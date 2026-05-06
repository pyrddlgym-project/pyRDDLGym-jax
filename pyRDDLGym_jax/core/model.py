from collections import deque
from copy import deepcopy
from enum import Enum
from functools import partial
import sys
import time
from tqdm import tqdm
from typing import Any, Callable, Dict, Generator, Iterable, Optional, Tuple

import jax
import jax.nn.initializers as initializers
import jax.numpy as jnp
import jax.random as random
import numpy as np
import optax

from pyRDDLGym.core.compiler.model import RDDLLiftedModel

from pyRDDLGym_jax.core.compiler import JaxRDDLSimState
from pyRDDLGym_jax.core.logic import JaxRDDLCompilerWithGrad
from pyRDDLGym_jax.core.planner import _jax_bound_action

Kwargs = Dict[str, Any]
State = Dict[str, np.ndarray]
Action = Dict[str, np.ndarray]
DataStream = Iterable[Tuple[State, Action, State]]
Params = Dict[str, np.ndarray]
Callback = Dict[str, Any]
LossFunction = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]


# ***********************************************************************
# ALL VERSIONS OF LOSS FUNCTIONS
#
# - loss functions based on specific likelihood assumptions (MSE, cross-entropy)
# 
# ***********************************************************************


def mean_squared_error() -> LossFunction:
    def _jax_wrapped_mse_loss(target, pred):
        return jnp.square(target - pred)
    return jax.jit(_jax_wrapped_mse_loss)


def binary_cross_entropy(eps: float=1e-8) -> LossFunction:
    def _jax_wrapped_binary_cross_entropy_loss(target, pred):
        pred = jnp.clip(pred, eps, 1.0 - eps)
        log_pred = jnp.log(pred)
        log_not_pred = jnp.log(1.0 - pred)
        return -target * log_pred - (1.0 - target) * log_not_pred
    return jax.jit(_jax_wrapped_binary_cross_entropy_loss)


def optax_loss(loss_fn: LossFunction, **kwargs) -> LossFunction:
    def _jax_wrapped_optax_loss(target, pred):
        return loss_fn(pred, target, **kwargs)
    return jax.jit(_jax_wrapped_optax_loss)


# ***********************************************************************
# ALL VERSIONS OF JAX MODEL LEARNER
#
# - gradient based model learning
# 
# ***********************************************************************


class JaxLearnerStatus(Enum):
    '''Represents the status of a parameter update from the JAX model learner, 
    including whether the update resulted in nan gradient, 
    whether progress was made, budget was reached, or other information that
    can be used to monitor and act based on the learner's progress.'''
    
    NORMAL = 0
    NO_PROGRESS = 1
    INVALID_GRADIENT = 2
    TIME_BUDGET_REACHED = 3
    ITER_BUDGET_REACHED = 4
    
    def is_terminal(self) -> bool:
        return self.value >= 2
    

class JaxModelLearner:
    '''A class for data-driven estimation of unknown parameters in a given RDDL MDP using 
    gradient descent.'''

    def __init__(self, rddl: RDDLLiftedModel, 
                 param_ranges: Dict[str, Tuple[Optional[np.ndarray], Optional[np.ndarray]]],
                 batch_size_train: int=32,
                 samples_per_datapoint: int=1,
                 optimizer: Callable[..., optax.GradientTransformation]=optax.rmsprop,
                 optimizer_kwargs: Optional[Kwargs]=None,
                 initializer: initializers.Initializer = initializers.normal(),
                 wrap_non_bool: bool=True,
                 bool_fluent_loss: LossFunction=binary_cross_entropy(),
                 real_fluent_loss: LossFunction=mean_squared_error(),
                 int_fluent_loss: LossFunction=mean_squared_error(),
                 compiler: JaxRDDLCompilerWithGrad=JaxRDDLCompilerWithGrad,
                 compiler_kwargs: Optional[Kwargs]=None,
                 model_params_reduction: Callable=lambda x: x[0]) -> None:
        '''Creates a new gradient-based algorithm for inferring unknown non-fluents
        in a RDDL domain from a data set or stream coming from the real environment.
        
        :param rddl: the RDDL domain to learn
        :param param_ranges: the ranges of all learnable non-fluents
        :param batch_size_train: how many transitions to compute per optimization 
        step
        :param samples_per_datapoint: how many random samples to produce from the step
        function per data point during training
        :param optimizer: a factory for an optax SGD algorithm
        :param optimizer_kwargs: a dictionary of parameters to pass to the SGD
        factory (e.g. which parameters are controllable externally)
        :param initializer: how to initialize non-fluents
        :param wrap_non_bool: whether to wrap non-boolean trainable parameters to satisfy
        required ranges as specified in param_ranges (use a projected gradient otherwise)
        :param bool_fluent_loss: loss function to optimize for bool-valued fluents
        :param real_fluent_loss: loss function to optimize for real-valued fluents
        :param int_fluent_loss: loss function to optimize for int-valued fluents
        :param compiler: compiler instance to use for planning
        :param compiler_kwargs: compiler instances kwargs for initialization
        :param model_params_reduction: how to aggregate updated model_params across runs
        in the batch (defaults to selecting the first element's parameters in the batch)
        '''
        self.rddl = rddl
        self.param_ranges = param_ranges.copy()
        self.batch_size_train = batch_size_train
        self.samples_per_datapoint = samples_per_datapoint
        if optimizer_kwargs is None:
            optimizer_kwargs = {'learning_rate': 0.001}
        self.optimizer_kwargs = optimizer_kwargs
        self.initializer = initializer
        self.wrap_non_bool = wrap_non_bool
        self.bool_fluent_loss = bool_fluent_loss
        self.real_fluent_loss = real_fluent_loss
        self.int_fluent_loss = int_fluent_loss
        self.model_params_reduction = model_params_reduction

        # validate param_ranges
        for (name, values) in param_ranges.items():
            if name not in rddl.non_fluents:
                raise ValueError(
                    f'param_ranges key <{name}> is not a valid non-fluent '
                    f'in the current rddl.')
            if not isinstance(values, (tuple, list)):
                raise ValueError(
                    f'param_ranges values with key <{name}> are not a tuple or list.')
            if len(values) != 2:
                raise ValueError(
                    f'param_ranges values with key <{name}> must be of length 2, '
                    f'got length {len(values)}.')
            lower, upper = values
            if lower is not None and upper is not None and not np.all(lower <= upper):
                raise ValueError(
                    f'param_ranges values with key <{name}> do not satisfy lower <= upper.')

        # build the optimizer
        optimizer = optimizer(**optimizer_kwargs)
        pipeline = [optimizer]
        self.optimizer = optax.chain(*pipeline)

        # build the computation graph
        if compiler_kwargs is None:
            compiler_kwargs = {}
        self.compiler_kwargs = compiler_kwargs
        self.compiler_type = compiler

        self.step_fn = self._jax_compile_rddl()
        self.map_fn = self._jax_map()
        self.loss_fn = self._jax_loss(map_fn=self.map_fn, step_fn=self.step_fn)
        self.update_fn, self.project_fn = self._jax_update(loss_fn=self.loss_fn)
        self.init_fn = self._jax_init(project_fn=self.project_fn)
    
    # ===========================================================================
    # COMPILATION SUBROUTINES
    # ===========================================================================

    def _jax_compile_rddl(self):

        # compile the RDDL model
        self.compiled = self.compiler_type(
            self.rddl,
            **self.compiler_kwargs
        )
        self.compiled.compile(log_jax_expr=True, heading='RELAXED MODEL')

        # compile the transition step function
        step_fn = self.compiled.compile_transition()

        def _jax_wrapped_step(sim_state, param_fluents, states, actions):
            nfls = sim_state.nfls.copy()
            nfls.update(param_fluents)
            fls = sim_state.fls.copy()
            fls.update(states)
            sim_state = sim_state.replace(nfls=nfls, fls=fls)
            sim_state, _ = step_fn(sim_state, actions)
            return sim_state

        # batched step function
        def _jax_wrapped_batched_step(sim_state, param_fluents, states, actions):
            keys = random.split(sim_state.key, num=self.batch_size_train)
            sim_state = sim_state.replace(key=keys)
            sim_state = jax.vmap(
                _jax_wrapped_step, 
                in_axes=(JaxRDDLSimState(key=0, fls=0), None, 0, 0)
            )(sim_state, param_fluents, states, actions)
            model_params = jax.tree_util.tree_map(
                self.model_params_reduction, sim_state.model_params)
            sim_state = sim_state.replace(model_params=model_params)
            return sim_state

        # batched step function with parallel samples per data point
        def _jax_wrapped_batched_parallel_step(sim_state, param_fluents, states, actions):
            keys = random.split(sim_state.key, num=self.samples_per_datapoint)
            sim_state = sim_state.replace(key=keys)
            sim_state = jax.vmap(
                _jax_wrapped_batched_step, 
                in_axes=(JaxRDDLSimState(key=0), None, None, None)
            )(sim_state, param_fluents, states, actions)
            model_params = jax.tree_util.tree_map(
                self.model_params_reduction, sim_state.model_params)
            sim_state = sim_state.replace(model_params=model_params)
            return sim_state
        return jax.jit(_jax_wrapped_batched_parallel_step)

    def _jax_map(self):

        # compute case indices for bounding
        case_indices = {}
        if self.wrap_non_bool:
            for (name, (lower, upper)) in self.param_ranges.items():
                if lower is None: 
                    lower = -np.inf
                if upper is None: 
                    upper = +np.inf
                self.param_ranges[name] = (lower, upper)
                case_indices[name] = [
                    np.isfinite(lower) & np.isfinite(upper),
                    np.isfinite(lower) & ~np.isfinite(upper),
                    ~np.isfinite(lower) & np.isfinite(upper),
                    ~np.isfinite(lower) & ~np.isfinite(upper)
                ]
        
        # map trainable parameters to their non-fluent values
        def _jax_wrapped_params_to_fluents(params):
            result = {}
            for (name, param) in params.items():
                if self.rddl.variable_ranges[name] == 'bool':
                    result[name] = jax.nn.sigmoid(param)
                else:
                    if self.wrap_non_bool:
                        lower, upper = self.param_ranges[name]
                        indices = case_indices[name]
                        result[name] = _jax_bound_action(indices, lower, upper, param)
                    else:
                        result[name] = param
            return result
        return jax.jit(_jax_wrapped_params_to_fluents)

    def _jax_loss(self, map_fn, step_fn):

        # use binary cross entropy for bool fluents
        # mean squared error for continuous and integer fluents
        def _jax_wrapped_batched_model_loss(
                sim_state, params, states, actions, next_fluents):
            param_fluents = map_fn(params)
            next_sim_state = step_fn(sim_state, param_fluents, states, actions)
            
            total_loss = 0.0
            for (name, value) in next_fluents.items():
                preds = jnp.asarray(next_sim_state.fls[name], dtype=self.compiled.REAL)
                targets = jnp.asarray(value, dtype=self.compiled.REAL)[jnp.newaxis, ...]
                prange = self.rddl.variable_ranges[name]
                if prange == 'bool':
                    loss = self.bool_fluent_loss(targets, preds)
                elif prange == 'real':
                    loss = self.real_fluent_loss(targets, preds)
                else:
                    loss = self.int_fluent_loss(targets, preds)
                total_loss += jnp.mean(loss) / len(next_fluents)
            return total_loss
        return jax.jit(_jax_wrapped_batched_model_loss)
    
    def _jax_init(self, project_fn):
        
        # initialize both the non-fluents and optimizer
        def _jax_wrapped_init_params_optimizer(key, guess):
            if guess is None:
                params = {}
                for name in self.param_ranges:
                    shape = jnp.shape(self.compiled.init_values[name])
                    key, subkey = random.split(key)
                    params[name] = self.initializer(subkey, shape, dtype=self.compiled.REAL)
            else:
                params = guess
            params = project_fn(params)
            opt_state = self.optimizer.init(params)
            return params, opt_state
        return jax.jit(_jax_wrapped_init_params_optimizer)
                    
    def _jax_update(self, loss_fn):

        # projected gradient trick to satisfy box constraints on params
        def _jax_wrapped_project_params(params):
            if self.wrap_non_bool:
                return params
            else:
                new_params = {}
                for (name, value) in params.items():
                    if self.rddl.variable_ranges[name] == 'bool':
                        new_params[name] = value
                    else:
                        new_params[name] = jnp.clip(value, *self.param_ranges[name])
                return new_params
        project_fn = jax.jit(_jax_wrapped_project_params)

        # gradient descent update
        def _jax_wrapped_params_update(
                sim_state, params, states, actions, next_fluents, opt_state):
            loss_val, grad = jax.value_and_grad(loss_fn, argnums=1)(
                sim_state, params, states, actions, next_fluents)
            updates, opt_state = self.optimizer.update(grad, opt_state)
            params = optax.apply_updates(params, updates)
            params = _jax_wrapped_project_params(params)
            zero_grads = jax.tree_util.tree_map(partial(jnp.allclose, b=0), grad)
            return opt_state, params, loss_val, zero_grads
        update_fn = jax.jit(_jax_wrapped_params_update)

        return update_fn, project_fn
    
    def _batched_init_subs(self): 
        init_fls, init_nfls = {}, {}
        for (name, value) in self.compiled.init_values.items():
            value = np.asarray(value, dtype=self.compiled.REAL)
            if name in self.rddl.non_fluents:
                init_nfls[name] = value
            else:
                init_fls[name] = np.repeat(
                    value[np.newaxis, ...], repeats=self.batch_size_train, axis=0)
        for (state, next_state) in self.rddl.next_state.items():
            init_fls[next_state] = init_fls[state]
        return init_fls, init_nfls
    
    # ===========================================================================
    # ESTIMATE API
    # ===========================================================================

    def optimize(self, *args, **kwargs) -> Optional[Callback]:
        '''Estimate the unknown parameters from the given data set. 
        Return the callback from training.

        :param data: a data stream represented as a (possibly infinite) sequence of 
        transition batches of the form (states, actions, next-states), where each element
        is a numpy array of leading dimension equal to batch_size_train
        :param key: JAX PRNG key (derived from clock if not provided)
        :param epochs: the maximum number of steps of gradient descent
        :param train_seconds: total time allocated for gradient descent  
        :param guess: initial non-fluent parameters: if None will use the initializer
        specified in this instance   
        :param print_progress: whether to print the progress bar during training
        '''
        it = self.optimize_generator(*args, **kwargs)
        
        # https://stackoverflow.com/questions/50937966/fastest-most-pythonic-way-to-consume-an-iterator
        callback = None
        if sys.implementation.name == 'cpython':
            last_callback = deque(it, maxlen=1)
            if last_callback:
                callback = last_callback.pop()
        else:
            for callback in it:
                pass
        return callback

    def optimize_generator(self, data: DataStream,
                           key: Optional[random.PRNGKey]=None,
                           epochs: int=999999,
                           train_seconds: float=120.,
                           guess: Optional[Params]=None,
                           print_progress: bool=True) -> Generator[Callback, None, None]:
        '''Return a generator for estimating the unknown parameters from the given data set. 
        Generator can be iterated over to lazily estimate the parameters, yielding
        a dictionary of intermediate computations.

        :param data: a data stream represented as a (possibly infinite) sequence of 
        transition batches of the form (states, actions, next-states), where each element
        is a numpy array of leading dimension equal to batch_size_train
        :param key: JAX PRNG key (derived from clock if not provided)
        :param epochs: the maximum number of steps of gradient descent
        :param train_seconds: total time allocated for gradient descent
        :param guess: initial non-fluent parameters: if None will use the initializer
        specified in this instance
        :param print_progress: whether to print the progress bar during training
        '''
        start_time = time.time()
        elapsed_outside_loop = 0

        # if PRNG key is not provided
        if key is None:
            key = random.PRNGKey(round(time.time() * 1000))
        
        # prepare initial subs
        fls, nfls = self._batched_init_subs()
        hyperparams = self.compiled.model_aux['params']
        sim_state = JaxRDDLSimState(key=key, fls=fls, nfls=nfls, model_params=hyperparams)

        # initialize parameter fluents to optimize
        params, opt_state = self.init_fn(key, guess)
        
        # progress bar
        if print_progress:
            progress_bar = tqdm(
                None, total=100, bar_format='{l_bar}{bar}| {elapsed} {postfix}')
        else:
            progress_bar = None

        # main training loop
        for (it, (states, actions, next_states)) in enumerate(data):
            status = JaxLearnerStatus.NORMAL

            # gradient update
            key, subkey = random.split(key)
            sim_state = sim_state.replace(key=subkey)
            opt_state, params, loss, zero_grads = self.update_fn(
                sim_state, params, states, actions, next_states, opt_state)
            
            # extract non-fluent values from the trainable parameters
            param_fluents = self.map_fn(params)
            param_fluents = {name: param_fluents[name] for name in self.param_ranges}

            # check for learnability
            if any(zero_grads.values()):
                status = JaxLearnerStatus.NO_PROGRESS

            # reached computation budget
            elapsed = time.time() - start_time - elapsed_outside_loop
            if elapsed >= train_seconds:
                status = JaxLearnerStatus.TIME_BUDGET_REACHED
            if it >= epochs - 1:
                status = JaxLearnerStatus.ITER_BUDGET_REACHED
            
            # build a callback
            progress_percent = 100 * min(
                1, max(0, elapsed / train_seconds, it / (epochs - 1)))
            callback = {
                'status': status,
                'iteration': it,
                'train_loss': loss,
                'params': params,
                'param_fluents': param_fluents,
                'key': sim_state.key,
                'progress': progress_percent
            }
            
            # update progress
            if print_progress:
                progress_bar.set_description(
                    f'{it:7} it / {loss:13.8f} train / {status.value} status', refresh=False)
                progress_bar.set_postfix_str(
                    f'{(it + 1) / (elapsed + 1e-6):.2f}it/s', refresh=False)
                progress_bar.update(progress_percent - progress_bar.n)
            
            # yield the callback
            start_time_outside = time.time()
            yield callback
            elapsed_outside_loop += (time.time() - start_time_outside)

            # abortion check
            if status.is_terminal():
                break
    
    def learned_model(self, param_fluents: Params) -> RDDLLiftedModel:
        '''Substitutes the given learned non-fluent values into the RDDL model and returns
        the new model.

        :param param_fluents: the learned non-fluent values
        '''
        model = deepcopy(self.rddl)
        for (name, values) in param_fluents.items():
            prange = model.variable_ranges[name]
            if prange == 'bool':
                values = values > 0.5
            elif prange != 'real':
                values = np.asarray(values, dtype=self.compiled.INT)
            values = np.ravel(values, order='C').tolist()
            if not self.rddl.variable_params[name]:
                assert (len(values) == 1)
                values = values[0]
            model.non_fluents[name] = values
        return model


def generate_rollouts(env, policy, episodes=10, max_steps=100):
    '''Generates rollouts from the given environment and policy.'''
    model = env.model
    states = {k: [] for k in model.state_fluents}
    actions = {k: [] for k in model.action_fluents}
    next_states = {k: [] for k in model.state_fluents}

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        while not done and steps < max_steps:
            action = policy(obs)
            next_obs, reward, term, trunc, info = env.step(action)
            for key in states:
                states[key].append(obs[key])
                next_states[key].append(next_obs[key])
            for key in actions:
                actions[key].append(action[key])
            obs = next_obs
            done = term or trunc
            steps += 1
    
    state_shapes = {k: (-1, *np.shape(v)) for k, v in model.state_fluents.items()}
    action_shapes = {k: (-1, *np.shape(v)) for k, v in model.action_fluents.items()}
    states = {k: np.reshape(v, state_shapes[k]) for k, v in states.items()}
    actions = {k: np.reshape(v, action_shapes[k]) for k, v in actions.items()}
    next_states = {k: np.reshape(v, state_shapes[k]) for k, v in next_states.items()}
    return states, actions, next_states


def batch_sampler(states, actions, next_states, batch_size=32):
    '''Yields batches of transitions from the given states, actions, and next_states.'''
    num_transitions = len(next(iter(states.values())))
    indices = np.arange(num_transitions)
    while True:
        np.random.shuffle(indices)
        for start in range(0, num_transitions - batch_size, batch_size):
            end = start + batch_size
            batch_id = indices[start:end]
            batch_states = {k: v[batch_id] for k, v in states.items()}
            batch_actions = {k: v[batch_id] for k, v in actions.items()}
            batch_next = {k: v[batch_id] for k, v in next_states.items()}
            yield batch_states, batch_actions, batch_next


if __name__ == '__main__':
    import os
    import pyRDDLGym
    from pyRDDLGym_jax.core.planner import load_config, JaxBackpropPlanner, JaxOfflineController
    
    # make some data
    policy = lambda obs: {'force': 5 if obs['ang-pos'] + obs['ang-vel'] > 0 else -5}
    env = pyRDDLGym.make('CartPole_Continuous_gym', '0', vectorized=True)
    states, actions, next_states = generate_rollouts(env, policy, episodes=10, max_steps=100)
    data_iterator = batch_sampler(states, actions, next_states, batch_size=32)

    # train it
    model_learner = JaxModelLearner(rddl=env.model, 
                                    param_ranges={
                                        'GRAVITY': (0., 20.)
                                    }, 
                                    batch_size_train=32, 
                                    optimizer_kwargs = {'learning_rate': 0.001})
    for cb in model_learner.optimize_generator(data_iterator, epochs=10000):
        if cb['iteration'] % 100 == 0:
            print(cb['param_fluents'])

    # planning in the trained model
    model = model_learner.learned_model(cb['param_fluents'])
    abs_path = os.path.dirname(os.path.abspath(__file__))        
    config_path = os.path.join(
        os.path.dirname(abs_path), 'examples', 'configs', 'default_drp.cfg') 
    planner_args, _, train_args = load_config(config_path)
    planner = JaxBackpropPlanner(rddl=model, **planner_args)
    controller = JaxOfflineController(planner, **train_args)

    # evaluation of the plan
    test_env = pyRDDLGym.make('CartPole_Continuous_gym', '0', vectorized=True)
    controller.evaluate(test_env, episodes=1, verbose=True, render=True)
