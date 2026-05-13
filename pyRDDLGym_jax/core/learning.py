from collections import deque
from copy import deepcopy
from enum import Enum
import gymnasium as gym
from functools import partial
from itertools import chain, islice
import sys
import time
from tqdm import tqdm
from typing import Any, Callable, Dict, Generator, Iterable, Optional, Tuple, Union

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

Floating = Union[np.ndarray, float]
Kwargs = Dict[str, Any]
State = Dict[str, np.ndarray]
Action = Dict[str, np.ndarray]
Transition = Tuple[State, Action, State]
DataStream = Iterable[Transition]
Params = Dict[str, np.ndarray]
Callback = Dict[str, Any]


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
                 param_ranges: Dict[str, Tuple[Optional[Floating], Optional[Floating]]],
                 batch_size_train: int=32,
                 samples_per_datapoint: int=16,
                 antithetic_sampling: bool=True,
                 optimizer: Callable[..., optax.GradientTransformation]=optax.rmsprop,
                 optimizer_kwargs: Optional[Kwargs]=None,
                 initializer: initializers.Initializer = initializers.normal(),
                 wrap_non_bool: bool=False,
                 normalize_next_states: bool=True,
                 compiler: type=JaxRDDLCompilerWithGrad,
                 compiler_kwargs: Optional[Kwargs]=None,
                 model_params_reduction: Callable=lambda x: x[0]) -> None:
        '''Creates a new gradient-based algorithm for inferring unknown non-fluents
        in a RDDL domain from a data set or stream coming from the real environment.
        
        :param rddl: the RDDL domain to learn
        :param param_ranges: the ranges of all learnable non-fluents
        :param batch_size_train: how many transitions to compute per optimization step
        :param samples_per_datapoint: how many random samples to produce from the step
        function per data point during training
        :param antithetic_sampling: whether to pair Monte Carlo samples with
        deterministic key-space mirrors to reduce gradient variance
        :param optimizer: a factory for an optax SGD algorithm
        :param optimizer_kwargs: a dictionary of parameters to pass to the SGD
        factory (e.g. which parameters are controllable externally)
        :param initializer: how to initialize non-fluents
        :param wrap_non_bool: whether to wrap non-boolean trainable parameters to satisfy
        required ranges as specified in param_ranges (use a projected gradient otherwise)
        :param compiler: compiler instance to use for planning
        :param compiler_kwargs: compiler instances kwargs for initialization
        :param model_params_reduction: how to aggregate updated model_params across runs
        in the batch (defaults to selecting the first element's parameters in the batch)
        '''
        self.rddl = rddl
        self.param_ranges = param_ranges.copy()
        self.batch_size_train = batch_size_train
        self.samples_per_datapoint = samples_per_datapoint
        self.antithetic_sampling = antithetic_sampling
        if optimizer_kwargs is None:
            optimizer_kwargs = {'learning_rate': 0.001}
        self.optimizer_kwargs = optimizer_kwargs
        self.initializer = initializer
        self.wrap_non_bool = wrap_non_bool
        self.normalize_next_states = normalize_next_states
        self.model_params_reduction = model_params_reduction
        self.state_keys = tuple(sorted(rddl.state_fluents))

        # validate param_ranges
        for (name, values) in param_ranges.items():
            if name not in rddl.non_fluents:
                raise ValueError(f'param_ranges key <{name}> is not a valid non-fluent.')
            if not isinstance(values, (tuple, list)):
                raise ValueError(f'param_ranges key <{name}> is not a tuple or list.')
            if len(values) != 2:
                raise ValueError(f'param_ranges key <{name}> must be of length 2, '
                                 f'got length {len(values)}.')
            lower, upper = values
            if lower is not None and upper is not None and not np.all(lower <= upper):
                raise ValueError(f'param_ranges key <{name}> do not satisfy lower <= upper.')

        # build the optimizer
        optimizer = optimizer(**optimizer_kwargs)
        pipeline = [optax.clip_by_global_norm(1.0), optimizer]
        self.optimizer = optax.chain(*pipeline)

        # build the computation graph
        if compiler_kwargs is None:
            compiler_kwargs = {}
        self.compiler_kwargs = compiler_kwargs
        self.compiler_type = compiler
        self.step_fn = self._jax_compile_rddl()
        self.map_fn = self._jax_map()

        # build the loss and update functions
        self.loss_fn = self._jax_loss(map_fn=self.map_fn, step_fn=self.step_fn)
        self.update_fn, self.project_fn = self._jax_update(loss_fn=self.loss_fn)
        self.init_fn = self._jax_init(project_fn=self.project_fn)
    
    # ===========================================================================
    # RDDL COMPILATION SUBROUTINES
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

        # wrap the step function to take in the trainable parameters and batch dimension
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
        return jax.jit(_jax_wrapped_batched_step)

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

    def _fit_norm_stats(self, batches):

        # fit sufficient stats for the next states from the provided batches of transitions
        totals, totals_sq, counts = {}, {}, {}
        for (_, _, next_states) in batches:
            for (name, value) in next_states.items():
                value = np.asarray(value, dtype=np.float32)
                if value.ndim == 0:
                    value = value.reshape((1,))
                totals[name] = totals.get(name, 0) + np.sum(value, axis=0)
                totals_sq[name] = totals_sq.get(name, 0) + np.sum(np.square(value), axis=0)
                counts[name] = counts.get(name, 0) + value.shape[0]

        # compute mean and std from the sufficient stats
        stats = {}
        for name in totals:
            count = max(1, counts[name])
            mean = totals[name] / count
            var = np.maximum(totals_sq[name] / count - np.square(mean), 0.0)
            std = np.sqrt(var)
            stats[name] = (mean, std)
        return stats

    # ===========================================================================
    # LOSS CALCULATION AND TRAINING SUBROUTINES
    # ===========================================================================

    def _jax_loss(self, map_fn, step_fn):

        # flatten the samples into a single vector
        def _jax_wrapped_flatten_and_norm(values, is_target, norm_stats):
            parts = []
            for key in self.state_keys:
                value = jnp.asarray(values[key])
                if norm_stats is not None:
                    mean, std = norm_stats[key]
                    value = (value - mean) / (std + 1e-10)
                if is_target:
                    value = jnp.reshape(value, (value.shape[0], -1))
                else:
                    value = jnp.reshape(value, (value.shape[0], value.shape[1], -1))
                parts.append(value)
            return jnp.concatenate(parts, axis=-1)
        
        # energy score E||X - y||^2 - 0.5 E||X - X'||^2
        def _jax_wrapped_energy_score_loss(next_states, pred_next_states, norm_stats):
            target_vec = _jax_wrapped_flatten_and_norm(next_states, True, norm_stats)
            pred_vec = _jax_wrapped_flatten_and_norm(pred_next_states, False, norm_stats)
            diff_to_target = pred_vec - jnp.expand_dims(target_vec, axis=0)
            first_term = jnp.mean(jnp.sum(jnp.square(diff_to_target), axis=-1))
            pw = pred_vec[:, jnp.newaxis, :, :] - pred_vec[jnp.newaxis, :, :, :]
            second_term = 0.5 * jnp.mean(jnp.sum(jnp.square(pw), axis=-1))
            return first_term - second_term

        # draw independent transition samples for energy score
        def _jax_wrapped_batched_model_loss(sim_state, params, states, actions, 
                                            next_fluents, norm_stats):
            param_fluents = map_fn(params)
            def _jax_wrapped_single_sample(subkey):
                sample_state = sim_state.replace(key=subkey)
                sample_next_state = step_fn(sample_state, param_fluents, states, actions)
                return sample_next_state.fls

            # For energy score, we use at least two samples and optionally pair
            # each sampled key with a deterministic mirror key.
            num_samples = max(2, self.samples_per_datapoint)
            if self.antithetic_sampling:
                primary_count = (num_samples + 1) // 2
                primary_keys = random.split(sim_state.key, num=primary_count)
                mirror_mask = jnp.asarray([0xFFFFFFFF, 0x9E3779B9], dtype=primary_keys.dtype)
                mirror_keys = jax.vmap(lambda k: jnp.bitwise_xor(k, mirror_mask))(primary_keys)
                sample_keys = jnp.concatenate([primary_keys, mirror_keys], axis=0)[:num_samples]
            else:
                sample_keys = random.split(sim_state.key, num=num_samples)

            pred_samples = jax.vmap(_jax_wrapped_single_sample)(sample_keys)
            loss = _jax_wrapped_energy_score_loss(next_fluents, pred_samples, norm_stats)
            return loss
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
        def _jax_wrapped_params_update(sim_state, params, states, actions, next_fluents, 
                                       norm_stats, opt_state):
            loss_val, grad = jax.value_and_grad(loss_fn, argnums=1)(
                sim_state, params, states, actions, next_fluents, norm_stats)
            updates, opt_state = self.optimizer.update(grad, opt_state)
            params = optax.apply_updates(params, updates)
            params = _jax_wrapped_project_params(params)
            zero_grads = jax.tree_util.tree_map(partial(jnp.allclose, b=0), grad)
            return opt_state, params, loss_val, zero_grads
        update_fn = jax.jit(_jax_wrapped_params_update)

        return update_fn, project_fn
    
    # ===========================================================================
    # ESTIMATE API
    # ===========================================================================

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
        :param norm_estimation_batches: number of batches to use for normalization
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
                           print_progress: bool=True,
                           norm_estimation_batches: int=32) -> Generator[Callback, None, None]:
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
        :param norm_estimation_batches: number of batches to use for normalization
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

        # infer normalization from the provided dataset stream
        if self.normalize_next_states:
            norm_batches = list(islice(data, norm_estimation_batches))
            norm_stats = self._fit_norm_stats(norm_batches)
            data = chain(norm_batches, data)
        else:
            norm_stats = None

        # main training loop
        for (it, (states, actions, next_states)) in enumerate(data):
            status = JaxLearnerStatus.NORMAL

            # gradient update
            key, subkey = random.split(key)
            sim_state = sim_state.replace(key=subkey)
            opt_state, params, loss, zero_grads = self.update_fn(
                sim_state, params, states, actions, next_states, norm_stats, opt_state)
            
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
            progress = 100 * min(1, max(0, elapsed / train_seconds, it / (epochs - 1)))
            callback = {
                'status': status,
                'iteration': it,
                'train_loss': loss,
                'params': params,
                'param_fluents': param_fluents,
                'key': sim_state.key,
                'progress': progress
            }
            
            # update progress
            if print_progress:
                progress_bar.set_description(
                    f'{it:7} it / {loss:13.8f} train / {status.value} status', refresh=False)
                progress_bar.set_postfix_str(
                    f'{(it + 1) / (elapsed + 1e-6):.2f}it/s', refresh=False)
                progress_bar.update(progress - progress_bar.n)
            
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


# ***********************************************************************
# DATA GENERATION HELPERS
#
# - rollout generation
# - batched sampling
# 
# ***********************************************************************

def generate_rollouts(env: gym.Env, policy: Callable, episodes: int, max_steps: int
                      ) -> Transition:
    '''Generates rollouts from the given environment and policy.
    
    :param env: the environment to generate rollouts from
    :param policy: a function that takes in an observation and returns an action
    :param episodes: how many episodes to generate
    :param max_steps: the maximum number of steps per episode
    '''
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
    
    # reshape the data into arrays of shape (num_transitions, fluent_size)
    state_shapes = {k: (-1, *np.shape(v)) for k, v in model.state_fluents.items()}
    action_shapes = {k: (-1, *np.shape(v)) for k, v in model.action_fluents.items()}
    states = {k: np.reshape(v, state_shapes[k]) for k, v in states.items()}
    actions = {k: np.reshape(v, action_shapes[k]) for k, v in actions.items()}
    next_states = {k: np.reshape(v, state_shapes[k]) for k, v in next_states.items()}
    return states, actions, next_states


def batch_sampler(states: State, actions: Action, next_states: State, batch_size: int=32
                  ) -> Generator[Transition, None, None]:
    '''Yields batches of transitions from the given states, actions, and next_states.
    
    :param states: a dictionary mapping state fluent names to numpy arrays
    :param actions: a dictionary mapping action fluent names to numpy arrays
    :param next_states: a dictionary mapping state fluent names to numpy arrays
    :param batch_size: how many transitions to include in each batch
    '''
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
    policy = lambda obs: {'release': np.random.uniform(0.0, 20., size=(3,))}
    env = pyRDDLGym.make('Reservoir_Continuous', '0', vectorized=True)
    transitions = generate_rollouts(env, policy, episodes=100, max_steps=40)
    data_iterator = batch_sampler(*transitions, batch_size=32)

    # train it
    model_learner = JaxModelLearner(rddl=env.model, 
                                    param_ranges={
                                        'RAIN_VAR': (0., 10.)
                                    }, 
                                    batch_size_train=32, 
                                    optimizer_kwargs = {'learning_rate': 0.0003})
    for cb in model_learner.optimize_generator(data_iterator, epochs=50000, print_progress=True):
        if cb['iteration'] % 2000 == 0:
            print(cb['param_fluents'])
        
    # planning in the trained model
    model = model_learner.learned_model(cb['param_fluents'])
    abs_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   
    config_path = os.path.join(abs_path, 'examples', 'configs', 'default_drp.cfg') 
    planner_args, _, train_args = load_config(config_path)
    planner = JaxBackpropPlanner(model, **planner_args)
    controller = JaxOfflineController(planner, **train_args)

    # evaluation of the plan
    test_env = pyRDDLGym.make('Reservoir_Continuous', '0', vectorized=True)
    controller.evaluate(test_env, episodes=1, verbose=True, render=True)
