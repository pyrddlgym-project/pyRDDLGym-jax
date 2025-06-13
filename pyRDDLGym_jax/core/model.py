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

from pyRDDLGym_jax.core.logic import Logic, ExactLogic
from pyRDDLGym_jax.core.planner import JaxRDDLCompilerWithGrad

Kwargs = Dict[str, Any]
State = Dict[str, np.ndarray]
Action = Dict[str, np.ndarray]
Params = Dict[str, np.ndarray]
Callback = Dict[str, Any]
LossFunction = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]


# ***********************************************************************
# ALL VERSIONS OF LOSS FUNCTIONS
#
# - loss functions based on specific likelihood assumptions (MSE, Huber, cross-entropy)
# 
# ***********************************************************************


def mean_squared_error() -> LossFunction:
    def _jax_wrapped_mse_loss(target, pred):
        loss_values = jnp.square(target - pred)
        return loss_values
    return jax.jit(_jax_wrapped_mse_loss)


def huber_loss(delta: float=1.0) -> LossFunction:
    def _jax_wrapped_huber_loss(target, pred):
        loss_values = optax.losses.huber_loss(pred, target, delta=delta)
        return loss_values
    return jax.jit(_jax_wrapped_huber_loss)


def binary_cross_entropy(eps: float=1e-6) -> LossFunction:
    def _jax_wrapped_binary_cross_entropy_loss(target, pred):
        pred = jnp.clip(pred, eps, 1.0 - eps)
        log_pred = jnp.log(pred)
        log_not_pred = jnp.log(1.0 - pred)
        loss_values = -target * log_pred - (1.0 - target) * log_not_pred
        return loss_values
    return jax.jit(_jax_wrapped_binary_cross_entropy_loss)


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
                 use64bit: bool=False,
                 bool_fluent_loss: LossFunction=binary_cross_entropy(),
                 real_fluent_loss: LossFunction=mean_squared_error(),
                 int_fluent_loss: LossFunction=mean_squared_error(),
                 logic: Logic=ExactLogic()) -> None:
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
        :param use64bit: whether to perform arithmetic in 64 bit
        :param bool_fluent_loss: loss function to optimize for bool-valued fluents
        :param real_fluent_loss: loss function to optimize for real-valued fluents
        :param int_fluent_loss: loss function to optimize for int-valued fluents
        :param logic: a subclass of Logic for mapping exact mathematical
        operations to their differentiable counterparts 
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
        self.use64bit = use64bit
        self.bool_fluent_loss = bool_fluent_loss
        self.real_fluent_loss = real_fluent_loss
        self.int_fluent_loss = int_fluent_loss
        self.logic = logic

        # build the optimizer
        optimizer = optimizer(**optimizer_kwargs)
        pipeline = [optimizer]
        self.optimizer = optax.chain(*pipeline)

        # build the computation graph
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
        self.compiled = JaxRDDLCompilerWithGrad(
            rddl=self.rddl,
            logic=self.logic,
            use64bit=self.use64bit,
            compile_non_fluent_exact=False,
            print_warnings=True
        )
        self.compiled.compile(log_jax_expr=True, heading='RELAXED MODEL')

        # compile the transition step function
        step_fn = self.compiled.compile_transition()

        def _jax_wrapped_step(key, param_fluents, subs, actions, hyperparams):
            for (name, param) in param_fluents.items():
                subs[name] = param
            subs, _, hyperparams = step_fn(key, actions, subs, hyperparams)
            return subs, hyperparams

        # batched step function
        # TODO: come up with a better way to reduce the hyperparams batch dim
        def _jax_wrapped_batched_step(key, param_fluents, subs, actions, hyperparams):
            keys = jnp.asarray(random.split(key, num=self.batch_size_train))
            subs, hyperparams = jax.vmap(
                _jax_wrapped_step, in_axes=(0, None, 0, 0, None)
            )(keys, param_fluents, subs, actions, hyperparams)
            hyperparams = jax.tree_util.tree_map(partial(jnp.mean, axis=0), hyperparams)
            return subs, hyperparams

        # batched step function with parallel samples per data point
        # TODO: come up with a better way to reduce the hyperparams batch dim
        def _jax_wrapped_batched_parallel_step(key, param_fluents, subs, actions, hyperparams):
            keys = jnp.asarray(random.split(key, num=self.samples_per_datapoint))
            subs, hyperparams = jax.vmap(
                _jax_wrapped_batched_step, in_axes=(0, None, None, None, None)
            )(keys, param_fluents, subs, actions, hyperparams)
            hyperparams = jax.tree_util.tree_map(partial(jnp.mean, axis=0), hyperparams)
            return subs, hyperparams

        batched_step_fn = jax.jit(_jax_wrapped_batched_parallel_step)
        return batched_step_fn        

    def _jax_map(self):

        # compute case indices for bounding
        case_indices = {}
        if self.wrap_non_bool:
            for (name, (lower, upper)) in self.param_ranges.items():
                if lower is None: lower = -np.inf
                if upper is None: upper = +np.inf
                self.param_ranges[name] = (lower, upper)
                case_indices[name] = (
                    0 * (np.isfinite(lower) & np.isfinite(upper)) +
                    1 * (np.isfinite(lower) & ~np.isfinite(upper)) +
                    2 * (~np.isfinite(lower) & np.isfinite(upper)) +
                    3 * (~np.isfinite(lower) & ~np.isfinite(upper))
                )
        
        # map trainable parameters to their non-fluent values
        def _jax_wrapped_params_to_fluents(params):
            param_fluents = {}
            for (name, param) in params.items():
                if self.rddl.variable_ranges[name] == 'bool':
                    param_fluents[name] = jax.nn.sigmoid(param)
                else:
                    if self.wrap_non_bool:
                        lower, upper = self.param_ranges[name]
                        cases = [
                            lambda x: lower + (upper - lower) * jax.nn.sigmoid(x),
                            lambda x: lower + (jax.nn.elu(x) + 1.0),
                            lambda x: upper - (jax.nn.elu(-x) + 1.0),
                            lambda x: x
                        ]
                        indices = case_indices[name]
                        param_fluents[name] = jax.lax.switch(indices, cases, param)
                    else:
                        param_fluents[name] = param
            return param_fluents
        
        map_fn = jax.jit(_jax_wrapped_params_to_fluents)
        return map_fn

    def _jax_loss(self, map_fn, step_fn):

        # use binary cross entropy for bool fluents
        # mean squared error for continuous and integer fluents
        def _jax_wrapped_batched_model_loss(key, param_fluents, subs, actions, next_fluents, 
                                            hyperparams):
            next_subs, hyperparams = step_fn(key, param_fluents, subs, actions, hyperparams)
            total_loss = 0.0
            for (name, next_value) in next_fluents.items():
                preds = jnp.asarray(next_subs[name], dtype=self.compiled.REAL)
                targets = jnp.asarray(next_value, dtype=self.compiled.REAL)[jnp.newaxis, ...]
                if self.rddl.variable_ranges[name] == 'bool':
                    loss_values = self.bool_fluent_loss(targets, preds)
                elif self.rddl.variable_ranges[name] == 'real':
                    loss_values = self.real_fluent_loss(targets, preds)
                else:
                    loss_values = self.int_fluent_loss(targets, preds)
                total_loss += jnp.mean(loss_values) / len(next_fluents)
            return total_loss, hyperparams
        
        # loss with the parameters mapped to their fluents
        def _jax_wrapped_batched_loss(key, params, subs, actions, next_fluents, hyperparams):
            param_fluents = map_fn(params)
            loss, hyperparams = _jax_wrapped_batched_model_loss(
                key, param_fluents, subs, actions, next_fluents, hyperparams)
            return loss, hyperparams

        loss_fn = jax.jit(_jax_wrapped_batched_loss)
        return loss_fn
    
    def _jax_init(self, project_fn):
        optimizer = self.optimizer
        
        # initialize both the non-fluents and optimizer
        def _jax_wrapped_init_params_optimizer(key):
            params = {}
            for name in self.param_ranges:
                shape = jnp.shape(self.compiled.init_values[name])
                key, subkey = random.split(key)
                params[name] = self.initializer(subkey, shape, dtype=self.compiled.REAL)
            params = project_fn(params)
            opt_state = optimizer.init(params)
            return params, opt_state
        
        init_fn = jax.jit(_jax_wrapped_init_params_optimizer)
        return init_fn
                    
    def _jax_update(self, loss_fn):
        optimizer = self.optimizer

        # projected gradient trick to satisfy box constraints on params
        if self.wrap_non_bool:
            def _jax_wrapped_project_params(params):
                return params
        else:
            def _jax_wrapped_project_params(params):
                new_params = {}
                for (name, value) in params.items():
                    if self.rddl.variable_ranges[name] == 'bool':
                        new_params[name] = value
                    else:
                        lower, upper = self.param_ranges[name]
                        new_params[name] = jnp.clip(value, lower, upper)
                return new_params

        # gradient descent update
        def _jax_wrapped_params_update(key, params, subs, actions, next_fluents, 
                                       hyperparams, opt_state):
            (loss_val, hyperparams), grad = jax.value_and_grad(
                loss_fn, argnums=1, has_aux=True
            )(key, params, subs, actions, next_fluents, hyperparams)
            updates, opt_state = optimizer.update(grad, opt_state)
            params = optax.apply_updates(params, updates)
            params = _jax_wrapped_project_params(params)
            zero_grads = jax.tree_util.tree_map(partial(jnp.allclose, b=0.0), grad)
            return params, opt_state, loss_val, zero_grads, hyperparams
    
        update_fn = jax.jit(_jax_wrapped_params_update)
        project_fn = jax.jit(_jax_wrapped_project_params)
        return update_fn, project_fn
    
    def _batched_init_subs(self): 
        init_train = {}
        for (name, value) in self.compiled.init_values.items():
            value = np.reshape(value, np.shape(value))[np.newaxis, ...]
            value = np.repeat(value, repeats=self.batch_size_train, axis=0)
            value = np.asarray(value, dtype=self.compiled.REAL)
            init_train[name] = value
        for (state, next_state) in self.rddl.next_state.items():
            init_train[next_state] = init_train[state]
        return init_train
    
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

    def optimize_generator(self, data: Iterable[Tuple[State, Action, State]],
                           key: Optional[random.PRNGKey]=None,
                           epochs: int=999999,
                           train_seconds: float=120.,
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
        :param print_progress: whether to print the progress bar during training
        '''
        start_time = time.time()
        elapsed_outside_loop = 0

        # if PRNG key is not provided
        if key is None:
            key = random.PRNGKey(round(time.time() * 1000))
        
        # prepare initial subs
        subs = self._batched_init_subs()

        # initialize parameter fluents to optimize
        key, subkey = random.split(key)
        params, opt_state = self.init_fn(subkey)
        hyperparams = self.compiled.model_params

        # progress bar
        if print_progress:
            progress_bar = tqdm(
                None, total=100, bar_format='{l_bar}{bar}| {elapsed} {postfix}')
        else:
            progress_bar = None

        status = JaxLearnerStatus.NORMAL

        # main training loop
        for (it, (states, actions, next_states)) in enumerate(data):
            status = JaxLearnerStatus.NORMAL

            # gradient update
            subs.update(states)
            key, subkey = random.split(key)
            params, opt_state, loss, zero_grads, hyperparams = self.update_fn(
                subkey, params, subs, actions, next_states, hyperparams, opt_state)
            param_fluents = self.map_fn(params)
            param_fluents = {name: param_fluents[name] for name in self.param_ranges}

            # check for learnability
            params_zero_grads = {
                name for (name, zero_grad) in zero_grads.items() if zero_grad}
            if params_zero_grads:
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
                'param_fluents': param_fluents,
                'key': key,
                'progress': progress_percent
            }
            
            # update progress
            if print_progress:
                progress_bar.set_description(
                    f'{it:7} it / {loss:12.8f} train / {status.value} status', refresh=False)
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
    
    def evaluate_loss(self, data: Iterable[Tuple[State, Action, State]],
                      key: random.PRNGKey, 
                      param_fluents: Params) -> float:
        '''Evaluates the model loss of the given learned non-fluent values and the data.

        :param data: a data stream represented as a (possibly infinite) sequence of 
        transition batches of the form (states, actions, next-states), where each element
        is a numpy array of leading dimension equal to batch_size_train
        :param key: JAX PRNG key (derived from clock if not provided)
        :param param_fluents: the learned non-fluent values
        '''
        if key is None:
            key = random.PRNGKey(round(time.time() * 1000))
        subs = self._batched_init_subs()
        hyperparams = self.compiled.model_params
        mean_loss = 0.0
        loss_count = 0
        for (states, actions, next_states) in data:
            subs.update(states)
            key, subkey = random.split(key)
            loss_value, _ = self.loss_fn(
                subkey, param_fluents, subs, actions, next_states, hyperparams)
            loss_count += 1
            mean_loss += (loss_value - mean_loss) / loss_count
        return mean_loss

    def learned_model(self, param_fluents: Params) -> RDDLLiftedModel:
        '''Substitutes the given learned non-fluent values into the RDDL model and returns
        the new model.

        :param param_fluents: the learned non-fluent values
        '''
        model = deepcopy(self.rddl)
        for (name, values) in param_fluents.items():
            prange = model.variable_ranges[name]
            if prange == 'real':
                pass
            elif prange == 'bool':
                values = values > 0.5
            else:
                values = np.asarray(values, dtype=self.compiled.INT)
            values = np.ravel(values, order='C').tolist()
            model.non_fluents[name] = values
        return model


if __name__ == '__main__':
    import pyRDDLGym
    from pyRDDLGym.core.debug.decompiler import RDDLDecompiler
    bs = 32

    # make some data
    def data_iterator():
        env = pyRDDLGym.make('CartPole_Continuous_gym', '0', vectorized=True)
        model = JaxModelLearner(rddl=env.model, param_ranges={}, batch_size_train=bs)
        key = random.PRNGKey(round(time.time() * 1000))
        subs = model._batched_init_subs()
        epoch = 0
        param_fluents = {}
        while True:
            states = {k: np.asarray(subs[k]) for k in model.rddl.state_fluents}
            actions = {
                'force': np.random.uniform(-10., 10., (bs,))
            }
            key, subkey = random.split(key)
            subs, _ = model.step_fn(subkey, param_fluents, subs, actions, {})
            subs = {k: np.asarray(v)[0, ...] for k, v in subs.items()}
            next_states = {k: subs[k] for k in model.rddl.state_fluents}
            epoch += 1
            if epoch > env.horizon:
                subs = model._batched_init_subs()
                epoch = 0
            yield (states, actions, next_states)
    
    # train it
    env = pyRDDLGym.make('TestJax', '0', vectorized=True)
    model_learner = JaxModelLearner(rddl=env.model, 
                                    param_ranges={
                                        'w1': (None, None), 'b1': (None, None),
                                        'w2': (None, None), 'b2': (None, None),
                                        'w1o': (None, None), 'b1o': (None, None),
                                        'w2o': (None, None), 'b2o': (None, None)
                                    }, 
                                    batch_size_train=bs, 
                                    optimizer_kwargs = {'learning_rate': 0.0003})
    for cb in model_learner.optimize_generator(data_iterator(), epochs=10000):
        if cb['iteration'] > 0 and cb['iteration'] % 100 == 0:
            print(cb['train_loss'])
    learned = model_learner.learned_model(cb['param_fluents'])
    decompiler = RDDLDecompiler()
    print(decompiler.decompile_domain(learned))
    print(decompiler.decompile_instance(learned))
