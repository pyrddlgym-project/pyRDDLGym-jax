from collections import deque
from enum import Enum
import sys
import time
from tqdm import tqdm
from typing import Any, Callable, Dict, Generator, Iterable, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import optax
import termcolor

from pyRDDLGym.core.compiler.model import RDDLLiftedModel

from pyRDDLGym_jax.core.logic import Logic, ExactLogic
from pyRDDLGym_jax.core.planner import JaxRDDLCompilerWithGrad

Kwargs = Dict[str, Any]
State = Dict[str, np.ndarray]
Action = Dict[str, np.ndarray]
Callback = Dict[str, Any]


class JaxLearnerStatus(Enum):
    '''Represents the status of a parameter update from the JAX model learner, 
    including whether the update resulted in nan gradient, 
    whether progress was made, budget was reached, or other information that
    can be used to monitor and act based on the learner's progress.'''
    
    NORMAL = 0
    INVALID_GRADIENT = 4
    TIME_BUDGET_REACHED = 5
    ITER_BUDGET_REACHED = 6
    
    def is_terminal(self) -> bool:
        return self.value == 1 or self.value >= 4
    

class JaxModelLearner:
    '''A class for data-driven estimation of unknown parameters in a given RDDL MDP using 
    gradient descent.'''

    def __init__(self, rddl: RDDLLiftedModel, 
                 param_ranges: Dict[str, Tuple[Optional[np.ndarray], Optional[np.ndarray]]],
                 batch_size_train: int=32,
                 use64bit: bool=False,
                 optimizer: Callable[..., optax.GradientTransformation]=optax.rmsprop,
                 optimizer_kwargs: Optional[Kwargs]=None,
                 logic: Logic=ExactLogic()) -> None:
        '''Creates a new gradient-based algorithm for inferring unknown non-fluents
        in a RDDL domain from a data set or stream coming from the real environment.
        
        :param rddl: the RDDL domain to learn
        :param param_ranges: the ranges of all learnable non-fluents
        :param batch_size_train: how many transitions to compute per optimization 
        step
        :param use64bit: whether to perform arithmetic in 64 bi
        :param optimizer: a factory for an optax SGD algorithm
        :param optimizer_kwargs: a dictionary of parameters to pass to the SGD
        factory (e.g. which parameters are controllable externally)
        :param logic: a subclass of Logic for mapping exact mathematical
        operations to their differentiable counterparts 
        '''
        self.rddl = rddl
        self.param_ranges = param_ranges
        self.batch_size_train = batch_size_train
        self.use64bit = use64bit
        if optimizer_kwargs is None:
            optimizer_kwargs = {'learning_rate': 0.001}
        self.optimizer_kwargs = optimizer_kwargs
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

        def _jax_wrapped_step(key, param_fluents, subs, actions):
            for (name, param) in param_fluents.items():
                subs[name] = param
            return step_fn(key, actions, subs, {})

        # batched step function
        def _jax_wrapped_batched_step(key, param_fluents, subs, actions):
            keys = jnp.asarray(random.split(key, num=self.batch_size_train))
            _, log, _ = jax.vmap(_jax_wrapped_step, in_axes=(0, None, 0, 0))(
                keys, param_fluents, subs, actions)
            fluents = log['fluents']
            return fluents

        batched_step_fn = jax.jit(_jax_wrapped_batched_step)
        return batched_step_fn        

    def _jax_map(self):
        
        # map trainable parameters to their non-fluent values
        def _jax_wrapped_params_to_fluents(params):
            param_fluents = {}
            for (name, param) in params.items():
                if self.rddl.variable_ranges[name] == 'bool':
                    param_fluents[name] = jax.nn.sigmoid(param)
                else:
                    param_fluents[name] = param
            return param_fluents
        
        map_fn = jax.jit(_jax_wrapped_params_to_fluents)
        return map_fn

    def _jax_loss(self, map_fn, step_fn, EPS=1e-6):

        # use binary cross entropy for bool fluents
        # mean squared error for continuous and integer fluents
        def _jax_wrapped_batched_model_loss(key, param_fluents, subs, actions, next_fluents):
            fluents = step_fn(key, param_fluents, subs, actions)
            total_loss = 0.0
            for (name, next_value) in next_fluents.items():
                preds = jnp.asarray(fluents[name], dtype=self.compiled.REAL)
                targets = jnp.asarray(next_value, dtype=self.compiled.REAL)
                if self.rddl.variable_ranges[name] == 'bool':
                    preds = jnp.clip(preds, EPS, 1.0 - EPS)
                    log_preds = jnp.log(preds)
                    log_not_preds = jnp.log(1.0 - preds)
                    loss_values = -targets * log_preds - (1.0 - targets) * log_not_preds
                else:
                    loss_values = jnp.square(preds - targets)
                total_loss += jnp.mean(loss_values)
            return total_loss
        
        # loss with the parameters mapped to their fluents
        def _jax_wrapped_batched_loss(key, params, subs, actions, next_fluents):
            param_fluents = map_fn(params)
            loss = _jax_wrapped_batched_model_loss(
                key, param_fluents, subs, actions, next_fluents)
            return loss

        loss_fn = jax.jit(_jax_wrapped_batched_loss)
        return loss_fn
    
    def _jax_init(self, project_fn, SCALE=0.01):
        optimizer = self.optimizer
        
        # initialize both the non-fluents and optimizer
        def _jax_wrapped_init_params_optimizer(key):
            params = {}
            for name in self.param_ranges:
                shape = jnp.shape(self.compiled.init_values[name])
                key, subkey = random.split(key)
                params[name] = SCALE * random.normal(key=subkey, shape=shape)
            params = project_fn(params)
            opt_state = optimizer.init(params)
            return params, opt_state
        
        init_fn = jax.jit(_jax_wrapped_init_params_optimizer)
        return init_fn
                    
    def _jax_update(self, loss_fn):
        optimizer = self.optimizer

        # projected gradient trick to satisfy box constraints on params
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
        def _jax_wrapped_params_update(key, params, subs, actions, next_fluents, opt_state):
            loss_val, grad = jax.value_and_grad(loss_fn, argnums=1)(
                key, params, subs, actions, next_fluents)
            updates, opt_state = optimizer.update(grad, opt_state)
            params = optax.apply_updates(params, updates)
            params = _jax_wrapped_project_params(params)
            return params, opt_state, loss_val
    
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

    def estimate(self, *args, **kwargs) -> Optional[Callback]:
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
        it = self.estimate_generator(*args, **kwargs)
        
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

    def estimate_generator(self, data: Iterable[Tuple[State, Action, State]],
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
            params, opt_state, loss = self.update_fn(
                subkey, params, subs, actions, next_states, opt_state)
            param_fluents = self.map_fn(params)
            param_fluents = {name: param_fluents[name] for name in self.param_ranges}

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
                'iteration': it,
                'train_loss': loss,
                'param_fluents': param_fluents,
                'key': key,
                'progress': progress_percent
            }
            
            # update progress
            if print_progress:
                progress_bar.set_description(
                    f'{it:7} it / {loss:12.8f} train', refresh=False)
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
            

if __name__ == '__main__':
    import pyRDDLGym
    env = pyRDDLGym.make('RaceCar_ippc2023', '1', vectorized=True)
    model = JaxModelLearner(rddl=env.model, 
                            param_ranges={
                                'MASS': (0.001, None), 
                                'DT': (0.001, None), 
                                'X0': (None, None)}, 
                            batch_size_train=32, 
                            optimizer_kwargs = {'learning_rate': 0.003})

    # make some data
    def data_iterator():
        key = random.PRNGKey(42)
        subs = model._batched_init_subs()
        param_fluents = {'MASS': np.array(3.0), 'DT': np.array(0.3), 'X0': np.array(0.1)}
        while True:
            states = {'x': np.random.uniform(-1., 1., (32,)),
                      'y': np.random.uniform(-1., 1., (32,)),
                      'vx': np.random.uniform(-1., 1., (32,)),
                      'vy': np.random.uniform(-1., 1., (32,))}
            actions = {'fx': np.random.uniform(-1., 1., (32,)),
                       'fy': np.random.uniform(-1., 1., (32,))}
            key, subkey = random.split(key)
            subs.update(states)
            for (state, next_state) in model.rddl.next_state.items():
                subs[next_state] = subs[state] 
            next_states = model.step_fn(subkey, param_fluents, subs, actions)
            yield (states, actions, next_states)
    
    # train it
    for cb in model.estimate_generator(data_iterator(), epochs=2000):
        print(cb['param_fluents'])
