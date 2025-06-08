from collections import deque
import sys
import time
from tqdm import tqdm
from typing import Any, Callable, Dict, Generator, Iterable, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import optax

from pyRDDLGym.core.compiler.model import RDDLLiftedModel

from pyRDDLGym_jax.core.compiler import JaxRDDLCompiler

Kwargs = Dict[str, Any]
State = Dict[str, np.ndarray]
Action = Dict[str, np.ndarray]
Callback = Dict[str, Any]


class JaxModelLearner:
    '''A class for data-driven estimation of unknown parameters in a given RDDL MDP using 
    gradient descent.'''

    def __init__(self, rddl: RDDLLiftedModel, 
                 params_list: Iterable[str],
                 batch_size_train: int=32,
                 optimizer: Callable[..., optax.GradientTransformation]=optax.rmsprop,
                 optimizer_kwargs: Optional[Kwargs]=None) -> None:
        '''Creates a new gradient-based algorithm for inferring unknown non-fluents
        in a RDDL domain from a data set or stream coming from the real environment.
        
        :param rddl: the RDDL domain to learn
        :param params_list: the list of non-fluents that are set to learnable (unknown)
        :param batch_size_train: how many transitions to compute per optimization 
        step
        :param optimizer: a factory for an optax SGD algorithm
        :param optimizer_kwargs: a dictionary of parameters to pass to the SGD
        factory (e.g. which parameters are controllable externally)
        '''
        self.rddl = rddl
        self.params_list = list(params_list)
        self.batch_size_train = batch_size_train
        if optimizer_kwargs is None:
            optimizer_kwargs = {'learning_rate': 0.001}
        self.optimizer_kwargs = optimizer_kwargs

        # build the optimizer
        optimizer = optimizer(**optimizer_kwargs)
        pipeline = [optimizer]
        self.optimizer = optax.chain(*pipeline)

        # build the computation graph
        self.step_fn = self._jax_compile_rddl()
        self.loss_fn = self._jax_loss(step_fn=self.step_fn)
        self.update_fn = self._jax_update(loss_fn=self.loss_fn)
        self.init_fn = self._jax_init()
    
    # ===========================================================================
    # COMPILATION SUBROUTINES
    # ===========================================================================

    def _jax_compile_rddl(self):

        # compile the RDDL model
        self.compiled = JaxRDDLCompiler(rddl=self.rddl)
        self.compiled.compile(log_jax_expr=True, heading='EXACT MODEL')

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

    def _jax_loss(self, step_fn, EPS=1e-6):

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

        loss_fn = jax.jit(_jax_wrapped_batched_model_loss)
        return loss_fn
    
    def _jax_init(self):
        optimizer = self.optimizer
        init_values = self.compiled.init_values
        
        # initialize both the non-fluents and its optimizer
        def _jax_wrapped_init_param_fluents_optimizer():
            param_fluents = {name: init_values[name] for name in self.params_list}
            opt_state = optimizer.init(param_fluents)
            return param_fluents, opt_state
        
        init_fn = jax.jit(_jax_wrapped_init_param_fluents_optimizer)
        return init_fn
                    
    def _jax_update(self, loss_fn):
        optimizer = self.optimizer

        def _jax_wrapped_param_fluents_update(key, param_fluents, subs, actions, 
                                              next_fluents, opt_state):
            loss_val, grad = jax.value_and_grad(loss_fn, argnums=1, allow_int=True)(
                key, param_fluents, subs, actions, next_fluents)
            updates, opt_state = optimizer.update(grad, opt_state)
            param_fluents = optax.apply_updates(param_fluents, updates)
            return param_fluents, opt_state, loss_val
    
        update_fn = jax.jit(_jax_wrapped_param_fluents_update)
        return update_fn
    
    def _batched_init_subs(self): 
        init_train = {}
        for (name, value) in self.compiled.init_values.items():
            value = np.reshape(value, np.shape(value))[np.newaxis, ...]
            value = np.repeat(value, repeats=self.batch_size_train, axis=0)
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
        param_fluents, opt_state = self.init_fn()

        # progress bar
        if print_progress:
            progress_bar = tqdm(
                None, total=100, bar_format='{l_bar}{bar}| {elapsed} {postfix}')
        else:
            progress_bar = None

        # main training loop
        for (it, (states, actions, next_states)) in enumerate(data):
            subs.update(states)
            key, subkey = random.split(key)
            param_fluents, opt_state, loss = self.update_fn(
                subkey, param_fluents, subs, actions, next_states, opt_state)

            # reached computation budget
            elapsed = time.time() - start_time - elapsed_outside_loop
            
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
                    f'{it:7} it / {loss:10.6f} train', refresh=False)
                progress_bar.set_postfix_str(
                    f'{(it + 1) / (elapsed + 1e-6):.2f}it/s', refresh=False)
                progress_bar.update(progress_percent - progress_bar.n)
            
            # yield the callback
            start_time_outside = time.time()
            yield callback
            elapsed_outside_loop += (time.time() - start_time_outside)

            # abortion check
            if elapsed >= train_seconds or it >= epochs:
                break
            

if __name__ == '__main__':
    import pyRDDLGym
    env = pyRDDLGym.make('RaceCar_ippc2023', '1', vectorized=True)
    model = JaxModelLearner(rddl=env.model, params_list=['MASS', 'DT'], 
                            batch_size_train=32, optimizer_kwargs = {'learning_rate': 0.003})

    # make some data
    def data_iterator():
        key = random.PRNGKey(42)
        subs = model._batched_init_subs()
        param_fluents = {'MASS': np.array(3.0), 'DT': np.array(0.3)}
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
    for cb in model.estimate_generator(data_iterator(), epochs=1000):
        print(cb['param_fluents'])
