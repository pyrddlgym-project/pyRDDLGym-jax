from collections import deque
import hashlib
import math
import numpy as np
from tqdm import tqdm
from typing import Any, Dict, Generator, Optional, Tuple
import sys
import time

import jax
import jax.random as random
import jax.numpy as jnp

import pyRDDLGym
from pyRDDLGym.core.compiler.model import RDDLLiftedModel
from pyRDDLGym.core.policy import BaseAgent

from pyRDDLGym_jax.core.compiler import JaxRDDLCompiler
from pyRDDLGym_jax.core.logic import Logic, FuzzyLogic
from pyRDDLGym_jax.core.planner import JaxStraightLinePlan, JaxRDDLCompilerWithGrad

ActionType = Dict[str, Any]
StateType = Dict[str, Any]
Fluents = Dict[str, Any]
Callback = Dict[str, Any]


class HashableDict:
    '''A wrapper for a dict with hash support to be used as a key.'''

    def __init__(self, value: Fluents) -> None:
        self.value = value
    
    def __hash__(self) -> int:
        combined_hash = 0 
        for (key, value) in self.value.items(): 
            value_hash = hashlib.sha256(value.tobytes()).hexdigest()
            combined_hash ^= hash(key) ^ int(value_hash, 16)
        return combined_hash
    
    def __eq__(self, other: Any) -> bool:
        if isinstance(other, HashableDict):
            for (key, arr) in self.value.items():
                if not np.array_equal(arr, np.asarray(other.value[key])):
                    return False
            return True
        return False


class ChanceNode:
    '''A chance node represents an action/decision in the search tree.'''
    
    def __init__(self, action: Optional[HashableDict]=None) -> None:
        self.action = action
        self.init_action = action.value
        self.children = {}
    
    def least_visited_node(self) -> 'DecisionNode':
        least, leastvisit = None, math.inf
        for vD in self.children.values():
            if vD.N < leastvisit:
                least, leastvisit = vD, vD.N
        return least
    

class DecisionNode:
    '''A decision node represents a transition following some decision made.'''
    
    def __init__(self, xi: Optional[random.PRNGKey]=None, 
                 tr: Optional[Tuple[StateType, float, bool]]=None) -> None:
        self.xi = xi
        self.tr = tr
        self.N = 0
        self.children = {}
        self.n_chance_subnodes = 0
        self.n_decision_subnodes = 0
    
    def uct(self, c: float) -> 'ChanceNode':
        logN = math.log(self.N)
        best, bestscore = None, -math.inf
        for (vC, Nc, Qc) in self.children.values():
            if c > 0:
                score = Qc + c * math.sqrt(2 * logN / Nc)
            else:
                score = Qc
            if score > bestscore:
                best, bestscore = vC, score
        return best
    
    def best(self) -> 'ChanceNode':
        return self.uct(0)
    
    def update_statistic(self, vC: 'ChanceNode', R: float) -> None:
        self.N += 1
        vC, Nc, Qc = self.children[vC.action]
        Nc = Nc + 1
        Qc = Qc + (R - Qc) / Nc
        self.children[vC.action] = (vC, Nc, Qc)


class JaxMCTSPlanner:
    '''Represents a hybrid approach between UCT-MCTS and planning by backprop.'''
    
    def __init__(self, rddl: RDDLLiftedModel,
                 batch_size_rollout: int=8,
                 rollout_horizon: Optional[int]=None,
                 alpha_start: float=0.9,
                 alpha_end: float=0.1,
                 beta_start: float=0.9,
                 beta_end: float=0.1,
                 c: float=math.sqrt(2.0),
                 grad_updates: bool=False,
                 adapt_c: bool=True,
                 learning_rate: float=0.001,
                 delta: float=0.1,
                 logic: Logic=FuzzyLogic()) -> None:
        '''Creates a new hybrid backprop + UCT-MCTS planner.
        
        :param rddl: the rddl domain to optimize
        :param batch_size_rollout: batch size for rollout generation
        :param rollout_horizon: how many steps to plan ahead
        :param alpha: the growth factor for chance nodes
        :param beta: the growth factor for decision nodes
        :param c: initial scaling factor for UCT
        :param grad_updates: whether to do gradient-based action refinement
        :param adapt_c: whether to adapt c to the scale of the reward
        :param learning_rate: learning rate for gradient-based action refinement
        :param delta: budget for gradient-based action refinement
        :param logic: a subclass of Logic for mapping exact mathematical
        operations to their differentiable counterparts 
        '''
        self.rddl = rddl
        self.batch_size_rollout = batch_size_rollout
        if rollout_horizon is None:
            rollout_horizon = rddl.horizon
        self.T = rollout_horizon
        self.alpha_start = alpha_start
        self.alpha_decay = (alpha_end / alpha_start) ** 0.01
        self.beta_start = beta_start
        self.beta_decay = (beta_end / beta_start) ** 0.01
        self.c = c
        self.adapt_c = adapt_c
        self.grad_updates = grad_updates
        self.lr = learning_rate
        self.delta = delta
        self.logic = logic

        self._jax_compile_rddl()
                    
    # ===========================================================================
    # JAX COMPILATIONS
    # ===========================================================================

    def _jax_compile_rddl(self):

        # compile exact RDDL model
        self.compiled = JaxRDDLCompiler(rddl)
        self.compiled.compile()
        self.noop_actions = {var: values 
                             for (var, values) in self.compiled.init_values.items() 
                             if rddl.variable_types[var] == 'action-fluent'}
        self.action_bounds = self.compiled.constraints.bounds

        # compile rollout functions
        self.policy_fn = self._jax_compile_policy()
        self.single_step_fn = jax.jit(
            self.compiled.compile_transition(check_constraints=True))
        self.rollout_fn = self._jax_compile_exact_rollout()

        # compile gradient planner
        if self.grad_updates:
            self.update_slp_fn = self._jax_compile_gradient_update()
        else:
            self.update_slp_fn = None

    def _jax_compile_policy(self):
        rddl = self.rddl
        num_actions = len(rddl.action_fluents)
        action_keys = list(rddl.action_fluents.keys())
        
        def _jax_wrapped_sample_num_modified(key):
            result = jnp.zeros(shape=(num_actions,), dtype=jnp.int32)
            budget = rddl.max_allowed_actions
            keys = random.split(key, num=1 + num_actions)
            for i in range(num_actions):
                num_modified = random.randint(
                    keys[i + 1], shape=(), minval=0, maxval=budget + 1)
                result = result.at[i].set(num_modified)
                budget = budget - num_modified
            result = random.permutation(keys[0], result)
            return dict(zip(action_keys, result))

        def _jax_wrapped_sample_action(key, policy_params, hyperparams, step, subs):
            actions = {}
            if 0 < rddl.max_allowed_actions < jnp.inf:
                num_modified_dict = _jax_wrapped_sample_num_modified(key)
                for action in action_keys:
                    prange = rddl.action_ranges[action]
                    noop_action = jnp.ravel(self.noop_actions[action])
                    num_actions = noop_action.size
                    key, subkey1, subkey2 = random.split(key, num=3)
                    permuted_index = random.permutation(subkey1, jnp.arange(num_actions))
                    if prange == 'bool':
                        new_values = random.uniform(subkey2, shape=(num_actions,)) < 0.5
                    elif prange == 'int':
                        lower, upper = self.action_bounds[action]
                        lower, upper = jnp.ravel(lower), jnp.ravel(upper)
                        new_values = random.randint(
                            subkey2, shape=(num_actions,), minval=lower, maxval=upper + 1)
                    else:
                        lower, upper = self.action_bounds[action]
                        lower, upper = jnp.ravel(lower), jnp.ravel(upper)
                        new_values = random.uniform(
                            subkey2, shape=(num_actions,), minval=lower, maxval=upper)
                    action_values = jnp.where(
                        permuted_index < num_modified_dict[action], new_values, noop_action)
                    actions[action] = jnp.reshape(
                        action_values, jnp.shape(self.noop_actions[action]))
            else:
                for action in action_keys:
                    prange = rddl.action_ranges[action]
                    ptypes = rddl.variable_params[action]
                    shape = rddl.object_counts(ptypes) if ptypes else ()
                    key, subkey = random.split(key)
                    if prange == 'bool':
                        action_value = random.uniform(subkey, shape=shape) < 0.5
                    else:
                        lower, upper = self.action_bounds[action]
                        if prange == 'int':
                            action_value = random.randint(
                                subkey, shape=shape, minval=lower, maxval=upper + 1)
                        else:
                            action_value = random.uniform(
                                subkey, shape=shape, minval=lower, maxval=upper)
                    actions[action] = action_value
            return actions        
        
        return jax.jit(_jax_wrapped_sample_action)
    
    def _jax_compile_exact_rollout(self):
        rddl = self.rddl
        gamma = rddl.discount

        # compile the exact rollout function
        rollouts = self.compiled.compile_rollouts(
            policy=self.policy_fn, 
            n_steps=self.T, 
            n_batch=self.batch_size_rollout, 
            check_constraints=True
        )   

        # compile the return estimation function
        def _jax_wrapped_batched_subs(subs):
            result = {}
            for (name, value) in subs.items():
                init_value = self.compiled.init_values[name]
                value = jnp.reshape(value, jnp.shape(init_value))[jnp.newaxis, ...]
                result[name] = jnp.repeat(value, repeats=self.batch_size_rollout, axis=0)
            for (state, next_state) in rddl.next_state.items():
                result[next_state] = result[state]
            return result
        
        def _jax_wrapped_return(subs, t, key):
            key, subkey = random.split(key)
            subs = _jax_wrapped_batched_subs(subs)            
            log, _ = rollouts(subkey, None, None, subs, None)
            rewards = log['reward'] * (1 - log['termination'])
            if gamma != 1:
                horizon = rewards.shape[1]
                discount = jnp.power(gamma, jnp.arange(horizon))
                rewards = rewards * discount[jnp.newaxis, ...]
            mean_return = jnp.mean(jnp.sum(rewards, axis=1))
            actions = {name: log['fluents'][name][0, ...] 
                       for name in rddl.action_fluents}
            length = self.T
            return mean_return, key, actions, length

        return jax.jit(_jax_wrapped_return)    

    def _jax_compile_gradient_update(self):
        rddl = self.rddl
        gamma = rddl.discount
        
        # compile the model approximation
        compiled_with_grad = JaxRDDLCompilerWithGrad(rddl=rddl, logic=self.logic)
        compiled_with_grad.compile()
        model_params = compiled_with_grad.model_params

        # compile the straight line plan
        slp = JaxStraightLinePlan(wrap_sigmoid=False)
        slp.compile(compiled_with_grad, _bounds={}, horizon=self.T)
        projection = slp.projection
        test_policy = slp.test_policy

        # compile the rollout function
        rollouts = compiled_with_grad.compile_rollouts(
            policy=slp.train_policy, 
            n_steps=self.T, 
            n_batch=1
        )

        # compile the gradient update function
        def _jax_wrapped_plan_loss(key, policy_params, subs):
            log, _ = rollouts(key, policy_params, None, subs, model_params)
            rewards = log['reward']
            if gamma != 1:
                horizon = rewards.shape[1]
                discount = jnp.power(gamma, jnp.arange(horizon))
                rewards = rewards * discount[jnp.newaxis, ...]
            return jnp.mean(jnp.sum(rewards, axis=1))
        
        def _jax_wrapped_plan_clip(test_actions, init_actions):
            new_actions = {}
            for (name, action) in test_actions.items():
                if rddl.variable_ranges[name] == 'real':
                    lower = init_actions[name] - self.delta
                    upper = init_actions[name] + self.delta
                    new_actions[name] = jnp.clip(action, lower, upper)
                else:
                    new_actions[name] = action
            return new_actions

        def _jax_wrapped_check_action_valid(test_actions):
            if rddl.max_allowed_actions < jnp.inf:
                count_non_equal = 0
                for (name, default) in self.noop_actions.items():
                    count_non_equal += jnp.sum(jnp.not_equal(default, test_actions[name]))
                return count_non_equal <= rddl.max_allowed_actions
            else:
                return True
            
        def _jax_wrapped_plan_update(key, policy_params, subs, init_actions):
            policy_params = {name: param.astype(compiled_with_grad.REAL)
                             for (name, param) in policy_params.items()}
            subs = {name: value[jnp.newaxis, ...].astype(compiled_with_grad.REAL) 
                    for (name, value) in subs.items()}
            grads = jax.grad(_jax_wrapped_plan_loss, argnums=1)(key, policy_params, subs)
            policy_params = {name: param + self.lr * grads[name]
                             for (name, param) in policy_params.items()}
            policy_params, _ = projection(policy_params, None)
            test_actions = test_policy(key, policy_params, None, 0, subs)
            test_actions = _jax_wrapped_plan_clip(test_actions, init_actions)
            valid_action = _jax_wrapped_check_action_valid(test_actions)
            return test_actions, valid_action        

        return jax.jit(_jax_wrapped_plan_update)
    
    # ===========================================================================
    # MCTS SUBROUTINES
    # ===========================================================================

    def _select_action(self, subs, vD, c, key, alpha):
        if int(vD.N ** alpha) >= len(vD.children):
            key, subkey = random.split(key)
            a = HashableDict(self.policy_fn(subkey, None, None, None, subs))
            child = vD.children.get(a, None)
            if child is None:
                vC = ChanceNode(action=a)
                vD.children[a] = (vC, 0, 0.0)
                rollout = True
            else:
                vC, _, _ = child
                rollout = False
        else:
            vC = vD.uct(c)
            rollout = False
        return vC, rollout, key

    def _select_state(self, vD, vC, subs, key, beta):
        _, Nc, _ = vD.children[vC.action]
        if int(Nc ** beta) >= len(vC.children):
            key, subkey = random.split(key)
            subs, log, _ = self.single_step_fn(subkey, vC.action.value, subs, None)
            s = HashableDict(subs)
            r = log['reward'].item()
            done = log['termination'].item()
            child = vC.children.get(s, None)
            if child is None:                
                vD2 = DecisionNode(xi=subkey, tr=(subs, r, done))
                vC.children[s] = vD2
                added = True
            else:
                vD2 = child
                added = False
        else:
            vD2 = vC.least_visited_node()
            added = False
        return vD2, added, key

    def _grad_update(self, vD, vC, actions, length, subs, key):
        plan = {name: action[np.newaxis, ...] 
                for (name, action) in vC.action.value.items()}
        length = min(length + 1, self.T)
        update = 0
        if self.grad_updates and actions is not None:
            plan = {name: np.concatenate([action, actions[name]], axis=0)[:self.T, ...]
                    for (name, action) in plan.items()}            
            if length >= self.T:
                new_action, valid = self.update_slp_fn(key, plan, subs, vC.init_action)
                if valid:
                    new_action = HashableDict(new_action)
                    vD.children[new_action] = vD.children.pop(vC.action)
                    vC.action = new_action
                    for (name, action) in plan.items():
                        action[0] = new_action.value[name]
                    update = 1
        return plan, length, update
    
    def _simulate(self, subs, vD, t, c, key, alpha, beta):

        # update the MCTS tree normally
        if t == self.T:
            return 0.0, key, None, 0, 0, 0, 0, 0
        vC, rollout, key = self._select_action(subs, vD, c, key, alpha)
        vD2, added, key = self._select_state(vD, vC, subs, key, beta)
        subs0 = subs
        subs, r, done = vD2.tr
        total_depth = 1
        total_d_subnodes = int(added)
        total_c_subnodes = int(rollout)
        total_grad_updates = 0
        if done:
            R2, actions, length = 0.0, None, 0
        elif rollout:
            R2, key, actions, length = self.rollout_fn(subs, t + 1, key)
        else:
            R2, key, actions, length, depth, d_subnodes, c_subnodes, grad_updates = \
                self._simulate(subs, vD2, t + 1, c, key, alpha, beta)
            total_depth += depth
            total_d_subnodes += d_subnodes
            total_c_subnodes += c_subnodes
            total_grad_updates += grad_updates
        R = r + self.rddl.discount * R2
        vD.update_statistic(vC, R)
        vD.n_decision_subnodes += total_d_subnodes
        vD.n_chance_subnodes += total_c_subnodes
       
        # perform an optional gradient update on the immediate action
        actions, length, updates = self._grad_update(vD, vC, actions, length, subs0, key)
        total_grad_updates += updates

        return R, key, actions, length, \
            total_depth, total_d_subnodes, total_c_subnodes, total_grad_updates
    
    def _state_to_subs(self, subs):
        rddl = self.rddl
        if subs is None:
            subs = {}
        else:
            subs = subs.copy()
        for (var, value) in self.compiled.init_values.items():
            if var not in subs:
                subs[var] = value
            shape = rddl.object_counts(rddl.variable_params[var])
            subs[var] = np.reshape(subs[var], shape)
        return subs

    def optimize_generator(self, key: Optional[random.PRNGKey]=None,  
                           epochs: int=999999,
                           train_seconds: float=1.0,
                           subs: Optional[Fluents]=None,
                           c_guess: Optional[float]=None,
                           root: Optional[DecisionNode]=None) -> Generator[Callback, None, None]:
        '''Performs a search using the current planner, and returns a generator.
        
        :param key: jax RNG key
        :param epochs: how many iterations of MCTS to perform
        :param train_seconds: how many seconds to limit training
        :param subs: initial pvariables (e.g. state, non-fluents) to begin from
        :param c_guess: initialize the c parameter to this value
        :param root: initialize the root of the tree to this value
        '''
        # if PRNG key is not provided
        if key is None:
            key = random.PRNGKey(round(time.time() * 1000))
        
        # initialization of MCTS tree
        subs = self._state_to_subs(subs)
        vD = DecisionNode() if root is None else root
        c = self.c if c_guess is None else c_guess

        # initialize statistics
        avg_depth = 0
        total_grad_updates = 0
        start_time = time.time()
        elapsed_outside_loop = 0
        progress_percent = 0

        # main optimization loop
        iters = tqdm(range(epochs), total=100)
        for it in iters:

            # update MCTS tree
            alpha = self.alpha_start * self.alpha_decay ** progress_percent
            beta = self.beta_start * self.beta_decay ** progress_percent
            R, key, _, _, depth, _, _, grad_updates = self._simulate(
                subs, vD, 0, c, key, alpha, beta)
            if self.adapt_c:
                c += (abs(R) - c) / (it + 1)
            
            # update statistics
            avg_depth += (depth - avg_depth) / (it + 1)
            total_grad_updates += grad_updates

            # update progress bar
            elapsed = time.time() - start_time - elapsed_outside_loop
            progress_percent = 100 * min(1, max(elapsed / train_seconds, it / epochs))
            iters.set_description(
                f'{it:6} it / {R:14.4f} reward / {c:14.4f} c / '
                f'{len(vD.children):3} width / {avg_depth:6.2f} depth / '
                f'{vD.n_decision_subnodes:6} nD / {vD.n_chance_subnodes:6} nC / '
                f'{total_grad_updates:6} grad'
            )
            iters.n = progress_percent

            # callback
            vC = vD.best()
            callback = {
                'return': R,
                'action': vC.action.value,
                'best': vC,
                'root': vD,
                'iter': it,
                'c': c
            }
            start_time_outside = time.time()
            yield callback
            elapsed_outside_loop += (time.time() - start_time_outside)

            # breaks
            if it >= epochs - 1: 
                break
            if elapsed >= train_seconds:
                break
            
    def optimize(self, *args, **kwargs) -> Callback: 
        '''Performs a search using the current planner.
        
        :param key: jax RNG key
        :param epochs: how many iterations of MCTS to perform
        :param train_seconds: how many seconds to limit training
        :param subs: initial pvariables (e.g. state, non-fluents) to begin from
        :param c_guess: initialize the c parameter to this value
        :param root: initialize the root of the tree to this value
        '''
        it = self.optimize_generator(*args, **kwargs)
        callback = None
        if sys.implementation.name == 'cpython':
            last_callback = deque(it, maxlen=1)
            if last_callback:
                callback = last_callback.pop()
        else:
            for callback in it:
                pass
        return callback
    
    def next_epoch_root(self, vC: ChanceNode, state: StateType) -> Optional[DecisionNode]:
        rddl = self.rddl
        state = self._state_to_subs(state)
        for (key, vD2) in vC.children.items():
            equal = True
            for (name, value) in state.items():
                if rddl.variable_types[name] in ('state-fluent', 'observ-fluent') and \
                not np.allclose(value, key.value[name]):
                    equal = False
                    break
            if equal:
                return vD2
        return None


class JaxMCTSController(BaseAgent):
    '''A container class for a Jax MCTS controller continuously updated using state 
    feedback.'''
    
    use_tensor_obs = True
    
    def __init__(self, planner: JaxMCTSPlanner,
                 key: Optional[random.PRNGKey]=None,
                 warm_start: bool=True,
                 **train_kwargs) -> None:
        '''Creates a new JAX MCTS control policy that is trained online in a closed-
        loop fashion.
        
        :param planner: underlying MCTS algorithm for optimizing actions
        :param key: the RNG key to seed randomness (derives from clock if not
        provided)
        :param warm_start: whether to use the previous decision epoch search tree
        to warm the next decision epoch
        :param **train_kwargs: any keyword arguments to be passed to the planner
        for optimization
        '''
        self.planner = planner
        if key is None:
            key = random.PRNGKey(round(time.time() * 1000))
        self.key = key
        self.warm_start = warm_start
        self.train_kwargs = train_kwargs
        self.reset()
     
    def sample_action(self, state: StateType) -> ActionType:
        planner = self.planner
        self.key, subkey = random.split(self.key)
        if self.callback is None: 
            c_guess = None
            root = None
        else:
            c_guess = self.callback['c']
            if self.warm_start:
                root = self.planner.next_epoch_root(self.callback['best'], state)
                if root is not None:
                    print(f'Warm starting with search sub-tree containing '
                          f'{root.n_decision_subnodes} decision nodes '
                          f'and {root.n_chance_subnodes} chance nodes.')
            else:
                root = None
        self.callback = planner.optimize(
            key=subkey,
            subs=state,
            c_guess=c_guess,
            root=root,
            **self.train_kwargs
        )
        return self.callback['action']
        
    def reset(self) -> None:
        self.callback = None


if __name__ == '__main__':
    env = pyRDDLGym.make('Elevators', '0', vectorized=True)
    rddl = env.model
    rddl.horizon = 120
    
    world = env
    agent = JaxMCTSController(
        JaxMCTSPlanner(
            rddl, 
            rollout_horizon=20, 
            delta=0.1,
            learning_rate=0.1,
            grad_updates=True),
        train_seconds=2
    )
    
    agent.evaluate(env, episodes=1, verbose=True, render=True)