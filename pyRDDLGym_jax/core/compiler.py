# ***********************************************************************
# JAXPLAN
# 
# Author: Michael Gimelfarb
#
# REFERENCES:
# 
# [1] Gimelfarb, Michael, Ayal Taitler, and Scott Sanner. "JaxPlan and GurobiPlan: 
# Optimization Baselines for Replanning in Discrete and Mixed Discrete-Continuous 
# Probabilistic Domains." Proceedings of the International Conference on Automated 
# Planning and Scheduling. Vol. 34. 2024.
#
# ***********************************************************************


from functools import partial
import termcolor
import traceback
from typing import Any, Callable, Dict, List, Optional

import jax
import jax.numpy as jnp
import jax.random as random
import jax.scipy as scipy 

from pyRDDLGym.core.compiler.initializer import RDDLValueInitializer
from pyRDDLGym.core.compiler.levels import RDDLLevelAnalysis
from pyRDDLGym.core.compiler.model import RDDLLiftedModel
from pyRDDLGym.core.compiler.tracer import RDDLObjectsTracer
from pyRDDLGym.core.constraints import RDDLConstraints
from pyRDDLGym.core.debug.exception import (
    print_stack_trace, 
    raise_warning,
    RDDLInvalidNumberOfArgumentsError, 
    RDDLNotImplementedError,
    RDDLUndefinedVariableError
)
from pyRDDLGym.core.debug.logger import Logger
from pyRDDLGym.core.simulator import RDDLSimulatorPrecompiled

# more robust approach - if user does not have this or broken try to continue
try:
    from tensorflow_probability.substrates import jax as tfp
except Exception:
    raise_warning('Failed to import tensorflow-probability: '
                  'compilation of some probability distributions will fail.', 'red')
    traceback.print_exc()
    tfp = None


class JaxRDDLCompiler:
    '''Compiles a RDDL AST representation into an equivalent JAX representation.
    All operations are identical to their numpy equivalents.
    '''
    
    def __init__(self, rddl: RDDLLiftedModel, *args, 
                 allow_synchronous_state: bool=True,
                 logger: Optional[Logger]=None,
                 use64bit: bool=False,
                 stochastic_is_fluent: bool=False,
                 python_functions: Optional[Dict[str, Callable]]=None, **kwargs) -> None:
        '''Creates a new RDDL to Jax compiler.
        
        :param rddl: the RDDL model to compile into Jax
        :param allow_synchronous_state: whether next-state components can depend
        on each other
        :param logger: to log information about compilation to file
        :param use64bit: whether to use 64 bit arithmetic
        :param stochastic_is_fluent: whether stochastic traced nodes are considered fluent
        even when their arguments are all non-fluent
        :param python_functions: dictionary of external Python functions to call from RDDL
        '''

        # warn about unused parameters
        if args:
            print(termcolor.colored(
                f'[WARN] JaxRDDLCompiler received invalid args {args}.',  'yellow'))
        if kwargs:
            print(termcolor.colored(
                f'[WARN] JaxRDDLCompiler received invalid kwargs {kwargs}.',  'yellow'))

        self.rddl = rddl
        self.logger = logger
        # jax.config.update('jax_log_compiles', True) # for testing ONLY
        
        self.use64bit = use64bit
        if use64bit:
            self.INT = jnp.int64
            self.REAL = jnp.float64
        else:
            self.INT = jnp.int32
            self.REAL = jnp.float32
        jax.config.update('jax_enable_x64', use64bit)
        self.ONE = jnp.array(1, dtype=self.INT)
        self.JAX_TYPES = {
            'int': self.INT,
            'real': self.REAL,
            'bool': jnp.bool_
        }

        # compile initial values
        initializer = RDDLValueInitializer(rddl)
        self.init_values = initializer.initialize()
        
        # compute dependency graph for CPFs and sort them by evaluation order
        self.allow_synchronous_state = allow_synchronous_state
        sorter = RDDLLevelAnalysis(rddl, allow_synchronous_state=allow_synchronous_state)
        self.levels = sorter.compute_levels()  
        self.policy_deps = sorter.build_policy_call_graph(include_non_fluents=True)      
        
        # trace expressions to cache information to be used later
        tracer = RDDLObjectsTracer(
            rddl, cpf_levels=self.levels, stochastic_is_fluent=stochastic_is_fluent)
        self.traced = tracer.trace()
        
        # external python functions
        if python_functions is None:
            python_functions = {}
        self.python_functions = python_functions

        # extract the box constraints on actions
        simulator = RDDLSimulatorPrecompiled(
            rddl=self.rddl,
            init_values=self.init_values,
            levels=self.levels,
            trace_info=self.traced,
            python_functions=python_functions
        )  
        self.constraints = RDDLConstraints(simulator, vectorized=True)
    
    def get_kwargs(self) -> Dict[str, Any]:
        '''Returns a dictionary of configurable parameter name: parameter value pairs.
        '''
        return {
            'allow_synchronous_state': self.allow_synchronous_state,
            'use64bit': self.use64bit,
            'python_functions': self.python_functions
        }

    def split_fluent_nonfluent(self, values):
        '''Splits given values dictionary into fluent and non-fluent dictionaries.
        '''
        nonfluents = self.rddl.non_fluents
        fls = {name: value for (name, value) in values.items() if name not in nonfluents}
        nfls = {name: value for (name, value) in values.items() if name in nonfluents}
        return fls, nfls
        
    # ===========================================================================
    # main compilation subroutines
    # ===========================================================================
     
    def compile(self, log_jax_expr: bool=False, 
                heading: str='',
                extra_aux: Dict[str, Any]={},
                compile_constraints: bool=True,
                compile_policy_block: bool=True) -> None: 
        '''Compiles the current RDDL into Jax expressions.
        
        :param log_jax_expr: whether to pretty-print the compiled Jax functions
        to the log file
        :param heading: the heading to print before compilation information
        :param extra_aux: extra info to save during compilations
        :param compile_constraints: whether to compile constraints
        :param compile_policy_block: whether to compile the policy block
        '''
        self.model_aux = {'params': {}, 'overriden': {}, 'exact': set()}
        self.model_aux.update(extra_aux)
        
        if compile_constraints:
            self.invariants = self._compile_constraints(self.rddl.invariants, self.model_aux)
            self.preconditions = self._compile_constraints(self.rddl.preconditions, self.model_aux)
            self.terminations = self._compile_constraints(self.rddl.terminations, self.model_aux)
        else:
            self.invariants, self.preconditions, self.terminations = [], [], []
        
        self.cpfs = self._compile_cpfs(self.model_aux)
        self.reward = self._compile_reward(self.model_aux)

        if compile_policy_block:
            self.policy = self._compile_policy_block(self.model_aux)
        else:
            self.policy = {}

        # add compiled jax expression to logger
        if log_jax_expr and self.logger is not None:
            self._log_printed_jax(heading)
    
    def _log_printed_jax(self, heading=''):        
        printed = self.print_jax()
        printed_cpfs = '\n\n'.join(f'{k}: {v}' for (k, v) in printed['cpfs'].items())
        printed_reward = printed['reward']
        printed_invariants = '\n\n'.join(v for v in printed['invariants'])
        printed_preconds = '\n\n'.join(v for v in printed['preconditions'])
        printed_terminals = '\n\n'.join(v for v in printed['terminations'])
        printed_params = '\n'.join(f'{k}: {v}' for (k, v) in self.model_aux['params'].items())
        self.logger.log(
            f'[info] {heading}\n'
            f'[info] compiled JAX CPFs:\n\n'
            f'{printed_cpfs}\n\n'
            f'[info] compiled JAX reward:\n\n'
            f'{printed_reward}\n\n'
            f'[info] compiled JAX invariants:\n\n'
            f'{printed_invariants}\n\n'
            f'[info] compiled JAX preconditions:\n\n'
            f'{printed_preconds}\n\n'
            f'[info] compiled JAX terminations:\n\n'
            f'{printed_terminals}\n'
            f'[info] model parameters:\n'
            f'{printed_params}\n'
        )
    
    def _compile_constraints(self, constraints, aux):
        return [self._jax(expr, aux, dtype=jnp.bool_) for expr in constraints]
        
    def _compile_cpfs(self, aux):
        jax_cpfs = {}
        for cpfs in self.levels.values():
            for cpf in cpfs:
                _, expr = self.rddl.cpfs[cpf]
                dtype = self.JAX_TYPES.get(self.rddl.variable_ranges[cpf], self.INT)
                jax_cpfs[cpf] = self._jax(expr, aux, dtype=dtype)
        return jax_cpfs
    
    def _compile_reward(self, aux):
        return self._jax(self.rddl.reward, aux, dtype=self.REAL)
    
    def _compile_policy_block(self, aux):
        jax_actions = {}
        for (name, (_, expr)) in self.rddl.policy.items():
            dtype = self.JAX_TYPES.get(self.rddl.variable_ranges[name], self.INT)
            jax_actions[name] = self._jax(expr, aux, dtype=dtype)
        return jax_actions

    def _extract_inequality_constraint(self, expr):
        result = []
        etype, op = expr.etype
        if etype == 'relational':
            left, right = expr.args
            if op == '<' or op == '<=':
                result.append((left, right))
            elif op == '>' or op == '>=':
                result.append((right, left))
        elif etype == 'boolean' and op == '^':
            for arg in expr.args:
                result.extend(self._extract_inequality_constraint(arg))
        return result
    
    def _extract_equality_constraint(self, expr):
        result = []
        etype, op = expr.etype
        if etype == 'relational':
            left, right = expr.args
            if op == '==':
                result.append((left, right))
        elif etype == 'boolean' and op == '^':
            for arg in expr.args:
                result.extend(self._extract_equality_constraint(arg))
        return result
            
    def _jax_nonlinear_constraints(self, aux):         
        jax_equalities, jax_inequalities = [], []
        for (i, expr) in enumerate(self.rddl.preconditions):
            if not self.constraints.is_box_preconditions[i]:

                # compile inequalities to JAX and write as h(s, a) <= 0
                for (left, right) in self._extract_inequality_constraint(expr):
                    jax_lhs = self._jax(left, aux)
                    jax_rhs = self._jax(right, aux)
                    jax_constr = self._jax_binary(
                        jax_lhs, jax_rhs, jnp.subtract, at_least_int=True)
                    jax_inequalities.append(jax_constr)

                # compile equalities to JAX and write as g(s, a) == 0
                for (left, right) in self._extract_equality_constraint(expr):
                    jax_lhs = self._jax(left, aux)
                    jax_rhs = self._jax(right, aux)
                    jax_constr = self._jax_binary(
                        jax_lhs, jax_rhs, jnp.subtract, at_least_int=True)
                    jax_equalities.append(jax_constr)
        return jax_inequalities, jax_equalities
    
    def _jax_preconditions(self):
        preconds = self.preconditions
        def _jax_wrapped_preconditions(key, errors, fls, nfls, params):
            precond_check = jnp.array(True, dtype=jnp.bool_)
            for precond in preconds:
                sample, key, err, params = precond(fls, nfls, params, key)
                precond_check = jnp.logical_and(precond_check, sample)
                errors = errors | err
            return precond_check, key, errors, params 
        return _jax_wrapped_preconditions

    def _jax_inequalities(self, aux_constr):
        inequality_fns, equality_fns = self._jax_nonlinear_constraints(aux_constr)
        def _jax_wrapped_inequalities(key, errors, fls, nfls, params):
            inequalities, equalities = [], []
            for constraint in inequality_fns:
                sample, key, err, params = constraint(fls, nfls, params, key)
                inequalities.append(sample)
                errors = errors | err
            for constraint in equality_fns:
                sample, key, err, params = constraint(fls, nfls, params, key)
                equalities.append(sample)
                errors = errors | err
            return (inequalities, equalities), key, errors, params
        return _jax_wrapped_inequalities

    def _jax_cpfs(self):
        cpfs = self.cpfs
        def _jax_wrapped_cpfs(key, errors, fls, nfls, params):
            fls = fls.copy()
            for (name, cpf) in cpfs.items():
                fls[name], key, err, params = cpf(fls, nfls, params, key)
                errors = errors | err  
            return fls, key, errors, params
        return _jax_wrapped_cpfs

    def _jax_reward(self):
        reward_fn = self.reward
        def _jax_wrapped_reward(key, errors, fls, nfls, params):
            reward, key, err, params = reward_fn(fls, nfls, params, key)
            errors = errors | err
            return reward, key, errors, params
        return _jax_wrapped_reward

    def _jax_policy(self):
        policy = self.policy
        def _jax_wrapped_actions(key, errors, fls, nfls, params):
            actions = {}
            for (name, jax_act) in policy.items():
                actions[name], key, err, params = jax_act(fls, nfls, params, key)
                errors = errors | err  
            return actions, key, errors, params
        return _jax_wrapped_actions

    def _jax_invariants(self):
        invariants = self.invariants
        def _jax_wrapped_invariants(key, errors, fls, nfls, params):
            invariant_check = jnp.array(True, dtype=jnp.bool_)
            for invariant in invariants:
                sample, key, err, params = invariant(fls, nfls, params, key)
                invariant_check = jnp.logical_and(invariant_check, sample)
                errors = errors | err
            return invariant_check, key, errors, params
        return _jax_wrapped_invariants

    def _jax_terminations(self):
        terminations = self.terminations
        def _jax_wrapped_terminations(key, errors, fls, nfls, params):
            terminated_check = jnp.array(False, dtype=jnp.bool_)
            for terminal in terminations:
                sample, key, err, params = terminal(fls, nfls, params, key)
                terminated_check = jnp.logical_or(terminated_check, sample)
                errors = errors | err
            return terminated_check, key, errors, params
        return _jax_wrapped_terminations

    def compile_transition(self, check_constraints: bool=False,
                           constraint_func: bool=False, 
                           cache_path_info: bool=False,
                           aux_constr: Dict[str, Any]={}) -> Callable:
        '''Compiles the current RDDL model into a JAX transition function that 
        samples the next state.
        
        The arguments of the returned function is:
            - key is the PRNG key
            - actions is the dict of action tensors
            - fls is the dict of current fluent pvar tensors
            - nfls is the dict of nonfluent pvar tensors
            - params is a dict of parameters for the relaxed model.
        
        The returned value of the function is:
            - fls is the returned next epoch fluent values
            - log includes all the auxiliary information about constraints 
              satisfied, errors, etc.
            
        constraint_func provides the option to compile nonlinear constraints:
        
            1. f(s, a) ?? g(s, a)
            2. f1(s, a) ^ f2(s, a) ^ ... ?? g(s, a)
            3. forall_{?p1, ...} f(s, a, ?p1, ...) ?? g(s, a) where f is of the
               form 1 or 2 above.
        
        and where ?? is <, <=, > or >= into JAX expressions h(s, a) representing 
        the constraints of the form: 
            
            h(s, a) <= 0
            g(s, a) == 0
                
        for which a penalty or barrier-type method can be used to enforce 
        constraint satisfaction. A list is returned containing values for all
        non-box inequality constraints.
        
        :param check_constraints: whether state, action and termination 
        conditions should be checked on each time step: this info is stored in the
        returned log and does not raise an exception
        :param constraint_func: produces the h(s, a) function described above
        in addition to the usual outputs
        :param cache_path_info: whether to save full path traces as part of the log
        :param aux_constr: auxiliary info to pass when compiling constraint_func
        '''
        NORMAL = JaxRDDLCompiler.ERROR_CODES['NORMAL']      
        
        # compile all components of the RDDL
        cpf_fn = self._jax_cpfs()
        reward_fn = self._jax_reward()

        # compile optional constraints
        if check_constraints:
            precond_fn = self._jax_preconditions()
            invariant_fn = self._jax_invariants()
            terminal_fn = self._jax_terminations()
        else:
            precond_fn = invariant_fn = terminal_fn = None
        
        # compile optional inequalities
        if constraint_func:
            ineq_fn = self._jax_inequalities(aux_constr)
        else:
            ineq_fn = None

        # do a single step update from the RDDL model
        def _jax_wrapped_single_step(key, actions, fls, nfls, params):
            errors = NORMAL
            
            fls = fls.copy()
            fls.update(actions)
            
            # check action preconditions
            if check_constraints:
                precond, key, errors, params = precond_fn(key, errors, fls, nfls, params)
            else:
                precond = jnp.array(True, dtype=jnp.bool_)
            
            # compute h(s, a) <= 0 and g(s, a) == 0 constraint functions
            if constraint_func:
                (inequalities, equalities), key, errors, params = ineq_fn(
                    key, errors, fls, nfls, params)
            else:
                inequalities, equalities = [], []
                
            # calculate CPFs in topological order
            fls, key, errors, params = cpf_fn(key, errors, fls, nfls, params)                
            fluents = fls if cache_path_info else {}
            
            # calculate the immediate reward
            reward, key, errors, params = reward_fn(key, errors, fls, nfls, params)
            
            # set the next state to the current state
            for (state, next_state) in self.rddl.next_state.items():
                fls[state] = fls[next_state]
            
            # check the state invariants and termination
            if check_constraints:
                invariant, key, errors, params = invariant_fn(key, errors, fls, nfls, params)
                terminated, key, errors, params = terminal_fn(key, errors, fls, nfls, params)
            else:
                invariant = jnp.array(True, dtype=jnp.bool_)
                terminated = jnp.array(False, dtype=jnp.bool_)
               
            # prepare the return value
            log = {
                'fluents': fluents,
                'reward': reward,
                'error': errors,
                'precondition': precond,
                'invariant': invariant,
                'termination': terminated,
                'inequalities': inequalities,
                'equalities': equalities
            }                
            return fls, log, params
        
        return _jax_wrapped_single_step        
    
    def _compile_policy_step(self, policy, transition_fn):
        def _jax_wrapped_policy_step(key, policy_params, hyperparams, step, fls, nfls, 
                                     model_params, fls_hist):
            key, subkey = random.split(key)
            actions = policy(key, policy_params, hyperparams, step, fls, fls_hist)
            return transition_fn(subkey, actions, fls, nfls, model_params)
        return _jax_wrapped_policy_step

    def _compile_batched_policy_step(self, policy_step_fn, n_batch, model_params_reduction):
        def _jax_wrapped_batched_policy_step(carry, step):
            key, policy_params, hyperparams, fls, nfls, model_params, fls_hist = carry  

            # update history 
            if fls_hist:
                fls_hist = jax.tree_util.tree_map(
                    lambda h, f: h.at[step].set(f), fls_hist, fls)

            # perform a batched single step from the model
            keys = random.split(key, num=1 + n_batch)
            key, subkeys = keys[0], keys[1:]
            fls, log, model_params = jax.vmap(
                policy_step_fn, in_axes=(0, None, None, None, 0, None, None, 1)
            )(subkeys, policy_params, hyperparams, step, fls, nfls, model_params, fls_hist)

            # must aggregate model params across branches into a single pytree
            model_params = jax.tree_util.tree_map(model_params_reduction, model_params)

            carry = (key, policy_params, hyperparams, fls, nfls, model_params, fls_hist)
            return carry, log 
        return _jax_wrapped_batched_policy_step
        
    def _compile_unrolled_policy_step(self, batched_policy_step_fn, n_steps, history_dependent):
        def _jax_wrapped_batched_policy_rollout(key, policy_params, hyperparams, fls, nfls, 
                                                model_params):            
            # initialize history the policy sees
            if history_dependent:
                fls_hist = jax.tree_util.tree_map(
                    lambda f: jnp.zeros((n_steps,) + jnp.shape(f), dtype=f.dtype), fls)
            else:
                fls_hist = {}

            # perform the trajectory unrolling in batched mode
            start = (key, policy_params, hyperparams, fls, nfls, model_params, fls_hist)
            steps = jnp.arange(n_steps)
            end, log = jax.lax.scan(batched_policy_step_fn, start, steps)

            # swap the batch and time axis so the batch axis is first
            log = jax.tree_util.tree_map(partial(jnp.swapaxes, axis1=0, axis2=1), log)
            if not log['fluents']:
                fls_end = end[3]
                log['fluents'] = {name: jnp.expand_dims(fl, axis=1) 
                                  for (name, fl) in fls_end.items()}
            model_params = end[-2]
            return log, model_params        
        return _jax_wrapped_batched_policy_rollout
    
    def compile_rollouts(self, policy: Callable,
                         n_steps: int,
                         n_batch: int,
                         history_dependent: bool=False,
                         check_constraints: bool=False,
                         constraint_func: bool=False, 
                         cache_path_info: bool=False,
                         model_params_reduction: Callable=lambda x: x[0],
                         aux_constr: Dict[str, Any]={}) -> Callable:
        '''Compiles the current RDDL model into a JAX transition function that 
        samples trajectories with a fixed horizon from a policy.
        
        The arguments of the returned function is:
            - key is the PRNG key (used by a stochastic policy)
            - policy_params is a pytree of trainable policy weights
            - hyperparams is a pytree of (optional) fixed policy hyper-parameters
            - fls is the dictionary of current fluent tensor values
            - nfls is the dictionary of non-fluent tensor value
            - model_params is a dict of model hyperparameters.
        
        The returned value of the returned function is:
            - log is the dictionary of all trajectory information, including
              constraints that were satisfied, errors, etc
            - model_params is the final set of model parameters.
            
        The arguments of the policy function is:
            - key is the PRNG key (used by a stochastic policy)
            - params is a pytree of trainable policy weights
            - hyperparams is a pytree of (optional) fixed policy hyper-parameters
            - step is the time index of the decision in the current rollout
            - fls is a dict of fluent tensors for the current epoch
            - fls_hist is a dict of fluent tensors up to the current epoch
        
        :param policy: a Jax compiled function for the policy as described above
        decision epoch, state dict, and an RNG key and returns an action dict
        :param n_steps: the rollout horizon
        :param n_batch: how many rollouts each batch performs
        :param history_dependent: whether the policy is fed a history of past fluent values
        :param check_constraints: whether state, action and termination 
        conditions should be checked on each time step: this info is stored in the
        returned log and does not raise an exception
        :param constraint_func: produces the h(s, a) constraint function
        in addition to the usual outputs
        :param cache_path_info: whether to save full path traces as part of the log
        :param model_params_reduction: how to aggregate updated model_params across runs
        in the batch (defaults to selecting the first element's parameters in the batch)
        :param aux_constr: auxiliary info to pass when compiling constraint_func
        '''
        jax_fn = self.compile_transition(
            check_constraints, constraint_func, cache_path_info, aux_constr)
        jax_fn = self._compile_policy_step(policy, jax_fn)
        jax_fn = self._compile_batched_policy_step(jax_fn, n_batch, model_params_reduction)
        jax_fn = self._compile_unrolled_policy_step(jax_fn, n_steps, history_dependent)
        return jax_fn
    
    # ===========================================================================
    # error checks and prints
    # ===========================================================================
    
    def print_jax(self) -> Dict[str, Any]:
        '''Returns a dictionary containing the string representations of all 
        Jax compiled expressions from the RDDL file.
        '''
        fls, nfls = self.split_fluent_nonfluent(self.init_values)
        params = self.model_aux['params']
        key = jax.random.PRNGKey(42)
        printed = {
            'cpfs': {
                name: str(jax.make_jaxpr(expr)(fls, nfls, params, key))
                for (name, expr) in self.cpfs.items()
            },
            'reward': str(jax.make_jaxpr(self.reward)(fls, nfls, params, key)),
            'invariants': [
                str(jax.make_jaxpr(expr)(fls, nfls, params, key))
                for expr in self.invariants
            ],
            'preconditions': [
                str(jax.make_jaxpr(expr)(fls, nfls, params, key))
                for expr in self.preconditions
            ],
            'terminations': [
                str(jax.make_jaxpr(expr)(fls, nfls, params, key))
                for expr in self.terminations
            ]
        }
        return printed
    
    def model_parameter_info(self) -> Dict[str, Dict[str, Any]]:
        '''Returns a dictionary of additional information about model parameters.'''
        result = {}
        for (id, value) in self.model_aux['params'].items():
            expr = self.traced.lookup(id)
            result[id] = {
                'id': id, 
                'rddl_op': ' '.join(expr.etype), 
                'init_value': value
            }
        return result
        
    def overriden_ops_info(self) -> Dict[str, Dict[str, List[int]]]:
        '''Returns a dictionary of operations overriden by another class.'''
        result = {}
        for (id, class_) in self.model_aux['overriden'].items():
            expr = self.traced.lookup(id)
            rddl_op = ' '.join(expr.etype)
            result.setdefault(class_, {}).setdefault(rddl_op, []).append(id)
        return result
    
    def exact_ops_info(self) -> Dict[str, List[int]]:
        '''Returns a dictionary of operations that use the exact operations in this class.'''
        result = {}
        for id in self.model_aux['exact']:
            expr = self.traced.lookup(id)
            rddl_op = ' '.join(expr.etype)
            result.setdefault(rddl_op, []).append(id)
        return result

    @staticmethod
    def _check_valid_op(expr, valid_ops):
        etype, op = expr.etype
        if op not in valid_ops:
            valid_op_str = ','.join(valid_ops)
            raise RDDLNotImplementedError(
                f'{etype} operator {op} is not supported: '
                f'must be one of {valid_op_str}.\n' + print_stack_trace(expr))
    
    @staticmethod
    def _check_num_args(expr, required_args):
        actual_args = len(expr.args)
        if actual_args != required_args:
            etype, op = expr.etype
            raise RDDLInvalidNumberOfArgumentsError(
                f'{etype} operator {op} requires {required_args} arguments, '
                f'got {actual_args}.\n' + print_stack_trace(expr))
    
    @staticmethod
    def _check_num_args_min(expr, required_args):
        actual_args = len(expr.args)
        if actual_args < required_args:
            etype, op = expr.etype
            raise RDDLInvalidNumberOfArgumentsError(
                f'{etype} operator {op} requires at least {required_args} arguments, '
                f'got {actual_args}.\n' + print_stack_trace(expr))
        
    ERROR_CODES = {
        'NORMAL': 0,
        'INVALID_CAST': 2 ** 0,
        'INVALID_PARAM_UNIFORM': 2 ** 1,
        'INVALID_PARAM_NORMAL': 2 ** 2,
        'INVALID_PARAM_EXPONENTIAL': 2 ** 3,
        'INVALID_PARAM_WEIBULL': 2 ** 4,
        'INVALID_PARAM_BERNOULLI': 2 ** 5,
        'INVALID_PARAM_POISSON': 2 ** 6,
        'INVALID_PARAM_GAMMA': 2 ** 7,
        'INVALID_PARAM_BETA': 2 ** 8,
        'INVALID_PARAM_GEOMETRIC': 2 ** 9,
        'INVALID_PARAM_PARETO': 2 ** 10,
        'INVALID_PARAM_STUDENT': 2 ** 11,
        'INVALID_PARAM_GUMBEL': 2 ** 12,
        'INVALID_PARAM_LAPLACE': 2 ** 13,
        'INVALID_PARAM_CAUCHY': 2 ** 14,
        'INVALID_PARAM_GOMPERTZ': 2 ** 15,
        'INVALID_PARAM_CHISQUARE': 2 ** 16,
        'INVALID_PARAM_KUMARASWAMY': 2 ** 17,
        'INVALID_PARAM_DISCRETE': 2 ** 18,
        'INVALID_PARAM_KRON_DELTA': 2 ** 19,
        'INVALID_PARAM_DIRICHLET': 2 ** 20,
        'INVALID_PARAM_MULTIVARIATE_STUDENT': 2 ** 21,
        'INVALID_PARAM_MULTINOMIAL': 2 ** 22,
        'INVALID_PARAM_BINOMIAL': 2 ** 23,
        'INVALID_PARAM_NEGATIVE_BINOMIAL': 2 ** 24        
    }
    
    INVERSE_ERROR_CODES = {
        0: 'Casting occurred that could result in loss of precision.',
        1: 'Found Uniform(a, b) distribution where a > b.',
        2: 'Found Normal(m, v^2) distribution where v < 0.',
        3: 'Found Exponential(s) distribution where s <= 0.',
        4: 'Found Weibull(k, l) distribution where either k <= 0 or l <= 0.',
        5: 'Found Bernoulli(p) distribution where either p < 0 or p > 1.',
        6: 'Found Poisson(l) distribution where l < 0.',
        7: 'Found Gamma(k, l) distribution where either k <= 0 or l <= 0.',
        8: 'Found Beta(a, b) distribution where either a <= 0 or b <= 0.',
        9: 'Found Geometric(p) distribution where either p < 0 or p > 1.',
        10: 'Found Pareto(k, l) distribution where either k <= 0 or l <= 0.',
        11: 'Found Student(df) distribution where df <= 0.',
        12: 'Found Gumbel(m, s) distribution where s <= 0.',
        13: 'Found Laplace(m, s) distribution where s <= 0.',
        14: 'Found Cauchy(m, s) distribution where s <= 0.',
        15: 'Found Gompertz(k, l) distribution where either k <= 0 or l <= 0.',
        16: 'Found ChiSquare(df) distribution where df <= 0.',
        17: 'Found Kumaraswamy(a, b) distribution where either a <= 0 or b <= 0.',
        18: 'Found Discrete(p) distribution where either p < 0 or p does not sum to 1.',
        19: 'Found KronDelta(x) distribution where x is not int nor bool.',
        20: 'Found Dirichlet(alpha) distribution where alpha < 0.',
        21: 'Found MultivariateStudent(mean, cov, df) distribution where df <= 0.',
        22: 'Found Multinomial(n, p) distribution where either p < 0, p does not sum to 1, or n <= 0.',
        23: 'Found Binomial(n, p) distribution where either p < 0, p > 1, or n <= 0.',
        24: 'Found NegativeBinomial(n, p) distribution where either p < 0, p > 1, or n <= 0.'        
    }
    
    @staticmethod
    def get_error_codes(error: int) -> List[int]:
        '''Given a compacted integer error flag from the execution of Jax, and 
        decomposes it into individual error codes.
        '''
        binary = reversed(bin(error)[2:])
        return [i for (i, c) in enumerate(binary) if c == '1']
    
    @staticmethod
    def get_error_messages(error: int) -> List[str]:
        '''Given a compacted integer error flag from the execution of Jax, and 
        decomposes it into error strings.
        '''
        codes = JaxRDDLCompiler.get_error_codes(error)
        return [JaxRDDLCompiler.INVERSE_ERROR_CODES[i] for i in codes]
    
    # ===========================================================================
    # expression compilation
    # ===========================================================================
    
    def _jax(self, expr, aux, dtype=None):
        etype, _ = expr.etype
        if etype == 'constant':
            jax_expr = self._jax_constant(expr, aux)
        elif etype == 'pvar':
            jax_expr = self._jax_pvar(expr, aux)
        elif etype == 'arithmetic':
            jax_expr = self._jax_arithmetic(expr, aux)
        elif etype == 'relational':
            jax_expr = self._jax_relational(expr, aux)
        elif etype == 'boolean':
            jax_expr = self._jax_logical(expr, aux)
        elif etype == 'aggregation':
            jax_expr = self._jax_aggregation(expr, aux)
        elif etype == 'func':
            jax_expr = self._jax_function(expr, aux)
        elif etype == 'pyfunc':
            jax_expr = self._jax_pyfunc(expr, aux)
        elif etype == 'control':
            jax_expr = self._jax_control(expr, aux)
        elif etype == 'randomvar':
            jax_expr = self._jax_random(expr, aux)
        elif etype == 'randomvector':
            jax_expr = self._jax_random_vector(expr, aux)
        elif etype == 'matrix':
            jax_expr = self._jax_matrix(expr, aux)
        else:
            raise RDDLNotImplementedError(
                f'Expression type {expr} is not supported.\n' + print_stack_trace(expr))
            
        # force type cast of tensor as required by caller
        if dtype is not None:
            jax_expr = self._jax_cast(jax_expr, dtype)
        return jax_expr
            
    def _jax_cast(self, jax_expr, dtype):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_CAST']

        def _jax_wrapped_cast(fls, nfls, params, key):
            val, key, err, params = jax_expr(fls, nfls, params, key)
            sample = jnp.asarray(val, dtype=dtype)
            invalid_cast = jnp.logical_and(
                jnp.logical_not(jnp.can_cast(val, dtype)),
                jnp.any(sample != val)
            )
            err = err | (invalid_cast * ERR)
            return sample, key, err, params
        return _jax_wrapped_cast
   
    def _fix_dtype(self, value):
        dtype = jnp.result_type(value)
        if jnp.issubdtype(dtype, jnp.integer):
            return self.INT
        elif jnp.issubdtype(dtype, jnp.floating):
            return self.REAL
        elif jnp.issubdtype(dtype, jnp.bool_):
            return jnp.bool_
        else:
            raise TypeError(f'dtype {dtype} of {value} is not valid.')
       
    # ===========================================================================
    # leaves
    # ===========================================================================
    
    def _jax_constant(self, expr, aux):
        NORMAL = JaxRDDLCompiler.ERROR_CODES['NORMAL']
        cached_value = self.traced.cached_sim_info(expr)
        dtype = self._fix_dtype(cached_value)
        
        def _jax_wrapped_constant(fls, nfls, params, key):
            sample = jnp.asarray(cached_value, dtype=dtype)
            return sample, key, NORMAL, params
        return _jax_wrapped_constant
    
    def _jax_pvar_slice(self, slice_):
        NORMAL = JaxRDDLCompiler.ERROR_CODES['NORMAL']
        
        def _jax_wrapped_pvar_slice(fls, nfls, params, key):
            return slice_, key, NORMAL, params
        return _jax_wrapped_pvar_slice
            
    def _jax_pvar(self, expr, aux):
        NORMAL = JaxRDDLCompiler.ERROR_CODES['NORMAL']
        var, pvars = expr.args  
        is_value, cached_info = self.traced.cached_sim_info(expr)
        
        # boundary case: free variable is converted to array (0, 1, 2...)
        # boundary case: domain object is converted to canonical integer index
        if is_value:
            cached_value = cached_info
            dtype = self._fix_dtype(cached_value)
            
            def _jax_wrapped_object(fls, nfls, params, key):
                sample = jnp.asarray(cached_value, dtype=dtype)
                return sample, key, NORMAL, params
            return _jax_wrapped_object
        
        # boundary case: no shape information (e.g. scalar pvar)
        elif cached_info is None:
            def _jax_wrapped_pvar_scalar(fls, nfls, params, key):
                value = fls[var] if var in fls else nfls[var]
                sample = jnp.asarray(value, dtype=self._fix_dtype(value))
                return sample, key, NORMAL, params
            return _jax_wrapped_pvar_scalar
        
        # must slice and/or reshape value tensor to match free variables
        else:
            slices, axis, shape, op_code, op_args = cached_info 
        
            # compile nested expressions
            if slices and op_code == RDDLObjectsTracer.NUMPY_OP_CODE.NESTED_SLICE:
                jax_nested_expr = [
                    (self._jax(arg, aux) if slice_ is None else self._jax_pvar_slice(slice_))
                    for (arg, slice_) in zip(pvars, slices)
                ]    
                
                def _jax_wrapped_pvar_tensor_nested(fls, nfls, params, key):
                    error = NORMAL
                    value = fls[var] if var in fls else nfls[var]
                    sample = jnp.asarray(value, dtype=self._fix_dtype(value))
                    new_slices = []
                    for jax_expr in jax_nested_expr:
                        new_slice, key, err, params = jax_expr(fls, nfls, params, key)
                        new_slice = jnp.asarray(new_slice, dtype=self.INT)
                        new_slices.append(new_slice)
                        error = error | err
                    sample = sample[tuple(new_slices)]
                    return sample, key, error, params
                return _jax_wrapped_pvar_tensor_nested
                
            # tensor variable but no nesting  
            else:
                def _jax_wrapped_pvar_tensor_non_nested(fls, nfls, params, key):
                    value = fls[var] if var in fls else nfls[var]
                    sample = jnp.asarray(value, dtype=self._fix_dtype(value))
                    if slices:
                        sample = sample[slices]
                    if axis:
                        sample = jnp.expand_dims(sample, axis=axis)
                        sample = jnp.broadcast_to(sample, shape=shape)
                    if op_code == RDDLObjectsTracer.NUMPY_OP_CODE.EINSUM:
                        sample = jnp.einsum(sample, *op_args)
                    elif op_code == RDDLObjectsTracer.NUMPY_OP_CODE.TRANSPOSE:
                        sample = jnp.transpose(sample, axes=op_args)
                    return sample, key, NORMAL, params
                return _jax_wrapped_pvar_tensor_non_nested
    
    # ===========================================================================
    # boilerplate helper functions
    # ===========================================================================
    
    def _jax_unary(self, jax_expr, jax_op, at_least_int=False, check_dtype=None):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_CAST']

        def _jax_wrapped_unary_op(fls, nfls, params, key):
            sample, key, err, params = jax_expr(fls, nfls, params, key)
            if at_least_int:
                sample = self.ONE * sample
            if check_dtype is not None:
                invalid_cast = jnp.logical_not(jnp.can_cast(sample, check_dtype))
                err = err | (invalid_cast * ERR)
            sample = jax_op(sample)
            return sample, key, err, params
        return _jax_wrapped_unary_op
    
    def _jax_binary(self, jax_lhs, jax_rhs, jax_op, at_least_int=False, check_dtype=None):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_CAST']
        
        def _jax_wrapped_binary_op(fls, nfls, params, key):
            sample1, key, err1, params = jax_lhs(fls, nfls, params, key)
            sample2, key, err2, params = jax_rhs(fls, nfls, params, key)
            if at_least_int:
                sample1 = self.ONE * sample1
                sample2 = self.ONE * sample2
            sample = jax_op(sample1, sample2)
            err = err1 | err2
            if check_dtype is not None:
                invalid_cast = jnp.logical_not(jnp.logical_and(
                    jnp.can_cast(sample1, check_dtype),
                    jnp.can_cast(sample2, check_dtype))
                )
                err = err | (invalid_cast * ERR)
            return sample, key, err, params
        return _jax_wrapped_binary_op
    
    def _jax_unary_helper(self, expr, aux, jax_op, at_least_int=False, check_dtype=None):
        JaxRDDLCompiler._check_num_args(expr, 1)
        arg, = expr.args
        jax_expr = self._jax(arg, aux)
        return self._jax_unary(
            jax_expr, jax_op, at_least_int=at_least_int, check_dtype=check_dtype)

    def _jax_binary_helper(self, expr, aux, jax_op, at_least_int=False, check_dtype=None):
        JaxRDDLCompiler._check_num_args(expr, 2)
        lhs, rhs = expr.args
        jax_lhs = self._jax(lhs, aux)
        jax_rhs = self._jax(rhs, aux)
        return self._jax_binary(
            jax_lhs, jax_rhs, jax_op, at_least_int=at_least_int, check_dtype=check_dtype)
    
    def _jax_nary_helper(self, expr, aux, jax_op, at_least_int=False, check_dtype=None):
        JaxRDDLCompiler._check_num_args_min(expr, 2)
        args = expr.args
        jax_exprs = [self._jax(arg, aux) for arg in args]
        result = jax_exprs[0]
        for jax_rhs in jax_exprs[1:]:
            result = self._jax_binary(
                result, jax_rhs, jax_op, at_least_int=at_least_int, check_dtype=check_dtype)
        return result
    
    # ===========================================================================
    # arithmetic
    # ===========================================================================
    
    def _jax_arithmetic(self, expr, aux):
        JaxRDDLCompiler._check_valid_op(expr, {'-', '+', '*', '/'})
        _, op = expr.etype
        if op == '-':
            if len(expr.args) == 1:
                return self._jax_negate(expr, aux)
            else:
                return self._jax_subtract(expr, aux)
        elif op == '/':
            return self._jax_divide(expr, aux)
        elif op == '+':
            return self._jax_add(expr, aux)
        elif op == '*':
            return self._jax_multiply(expr, aux)
    
    def _jax_negate(self, expr, aux):
        aux['exact'].add(expr.id)
        return self._jax_unary_helper(expr, aux, jnp.negative, at_least_int=True)
    
    def _jax_add(self, expr, aux):
        aux['exact'].add(expr.id)
        return self._jax_nary_helper(expr, aux, jnp.add, at_least_int=True)
    
    def _jax_subtract(self, expr, aux):
        aux['exact'].add(expr.id)
        return self._jax_binary_helper(expr, aux, jnp.subtract, at_least_int=True)
    
    def _jax_multiply(self, expr, aux):
        aux['exact'].add(expr.id)
        return self._jax_nary_helper(expr, aux, jnp.multiply, at_least_int=True)
    
    def _jax_divide(self, expr, aux):
        aux['exact'].add(expr.id)
        return self._jax_binary_helper(expr, aux, jnp.divide, at_least_int=True)
    
    # ===========================================================================
    # relational
    # ===========================================================================
    
    def _jax_relational(self, expr, aux):
        JaxRDDLCompiler._check_valid_op(expr, {'>=', '<=', '>', '<', '==', '~='})
        _, op = expr.etype
        if op == '>=':
            return self._jax_greater_equal(expr, aux)
        elif op == '<=':
            return self._jax_less_equal(expr, aux)
        elif op == '>':
            return self._jax_greater(expr, aux)
        elif op == '<':
            return self._jax_less(expr, aux)
        elif op == '==':
            return self._jax_equal(expr, aux)
        elif op == '~=':
            return self._jax_not_equal(expr, aux)

    def _jax_greater_equal(self, expr, aux):
        aux['exact'].add(expr.id)
        return self._jax_binary_helper(expr, aux, jnp.greater_equal, at_least_int=True)
        
    def _jax_less_equal(self, expr, aux):
        aux['exact'].add(expr.id)
        return self._jax_binary_helper(expr, aux, jnp.less_equal, at_least_int=True)

    def _jax_greater(self, expr, aux):
        aux['exact'].add(expr.id)
        return self._jax_binary_helper(expr, aux, jnp.greater, at_least_int=True)
    
    def _jax_less(self, expr, aux):
        aux['exact'].add(expr.id)
        return self._jax_binary_helper(expr, aux, jnp.less, at_least_int=True)
    
    def _jax_equal(self, expr, aux):
        aux['exact'].add(expr.id)
        return self._jax_binary_helper(expr, aux, jnp.equal, at_least_int=True)
    
    def _jax_not_equal(self, expr, aux):
        aux['exact'].add(expr.id)
        return self._jax_binary_helper(expr, aux, jnp.not_equal, at_least_int=True)
    
    # ===========================================================================
    # logical
    # ===========================================================================

    def _jax_logical(self, expr, aux):
        JaxRDDLCompiler._check_valid_op(expr, {'^', '&', '|', '~', '=>', '<=>'})
        _, op = expr.etype
        if op == '~':
            if len(expr.args) == 1:
                return self._jax_not(expr, aux)
            else:
                return self._jax_xor(expr, aux)
        elif op == '^' or op == '&':
            return self._jax_and(expr, aux)
        elif op == '|':
            return self._jax_or(expr, aux)
        elif op == '=>':
            return self._jax_implies(expr, aux)
        elif op == '<=>':
            return self._jax_equiv(expr, aux)        
    
    def _jax_not(self, expr, aux):
        aux['exact'].add(expr.id)
        return self._jax_unary_helper(expr, aux, jnp.logical_not, check_dtype=jnp.bool_)
    
    def _jax_and(self, expr, aux):
        aux['exact'].add(expr.id)
        return self._jax_nary_helper(expr, aux, jnp.logical_and, check_dtype=jnp.bool_)
    
    def _jax_or(self, expr, aux):
        aux['exact'].add(expr.id)
        return self._jax_nary_helper(expr, aux, jnp.logical_or, check_dtype=jnp.bool_)
    
    def _jax_xor(self, expr, aux):
        aux['exact'].add(expr.id)
        return self._jax_binary_helper(expr, aux, jnp.logical_xor, check_dtype=jnp.bool_)
    
    def _jax_implies(self, expr, aux):
        aux['exact'].add(expr.id)
        def implies_op(x, y):
            return jnp.logical_or(jnp.logical_not(x), y)
        return self._jax_binary_helper(expr, aux, implies_op, check_dtype=jnp.bool_)
    
    def _jax_equiv(self, expr, aux):
        aux['exact'].add(expr.id)
        return self._jax_binary_helper(expr, aux, jnp.equal, check_dtype=jnp.bool_)
    
    # ===========================================================================
    # aggregation
    # ===========================================================================
    
    def _jax_aggregation(self, expr, aux):
        JaxRDDLCompiler._check_valid_op(expr, {'sum', 'avg', 'prod', 'minimum', 'maximum', 
                                               'forall', 'exists', 'argmin', 'argmax'})
        _, op = expr.etype
        if op == 'sum':
            return self._jax_sum(expr, aux)
        elif op == 'avg':
            return self._jax_avg(expr, aux)
        elif op == 'prod':
            return self._jax_prod(expr, aux)
        elif op == 'minimum':
            return self._jax_minimum(expr, aux)
        elif op == 'maximum':
            return self._jax_maximum(expr, aux)
        elif op == 'forall':
            return self._jax_forall(expr, aux)
        elif op == 'exists':
            return self._jax_exists(expr, aux)
        elif op == 'argmin':
            return self._jax_argmin(expr, aux)
        elif op == 'argmax':
            return self._jax_argmax(expr, aux)
        
    def _jax_aggregation_helper(self, expr, aux, jax_op, is_bool=False):
        arg = expr.args[-1]
        _, axes = self.traced.cached_sim_info(expr)   
        jax_expr = self._jax(arg, aux) 
        return self._jax_unary(
            jax_expr, 
            jax_op=partial(jax_op, axis=axes), 
            at_least_int=not is_bool, 
            check_dtype=jnp.bool_ if is_bool else None
        )

    def _jax_sum(self, expr, aux):
        aux['exact'].add(expr.id)
        return self._jax_aggregation_helper(expr, aux, jnp.sum)
    
    def _jax_avg(self, expr, aux):
        aux['exact'].add(expr.id)
        return self._jax_aggregation_helper(expr, aux, jnp.mean)
    
    def _jax_prod(self, expr, aux):
        aux['exact'].add(expr.id)
        return self._jax_aggregation_helper(expr, aux, jnp.prod)
    
    def _jax_minimum(self, expr, aux):
        aux['exact'].add(expr.id)
        return self._jax_aggregation_helper(expr, aux, jnp.min)
    
    def _jax_maximum(self, expr, aux):
        aux['exact'].add(expr.id)
        return self._jax_aggregation_helper(expr, aux, jnp.max)
    
    def _jax_forall(self, expr, aux):
        aux['exact'].add(expr.id)
        return self._jax_aggregation_helper(expr, aux, jnp.all, is_bool=True)
    
    def _jax_exists(self, expr, aux):
        aux['exact'].add(expr.id)
        return self._jax_aggregation_helper(expr, aux, jnp.any, is_bool=True)
    
    def _jax_argmin(self, expr, aux):
        aux['exact'].add(expr.id)
        return self._jax_aggregation_helper(expr, aux, jnp.argmin)
    
    def _jax_argmax(self, expr, aux):
        aux['exact'].add(expr.id)
        return self._jax_aggregation_helper(expr, aux, jnp.argmax)
    
    # ===========================================================================
    # function
    # ===========================================================================
    
    def _jax_function(self, expr, aux):
        JaxRDDLCompiler._check_valid_op(expr, {'abs', 'sgn', 'round', 'floor', 'ceil', 
                                               'cos', 'sin', 'tan', 'acos', 'asin', 'atan',
                                               'cosh', 'sinh', 'tanh', 'exp', 'ln', 'sqrt',
                                               'lngamma', 'gamma',
                                               'div', 'mod', 'fmod', 'min', 'max', 
                                               'pow', 'log', 'hypot'})
        _, op = expr.etype

        # unary functions
        if op == 'abs':
            return self._jax_abs(expr, aux)
        elif op == 'sgn':
            return self._jax_sgn(expr, aux)
        elif op == 'round':
            return self._jax_round(expr, aux)
        elif op == 'floor':
            return self._jax_floor(expr, aux)
        elif op == 'ceil':
            return self._jax_ceil(expr, aux)
        elif op == 'cos':
            return self._jax_cos(expr, aux)
        elif op == 'sin':
            return self._jax_sin(expr, aux)
        elif op == 'tan':
            return self._jax_tan(expr, aux)
        elif op == 'acos':
            return self._jax_acos(expr, aux)
        elif op == 'asin':
            return self._jax_asin(expr, aux)
        elif op == 'atan':
            return self._jax_atan(expr, aux)
        elif op == 'cosh':
            return self._jax_cosh(expr, aux)
        elif op == 'sinh':
            return self._jax_sinh(expr, aux)
        elif op == 'tanh':
            return self._jax_tanh(expr, aux)
        elif op == 'exp':
            return self._jax_exp(expr, aux)
        elif op == 'ln':
            return self._jax_ln(expr, aux)
        elif op == 'sqrt':
            return self._jax_sqrt(expr, aux)
        elif op == 'lngamma':
            return self._jax_lngamma(expr, aux)
        elif op == 'gamma':
            return self._jax_gamma(expr, aux)
        
        # binary functions
        elif op == 'div':
            return self._jax_div(expr, aux)
        elif op == 'mod':
            return self._jax_mod(expr, aux)
        elif op == 'fmod':
            return self._jax_fmod(expr, aux)
        elif op == 'min':
            return self._jax_min(expr, aux)
        elif op == 'max':
            return self._jax_max(expr, aux)
        elif op == 'pow':
            return self._jax_pow(expr, aux)
        elif op == 'log':
            return self._jax_log(expr, aux)
        elif op == 'hypot':
            return self._jax_hypot(expr, aux)
        
    def _jax_abs(self, expr, aux):
        aux['exact'].add(expr.id)
        return self._jax_unary_helper(expr, aux, jnp.abs, at_least_int=True)
    
    def _jax_sgn(self, expr, aux):
        aux['exact'].add(expr.id)
        return self._jax_unary_helper(expr, aux, jnp.sign, at_least_int=True)
    
    def _jax_round(self, expr, aux):
        aux['exact'].add(expr.id)
        return self._jax_unary_helper(expr, aux, jnp.round, at_least_int=True)
    
    def _jax_floor(self, expr, aux):
        aux['exact'].add(expr.id)
        return self._jax_unary_helper(expr, aux, jnp.floor, at_least_int=True)
    
    def _jax_ceil(self, expr, aux):
        aux['exact'].add(expr.id)
        return self._jax_unary_helper(expr, aux, jnp.ceil, at_least_int=True)
    
    def _jax_cos(self, expr, aux):
        aux['exact'].add(expr.id)
        return self._jax_unary_helper(expr, aux, jnp.cos, at_least_int=True)
    
    def _jax_sin(self, expr, aux):
        aux['exact'].add(expr.id)
        return self._jax_unary_helper(expr, aux, jnp.sin, at_least_int=True)
    
    def _jax_tan(self, expr, aux):
        aux['exact'].add(expr.id)
        return self._jax_unary_helper(expr, aux, jnp.tan, at_least_int=True)
    
    def _jax_acos(self, expr, aux):
        aux['exact'].add(expr.id)
        return self._jax_unary_helper(expr, aux, jnp.arccos, at_least_int=True)
    
    def _jax_asin(self, expr, aux):
        aux['exact'].add(expr.id)
        return self._jax_unary_helper(expr, aux, jnp.arcsin, at_least_int=True)
    
    def _jax_atan(self, expr, aux):
        aux['exact'].add(expr.id)
        return self._jax_unary_helper(expr, aux, jnp.arctan, at_least_int=True)
    
    def _jax_cosh(self, expr, aux):
        aux['exact'].add(expr.id)
        return self._jax_unary_helper(expr, aux, jnp.cosh, at_least_int=True)
    
    def _jax_sinh(self, expr, aux):
        aux['exact'].add(expr.id)
        return self._jax_unary_helper(expr, aux, jnp.sinh, at_least_int=True)
    
    def _jax_tanh(self, expr, aux):
        aux['exact'].add(expr.id)
        return self._jax_unary_helper(expr, aux, jnp.tanh, at_least_int=True)
    
    def _jax_exp(self, expr, aux):
        aux['exact'].add(expr.id)
        return self._jax_unary_helper(expr, aux, jnp.exp, at_least_int=True)
    
    def _jax_ln(self, expr, aux):
        aux['exact'].add(expr.id)
        return self._jax_unary_helper(expr, aux, jnp.ln, at_least_int=True)
    
    def _jax_sqrt(self, expr, aux):
        aux['exact'].add(expr.id)
        return self._jax_unary_helper(expr, aux, jnp.sqrt, at_least_int=True)
    
    def _jax_lngamma(self, expr, aux):
        aux['exact'].add(expr.id)
        return self._jax_unary_helper(expr, aux, scipy.special.gammaln, at_least_int=True)
    
    def _jax_gamma(self, expr, aux):
        aux['exact'].add(expr.id)
        return self._jax_unary_helper(expr, aux, scipy.special.gamma, at_least_int=True)

    def _jax_div(self, expr, aux):
        aux['exact'].add(expr.id)
        return self._jax_binary_helper(expr, aux, jnp.floor_divide, at_least_int=True)
    
    def _jax_mod(self, expr, aux):
        aux['exact'].add(expr.id)
        return self._jax_binary_helper(expr, aux, jnp.mod, at_least_int=True)
    
    def _jax_fmod(self, expr, aux):
        aux['exact'].add(expr.id)
        return self._jax_binary_helper(expr, aux, jnp.fmod, at_least_int=True)
    
    def _jax_min(self, expr, aux):
        aux['exact'].add(expr.id)
        return self._jax_binary_helper(expr, aux, jnp.minimum, at_least_int=True)
    
    def _jax_max(self, expr, aux):
        aux['exact'].add(expr.id)
        return self._jax_binary_helper(expr, aux, jnp.maximum, at_least_int=True)
    
    def _jax_pow(self, expr, aux):
        aux['exact'].add(expr.id)
        return self._jax_binary_helper(expr, aux, jnp.power, at_least_int=True)
    
    def _jax_log(self, expr, aux):
        aux['exact'].add(expr.id)
        def log_op(x, y):
            return jnp.log(x) / jnp.log(y)
        return self._jax_binary_helper(expr, aux, log_op, at_least_int=True)
    
    def _jax_hypot(self, expr, aux):
        aux['exact'].add(expr.id)
        return self._jax_binary_helper(expr, aux, jnp.hypot, at_least_int=True)
    
    # ===========================================================================
    # external function
    # ===========================================================================
    
    def _jax_pyfunc(self, expr, aux):
        aux['exact'].add(expr.id)
        NORMAL = JaxRDDLCompiler.ERROR_CODES['NORMAL']

        # get the Python function by name
        _, pyfunc_name = expr.etype
        pyfunc = self.python_functions.get(pyfunc_name)
        if pyfunc is None:
            raise RDDLUndefinedVariableError(
                f'Undefined external Python function <{pyfunc_name}>, '
                f'must be one of {list(self.python_functions.keys())}.\n' +  
                print_stack_trace(expr))
        
        captured_vars, args = expr.args
        scope_vars = self.traced.cached_objects_in_scope(expr)
        dest_indices = self.traced.cached_sim_info(expr)
        free_vars = [p for p in scope_vars if p[0] not in captured_vars]
        free_dims = self.rddl.object_counts(p for (_, p) in free_vars)
        num_free_vars = len(free_vars)
        captured_types = [t for (p, t) in scope_vars if p in captured_vars]
        require_dims = self.rddl.object_counts(captured_types)
        source_indices = [num_free_vars + i for i in range(len(require_dims))]

        # compile the inputs to the function
        jax_inputs = [self._jax(arg, aux) for arg in args]

        # compile the function evaluation function
        def _jax_wrapped_external_function(fls, nfls, params, key):

            # evaluate inputs to the function
            # first dimensions are non-captured vars in outer scope followed by all the _
            error = NORMAL
            flat_samples = []
            for jax_expr in jax_inputs:
                sample, key, err, params = jax_expr(fls, nfls, params, key)
                new_shape = (-1,) + jnp.shape(sample)[num_free_vars:]
                flat_sample = jnp.reshape(sample, new_shape)
                flat_samples.append(flat_sample)
                error = error | err

            # now all the inputs have dimensions equal to (k,) + the number of _ occurences
            # k is the number of possible non-captured object combinations
            # evaluate the function independently for each combination
            # output dimension for each combination is captured variables (n1, n2, ...)
            # so the total dimension of the output array is (k, n1, n2, ...)
            sample = jax.vmap(pyfunc, in_axes=0)(*flat_samples)
            if not isinstance(sample, jnp.ndarray):
                raise ValueError(
                    f'Output of external Python function <{pyfunc_name}> '
                    f'is not a JAX array.\n' + print_stack_trace(expr)
                )                        
            
            pyfunc_dims = jnp.shape(sample)[1:]
            if len(require_dims) != len(pyfunc_dims):
                raise ValueError(
                    f'External Python function <{pyfunc_name}> returned array with '
                    f'{len(pyfunc_dims)} dimensions, which does not match the '
                    f'number of captured parameter(s) {len(require_dims)}.\n' +  
                    print_stack_trace(expr)
                )
            for (param, require_dim, actual_dim) in zip(captured_vars, require_dims, pyfunc_dims):
                if require_dim != actual_dim:
                    raise ValueError(
                        f'External Python function <{pyfunc_name}> returned array with '
                        f'{actual_dim} elements for captured parameter <{param}>, '
                        f'which does not match the number of objects {require_dim}.\n' + 
                        print_stack_trace(expr)
                    )

            # unravel the combinations k back into their original dimensions
            sample = jnp.reshape(sample, free_dims + pyfunc_dims)
            
            # rearrange the output dimensions to match the outer scope
            sample = jnp.moveaxis(sample, source=source_indices, destination=dest_indices)
            return sample, key, error, params
        
        return _jax_wrapped_external_function

    # ===========================================================================
    # control flow
    # ===========================================================================
    
    def _jax_control(self, expr, aux):
        JaxRDDLCompiler._check_valid_op(expr, {'if', 'switch'})
        _, op = expr.etype        
        if op == 'if':
            return self._jax_if(expr, aux)
        elif op == 'switch':
            return self._jax_switch(expr, aux) 
    
    def _jax_if(self, expr, aux):
        aux['exact'].add(expr.id)
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_CAST']
        JaxRDDLCompiler._check_num_args(expr, 3)
        pred, if_true, if_false = expr.args     
        
        # recursively compile arguments   
        jax_pred = self._jax(pred, aux)
        jax_true = self._jax(if_true, aux)
        jax_false = self._jax(if_false, aux)
        
        def _jax_wrapped_if_then_else(fls, nfls, params, key):
            sample1, key, err1, params = jax_pred(fls, nfls, params, key)
            sample2, key, err2, params = jax_true(fls, nfls, params, key)
            sample3, key, err3, params = jax_false(fls, nfls, params, key)
            sample = jnp.where(sample1 > 0.5, sample2, sample3)
            err = err1 | err2 | err3
            invalid_cast = jnp.logical_not(jnp.can_cast(sample1, jnp.bool_))
            err = err | (invalid_cast * ERR)
            return sample, key, err, params
        return _jax_wrapped_if_then_else
    
    def _jax_switch(self, expr, aux):
        aux['exact'].add(expr.id)

        # recursively compile predicate
        pred = expr.args[0]
        jax_pred = self._jax(pred, aux)
        
        # recursively compile cases
        cases, default = self.traced.cached_sim_info(expr) 
        jax_default = None if default is None else self._jax(default, aux)
        jax_cases = [
            (jax_default if _case is None else self._jax(_case, aux))
            for _case in cases
        ]
                    
        def _jax_wrapped_switch(fls, nfls, params, key):
            
            # sample predicate
            sample_pred, key, err, params = jax_pred(fls, nfls, params, key) 
            
            # sample cases
            sample_cases = []
            for jax_case in jax_cases:
                sample, key, err_case, params = jax_case(fls, nfls, params, key)
                sample_cases.append(sample)
                err = err | err_case      
            sample_cases = jnp.asarray(sample_cases)          
            sample_cases = jnp.asarray(sample_cases, dtype=self._fix_dtype(sample_cases))
            
            # predicate (enum) is an integer - use it to extract from case array
            sample_pred = jnp.asarray(sample_pred[jnp.newaxis, ...], dtype=self.INT)
            sample = jnp.take_along_axis(sample_cases, sample_pred, axis=0)
            assert jnp.shape(sample)[0] == 1
            sample = sample[0, ...]
            return sample, key, err, params
        return _jax_wrapped_switch
    
    # ===========================================================================
    # random variables
    # ===========================================================================
    
    # distributions which seem to support backpropagation (need more testing):
    # Beta
    # Student
    # Gamma
    # ChiSquare   
    # Dirichlet

    # distributions with incomplete reparameterization support (TODO):
    # Multinomial
    
    def _jax_random(self, expr, aux):
        _, name = expr.etype
        if name == 'KronDelta':
            return self._jax_kron(expr, aux)        
        elif name == 'DiracDelta':
            return self._jax_dirac(expr, aux)
        elif name == 'Uniform':
            return self._jax_uniform(expr, aux)
        elif name == 'Bernoulli':
            return self._jax_bernoulli(expr, aux)
        elif name == 'Normal':
            return self._jax_normal(expr, aux)  
        elif name == 'Poisson':
            return self._jax_poisson(expr, aux)
        elif name == 'Exponential':
            return self._jax_exponential(expr, aux)
        elif name == 'Weibull':
            return self._jax_weibull(expr, aux) 
        elif name == 'Gamma':
            return self._jax_gamma(expr, aux)
        elif name == 'Binomial':
            return self._jax_binomial(expr, aux)
        elif name == 'NegativeBinomial':
            return self._jax_negative_binomial(expr, aux)
        elif name == 'Beta':
            return self._jax_beta(expr, aux)
        elif name == 'Geometric':
            return self._jax_geometric(expr, aux)
        elif name == 'Pareto':
            return self._jax_pareto(expr, aux)
        elif name == 'Student':
            return self._jax_student(expr, aux)
        elif name == 'Gumbel':
            return self._jax_gumbel(expr, aux)
        elif name == 'Laplace':
            return self._jax_laplace(expr, aux)
        elif name == 'Cauchy':
            return self._jax_cauchy(expr, aux)
        elif name == 'Gompertz':
            return self._jax_gompertz(expr, aux)
        elif name == 'ChiSquare':
            return self._jax_chisquare(expr, aux)
        elif name == 'Kumaraswamy':
            return self._jax_kumaraswamy(expr, aux)
        elif name == 'Discrete':
            return self._jax_discrete(expr, aux, unnorm=False)
        elif name == 'UnnormDiscrete':
            return self._jax_discrete(expr, aux, unnorm=True)
        elif name == 'Discrete(p)':
            return self._jax_discrete_pvar(expr, aux, unnorm=False)
        elif name == 'UnnormDiscrete(p)':
            return self._jax_discrete_pvar(expr, aux, unnorm=True)
        else:
            raise RDDLNotImplementedError(
                f'Distribution {name} is not supported.\n' + print_stack_trace(expr))
        
    def _jax_kron(self, expr, aux):
        aux['exact'].add(expr.id)
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_KRON_DELTA']
        JaxRDDLCompiler._check_num_args(expr, 1)
        arg, = expr.args
        arg = self._jax(arg, aux)
        
        # just check that the sample can be cast to int
        def _jax_wrapped_distribution_kron(fls, nfls, params, key):
            sample, key, err, params = arg(fls, nfls, params, key)
            invalid_cast = jnp.logical_not(jnp.can_cast(sample, self.INT))
            err = err | (invalid_cast * ERR)
            return sample, key, err, params
        return _jax_wrapped_distribution_kron
    
    def _jax_dirac(self, expr, aux):
        aux['exact'].add(expr.id)
        JaxRDDLCompiler._check_num_args(expr, 1)
        arg, = expr.args
        arg = self._jax(arg, aux, dtype=self.REAL)
        return arg
    
    def _jax_uniform(self, expr, aux):
        aux['exact'].add(expr.id)
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_UNIFORM']
        JaxRDDLCompiler._check_num_args(expr, 2)
        
        arg_lb, arg_ub = expr.args
        jax_lb = self._jax(arg_lb, aux)
        jax_ub = self._jax(arg_ub, aux)
        
        # reparameterization trick U(a, b) = a + (b - a) * U(0, 1)
        def _jax_wrapped_distribution_uniform(fls, nfls, params, key):
            lb, key, err1, params = jax_lb(fls, nfls, params, key)
            ub, key, err2, params = jax_ub(fls, nfls, params, key)
            key, subkey = random.split(key)
            U = random.uniform(key=subkey, shape=jnp.shape(lb), dtype=self.REAL)
            sample = lb + (ub - lb) * U
            out_of_bounds = jnp.logical_not(jnp.all(lb <= ub))
            err = err1 | err2 | (out_of_bounds * ERR)
            return sample, key, err, params        
        return _jax_wrapped_distribution_uniform
    
    def _jax_normal(self, expr, aux):
        aux['exact'].add(expr.id)
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_NORMAL']
        JaxRDDLCompiler._check_num_args(expr, 2)
        
        arg_mean, arg_var = expr.args
        jax_mean = self._jax(arg_mean, aux)
        jax_var = self._jax(arg_var, aux)
        
        # reparameterization trick N(m, s^2) = m + s * N(0, 1)
        def _jax_wrapped_distribution_normal(fls, nfls, params, key):
            mean, key, err1, params = jax_mean(fls, nfls, params, key)
            var, key, err2, params = jax_var(fls, nfls, params, key)
            std = jnp.sqrt(var)
            key, subkey = random.split(key)
            Z = random.normal(key=subkey, shape=jnp.shape(mean), dtype=self.REAL)
            sample = mean + std * Z
            out_of_bounds = jnp.logical_not(jnp.all(var >= 0))
            err = err1 | err2 | (out_of_bounds * ERR)
            return sample, key, err, params        
        return _jax_wrapped_distribution_normal
    
    def _jax_exponential(self, expr, aux):
        aux['exact'].add(expr.id)
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_EXPONENTIAL']
        JaxRDDLCompiler._check_num_args(expr, 1)
        
        arg_scale, = expr.args
        jax_scale = self._jax(arg_scale, aux)
        
        # reparameterization trick Exp(s) = s * Exp(1)
        def _jax_wrapped_distribution_exp(fls, nfls, params, key):
            scale, key, err, params = jax_scale(fls, nfls, params, key)
            key, subkey = random.split(key)
            exp = random.exponential(key=subkey, shape=jnp.shape(scale), dtype=self.REAL)
            sample = scale * exp
            out_of_bounds = jnp.logical_not(jnp.all(scale > 0))
            err = err | (out_of_bounds * ERR)
            return sample, key, err, params       
        return _jax_wrapped_distribution_exp
    
    def _jax_weibull(self, expr, aux):
        aux['exact'].add(expr.id)
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_WEIBULL']
        JaxRDDLCompiler._check_num_args(expr, 2)
        
        arg_shape, arg_scale = expr.args
        jax_shape = self._jax(arg_shape, aux)
        jax_scale = self._jax(arg_scale, aux)
        
        # reparameterization trick W(s, r) = r * (-ln(1 - U(0, 1))) ** (1 / s)
        def _jax_wrapped_distribution_weibull(fls, nfls, params, key):
            shape, key, err1, params = jax_shape(fls, nfls, params, key)
            scale, key, err2, params = jax_scale(fls, nfls, params, key)
            key, subkey = random.split(key)
            sample = random.weibull_min(
                key=subkey, scale=scale, concentration=shape, dtype=self.REAL)
            out_of_bounds = jnp.logical_not(jnp.all(jnp.logical_and(shape > 0, scale > 0)))
            err = err1 | err2 | (out_of_bounds * ERR)
            return sample, key, err, params        
        return _jax_wrapped_distribution_weibull
    
    def _jax_bernoulli(self, expr, aux):
        aux['exact'].add(expr.id)
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_BERNOULLI']
        JaxRDDLCompiler._check_num_args(expr, 1)
        arg_prob, = expr.args
        
        # recursively compile arguments
        jax_prob = self._jax(arg_prob, aux)
        
        def _jax_wrapped_distribution_bernoulli(fls, nfls, params, key):
            prob, key, err, params = jax_prob(fls, nfls, params, key)
            key, subkey = random.split(key)
            sample = random.bernoulli(subkey, prob)
            out_of_bounds = jnp.logical_not(jnp.all(jnp.logical_and(prob >= 0, prob <= 1)))
            err = err | (out_of_bounds * ERR)
            return sample, key, err, params        
        return _jax_wrapped_distribution_bernoulli
    
    def _jax_poisson(self, expr, aux):
        aux['exact'].add(expr.id)
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_POISSON']
        JaxRDDLCompiler._check_num_args(expr, 1)
        arg_rate, = expr.args
        
        # recursively compile arguments
        jax_rate = self._jax(arg_rate, aux)
        
        # uses the implicit JAX subroutine
        def _jax_wrapped_distribution_poisson(fls, nfls, params, key):
            rate, key, err, params = jax_rate(fls, nfls, params, key)
            key, subkey = random.split(key)
            sample = random.poisson(key=subkey, lam=rate, dtype=self.INT)
            out_of_bounds = jnp.logical_not(jnp.all(rate >= 0))
            err = err | (out_of_bounds * ERR)
            return sample, key, err, params        
        return _jax_wrapped_distribution_poisson
    
    def _jax_gamma(self, expr, aux):
        aux['exact'].add(expr.id)
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_GAMMA']
        JaxRDDLCompiler._check_num_args(expr, 2)
        
        arg_shape, arg_scale = expr.args
        jax_shape = self._jax(arg_shape, aux)
        jax_scale = self._jax(arg_scale, aux)
        
        # partial reparameterization trick Gamma(s, r) = r * Gamma(s, 1)
        # uses the implicit JAX subroutine for Gamma(s, 1) 
        def _jax_wrapped_distribution_gamma(fls, nfls, params, key):
            shape, key, err1, params = jax_shape(fls, nfls, params, key)
            scale, key, err2, params = jax_scale(fls, nfls, params, key)
            key, subkey = random.split(key)
            Gamma = random.gamma(key=subkey, a=shape, dtype=self.REAL)
            sample = scale * Gamma
            out_of_bounds = jnp.logical_not(jnp.all(jnp.logical_and(shape > 0, scale > 0)))
            err = err1 | err2 | (out_of_bounds * ERR)
            return sample, key, err, params        
        return _jax_wrapped_distribution_gamma
    
    def _jax_binomial(self, expr, aux):
        aux['exact'].add(expr.id)
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_BINOMIAL']
        JaxRDDLCompiler._check_num_args(expr, 2)
        arg_trials, arg_prob = expr.args

        jax_trials = self._jax(arg_trials, aux)
        jax_prob = self._jax(arg_prob, aux)

        # uses reduction for constant trials
        def _jax_wrapped_distribution_binomial(fls, nfls, params, key):
            trials, key, err2, params = jax_trials(fls, nfls, params, key)       
            prob, key, err1, params = jax_prob(fls, nfls, params, key)
            key, subkey = random.split(key)
            trials = jnp.asarray(trials, dtype=self.REAL)
            prob = jnp.asarray(prob, dtype=self.REAL)
            sample = random.binomial(key=subkey, n=trials, p=prob, dtype=self.REAL)
            sample = jnp.asarray(sample, dtype=self.INT)
            out_of_bounds = jnp.logical_not(jnp.all(
                jnp.logical_and(jnp.logical_and(prob >= 0, prob <= 1), trials >= 0)))
            err = err1 | err2 | (out_of_bounds * ERR)
            return sample, key, err, params        
        return _jax_wrapped_distribution_binomial
    
    def _jax_negative_binomial(self, expr, aux):
        aux['exact'].add(expr.id)
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_NEGATIVE_BINOMIAL']
        JaxRDDLCompiler._check_num_args(expr, 2)
        arg_trials, arg_prob = expr.args

        jax_trials = self._jax(arg_trials, aux)
        jax_prob = self._jax(arg_prob, aux)
        
        # uses tensorflow-probability
        def _jax_wrapped_distribution_negative_binomial(fls, nfls, params, key):
            trials, key, err2, params = jax_trials(fls, nfls, params, key)       
            prob, key, err1, params = jax_prob(fls, nfls, params, key)
            key, subkey = random.split(key)
            trials = jnp.asarray(trials, dtype=self.REAL)
            prob = jnp.asarray(prob, dtype=self.REAL)
            dist = tfp.distributions.NegativeBinomial(total_count=trials, probs=1. - prob)
            sample = jnp.asarray(dist.sample(seed=subkey), dtype=self.INT)
            out_of_bounds = jnp.logical_not(jnp.all(
                jnp.logical_and(jnp.logical_and(prob >= 0, prob <= 1), trials > 0)))
            err = err1 | err2 | (out_of_bounds * ERR)
            return sample, key, err, params        
        return _jax_wrapped_distribution_negative_binomial    
        
    def _jax_beta(self, expr, aux):
        aux['exact'].add(expr.id)
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_BETA']
        JaxRDDLCompiler._check_num_args(expr, 2)
        
        arg_shape, arg_rate = expr.args
        jax_shape = self._jax(arg_shape, aux)
        jax_rate = self._jax(arg_rate, aux)
        
        # uses the implicit JAX subroutine
        def _jax_wrapped_distribution_beta(fls, nfls, params, key):
            shape, key, err1, params = jax_shape(fls, nfls, params, key)
            rate, key, err2, params = jax_rate(fls, nfls, params, key)
            key, subkey = random.split(key)
            sample = random.beta(key=subkey, a=shape, b=rate, dtype=self.REAL)
            out_of_bounds = jnp.logical_not(jnp.all(jnp.logical_and(shape > 0, rate > 0)))
            err = err1 | err2 | (out_of_bounds * ERR)
            return sample, key, err, params        
        return _jax_wrapped_distribution_beta
    
    def _jax_geometric(self, expr, aux):
        aux['exact'].add(expr.id)
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_GEOMETRIC']
        JaxRDDLCompiler._check_num_args(expr, 1)        
        arg_prob, = expr.args
        
        # recursively compile arguments        
        jax_prob = self._jax(arg_prob, aux)
        
        def _jax_wrapped_distribution_geometric(fls, nfls, params, key):
            prob, key, err, params = jax_prob(fls, nfls, params, key)
            key, subkey = random.split(key)
            sample = random.geometric(key=subkey, p=prob, dtype=self.INT)
            out_of_bounds = jnp.logical_not(jnp.all(jnp.logical_and(prob >= 0, prob <= 1)))
            err = err | (out_of_bounds * ERR)
            return sample, key, err, params        
        return _jax_wrapped_distribution_geometric
    
    def _jax_pareto(self, expr, aux):
        aux['exact'].add(expr.id)
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_PARETO']
        JaxRDDLCompiler._check_num_args(expr, 2)
        
        arg_shape, arg_scale = expr.args
        jax_shape = self._jax(arg_shape, aux)
        jax_scale = self._jax(arg_scale, aux)
        
        # partial reparameterization trick Pareto(s, r) = r * Pareto(s, 1)
        # uses the implicit JAX subroutine for Pareto(s, 1) 
        def _jax_wrapped_distribution_pareto(fls, nfls, params, key):
            shape, key, err1, params = jax_shape(fls, nfls, params, key)
            scale, key, err2, params = jax_scale(fls, nfls, params, key)
            key, subkey = random.split(key)
            sample = scale * random.pareto(key=subkey, b=shape, dtype=self.REAL)
            out_of_bounds = jnp.logical_not(jnp.all(jnp.logical_and(shape > 0, scale > 0)))
            err = err1 | err2 | (out_of_bounds * ERR)
            return sample, key, err, params        
        return _jax_wrapped_distribution_pareto
    
    def _jax_student(self, expr, aux):
        aux['exact'].add(expr.id)
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_STUDENT']
        JaxRDDLCompiler._check_num_args(expr, 1)
        
        arg_df, = expr.args
        jax_df = self._jax(arg_df, aux)
        
        # uses the implicit JAX subroutine for student(df)
        def _jax_wrapped_distribution_t(fls, nfls, params, key):
            df, key, err, params = jax_df(fls, nfls, params, key)
            key, subkey = random.split(key)
            sample = random.t(key=subkey, df=df, shape=jnp.shape(df), dtype=self.REAL)
            out_of_bounds = jnp.logical_not(jnp.all(df > 0))
            err = err | (out_of_bounds * ERR)
            return sample, key, err, params        
        return _jax_wrapped_distribution_t
    
    def _jax_gumbel(self, expr, aux):
        aux['exact'].add(expr.id)
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_GUMBEL']
        JaxRDDLCompiler._check_num_args(expr, 2)
        
        arg_mean, arg_scale = expr.args
        jax_mean = self._jax(arg_mean, aux)
        jax_scale = self._jax(arg_scale, aux)
        
        # reparameterization trick Gumbel(m, s) = m + s * Gumbel(0, 1)
        def _jax_wrapped_distribution_gumbel(fls, nfls, params, key):
            mean, key, err1, params = jax_mean(fls, nfls, params, key)
            scale, key, err2, params = jax_scale(fls, nfls, params, key)
            key, subkey = random.split(key)
            g = random.gumbel(key=subkey, shape=jnp.shape(mean), dtype=self.REAL)
            sample = mean + scale * g
            out_of_bounds = jnp.logical_not(jnp.all(scale > 0))
            err = err1 | err2 | (out_of_bounds * ERR)
            return sample, key, err, params        
        return _jax_wrapped_distribution_gumbel
    
    def _jax_laplace(self, expr, aux):
        aux['exact'].add(expr.id)
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_LAPLACE']
        JaxRDDLCompiler._check_num_args(expr, 2)
        
        arg_mean, arg_scale = expr.args
        jax_mean = self._jax(arg_mean, aux)
        jax_scale = self._jax(arg_scale, aux)
        
        # reparameterization trick Laplace(m, s) = m + s * Laplace(0, 1)
        def _jax_wrapped_distribution_laplace(fls, nfls, params, key):
            mean, key, err1, params = jax_mean(fls, nfls, params, key)
            scale, key, err2, params = jax_scale(fls, nfls, params, key)
            key, subkey = random.split(key)
            lp = random.laplace(key=subkey, shape=jnp.shape(mean), dtype=self.REAL)
            sample = mean + scale * lp
            out_of_bounds = jnp.logical_not(jnp.all(scale > 0))
            err = err1 | err2 | (out_of_bounds * ERR)
            return sample, key, err, params        
        return _jax_wrapped_distribution_laplace
    
    def _jax_cauchy(self, expr, aux):
        aux['exact'].add(expr.id)
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_CAUCHY']
        JaxRDDLCompiler._check_num_args(expr, 2)
        
        arg_mean, arg_scale = expr.args
        jax_mean = self._jax(arg_mean, aux)
        jax_scale = self._jax(arg_scale, aux)
        
        # reparameterization trick Cauchy(m, s) = m + s * Cauchy(0, 1)
        def _jax_wrapped_distribution_cauchy(fls, nfls, params, key):
            mean, key, err1, params = jax_mean(fls, nfls, params, key)
            scale, key, err2, params = jax_scale(fls, nfls, params, key)
            key, subkey = random.split(key)
            cauchy = random.cauchy(key=subkey, shape=jnp.shape(mean), dtype=self.REAL)
            sample = mean + scale * cauchy
            out_of_bounds = jnp.logical_not(jnp.all(scale > 0))
            err = err1 | err2 | (out_of_bounds * ERR)
            return sample, key, err, params        
        return _jax_wrapped_distribution_cauchy
    
    def _jax_gompertz(self, expr, aux):
        aux['exact'].add(expr.id)
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_GOMPERTZ']
        JaxRDDLCompiler._check_num_args(expr, 2)
        
        arg_shape, arg_scale = expr.args
        jax_shape = self._jax(arg_shape, aux)
        jax_scale = self._jax(arg_scale, aux)
        
        # reparameterization trick Gompertz(s, r) = ln(1 - log(U(0, 1)) / s) / r
        def _jax_wrapped_distribution_gompertz(fls, nfls, params, key):
            shape, key, err1, params = jax_shape(fls, nfls, params, key)
            scale, key, err2, params = jax_scale(fls, nfls, params, key)
            key, subkey = random.split(key)
            U = random.uniform(key=subkey, shape=jnp.shape(scale), dtype=self.REAL)
            sample = jnp.log(1.0 - jnp.log1p(-U) / shape) / scale
            out_of_bounds = jnp.logical_not(jnp.all(jnp.logical_and(shape > 0, scale > 0)))
            err = err1 | err2 | (out_of_bounds * ERR)
            return sample, key, err, params        
        return _jax_wrapped_distribution_gompertz
    
    def _jax_chisquare(self, expr, aux):
        aux['exact'].add(expr.id)
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_CHISQUARE']
        JaxRDDLCompiler._check_num_args(expr, 1)
        
        arg_df, = expr.args
        jax_df = self._jax(arg_df, aux)
        
        # use the fact that ChiSquare(df) = Gamma(df/2, 2)
        def _jax_wrapped_distribution_chisquare(fls, nfls, params, key):
            df, key, err1, params = jax_df(fls, nfls, params, key)
            key, subkey = random.split(key)
            shape = 0.5 * df
            sample = 2.0 * random.gamma(key=subkey, a=shape, dtype=self.REAL)
            out_of_bounds = jnp.logical_not(jnp.all(df > 0))
            err = err1 | (out_of_bounds * ERR)
            return sample, key, err, params        
        return _jax_wrapped_distribution_chisquare
    
    def _jax_kumaraswamy(self, expr, aux):
        aux['exact'].add(expr.id)
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_KUMARASWAMY']
        JaxRDDLCompiler._check_num_args(expr, 2)
        
        arg_a, arg_b = expr.args
        jax_a = self._jax(arg_a, aux)
        jax_b = self._jax(arg_b, aux)
        
        # uses the reparameterization K(a, b) = (1 - (1 - U(0, 1))^{1/b})^{1/a}
        def _jax_wrapped_distribution_kumaraswamy(fls, nfls, params, key):
            a, key, err1, params = jax_a(fls, nfls, params, key)
            b, key, err2, params = jax_b(fls, nfls, params, key)
            key, subkey = random.split(key)
            U = random.uniform(key=subkey, shape=jnp.shape(a), dtype=self.REAL)            
            sample = jnp.power(1.0 - jnp.power(U, 1.0 / b), 1.0 / a)
            out_of_bounds = jnp.logical_not(jnp.all(jnp.logical_and(a > 0, b > 0)))
            err = err1 | err2 | (out_of_bounds * ERR)
            return sample, key, err, params        
        return _jax_wrapped_distribution_kumaraswamy
    
    # ===========================================================================
    # random variables with enum support
    # ===========================================================================
    
    @staticmethod
    def _jax_update_discrete_oob_error(err, prob):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_DISCRETE']
        out_of_bounds = jnp.logical_not(jnp.logical_and(
            jnp.all(prob >= 0),
            jnp.allclose(jnp.sum(prob, axis=-1), 1.0)
        ))
        error = err | (out_of_bounds * ERR)
        return error
    
    def _jax_discrete_prob(self, jax_probs, unnormalized):
        def _jax_wrapped_calc_discrete_prob(fls, nfls, params, key):

            # calculate probability expressions
            error = JaxRDDLCompiler.ERROR_CODES['NORMAL']
            prob = []
            for jax_prob in jax_probs:
                sample, key, error_pdf, params = jax_prob(fls, nfls, params, key)
                prob.append(sample)
                error = error | error_pdf
            prob = jnp.stack(prob, axis=-1)

            # normalize them if required
            if unnormalized:
                normalizer = jnp.sum(prob, axis=-1, keepdims=True)
                prob = prob / normalizer
            return prob, key, error, params
        return _jax_wrapped_calc_discrete_prob
    
    def _jax_discrete(self, expr, aux, unnorm):
        aux['exact'].add(expr.id)
        ordered_args = self.traced.cached_sim_info(expr)
        jax_probs = [self._jax(arg, aux) for arg in ordered_args]
        prob_fn = self._jax_discrete_prob(jax_probs, unnorm)
        
        def _jax_wrapped_distribution_discrete(fls, nfls, params, key):
            prob, key, error, params = prob_fn(fls, nfls, params, key)
            key, subkey = random.split(key)
            sample = random.categorical(key=subkey, logits=jnp.log(prob), axis=-1)
            error = JaxRDDLCompiler._jax_update_discrete_oob_error(error, prob)
            return sample, key, error, params        
        return _jax_wrapped_distribution_discrete

    @staticmethod
    def _jax_discrete_pvar_prob(jax_probs, unnormalized):
        def _jax_wrapped_calc_discrete_prob(fls, nfls, params, key):
            prob, key, error, params = jax_probs(fls, nfls, params, key)
            if unnormalized:
                normalizer = jnp.sum(prob, axis=-1, keepdims=True)
                prob = prob / normalizer
            return prob, key, error, params
        return _jax_wrapped_calc_discrete_prob

    def _jax_discrete_pvar(self, expr, aux, unnorm):
        aux['exact'].add(expr.id)
        JaxRDDLCompiler._check_num_args(expr, 2)
        _, args = expr.args
        arg, = args
        jax_probs = self._jax(arg, aux)
        prob_fn = self._jax_discrete_pvar_prob(jax_probs, unnorm)

        def _jax_wrapped_distribution_discrete_pvar(fls, nfls, params, key):
            prob, key, error, params = prob_fn(fls, nfls, params, key)
            key, subkey = random.split(key)
            sample = random.categorical(key=subkey, logits=jnp.log(prob), axis=-1)
            error = JaxRDDLCompiler._jax_update_discrete_oob_error(error, prob)
            return sample, key, error, params        
        return _jax_wrapped_distribution_discrete_pvar

    # ===========================================================================
    # random vectors
    # ===========================================================================
    
    def _jax_random_vector(self, expr, aux):
        _, name = expr.etype
        if name == 'MultivariateNormal':
            return self._jax_multivariate_normal(expr, aux)   
        elif name == 'MultivariateStudent':
            return self._jax_multivariate_student(expr, aux)  
        elif name == 'Dirichlet':
            return self._jax_dirichlet(expr, aux)
        elif name == 'Multinomial':
            return self._jax_multinomial(expr, aux)
        else:
            raise RDDLNotImplementedError(
                f'Distribution {name} is not supported.\n' + print_stack_trace(expr))
    
    def _jax_multivariate_normal(self, expr, aux): 
        aux['exact'].add(expr.id)
        _, args = expr.args
        mean, cov = args
        jax_mean = self._jax(mean, aux)
        jax_cov = self._jax(cov, aux)
        index, = self.traced.cached_sim_info(expr)
        
        # reparameterization trick MN(m, LL') = LZ + m, where Z ~ Normal(0, 1)
        def _jax_wrapped_distribution_multivariate_normal(fls, nfls, params, key):
            
            # sample the mean and covariance
            sample_mean, key, err1, params = jax_mean(fls, nfls, params, key)
            sample_cov, key, err2, params = jax_cov(fls, nfls, params, key)
            
            # sample Normal(0, 1)
            key, subkey = random.split(key)
            Z = random.normal(
                key=subkey, shape=jnp.shape(sample_mean) + (1,), dtype=self.REAL)       
            
            # compute L s.t. cov = L * L' and reparameterize
            L = jnp.linalg.cholesky(sample_cov)
            sample = jnp.matmul(L, Z)[..., 0] + sample_mean
            sample = jnp.moveaxis(sample, source=-1, destination=index)
            err = err1 | err2
            return sample, key, err, params        
        return _jax_wrapped_distribution_multivariate_normal
    
    def _jax_multivariate_student(self, expr, aux):
        aux['exact'].add(expr.id)
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_MULTIVARIATE_STUDENT']
        
        _, args = expr.args
        mean, cov, df = args
        jax_mean = self._jax(mean, aux)
        jax_cov = self._jax(cov, aux)
        jax_df = self._jax(df, aux)
        index, = self.traced.cached_sim_info(expr)
        
        # reparameterization trick MN(m, LL') = LZ + m, where Z ~ StudentT(0, 1)
        def _jax_wrapped_distribution_multivariate_student(fls, nfls, params, key):
            
            # sample the mean and covariance and degrees of freedom
            sample_mean, key, err1, params = jax_mean(fls, nfls, params, key)
            sample_cov, key, err2, params = jax_cov(fls, nfls, params, key)
            sample_df, key, err3, params = jax_df(fls, nfls, params, key)
            out_of_bounds = jnp.logical_not(jnp.all(sample_df > 0))
            
            # sample StudentT(0, 1, df) -- broadcast df to same shape as cov
            sample_df = sample_df[..., jnp.newaxis, jnp.newaxis]
            sample_df = jnp.broadcast_to(sample_df, shape=jnp.shape(sample_mean) + (1,))
            key, subkey = random.split(key)
            Z = random.t(
                key=subkey, 
                df=sample_df, 
                shape=jnp.shape(sample_df),
                dtype=self.REAL
            )   
            
            # compute L s.t. cov = L * L' and reparameterize
            L = jnp.linalg.cholesky(sample_cov)
            sample = jnp.matmul(L, Z)[..., 0] + sample_mean
            sample = jnp.moveaxis(sample, source=-1, destination=index)
            error = err1 | err2 | err3 | (out_of_bounds * ERR)
            return sample, key, error, params        
        return _jax_wrapped_distribution_multivariate_student
    
    def _jax_dirichlet(self, expr, aux):
        aux['exact'].add(expr.id)
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_DIRICHLET']
        
        _, args = expr.args
        alpha, = args
        jax_alpha = self._jax(alpha, aux)
        index, = self.traced.cached_sim_info(expr)
        
        # sample Gamma(alpha_i, 1) and normalize across i
        def _jax_wrapped_distribution_dirichlet(fls, nfls, params, key):
            alpha, key, error, params = jax_alpha(fls, nfls, params, key)
            out_of_bounds = jnp.logical_not(jnp.all(alpha > 0))
            error = error | (out_of_bounds * ERR)
            key, subkey = random.split(key)
            gamma = random.gamma(key=subkey, a=alpha, dtype=self.REAL)
            sample = gamma / jnp.sum(gamma, axis=-1, keepdims=True)
            sample = jnp.moveaxis(sample, source=-1, destination=index)
            return sample, key, error, params        
        return _jax_wrapped_distribution_dirichlet
    
    def _jax_multinomial(self, expr, aux):
        aux['exact'].add(expr.id)
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_MULTINOMIAL']
        
        _, args = expr.args
        trials, prob = args
        jax_trials = self._jax(trials, aux)
        jax_prob = self._jax(prob, aux)
        index, = self.traced.cached_sim_info(expr)
        
        def _jax_wrapped_distribution_multinomial(fls, nfls, params, key):
            trials, key, err1, params = jax_trials(fls, nfls, params, key)
            prob, key, err2, params = jax_prob(fls, nfls, params, key)
            trials = jnp.asarray(trials, dtype=self.REAL)
            prob = jnp.asarray(prob, dtype=self.REAL)
            key, subkey = random.split(key)
            dist = tfp.distributions.Multinomial(total_count=trials, probs=prob)
            sample = jnp.asarray(dist.sample(seed=subkey), dtype=self.INT)
            sample = jnp.moveaxis(sample, source=-1, destination=index)
            out_of_bounds = jnp.logical_not(jnp.all(
                jnp.logical_and(
                    jnp.logical_and(prob >= 0, jnp.allclose(jnp.sum(prob, axis=-1), 1.)),
                    trials >= 0
                )
            ))
            error = err1 | err2 | (out_of_bounds * ERR)
            return sample, key, error, params                  
        return _jax_wrapped_distribution_multinomial
    
    # ===========================================================================
    # matrix algebra
    # ===========================================================================
    
    def _jax_matrix(self, expr, aux):
        _, op = expr.etype
        if op == 'det':
            return self._jax_matrix_det(expr, aux)
        elif op == 'inverse':
            return self._jax_matrix_inv(expr, aux, pseudo=False)
        elif op == 'pinverse':
            return self._jax_matrix_inv(expr, aux, pseudo=True)
        elif op == 'cholesky':
            return self._jax_matrix_cholesky(expr, aux)
        else:
            raise RDDLNotImplementedError(
                f'Matrix operation {op} is not supported.\n' + print_stack_trace(expr))
    
    def _jax_matrix_det(self, expr, aux):
        aux['exact'].add(expr.id)
        arg = expr.args[-1]
        jax_arg = self._jax(arg, aux)
        
        def _jax_wrapped_matrix_operation_det(fls, nfls, params, key):
            sample_arg, key, error, params = jax_arg(fls, nfls, params, key)
            sample = jnp.linalg.det(sample_arg)
            return sample, key, error, params        
        return _jax_wrapped_matrix_operation_det
    
    def _jax_matrix_inv(self, expr, aux, pseudo):
        aux['exact'].add(expr.id)
        _, arg = expr.args
        jax_arg = self._jax(arg, aux)
        indices = self.traced.cached_sim_info(expr)
        op = jnp.linalg.pinv if pseudo else jnp.linalg.inv
        
        def _jax_wrapped_matrix_operation_inv(fls, nfls, params, key):
            sample_arg, key, error, params = jax_arg(fls, nfls, params, key)
            sample = op(sample_arg)
            sample = jnp.moveaxis(sample, source=(-2, -1), destination=indices)
            return sample, key, error, params        
        return _jax_wrapped_matrix_operation_inv
    
    def _jax_matrix_cholesky(self, expr, aux):
        aux['exact'].add(expr.id)
        _, arg = expr.args
        jax_arg = self._jax(arg, aux)
        indices = self.traced.cached_sim_info(expr)
        op = jnp.linalg.cholesky
        
        def _jax_wrapped_matrix_operation_cholesky(fls, nfls, params, key):
            sample_arg, key, error, params = jax_arg(fls, nfls, params, key)
            sample = op(sample_arg)
            sample = jnp.moveaxis(sample, source=(-2, -1), destination=indices)
            return sample, key, error, params        
        return _jax_wrapped_matrix_operation_cholesky

            