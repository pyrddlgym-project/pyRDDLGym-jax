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
import traceback
from typing import Any, Callable, Dict, List, Optional

import jax
import jax.numpy as jnp
import jax.random as random

from pyRDDLGym.core.compiler.initializer import RDDLValueInitializer
from pyRDDLGym.core.compiler.levels import RDDLLevelAnalysis
from pyRDDLGym.core.compiler.model import RDDLLiftedModel
from pyRDDLGym.core.compiler.tracer import RDDLObjectsTracer
from pyRDDLGym.core.constraints import RDDLConstraints
from pyRDDLGym.core.debug.exception import (
    print_stack_trace, 
    raise_warning,
    RDDLInvalidNumberOfArgumentsError, 
    RDDLNotImplementedError
)
from pyRDDLGym.core.debug.logger import Logger
from pyRDDLGym.core.simulator import RDDLSimulatorPrecompiled

from pyRDDLGym_jax.core.logic import ExactLogic

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
    
    def __init__(self, rddl: RDDLLiftedModel,
                 allow_synchronous_state: bool=True,
                 logger: Optional[Logger]=None,
                 use64bit: bool=False,
                 compile_non_fluent_exact: bool=True,
                 python_functions: Optional[Dict[str, Callable]]=None) -> None:
        '''Creates a new RDDL to Jax compiler.
        
        :param rddl: the RDDL model to compile into Jax
        :param allow_synchronous_state: whether next-state components can depend
        on each other
        :param logger: to log information about compilation to file
        :param use64bit: whether to use 64 bit arithmetic
        :param compile_non_fluent_exact: whether non-fluent expressions 
        are always compiled using exact JAX expressions
        :param python_functions: dictionary of external Python functions to call from RDDL
        '''
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
        self.ONE = jnp.asarray(1, dtype=self.INT)
        self.JAX_TYPES = {
            'int': self.INT,
            'real': self.REAL,
            'bool': bool
        }

        # compile initial values
        initializer = RDDLValueInitializer(rddl)
        self.init_values = initializer.initialize()
        
        # compute dependency graph for CPFs and sort them by evaluation order
        sorter = RDDLLevelAnalysis(rddl, allow_synchronous_state=allow_synchronous_state)
        self.levels = sorter.compute_levels()        
        
        # trace expressions to cache information to be used later
        tracer = RDDLObjectsTracer(rddl, cpf_levels=self.levels)
        self.traced = tracer.trace()
        
        # extract the box constraints on actions
        if python_functions is None:
            python_functions = {}
        self.python_functions = python_functions
        simulator = RDDLSimulatorPrecompiled(
            rddl=self.rddl,
            init_values=self.init_values,
            levels=self.levels,
            trace_info=self.traced,
            python_functions=python_functions
        )  
        constraints = RDDLConstraints(simulator, vectorized=True)
        self.constraints = constraints
        
        # basic operations - these can be override in subclasses
        self.compile_non_fluent_exact = compile_non_fluent_exact
        self.AGGREGATION_BOOL = {'forall', 'exists'}
        self.EXACT_OPS = ExactLogic(use64bit=self.use64bit).get_operator_dicts()
        self.OPS = self.EXACT_OPS
        
    # ===========================================================================
    # main compilation subroutines
    # ===========================================================================
     
    def compile(self, log_jax_expr: bool=False, heading: str='') -> None: 
        '''Compiles the current RDDL into Jax expressions.
        
        :param log_jax_expr: whether to pretty-print the compiled Jax functions
        to the log file
        :param heading: the heading to print before compilation information
        '''
        init_params = {}
        self.invariants = self._compile_constraints(self.rddl.invariants, init_params)
        self.preconditions = self._compile_constraints(self.rddl.preconditions, init_params)
        self.terminations = self._compile_constraints(self.rddl.terminations, init_params)
        self.cpfs = self._compile_cpfs(init_params)
        self.reward = self._compile_reward(init_params)
        self.model_params = init_params
        
        if log_jax_expr and self.logger is not None:
            printed = self.print_jax()
            printed_cpfs = '\n\n'.join(f'{k}: {v}' 
                                       for (k, v) in printed['cpfs'].items())
            printed_reward = printed['reward']
            printed_invariants = '\n\n'.join(v for v in printed['invariants'])
            printed_preconds = '\n\n'.join(v for v in printed['preconditions'])
            printed_terminals = '\n\n'.join(v for v in printed['terminations'])
            printed_params = '\n'.join(f'{k}: {v}' for (k, v) in init_params.items())
            message = (
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
            self.logger.log(message)
    
    def _compile_constraints(self, constraints, init_params):
        return [self._jax(expr, init_params, dtype=bool) for expr in constraints]
        
    def _compile_cpfs(self, init_params):
        jax_cpfs = {}
        for cpfs in self.levels.values():
            for cpf in cpfs:
                _, expr = self.rddl.cpfs[cpf]
                prange = self.rddl.variable_ranges[cpf]
                dtype = self.JAX_TYPES.get(prange, self.INT)
                jax_cpfs[cpf] = self._jax(expr, init_params, dtype=dtype)
        return jax_cpfs
    
    def _compile_reward(self, init_params):
        return self._jax(self.rddl.reward, init_params, dtype=self.REAL)
    
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
            
    def _jax_nonlinear_constraints(self, init_params): 
        rddl = self.rddl
        
        # extract the non-box inequality constraints on actions
        inequalities = [constr 
                        for (i, expr) in enumerate(rddl.preconditions)
                        for constr in self._extract_inequality_constraint(expr)
                        if not self.constraints.is_box_preconditions[i]]
        
        # compile them to JAX and write as h(s, a) <= 0
        jax_op = ExactLogic.exact_binary_function(jnp.subtract)    
        jax_inequalities = []
        for (left, right) in inequalities:
            jax_lhs = self._jax(left, init_params)
            jax_rhs = self._jax(right, init_params)
            jax_constr = self._jax_binary(jax_lhs, jax_rhs, jax_op, at_least_int=True)
            jax_inequalities.append(jax_constr)
        
        # extract the non-box equality constraints on actions
        equalities = [constr 
                      for (i, expr) in enumerate(rddl.preconditions)
                      for constr in self._extract_equality_constraint(expr)
                      if not self.constraints.is_box_preconditions[i]]
        
        # compile them to JAX and write as g(s, a) == 0
        jax_equalities = []
        for (left, right) in equalities:
            jax_lhs = self._jax(left, init_params)
            jax_rhs = self._jax(right, init_params)
            jax_constr = self._jax_binary(jax_lhs, jax_rhs, jax_op, at_least_int=True)
            jax_equalities.append(jax_constr)
            
        return jax_inequalities, jax_equalities
    
    def compile_transition(self, check_constraints: bool=False,
                           constraint_func: bool=False, 
                           init_params_constr: Dict[str, Any]={},
                           cache_path_info: bool=False) -> Callable:
        '''Compiles the current RDDL model into a JAX transition function that 
        samples the next state.
        
        The arguments of the returned function is:
            - key is the PRNG key
            - actions is the dict of action tensors
            - subs is the dict of current pvar value tensors
            - model_params is a dict of parameters for the relaxed model.
        
        The returned value of the function is:
            - subs is the returned next epoch fluent values
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
        '''
        NORMAL = JaxRDDLCompiler.ERROR_CODES['NORMAL']        
        rddl = self.rddl
        reward_fn, cpfs, preconds, invariants, terminals = \
            self.reward, self.cpfs, self.preconditions, self.invariants, self.terminations
        
        # compile constraint information
        if constraint_func:
            inequality_fns, equality_fns = self._jax_nonlinear_constraints(
                init_params_constr)
        else:
            inequality_fns, equality_fns = None, None
        
        # do a single step update from the RDDL model
        def _jax_wrapped_single_step(key, actions, subs, model_params):
            errors = NORMAL
            subs.update(actions)
            
            # check action preconditions
            precond_check = True
            if check_constraints:
                for precond in preconds:
                    sample, key, err, model_params = precond(subs, model_params, key)
                    precond_check = jnp.logical_and(precond_check, sample)
                    errors |= err
            
            # compute h(s, a) <= 0 and g(s, a) == 0 constraint functions
            inequalities, equalities = [], []
            if constraint_func:
                for constraint in inequality_fns:
                    sample, key, err, model_params = constraint(subs, model_params, key)
                    inequalities.append(sample)
                    errors |= err
                for constraint in equality_fns:
                    sample, key, err, model_params = constraint(subs, model_params, key)
                    equalities.append(sample)
                    errors |= err
                
            # calculate CPFs in topological order
            for (name, cpf) in cpfs.items():
                subs[name], key, err, model_params = cpf(subs, model_params, key)
                errors |= err                
                
            # calculate the immediate reward
            reward, key, err, model_params = reward_fn(subs, model_params, key)
            errors |= err
            
            # calculate fluent values
            if cache_path_info:
                fluents = {name: values for (name, values) in subs.items() 
                           if name not in rddl.non_fluents}
            else:
                fluents = {}
            
            # set the next state to the current state
            for (state, next_state) in rddl.next_state.items():
                subs[state] = subs[next_state]
            
            # check the state invariants
            invariant_check = True
            if check_constraints:
                for invariant in invariants:
                    sample, key, err, model_params = invariant(subs, model_params, key)
                    invariant_check = jnp.logical_and(invariant_check, sample)
                    errors |= err
            
            # check the termination (TODO: zero out reward in s if terminated)
            terminated_check = False
            if check_constraints:
                for terminal in terminals:
                    sample, key, err, model_params = terminal(subs, model_params, key)
                    terminated_check = jnp.logical_or(terminated_check, sample)
                    errors |= err
            
            # prepare the return value
            log = {
                'fluents': fluents,
                'reward': reward,
                'error': errors,
                'precondition': precond_check,
                'invariant': invariant_check,
                'termination': terminated_check
            }            
            if constraint_func:
                log['inequalities'] = inequalities
                log['equalities'] = equalities
                
            return subs, log, model_params
        
        return _jax_wrapped_single_step        
    
    def compile_rollouts(self, policy: Callable,
                         n_steps: int,
                         n_batch: int,
                         check_constraints: bool=False,
                         constraint_func: bool=False, 
                         init_params_constr: Dict[str, Any]={},
                         model_params_reduction: Callable=lambda x: x[0],
                         cache_path_info: bool=False) -> Callable:
        '''Compiles the current RDDL model into a JAX transition function that 
        samples trajectories with a fixed horizon from a policy.
        
        The arguments of the returned function is:
            - key is the PRNG key (used by a stochastic policy)
            - policy_params is a pytree of trainable policy weights
            - hyperparams is a pytree of (optional) fixed policy hyper-parameters
            - subs is the dictionary of current fluent tensor values
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
            - states is a dict of tensors for the current observation.
        
        :param policy: a Jax compiled function for the policy as described above
        decision epoch, state dict, and an RNG key and returns an action dict
        :param n_steps: the rollout horizon
        :param n_batch: how many rollouts each batch performs
        :param check_constraints: whether state, action and termination 
        conditions should be checked on each time step: this info is stored in the
        returned log and does not raise an exception
        :param constraint_func: produces the h(s, a) constraint function
        in addition to the usual outputs
        :param model_params_reduction: how to aggregate updated model_params across runs
        in the batch (defaults to selecting the first element's parameters in the batch)
        :param cache_path_info: whether to save full path traces as part of the log
        '''
        rddl = self.rddl
        jax_step_fn = self.compile_transition(
            check_constraints, constraint_func, init_params_constr, cache_path_info)
        
        # for POMDP only observ-fluents are assumed visible to the policy
        if rddl.observ_fluents:
            observed_vars = rddl.observ_fluents
        else:
            observed_vars = rddl.state_fluents
            
        # evaluate the step from the policy
        def _jax_wrapped_single_step_policy(key, policy_params, hyperparams, 
                                            step, subs, model_params):
            states = {var: values 
                      for (var, values) in subs.items()
                      if var in observed_vars}
            actions = policy(key, policy_params, hyperparams, step, states)
            key, subkey = random.split(key)
            return jax_step_fn(subkey, actions, subs, model_params)
                        
        # do a batched step update from the policy
        def _jax_wrapped_batched_step_policy(carry, step):
            key, policy_params, hyperparams, subs, model_params = carry  
            key, *subkeys = random.split(key, num=1 + n_batch)
            keys = jnp.asarray(subkeys)
            subs, log, model_params = jax.vmap(
                _jax_wrapped_single_step_policy,
                in_axes=(0, None, None, None, 0, None)
            )(keys, policy_params, hyperparams, step, subs, model_params)
            model_params = jax.tree_util.tree_map(model_params_reduction, model_params)
            carry = (key, policy_params, hyperparams, subs, model_params)
            return carry, log            
            
        # do a batched roll-out from the policy
        def _jax_wrapped_batched_rollout(key, policy_params, hyperparams, 
                                         subs, model_params):
            start = (key, policy_params, hyperparams, subs, model_params)
            steps = jnp.arange(n_steps)
            end, log = jax.lax.scan(_jax_wrapped_batched_step_policy, start, steps)
            log = jax.tree_util.tree_map(partial(jnp.swapaxes, axis1=0, axis2=1), log)
            model_params = end[-1]
            return log, model_params
        
        return _jax_wrapped_batched_rollout
    
    # ===========================================================================
    # error checks and prints
    # ===========================================================================
    
    def print_jax(self) -> Dict[str, Any]:
        '''Returns a dictionary containing the string representations of all 
        Jax compiled expressions from the RDDL file.
        '''
        subs = self.init_values
        init_params = self.model_params
        key = jax.random.PRNGKey(42)
        printed = {
            'cpfs': {name: str(jax.make_jaxpr(expr)(subs, init_params, key))
                     for (name, expr) in self.cpfs.items()},
            'reward': str(jax.make_jaxpr(self.reward)(subs, init_params, key)),
            'invariants': [str(jax.make_jaxpr(expr)(subs, init_params, key))
                           for expr in self.invariants],
            'preconditions': [str(jax.make_jaxpr(expr)(subs, init_params, key))
                              for expr in self.preconditions],
            'terminations': [str(jax.make_jaxpr(expr)(subs, init_params, key))
                             for expr in self.terminations]
        }
        return printed
    
    def model_parameter_info(self) -> Dict[str, Dict[str, Any]]:
        '''Returns a dictionary of additional information about model parameters.'''
        result = {}
        for (id, value) in self.model_params.items():
            expr_id = int(str(id).split('_')[0])
            expr = self.traced.lookup(expr_id)
            result[id] = {
                'id': expr_id, 
                'rddl_op': ' '.join(expr.etype), 
                'init_value': value
            }
        return result
        
    @staticmethod
    def _check_valid_op(expr, valid_ops):
        etype, op = expr.etype
        if op not in valid_ops:
            valid_op_str = ','.join(valid_ops.keys())
            raise RDDLNotImplementedError(
                f'{etype} operator {op} is not supported: '
                f'must be in {valid_op_str}.\n' + print_stack_trace(expr))
    
    @staticmethod
    def _check_num_args(expr, required_args):
        actual_args = len(expr.args)
        if actual_args != required_args:
            etype, op = expr.etype
            raise RDDLInvalidNumberOfArgumentsError(
                f'{etype} operator {op} requires {required_args} arguments, '
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
        errors = [i for (i, c) in enumerate(binary) if c == '1']
        return errors
    
    @staticmethod
    def get_error_messages(error: int) -> List[str]:
        '''Given a compacted integer error flag from the execution of Jax, and 
        decomposes it into error strings.
        '''
        codes = JaxRDDLCompiler.get_error_codes(error)
        messages = [JaxRDDLCompiler.INVERSE_ERROR_CODES[i] for i in codes]
        return messages
    
    # ===========================================================================
    # expression compilation
    # ===========================================================================
    
    def _jax(self, expr, init_params, dtype=None):
        etype, _ = expr.etype
        if etype == 'constant':
            jax_expr = self._jax_constant(expr, init_params)
        elif etype == 'pvar':
            jax_expr = self._jax_pvar(expr, init_params)
        elif etype == 'arithmetic':
            jax_expr = self._jax_arithmetic(expr, init_params)
        elif etype == 'relational':
            jax_expr = self._jax_relational(expr, init_params)
        elif etype == 'boolean':
            jax_expr = self._jax_logical(expr, init_params)
        elif etype == 'aggregation':
            jax_expr = self._jax_aggregation(expr, init_params)
        elif etype == 'func':
            jax_expr = self._jax_functional(expr, init_params)
        elif etype == 'pyfunc':
            jax_expr = self._jax_pyfunc(expr, init_params)
        elif etype == 'control':
            jax_expr = self._jax_control(expr, init_params)
        elif etype == 'randomvar':
            jax_expr = self._jax_random(expr, init_params)
        elif etype == 'randomvector':
            jax_expr = self._jax_random_vector(expr, init_params)
        elif etype == 'matrix':
            jax_expr = self._jax_matrix(expr, init_params)
        else:
            raise RDDLNotImplementedError(
                f'Internal error: expression type {expr} is not supported.\n' + 
                print_stack_trace(expr))
            
        # force type cast of tensor as required by caller
        if dtype is not None:
            jax_expr = self._jax_cast(jax_expr, dtype)
        
        return jax_expr
            
    def _jax_cast(self, jax_expr, dtype):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_CAST']
        
        def _jax_wrapped_cast(x, params, key):
            val, key, err, params = jax_expr(x, params, key)
            sample = jnp.asarray(val, dtype=dtype)
            invalid_cast = jnp.logical_and(
                jnp.logical_not(jnp.can_cast(val, dtype)),
                jnp.any(sample != val)
            )
            err |= (invalid_cast * ERR)
            return sample, key, err, params
        
        return _jax_wrapped_cast
   
    def _fix_dtype(self, value):
        dtype = jnp.result_type(value)
        if jnp.issubdtype(dtype, jnp.integer):
            return self.INT
        elif jnp.issubdtype(dtype, jnp.floating):
            return self.REAL
        elif jnp.issubdtype(dtype, jnp.bool_) or jnp.issubdtype(dtype, bool):
            return bool
        else:
            raise TypeError(f'Invalid type {dtype} of {value}.')
       
    # ===========================================================================
    # leaves
    # ===========================================================================
    
    def _jax_constant(self, expr, init_params):
        NORMAL = JaxRDDLCompiler.ERROR_CODES['NORMAL']
        cached_value = self.traced.cached_sim_info(expr)
        
        def _jax_wrapped_constant(x, params, key):
            sample = jnp.asarray(cached_value, dtype=self._fix_dtype(cached_value))
            return sample, key, NORMAL, params

        return _jax_wrapped_constant
    
    def _jax_pvar_slice(self, _slice):
        NORMAL = JaxRDDLCompiler.ERROR_CODES['NORMAL']
        
        def _jax_wrapped_pvar_slice(x, params, key):
            return _slice, key, NORMAL, params
        
        return _jax_wrapped_pvar_slice
            
    def _jax_pvar(self, expr, init_params):
        NORMAL = JaxRDDLCompiler.ERROR_CODES['NORMAL']
        var, pvars = expr.args  
        is_value, cached_info = self.traced.cached_sim_info(expr)
        
        # boundary case: free variable is converted to array (0, 1, 2...)
        # boundary case: domain object is converted to canonical integer index
        if is_value:
            cached_value = cached_info

            def _jax_wrapped_object(x, params, key):
                sample = jnp.asarray(cached_value, dtype=self._fix_dtype(cached_value))
                return sample, key, NORMAL, params
            
            return _jax_wrapped_object
        
        # boundary case: no shape information (e.g. scalar pvar)
        elif cached_info is None:
            
            def _jax_wrapped_pvar_scalar(x, params, key):
                value = x[var]
                sample = jnp.asarray(value, dtype=self._fix_dtype(value))
                return sample, key, NORMAL, params
            
            return _jax_wrapped_pvar_scalar
        
        # must slice and/or reshape value tensor to match free variables
        else:
            slices, axis, shape, op_code, op_args = cached_info 
        
            # compile nested expressions
            if slices and op_code == RDDLObjectsTracer.NUMPY_OP_CODE.NESTED_SLICE:
                
                jax_nested_expr = [(self._jax(arg, init_params) 
                                    if _slice is None 
                                    else self._jax_pvar_slice(_slice))
                                   for (arg, _slice) in zip(pvars, slices)]    
                
                def _jax_wrapped_pvar_tensor_nested(x, params, key):
                    error = NORMAL
                    value = x[var]
                    sample = jnp.asarray(value, dtype=self._fix_dtype(value))
                    new_slices = [None] * len(jax_nested_expr)
                    for (i, jax_expr) in enumerate(jax_nested_expr):
                        new_slice, key, err, params = jax_expr(x, params, key)
                        if not jnp.issubdtype(jnp.result_type(new_slice), jnp.integer):
                            new_slice = jnp.asarray(new_slice, dtype=self.INT)
                        new_slices[i] = new_slice
                        error |= err
                    new_slices = tuple(new_slices)
                    sample = sample[new_slices]
                    return sample, key, error, params
                
                return _jax_wrapped_pvar_tensor_nested
                
            # tensor variable but no nesting  
            else:
    
                def _jax_wrapped_pvar_tensor_non_nested(x, params, key):
                    value = x[var]
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
    # mathematical
    # ===========================================================================
    
    def _jax_unary(self, jax_expr, jax_op, at_least_int=False, check_dtype=None):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_CAST']

        def _jax_wrapped_unary_op(x, params, key):
            sample, key, err, params = jax_expr(x, params, key)
            if at_least_int:
                sample = self.ONE * sample
            sample, params = jax_op(sample, params)
            if check_dtype is not None:
                invalid_cast = jnp.logical_not(jnp.can_cast(sample, check_dtype))
                err |= (invalid_cast * ERR)
            return sample, key, err, params
        
        return _jax_wrapped_unary_op
    
    def _jax_binary(self, jax_lhs, jax_rhs, jax_op, at_least_int=False, check_dtype=None):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_CAST']
        
        def _jax_wrapped_binary_op(x, params, key):
            sample1, key, err1, params = jax_lhs(x, params, key)
            sample2, key, err2, params = jax_rhs(x, params, key)
            if at_least_int:
                sample1 = self.ONE * sample1
                sample2 = self.ONE * sample2
            sample, params = jax_op(sample1, sample2, params)
            err = err1 | err2
            if check_dtype is not None:
                invalid_cast = jnp.logical_not(jnp.logical_and(
                    jnp.can_cast(sample1, check_dtype),
                    jnp.can_cast(sample2, check_dtype))
                )
                err |= (invalid_cast * ERR)
            return sample, key, err, params
        
        return _jax_wrapped_binary_op
    
    def _jax_arithmetic(self, expr, init_params):
        _, op = expr.etype
        
        # if expression is non-fluent, always use the exact operation
        if self.compile_non_fluent_exact and not self.traced.cached_is_fluent(expr):
            valid_ops = self.EXACT_OPS['arithmetic']
            negative_op = self.EXACT_OPS['negative']
        else:
            valid_ops = self.OPS['arithmetic']
            negative_op = self.OPS['negative']            
        JaxRDDLCompiler._check_valid_op(expr, valid_ops)
        
        # recursively compile arguments
        args = expr.args
        n = len(args)
        if n == 1 and op == '-':
            arg, = args
            jax_expr = self._jax(arg, init_params)
            jax_op = negative_op(expr.id, init_params)
            return self._jax_unary(jax_expr, jax_op, at_least_int=True)
                    
        elif n == 2 or (n >= 2 and op in {'*', '+'}):
            jax_exprs = [self._jax(arg, init_params) for arg in args]
            result = jax_exprs[0]
            for (i, jax_rhs) in enumerate(jax_exprs[1:]):
                jax_op = valid_ops[op](f'{expr.id}_{op}{i}', init_params)
                result = self._jax_binary(result, jax_rhs, jax_op, at_least_int=True)
            return result
        
        JaxRDDLCompiler._check_num_args(expr, 2)
    
    def _jax_relational(self, expr, init_params):
        _, op = expr.etype
        
        # if expression is non-fluent, always use the exact operation
        if self.compile_non_fluent_exact and not self.traced.cached_is_fluent(expr):
            valid_ops = self.EXACT_OPS['relational']
        else:
            valid_ops = self.OPS['relational']
        JaxRDDLCompiler._check_valid_op(expr, valid_ops)
        
        # recursively compile arguments
        JaxRDDLCompiler._check_num_args(expr, 2)
        lhs, rhs = expr.args
        jax_lhs = self._jax(lhs, init_params)
        jax_rhs = self._jax(rhs, init_params)
        jax_op = valid_ops[op](expr.id, init_params)
        return self._jax_binary(jax_lhs, jax_rhs, jax_op, at_least_int=True)
           
    def _jax_logical(self, expr, init_params):
        _, op = expr.etype
        
        # if expression is non-fluent, always use the exact operation
        if self.compile_non_fluent_exact and not self.traced.cached_is_fluent(expr):
            valid_ops = self.EXACT_OPS['logical']  
            logical_not_op = self.EXACT_OPS['logical_not']
        else:
            valid_ops = self.OPS['logical']
            logical_not_op = self.OPS['logical_not'] 
        JaxRDDLCompiler._check_valid_op(expr, valid_ops)
                
        # recursively compile arguments
        args = expr.args
        n = len(args)        
        if n == 1 and op == '~':
            arg, = args
            jax_expr = self._jax(arg, init_params)
            jax_op = logical_not_op(expr.id, init_params)
            return self._jax_unary(jax_expr, jax_op, check_dtype=bool)
        
        elif n == 2 or (n >= 2 and op in {'^', '&', '|'}):
            jax_exprs = [self._jax(arg, init_params) for arg in args]
            result = jax_exprs[0]
            for i, jax_rhs in enumerate(jax_exprs[1:]):
                jax_op = valid_ops[op](f'{expr.id}_{op}{i}', init_params)
                result = self._jax_binary(result, jax_rhs, jax_op, check_dtype=bool)
            return result
        
        JaxRDDLCompiler._check_num_args(expr, 2)
    
    def _jax_aggregation(self, expr, init_params):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_CAST']
        _, op = expr.etype
        
        # if expression is non-fluent, always use the exact operation
        if self.compile_non_fluent_exact and not self.traced.cached_is_fluent(expr):
            valid_ops = self.EXACT_OPS['aggregation']   
        else:
            valid_ops = self.OPS['aggregation']   
        JaxRDDLCompiler._check_valid_op(expr, valid_ops) 
        is_floating = op not in self.AGGREGATION_BOOL
        
        # recursively compile arguments
        * _, arg = expr.args  
        _, axes = self.traced.cached_sim_info(expr)        
        jax_expr = self._jax(arg, init_params)
        jax_op = valid_ops[op](expr.id, init_params)
        
        def _jax_wrapped_aggregation(x, params, key):
            sample, key, err, params = jax_expr(x, params, key)
            if is_floating:
                sample = self.ONE * sample
            else:
                invalid_cast = jnp.logical_not(jnp.can_cast(sample, bool))
                err |= (invalid_cast * ERR)
            sample, params = jax_op(sample, axis=axes, params=params)
            return sample, key, err, params
        
        return _jax_wrapped_aggregation
               
    def _jax_functional(self, expr, init_params):
        _, op = expr.etype
        
        # if expression is non-fluent, always use the exact operation
        if self.compile_non_fluent_exact and not self.traced.cached_is_fluent(expr):            
            unary_ops = self.EXACT_OPS['unary']
            binary_ops = self.EXACT_OPS['binary']
        else:
            unary_ops = self.OPS['unary']
            binary_ops = self.OPS['binary']
        
        # recursively compile arguments
        if op in unary_ops:
            JaxRDDLCompiler._check_num_args(expr, 1)                            
            arg, = expr.args
            jax_expr = self._jax(arg, init_params)
            jax_op = unary_ops[op](expr.id, init_params)
            return self._jax_unary(jax_expr, jax_op, at_least_int=True)
            
        elif op in binary_ops:
            JaxRDDLCompiler._check_num_args(expr, 2)                
            lhs, rhs = expr.args
            jax_lhs = self._jax(lhs, init_params)
            jax_rhs = self._jax(rhs, init_params)
            jax_op = binary_ops[op](expr.id, init_params)
            return self._jax_binary(jax_lhs, jax_rhs, jax_op, at_least_int=True)
        
        raise RDDLNotImplementedError(
            f'Function {op} is not supported.\n' + print_stack_trace(expr))   
    
    def _jax_pyfunc(self, expr, init_params):
        NORMAL = JaxRDDLCompiler.ERROR_CODES['NORMAL']

        # get the Python function by name
        _, pyfunc_name = expr.etype
        pyfunc = self.python_functions.get(pyfunc_name)
        if pyfunc is None:
            raise ValueError(
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

        # compile the inputs to the function
        jax_inputs = [self._jax(arg, init_params) for arg in args]

        # compile the function evaluation function
        def _jax_wrapped_external_function(x, params, key):

            # evaluate inputs to the function
            # first dimensions are non-captured vars in outer scope followed by all the _
            error = NORMAL
            flat_samples = []
            for jax_expr in jax_inputs:
                sample, key, err, params = jax_expr(x, params, key)
                shape = jnp.shape(sample)
                first_dim = 1
                for dim in shape[:num_free_vars]:
                    first_dim *= dim
                new_shape = (first_dim,) + shape[num_free_vars:]
                flat_sample = jnp.reshape(sample, new_shape)
                flat_samples.append(flat_sample)
                error |= err

            # now all the inputs have dimensions equal to (k,) + the number of _ occurences
            # k is the number of possible non-captured object combinations
            # evaluate the function independently for each combination
            # output dimension for each combination is captured variables (n1, n2, ...)
            # so the total dimension of the output array is (k, n1, n2, ...)
            sample = jax.vmap(pyfunc, in_axes=0)(*flat_samples)
            pyfunc_dims = jnp.shape(sample)[1:]
                        
            if len(require_dims) != len(pyfunc_dims):
                raise ValueError(
                    f'Output of external Python function <{pyfunc_name}> returned array with '
                    f'{len(pyfunc_dims)} dimensions, which does not match the '
                    f'number of captured parameter(s) {len(require_dims)}.\n' +  
                    print_stack_trace(expr))
            for (param, require_dim, actual_dim) in zip(captured_vars, require_dims, pyfunc_dims):
                if require_dim != actual_dim:
                    raise ValueError(
                        f'Output of external Python function <{pyfunc_name}> returned array with '
                        f'{actual_dim} elements for captured parameter <{param}>, '
                        f'which does not match the number of objects {require_dim}.\n' + 
                        print_stack_trace(expr))

            # unravel the combinations k back into their original dimensions
            sample = jnp.reshape(sample, free_dims + pyfunc_dims)
            
            # rearrange the output dimensions to match the outer scope
            source_indices = [num_free_vars + i for i in range(len(pyfunc_dims))]
            sample = jnp.moveaxis(sample, source=source_indices, destination=dest_indices)
            return sample, key, error, params
        
        return _jax_wrapped_external_function

    # ===========================================================================
    # control flow
    # ===========================================================================
    
    def _jax_control(self, expr, init_params):
        _, op = expr.etype        
        if op == 'if':
            return self._jax_if(expr, init_params)
        elif op == 'switch':
            return self._jax_switch(expr, init_params)
        
        raise RDDLNotImplementedError(
            f'Control operator {op} is not supported.\n' + print_stack_trace(expr))   
    
    def _jax_if(self, expr, init_params):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_CAST']
        JaxRDDLCompiler._check_num_args(expr, 3)
        pred, if_true, if_false = expr.args     
        
        # if predicate is non-fluent, always use the exact operation
        if self.compile_non_fluent_exact and not self.traced.cached_is_fluent(pred):
            if_op = self.EXACT_OPS['control']['if']
        else:
            if_op = self.OPS['control']['if']
        jax_op = if_op(expr.id, init_params)
        
        # recursively compile arguments   
        jax_pred = self._jax(pred, init_params)
        jax_true = self._jax(if_true, init_params)
        jax_false = self._jax(if_false, init_params)
        
        def _jax_wrapped_if_then_else(x, params, key):
            sample1, key, err1, params = jax_pred(x, params, key)
            sample2, key, err2, params = jax_true(x, params, key)
            sample3, key, err3, params = jax_false(x, params, key)
            sample, params = jax_op(sample1, sample2, sample3, params)
            err = err1 | err2 | err3
            invalid_cast = jnp.logical_not(jnp.can_cast(sample1, bool))
            err |= (invalid_cast * ERR)
            return sample, key, err, params
            
        return _jax_wrapped_if_then_else
    
    def _jax_switch(self, expr, init_params):
        pred, *_ = expr.args
             
        # if predicate is non-fluent, always use the exact operation
        # case conditions are currently only literals so they are non-fluent
        if self.compile_non_fluent_exact and not self.traced.cached_is_fluent(pred):
            switch_op = self.EXACT_OPS['control']['switch']
        else:
            switch_op = self.OPS['control']['switch']
        jax_op = switch_op(expr.id, init_params)
        
        # recursively compile predicate
        jax_pred = self._jax(pred, init_params)
        
        # recursively compile cases
        cases, default = self.traced.cached_sim_info(expr) 
        jax_default = None if default is None else self._jax(default, init_params)
        jax_cases = [(jax_default if _case is None else self._jax(_case, init_params))
                     for _case in cases]
                    
        def _jax_wrapped_switch(x, params, key):
            
            # sample predicate
            sample_pred, key, err, params = jax_pred(x, params, key) 
            
            # sample cases
            sample_cases = [None] * len(jax_cases)
            for (i, jax_case) in enumerate(jax_cases):
                sample_cases[i], key, err_case, params = jax_case(x, params, key)
                err |= err_case      
            sample_cases = jnp.asarray(sample_cases)          
            sample_cases = jnp.asarray(sample_cases, dtype=self._fix_dtype(sample_cases))
            
            # predicate (enum) is an integer - use it to extract from case array
            sample, params = jax_op(sample_pred, sample_cases, params)
            return sample, key, err, params
        
        return _jax_wrapped_switch
    
    # ===========================================================================
    # random variables
    # ===========================================================================
    
    # distributions with complete reparameterization support:
    # KronDelta: complete
    # DiracDelta: complete
    # Uniform: complete
    # Bernoulli: complete (subclass uses Gumbel-softmax)
    # Normal: complete
    # Exponential: complete
    # Geometric: complete
    # Weibull: complete
    # Pareto: complete
    # Gumbel: complete
    # Laplace: complete
    # Cauchy: complete
    # Gompertz: complete
    # Kumaraswamy: complete
    # Discrete: complete (subclass uses Gumbel-softmax)
    # UnnormDiscrete: complete (subclass uses Gumbel-softmax)
    # Discrete(p): complete (subclass uses Gumbel-softmax)
    # UnnormDiscrete(p): complete (subclass uses Gumbel-softmax)
    # Poisson (subclass uses Gumbel-softmax or Poisson process trick)
    # Binomial (subclass uses Gumbel-softmax or Normal approximation)
    # NegativeBinomial (subclass uses Poisson-Gamma mixture)
    
    # distributions which seem to support backpropagation (need more testing):
    # Beta
    # Student
    # Gamma
    # ChiSquare   
    # Dirichlet

    # distributions with incomplete reparameterization support (TODO):
    # Multinomial
    
    def _jax_random(self, expr, init_params):
        _, name = expr.etype
        if name == 'KronDelta':
            return self._jax_kron(expr, init_params)        
        elif name == 'DiracDelta':
            return self._jax_dirac(expr, init_params)
        elif name == 'Uniform':
            return self._jax_uniform(expr, init_params)
        elif name == 'Bernoulli':
            return self._jax_bernoulli(expr, init_params)
        elif name == 'Normal':
            return self._jax_normal(expr, init_params)  
        elif name == 'Poisson':
            return self._jax_poisson(expr, init_params)
        elif name == 'Exponential':
            return self._jax_exponential(expr, init_params)
        elif name == 'Weibull':
            return self._jax_weibull(expr, init_params) 
        elif name == 'Gamma':
            return self._jax_gamma(expr, init_params)
        elif name == 'Binomial':
            return self._jax_binomial(expr, init_params)
        elif name == 'NegativeBinomial':
            return self._jax_negative_binomial(expr, init_params)
        elif name == 'Beta':
            return self._jax_beta(expr, init_params)
        elif name == 'Geometric':
            return self._jax_geometric(expr, init_params)
        elif name == 'Pareto':
            return self._jax_pareto(expr, init_params)
        elif name == 'Student':
            return self._jax_student(expr, init_params)
        elif name == 'Gumbel':
            return self._jax_gumbel(expr, init_params)
        elif name == 'Laplace':
            return self._jax_laplace(expr, init_params)
        elif name == 'Cauchy':
            return self._jax_cauchy(expr, init_params)
        elif name == 'Gompertz':
            return self._jax_gompertz(expr, init_params)
        elif name == 'ChiSquare':
            return self._jax_chisquare(expr, init_params)
        elif name == 'Kumaraswamy':
            return self._jax_kumaraswamy(expr, init_params)
        elif name == 'Discrete':
            return self._jax_discrete(expr, init_params, unnorm=False)
        elif name == 'UnnormDiscrete':
            return self._jax_discrete(expr, init_params, unnorm=True)
        elif name == 'Discrete(p)':
            return self._jax_discrete_pvar(expr, init_params, unnorm=False)
        elif name == 'UnnormDiscrete(p)':
            return self._jax_discrete_pvar(expr, init_params, unnorm=True)
        else:
            raise RDDLNotImplementedError(
                f'Distribution {name} is not supported.\n' + print_stack_trace(expr))
        
    def _jax_kron(self, expr, init_params):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_KRON_DELTA']
        JaxRDDLCompiler._check_num_args(expr, 1)
        arg, = expr.args
        arg = self._jax(arg, init_params)
        
        # just check that the sample can be cast to int
        def _jax_wrapped_distribution_kron(x, params, key):
            sample, key, err, params = arg(x, params, key)
            invalid_cast = jnp.logical_not(jnp.can_cast(sample, self.INT))
            err |= (invalid_cast * ERR)
            return sample, key, err, params
                        
        return _jax_wrapped_distribution_kron
    
    def _jax_dirac(self, expr, init_params):
        JaxRDDLCompiler._check_num_args(expr, 1)
        arg, = expr.args
        arg = self._jax(arg, init_params, dtype=self.REAL)
        return arg
    
    def _jax_uniform(self, expr, init_params):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_UNIFORM']
        JaxRDDLCompiler._check_num_args(expr, 2)
        
        arg_lb, arg_ub = expr.args
        jax_lb = self._jax(arg_lb, init_params)
        jax_ub = self._jax(arg_ub, init_params)
        
        # reparameterization trick U(a, b) = a + (b - a) * U(0, 1)
        def _jax_wrapped_distribution_uniform(x, params, key):
            lb, key, err1, params = jax_lb(x, params, key)
            ub, key, err2, params = jax_ub(x, params, key)
            key, subkey = random.split(key)
            U = random.uniform(key=subkey, shape=jnp.shape(lb), dtype=self.REAL)
            sample = lb + (ub - lb) * U
            out_of_bounds = jnp.logical_not(jnp.all(lb <= ub))
            err = err1 | err2 | (out_of_bounds * ERR)
            return sample, key, err, params
        
        return _jax_wrapped_distribution_uniform
    
    def _jax_normal(self, expr, init_params):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_NORMAL']
        JaxRDDLCompiler._check_num_args(expr, 2)
        
        arg_mean, arg_var = expr.args
        jax_mean = self._jax(arg_mean, init_params)
        jax_var = self._jax(arg_var, init_params)
        
        # reparameterization trick N(m, s^2) = m + s * N(0, 1)
        def _jax_wrapped_distribution_normal(x, params, key):
            mean, key, err1, params = jax_mean(x, params, key)
            var, key, err2, params = jax_var(x, params, key)
            std = jnp.sqrt(var)
            key, subkey = random.split(key)
            Z = random.normal(key=subkey, shape=jnp.shape(mean), dtype=self.REAL)
            sample = mean + std * Z
            out_of_bounds = jnp.logical_not(jnp.all(var >= 0))
            err = err1 | err2 | (out_of_bounds * ERR)
            return sample, key, err, params
        
        return _jax_wrapped_distribution_normal
    
    def _jax_exponential(self, expr, init_params):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_EXPONENTIAL']
        JaxRDDLCompiler._check_num_args(expr, 1)
        
        arg_scale, = expr.args
        jax_scale = self._jax(arg_scale, init_params)
        
        # reparameterization trick Exp(s) = s * Exp(1)
        def _jax_wrapped_distribution_exp(x, params, key):
            scale, key, err, params = jax_scale(x, params, key)
            key, subkey = random.split(key)
            Exp1 = random.exponential(key=subkey, shape=jnp.shape(scale), dtype=self.REAL)
            sample = scale * Exp1
            out_of_bounds = jnp.logical_not(jnp.all(scale > 0))
            err |= (out_of_bounds * ERR)
            return sample, key, err, params
        
        return _jax_wrapped_distribution_exp
    
    def _jax_weibull(self, expr, init_params):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_WEIBULL']
        JaxRDDLCompiler._check_num_args(expr, 2)
        
        arg_shape, arg_scale = expr.args
        jax_shape = self._jax(arg_shape, init_params)
        jax_scale = self._jax(arg_scale, init_params)
        
        # reparameterization trick W(s, r) = r * (-ln(1 - U(0, 1))) ** (1 / s)
        def _jax_wrapped_distribution_weibull(x, params, key):
            shape, key, err1, params = jax_shape(x, params, key)
            scale, key, err2, params = jax_scale(x, params, key)
            key, subkey = random.split(key)
            sample = random.weibull_min(
                key=subkey, scale=scale, concentration=shape, dtype=self.REAL)
            out_of_bounds = jnp.logical_not(jnp.all((shape > 0) & (scale > 0)))
            err = err1 | err2 | (out_of_bounds * ERR)
            return sample, key, err, params
        
        return _jax_wrapped_distribution_weibull
    
    def _jax_bernoulli(self, expr, init_params):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_BERNOULLI']
        JaxRDDLCompiler._check_num_args(expr, 1)
        arg_prob, = expr.args
        
        # if probability is non-fluent, always use the exact operation
        if self.compile_non_fluent_exact and not self.traced.cached_is_fluent(arg_prob):
            bern_op = self.EXACT_OPS['sampling']['Bernoulli']
        else:
            bern_op = self.OPS['sampling']['Bernoulli']
        jax_op = bern_op(expr.id, init_params)
        
        # recursively compile arguments
        jax_prob = self._jax(arg_prob, init_params)
        
        def _jax_wrapped_distribution_bernoulli(x, params, key):
            prob, key, err, params = jax_prob(x, params, key)
            key, subkey = random.split(key)
            sample, params = jax_op(subkey, prob, params)
            out_of_bounds = jnp.logical_not(jnp.all((prob >= 0) & (prob <= 1)))
            err |= (out_of_bounds * ERR)
            return sample, key, err, params
        
        return _jax_wrapped_distribution_bernoulli
    
    def _jax_poisson(self, expr, init_params):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_POISSON']
        JaxRDDLCompiler._check_num_args(expr, 1)
        arg_rate, = expr.args
        
        # if rate is non-fluent, always use the exact operation
        if self.compile_non_fluent_exact and not self.traced.cached_is_fluent(arg_rate):
            poisson_op = self.EXACT_OPS['sampling']['Poisson']
        else:
            poisson_op = self.OPS['sampling']['Poisson']
        jax_op = poisson_op(expr.id, init_params)
        
        # recursively compile arguments
        jax_rate = self._jax(arg_rate, init_params)
        
        # uses the implicit JAX subroutine
        def _jax_wrapped_distribution_poisson(x, params, key):
            rate, key, err, params = jax_rate(x, params, key)
            key, subkey = random.split(key)
            sample, params = jax_op(subkey, rate, params)
            out_of_bounds = jnp.logical_not(jnp.all(rate >= 0))
            err |= (out_of_bounds * ERR)
            return sample, key, err, params
        
        return _jax_wrapped_distribution_poisson
    
    def _jax_gamma(self, expr, init_params):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_GAMMA']
        JaxRDDLCompiler._check_num_args(expr, 2)
        
        arg_shape, arg_scale = expr.args
        jax_shape = self._jax(arg_shape, init_params)
        jax_scale = self._jax(arg_scale, init_params)
        
        # partial reparameterization trick Gamma(s, r) = r * Gamma(s, 1)
        # uses the implicit JAX subroutine for Gamma(s, 1) 
        def _jax_wrapped_distribution_gamma(x, params, key):
            shape, key, err1, params = jax_shape(x, params, key)
            scale, key, err2, params = jax_scale(x, params, key)
            key, subkey = random.split(key)
            Gamma = random.gamma(key=subkey, a=shape, dtype=self.REAL)
            sample = scale * Gamma
            out_of_bounds = jnp.logical_not(jnp.all((shape > 0) & (scale > 0)))
            err = err1 | err2 | (out_of_bounds * ERR)
            return sample, key, err, params
        
        return _jax_wrapped_distribution_gamma
    
    def _jax_binomial(self, expr, init_params):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_BINOMIAL']
        JaxRDDLCompiler._check_num_args(expr, 2)
        arg_trials, arg_prob = expr.args

        # if prob is non-fluent, always use the exact operation
        if self.compile_non_fluent_exact \
        and not self.traced.cached_is_fluent(arg_trials) \
        and not self.traced.cached_is_fluent(arg_prob):
            bin_op = self.EXACT_OPS['sampling']['Binomial']
        else:
            bin_op = self.OPS['sampling']['Binomial']
        jax_op = bin_op(expr.id, init_params)

        jax_trials = self._jax(arg_trials, init_params)
        jax_prob = self._jax(arg_prob, init_params)

        # uses reduction for constant trials
        def _jax_wrapped_distribution_binomial(x, params, key):
            trials, key, err2, params = jax_trials(x, params, key)       
            prob, key, err1, params = jax_prob(x, params, key)
            key, subkey = random.split(key)
            sample, params = jax_op(subkey, trials, prob, params)
            out_of_bounds = jnp.logical_not(jnp.all(
                (prob >= 0) & (prob <= 1) & (trials >= 0)))
            err = err1 | err2 | (out_of_bounds * ERR)
            return sample, key, err, params
        
        return _jax_wrapped_distribution_binomial
    
    def _jax_negative_binomial(self, expr, init_params):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_NEGATIVE_BINOMIAL']
        JaxRDDLCompiler._check_num_args(expr, 2)
        arg_trials, arg_prob = expr.args

        # if prob is non-fluent, always use the exact operation
        if self.compile_non_fluent_exact \
        and not self.traced.cached_is_fluent(arg_trials) \
        and not self.traced.cached_is_fluent(arg_prob):
            negbin_op = self.EXACT_OPS['sampling']['NegativeBinomial']
        else:
            negbin_op = self.OPS['sampling']['NegativeBinomial']
        jax_op = negbin_op(expr.id, init_params)

        jax_trials = self._jax(arg_trials, init_params)
        jax_prob = self._jax(arg_prob, init_params)
        
        # uses the JAX substrate of tensorflow-probability
        def _jax_wrapped_distribution_negative_binomial(x, params, key):
            trials, key, err2, params = jax_trials(x, params, key)       
            prob, key, err1, params = jax_prob(x, params, key)
            key, subkey = random.split(key)
            sample, params = jax_op(subkey, trials, prob, params)
            out_of_bounds = jnp.logical_not(jnp.all(
                (prob >= 0) & (prob <= 1) & (trials > 0)))
            err = err1 | err2 | (out_of_bounds * ERR)
            return sample, key, err, params
        
        return _jax_wrapped_distribution_negative_binomial    
        
    def _jax_beta(self, expr, init_params):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_BETA']
        JaxRDDLCompiler._check_num_args(expr, 2)
        
        arg_shape, arg_rate = expr.args
        jax_shape = self._jax(arg_shape, init_params)
        jax_rate = self._jax(arg_rate, init_params)
        
        # uses the implicit JAX subroutine
        def _jax_wrapped_distribution_beta(x, params, key):
            shape, key, err1, params = jax_shape(x, params, key)
            rate, key, err2, params = jax_rate(x, params, key)
            key, subkey = random.split(key)
            sample = random.beta(key=subkey, a=shape, b=rate, dtype=self.REAL)
            out_of_bounds = jnp.logical_not(jnp.all((shape > 0) & (rate > 0)))
            err = err1 | err2 | (out_of_bounds * ERR)
            return sample, key, err, params
        
        return _jax_wrapped_distribution_beta
    
    def _jax_geometric(self, expr, init_params):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_GEOMETRIC']
        JaxRDDLCompiler._check_num_args(expr, 1)        
        arg_prob, = expr.args
        
        # if prob is non-fluent, always use the exact operation
        if self.compile_non_fluent_exact and not self.traced.cached_is_fluent(arg_prob):
            geom_op = self.EXACT_OPS['sampling']['Geometric']
        else:
            geom_op = self.OPS['sampling']['Geometric']
        jax_op = geom_op(expr.id, init_params)
        
        # recursively compile arguments        
        jax_prob = self._jax(arg_prob, init_params)
        
        def _jax_wrapped_distribution_geometric(x, params, key):
            prob, key, err, params = jax_prob(x, params, key)
            key, subkey = random.split(key)
            sample, params = jax_op(subkey, prob, params)
            out_of_bounds = jnp.logical_not(jnp.all((prob >= 0) & (prob <= 1)))
            err |= (out_of_bounds * ERR)
            return sample, key, err, params
        
        return _jax_wrapped_distribution_geometric
    
    def _jax_pareto(self, expr, init_params):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_PARETO']
        JaxRDDLCompiler._check_num_args(expr, 2)
        
        arg_shape, arg_scale = expr.args
        jax_shape = self._jax(arg_shape, init_params)
        jax_scale = self._jax(arg_scale, init_params)
        
        # partial reparameterization trick Pareto(s, r) = r * Pareto(s, 1)
        # uses the implicit JAX subroutine for Pareto(s, 1) 
        def _jax_wrapped_distribution_pareto(x, params, key):
            shape, key, err1, params = jax_shape(x, params, key)
            scale, key, err2, params = jax_scale(x, params, key)
            key, subkey = random.split(key)
            sample = scale * random.pareto(key=subkey, b=shape, dtype=self.REAL)
            out_of_bounds = jnp.logical_not(jnp.all((shape > 0) & (scale > 0)))
            err = err1 | err2 | (out_of_bounds * ERR)
            return sample, key, err, params
        
        return _jax_wrapped_distribution_pareto
    
    def _jax_student(self, expr, init_params):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_STUDENT']
        JaxRDDLCompiler._check_num_args(expr, 1)
        
        arg_df, = expr.args
        jax_df = self._jax(arg_df, init_params)
        
        # uses the implicit JAX subroutine for student(df)
        def _jax_wrapped_distribution_t(x, params, key):
            df, key, err, params = jax_df(x, params, key)
            key, subkey = random.split(key)
            sample = random.t(key=subkey, df=df, shape=jnp.shape(df), dtype=self.REAL)
            out_of_bounds = jnp.logical_not(jnp.all(df > 0))
            err |= (out_of_bounds * ERR)
            return sample, key, err, params
        
        return _jax_wrapped_distribution_t
    
    def _jax_gumbel(self, expr, init_params):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_GUMBEL']
        JaxRDDLCompiler._check_num_args(expr, 2)
        
        arg_mean, arg_scale = expr.args
        jax_mean = self._jax(arg_mean, init_params)
        jax_scale = self._jax(arg_scale, init_params)
        
        # reparameterization trick Gumbel(m, s) = m + s * Gumbel(0, 1)
        def _jax_wrapped_distribution_gumbel(x, params, key):
            mean, key, err1, params = jax_mean(x, params, key)
            scale, key, err2, params = jax_scale(x, params, key)
            key, subkey = random.split(key)
            Gumbel01 = random.gumbel(key=subkey, shape=jnp.shape(mean), dtype=self.REAL)
            sample = mean + scale * Gumbel01
            out_of_bounds = jnp.logical_not(jnp.all(scale > 0))
            err = err1 | err2 | (out_of_bounds * ERR)
            return sample, key, err, params
        
        return _jax_wrapped_distribution_gumbel
    
    def _jax_laplace(self, expr, init_params):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_LAPLACE']
        JaxRDDLCompiler._check_num_args(expr, 2)
        
        arg_mean, arg_scale = expr.args
        jax_mean = self._jax(arg_mean, init_params)
        jax_scale = self._jax(arg_scale, init_params)
        
        # reparameterization trick Laplace(m, s) = m + s * Laplace(0, 1)
        def _jax_wrapped_distribution_laplace(x, params, key):
            mean, key, err1, params = jax_mean(x, params, key)
            scale, key, err2, params = jax_scale(x, params, key)
            key, subkey = random.split(key)
            Laplace01 = random.laplace(key=subkey, shape=jnp.shape(mean), dtype=self.REAL)
            sample = mean + scale * Laplace01
            out_of_bounds = jnp.logical_not(jnp.all(scale > 0))
            err = err1 | err2 | (out_of_bounds * ERR)
            return sample, key, err, params
        
        return _jax_wrapped_distribution_laplace
    
    def _jax_cauchy(self, expr, init_params):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_CAUCHY']
        JaxRDDLCompiler._check_num_args(expr, 2)
        
        arg_mean, arg_scale = expr.args
        jax_mean = self._jax(arg_mean, init_params)
        jax_scale = self._jax(arg_scale, init_params)
        
        # reparameterization trick Cauchy(m, s) = m + s * Cauchy(0, 1)
        def _jax_wrapped_distribution_cauchy(x, params, key):
            mean, key, err1, params = jax_mean(x, params, key)
            scale, key, err2, params = jax_scale(x, params, key)
            key, subkey = random.split(key)
            Cauchy01 = random.cauchy(key=subkey, shape=jnp.shape(mean), dtype=self.REAL)
            sample = mean + scale * Cauchy01
            out_of_bounds = jnp.logical_not(jnp.all(scale > 0))
            err = err1 | err2 | (out_of_bounds * ERR)
            return sample, key, err, params
        
        return _jax_wrapped_distribution_cauchy
    
    def _jax_gompertz(self, expr, init_params):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_GOMPERTZ']
        JaxRDDLCompiler._check_num_args(expr, 2)
        
        arg_shape, arg_scale = expr.args
        jax_shape = self._jax(arg_shape, init_params)
        jax_scale = self._jax(arg_scale, init_params)
        
        # reparameterization trick Gompertz(s, r) = ln(1 - log(U(0, 1)) / s) / r
        def _jax_wrapped_distribution_gompertz(x, params, key):
            shape, key, err1, params = jax_shape(x, params, key)
            scale, key, err2, params = jax_scale(x, params, key)
            key, subkey = random.split(key)
            U = random.uniform(key=subkey, shape=jnp.shape(scale), dtype=self.REAL)
            sample = jnp.log(1.0 - jnp.log1p(-U) / shape) / scale
            out_of_bounds = jnp.logical_not(jnp.all((shape > 0) & (scale > 0)))
            err = err1 | err2 | (out_of_bounds * ERR)
            return sample, key, err, params
        
        return _jax_wrapped_distribution_gompertz
    
    def _jax_chisquare(self, expr, init_params):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_CHISQUARE']
        JaxRDDLCompiler._check_num_args(expr, 1)
        
        arg_df, = expr.args
        jax_df = self._jax(arg_df, init_params)
        
        # use the fact that ChiSquare(df) = Gamma(df/2, 2)
        def _jax_wrapped_distribution_chisquare(x, params, key):
            df, key, err1, params = jax_df(x, params, key)
            key, subkey = random.split(key)
            shape = df / 2.0
            Gamma = random.gamma(key=subkey, a=shape, dtype=self.REAL)
            sample = 2.0 * Gamma
            out_of_bounds = jnp.logical_not(jnp.all(df > 0))
            err = err1 | (out_of_bounds * ERR)
            return sample, key, err, params
        
        return _jax_wrapped_distribution_chisquare
    
    def _jax_kumaraswamy(self, expr, init_params):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_KUMARASWAMY']
        JaxRDDLCompiler._check_num_args(expr, 2)
        
        arg_a, arg_b = expr.args
        jax_a = self._jax(arg_a, init_params)
        jax_b = self._jax(arg_b, init_params)
        
        # uses the reparameterization K(a, b) = (1 - (1 - U(0, 1))^{1/b})^{1/a}
        def _jax_wrapped_distribution_kumaraswamy(x, params, key):
            a, key, err1, params = jax_a(x, params, key)
            b, key, err2, params = jax_b(x, params, key)
            key, subkey = random.split(key)
            U = random.uniform(key=subkey, shape=jnp.shape(a), dtype=self.REAL)            
            sample = jnp.power(1.0 - jnp.power(U, 1.0 / b), 1.0 / a)
            out_of_bounds = jnp.logical_not(jnp.all((a > 0) & (b > 0)))
            err = err1 | err2 | (out_of_bounds * ERR)
            return sample, key, err, params
        
        return _jax_wrapped_distribution_kumaraswamy
    
    # ===========================================================================
    # random variables with enum support
    # ===========================================================================
    
    def _jax_discrete(self, expr, init_params, unnorm):
        NORMAL = JaxRDDLCompiler.ERROR_CODES['NORMAL']
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_DISCRETE']
        ordered_args = self.traced.cached_sim_info(expr)
        
        # if all probabilities are non-fluent, then always sample exact
        has_fluent_arg = any(self.traced.cached_is_fluent(arg) 
                             for arg in ordered_args)
        if self.compile_non_fluent_exact and not has_fluent_arg:
            discrete_op = self.EXACT_OPS['sampling']['Discrete']
        else:
            discrete_op = self.OPS['sampling']['Discrete']
        jax_op = discrete_op(expr.id, init_params)
        
        # compile probability expressions
        jax_probs = [self._jax(arg, init_params) for arg in ordered_args]
        
        def _jax_wrapped_distribution_discrete(x, params, key):
            
            # sample case probabilities and normalize as needed
            error = NORMAL
            prob = [None] * len(jax_probs)
            for (i, jax_prob) in enumerate(jax_probs):
                prob[i], key, error_pdf, params = jax_prob(x, params, key)
                error |= error_pdf
            prob = jnp.stack(prob, axis=-1)
            if unnorm:
                normalizer = jnp.sum(prob, axis=-1, keepdims=True)
                prob = prob / normalizer
            
            # dispatch to sampling subroutine
            key, subkey = random.split(key)
            sample, params = jax_op(subkey, prob, params)
            out_of_bounds = jnp.logical_not(jnp.logical_and(
                jnp.all(prob >= 0),
                jnp.allclose(jnp.sum(prob, axis=-1), 1.0)
            ))
            error |= (out_of_bounds * ERR)
            return sample, key, error, params
        
        return _jax_wrapped_distribution_discrete
    
    def _jax_discrete_pvar(self, expr, init_params, unnorm):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_DISCRETE']
        JaxRDDLCompiler._check_num_args(expr, 2)
        _, args = expr.args
        arg, = args
        
        # if probabilities are non-fluent, then always sample exact
        if self.compile_non_fluent_exact and not self.traced.cached_is_fluent(arg):
            discrete_op = self.EXACT_OPS['sampling']['Discrete']
        else:
            discrete_op = self.OPS['sampling']['Discrete']
        jax_op = discrete_op(expr.id, init_params)
        
        # compile probability function
        jax_probs = self._jax(arg, init_params)

        def _jax_wrapped_distribution_discrete_pvar(x, params, key):
            
            # sample probabilities
            prob, key, error, params = jax_probs(x, params, key)
            if unnorm:
                normalizer = jnp.sum(prob, axis=-1, keepdims=True)
                prob = prob / normalizer
            
            # dispatch to sampling subroutine
            key, subkey = random.split(key)
            sample, params = jax_op(subkey, prob, params)
            out_of_bounds = jnp.logical_not(jnp.logical_and(
                jnp.all(prob >= 0),
                jnp.allclose(jnp.sum(prob, axis=-1), 1.0)
            ))
            error |= (out_of_bounds * ERR)
            return sample, key, error, params
        
        return _jax_wrapped_distribution_discrete_pvar

    # ===========================================================================
    # random vectors
    # ===========================================================================
    
    def _jax_random_vector(self, expr, init_params):
        _, name = expr.etype
        if name == 'MultivariateNormal':
            return self._jax_multivariate_normal(expr, init_params)   
        elif name == 'MultivariateStudent':
            return self._jax_multivariate_student(expr, init_params)  
        elif name == 'Dirichlet':
            return self._jax_dirichlet(expr, init_params)
        elif name == 'Multinomial':
            return self._jax_multinomial(expr, init_params)
        else:
            raise RDDLNotImplementedError(
                f'Distribution {name} is not supported.\n' + print_stack_trace(expr))
    
    def _jax_multivariate_normal(self, expr, init_params): 
        _, args = expr.args
        mean, cov = args
        jax_mean = self._jax(mean, init_params)
        jax_cov = self._jax(cov, init_params)
        index, = self.traced.cached_sim_info(expr)
        
        # reparameterization trick MN(m, LL') = LZ + m, where Z ~ Normal(0, 1)
        def _jax_wrapped_distribution_multivariate_normal(x, params, key):
            
            # sample the mean and covariance
            sample_mean, key, err1, params = jax_mean(x, params, key)
            sample_cov, key, err2, params = jax_cov(x, params, key)
            
            # sample Normal(0, 1)
            key, subkey = random.split(key)
            Z = random.normal(
                key=subkey,
                shape=jnp.shape(sample_mean) + (1,),
                dtype=self.REAL
            )       
            
            # compute L s.t. cov = L * L' and reparameterize
            L = jnp.linalg.cholesky(sample_cov)
            sample = jnp.matmul(L, Z)[..., 0] + sample_mean
            sample = jnp.moveaxis(sample, source=-1, destination=index)
            err = err1 | err2
            return sample, key, err, params
        
        return _jax_wrapped_distribution_multivariate_normal
    
    def _jax_multivariate_student(self, expr, init_params):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_MULTIVARIATE_STUDENT']
        
        _, args = expr.args
        mean, cov, df = args
        jax_mean = self._jax(mean, init_params)
        jax_cov = self._jax(cov, init_params)
        jax_df = self._jax(df, init_params)
        index, = self.traced.cached_sim_info(expr)
        
        # reparameterization trick MN(m, LL') = LZ + m, where Z ~ StudentT(0, 1)
        def _jax_wrapped_distribution_multivariate_student(x, params, key):
            
            # sample the mean and covariance and degrees of freedom
            sample_mean, key, err1, params = jax_mean(x, params, key)
            sample_cov, key, err2, params = jax_cov(x, params, key)
            sample_df, key, err3, params = jax_df(x, params, key)
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
    
    def _jax_dirichlet(self, expr, init_params):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_DIRICHLET']
        
        _, args = expr.args
        alpha, = args
        jax_alpha = self._jax(alpha, init_params)
        index, = self.traced.cached_sim_info(expr)
        
        # sample Gamma(alpha_i, 1) and normalize across i
        def _jax_wrapped_distribution_dirichlet(x, params, key):
            alpha, key, error, params = jax_alpha(x, params, key)
            out_of_bounds = jnp.logical_not(jnp.all(alpha > 0))
            error |= (out_of_bounds * ERR)
            key, subkey = random.split(key)
            Gamma = random.gamma(key=subkey, a=alpha, dtype=self.REAL)
            sample = Gamma / jnp.sum(Gamma, axis=-1, keepdims=True)
            sample = jnp.moveaxis(sample, source=-1, destination=index)
            return sample, key, error, params
        
        return _jax_wrapped_distribution_dirichlet
    
    def _jax_multinomial(self, expr, init_params):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_MULTINOMIAL']
        
        _, args = expr.args
        trials, prob = args
        jax_trials = self._jax(trials, init_params)
        jax_prob = self._jax(prob, init_params)
        index, = self.traced.cached_sim_info(expr)
        
        def _jax_wrapped_distribution_multinomial(x, params, key):
            trials, key, err1, params = jax_trials(x, params, key)
            prob, key, err2, params = jax_prob(x, params, key)
            trials = jnp.asarray(trials, dtype=self.REAL)
            prob = jnp.asarray(prob, dtype=self.REAL)
            key, subkey = random.split(key)
            dist = tfp.distributions.Multinomial(total_count=trials, probs=prob)
            sample = jnp.asarray(dist.sample(seed=subkey), dtype=self.INT)
            sample = jnp.moveaxis(sample, source=-1, destination=index)
            out_of_bounds = jnp.logical_not(jnp.all(
                (prob >= 0)
                & jnp.allclose(jnp.sum(prob, axis=-1), 1.0)
                & (trials >= 0)
            ))
            error = err1 | err2 | (out_of_bounds * ERR)
            return sample, key, error, params          
        
        return _jax_wrapped_distribution_multinomial
    
    # ===========================================================================
    # matrix algebra
    # ===========================================================================
    
    def _jax_matrix(self, expr, init_params):
        _, op = expr.etype
        if op == 'det':
            return self._jax_matrix_det(expr, init_params)
        elif op == 'inverse':
            return self._jax_matrix_inv(expr, init_params, pseudo=False)
        elif op == 'pinverse':
            return self._jax_matrix_inv(expr, init_params, pseudo=True)
        elif op == 'cholesky':
            return self._jax_matrix_cholesky(expr, init_params)
        else:
            raise RDDLNotImplementedError(
                f'Matrix operation {op} is not supported.\n' + 
                print_stack_trace(expr))
    
    def _jax_matrix_det(self, expr, init_params):
        * _, arg = expr.args
        jax_arg = self._jax(arg, init_params)
        
        def _jax_wrapped_matrix_operation_det(x, params, key):
            sample_arg, key, error, params = jax_arg(x, params, key)
            sample = jnp.linalg.det(sample_arg)
            return sample, key, error, params
        
        return _jax_wrapped_matrix_operation_det
    
    def _jax_matrix_inv(self, expr, init_params, pseudo):
        _, arg = expr.args
        jax_arg = self._jax(arg, init_params)
        indices = self.traced.cached_sim_info(expr)
        op = jnp.linalg.pinv if pseudo else jnp.linalg.inv
        
        def _jax_wrapped_matrix_operation_inv(x, params, key):
            sample_arg, key, error, params = jax_arg(x, params, key)
            sample = op(sample_arg)
            sample = jnp.moveaxis(sample, source=(-2, -1), destination=indices)
            return sample, key, error, params
        
        return _jax_wrapped_matrix_operation_inv
    
    def _jax_matrix_cholesky(self, expr, init_params):
        _, arg = expr.args
        jax_arg = self._jax(arg, init_params)
        indices = self.traced.cached_sim_info(expr)
        op = jnp.linalg.cholesky
        
        def _jax_wrapped_matrix_operation_cholesky(x, params, key):
            sample_arg, key, error, params = jax_arg(x, params, key)
            sample = op(sample_arg)
            sample = jnp.moveaxis(sample, source=(-2, -1), destination=indices)
            return sample, key, error, params
        
        return _jax_wrapped_matrix_operation_cholesky
            
