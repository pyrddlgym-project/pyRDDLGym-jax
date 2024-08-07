from functools import partial
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
    RDDLNotImplementedError
)
from pyRDDLGym.core.debug.logger import Logger
from pyRDDLGym.core.simulator import RDDLSimulatorPrecompiled

# more robust approach - if user does not have this or broken try to continue
try:
    from tensorflow_probability.substrates import jax as tfp
except Exception:
    raise_warning('Failed to import tensorflow-probability: '
                  'compilation of some complex distributions '
                  '(Binomial, Negative-Binomial, Multinomial) will fail.', 'red')
    traceback.print_exc()
    tfp = None


# ===========================================================================
# EXACT RDDL TO JAX COMPILATION RULES
# ===========================================================================
     
def _function_unary_exact_named(op, name):
        
    def _jax_wrapped_unary_fn_exact(x, param):
        return op(x)
        
    return _jax_wrapped_unary_fn_exact
        
        
def _function_unary_exact_named_gamma():
        
    def _jax_wrapped_unary_gamma_exact(x, param):
        return jnp.exp(scipy.special.gammaln(x))
        
    return _jax_wrapped_unary_gamma_exact        
    
    
def _function_binary_exact_named(op, name):
        
    def _jax_wrapped_binary_fn_exact(x, y, param):
        return op(x, y)
        
    return _jax_wrapped_binary_fn_exact
    
    
def _function_binary_exact_named_implies():
        
    def _jax_wrapped_binary_implies_exact(x, y, param):
        return jnp.logical_or(jnp.logical_not(x), y)
        
    return _jax_wrapped_binary_implies_exact
    
    
def _function_binary_exact_named_log():
        
    def _jax_wrapped_binary_log_exact(x, y, param):
        return jnp.log(x) / jnp.log(y)
        
    return _jax_wrapped_binary_log_exact
    
    
def _function_aggregation_exact_named(op, name):
        
    def _jax_wrapped_aggregation_fn_exact(x, axis, param):
        return op(x, axis=axis)
        
    return _jax_wrapped_aggregation_fn_exact
    
    
def _function_if_exact_named():
        
    def _jax_wrapped_if_exact(c, a, b, param):
        return jnp.where(c > 0.5, a, b)
        
    return _jax_wrapped_if_exact
    
    
def _function_switch_exact_named():
        
    def _jax_wrapped_switch_exact(pred, cases, param):
        pred = pred[jnp.newaxis, ...]
        sample = jnp.take_along_axis(cases, pred, axis=0)
        assert sample.shape[0] == 1
        return sample[0, ...]

    return _jax_wrapped_switch_exact
    
    
def _function_bernoulli_exact_named():
        
    def _jax_wrapped_bernoulli_exact(key, prob, param):
        return random.bernoulli(key, prob)
        
    return _jax_wrapped_bernoulli_exact
    
    
def _function_discrete_exact_named():
        
    def _jax_wrapped_discrete_exact(key, prob, param):
        return random.categorical(key=key, logits=jnp.log(prob), axis=-1)
        
    return _jax_wrapped_discrete_exact


def _function_poisson_exact_named():
    
    def _jax_wrapped_poisson_exact(key, rate, param):
        return random.poisson(key=key, lam=rate, dtype=jnp.int64)
    
    return _jax_wrapped_poisson_exact


def _function_geometric_exact_named():
    
    def _jax_wrapped_geometric_exact(key, prob, param):
        return random.geometric(key=key, p=prob, dtype=jnp.int64)
    
    return _jax_wrapped_geometric_exact


class JaxRDDLCompiler:
    '''Compiles a RDDL AST representation into an equivalent JAX representation.
    All operations are identical to their numpy equivalents.
    '''
    
    MODEL_PARAM_TAG_SEPARATOR = '___'
    
    # ===========================================================================
    # EXACT RDDL TO JAX COMPILATION RULES BY DEFAULT
    # ===========================================================================

    EXACT_RDDL_TO_JAX_NEGATIVE = _function_unary_exact_named(jnp.negative, 'negative')
    
    EXACT_RDDL_TO_JAX_ARITHMETIC = {
        '+': _function_binary_exact_named(jnp.add, 'add'),
        '-': _function_binary_exact_named(jnp.subtract, 'subtract'),
        '*': _function_binary_exact_named(jnp.multiply, 'multiply'),
        '/': _function_binary_exact_named(jnp.divide, 'divide')
    }    
        
    EXACT_RDDL_TO_JAX_RELATIONAL = {
        '>=': _function_binary_exact_named(jnp.greater_equal, 'greater_equal'),
        '<=': _function_binary_exact_named(jnp.less_equal, 'less_equal'),
        '<': _function_binary_exact_named(jnp.less, 'less'),
        '>': _function_binary_exact_named(jnp.greater, 'greater'),
        '==': _function_binary_exact_named(jnp.equal, 'equal'),
        '~=': _function_binary_exact_named(jnp.not_equal, 'not_equal')
    }
        
    EXACT_RDDL_TO_JAX_LOGICAL = {
        '^': _function_binary_exact_named(jnp.logical_and, 'and'),
        '&': _function_binary_exact_named(jnp.logical_and, 'and'),
        '|': _function_binary_exact_named(jnp.logical_or, 'or'),
        '~': _function_binary_exact_named(jnp.logical_xor, 'xor'),
        '=>': _function_binary_exact_named_implies(),
        '<=>': _function_binary_exact_named(jnp.equal, 'iff')
    }
    
    EXACT_RDDL_TO_JAX_LOGICAL_NOT = _function_unary_exact_named(jnp.logical_not, 'not')
    
    EXACT_RDDL_TO_JAX_AGGREGATION = {
        'sum': _function_aggregation_exact_named(jnp.sum, 'sum'),
        'avg': _function_aggregation_exact_named(jnp.mean, 'avg'),
        'prod': _function_aggregation_exact_named(jnp.prod, 'prod'),
        'minimum': _function_aggregation_exact_named(jnp.min, 'minimum'),
        'maximum': _function_aggregation_exact_named(jnp.max, 'maximum'),
        'forall': _function_aggregation_exact_named(jnp.all, 'forall'),
        'exists': _function_aggregation_exact_named(jnp.any, 'exists'),
        'argmin': _function_aggregation_exact_named(jnp.argmin, 'argmin'),
        'argmax': _function_aggregation_exact_named(jnp.argmax, 'argmax')
    }
    
    EXACT_RDDL_TO_JAX_UNARY = {        
        'abs': _function_unary_exact_named(jnp.abs, 'abs'),
        'sgn': _function_unary_exact_named(jnp.sign, 'sgn'),
        'round': _function_unary_exact_named(jnp.round, 'round'),
        'floor': _function_unary_exact_named(jnp.floor, 'floor'),
        'ceil': _function_unary_exact_named(jnp.ceil, 'ceil'),
        'cos': _function_unary_exact_named(jnp.cos, 'cos'),
        'sin': _function_unary_exact_named(jnp.sin, 'sin'),
        'tan': _function_unary_exact_named(jnp.tan, 'tan'),
        'acos': _function_unary_exact_named(jnp.arccos, 'acos'),
        'asin': _function_unary_exact_named(jnp.arcsin, 'asin'),
        'atan': _function_unary_exact_named(jnp.arctan, 'atan'),
        'cosh': _function_unary_exact_named(jnp.cosh, 'cosh'),
        'sinh': _function_unary_exact_named(jnp.sinh, 'sinh'),
        'tanh': _function_unary_exact_named(jnp.tanh, 'tanh'),
        'exp': _function_unary_exact_named(jnp.exp, 'exp'),
        'ln': _function_unary_exact_named(jnp.log, 'ln'),
        'sqrt': _function_unary_exact_named(jnp.sqrt, 'sqrt'),
        'lngamma': _function_unary_exact_named(scipy.special.gammaln, 'lngamma'),
        'gamma': _function_unary_exact_named_gamma()
    }      
        
    EXACT_RDDL_TO_JAX_BINARY = {
        'div': _function_binary_exact_named(jnp.floor_divide, 'div'),
        'mod': _function_binary_exact_named(jnp.mod, 'mod'),
        'fmod': _function_binary_exact_named(jnp.mod, 'fmod'),
        'min': _function_binary_exact_named(jnp.minimum, 'min'),
        'max': _function_binary_exact_named(jnp.maximum, 'max'),
        'pow': _function_binary_exact_named(jnp.power, 'pow'),
        'log': _function_binary_exact_named_log(),
        'hypot': _function_binary_exact_named(jnp.hypot, 'hypot'),
    }
    
    EXACT_RDDL_TO_JAX_IF = _function_if_exact_named()
    EXACT_RDDL_TO_JAX_SWITCH = _function_switch_exact_named()
    
    EXACT_RDDL_TO_JAX_BERNOULLI = _function_bernoulli_exact_named()
    EXACT_RDDL_TO_JAX_DISCRETE = _function_discrete_exact_named()
    EXACT_RDDL_TO_JAX_POISSON = _function_poisson_exact_named()
    EXACT_RDDL_TO_JAX_GEOMETRIC = _function_geometric_exact_named()

    def __init__(self, rddl: RDDLLiftedModel,
                 allow_synchronous_state: bool=True,
                 logger: Optional[Logger]=None,
                 use64bit: bool=False,
                 compile_non_fluent_exact: bool=True) -> None:
        '''Creates a new RDDL to Jax compiler.
        
        :param rddl: the RDDL model to compile into Jax
        :param allow_synchronous_state: whether next-state components can depend
        on each other
        :param logger: to log information about compilation to file
        :param use64bit: whether to use 64 bit arithmetic
        :param compile_non_fluent_exact: whether non-fluent expressions 
        are always compiled using exact JAX expressions.
        '''
        self.rddl = rddl
        self.logger = logger
        # jax.config.update('jax_log_compiles', True) # for testing ONLY
        
        self.use64bit = use64bit
        if use64bit:
            self.INT = jnp.int64
            self.REAL = jnp.float64
            jax.config.update('jax_enable_x64', True)
        else:
            self.INT = jnp.int32
            self.REAL = jnp.float32
            jax.config.update('jax_enable_x64', False)
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
        sorter = RDDLLevelAnalysis(
            rddl, allow_synchronous_state=allow_synchronous_state)
        self.levels = sorter.compute_levels()        
        
        # trace expressions to cache information to be used later
        tracer = RDDLObjectsTracer(rddl, cpf_levels=self.levels)
        self.traced = tracer.trace()
        
        # extract the box constraints on actions
        simulator = RDDLSimulatorPrecompiled(
            rddl=self.rddl,
            init_values=self.init_values,
            levels=self.levels,
            trace_info=self.traced)  
        constraints = RDDLConstraints(simulator, vectorized=True)
        self.constraints = constraints
        
        # basic operations - these can be override in subclasses
        self.compile_non_fluent_exact = compile_non_fluent_exact
        self.NEGATIVE = JaxRDDLCompiler.EXACT_RDDL_TO_JAX_NEGATIVE
        self.ARITHMETIC_OPS = JaxRDDLCompiler.EXACT_RDDL_TO_JAX_ARITHMETIC.copy()
        self.RELATIONAL_OPS = JaxRDDLCompiler.EXACT_RDDL_TO_JAX_RELATIONAL.copy()
        self.LOGICAL_NOT = JaxRDDLCompiler.EXACT_RDDL_TO_JAX_LOGICAL_NOT
        self.LOGICAL_OPS = JaxRDDLCompiler.EXACT_RDDL_TO_JAX_LOGICAL.copy()
        self.AGGREGATION_OPS = JaxRDDLCompiler.EXACT_RDDL_TO_JAX_AGGREGATION.copy()
        self.AGGREGATION_BOOL = {'forall', 'exists'}
        self.KNOWN_UNARY = JaxRDDLCompiler.EXACT_RDDL_TO_JAX_UNARY.copy()
        self.KNOWN_BINARY = JaxRDDLCompiler.EXACT_RDDL_TO_JAX_BINARY.copy()
        self.IF_HELPER = JaxRDDLCompiler.EXACT_RDDL_TO_JAX_IF
        self.SWITCH_HELPER = JaxRDDLCompiler.EXACT_RDDL_TO_JAX_SWITCH
        self.BERNOULLI_HELPER = JaxRDDLCompiler.EXACT_RDDL_TO_JAX_BERNOULLI
        self.DISCRETE_HELPER = JaxRDDLCompiler.EXACT_RDDL_TO_JAX_DISCRETE
        self.POISSON_HELPER = JaxRDDLCompiler.EXACT_RDDL_TO_JAX_POISSON
        self.GEOMETRIC_HELPER = JaxRDDLCompiler.EXACT_RDDL_TO_JAX_GEOMETRIC
    
    # ===========================================================================
    # main compilation subroutines
    # ===========================================================================
     
    def compile(self, log_jax_expr: bool=False, heading: str='') -> None: 
        '''Compiles the current RDDL into Jax expressions.
        
        :param log_jax_expr: whether to pretty-print the compiled Jax functions
        to the log file
        :param heading: the heading to print before compilation information
        '''
        info = ({}, [])
        self.invariants = self._compile_constraints(self.rddl.invariants, info)
        self.preconditions = self._compile_constraints(self.rddl.preconditions, info)
        self.terminations = self._compile_constraints(self.rddl.terminations, info)
        self.cpfs = self._compile_cpfs(info)
        self.reward = self._compile_reward(info)
        self.model_params = {key: value 
                             for (key, (value, *_)) in info[0].items()}
        self.relaxations = info[1]
        
        if log_jax_expr and self.logger is not None:
            printed = self.print_jax()
            printed_cpfs = '\n\n'.join(f'{k}: {v}' 
                                       for (k, v) in printed['cpfs'].items())
            printed_reward = printed['reward']
            printed_invariants = '\n\n'.join(v for v in printed['invariants'])
            printed_preconds = '\n\n'.join(v for v in printed['preconditions'])
            printed_terminals = '\n\n'.join(v for v in printed['terminations'])
            printed_params = '\n'.join(f'{k}: {v}' for (k, v) in info.items())
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
    
    def _compile_constraints(self, constraints, info):
        return [self._jax(expr, info, dtype=bool) for expr in constraints]
        
    def _compile_cpfs(self, info):
        jax_cpfs = {}
        for cpfs in self.levels.values():
            for cpf in cpfs:
                _, expr = self.rddl.cpfs[cpf]
                prange = self.rddl.variable_ranges[cpf]
                dtype = self.JAX_TYPES.get(prange, self.INT)
                jax_cpfs[cpf] = self._jax(expr, info, dtype=dtype)
        return jax_cpfs
    
    def _compile_reward(self, info):
        return self._jax(self.rddl.reward, info, dtype=self.REAL)
    
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
            
    def _jax_nonlinear_constraints(self): 
        rddl = self.rddl
        
        # extract the non-box inequality constraints on actions
        inequalities = [constr 
                        for (i, expr) in enumerate(rddl.preconditions)
                        for constr in self._extract_inequality_constraint(expr)
                        if not self.constraints.is_box_preconditions[i]]
        
        # compile them to JAX and write as h(s, a) <= 0
        op = self.ARITHMETIC_OPS['-']        
        jax_inequalities = []
        for (left, right) in inequalities:
            jax_lhs = self._jax(left, {})
            jax_rhs = self._jax(right, {})
            jax_constr = self._jax_binary(jax_lhs, jax_rhs, op, '', at_least_int=True)
            jax_inequalities.append(jax_constr)
        
        # extract the non-box equality constraints on actions
        equalities = [constr 
                      for (i, expr) in enumerate(rddl.preconditions)
                      for constr in self._extract_equality_constraint(expr)
                      if not self.constraints.is_box_preconditions[i]]
        
        # compile them to JAX and write as g(s, a) == 0
        jax_equalities = []
        for (left, right) in equalities:
            jax_lhs = self._jax(left, {})
            jax_rhs = self._jax(right, {})
            jax_constr = self._jax_binary(jax_lhs, jax_rhs, op, '', at_least_int=True)
            jax_equalities.append(jax_constr)
            
        return jax_inequalities, jax_equalities
    
    def compile_transition(self, check_constraints: bool=False,
                           constraint_func: bool=False) -> Callable:
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
        '''
        NORMAL = JaxRDDLCompiler.ERROR_CODES['NORMAL']        
        rddl = self.rddl
        reward_fn, cpfs = self.reward, self.cpfs
        preconds, invariants, terminals = \
            self.preconditions, self.invariants, self.terminations
        
        # compile constraint information
        if constraint_func:
            inequality_fns, equality_fns = self._jax_nonlinear_constraints()
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
                    sample, key, err = precond(subs, model_params, key)
                    precond_check = jnp.logical_and(precond_check, sample)
                    errors |= err
            
            # compute h(s, a) <= 0 and g(s, a) == 0 constraint functions
            inequalities, equalities = [], []
            if constraint_func:
                for constraint in inequality_fns:
                    sample, key, err = constraint(subs, model_params, key)
                    inequalities.append(sample)
                    errors |= err
                for constraint in equality_fns:
                    sample, key, err = constraint(subs, model_params, key)
                    equalities.append(sample)
                    errors |= err
                
            # calculate CPFs in topological order
            for (name, cpf) in cpfs.items():
                subs[name], key, err = cpf(subs, model_params, key)
                errors |= err                
                
            # calculate the immediate reward
            reward, key, err = reward_fn(subs, model_params, key)
            errors |= err
            
            # calculate fluent values
            fluents = {name: values for (name, values) in subs.items() 
                       if name not in rddl.non_fluents}
            
            # set the next state to the current state
            for (state, next_state) in rddl.next_state.items():
                subs[state] = subs[next_state]
            
            # check the state invariants
            invariant_check = True
            if check_constraints:
                for invariant in invariants:
                    sample, key, err = invariant(subs, model_params, key)
                    invariant_check = jnp.logical_and(invariant_check, sample)
                    errors |= err
            
            # check the termination (TODO: zero out reward in s if terminated)
            terminated_check = False
            if check_constraints:
                for terminal in terminals:
                    sample, key, err = terminal(subs, model_params, key)
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
                
            return subs, log
        
        return _jax_wrapped_single_step        
    
    def compile_rollouts(self, policy: Callable,
                         n_steps: int,
                         n_batch: int,
                         check_constraints: bool=False,
                         constraint_func: bool=False) -> Callable:
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
              constraints that were satisfied, errors, etc. 
            
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
        '''
        rddl = self.rddl
        jax_step_fn = self.compile_transition(check_constraints, constraint_func)
        
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
            subs, log = jax_step_fn(subkey, actions, subs, model_params)
            return subs, log
                        
        # do a batched step update from the policy
        def _jax_wrapped_batched_step_policy(carry, step):
            key, policy_params, hyperparams, subs, model_params = carry  
            key, *subkeys = random.split(key, num=1 + n_batch)
            keys = jnp.asarray(subkeys)
            subs, log = jax.vmap(
                _jax_wrapped_single_step_policy,
                in_axes=(0, None, None, None, 0, None)
            )(keys, policy_params, hyperparams, step, subs, model_params)
            carry = (key, policy_params, hyperparams, subs, model_params)
            return carry, log            
            
        # do a batched roll-out from the policy
        def _jax_wrapped_batched_rollout(key, policy_params, hyperparams, 
                                         subs, model_params):
            start = (key, policy_params, hyperparams, subs, model_params)
            steps = jnp.arange(n_steps)
            _, log = jax.lax.scan(_jax_wrapped_batched_step_policy, start, steps)
            log = jax.tree_map(partial(jnp.swapaxes, axis1=0, axis2=1), log)
            return log
        
        return _jax_wrapped_batched_rollout
    
    # ===========================================================================
    # error checks
    # ===========================================================================
    
    def print_jax(self) -> Dict[str, Any]:
        '''Returns a dictionary containing the string representations of all 
        Jax compiled expressions from the RDDL file.
        '''
        subs = self.init_values
        params = self.model_params
        key = jax.random.PRNGKey(42)
        printed = {}
        printed['cpfs'] = {name: str(jax.make_jaxpr(expr)(subs, params, key))
                           for (name, expr) in self.cpfs.items()}
        printed['reward'] = str(jax.make_jaxpr(self.reward)(subs, params, key))
        printed['invariants'] = [str(jax.make_jaxpr(expr)(subs, params, key))
                                 for expr in self.invariants]
        printed['preconditions'] = [str(jax.make_jaxpr(expr)(subs, params, key))
                                    for expr in self.preconditions]
        printed['terminations'] = [str(jax.make_jaxpr(expr)(subs, params, key))
                                   for expr in self.terminations]
        return printed
        
    @staticmethod
    def _check_valid_op(expr, valid_ops):
        etype, op = expr.etype
        if op not in valid_ops:
            valid_op_str = ','.join(valid_ops.keys())
            raise RDDLNotImplementedError(
                f'{etype} operator {op} is not supported: '
                f'must be in {valid_op_str}.\n' + 
                print_stack_trace(expr))
    
    @staticmethod
    def _check_num_args(expr, required_args):
        actual_args = len(expr.args)
        if actual_args != required_args:
            etype, op = expr.etype
            raise RDDLInvalidNumberOfArgumentsError(
                f'{etype} operator {op} requires {required_args} arguments, '
                f'got {actual_args}.\n' + 
                print_stack_trace(expr))
        
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
    # handling of auxiliary data (e.g. model tuning parameters)
    # ===========================================================================
    
    def _unwrap(self, op, expr_id, info):
        jax_op, name = op, None
        model_params, relaxed_list = info
        if isinstance(op, tuple):
            jax_op, param = op
            if param is not None:
                tags, values = param
                sep = JaxRDDLCompiler.MODEL_PARAM_TAG_SEPARATOR
                if isinstance(tags, tuple):
                    name = sep.join(tags)
                else:
                    name = str(tags)
                name = f'{name}{sep}{expr_id}'
                if name in model_params:
                    raise RuntimeError(
                        f'Internal error: model parameter {name} is already defined.')
                model_params[name] = (values, tags, expr_id, jax_op.__name__)
            relaxed_list.append((param, expr_id, jax_op.__name__))
        return jax_op, name
    
    def summarize_model_relaxations(self) -> str:
        '''Returns a string of information about model relaxations in the
        compiled model.'''
        occurence_by_type = {}
        for (_, expr_id, jax_op) in self.relaxations:
            etype = self.traced.lookup(expr_id).etype
            source = f'{etype[1]} ({etype[0]})'
            sub = f'{source:<30} --> {jax_op}'
            occurence_by_type[sub] = occurence_by_type.get(sub, 0) + 1        
        col = "{:<80} {:<10}\n"
        table = col.format('Substitution', 'Count')
        for (sub, occurs) in occurence_by_type.items():
            table += col.format(sub, occurs)
        return table
        
    # ===========================================================================
    # expression compilation
    # ===========================================================================
    
    def _jax(self, expr, info, dtype=None):
        etype, _ = expr.etype
        if etype == 'constant':
            jax_expr = self._jax_constant(expr, info)
        elif etype == 'pvar':
            jax_expr = self._jax_pvar(expr, info)
        elif etype == 'arithmetic':
            jax_expr = self._jax_arithmetic(expr, info)
        elif etype == 'relational':
            jax_expr = self._jax_relational(expr, info)
        elif etype == 'boolean':
            jax_expr = self._jax_logical(expr, info)
        elif etype == 'aggregation':
            jax_expr = self._jax_aggregation(expr, info)
        elif etype == 'func':
            jax_expr = self._jax_functional(expr, info)
        elif etype == 'control':
            jax_expr = self._jax_control(expr, info)
        elif etype == 'randomvar':
            jax_expr = self._jax_random(expr, info)
        elif etype == 'randomvector':
            jax_expr = self._jax_random_vector(expr, info)
        elif etype == 'matrix':
            jax_expr = self._jax_matrix(expr, info)
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
            val, key, err = jax_expr(x, params, key)
            sample = jnp.asarray(val, dtype=dtype)
            invalid_cast = jnp.logical_and(
                jnp.logical_not(jnp.can_cast(val, dtype)),
                jnp.any(sample != val))
            err |= (invalid_cast * ERR)
            return sample, key, err
        
        return _jax_wrapped_cast
   
    def _fix_dtype(self, value):
        dtype = jnp.atleast_1d(value).dtype
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
    
    def _jax_constant(self, expr, info):
        NORMAL = JaxRDDLCompiler.ERROR_CODES['NORMAL']
        cached_value = self.traced.cached_sim_info(expr)
        
        def _jax_wrapped_constant(x, params, key):
            sample = jnp.asarray(cached_value, dtype=self._fix_dtype(cached_value))
            return sample, key, NORMAL

        return _jax_wrapped_constant
    
    def _jax_pvar_slice(self, _slice):
        NORMAL = JaxRDDLCompiler.ERROR_CODES['NORMAL']
        
        def _jax_wrapped_pvar_slice(x, params, key):
            return _slice, key, NORMAL
        
        return _jax_wrapped_pvar_slice
            
    def _jax_pvar(self, expr, info):
        NORMAL = JaxRDDLCompiler.ERROR_CODES['NORMAL']
        var, pvars = expr.args  
        is_value, cached_info = self.traced.cached_sim_info(expr)
        
        # boundary case: free variable is converted to array (0, 1, 2...)
        # boundary case: domain object is converted to canonical integer index
        if is_value:
            cached_value = cached_info

            def _jax_wrapped_object(x, params, key):
                sample = jnp.asarray(cached_value, dtype=self._fix_dtype(cached_value))
                return sample, key, NORMAL
            
            return _jax_wrapped_object
        
        # boundary case: no shape information (e.g. scalar pvar)
        elif cached_info is None:
            
            def _jax_wrapped_pvar_scalar(x, params, key):
                value = x[var]
                sample = jnp.asarray(value, dtype=self._fix_dtype(value))
                return sample, key, NORMAL
            
            return _jax_wrapped_pvar_scalar
        
        # must slice and/or reshape value tensor to match free variables
        else:
            slices, axis, shape, op_code, op_args = cached_info 
        
            # compile nested expressions
            if slices and op_code == RDDLObjectsTracer.NUMPY_OP_CODE.NESTED_SLICE:
                
                jax_nested_expr = [(self._jax(arg, info) 
                                    if _slice is None 
                                    else self._jax_pvar_slice(_slice))
                                   for (arg, _slice) in zip(pvars, slices)]    
                
                def _jax_wrapped_pvar_tensor_nested(x, params, key):
                    error = NORMAL
                    value = x[var]
                    sample = jnp.asarray(value, dtype=self._fix_dtype(value))
                    new_slices = [None] * len(jax_nested_expr)
                    for (i, jax_expr) in enumerate(jax_nested_expr):
                        new_slices[i], key, err = jax_expr(x, params, key)
                        error |= err
                    new_slices = tuple(new_slices)
                    sample = sample[new_slices]
                    return sample, key, error
                
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
                    return sample, key, NORMAL
                
                return _jax_wrapped_pvar_tensor_non_nested
    
    # ===========================================================================
    # mathematical
    # ===========================================================================
    
    def _jax_unary(self, jax_expr, jax_op, jax_param,
                   at_least_int=False, check_dtype=None):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_CAST']

        def _jax_wrapped_unary_op(x, params, key):
            sample, key, err = jax_expr(x, params, key)
            if at_least_int:
                sample = self.ONE * sample
            param = params.get(jax_param, None)
            sample = jax_op(sample, param)
            if check_dtype is not None:
                invalid_cast = jnp.logical_not(jnp.can_cast(sample, check_dtype))
                err |= (invalid_cast * ERR)
            return sample, key, err
        
        return _jax_wrapped_unary_op
    
    def _jax_binary(self, jax_lhs, jax_rhs, jax_op, jax_param,
                    at_least_int=False, check_dtype=None):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_CAST']
        
        def _jax_wrapped_binary_op(x, params, key):
            sample1, key, err1 = jax_lhs(x, params, key)
            sample2, key, err2 = jax_rhs(x, params, key)
            if at_least_int:
                sample1 = self.ONE * sample1
                sample2 = self.ONE * sample2
            param = params.get(jax_param, None)
            sample = jax_op(sample1, sample2, param)
            err = err1 | err2
            if check_dtype is not None:
                invalid_cast = jnp.logical_not(jnp.logical_and(
                    jnp.can_cast(sample1, check_dtype),
                    jnp.can_cast(sample2, check_dtype)))
                err |= (invalid_cast * ERR)
            return sample, key, err
        
        return _jax_wrapped_binary_op
    
    def _jax_arithmetic(self, expr, info):
        _, op = expr.etype
        
        # if expression is non-fluent, always use the exact operation
        if self.compile_non_fluent_exact and not self.traced.cached_is_fluent(expr):
            valid_ops = JaxRDDLCompiler.EXACT_RDDL_TO_JAX_ARITHMETIC
            negative_op = JaxRDDLCompiler.EXACT_RDDL_TO_JAX_NEGATIVE
        else:
            valid_ops = self.ARITHMETIC_OPS
            negative_op = self.NEGATIVE            
        JaxRDDLCompiler._check_valid_op(expr, valid_ops)
        
        # recursively compile arguments
        args = expr.args
        n = len(args)
        if n == 1 and op == '-':
            arg, = args
            jax_expr = self._jax(arg, info)
            jax_op, jax_param = self._unwrap(negative_op, expr.id, info)
            return self._jax_unary(jax_expr, jax_op, jax_param, at_least_int=True)
                    
        elif n == 2 or (n >= 2 and op in {'*', '+'}):
            jax_exprs = [self._jax(arg, info) for arg in args]
            jax_op, jax_param = self._unwrap(valid_ops[op], expr.id, info)
            result = jax_exprs[0]
            for jax_rhs in jax_exprs[1:]:
                result = self._jax_binary(
                    result, jax_rhs, jax_op, jax_param, at_least_int=True)
            return result
        
        JaxRDDLCompiler._check_num_args(expr, 2)
    
    def _jax_relational(self, expr, info):
        _, op = expr.etype
        
        # if expression is non-fluent, always use the exact operation
        if self.compile_non_fluent_exact and not self.traced.cached_is_fluent(expr):
            valid_ops = JaxRDDLCompiler.EXACT_RDDL_TO_JAX_RELATIONAL
        else:
            valid_ops = self.RELATIONAL_OPS
        JaxRDDLCompiler._check_valid_op(expr, valid_ops)
        jax_op, jax_param = self._unwrap(valid_ops[op], expr.id, info)
        
        # recursively compile arguments
        JaxRDDLCompiler._check_num_args(expr, 2)
        lhs, rhs = expr.args
        jax_lhs = self._jax(lhs, info)
        jax_rhs = self._jax(rhs, info)
        return self._jax_binary(
            jax_lhs, jax_rhs, jax_op, jax_param, at_least_int=True)
           
    def _jax_logical(self, expr, info):
        _, op = expr.etype
        
        # if expression is non-fluent, always use the exact operation
        if self.compile_non_fluent_exact and not self.traced.cached_is_fluent(expr):
            valid_ops = JaxRDDLCompiler.EXACT_RDDL_TO_JAX_LOGICAL  
            logical_not_op = JaxRDDLCompiler.EXACT_RDDL_TO_JAX_LOGICAL_NOT
        else:
            valid_ops = self.LOGICAL_OPS 
            logical_not_op = self.LOGICAL_NOT 
        JaxRDDLCompiler._check_valid_op(expr, valid_ops)
                
        # recursively compile arguments
        args = expr.args
        n = len(args)        
        if n == 1 and op == '~':
            arg, = args
            jax_expr = self._jax(arg, info)
            jax_op, jax_param = self._unwrap(logical_not_op, expr.id, info)
            return self._jax_unary(jax_expr, jax_op, jax_param, check_dtype=bool)
        
        elif n == 2 or (n >= 2 and op in {'^', '&', '|'}):
            jax_exprs = [self._jax(arg, info) for arg in args]
            jax_op, jax_param = self._unwrap(valid_ops[op], expr.id, info)
            result = jax_exprs[0]
            for jax_rhs in jax_exprs[1:]:
                result = self._jax_binary(
                    result, jax_rhs, jax_op, jax_param, check_dtype=bool)
            return result
        
        JaxRDDLCompiler._check_num_args(expr, 2)
    
    def _jax_aggregation(self, expr, info):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_CAST']
        _, op = expr.etype
        
        # if expression is non-fluent, always use the exact operation
        if self.compile_non_fluent_exact and not self.traced.cached_is_fluent(expr):
            valid_ops = JaxRDDLCompiler.EXACT_RDDL_TO_JAX_AGGREGATION   
        else:
            valid_ops = self.AGGREGATION_OPS   
        JaxRDDLCompiler._check_valid_op(expr, valid_ops) 
        jax_op, jax_param = self._unwrap(valid_ops[op], expr.id, info)
        
        # recursively compile arguments
        is_floating = op not in self.AGGREGATION_BOOL
        * _, arg = expr.args  
        _, axes = self.traced.cached_sim_info(expr)        
        jax_expr = self._jax(arg, info)
        
        def _jax_wrapped_aggregation(x, params, key):
            sample, key, err = jax_expr(x, params, key)
            if is_floating:
                sample = self.ONE * sample
            else:
                invalid_cast = jnp.logical_not(jnp.can_cast(sample, bool))
                err |= (invalid_cast * ERR)
            param = params.get(jax_param, None)
            sample = jax_op(sample, axis=axes, param=param)
            return sample, key, err
        
        return _jax_wrapped_aggregation
               
    def _jax_functional(self, expr, info):
        _, op = expr.etype
        
        # if expression is non-fluent, always use the exact operation
        if self.compile_non_fluent_exact and not self.traced.cached_is_fluent(expr):            
            unary_ops = JaxRDDLCompiler.EXACT_RDDL_TO_JAX_UNARY
            binary_ops = JaxRDDLCompiler.EXACT_RDDL_TO_JAX_BINARY
        else:
            unary_ops = self.KNOWN_UNARY
            binary_ops = self.KNOWN_BINARY
        
        # recursively compile arguments
        if op in unary_ops:
            JaxRDDLCompiler._check_num_args(expr, 1)                            
            arg, = expr.args
            jax_expr = self._jax(arg, info)
            jax_op, jax_param = self._unwrap(unary_ops[op], expr.id, info)
            return self._jax_unary(jax_expr, jax_op, jax_param, at_least_int=True)
            
        elif op in binary_ops:
            JaxRDDLCompiler._check_num_args(expr, 2)                
            lhs, rhs = expr.args
            jax_lhs = self._jax(lhs, info)
            jax_rhs = self._jax(rhs, info)
            jax_op, jax_param = self._unwrap(binary_ops[op], expr.id, info)
            return self._jax_binary(
                jax_lhs, jax_rhs, jax_op, jax_param, at_least_int=True)
        
        raise RDDLNotImplementedError(
            f'Function {op} is not supported.\n' + 
            print_stack_trace(expr))   
    
    # ===========================================================================
    # control flow
    # ===========================================================================
    
    def _jax_control(self, expr, info):
        _, op = expr.etype        
        if op == 'if':
            return self._jax_if(expr, info)
        elif op == 'switch':
            return self._jax_switch(expr, info)
        
        raise RDDLNotImplementedError(
            f'Control operator {op} is not supported.\n' + 
            print_stack_trace(expr))   
    
    def _jax_if(self, expr, info):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_CAST']
        JaxRDDLCompiler._check_num_args(expr, 3)
        pred, if_true, if_false = expr.args     
        
        # if predicate is non-fluent, always use the exact operation
        if self.compile_non_fluent_exact and not self.traced.cached_is_fluent(pred):
            if_op = JaxRDDLCompiler.EXACT_RDDL_TO_JAX_IF
        else:
            if_op = self.IF_HELPER
        jax_if, jax_param = self._unwrap(if_op, expr.id, info)
        
        # recursively compile arguments   
        jax_pred = self._jax(pred, info)
        jax_true = self._jax(if_true, info)
        jax_false = self._jax(if_false, info)
        
        def _jax_wrapped_if_then_else(x, params, key):
            sample1, key, err1 = jax_pred(x, params, key)
            sample2, key, err2 = jax_true(x, params, key)
            sample3, key, err3 = jax_false(x, params, key)
            param = params.get(jax_param, None)
            sample = jax_if(sample1, sample2, sample3, param)
            err = err1 | err2 | err3
            invalid_cast = jnp.logical_not(jnp.can_cast(sample1, bool))
            err |= (invalid_cast * ERR)
            return sample, key, err
            
        return _jax_wrapped_if_then_else
    
    def _jax_switch(self, expr, info):
        pred, *_ = expr.args
             
        # if predicate is non-fluent, always use the exact operation
        # case conditions are currently only literals so they are non-fluent
        if self.compile_non_fluent_exact and not self.traced.cached_is_fluent(pred):
            switch_op = JaxRDDLCompiler.EXACT_RDDL_TO_JAX_SWITCH
        else:
            switch_op = self.SWITCH_HELPER
        jax_switch, jax_param = self._unwrap(switch_op, expr.id, info)
        
        # recursively compile predicate
        jax_pred = self._jax(pred, info)
        
        # recursively compile cases
        cases, default = self.traced.cached_sim_info(expr) 
        jax_default = None if default is None else self._jax(default, info)
        jax_cases = [(jax_default if _case is None else self._jax(_case, info))
                     for _case in cases]
                    
        def _jax_wrapped_switch(x, params, key):
            
            # sample predicate
            sample_pred, key, err = jax_pred(x, params, key) 
            
            # sample cases
            sample_cases = [None] * len(jax_cases)
            for (i, jax_case) in enumerate(jax_cases):
                sample_cases[i], key, err_case = jax_case(x, params, key)
                err |= err_case                
            sample_cases = jnp.asarray(
                sample_cases, dtype=self._fix_dtype(sample_cases))
            
            # predicate (enum) is an integer - use it to extract from case array
            param = params.get(jax_param, None)
            sample = jax_switch(sample_pred, sample_cases, param)
            return sample, key, err    
        
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
    
    # distributions with incomplete reparameterization support (TODO):
    # Binomial: (use truncation and Gumbel-softmax)
    # NegativeBinomial: (no reparameterization)
    # Poisson: (use truncation and Gumbel-softmax)
    # Gamma, ChiSquare: (no shape reparameterization)
    # Beta: (no reparameterization)
    # Geometric: (implement safe floor)
    # Student: (no reparameterization)
    
    def _jax_random(self, expr, info):
        _, name = expr.etype
        if name == 'KronDelta':
            return self._jax_kron(expr, info)        
        elif name == 'DiracDelta':
            return self._jax_dirac(expr, info)
        elif name == 'Uniform':
            return self._jax_uniform(expr, info)
        elif name == 'Bernoulli':
            return self._jax_bernoulli(expr, info)
        elif name == 'Normal':
            return self._jax_normal(expr, info)  
        elif name == 'Poisson':
            return self._jax_poisson(expr, info)
        elif name == 'Exponential':
            return self._jax_exponential(expr, info)
        elif name == 'Weibull':
            return self._jax_weibull(expr, info) 
        elif name == 'Gamma':
            return self._jax_gamma(expr, info)
        elif name == 'Binomial':
            return self._jax_binomial(expr, info)
        elif name == 'NegativeBinomial':
            return self._jax_negative_binomial(expr, info)
        elif name == 'Beta':
            return self._jax_beta(expr, info)
        elif name == 'Geometric':
            return self._jax_geometric(expr, info)
        elif name == 'Pareto':
            return self._jax_pareto(expr, info)
        elif name == 'Student':
            return self._jax_student(expr, info)
        elif name == 'Gumbel':
            return self._jax_gumbel(expr, info)
        elif name == 'Laplace':
            return self._jax_laplace(expr, info)
        elif name == 'Cauchy':
            return self._jax_cauchy(expr, info)
        elif name == 'Gompertz':
            return self._jax_gompertz(expr, info)
        elif name == 'ChiSquare':
            return self._jax_chisquare(expr, info)
        elif name == 'Kumaraswamy':
            return self._jax_kumaraswamy(expr, info)
        elif name == 'Discrete':
            return self._jax_discrete(expr, info, unnorm=False)
        elif name == 'UnnormDiscrete':
            return self._jax_discrete(expr, info, unnorm=True)
        elif name == 'Discrete(p)':
            return self._jax_discrete_pvar(expr, info, unnorm=False)
        elif name == 'UnnormDiscrete(p)':
            return self._jax_discrete_pvar(expr, info, unnorm=True)
        else:
            raise RDDLNotImplementedError(
                f'Distribution {name} is not supported.\n' + 
                print_stack_trace(expr))
        
    def _jax_kron(self, expr, info):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_KRON_DELTA']
        JaxRDDLCompiler._check_num_args(expr, 1)
        arg, = expr.args
        arg = self._jax(arg, info)
        
        # just check that the sample can be cast to int
        def _jax_wrapped_distribution_kron(x, params, key):
            sample, key, err = arg(x, params, key)
            invalid_cast = jnp.logical_not(jnp.can_cast(sample, self.INT))
            err |= (invalid_cast * ERR)
            return sample, key, err
                        
        return _jax_wrapped_distribution_kron
    
    def _jax_dirac(self, expr, info):
        JaxRDDLCompiler._check_num_args(expr, 1)
        arg, = expr.args
        arg = self._jax(arg, info, dtype=self.REAL)
        return arg
    
    def _jax_uniform(self, expr, info):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_UNIFORM']
        JaxRDDLCompiler._check_num_args(expr, 2)
        
        arg_lb, arg_ub = expr.args
        jax_lb = self._jax(arg_lb, info)
        jax_ub = self._jax(arg_ub, info)
        
        # reparameterization trick U(a, b) = a + (b - a) * U(0, 1)
        def _jax_wrapped_distribution_uniform(x, params, key):
            lb, key, err1 = jax_lb(x, params, key)
            ub, key, err2 = jax_ub(x, params, key)
            key, subkey = random.split(key)
            U = random.uniform(key=subkey, shape=jnp.shape(lb), dtype=self.REAL)
            sample = lb + (ub - lb) * U
            out_of_bounds = jnp.logical_not(jnp.all(lb <= ub))
            err = err1 | err2 | (out_of_bounds * ERR)
            return sample, key, err
        
        return _jax_wrapped_distribution_uniform
    
    def _jax_normal(self, expr, info):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_NORMAL']
        JaxRDDLCompiler._check_num_args(expr, 2)
        
        arg_mean, arg_var = expr.args
        jax_mean = self._jax(arg_mean, info)
        jax_var = self._jax(arg_var, info)
        
        # reparameterization trick N(m, s^2) = m + s * N(0, 1)
        def _jax_wrapped_distribution_normal(x, params, key):
            mean, key, err1 = jax_mean(x, params, key)
            var, key, err2 = jax_var(x, params, key)
            std = jnp.sqrt(var)
            key, subkey = random.split(key)
            Z = random.normal(key=subkey, shape=jnp.shape(mean), dtype=self.REAL)
            sample = mean + std * Z
            out_of_bounds = jnp.logical_not(jnp.all(var >= 0))
            err = err1 | err2 | (out_of_bounds * ERR)
            return sample, key, err
        
        return _jax_wrapped_distribution_normal
    
    def _jax_exponential(self, expr, info):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_EXPONENTIAL']
        JaxRDDLCompiler._check_num_args(expr, 1)
        
        arg_scale, = expr.args
        jax_scale = self._jax(arg_scale, info)
        
        # reparameterization trick Exp(s) = s * Exp(1)
        def _jax_wrapped_distribution_exp(x, params, key):
            scale, key, err = jax_scale(x, params, key)
            key, subkey = random.split(key)
            Exp1 = random.exponential(
                key=subkey, shape=jnp.shape(scale), dtype=self.REAL)
            sample = scale * Exp1
            out_of_bounds = jnp.logical_not(jnp.all(scale > 0))
            err |= (out_of_bounds * ERR)
            return sample, key, err
        
        return _jax_wrapped_distribution_exp
    
    def _jax_weibull(self, expr, info):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_WEIBULL']
        JaxRDDLCompiler._check_num_args(expr, 2)
        
        arg_shape, arg_scale = expr.args
        jax_shape = self._jax(arg_shape, info)
        jax_scale = self._jax(arg_scale, info)
        
        # reparameterization trick W(s, r) = r * (-ln(1 - U(0, 1))) ** (1 / s)
        def _jax_wrapped_distribution_weibull(x, params, key):
            shape, key, err1 = jax_shape(x, params, key)
            scale, key, err2 = jax_scale(x, params, key)
            key, subkey = random.split(key)
            U = random.uniform(key=subkey, shape=jnp.shape(scale), dtype=self.REAL)
            sample = scale * jnp.power(-jnp.log(U), 1.0 / shape)
            out_of_bounds = jnp.logical_not(jnp.all((shape > 0) & (scale > 0)))
            err = err1 | err2 | (out_of_bounds * ERR)
            return sample, key, err
        
        return _jax_wrapped_distribution_weibull
    
    def _jax_bernoulli(self, expr, info):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_BERNOULLI']
        JaxRDDLCompiler._check_num_args(expr, 1)
        arg_prob, = expr.args
        
        # if probability is non-fluent, always use the exact operation
        if self.compile_non_fluent_exact and not self.traced.cached_is_fluent(arg_prob):
            bern_op = JaxRDDLCompiler.EXACT_RDDL_TO_JAX_BERNOULLI
        else:
            bern_op = self.BERNOULLI_HELPER
        jax_bern, jax_param = self._unwrap(bern_op, expr.id, info)
        
        # recursively compile arguments
        jax_prob = self._jax(arg_prob, info)
        
        def _jax_wrapped_distribution_bernoulli(x, params, key):
            prob, key, err = jax_prob(x, params, key)
            key, subkey = random.split(key)
            param = params.get(jax_param, None)
            sample = jax_bern(subkey, prob, param)
            out_of_bounds = jnp.logical_not(jnp.all((prob >= 0) & (prob <= 1)))
            err |= (out_of_bounds * ERR)
            return sample, key, err
        
        return _jax_wrapped_distribution_bernoulli
    
    def _jax_poisson(self, expr, info):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_POISSON']
        JaxRDDLCompiler._check_num_args(expr, 1)
        arg_rate, = expr.args
        
        # if rate is non-fluent, always use the exact operation
        if self.compile_non_fluent_exact and not self.traced.cached_is_fluent(arg_rate):
            poisson_op = JaxRDDLCompiler.EXACT_RDDL_TO_JAX_POISSON
        else:
            poisson_op = self.POISSON_HELPER
        jax_poisson, jax_param = self._unwrap(poisson_op, expr.id, info)
        
        # recursively compile arguments
        jax_rate = self._jax(arg_rate, info)
        
        # uses the implicit JAX subroutine
        def _jax_wrapped_distribution_poisson(x, params, key):
            rate, key, err = jax_rate(x, params, key)
            key, subkey = random.split(key)
            param = params.get(jax_param, None)
            sample = jax_poisson(subkey, rate, param).astype(self.INT)
            out_of_bounds = jnp.logical_not(jnp.all(rate >= 0))
            err |= (out_of_bounds * ERR)
            return sample, key, err
        
        return _jax_wrapped_distribution_poisson
    
    def _jax_gamma(self, expr, info):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_GAMMA']
        JaxRDDLCompiler._check_num_args(expr, 2)
        
        arg_shape, arg_scale = expr.args
        jax_shape = self._jax(arg_shape, info)
        jax_scale = self._jax(arg_scale, info)
        
        # partial reparameterization trick Gamma(s, r) = r * Gamma(s, 1)
        # uses the implicit JAX subroutine for Gamma(s, 1) 
        def _jax_wrapped_distribution_gamma(x, params, key):
            shape, key, err1 = jax_shape(x, params, key)
            scale, key, err2 = jax_scale(x, params, key)
            key, subkey = random.split(key)
            Gamma = random.gamma(key=subkey, a=shape, dtype=self.REAL)
            sample = scale * Gamma
            out_of_bounds = jnp.logical_not(jnp.all((shape > 0) & (scale > 0)))
            err = err1 | err2 | (out_of_bounds * ERR)
            return sample, key, err
        
        return _jax_wrapped_distribution_gamma
    
    def _jax_binomial(self, expr, info):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_BINOMIAL']
        JaxRDDLCompiler._check_num_args(expr, 2)
        
        arg_trials, arg_prob = expr.args
        jax_trials = self._jax(arg_trials, info)
        jax_prob = self._jax(arg_prob, info)
        
        # uses the JAX substrate of tensorflow-probability
        def _jax_wrapped_distribution_binomial(x, params, key):
            trials, key, err2 = jax_trials(x, params, key)       
            prob, key, err1 = jax_prob(x, params, key)
            trials = jnp.asarray(trials, dtype=self.REAL)
            prob = jnp.asarray(prob, dtype=self.REAL)
            key, subkey = random.split(key)
            dist = tfp.distributions.Binomial(total_count=trials, probs=prob)
            sample = dist.sample(seed=subkey).astype(self.INT)
            out_of_bounds = jnp.logical_not(jnp.all(
                (prob >= 0) & (prob <= 1) & (trials >= 0)))
            err = err1 | err2 | (out_of_bounds * ERR)
            return sample, key, err
        
        return _jax_wrapped_distribution_binomial
    
    def _jax_negative_binomial(self, expr, info):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_NEGATIVE_BINOMIAL']
        JaxRDDLCompiler._check_num_args(expr, 2)
        
        arg_trials, arg_prob = expr.args
        jax_trials = self._jax(arg_trials, info)
        jax_prob = self._jax(arg_prob, info)
        
        # uses the JAX substrate of tensorflow-probability
        def _jax_wrapped_distribution_negative_binomial(x, params, key):
            trials, key, err2 = jax_trials(x, params, key)       
            prob, key, err1 = jax_prob(x, params, key)
            trials = jnp.asarray(trials, dtype=self.REAL)
            prob = jnp.asarray(prob, dtype=self.REAL)
            key, subkey = random.split(key)
            dist = tfp.distributions.NegativeBinomial(total_count=trials, probs=prob)
            sample = dist.sample(seed=subkey).astype(self.INT)
            out_of_bounds = jnp.logical_not(jnp.all(
                (prob >= 0) & (prob <= 1) & (trials > 0)))
            err = err1 | err2 | (out_of_bounds * ERR)
            return sample, key, err
        
        return _jax_wrapped_distribution_negative_binomial    
        
    def _jax_beta(self, expr, info):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_BETA']
        JaxRDDLCompiler._check_num_args(expr, 2)
        
        arg_shape, arg_rate = expr.args
        jax_shape = self._jax(arg_shape, info)
        jax_rate = self._jax(arg_rate, info)
        
        # uses the implicit JAX subroutine
        def _jax_wrapped_distribution_beta(x, params, key):
            shape, key, err1 = jax_shape(x, params, key)
            rate, key, err2 = jax_rate(x, params, key)
            key, subkey = random.split(key)
            sample = random.beta(key=subkey, a=shape, b=rate, dtype=self.REAL)
            out_of_bounds = jnp.logical_not(jnp.all((shape > 0) & (rate > 0)))
            err = err1 | err2 | (out_of_bounds * ERR)
            return sample, key, err
        
        return _jax_wrapped_distribution_beta
    
    def _jax_geometric(self, expr, info):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_GEOMETRIC']
        JaxRDDLCompiler._check_num_args(expr, 1)        
        arg_prob, = expr.args
        
        # if prob is non-fluent, always use the exact operation
        if self.compile_non_fluent_exact and not self.traced.cached_is_fluent(arg_prob):
            geom_op = JaxRDDLCompiler.EXACT_RDDL_TO_JAX_GEOMETRIC
        else:
            geom_op = self.GEOMETRIC_HELPER
        jax_geom, jax_param = self._unwrap(geom_op, expr.id, info)
        
        # recursively compile arguments        
        jax_prob = self._jax(arg_prob, info)
        
        def _jax_wrapped_distribution_geometric(x, params, key):
            prob, key, err = jax_prob(x, params, key)
            key, subkey = random.split(key)
            param = params.get(jax_param, None)
            sample = jax_geom(subkey, prob, param).astype(self.INT)
            out_of_bounds = jnp.logical_not(jnp.all((prob >= 0) & (prob <= 1)))
            err |= (out_of_bounds * ERR)
            return sample, key, err
        
        return _jax_wrapped_distribution_geometric
    
    def _jax_pareto(self, expr, info):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_PARETO']
        JaxRDDLCompiler._check_num_args(expr, 2)
        
        arg_shape, arg_scale = expr.args
        jax_shape = self._jax(arg_shape, info)
        jax_scale = self._jax(arg_scale, info)
        
        # partial reparameterization trick Pareto(s, r) = r * Pareto(s, 1)
        # uses the implicit JAX subroutine for Pareto(s, 1) 
        def _jax_wrapped_distribution_pareto(x, params, key):
            shape, key, err1 = jax_shape(x, params, key)
            scale, key, err2 = jax_scale(x, params, key)
            key, subkey = random.split(key)
            sample = scale * random.pareto(key=subkey, b=shape, dtype=self.REAL)
            out_of_bounds = jnp.logical_not(jnp.all((shape > 0) & (scale > 0)))
            err = err1 | err2 | (out_of_bounds * ERR)
            return sample, key, err
        
        return _jax_wrapped_distribution_pareto
    
    def _jax_student(self, expr, info):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_STUDENT']
        JaxRDDLCompiler._check_num_args(expr, 1)
        
        arg_df, = expr.args
        jax_df = self._jax(arg_df, info)
        
        # uses the implicit JAX subroutine for student(df)
        def _jax_wrapped_distribution_t(x, params, key):
            df, key, err = jax_df(x, params, key)
            key, subkey = random.split(key)
            sample = random.t(
                key=subkey, df=df, shape=jnp.shape(df), dtype=self.REAL)
            out_of_bounds = jnp.logical_not(jnp.all(df > 0))
            err |= (out_of_bounds * ERR)
            return sample, key, err
        
        return _jax_wrapped_distribution_t
    
    def _jax_gumbel(self, expr, info):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_GUMBEL']
        JaxRDDLCompiler._check_num_args(expr, 2)
        
        arg_mean, arg_scale = expr.args
        jax_mean = self._jax(arg_mean, info)
        jax_scale = self._jax(arg_scale, info)
        
        # reparameterization trick Gumbel(m, s) = m + s * Gumbel(0, 1)
        def _jax_wrapped_distribution_gumbel(x, params, key):
            mean, key, err1 = jax_mean(x, params, key)
            scale, key, err2 = jax_scale(x, params, key)
            key, subkey = random.split(key)
            Gumbel01 = random.gumbel(
                key=subkey, shape=jnp.shape(mean), dtype=self.REAL)
            sample = mean + scale * Gumbel01
            out_of_bounds = jnp.logical_not(jnp.all(scale > 0))
            err = err1 | err2 | (out_of_bounds * ERR)
            return sample, key, err
        
        return _jax_wrapped_distribution_gumbel
    
    def _jax_laplace(self, expr, info):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_LAPLACE']
        JaxRDDLCompiler._check_num_args(expr, 2)
        
        arg_mean, arg_scale = expr.args
        jax_mean = self._jax(arg_mean, info)
        jax_scale = self._jax(arg_scale, info)
        
        # reparameterization trick Laplace(m, s) = m + s * Laplace(0, 1)
        def _jax_wrapped_distribution_laplace(x, params, key):
            mean, key, err1 = jax_mean(x, params, key)
            scale, key, err2 = jax_scale(x, params, key)
            key, subkey = random.split(key)
            Laplace01 = random.laplace(
                key=subkey, shape=jnp.shape(mean), dtype=self.REAL)
            sample = mean + scale * Laplace01
            out_of_bounds = jnp.logical_not(jnp.all(scale > 0))
            err = err1 | err2 | (out_of_bounds * ERR)
            return sample, key, err
        
        return _jax_wrapped_distribution_laplace
    
    def _jax_cauchy(self, expr, info):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_CAUCHY']
        JaxRDDLCompiler._check_num_args(expr, 2)
        
        arg_mean, arg_scale = expr.args
        jax_mean = self._jax(arg_mean, info)
        jax_scale = self._jax(arg_scale, info)
        
        # reparameterization trick Cauchy(m, s) = m + s * Cauchy(0, 1)
        def _jax_wrapped_distribution_cauchy(x, params, key):
            mean, key, err1 = jax_mean(x, params, key)
            scale, key, err2 = jax_scale(x, params, key)
            key, subkey = random.split(key)
            Cauchy01 = random.cauchy(
                key=subkey, shape=jnp.shape(mean), dtype=self.REAL)
            sample = mean + scale * Cauchy01
            out_of_bounds = jnp.logical_not(jnp.all(scale > 0))
            err = err1 | err2 | (out_of_bounds * ERR)
            return sample, key, err
        
        return _jax_wrapped_distribution_cauchy
    
    def _jax_gompertz(self, expr, info):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_GOMPERTZ']
        JaxRDDLCompiler._check_num_args(expr, 2)
        
        arg_shape, arg_scale = expr.args
        jax_shape = self._jax(arg_shape, info)
        jax_scale = self._jax(arg_scale, info)
        
        # reparameterization trick Gompertz(s, r) = ln(1 - log(U(0, 1)) / s) / r
        def _jax_wrapped_distribution_gompertz(x, params, key):
            shape, key, err1 = jax_shape(x, params, key)
            scale, key, err2 = jax_scale(x, params, key)
            key, subkey = random.split(key)
            U = random.uniform(key=subkey, shape=jnp.shape(scale), dtype=self.REAL)
            sample = jnp.log(1.0 - jnp.log(U) / shape) / scale
            out_of_bounds = jnp.logical_not(jnp.all((shape > 0) & (scale > 0)))
            err = err1 | err2 | (out_of_bounds * ERR)
            return sample, key, err
        
        return _jax_wrapped_distribution_gompertz
    
    def _jax_chisquare(self, expr, info):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_CHISQUARE']
        JaxRDDLCompiler._check_num_args(expr, 1)
        
        arg_df, = expr.args
        jax_df = self._jax(arg_df, info)
        
        # use the fact that ChiSquare(df) = Gamma(df/2, 2)
        def _jax_wrapped_distribution_chisquare(x, params, key):
            df, key, err1 = jax_df(x, params, key)
            key, subkey = random.split(key)
            shape = df / 2.0
            Gamma = random.gamma(key=subkey, a=shape, dtype=self.REAL)
            sample = 2.0 * Gamma
            out_of_bounds = jnp.logical_not(jnp.all(df > 0))
            err = err1 | (out_of_bounds * ERR)
            return sample, key, err
        
        return _jax_wrapped_distribution_chisquare
    
    def _jax_kumaraswamy(self, expr, info):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_KUMARASWAMY']
        JaxRDDLCompiler._check_num_args(expr, 2)
        
        arg_a, arg_b = expr.args
        jax_a = self._jax(arg_a, info)
        jax_b = self._jax(arg_b, info)
        
        # uses the reparameterization K(a, b) = (1 - (1 - U(0, 1))^{1/b})^{1/a}
        def _jax_wrapped_distribution_kumaraswamy(x, params, key):
            a, key, err1 = jax_a(x, params, key)
            b, key, err2 = jax_b(x, params, key)
            key, subkey = random.split(key)
            U = random.uniform(key=subkey, shape=jnp.shape(a), dtype=self.REAL)            
            sample = jnp.power(1.0 - jnp.power(U, 1.0 / b), 1.0 / a)
            out_of_bounds = jnp.logical_not(jnp.all((a > 0) & (b > 0)))
            err = err1 | err2 | (out_of_bounds * ERR)
            return sample, key, err
        
        return _jax_wrapped_distribution_kumaraswamy
    
    # ===========================================================================
    # random variables with enum support
    # ===========================================================================
    
    def _jax_discrete(self, expr, info, unnorm):
        NORMAL = JaxRDDLCompiler.ERROR_CODES['NORMAL']
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_DISCRETE']
        ordered_args = self.traced.cached_sim_info(expr)
        
        # if all probabilities are non-fluent, then always sample exact
        has_fluent_arg = any(self.traced.cached_is_fluent(arg) 
                             for arg in ordered_args)
        if self.compile_non_fluent_exact and not has_fluent_arg:
            discrete_op = JaxRDDLCompiler.EXACT_RDDL_TO_JAX_DISCRETE
        else:
            discrete_op = self.DISCRETE_HELPER
        jax_discrete, jax_param = self._unwrap(discrete_op, expr.id, info)
        
        # compile probability expressions
        jax_probs = [self._jax(arg, info) for arg in ordered_args]
        
        def _jax_wrapped_distribution_discrete(x, params, key):
            
            # sample case probabilities and normalize as needed
            error = NORMAL
            prob = [None] * len(jax_probs)
            for (i, jax_prob) in enumerate(jax_probs):
                prob[i], key, error_pdf = jax_prob(x, params, key)
                error |= error_pdf
            prob = jnp.stack(prob, axis=-1)
            if unnorm:
                normalizer = jnp.sum(prob, axis=-1, keepdims=True)
                prob = prob / normalizer
            
            # dispatch to sampling subroutine
            key, subkey = random.split(key)
            param = params.get(jax_param, None)
            sample = jax_discrete(subkey, prob, param)
            out_of_bounds = jnp.logical_not(jnp.logical_and(
                jnp.all(prob >= 0),
                jnp.allclose(jnp.sum(prob, axis=-1), 1.0)))
            error |= (out_of_bounds * ERR)
            return sample, key, error
        
        return _jax_wrapped_distribution_discrete
    
    def _jax_discrete_pvar(self, expr, info, unnorm):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_DISCRETE']
        JaxRDDLCompiler._check_num_args(expr, 2)
        _, args = expr.args
        arg, = args
        
        # if probabilities are non-fluent, then always sample exact
        if self.compile_non_fluent_exact and not self.traced.cached_is_fluent(arg):
            discrete_op = JaxRDDLCompiler.EXACT_RDDL_TO_JAX_DISCRETE
        else:
            discrete_op = self.DISCRETE_HELPER
        jax_discrete, jax_param = self._unwrap(discrete_op, expr.id, info)
        
        # compile probability function
        jax_probs = self._jax(arg, info)

        def _jax_wrapped_distribution_discrete_pvar(x, params, key):
            
            # sample probabilities
            prob, key, error = jax_probs(x, params, key)
            if unnorm:
                normalizer = jnp.sum(prob, axis=-1, keepdims=True)
                prob = prob / normalizer
            
            # dispatch to sampling subroutine
            key, subkey = random.split(key)
            param = params.get(jax_param, None)
            sample = jax_discrete(subkey, prob, param)
            out_of_bounds = jnp.logical_not(jnp.logical_and(
                jnp.all(prob >= 0),
                jnp.allclose(jnp.sum(prob, axis=-1), 1.0)))
            error |= (out_of_bounds * ERR)
            return sample, key, error
        
        return _jax_wrapped_distribution_discrete_pvar

    # ===========================================================================
    # random vectors
    # ===========================================================================
    
    def _jax_random_vector(self, expr, info):
        _, name = expr.etype
        if name == 'MultivariateNormal':
            return self._jax_multivariate_normal(expr, info)   
        elif name == 'MultivariateStudent':
            return self._jax_multivariate_student(expr, info)  
        elif name == 'Dirichlet':
            return self._jax_dirichlet(expr, info)
        elif name == 'Multinomial':
            return self._jax_multinomial(expr, info)
        else:
            raise RDDLNotImplementedError(
                f'Distribution {name} is not supported.\n' + 
                print_stack_trace(expr))
    
    def _jax_multivariate_normal(self, expr, info): 
        _, args = expr.args
        mean, cov = args
        jax_mean = self._jax(mean, info)
        jax_cov = self._jax(cov, info)
        index, = self.traced.cached_sim_info(expr)
        
        # reparameterization trick MN(m, LL') = LZ + m, where Z ~ Normal(0, 1)
        def _jax_wrapped_distribution_multivariate_normal(x, params, key):
            
            # sample the mean and covariance
            sample_mean, key, err1 = jax_mean(x, params, key)
            sample_cov, key, err2 = jax_cov(x, params, key)
            
            # sample Normal(0, 1)
            key, subkey = random.split(key)
            Z = random.normal(
                key=subkey,
                shape=jnp.shape(sample_mean) + (1,),
                dtype=self.REAL)       
            
            # compute L s.t. cov = L * L' and reparameterize
            L = jnp.linalg.cholesky(sample_cov)
            sample = jnp.matmul(L, Z)[..., 0] + sample_mean
            sample = jnp.moveaxis(sample, source=-1, destination=index)
            err = err1 | err2
            return sample, key, err
        
        return _jax_wrapped_distribution_multivariate_normal
    
    def _jax_multivariate_student(self, expr, info):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_MULTIVARIATE_STUDENT']
        
        _, args = expr.args
        mean, cov, df = args
        jax_mean = self._jax(mean, info)
        jax_cov = self._jax(cov, info)
        jax_df = self._jax(df, info)
        index, = self.traced.cached_sim_info(expr)
        
        # reparameterization trick MN(m, LL') = LZ + m, where Z ~ StudentT(0, 1)
        def _jax_wrapped_distribution_multivariate_student(x, params, key):
            
            # sample the mean and covariance and degrees of freedom
            sample_mean, key, err1 = jax_mean(x, params, key)
            sample_cov, key, err2 = jax_cov(x, params, key)
            sample_df, key, err3 = jax_df(x, params, key)
            out_of_bounds = jnp.logical_not(jnp.all(sample_df > 0))
            
            # sample StudentT(0, 1, df) -- broadcast df to same shape as cov
            sample_df = sample_df[..., jnp.newaxis, jnp.newaxis]
            sample_df = jnp.broadcast_to(sample_df, shape=sample_mean.shape + (1,))
            key, subkey = random.split(key)
            Z = random.t(
                key=subkey, 
                df=sample_df, 
                shape=jnp.shape(sample_df),
                dtype=self.REAL)   
            
            # compute L s.t. cov = L * L' and reparameterize
            L = jnp.linalg.cholesky(sample_cov)
            sample = jnp.matmul(L, Z)[..., 0] + sample_mean
            sample = jnp.moveaxis(sample, source=-1, destination=index)
            error = err1 | err2 | err3 | (out_of_bounds * ERR)
            return sample, key, error
        
        return _jax_wrapped_distribution_multivariate_student
    
    def _jax_dirichlet(self, expr, info):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_DIRICHLET']
        
        _, args = expr.args
        alpha, = args
        jax_alpha = self._jax(alpha, info)
        index, = self.traced.cached_sim_info(expr)
        
        # sample Gamma(alpha_i, 1) and normalize across i
        def _jax_wrapped_distribution_dirichlet(x, params, key):
            alpha, key, error = jax_alpha(x, params, key)
            out_of_bounds = jnp.logical_not(jnp.all(alpha > 0))
            error |= (out_of_bounds * ERR)
            key, subkey = random.split(key)
            Gamma = random.gamma(key=subkey, a=alpha, dtype=self.REAL)
            sample = Gamma / jnp.sum(Gamma, axis=-1, keepdims=True)
            sample = jnp.moveaxis(sample, source=-1, destination=index)
            return sample, key, error
        
        return _jax_wrapped_distribution_dirichlet
    
    def _jax_multinomial(self, expr, info):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_MULTINOMIAL']
        
        _, args = expr.args
        trials, prob = args
        jax_trials = self._jax(trials, info)
        jax_prob = self._jax(prob, info)
        index, = self.traced.cached_sim_info(expr)
        
        def _jax_wrapped_distribution_multinomial(x, params, key):
            trials, key, err1 = jax_trials(x, params, key)
            prob, key, err2 = jax_prob(x, params, key)
            trials = jnp.asarray(trials, dtype=self.REAL)
            prob = jnp.asarray(prob, dtype=self.REAL)
            key, subkey = random.split(key)
            dist = tfp.distributions.Multinomial(total_count=trials, probs=prob)
            sample = dist.sample(seed=subkey).astype(self.INT)
            sample = jnp.moveaxis(sample, source=-1, destination=index)
            out_of_bounds = jnp.logical_not(jnp.all(
                (prob >= 0)
                & jnp.allclose(jnp.sum(prob, axis=-1), 1.0)
                & (trials >= 0)))
            error = err1 | err2 | (out_of_bounds * ERR)
            return sample, key, error            
        
        return _jax_wrapped_distribution_multinomial
    
    # ===========================================================================
    # matrix algebra
    # ===========================================================================
    
    def _jax_matrix(self, expr, info):
        _, op = expr.etype
        if op == 'det':
            return self._jax_matrix_det(expr, info)
        elif op == 'inverse':
            return self._jax_matrix_inv(expr, info, pseudo=False)
        elif op == 'pinverse':
            return self._jax_matrix_inv(expr, info, pseudo=True)
        elif op == 'cholesky':
            return self._jax_matrix_cholesky(expr, info)
        else:
            raise RDDLNotImplementedError(
                f'Matrix operation {op} is not supported.\n' + 
                print_stack_trace(expr))
    
    def _jax_matrix_det(self, expr, info):
        * _, arg = expr.args
        jax_arg = self._jax(arg, info)
        
        def _jax_wrapped_matrix_operation_det(x, params, key):
            sample_arg, key, error = jax_arg(x, params, key)
            sample = jnp.linalg.det(sample_arg)
            return sample, key, error
        
        return _jax_wrapped_matrix_operation_det
    
    def _jax_matrix_inv(self, expr, info, pseudo):
        _, arg = expr.args
        jax_arg = self._jax(arg, info)
        indices = self.traced.cached_sim_info(expr)
        op = jnp.linalg.pinv if pseudo else jnp.linalg.inv
        
        def _jax_wrapped_matrix_operation_inv(x, params, key):
            sample_arg, key, error = jax_arg(x, params, key)
            sample = op(sample_arg)
            sample = jnp.moveaxis(sample, source=(-2, -1), destination=indices)
            return sample, key, error
        
        return _jax_wrapped_matrix_operation_inv
    
    def _jax_matrix_cholesky(self, expr, info):
        _, arg = expr.args
        jax_arg = self._jax(arg, info)
        indices = self.traced.cached_sim_info(expr)
        op = jnp.linalg.cholesky
        
        def _jax_wrapped_matrix_operation_cholesky(x, params, key):
            sample_arg, key, error = jax_arg(x, params, key)
            sample = op(sample_arg)
            sample = jnp.moveaxis(sample, source=(-2, -1), destination=indices)
            return sample, key, error
        
        return _jax_wrapped_matrix_operation_cholesky
            
