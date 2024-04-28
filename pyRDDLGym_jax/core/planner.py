from ast import literal_eval
from collections import deque
import configparser
import haiku as hk
import jax
import jax.numpy as jnp
import jax.random as random
import jax.nn.initializers as initializers
import numpy as np
import optax
import os
import sys
import termcolor
import time
from tqdm import tqdm
from typing import Callable, Dict, Generator, Optional, Set, Sequence, Tuple

from pyRDDLGym.core.compiler.model import RDDLPlanningModel, RDDLLiftedModel
from pyRDDLGym.core.debug.logger import Logger
from pyRDDLGym.core.debug.exception import (
    raise_warning,
    RDDLNotImplementedError,
    RDDLUndefinedVariableError,
    RDDLTypeError
)
from pyRDDLGym.core.policy import BaseAgent

from pyRDDLGym_jax.core.compiler import JaxRDDLCompiler
from pyRDDLGym_jax.core import logic
from pyRDDLGym_jax.core.logic import FuzzyLogic


# ***********************************************************************
# CONFIG FILE MANAGEMENT
# 
# - read config files from file path
# - extract experiment settings
# - instantiate planner
#
# ***********************************************************************
def _parse_config_file(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f'File {path} does not exist.')
    config = configparser.RawConfigParser()
    config.optionxform = str 
    config.read(path)
    args = {k: literal_eval(v) 
            for section in config.sections()
            for (k, v) in config.items(section)}
    return config, args


def _parse_config_string(value: str):
    config = configparser.RawConfigParser()
    config.optionxform = str 
    config.read_string(value)
    args = {k: literal_eval(v) 
            for section in config.sections()
            for (k, v) in config.items(section)}
    return config, args


def _load_config(config, args):
    model_args = {k: args[k] for (k, _) in config.items('Model')}
    planner_args = {k: args[k] for (k, _) in config.items('Optimizer')}
    train_args = {k: args[k] for (k, _) in config.items('Training')}    
    
    train_args['key'] = jax.random.PRNGKey(train_args['key'])
    
    # read the model settings
    logic_name = model_args.get('logic', 'FuzzyLogic')
    logic_kwargs = model_args.get('logic_kwargs', {})
    tnorm_name = model_args.get('tnorm', 'ProductTNorm')
    tnorm_kwargs = model_args.get('tnorm_kwargs', {})
    comp_name = model_args.get('complement', 'StandardComplement')
    comp_kwargs = model_args.get('complement_kwargs', {})
    logic_kwargs['tnorm'] = getattr(logic, tnorm_name)(**tnorm_kwargs)
    logic_kwargs['complement'] = getattr(logic, comp_name)(**comp_kwargs)
    
    # read the optimizer settings
    plan_method = planner_args.pop('method')
    plan_kwargs = planner_args.pop('method_kwargs', {})  
    
    if 'initializer' in plan_kwargs:  # weight initialization
        init_name = plan_kwargs['initializer']
        init_class = getattr(initializers, init_name)
        init_kwargs = plan_kwargs.pop('initializer_kwargs', {})
        try: 
            plan_kwargs['initializer'] = init_class(**init_kwargs)
        except:
            raise_warning(f'ignoring arguments for initializer <{init_name}>')
            plan_kwargs['initializer'] = init_class
               
    if 'activation' in plan_kwargs:  # activation function
        plan_kwargs['activation'] = getattr(jax.nn, plan_kwargs['activation'])
    
    planner_args['logic'] = getattr(logic, logic_name)(**logic_kwargs)
    planner_args['plan'] = getattr(sys.modules[__name__], plan_method)(**plan_kwargs)
    if 'optimizer' in planner_args:
        planner_args['optimizer'] = getattr(optax, planner_args['optimizer'])
    
    return planner_args, plan_kwargs, train_args


def load_config(path: str) -> Tuple[Dict[str, object], ...]:
    '''Loads a config file at the specified file path.'''
    config, args = _parse_config_file(path)
    return _load_config(config, args)


def load_config_from_string(value: str) -> Tuple[Dict[str, object], ...]:
    '''Loads config file contents specified explicitly as a string value.'''
    config, args = _parse_config_string(value)
    return _load_config(config, args)

    
# ***********************************************************************
# MODEL RELAXATIONS
# 
# - replace discrete ops in state dynamics/reward with differentiable ones
#
# ***********************************************************************
class JaxRDDLCompilerWithGrad(JaxRDDLCompiler):
    '''Compiles a RDDL AST representation to an equivalent JAX representation. 
    Unlike its parent class, this class treats all fluents as real-valued, and
    replaces all mathematical operations by equivalent ones with a well defined 
    (e.g. non-zero) gradient where appropriate. 
    '''
    
    def __init__(self, *args,
                 logic: FuzzyLogic=FuzzyLogic(),
                 cpfs_without_grad: Optional[set]=None,
                 **kwargs) -> None:
        '''Creates a new RDDL to Jax compiler, where operations that are not
        differentiable are converted to approximate forms that have defined 
        gradients.
        
        :param *args: arguments to pass to base compiler
        :param logic: Fuzzy logic object that specifies how exact operations
        are converted to their approximate forms: this class may be subclassed
        to customize these operations
        :param cpfs_without_grad: which CPFs do not have gradients (use straight
        through gradient trick)
        :param *kwargs: keyword arguments to pass to base compiler
        '''
        super(JaxRDDLCompilerWithGrad, self).__init__(*args, **kwargs)
        self.logic = logic
        if cpfs_without_grad is None:
            cpfs_without_grad = set()
        self.cpfs_without_grad = cpfs_without_grad
        
        # actions and CPFs must be continuous
        raise_warning('Initial values of pvariables will be cast to real.')   
        for (var, values) in self.init_values.items():
            self.init_values[var] = np.asarray(values, dtype=self.REAL) 
        
        # overwrite basic operations with fuzzy ones
        self.RELATIONAL_OPS = {
            '>=': logic.greaterEqual(),
            '<=': logic.lessEqual(),
            '<': logic.less(),
            '>': logic.greater(),
            '==': logic.equal(),
            '~=': logic.notEqual()
        }
        self.LOGICAL_NOT = logic.Not()
        self.LOGICAL_OPS = {
            '^': logic.And(),
            '&': logic.And(),
            '|': logic.Or(),
            '~': logic.xor(),
            '=>': logic.implies(),
            '<=>': logic.equiv()
        }
        self.AGGREGATION_OPS['forall'] = logic.forall()
        self.AGGREGATION_OPS['exists'] = logic.exists()
        self.AGGREGATION_OPS['argmin'] = logic.argmin()
        self.AGGREGATION_OPS['argmax'] = logic.argmax()
        self.KNOWN_UNARY['sgn'] = logic.signum()
        self.KNOWN_UNARY['floor'] = logic.floor()   
        self.KNOWN_UNARY['ceil'] = logic.ceil()   
        self.KNOWN_UNARY['round'] = logic.round()
        self.KNOWN_UNARY['sqrt'] = logic.sqrt()
        self.KNOWN_BINARY['div'] = logic.floorDiv()
        self.KNOWN_BINARY['mod'] = logic.mod()
        self.KNOWN_BINARY['fmod'] = logic.mod()
    
    def _jax_stop_grad(self, jax_expr):
        
        def _jax_wrapped_stop_grad(x, params, key):
            sample, key, error = jax_expr(x, params, key)
            sample = jax.lax.stop_gradient(sample)
            return sample, key, error
        
        return _jax_wrapped_stop_grad
        
    def _compile_cpfs(self, info):
        raise_warning('CPFs outputs will be cast to real.')      
        jax_cpfs = {}
        for (_, cpfs) in self.levels.items():
            for cpf in cpfs:
                _, expr = self.rddl.cpfs[cpf]
                jax_cpfs[cpf] = self._jax(expr, info, dtype=self.REAL)
                if cpf in self.cpfs_without_grad:
                    raise_warning(f'CPF <{cpf}> stops gradient.')      
                    jax_cpfs[cpf] = self._jax_stop_grad(jax_cpfs[cpf])
        return jax_cpfs
    
    def _jax_if_helper(self):
        return self.logic.If()
    
    def _jax_switch_helper(self):
        return self.logic.Switch()
        
    def _jax_kron(self, expr, info):
        if self.logic.verbose:
            raise_warning('KronDelta will be ignored.')            
                       
        arg, = expr.args
        arg = self._jax(arg, info)
        return arg
    
    def _jax_bernoulli_helper(self):
        return self.logic.bernoulli()
    
    def _jax_discrete_helper(self):
        jax_discrete, jax_param = self.logic.discrete()

        def _jax_wrapped_discrete_calc_approx(key, prob, params):
            sample = jax_discrete(key, prob, params)
            out_of_bounds = jnp.logical_not(jnp.logical_and(
                jnp.all(prob >= 0),
                jnp.allclose(jnp.sum(prob, axis=-1), 1.0)))
            return sample, out_of_bounds
        
        return _jax_wrapped_discrete_calc_approx, jax_param


# ***********************************************************************
# ALL VERSIONS OF JAX PLANS
# 
# - straight line plan
# - deep reactive policy
#
# ***********************************************************************
class JaxPlan:
    '''Base class for all JAX policy representations.'''
    
    def __init__(self) -> None:
        self._initializer = None
        self._train_policy = None
        self._test_policy = None
        self._projection = None
    
    def summarize_hyperparameters(self):
        pass
        
    def compile(self, compiled: JaxRDDLCompilerWithGrad,
                _bounds: Dict,
                horizon: int) -> None:
        raise NotImplementedError
    
    def guess_next_epoch(self, params: Dict) -> Dict:
        raise NotImplementedError
    
    @property
    def initializer(self):
        return self._initializer

    @initializer.setter
    def initializer(self, value):
        self._initializer = value
    
    @property
    def train_policy(self):
        return self._train_policy

    @train_policy.setter
    def train_policy(self, value):
        self._train_policy = value
        
    @property
    def test_policy(self):
        return self._test_policy

    @test_policy.setter
    def test_policy(self, value):
        self._test_policy = value
         
    @property
    def projection(self):
        return self._projection

    @projection.setter
    def projection(self, value):
        self._projection = value
    
    def _calculate_action_info(self, compiled: JaxRDDLCompilerWithGrad,
                               user_bounds: Dict[str, object], horizon: int):
        shapes, bounds, bounds_safe, cond_lists = {}, {}, {}, {}
        for (name, prange) in compiled.rddl.variable_ranges.items():
            if compiled.rddl.variable_types[name] != 'action-fluent':
                continue
            
            # check invalid type
            if prange not in compiled.JAX_TYPES:
                raise RDDLTypeError(
                    f'Invalid range <{prange}. of action-fluent <{name}>, '
                    f'must be one of {set(compiled.JAX_TYPES.keys())}.')
                
            # clip boolean to (0, 1), otherwise use the RDDL action bounds
            # or the user defined action bounds if provided
            shapes[name] = (horizon,) + np.shape(compiled.init_values[name])
            if prange == 'bool':
                lower, upper = None, None
            else:
                lower, upper = compiled.constraints.bounds[name]
                lower, upper = user_bounds.get(name, (lower, upper))
                lower = np.asarray(lower, dtype=np.float32)
                upper = np.asarray(upper, dtype=np.float32)
                lower_finite = np.isfinite(lower)
                upper_finite = np.isfinite(upper)
                bounds_safe[name] = (np.where(lower_finite, lower, 0.0),
                                     np.where(upper_finite, upper, 0.0))
                cond_lists[name] = [lower_finite & upper_finite,
                                    lower_finite & ~upper_finite,
                                    ~lower_finite & upper_finite,
                                    ~lower_finite & ~upper_finite]
            bounds[name] = (lower, upper)
            raise_warning(f'Bounds of action fluent <{name}> set to {bounds[name]}.')
        return shapes, bounds, bounds_safe, cond_lists
    
    def _count_bool_actions(self, rddl: RDDLLiftedModel):
        constraint = rddl.max_allowed_actions
        num_bool_actions = sum(np.size(values)
                               for (var, values) in rddl.action_fluents.items()
                               if rddl.variable_ranges[var] == 'bool')
        return num_bool_actions, constraint

    
class JaxStraightLinePlan(JaxPlan):
    '''A straight line plan implementation in JAX'''
    
    def __init__(self, initializer: initializers.Initializer=initializers.normal(),
                 wrap_sigmoid: bool=True,
                 min_action_prob: float=1e-5,
                 wrap_non_bool: bool=False,
                 wrap_softmax: bool=False,
                 use_new_projection: bool=False,
                 max_constraint_iter: int=100) -> None:
        '''Creates a new straight line plan in JAX.
        
        :param initializer: a Jax Initializer for setting the initial actions
        :param wrap_sigmoid: wrap bool action parameters with sigmoid 
        (uses gradient clipping instead of sigmoid if None; this flag is ignored
        if wrap_softmax = True)
        :param min_action_prob: minimum value a soft boolean action can take
        (maximum is 1 - min_action_prob); required positive if wrap_sigmoid = True
        :param wrap_non_bool: whether to wrap real or int action fluent parameters
        with non-linearity (e.g. sigmoid or ELU) to satisfy box constraints
        :param wrap_softmax: whether to use softmax activation approach 
        (note, this is limited to max-nondef-actions = 1) instead of projected
        gradient to satisfy action constraints 
        :param use_new_projection: whether to use non-iterative (e.g. sort-based)
        projection method, or modified SOGBOFA projection method to satisfy
        action concurrency constraint
        :param max_constraint_iter: max iterations of projected 
        gradient for ensuring actions satisfy constraints, only required if 
        use_new_projection = True
        '''
        super(JaxStraightLinePlan, self).__init__()
        self._initializer_base = initializer
        self._initializer = initializer
        self._wrap_sigmoid = wrap_sigmoid
        self._min_action_prob = min_action_prob
        self._wrap_non_bool = wrap_non_bool
        self._wrap_softmax = wrap_softmax
        self._use_new_projection = use_new_projection
        self._max_constraint_iter = max_constraint_iter
        
    def summarize_hyperparameters(self):
        print(f'policy hyper-parameters:\n'
              f'    initializer          ={type(self._initializer_base).__name__}\n'
              f'constraint-sat strategy (simple):\n'
              f'    wrap_sigmoid         ={self._wrap_sigmoid}\n'
              f'    wrap_sigmoid_min_prob={self._min_action_prob}\n'
              f'    wrap_non_bool        ={self._wrap_non_bool}\n'
              f'constraint-sat strategy (complex):\n'
              f'    wrap_softmax         ={self._wrap_softmax}\n'
              f'    use_new_projection   ={self._use_new_projection}')
    
    def compile(self, compiled: JaxRDDLCompilerWithGrad,
                _bounds: Dict, horizon: int) -> None:
        rddl = compiled.rddl
        
        # calculate the correct action box bounds
        shapes, bounds, bounds_safe, cond_lists = self._calculate_action_info(
            compiled, _bounds, horizon)
        self.bounds = bounds
        
        # action concurrency check
        bool_action_count, allowed_actions = self._count_bool_actions(rddl)
        use_constraint_satisfaction = allowed_actions < bool_action_count        
        if use_constraint_satisfaction: 
            raise_warning(f'Using projected gradient trick to satisfy '
                          f'max_nondef_actions: total boolean actions '
                          f'{bool_action_count} > max_nondef_actions '
                          f'{allowed_actions}.')
            
        noop = {var: (values[0] if isinstance(values, list) else values)
                for (var, values) in rddl.action_fluents.items()}
        bool_key = 'bool__'
        
        # ***********************************************************************
        # STRAIGHT-LINE PLAN
        #
        # ***********************************************************************
        
        # define the mapping between trainable parameter and action
        wrap_sigmoid = self._wrap_sigmoid
        bool_threshold = 0.0 if wrap_sigmoid else 0.5
        
        def _jax_bool_param_to_action(var, param, hyperparams):
            if wrap_sigmoid:
                weight = hyperparams[var]
                return jax.nn.sigmoid(weight * param)
            else:
                return param 
        
        def _jax_bool_action_to_param(var, action, hyperparams):
            if wrap_sigmoid:
                weight = hyperparams[var]
                return (-1.0 / weight) * jnp.log1p(1.0 / action - 2.0)
            else:
                return action
            
        wrap_non_bool = self._wrap_non_bool
        
        def _jax_non_bool_param_to_action(var, param, hyperparams):
            if wrap_non_bool:
                lower, upper = bounds_safe[var]
                action = jnp.select(
                    condlist=cond_lists[var],
                    choicelist=[
                        lower + (upper - lower) * jax.nn.sigmoid(param),
                        lower + (jax.nn.elu(param) + 1.0),
                        upper - (jax.nn.elu(-param) + 1.0),
                        param
                    ]
                )
            else:
                action = param
            return action
        
        # handle box constraints    
        min_action = self._min_action_prob
        max_action = 1.0 - min_action
        
        def _jax_project_bool_to_box(var, param, hyperparams):
            lower = _jax_bool_action_to_param(var, min_action, hyperparams)
            upper = _jax_bool_action_to_param(var, max_action, hyperparams)
            valid_param = jnp.clip(param, lower, upper)
            return valid_param
        
        ranges = rddl.variable_ranges
        
        def _jax_wrapped_slp_project_to_box(params, hyperparams):
            new_params = {}
            for (var, param) in params.items():
                if var == bool_key:
                    new_params[var] = param
                elif ranges[var] == 'bool':
                    new_params[var] = _jax_project_bool_to_box(var, param, hyperparams)
                elif wrap_non_bool:
                    new_params[var] = param
                else:
                    new_params[var] = jnp.clip(param, *bounds[var])
            return new_params, True
        
        # convert softmax action back to action dict
        action_sizes = {var: np.prod(shape[1:], dtype=int) 
                        for (var, shape) in shapes.items()
                        if ranges[var] == 'bool'}
        
        def _jax_unstack_bool_from_softmax(output):
            actions = {}
            start = 0
            for (name, size) in action_sizes.items():
                action = output[..., start:start + size]
                action = jnp.reshape(action, newshape=shapes[name][1:])
                if noop[name]:
                    action = 1.0 - action
                actions[name] = action
                start += size
            return actions
                
        # train plan prediction (TODO: implement one-hot for integer actions)        
        def _jax_wrapped_slp_predict_train(key, params, hyperparams, step, subs):
            actions = {}
            for (var, param) in params.items():
                action = jnp.asarray(param[step, ...], dtype=compiled.REAL)
                if var == bool_key:
                    output = jax.nn.softmax(action)
                    bool_actions = _jax_unstack_bool_from_softmax(output)
                    actions.update(bool_actions)
                elif ranges[var] == 'bool':
                    actions[var] = _jax_bool_param_to_action(var, action, hyperparams)
                else:
                    actions[var] = _jax_non_bool_param_to_action(var, action, hyperparams)
            return actions
        
        # test plan prediction
        def _jax_wrapped_slp_predict_test(key, params, hyperparams, step, subs):
            actions = {}
            for (var, param) in params.items():
                action = jnp.asarray(param[step, ...])
                if var == bool_key:
                    output = jax.nn.softmax(action)
                    bool_actions = _jax_unstack_bool_from_softmax(output)
                    for (bool_var, bool_action) in bool_actions.items():
                        actions[bool_var] = bool_action > 0.5
                elif ranges[var] == 'bool':
                    actions[var] = action > bool_threshold
                else:
                    action = _jax_non_bool_param_to_action(var, action, hyperparams)
                    action = jnp.clip(action, *bounds[var])
                    if ranges[var] == 'int':
                        action = jnp.round(action).astype(compiled.INT)
                    actions[var] = action
            return actions
        
        self.train_policy = _jax_wrapped_slp_predict_train
        self.test_policy = _jax_wrapped_slp_predict_test
        
        # ***********************************************************************
        # ACTION CONSTRAINT SATISFACTION
        #
        # ***********************************************************************
        
        # use a softmax output activation
        if use_constraint_satisfaction and self._wrap_softmax:
            
            # only allow one action non-noop for now
            if 1 < allowed_actions < bool_action_count:
                raise RDDLNotImplementedError(
                    f'Straight-line plans with wrap_softmax currently '
                    f'do not support max-nondef-actions = {allowed_actions} > 1.')
                
            # potentially apply projection but to non-bool actions only
            self.projection = _jax_wrapped_slp_project_to_box
            
        # use new gradient projection method...
        elif use_constraint_satisfaction and self._use_new_projection:
            
            # shift the boolean actions uniformly, clipping at the min/max values
            # the amount to move is such that only top allowed_actions actions
            # are still active (e.g. not equal to noop) after the shift
            def _jax_wrapped_sorting_project(params, hyperparams):
                
                # find the amount to shift action parameters
                # if noop is True pretend it is False and reflect the parameter
                scores = []
                for (var, param) in params.items():
                    if ranges[var] == 'bool':
                        param_flat = jnp.ravel(param)
                        if noop[var]:
                            param_flat = (-param_flat) if wrap_sigmoid else 1.0 - param_flat
                        scores.append(param_flat)
                scores = jnp.concatenate(scores)
                descending = jnp.sort(scores)[::-1]
                kplus1st_greatest = descending[allowed_actions]
                surplus = jnp.maximum(kplus1st_greatest - bool_threshold, 0.0)
                    
                # perform the shift
                new_params = {}
                for (var, param) in params.items():
                    if ranges[var] == 'bool':
                        new_param = param + (surplus if noop[var] else -surplus)
                        new_param = _jax_project_bool_to_box(var, new_param, hyperparams)
                    else:
                        new_param = param
                    new_params[var] = new_param
                return new_params, True
                
            # clip actions to valid bounds and satisfy constraint on max actions
            def _jax_wrapped_slp_project_to_max_constraint(params, hyperparams):
                params, _ = _jax_wrapped_slp_project_to_box(params, hyperparams)
                project_over_horizon = jax.vmap(
                    _jax_wrapped_sorting_project, in_axes=(0, None)
                )(params, hyperparams)
                return project_over_horizon
            
            self.projection = _jax_wrapped_slp_project_to_max_constraint
        
        # use SOGBOFA projection method...
        elif use_constraint_satisfaction and not self._use_new_projection:
            
            # calculate the surplus of actions above max-nondef-actions
            def _jax_wrapped_sogbofa_surplus(params, hyperparams):
                sum_action, count = 0.0, 0
                for (var, param) in params.items():
                    if ranges[var] == 'bool':
                        action = _jax_bool_param_to_action(var, param, hyperparams)                        
                        if noop[var]:
                            sum_action += jnp.size(action) - jnp.sum(action)
                            count += jnp.sum(action < 1)
                        else:
                            sum_action += jnp.sum(action)
                            count += jnp.sum(action > 0)
                surplus = jnp.maximum(sum_action - allowed_actions, 0.0)
                count = jnp.maximum(count, 1)
                return surplus / count
                
            # return whether the surplus is positive or reached compute limit
            max_constraint_iter = self._max_constraint_iter
        
            def _jax_wrapped_sogbofa_continue(values):
                it, _, _, surplus = values
                return jnp.logical_and(it < max_constraint_iter, surplus > 0)
                
            # reduce all bool action values by the surplus clipping at minimum
            # for no-op = True, do the opposite, i.e. increase all
            # bool action values by surplus clipping at maximum
            def _jax_wrapped_sogbofa_subtract_surplus(values):
                it, params, hyperparams, surplus = values
                new_params = {}
                for (var, param) in params.items():
                    if ranges[var] == 'bool':
                        action = _jax_bool_param_to_action(var, param, hyperparams)
                        new_action = action + (surplus if noop[var] else -surplus)
                        new_action = jnp.clip(new_action, min_action, max_action)
                        new_param = _jax_bool_action_to_param(var, new_action, hyperparams)
                    else:
                        new_param = param
                    new_params[var] = new_param
                new_surplus = _jax_wrapped_sogbofa_surplus(new_params, hyperparams)
                new_it = it + 1
                return new_it, new_params, hyperparams, new_surplus
                
            # apply the surplus to the actions until it becomes zero
            def _jax_wrapped_sogbofa_project(params, hyperparams):
                surplus = _jax_wrapped_sogbofa_surplus(params, hyperparams)
                _, params, _, surplus = jax.lax.while_loop(
                    cond_fun=_jax_wrapped_sogbofa_continue,
                    body_fun=_jax_wrapped_sogbofa_subtract_surplus,
                    init_val=(0, params, hyperparams, surplus)
                )
                converged = jnp.logical_not(surplus > 0)
                return params, converged
                
            # clip actions to valid bounds and satisfy constraint on max actions
            def _jax_wrapped_slp_project_to_max_constraint(params, hyperparams):
                params, _ = _jax_wrapped_slp_project_to_box(params, hyperparams)
                project_over_horizon = jax.vmap(
                    _jax_wrapped_sogbofa_project, in_axes=(0, None)
                )(params, hyperparams)
                return project_over_horizon
            
            self.projection = _jax_wrapped_slp_project_to_max_constraint
        
        # just project to box constraints
        else: 
            self.projection = _jax_wrapped_slp_project_to_box
            
        # ***********************************************************************
        # PLAN INITIALIZATION
        #
        # ***********************************************************************
        
        init = self._initializer
        stack_bool_params = use_constraint_satisfaction and self._wrap_softmax
        
        def _jax_wrapped_slp_init(key, hyperparams, subs):
            params = {}
            for (var, shape) in shapes.items():
                if ranges[var] != 'bool' or not stack_bool_params: 
                    key, subkey = random.split(key)
                    param = init(subkey, shape, dtype=compiled.REAL)
                    if ranges[var] == 'bool':
                        param += bool_threshold
                    params[var] = param
            if stack_bool_params:
                key, subkey = random.split(key)
                bool_shape = (horizon, bool_action_count)
                bool_param = init(subkey, bool_shape, dtype=compiled.REAL)
                params[bool_key] = bool_param
            params, _ = _jax_wrapped_slp_project_to_box(params, hyperparams)
            return params
        
        self.initializer = _jax_wrapped_slp_init
    
    @staticmethod
    @jax.jit
    def _guess_next_epoch(param):
        # "progress" the plan one step forward and set last action to second-last
        return jnp.append(param[1:, ...], param[-1:, ...], axis=0)

    def guess_next_epoch(self, params: Dict) -> Dict:
        next_fn = JaxStraightLinePlan._guess_next_epoch
        return jax.tree_map(next_fn, params)


class JaxDeepReactivePolicy(JaxPlan):
    '''A deep reactive policy network implementation in JAX.'''
    
    def __init__(self, topology: Sequence[int],
                 activation: Callable=jax.nn.relu,
                 initializer: hk.initializers.Initializer=hk.initializers.VarianceScaling(scale=2.0),
                 normalize: bool=True,
                 wrap_non_bool: bool=False) -> None:
        '''Creates a new deep reactive policy in JAX.
        
        :param neurons: sequence consisting of the number of neurons in each
        layer of the policy
        :param activation: function to apply after each layer of the policy
        :param initializer: weight initialization
        :param normalize: whether to apply layer norm to the inputs
        :param wrap_non_bool: whether to wrap real or int action fluent parameters
        with non-linearity (e.g. sigmoid or ELU) to satisfy box constraints
        '''
        super(JaxDeepReactivePolicy, self).__init__()
        self._topology = topology
        self._activations = [activation for _ in topology]
        self._initializer_base = initializer
        self._initializer = initializer
        self._normalize = normalize
        self._wrap_non_bool = wrap_non_bool
            
    def summarize_hyperparameters(self):
        print(f'policy hyper-parameters:\n'
              f'    topology        ={self._topology}\n'
              f'    activation_fn   ={self._activations[0].__name__}\n'
              f'    initializer     ={type(self._initializer_base).__name__}\n'
              f'    apply_layer_norm={self._normalize}\n'
              f'    wrap_non_bool   ={self._wrap_non_bool}')
    
    def compile(self, compiled: JaxRDDLCompilerWithGrad,
                _bounds: Dict, horizon: int) -> None:
        rddl = compiled.rddl
        
        # calculate the correct action box bounds
        shapes, bounds, bounds_safe, cond_lists = self._calculate_action_info(
            compiled, _bounds, horizon)
        shapes = {var: value[1:] for (var, value) in shapes.items()}
        self.bounds = bounds
        
        # action concurrency check - only allow one action non-noop for now
        bool_action_count, allowed_actions = self._count_bool_actions(rddl)
        if 1 < allowed_actions < bool_action_count:
            raise RDDLNotImplementedError(
                f'Deep reactive policies currently do not support '
                f'max-nondef-actions = {allowed_actions} > 1.')
        use_constraint_satisfaction = allowed_actions < bool_action_count
            
        noop = {var: (values[0] if isinstance(values, list) else values)
                for (var, values) in rddl.action_fluents.items()}                   
        bool_key = 'bool__'
        
        # ***********************************************************************
        # POLICY NETWORK PREDICTION
        #
        # ***********************************************************************
                   
        ranges = rddl.variable_ranges
        normalize = self._normalize
        wrap_non_bool = self._wrap_non_bool
        init = self._initializer
        layers = list(enumerate(zip(self._topology, self._activations)))
        layer_sizes = {var: np.prod(shape, dtype=int) 
                       for (var, shape) in shapes.items()}
        layer_names = {var: f'output_{var}'.replace('-', '_') for var in shapes}
        
        # predict actions from the policy network for current state
        def _jax_wrapped_policy_network_predict(state):
            
            # apply layer norm
            if normalize:
                normalizer = hk.LayerNorm(
                    axis=-1, param_axis=-1,
                    create_offset=True, create_scale=True,
                    name='input_norm')
                state = normalizer(state)
            
            # feed state vector through hidden layers
            hidden = state
            for (i, (num_neuron, activation)) in layers:
                linear = hk.Linear(num_neuron, name=f'hidden_{i}', w_init=init)
                hidden = activation(linear(hidden))
            
            # each output is a linear layer reshaped to original lifted shape
            actions = {}
            for (var, size) in layer_sizes.items():
                linear = hk.Linear(size, name=layer_names[var], w_init=init)
                reshape = hk.Reshape(output_shape=shapes[var], preserve_dims=-1,
                                     name=f'reshape_{layer_names[var]}')
                output = reshape(linear(hidden))
                if not shapes[var]:
                    output = jnp.squeeze(output)
                
                # project action output to valid box constraints 
                if ranges[var] == 'bool':
                    if not use_constraint_satisfaction:
                        actions[var] = jax.nn.sigmoid(output)
                else:
                    if wrap_non_bool:
                        lower, upper = bounds_safe[var]
                        action = jnp.select(
                            condlist=cond_lists[var],
                            choicelist=[
                                lower + (upper - lower) * jax.nn.sigmoid(output),
                                lower + (jax.nn.elu(output) + 1.0),
                                upper - (jax.nn.elu(-output) + 1.0),
                                output
                            ]
                        )
                    else:
                        action = output
                    actions[var] = action
            
            # for constraint satisfaction wrap bool actions with softmax
            if use_constraint_satisfaction:
                linear = hk.Linear(
                    bool_action_count, name='output_bool', w_init=init)
                output = jax.nn.softmax(linear(hidden))
                actions[bool_key] = output
             
            return actions
        
        predict_fn = hk.transform(_jax_wrapped_policy_network_predict)
        predict_fn = hk.without_apply_rng(predict_fn)            
        
        # convert softmax action back to action dict
        def _jax_unstack_bool_from_softmax(output):
            actions = {}
            start = 0
            for (name, size) in layer_sizes.items():
                if ranges[name] == 'bool':
                    action = output[..., start:start + size]
                    action = jnp.reshape(action, newshape=shapes[name])
                    if noop[name]:
                        action = 1.0 - action
                    actions[name] = action
                    start += size
            return actions
        
        if rddl.observ_fluents:
            observed_vars = rddl.observ_fluents
        else:
            observed_vars = rddl.state_fluents
             
        # state is concatenated into single tensor
        def _jax_wrapped_subs_to_state(subs):
            subs = {var: value
                    for (var, value) in subs.items()
                    if var in observed_vars}
            flat_subs = jax.tree_map(jnp.ravel, subs)
            states = list(flat_subs.values())
            state = jnp.concatenate(states)
            return state
        
        # train action prediction
        def _jax_wrapped_drp_predict_train(key, params, hyperparams, step, subs):
            state = _jax_wrapped_subs_to_state(subs)
            actions = predict_fn.apply(params, state)
            if not wrap_non_bool:
                for (var, action) in actions.items():
                    if var != bool_key and ranges[var] != 'bool':
                        actions[var] = jnp.clip(action, *bounds[var])
            if use_constraint_satisfaction:
                bool_actions = _jax_unstack_bool_from_softmax(actions[bool_key])
                actions.update(bool_actions)
                del actions[bool_key]
            return actions
        
        # test action prediction
        def _jax_wrapped_drp_predict_test(key, params, hyperparams, step, subs):
            actions = _jax_wrapped_drp_predict_train(
                key, params, hyperparams, step, subs)
            new_actions = {}
            for (var, action) in actions.items():
                prange = ranges[var]
                if prange == 'bool':
                    new_action = action > 0.5
                elif prange == 'int':
                    action = jnp.clip(action, *bounds[var])
                    new_action = jnp.round(action).astype(compiled.INT)
                else:
                    new_action = jnp.clip(action, *bounds[var])
                new_actions[var] = new_action
            return new_actions
        
        self.train_policy = _jax_wrapped_drp_predict_train
        self.test_policy = _jax_wrapped_drp_predict_test
        
        # ***********************************************************************
        # ACTION CONSTRAINT SATISFACTION
        #
        # ***********************************************************************
        
        # no projection applied since the actions are already constrained
        def _jax_wrapped_drp_no_projection(params, hyperparams):
            return params, True
        
        self.projection = _jax_wrapped_drp_no_projection
    
        # ***********************************************************************
        # POLICY NETWORK INITIALIZATION
        #
        # ***********************************************************************
        
        def _jax_wrapped_drp_init(key, hyperparams, subs):
            subs = {var: value[0, ...] 
                    for (var, value) in subs.items()
                    if var in observed_vars}
            state = _jax_wrapped_subs_to_state(subs)
            params = predict_fn.init(key, state)
            return params
        
        self.initializer = _jax_wrapped_drp_init
        
    def guess_next_epoch(self, params: Dict) -> Dict:
        return params


class JaxQValueFunction:
    
    def __init__(self, topology_state: Sequence[int],
                 topology_action: Sequence[int],
                 topology_common: Sequence[int],
                 activation: Callable=jax.nn.relu,
                 initializer: hk.initializers.Initializer=hk.initializers.VarianceScaling(scale=2.0)):
        self._topology_state = topology_state
        self._topology_action = topology_action
        self._topology_common = topology_common
        self._activation = activation
        self._initializer_base = initializer
        self._initializer = initializer
    
    def summarize_hyperparameters(self):
        print(f'Q-function hyper-parameters:\n'
              f'    topology_state ={self._topology_state}\n'
              f'    topology_action={self._topology_action}\n'
              f'    topology_common={self._topology_common}\n'
              f'    activation_fn  ={self._activation.__name__}\n'
              f'    initializer    ={type(self._initializer_base).__name__}')
    
    def compile(self, compiled: JaxRDDLCompilerWithGrad) -> None:
        rddl = compiled.rddl
        
        # ***********************************************************************
        # Q-VALUE NETWORK PREDICTION
        #
        # ***********************************************************************
                   
        init = self._initializer
        
        # predict actions from the policy network for current state-action pair
        def _jax_wrapped_q_network_predict(state, action):
            
            # feed state through hidden layers
            hidden_state = state
            for (i, num_neuron) in enumerate(self._topology_state):
                linear = hk.Linear(num_neuron, name=f'hidden_state_{i}', w_init=init)
                hidden_state = self._activation(linear(hidden_state))
            
            # feed action through hidden layers
            hidden_action = action
            for (i, num_neuron) in enumerate(self._topology_action):
                linear = hk.Linear(num_neuron, name=f'hidden_action_{i}', w_init=init)
                hidden_action = self._activation(linear(hidden_action))
            
            # continue feeding through shared hidden layers
            hidden = jnp.concatenate([hidden_state, hidden_action], axis=-1)
            for (i, num_neuron) in enumerate(self._topology_common):
                linear = hk.Linear(num_neuron, name=f'hidden_shared_{i}', w_init=init)
                hidden = self._activation(linear(hidden))            
            linear = hk.Linear(1, name='output', w_init=init)
            value = linear(hidden)            
            return value
        
        predict_fn = hk.transform(_jax_wrapped_q_network_predict)
        predict_fn = hk.without_apply_rng(predict_fn)       
        
        # extract only the observed variables from the subs dictionary
        if rddl.observ_fluents:
            observed_vars = rddl.observ_fluents
        else:
            observed_vars = rddl.state_fluents
             
        def _jax_wrapped_flatten_subs(subs):
            flat_subs = jax.tree_map(jnp.ravel, subs)
            values = list(flat_subs.values())
            return jnp.concatenate(values, axis=None)
        
        def _jax_wrapped_q_predict(params, state, action):
            state = {var: value
                     for (var, value) in state.items()
                     if var in observed_vars}
            state = _jax_wrapped_flatten_subs(state)
            action = _jax_wrapped_flatten_subs(action)
            q_values = predict_fn.apply(params, state, action)
            return q_values
        
        self.predict_q = _jax_wrapped_q_predict
               
        # ***********************************************************************
        # Q-VALUE NETWORK INITIALIZATION
        #
        # ***********************************************************************
        
        def _jax_wrapped_q_init(key, state, action):
            state = {var: value[0, ...] 
                     for (var, value) in state.items()
                     if var in observed_vars}
            state = _jax_wrapped_flatten_subs(state)
            action = _jax_wrapped_flatten_subs(action)
            params = predict_fn.init(key, state, action)
            return params
        
        self.initializer = _jax_wrapped_q_init


# ***********************************************************************
# ALL VERSIONS OF JAX PLANNER
# 
# - simple gradient descent based planner
# - more stable but slower line search based planner
#
# ***********************************************************************
class RollingMean:
    
    def __init__(self, window_size: int) -> None:
        self._window_size = window_size
        self._memory = deque(maxlen=window_size)
        self._total = 0
    
    def update(self, x: float) -> float:
        memory = self._memory
        self._total += x
        if len(memory) == self._window_size:
            self._total -= memory.popleft()
        memory.append(x)
        return self._total / len(memory)


class JaxBackpropPlanner:
    '''A class for optimizing an action sequence in the given RDDL MDP using 
    gradient descent.'''
    
    def __init__(self, rddl: RDDLLiftedModel,
                 plan: JaxPlan,
                 batch_size_train: int=32,
                 batch_size_test: Optional[int]=None,
                 rollout_horizon: Optional[int]=None,
                 use64bit: bool=False,
                 action_bounds: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]]=None,
                 optimizer: Callable[..., optax.GradientTransformation]=optax.rmsprop,
                 optimizer_kwargs: Optional[Dict[str, object]]=None,
                 clip_grad: Optional[float]=None,
                 logic: FuzzyLogic=FuzzyLogic(),
                 use_symlog_reward: bool=False,
                 utility=jnp.mean,
                 cpfs_without_grad: Optional[Set]=None,
                 logger: Optional[Logger]=None) -> None:
        '''Creates a new gradient-based algorithm for optimizing action sequences
        (plan) in the given RDDL. Some operations will be converted to their
        differentiable counterparts; the specific operations can be customized
        by providing a subclass of FuzzyLogic.
        
        :param rddl: the RDDL domain to optimize
        :param plan: the policy/plan representation to optimize
        :param batch_size_train: how many rollouts to perform per optimization 
        step
        :param batch_size_test: how many rollouts to use to test the plan at each
        optimization step
        :param rollout_horizon: lookahead planning horizon: None uses the
        :param use64bit: whether to perform arithmetic in 64 bit
        horizon parameter in the RDDL instance
        :param action_bounds: box constraints on actions
        :param optimizer: a factory for an optax SGD algorithm
        :param optimizer_kwargs: a dictionary of parameters to pass to the SGD
        factory (e.g. which parameters are controllable externally)
        :param clip_grad: maximum magnitude of gradient updates
        :param logic: a subclass of FuzzyLogic for mapping exact mathematical
        operations to their differentiable counterparts 
        :param use_symlog_reward: whether to use the symlog transform on the 
        reward as a form of normalization
        :param utility: how to aggregate return observations to compute utility
        of a policy or plan
        :param cpfs_without_grad: which CPFs do not have gradients (use straight
        through gradient trick)
        :param logger: to log information about compilation to file
        '''
        self.rddl = rddl
        self.plan = plan
        self.batch_size_train = batch_size_train
        if batch_size_test is None:
            batch_size_test = batch_size_train
        self.batch_size_test = batch_size_test
        if rollout_horizon is None:
            rollout_horizon = rddl.horizon
        self.horizon = rollout_horizon
        if action_bounds is None:
            action_bounds = {}
        self._action_bounds = action_bounds
        self.use64bit = use64bit
        self._optimizer_name = optimizer
        if optimizer_kwargs is None:
            optimizer_kwargs = {'learning_rate': 0.1}
        self._optimizer_kwargs = optimizer_kwargs
        self.clip_grad = clip_grad
        
        # set optimizer
        try:
            optimizer = optax.inject_hyperparams(optimizer)(**optimizer_kwargs)
        except:
            raise_warning(
                'Failed to inject hyperparameters into optax optimizer, '
                'rolling back to safer method: please note that modification of '
                'optimizer hyperparameters will not work, and it is '
                'recommended to update your packages and Python distribution.',
                'red')
            optimizer = optimizer(**optimizer_kwargs)     
        if clip_grad is None:
            self.optimizer = optimizer
        else:
            self.optimizer = optax.chain(
                optax.clip(clip_grad),
                optimizer
            )
            
        self.logic = logic
        self.use_symlog_reward = use_symlog_reward
        self.utility = utility
        if cpfs_without_grad is None:
            cpfs_without_grad = set()
        self.cpfs_without_grad = cpfs_without_grad
        self.logger = logger
        
        self._jax_compile_rddl()        
        self._jax_compile_optimizer()
        
    def summarize_hyperparameters(self):
        print(f'objective and relaxations:\n'
              f'    objective_fn    ={self.utility.__name__}\n'
              f'    use_symlog      ={self.use_symlog_reward}\n'
              f'    lookahead       ={self.horizon}\n'
              f'    model relaxation={type(self.logic).__name__}\n'
              f'    action_bounds   ={self._action_bounds}\n'
              f'    cpfs_no_gradient={self.cpfs_without_grad}\n'
              f'optimizer hyper-parameters:\n'
              f'    use_64_bit      ={self.use64bit}\n'
              f'    optimizer       ={self._optimizer_name.__name__}\n'
              f'    optimizer args  ={self._optimizer_kwargs}\n'
              f'    clip_gradient   ={self.clip_grad}\n'
              f'    batch_size_train={self.batch_size_train}\n'
              f'    batch_size_test ={self.batch_size_test}')
        self.plan.summarize_hyperparameters()
        self.logic.summarize_hyperparameters()
        
    def _jax_compile_rddl(self):
        rddl = self.rddl
        
        # Jax compilation of the differentiable RDDL for training
        self.compiled = JaxRDDLCompilerWithGrad(
            rddl=rddl,
            logic=self.logic,
            logger=self.logger,
            use64bit=self.use64bit,
            cpfs_without_grad=self.cpfs_without_grad)
        self.compiled.compile(log_jax_expr=True, heading='RELAXED MODEL')
        
        # Jax compilation of the exact RDDL for testing
        self.test_compiled = JaxRDDLCompiler(
            rddl=rddl, 
            logger=self.logger,
            use64bit=self.use64bit)
        self.test_compiled.compile(log_jax_expr=True, heading='EXACT MODEL')
        
    def _jax_compile_optimizer(self):
        
        # policy
        self.plan.compile(self.compiled,
                          _bounds=self._action_bounds,
                          horizon=self.horizon)
        self.train_policy = jax.jit(self.plan.train_policy)
        self.test_policy = jax.jit(self.plan.test_policy)
        
        # roll-outs
        train_rollouts = self.compiled.compile_rollouts(
            policy=self.plan.train_policy,
            n_steps=self.horizon,
            n_batch=self.batch_size_train)
        
        test_rollouts = self.test_compiled.compile_rollouts(
            policy=self.plan.test_policy,
            n_steps=self.horizon,
            n_batch=self.batch_size_test)
        self.test_rollouts = jax.jit(test_rollouts)
        
        # initialization
        self.initialize = jax.jit(self._jax_init())
        
        # losses
        train_loss = self._jax_loss(train_rollouts, use_symlog=self.use_symlog_reward)
        self.train_loss = jax.jit(train_loss)
        self.test_loss = jax.jit(self._jax_loss(test_rollouts, use_symlog=False))
        
        # optimization
        self.update = jax.jit(self._jax_update(train_loss))
    
    def _jax_return(self, use_symlog):
        gamma = self.rddl.discount
        
        # apply discounting of future reward and then optional symlog transform
        def _jax_wrapped_returns(rewards):
            if gamma != 1:
                horizon = rewards.shape[1]
                discount = jnp.power(gamma, jnp.arange(horizon))
                rewards = rewards * discount[jnp.newaxis, ...]
            returns = jnp.sum(rewards, axis=1)
            if use_symlog:
                returns = jnp.sign(returns) * jnp.log1p(jnp.abs(returns))
            return returns
        
        return _jax_wrapped_returns
        
    def _jax_loss(self, rollouts, use_symlog=False): 
        utility_fn = self.utility        
        _jax_wrapped_returns = self._jax_return(use_symlog)
        
        # the loss is the average cumulative reward across all roll-outs
        def _jax_wrapped_plan_loss(key, policy_params, hyperparams,
                                   subs, model_params):
            log = rollouts(key, policy_params, hyperparams, subs, model_params)
            rewards = log['reward']
            returns = _jax_wrapped_returns(rewards)
            utility = utility_fn(returns)
            loss = -utility
            return loss, log
        
        return _jax_wrapped_plan_loss
    
    def _jax_init(self):
        init = self.plan.initializer
        optimizer = self.optimizer
        
        def _jax_wrapped_init_policy(key, hyperparams, subs):
            policy_params = init(key, hyperparams, subs)
            opt_state = optimizer.init(policy_params)
            return policy_params, opt_state
        
        return _jax_wrapped_init_policy
        
    def _jax_update(self, loss):
        optimizer = self.optimizer
        projection = self.plan.projection
        
        # calculate the plan gradient w.r.t. return loss and update optimizer
        # also perform a projection step to satisfy constraints on actions
        def _jax_wrapped_plan_update(key, policy_params, hyperparams,
                                     subs, model_params, opt_state):
            grad_fn = jax.grad(loss, argnums=1, has_aux=True)
            grad, log = grad_fn(key, policy_params, hyperparams, subs, model_params)  
            updates, opt_state = optimizer.update(grad, opt_state) 
            policy_params = optax.apply_updates(policy_params, updates)
            policy_params, converged = projection(policy_params, hyperparams)
            log['grad'] = grad
            log['updates'] = updates
            return policy_params, converged, opt_state, log
        
        return _jax_wrapped_plan_update
            
    def _batched_init_subs(self, subs): 
        rddl = self.rddl
        n_train, n_test = self.batch_size_train, self.batch_size_test
        
        # batched subs
        init_train, init_test = {}, {}
        for (name, value) in subs.items():
            init_value = self.test_compiled.init_values.get(name, None)
            if init_value is None:
                raise RDDLUndefinedVariableError(
                    f'Variable <{name}> in subs argument is not a '
                    f'valid p-variable, must be one of '
                    f'{set(self.test_compiled.init_values.keys())}.')
            value = np.reshape(value, newshape=np.shape(init_value))[np.newaxis, ...]
            train_value = np.repeat(value, repeats=n_train, axis=0)
            train_value = train_value.astype(self.compiled.REAL)
            init_train[name] = train_value
            init_test[name] = np.repeat(value, repeats=n_test, axis=0)
        
        # make sure next-state fluents are also set
        for (state, next_state) in rddl.next_state.items():
            init_train[next_state] = init_train[state]
            init_test[next_state] = init_test[state]
        
        return init_train, init_test
    
    def optimize(self, *args, return_callback: bool=False, **kwargs) -> Dict[str, object]:
        ''' Compute an optimal straight-line plan. Returns the parameters
        for the optimized policy.
        
        :param return_callback: whether to return the callback from training
        instead of the parameters
        :param key: JAX PRNG key   
        :param epochs: the maximum number of steps of gradient descent
        :param the maximum number of steps of gradient descent     
        :param train_seconds: total time allocated for gradient descent
        :param plot_step: frequency to plot the plan and save result to disk
        :param model_params: optional model-parameters to override default
        :param policy_hyperparams: hyper-parameters for the policy/plan, such as
        weights for sigmoid wrapping boolean actions
        :param subs: dictionary mapping initial state and non-fluents to 
        their values: if None initializes all variables from the RDDL instance
        :param guess: initial policy parameters: if None will use the initializer
        specified in this instance
        :param verbose: not print (0), print summary (1), print progress (2)
        :param test_rolling_window: the test return is averaged on a rolling 
        window of the past test_rolling_window returns when updating the best
        parameters found so far
        :param tqdm_position: position of tqdm progress bar (for multiprocessing)
        '''
        it = self.optimize_generator(*args, **kwargs)
        callback = deque(it, maxlen=1).pop()
        if return_callback:
            return callback
        else:
            return callback['best_params']
    
    def optimize_generator(self, key: random.PRNGKey,
                           epochs: int=999999,
                           train_seconds: float=120.,
                           plot_step: Optional[int]=None,
                           model_params: Optional[Dict[str, object]]=None,
                           policy_hyperparams: Optional[Dict[str, object]]=None,
                           subs: Optional[Dict[str, object]]=None,
                           guess: Optional[Dict[str, object]]=None,
                           verbose: int=2,
                           test_rolling_window: int=10,
                           tqdm_position: Optional[int]=None) -> Generator[Dict[str, object], None, None]:
        '''Returns a generator for computing an optimal straight-line plan. 
        Generator can be iterated over to lazily optimize the plan, yielding
        a dictionary of intermediate computations.
        
        :param key: JAX PRNG key   
        :param epochs: the maximum number of steps of gradient descent
        :param the maximum number of steps of gradient descent     
        :param train_seconds: total time allocated for gradient descent
        :param plot_step: frequency to plot the plan and save result to disk
        :param model_params: optional model-parameters to override default
        :param policy_hyperparams: hyper-parameters for the policy/plan, such as
        weights for sigmoid wrapping boolean actions
        :param subs: dictionary mapping initial state and non-fluents to 
        their values: if None initializes all variables from the RDDL instance
        :param guess: initial policy parameters: if None will use the initializer
        specified in this instance
        :param verbose: not print (0), print summary (1), print progress (2)
        :param test_rolling_window: the test return is averaged on a rolling 
        window of the past test_rolling_window returns when updating the best
        parameters found so far
        :param tqdm_position: position of tqdm progress bar (for multiprocessing)
        '''
        verbose = int(verbose)
        start_time = time.time()
        elapsed_outside_loop = 0
        
        # if policy_hyperparams is not provided
        if policy_hyperparams is None:
            raise_warning('policy_hyperparams is not set, setting 1.0 for '
                          'all action-fluents which could be suboptimal.', 'red')
            policy_hyperparams = {action: 1.0 
                                  for action in self.rddl.action_fluents}
        
        # if policy_hyperparams is a scalar
        elif isinstance(policy_hyperparams, (int, float, np.number)):
            raise_warning(f'policy_hyperparams is {policy_hyperparams}, '
                          'setting this value for all action-fluents.')
            hyperparam_value = float(policy_hyperparams)
            policy_hyperparams = {action: hyperparam_value
                                  for action in self.rddl.action_fluents}
            
        # print summary of parameters:
        if verbose >= 1:
            print('==============================================\n'
                  'JAX PLANNER PARAMETER SUMMARY\n'
                  '==============================================')
            self.summarize_hyperparameters()
            print(f'optimize() call hyper-parameters:\n'
                  f'    max_iterations     ={epochs}\n'
                  f'    max_seconds        ={train_seconds}\n'
                  f'    model_params       ={model_params}\n'
                  f'    policy_hyper_params={policy_hyperparams}\n'
                  f'    override_subs_dict ={subs is not None}\n'
                  f'    provide_param_guess={guess is not None}\n'
                  f'    test_rolling_window={test_rolling_window}\n' 
                  f'    plot_frequency     ={plot_step}\n')
            
        # compute a batched version of the initial values
        if subs is None:
            subs = self.test_compiled.init_values
        else:
            # if some p-variables are not provided, add their default values
            subs = subs.copy()
            added_pvars_to_subs = []
            for (var, value) in self.test_compiled.init_values.items():
                if var not in subs:
                    subs[var] = value
                    added_pvars_to_subs.append(var)
            if added_pvars_to_subs:
                raise_warning(f'p-variables {added_pvars_to_subs} not in '
                              'provided subs, using their initial values '
                              'from the RDDL files.')
        train_subs, test_subs = self._batched_init_subs(subs)
        
        # initialize, model parameters
        if model_params is None:
            model_params = self.compiled.model_params
        model_params_test = self.test_compiled.model_params
        
        # initialize policy parameters
        if guess is None:
            key, subkey = random.split(key)
            policy_params, opt_state = self.initialize(
                subkey, policy_hyperparams, train_subs)
        else:
            policy_params = guess
            opt_state = self.optimizer.init(policy_params)
        
        # initialize running statistics
        best_params, best_loss, best_grad = policy_params, jnp.inf, jnp.inf
        last_iter_improve = 0
        rolling_test_loss = RollingMean(test_rolling_window)
        log = {}
        
        # training loop
        iters = range(epochs)
        if verbose >= 2:
            iters = tqdm(iters, total=100, position=tqdm_position)
        
        for it in iters:
            
            # update the parameters of the plan
            key, subkey1, subkey2, subkey3 = random.split(key, num=4)
            policy_params, converged, opt_state, train_log = self.update(
                subkey1, policy_params, policy_hyperparams,
                train_subs, model_params, opt_state)
            if not np.all(converged):
                raise_warning(
                    'Projected gradient method for satisfying action concurrency '
                    'constraints reached the iteration limit: plan is possibly '
                    'invalid for the current instance.', 'red')
            
            # evaluate losses
            train_loss, _ = self.train_loss(
                subkey2, policy_params, policy_hyperparams,
                train_subs, model_params)
            test_loss, log = self.test_loss(
                subkey3, policy_params, policy_hyperparams,
                test_subs, model_params_test)
            test_loss = rolling_test_loss.update(test_loss)
            
            # record the best plan so far
            if test_loss < best_loss:
                best_params, best_loss, best_grad = \
                    policy_params, test_loss, train_log['grad']
                last_iter_improve = it
            
            # save the plan figure
            if plot_step is not None and it % plot_step == 0:
                self._plot_actions(
                    key, policy_params, policy_hyperparams, test_subs, it)
            
            # if the progress bar is used
            elapsed = time.time() - start_time - elapsed_outside_loop
            if verbose >= 2:
                iters.n = int(100 * min(1, max(elapsed / train_seconds, it / epochs)))
                iters.set_description(
                    f'[{tqdm_position}] {it:6} it / {-train_loss:14.4f} train / '
                    f'{-test_loss:14.4f} test / {-best_loss:14.4f} best')
            
            # return a callback
            start_time_outside = time.time()
            yield {
                'iteration': it,
                'train_return':-train_loss,
                'test_return':-test_loss,
                'best_return':-best_loss,
                'params': policy_params,
                'best_params': best_params,
                'last_iteration_improved': last_iter_improve,
                'grad': train_log['grad'],
                'best_grad': best_grad,
                'updates': train_log['updates'],
                'elapsed_time': elapsed,
                'key': key,
                **log
            }
            elapsed_outside_loop += (time.time() - start_time_outside)
            
            # reached time budget
            if elapsed >= train_seconds:
                break
            
            # numerical error
            if not np.isfinite(train_loss):
                raise_warning(
                    f'Aborting optimization due to invalid train loss {train_loss}.',
                    'red')
                break
        
        if verbose >= 2:
            iters.close()
        
        # validate the test return
        if log:
            messages = set()
            for error_code in np.unique(log['error']):
                messages.update(JaxRDDLCompiler.get_error_messages(error_code))
            if messages:
                messages = '\n'.join(messages)
                raise_warning('The JAX compiler encountered the following '
                              'problems in the original RDDL '
                              f'during test evaluation:\n{messages}', 'red')                               
        
        # summarize and test for convergence
        if verbose >= 1:
            grad_norm = jax.tree_map(
                lambda x: np.array(jnp.linalg.norm(x)).item(), best_grad)
            diagnosis = self._perform_diagnosis(
                last_iter_improve, -train_loss, -test_loss, -best_loss, grad_norm)
            print(f'summary of optimization:\n'
                  f'    time_elapsed  ={elapsed}\n'
                  f'    iterations    ={it}\n'
                  f'    best_objective={-best_loss}\n'
                  f'    grad_norm     ={grad_norm}\n'
                  f'diagnosis: {diagnosis}\n')
    
    def _perform_diagnosis(self, last_iter_improve, 
                           train_return, test_return, best_return, grad_norm):
        max_grad_norm = max(jax.tree_util.tree_leaves(grad_norm))
        grad_is_zero = np.allclose(max_grad_norm, 0)
        
        validation_error = 100 * abs(test_return - train_return) / \
                            max(abs(train_return), abs(test_return))
        
        # divergence if the solution is not finite
        if not np.isfinite(train_return):
            return termcolor.colored('[FAILURE] training loss diverged.', 'red')
            
        # hit a plateau is likely IF:
        # 1. planner does not improve at all
        # 2. the gradient norm at the best solution is zero
        if last_iter_improve <= 1:
            if grad_is_zero:
                return termcolor.colored(
                    '[FAILURE] no progress was made, '
                    f'and max grad norm = {max_grad_norm}, '
                    'likely stuck in a plateau.', 'red')
            else:
                return termcolor.colored(
                    '[FAILURE] no progress was made, '
                    f'but max grad norm = {max_grad_norm} > 0, '
                    'likely due to bad l.r. or other hyper-parameter.', 'red')
        
        # model is likely poor IF:
        # 1. the train and test return disagree
        if not (validation_error < 20):
            return termcolor.colored(
                '[WARNING] progress was made, '
                f'but relative train test error = {validation_error} is high, '
                'likely poor model relaxation around the solution, '
                'or the batch size is too small.', 'yellow')
        
        # model likely did not converge IF:
        # 1. the max grad relative to the return is high
        if not grad_is_zero:
            return_to_grad_norm = abs(best_return) / max_grad_norm
            if not (return_to_grad_norm > 1):
                return termcolor.colored(
                    '[WARNING] progress was made, '
                    f'but max grad norm = {max_grad_norm} is high, '
                    'likely indicates the solution is not locally optimal, '
                    'or the model is not smooth around the solution, '
                    'or the batch size is too small.', 'yellow')
        
        # likely successful
        return termcolor.colored(
            '[SUCCESS] planner appears to have converged successfully '
            '(note: not all potential problems can be ruled out).', 'green')
        
    def get_action(self, key: random.PRNGKey,
                   params: Dict,
                   step: int,
                   subs: Dict,
                   policy_hyperparams: Optional[Dict[str, object]]=None) -> Dict[str, object]:
        '''Returns an action dictionary from the policy or plan with the given
        parameters.
        
        :param key: the JAX PRNG key
        :param params: the trainable parameter PyTree of the policy
        :param step: the time step at which decision is made
        :param policy_hyperparams: hyper-parameters for the policy/plan, such as
        weights for sigmoid wrapping boolean actions
        :param subs: the dict of pvariables
        '''
        
        # check compatibility of the subs dictionary
        for (var, values) in subs.items():
            
            # must not be grounded
            if RDDLPlanningModel.FLUENT_SEP in var \
            or RDDLPlanningModel.OBJECT_SEP in var:
                raise ValueError(f'State dictionary passed to the JAX policy is '
                                 f'grounded, since it contains the key <{var}>, '
                                 f'but a vectorized environment is required: '
                                 f'please make sure vectorized=True in the RDDLEnv.')
            
            # must be numeric array
            # exception is for POMDPs at 1st epoch when observ-fluents are None
            if not jnp.issubdtype(values.dtype, jnp.number) \
            and not jnp.issubdtype(values.dtype, jnp.bool_):
                if step == 0 and var in self.rddl.observ_fluents:
                    subs[var] = self.test_compiled.init_values[var]
                else:
                    raise ValueError(f'Values assigned to pvariable {var} are '
                                     f'non-numeric of type {values.dtype}: {values}.')
            
        # cast device arrays to numpy
        actions = self.test_policy(key, params, policy_hyperparams, step, subs)
        actions = jax.tree_map(np.asarray, actions)
        return actions      
            
    def _plot_actions(self, key, params, hyperparams, subs, it):
        try:
            import matplotlib.pyplot as plt
        except Exception:
            raise_warning('matplotlib is not installed, aborting plot.', 'red')
            return
            
        # predict actions from the trained policy or plan
        actions = self.test_rollouts(key, params, hyperparams, subs, {})['action']
            
        # plot the action sequences as color maps
        fig, axs = plt.subplots(nrows=len(actions), constrained_layout=True)
        if len(actions) == 1:
            axs = [axs]
        
        rddl = self.rddl
        for (ax, name) in zip(axs, actions):
            action = np.mean(actions[name], axis=0, dtype=float)
            action = np.reshape(action, newshape=(action.shape[0], -1)).T
            if rddl.variable_ranges[name] == 'bool':
                vmin, vmax = 0.0, 1.0
            else:
                vmin, vmax = None, None                
            img = ax.imshow(
                action, vmin=vmin, vmax=vmax, cmap='seismic', aspect='auto')
            ax.set_xlabel('time')
            ax.set_ylabel(name)
            plt.colorbar(img, ax=ax)
            
        # write plot to disk
        plt.savefig(f'plan_{rddl.domain_name}_{rddl.instance_name}_{it}.pdf',
                    bbox_inches='tight')
        plt.clf()
        plt.close(fig)


class Buffer:
    
    def __init__(self, n_samples:int=10000, n_batch:int=32) -> None:
        self.n_samples = n_samples
        self.n_batch = n_batch
        
    def compile(self, rddl):
        
        def _reshape(arr):
            return jnp.reshape(arr, newshape=(-1,) + arr.shape[2:])
        
        # convert trajectories in log into tensors whose leading dimension
        # combines the batch and time dimensions of the original trajectories
        def _jax_process_trajectories(log):
            fluents, rewards, dones = \
                log['fluents'], log['reward'], log['termination']
            states = {name: _reshape(values) 
                      for (name, values) in fluents.items()
                      if name in rddl.state_fluents}
            actions = {name: _reshape(values)
                       for (name, values) in fluents.items()
                       if name in rddl.action_fluents}
            next_states = {rddl.prev_state[name]: _reshape(values)
                           for (name, values) in fluents.items()
                           if name in rddl.prev_state}
            rewards = _reshape(rewards)
            dones = _reshape(dones)
            return states, actions, rewards, next_states, dones
        
        self.process_fn = jax.jit(_jax_process_trajectories)
                      
    def summarize_hyperparameters(self):
        print(f'replay buffer hyper-parameters:\n'
              f'    capacity  ={self.n_samples}\n'
              f'    batch_size={self.n_batch}')
     
    def reset(self) -> None:
        self.states = {}
        self.actions = {}
        self.rewards = None
        self.next_states = {}
        self.dones = None
        self.index = 0
        self.size = 0
    
    def replay(self) -> Tuple[np.ndarray, ...]:
        if self.size < self.n_batch: 
            return None
        indices = np.random.randint(low=0, high=self.size, size=(self.n_batch,))
        states = {name: values[indices] for (name, values) in self.states.items()}
        actions = {name: values[indices] for (name, values) in self.actions.items()}
        rewards = self.rewards[indices]
        next_states = {name: values[indices] for (name, values) in self.next_states.items()}
        dones = self.dones[indices]
        return states, actions, rewards, next_states, dones
    
    def _initialize(self, states, actions, next_states):
        self.states = {
            name: np.empty((self.n_samples,) + values.shape[1:], dtype=values.dtype)
            for (name, values) in states.items()
        }
        self.actions = {
            name: np.empty((self.n_samples,) + values.shape[1:], dtype=values.dtype)
            for (name, values) in actions.items()
        }
        self.rewards = np.empty((self.n_samples,), dtype=float)
        self.next_states = {
            name: np.empty((self.n_samples,) + values.shape[1:], dtype=values.dtype)
            for (name, values) in next_states.items()
        }
        self.dones = np.empty((self.n_samples,), dtype=bool)
    
    def append(self, log):
        states, actions, rewards, next_states, dones = self.process_fn(log)
        if not self.size:
            self._initialize(states, actions, next_states)
        n_steps = rewards.size
        indices = np.arange(self.index, self.index + n_steps) % self.n_samples
        for (name, values) in states.items():
            self.states[name][indices] = np.asarray(values)
        for (name, values) in actions.items():
            self.actions[name][indices] = np.asarray(values)
        self.rewards[indices] = np.asarray(rewards)
        for (name, values) in next_states.items():
            self.next_states[name][indices] = np.asarray(values)
        self.dones[indices] = np.asarray(dones)       
        self.index += n_steps
        self.size = min(self.size + n_steps, self.n_samples)
        
        
class JaxBackpropPlannerWithQ(JaxBackpropPlanner):
    
    def __init__(self, *args,
                 q: JaxQValueFunction,
                 optimizer_q: Callable[..., optax.GradientTransformation]=optax.adam,
                 optimizer_q_kwargs: Optional[Dict[str, object]]=None,
                 buffer:Buffer=Buffer(),
                 tau:float=0.1,
                 num_q_updates:int=1,
                 reg_penalty:float=0.,
                 **kwargs) -> None:
        self.q = q
        if optimizer_q_kwargs is None:
            optimizer_q_kwargs = {'learning_rate': 0.0005}
        self.optimizer_q_kwargs = optimizer_q_kwargs  
        self._optimizer_q_name = optimizer_q
        self.optimizer_q = optimizer_q(**optimizer_q_kwargs)   
        self.buffer = buffer
        self.tau = tau
        self.num_q_updates = num_q_updates
        self.reg_penalty = reg_penalty
                        
        super(JaxBackpropPlannerWithQ, self).__init__(*args, **kwargs)
        
    def summarize_hyperparameters(self):
        super(JaxBackpropPlannerWithQ, self).summarize_hyperparameters()
        self.q.summarize_hyperparameters()
        self.buffer.summarize_hyperparameters()
        print(f'Q-function optimizer hyper-parameters:\n'
              f'    optimizer          ={self._optimizer_q_name.__name__}\n'
              f'    optimizer args     ={self.optimizer_q_kwargs}\n'
              f'    target_weight (tau)={self.tau}\n'
              f'    num_updates        ={self.num_q_updates}\n'
              f'    l2_regularization  ={self.reg_penalty}')
                
    def _jax_compile_optimizer(self):
        
        # policy
        self.plan.compile(self.compiled,
                          _bounds=self._action_bounds,
                          horizon=self.horizon)
        self.train_policy = jax.jit(self.plan.train_policy)
        self.test_policy = jax.jit(self.plan.test_policy)
        
        # value
        self.q.compile(self.compiled)
        
        # roll-outs
        train_rollouts = self.compiled.compile_rollouts(
            policy=self.plan.train_policy,
            n_steps=self.horizon,
            n_batch=self.batch_size_train)
        
        test_rollouts = self.test_compiled.compile_rollouts(
            policy=self.plan.test_policy,
            n_steps=self.horizon,
            n_batch=self.batch_size_test)
        self.test_rollouts = jax.jit(test_rollouts)
        
        # initialization
        self.initialize = jax.jit(self._jax_init())
        self.initialize_q = jax.jit(self._jax_init_q())
        
        # policy losses
        train_loss = self._jax_loss(
            train_rollouts, self.train_policy, 
            bootstrap=False, use_symlog=self.use_symlog_reward)
        self.train_loss = jax.jit(train_loss)
        self.test_loss = jax.jit(self._jax_loss(
            test_rollouts, self.test_policy, bootstrap=False, use_symlog=False))
        
        # q-value losses
        q_loss = self._jax_loss_q()
        self.q_loss = jax.jit(q_loss)
        
        # optimization
        self.update = jax.jit(self._jax_update(train_loss))
        self.update_q = jax.jit(self._jax_update_q(q_loss))
    
    def _jax_init_q(self):        
        init_q = self.q.initializer
        optimizer_q = self.optimizer_q
        policy = self.train_policy
        
        def _jax_wrapped_init_q(key, policy_params, hyperparams, subs):            
            init_subs = {name: values[0, ...] for (name, values) in subs.items()}
            action = policy(key, policy_params, hyperparams, 0, init_subs)  
            q_params = init_q(key, subs, action)
            q_opt_state = optimizer_q.init(q_params)
            return q_params, q_opt_state
        
        return _jax_wrapped_init_q
        
    def _jax_loss(self, rollouts, policy, bootstrap=True, use_symlog=False): 
        utility_fn = self.utility        
        _jax_wrapped_returns = self._jax_return(use_symlog)
        q_fn = self.q.predict_q
        rddl = self.rddl
        gamma = self.rddl.discount
        
        # the loss is the average cumulative reward across all roll-outs
        # optionally add Q(state, policy(state)) for the last state
        def _jax_wrapped_plan_loss(key, policy_params, hyperparams,
                                   subs, model_params, q_params):
            key, subkey = random.split(key)
            log = rollouts(subkey, policy_params, hyperparams, subs, model_params)
            rewards = log['reward']
            returns = _jax_wrapped_returns(rewards)
            if bootstrap:     
                n_batch, n_steps = rewards.shape
                key, *subkeys = random.split(key, num=1 + n_batch)
                keys = jnp.asarray(subkeys)
                last_states = {name: values[:, -1, ...]
                               for (name, values) in log['fluents'].items()
                               if name in rddl.prev_state}
                last_actions = jax.vmap(policy, in_axes=(0, None, None, None, 0))(
                    keys, policy_params, hyperparams, n_steps, last_states)
                q_values = jax.vmap(q_fn, in_axes=(None, 0, 0))(
                    q_params, last_states, last_actions)                
                q_values = jnp.reshape(q_values, newshape=returns.shape)      
                returns += jnp.power(gamma, n_steps) * q_values
            utility = utility_fn(returns)
            loss = -utility
            return loss, log
        
        return _jax_wrapped_plan_loss
    
    def _jax_loss_q(self): 
        q_fn = self.q.predict_q
        gamma = self.rddl.discount
        reg_penalty = self.reg_penalty
        
        # the error between current Q-value and target value
        def _jax_wrapped_q_error(q_params, q_target_params, state, action, reward,
                                 next_state, next_action, done):
            q_value = q_fn(q_params, state, action)
            next_q_value = q_fn(q_target_params, next_state, next_action)
            target = reward + (1 - done) * gamma * next_q_value
            error = q_value - jax.lax.stop_gradient(target)
            return error
        
        # batched MSE loss function for training Q-value function
        def _jax_wrapped_q_loss_batched(q_params, q_target_params, 
                                        states, actions, rewards,
                                        next_states, next_actions, dones):
            batched_error_fn = jax.vmap(
                _jax_wrapped_q_error, in_axes=(None, None, 0, 0, 0, 0, 0, 0))
            errors = batched_error_fn(
                q_params, q_target_params, states, actions, rewards, 
                next_states, next_actions, dones)
            loss = jnp.mean(jnp.square(errors))
            for param in jax.tree_util.tree_leaves(q_params):
                loss += reg_penalty * jnp.mean(jnp.square(param))
            return loss
            
        return _jax_wrapped_q_loss_batched
        
    def _jax_update(self, loss):
        optimizer = self.optimizer
        projection = self.plan.projection
        
        # calculate the plan gradient w.r.t. return loss and update optimizer
        # also perform a projection step to satisfy constraints on actions
        def _jax_wrapped_plan_update(key, policy_params, hyperparams,
                                     subs, model_params, opt_state, q_params):
            grad, log = jax.grad(loss, argnums=1, has_aux=True)(
                key, policy_params, hyperparams, subs, model_params, q_params)  
            updates, opt_state = optimizer.update(grad, opt_state) 
            policy_params = optax.apply_updates(policy_params, updates)
            policy_params, converged = projection(policy_params, hyperparams)
            log['grad'] = grad
            log['updates'] = updates
            return policy_params, converged, opt_state, log
        
        return _jax_wrapped_plan_update
            
    def _jax_update_q(self, q_loss):
        optimizer_q = self.optimizer_q
        policy = self.test_policy
        tau = self.tau
        
        # update the q-value parameters and the target parameters
        def _jax_wrapped_q_update(q_params, q_opt_state, q_target_params,
                                  states, actions, rewards, 
                                  next_states, next_actions, dones):
            loss, grad = jax.value_and_grad(q_loss, argnums=0)(
                q_params, q_target_params, 
                states, actions, rewards, next_states, next_actions, dones)
            updates, q_opt_state = optimizer_q.update(grad, q_opt_state)
            q_params = optax.apply_updates(q_params, updates)
            q_target_params = jax.tree_map(
                lambda x, y: tau * x + (1. - tau) * y, q_params, q_target_params)
            return q_params, q_opt_state, q_target_params, loss        
        
        # update the q-value parameters and the target parameters using the actor
        def _jax_wrapped_q_update_policy(q_params, q_opt_state, q_target_params,
                                         states, actions, rewards, 
                                         next_states, dones, 
                                         key, policy_params, hyperparams):
            key, *subkeys = random.split(key, num=1 + dones.size)
            keys = jnp.asarray(subkeys)
            next_actions = jax.vmap(policy, in_axes=(0, None, None, None, 0))(
                keys, policy_params, hyperparams, 0, next_states)
            return _jax_wrapped_q_update(q_params, q_opt_state, q_target_params, 
                                         states, actions, rewards, 
                                         next_states, next_actions, dones)
        
        return _jax_wrapped_q_update_policy
            
    def optimize_generator(self, key: random.PRNGKey,
                           epochs: int=999999,
                           train_seconds: float=120.,
                           plot_step: Optional[int]=None,
                           model_params: Optional[Dict[str, object]]=None,
                           policy_hyperparams: Optional[Dict[str, object]]=None,
                           subs: Optional[Dict[str, object]]=None,
                           guess: Optional[Dict[str, object]]=None,
                           verbose: int=2,
                           test_rolling_window: int=10,
                           tqdm_position: Optional[int]=None) -> Generator[Dict[str, object], None, None]:
        verbose = int(verbose)
        start_time = time.time()
        elapsed_outside_loop = 0
        
        # if policy_hyperparams is not provided
        if policy_hyperparams is None:
            raise_warning('policy_hyperparams is not set, setting 1.0 for '
                          'all action-fluents which could be suboptimal.', 'red')
            policy_hyperparams = {action: 1.0 
                                  for action in self.rddl.action_fluents}
        
        # if policy_hyperparams is a scalar
        elif isinstance(policy_hyperparams, (int, float, np.number)):
            raise_warning(f'policy_hyperparams is {policy_hyperparams}, '
                          'setting this value for all action-fluents.')
            hyperparam_value = float(policy_hyperparams)
            policy_hyperparams = {action: hyperparam_value
                                  for action in self.rddl.action_fluents}
            
        # print summary of parameters:
        if verbose >= 1:
            print('==============================================\n'
                  'JAX PLANNER PARAMETER SUMMARY\n'
                  '==============================================')
            self.summarize_hyperparameters()
            print(f'optimize() call hyper-parameters:\n'
                  f'    max_iterations     ={epochs}\n'
                  f'    max_seconds        ={train_seconds}\n'
                  f'    model_params       ={model_params}\n'
                  f'    policy_hyper_params={policy_hyperparams}\n'
                  f'    override_subs_dict ={subs is not None}\n'
                  f'    provide_param_guess={guess is not None}\n'
                  f'    test_rolling_window={test_rolling_window}\n' 
                  f'    plot_frequency     ={plot_step}\n')
            
        # compute a batched version of the initial values
        if subs is None:
            subs = self.test_compiled.init_values
        else:
            # if some p-variables are not provided, add their default values
            subs = subs.copy()
            added_pvars_to_subs = []
            for (var, value) in self.test_compiled.init_values.items():
                if var not in subs:
                    subs[var] = value
                    added_pvars_to_subs.append(var)
            if added_pvars_to_subs:
                raise_warning(f'p-variables {added_pvars_to_subs} not in '
                              'provided subs, using their initial values '
                              'from the RDDL files.')
        train_subs, test_subs = self._batched_init_subs(subs)
        
        # initialize model parameters
        if model_params is None:
            model_params = self.compiled.model_params
        model_params_test = self.test_compiled.model_params
        
        # initialize policy parameters
        if guess is None:
            key, subkey = random.split(key)
            policy_params, opt_state = self.initialize(
                subkey, policy_hyperparams, train_subs)
        else:
            policy_params = guess
            opt_state = self.optimizer.init(policy_params)
        
        # initialize replay buffer
        self.buffer.compile(self.rddl)
        self.buffer.reset()
        q_losses = []
        
        # initialize q-value parameters
        key, subkey = random.split(key)
        q_params, q_opt_state = self.initialize_q(
            subkey, policy_params, policy_hyperparams, train_subs)
        q_target_params = q_params
        
        # initialize running statistics
        best_params, best_loss, best_grad = policy_params, jnp.inf, jnp.inf
        last_iter_improve = 0
        rolling_test_loss = RollingMean(test_rolling_window)
        log = {}
        
        # training loop
        iters = range(epochs)
        if verbose >= 2:
            iters = tqdm(iters, total=100, position=tqdm_position)
        
        for it in iters:
            
            # update the parameters of the plan
            key, subkey1, subkey2, subkey3 = random.split(key, num=4)
            policy_params, converged, opt_state, train_log = self.update(
                subkey1, policy_params, policy_hyperparams,
                train_subs, model_params, opt_state, q_params)
            if not np.all(converged):
                raise_warning(
                    'Projected gradient method for satisfying action concurrency '
                    'constraints reached the iteration limit: plan is possibly '
                    'invalid for the current instance.', 'red')
            
            # evaluate losses
            train_loss, _ = self.train_loss(
                subkey2, policy_params, policy_hyperparams, 
                train_subs, model_params, q_params)
            test_loss, log = self.test_loss(
                subkey3, policy_params, policy_hyperparams,
                test_subs, model_params_test, q_params)
            test_loss = rolling_test_loss.update(test_loss)
            
            # update Q-values
            self.buffer.append(log)
            key, *subkeys = random.split(key, num=1 + self.num_q_updates)
            q_loss_mean = 0.
            for subkey in subkeys:
                batch = self.buffer.replay()
                if batch is not None:
                    q_params, q_opt_state, q_target_params, q_loss_val = self.update_q(
                        q_params, q_opt_state, q_target_params, *batch, 
                        subkey, policy_params, policy_hyperparams)
                    q_loss_mean += q_loss_val / self.num_q_updates
            q_losses.append(q_loss_mean)
            
            if it % 200 == 0:
                n = 30
                grid_values = np.zeros((n, n))
                for i1, r1 in enumerate(np.linspace(0, 100, n)):
                    for i2, r2 in enumerate(np.linspace(0, 100, n)):
                        state_r1_r2 = {'rlevel': np.asarray([r1, r2])}
                        action_r1_r2 = {'release': np.asarray([0., 0.])}
                        grid_values[i1, i2] = self.q.predict_q(q_params, state_r1_r2, action_r1_r2)
                import matplotlib.pyplot as plt
                plt.imshow(grid_values, extent=[0, 100, 0, 100], origin='lower')
                plt.savefig(f'grid_{it}.pdf')
                    
            # record the best plan so far
            if test_loss < best_loss:
                best_params, best_loss, best_grad = \
                    policy_params, test_loss, train_log['grad']
                last_iter_improve = it
            
            # save the plan figure
            if plot_step is not None and it % plot_step == 0:
                self._plot_actions(
                    key, policy_params, policy_hyperparams, test_subs, it)
            
            # if the progress bar is used
            elapsed = time.time() - start_time - elapsed_outside_loop
            if verbose >= 2:
                iters.n = int(100 * min(1, max(elapsed / train_seconds, it / epochs)))
                iters.set_description(
                    f'[{tqdm_position}] {it:6} it / {-train_loss:14.4f} train / '
                    f'{-test_loss:14.4f} test / {-best_loss:14.4f} best / '
                    f'{q_loss_mean:14.4f} q-loss')
            
            # return a callback
            start_time_outside = time.time()
            yield {
                'iteration': it,
                'train_return':-train_loss,
                'test_return':-test_loss,
                'best_return':-best_loss,
                'params': policy_params,
                'best_params': best_params,
                'last_iteration_improved': last_iter_improve,
                'grad': train_log['grad'],
                'best_grad': best_grad,
                'updates': train_log['updates'],
                'elapsed_time': elapsed,
                'key': key,
                **log
            }
            elapsed_outside_loop += (time.time() - start_time_outside)
            
            # reached time budget
            if elapsed >= train_seconds:
                break
            
            # numerical error
            if not np.isfinite(train_loss):
                raise_warning(
                    f'Aborting optimization due to invalid train loss {train_loss}.',
                    'red')
                break
            if not np.isfinite(q_loss_mean):
                raise_warning(
                    f'Aborting optimization due to invalid q loss {q_loss_mean}.',
                    'red')
                break
        
        if verbose >= 2:
            iters.close()
        
        # validate the test return
        if log:
            messages = set()
            for error_code in np.unique(log['error']):
                messages.update(JaxRDDLCompiler.get_error_messages(error_code))
            if messages:
                messages = '\n'.join(messages)
                raise_warning('The JAX compiler encountered the following '
                              'problems in the original RDDL '
                              f'during test evaluation:\n{messages}', 'red')                               
        
        # summarize and test for convergence
        if verbose >= 1:
            grad_norm = jax.tree_map(
                lambda x: np.array(jnp.linalg.norm(x)).item(), best_grad)
            diagnosis = self._perform_diagnosis(
                last_iter_improve, -train_loss, -test_loss, -best_loss, grad_norm)
            print(f'summary of optimization:\n'
                  f'    time_elapsed  ={elapsed}\n'
                  f'    iterations    ={it}\n'
                  f'    best_objective={-best_loss}\n'
                  f'    grad_norm     ={grad_norm}\n'
                  f'diagnosis: {diagnosis}\n')
        
        n_smooth = 10
        import matplotlib.pyplot as plt
        q_losses = np.asarray(q_losses)
        ret = np.cumsum(q_losses, dtype=float)
        ret[n_smooth:] = ret[n_smooth:] - ret[:-n_smooth]
        q_losses = ret[n_smooth - 1:] / n_smooth
        plt.plot(q_losses)
        plt.savefig('q_value.pdf')


class JaxArmijoLineSearchPlanner(JaxBackpropPlanner):
    '''A class for optimizing an action sequence in the given RDDL MDP using 
    Armijo linear search gradient descent.'''
    
    def __init__(self, *args,
                 optimizer: Callable[..., optax.GradientTransformation]=optax.sgd,
                 optimizer_kwargs: Dict[str, object]={'learning_rate': 1.0},
                 beta: float=0.8,
                 c: float=0.1,
                 lrmax: float=1.0,
                 lrmin: float=1e-5,
                 **kwargs) -> None:
        '''Creates a new gradient-based algorithm for optimizing action sequences
        (plan) in the given RDDL using Armijo line search. All arguments are the
        same as in the parent class, except:
        
        :param beta: reduction factor of learning rate per line search iteration
        :param c: coefficient in Armijo condition
        :param lrmax: initial learning rate for line search
        :param lrmin: minimum possible learning rate (line search halts)
        '''
        self.beta = beta
        self.c = c
        self.lrmax = lrmax
        self.lrmin = lrmin
        super(JaxArmijoLineSearchPlanner, self).__init__(
            *args,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            **kwargs)
        
    def summarize_hyperparameters(self):
        super(JaxArmijoLineSearchPlanner, self).summarize_hyperparameters()
        print(f'linesearch hyper-parameters:\n'
              f'    beta    ={self.beta}\n'
              f'    c       ={self.c}\n'
              f'    lr_range=({self.lrmin}, {self.lrmax})\n')
    
    def _jax_update(self, loss):
        optimizer = self.optimizer
        projection = self.plan.projection
        beta, c, lrmax, lrmin = self.beta, self.c, self.lrmax, self.lrmin
        
        # continue line search if Armijo condition not satisfied and learning
        # rate can be further reduced
        def _jax_wrapped_line_search_armijo_check(val):
            (_, old_f, _, old_norm_g2, _), (_, new_f, lr, _), _, _ = val            
            return jnp.logical_and(
                new_f >= old_f - c * lr * old_norm_g2,
                lr >= lrmin / beta)
            
        def _jax_wrapped_line_search_iteration(val):
            old, new, best, aux = val
            old_x, _, old_g, _, old_state = old
            _, _, lr, iters = new
            _, best_f, _, _ = best
            key, hyperparams, *other = aux
            
            # anneal learning rate and apply a gradient step
            new_lr = beta * lr
            old_state.hyperparams['learning_rate'] = new_lr
            updates, new_state = optimizer.update(old_g, old_state)
            new_x = optax.apply_updates(old_x, updates)
            new_x, _ = projection(new_x, hyperparams)
            
            # evaluate new loss and record best so far
            new_f, _ = loss(key, new_x, hyperparams, *other)
            new = (new_x, new_f, new_lr, iters + 1)
            best = jax.lax.cond(
                new_f < best_f,
                lambda: (new_x, new_f, new_lr, new_state),
                lambda: best
            )
            return old, new, best, aux
            
        def _jax_wrapped_plan_update(key, policy_params, hyperparams,
                                     subs, model_params, opt_state):
            
            # calculate initial loss value, gradient and squared norm
            old_x = policy_params
            loss_and_grad_fn = jax.value_and_grad(loss, argnums=1, has_aux=True)
            (old_f, log), old_g = loss_and_grad_fn(
                key, old_x, hyperparams, subs, model_params)            
            old_norm_g2 = jax.tree_map(lambda x: jnp.sum(jnp.square(x)), old_g)
            old_norm_g2 = jax.tree_util.tree_reduce(jnp.add, old_norm_g2)
            log['grad'] = old_g
            
            # initialize learning rate to maximum
            new_lr = lrmax / beta
            old = (old_x, old_f, old_g, old_norm_g2, opt_state)
            new = (old_x, old_f, new_lr, 0)            
            best = (old_x, jnp.inf, jnp.nan, opt_state)
            aux = (key, hyperparams, subs, model_params)
            
            # do a single line search step with the initial learning rate
            init_val = (old, new, best, aux)            
            init_val = _jax_wrapped_line_search_iteration(init_val)
            
            # continue to anneal the learning rate until Armijo condition holds
            # or the learning rate becomes too small, then use the best parameter
            _, (*_, iters), (best_params, _, best_lr, best_state), _ = \
            jax.lax.while_loop(
                cond_fun=_jax_wrapped_line_search_armijo_check,
                body_fun=_jax_wrapped_line_search_iteration,
                init_val=init_val
            )
            best_state.hyperparams['learning_rate'] = best_lr
            log['updates'] = None
            log['line_search_iters'] = iters
            log['learning_rate'] = best_lr
            return best_params, True, best_state, log
            
        return _jax_wrapped_plan_update


# ***********************************************************************
# ALL VERSIONS OF CONTROLLERS
# 
# - offline controller is the straight-line planner
# - online controller is the replanning mode
#
# ***********************************************************************
class JaxOfflineController(BaseAgent):
    '''A container class for a Jax policy trained offline.'''
    use_tensor_obs = True
    
    def __init__(self, planner: JaxBackpropPlanner, 
                 key: random.PRNGKey,
                 eval_hyperparams: Optional[Dict[str, object]]=None,
                 params: Optional[Dict[str, object]]=None,
                 train_on_reset: bool=False,
                 **train_kwargs) -> None:
        '''Creates a new JAX offline control policy that is trained once, then
        deployed later.
        
        :param planner: underlying planning algorithm for optimizing actions
        :param key: the RNG key to seed randomness
        :param eval_hyperparams: policy hyperparameters to apply for evaluation
        or whenever sample_action is called
        :param params: use the specified policy parameters instead of calling
        planner.optimize()
        :param train_on_reset: retrain policy parameters on every episode reset
        :param **train_kwargs: any keyword arguments to be passed to the planner
        for optimization
        '''
        self.planner = planner
        self.key = key
        self.eval_hyperparams = eval_hyperparams
        self.train_on_reset = train_on_reset
        self.train_kwargs = train_kwargs        
        self.params_given = params is not None
        
        self.step = 0
        if not self.train_on_reset and not self.params_given:
            params = self.planner.optimize(key=self.key, **self.train_kwargs) 
        self.params = params  
        
    def sample_action(self, state):
        self.key, subkey = random.split(self.key)
        actions = self.planner.get_action(
            subkey, self.params, self.step, state, self.eval_hyperparams)
        self.step += 1
        return actions
        
    def reset(self):
        self.step = 0
        if self.train_on_reset and not self.params_given:
            self.params = self.planner.optimize(key=self.key, **self.train_kwargs)


class JaxOnlineController(BaseAgent):
    '''A container class for a Jax controller continuously updated using state 
    feedback.'''
    use_tensor_obs = True
    
    def __init__(self, planner: JaxBackpropPlanner, 
                 key: random.PRNGKey,
                 eval_hyperparams: Optional[Dict[str, object]]=None,
                 warm_start: bool=True,
                 **train_kwargs) -> None:
        '''Creates a new JAX control policy that is trained online in a closed-
        loop fashion.
        
        :param planner: underlying planning algorithm for optimizing actions
        :param key: the RNG key to seed randomness
        :param eval_hyperparams: policy hyperparameters to apply for evaluation
        or whenever sample_action is called
        :param **train_kwargs: any keyword arguments to be passed to the planner
        for optimization
        '''
        self.planner = planner
        self.key = key
        self.eval_hyperparams = eval_hyperparams
        self.warm_start = warm_start
        self.train_kwargs = train_kwargs
        self.reset()
     
    def sample_action(self, state):
        planner = self.planner
        params = planner.optimize(
            key=self.key,
            guess=self.guess,
            subs=state,
            **self.train_kwargs)
        self.key, subkey = random.split(self.key)
        actions = planner.get_action(subkey, params, 0, state, self.eval_hyperparams)
        if self.warm_start:
            self.guess = planner.plan.guess_next_epoch(params)
        return actions
        
    def reset(self):
        self.guess = None
    
