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
# [2] Patton, Noah, Jihwan Jeong, Mike Gimelfarb, and Scott Sanner. "A Distributional 
# Framework for Risk-Sensitive End-to-End Planning in Continuous MDPs." In Proceedings of 
# the AAAI Conference on Artificial Intelligence, vol. 36, no. 9, pp. 9894-9901. 2022.
#
# [3] Bueno, Thiago P., Leliane N. de Barros, Denis D. Mauá, and Scott Sanner. "Deep 
# reactive policies for planning in stochastic nonlinear domains." In Proceedings of the 
# AAAI Conference on Artificial Intelligence, vol. 33, no. 01, pp. 7530-7537. 2019.
#
# [4] Cui, Hao, Thomas Keller, and Roni Khardon. "Stochastic planning with lifted symbolic 
# trajectory optimization." In Proceedings of the International Conference on Automated 
# Planning and Scheduling, vol. 29, pp. 119-127. 2019.
#
# [5] Wu, Ga, Buser Say, and Scott Sanner. "Scalable planning with tensorflow for hybrid 
# nonlinear domains." Advances in Neural Information Processing Systems 30 (2017).
#
# [6] Sehnke, Frank, and Tingting Zhao. "Baseline-free sampling in parameter exploring 
# policy gradients: Super symmetric pgpe." Artificial Neural Networks: Methods and 
# Applications in Bio-/Neuroinformatics. Springer International Publishing, 2015.
#
# ***********************************************************************


from abc import ABCMeta, abstractmethod
from ast import literal_eval
from collections import deque
import configparser
from enum import Enum
from functools import partial
import os
import pickle
import sys
import time
import traceback
from typing import Any, Callable, Dict, Generator, Optional, Sequence, Type, Tuple, Union

import haiku as hk
import jax
import jax.nn.initializers as initializers
import jax.numpy as jnp
import jax.random as random
import numpy as np
import optax
import termcolor
from tqdm import tqdm, TqdmWarning
import warnings
warnings.filterwarnings("ignore", category=TqdmWarning)

from pyRDDLGym.core.compiler.initializer import RDDLValueInitializer
from pyRDDLGym.core.compiler.model import RDDLPlanningModel, RDDLLiftedModel
from pyRDDLGym.core.debug.logger import Logger
from pyRDDLGym.core.debug.exception import (
    raise_warning,
    RDDLNotImplementedError,
    RDDLUndefinedVariableError,
    RDDLTypeError
)
from pyRDDLGym.core.policy import BaseAgent

from pyRDDLGym_jax import __version__
from pyRDDLGym_jax.core.compiler import JaxRDDLCompiler
from pyRDDLGym_jax.core import logic
from pyRDDLGym_jax.core.logic import (
    JaxRDDLCompilerWithGrad, DefaultJaxRDDLCompilerWithGrad, stable_sigmoid
)

# try to load the dash board
try:
    from pyRDDLGym_jax.core.visualization import JaxPlannerDashboard
except Exception:
    raise_warning('Failed to load the dashboard visualization tool: '
                  'ensure all prerequisite packages are installed.', 'red')
    traceback.print_exc()
    JaxPlannerDashboard = None

Activation = Callable[[jnp.ndarray], jnp.ndarray]
Bounds = Dict[str, Tuple[np.ndarray, np.ndarray]]
Kwargs = Dict[str, Any]
Pytree = Any


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
    args = {section: {k: literal_eval(v) for (k, v) in config.items(section)}
            for section in config.sections()}
    return config, args


def _parse_config_string(value: str):
    config = configparser.RawConfigParser()
    config.optionxform = str 
    config.read_string(value)
    args = {section: {k: literal_eval(v) for (k, v) in config.items(section)}
            for section in config.sections()}
    return config, args


def _getattr_any(packages, item):
    for package in packages:
        loaded = getattr(package, item, None)
        if loaded is not None:
            return loaded
    return None


def _load_config(config, args):
    compiler_kwargs = {k: args['Compiler'][k] for (k, _) in config.items('Compiler')}
    planner_args = {k: args['Planner'][k] for (k, _) in config.items('Planner')}
    train_args = {k: args['Optimize'][k] for (k, _) in config.items('Optimize')}    
    
    # read the compiler settings
    compiler_name = compiler_kwargs.pop('method', 'DefaultJaxRDDLCompilerWithGrad')
    planner_args['compiler'] = getattr(logic, compiler_name)
    planner_args['compiler_kwargs'] = compiler_kwargs

    # read the policy settings
    plan_method = planner_args.pop('method')
    plan_kwargs = planner_args.pop('method_kwargs', {})  
    
    # policy initialization
    plan_initializer = plan_kwargs.get('initializer', None)
    if plan_initializer is not None:
        initializer = _getattr_any(
            packages=[initializers, hk.initializers], item=plan_initializer)
        if initializer is None:
            raise ValueError(f'Invalid initializer <{plan_initializer}>.')
        else:
            init_kwargs = plan_kwargs.pop('initializer_kwargs', {})
            try: 
                plan_kwargs['initializer'] = initializer(**init_kwargs)
            except Exception as _:
                raise ValueError(f'Invalid initializer kwargs <{init_kwargs}>.')
    
    # policy activation
    plan_activation = plan_kwargs.get('activation', None)
    if plan_activation is not None:
        activation = _getattr_any(packages=[jax.nn, jax.numpy], item=plan_activation)
        if activation is None:
            raise ValueError(f'Invalid activation <{plan_activation}>.')
        else:
            plan_kwargs['activation'] = activation
    
    # read the planner settings
    planner_args['plan'] = getattr(sys.modules[__name__], plan_method)(**plan_kwargs)
    
    # planner optimizer
    planner_optimizer = planner_args.get('optimizer', None)
    if planner_optimizer is not None:
        optimizer = _getattr_any(packages=[optax], item=planner_optimizer)
        if optimizer is None:
            raise ValueError(f'Invalid optimizer <{planner_optimizer}>.')
        else:
            planner_args['optimizer'] = optimizer

    # pgpe optimizer
    pgpe_method = planner_args.get('pgpe', 'GaussianPGPE')
    pgpe_kwargs = planner_args.pop('pgpe_kwargs', {})
    if pgpe_method is not None:
        if 'optimizer' in pgpe_kwargs:
            pgpe_optimizer = _getattr_any(packages=[optax], item=pgpe_kwargs['optimizer'])
            if pgpe_optimizer is None:
                raise ValueError(f'Invalid optimizer <{pgpe_optimizer}>.')
            else:
                pgpe_kwargs['optimizer'] = pgpe_optimizer
        planner_args['pgpe'] = getattr(sys.modules[__name__], pgpe_method)(**pgpe_kwargs)

    # preprocessor settings
    preproc_method = planner_args.get('preprocessor', None)
    preproc_kwargs = planner_args.pop('preprocessor_kwargs', {})
    if preproc_method is not None:
        planner_args['preprocessor'] = getattr(
            sys.modules[__name__], preproc_method)(**preproc_kwargs)

    # optimize call RNG key
    planner_key = train_args.get('key', None)
    if planner_key is not None:
        train_args['key'] = random.PRNGKey(planner_key)
    
    # dashboard
    dashboard_key = planner_args.get('dashboard', None)
    if dashboard_key is not None and dashboard_key and JaxPlannerDashboard is not None:
        planner_args['dashboard'] = JaxPlannerDashboard()
    elif dashboard_key is not None:
        del planner_args['dashboard']        
    
    # optimize call stopping rule
    stopping_rule = train_args.get('stopping_rule', None)
    if stopping_rule is not None:
        stopping_rule_kwargs = train_args.pop('stopping_rule_kwargs', {})
        train_args['stopping_rule'] = getattr(
            sys.modules[__name__], stopping_rule)(**stopping_rule_kwargs)
    
    return planner_args, plan_kwargs, train_args


def load_config(path: str) -> Tuple[Kwargs, ...]:
    '''Loads a config file at the specified file path.
    
    :param path: the path of the config file to load and parse
    '''
    config, args = _parse_config_file(path)
    return _load_config(config, args)


def load_config_from_string(value: str) -> Tuple[Kwargs, ...]:
    '''Loads config file contents specified explicitly as a string value.
    
    :param value: the string in json format containing the config contents to parse
    '''
    config, args = _parse_config_string(value)
    return _load_config(config, args)
    

# ***********************************************************************
# ALL VERSIONS OF STATE PREPROCESSING FOR DRP
# 
# - static normalization
# 
# ***********************************************************************


class Preprocessor(metaclass=ABCMeta):
    '''Base class for all state preprocessors.'''

    HYPERPARAMS_KEY = 'preprocessor__'

    def __init__(self) -> None:
        self._initializer = None
        self._update = None
        self._transform = None

    @property
    def initialize(self) -> Callable:
        return self._initializer

    @property
    def update(self) -> Callable:
        return self._update
    
    @property
    def transform(self) -> Callable:
        return self._transform
    
    @abstractmethod
    def compile(self, compiled: JaxRDDLCompilerWithGrad) -> None:
        pass


class StaticNormalizer(Preprocessor):
    '''Normalize values by box constraints on fluents computed from the RDDL domain.'''

    def __init__(self, fluent_bounds: Dict[str, Tuple[np.ndarray, np.ndarray]]={}) -> None:
        '''Create a new instance of the static normalizer.

        :param fluent_bounds: optional bounds on fluents to overwrite default values.
        '''
        self.fluent_bounds = fluent_bounds

    def compile(self, compiled: JaxRDDLCompilerWithGrad) -> None:
        
        # adjust for partial observability
        rddl = compiled.rddl
        if rddl.observ_fluents:
            observed_vars = rddl.observ_fluents
        else:
            observed_vars = rddl.state_fluents

        # ignore boolean fluents and infinite bounds
        bounded_vars = {}
        for var in observed_vars:
            if rddl.variable_ranges[var] != 'bool':
                lower, upper = compiled.constraints.bounds[var]
                if np.all(np.isfinite(lower) & np.isfinite(upper) & (lower < upper)):
                    bounded_vars[var] = (lower, upper)
                user_bounds = self.fluent_bounds.get(var, None)
                if user_bounds is not None:
                    bounded_vars[var] = tuple(user_bounds)
        
        # initialize to ranges computed by the constraint parser
        def _jax_wrapped_normalizer_init():
            return bounded_vars        
        self._initializer = jax.jit(_jax_wrapped_normalizer_init)

        # static bounds
        def _jax_wrapped_normalizer_update(fls, stats):
            return {var: (jnp.asarray(lower, dtype=compiled.REAL), 
                          jnp.asarray(upper, dtype=compiled.REAL)) 
                    for (var, (lower, upper)) in bounded_vars.items()}
        self._update = jax.jit(_jax_wrapped_normalizer_update)

        # apply min max scaling
        def _jax_wrapped_normalizer_transform(fls, stats):
            new_fls = {}
            for (var, values) in fls.items():
                if var in stats:
                    lower, upper = stats[var]
                    new_dims = jnp.ndim(values) - jnp.ndim(lower)
                    lower = lower[(jnp.newaxis,) * new_dims + (...,)]
                    upper = upper[(jnp.newaxis,) * new_dims + (...,)]
                    new_fls[var] = (values - lower) / (upper - lower)
                else:
                    new_fls[var] = values
            return new_fls
        self._transform = jax.jit(_jax_wrapped_normalizer_transform)


# ***********************************************************************
# ALL VERSIONS OF JAX PLANS
# 
# - straight line plan
# - deep reactive policy
#
# ***********************************************************************


class JaxPlan(metaclass=ABCMeta):
    '''Base class for all JAX policy representations.'''
    
    def __init__(self) -> None:
        self._initializer = None
        self._train_policy = None
        self._test_policy = None
        self._projection = None
        self.bounds = None
        
    def summarize_hyperparameters(self) -> str:
        return self.__str__()
    
    @abstractmethod
    def compile(self, compiled: JaxRDDLCompilerWithGrad,
                _bounds: Bounds,
                horizon: int,
                preprocessor: Optional[Preprocessor]=None) -> None:
        pass
    
    @abstractmethod
    def guess_next_epoch(self, params: Pytree) -> Pytree:
        pass
    
    @property
    def initializer(self) -> Callable:
        return self._initializer

    @initializer.setter
    def initializer(self, value: Callable) -> None:
        self._initializer = value
    
    @property
    def train_policy(self) -> Callable:
        return self._train_policy

    @train_policy.setter
    def train_policy(self, value: Callable) -> None:
        self._train_policy = value
        
    @property
    def test_policy(self) -> Callable:
        return self._test_policy

    @test_policy.setter
    def test_policy(self, value: Callable) -> None:
        self._test_policy = value
         
    @property
    def projection(self) -> Callable:
        return self._projection

    @projection.setter
    def projection(self, value: Callable) -> None:
        self._projection = value
    
    def _calculate_action_info(self, compiled, user_bounds, horizon):
        shapes, bounds, bounds_safe, cond_lists = {}, {}, {}, {}
        for (name, prange) in compiled.rddl.variable_ranges.items():
            if compiled.rddl.variable_types[name] != 'action-fluent':
                continue
            
            # check invalid type
            if prange not in compiled.JAX_TYPES and prange not in compiled.rddl.enum_types:
                keys = list(compiled.JAX_TYPES.keys()) + list(compiled.rddl.enum_types)
                raise RDDLTypeError(
                    f'Invalid range <{prange}> of action-fluent <{name}>, '
                    f'must be one of {keys}.'
                )
                
            # clip boolean to (0, 1), otherwise use the RDDL action bounds
            # or the user defined action bounds if provided
            shapes[name] = (horizon,) + np.shape(compiled.init_values[name])
            if prange == 'bool':
                lower, upper = None, None
            else:

                # enum values are ordered from 0 to number of objects - 1
                if prange in compiled.rddl.enum_types:
                    lower = np.zeros(shape=shapes[name][1:])
                    upper = len(compiled.rddl.type_to_objects[prange]) - 1
                    upper = np.ones(shape=shapes[name][1:]) * upper
                else:
                    lower, upper = compiled.constraints.bounds[name]
                
                # override with user defined bounds
                lower, upper = user_bounds.get(name, (lower, upper))
                lower = np.asarray(lower, dtype=compiled.REAL)
                upper = np.asarray(upper, dtype=compiled.REAL)

                # get masks for a jax conditional statement to avoid numerical errors
                # for infinite values
                lower_finite = np.isfinite(lower)
                upper_finite = np.isfinite(upper)
                bounds_safe[name] = (np.where(lower_finite, lower, 0.0),
                                     np.where(upper_finite, upper, 0.0))
                cond_lists[name] = [lower_finite & upper_finite,
                                    lower_finite & ~upper_finite,
                                    ~lower_finite & upper_finite,
                                    ~lower_finite & ~upper_finite]
            bounds[name] = (lower, upper)

            if compiled.print_warnings:
                print(termcolor.colored(
                    f'[INFO] Bounds of action-fluent <{name}> set to {bounds[name]}.', 
                    'dark_grey'
                ))
        return shapes, bounds, bounds_safe, cond_lists
    
    def _count_bool_actions(self, rddl):
        constraint = rddl.max_allowed_actions
        num_bool_actions = sum(np.size(values)
                               for (var, values) in rddl.action_fluents.items()
                               if rddl.variable_ranges[var] == 'bool')
        return num_bool_actions, constraint


class JaxActionProjection(metaclass=ABCMeta):
    '''Base of all straight-line plan action projections.'''

    @abstractmethod
    def compile(self, *args, **kwargs) -> Callable:
        pass
    

class JaxSortingActionProjection(JaxActionProjection):
    '''Action projection using sorting method.'''

    def compile(self, ranges: Dict[str, str], noop: Dict[str, Any], 
                wrap_sigmoid: bool, allowed_actions: int, bool_threshold: float, 
                jax_bool_to_box: Callable, *args, **kwargs) -> Callable:

        # shift the boolean actions uniformly, clipping at the min/max values
        # the amount to move is such that only top allowed_actions actions
        # are still active (e.g. not equal to noop) after the shift
        def _jax_wrapped_sorting_project(params, hyperparams):
            
            # find the amount to shift action parameters: if noop=True reflect parameter
            scores = []
            for (var, param) in params.items():
                if ranges[var] == 'bool':
                    param_flat = jnp.ravel(param, order='C')
                    if noop[var]:
                        if wrap_sigmoid:
                            param_flat = -param_flat
                        else:
                            param_flat = 1.0 - param_flat
                    scores.append(param_flat)
            scores = jnp.concatenate(scores)
            descending = jnp.sort(scores)[::-1]
            kplus1st_greatest = descending[allowed_actions]
            surplus = jnp.maximum(kplus1st_greatest - bool_threshold, 0.0)
                
            # perform the shift
            new_params = {}
            for (var, param) in params.items():
                if ranges[var] == 'bool':
                    if noop[var]:
                        new_param = param + surplus
                    else:
                        new_param = param - surplus
                    new_params[var] = jax_bool_to_box(var, new_param, hyperparams)
                else:
                    new_params[var] = param
            converged = jnp.array(True, dtype=jnp.bool_)
            return new_params, converged
        return _jax_wrapped_sorting_project


class JaxSogbofaActionProjection(JaxActionProjection):
    '''Action projection using the SOGBOFA method.'''

    def compile(self, ranges: Dict[str, str], noop: Dict[str, Any], 
                allowed_actions: int, max_constraint_iter: int, 
                jax_param_to_action: Callable, jax_action_to_param: Callable, 
                min_action: float, max_action: float, real_dtype: type, 
                *args, **kwargs) -> Callable:
        
        # calculate the surplus of actions above max-nondef-actions
        def _jax_wrapped_sogbofa_surplus(actions):
            sum_action = jnp.array(0.0, dtype=real_dtype)
            k = jnp.array(0, dtype=jnp.int32)
            for (var, action) in actions.items():
                if ranges[var] == 'bool':
                    if noop[var]:
                        action = 1 - action                       
                    sum_action = sum_action + jnp.sum(action)
                    k = k + jnp.count_nonzero(action)
            surplus = jnp.maximum(sum_action - allowed_actions, 0.0)
            return surplus, k
            
        # return whether the surplus is positive or reached compute limit    
        def _jax_wrapped_sogbofa_continue(values):
            it, _, surplus, k = values
            return jnp.logical_and(
                it < max_constraint_iter, jnp.logical_and(surplus > 0, k > 0))
            
        # reduce all bool action values by the surplus clipping at minimum
        # for no-op = True, do the opposite, i.e. increase all
        # bool action values by surplus clipping at maximum
        def _jax_wrapped_sogbofa_subtract_surplus(values):
            it, actions, surplus, k = values
            amount = surplus / k
            new_actions = {}
            for (var, action) in actions.items():
                if ranges[var] == 'bool':
                    if noop[var]:
                        new_actions[var] = jnp.minimum(action + amount, 1)
                    else:
                        new_actions[var] = jnp.maximum(action - amount, 0)
                else:
                    new_actions[var] = action
            new_surplus, new_k = _jax_wrapped_sogbofa_surplus(new_actions)
            new_it = it + 1
            return new_it, new_actions, new_surplus, new_k
            
        # apply the surplus to the actions until it becomes zero
        def _jax_wrapped_sogbofa_project(params, hyperparams):

            # convert parameters to actions
            actions = {}
            for (var, param) in params.items():
                if ranges[var] == 'bool':
                    actions[var] = jax_param_to_action(var, param, hyperparams)
                else:
                    actions[var] = param
            
            # run SOGBOFA loop on the actions to get adjusted actions
            surplus, k = _jax_wrapped_sogbofa_surplus(actions)
            _, actions, surplus, k = jax.lax.while_loop(
                cond_fun=_jax_wrapped_sogbofa_continue,
                body_fun=_jax_wrapped_sogbofa_subtract_surplus,
                init_val=(0, actions, surplus, k)
            )
            converged = jnp.logical_not(surplus > 0)

            # check for any remaining constraint violation
            total_bool = jnp.array(0, dtype=jnp.int32)
            for (var, action) in actions.items():
                if ranges[var] == 'bool':
                    if noop[var]:
                        total_bool = total_bool + jnp.count_nonzero(action < 0.5)
                    else:
                        total_bool = total_bool + jnp.count_nonzero(action > 0.5)
            excess = jnp.maximum(total_bool - allowed_actions, 0)
            
            # convert the adjusted actions back to parameters
            # reduce the excess number of parameters that are non-noop above constraint
            new_params = {}            
            for (var, action) in actions.items():
                if ranges[var] == 'bool':
                    action = jnp.clip(action, min_action, max_action)
                    flat_action = jnp.ravel(action, order='C')
                    if noop[var]:
                        ranks = jnp.cumsum(flat_action < 0.5)
                        replace_mask = (flat_action < 0.5) & (ranks <= excess)
                    else:
                        ranks = jnp.cumsum(flat_action > 0.5)
                        replace_mask = (flat_action > 0.5) & (ranks <= excess)
                    flat_action = jnp.where(replace_mask, 0.5, flat_action)
                    action = jnp.reshape(flat_action, jnp.shape(action))
                    new_params[var] = jax_action_to_param(var, action, hyperparams)
                    excess = jnp.maximum(excess - jnp.count_nonzero(replace_mask), 0)
                else:
                    new_params[var] = action              
            return new_params, converged
        return _jax_wrapped_sogbofa_project


class JaxStraightLinePlan(JaxPlan):
    '''A straight line plan implementation in JAX'''
    
    def __init__(self, initializer: initializers.Initializer=initializers.normal(),
                 wrap_sigmoid: bool=True,
                 min_action_prob: float=1e-6,
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
    
    def __str__(self) -> str:
        bounds = '\n        '.join(
            map(lambda kv: f'{kv[0]}: {kv[1]}', self.bounds.items()))
        return (f'[INFO] policy hyper-parameters:\n'
                f'    initializer={self._initializer_base}\n'
                f'    constraint-sat strategy (simple):\n'
                f'        parsed_action_bounds =\n        {bounds}\n'
                f'        wrap_sigmoid         ={self._wrap_sigmoid}\n'
                f'        wrap_sigmoid_min_prob={self._min_action_prob}\n'
                f'        wrap_non_bool        ={self._wrap_non_bool}\n'
                f'    constraint-sat strategy (complex):\n'
                f'        wrap_softmax        ={self._wrap_softmax}\n'
                f'        use_new_projection  ={self._use_new_projection}\n'
                f'        max_projection_iters={self._max_constraint_iter}\n')
    
    def compile(self, compiled: JaxRDDLCompilerWithGrad,
                _bounds: Bounds,
                horizon: int,
                preprocessor: Optional[Preprocessor]=None) -> None:
        rddl = compiled.rddl
        
        # calculate the correct action box bounds
        shapes, bounds, bounds_safe, cond_lists = self._calculate_action_info(
            compiled, _bounds, horizon)
        self.bounds = bounds
        
        # get the noop action values
        noop = {var: (values[0] if isinstance(values, list) else values)
                for (var, values) in rddl.action_fluents.items()}
        bool_key = 'bool__'
        
        # ***********************************************************************
        # STRAIGHT-LINE PLAN
        #
        # ***********************************************************************
        
        # boolean actions are parameters wrapped by sigmoid to ensure [0, 1]:
        #
        #   action = sigmoid(weight * param)
        #
        # here weight is a hyper-parameter and param is the trainable policy parameter
        wrap_sigmoid = self._wrap_sigmoid
        bool_threshold = 0.0 if wrap_sigmoid else 0.5
        
        def _jax_bool_param_to_action(var, param, hyperparams):
            if wrap_sigmoid:
                weight = hyperparams[var]
                return stable_sigmoid(weight * param)
            else:
                return param 
        
        def _jax_bool_action_to_param(var, action, hyperparams):
            if wrap_sigmoid:
                weight = hyperparams[var]
                return jax.scipy.special.logit(action) / weight
            else:
                return action
        
        # the same technique could be applied to non-bool actions following Bueno et al.
        # this is disabled by default since the gradient projection trick seems to work
        # better, especially for one-sided bounds (-inf, B) or (B, +inf)
        wrap_non_bool = self._wrap_non_bool
        
        def _jax_non_bool_param_to_action(var, param, hyperparams):
            if wrap_non_bool:
                lower, upper = bounds_safe[var]
                mb, ml, mu, mn = [jnp.asarray(mask, dtype=compiled.REAL) 
                                  for mask in cond_lists[var]]       
                action = (
                    mb * (lower + (upper - lower) * stable_sigmoid(param)) + 
                    ml * (lower + jax.nn.softplus(param)) + 
                    mu * (upper - jax.nn.softplus(-param)) + 
                    mn * param
                )
            else:
                action = param
            return action
                
        # a different option to handle boolean action concurrency constraints with |A| = 1
        # is to use a softmax activation layer over pooled action parameters
        ranges = rddl.variable_ranges
        action_sizes = {var: np.prod(shape[1:], dtype=np.int64) 
                        for (var, shape) in shapes.items()
                        if ranges[var] == 'bool'}
        
        # given a softmax output, this simply unpacks the result of the softmax back into
        # the original action fluent dictionary
        def _jax_unstack_bool_from_softmax(output):
            actions = {}
            start = 0
            for (name, size) in action_sizes.items():
                action = output[..., start:start + size]
                action = jnp.reshape(action, shapes[name][1:])
                if noop[name]:
                    action = 1.0 - action
                actions[name] = action
                start = start + size
            return actions
                
        # the main subroutine to compute the trainable rddl actions from the trainable
        # parameters (TODO: implement one-hot for integer actions)        
        def _jax_wrapped_slp_predict_train(key, params, hyperparams, step, fls):
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
        self.train_policy = _jax_wrapped_slp_predict_train
        
        # the main subroutine to compute the test rddl actions from the trainable 
        # parameters: the difference here is that actions are converted to their required
        # types (i.e. bool, int, float)
        def _jax_wrapped_slp_predict_test(key, params, hyperparams, step, fls):
            actions = {}
            for (var, param) in params.items():
                action = jnp.asarray(param[step, ...], dtype=compiled.REAL)
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
                    if ranges[var] == 'int' or ranges[var] in rddl.enum_types:
                        action = jnp.asarray(jnp.round(action), dtype=compiled.INT)
                    actions[var] = action
            return actions
        self.test_policy = _jax_wrapped_slp_predict_test
        
        # ***********************************************************************
        # ACTION CONSTRAINT SATISFACTION
        #
        # ***********************************************************************
        
        # if the user wants min/max values for clipping boolean action parameters
        # this might be a good idea to avoid saturation of action-fluents since the
        # gradient could vanish as a result  
        min_action = self._min_action_prob
        max_action = 1.0 - min_action
        
        def _jax_project_bool_to_box(var, param, hyperparams):
            lower = _jax_bool_action_to_param(var, min_action, hyperparams)
            upper = _jax_bool_action_to_param(var, max_action, hyperparams)
            return jnp.clip(param, lower, upper)
        
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
            converged = jnp.array(True, dtype=jnp.bool_)
            return new_params, converged
        
        # enable constraint satisfaction subroutines during optimization 
        # if there are nontrivial concurrency constraints in the problem description 
        bool_action_count, allowed_actions = self._count_bool_actions(rddl)
        use_constraint_satisfaction = allowed_actions < bool_action_count        
        if compiled.print_warnings and use_constraint_satisfaction: 
            print(termcolor.colored(
                f'[INFO] Number of boolean actions {bool_action_count} '
                f'> max_nondef_actions {allowed_actions}: enabling projected gradient to '
                f'satisfy constraints on action-fluents.', 'dark_grey'
            ))
        
        # use a softmax output activation: only allow one action non-noop for now
        if use_constraint_satisfaction and self._wrap_softmax:
            if 1 < allowed_actions < bool_action_count:
                raise RDDLNotImplementedError(
                    f'SLPs with wrap_softmax currently '
                    f'do not support max-nondef-actions {allowed_actions} > 1.'
                )
            self.projection = _jax_wrapped_slp_project_to_box
            
        # use new gradient projection method
        elif use_constraint_satisfaction and self._use_new_projection:
            jax_project_fn = JaxSortingActionProjection().compile(
                ranges, noop, wrap_sigmoid, allowed_actions, bool_threshold, 
                _jax_project_bool_to_box
            )

            # clip actions to valid bounds and satisfy constraint on max actions
            def _jax_wrapped_slp_project_to_max_constraint(params, hyperparams):
                params, _ = _jax_wrapped_slp_project_to_box(params, hyperparams)
                return jax.vmap(jax_project_fn, in_axes=(0, None))(params, hyperparams)
            self.projection = _jax_wrapped_slp_project_to_max_constraint
        
        # use SOGBOFA projection method
        elif use_constraint_satisfaction and not self._use_new_projection:
            jax_project_fn = JaxSogbofaActionProjection().compile(
                ranges, noop, allowed_actions, self._max_constraint_iter, 
                _jax_bool_param_to_action, _jax_bool_action_to_param, 
                min_action, max_action, compiled.REAL
            )
            
            # clip actions to valid bounds and satisfy constraint on max actions
            def _jax_wrapped_slp_project_to_max_constraint(params, hyperparams):
                params, _ = _jax_wrapped_slp_project_to_box(params, hyperparams)
                return jax.vmap(jax_project_fn, in_axes=(0, None))(params, hyperparams)
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
        
        # use the user required initializer and project actions to feasible range
        def _jax_wrapped_slp_init(key, hyperparams, fls):
            params = {}
            for (var, shape) in shapes.items():
                if ranges[var] != 'bool' or not stack_bool_params: 
                    key, subkey = random.split(key)
                    param = init(key=subkey, shape=shape, dtype=compiled.REAL)
                    if ranges[var] == 'bool':
                        param = param + bool_threshold
                    params[var] = param
            if stack_bool_params:
                key, subkey = random.split(key)
                bool_shape = (horizon, bool_action_count)
                params[bool_key] = init(key=subkey, shape=bool_shape, dtype=compiled.REAL)
            params, _ = _jax_wrapped_slp_project_to_box(params, hyperparams)
            return params
        self.initializer = _jax_wrapped_slp_init
    
    @staticmethod
    @jax.jit
    def _guess_next_epoch(param):
        return jnp.append(param[1:, ...], param[-1:, ...], axis=0)

    def guess_next_epoch(self, params: Pytree) -> Pytree:
        # "progress" the plan one step forward and set last action to second-last
        return jax.tree_util.tree_map(JaxStraightLinePlan._guess_next_epoch, params)


class JaxDeepReactivePolicy(JaxPlan):
    '''A deep reactive policy network implementation in JAX.'''
    
    def __init__(self, topology: Optional[Sequence[int]]=None,
                 activation: Activation=jnp.tanh,
                 initializer: hk.initializers.Initializer=hk.initializers.VarianceScaling(scale=2.0),
                 normalize: bool=False,
                 normalize_per_layer: bool=False,
                 normalizer_kwargs: Optional[Kwargs]=None,
                 wrap_non_bool: bool=False) -> None:
        '''Creates a new deep reactive policy in JAX.
        
        :param neurons: sequence consisting of the number of neurons in each
        layer of the policy
        :param activation: function to apply after each layer of the policy
        :param initializer: weight initialization
        :param normalize: whether to apply layer norm to the inputs
        :param normalize_per_layer: whether to apply layer norm to each input
        individually (only active if normalize is True)
        :param normalizer_kwargs: if normalize is True, apply additional arguments
        to layer norm
        :param wrap_non_bool: whether to wrap real or int action fluent parameters
        with non-linearity (e.g. sigmoid or ELU) to satisfy box constraints
        '''
        super(JaxDeepReactivePolicy, self).__init__()
        
        if topology is None:
            topology = [128, 64]
        self._topology = topology
        self._activations = [activation for _ in topology]
        self._initializer_base = initializer
        self._initializer = initializer
        self._normalize = normalize
        self._normalize_per_layer = normalize_per_layer
        if normalizer_kwargs is None:
            normalizer_kwargs = {'create_offset': True, 'create_scale': True}
        self._normalizer_kwargs = normalizer_kwargs
        self._wrap_non_bool = wrap_non_bool
    
    def __str__(self) -> str:
        bounds = '\n        '.join(
            map(lambda kv: f'{kv[0]}: {kv[1]}', self.bounds.items()))
        return (f'[INFO] policy hyper-parameters:\n'
                f'    topology     ={self._topology}\n'
                f'    activation_fn={self._activations[0].__name__}\n'
                f'    initializer  ={type(self._initializer_base).__name__}\n'
                f'    input norm:\n'
                f'        apply_input_norm    ={self._normalize}\n'
                f'        input_norm_layerwise={self._normalize_per_layer}\n'
                f'        input_norm_args     ={self._normalizer_kwargs}\n'
                f'    constraint-sat strategy:\n'
                f'        parsed_action_bounds=\n        {bounds}\n'
                f'        wrap_non_bool       ={self._wrap_non_bool}\n')
        
    def compile(self, compiled: JaxRDDLCompilerWithGrad,
                _bounds: Bounds,
                horizon: int,
                preprocessor: Optional[Preprocessor]=None) -> None:
        rddl = compiled.rddl
        
        # calculate the correct action box bounds
        shapes, bounds, bounds_safe, cond_lists = self._calculate_action_info(
            compiled, _bounds, horizon)
        shapes = {var: value[1:] for (var, value) in shapes.items()}
        self.bounds = bounds
        
        # enable constraint satisfaction subroutines during optimization 
        # if there are nontrivial concurrency constraints in the problem description
        # only handles the case where |A| = 1 for now, as there is no way to do projection
        # currently (TODO: fix this)
        bool_action_count, allowed_actions = self._count_bool_actions(rddl)
        if 1 < allowed_actions < bool_action_count:
            raise RDDLNotImplementedError(
                f'DRPs currently do not support max-nondef-actions {allowed_actions} > 1.'
            )
        use_constraint_satisfaction = allowed_actions < bool_action_count
        
        # get the noop action values
        noop = {var: (values[0] if isinstance(values, list) else values)
                for (var, values) in rddl.action_fluents.items()}                   
        bool_key = 'bool__'
        
        # ***********************************************************************
        # POLICY NETWORK PREDICTION
        #
        # ***********************************************************************
            
        # compute the correct shapes of the output layers based on the action-fluent shape
        ranges = rddl.variable_ranges
        normalize = self._normalize
        normalize_per_layer = self._normalize_per_layer
        wrap_non_bool = self._wrap_non_bool
        init = self._initializer
        layers = list(enumerate(zip(self._topology, self._activations)))
        layer_sizes = {var: np.prod(shape, dtype=np.int64) 
                       for (var, shape) in shapes.items()}
        layer_names = {var: f'output_{var}'.replace('-', '_') for var in shapes}
        
        # inputs for the policy network are states for fully observed and obs for POMDPs
        if rddl.observ_fluents:
            observed_vars = rddl.observ_fluents
        else:
            observed_vars = rddl.state_fluents
        input_names = {var: f'{var}'.replace('-', '_') for var in observed_vars}
        
        # catch if input norm is applied to size 1 tensor:
        # this leads to incorrect behavior as the input is always "1"
        if normalize:
            non_bool_dims = 0
            for (var, values) in observed_vars.items():
                if ranges[var] != 'bool':
                    value_size = np.size(values)
                    if normalize_per_layer and value_size == 1:
                        if compiled.print_warnings:
                            print(termcolor.colored(
                                f'[WARN] Cannot apply layer norm to state-fluent <{var}> '
                                f'of size 1: setting normalize_per_layer = False.', 'yellow'
                            ))
                        normalize_per_layer = False
                    non_bool_dims += value_size
            if not normalize_per_layer and non_bool_dims == 1:
                if compiled.print_warnings:
                    print(termcolor.colored(
                        '[WARN] Cannot apply layer norm to state-fluents of total size 1: '
                        'setting normalize = False.', 'yellow'
                    ))
                normalize = False
        
        # convert fluents dictionary into a state vector to feed to the MLP
        def _jax_wrapped_policy_input(fls, hyperparams):

            # optional state preprocessing
            if preprocessor is not None:
                stats = hyperparams[preprocessor.HYPERPARAMS_KEY]
                fls = preprocessor.transform(fls, stats)
            
            # concatenate all state variables into a single vector
            # optionally apply layer norm to each input tensor
            states_bool, states_non_bool = [], []
            non_bool_dims = 0
            for (var, value) in fls.items():
                if var in observed_vars:
                    state = jnp.ravel(value, order='C')
                    if ranges[var] == 'bool':
                        states_bool.append(state)
                    else:
                        if normalize and normalize_per_layer:
                            normalizer = hk.LayerNorm(
                                axis=-1, 
                                param_axis=-1,
                                name=f'input_norm_{input_names[var]}',
                                **self._normalizer_kwargs
                            )
                            state = normalizer(state)
                        states_non_bool.append(state)
                        non_bool_dims = non_bool_dims + state.size
            state = jnp.concatenate(states_non_bool + states_bool)
            
            # optionally perform layer normalization on the non-bool inputs
            if normalize and not normalize_per_layer and non_bool_dims:
                normalizer = hk.LayerNorm(
                    axis=-1, 
                    param_axis=-1, 
                    name='input_norm',
                    **self._normalizer_kwargs
                )
                normalized = normalizer(state[:non_bool_dims])
                state = state.at[:non_bool_dims].set(normalized)
            return state
            
        # predict actions from the policy network for current state
        def _jax_wrapped_policy_network_predict(fls, hyperparams):
            state = _jax_wrapped_policy_input(fls, hyperparams)
            
            # feed state vector through hidden layers
            hidden = state
            for (i, (num_neuron, activation)) in layers:
                linear = hk.Linear(num_neuron, name=f'hidden_{i}', w_init=init)
                hidden = activation(linear(hidden))
            
            # each output is a linear layer reshaped to original lifted shape
            actions = {}
            for (var, size) in layer_sizes.items():
                linear = hk.Linear(size, name=layer_names[var], w_init=init)
                reshape = hk.Reshape(output_shape=shapes[var], 
                                     preserve_dims=-1,
                                     name=f'reshape_{layer_names[var]}')
                output = reshape(linear(hidden))
                if not shapes[var]:
                    output = jnp.squeeze(output)
                
                # project action output to valid box constraints following Bueno et. al.
                if ranges[var] == 'bool':
                    if not use_constraint_satisfaction:
                        actions[var] = stable_sigmoid(output)
                else:
                    if wrap_non_bool:
                        lower, upper = bounds_safe[var]
                        mb, ml, mu, mn = [jnp.asarray(mask, dtype=compiled.REAL) 
                                          for mask in cond_lists[var]]       
                        actions[var] = (
                            mb * (lower + (upper - lower) * stable_sigmoid(output)) + 
                            ml * (lower + jax.nn.softplus(output)) + 
                            mu * (upper - jax.nn.softplus(-output)) + 
                            mn * output
                        )
                    else:
                        actions[var] = output
            
            # for constraint satisfaction wrap bool actions with softmax:
            # this only works when |A| = 1
            if use_constraint_satisfaction:
                linear = hk.Linear(bool_action_count, name='output_bool', w_init=init)
                actions[bool_key] = jax.nn.softmax(linear(hidden))
            return actions
        
        # we need pure JAX functions for the policy network prediction
        predict_fn = hk.transform(_jax_wrapped_policy_network_predict)
        predict_fn = hk.without_apply_rng(predict_fn)            
        
        # given a softmax output, this simply unpacks the result of the softmax back into
        # the original action fluent dictionary
        def _jax_unstack_bool_from_softmax(output):
            actions = {}
            start = 0
            for (name, size) in layer_sizes.items():
                if ranges[name] == 'bool':
                    action = output[..., start:start + size]
                    action = jnp.reshape(action, shapes[name])
                    if noop[name]:
                        action = 1.0 - action
                    actions[name] = action
                    start = start + size
            return actions
        
        # the main subroutine to compute the trainable rddl actions from the trainable
        # parameters and the current state/obs dictionary       
        def _jax_wrapped_drp_predict_train(key, params, hyperparams, step, fls):
            actions = predict_fn.apply(params, fls, hyperparams)
            if not wrap_non_bool:
                for (var, action) in actions.items():
                    if var != bool_key and ranges[var] != 'bool':
                        actions[var] = jnp.clip(action, *bounds[var])
            if use_constraint_satisfaction:
                bool_actions = _jax_unstack_bool_from_softmax(actions[bool_key])
                actions.update(bool_actions)
                del actions[bool_key]
            return actions
        self.train_policy = _jax_wrapped_drp_predict_train
        
        # the main subroutine to compute the test rddl actions from the trainable 
        # parameters and state/obs dict: the difference here is that actions are converted 
        # to their required types (i.e. bool, int, float)
        def _jax_wrapped_drp_predict_test(key, params, hyperparams, step, fls):
            actions = _jax_wrapped_drp_predict_train(key, params, hyperparams, step, fls)
            new_actions = {}
            for (var, action) in actions.items():
                prange = ranges[var]
                if prange == 'bool':
                    new_action = action > 0.5
                elif prange == 'int' or prange in rddl.enum_types:
                    action = jnp.clip(action, *bounds[var])
                    new_action = jnp.asarray(jnp.round(action), dtype=compiled.INT)
                else:
                    new_action = jnp.clip(action, *bounds[var])
                new_actions[var] = new_action
            return new_actions
        self.test_policy = _jax_wrapped_drp_predict_test
        
        # ***********************************************************************
        # ACTION CONSTRAINT SATISFACTION
        #
        # ***********************************************************************
        
        # no projection applied since the actions are already constrained
        def _jax_wrapped_drp_no_projection(params, hyperparams):
            converged = jnp.array(True, dtype=jnp.bool_)
            return params, converged
        self.projection = _jax_wrapped_drp_no_projection
    
        # ***********************************************************************
        # POLICY NETWORK INITIALIZATION
        #
        # ***********************************************************************
        
        # initialize policy parameters according to user-desired weight initializer
        def _jax_wrapped_drp_init(key, hyperparams, fls):
            obs_vars = {var: value[0, ...] 
                        for (var, value) in fls.items()
                        if var in observed_vars}
            return predict_fn.init(key, obs_vars, hyperparams)
        self.initializer = _jax_wrapped_drp_init
        
    def guess_next_epoch(self, params: Pytree) -> Pytree:
        # this is easy: just warm-start from the previously obtained policy
        return params
    
    
# ***********************************************************************
# SUPPORTING FUNCTIONS
# 
# - smoothed mean calculation
# - planner status
# - stopping criteria
#
# ***********************************************************************


class RollingMean:
    '''Maintains the rolling mean of a stream of real-valued observations.'''
    
    def __init__(self, window_size: int) -> None:
        self._window_size = window_size
        self._memory = deque(maxlen=window_size)
        self._total = 0
    
    def update(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        memory = self._memory
        self._total = self._total + x
        if len(memory) == self._window_size:
            self._total = self._total - memory.popleft()
        memory.append(x)
        return self._total / len(memory)


class JaxPlannerStatus(Enum):
    '''Represents the status of a policy update from the JAX planner, 
    including whether the update resulted in nan gradient, 
    whether progress was made, budget was reached, or other information that
    can be used to monitor and act based on the planner's progress.'''
    
    NORMAL = 0
    STOPPING_RULE_REACHED = 1
    NO_PROGRESS = 2
    PRECONDITION_POSSIBLY_UNSATISFIED = 3
    INVALID_GRADIENT = 4
    TIME_BUDGET_REACHED = 5
    ITER_BUDGET_REACHED = 6
    
    def is_terminal(self) -> bool:
        return self.value == 1 or self.value >= 4


class JaxPlannerStoppingRule(metaclass=ABCMeta):
    '''The base class of all planner stopping rules.'''
    
    @abstractmethod
    def reset(self) -> None:
        pass
    
    @abstractmethod
    def monitor(self, callback: Dict[str, Any]) -> bool:
        pass
    

class NoImprovementStoppingRule(JaxPlannerStoppingRule):
    '''Stopping rule based on no improvement for a fixed number of iterations.'''
    
    def __init__(self, patience: int) -> None:
        self.patience = patience
    
    def reset(self) -> None:
        self.callback = None
        self.iters_since_last_update = 0
        
    def monitor(self, callback: Dict[str, Any]) -> bool:
        if self.callback is None or callback['best_return'] > self.callback['best_return']:
            self.callback = callback
            self.iters_since_last_update = 0
        else:
            self.iters_since_last_update += 1
        return self.iters_since_last_update >= self.patience
    
    def __str__(self) -> str:
        return f'No improvement for {self.patience} iterations'
        

# ***********************************************************************
# PARAMETER EXPLORING POLICY GRADIENTS (PGPE)
# 
# - simple Gaussian PGPE
#
# ***********************************************************************


class PGPE(metaclass=ABCMeta):
    """Base class for all PGPE strategies."""

    def __init__(self) -> None:
        self._initializer = None
        self._update = None

    @property
    def initialize(self) -> Callable:
        return self._initializer

    @property
    def update(self) -> Callable:
        return self._update

    @abstractmethod
    def compile(self, loss_fn: Callable, projection: Callable, real_dtype: Type,
                print_warnings: bool, parallel_updates: int=1) -> None:
        pass


class GaussianPGPE(PGPE):
    '''PGPE with a Gaussian parameter distribution.'''

    def __init__(self, batch_size: int=1, 
                 init_sigma: float=1.0,
                 sigma_range: Tuple[float, float]=(1e-5, 1e5),
                 scale_reward: bool=True,
                 min_reward_scale: float=1e-5,
                 super_symmetric: bool=True,
                 super_symmetric_accurate: bool=True,
                 optimizer: Callable[..., optax.GradientTransformation]=optax.adam,
                 optimizer_kwargs_mu: Optional[Kwargs]=None,
                 optimizer_kwargs_sigma: Optional[Kwargs]=None,
                 start_entropy_coeff: float=1e-3,
                 end_entropy_coeff: float=1e-8,
                 max_kl_update: Optional[float]=None) -> None:
        '''Creates a new Gaussian PGPE planner.
        
        :param batch_size: how many policy parameters to sample per optimization step
        :param init_sigma: initial standard deviation of Gaussian
        :param sigma_range: bounds to constrain standard deviation
        :param scale_reward: whether to apply reward scaling as in the paper
        :param min_reward_scale: minimum reward scaling to avoid underflow
        :param super_symmetric: whether to use super-symmetric sampling as in the paper
        :param super_symmetric_accurate: whether to use the accurate formula for super-
        symmetric sampling or the simplified but biased formula
        :param optimizer: a factory for an optax SGD algorithm
        :param optimizer_kwargs_mu: a dictionary of parameters to pass to the SGD
        factory for the mean optimizer
        :param optimizer_kwargs_sigma: a dictionary of parameters to pass to the SGD
        factory for the standard deviation optimizer
        :param start_entropy_coeff: starting entropy regularization coeffient for Gaussian
        :param end_entropy_coeff: ending entropy regularization coeffient for Gaussian
        :param max_kl_update: bound on kl-divergence between parameter updates
        '''
        super().__init__()

        self.batch_size = batch_size
        self.init_sigma = init_sigma
        self.sigma_range = sigma_range
        self.scale_reward = scale_reward
        self.min_reward_scale = min_reward_scale
        self.super_symmetric = super_symmetric
        self.super_symmetric_accurate = super_symmetric_accurate

        # entropy regularization penalty is decayed exponentially between these values
        self.start_entropy_coeff = start_entropy_coeff
        self.end_entropy_coeff = end_entropy_coeff
        
        # set optimizers
        if optimizer_kwargs_mu is None:
            optimizer_kwargs_mu = {'learning_rate': 0.1}
        self.optimizer_kwargs_mu = optimizer_kwargs_mu
        if optimizer_kwargs_sigma is None:
            optimizer_kwargs_sigma = {'learning_rate': 0.1}
        self.optimizer_kwargs_sigma = optimizer_kwargs_sigma
        self.optimizer_name = optimizer
        try:
            mu_optimizer = optax.inject_hyperparams(optimizer)(**optimizer_kwargs_mu)
            sigma_optimizer = optax.inject_hyperparams(optimizer)(**optimizer_kwargs_sigma)
        except Exception as _:
            print(termcolor.colored(
                '[WARN] Could not inject hyperparameters into PGPE optimizer: '
                'kl-divergence constraint will be disabled.', 'yellow'
            ))
            mu_optimizer = optimizer(**optimizer_kwargs_mu)   
            sigma_optimizer = optimizer(**optimizer_kwargs_sigma) 
            max_kl_update = None
        self.optimizers = (mu_optimizer, sigma_optimizer)
        self.max_kl = max_kl_update
    
    def __str__(self) -> str:
        return (f'[INFO] PGPE hyper-parameters:\n'
                f'    method             ={self.__class__.__name__}\n'
                f'    batch_size         ={self.batch_size}\n'
                f'    init_sigma         ={self.init_sigma}\n'
                f'    sigma_range        ={self.sigma_range}\n'
                f'    scale_reward       ={self.scale_reward}\n'
                f'    min_reward_scale   ={self.min_reward_scale}\n'
                f'    super_symmetric    ={self.super_symmetric}\n'
                f'        accurate       ={self.super_symmetric_accurate}\n'
                f'    optimizer          ={self.optimizer_name}\n'
                f'    optimizer_kwargs:\n'
                f'        mu   ={self.optimizer_kwargs_mu}\n'
                f'        sigma={self.optimizer_kwargs_sigma}\n'
                f'    start_entropy_coeff={self.start_entropy_coeff}\n'
                f'    end_entropy_coeff  ={self.end_entropy_coeff}\n'
                f'    max_kl_update      ={self.max_kl}\n'
        )

    def compile(self, loss_fn: Callable, 
                projection: Callable, 
                real_dtype: Type,
                print_warnings: bool,
                parallel_updates: int=1) -> None:
        sigma0 = self.init_sigma
        sigma_lo, sigma_hi = self.sigma_range
        scale_reward = self.scale_reward
        min_reward_scale = self.min_reward_scale
        super_symmetric = self.super_symmetric
        super_symmetric_accurate = self.super_symmetric_accurate
        batch_size = self.batch_size
        mu_optimizer, sigma_optimizer = self.optimizers
        max_kl = self.max_kl
        
        # entropy regularization penalty is decayed exponentially by elapsed budget
        # this uses the optimizer progress (as percentage) to move the decay
        start_entropy_coeff = self.start_entropy_coeff
        if start_entropy_coeff == 0:
            entropy_coeff_decay = 0
        else:
            entropy_coeff_decay = (self.end_entropy_coeff / start_entropy_coeff) ** 0.01
        
        # ***********************************************************************
        # INITIALIZATION OF POLICY
        #
        # ***********************************************************************
        
        # use the default initializer for the (mean, sigma) parameters
        # these parameters define the sampling distribution over policy parameters
        def _jax_wrapped_pgpe_init(key, policy_params):
            mu = policy_params
            sigma = jax.tree_util.tree_map(partial(jnp.full_like, fill_value=sigma0), mu)
            pgpe_params = (mu, sigma)
            pgpe_opt_state = (mu_optimizer.init(mu), sigma_optimizer.init(sigma))
            r_max = -jnp.inf
            return pgpe_params, pgpe_opt_state, r_max
        
        # for parallel policy update, initialize multiple indepdendent (mean, sigma)
        # gaussians that will be optimized in parallel
        def _jax_wrapped_batched_pgpe_init(key, policy_params):
            keys = random.split(key, num=parallel_updates)
            return jax.vmap(_jax_wrapped_pgpe_init, in_axes=0)(keys, policy_params)
    
        self._initializer = jax.jit(_jax_wrapped_batched_pgpe_init)

        # ***********************************************************************
        # PARAMETER SAMPLING FUNCTIONS
        #
        # ***********************************************************************

        # sample from i.i.d. Normal(0, sigma)
        def _jax_wrapped_mu_noise(key, sigma):
            return sigma * random.normal(key, shape=jnp.shape(sigma), dtype=real_dtype)

        # this samples a noise variable epsilon* from epsilon with the N(0, 1) density
        # according to super-symmetric sampling paper:
        # the paper presents a more accurate formula which is used by default
        def _jax_wrapped_epsilon_star(sigma, epsilon):
            phi = 0.67449 * sigma
            a = (sigma - jnp.abs(epsilon)) / sigma

            # more accurate formula
            if super_symmetric_accurate:
                aa = jnp.abs(a)
                atol = 1e-10
                c1, c2, c3 = -0.06655, -0.9706, 0.124
                term_neg_log = c1 * (aa * aa - 1.0) / jnp.log(aa + atol) + c2
                term_pos_log = 1.0 - c3 * jnp.log1p(-aa ** 3 + atol)
                epsilon_star = jnp.sign(epsilon) * phi * jnp.exp(
                    aa * jnp.where(a <= 0, term_neg_log, term_pos_log))
            
            # less accurate and simple formula
            else:
                epsilon_star = jnp.sign(epsilon) * phi * jnp.exp(a)
            return epsilon_star

        # implements baseline-free super-symmetric sampling to generate 4 trajectories
        # this type of sampling removes the need for the baseline completely
        def _jax_wrapped_sample_params(key, mu, sigma):

            # this samples the basic two policy parameters from Gaussian(mean, sigma)
            # using the control variates
            treedef = jax.tree_util.tree_structure(sigma)
            keys = random.split(key, num=treedef.num_leaves)
            keys_pytree = jax.tree_util.tree_unflatten(treedef=treedef, leaves=keys)
            epsilon = jax.tree_util.tree_map(_jax_wrapped_mu_noise, keys_pytree, sigma)
            p1 = jax.tree_util.tree_map(jnp.add, mu, epsilon)
            p2 = jax.tree_util.tree_map(jnp.subtract, mu, epsilon)
            
            # sumer-symmetric sampling removes the need for a baseline but requires
            # two additional policies to be sampled
            if super_symmetric:
                epsilon_star = jax.tree_util.tree_map(
                    _jax_wrapped_epsilon_star, sigma, epsilon)     
                p3 = jax.tree_util.tree_map(jnp.add, mu, epsilon_star)
                p4 = jax.tree_util.tree_map(jnp.subtract, mu, epsilon_star)
            else:
                epsilon_star, p3, p4 = epsilon, p1, p2
            return p1, p2, p3, p4, epsilon, epsilon_star
                        
        # ***********************************************************************
        # POLICY GRADIENT CALCULATION
        #
        # ***********************************************************************

        # gradient with respect to mean
        def _jax_wrapped_mu_grad(epsilon, epsilon_star, r1, r2, r3, r4, m):

            # for super symmetric sampling
            if super_symmetric:
                if scale_reward:
                    scale1 = jnp.maximum(min_reward_scale, m - (r1 + r2) / 2)
                    scale2 = jnp.maximum(min_reward_scale, m - (r3 + r4) / 2)
                else:
                    scale1 = scale2 = 1.0
                r_mu1 = (r1 - r2) / (2 * scale1)
                r_mu2 = (r3 - r4) / (2 * scale2)
                grad = -(r_mu1 * epsilon + r_mu2 * epsilon_star)
            
            # for the basic pgpe
            else:
                if scale_reward:
                    scale = jnp.maximum(min_reward_scale, m - (r1 + r2) / 2)
                else:
                    scale = 1.0
                r_mu = (r1 - r2) / (2 * scale)
                grad = -r_mu * epsilon
            return grad
        
        #  gradient with respect to std. deviation
        def _jax_wrapped_sigma_grad(epsilon, epsilon_star, sigma, r1, r2, r3, r4, m, ent):

            # for super symmetric sampling
            if super_symmetric:
                mask = r1 + r2 >= r3 + r4
                epsilon_tau = mask * epsilon + (1 - mask) * epsilon_star
                s = jnp.square(epsilon_tau) / sigma - sigma
                if scale_reward:
                    scale = jnp.maximum(min_reward_scale, m - (r1 + r2 + r3 + r4) / 4)
                else:
                    scale = 1.0
                r_sigma = ((r1 + r2) - (r3 + r4)) / (4 * scale)
            
            # for basic pgpe
            else:
                s = jnp.square(epsilon) / sigma - sigma
                if scale_reward:
                    scale = jnp.maximum(min_reward_scale, jnp.abs(m))
                else:
                    scale = 1.0
                r_sigma = (r1 + r2) / (2 * scale)

            return -(r_sigma * s + ent / sigma)
            
        # calculate the policy gradients
        def _jax_wrapped_pgpe_grad(key, mu, sigma, r_max, ent,
                                   policy_hyperparams, fls, nfls, 
                                   model_params, critic_params):
            
            # basic pgpe sampling with return estimation
            key, subkey = random.split(key)
            p1, p2, p3, p4, epsilon, epsilon_star = _jax_wrapped_sample_params(
                key, mu, sigma)
            r1 = -loss_fn(subkey, p1, policy_hyperparams, fls, nfls, model_params, critic_params)[0]
            r2 = -loss_fn(subkey, p2, policy_hyperparams, fls, nfls, model_params, critic_params)[0]

            # do a return normalization for optimizer stability
            r_max = jnp.maximum(r_max, r1)
            r_max = jnp.maximum(r_max, r2)            

            # super symmetric sampling requires two more trajectories and their returns
            if super_symmetric:
                r3 = -loss_fn(subkey, p3, policy_hyperparams, fls, nfls, model_params, critic_params)[0]
                r4 = -loss_fn(subkey, p4, policy_hyperparams, fls, nfls, model_params, critic_params)[0]
                r_max = jnp.maximum(r_max, r3)
                r_max = jnp.maximum(r_max, r4)       
            else:
                r3, r4 = r1, r2     

            # calculate gradient with respect to the mean
            grad_mu = jax.tree_util.tree_map(
                partial(_jax_wrapped_mu_grad, r1=r1, r2=r2, r3=r3, r4=r4, m=r_max), 
                epsilon, epsilon_star
            ) 

            # calculate gradient with respect to the sigma
            grad_sigma = jax.tree_util.tree_map(
                partial(_jax_wrapped_sigma_grad, 
                        r1=r1, r2=r2, r3=r3, r4=r4, m=r_max, ent=ent), 
                epsilon, epsilon_star, sigma
            )
            return grad_mu, grad_sigma, r_max

        # calculate the policy gradients with batching on the first dimension
        def _jax_wrapped_pgpe_grad_batched(key, pgpe_params, r_max, ent,
                                           policy_hyperparams, fls, nfls, 
                                           model_params, critic_params):
            mu, sigma = pgpe_params

            # no batching required
            if batch_size == 1:
                mu_grad, sigma_grad, new_r_max = _jax_wrapped_pgpe_grad(
                    key, mu, sigma, r_max, ent, policy_hyperparams, fls, nfls, 
                    model_params, critic_params)
            
            # for batching need to handle how meta-gradients of mean, sigma are aggregated
            else:
                # do the batched calculation of mean and sigma gradients
                keys = random.split(key, num=batch_size)
                mu_grads, sigma_grads, r_maxs = jax.vmap(
                    _jax_wrapped_pgpe_grad, 
                    in_axes=(0, None, None, None, None, None, None, None, None, None)
                )(keys, mu, sigma, r_max, ent, policy_hyperparams, fls, nfls, 
                  model_params, critic_params)

                # calculate the average gradient for aggregation
                mu_grad, sigma_grad = jax.tree_util.tree_map(
                    partial(jnp.mean, axis=0), (mu_grads, sigma_grads))
                new_r_max = jnp.max(r_maxs)

            return mu_grad, sigma_grad, new_r_max
        
        # ***********************************************************************
        # PARAMETER UPDATE
        #
        # ***********************************************************************

        # estimate KL divergence between two updates
        def _jax_wrapped_pgpe_kl_term(mu, sigma, old_mu, old_sigma):
            return 0.5 * jnp.sum(2 * jnp.log(sigma / old_sigma) + 
                                 jnp.square(old_sigma / sigma) + 
                                 jnp.square((mu - old_mu) / sigma) - 1)
        
        # update mean and std. deviation with a gradient step
        def _jax_wrapped_pgpe_update_helper(mu, sigma, mu_grad, sigma_grad, 
                                            mu_state, sigma_state):
            mu_updates, new_mu_state = mu_optimizer.update(mu_grad, mu_state, params=mu) 
            sigma_updates, new_sigma_state = sigma_optimizer.update(
                sigma_grad, sigma_state, params=sigma) 
            new_mu = optax.apply_updates(mu, mu_updates)
            new_sigma = optax.apply_updates(sigma, sigma_updates)
            new_sigma = jax.tree_util.tree_map(
                partial(jnp.clip, min=sigma_lo, max=sigma_hi), new_sigma)
            return new_mu, new_sigma, new_mu_state, new_sigma_state

        def _jax_wrapped_pgpe_update(key, pgpe_params, r_max, progress,
                                     policy_hyperparams, fls, nfls, 
                                     model_params, critic_params, pgpe_opt_state):
            # regular update for pgpe
            mu, sigma = pgpe_params
            mu_state, sigma_state = pgpe_opt_state
            ent = start_entropy_coeff * jnp.power(entropy_coeff_decay, progress)
            mu_grad, sigma_grad, new_r_max = _jax_wrapped_pgpe_grad_batched(
                key, pgpe_params, r_max, ent, policy_hyperparams, fls, nfls, 
                model_params, critic_params)
            new_mu, new_sigma, new_mu_state, new_sigma_state = _jax_wrapped_pgpe_update_helper(
                mu, sigma, mu_grad, sigma_grad, mu_state, sigma_state)
            
            # respect KL divergence contraint with old parameters
            if max_kl is not None:
                old_mu_lr = new_mu_state.hyperparams['learning_rate']
                old_sigma_lr = new_sigma_state.hyperparams['learning_rate']
                kl_terms = jax.tree_util.tree_map(
                    _jax_wrapped_pgpe_kl_term, new_mu, new_sigma, mu, sigma)
                total_kl = jax.tree_util.tree_reduce(jnp.add, kl_terms)
                kl_reduction = jnp.minimum(1.0, jnp.sqrt(max_kl / total_kl))
                mu_state.hyperparams['learning_rate'] = old_mu_lr * kl_reduction
                sigma_state.hyperparams['learning_rate'] = old_sigma_lr * kl_reduction
                new_mu, new_sigma, new_mu_state, new_sigma_state = _jax_wrapped_pgpe_update_helper(
                    mu, sigma, mu_grad, sigma_grad, mu_state, sigma_state)
                new_mu_state.hyperparams['learning_rate'] = old_mu_lr
                new_sigma_state.hyperparams['learning_rate'] = old_sigma_lr

            # apply projection step to the sampled policy
            new_mu, converged = projection(new_mu, policy_hyperparams)

            new_pgpe_params = (new_mu, new_sigma)
            new_pgpe_opt_state = (new_mu_state, new_sigma_state)
            policy_params = new_mu
            return new_pgpe_params, new_r_max, new_pgpe_opt_state, policy_params, converged

        # for parallel policy update
        def _jax_wrapped_batched_pgpe_updates(key, pgpe_params, r_max, progress,
                                              policy_hyperparams, fls, nfls, 
                                              model_params, critic_params, pgpe_opt_state):
            keys = random.split(key, num=parallel_updates)
            return jax.vmap(
                _jax_wrapped_pgpe_update, 
                in_axes=(0, 0, 0, None, None, None, None, 0, None, 0)
            )(keys, pgpe_params, r_max, progress, policy_hyperparams, fls, nfls, 
              model_params, critic_params, pgpe_opt_state)
        
        self._update = jax.jit(_jax_wrapped_batched_pgpe_updates)


# ***********************************************************************
# ALL VERSIONS OF RISK FUNCTIONS
# 
# Based on the original paper "A Distributional Framework for Risk-Sensitive 
# End-to-End Planning in Continuous MDPs" by Patton et al., AAAI 2022.
#
# Original risk functions:
# - entropic utility
# - mean-variance
# - mean-semideviation
# - conditional value at risk with straight-through gradient trick
#
# ***********************************************************************


@jax.jit
def entropic_utility(returns: jnp.ndarray, beta: float) -> float:
    return (-1.0 / beta) * jax.scipy.special.logsumexp(
        -beta * returns, b=1.0 / returns.size)


@jax.jit
def mean_variance_utility(returns: jnp.ndarray, beta: float) -> float:
    return jnp.mean(returns) - 0.5 * beta * jnp.var(returns)


@jax.jit
def mean_deviation_utility(returns: jnp.ndarray, beta: float) -> float:
    return jnp.mean(returns) - 0.5 * beta * jnp.std(returns)


@jax.jit
def mean_semideviation_utility(returns: jnp.ndarray, beta: float) -> float:
    mu = jnp.mean(returns)
    msd = jnp.sqrt(jnp.mean(jnp.square(jnp.minimum(0.0, returns - mu))))
    return mu - 0.5 * beta * msd


@jax.jit
def mean_semivariance_utility(returns: jnp.ndarray, beta: float) -> float:
    mu = jnp.mean(returns)
    msv = jnp.mean(jnp.square(jnp.minimum(0.0, returns - mu)))
    return mu - 0.5 * beta * msv


@jax.jit
def sharpe_utility(returns: jnp.ndarray, risk_free: float) -> float:
    return (jnp.mean(returns) - risk_free) / (jnp.std(returns) + 1e-10)


@jax.jit
def var_utility(returns: jnp.ndarray, alpha: float) -> float:
    return jnp.percentile(returns, q=100 * alpha)


@jax.jit
def cvar_utility(returns: jnp.ndarray, alpha: float) -> float:
    var = jnp.percentile(returns, q=100 * alpha)
    mask = returns <= var
    return jnp.sum(returns * mask) / jnp.maximum(1, jnp.count_nonzero(mask))


# set of all currently valid built-in utility functions
UTILITY_LOOKUP = {
    'mean': jnp.mean,
    'mean_var': mean_variance_utility,
    'mean_std': mean_deviation_utility,
    'mean_semivar': mean_semivariance_utility,
    'mean_semidev': mean_semideviation_utility,
    'sharpe': sharpe_utility,
    'entropic': entropic_utility,
    'exponential': entropic_utility,
    'var': var_utility,
    'cvar': cvar_utility
}


# ***********************************************************************
# ALL VERSIONS OF JAX PLANNER
# 
# - simple gradient descent based planner
# 
# ***********************************************************************


@jax.jit
def pytree_at(tree: Pytree, i: int) -> Pytree:
    return jax.tree_util.tree_map(lambda x: x[i], tree)


class JaxBackpropPlanner:
    '''A class for optimizing an action sequence in the given RDDL MDP using 
    gradient descent.'''
    
    def __init__(self, rddl: RDDLLiftedModel,
                 plan: JaxPlan,
                 batch_size_train: int=32,
                 batch_size_test: Optional[int]=None,
                 rollout_horizon: Optional[int]=None,
                 parallel_updates: int=1,
                 action_bounds: Optional[Bounds]=None,
                 optimizer: Callable[..., optax.GradientTransformation]=optax.rmsprop,
                 optimizer_kwargs: Optional[Kwargs]=None,
                 clip_grad: Optional[float]=None,
                 line_search_kwargs: Optional[Kwargs]=None,
                 noise_kwargs: Optional[Kwargs]=None,
                 ema_decay: Optional[float]=None,
                 pgpe: Optional[PGPE]=GaussianPGPE(),
                 compiler: JaxRDDLCompilerWithGrad=DefaultJaxRDDLCompilerWithGrad,
                 compiler_kwargs: Optional[Kwargs]=None,
                 use_symlog_reward: bool=False,
                 utility: Union[Callable[[jnp.ndarray], float], str]='mean',
                 utility_kwargs: Optional[Kwargs]=None,
                 critic: Optional[Callable]=None,
                 logger: Optional[Logger]=None,
                 dashboard: Optional[Any]=None,
                 dashboard_viz: Optional[Any]=None,
                 preprocessor: Optional[Preprocessor]=None,
                 python_functions: Optional[Dict[str, Callable]]=None) -> None:
        '''Creates a new gradient-based algorithm for optimizing action sequences
        (plan) in the given RDDL. Some operations will be converted to their
        differentiable counterparts; the specific operations can be customized
        by providing a tailored compiler instance.
        
        :param rddl: the RDDL domain to optimize
        :param plan: the policy/plan representation to optimize
        :param batch_size_train: how many rollouts to perform per optimization 
        step
        :param batch_size_test: how many rollouts to use to test the plan at each
        optimization step
        :param rollout_horizon: lookahead planning horizon: None uses the env horizon
        :param parallel_updates: how many optimizers to run independently in parallel
        :param action_bounds: box constraints on actions
        :param optimizer: a factory for an optax SGD algorithm
        :param optimizer_kwargs: a dictionary of parameters to pass to the SGD
        factory (e.g. which parameters are controllable externally)
        :param clip_grad: maximum magnitude of gradient updates
        :param line_search_kwargs: parameters to pass to optional line search
        method to scale learning rate
        :param noise_kwargs: parameters of optional gradient noise
        :param ema_decay: optional exponential moving average of past parameters
        :param pgpe: optional policy gradient to run alongside the planner
        :param compiler: compiler instance to use for planning
        :param compiler_kwargs: compiler instances kwargs for initialization
        :param use_symlog_reward: whether to use the symlog transform on the 
        reward as a form of normalization
        :param utility: how to aggregate return observations to compute utility
        of a policy or plan; must be either a function mapping jax array to a 
        scalar, or a a string identifying the utility function by name
        :param utility_kwargs: additional keyword arguments to pass hyper-
        parameters to the utility function call
        :param critic: optional critic to estimate the tail return
        :param logger: to log information about compilation to file
        :param dashboard: optional dashboard to display training progress and results
        :param dashboard_viz: optional visualizer object from the environment
        to pass to the dashboard to visualize the policy
        :param preprocessor: optional preprocessor for state inputs to plan
        :param python_functions: dictionary of external Python functions to call from RDDL
        '''
        self.rddl = rddl
        self.plan = plan
        self.batch_size_train = batch_size_train
        if batch_size_test is None:
            batch_size_test = batch_size_train
        self.batch_size_test = batch_size_test
        self.parallel_updates = parallel_updates
        if rollout_horizon is None:
            rollout_horizon = rddl.horizon
        self.horizon = rollout_horizon
        if action_bounds is None:
            action_bounds = {}
        self._action_bounds = action_bounds
        self.optimizer_name = optimizer
        if optimizer_kwargs is None:
            optimizer_kwargs = {'learning_rate': 0.1}
        self.optimizer_kwargs = optimizer_kwargs
        self.clip_grad = clip_grad
        self.line_search_kwargs = line_search_kwargs
        self.noise_kwargs = noise_kwargs
        self.ema_decay = ema_decay
        self.pgpe = pgpe
        self.use_pgpe = pgpe is not None
        self.preprocessor = preprocessor
        if python_functions is None:
            python_functions = {}
        self.python_functions = python_functions

        # set optimizer
        try:
            optimizer = optax.inject_hyperparams(optimizer)(**optimizer_kwargs)
        except Exception as _:
            print(termcolor.colored(
                '[WARN] Could not inject hyperparameters into JaxPlan optimizer: '
                'runtime modification of hyperparameters will be disabled.', 'yellow'
            ))
            optimizer = optimizer(**optimizer_kwargs)   
        
        # apply optimizer chain of transformations
        pipeline = []  
        if clip_grad is not None:
            pipeline.append(optax.clip(clip_grad))
        if noise_kwargs is not None:
            pipeline.append(optax.add_noise(**noise_kwargs))
        pipeline.append(optimizer)
        if line_search_kwargs is not None:
            pipeline.append(optax.scale_by_zoom_linesearch(**line_search_kwargs))
        if ema_decay is not None:
            pipeline.append(optax.ema(ema_decay))
        self.optimizer = optax.chain(*pipeline)
        
        # set utility
        if isinstance(utility, str):
            utility = utility.lower()
            utility_fn = UTILITY_LOOKUP.get(utility, None)
            if utility_fn is None:
                raise RDDLNotImplementedError(
                    f'Utility function <{utility}> is not supported, '
                    f'must be one of {list(UTILITY_LOOKUP.keys())}.'
                )
        else:
            utility_fn = utility
        self.utility = utility_fn
        if utility_kwargs is None:
            utility_kwargs = {}
        self.utility_kwargs = utility_kwargs    
        
        if compiler_kwargs is None:
            compiler_kwargs = {}
        self.compiler_type = compiler
        self.compiler_kwargs = compiler_kwargs
        self.use_symlog_reward = use_symlog_reward
        self.critic_fn = critic

        self.logger = logger
        self.dashboard = dashboard
        self.dashboard_viz = dashboard_viz
        
        self._jax_compile_graph()
    
    @staticmethod
    def summarize_system() -> str:
        '''Returns a string containing information about the system, Python version 
        and jax-related packages that are relevant to the current planner.
        '''         
        devices = jax.devices()
        default_device = devices[0] if devices else 'n/a'
        return termcolor.colored(
            '\n'
            f'Starting JaxPlan v{__version__} '
            f'on device {default_device.platform}{default_device.id}\n', attrs=['bold']
        )
    
    def summarize_relaxations(self) -> str:
        '''Returns a summary table containing all non-differentiable operators
        and their relaxations.
        '''
        result = ''
        overriden_ops_info = self.compiled.overriden_ops_info()
        if overriden_ops_info:
            result += ('[INFO] Some RDDL operations are non-differentiable '
                       'and will be approximated as follows:' + '\n')
            for (class_, op_to_ids_dict) in overriden_ops_info.items():
                result += f'    {class_}:\n'
                for (op, ids) in op_to_ids_dict.items():
                    result += (
                        f'        {op} ' + 
                        termcolor.colored(f'[{len(ids)} occurences]\n', 'dark_grey')
                    )
        return result
        
    def summarize_hyperparameters(self) -> str:
        '''Returns a string summarizing the hyper-parameters of the current planner 
        instance.
        '''
        result = (f'[INFO] objective hyper-parameters:\n'
                  f'    utility_fn        ={self.utility.__name__}\n'
                  f'    utility args      ={self.utility_kwargs}\n'
                  f'    use_symlog        ={self.use_symlog_reward}\n'
                  f'    use_critic        ={self.critic_fn is not None}\n'
                  f'    lookahead         ={self.horizon}\n'
                  f'    user_action_bounds={self._action_bounds}\n'
                  f'[INFO] optimizer hyper-parameters:\n'
                  f'    optimizer         ={self.optimizer_name}\n'
                  f'    optimizer args    ={self.optimizer_kwargs}\n'
                  f'    clip_gradient     ={self.clip_grad}\n'
                  f'    line_search_kwargs={self.line_search_kwargs}\n'
                  f'    noise_kwargs      ={self.noise_kwargs}\n'
                  f'    ema_decay         ={self.ema_decay}\n'
                  f'    batch_size_train  ={self.batch_size_train}\n'
                  f'    batch_size_test   ={self.batch_size_test}\n'
                  f'    parallel_updates  ={self.parallel_updates}\n'
                  f'    preprocessor      ={self.preprocessor}\n')
        result += str(self.plan)
        if self.use_pgpe:
            result += str(self.pgpe)
        result += 'test compiler:\n'
        for k, v in self.test_compiled.get_kwargs().items():
            result += f'    {k}={v}\n'
        result += 'train compiler:\n'
        for k, v in self.compiled.get_kwargs().items():
            result += f'    {k}={v}\n'
        return result
        
    # ===========================================================================
    # COMPILE RDDL
    # ===========================================================================

    def _jax_compile_rddl(self):
        self.compiled = self.compiler_type(
            rddl=self.rddl,
            logger=self.logger,
            python_functions=self.python_functions,
            **self.compiler_kwargs
        )
        self.print_warnings = self.compiled.print_warnings
        self.compiled.compile(log_jax_expr=True, heading='RELAXED MODEL')
    
        self.test_compiled = JaxRDDLCompiler(
            rddl=self.rddl,
            allow_synchronous_state=True,
            logger=self.logger,
            use64bit=self.compiled.use64bit,
            python_functions=self.python_functions
        )
        self.test_compiled.compile(log_jax_expr=True, heading='EXACT MODEL')

    def _jax_compile_policy(self):
        if self.preprocessor is not None:
            self.preprocessor.compile(self.compiled)   
        self.plan.compile(self.compiled,
                          _bounds=self._action_bounds,
                          horizon=self.horizon,
                          preprocessor=self.preprocessor)
        self.train_policy = jax.jit(self.plan.train_policy)
        self.test_policy = jax.jit(self.plan.test_policy)

    def _jax_compile_rollouts(self):
        self.train_rollouts = self.compiled.compile_rollouts(
            policy=self.plan.train_policy,
            n_steps=self.horizon,
            n_batch=self.batch_size_train,
            cache_path_info=self.preprocessor is not None or self.dashboard is not None
        )
        test_rollouts = self.test_compiled.compile_rollouts(
            policy=self.plan.test_policy,
            n_steps=self.horizon,
            n_batch=self.batch_size_test,
            cache_path_info=self.dashboard is not None
        )
        self.test_rollouts = jax.jit(test_rollouts)

    def _jax_compile_train_update(self):
        self.initialize, self.init_optimizer = self._jax_init_optimizer()
        train_loss = self._jax_loss(self.train_rollouts, use_symlog=self.use_symlog_reward)
        self.single_train_loss = train_loss
        self.update = self._jax_update(train_loss)
    
    def _jax_compile_test_loss(self):
        test_loss = self._jax_loss(self.test_rollouts, use_symlog=False)
        self.single_test_loss = test_loss
        self.test_loss = jax.jit(jax.vmap(
            test_loss, in_axes=(None, 0, None, None, None, 0, None)))

    def _jax_compile_pgpe(self):
        if self.use_pgpe:
            self.pgpe.compile(
                loss_fn=self.single_test_loss, 
                projection=self.plan.projection, 
                real_dtype=self.test_compiled.REAL,
                print_warnings=self.print_warnings,
                parallel_updates=self.parallel_updates
            )
            self.merge_pgpe = self._jax_merge_pgpe_jaxplan()
        else:
            self.merge_pgpe = None
    
    def _jax_compile_graph(self):
        self._jax_compile_rddl()
        self._jax_compile_policy()
        self._jax_compile_rollouts()
        self._jax_compile_train_update()
        self._jax_compile_test_loss()
        self._jax_compile_pgpe()
    
    def _jax_critic(self):
        critic_fn = self.critic_fn

        def _jax_wrapped_critic(critic_params, key, policy_params, policy_hyperparams, fls):

            # fluent tensors have leading time dimension, take last observation
            if self.rddl.observ_fluents:
                obs = {name: fls[name][-1, ...] for name in self.rddl.observ_fluents}
            else:
                obs = {state: fls[next_state][-1, ...] 
                    for (state, next_state) in self.rddl.next_state.items()}    

            # calculate action of the last observation   
            act = self.train_policy(key, policy_params, policy_hyperparams, 0, obs)    

            # evaluate and validate critic
            critic_value = jnp.squeeze(critic_fn(critic_params, obs, act))
            assert jnp.isscalar(critic_value), 'Critic value must be a scalar.'
            return critic_value
        return _jax_wrapped_critic            

    def _jax_return(self, use_symlog):
        gamma = self.rddl.discount
        
        # apply discounting of future reward and then optional symlog transform
        def _jax_wrapped_returns(rewards, critic_values):

            # apply optional discounting
            if gamma != 1:
                horizon = jnp.shape(rewards)[-1]
                discount = jnp.power(gamma, jnp.arange(horizon))
                rewards = rewards * discount[jnp.newaxis, ...]
                critic_values = critic_values * jnp.power(gamma, horizon)
            
            # evaluate return and apply optional symlog transform
            returns = jnp.sum(rewards, axis=1) + critic_values
            if use_symlog:
                returns = jnp.sign(returns) * jnp.log1p(jnp.abs(returns))
            return returns
        return _jax_wrapped_returns
        
    def _jax_loss(self, rollouts, use_symlog=False): 
        utility_fn = self.utility    
        utility_kwargs = self.utility_kwargs 
        _jax_wrapped_returns = self._jax_return(use_symlog)
        _jax_wrapped_critic = self._jax_critic()
        
        # the loss is the average cumulative reward across all roll-outs
        # but applies a utility function if requested to each return observation:
        # by default, the utility function is the mean
        def _jax_wrapped_plan_loss(key, policy_params, policy_hyperparams, fls, nfls, 
                                   model_params, critic_params):
            
            # generate sample rollouts
            log, model_params = rollouts(
                key, policy_params, policy_hyperparams, fls, nfls, model_params)
            
            # evaluate optional critic network at termination of rollouts
            batch_size = jnp.shape(log['reward'])[0]
            if self.critic_fn is None:
                critic_values = jnp.zeros((batch_size,), dtype=self.compiled.REAL)
            else:
                keys = random.split(key, num=batch_size)
                critic_values = jax.vmap(
                    _jax_wrapped_critic, in_axes=(None, 0, None, None, 0)
                )(critic_params, keys, policy_params, policy_hyperparams, log['fluents'])
            
            # evaluate cumulative return per rollout
            returns = _jax_wrapped_returns(log['reward'], critic_values)
            utility = utility_fn(returns, **utility_kwargs)
            loss = -utility
            aux = (log, model_params)
            return loss, aux
        return _jax_wrapped_plan_loss
    
    def _jax_init_optimizer(self):
        init = self.plan.initializer
        optimizer = self.optimizer
        num_parallel = self.parallel_updates
        
        # initialize both the policy and its optimizer
        def _jax_wrapped_init_policy(key, policy_hyperparams, fls):
            policy_params = init(key, policy_hyperparams, fls)
            opt_state = optimizer.init(policy_params)
            return policy_params, opt_state, {}    
        
        # initialize just the optimizer from the policy
        def _jax_wrapped_init_opt(policy_params):
            opt_state = jax.vmap(optimizer.init, in_axes=0)(policy_params)
            return opt_state, {}

        # initialize multiple policies to be optimized in parallel
        def _jax_wrapped_batched_init_policy(key, policy_hyperparams, fls):
            keys = random.split(key, num=num_parallel)
            return jax.vmap(
                _jax_wrapped_init_policy, in_axes=(0, None, None)
            )(keys, policy_hyperparams, fls)      
          
        return jax.jit(_jax_wrapped_batched_init_policy), jax.jit(_jax_wrapped_init_opt)
        
    def _jax_update(self, loss):
        optimizer = self.optimizer
        projection = self.plan.projection
        use_ls = self.line_search_kwargs is not None
        num_parallel = self.parallel_updates
        
        # check if the gradients are all zeros
        def _jax_wrapped_zero_gradients(grad):
            leaves, _ = jax.tree_util.tree_flatten(
                jax.tree_util.tree_map(partial(jnp.allclose, b=0), grad))
            return jnp.all(jnp.asarray(leaves))
        
        # calculate the plan gradient w.r.t. return loss and update optimizer
        # also perform a projection step to satisfy constraints on actions
        def _jax_wrapped_loss_swapped(policy_params, key, policy_hyperparams, fls, nfls, 
                                      model_params, critic_params):
            return loss(key, policy_params, policy_hyperparams, fls, nfls, 
                        model_params, critic_params)[0]
            
        def _jax_wrapped_plan_update(key, policy_params, policy_hyperparams, fls, nfls, 
                                     model_params, critic_params, opt_state, opt_aux):
            
            # calculate the gradient of the loss with respect to the policy
            grad_fn = jax.value_and_grad(loss, argnums=1, has_aux=True)
            (loss_val, (log, model_params)), grad = grad_fn(
                key, policy_params, policy_hyperparams, fls, nfls, 
                model_params, critic_params
            )
            
            # require a slightly different update if line search is used
            if use_ls:
                updates, opt_state = optimizer.update(
                    grad, opt_state, params=policy_params, 
                    value=loss_val, grad=grad, value_fn=_jax_wrapped_loss_swapped,
                    key=key, policy_hyperparams=policy_hyperparams, fls=fls, nfls=nfls,
                    model_params=model_params, critic_params=critic_params
                )
            else:
                updates, opt_state = optimizer.update(grad, opt_state, params=policy_params) 

            # apply optimizer and optional policy projection
            policy_params = optax.apply_updates(policy_params, updates)
            policy_params, converged = projection(policy_params, policy_hyperparams)
            
            log['grad'] = grad
            log['updates'] = updates
            zero_grads = _jax_wrapped_zero_gradients(grad)
            return (policy_params, converged, opt_state, opt_aux, 
                    loss_val, log, model_params, zero_grads)
        
        # for parallel policy update, just do each policy update in parallel
        def _jax_wrapped_batched_plan_update(key, policy_params, policy_hyperparams,
                                             fls, nfls, model_params, critic_params, 
                                             opt_state, opt_aux):
            keys = random.split(key, num=num_parallel)
            return jax.vmap(
                _jax_wrapped_plan_update, in_axes=(0, 0, None, None, None, 0, None, 0, 0)
            )(keys, policy_params, policy_hyperparams, fls, nfls, 
              model_params, critic_params, opt_state, opt_aux)
        return jax.jit(_jax_wrapped_batched_plan_update)
            
    def _jax_merge_pgpe_jaxplan(self):

        # currently implements a hard replacement where the jaxplan parameter
        # is replaced by the PGPE parameter if the latter is an improvement
        def _jax_wrapped_batched_pgpe_merge(pgpe_mask, pgpe_param, policy_params, 
                                            pgpe_loss, test_loss, 
                                            pgpe_loss_smooth, test_loss_smooth, 
                                            pgpe_converged, converged):
            mask_tree = jax.tree_util.tree_map(
                lambda leaf: pgpe_mask[(...,) + (jnp.newaxis,) * (jnp.ndim(leaf) - 1)],
                pgpe_param)
            policy_params = jax.tree_util.tree_map(
                jnp.where, mask_tree, pgpe_param, policy_params)
            test_loss = jnp.where(pgpe_mask, pgpe_loss, test_loss)
            test_loss_smooth = jnp.where(pgpe_mask, pgpe_loss_smooth, test_loss_smooth)
            converged = jnp.where(pgpe_mask, pgpe_converged, converged)
            return policy_params, test_loss, test_loss_smooth, converged
        return jax.jit(_jax_wrapped_batched_pgpe_merge)

    def _batched_init_subs(self, init_values): 
        rddl = self.rddl
        n_train, n_test = self.batch_size_train, self.batch_size_test
        
        init_train_fls, init_train_nfls, init_test_fls, init_test_nfls = {}, {}, {}, {}
        for (name, value) in init_values.items():

            # get the initial fluent values and check validity
            init_value = self.test_compiled.init_values.get(name, None)
            if init_value is None:
                raise RDDLUndefinedVariableError(
                    f'Variable <{name}> in init_values argument is not a '
                    f'valid p-variable, must be one of '
                    f'{set(self.test_compiled.init_values.keys())}.'
                )         
            
            # for enum types need to convert the string values to integer indices
            if np.size(value) != np.size(init_value):
                value = init_value
            value = np.reshape(value, np.shape(init_value))
            if value.dtype.type is np.str_:
                value = rddl.object_string_to_index_array(rddl.variable_ranges[name], value)
            
            # train and test fluents have a batch dimension added, non-fluents do not
            # train fluents are also converted to float
            if name not in rddl.non_fluents:
                train_value = np.repeat(value[np.newaxis, ...], repeats=n_train, axis=0)
                init_train_fls[name] = np.asarray(train_value, dtype=self.compiled.REAL)
                init_test_fls[name] = np.repeat(value[np.newaxis, ...], repeats=n_test, axis=0)
            else:
                init_train_nfls[name] = np.asarray(value, dtype=self.compiled.REAL)
                init_test_nfls[name] = value

            # safely cast test variable to required type in case the type is wrong
            if name in rddl.variable_ranges:
                required_type = RDDLValueInitializer.NUMPY_TYPES.get(
                    rddl.variable_ranges[name], RDDLValueInitializer.INT)
                init_test = init_test_nfls if name in rddl.non_fluents else init_test_fls
                if np.result_type(init_test[name]) != required_type:
                    init_test[name] = np.asarray(init_test[name], dtype=required_type)
        
        # make sure next-state fluents are also set
        for (state, next_state) in rddl.next_state.items():
            init_train_fls[next_state] = init_train_fls[state]
            init_test_fls[next_state] = init_test_fls[state]
        return (init_train_fls, init_train_nfls), (init_test_fls, init_test_nfls)
    
    def _broadcast_pytree(self, pytree):
        def make_batched(x):
            x = np.asarray(x)
            x = np.broadcast_to(
                x[np.newaxis, ...], shape=(self.parallel_updates,) + np.shape(x))
            return x
        return jax.tree_util.tree_map(make_batched, pytree)
    
    def as_optimization_problem(
            self, key: Optional[random.PRNGKey]=None,
            policy_hyperparams: Optional[Pytree]=None,
            critic_params: Optional[Pytree]=None,
            loss_function_updates_key: bool=True,
            grad_function_updates_key: bool=False) -> Tuple[Callable, Callable, np.ndarray, Callable]:
        '''Returns a function that computes the loss and a function that 
        computes gradient of the return as a 1D vector given a 1D representation 
        of policy parameters. These functions are designed to be compatible with 
        off-the-shelf optimizers such as scipy. 
        
        Also returns the initial parameter vector to seed an optimizer, 
        as well as a mapping that recovers the parameter pytree from the vector.
        The PRNG key is updated internally starting from the optional given key.
        
        Constraints on actions, if they are required, cannot be constructed 
        automatically in the general case. The user should build constraints
        for each problem in the format required by the downstream optimizer.
        
        :param key: JAX PRNG key (derived from clock if not provided)
        :param policy_hyperparameters: hyper-parameters for the policy/plan, 
        such as weights for sigmoid wrapping boolean actions (defaults to 1
        for all action-fluents if not provided)
        :param critic_params: optional critic network parameters
        :param loss_function_updates_key: if True, the loss function 
        updates the PRNG key internally independently of the grad function
        :param grad_function_updates_key: if True, the gradient function
        updates the PRNG key internally independently of the loss function.
        '''

        # make sure parallel updates are disabled
        if self.parallel_updates > 1:
            raise ValueError('Cannot compile static optimization problem '
                             'when parallel_updates is not None.')
        
        # if PRNG key is not provided
        if key is None:
            key = random.PRNGKey(round(time.time() * 1000))
            
        # initialize the initial fluents, model parameters, policy hyper-params
        (fls, nfls), _ = self._batched_init_subs(self.test_compiled.init_values)
        model_params = self.compiled.model_aux['params']
        if policy_hyperparams is None:
            if self.print_warnings:
                print(termcolor.colored(
                    '[WARN] policy_hyperparams is not set: setting values to 1.0 for '
                    'all action-fluents, which could be suboptimal.', 'yellow'
                ))
            policy_hyperparams = {action: 1.0 for action in self.rddl.action_fluents}
                
        # initialize the policy parameters
        params_guess = self.initialize(key, policy_hyperparams, fls)[0]
        params_guess = pytree_at(params_guess, 0)

        # get the params mapping to a 1D vector
        guess_1d, unravel_fn = jax.flatten_util.ravel_pytree(params_guess)  
        guess_1d = np.asarray(guess_1d)      
        
        # computes the training loss function in a 1D vector
        @jax.jit
        def _loss_with_key(key, params_1d, model_params):
            policy_params = unravel_fn(params_1d)
            loss_val, (_, model_params) = self.single_train_loss(
                key, policy_params, policy_hyperparams, fls, nfls, 
                model_params, critic_params
            )
            return loss_val, model_params
        
        # computes the training loss gradient function in a 1D vector
        grad_fn = jax.grad(self.single_train_loss, argnums=1, has_aux=True)

        @jax.jit
        def _grad_with_key(key, params_1d, model_params):
            policy_params = unravel_fn(params_1d)
            grad_val, (_, model_params) = grad_fn(
                key, policy_params, policy_hyperparams, fls, nfls, 
                model_params, critic_params
            )
            grad_val = jax.flatten_util.ravel_pytree(grad_val)[0]
            return grad_val, model_params 
        
        # store a global reference to the key on every JAX function call and pass when
        # required by JAX, then update it upon return
        def _loss_function(params_1d):
            nonlocal key
            nonlocal model_params
            if loss_function_updates_key:
                key, subkey = random.split(key)
            else:
                subkey = key
            loss_val, model_params = _loss_with_key(subkey, params_1d, model_params)
            loss_val = float(loss_val)
            return loss_val
        
        def _grad_function(params_1d):
            nonlocal key
            nonlocal model_params
            if grad_function_updates_key:
                key, subkey = random.split(key)
            else:
                subkey = key
            grad, model_params = _grad_with_key(subkey, params_1d, model_params)
            grad = np.asarray(grad)
            return grad
        
        return _loss_function, _grad_function, guess_1d, jax.jit(unravel_fn)
                
    # ===========================================================================
    # OPTIMIZE API
    # ===========================================================================

    def optimize(self, *args, **kwargs) -> Dict[str, Any]:
        '''Compute an optimal policy or plan. Return the callback from training.
        
        :param key: JAX PRNG key (derived from clock if not provided)
        :param epochs: the maximum number of steps of gradient descent
        :param train_seconds: total time allocated for gradient descent    
        :param model_params: optional model-parameters to override default
        :param policy_hyperparams: hyper-parameters for the policy/plan, such as
        weights for sigmoid wrapping boolean actions
        :param subs: dictionary mapping initial state and non-fluents to 
        their values: if None initializes all variables from the RDDL instance
        :param guess: initial policy parameters: if None will use the initializer
        specified in this instance
        :param critic_params: parameters of the critic function
        :param print_summary: whether to print planner header and diagnosis
        :param print_progress: whether to print the progress bar during training
        :param print_hyperparams: whether to print list of hyper-parameter settings   
        :param dashboard_id: experiment id for the dashboard
        :param stopping_rule: stopping criterion
        :param test_rolling_window: the test return is averaged on a rolling 
        window of the past test_rolling_window returns when updating the best
        parameters found so far
        :param tqdm_position: position of tqdm progress bar (for multiprocessing)
        '''
        it = self.optimize_generator(*args, **kwargs)
        
        # if the python is C-compiled then the deque is native C and much faster
        # than naively exhausting iterator, but not if the python is some other
        # version (e.g. PyPi); for details, see
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
    
    def optimize_generator(self, key: Optional[random.PRNGKey]=None,
                           epochs: int=999999,
                           train_seconds: float=120.,
                           model_params: Optional[Dict[str, Any]]=None,
                           policy_hyperparams: Optional[Dict[str, Any]]=None,
                           subs: Optional[Dict[str, Any]]=None,
                           guess: Optional[Pytree]=None,
                           critic_params: Optional[Pytree]=None,
                           print_summary: bool=True,
                           print_progress: bool=True,
                           print_hyperparams: bool=False,
                           dashboard_id: Optional[str]=None,
                           stopping_rule: Optional[JaxPlannerStoppingRule]=None,
                           test_rolling_window: int=10,
                           tqdm_position: Optional[int]=None) -> Generator[Dict[str, Any], None, None]:
        '''Returns a generator for computing an optimal policy or plan. 
        Generator can be iterated over to lazily optimize the plan, yielding
        a dictionary of intermediate computations.
        
        :param key: JAX PRNG key (derived from clock if not provided)
        :param epochs: the maximum number of steps of gradient descent
        :param train_seconds: total time allocated for gradient descent 
        :param model_params: optional model-parameters to override default
        :param policy_hyperparams: hyper-parameters for the policy/plan, such as
        weights for sigmoid wrapping boolean actions
        :param subs: dictionary mapping initial state and non-fluents to 
        their values: if None initializes all variables from the RDDL instance
        :param guess: initial policy parameters: if None will use the initializer
        specified in this instance        
        :param critic_params: parameters of the critic function
        :param print_summary: whether to print planner header and diagnosis
        :param print_progress: whether to print the progress bar during training
        :param print_hyperparams: whether to print list of hyper-parameter settings
        :param dashboard_id: experiment id for the dashboard
        :param stopping_rule: stopping criterion
        :param test_rolling_window: the test return is averaged on a rolling 
        window of the past test_rolling_window returns when updating the best
        parameters found so far
        :param tqdm_position: position of tqdm progress bar (for multiprocessing)
        '''

        # start measuring execution time here, including time spent outside optimize loop
        start_time = time.time()
        elapsed_outside_loop = 0
        
        # ======================================================================
        # INITIALIZATION OF HYPER-PARAMETERS
        # ======================================================================

        # if PRNG key is not provided
        if key is None:
            key = random.PRNGKey(round(time.time() * 1000))
            if self.print_warnings:
                print(termcolor.colored(
                    '[WARN] PRNG key is not set: setting from clock.', 'yellow'
                ))
        dash_key = key[1].item()
            
        # if policy_hyperparams is not provided
        if policy_hyperparams is None:
            if self.print_warnings:
                print(termcolor.colored(
                    '[WARN] policy_hyperparams is not set: setting values to 1.0 for '
                    'all action-fluents, which could be suboptimal.', 'yellow'
                ))
            policy_hyperparams = {action: 1.0 for action in self.rddl.action_fluents}
        
        # if policy_hyperparams is a scalar
        elif isinstance(policy_hyperparams, (int, float, np.number)):
            policy_hyperparams = {action: float(policy_hyperparams) 
                                  for action in self.rddl.action_fluents}
        
        # fill in missing entries
        elif isinstance(policy_hyperparams, dict):
            for action in self.rddl.action_fluents:
                if action not in policy_hyperparams:
                    if self.print_warnings:
                        print(termcolor.colored(
                            f'[WARN] policy_hyperparams[{action}] is not set: '
                            f'setting values to 1.0 for missing action-fluents, '
                            f'which could be suboptimal.', 'yellow'
                        ))
                    policy_hyperparams[action] = 1.0
        
        # initialize preprocessor        
        if self.preprocessor is None:
            preproc_key = None
        else:
            preproc_key = self.preprocessor.HYPERPARAMS_KEY
            policy_hyperparams[preproc_key] = self.preprocessor.initialize()

        # print summary of parameters:
        if print_summary:
            print(self.summarize_system())
            print(self.summarize_relaxations())
        if print_hyperparams:
            print(self.summarize_hyperparameters())
            print(
                f'[INFO] optimize call hyper-parameters:\n'
                f'    PRNG key           ={key}\n'
                f'    max_iterations     ={epochs}\n'
                f'    max_seconds        ={train_seconds}\n'
                f'    model_params       ={model_params}\n'
                f'    policy_hyper_params={policy_hyperparams}\n'
                f'    override_subs_dict ={subs is not None}\n'
                f'    provide_param_guess={guess is not None}\n'
                f'    critic_params      ={critic_params is not None}\n'
                f'    test_rolling_window={test_rolling_window}\n'
                f'    print_summary      ={print_summary}\n'
                f'    print_progress     ={print_progress}\n'
                f'    dashboard_id       ={dashboard_id}\n'
                f'    stopping_rule      ={stopping_rule}\n'
            )
        
        # ======================================================================
        # INITIALIZATION OF STATE AND POLICY
        # ======================================================================
        
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
            if self.print_warnings and added_pvars_to_subs:
                print(termcolor.colored(
                    f'[INFO] p-variable(s) {added_pvars_to_subs} are not in '
                    f'provided subs: using their initial values.', 'dark_grey'
                ))
        train_subs, test_subs = self._batched_init_subs(subs)
        
        # initialize model parameters
        if model_params is None:
            model_params = self.compiled.model_aux['params']
        model_params = self._broadcast_pytree(model_params)
        model_params_test = self._broadcast_pytree(self.test_compiled.model_aux['params'])
        
        # initialize policy parameters
        if guess is None:
            key, subkey = random.split(key)
            policy_params, opt_state, opt_aux = self.initialize(
                subkey, policy_hyperparams, train_subs[0])
        else:
            policy_params = self._broadcast_pytree(guess)
            opt_state, opt_aux = self.init_optimizer(policy_params)
        
        # initialize pgpe parameters
        if self.use_pgpe:
            pgpe_params, pgpe_opt_state, r_max = self.pgpe.initialize(key, policy_params)
            rolling_pgpe_loss = RollingMean(test_rolling_window)
        else:
            pgpe_params = pgpe_opt_state = r_max = rolling_pgpe_loss = None
        total_pgpe_it = 0

        # ======================================================================
        # INITIALIZATION OF RUNNING STATISTICS
        # ======================================================================
        
        # initialize running statistics
        best_params = pytree_at(policy_params, 0)
        best_loss, pbest_loss, best_grad = np.inf, np.inf, None
        best_index = 0
        last_iter_improve = 0
        rolling_test_loss = RollingMean(test_rolling_window)
        status = JaxPlannerStatus.NORMAL
        progress_percent = 0
        
        # initialize stopping criterion
        if stopping_rule is not None:
            stopping_rule.reset()
            
        # initialize dashboard 
        dashboard = self.dashboard
        if dashboard is not None:
            dashboard_id = dashboard.register_experiment(
                dashboard_id, 
                dashboard.get_planner_info(self), 
                key=dash_key, 
                viz=self.dashboard_viz
            )
        
        # progress bar
        if print_progress:
            progress_bar = tqdm(None, total=100, position=tqdm_position, 
                                bar_format='{l_bar}{bar}| {elapsed} {postfix}')
        else:
            progress_bar = None
        position_str = '' if tqdm_position is None else f'[{tqdm_position}]'

        # error handlers (to avoid spam messaging)
        policy_constraint_msg_shown = False
        jax_train_msg_shown = set()
        jax_test_msg_shown = set()
        
        # ======================================================================
        # MAIN TRAINING LOOP BEGINS
        # ======================================================================
        
        for it in range(epochs):
            
            # ==================================================================
            # JAXPLAN GRADIENT DESCENT STEP
            # ==================================================================
            
            status = JaxPlannerStatus.NORMAL
            
            # update the parameters of the plan
            key, subkey = random.split(key)
            (policy_params, converged, opt_state, opt_aux, train_loss, train_log, 
             model_params, zero_grads) = self.update(
                 subkey, policy_params, policy_hyperparams, *train_subs, 
                 model_params, critic_params, opt_state, opt_aux
            )
            
            # update the preprocessor
            if self.preprocessor is not None:                
                policy_hyperparams[preproc_key] = self.preprocessor.update(
                    train_log['fluents'], policy_hyperparams[preproc_key]
                )

            # evaluate
            test_loss, (test_log, model_params_test) = self.test_loss(
                subkey, policy_params, policy_hyperparams, *test_subs, 
                model_params_test, critic_params
            )
            train_loss = np.asarray(train_loss)
            test_loss = np.asarray(test_loss)
            test_loss_smooth = rolling_test_loss.update(test_loss)

            # ==================================================================
            # PGPE GRADIENT DESCENT STEP
            # ==================================================================
            
            pgpe_improve = False
            if self.use_pgpe:

                # pgpe update of the plan
                key, subkey = random.split(key)
                pgpe_params, r_max, pgpe_opt_state, pgpe_param, pgpe_converged = self.pgpe.update(
                    subkey, pgpe_params, r_max, progress_percent, 
                    policy_hyperparams, *test_subs, model_params_test, critic_params, 
                    pgpe_opt_state
                )
                
                # evaluate
                pgpe_loss, _ = self.test_loss(
                    subkey, pgpe_param, policy_hyperparams, *test_subs, 
                    model_params_test, critic_params
                )
                pgpe_loss = np.asarray(pgpe_loss)
                pgpe_loss_smooth = rolling_pgpe_loss.update(pgpe_loss)
                pgpe_return = -pgpe_loss_smooth

                # replace JaxPlan with PGPE if new minimum reached or train loss invalid
                pgpe_mask = (pgpe_loss_smooth < pbest_loss) | ~np.isfinite(train_loss)
                if np.any(pgpe_mask):
                    policy_params, test_loss, test_loss_smooth, converged = self.merge_pgpe(
                        pgpe_mask, pgpe_param, policy_params, 
                        pgpe_loss, test_loss, pgpe_loss_smooth, test_loss_smooth, 
                        pgpe_converged, converged
                    )
                    pgpe_improve = True
                    total_pgpe_it += 1  
            else:
                pgpe_loss = pgpe_loss_smooth = pgpe_return = None

            # ==================================================================
            # STATUS CHECKS AND LOGGING
            # ==================================================================

            # evaluate test losses and record best parameters so far
            best_index = np.argmin(test_loss_smooth)
            if test_loss_smooth[best_index] < best_loss:
                best_params = pytree_at(policy_params, best_index)
                best_grad = pytree_at(train_log['grad'], best_index)
                best_loss = test_loss_smooth[best_index]
                last_iter_improve = it
            pbest_loss = np.minimum(pbest_loss, test_loss_smooth)
                        
            # no progress
            if (not pgpe_improve) and np.all(zero_grads):
                status = JaxPlannerStatus.NO_PROGRESS
            
            # constraint satisfaction problem
            if not np.all(converged):                
                if progress_bar is not None and not policy_constraint_msg_shown:
                    progress_bar.write(termcolor.colored(
                        '[FAIL] Policy update violated action constraints.', 'red'
                    ))
                    policy_constraint_msg_shown = True
                status = JaxPlannerStatus.PRECONDITION_POSSIBLY_UNSATISFIED
            
            # numerical error
            invalid_loss = not np.any(np.isfinite(train_loss))
            if self.use_pgpe:
                invalid_loss = invalid_loss and not np.any(np.isfinite(pgpe_loss))
            if invalid_loss:
                if progress_bar is not None:
                    progress_bar.write(termcolor.colored(
                        f'[FAIL] Planner aborted early with train loss {train_loss}.', 'red'
                    ))
                status = JaxPlannerStatus.INVALID_GRADIENT
              
            # problem in the model compilation
            if progress_bar is not None:

                # train model
                for error_code in np.unique(train_log['error']):
                    if error_code not in jax_train_msg_shown:
                        jax_train_msg_shown.add(error_code)
                        for message in JaxRDDLCompiler.get_error_messages(error_code):
                            progress_bar.write(termcolor.colored(
                                '[FAIL] Training model error: ' + message, 'red'))

                # test model
                for error_code in np.unique(test_log['error']):
                    if error_code not in jax_test_msg_shown:
                        jax_test_msg_shown.add(error_code)
                        for message in JaxRDDLCompiler.get_error_messages(error_code):
                            progress_bar.write(termcolor.colored(
                                '[FAIL] Testing model error: ' + message, 'red'))
        
            # reached computation budget
            elapsed = time.time() - start_time - elapsed_outside_loop
            if elapsed >= train_seconds:
                status = JaxPlannerStatus.TIME_BUDGET_REACHED
            if it >= epochs - 1:
                status = JaxPlannerStatus.ITER_BUDGET_REACHED
            
            # build a callback
            progress_percent = 100 * min(
                1, max(0, elapsed / train_seconds, it / (epochs - 1)))
            callback = {
                'iteration': it,
                'elapsed_time': elapsed,
                'progress': progress_percent,
                'status': status,
                'key': key,
                'train_return':-train_loss,
                'test_return':-test_loss_smooth,
                'best_return':-best_loss,
                'pgpe_return': pgpe_return,
                'last_iteration_improved': last_iter_improve,
                'pgpe_improved': pgpe_improve,
                'params': policy_params,
                'best_params': best_params,
                'best_index': best_index,
                'pgpe_params': pgpe_params,
                'model_params': model_params,
                'policy_hyperparams': policy_hyperparams,
                'grad': train_log['grad'],
                'best_grad': best_grad,
                'train_log': train_log,
                'test_log': test_log
            }

            # stopping condition reached
            if stopping_rule is not None and stopping_rule.monitor(callback):
                if self.print_warnings and progress_bar is not None:
                    progress_bar.write(termcolor.colored(
                        '[SUCC] Stopping rule has been reached.', 'green'
                    ))
                callback['status'] = status = JaxPlannerStatus.STOPPING_RULE_REACHED  
            
            # if the progress bar is used
            if print_progress:
                progress_bar.set_description(
                    f'{position_str} {it} it | {-np.min(train_loss):13.5f} train | '
                    f'{-np.min(test_loss_smooth):13.5f} test | '
                    f'{-best_loss:13.5f} best | '
                    f'{total_pgpe_it} pgpe | {status.value} status', 
                    refresh=False
                )
                progress_bar.set_postfix_str(
                    f'{(it + 1) / (elapsed + 1e-6):.2f}it/s', refresh=False
                )
                progress_bar.update(progress_percent - progress_bar.n)
            
            # dashboard
            if dashboard is not None:            
                dashboard.update_experiment(dashboard_id, callback)
                        
            # yield the callback
            start_time_outside = time.time()
            yield callback
            elapsed_outside_loop += (time.time() - start_time_outside)
            
            # abortion check
            if status.is_terminal():
                break             
        
        # ======================================================================
        # POST-PROCESSING AND CLEANUP
        # ====================================================================== 

        # release resources
        if print_progress:
            progress_bar.close()
        
        # summarize and test for convergence
        if print_summary:

            # calculate gradient norm
            grad_norm = jax.tree_util.tree_map(lambda x: np.linalg.norm(x).item(), best_grad)
            grad_norms = jax.tree_util.tree_leaves(grad_norm)
            max_grad_norm = max(grad_norms) if grad_norms else np.nan

            # calculate best policy return
            _, (final_log, _) = self.test_loss(
                key, self._broadcast_pytree(best_params), policy_hyperparams, 
                *test_subs, model_params_test, critic_params
            )
            best_returns = np.ravel(np.sum(final_log['reward'], axis=2))
            mean, rlo, rhi = self.ci_bootstrap(best_returns)            

            # diagnosis
            diagnosis = self._perform_diagnosis(
                last_iter_improve, -np.min(train_loss), -np.min(test_loss_smooth), 
                -best_loss, max_grad_norm
            )
            print(
                f'[INFO] Summary of optimization:\n'
                f'    status:           {status}\n'
                f'    time:             {elapsed:.2f} seconds\n'
                f'    iterations:       {it}\n'
                f'    best objective:   {-best_loss:.5f}\n'
                f'    best grad norm:   {max_grad_norm:.5f}\n'
                f'    best cuml reward: Mean = {mean:.5f}, 95% CI [{rlo:.5f}, {rhi:.5f}]\n'
                f'    diagnosis:        {diagnosis}\n'
            )
    
    @staticmethod
    def ci_bootstrap(returns, confidence=0.95, n_boot=10000):
        means = np.zeros((n_boot,))
        for i in range(n_boot):
            means[i] = np.mean(np.random.choice(returns, size=len(returns), replace=True))
        lower = np.percentile(means, 100 * (1 - confidence) / 2)
        upper = np.percentile(means, 100 * (1 + confidence) / 2)
        mean = np.mean(returns)
        return mean, lower, upper

    def _perform_diagnosis(self, last_iter_improve, train_return, test_return, best_return, 
                           max_grad_norm):
        
        # divergence if the solution is not finite
        if not np.isfinite(train_return):
            return termcolor.colored('[FAIL] Training loss diverged.', 'red')
            
        # hit a plateau is likely IF:
        # 1. planner does not improve at all
        # 2. the gradient norm at the best solution is zero
        grad_is_zero = np.allclose(max_grad_norm, 0)
        if last_iter_improve <= 1:
            if grad_is_zero:
                return termcolor.colored(
                    f'[FAIL] No progress and ||g||={max_grad_norm:.4f}, '
                    f'solver initialized in a plateau.', 'red'
                )
            else:
                return termcolor.colored(
                    f'[FAIL] No progress and ||g||={max_grad_norm:.4f}, '
                    f'adjust learning rate or other parameters.', 'red'
                )
        
        # model is likely poor IF:
        # 1. the train and test return disagree
        validation_error = abs(test_return - train_return)
        validation_error_norm = max(abs(train_return), abs(test_return))
        if not (validation_error < 0.2 * validation_error_norm):
            return termcolor.colored(
                f'[WARN] Progress but large rel. train/test error {validation_error:.4f}, '
                f'adjust model or batch size.', 'yellow'
            )
        
        # model likely did not converge IF:
        # 1. the max grad relative to the return is high
        if not grad_is_zero:
            if not (abs(best_return) > 1.0 * max_grad_norm):
                return termcolor.colored(
                    f'[WARN] Progress but large ||g||={max_grad_norm:.4f}, '
                    f'adjust learning rate or budget.', 'yellow'
                )
        
        # likely successful
        return termcolor.colored(
            '[SUCC] No convergence problems found.', 'green'
        )
        
    def get_action(self, key: random.PRNGKey,
                   params: Pytree,
                   step: int,
                   state: Dict[str, Any],
                   policy_hyperparams: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
        '''Returns an action dictionary from the policy or plan with the given parameters.
        
        :param key: the JAX PRNG key
        :param params: the trainable parameter PyTree of the policy
        :param step: the time step at which decision is made
        :param state: the dict of state p-variables
        :param policy_hyperparams: hyper-parameters for the policy/plan, such as
        weights for sigmoid wrapping boolean actions (optional)
        '''
        state = state.copy()
        
        # check compatibility of the state dictionary
        for (var, values) in state.items():
            
            # must not be grounded
            if RDDLPlanningModel.FLUENT_SEP in var or RDDLPlanningModel.OBJECT_SEP in var:
                raise ValueError(f'State dictionary passed to the JAX policy is '
                                 f'grounded, since it contains the key <{var}>, '
                                 f'but a vectorized environment is required: '
                                 f'make sure vectorized = True in the RDDLEnv.')
            
            # must be numeric array
            # exception is for POMDPs at 1st epoch when observ-fluents are None
            dtype = np.result_type(values)
            if not np.issubdtype(dtype, np.number) and not np.issubdtype(dtype, np.bool_):
                if step == 0 and var in self.rddl.observ_fluents:
                    state[var] = self.test_compiled.init_values[var]
                else:
                    if dtype.type is np.str_:
                        prange = self.rddl.variable_ranges[var]
                        state[var] = self.rddl.object_string_to_index_array(prange, state[var])     
                    else:               
                        raise ValueError(
                            f'Values {values} assigned to p-variable <{var}> are '
                            f'non-numeric of type {dtype}.'
                        )
            
        # cast device arrays to numpy
        actions = self.test_policy(key, params, policy_hyperparams, step, state)
        actions = jax.tree_util.tree_map(np.asarray, actions)
        return actions      
       

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
                 key: Optional[random.PRNGKey]=None,
                 eval_hyperparams: Optional[Dict[str, Any]]=None,
                 params: Optional[Union[str, Pytree]]=None,
                 train_on_reset: bool=False,
                 save_path: Optional[str]=None,
                 **train_kwargs) -> None:
        '''Creates a new JAX offline control policy that is trained once, then
        deployed later.
        
        :param planner: underlying planning algorithm for optimizing actions
        :param key: the RNG key to seed randomness (derives from clock if not
        provided)
        :param eval_hyperparams: policy hyperparameters to apply for evaluation
        or whenever sample_action is called
        :param params: use the specified policy parameters instead of calling
        planner.optimize(); can be a string pointing to a valid file path where params
        have been saved, or a pytree of parameters
        :param train_on_reset: retrain policy parameters on every episode reset
        :param save_path: optional path to save parameters to
        :param **train_kwargs: any keyword arguments to be passed to the planner
        for optimization
        '''
        self.planner = planner
        if key is None:
            key = random.PRNGKey(round(time.time() * 1000))
        self.key = key
        self.eval_hyperparams = eval_hyperparams
        self.train_on_reset = train_on_reset
        self.train_kwargs = train_kwargs        
        self.params_given = params is not None
        self.hyperparams_given = eval_hyperparams is not None
        
        # load the policy from file
        if not self.train_on_reset and params is not None and isinstance(params, str):
            with open(params, 'rb') as file:
                params = pickle.load(file)

        # train the policy once before starting to step() through the environment
        # and then execute this policy in open-loop fashion
        self.step = 0
        self.callback = None
        if not self.train_on_reset and not self.params_given:
            callback = self.planner.optimize(key=self.key, **self.train_kwargs)
            self.callback = callback
            params = callback['best_params'] 
            if not self.hyperparams_given:
                self.eval_hyperparams = callback['policy_hyperparams']

            # save the policy
            if save_path is not None:
                with open(save_path, 'wb') as file:
                    pickle.dump(params, file)

        self.params = params  
        
    def sample_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        self.key, subkey = random.split(self.key)
        actions = self.planner.get_action(
            subkey, self.params, self.step, state, self.eval_hyperparams)
        self.step += 1
        return actions
        
    def reset(self) -> None:
        self.step = 0

        # train the policy if required to reset at the start of every episode
        if self.train_on_reset and not self.params_given:
            callback = self.planner.optimize(key=self.key, **self.train_kwargs)
            self.callback = callback
            self.params = callback['best_params']
            if not self.hyperparams_given:
                self.eval_hyperparams = callback['policy_hyperparams']


class JaxOnlineController(BaseAgent):
    '''A container class for a Jax controller continuously updated using state feedback.'''
    
    use_tensor_obs = True
    
    def __init__(self, planner: JaxBackpropPlanner,
                 key: Optional[random.PRNGKey]=None,
                 eval_hyperparams: Optional[Dict[str, Any]]=None,
                 warm_start: bool=True,
                 max_attempts: int=3,
                 **train_kwargs) -> None:
        '''Creates a new JAX control policy that is trained online in a closed-
        loop fashion.
        
        :param planner: underlying planning algorithm for optimizing actions
        :param key: the RNG key to seed randomness (derives from clock if not provided)
        :param eval_hyperparams: policy hyperparameters to apply for evaluation
        or whenever sample_action is called
        :param warm_start: whether to use the previous decision epoch final
        policy parameters to warm the next decision epoch
        :param max_attempts: maximum attempted restarts of the optimizer when the total
        iteration count is 1 (i.e. the execution time is dominated by the jit compilation)
        :param **train_kwargs: any keyword arguments to be passed to the planner
        for optimization
        '''
        self.planner = planner
        if key is None:
            key = random.PRNGKey(round(time.time() * 1000))
        self.key = key
        self.eval_hyperparams = eval_hyperparams
        self.hyperparams_given = eval_hyperparams is not None
        self.warm_start = warm_start
        self.train_kwargs = train_kwargs
        self.max_attempts = max_attempts
        self.reset()
     
    def sample_action(self, state: Dict[str, Any]) -> Dict[str, Any]:

        # we train the policy from the current state every time we step()
        planner = self.planner
        callback = planner.optimize(
            key=self.key, guess=self.guess, subs=state, **self.train_kwargs)
        
        # optimize again if jit compilation takes up the entire time budget
        # this can be done for several attempts until the optimizer has traced the 
        # computation graph: we report the callback of the successful attempt (if exists)
        attempts = 0
        while attempts < self.max_attempts and callback['iteration'] <= 1:
            attempts += 1
            if self.planner.print_warnings:
                print(termcolor.colored(
                    f'[INFO] JIT compilation dominated the execution time: '
                    f'executing the optimizer again on the traced model '
                    f'[attempt {attempts}].', 'dark_grey'
                ))
            callback = planner.optimize(
                key=self.key, guess=self.guess, subs=state, **self.train_kwargs)   
        if callback['iteration'] <= 1 and self.planner.print_warnings:
            print(termcolor.colored(
                f'[FAIL] JIT compilation dominated the execution time and '
                f'ran out of attempts: increase max_attempts or the training time.', 'red'
            )) 
        
        # use the last callback obtained
        self.callback = callback
        params = callback['best_params']
        if not self.hyperparams_given:
            self.eval_hyperparams = callback['policy_hyperparams']

        # get the action from the parameters for the current state
        self.key, subkey = random.split(self.key)
        actions = planner.get_action(subkey, params, 0, state, self.eval_hyperparams)

        # apply warm start for the next epoch
        if self.warm_start:
            self.guess = planner.plan.guess_next_epoch(params)
        return actions
        
    def reset(self) -> None:
        self.guess = None
        self.callback = None
    
