# ***********************************************************************
# JAXPLAN
# 
# Author: Michael Gimelfarb
#
# REFERENCES:
#
# [1] Bueno, Thiago P., Leliane N. de Barros, Denis D. Mauá, and Scott Sanner. "Deep 
# reactive policies for planning in stochastic nonlinear domains." In Proceedings of the 
# AAAI Conference on Artificial Intelligence, vol. 33, no. 01, pp. 7530-7537. 2019.
#
# [2] Cui, Hao, Thomas Keller, and Roni Khardon. "Stochastic planning with lifted symbolic 
# trajectory optimization." In Proceedings of the International Conference on Automated 
# Planning and Scheduling, vol. 29, pp. 119-127. 2019.
#
# [3] Wu, Ga, Buser Say, and Scott Sanner. "Scalable planning with tensorflow for hybrid 
# nonlinear domains." Advances in Neural Information Processing Systems 30 (2017).
#
# ***********************************************************************


from abc import ABCMeta, abstractmethod
from functools import partial
from typing import Any, Callable, Dict, Optional, Sequence, Type, Tuple

import jax
import jax.nn.initializers as initializers
import jax.numpy as jnp
import jax.random as random
import flax.linen as nn
import numpy as np
import termcolor

from pyRDDLGym.core.debug.exception import (
    RDDLNotImplementedError,
    RDDLTypeError
)

from pyRDDLGym_jax.core.compiler import JaxRDDLCompiler, safe_log
from pyRDDLGym_jax.core.logic import JaxRDDLCompilerWithGrad

Activation = Callable[[jnp.ndarray], jnp.ndarray]
Bounds = Dict[str, Tuple[np.ndarray, np.ndarray]]
Kwargs = Dict[str, Any]
Pytree = Any


# ***********************************************************************
# ALL VERSIONS OF STATE PREPROCESSING FOR DRP
# 
# - static normalization
# 
# ***********************************************************************


class Preprocessor(metaclass=ABCMeta):
    '''Base class for all state preprocessors.'''

    HYPERPARAMS_KEY = '__preprocessor__'

    def __init__(self) -> None:
        self._initializer = None
        self._update = None
        self._transform = None

    # signature is initialize() -> Pytree
    @property
    def initialize(self) -> Callable:
        return self._initializer

    # signature is update(fls: Pytree, stats: Pytree) -> Pytree
    @property
    def update(self) -> Callable:
        return self._update
    
    # signature is transform(fls: Pytree, stats: Pytree) -> Pytree
    @property
    def transform(self) -> Callable:
        return self._transform
    
    @abstractmethod
    def compile(self, compiled: JaxRDDLCompilerWithGrad) -> None:
        pass


class StaticNormalizer(Preprocessor):
    '''Normalize values by box constraints on fluents computed from the RDDL domain.'''

    def __init__(self, fluent_bounds: Optional[Bounds]=None) -> None:
        if fluent_bounds is None:
            fluent_bounds = {}
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
                if np.all(np.isfinite(lower) & np.isfinite(upper) & np.less(lower, upper)):
                    bounded_vars[var] = (lower, upper)
                user_bounds = self.fluent_bounds.get(var, None)
                if user_bounds is not None:
                    bounded_vars[var] = tuple(user_bounds)
        bounded_vars = jax.tree_util.tree_map(
            partial(np.asarray, dtype=compiled.REAL), bounded_vars)
        
        # initialize to ranges computed by the constraint parser
        def _jax_wrapped_normalizer_init():
            return bounded_vars        
        self._initializer = jax.jit(_jax_wrapped_normalizer_init)

        # static bounds
        def _jax_wrapped_normalizer_update(fls, stats):
            return stats
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
                    new_fls[var] = jnp.divide(values - lower, upper - lower)
                else:
                    new_fls[var] = values
            return new_fls
        self._transform = jax.jit(_jax_wrapped_normalizer_transform)


class RunningStandardizer(Preprocessor):
    '''Standardize values by a running (bias-corrected EMA) estimate of the mean and
    standard deviation of each fluent, computed from the training rollouts.'''

    def __init__(self, decay: float=0.99, epsilon: float=1e-8,
                 clip: Optional[float]=20.) -> None:
        '''Creates a new running standardizer.

        :param decay: decay factor of the running mean and variance per update
        :param epsilon: added to the variance for numerical stability
        :param clip: optional bound on the absolute value of the standardized fluents
        '''
        super(RunningStandardizer, self).__init__()
        self.decay = decay
        self.epsilon = epsilon
        self.clip = clip

    def compile(self, compiled: JaxRDDLCompilerWithGrad) -> None:

        # adjust for partial observability
        rddl = compiled.rddl
        if rddl.observ_fluents:
            observed_vars = rddl.observ_fluents
        else:
            observed_vars = rddl.state_fluents

        # ignore boolean fluents
        shapes = {var: np.shape(compiled.init_values[var])
                  for var in observed_vars if rddl.variable_ranges[var] != 'bool'}
        decay, epsilon, clip = self.decay, self.epsilon, self.clip

        # running first and second moments start at zero
        def _jax_wrapped_standardizer_init():
            return {var: {'mean': jnp.zeros(shape, dtype=compiled.REAL),
                          'm2': jnp.zeros(shape, dtype=compiled.REAL),
                          'count': jnp.zeros((), dtype=compiled.REAL)}
                    for (var, shape) in shapes.items()}
        self._initializer = jax.jit(_jax_wrapped_standardizer_init)

        # update the moments with the fluents of the training rollouts, averaging over
        # all leading (e.g. batch and time) dimensions
        def _jax_wrapped_standardizer_update(fls, stats):
            new_stats = {}
            for (var, stat) in stats.items():
                values = jnp.asarray(fls[var], dtype=compiled.REAL)
                axes = tuple(range(jnp.ndim(values) - len(shapes[var])))
                new_stats[var] = {
                    'mean': decay * stat['mean'] + (1. - decay) * jnp.mean(values, axis=axes),
                    'm2': decay * stat['m2'] + (1. - decay) * jnp.mean(
                        jnp.square(values), axis=axes),
                    'count': stat['count'] + 1.
                }
            return new_stats
        self._update = jax.jit(_jax_wrapped_standardizer_update)

        # apply standardization once at least one update has been made
        def _jax_wrapped_standardizer_transform(fls, stats):
            new_fls = dict(fls)
            for (var, stat) in stats.items():
                values = fls[var]
                correction = jnp.maximum(1. - decay ** stat['count'], epsilon)
                mean = stat['mean'] / correction
                var_ = jnp.maximum(stat['m2'] / correction - jnp.square(mean), 0.)
                new_dims = jnp.ndim(values) - jnp.ndim(mean)
                mean = mean[(jnp.newaxis,) * new_dims + (...,)]
                var_ = var_[(jnp.newaxis,) * new_dims + (...,)]
                standardized = (values - mean) / jnp.sqrt(var_ + epsilon)
                if clip is not None:
                    standardized = jnp.clip(standardized, -clip, clip)
                new_fls[var] = jnp.where(stat['count'] > 0, standardized, values)
            return new_fls
        self._transform = jax.jit(_jax_wrapped_standardizer_transform)


# ***********************************************************************
# JAX BASE PLAN TEMPLATE
#
# ***********************************************************************


class JaxPlan(metaclass=ABCMeta):
    '''Base class for all JAX policy representations.'''

    history_dependent = False
    
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
                test_compiled: JaxRDDLCompiler,
                _bounds: Bounds,
                horizon: int,
                preprocessor: Optional[Preprocessor]=None) -> None:
        pass
    
    @abstractmethod
    def guess_next_epoch(self, params: Pytree) -> Pytree:
        pass
    
    # signature is initializer(sim_state) -> (params: Pytree, hyperparams: Pytree)
    @property
    def initializer(self) -> Callable:
        return self._initializer

    @initializer.setter
    def initializer(self, value: Callable) -> None:
        self._initializer = value
    
    # signature is train_policy(sim_state, planner_state) -> (actions: Dict)
    @property
    def train_policy(self) -> Callable:
        return self._train_policy

    @train_policy.setter
    def train_policy(self, value: Callable) -> None:
        self._train_policy = value
    
    # signature is test_policy(sim_state, planner_state) -> (actions: Dict)
    @property
    def test_policy(self) -> Callable:
        return self._test_policy

    @test_policy.setter
    def test_policy(self, value: Callable) -> None:
        self._test_policy = value
        
    # signature is projection(planner_state) -> (params: Pytree, converged: bool)
    @property
    def projection(self) -> Callable:
        return self._projection

    @projection.setter
    def projection(self, value: Callable) -> None:
        self._projection = value


def get_action_info(compiled: JaxRDDLCompiler, user_bounds: Bounds, horizon: int) -> Tuple[Any, Any, Any]:
    '''Gets the shapes, bounds, and conditional lists for action-fluents in the RDDL domain.'''
    shapes, bounds, cond_lists = {}, {}, {}
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
                upper_value = len(compiled.rddl.type_to_objects[prange]) - 1
                upper = np.ones(shape=shapes[name][1:]) * upper_value
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
    return shapes, bounds, cond_lists
   

# ***********************************************************************
# JAX STRAIGHT LINE PLAN
#
# ***********************************************************************


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
        def _jax_wrapped_sorting_project(bool_params, hyperparams):
            
            # find the amount to shift action parameters: if noop=True reflect parameter
            scores = []
            for (var, param) in bool_params.items():
                logit_flat = jnp.ravel(param['logit'])
                if noop[var]:
                    if wrap_sigmoid:
                        logit_flat = -logit_flat
                    else:
                        logit_flat = 1. - logit_flat
                scores.append(logit_flat)
            scores = jnp.concatenate(scores)
            descending = jnp.sort(scores)[::-1]
            kplus1st_greatest = descending[allowed_actions]
            surplus = jnp.maximum(kplus1st_greatest - bool_threshold, 0.)
                
            # perform the shift
            new_bool_params = {}
            for (var, param) in bool_params.items():
                if noop[var]:
                    shifted_logit = param['logit'] + surplus
                else:
                    shifted_logit = param['logit'] - surplus
                new_logit = jax_bool_to_box(var, shifted_logit, hyperparams)
                new_bool_params[var] = {'logit': new_logit}
            converged = jnp.array(True, dtype=jnp.bool_)
            return new_bool_params, converged
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
            amount = jnp.divide(surplus, k)
            new_actions = {}
            for (var, action) in actions.items():
                if noop[var]:
                    new_actions[var] = jnp.minimum(action + amount, 1)
                else:
                    new_actions[var] = jnp.maximum(action - amount, 0)
            new_surplus, new_k = _jax_wrapped_sogbofa_surplus(new_actions)
            new_it = it + 1
            return new_it, new_actions, new_surplus, new_k
            
        # apply the surplus to the actions until it becomes zero
        def _jax_wrapped_sogbofa_project(bool_params, hyperparams):

            # convert parameters to actions
            actions = {}
            for (var, param) in bool_params.items():
                actions[var] = jax_param_to_action(var, param['logit'], hyperparams)
            
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
                if noop[var]:
                    total_bool = total_bool + jnp.count_nonzero(action < 0.5)
                else:
                    total_bool = total_bool + jnp.count_nonzero(action > 0.5)
            excess = jnp.maximum(total_bool - allowed_actions, 0)
            
            # convert the adjusted actions back to parameters
            # reduce the excess number of parameters that are non-noop above constraint
            new_bool_params = {}            
            for (var, action) in actions.items():
                action = jnp.clip(action, min_action, max_action)
                flat_action = jnp.ravel(action)
                if noop[var]:
                    ranks = jnp.cumsum(flat_action < 0.5)
                    replace_mask = (flat_action < 0.5) & (ranks <= excess)
                else:
                    ranks = jnp.cumsum(flat_action > 0.5)
                    replace_mask = (flat_action > 0.5) & (ranks <= excess)
                flat_action = jnp.where(replace_mask, 0.5, flat_action)
                action = jnp.reshape(flat_action, jnp.shape(action))
                new_logit = jax_action_to_param(var, action, hyperparams)
                new_bool_params[var] = {'logit': new_logit}
                excess = jnp.maximum(excess - jnp.count_nonzero(replace_mask), 0)
            return new_bool_params, converged
        return _jax_wrapped_sogbofa_project


# clip whose gradient outside the bounds is kept only if descent moves back inside
# (i.e. projected gradient), so that saturated values can still recover
@jax.custom_vjp
def _jax_clip_inward(x, lower, upper):
    return jnp.clip(x, lower, upper)


def _jax_clip_inward_fwd(x, lower, upper):
    return jnp.clip(x, lower, upper), (x, lower, upper)


def _jax_clip_inward_bwd(res, grad):
    x, lower, upper = res
    lower = -jnp.inf if lower is None else lower
    upper = jnp.inf if upper is None else upper
    keep = ((x >= lower) & (x <= upper)) | ((x > upper) & (grad > 0)) | ((x < lower) & (grad < 0))
    return jnp.where(keep, grad, 0.), None, None


_jax_clip_inward.defvjp(_jax_clip_inward_fwd, _jax_clip_inward_bwd)


def _jax_bound_action(branches, lower, upper, param):
    lower = jnp.where(jnp.isfinite(lower), lower, 0.0)
    upper = jnp.where(jnp.isfinite(upper), upper, 0.0)
    f_both = lower + (upper - lower) * jax.nn.sigmoid(param)
    f_lowe = lower + jax.nn.softplus(param)
    f_uppe = upper - jax.nn.softplus(-param)
    f_none = param
    action = jnp.select(branches, [f_both, f_lowe, f_uppe, f_none])
    return action
    

class JaxStraightLinePlan(JaxPlan):
    '''A straight line plan implementation in JAX'''

    history_dependent = False
    
    def __init__(self, initializer: initializers.Initializer=initializers.normal(),
                 wrap_sigmoid: bool=True,
                 min_action_prob: float=1e-6,
                 sigmoid_weight: float=1.0,
                 wrap_non_bool: bool=False,
                 wrap_softmax: bool=False,
                 softmax_weight: float=1.0,
                 use_new_projection: bool=False,
                 max_constraint_iter: int=100) -> None:
        '''Creates a new straight line plan in JAX.
        
        :param initializer: a Jax Initializer for setting the initial actions
        :param wrap_sigmoid: wrap bool action parameters with sigmoid 
        (uses gradient clipping instead of sigmoid if None; this flag is ignored
        if wrap_softmax = True)
        :param min_action_prob: minimum value a soft boolean action can take
        (maximum is 1 - min_action_prob); required positive if wrap_sigmoid = True
        :param sigmoid_weight: weight for wrap_sigmoid operation
        :param wrap_non_bool: whether to wrap real or int action fluent parameters
        with non-linearity (e.g. sigmoid or ELU) to satisfy box constraints
        :param wrap_softmax: whether to use softmax activation approach 
        (note, this is limited to max-nondef-actions = 1) instead of projected
        gradient to satisfy action constraints 
        :param softmax_weight: weight for wrap_softmax operation
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
        self._sigmoid_weight = sigmoid_weight
        self._wrap_non_bool = wrap_non_bool
        self._wrap_softmax = wrap_softmax
        self._softmax_weight = softmax_weight
        self._use_new_projection = use_new_projection
        self._max_constraint_iter = max_constraint_iter

    def __str__(self) -> str:
        bounds = '\n        '.join(
            map(lambda kv: f'{kv[0]}: {kv[1]}', self.bounds.items()))
        return (f'[INFO] policy hyper-parameters:\n'
                f'    initializer       ={self._initializer_base}\n'
                f'    constraint-sat strategy (simple):\n'
                f'        parsed_action_bounds =\n        {bounds}\n'
                f'        wrap_sigmoid         ={self._wrap_sigmoid}\n'
                f'        wrap_sigmoid_min_prob={self._min_action_prob}\n'
                f'        sigmoid_weight       ={self._sigmoid_weight}\n'
                f'        wrap_non_bool        ={self._wrap_non_bool}\n'
                f'    constraint-sat strategy (complex):\n'
                f'        wrap_softmax        ={self._wrap_softmax}\n'
                f'        softmax_weight      ={self._softmax_weight}\n'
                f'        use_new_projection  ={self._use_new_projection}\n'
                f'        max_projection_iters={self._max_constraint_iter}\n')
    
    def compile(self, compiled: JaxRDDLCompilerWithGrad, 
                test_compiled: JaxRDDLCompiler,
                _bounds: Bounds,
                horizon: int,
                preprocessor: Optional[Preprocessor]=None) -> None:
        rddl = compiled.rddl
        
        # calculate the correct action box bounds
        shapes, bounds, cond_lists = get_action_info(compiled, _bounds, horizon)
        self.bounds = bounds
        
        # get the noop action values
        noop = {var: (values[0] if isinstance(values, list) else values)
                for (var, values) in rddl.action_fluents.items()}
        BOOL_KEY = '__bool__'
        
        # ***********************************************************************
        # STRAIGHT-LINE PLAN
        #
        # ***********************************************************************
        
        # boolean actions are parameters wrapped by sigmoid to ensure [0, 1]:
        #   action = sigmoid(weight * param)
        # here weight is a hyper-parameter and param is the trainable policy parameter
        bool_threshold = 0.0 if self._wrap_sigmoid else 0.5
        
        def _jax_bool_param_to_action(var, param, hyperparams):
            if self._wrap_sigmoid:
                return jax.nn.sigmoid(hyperparams[var] * param)
            else:
                return param 
        
        def _jax_bool_action_to_param(var, action, hyperparams):
            if self._wrap_sigmoid:
                return jnp.divide(jax.scipy.special.logit(action), hyperparams[var])
            else:
                return action
        
        # the same technique could be applied to non-bool actions following Bueno et al.
        # this is disabled by default since the gradient projection trick seems to work
        # better, especially for one-sided bounds (-inf, B) or (B, +inf)
        def _jax_non_bool_param_to_action(var, param):
            if self._wrap_non_bool:
                lower, upper = bounds[var]
                branches = [jnp.asarray(mask, dtype=compiled.REAL) 
                            for mask in cond_lists[var]]       
                return _jax_bound_action(branches, lower, upper, param)
            else:
                return param
                
        # a different option to handle boolean action concurrency constraints with |A| = 1
        # is to use a softmax activation layer over pooled action parameters
        # one noop slot is appended per allowed action, and is dropped when unstacking
        ranges = rddl.variable_ranges
        action_sizes = {var: np.prod(shape[1:], dtype=np.int64) 
                        for (var, shape) in shapes.items()
                        if ranges[var] == 'bool'}
        
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
        def _jax_wrapped_slp_predict_train(sim_state, planner_state):
            step = sim_state.step
            hyperparams = planner_state.hyperparams
            actions = {}

            # handle bool actions
            for (var, param) in planner_state.policy_params['bool'].items():
                logits = jnp.asarray(param['logit'][step, ...], dtype=compiled.REAL)
                if var == BOOL_KEY:
                    softmax_weight = hyperparams[BOOL_KEY]
                    output = jax.nn.softmax(softmax_weight * logits)
                    bool_actions = _jax_unstack_bool_from_softmax(output)
                    actions.update(bool_actions)
                else:
                    actions[var] = _jax_bool_param_to_action(var, logits, hyperparams)
                    
            # handle real and int actions
            for (var, param) in planner_state.policy_params['real'].items():
                action = jnp.asarray(param['mu'][step, ...], dtype=compiled.REAL)
                actions[var] = _jax_non_bool_param_to_action(var, action)

            return actions
        self.train_policy = _jax_wrapped_slp_predict_train
        
        # the main subroutine to compute the test rddl actions from the trainable 
        # parameters: actions are converted to correct ranges (i.e. bool, int, float)
        def _jax_wrapped_slp_predict_test(sim_state, planner_state):
            step = sim_state.step
            actions = {}

            # handle bool actions
            for (var, param) in planner_state.policy_params['bool'].items():
                logits = jnp.asarray(param['logit'][step, ...], dtype=compiled.REAL)
                if var == BOOL_KEY:
                    output = jax.nn.one_hot(
                        jnp.argmax(logits), jnp.shape(logits)[-1], dtype=compiled.REAL)
                    bool_actions = _jax_unstack_bool_from_softmax(output)
                    for (bool_var, bool_action) in bool_actions.items():
                        actions[bool_var] = bool_action > 0.5
                else:
                    actions[var] = logits > bool_threshold

            # handle real and int actions
            for (var, param) in planner_state.policy_params['real'].items():
                action = jnp.asarray(param['mu'][step, ...], dtype=compiled.REAL)
                action = _jax_non_bool_param_to_action(var, action)
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
        def _jax_project_bool_to_box(var, param, hyperparams):
            min_p = self._min_action_prob
            lower = _jax_bool_action_to_param(var, min_p, hyperparams)
            upper = _jax_bool_action_to_param(var, 1. - min_p, hyperparams)
            return jnp.clip(param, lower, upper)
        
        def _jax_project_to_box(params, hyperparams):
            new_params = {'bool': {}, 'real': {}}

            # box projection for boolean actions
            for (var, param) in params['bool'].items():
                if var == BOOL_KEY:
                    new_params['bool'][var] = param
                else:
                    new_logit = _jax_project_bool_to_box(var, param['logit'], hyperparams)
                    new_params['bool'][var] = {'logit': new_logit}
                
            # box projection for real actions
            for (var, param) in params['real'].items():
                new_mean = param['mu']
                if not self._wrap_non_bool:
                    new_mean = jnp.clip(new_mean, *bounds[var])
                new_params['real'][var] = {'mu': new_mean}
            return new_params
        
        def _jax_wrapped_slp_project_to_box(planner_state):
            params = _jax_project_to_box(
                planner_state.policy_params, planner_state.hyperparams)  
            converged = jnp.array(True, dtype=jnp.bool_)
            return params, converged  
        
        # enable constraint satisfaction subroutines during optimization 
        # if there are nontrivial concurrency constraints in the problem description 
        allowed_actions = rddl.max_allowed_actions
        bool_action_count = sum(np.size(values)
                                for (var, values) in rddl.action_fluents.items()
                                if rddl.variable_ranges[var] == 'bool')    
        
        if compiled.print_warnings and allowed_actions < bool_action_count: 
            print(termcolor.colored(
                f'[INFO] Number of boolean actions {bool_action_count} '
                f'> cardinality {allowed_actions}: enabling projected gradient to '
                f'satisfy constraints on action-fluents.', 'dark_grey'
            ))
        
        # use a softmax output activation: only allow one action non-noop for now
        if allowed_actions < bool_action_count and self._wrap_softmax:
            if allowed_actions > 1:
                raise RDDLNotImplementedError(
                    f'SLPs with wrap_softmax currently '
                    f'do not support action cardinality constraint {allowed_actions} > 1.'
                )
            self.projection = _jax_wrapped_slp_project_to_box
            
        # use gradient projection method
        elif allowed_actions < bool_action_count:

            # apply chosen projection method
            if self._use_new_projection:
                jax_project_fn = JaxSortingActionProjection().compile(
                    ranges, noop, self._wrap_sigmoid, allowed_actions, bool_threshold, 
                    _jax_project_bool_to_box
                )
            else:
                 jax_project_fn = JaxSogbofaActionProjection().compile(
                    ranges, noop, allowed_actions, self._max_constraint_iter, 
                    _jax_bool_param_to_action, _jax_bool_action_to_param, 
                    self._min_action_prob, 1. - self._min_action_prob, compiled.REAL
                )

            # clip actions to valid bounds and satisfy constraint on max actions
            def _jax_wrapped_slp_project_to_max_constraint(planner_state):
                params, _ = _jax_wrapped_slp_project_to_box(planner_state)
                bool_params, converged = jax.vmap(jax_project_fn, in_axes=(0, None))(
                    params['bool'], planner_state.hyperparams)
                new_params = {'bool': bool_params, 'real': params['real']}
                converged = jnp.all(converged)
                return new_params, converged
            self.projection = _jax_wrapped_slp_project_to_max_constraint
        
        # just project to box constraints
        else: 
            self.projection = _jax_wrapped_slp_project_to_box
            
        # ***********************************************************************
        # PLAN INITIALIZATION
        #
        # ***********************************************************************
        
        # use the user required initializer and project actions to feasible range
        init_fn = self._initializer
        stack_bool_params = allowed_actions < bool_action_count and self._wrap_softmax

        def _jax_wrapped_slp_init(sim_state):
            key = sim_state.key
            hyperparams = {var: self._sigmoid_weight for var in rddl.action_fluents}
            params = {'bool': {}, 'real': {}}
            for (var, shape) in shapes.items():

                # bool logit initialization
                if ranges[var] == 'bool':
                    if not stack_bool_params:
                        key, subkey = random.split(key)
                        param = init_fn(key=subkey, shape=shape, dtype=compiled.REAL)
                        param = param + bool_threshold
                        params['bool'][var] = {'logit': param}
                
                # non-bool action initialization
                else:
                    key, subkey = random.split(key)
                    param_mu = init_fn(key=subkey, shape=shape, dtype=compiled.REAL)
                    params['real'][var] = {'mu': param_mu}
            
            # init stacked bool actions as one tensor
            if stack_bool_params:
                shape = (horizon, bool_action_count + allowed_actions)
                key, subkey = random.split(key)
                param = init_fn(key=subkey, shape=shape, dtype=compiled.REAL)
                params['bool'][BOOL_KEY] = {'logit': param}
                hyperparams[BOOL_KEY] = self._softmax_weight

            params = _jax_project_to_box(params, hyperparams)
            return params, hyperparams
        self.initializer = _jax_wrapped_slp_init
    
    @staticmethod
    @jax.jit
    def _guess_next_epoch(param):
        return jnp.append(param[1:, ...], param[-1:, ...], axis=0)

    def guess_next_epoch(self, params: Pytree) -> Pytree:
        # "progress" the plan one step forward and set last action to second-last
        return jax.tree_util.tree_map(JaxStraightLinePlan._guess_next_epoch, params)


# ***********************************************************************
# JAX DEEP REACTIVE POLICY
#
# ***********************************************************************


class GumbelSoftmaxTopK(nn.Module):
    '''A differentiable top-k generalization of gumbel-softmax.'''

    allowed_actions: int
    dtype: Any
    name: str='topk'

    @nn.compact
    def __call__(self, rng_key, action_logits, weight: float, train: bool):
        
        # forward relaxation (for k = 1 this is the gumbel-softmax)
        def soft_top_k(key, logits, w):
            g = random.gumbel(key=key, shape=jnp.shape(logits), dtype=self.dtype)
            logits = logits + g
            def body_fun(i, carry):
                khot, logits_i = carry
                logits_new = logits_i + safe_log(1.0 - khot)
                khot_new = khot + jax.nn.softmax(w * logits_new)
                return (khot_new, logits_new)
            khot0 = jnp.zeros_like(logits)
            khot_final, _ = jax.lax.fori_loop(
                lower=0, upper=self.allowed_actions, 
                body_fun=body_fun, 
                init_val=(khot0, logits)
            )
            return khot_final
        
        # forward exact evaluation
        def hard_top_k(key, logits, w):
            def body_fun(i, carry):
                khot, logits_i = carry
                idx = jnp.argmax(logits_i)
                logits_new = logits_i.at[idx].set(-jnp.inf)
                khot_new = khot.at[idx].set(1.0)
                return (khot_new, logits_new)
            khot0 = jnp.zeros_like(logits)
            khot_final, _ = jax.lax.fori_loop(
                lower=0, upper=self.allowed_actions, 
                body_fun=body_fun, 
                init_val=(khot0, logits)
            )
            return khot_final

        return jax.lax.switch(train, [hard_top_k, soft_top_k], rng_key, action_logits, weight)


class FiLM(nn.Module):
    '''FiLM layer for time conditioning a dense layer.'''
    hidden_dim: int
    name: str

    @nn.compact
    def __call__(self, h, cond):
        gamma = nn.Dense(self.hidden_dim, name='gamma')(cond)
        beta = nn.Dense(self.hidden_dim, name='beta')(cond)
        return gamma * h + beta


class FixedTimeEmbedding(nn.Module):
    '''Learnable fixed time embedding layer.'''
    max_t: int=500
    dim: int=8
    init: object=nn.initializers.normal()
    name: str='fixed_time_embed'

    @nn.compact
    def __call__(self, t):
        emb = self.param("kernel", self.init, (self.max_t + 1, self.dim))
        return emb[t]
    

class SinusoidalTimeEmbedding(nn.Module):
    '''Sinusoidal positional encoding for time.'''
    dim: int=8
    name: str = 'sin_time_embed'

    @nn.compact
    def __call__(self, t):
        t = jnp.asarray(t, dtype=jnp.float32)
        freqs = 10000.0 ** (-2.0 * jnp.arange(self.dim) / self.dim)
        angles = t * freqs
        emb = jnp.stack([jnp.sin(angles), jnp.cos(angles)], axis=-1)
        return jnp.ravel(emb)[:self.dim]
    
            
class JaxDeepReactivePolicy(JaxPlan):
    '''A deep reactive policy network implementation in JAX.'''

    history_dependent = False
    
    def __init__(self, topology: Optional[Sequence[int]]=None,
                 activation: Activation=jnp.tanh,
                 initializer: nn.initializers.Initializer=nn.initializers.variance_scaling(
                    scale=2.0, mode="fan_in", distribution="truncated_normal"),
                 normalize: bool=False,
                 normalize_per_layer: bool=False,
                 normalizer_kwargs: Optional[Kwargs]=None,
                 sigmoid_weight: float=1.0,
                 wrap_non_bool: bool=False,
                 softmax_weight: float=1.0,
                 time_dependent: bool=False,
                 time_embedding: Optional[Type]=SinusoidalTimeEmbedding,
                 time_embedding_kwargs: Optional[Kwargs]=None) -> None:
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
        :param sigmoid_weight: weight for sigmoid operation on boolean action parameters
        :param wrap_non_bool: whether to wrap real or int action fluent parameters
        with non-linearity (e.g. sigmoid or ELU) to satisfy box constraints
        :param softmax_weight: weight in softmax action constraint satisfaction
        :param time_dependent: whether to make the DRP time_dependent
        :param time_embedding: time embedding for time dependent policy
        :param time_embedding_kwargs: arguments to pass to time embedding on initialization
        '''
        super(JaxDeepReactivePolicy, self).__init__()
        
        if topology is None:
            topology = [64, 64]
        self._topology = topology
        self._activations = [activation for _ in topology]
        self._initializer_base = initializer
        self._initializer = initializer
        self._normalize = normalize
        self._normalize_per_layer = normalize_per_layer
        if normalizer_kwargs is None:
            normalizer_kwargs = {'use_bias': True, 'use_scale': True}
        self._normalizer_kwargs = normalizer_kwargs
        self._sigmoid_weight = sigmoid_weight
        self._wrap_non_bool = wrap_non_bool
        self._softmax_output_weight = softmax_weight

        # for time embedding
        self._time_dependent = time_dependent
        self._time_embedding = time_embedding
        if time_embedding_kwargs is None:
            time_embedding_kwargs = {}
        self._time_embedding_kwargs = time_embedding_kwargs
    
    def __str__(self) -> str:
        bounds = '\n        '.join(
            map(lambda kv: f'{kv[0]}: {kv[1]}', self.bounds.items()))
        return (f'[INFO] policy hyper-parameters:\n'
                f'    topology          ={self._topology}\n'
                f'    activation_fn     ={self._activations[0].__name__}\n'
                f'    initializer       ={type(self._initializer_base).__name__}\n'
                f'    time embedding:\n'
                f'        time_dependent   ={self._time_dependent}\n'
                f'        time_embedding   ={self._time_embedding}\n'
                f'        time_embed_kwargs={self._time_embedding_kwargs}\n'
                f'    input norm:\n'
                f'        apply_input_norm    ={self._normalize}\n'
                f'        input_norm_layerwise={self._normalize_per_layer}\n'
                f'        input_norm_args     ={self._normalizer_kwargs}\n'
                f'    constraint-sat strategy:\n'
                f'        parsed_action_bounds=\n        {bounds}\n'
                f'        sigmoid_weight      ={self._sigmoid_weight}\n'
                f'        wrap_non_bool       ={self._wrap_non_bool}\n'
                f'        softmax_weight      ={self._softmax_output_weight}\n')
        
    def compile(self, compiled: JaxRDDLCompilerWithGrad, 
                test_compiled: JaxRDDLCompiler,
                _bounds: Bounds,
                horizon: int,
                preprocessor: Optional[Preprocessor]=None) -> None:
        rddl = compiled.rddl
        
        # calculate the correct action box bounds
        shapes, bounds, cond_lists = get_action_info(compiled, _bounds, horizon)
        shapes = {var: value[1:] for (var, value) in shapes.items()}
        self.bounds = bounds
        
        # get the noop action values
        noop = {var: (values[0] if isinstance(values, list) else values)
                for (var, values) in rddl.action_fluents.items()}                   
        BOOL_KEY = '__bool__'
        
        # ***********************************************************************
        # POLICY NETWORK PREDICTION
        #
        # ***********************************************************************
            
        # compute the correct shapes of the output layers based on the action-fluent shape
        ranges = rddl.variable_ranges
        layers = list(enumerate(zip(self._topology, self._activations)))
        layer_sizes = {var: np.prod(shape, dtype=np.int64) 
                       for (var, shape) in shapes.items()}
        layer_names = {var: var.replace('-', '_') for var in shapes}
        
        # inputs for the policy network are states for fully observed and obs for POMDPs
        if rddl.observ_fluents:
            observed_vars = rddl.observ_fluents
        else:
            observed_vars = rddl.state_fluents
        input_names = {var: var.replace('-', '_') for var in observed_vars}
        
        # catch if input norm is applied to size 1 tensor:
        # this leads to incorrect behavior as the input is always "1"
        normalize = self._normalize
        normalize_per_layer = self._normalize_per_layer
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
        normalizer_kwargs = self._normalizer_kwargs

        class DRPStateLayer(nn.Module):
            name: str='inputs'

            @nn.compact
            def __call__(self, fls, hyperparams):

                # optional state preprocessing
                if preprocessor is not None:
                    stats = hyperparams[Preprocessor.HYPERPARAMS_KEY]
                    fls = preprocessor.transform(fls, stats)
                
                # concatenate all state variables into a single vector
                # optionally apply layer norm to each input tensor
                states_bool, states_non_bool = [], []
                non_bool_dims = 0
                for (var, value) in fls.items():
                    if var in observed_vars:
                        state = jnp.ravel(value)
                        if ranges[var] == 'bool':
                            states_bool.append(state)
                        else:
                            if normalize and normalize_per_layer:
                                name = f'norm_{input_names[var]}'
                                state = nn.LayerNorm(name=name, **normalizer_kwargs)(state)
                            states_non_bool.append(state)
                            non_bool_dims = non_bool_dims + state.size
                state = jnp.concatenate(states_non_bool + states_bool)
                
                # optionally perform layer normalization on the non-bool inputs
                if normalize and not normalize_per_layer and non_bool_dims:
                    norm_layer = nn.LayerNorm(name='norm', **normalizer_kwargs)
                    normalized = norm_layer(state[:non_bool_dims])
                    state = state.at[:non_bool_dims].set(normalized)
                return state
        
        # enable constraint satisfaction subroutines during optimization 
        # if there are nontrivial concurrency constraints in the problem description
        bool_action_count = sum(np.size(values)
                                for (var, values) in rddl.action_fluents.items()
                                if rddl.variable_ranges[var] == 'bool')
        use_constraint_satisfaction = rddl.max_allowed_actions < bool_action_count
        SOFTMAX_KEY = 'softmax_weight'
        SIGMOID_KEY = 'sigmoid_weight'

        # predict actions from the policy network for current state
        wrap_non_bool = self._wrap_non_bool
        time_dependent = self._time_dependent
        time_embedding = self._time_embedding
        time_embedding_kwargs = self._time_embedding_kwargs
        init = self._initializer

        class DRP(nn.Module):
            name: str='drp'

            @nn.compact
            def __call__(self, fls, hyperparams, step, train):
                key = self.make_rng("policy")

                # input layer
                state = DRPStateLayer(name='inputs')(fls, hyperparams)
                
                # time embedding
                if time_dependent:
                    t_emb = time_embedding(**time_embedding_kwargs)(step)

                # feed state vector through hidden layers
                hidden = state
                for (i, (num_neuron, activation)) in layers:
                    name = f'dense_{i}'
                    linear = nn.Dense(features=num_neuron, kernel_init=init, name=name)
                    hidden = activation(linear(hidden))
                    if time_dependent:
                        hidden = FiLM(num_neuron, name=f'film_{i}')(hidden, t_emb)
                
                # each output is a linear layer reshaped to original lifted shape
                actions = {}
                for (var, size) in layer_sizes.items():
                    
                    # for bool action, create logit head with sigmoid activation
                    if ranges[var] == 'bool':
                        if not use_constraint_satisfaction:

                            # linear head for logits
                            name = f'dense_{layer_names[var]}_logit'
                            linear = nn.Dense(features=size, kernel_init=init, name=name)
                            logits = linear(hidden)
                            logits = jnp.reshape(logits, logits.shape[:-1] + shapes[var])
                            if not shapes[var]:
                                logits = jnp.squeeze(logits)
                            actions[var] = jax.nn.sigmoid(hyperparams[SIGMOID_KEY] * logits)
                    
                    # for real and int action, create mean head
                    # project to valid ranges using Bueno et al. 2019 method
                    else:
                        # linear head for mean action
                        name = f'dense_{layer_names[var]}_mu'
                        linear = nn.Dense(features=size, kernel_init=init, name=name)
                        action = linear(hidden)
                        action = jnp.reshape(action, action.shape[:-1] + shapes[var])
                        if not shapes[var]:
                            action = jnp.squeeze(action)

                        # project to valid ranges using Bueno et al. 2019 method
                        if wrap_non_bool:
                            lower, upper = bounds[var]
                            branches = [jnp.asarray(mask, dtype=compiled.REAL) 
                                        for mask in cond_lists[var]]       
                            actions[var] = _jax_bound_action(branches, lower, upper, action)
                        else:
                            actions[var] = action
                
                # for constraint satisfaction wrap bool actions with softmax
                # one noop slot is added per allowed action so that up to k actions are
                # selected, and the noop slots are dropped after selection
                if use_constraint_satisfaction:
                    num_noop_slots = rddl.max_allowed_actions
                    linear = nn.Dense(
                        features=bool_action_count + num_noop_slots, kernel_init=init,
                        name='dense_topk')
                    logits = linear(hidden)
                    weight = hyperparams[SOFTMAX_KEY]
                    topk = GumbelSoftmaxTopK(
                        allowed_actions=rddl.max_allowed_actions, dtype=compiled.REAL)
                    output = topk(key, logits, weight, train)
                    actions[BOOL_KEY] = output[..., :bool_action_count]
                return actions
        predict_fn = DRP(name='drp')       
        
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
        def _jax_wrapped_drp_predict_train(sim_state, planner_state, train=1):
            actions = predict_fn.apply(
                planner_state.policy_params, sim_state.fls, planner_state.hyperparams, 
                sim_state.step, train=train, rngs={"policy": sim_state.key})
            if not wrap_non_bool:
                for (var, action) in actions.items():
                    if var != BOOL_KEY and ranges[var] != 'bool':
                        actions[var] = _jax_clip_inward(action, *bounds[var])
            if use_constraint_satisfaction:
                bool_actions = _jax_unstack_bool_from_softmax(actions[BOOL_KEY])
                actions.update(bool_actions)
                del actions[BOOL_KEY]
            return actions
        self.train_policy = _jax_wrapped_drp_predict_train
        
        # the main subroutine to compute the test rddl actions from the trainable 
        # parameters and state/obs dict: the difference here is that actions are converted 
        # to their required types (i.e. bool, int, float)
        def _jax_wrapped_drp_predict_test(sim_state, planner_state):
            actions = _jax_wrapped_drp_predict_train(sim_state, planner_state, train=0)
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
        def _jax_wrapped_drp_no_projection(planner_state):
            params = planner_state.policy_params
            converged = jnp.array(True, dtype=jnp.bool_)
            return params, converged
        self.projection = _jax_wrapped_drp_no_projection
    
        # ***********************************************************************
        # POLICY NETWORK INITIALIZATION
        #
        # ***********************************************************************
        
        # initialize policy parameters according to user-desired weight initializer
        def _jax_wrapped_drp_init(sim_state):
            hyperparams = {SOFTMAX_KEY: self._softmax_output_weight,
                           SIGMOID_KEY: self._sigmoid_weight}
            if preprocessor is not None:
                hyperparams[Preprocessor.HYPERPARAMS_KEY] = preprocessor.initialize()
            obs_vars = {var: value[0, ...] 
                        for (var, value) in sim_state.fls.items()
                        if var in observed_vars}
            params = predict_fn.init(sim_state.key, obs_vars, hyperparams, 0, train=1)
            return params, hyperparams
        self.initializer = _jax_wrapped_drp_init
        
    def guess_next_epoch(self, params: Pytree) -> Pytree:
        # this is easy: just warm-start from the previously obtained policy
        return params


# ***********************************************************************
# JAX RDDL policy
#
# ***********************************************************************


class JaxRDDLPolicy(JaxPlan):
    '''A structured policy whose structure is defined in RDDL policy block.'''

    history_dependent = False
    
    def compile(self, compiled: JaxRDDLCompilerWithGrad, 
                test_compiled: JaxRDDLCompiler,
                _bounds: Bounds,
                horizon: int,
                preprocessor: Optional[Preprocessor]=None) -> None:
        rddl = compiled.rddl

        # calculate the correct action box bounds
        shapes, bounds, _ = get_action_info(compiled, _bounds, horizon)
        shapes = {var: value[1:] for (var, value) in shapes.items()}
        self.bounds = bounds

        # ***********************************************************************
        # POLICY PREDICTION
        #
        # ***********************************************************************
            
        if rddl.policy is None:
            raise ValueError('RDDL domain does not have a valid policy block.')

        # trainable parameters are the non-fluents in the policy
        param_fluent_vars = set(rddl.policy.param_fluents.keys())
        if not param_fluent_vars:
            raise ValueError('RDDL policy has no param-fluent(s) to optimize.')
        print(termcolor.colored(
            f'[INFO] JaxPlan will use the policy defined in the policy block: '
            f'param-fluents {param_fluent_vars} will be optimized.', 'dark_grey'
        ))

        # functions that evaluate the policy come directly from the compiled policy block
        train_policy_fn = compiled._jax_policy()
        test_policy_fn = test_compiled._jax_policy()

        # to evaluate the train and test action, params -> nfls, hyperparams -> params
        def _jax_wrapped_rddl_predict_train(sim_state, planner_state):
            nfls = {**sim_state.nfls, **planner_state.policy_params}
            policy_fls = train_policy_fn(
                sim_state.key, 0, sim_state.fls, nfls, planner_state.hyperparams)[0]
            new_fls = {}
            for (var, value) in policy_fls.items():
                if var in rddl.action_fluents:
                    if not shapes[var]:
                        value = jnp.squeeze(value)
                    value = _jax_clip_inward(value, *bounds[var])
                    value = jnp.asarray(value, dtype=compiled.REAL)
                new_fls[var] = value
            return new_fls
        self.train_policy = _jax_wrapped_rddl_predict_train

        def _jax_wrapped_rddl_cast_param_fluents(params):
            new_params = {}
            for (var, value) in params.items():
                prange = rddl.policy.variable_ranges[var]
                if prange == 'real':
                    new_params[var] = value
                elif prange == 'bool':
                    new_params[var] = value > 0.5
                else:
                    new_params[var] = jnp.asarray(jnp.round(value), dtype=test_compiled.INT)
            return new_params

        def _jax_wrapped_rddl_predict_test(sim_state, planner_state):
            params = _jax_wrapped_rddl_cast_param_fluents(planner_state.policy_params)
            nfls = {**sim_state.nfls, **params}
            policy_fls = test_policy_fn(
                sim_state.key, 0, sim_state.fls, nfls, planner_state.hyperparams)[0]
            new_fls = {}
            for (var, value) in policy_fls.items():
                if var in rddl.action_fluents:
                    if not shapes[var]:
                        value = jnp.squeeze(value)
                    prange = rddl.variable_ranges[var]
                    if prange == 'bool':
                        value = value > 0.5
                    elif prange == 'real':
                        value = jnp.clip(value, *bounds[var])
                        value = jnp.asarray(value, dtype=test_compiled.REAL)
                    elif prange == 'int' or prange in rddl.enum_types:
                        value = jnp.clip(value, *bounds[var])
                        value = jnp.asarray(jnp.round(value), dtype=test_compiled.INT)
                new_fls[var] = value
            return new_fls
        self.test_policy = _jax_wrapped_rddl_predict_test
        
        # ***********************************************************************
        # ACTION CONSTRAINT SATISFACTION
        #
        # ***********************************************************************
        
        # params are optimized non-fluents and must be clipped to their valid ranges
        # currently this is just done for bool
        def _jax_wrapped_rddl_clip_projection(planner_state):
            params = {}
            for (var, param) in planner_state.policy_params.items():
                if rddl.policy.variable_ranges[var] == 'bool':
                    param = jnp.clip(param, 0., 1.)
                params[var] = param
            converged = jnp.array(True, dtype=jnp.bool_)
            return params, converged
        self.projection = _jax_wrapped_rddl_clip_projection

        # ***********************************************************************
        # POLICY NETWORK INITIALIZATION
        #
        # ***********************************************************************
        
        # params <- default non-fluent values, hyperparams <- model-params
        def _jax_wrapped_rddl_init(sim_state):
            hyperparams = compiled.model_aux['params']
            params = {
                name: jnp.asarray(compiled.init_policy_values[name], dtype=compiled.REAL) 
                for name in param_fluent_vars
            }
            return params, hyperparams
        self.initializer = _jax_wrapped_rddl_init

    def guess_next_epoch(self, params: Pytree) -> Pytree:
        # this is easy: just warm-start from the previously obtained policy
        return params
