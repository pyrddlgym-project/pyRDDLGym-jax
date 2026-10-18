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
# [7] Schaul, Tom, Tobias Glasmachers, and Jurgen Schmidhuber. "High dimensions and 
# heavy tails for natural evolution strategies." In Proceedings of the 13th Annual 
# Conference on Genetic and Evolutionary Computation, pp. 845-852. 2011.
#
# [8] Loshchilov, Ilya, Tobias Glasmachers, and Hans-Georg Beyer. "Large scale black-box
# optimization by limited-memory matrix adaptation." IEEE Transactions on Evolutionary
# Computation 23, no. 2 (2019): 353-358.
#
# [9] Akimoto, Youhei, and Nikolaus Hansen. "Projection-based restricted covariance
# matrix adaptation for high dimension." In Proceedings of the Genetic and Evolutionary
# Computation Conference 2016, pp. 197-204. 2016.
#
# [10] Wang, Hao, Michael Emmerich, and Thomas Back. "Mirrored orthogonal sampling for
# covariance matrix adaptation evolution strategies." Evolutionary Computation 27,
# no. 4 (2019): 699-725.
#
# ***********************************************************************


from abc import ABCMeta, abstractmethod
from collections import deque
import copy
from enum import Enum
from functools import partial
import inspect
import sys
import time
from typing import Any, Callable, Dict, Generator, Optional, Tuple, Union

import jax
import jax.flatten_util
import jax.numpy as jnp
import jax.random as random
import flax
import numpy as np
import optax
import termcolor
from tqdm import tqdm, TqdmWarning
import warnings
warnings.filterwarnings("ignore", category=TqdmWarning)

from pyRDDLGym.core.compiler.model import RDDLPlanningModel, RDDLLiftedModel
from pyRDDLGym.core.debug.logger import Logger

from pyRDDLGym_jax import __version__
from pyRDDLGym_jax.core.compiler import JaxRDDLCompiler, JaxRDDLSimState
from pyRDDLGym_jax.core.logic import JaxRDDLCompilerWithGrad, DefaultJaxRDDLCompilerWithGrad
from pyRDDLGym_jax.core.plan import Preprocessor, JaxPlan
from pyRDDLGym_jax.core.utility import create_utility

Bounds = Dict[str, Tuple[np.ndarray, np.ndarray]]
Kwargs = Dict[str, Any]
Pytree = Any


# ***********************************************************************
# SUPPORTING FUNCTIONS
# 
# - smoothed mean calculation
# - planner status
# - stopping criteria
# - optimizer construction and chaining with optax  
# - action repair for contraints
#
# ***********************************************************************


@flax.struct.dataclass
class JaxPlannerState:
    '''The JAX planner state representation.'''
    policy_params: Pytree=None
    hyperparams: Pytree=None
    opt_state: Pytree=None
    opt_aux: Pytree=None
    critic_params: Pytree=None
    progress: float=None
    reward_mask: jnp.ndarray=None


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
    
    def get_type(self) -> str:
        if self.value in {5, 6}:
            return 'info'
        elif self.value in {0, 1}:
            return 'success'
        elif self.value in {2, 3}:
            return 'warning'
        elif self.value == 4:
            return 'error'
        else:
            raise Exception(f'Invalid value {self.value}.')


class RollingMean:
    '''Maintains the rolling mean of a stream of real-valued observations.'''
    
    def __init__(self, window_size: int, exact: bool=False) -> None:
        self._window_size = window_size
        self._exact = exact
        self._memory = deque(maxlen=window_size)
        self._total = 0
    
    def update(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        memory = self._memory
        self._total = self._total + x
        if len(memory) == self._window_size:
            self._total = self._total - memory.popleft()
        memory.append(x)
        if self._exact or not np.all(np.isfinite(self._total)):
            self._total = np.sum(np.stack(memory), axis=0)
        return self._total / len(memory)
    

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
        if self.callback is None or \
        callback['last_iteration_improved'] > self.callback['last_iteration_improved']:
            self.callback = callback
            self.iters_since_last_update = 0
        else:
            self.iters_since_last_update += 1
        return self.iters_since_last_update >= self.patience
    
    def __str__(self) -> str:
        return f'No improvement for {self.patience} iterations'
        

def _build_optax_optimizer(optimizer, optimizer_kwargs, clip_grad, noise_kwargs,
                           line_search_kwargs, ema_decay, reduce_on_plateau_kwargs):

    # set optimizer: integer arguments cannot be traced
    try:
        arguments = inspect.signature(optimizer).bind(**optimizer_kwargs)
        arguments.apply_defaults()
        static_args = [name for (name, value) in arguments.arguments.items()
                       if isinstance(value, int) and not isinstance(value, bool)]
        optimizer = optax.inject_hyperparams(
            optimizer, static_args=static_args)(**optimizer_kwargs)
        injected = True
    except Exception as _:
        print(termcolor.colored(
            '[WARN] Could not inject hyperparameters into JaxPlan optimizer: '
            'runtime modification of hyperparameters will be disabled.', 'yellow'
        ))
        optimizer = optimizer(**optimizer_kwargs)   
        injected = False
    
    # apply optimizer chain of transformations
    pipeline = []
    use_loss_metric = False
    if clip_grad is not None:
        pipeline.append(optax.clip_by_global_norm(clip_grad))
    if noise_kwargs is not None:
        pipeline.append(optax.add_noise(**noise_kwargs))
    pipeline.append(optimizer)
    if line_search_kwargs is not None:
        pipeline.append(optax.scale_by_zoom_linesearch(**line_search_kwargs))
    if reduce_on_plateau_kwargs is not None:
        rop = getattr(getattr(optax, 'contrib', None), 'reduce_on_plateau', None)
        if rop is None:
            print(termcolor.colored(
                '[WARN] optax.contrib.reduce_on_plateau is unavailable: '
                'disabling reduce_on_plateau_kwargs.', 'yellow'
            ))
        else:
            pipeline.append(rop(**reduce_on_plateau_kwargs))
            use_loss_metric = True
    if ema_decay is not None:
        pipeline.append(optax.ema(ema_decay))
    return optax.chain(*pipeline), injected, use_loss_metric


@jax.jit
def _unbatched_pytree(tree: Pytree, i: int) -> Pytree:
    return jax.tree_util.tree_map(lambda x: x[i], tree)


def _jax_zero_tree(tree: Pytree) -> bool:
    leaves, _ = jax.tree_util.tree_flatten(
        jax.tree_util.tree_map(partial(jnp.allclose, b=0), tree))
    return jnp.all(jnp.asarray(leaves))


class JaxActionRepair:
    '''Repairs actions to satisfy the action preconditions of the exact model by moving
    them towards the default actions.'''

    def __init__(self, compiled: JaxRDDLCompiler, print_warnings: bool=True) -> None:
        '''Creates a new repair of the actions of the given exact model, which must be
        compiled with its constraints.

        :param compiled: the compiled exact model
        :param print_warnings: whether to print warnings
        '''
        self.compiled = compiled

        # default actions, with enum values as their indices
        rddl = compiled.rddl
        ranges = rddl.variable_ranges
        defaults = {}
        for var in rddl.action_fluents:
            value = np.asarray(compiled.init_values[var])
            if value.dtype.type is np.str_:
                value = rddl.object_string_to_index_array(ranges[var], value)
            defaults[var] = value
        self._defaults = defaults

        # repaired actions are guaranteed feasible only if the default actions are
        if print_warnings:
            default_actions = {var: compiled.init_values[var] for var in defaults}
            if not compiled.constraints.sim.check_action_preconditions(
                    default_actions, silent=True):
                print(termcolor.colored(
                    '[WARN] The default actions violate the preconditions in the initial '
                    'state: repaired actions may still violate them.', 'yellow'))

        self._repair = self._jax_repair()

    def _jax_repair(self) -> Callable:
        '''Returns the function (sim_state, actions, scores) -> (actions, info, kept, step)
        that repairs the actions in the state of the given simulation state, where the
        scores are the values in [0, 1] of the boolean actions.'''
        compiled = self.compiled
        REAL, INT = compiled.REAL, compiled.INT
        NORMAL = JaxRDDLCompiler.ERROR_CODES['NORMAL']
        precond_fn = compiled._jax_preconditions()
        violations_fn = compiled._jax_precondition_violations(compiled.model_aux)
        params = compiled.model_aux['params']
        ranges = compiled.rddl.variable_ranges
        shapes = {var: np.shape(value) for (var, value) in compiled.init_values.items()}
        sizes = {var: int(np.size(value)) for (var, value) in compiled.init_values.items()}

        # flat default actions
        defaults = self._defaults
        bool_vars = [var for var in defaults if ranges[var] == 'bool']
        other_vars = [var for var in defaults if ranges[var] != 'bool']
        bool_default = np.asarray(np.concatenate(
            [np.ravel(defaults[var]) for var in bool_vars] + [np.zeros((0,), dtype=bool)]),
            dtype=bool)
        other_default = np.asarray(np.concatenate(
            [np.ravel(defaults[var]) for var in other_vars] + [np.zeros((0,))]), 
            dtype=REAL)
        bool_splits = np.cumsum([sizes[var] for var in bool_vars])[:-1]
        other_splits = np.cumsum([sizes[var] for var in other_vars])[:-1]

        # bisection stops at the square root of the precision of the model, so that the
        # repaired actions keep a margin from the boundary of the preconditions against
        # rounding differences between evaluations of the preconditions (e.g. in float64)
        num_bisections = (np.finfo(REAL).nmant + 1) // 2

        def _jax_wrapped_feasible(sim_state, fls, actions):
            fls = {**fls, **actions}
            return precond_fn(sim_state.key, NORMAL, fls, sim_state.nfls, params)[0]

        def _jax_wrapped_violations(sim_state, fls, actions):
            fls = {**fls, **actions}
            return violations_fn(fls, sim_state.nfls, params, sim_state.key)

        def _jax_wrapped_unflatten(flat, splits, names):
            parts = jnp.split(flat, splits)
            return {var: jnp.reshape(part, shapes[var]) 
                    for (var, part) in zip(names, parts)}

        # the state given to the policy outside the rollouts has the shapes of the
        # environment, which are made the shapes of the compiled model
        def _jax_wrapped_model_fls(fls):
            result = {}
            for (var, value) in fls.items():
                if var in shapes and jnp.shape(value) != shapes[var] and \
                jnp.size(value) == sizes[var]:
                    value = jnp.reshape(value, shapes[var])
                result[var] = value
            return result

        def _jax_wrapped_repair(sim_state, actions, scores):
            fls = {**_jax_wrapped_model_fls(sim_state.fls), **actions}
            infeasible = jnp.logical_not(_jax_wrapped_feasible(sim_state, fls, {}))
            resets = jnp.zeros((), dtype=REAL)
            kept = {var: jnp.ones(shapes[var], dtype=jnp.bool_) for var in bool_vars}
            step = {}

            # reset the boolean actions that differ from defaults in order of confidence
            if bool_vars:
                flat = jnp.concatenate(
                    [jnp.ravel(jnp.asarray(actions[var], dtype=jnp.bool_))
                     for var in bool_vars])
                score = jnp.concatenate(
                    [jnp.ravel(jnp.asarray(scores[var], dtype=REAL)) 
                     for var in bool_vars])
                flat_default = jnp.asarray(bool_default)
                active = flat != flat_default
                order = jnp.argsort(jnp.where(active, jnp.abs(score - 0.5), jnp.inf))

                def _jax_wrapped_targeted_reset(i, carry):
                    flat, violations = carry
                    index = order[i]
                    candidate = flat.at[index].set(flat_default[index])
                    unflat = _jax_wrapped_unflatten(candidate, bool_splits, bool_vars)
                    cand_violations = _jax_wrapped_violations(sim_state, fls, unflat)
                    reset = active[index] & (violations > 0) & (cand_violations < violations)
                    return (jnp.where(reset, candidate, flat),
                            jnp.where(reset, cand_violations, violations))

                def _jax_wrapped_reset(i, flat):
                    index = order[i]
                    unflat = _jax_wrapped_unflatten(flat, bool_splits, bool_vars)
                    unflat_feas = _jax_wrapped_feasible(sim_state, fls, unflat)
                    reset = active[index] & jnp.logical_not(unflat_feas)
                    candidate = flat.at[index].set(flat_default[index])
                    return jnp.where(reset, candidate, flat)

                initial = flat
                unflats = _jax_wrapped_unflatten(flat, bool_splits, bool_vars)
                violations = _jax_wrapped_violations(sim_state, fls, unflats)
                flat, _ = jax.lax.fori_loop(
                    0, flat.size, _jax_wrapped_targeted_reset, (flat, violations))
                flat = jax.lax.fori_loop(0, flat.size, _jax_wrapped_reset, flat)
                resets = jnp.sum(flat != initial).astype(REAL)
                kept = _jax_wrapped_unflatten(flat == initial, bool_splits, bool_vars)
                fls = {**fls, **_jax_wrapped_unflatten(flat, bool_splits, bool_vars)}

            # move the other actions towards their defaults by bisection
            shrink = jnp.zeros((), dtype=REAL)
            if other_vars:
                origin = jnp.asarray(other_default)
                proposal = jnp.concatenate(
                    [jnp.ravel(jnp.asarray(fls[var], dtype=REAL)) for var in other_vars])

                def _jax_wrapped_interpolate(t, involved):
                    return jnp.where(involved, origin + t * (proposal - origin), proposal)

                def _jax_wrapped_flat_actions(flat):
                    new_actions = _jax_wrapped_unflatten(flat, other_splits, other_vars)
                    for var in other_vars:
                        if ranges[var] != 'real':
                            new_actions[var] = jnp.asarray(
                                jnp.round(new_actions[var]), dtype=INT)
                    return new_actions

                def _jax_wrapped_reset_violations(i):
                    reset = jnp.zeros(proposal.shape, dtype=jnp.bool_).at[i].set(True)
                    interp_0 = _jax_wrapped_flat_actions(_jax_wrapped_interpolate(0., reset))
                    return _jax_wrapped_violations(sim_state, fls, interp_0)

                # the bisection keeps the last candidate found feasible
                def _jax_wrapped_bisect(_, bounds):
                    low, high, best = bounds
                    mid = 0.5 * (low + high)
                    candidate = _jax_wrapped_interpolate(mid, involved)
                    cand_flat = _jax_wrapped_flat_actions(candidate)
                    feasible = _jax_wrapped_feasible(sim_state, fls, cand_flat)
                    return (jnp.where(feasible, mid, low), 
                            jnp.where(feasible, high, mid),
                            jnp.where(feasible, candidate, best))

                violations = _jax_wrapped_violations(sim_state, fls, {})
                reset_violations = jax.vmap(_jax_wrapped_reset_violations)(
                    jnp.arange(proposal.size))
                involved = (proposal != origin) & (reset_violations < violations)
                interp_0 = _jax_wrapped_flat_actions(_jax_wrapped_interpolate(0., involved))
                feasible_0 = _jax_wrapped_feasible(sim_state, fls, interp_0)
                involved = jnp.where(feasible_0, involved, True)
                involved_0_1 = _jax_wrapped_interpolate(0., involved)
                low, _, best = jax.lax.fori_loop(
                    0, num_bisections, _jax_wrapped_bisect,
                    (jnp.zeros((), dtype=REAL), jnp.ones((), dtype=REAL), involved_0_1))
                feasible = _jax_wrapped_feasible(sim_state, fls, {})
                repaired = _jax_wrapped_flat_actions(best)
                fls = {**fls, **{var: jnp.where(feasible, fls[var], repaired[var])
                                 for var in other_vars}}
                shrink = jnp.where(feasible, 0., 1. - low)
                step = _jax_wrapped_unflatten(
                    jnp.where(feasible | jnp.logical_not(involved), 1., low),
                    other_splits, other_vars)

            info = {'infeasible': infeasible, 'resets': resets, 'shrink': shrink,
                    'failed': jnp.logical_not(_jax_wrapped_feasible(sim_state, fls, {}))}
            actions = {**actions, **{var: fls[var] for var in defaults}}
            return actions, info, kept, step
        return _jax_wrapped_repair

    def exact(self, score_policy: Callable) -> Callable:
        '''Returns the function (sim_state, planner_state, actions) -> (actions, info)
        that repairs the actions on the exact model.

        :param score_policy: a policy whose boolean actions are in [0, 1]
        '''
        def _jax_wrapped_repair_exact_actions(sim_state, planner_state, actions):
            scores = score_policy(sim_state, planner_state)
            actions, info, _, _ = self._repair(sim_state, actions, scores)
            return actions, info
        return _jax_wrapped_repair_exact_actions

    def relaxed(self) -> Callable:
        '''Returns the function (sim_state, planner_state, actions) -> (actions, info)
        that repairs relaxed actions as their exact (rounded) values are repaired.'''
        repair_fn = self._repair
        INT = self.compiled.INT
        ranges = self.compiled.rddl.variable_ranges
        defaults = {var: np.asarray(value, dtype=self.compiled.REAL)
                    for (var, value) in self._defaults.items()}

        def _jax_wrapped_repair_relaxed_actions(sim_state, planner_state, actions):
            exact = {}
            for var in defaults:
                if ranges[var] == 'bool':
                    exact[var] = actions[var] > 0.5
                elif ranges[var] == 'real':
                    exact[var] = actions[var]
                else:
                    exact[var] = jnp.asarray(jnp.round(actions[var]), dtype=INT)
            _, info, kept, step = repair_fn(sim_state, {**actions, **exact}, actions)
            kept, step = jax.lax.stop_gradient((kept, step))
            repaired = dict(actions)
            for var in defaults:
                if ranges[var] == 'bool':
                    repaired[var] = jnp.where(kept[var], actions[var], defaults[var])
                else:
                    repaired[var] = defaults[var] + step[var] * (actions[var] - defaults[var])
            return repaired, info
        return _jax_wrapped_repair_relaxed_actions

    @staticmethod
    def statistics(log: Dict[str, Any]) -> Dict[str, float]:
        '''Returns statistics of the repairs in the given log of rollouts.'''
        info = log['repair']
        infeasible = np.asarray(info['infeasible'])
        num_repairs = max(int(np.sum(infeasible)), 1)
        return {
            'repaired': float(np.mean(infeasible)),
            'resets_per_repair': float(np.sum(np.asarray(info['resets'])) / num_repairs),
            'shrink_per_repair': float(np.sum(np.asarray(info['shrink'])) / num_repairs),
            'failed': float(np.mean(np.asarray(info['failed'])))
        }


# ***********************************************************************
# ALL VERSIONS OF JAX PLANNER
# 
# - base class of all planners (compilation, optimization loop and actions)
# - gradient descent through the relaxed model
# - parameter exploring policy gradients (PGPE)
# - limited-memory matrix adaptation evolution strategy (LM-MA-ES), which includes
#   separable natural evolution strategies (SNES)
# 
# ***********************************************************************


class JaxPlanner(metaclass=ABCMeta):
    '''Base class for all planners that optimize a policy or plan in a RDDL MDP.

    The base class compiles the relaxed (train) and exact (test) models, the policy and
    the losses, runs the optimization loop and computes actions. Subclasses only define
    how policy parameters are initialized and updated, by implementing
    _jax_compile_optimizer, which must set:

        initialize(sim_state, planner_state, init_policy_params) -> planner_state
            initializes the optimizer state (and the policy parameters if
            init_policy_params)
            of all parallel updates, given the train simulation state
        update(train_sim_state, test_sim_state, planner_state)
            -> (train_sim_state, test_sim_state, planner_state, loss, converged, log,
                zero_grads)
            performs one update of all parallel policies; the log is a dictionary that
            may contain the 'grad' (descent direction) of the update, as well as the
            'fluents' and 'error' codes of the rollouts on the relaxed model

    A planner can run another planner alongside it as an auxiliary planner, which then
    shares the models, policy and losses of the planner it is passed to: the policy
    parameters of the auxiliary planner replace those of this planner whenever they
    achieve a new best test return. A planner that is used as an auxiliary planner is
    created without the RDDL model and planning problem arguments.
    '''

    # attributes an auxiliary planner shares with the planner that it is passed to
    _SHARED_ATTRIBUTES = (
        'rddl', 'plan', 'batch_size_train', 'batch_size_test', 'horizon',
        'parallel_updates', '_action_bounds', 'utility', 'utility_kwargs',
        'compiler_type', 'compiler_kwargs', 'use_symlog_reward', 'critic_fn', 'logger',
        'preprocessor', 'python_functions', 'cache_full_train_rollouts',
        'cache_full_test_rollouts', 'compiled', 'test_compiled', 'print_warnings',
        'check_preconditions',
        'init_test_nfls', 'train_policy', 'test_policy', 'train_rollouts',
        'test_rollouts', 'init_policy', 'single_train_loss', 'single_test_loss',
        'test_loss'
    )

    def __init__(self, rddl: Optional[RDDLLiftedModel]=None,
                 plan: Optional[JaxPlan]=None,
                 batch_size_train: int=32,
                 batch_size_test: Optional[int]=None,
                 rollout_horizon: Optional[int]=None,
                 parallel_updates: int=1,
                 shard_parallel_updates: bool=False,
                 model_param_ranges: Optional[Dict[str, Tuple[float, float]]]=None,
                 complex_preconditions: bool=True,
                 action_bounds: Optional[Bounds]=None,
                 auxiliary: Optional['JaxPlanner']=None,
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
                 python_functions: Optional[Dict[str, Callable]]=None,
                 cache_full_train_rollouts: bool=False,
                 cache_full_test_rollouts: bool=False) -> None:
        '''Creates a new planner for optimizing a policy or plan in the given RDDL.
        Some operations will be converted to their differentiable counterparts in the
        relaxed model; the specific operations can be customized by providing a
        tailored compiler instance.
        
        :param rddl: the RDDL domain to optimize: None if the planner is to be used
        as an auxiliary planner, in which case the remaining arguments are ignored
        :param plan: the policy/plan representation to optimize
        :param batch_size_train: how many rollouts to perform per optimization step
        :param batch_size_test: how many rollouts to use to test the plan at each step
        :param rollout_horizon: lookahead planning horizon: None uses the env horizon
        :param parallel_updates: how many optimizers to run independently in parallel
        :param shard_parallel_updates: whether to shard the parallel optimizers over all
        devices, if parallel_updates is a multiple of the number of devices
        :param model_param_ranges: optional ranges (low, high) of parameters of the
        relaxed model (e.g. the weights of relaxations), each keyed by a parameter name
        :param complex_preconditions: whether the action preconditions that are
        not box constraints on actions are part of the problem
        :param action_bounds: box constraints on actions
        :param auxiliary: optional planner to run alongside this planner
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
        :param cache_full_train_rollouts: whether to cache full training rollouts in log
        :param cache_full_test_rollouts: whether to cache full testing rollouts to log
        '''
        self.rddl = rddl
        self.plan = plan
        self.batch_size_train = batch_size_train
        if batch_size_test is None:
            batch_size_test = batch_size_train
        self.batch_size_test = batch_size_test
        self.parallel_updates = parallel_updates
        self.shard_parallel_updates = shard_parallel_updates
        if model_param_ranges is not None:
            if parallel_updates < 2:
                raise ValueError('model_param_ranges requires parallel_updates > 1.')
            for (name, (low, high)) in model_param_ranges.items():
                if not 0 < low <= high:
                    raise ValueError(f'Range of model parameter <{name}> must satisfy '
                                     f'0 < low <= high, got ({low}, {high}).')
        self.model_param_ranges = model_param_ranges
        self.complex_preconditions = complex_preconditions
        if rollout_horizon is None and rddl is not None:
            rollout_horizon = rddl.horizon
        self.horizon = rollout_horizon
        if action_bounds is None:
            action_bounds = {}
        self._action_bounds = action_bounds
        self.preprocessor = preprocessor
        if python_functions is None:
            python_functions = {}
        self.python_functions = python_functions
        
        # set utility
        if isinstance(utility, str):
            utility = create_utility(utility)
        self.utility = utility
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
        
        # caching flags
        if (not cache_full_train_rollouts) \
        and (self.preprocessor is not None or self.dashboard is not None):
            cache_full_train_rollouts = True
            print(termcolor.colored(
                '[INFO] cache_full_train_rollouts will be set to True '
                'because state processor or dashboard requested. ', 'dark_grey'
            ))
        if (not cache_full_test_rollouts) and self.dashboard is not None:
            cache_full_test_rollouts = True
            print(termcolor.colored(
                '[INFO] cache_full_test_rollouts will be set to True '
                'because dashboard requested. ', 'dark_grey'
            ))
        self.cache_full_train_rollouts = cache_full_train_rollouts
        self.cache_full_test_rollouts = cache_full_test_rollouts

        # auxiliary planner shares the model of this planner: each planner owns a copy
        if auxiliary is not None:
            if auxiliary.rddl is not None:
                raise ValueError('Auxiliary planner must be created without a RDDL model, '
                                 'since it shares the model of the planner it is passed to.')
            if auxiliary.auxiliary is not None:
                raise ValueError('Auxiliary planner cannot have its own auxiliary planner.')
            auxiliary = copy.copy(auxiliary)
        self.auxiliary = auxiliary

        if rddl is not None:
            if plan is None:
                raise ValueError('plan must be provided together with the RDDL model.')
            self._jax_compile_graph()

    def __str__(self) -> str:
        return self.summarize_hyperparameters()
    
    @staticmethod
    def summarize_system() -> str:
        '''Returns a string containing information about the system, Python version 
        and jax-related packages that are relevant to the current planner.'''         
        devices = jax.devices()
        default_device = devices[0] if devices else 'n/a'
        return termcolor.colored(
            '\n'
            f'Starting JaxPlan v{__version__} '
            f'on device {default_device.platform}{default_device.id}\n', attrs=['bold']
        )
    
    def summarize_relaxations(self) -> str:
        '''Returns a summary table containing all non-differentiable operators
        and their relaxations.'''
        result = ''
        overriden_ops_info = self.compiled.overriden_ops_info()
        exact_ops_info = self.compiled.exact_ops_info()
        if overriden_ops_info:
            result += ('[INFO] Some RDDL operations are non-differentiable '
                       'and will be approximated as follows:' + '\n')
            for (class_, op_to_ids_dict) in overriden_ops_info.items():
                result += f'    {class_}:\n'
                for (op, relaxed_ids) in op_to_ids_dict.items():
                    exact_ids = exact_ops_info.get(op, [])
                    result += (
                        f'        {op} ' + 
                        termcolor.colored(
                            f'[{len(relaxed_ids)} relaxed, {len(exact_ids)} exact]\n', 
                            'dark_grey'
                        )
                    )
        return result
        
    @abstractmethod
    def summarize_optimizer(self) -> str:
        '''Returns a string summarizing the hyper-parameters of the optimizer that is
        specific to this planner.'''
        pass

    def summarize_hyperparameters(self) -> str:
        '''Returns a string summarizing the hyper-parameters of the current planner 
        instance.'''
        result = (f'[INFO] objective hyper-parameters:\n'
                  f'    utility_fn         ={self.utility.__name__}\n'
                  f'    utility args       ={self.utility_kwargs}\n'
                  f'    use_symlog         ={self.use_symlog_reward}\n'
                  f'    use_critic         ={self.critic_fn is not None}\n'
                  f'    lookahead          ={self.horizon}\n'
                  f'    user_action_bounds ={self._action_bounds}\n'
                  f'[INFO] planner hyper-parameters:\n'
                  f'    method            ={self.__class__.__name__}\n'
                  f'    batch_size_train  ={self.batch_size_train}\n'
                  f'    batch_size_test   ={self.batch_size_test}\n'
                  f'    parallel_updates  ={self.parallel_updates}\n'
                  f'    shard_parallel    ={self.shard_parallel_updates}\n'
                  f'    model_param_ranges={self.model_param_ranges}\n'
                  f'    complex_preconds  ={self.complex_preconditions}\n'
                  f'    preprocessor      ={self.preprocessor}\n')
        result += self.summarize_optimizer()
        result += str(self.plan)
        if self.auxiliary is not None:
            result += (f'[INFO] auxiliary planner:\n'
                       f'    method            ={self.auxiliary.__class__.__name__}\n')
            result += self.auxiliary.summarize_optimizer()
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

        # compile relaxed model
        self.compiled = self.compiler_type(
            rddl=self.rddl,
            logger=self.logger,
            python_functions=self.python_functions,
            **self.compiler_kwargs
        )
        self.print_warnings = self.compiled.print_warnings
        self.compiled.compile(log_jax_expr=True, heading='RELAXED MODEL', 
                              compile_constraints=False)
    
        # compile exact model for testing
        self.test_compiled = JaxRDDLCompiler(
            rddl=self.rddl,
            allow_synchronous_state=True,
            logger=self.logger,
            use64bit=self.compiled.use64bit,
            python_functions=self.python_functions
        )
        self.check_preconditions = self.complex_preconditions and not all(
            self.test_compiled.constraints.is_box_preconditions)
        if self.check_preconditions and self.print_warnings:
            print(termcolor.colored(
                '[INFO] Some action preconditions are not box constraints: they will be '
                'checked on the exact model.', 'dark_grey'
            ))
        self.test_compiled.compile(log_jax_expr=True, heading='EXACT MODEL',
                                   compile_constraints=self.check_preconditions)
        
        self.init_test_nfls = self.test_compiled.init_sim_state(None, None, 1, False).nfls

    def _jax_compile_policy(self):
        if self.preprocessor is not None:
            self.preprocessor.compile(self.compiled)   
        self.plan.compile(self.compiled, self.test_compiled,
                          _bounds=self._action_bounds,
                          horizon=self.horizon,
                          preprocessor=self.preprocessor)
        self.train_policy = jax.jit(self.plan.train_policy)

        # the actions on the exact model are repaired to satisfy the preconditions,
        # with the boolean actions of the train policy as their scores
        self.action_repair = None
        test_policy = self.plan.test_policy
        if self.check_preconditions:
            self.action_repair = JaxActionRepair(
                self.test_compiled, print_warnings=self.print_warnings)
            exact_repair = self.action_repair.exact(self.plan.train_policy)
            def _jax_wrapped_repaired_test_policy(sim_state, planner_state):
                actions = self.plan.test_policy(sim_state, planner_state)
                return exact_repair(sim_state, planner_state, actions)[0]
            test_policy = _jax_wrapped_repaired_test_policy
        self.test_policy = jax.jit(test_policy)
        self.init_policy = self._jax_init_policy()

    def _jax_train_action_repair(self) -> Optional[Callable]:
        '''Returns the function that repairs the actions of the train policy in the
        rollouts on the relaxed model, or None if they are not repaired.'''
        return None

    def _jax_compile_rollouts(self):
        self.train_rollouts = self.compiled.compile_rollouts(
            policy=self.plan.train_policy,
            n_steps=self.horizon,
            n_batch=self.batch_size_train,
            history_dependent=self.plan.history_dependent,
            cache_path_info=self.cache_full_train_rollouts,
            action_repair=self._jax_train_action_repair()
        )
        test_rollouts = self.test_compiled.compile_rollouts(
            policy=self.plan.test_policy,
            n_steps=self.horizon,
            n_batch=self.batch_size_test,
            history_dependent=self.plan.history_dependent,
            check_constraints=self.check_preconditions,
            cache_path_info=self.cache_full_test_rollouts,
            action_repair=(None if self.action_repair is None
                           else self.action_repair.exact(self.plan.train_policy))
        )
        self.test_rollouts = jax.jit(test_rollouts)

    def _jax_compile_losses(self):
        self.single_train_loss = self._jax_loss(
            self.train_rollouts, self.compiled, use_symlog=self.use_symlog_reward)
        self.single_test_loss = self._jax_loss(
            self.test_rollouts, self.test_compiled, use_symlog=False)
        vmap_axes = (JaxRDDLSimState(model_params=0), JaxPlannerState(policy_params=0))
        self.test_loss = jax.jit(jax.vmap(self.single_test_loss, in_axes=vmap_axes))

    @abstractmethod
    def _jax_compile_optimizer(self) -> None:
        '''Compiles the initialize and update functions of this planner.'''
        pass

    def _jax_compile_auxiliary(self):
        if self.auxiliary is not None:
            self.auxiliary._jax_attach(self)
            self.merge_auxiliary = self._jax_merge_auxiliary()
        else:
            self.merge_auxiliary = None

    def _jax_attach(self, planner: 'JaxPlanner') -> None:
        '''Shares the model, policy and losses of the given planner with this planner,
        and compiles the optimizer of this planner on them.'''
        for name in self._SHARED_ATTRIBUTES:
            setattr(self, name, getattr(planner, name))
        self._jax_compile_optimizer()
    
    def _jax_compile_graph(self):
        self._jax_compile_rddl()
        self._jax_compile_policy()
        self._jax_compile_rollouts()
        self._jax_compile_losses()
        self._jax_compile_optimizer()
        self._jax_compile_auxiliary()

    def _jax_init_policy(self):
        plan_init_fn = self.plan.initializer
        num_parallel = self.parallel_updates

        # initialize multiple policies to be optimized in parallel
        def _jax_wrapped_batched_init_policy(sim_state):
            keys = random.split(sim_state.key, num=num_parallel)
            sim_state = sim_state.replace(key=keys)
            vmap_axes = (JaxRDDLSimState(key=0),)
            params, hyperparams = jax.vmap(plan_init_fn, in_axes=vmap_axes)(sim_state)
            return params, _unbatched_pytree(hyperparams, 0)
        return _jax_wrapped_batched_init_policy
    
    def _jax_critic(self):
        critic_fn = self.critic_fn
        policy_fn = self.train_policy

        def _jax_wrapped_critic(sim_state, planner_state):

            # fluent tensors have leading time dimension, take last observation
            if self.rddl.observ_fluents:
                obs = {name: sim_state.fls[name][-1, ...] 
                       for name in self.rddl.observ_fluents}
            else:
                obs = {state: sim_state.fls[next_state][-1, ...] 
                       for (state, next_state) in self.rddl.next_state.items()}    

            # calculate action of the last observation   
            # TODO: allow history dependent and nonstationary policies
            sim_state = sim_state.replace(step=0, fls=obs, fls_hist={})
            action = policy_fn(sim_state, planner_state)    

            # evaluate and validate critic
            critic_value = jnp.squeeze(critic_fn(planner_state.critic_params, obs, action))
            if not jnp.isscalar(critic_value):
                raise ValueError('Critic value must be a scalar.')
            return critic_value
        return _jax_wrapped_critic            

    def _jax_return(self, use_symlog):
        gamma = self.rddl.discount
        
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
        
    def _jax_loss(self, rollouts, compiled, use_symlog=False):
        utility_fn = self.utility    
        utility_kwargs = self.utility_kwargs 
        _jax_wrapped_returns = self._jax_return(use_symlog)
        _jax_wrapped_critic = self._jax_critic()
        
        def _jax_wrapped_plan_loss(sim_state, planner_state):
            
            # update the model parameters from statistics collected in the rollouts
            log, model_params = rollouts(sim_state, planner_state)
            model_params = compiled.update_model_params(model_params, log['model_stats'])
            
            # evaluate optional critic network at termination of rollouts
            batch_size = jnp.shape(log['reward'])[0]
            if self.critic_fn is None:
                critic_values = jnp.zeros((batch_size,), dtype=self.compiled.REAL)
            else:
                keys = random.split(sim_state.key, num=batch_size)
                sim_state = sim_state.replace(key=keys, fls=log['fluents'])
                vmap_axes = (JaxRDDLSimState(key=0, fls=0), None)
                critic_values = jax.vmap(
                    _jax_wrapped_critic, in_axes=vmap_axes)(sim_state, planner_state)

            # optional reward masking
            rewards = log['reward']
            if planner_state.reward_mask is not None:
                mask = planner_state.reward_mask
                if jnp.ndim(mask) != 1:
                    raise ValueError('Reward mask must be 1-dimensional.')
                if mask.size != rewards.shape[1]:
                    raise ValueError('Reward mask size must match reward dimension.')
                rewards = rewards * mask[jnp.newaxis, ...]

            # rewards only count until the episode terminates: termination is checked
            # on the next state, so the reward of the terminating step still counts
            terminated = jnp.asarray(log['termination'], dtype=rewards.dtype)
            not_terminated = jnp.cumprod(1. - terminated, axis=1)
            alive = jnp.concatenate(
                [jnp.ones_like(not_terminated[:, :1]), not_terminated[:, :-1]], axis=1)
            rewards = rewards * alive
            critic_values = critic_values * not_terminated[:, -1]

            # number of steps of each rollout at which the action preconditions are
            # violated until the episode terminates (zero if they are not checked)
            violated = 1. - jnp.asarray(log['precondition'], dtype=rewards.dtype)
            log['violations'] = jnp.sum(violated * alive, axis=1)

            # evaluate cumulative return per rollout
            returns = _jax_wrapped_returns(rewards, critic_values)
            loss = -utility_fn(returns, **utility_kwargs)
            aux = (log, model_params)
            return loss, aux
        return _jax_wrapped_plan_loss
    
    def _jax_merge_auxiliary(self):
        
        # currently implements a hard replacement where the parameter of this planner
        # is replaced by the auxiliary planner parameter if the latter is an improvement
        def _jax_wrapped_batched_auxiliary_merge(aux_mask, aux_policy_params, policy_params,
                                                 aux_loss, test_loss,
                                                 aux_loss_smooth, test_loss_smooth,
                                                 aux_converged, converged):
            mask_tree = jax.tree_util.tree_map(
                lambda leaf: aux_mask[(...,) + (jnp.newaxis,) * (jnp.ndim(leaf) - 1)],
                aux_policy_params)
            policy_params = jax.tree_util.tree_map(
                jnp.where, mask_tree, aux_policy_params, policy_params)
            test_loss = jnp.where(aux_mask, aux_loss, test_loss)
            test_loss_smooth = jnp.where(aux_mask, aux_loss_smooth, test_loss_smooth)
            converged = jnp.where(aux_mask, aux_converged, converged)
            return policy_params, test_loss, test_loss_smooth, converged
        return jax.jit(_jax_wrapped_batched_auxiliary_merge)
    
    # ===========================================================================
    # OPTIMIZE API
    # ===========================================================================

    @staticmethod
    def _is_better(violations, loss, best_violations, best_loss):
        '''Returns whether plans with the given mean numbers of violations of the action
        preconditions and losses improve on the best plans: a plan improves if it has
        fewer violations, or as many violations and a smaller loss.'''
        return np.less(violations, best_violations) | (
            np.equal(violations, best_violations) & np.less(loss, best_loss))

    def _batched_pytree(self, pytree):
        def make_batched(x):
            x = np.asarray(x)
            x = np.repeat(x[np.newaxis, ...], self.parallel_updates, axis=0)
            return x
        return jax.tree_util.tree_map(make_batched, pytree)

    def _parallel_sharding(self):
        '''Returns the sharding of the parallel optimizers over all devices, or None if
        sharding is off, there is a single device, or the parallel optimizers are not a
        multiple of the number of devices.'''
        devices = jax.devices()
        if not self.shard_parallel_updates or len(devices) < 2:
            return None
        if self.parallel_updates % len(devices) != 0:
            if self.print_warnings:
                print(termcolor.colored(
                    f'[WARN] parallel_updates={self.parallel_updates} is not a multiple of '
                    f'the {len(devices)} devices: parallel optimizers will run on a single '
                    f'device.', 'yellow'
                ))
            return None
        mesh = jax.sharding.Mesh(np.asarray(devices), ('parallel',))
        return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('parallel'))

    def _shard_parallel(self, pytree, sharding):
        '''Places the leaves of a pytree batched over the parallel optimizers with the
        given sharding along their leading axis.'''
        if sharding is None:
            return pytree
        return jax.tree_util.tree_map(
            lambda leaf: jax.device_put(leaf, sharding) if np.ndim(leaf) > 0 else leaf,
            pytree)

    def _sample_model_params(self, key, model_params):
        '''Samples the parameters of the relaxed model of the parallel optimizers that
        are matched by model_param_ranges from a Latin hypercube over their ranges on
        log scale, given the parameters batched over the parallel optimizers.'''
        ranges = self.model_param_ranges
        num_samples = self.parallel_updates
        leaves_with_paths, treedef = jax.tree_util.tree_flatten_with_path(model_params)
        leaves = [leaf for (_, leaf) in leaves_with_paths]

        # a parameter is matched by its name, the last key of its path
        names = [str(getattr(path[-1], 'key', path[-1]))
                 for (path, _) in leaves_with_paths]

        # names that match no parameter of this model are skipped
        unmatched = [name for name in ranges if name not in names]
        if unmatched and self.print_warnings:
            print(termcolor.colored(
                f'[WARN] Model parameters {unmatched} in model_param_ranges were not '
                f'found and will be ignored.', 'yellow'))

        # each name is a dimension of the Latin hypercube
        rng = np.random.default_rng(np.asarray(key))
        for (name, (low, high)) in ranges.items():
            strata = rng.permutation(num_samples)
            values = low * np.power(high / low, (strata + 0.5) / num_samples)
            for (index, leaf_name) in enumerate(names):
                if leaf_name == name:
                    leaf = np.asarray(leaves[index])
                    values_shape = (-1,) + (1,) * (np.ndim(leaf) - 1)
                    leaves[index] = np.broadcast_to(
                        np.reshape(values, values_shape), leaf.shape).astype(leaf.dtype)
        return jax.tree_util.tree_unflatten(treedef, leaves)
    
    def as_optimization_problem(self, key: Optional[random.PRNGKey]=None,
                                critic_params: Optional[Pytree]=None,
                                print_values: bool=False) -> Tuple[Callable, Callable, np.ndarray, Callable]:
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
        :param critic_params: optional critic network parameters
        :param print_values: whether to print objective values during function calls
        '''

        # make sure parallel updates are disabled
        if self.parallel_updates > 1:
            raise ValueError('Cannot compile static optimization problem '
                             'when parallel_updates is not None.')
        
        # if PRNG key is not provided
        if key is None:
            key = random.PRNGKey(round(time.time() * 1000))
            
        # initialize the simulation state
        sim_state = self.compiled.init_sim_state(key, None, self.batch_size_train)
        
        # initialize the planner state
        params, hyperparams = jax.jit(self.init_policy)(sim_state)
        policy_params = _unbatched_pytree(params, 0)
        planner_state = JaxPlannerState(policy_params=policy_params, hyperparams=hyperparams,
                                        critic_params=critic_params, progress=0.)
        
        # get the params mapping to a 1D vector
        guess_1d, unravel_fn = jax.flatten_util.ravel_pytree(policy_params)  
        guess_1d = np.asarray(guess_1d)      
        
        # computes the training loss function and gradient in a 1D vector
        @jax.jit
        def _loss_with_key(params_1d, _sim_state):
            key, subkey = random.split(_sim_state.key)
            _sim_state = _sim_state.replace(key=subkey)
            _planner_state = planner_state.replace(policy_params=unravel_fn(params_1d))
            loss_val, (_, model_params) = self.single_train_loss(_sim_state, _planner_state)
            _sim_state = _sim_state.replace(key=key, model_params=model_params)
            return loss_val, _sim_state
        
        _grad_with_key = jax.jit(jax.grad(_loss_with_key, has_aux=True))
        
        # store a global reference to the key on every JAX function call and pass when
        # required by JAX, then update it upon return
        def _loss_function(params_1d):
            nonlocal sim_state
            loss_val, sim_state = _loss_with_key(params_1d, sim_state)
            if print_values:
                print(f'Loss: {loss_val:.6f}')
            return float(loss_val)
        
        def _grad_function(params_1d):
            nonlocal sim_state
            grad, sim_state = _grad_with_key(params_1d, sim_state)
            return np.asarray(grad)
        
        return _loss_function, _grad_function, guess_1d, jax.jit(unravel_fn)
                
    def optimize(self, *args, **kwargs) -> Dict[str, Any]:
        '''Compute an optimal policy or plan. Return the callback from training.
        
        :param key: JAX PRNG key (derived from clock if not provided)
        :param epochs: the maximum number of optimization steps
        :param train_seconds: total time allocated for optimization
        :param model_params: optional model-parameters to override default
        :param policy_hyperparams: hyper-parameters for the policy/plan, such as
        weights for sigmoid wrapping boolean actions
        :param subs: dictionary mapping initial state and non-fluents to 
        their values: if None initializes all variables from the RDDL instance
        :param guess: initial policy parameters: if None will use the initializer
        specified in this instance
        :param critic_params: parameters of the critic function
        :param print_summary: whether to print planner header and summary
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
    
    @staticmethod
    @jax.jit
    def _jax_invalidate_loss(loss, policy_params):
        finite = [jnp.all(jnp.isfinite(leaf), axis=tuple(range(1, jnp.ndim(leaf))))
                  for leaf in jax.tree_util.tree_leaves(policy_params)]
        return jnp.where(jnp.all(jnp.stack(finite), axis=0), loss, jnp.inf)
        
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
        :param epochs: the maximum number of optimization steps
        :param train_seconds: total time allocated for optimization
        :param model_params: optional model-parameters to override default
        :param policy_hyperparams: hyper-parameters for the policy/plan, such as
        weights for sigmoid wrapping boolean actions
        :param subs: dictionary mapping initial state and non-fluents to 
        their values: if None initializes all variables from the RDDL instance
        :param guess: initial policy parameters: if None will use the initializer
        specified in this instance        
        :param critic_params: parameters of the critic function
        :param print_summary: whether to print planner header and summary
        :param print_progress: whether to print the progress bar during training
        :param print_hyperparams: whether to print list of hyper-parameter settings
        :param dashboard_id: experiment id for the dashboard
        :param stopping_rule: stopping criterion
        :param test_rolling_window: the test return is averaged on a rolling 
        window of the past test_rolling_window returns when updating the best
        parameters found so far
        :param tqdm_position: position of tqdm progress bar (for multiprocessing)
        '''
        if self.rddl is None:
            raise ValueError('Planner was created without a RDDL model: it can only be '
                             'used as an auxiliary planner.')

        # ======================================================================
        # INITIALIZATION OF HYPER-PARAMETERS
        # ======================================================================

        # start measuring execution time here, including time spent outside optimize loop
        start_time = time.time()
        elapsed_outside_loop = 0
        
        # if PRNG key is not provided
        if key is None:
            key = random.PRNGKey(round(time.time() * 1000))
            if self.print_warnings:
                print(termcolor.colored(
                    '[WARN] PRNG key is not set: setting from clock.', 'yellow'
                ))
        dash_key = key[1].item()
        
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
        # INITIALIZATION OF SIMULATOR STATE
        # ======================================================================
        
        # initialize simulation state
        train_sim_state = self.compiled.init_sim_state(
            key, subs, self.batch_size_train, print_warnings=self.print_warnings)
        test_sim_state = self.test_compiled.init_sim_state(
            key, subs, self.batch_size_test, print_warnings=False)

        # update model params from user
        if model_params is None:
            model_params = train_sim_state.model_params
        model_params = self._batched_pytree(model_params)
        if self.model_param_ranges is not None:
            model_params = self._sample_model_params(key, model_params)

        # the parallel optimizers are sharded over all devices
        sharding = self._parallel_sharding()
        if sharding is not None and print_summary:
            print(f'[INFO] Parallel updates are sharded over {len(jax.devices())} devices.')
        train_sim_state = train_sim_state.replace(
            model_params=self._shard_parallel(model_params, sharding))
        test_sim_state = test_sim_state.replace(model_params=self._shard_parallel(
            self._batched_pytree(test_sim_state.model_params), sharding))
                        
        # ======================================================================
        # INITIALIZATION OF PLANNER STATE
        # ======================================================================
        
        planner_state = JaxPlannerState(
            hyperparams=policy_hyperparams, critic_params=critic_params, progress=0.)
        if guess is not None:
            if policy_hyperparams is None:
                raise ValueError('guess was provided without policy_hyperparams.')
            planner_state = planner_state.replace(policy_params=self._batched_pytree(guess))
        planner_state = self.initialize(train_sim_state, planner_state, guess is None)
        planner_state = planner_state.replace(
            policy_params=self._shard_parallel(planner_state.policy_params, sharding),
            opt_state=self._shard_parallel(planner_state.opt_state, sharding),
            opt_aux=self._shard_parallel(planner_state.opt_aux, sharding))

        # initialize the auxiliary planner at the policy parameters of this planner
        auxiliary = self.auxiliary
        if auxiliary is not None:
            aux_state = auxiliary.initialize(train_sim_state, planner_state, False)
            aux_state = aux_state.replace(
                opt_state=self._shard_parallel(aux_state.opt_state, sharding),
                opt_aux=self._shard_parallel(aux_state.opt_aux, sharding))
            rolling_aux_loss = RollingMean(test_rolling_window)
            rolling_aux_violations = RollingMean(test_rolling_window, exact=True)
        else:
            aux_state = rolling_aux_loss = rolling_aux_violations = None

        # ======================================================================
        # INITIALIZATION OF RUNNING STATISTICS
        # ======================================================================
        
        # initialize running statistics
        best_params = _unbatched_pytree(planner_state.policy_params, 0)
        best_loss, pbest_loss, best_grad = np.inf, np.inf, None
        best_violations, pbest_violations = np.inf, np.inf
        best_index = 0
        best_found_by = ' '
        last_iter_improve = 0
        rolling_test_loss = RollingMean(test_rolling_window)
        rolling_test_violations = RollingMean(test_rolling_window, exact=True)
        status = JaxPlannerStatus.NORMAL
        progress_percent = 0
        total_aux_it = 0
        
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

        # a return is marked F if its plan satisfies the action preconditions
        def _feasible_marker(violations):
            if not self.check_preconditions:
                return ''
            else:
                return 'F' if violations == 0 else ' '

        # error handlers (to avoid spam messaging)
        policy_constraint_msg_shown = False
        jax_train_msg_shown = set()
        jax_test_msg_shown = set()
        
        # ======================================================================
        # MAIN TRAINING LOOP BEGINS
        # ======================================================================
        
        for it in range(epochs):
            
            # ==================================================================
            # PLANNER UPDATE STEP
            # ==================================================================
            
            # update the parameters of the plan
            (train_sim_state, test_sim_state, planner_state, train_loss, converged,
             train_log, zero_grads) = self.update(train_sim_state, test_sim_state, planner_state)

            # update the preprocessor    
            proc_key = Preprocessor.HYPERPARAMS_KEY
            if self.preprocessor is not None and proc_key in planner_state.hyperparams \
            and 'fluents' in train_log:
                hyperparams = dict(planner_state.hyperparams)
                hyperparams[proc_key] = self.preprocessor.update(
                    train_log['fluents'], hyperparams[proc_key])
                planner_state = planner_state.replace(hyperparams=hyperparams)

            # evaluate
            test_sim_state = test_sim_state.replace(key=train_sim_state.key)
            test_loss, (test_log, model_params_test) = self.test_loss(
                test_sim_state, planner_state)
            test_sim_state = test_sim_state.replace(model_params=model_params_test)
            train_loss = np.asarray(train_loss)
            test_loss = np.asarray(self._jax_invalidate_loss(
                test_loss, planner_state.policy_params))
            test_loss_smooth = rolling_test_loss.update(test_loss)

            # mean number of violations of the action preconditions per rollout
            test_violations = np.mean(np.asarray(test_log['violations']), axis=1)
            test_violations = np.where(np.isfinite(test_loss), test_violations, np.inf)
            test_violations_smooth = rolling_test_violations.update(test_violations)

            # ==================================================================
            # AUXILIARY PLANNER UPDATE STEP
            # ==================================================================
            
            aux_improve = False
            if auxiliary is not None:

                # auxiliary update of the plan
                aux_state = planner_state.replace(
                    policy_params=aux_state.policy_params,
                    opt_state=aux_state.opt_state,
                    opt_aux=aux_state.opt_aux
                )
                _, test_sim_state, aux_state, _, aux_converged, _, _ = auxiliary.update(
                    train_sim_state, test_sim_state, aux_state)
                
                # evaluate with the state of this planner to reuse the compiled test loss
                aux_loss, (aux_log, _) = self.test_loss(
                    test_sim_state, planner_state.replace(policy_params=aux_state.policy_params))
                aux_loss = np.asarray(self._jax_invalidate_loss(aux_loss, aux_state.policy_params))
                aux_loss_smooth = rolling_aux_loss.update(aux_loss)
                aux_return = -aux_loss_smooth
                aux_violations = np.mean(np.asarray(aux_log['violations']), axis=1)
                aux_violations = np.where(np.isfinite(aux_loss), aux_violations, np.inf)
                aux_violations_smooth = rolling_aux_violations.update(aux_violations)

                # replace plan with auxiliary if new min reached or train loss invalid
                aux_mask = self._is_better(
                    aux_violations_smooth, aux_loss_smooth, pbest_violations, pbest_loss
                ) | ~np.isfinite(train_loss)
                if np.any(aux_mask):
                    policy_params, test_loss, test_loss_smooth, converged = \
                        self.merge_auxiliary(
                            aux_mask, aux_state.policy_params, planner_state.policy_params,
                            aux_loss, test_loss, aux_loss_smooth, test_loss_smooth,
                            aux_converged, converged
                        )
                    test_violations = np.where(aux_mask, aux_violations, test_violations)
                    test_violations_smooth = np.where(
                        aux_mask, aux_violations_smooth, test_violations_smooth)
                    planner_state = planner_state.replace(policy_params=policy_params)
                    aux_improve = True
                    total_aux_it += 1
            else:
                aux_loss = aux_loss_smooth = aux_return = aux_violations_smooth = None

            # ==================================================================
            # STATUS CHECKS AND LOGGING
            # ==================================================================

            # evaluate test losses and record best parameters so far
            train_grad = train_log.get('grad', None)
            best_index = np.lexsort((test_loss_smooth, test_violations_smooth))[0]
            if self._is_better(test_violations_smooth[best_index],
                               test_loss_smooth[best_index], best_violations, best_loss):
                best_params = _unbatched_pytree(planner_state.policy_params, best_index)
                if train_grad is not None:
                    best_grad = _unbatched_pytree(train_grad, best_index)
                best_loss = test_loss_smooth[best_index]
                best_violations = test_violations_smooth[best_index]
                best_found_by = 'H' if aux_improve and aux_mask[best_index] else '*'
                last_iter_improve = it
            pbest_improve = self._is_better(
                test_violations_smooth, test_loss_smooth, pbest_violations, pbest_loss)
            pbest_loss = np.where(pbest_improve, test_loss_smooth, pbest_loss)
            pbest_violations = np.where(
                pbest_improve, test_violations_smooth, pbest_violations)
                        
            # no progress
            status = JaxPlannerStatus.NORMAL
            if (not aux_improve) and np.all(zero_grads):
                status = JaxPlannerStatus.NO_PROGRESS
            
            # constraint satisfaction problem
            if not np.all(converged):                
                if progress_bar is not None and not policy_constraint_msg_shown:
                    progress_bar.write(termcolor.colored(
                        '[FAIL] Policy update violated action constraints.', 'red'
                    ))
                    policy_constraint_msg_shown = True
                status = JaxPlannerStatus.PRECONDITION_POSSIBLY_UNSATISFIED
            if self.check_preconditions and best_violations > 0:
                status = JaxPlannerStatus.PRECONDITION_POSSIBLY_UNSATISFIED
            
            # numerical error
            invalid_loss = np.all(~np.isfinite(train_loss))
            if auxiliary is not None:
                invalid_loss = invalid_loss and np.all(~np.isfinite(aux_loss))
            if invalid_loss:
                if progress_bar is not None:
                    progress_bar.write(termcolor.colored(
                        f'[FAIL] Planner aborted early with train loss {train_loss}.', 'red'
                    ))
                status = JaxPlannerStatus.INVALID_GRADIENT
              
            # problem in the model compilation
            if progress_bar is not None:

                # train model
                for error_code in np.unique(train_log.get('error', [])):
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
            
            # measure percent of progress based on elapsed time and iterations completed
            progress_time = elapsed / train_seconds
            progress_it = it / (epochs - 1) if epochs > 1 else 0
            progress_percent = 100 * min(1, max(0, progress_time, progress_it))
            planner_state = planner_state.replace(progress=progress_percent)
            
            # build a callback
            callback = {
                'iteration': it,
                'elapsed_time': elapsed,
                'progress': progress_percent,
                'status': status,
                'key': train_sim_state.key,
                'test_key': test_sim_state.key,
                'train_return':-train_loss,
                'test_return':-test_loss_smooth,
                'best_return':-best_loss,
                'best_violations': best_violations,
                'last_iteration_improved': last_iter_improve,
                'params': planner_state.policy_params,
                'best_params': best_params,
                'best_index': best_index,
                'model_params': train_sim_state.model_params,
                'policy_hyperparams': planner_state.hyperparams,
                'grad': train_grad,
                'best_grad': best_grad,
                'train_log': train_log,
                'test_log': test_log,
                'planner_state': planner_state,
                'auxiliary_return': aux_return,
                'auxiliary_violations': aux_violations_smooth,
                'best_found_by': best_found_by,
                'auxiliary_improved': aux_improve,
                'auxiliary_params': None if aux_state is None else aux_state.policy_params,                
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
                test_info = (f'{-test_loss_smooth[best_index]:13.5f}'
                             f'{_feasible_marker(test_violations_smooth[best_index])}')
                if auxiliary is not None:
                    aux_index = np.lexsort((aux_loss_smooth, aux_violations_smooth))[0]
                    aux_info = (f'{-aux_loss_smooth[aux_index]:13.5f}'
                                f'{_feasible_marker(aux_violations_smooth[aux_index])} '
                                f'[{total_aux_it} it] aux | ')
                else:
                    aux_info = ''
                best_info = (f'{best_found_by} {-best_loss:13.5f}'
                             f'{_feasible_marker(best_violations)}')
                progress_bar.set_description(
                    f'{position_str} {it} it | {-np.min(train_loss):13.5f} train | '
                    f'{test_info} test | {aux_info}{best_info} best | {status.value} status', 
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
            grad_norm = optax.tree.norm(best_grad) if best_grad is not None else np.nan

            # calculate best policy return
            final_planner_state = planner_state.replace(
                policy_params=self._batched_pytree(best_params))
            _, (final_log, _) = self.test_loss(test_sim_state, final_planner_state)
            final_returns = np.sum(final_log['reward'], axis=2)
            best_returns = np.ravel(final_returns[best_index])
            mean, rlo, rhi = self._ci_bootstrap(best_returns, key[0].item())            
            if self.check_preconditions:
                final_violations = np.mean(final_log['violations'][best_index])
                repairs = JaxActionRepair.statistics(final_log)
                violations_info = (f'    best violations:  {final_violations:.5f} '
                                   f'steps per rollout\n'
                                   f'    best repairs:     {repairs["repaired"]:.1%} of '
                                   f'steps, {repairs["resets_per_repair"]:.2f} resets '
                                   f'per repair, {repairs["failed"]:.1%} of steps '
                                   f'infeasible after repair\n')
                if best_violations > 0:
                    print(termcolor.colored(
                        '[WARN] No plan satisfying the preconditions was found.', 'yellow'
                    ))
            else:
                violations_info = ''

            print(
                f'[INFO] Summary of optimization:\n'
                f'    status:             {status}\n'
                f'    time:               {elapsed:.2f} seconds\n'
                f'    iterations:         {it}\n'
                f'    best objective:     {-best_loss:.5f}\n'
                f'    best grad norm:     {grad_norm:.5f}\n'
                f'    best reward mean:   {mean:.5f}\n'
                f'    best reward 95% CI: [{rlo:.5f}, {rhi:.5f}]\n'
                f'{violations_info}'
            )
    
    @staticmethod
    def _ci_bootstrap(returns, seed, confidence=0.95, n_boot=8192):
        rng = np.random.default_rng(seed)
        returns = np.asarray(returns)
        resampled = rng.choice(returns, size=(n_boot, returns.shape[0]), replace=True)
        means = np.mean(resampled, axis=1)
        lower = np.percentile(means, 100 * 0.5 * (1 - confidence))
        upper = np.percentile(means, 100 * 0.5 * (1 + confidence))
        mean = np.mean(returns)
        return mean, lower, upper

    def get_action(self, key: random.PRNGKey,
                   params: Pytree,
                   step: int,
                   state: Dict[str, Any],
                   policy_hyperparams: Optional[Dict[str, Any]]=None, 
                   history: Optional[Dict[str, Any]]=None,
                   copy_state: bool=True) -> Dict[str, Any]:
        '''Returns an action dictionary from the policy or plan with the given parameters.
        
        :param key: the JAX PRNG key
        :param params: the trainable parameter PyTree of the policy
        :param step: the time step at which decision is made
        :param state: the dict of state p-variables
        :param policy_hyperparams: hyper-parameters for the policy/plan, such as
        weights for sigmoid wrapping boolean actions (optional)
        :param history: history of past fluents for history-dependent policy (optional)
        :param copy_state: whether to copy state before processing
        '''
        if copy_state:
            state = state.copy()

        if history is None: 
            if self.plan.history_dependent:
                raise ValueError(
                    'Policy has history_dependent = True, but history was not provided.')
            history = {}
        
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
            if not (np.issubdtype(dtype, np.number) or np.issubdtype(dtype, np.bool_)):
                if step == 0 and var in self.rddl.observ_fluents:
                    state[var] = self.test_compiled.init_values[var]
                else:
                    if dtype.type is np.str_:
                        prange = self.compiled._variable_range_from_rddl_or_policy(var)
                        state[var] = self.rddl.object_string_to_index_array(prange, state[var])     
                    else:               
                        raise ValueError(
                            f'Values {values} assigned to p-variable <{var}> are '
                            f'non-numeric of type {dtype}.'
                        )
            
        # get test fluents and cast to numpy
        sim_state = JaxRDDLSimState(
            key=key, step=step, fls=state, nfls=self.init_test_nfls, fls_hist=history)
        planner_state = JaxPlannerState(policy_params=params, hyperparams=policy_hyperparams)
        policy_fls = self.test_policy(sim_state, planner_state)
        policy_fls = jax.tree_util.tree_map(np.asarray, policy_fls)
        return policy_fls      
       

class JaxPGPEPlanner(JaxPlanner):
    '''Parameter exploring policy gradients (PGPE) with a Gaussian distribution over
    the policy parameters [6], that optimizes the policy on the exact model.'''

    def __init__(self, rddl: Optional[RDDLLiftedModel]=None,
                 plan: Optional[JaxPlan]=None,
                 batch_size: int=1,
                 steps_per_update: int=1,
                 init_sigma: float=1.0,
                 sigma_range: Tuple[float, float]=(1e-5, 1e5),
                 reward_shaping: Optional[str]='ranks',
                 min_reward_scale: float=1e-5,
                 super_symmetric: bool=True,
                 super_symmetric_accurate: bool=True,
                 project_samples: bool=False,
                 optimizer: Callable[..., optax.GradientTransformation]=optax.adam,
                 optimizer_sigma: Optional[Callable[..., optax.GradientTransformation]]=None,
                 optimizer_kwargs_mu: Optional[Kwargs]=None,
                 optimizer_kwargs_sigma: Optional[Kwargs]=None,
                 clip_grad_mu: Optional[float]=None,
                 clip_grad_sigma: Optional[float]=None,
                 ema_decay: Optional[float]=None,
                 start_entropy_coeff: float=1e-2,
                 end_entropy_coeff: float=1e-5,
                 max_kl_update: Optional[float]=None,
                 eps: float=1e-10,
                 **planner_kwargs) -> None:
        '''Creates a new Gaussian PGPE planner.

        :param rddl: the RDDL domain to optimize: None if used as an auxiliary planner
        :param plan: the policy/plan representation to optimize
        :param batch_size: how many policy parameters to sample per optimization step
        :param steps_per_update: how many optimization steps are performed per planner
        update (useful for partially jit compiling the update loop)
        :param init_sigma: initial standard deviation of Gaussian
        :param sigma_range: bounds to constrain standard deviation
        :param reward_shaping: how the returns of the sampled policy parameters are
        shaped: 'scale' applies the reward scaling of the paper, 'ranks' replaces the
        returns by their centered ranks in [-0.5, 0.5], and None uses the raw returns
        :param min_reward_scale: minimum reward scaling to avoid underflow ('scale' only)
        :param super_symmetric: whether to use super-symmetric sampling as in the paper
        :param super_symmetric_accurate: whether to use the accurate formula for super-
        symmetric sampling or the simplified but biased formula
        :param project_samples: whether to use projected gradient on the sampled
        plan parameters (possibly introduces bias into the update rule)
        :param optimizer: a factory for an optax SGD algorithm
        :param optimizer_sigma: a factory for an optax SGD algorithm of the standard
        deviation (the same as the optimizer of the mean if None)
        :param optimizer_kwargs_mu: a dictionary of parameters to pass to the SGD
        factory for the mean optimizer
        :param optimizer_kwargs_sigma: a dictionary of parameters to pass to the SGD
        factory for the standard deviation optimizer
        :param clip_grad_mu: gradient clipping for mean optimizer
        :param clip_grad_sigma: gradient clipping for standard deviation optimizer
        :param ema_decay: EMA decay of parameters during optimization
        :param start_entropy_coeff: starting entropy regularization coeffient for Gaussian
        :param end_entropy_coeff: ending entropy regularization coeffient for Gaussian
        :param max_kl_update: bound on kl-divergence between parameter updates
        :param eps: small value to avoid division by zero
        :param **planner_kwargs: arguments of the planning problem (see JaxPlanner)
        '''
        self.batch_size = batch_size
        self.steps_per_update = steps_per_update
        self.init_sigma = init_sigma
        self.sigma_range = sigma_range
        if reward_shaping not in ('scale', 'ranks', None):
            raise ValueError(f'Invalid reward_shaping <{reward_shaping}>: must be '
                             f'\'scale\', \'ranks\' or None.')
        if reward_shaping == 'ranks' and not super_symmetric and batch_size == 1:
            raise ValueError('reward_shaping=\'ranks\' requires super_symmetric=True or '
                             'batch_size > 1: the ranks of two returns always sum to zero, '
                             'so the standard deviation would receive no gradient.')
        self.reward_shaping = reward_shaping
        self.min_reward_scale = min_reward_scale
        self.super_symmetric = super_symmetric
        self.super_symmetric_accurate = super_symmetric_accurate
        self.clip_grad_mu = clip_grad_mu
        self.clip_grad_sigma = clip_grad_sigma
        self.project_samples = project_samples

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
        if optimizer_sigma is None:
            optimizer_sigma = optimizer
        self.optimizer_name = optimizer
        self.optimizer_sigma_name = optimizer_sigma
        mu_optimizer, inject_mu, _ = _build_optax_optimizer(
            optimizer, optimizer_kwargs_mu, clip_grad_mu, None, None, ema_decay, None)
        sigma_optimizer, inject_sigma, _ = _build_optax_optimizer(
            optimizer_sigma, optimizer_kwargs_sigma, clip_grad_sigma, None, None, ema_decay,
            None)
        self.optimizer = optax.multi_transform(
            {'mu': mu_optimizer, 'sigma': sigma_optimizer}, 
            param_labels={'mu': 'mu', 'sigma': 'sigma'}
        )
        self.max_kl = max_kl_update if inject_mu and inject_sigma else None
        self.eps = eps
    
        super(JaxPGPEPlanner, self).__init__(rddl=rddl, plan=plan, **planner_kwargs)

    def summarize_optimizer(self) -> str:
        return (f'[INFO] PGPE hyper-parameters:\n'
                f'    init_sigma         ={self.init_sigma}\n'
                f'    sigma_range        ={self.sigma_range}\n'
                f'    reward_shaping     ={self.reward_shaping}\n'
                f'    min_reward_scale   ={self.min_reward_scale}\n'
                f'    super_symmetric    ={self.super_symmetric}\n'
                f'        accurate       ={self.super_symmetric_accurate}\n'
                f'    project_samples    ={self.project_samples}\n'
                f'    eps                ={self.eps}\n'
                f'[INFO] PGPE optimizer hyper-parameters:\n'
                f'    optimizer          ={self.optimizer_name}\n'
                f'    optimizer_sigma    ={self.optimizer_sigma_name}\n'
                f'    optimizer_kwargs:\n'
                f'        mu   ={self.optimizer_kwargs_mu}\n'
                f'        sigma={self.optimizer_kwargs_sigma}\n'
                f'    batch_size         ={self.batch_size}\n'
                f'    steps_per_update   ={self.steps_per_update}\n'
                f'    clip_grad:\n'
                f'        mu   ={self.clip_grad_mu}\n'
                f'        sigma={self.clip_grad_sigma}\n'
                f'    start_entropy_coeff={self.start_entropy_coeff}\n'
                f'    end_entropy_coeff  ={self.end_entropy_coeff}\n'
                f'    max_kl_update      ={self.max_kl}\n'
        )

    def _jax_compile_optimizer(self) -> None:
        loss_fn = self.single_test_loss
        projection = self.plan.projection
        real_dtype = self.test_compiled.REAL
        parallel_updates = self.parallel_updates
        init_policy = self.init_policy
        sigma0 = self.init_sigma
        sigma_lo, sigma_hi = self.sigma_range
        centered_ranks = self.reward_shaping == 'ranks'
        scale_reward = self.reward_shaping == 'scale'
        min_reward_scale = self.min_reward_scale
        super_symmetric = self.super_symmetric
        super_symmetric_accurate = self.super_symmetric_accurate
        batch_size = self.batch_size
        optimizer = self.optimizer
        max_kl = self.max_kl
        project_samples = self.project_samples
        
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
        
        # the policy parameters are the mean of the Gaussian, and the optimizer state
        # holds its standard deviation and the largest return observed so far
        def _jax_wrapped_pgpe_init(mu):
            sigma = jax.tree_util.tree_map(
                partial(jnp.full_like, fill_value=sigma0, dtype=real_dtype), mu)
            opt_state = optimizer.init({'mu': mu, 'sigma': sigma})
            r_max = jnp.asarray(-jnp.inf, dtype=real_dtype)
            return {'sigma': sigma, 'r_max': r_max, 'optimizer': opt_state}

        # initialize independent (mean, sigma) per policy parameter instance
        def _jax_wrapped_batched_pgpe_init(sim_state, planner_state, init_policy_params=True):
            if init_policy_params:
                params, hyperparams = init_policy(sim_state)
                planner_state = planner_state.replace(
                    policy_params=params, hyperparams=hyperparams)
            opt_state = jax.vmap(_jax_wrapped_pgpe_init, in_axes=0)(
                planner_state.policy_params)
            return planner_state.replace(opt_state=opt_state)
        self.initialize = jax.jit(
            _jax_wrapped_batched_pgpe_init, static_argnames=['init_policy_params'])

        # ***********************************************************************
        # PARAMETER SAMPLING FUNCTIONS
        #
        # ***********************************************************************

        # sample from i.i.d. Normal(0, sigma)
        def _jax_wrapped_mu_noise(key, sigma):
            return sigma * random.normal(key, shape=jnp.shape(sigma), dtype=real_dtype)

        # this samples a noise variable epsilon* from epsilon with the N(0, 1) density
        # according to super-symmetric sampling paper
        def _jax_wrapped_eps(sigma, epsilon):
            phi = 0.67449 * sigma
            a = jnp.divide(sigma - jnp.abs(epsilon), sigma)

            # more accurate formula
            if super_symmetric_accurate:
                aa = jnp.abs(a)
                c1, c2, c3 = -0.06655, -0.9706, 0.124
                term_neg = jnp.exp(jnp.where(
                    aa == 1.,
                    2. * c1 + c2,
                    jnp.divide(c1 * aa * (aa * aa - 1.), jnp.log(aa)) + c2 * aa
                ))
                safe_base = jnp.maximum(1. - aa ** 3, jnp.finfo(aa.dtype).tiny)
                term_pos = jnp.divide(jnp.exp(aa), jnp.power(safe_base, c3 * aa))
                epsilon_star = jnp.where(
                    epsilon == 0., 
                    0., 
                    jnp.sign(epsilon) * phi * jnp.where(a < 0, term_neg, term_pos)
                )
                
            # less accurate and simple formula
            else:
                epsilon_star = jnp.sign(epsilon) * phi * jnp.exp(a)
            return epsilon_star

        # optional projection of sampled parameters to the valid set of parameters
        def _jax_wrapped_projection(params, hyperparams):
            planner_state = JaxPlannerState(policy_params=params, hyperparams=hyperparams)
            return projection(planner_state)

        # implements baseline-free super-symmetric sampling to generate 4 trajectories
        # this type of sampling removes the need for the baseline completely
        def _jax_wrapped_sample_params(key, mu, sigma, hyperparams):

            # this samples the basic two policy parameters from Gaussian(mean, sigma)
            # using the control variates
            treedef = jax.tree_util.tree_structure(sigma)
            keys = random.split(key, num=treedef.num_leaves)
            keys_pytree = jax.tree_util.tree_unflatten(treedef=treedef, leaves=keys)
            epsilon = jax.tree_util.tree_map(_jax_wrapped_mu_noise, keys_pytree, sigma)
            p1 = jax.tree_util.tree_map(jnp.add, mu, epsilon)
            p2 = jax.tree_util.tree_map(jnp.subtract, mu, epsilon)
            if project_samples:
                p1, _ = _jax_wrapped_projection(p1, hyperparams)
                p2, _ = _jax_wrapped_projection(p2, hyperparams)

            # sumer-symmetric sampling removes the need for a baseline but requires
            # two additional policies to be sampled
            if super_symmetric:
                epsilon_star = jax.tree_util.tree_map(_jax_wrapped_eps, sigma, epsilon)     
                p3 = jax.tree_util.tree_map(jnp.add, mu, epsilon_star)
                p4 = jax.tree_util.tree_map(jnp.subtract, mu, epsilon_star)
                if project_samples:
                    p3, _ = _jax_wrapped_projection(p3, hyperparams)
                    p4, _ = _jax_wrapped_projection(p4, hyperparams)
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
                r_mu1 = jnp.divide(r1 - r2, 2 * scale1)
                r_mu2 = jnp.divide(r3 - r4, 2 * scale2)
                grad = -(r_mu1 * epsilon + r_mu2 * epsilon_star)
            
            # for the basic pgpe
            else:
                if scale_reward:
                    scale = jnp.maximum(min_reward_scale, m - (r1 + r2) / 2)
                else:
                    scale = 1.0
                r_mu = jnp.divide(r1 - r2, 2 * scale)
                grad = -r_mu * epsilon
            return grad
        
        #  gradient with respect to std. deviation
        def _jax_wrapped_sigma_grad(epsilon, epsilon_star, sigma, r1, r2, r3, r4, m, ent):

            # for super symmetric sampling
            if super_symmetric:
                epsilon_tau = jnp.where(r1 + r2 >= r3 + r4, epsilon, epsilon_star)
                s = (jnp.divide(jnp.square(epsilon_tau), jnp.power(sigma, 3)) - 
                     jnp.divide(1., sigma))
                if scale_reward:
                    scale = jnp.maximum(min_reward_scale, m - (r1 + r2 + r3 + r4) / 4)
                else:
                    scale = 1.0
                r_sigma = jnp.divide((r1 + r2) - (r3 + r4), 4 * scale)
            
            # for basic pgpe
            else:
                s = (jnp.divide(jnp.square(epsilon), jnp.power(sigma, 3)) - 
                     jnp.divide(1., sigma))
                if scale_reward:
                    scale = jnp.maximum(min_reward_scale, jnp.abs(m))
                else:
                    scale = 1.0
                r_sigma = jnp.divide(r1 + r2, 2 * scale)

            return -(r_sigma * s + jnp.divide(ent, sigma))
            
        # sample policy parameters and evaluate their returns
        def _jax_wrapped_pgpe_returns(sim_state, planner_state, stats):
            mu, sigma = stats['mu'], stats['sigma']
            key, subkey = random.split(sim_state.key)
            p1, p2, p3, p4, epsilon, epsilon_star = _jax_wrapped_sample_params(
                key, mu, sigma, planner_state.hyperparams)
            sim_state = sim_state.replace(key=subkey)
            samples = (p1, p2, p3, p4) if super_symmetric else (p1, p2)
            samples = jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves), *samples)
            returns = -jax.vmap(lambda params: loss_fn(
                sim_state, planner_state.replace(policy_params=params))[0])(samples)
            return epsilon, epsilon_star, returns

        # centered ranks of the returns in [-0.5, 0.5]
        def _jax_wrapped_centered_ranks(returns):
            flat = jnp.ravel(jnp.where(jnp.isfinite(returns), returns, -jnp.inf))
            less = jnp.sum(flat[None, :] < flat[:, None], axis=1)
            equal = jnp.sum(flat[None, :] == flat[:, None], axis=1)
            ranks = (less + 0.5 * (equal - 1)) / (flat.size - 1) - 0.5
            return jnp.reshape(ranks, jnp.shape(returns))

        # calculate the policy gradients from the returns of the sampled parameters
        def _jax_wrapped_pgpe_grad(epsilon, epsilon_star, sigma, returns, r_max, ent):
            if super_symmetric:
                r1, r2, r3, r4 = returns
            else:
                r1, r2 = returns
                r3, r4 = r1, r2
            grad_mu = jax.tree_util.tree_map(
                partial(_jax_wrapped_mu_grad, r1=r1, r2=r2, r3=r3, r4=r4, m=r_max),
                epsilon, epsilon_star
            )
            grad_sigma = jax.tree_util.tree_map(
                partial(_jax_wrapped_sigma_grad, r1=r1, r2=r2, r3=r3, r4=r4, m=r_max, ent=ent),
                epsilon, epsilon_star, sigma
            )
            return grad_mu, grad_sigma

        # calculate the policy gradients with batching on the first dimension
        def _jax_wrapped_pgpe_grad_batched(sim_state, planner_state, stats, r_max, ent):

            # sample policy parameters and evaluate their returns
            if batch_size == 1:
                samples = _jax_wrapped_pgpe_returns(sim_state, planner_state, stats)
                epsilon, epsilon_star, returns = jax.tree_util.tree_map(
                    lambda x: x[None], samples)
            else:
                keys = random.split(sim_state.key, num=batch_size)
                sim_state = sim_state.replace(key=keys)
                vmap_axes = (JaxRDDLSimState(key=0), None, None)
                epsilon, epsilon_star, returns = jax.vmap(
                    _jax_wrapped_pgpe_returns, in_axes=vmap_axes)(
                        sim_state, planner_state, stats)
            loss = -jnp.mean(returns)

            # the samples give no gradient if any of their returns is invalid, unless
            # the returns are replaced by their centered ranks
            if centered_ranks:
                returns = _jax_wrapped_centered_ranks(returns)
            else:
                valid = jnp.all(jnp.isfinite(returns), axis=1, keepdims=True)
                r_max = jnp.maximum(r_max, jnp.max(jnp.where(valid, returns, -jnp.inf)))
                returns = jnp.where(valid, returns, 0.)

            # calculate the average gradient over the batch
            mu_grad, sigma_grad = jax.vmap(
                _jax_wrapped_pgpe_grad, in_axes=(0, 0, None, 0, None, None))(
                    epsilon, epsilon_star, stats['sigma'], returns, r_max, ent)
            mu_grad, sigma_grad = jax.tree_util.tree_map(
                partial(jnp.mean, axis=0), (mu_grad, sigma_grad))
            return {'mu': mu_grad, 'sigma': sigma_grad}, r_max, loss
        
        # ***********************************************************************
        # PARAMETER UPDATE
        #
        # ***********************************************************************

        # estimate KL divergence between two updates
        def _jax_wrapped_pgpe_kl_term(old_mu, old_sigma, mu, sigma):
            return 0.5 * jnp.sum(2 * jnp.log(jnp.divide(sigma, old_sigma)) + 
                                 jnp.square(jnp.divide(old_sigma, sigma)) + 
                                 jnp.square(jnp.divide(mu - old_mu, sigma)) - 1)
        
        # update mean and std. deviation with a gradient step
        def _jax_wrapped_pgpe_update_helper(stats, grad, opt_state):
            updates, new_opt_state = optimizer.update(grad, opt_state, stats)
            new_stats = optax.apply_updates(stats, updates)
            new_stats['sigma'] = jax.tree_util.tree_map(
                partial(jnp.clip, min=sigma_lo, max=sigma_hi), new_stats['sigma'])
            return new_stats, new_opt_state

        def _jax_wrapped_pgpe_update(sim_state, planner_state):
            
            # calculate entropy coefficient
            ent = start_entropy_coeff * jnp.power(entropy_coeff_decay, planner_state.progress)

            # do a single update step
            def _jax_wrapped_pgpe_update_step(_, carry):
                key, stats, r_max, opt_state, *_ = carry

                # do a pgpe update
                key, subkey = random.split(key)
                _sim_state = sim_state.replace(key=subkey)
                grad, new_r_max, loss = _jax_wrapped_pgpe_grad_batched(
                    _sim_state, planner_state, stats, r_max, ent)
                new_stats, new_opt_state = _jax_wrapped_pgpe_update_helper(
                    stats, grad, opt_state)
                
                # respect KL divergence constraint by scaling gradients
                if max_kl is not None:
                    kl = jax.tree_util.tree_map(
                        _jax_wrapped_pgpe_kl_term, 
                        stats['mu'], stats['sigma'], new_stats['mu'], new_stats['sigma']
                    )
                    sum_kl = jax.tree_util.tree_reduce(jnp.add, kl)
                    scale = jnp.minimum(1.0, jnp.sqrt(jnp.divide(max_kl, sum_kl + self.eps)))
                    grad = jax.tree_util.tree_map(lambda g: g * scale, grad)
                    new_stats, new_opt_state = _jax_wrapped_pgpe_update_helper(
                        stats, grad, opt_state)
                    
                # apply a gradient projection
                new_stats['mu'], converged = _jax_wrapped_projection(
                    new_stats['mu'], planner_state.hyperparams)
                return (key, new_stats, new_r_max, new_opt_state, converged, loss, grad['mu'])

            # do an unrolled update
            opt_state = planner_state.opt_state
            stats = {'mu': planner_state.policy_params, 'sigma': opt_state['sigma']}
            carry = (sim_state.key, stats, opt_state['r_max'], opt_state['optimizer'],
                     True, jnp.zeros((), dtype=real_dtype),
                     jax.tree_util.tree_map(jnp.zeros_like, stats['mu']))
            _, stats, r_max, optimizer_state, converged, loss, grad = jax.lax.fori_loop(
                lower=0, upper=self.steps_per_update, 
                body_fun=_jax_wrapped_pgpe_update_step, 
                init_val=carry
            )
            opt_state = {'sigma': stats['sigma'], 'r_max': r_max,
                         'optimizer': optimizer_state}
            return stats['mu'], opt_state, converged, loss, grad

        # for parallel policy update
        def _jax_wrapped_batched_pgpe_updates(sim_state, planner_state):
            keys = random.split(sim_state.key, num=1 + parallel_updates)
            sim_state = sim_state.replace(key=keys[1:])
            vmap_axes = (JaxRDDLSimState(key=0, model_params=0), 
                         JaxPlannerState(policy_params=0, opt_state=0))
            params, opt_state, converged, loss, grad = jax.vmap(
                _jax_wrapped_pgpe_update, in_axes=vmap_axes)(sim_state, planner_state)
            planner_state = planner_state.replace(policy_params=params, opt_state=opt_state)
            sim_state = sim_state.replace(key=keys[0])
            zero_grads = jax.vmap(_jax_zero_tree)(grad)
            return sim_state, planner_state, loss, converged, grad, zero_grads
        update_fn = jax.jit(_jax_wrapped_batched_pgpe_updates)

        # the exact model is optimized, which advances the test simulation state: the
        # key of the train simulation state is advanced with it
        def _jax_wrapped_update(train_sim_state, test_sim_state, planner_state):
            test_sim_state, planner_state, loss, converged, grad, zero_grads = update_fn(
                test_sim_state, planner_state)
            train_sim_state = train_sim_state.replace(key=test_sim_state.key)
            log = {'grad': grad}
            return (train_sim_state, test_sim_state, planner_state, loss, converged, log,
                    zero_grads)
        self.update = _jax_wrapped_update


class JaxLMMAESPlanner(JaxPlanner):
    '''Limited-memory matrix adaptation evolution strategy (LM-MA-ES) [8], that
    optimizes the policy on the exact model by searching with a Gaussian over the policy
    parameters, whose covariance is the identity transformed by a small number of
    evolution paths: this learns correlations between parameters at a cost linear in
    the number of parameters. Samples are orthogonal [10] and share the random numbers
    of the rollouts.

    Optionally, the covariance is scaled by a diagonal matrix D as in D (I + V V^T) D [9],
    whose log is adapted by the natural gradient of the expected rank-based utility as in
    separable natural evolution strategies (SNES) [7]. SNES is recovered with
    num_directions=0, step_size_adaptation=False, diagonal_scaling=True,
    mirrored_sampling=True, orthogonal_sampling=False and weights='utilities'.'''

    def __init__(self, rddl: Optional[RDDLLiftedModel]=None,
                 plan: Optional[JaxPlan]=None,
                 population_size: Optional[int]=None,
                 num_directions: Optional[int]=None,
                 init_sigma: float=1.0,
                 step_size_adaptation: bool=True,
                 diagonal_scaling: bool=False,
                 learning_rate_diagonal: Optional[float]=None,
                 mirrored_sampling: bool=True,
                 orthogonal_sampling: bool=True,
                 weights: str='truncation',
                 steps_per_update: int=1,
                 **planner_kwargs) -> None:
        '''Creates a new LM-MA-ES planner.

        :param rddl: the RDDL domain to optimize: None if used as an auxiliary planner
        :param plan: the policy/plan representation to optimize
        :param population_size: how many policy parameters to sample per optimization
        step: None uses the default 4 + floor(3 ln n) for n policy parameters (rounded
        up to an even number for mirrored sampling)
        :param num_directions: how many evolution paths define the covariance (0 for a
        diagonal covariance): None uses the default 4 + floor(3 ln n) for n policy
        parameters
        :param init_sigma: initial step size of the search
        :param step_size_adaptation: whether to adapt the step size from the length of
        its evolution path, otherwise it remains init_sigma
        :param diagonal_scaling: whether to adapt a diagonal scaling of the covariance
        :param learning_rate_diagonal: learning rate of the log diagonal scaling: None
        uses the default (3 + ln n) / (5 sqrt(n)) of SNES for n policy parameters
        :param mirrored_sampling: whether to sample mirrored pairs of parameters
        :param orthogonal_sampling: whether to sample orthogonal parameters (in blocks
        of the number of parameters), each with the length of a Gaussian sample
        :param weights: recombination weights of the samples sorted from best to worst,
        either 'truncation' for the log-decreasing weights of the best half of the
        samples, or 'utilities' for the rank-based utilities of all samples with a
        baseline as in SNES
        :param steps_per_update: how many optimization steps are performed per planner
        update (useful for partially jit compiling the update loop)
        :param **planner_kwargs: arguments of the planning problem (see JaxPlanner)
        '''
        if weights not in ('truncation', 'utilities'):
            raise ValueError(f'Invalid weights <{weights}>, must be truncation or utilities.')
        self.population_size = population_size
        self.num_directions = num_directions
        self.init_sigma = init_sigma
        self.step_size_adaptation = step_size_adaptation
        self.diagonal_scaling = diagonal_scaling
        self.learning_rate_diagonal = learning_rate_diagonal
        self.mirrored_sampling = mirrored_sampling
        self.orthogonal_sampling = orthogonal_sampling
        self.weights = weights
        self.steps_per_update = steps_per_update
        super(JaxLMMAESPlanner, self).__init__(rddl=rddl, plan=plan, **planner_kwargs)

    def summarize_optimizer(self) -> str:
        return (f'[INFO] LM-MA-ES hyper-parameters:\n'
                f'    population_size    ={self.population_size}\n'
                f'    num_directions     ={self.num_directions}\n'
                f'    init_sigma         ={self.init_sigma}\n'
                f'    step_size_adapt    ={self.step_size_adaptation}\n'
                f'    diagonal_scaling   ={self.diagonal_scaling}\n'
                f'    learning_rate_diag ={self.learning_rate_diagonal}\n'
                f'    mirrored_sampling  ={self.mirrored_sampling}\n'
                f'    orthogonal_samples ={self.orthogonal_sampling}\n'
                f'    weights            ={self.weights}\n'
                f'    steps_per_update   ={self.steps_per_update}\n')

    def _jax_compile_optimizer(self) -> None:
        loss_fn = self.single_test_loss
        projection = self.plan.projection
        real_dtype = self.test_compiled.REAL
        parallel_updates = self.parallel_updates
        init_policy = self.init_policy
        sigma0 = self.init_sigma
        use_step_size = self.step_size_adaptation
        use_diagonal = self.diagonal_scaling
        mirrored = self.mirrored_sampling
        orthogonal = self.orthogonal_sampling

        # the search operates on the policy parameters flattened to a vector
        sim_state = self.compiled.init_sim_state(
            random.PRNGKey(0), None, self.batch_size_train, print_warnings=False)
        param_shapes, _ = jax.eval_shape(self.plan.initializer, sim_state)
        param_template = jax.tree_util.tree_map(
            lambda leaf: jnp.zeros(leaf.shape, dtype=leaf.dtype), param_shapes)
        _, unravel = jax.flatten_util.ravel_pytree(param_template)
        ravel = lambda params: jax.flatten_util.ravel_pytree(params)[0]
        n = max(sum(int(np.prod(leaf.shape))
                    for leaf in jax.tree_util.tree_leaves(param_shapes)), 1)

        # default hyper-parameters of LM-MA-ES depend on the number of parameters
        population_size = self.population_size
        if population_size is None:
            population_size = 4 + int(3 * np.log(n))
        if mirrored:
            population_size = 2 * ((population_size + 1) // 2)
        num_directions = self.num_directions
        if num_directions is None:
            num_directions = 4 + int(3 * np.log(n))

        # recombination weights of the samples sorted from best to worst
        ranks = np.arange(1, population_size + 1)
        if self.weights == 'truncation':
            num_parents = max(population_size // 2, 1)
            weights = np.maximum(0., np.log(num_parents + 0.5) - np.log(ranks))
            weights[num_parents:] = 0.
            weights = weights / np.sum(weights)
        else:
            weights = np.maximum(0., np.log(population_size / 2 + 1) - np.log(ranks))
            weights = weights / np.sum(weights) - 1. / population_size
        mu_eff = 1. / np.sum(np.square(weights))

        # learning rates of the step size and of the directions
        c_sigma = 2. * population_size / n
        powers = np.arange(num_directions)
        c_d = 1. / (np.power(1.5, powers) * n)
        c_c = population_size / (np.power(4., powers) * n)
        path_scale_sigma = np.sqrt(mu_eff * c_sigma * (2. - c_sigma))
        path_scale_c = np.sqrt(mu_eff * c_c * (2. - c_c))
        weights = jnp.asarray(weights, dtype=real_dtype)
        c_d = jnp.asarray(c_d, dtype=real_dtype)
        c_c = jnp.asarray(c_c, dtype=real_dtype)
        path_scale_c = jnp.asarray(path_scale_c, dtype=real_dtype)

        # rank-based utilities of the samples for the diagonal scaling as in SNES
        lr_diagonal = self.learning_rate_diagonal
        if lr_diagonal is None:
            lr_diagonal = (3 + np.log(n)) / (5 * np.sqrt(n))
        utilities = np.maximum(0., np.log(population_size / 2 + 1) - np.log(ranks))
        utilities = utilities / np.sum(utilities) - 1. / population_size
        utilities = jnp.asarray(utilities, dtype=real_dtype)

        # ***********************************************************************
        # INITIALIZATION OF POLICY
        #
        # ***********************************************************************

        # the policy parameters are the mean of the search, and the optimizer state
        # holds the step size, its evolution path and the directions of the covariance
        def _jax_wrapped_batched_lmmaes_init(sim_state, planner_state, init_policy_params=True):
            if init_policy_params:
                params, hyperparams = init_policy(sim_state)
                planner_state = planner_state.replace(
                    policy_params=params, hyperparams=hyperparams)
            opt_state = {
                'sigma': jnp.full((parallel_updates,), sigma0, dtype=real_dtype),
                'path': jnp.zeros((parallel_updates, n), dtype=real_dtype),
                'directions': jnp.zeros(
                    (parallel_updates, num_directions, n), dtype=real_dtype),
                'iteration': jnp.zeros((parallel_updates,), dtype=jnp.int32)
            }
            if use_diagonal:
                opt_state['diagonal'] = jnp.ones((parallel_updates, n), dtype=real_dtype)
            return planner_state.replace(opt_state=opt_state)
        self.initialize = jax.jit(
            _jax_wrapped_batched_lmmaes_init, static_argnames=['init_policy_params'])

        # ***********************************************************************
        # PARAMETER UPDATE
        #
        # ***********************************************************************

        # transforms samples z ~ N(0, I) by the directions learned so far: each factor
        # (1 - c) I + c v v^T of the transformation M is symmetric
        def _jax_wrapped_transform(z, directions, iteration):
            if num_directions == 0:
                return z
            def _jax_wrapped_apply_direction(j, d):
                v = directions[j]
                d_new = (1. - c_d[j]) * d + c_d[j] * jnp.outer(d @ v, v)
                return jnp.where(j < iteration, d_new, d)
            return jax.lax.fori_loop(0, num_directions, _jax_wrapped_apply_direction, z)

        # applies the inverse transpose of M to z, inverting each factor in closed form
        def _jax_wrapped_inverse_transpose(z, directions, iteration):
            if num_directions == 0:
                return z
            def _jax_wrapped_invert_direction(j, e):
                v, c = directions[j], c_d[j]
                denom = (1. - c) + c * jnp.sum(jnp.square(v))
                e_new = (e - c * jnp.outer(e @ v, v) / denom) / (1. - c)
                return jnp.where(j < iteration, e_new, e)
            return jax.lax.fori_loop(0, num_directions, _jax_wrapped_invert_direction, z)

        # applies the inverse of M to z, inverting each factor in closed form in the
        # reverse order of the transformation
        def _jax_wrapped_inverse(z, directions, iteration):
            if num_directions == 0:
                return z
            def _jax_wrapped_invert_direction(i, e):
                j = num_directions - 1 - i
                v, c = directions[j], c_d[j]
                denom = (1. - c) + c * jnp.sum(jnp.square(v))
                e_new = (e - c * jnp.outer(e @ v, v) / denom) / (1. - c)
                return jnp.where(j < iteration, e_new, e)
            return jax.lax.fori_loop(0, num_directions, _jax_wrapped_invert_direction, z)

        # samples z ~ N(0, I) made orthogonal in blocks of at most n: each sample keeps
        # its length, which is independent of its direction
        def _jax_wrapped_orthogonal_normal(key, num_samples):
            block = min(num_samples, n)
            num_blocks = -(-num_samples // block)
            g = random.normal(key, shape=(num_blocks, block, n), dtype=real_dtype)
            q, r = jnp.linalg.qr(jnp.swapaxes(g, 1, 2))
            signs = jnp.sign(jnp.diagonal(r, axis1=1, axis2=2))
            q = jnp.swapaxes(q, 1, 2) * signs[..., None]
            z = q * jnp.linalg.norm(g, axis=2, keepdims=True)
            return jnp.reshape(z, (-1, n))[:num_samples]

        # samples z ~ N(0, I) and the transformed samples d = M z (optionally mirrored)
        def _jax_wrapped_sample(key, opt_state):
            num_samples = population_size // 2 if mirrored else population_size
            if orthogonal:
                z = _jax_wrapped_orthogonal_normal(key, num_samples)
            else:
                z = random.normal(key, shape=(num_samples, n), dtype=real_dtype)
            d = _jax_wrapped_transform(z, opt_state['directions'], opt_state['iteration'])
            if mirrored:
                z, d = jnp.concatenate([z, -z]), jnp.concatenate([d, -d])
            return z, d

        def _jax_wrapped_lmmaes_update(sim_state, planner_state):

            # do a single update step
            def _jax_wrapped_lmmaes_update_step(_, carry):
                key, mu, opt_state, *_ = carry
                sigma = opt_state['sigma']
                diagonal = opt_state['diagonal'] if use_diagonal else 1.
                key, sample_key, rollout_key = random.split(key, num=3)

                # evaluate samples with common random numbers in the rollouts
                mu_flat = ravel(mu)
                z, d = _jax_wrapped_sample(sample_key, opt_state)
                _sim_state = sim_state.replace(key=rollout_key)
                losses = jax.vmap(
                    lambda dd: loss_fn(_sim_state, planner_state.replace(
                        policy_params=unravel(mu_flat + sigma * diagonal * dd)))[0]
                )(d)

                # recombine the samples sorted from best to worst, with invalid returns
                # ranked last
                order = jnp.argsort(jnp.where(jnp.isfinite(losses), losses, jnp.inf))
                z_sorted, d_sorted = z[order], d[order]
                z_mean = weights @ z_sorted
                d_mean = weights @ d_sorted

                # update the mean, the evolution paths of the step size and covariance
                step = sigma * diagonal * d_mean
                mu = unravel(mu_flat + step)
                mu, converged = projection(planner_state.replace(policy_params=mu))

                # the evolution paths follow the step of the mean after projection, whose
                # length is at most that of the step before projection
                correction = ravel(mu) - (mu_flat + step)
                step = step + correction
                z_step = z_mean + _jax_wrapped_inverse(
                    (correction / jnp.maximum(sigma * diagonal, 1e-30))[None, :],
                    opt_state['directions'], opt_state['iteration'])[0]
                z_norm = jnp.sqrt(jnp.sum(jnp.square(z_step)))
                z_mean_norm = jnp.sqrt(jnp.sum(jnp.square(z_mean)))
                z_mean = jnp.where(z_norm > z_mean_norm,
                                   z_step * (z_mean_norm / jnp.maximum(z_norm, 1e-30)),
                                   z_step)
                new_opt_state = dict(opt_state)
                if num_directions > 0:
                    new_opt_state['directions'] = (
                        (1. - c_c)[:, None] * opt_state['directions'] +
                        path_scale_c[:, None] * z_mean[None, :])
                    new_opt_state['iteration'] = opt_state['iteration'] + 1
                if use_step_size:
                    path = (1. - c_sigma) * opt_state['path'] + path_scale_sigma * z_mean
                    new_opt_state['path'] = path
                    new_opt_state['sigma'] = sigma * jnp.exp(
                        0.5 * c_sigma * (jnp.sum(jnp.square(path)) / n - 1.))

                # natural gradient step of the log diagonal scaling: the score of the
                # log scaling of each parameter is d * (M^-T z) - 1
                if use_diagonal:
                    e_sorted = _jax_wrapped_inverse_transpose(
                        z_sorted, opt_state['directions'], opt_state['iteration'])
                    score = d_sorted * e_sorted - 1.
                    new_opt_state['diagonal'] = diagonal * jnp.exp(
                        0.5 * lr_diagonal * (utilities @ score))
                return (key, mu, new_opt_state, converged, jnp.mean(losses), unravel(-step))

            # do an unrolled update
            mu = planner_state.policy_params
            carry = (sim_state.key, mu, planner_state.opt_state, True,
                     jnp.zeros((), dtype=real_dtype),
                     jax.tree_util.tree_map(jnp.zeros_like, mu))
            _, mu, opt_state, converged, loss, grad = jax.lax.fori_loop(
                lower=0, upper=self.steps_per_update,
                body_fun=_jax_wrapped_lmmaes_update_step,
                init_val=carry
            )
            return mu, opt_state, converged, loss, grad

        # for parallel policy update
        def _jax_wrapped_batched_lmmaes_updates(sim_state, planner_state):
            keys = random.split(sim_state.key, num=1 + parallel_updates)
            sim_state = sim_state.replace(key=keys[1:])
            vmap_axes = (JaxRDDLSimState(key=0, model_params=0),
                         JaxPlannerState(policy_params=0, opt_state=0))
            params, opt_state, converged, loss, grad = jax.vmap(
                _jax_wrapped_lmmaes_update, in_axes=vmap_axes)(sim_state, planner_state)
            planner_state = planner_state.replace(policy_params=params, opt_state=opt_state)
            sim_state = sim_state.replace(key=keys[0])
            zero_grads = jax.vmap(_jax_zero_tree)(grad)
            return sim_state, planner_state, loss, converged, grad, zero_grads
        update_fn = jax.jit(_jax_wrapped_batched_lmmaes_updates)

        # the exact model is optimized, which advances the test simulation state: the
        # key of the train simulation state is advanced with it
        def _jax_wrapped_update(train_sim_state, test_sim_state, planner_state):
            test_sim_state, planner_state, loss, converged, grad, zero_grads = update_fn(
                test_sim_state, planner_state)
            train_sim_state = train_sim_state.replace(key=test_sim_state.key)
            log = {'grad': grad}
            return (train_sim_state, test_sim_state, planner_state, loss, converged, log,
                    zero_grads)
        self.update = _jax_wrapped_update


class JaxBackpropPlanner(JaxPlanner):
    '''A class for optimizing an action sequence in the given RDDL MDP using 
    gradient descent through the relaxed model.'''

    _DEFAULT_AUXILIARY = JaxLMMAESPlanner()

    def __init__(self, rddl: Optional[RDDLLiftedModel]=None,
                 plan: Optional[JaxPlan]=None,
                 steps_per_update: int=1,
                 optimizer: Callable[..., optax.GradientTransformation]=optax.rmsprop,
                 optimizer_kwargs: Optional[Kwargs]=None,
                 clip_grad: Optional[float]=None,
                 line_search_kwargs: Optional[Kwargs]=None,
                 noise_kwargs: Optional[Kwargs]=None,
                 ema_decay: Optional[float]=None,
                 reduce_on_plateau_kwargs: Optional[Kwargs]=None,
                 auxiliary: Optional[JaxPlanner]=_DEFAULT_AUXILIARY,
                 **planner_kwargs) -> None:
        '''Creates a new gradient-based algorithm for optimizing action sequences
        (plan) in the given RDDL.

        :param rddl: the RDDL domain to optimize: None if used as an auxiliary planner
        :param plan: the policy/plan representation to optimize
        :param steps_per_update: how many optimization steps are performed per planner
        update (useful for partially jit compiling the update loop)
        :param optimizer: a factory for an optax SGD algorithm
        :param optimizer_kwargs: a dictionary of parameters to pass to the SGD
        factory (e.g. which parameters are controllable externally)
        :param clip_grad: maximum magnitude of gradient updates
        :param line_search_kwargs: parameters to pass to optional line search
        method to scale learning rate
        :param noise_kwargs: parameters of optional gradient noise
        :param ema_decay: optional exponential moving average of past parameters
        :param reduce_on_plateau_kwargs: optional keyword arguments for
        optax.contrib.reduce_on_plateau to decay learning rate on objective plateaus
        :param auxiliary: optional planner to run alongside this planner (the default
        is not used when this planner is itself created as an auxiliary planner)
        :param **planner_kwargs: arguments of the planning problem (see JaxPlanner)
        '''
        if rddl is None and auxiliary is JaxBackpropPlanner._DEFAULT_AUXILIARY:
            auxiliary = None
        self.steps_per_update = steps_per_update
        self.optimizer_name = optimizer
        if optimizer_kwargs is None:
            optimizer_kwargs = {'learning_rate': 0.1}
        self.optimizer_kwargs = optimizer_kwargs
        self.clip_grad = clip_grad
        self.line_search_kwargs = line_search_kwargs
        self.noise_kwargs = noise_kwargs
        self.ema_decay = ema_decay
        self.reduce_on_plateau_kwargs = reduce_on_plateau_kwargs

        # set optimizer
        self.optimizer, _, self._optimizer_uses_loss = _build_optax_optimizer(
            optimizer, optimizer_kwargs, clip_grad, noise_kwargs, line_search_kwargs, 
            ema_decay, reduce_on_plateau_kwargs
        )
        super(JaxBackpropPlanner, self).__init__(
            rddl=rddl, plan=plan, auxiliary=auxiliary, **planner_kwargs)

    def summarize_optimizer(self) -> str:
        result = (f'[INFO] optimizer hyper-parameters:\n'
                  f'    optimizer         ={self.optimizer_name}\n'
                  f'    optimizer args    ={self.optimizer_kwargs}\n'
                  f'    clip_gradient     ={self.clip_grad}\n'
                  f'    line_search_kwargs={self.line_search_kwargs}\n'
                  f'    noise_kwargs      ={self.noise_kwargs}\n'
                  f'    ema_decay         ={self.ema_decay}\n'
                  f'    reduce_on_plateau ={self.reduce_on_plateau_kwargs}\n'
                  f'    steps_per_update  ={self.steps_per_update}\n')
        return result

    def _jax_train_action_repair(self) -> Optional[Callable]:
        if self.action_repair is not None:
            return self.action_repair.relaxed()
        else:
            return None

    def _jax_compile_optimizer(self) -> None:
        self.initialize = self._jax_init_optimizer()
        self.update = self._jax_update(self.single_train_loss)

    def _jax_init_optimizer(self):
        init_policy = self.init_policy
        optimizer = self.optimizer

        # initialize both the policy and its optimizer for all parallel policies
        def _jax_wrapped_batched_init_policy(sim_state, planner_state, init_policy_params=True):
            if init_policy_params:
                params, hyperparams = init_policy(sim_state)
                planner_state = planner_state.replace(
                    policy_params=params, hyperparams=hyperparams)
            opt_state = jax.vmap(optimizer.init, in_axes=0)(planner_state.policy_params)
            return planner_state.replace(opt_state=opt_state, opt_aux={})
        return jax.jit(_jax_wrapped_batched_init_policy, static_argnames=['init_policy_params'])
        
    def _jax_update(self, loss):
        optimizer = self.optimizer
        projection = self.plan.projection
        use_ls = self.line_search_kwargs is not None
        use_loss_metric = self._optimizer_uses_loss
        num_parallel = self.parallel_updates
        
        # calculate the plan gradient w.r.t. return loss and update optimizer
        # also perform a projection step to satisfy constraints on actions
        def _jax_wrapped_plan_update(sim_state, planner_state):

            # calculate gradient of the loss with respect to the policy parameters 
            def _jax_wrapped_plan_params_loss(policy_params, _sim_state):
                return loss(_sim_state, planner_state.replace(policy_params=policy_params))                                           
            grad_fn = jax.value_and_grad(_jax_wrapped_plan_params_loss, has_aux=True)

            # perform a single gradient descent update
            def _jax_wrapped_plan_update_step(_, carry):
                key, model_params, params, opt_state, *_ = carry

                # calculate the gradient of the loss with respect to the policy
                key, subkey = random.split(key)
                _sim_state = sim_state.replace(key=subkey, model_params=model_params)
                (loss_val, (log, model_params)), grad = grad_fn(params, _sim_state)
            
                # require a slightly different update if line search is used
                if use_ls:
                    updates, opt_state = optimizer.update(
                        grad, opt_state, params=params, value=loss_val, grad=grad, 
                        value_fn=lambda p, s: _jax_wrapped_plan_params_loss(p, s)[0], 
                        s=_sim_state
                    )
                elif use_loss_metric:
                    updates, opt_state = optimizer.update(
                        grad, opt_state, params=params, value=loss_val)
                else:
                    updates, opt_state = optimizer.update(grad, opt_state, params=params) 

                # apply optimizer and optional policy projection
                params = optax.apply_updates(params, updates)
                params, converged = projection(planner_state.replace(policy_params=params))
                log['grad'] = grad
                log['updates'] = updates
                return (key, model_params, params, opt_state, loss_val, converged, log)
            
            # do a single update then an unrolled update in JAX for any remaining steps
            # necessary to fill the log
            carry = (sim_state.key, sim_state.model_params, 
                     planner_state.policy_params, planner_state.opt_state, 0., True, {})
            carry = _jax_wrapped_plan_update_step(0, carry)
            if self.steps_per_update > 1:
                carry = jax.lax.fori_loop(
                    lower=0, upper=self.steps_per_update - 1, 
                    body_fun=_jax_wrapped_plan_update_step, 
                    init_val=carry
                )
            _, *rest, end_log = carry
            zero_grads = _jax_zero_tree(end_log['grad'])
            return (*rest, end_log, zero_grads)
        
        # for parallel policy update, just do each policy update in parallel
        def _jax_wrapped_batched_plan_update(sim_state, planner_state):
            keys = random.split(sim_state.key, num=1 + num_parallel)
            sim_state = sim_state.replace(key=keys[1:])
            vmap_axes = (JaxRDDLSimState(key=0, model_params=0), 
                         JaxPlannerState(policy_params=0, opt_state=0, opt_aux=0))
            model_params, params, opt_state, *rest = jax.vmap(
                _jax_wrapped_plan_update, in_axes=vmap_axes)(sim_state, planner_state)
            sim_state = sim_state.replace(key=keys[0], model_params=model_params)
            planner_state = planner_state.replace(policy_params=params, opt_state=opt_state)
            return (sim_state, planner_state, *rest)
        update_fn = jax.jit(_jax_wrapped_batched_plan_update)

        # the relaxed model is optimized, which advances the train simulation state only
        def _jax_wrapped_update(train_sim_state, test_sim_state, planner_state):
            train_sim_state, planner_state, *rest = update_fn(train_sim_state, planner_state)
            return (train_sim_state, test_sim_state, planner_state, *rest)
        return _jax_wrapped_update


def create_planner(rddl: RDDLLiftedModel, **planner_args) -> JaxPlanner:
    '''Creates a planner for the given RDDL from the planner arguments returned by
    load_config: the planner class is given by the 'planner' argument (the planner
    key in the config file), and is JaxBackpropPlanner if not specified.

    :param rddl: the RDDL domain to optimize
    :param **planner_args: arguments of the planner, and optionally its class
    '''
    planner_class = planner_args.pop('planner', JaxBackpropPlanner)
    return planner_class(rddl=rddl, **planner_args)
