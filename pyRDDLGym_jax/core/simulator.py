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
# [2] Taitler, Ayal, Michael Gimelfarb, Jihwan Jeong, Sriram Gopalakrishnan, Martin 
# Mladenov, Xiaotian Liu, and Scott Sanner. "pyRDDLGym: From RDDL to Gym Environments." 
# In PRL Workshop Series {\textendash} Bridging the Gap Between AI Planning and 
# Reinforcement Learning.
#
# ***********************************************************************


import time
import numpy as np
from typing import Callable, Dict, Optional, Union

import jax

from pyRDDLGym.core.compiler.initializer import RDDLValueInitializer
from pyRDDLGym.core.compiler.model import RDDLLiftedModel
from pyRDDLGym.core.debug.exception import (
    RDDLActionPreconditionNotSatisfiedError,
    RDDLInvalidExpressionError,
    RDDLStateInvariantNotSatisfiedError
)
from pyRDDLGym.core.debug.logger import Logger
from pyRDDLGym.core.parser.expr import Value
from pyRDDLGym.core.simulator import RDDLSimulator

from pyRDDLGym_jax.core.compiler import JaxRDDLCompiler

Args = Dict[str, Union[np.ndarray, Value]]


class JaxRDDLSimulator(RDDLSimulator):
        
    def __init__(self, rddl: RDDLLiftedModel,
                 key: Optional[jax.random.PRNGKey]=None,
                 raise_error: bool=True,
                 logger: Optional[Logger]=None,
                 keep_tensors: bool=False,
                 objects_as_strings: bool=True,
                 python_functions: Optional[Dict[str, Callable]]=None,
                 **compiler_args) -> None:
        '''Creates a new simulator for the given RDDL model with Jax as a backend.
        
        :param rddl: the RDDL model
        :param key: the Jax PRNG key for sampling random variables
        :param raise_error: whether errors are raised as they are encountered in
        the Jax program: unlike the numpy sim, errors cannot be caught in the 
        middle of evaluating a Jax expression; instead they are accumulated and
        returned to the user upon complete evaluation of the expression
        :param logger: to log information about compilation to file
        :param keep_tensors: whether the sampler takes actions and
        returns state in numpy array form
        :param objects_as_strings: whether to return object values as strings (defaults
        to integer indices if False)
        :param python_functions: dictionary of external Python functions to call from RDDL
        :param **compiler_args: keyword arguments to pass to the Jax compiler
        '''
        if key is None:
            key = jax.random.PRNGKey(round(time.time() * 1000))
        self.key = key
        self.raise_error = raise_error
        self.compiler_args = compiler_args
        
        # generate direct sampling with default numpy RNG and operations
        super(JaxRDDLSimulator, self).__init__(
            rddl, logger=logger, 
            keep_tensors=keep_tensors, objects_as_strings=objects_as_strings,
            python_functions=python_functions)
    
    def seed(self, seed: int) -> None:
        super(JaxRDDLSimulator, self).seed(seed)
        self.key = jax.random.PRNGKey(seed)
        
    def _compile(self):
        rddl = self.rddl
        
        # compilation
        compiled = JaxRDDLCompiler(
            rddl, 
            logger=self.logger, 
            python_functions=self.python_functions, 
            **self.compiler_args
        )
        compiled.compile(log_jax_expr=True, heading='SIMULATION MODEL')
        
        self.init_values = compiled.init_values
        self.levels = compiled.levels
        self.traced = compiled.traced
        
        self.invariants = jax.tree_util.tree_map(jax.jit, compiled.invariants)
        self.preconds = jax.tree_util.tree_map(jax.jit, compiled.preconditions)
        self.terminals = jax.tree_util.tree_map(jax.jit, compiled.terminations)
        self.reward = jax.jit(compiled.reward)
        jax_cpfs = jax.tree_util.tree_map(jax.jit, compiled.cpfs)
        self.model_params = compiled.model_aux['params']
        
        # level analysis
        self.cpfs = []  
        for cpfs in self.levels.values():
            for cpf in cpfs:
                expr = jax_cpfs[cpf]
                prange = rddl.variable_ranges[cpf]
                dtype = compiled.JAX_TYPES.get(prange, compiled.INT)
                self.cpfs.append((cpf, expr, dtype))
        
        # initialize all fluent and non-fluent values    
        self.subs = self.init_values.copy() 
        self.state = None 
        self.noop_actions = {var: values 
                             for (var, values) in self.init_values.items() 
                             if rddl.variable_types[var] == 'action-fluent'}
        self.grounded_noop_actions = rddl.ground_vars_with_values(self.noop_actions)
        self.grounded_action_ranges = rddl.ground_vars_with_value(rddl.action_ranges)
        self._pomdp = bool(rddl.observ_fluents)
        
        # cached for performance
        self.invariant_names = [f'Invariant {i}' for i in range(len(rddl.invariants))]        
        self.precond_names = [f'Precondition {i}' for i in range(len(rddl.preconditions))]
        self.terminal_names = [f'Termination {i}' for i in range(len(rddl.terminations))]
        
    def handle_error_code(self, error: int, msg: str) -> None:
        if self.raise_error:
            errors = JaxRDDLCompiler.get_error_messages(error)
            if errors:
                message = f'Internal error in evaluation of {msg}:\n'
                errors = '\n'.join(f'{i + 1}. {s}' for (i, s) in enumerate(errors))
                raise RDDLInvalidExpressionError(message + errors)
    
    def check_state_invariants(self, silent: bool=False) -> bool:
        '''Throws an exception if the state invariants are not satisfied.'''
        for (i, invariant) in enumerate(self.invariants):
            loc = self.invariant_names[i]
            sample, self.key, error, self.model_params = invariant(
                self.subs, self.model_params, self.key)
            self.handle_error_code(error, loc)            
            if not bool(sample):
                if not silent:
                    raise RDDLStateInvariantNotSatisfiedError(
                        f'{loc} is not satisfied.')
                return False
        return True
    
    def check_action_preconditions(self, actions: Args, silent: bool=False) -> bool:
        '''Throws an exception if the action preconditions are not satisfied.'''
        subs = self.subs
        subs.update(actions)
        
        for (i, precond) in enumerate(self.preconds):
            loc = self.precond_names[i]
            sample, self.key, error, self.model_params = precond(
                subs, self.model_params, self.key)
            self.handle_error_code(error, loc)            
            if not bool(sample):
                if not silent:
                    raise RDDLActionPreconditionNotSatisfiedError(
                        f'{loc} is not satisfied for actions {actions}.')
                return False
        return True
    
    def check_terminal_states(self) -> bool:
        '''return True if a terminal state has been reached.'''
        for (i, terminal) in enumerate(self.terminals):
            loc = self.terminal_names[i]
            sample, self.key, error, self.model_params = terminal(
                self.subs, self.model_params, self.key)
            self.handle_error_code(error, loc)
            if bool(sample):
                return True
        return False
    
    def sample_reward(self) -> float:
        '''Samples the current reward given the current state and action.'''
        reward, self.key, error, self.model_params = self.reward(
            self.subs, self.model_params, self.key)
        self.handle_error_code(error, 'reward function')
        return float(reward)
    
    def step(self, actions: Args) -> Args:
        '''Samples and returns the next state from the cpfs.
        
        :param actions: a dict mapping current action fluents to their values
        '''
        rddl = self.rddl
        keep_tensors = self.keep_tensors
        subs = self.subs
        subs.update(actions)
        
        # compute CPFs in topological order
        for (cpf, expr, _) in self.cpfs:
            subs[cpf], self.key, error, self.model_params = expr(
                subs, self.model_params, self.key)
            self.handle_error_code(error, f'CPF <{cpf}>')            
                
        # sample reward
        reward = self.sample_reward()
        
        # update state
        self.state = {}
        for (state, next_state) in rddl.next_state.items():

            # set state = state' for the next epoch
            subs[state] = subs[next_state]

            # convert object integer to string representation
            state_values = subs[state]
            if self.objects_as_strings:
                ptype = rddl.variable_ranges[state]
                if ptype not in RDDLValueInitializer.NUMPY_TYPES:
                    state_values = rddl.index_to_object_string_array(ptype, state_values)

            # optional grounding of state dictionary
            if keep_tensors:
                self.state[state] = state_values
            else:
                self.state.update(rddl.ground_var_with_values(state, state_values))
        
        # update observation
        if self._pomdp: 
            obs = {}
            for var in rddl.observ_fluents:

                # convert object integer to string representation
                obs_values = subs[var]
                if self.objects_as_strings:
                    ptype = rddl.variable_ranges[var]
                    if ptype not in RDDLValueInitializer.NUMPY_TYPES:
                        obs_values = rddl.index_to_object_string_array(ptype, obs_values)

                # optional grounding of observ-fluent dictionary    
                if keep_tensors:
                    obs[var] = obs_values
                else:
                    obs.update(rddl.ground_var_with_values(var, obs_values))
        else:
            obs = self.state
        
        done = self.check_terminal_states()        
        return obs, reward, done
        
