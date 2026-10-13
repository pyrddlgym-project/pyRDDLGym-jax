# ***********************************************************************
# JAXPLAN
#
# Author: Michael Gimelfarb
#
# ALL VERSIONS OF CONTROLLERS
# - offline controller is the straight-line planner
# - online controller is the replanning mode
#
# ***********************************************************************


import pickle
import time
from typing import Any, Dict, Optional, Union

import jax.random as random
import numpy as np
import termcolor

from pyRDDLGym.core.policy import BaseAgent

from pyRDDLGym_jax.core.planner import JaxPlanner

Pytree = Any


def _split_policy_actions(rddl, policy_fls):
    actions, fls = {}, {}
    for (var, value) in policy_fls.items():
        if rddl.variable_types.get(var, '') == 'action-fluent':
            actions[var] = value
        elif rddl.policy is not None and var in rddl.policy.variable_types:
            fls[var] = value
    return actions, fls


def _init_fls_hist(planner):
    fls_hist = {}
    if planner.plan.history_dependent:
        for name in planner.test_compiled.fls_hist_keys:
            values = np.asarray(planner.test_compiled.init_values[name])
            fls_hist[name] = np.zeros(
                (planner.rddl.horizon,) + np.shape(values), dtype=values.dtype)
    return fls_hist


def _update_fls_hist(fls_hist, step, states, actions):
    for (name, value) in fls_hist.items():
        if name in states:
            source = states[name]
        elif name in actions:
            source = actions[name]
        else:            
            continue
        if step >= np.shape(value)[0]:
            value[:-1] = value[1:]
        index = min(step, np.shape(value)[0] - 1)
        value[index] = np.reshape(source, np.shape(value[index]))


class JaxOfflineController(BaseAgent):
    '''A container class for a Jax policy trained offline.'''
    
    use_tensor_obs = True
    
    def __init__(self, planner: JaxPlanner,
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
                contents = pickle.load(file)
                params = contents['params']
                self.eval_hyperparams = contents['hyperparams']
                self.hyperparams_given = True

        # train the policy once before starting to step() through the environment
        # and then execute this policy in open-loop fashion
        self.step = 0
        self.callback = None
        self.fls_hist = _init_fls_hist(self.planner)
        if not self.train_on_reset and not self.params_given:
            callback = self.planner.optimize(key=self.key, **self.train_kwargs)
            self.callback = callback
            params = callback['best_params'] 
            if not self.hyperparams_given:
                self.eval_hyperparams = callback['policy_hyperparams']

            # save the policy
            if save_path is not None:
                with open(save_path, 'wb') as file:
                    contents = {'params': params, 'hyperparams': self.eval_hyperparams}
                    pickle.dump(contents, file)

        self.params = params  
        
    def sample_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        _update_fls_hist(self.fls_hist, self.step, state, {})
        state = {**state, **self.policy_state}
        self.key, subkey = random.split(self.key)
        policy_fls = self.planner.get_action(
            subkey, self.params, self.step, state, self.eval_hyperparams, self.fls_hist, 
            copy_state=False
        )
        actions, self.policy_state = _split_policy_actions(self.planner.rddl, policy_fls)
        _update_fls_hist(self.fls_hist, self.step, {}, actions)
        self.step += 1
        return actions
        
    def reset(self) -> None:
        self.step = 0
        self.fls_hist = _init_fls_hist(self.planner)
        self.policy_state = {
            var: value
            for (var, value) in self.planner.test_compiled.init_policy_values.items()
            if var not in self.planner.test_compiled.rddl.policy.param_fluents
        }

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
    
    def __init__(self, planner: JaxPlanner,
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
        _update_fls_hist(self.fls_hist, self.step, state, {})
        state = {**state, **self.policy_state}

        # we train the policy from the current state every time we step()
        planner = self.planner
        callback = planner.optimize(
            key=self.key, guess=self.guess, policy_hyperparams=self.hyperparams, 
            subs=state, **self.train_kwargs
        )
        
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
                key=self.key, guess=self.guess, policy_hyperparams=self.hyperparams, 
                subs=state, **self.train_kwargs
            )   
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
        # TODO: allow history carryover between real-time actions
        self.key, subkey = random.split(self.key)
        policy_fls = planner.get_action(
            subkey, params, 0, state, self.eval_hyperparams, self.fls_hist, 
            copy_state=False
        )
        actions, self.policy_state = _split_policy_actions(self.planner.rddl, policy_fls)
        _update_fls_hist(self.fls_hist, self.step, {}, actions)

        # apply warm start for the next epoch
        self.step += 1
        if self.warm_start:
            self.guess = planner.plan.guess_next_epoch(params)
            self.hyperparams = callback['policy_hyperparams']
        return actions
        
    def reset(self) -> None:
        self.step = 0
        self.guess = None
        self.hyperparams = None
        self.callback = None
        self.fls_hist = _init_fls_hist(self.planner)
        self.policy_state = {
            var: value
            for (var, value) in self.planner.test_compiled.init_policy_values.items()
            if var not in self.planner.test_compiled.rddl.policy.param_fluents
        }
        