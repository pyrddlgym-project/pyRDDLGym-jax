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


import csv
import datetime
import threading
import multiprocessing
import os
import termcolor
import time
import traceback
from typing import Any, Callable, Dict, Iterable, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from bayes_opt import BayesianOptimization
from bayes_opt.acquisition import AcquisitionFunction, UpperConfidenceBound
import jax
import numpy as np

from pyRDDLGym.core.debug.exception import raise_warning
from pyRDDLGym.core.env import RDDLEnv

from pyRDDLGym_jax.core.planner import (
    JaxBackpropPlanner,
    JaxOfflineController,
    JaxOnlineController,
    load_config_from_string
)

# try to load the dash board
try:
    from pyRDDLGym_jax.core.visualization import JaxPlannerDashboard
except Exception:
    raise_warning('Failed to load the dashboard visualization tool: '
                  'please make sure you have installed the required packages.', 'red')
    traceback.print_exc()
    JaxPlannerDashboard = None


class Hyperparameter:
    '''A generic hyper-parameter of the planner that can be tuned.'''
    
    def __init__(self, tag: str, lower_bound: float, upper_bound: float,
                 search_to_config_map: Callable) -> None:
        self.tag = tag
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.search_to_config_map = search_to_config_map
        
    def __str__(self) -> str:
        return (f'{self.search_to_config_map.__name__} '
                f': [{self.lower_bound}, {self.upper_bound}] -> {self.tag}')


Kwargs = Dict[str, Any]
ParameterValues = Dict[str, Any]
Hyperparameters = Iterable[Hyperparameter]
    
COLUMNS = ['pid', 'worker', 'iteration', 'target', 'best_target', 'acq_params']


class JaxParameterTuning:
    '''A general-purpose class for tuning a Jax planner.'''
    
    def __init__(self, env: RDDLEnv,
                 config_template: str,
                 hyperparams: Hyperparameters,
                 online: bool,
                 eval_trials: int=5,
                 rollouts_per_trial: int=1,
                 verbose: bool=True,
                 timeout_tuning: float=np.inf,
                 pool_context: str='spawn',
                 num_workers: int=1,
                 poll_frequency: float=0.2,
                 gp_iters: int=25,
                 acquisition: Optional[AcquisitionFunction]=None,
                 gp_init_kwargs: Optional[Kwargs]=None,
                 gp_params: Optional[Kwargs]=None) -> None:
        '''Creates a new instance for tuning hyper-parameters for Jax planners
        on the given RDDL domain and instance.
        
        :param env: the RDDLEnv describing the MDP to optimize
        :param config_template: base configuration file content to tune: regex
        matches are specified directly in the config and map to keys in the 
        hyperparams_dict field
        :param hyperparams: list of hyper-parameters to regex replace in the
        config template during tuning
        :param online: whether the planner is optimized online or offline
        :param timeout_tuning: the maximum amount of time to spend tuning 
        hyperparameters in general (in seconds)
        :param eval_trials: how many trials to perform independent training
        in order to estimate the return for each set of hyper-parameters
        :param rollouts_per_trial: how many rollouts to perform during evaluation
        at the end of each training trial (only applies when online=False)
        :param verbose: whether to print intermediate results of tuning
        :param pool_context: context for multiprocessing pool (default "spawn")
        :param num_workers: how many points to evaluate in parallel
        :param poll_frequency: how often (in seconds) to poll for completed
        jobs, necessary if num_workers > 1
        :param gp_iters: number of iterations of optimization
        :param acquisition: acquisition function for Bayesian optimizer
        :parm gp_init_kwargs: additional parameters to feed to Bayesian 
        during initialization  
        :param gp_params: additional parameters to feed to Bayesian optimizer 
        after initialization optimization
        '''        
        # objective parameters
        self.env = env
        self.config_template = config_template
        hyperparams_dict = {hyper_param.tag: hyper_param 
                            for hyper_param in hyperparams
                            if hyper_param.tag in config_template}
        self.hyperparams_dict = hyperparams_dict
        self.online = online
        self.eval_trials = eval_trials
        self.rollouts_per_trial = rollouts_per_trial
        self.verbose = verbose
        
        # Bayesian parameters
        self.timeout_tuning = timeout_tuning
        self.pool_context = pool_context
        self.num_workers = num_workers
        self.poll_frequency = poll_frequency
        self.gp_iters = gp_iters
        if gp_init_kwargs is None:
            gp_init_kwargs = {}
        self.gp_init_kwargs = gp_init_kwargs
        if gp_params is None:
            gp_params = {'n_restarts_optimizer': 25, 
                         'kernel': self.make_default_kernel()}
        self.gp_params = gp_params
        if acquisition is None:
            num_samples = self.gp_iters * self.num_workers
            acquisition = JaxParameterTuning.annealing_acquisition(num_samples)
        self.acquisition = acquisition
    
    @staticmethod
    def make_default_kernel():
        weight1 = ConstantKernel(1.0, (0.01, 100.0))
        weight2 = ConstantKernel(1.0, (0.01, 100.0))
        weight3 = ConstantKernel(1.0, (0.01, 100.0))
        kernel1 = Matern(length_scale=0.5, length_scale_bounds=(0.1, 0.5), nu=2.5)
        kernel2 = Matern(length_scale=1.0, length_scale_bounds=(0.5, 1.0), nu=2.5)
        kernel3 = Matern(length_scale=5.0, length_scale_bounds=(1.0, 5.0), nu=2.5)
        return weight1 * kernel1 + weight2 * kernel2 + weight3 * kernel3
        
    def summarize_hyperparameters(self) -> str:
        hyper_params_table = []
        for (_, param) in self.hyperparams_dict.items():
            hyper_params_table.append(f'        {str(param)}')
        hyper_params_table = '\n'.join(hyper_params_table)
        return (f'hyperparameter optimizer parameters:\n'
                f'    tuned_hyper_parameters    =\n{hyper_params_table}\n'
                f'    initialization_args       ={self.gp_init_kwargs}\n'
                f'    gp_params                 ={self.gp_params}\n'
                f'    tuning_iterations         ={self.gp_iters}\n'
                f'    tuning_timeout            ={self.timeout_tuning}\n'
                f'    tuning_batch_size         ={self.num_workers}\n'
                f'    mp_pool_context_type      ={self.pool_context}\n'
                f'    mp_pool_poll_frequency    ={self.poll_frequency}\n'
                f'meta-objective parameters:\n'
                f'    planning_trials_per_iter  ={self.eval_trials}\n'
                f'    rollouts_per_trial        ={self.rollouts_per_trial}\n'
                f'    acquisition_fn            ={self.acquisition}')
        
    @staticmethod
    def annealing_acquisition(n_samples: int, n_delay_samples: int=0,
                              kappa1: float=10.0, kappa2: float=1.0) -> UpperConfidenceBound:
        acq_fn = UpperConfidenceBound(
            kappa=kappa1,
            exploration_decay=(kappa2 / kappa1) ** (1.0 / (n_samples - n_delay_samples)),
            exploration_decay_delay=n_delay_samples
        )
        return acq_fn
    
    @staticmethod
    def search_to_config_params(hyper_params: Hyperparameters,
                                params: ParameterValues) -> ParameterValues:
        config_params = {
            tag: param.search_to_config_map(params[tag])
            for (tag, param) in hyper_params.items()
        }
        return config_params
    
    @staticmethod
    def config_from_template(config_template: str,
                             config_params: ParameterValues) -> str:
        config_string = config_template
        for (tag, param_value) in config_params.items():
            config_string = config_string.replace(tag, str(param_value))
        return config_string
    
    @property
    def best_config(self) -> str:
        return self.config_from_template(self.config_template, self.best_params)
    
    @staticmethod
    def queue_listener(queue, dashboard):
        while True:
            args = queue.get()
            if args is None:
                break
            elif len(args) == 2:
                dashboard.update_experiment(*args)
            else:
                dashboard.register_experiment(*args)
    
    @staticmethod
    def offline_trials(env, planner, train_args, key, iteration, index, num_trials, 
                       rollouts_per_trial, verbose, viz, queue):
        average_reward = 0.0
        for trial in range(num_trials):
            key, subkey = jax.random.split(key)
            
            # for the dashboard
            experiment_id = f'iter={iteration}, worker={index}, trial={trial}'
            if queue is not None and JaxPlannerDashboard is not None:
                queue.put((
                    experiment_id, 
                    JaxPlannerDashboard.get_planner_info(planner), 
                    subkey[0],
                    viz
                ))
            
            # train the policy
            callback = None
            for callback in planner.optimize_generator(key=subkey, **train_args):
                if queue is not None and queue.empty():
                    queue.put((experiment_id, callback))
            best_params = None if callback is None else callback['best_params']
            
            # evaluate the policy in the real environment
            policy = JaxOfflineController(
                planner=planner, key=subkey, tqdm_position=index, 
                params=best_params, train_on_reset=False)
            total_reward = policy.evaluate(env, episodes=rollouts_per_trial, 
                                           seed=np.array(subkey)[0])['mean']
            
            # update average reward
            if verbose:
                iters = None if callback is None else callback['iteration']
                print(f'    [{index}] trial {trial + 1}, key={subkey[0]}, '
                      f'reward={total_reward:.6f}, iters={iters}', flush=True)
            average_reward += total_reward / num_trials
            
        if verbose:
            print(f'[{index}] average reward={average_reward:.6f}', flush=True)
        return average_reward
    
    @staticmethod
    def online_trials(env, planner, train_args, key, iteration, index, num_trials, 
                      verbose, viz, queue):
        average_reward = 0.0
        for trial in range(num_trials):
            key, subkey = jax.random.split(key)
            
            # for the dashboard
            experiment_id = f'iter={iteration}, worker={index}, trial={trial}'
            if queue is not None and JaxPlannerDashboard is not None:
                queue.put((
                    experiment_id, 
                    JaxPlannerDashboard.get_planner_info(planner), 
                    subkey[0],
                    viz
                ))
            
            # initialize the online policy
            policy = JaxOnlineController(
                planner=planner, key=subkey, tqdm_position=index, **train_args)
            
            # evaluate the policy in the real environment
            total_reward = 0.0
            callback = None
            state, _ = env.reset(seed=np.array(subkey)[0])
            elapsed_time = 0.0
            for step in range(env.horizon):
                action = policy.sample_action(state)   
                next_state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                done = terminated or truncated
                state = next_state
                callback = policy.callback
                elapsed_time += callback['elapsed_time']
                callback['iteration'] = step
                callback['progress'] = int(100 * (step + 1.) / env.horizon)
                callback['elapsed_time'] = elapsed_time
                if queue is not None and queue.empty():
                    queue.put((experiment_id, callback))
                if done:
                    break
            
            # update average reward
            if verbose:
                iters = None if callback is None else callback['iteration']
                print(f'    [{index}] trial {trial + 1}, key={subkey[0]}, '
                      f'reward={total_reward:.6f}, iters={iters}', flush=True)
            average_reward += total_reward / num_trials
        
        if verbose:
            print(f'[{index}] average reward={average_reward:.6f}', flush=True)
        return average_reward            
        
    @staticmethod
    def objective_function(params: ParameterValues,
                           key: jax.random.PRNGKey,
                           index: int, 
                           iteration: int,
                           kwargs: Kwargs, 
                           queue: object) -> Tuple[ParameterValues, float, int, int]:
        '''A pickleable objective function to evaluate a single hyper-parameter 
        configuration.'''
        
        hyperparams_dict = kwargs['hyperparams_dict']
        config_template = kwargs['config_template']
        online = kwargs['online']
        domain = kwargs['domain']
        instance = kwargs['instance']
        num_trials = kwargs['eval_trials']
        rollouts_per_trial = kwargs['rollouts_per_trial']
        viz = kwargs['viz']
        verbose = kwargs['verbose']
        
        # config string substitution and parsing
        config_params = JaxParameterTuning.search_to_config_params(hyperparams_dict, params)
        if verbose:
            config_param_str = ', '.join(
                f'{k}={v}' for (k, v) in config_params.items())
            print(f'[{index}] key={key[0]}, {config_param_str}', flush=True)
        config_string = JaxParameterTuning.config_from_template(config_template, config_params)
        planner_args, _, train_args = load_config_from_string(config_string)
        
        # remove keywords that should not be in the tuner
        train_args.pop('dashboard', None)
        planner_args.pop('parallel_updates', None)
    
        # initialize env for evaluation (need fresh copy to avoid concurrency)
        env = RDDLEnv(domain, instance, vectorized=True, enforce_action_constraints=False)
    
        # run planning algorithm
        planner = JaxBackpropPlanner(rddl=env.model, **planner_args)
        if online:
            average_reward = JaxParameterTuning.online_trials(
                env, planner, train_args, key, iteration, index, num_trials, 
                verbose, viz, queue
            )
        else:
            average_reward = JaxParameterTuning.offline_trials(
                env, planner, train_args, key, iteration, index, 
                num_trials, rollouts_per_trial, verbose, viz, queue
            )
        
        pid = os.getpid()
        return params, average_reward, index, pid
    
    def tune_optimizer(self, optimizer: BayesianOptimization) -> None:
        '''Tunes the Bayesian optimization algorithm hyper-parameters.'''
        print(f'Kernel: {repr(optimizer._gp.kernel_)}.')
        
    def tune(self, key: int, 
             log_file: Optional[str]=None, 
             show_dashboard: bool=False, 
             print_hyperparams: bool=False) -> ParameterValues:
        '''Tunes the hyper-parameters for Jax planner, returns the best found.
        
        :param key: RNG key to seed the hyper-parameter optimizer
        :param log_file: optional path to file where tuning progress will be saved
        :param show_dashboard: whether to display tuning results in a dashboard
        :param print_hyperparams: whether to print a hyper-parameter summary of the 
        optimizer
        '''
        
        if self.verbose:
            print(JaxBackpropPlanner.summarize_system())
        if print_hyperparams:
            print(self.summarize_hyperparameters())
        
        # clear and prepare output file
        if log_file is not None:
            with open(log_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(COLUMNS + list(self.hyperparams_dict.keys()))
            
        # create a dash-board for visualizing experiment runs
        if show_dashboard and JaxPlannerDashboard is not None:
            dashboard = JaxPlannerDashboard()
            dashboard.launch()
            
        # objective function auxiliary data
        obj_kwargs = {
            'hyperparams_dict': self.hyperparams_dict,
            'config_template': self.config_template,
            'online': self.online,
            'domain': self.env.domain_text,
            'instance': self.env.instance_text,
            'eval_trials': self.eval_trials,
            'rollouts_per_trial': self.rollouts_per_trial,
            'viz': self.env._visualizer,
            'verbose': self.verbose
        }
        
        # create optimizer
        hyperparams_bounds = {
            tag: (param.lower_bound, param.upper_bound) 
            for (tag, param) in self.hyperparams_dict.items()
        }
        optimizer = BayesianOptimization(
            f=None,
            acquisition_function=self.acquisition,
            pbounds=hyperparams_bounds,
            allow_duplicate_points=True,  # to avoid crash
            random_state=np.random.RandomState(key),
            **self.gp_init_kwargs
        )
        optimizer.set_gp_params(**self.gp_params)
        
        # suggest initial parameters to evaluate
        num_workers = self.num_workers
        suggested_params, acq_params = [], []
        for _ in range(num_workers):
            probe = optimizer.suggest()
            suggested_params.append(probe) 
            acq_params.append(vars(optimizer.acquisition_function))
        
        with multiprocessing.Manager() as manager:
            
            # queue and parallel thread for handing render events
            if show_dashboard:
                queue = manager.Queue()
                dashboard_thread = threading.Thread(
                    target=JaxParameterTuning.queue_listener,
                    args=(queue, dashboard)
                )
                dashboard_thread.start()
            else:
                queue = None
            
            # start multiprocess evaluation
            worker_ids = list(range(num_workers))
            best_params, best_target = None, -np.inf
            key = jax.random.PRNGKey(key)
            start_time = time.time()
            
            for it in range(self.gp_iters): 
                
                # check if there is enough time left for another iteration
                elapsed = time.time() - start_time
                if elapsed >= self.timeout_tuning:
                    message = termcolor.colored(
                        f'[INFO] Global time limit reached at iteration {it}.', 'green')
                    print(message)
                    break
                
                # continue with next iteration
                print('\n' + '*' * 80 + 
                      f'\n[{datetime.timedelta(seconds=elapsed)}] ' + 
                      f'Starting iteration {it + 1}' + 
                      '\n' + '*' * 80)
                key, *subkeys = jax.random.split(key, num=num_workers + 1)
                rows = [None] * num_workers
                old_best_target = best_target
                
                # create worker pool: note each iteration must wait for all workers
                # to finish before moving to the next
                with multiprocessing.get_context(
                    self.pool_context).Pool(processes=num_workers) as pool:
                    
                    # assign jobs to worker pool
                    results = [
                        pool.apply_async(JaxParameterTuning.objective_function,
                                         obj_args + (it, obj_kwargs, queue))
                        for obj_args in zip(suggested_params, subkeys, worker_ids)
                    ]
                
                    # wait for all workers to complete
                    while results:
                        time.sleep(self.poll_frequency)
                        
                        # determine which jobs have completed
                        jobs_done = []
                        for (i, candidate) in enumerate(results):
                            if candidate.ready():
                                jobs_done.append(i)
                        
                        # get result from completed jobs
                        for i in jobs_done[::-1]:
                            
                            # extract and register the new evaluation
                            params, target, index, pid = results.pop(i).get()
                            optimizer.register(params, target)
                            optimizer._gp.fit(
                                optimizer.space.params, optimizer.space.target)
                            
                            # update acquisition function and suggest a new point
                            suggested_params[index] = optimizer.suggest()
                            old_acq_params = acq_params[index]
                            acq_params[index] = vars(optimizer.acquisition_function)
                            
                            # transform suggestion back to natural space
                            config_params = JaxParameterTuning.search_to_config_params(
                                self.hyperparams_dict, params)
                            
                            # update the best suggestion so far
                            if target > best_target:
                                best_params, best_target = config_params, target
                            
                            rows[index] = [pid, index, it, target,
                                           best_target, old_acq_params] + \
                                           list(config_params.values())
                
                # print best parameter if found
                if best_target > old_best_target:
                    message = termcolor.colored(
                        f'[INFO] Found new best average reward {best_target:.6f}.', 
                        'green')
                    print(message)
                
                # tune the optimizer here
                self.tune_optimizer(optimizer)
                
                # write results of all processes in current iteration to file
                if log_file is not None:
                    with open(log_file, 'a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerows(rows)
                    
                # update the dashboard tuning
                if show_dashboard:
                    dashboard.update_tuning(optimizer, hyperparams_bounds)
            
            # stop the queue listener thread
            if show_dashboard:
                queue.put(None) 
                dashboard_thread.join()
        
        # print summary of results
        elapsed = time.time() - start_time
        print(f'Summary of hyper-parameter optimization:\n'
              f'    time_elapsed         ={datetime.timedelta(seconds=elapsed)}\n'
              f'    iterations           ={it + 1}\n'
              f'    best_hyper_parameters={best_params}\n'
              f'    best_meta_objective  ={best_target}\n')        
        
        self.best_params = best_params
        self.optimizer = optimizer
        self.log_file = log_file
        return best_params
