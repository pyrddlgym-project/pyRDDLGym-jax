import csv
import datetime
from multiprocessing import get_context
import os
import time
from typing import Any, Callable, Dict, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

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

Kwargs = Dict[str, Any]


class Hyperparameter:
    '''A generic hyper-parameter of the planner that can be tuned.'''
    
    def __init__(self, name: str, lower_bound: float, upper_bound: float, 
                 search_to_config_map: Callable) -> None:
        self.name = name
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.search_to_config_map = search_to_config_map
        
    def __str__(self) -> str:
        return (f'{self.search_to_config_map.__name__} '
                f': [{self.lower_bound}, {self.upper_bound}] -> {self.name}')

    
COLUMNS = ['pid', 'worker', 'iteration', 'target', 'best_target', 'acq_params']


class JaxParameterTuning:
    '''A general-purpose class for tuning a Jax planner.'''
    
    def __init__(self, env: RDDLEnv,
                 config_template: str,
                 hyperparams_dict: Dict[str, Hyperparameter],
                 online: bool,
                 eval_trials: int=5,
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
        :param hyperparams_dict: dictionary mapping regex match in the config
        file template to a hyper-parameter object for the tuner to optimize
        :param online: whether the planner is optimized online or offline
        :param timeout_tuning: the maximum amount of time to spend tuning 
        hyperparameters in general (in seconds)
        :param eval_trials: how many trials to perform independent training
        in order to estimate the return for each set of hyper-parameters
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
        hyperparams_dict = {tag: hyper_param 
                            for (tag, hyper_param) in hyperparams_dict.items()
                            if tag in config_template}
        self.hyperparams_dict = hyperparams_dict
        self.online = online
        self.eval_trials = eval_trials
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
            gp_params = {'n_restarts_optimizer': 10}
        self.gp_params = gp_params
        if acquisition is None:
            num_samples = self.gp_iters * self.num_workers
            acquisition = JaxParameterTuning.annealing_acquisition(num_samples)
        self.acquisition = acquisition
    
    def summarize_hyperparameters(self) -> None:
        hyper_params_table = []
        for (tag, param) in self.hyperparams_dict.items():
            hyper_params_table.append(f'        {tag}: {str(param)}')
        hyper_params_table = '\n'.join(hyper_params_table)
        print(f'hyperparameter optimizer parameters:\n'
              f'    tuned_hyper_parameters    =\n{hyper_params_table}\n'
              f'    initialization_args       ={self.gp_init_kwargs}\n'
              f'    additional_args           ={self.gp_params}\n'
              f'    tuning_iterations         ={self.gp_iters}\n'
              f'    tuning_timeout            ={self.timeout_tuning}\n'
              f'    tuning_batch_size         ={self.num_workers}\n'
              f'    mp_pool_context_type      ={self.pool_context}\n'
              f'    mp_pool_poll_frequency    ={self.poll_frequency}\n'
              f'meta-objective parameters:\n'
              f'    planning_trials_per_iter  ={self.eval_trials}\n'
              f'    acquisition_fn            ={self.acquisition}')
        
    @staticmethod
    def annealing_acquisition(n_samples, n_delay_samples=0, kappa1=10.0, kappa2=1.0):
        acq_fn = UpperConfidenceBound(
            kappa=kappa1,
            exploration_decay=(kappa2 / kappa1) ** (1.0 / (n_samples - n_delay_samples)),
            exploration_decay_delay=n_delay_samples
        )
        return acq_fn
    
    @staticmethod
    def search_to_config_params(hyper_params: Dict[str, Hyperparameter], 
                                params: Dict[str, Any]) -> Dict[str, Any]:
        config_params = {
            tag: param.search_to_config_map(params[tag])
            for (tag, param) in hyper_params.items()
        }
        return config_params
    
    @staticmethod
    def config_from_template(config_template: str, 
                             config_params: Dict[str, Any]) -> str:
        config_string = config_template
        for (tag, param_value) in config_params.items():
            config_string = config_string.replace(tag, str(param_value))
        return config_string
    
    @property
    def best_config(self) -> str:
        return self.config_from_template(self.config_template, self.best_params)
        
    @staticmethod
    def objective_function(params, key, index, kwargs):
        '''A pickleable objective function to evaluate a single hyper-parameter 
        configuration.'''
        
        hyperparams_dict = kwargs['hyperparams_dict']
        config_template = kwargs['config_template']
        online = kwargs['online']
        domain = kwargs['domain']
        instance = kwargs['instance']
        num_trials = kwargs['eval_trials']
        verbose = kwargs['verbose']
        
        # config string substitution and parsing
        config_params = JaxParameterTuning.search_to_config_params(hyperparams_dict, params)
        if verbose:
            print(f'[{index}] key={key[0]}, hyper_params={config_params}', flush=True)
        config_string = JaxParameterTuning.config_from_template(config_template, config_params)
        planner_args, _, train_args = load_config_from_string(config_string)
    
        # initialize env for evaluation (need fresh copy to avoid concurrency)
        env = RDDLEnv(domain, instance, vectorized=True, enforce_action_constraints=False)
    
        # initialize planning algorithm
        planner = JaxBackpropPlanner(rddl=env.model, **planner_args)
        klass = JaxOnlineController if online else JaxOfflineController
        policy = klass(
            planner=planner, key=key,
            print_summary=False, print_progress=False, tqdm_position=index,
            **train_args
        )
        
        # perform training
        average_reward = 0.0
        for trial in range(num_trials):
            key, subkey = jax.random.split(key)
            total_reward = policy.evaluate(env, seed=np.array(subkey)[0])['mean']
            if verbose:
                print(f'    [{index}] trial {trial + 1} key={subkey[0]}, '
                      f'reward={total_reward}', flush=True)
            average_reward += total_reward / num_trials    
        if verbose:
            print(f'[{index}] average reward={average_reward}', flush=True)        
        
        pid = os.getpid()
        return params, average_reward, index, pid
    
    def tune(self, key: int,
             filename: str,
             save_plot: bool=False) -> Dict[str, Any]:
        '''Tunes the hyper-parameters for Jax planner, returns the best found.'''
        
        self.summarize_hyperparameters()
        
        # clear and prepare output file
        filename = self._filename(filename, 'csv')
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(COLUMNS + list(self.hyperparams_dict.keys()))
        
        # objective function auxiliary data
        obj_kwargs = {
            'hyperparams_dict': self.hyperparams_dict,
            'config_template': self.config_template,
            'online': self.online,
            'domain': self.env.domain_text,
            'instance': self.env.instance_text,
            'eval_trials': self.eval_trials,
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
        
        # start multiprocess evaluation
        worker_ids = list(range(num_workers))
        best_params, best_target = None, -np.inf
        key = jax.random.PRNGKey(key)
        start_time = time.time()
        
        for it in range(self.gp_iters): 
            
            # check if there is enough time left for another iteration
            elapsed = time.time() - start_time
            if elapsed >= self.timeout_tuning:
                print(f'global time limit reached at iteration {it}, aborting')
                break
            
            # continue with next iteration
            print('\n' + '*' * 25 + 
                  f'\n[{datetime.timedelta(seconds=elapsed)}] ' + 
                  f'starting iteration {it + 1}' + 
                  '\n' + '*' * 25)
            key, *subkeys = jax.random.split(key, num=num_workers + 1)
            rows = [None] * num_workers
            
            # create worker pool: note each iteration must wait for all workers
            # to finish before moving to the next
            with get_context(self.pool_context).Pool(processes=num_workers) as pool:
                
                # assign jobs to worker pool
                results = [
                    pool.apply_async(JaxParameterTuning.objective_function,
                                     obj_args + (obj_kwargs,))
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
                                       best_target, old_acq_params] \
                                        + list(config_params.values())
                        
            # write results of all processes in current iteration to file
            with open(filename, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(rows)
        
        # print summary of results
        elapsed = time.time() - start_time
        print(f'summary of hyper-parameter optimization:\n'
              f'    time_elapsed         ={datetime.timedelta(seconds=elapsed)}\n'
              f'    iterations           ={it + 1}\n'
              f'    best_hyper_parameters={best_params}\n'
              f'    best_meta_objective  ={best_target}\n')        
        if save_plot:
            self._save_plot(filename)
        
        self.best_params = best_params
        self.optimizer = optimizer
        return best_params

    def _filename(self, name, ext):
        domain_name = ''.join(c for c in self.env.model.domain_name 
                              if c.isalnum() or c == '_')
        instance_name = ''.join(c for c in self.env.model.instance_name 
                                if c.isalnum() or c == '_')
        filename = f'{name}_{domain_name}_{instance_name}.{ext}'
        return filename
    
    def _save_plot(self, filename):
        try:
            import matplotlib.pyplot as plt
            from sklearn.manifold import MDS
        except Exception as e:
            raise_warning(f'failed to import packages matplotlib or sklearn, '
                          f'aborting plot of search space\n{e}', 'red')
        else:
            with open(filename, 'r') as file:
                data_iter = csv.reader(file, delimiter=',')
                data = [row for row in data_iter]
            data = np.asarray(data, dtype=object)
            hparam = data[1:, len(COLUMNS):].astype(np.float64)
            target = data[1:, 3].astype(np.float64)
            target = (target - np.min(target)) / (np.max(target) - np.min(target))
            embedding = MDS(n_components=2, normalized_stress='auto')
            hparam_low = embedding.fit_transform(hparam)
            sc = plt.scatter(hparam_low[:, 0], hparam_low[:, 1], c=target, s=5,
                             cmap='seismic', edgecolor='gray', linewidth=0)
            ax = plt.gca()
            for i in range(len(target)):
                ax.annotate(str(i), (hparam_low[i, 0], hparam_low[i, 1]), fontsize=3)         
            plt.colorbar(sc)
            plt.savefig(self._filename('gp_points', 'pdf'))
            plt.clf()
            plt.close()
