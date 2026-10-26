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
# [2] Akiba, Takuya, et al. "Optuna: A next-generation hyperparameter optimization
# framework." Proceedings of the 25th ACM SIGKDD International Conference on Knowledge
# Discovery & Data Mining. 2019.
#
# ***********************************************************************


import atexit
import csv
import datetime
import importlib.util
import multiprocessing
import os
from queue import Queue
import re
import shutil
import tempfile
import threading
import time
import traceback
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Union
import webbrowser
from wsgiref.simple_server import make_server, WSGIRequestHandler

import jax
import numpy as np
import optuna
from optuna.distributions import (
    BaseDistribution, CategoricalDistribution, FloatDistribution, IntDistribution
)
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from optuna.trial import TrialState

from pyRDDLGym.core.debug.exception import raise_warning
from pyRDDLGym.core.env import RDDLEnv

from pyRDDLGym_jax.core.config import load_config_from_string
from pyRDDLGym_jax.core.controller import JaxOfflineController, JaxOnlineController
from pyRDDLGym_jax.core.planner import JaxPlanner, create_planner

# try to load the dash board
try:
    from pyRDDLGym_jax.core.visualization import JaxPlannerDashboard
except Exception:
    raise_warning('Failed to load the dashboard visualization tool: '
                  'please make sure you have installed the required packages.', 'red')
    traceback.print_exc()
    JaxPlannerDashboard = None


class Hyperparameter:
    '''A hyper-parameter of the planner that can be tuned: its tag in the config
    template is replaced by the (transformed) value suggested by the optimizer.'''

    def __init__(self, tag: str, distribution: BaseDistribution,
                 transform: Optional[Callable]=None) -> None:
        '''Creates a new hyper-parameter.

        :param tag: the pattern in the config template to replace
        :param distribution: the Optuna search space of the hyper-parameter
        :param transform: optional map from the searched value to the config value
        '''
        self.tag = tag
        self.distribution = distribution
        self.transform = transform

    @staticmethod
    def real(tag: str, low: float, high: float, log: bool=False,
             step: Optional[float]=None, transform: Optional[Callable]=None) -> 'Hyperparameter':
        '''A real-valued hyper-parameter in [low, high], searched in log-space if log.'''
        return Hyperparameter(tag, FloatDistribution(low, high, log=log, step=step), transform)

    @staticmethod
    def integer(tag: str, low: int, high: int, log: bool=False, step: int=1,
                transform: Optional[Callable]=None) -> 'Hyperparameter':
        '''An integer hyper-parameter in [low, high], searched in log-space if log.'''
        return Hyperparameter(tag, IntDistribution(low, high, log=log, step=step), transform)

    @staticmethod
    def categorical(tag: str, choices: Sequence[Any],
                    transform: Optional[Callable]=None) -> 'Hyperparameter':
        '''A hyper-parameter taking one of the given (unordered) choices.'''
        return Hyperparameter(tag, CategoricalDistribution(choices), transform)

    def to_config(self, value: Any) -> Any:
        return value if self.transform is None else self.transform(value)

    def __str__(self) -> str:
        transform = '' if self.transform is None else f' -> {self.transform.__name__}'
        return f'{self.tag}: {self.distribution}{transform}'


Kwargs = Dict[str, Any]
ParameterValues = Dict[str, Any]
Hyperparameters = Iterable[Hyperparameter]

COLUMNS = ['number', 'state', 'target', 'best_target', 'seconds']


def _tag_pattern(tag: str) -> re.Pattern:
    return re.compile(r'\b' + re.escape(tag) + r'\b')


class JaxParameterTuning:
    '''A general-purpose class for tuning a Jax planner with Optuna. Each configuration
    is evaluated in a fresh process by averaging the return of several independent
    training runs; the training seeds are shared across configurations (common random
    numbers), which makes the comparison between configurations paired.'''

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
                 num_configs: int=50,
                 sampler: Optional[optuna.samplers.BaseSampler]=None,
                 pruner: Optional[optuna.pruners.BasePruner]=None,
                 storage: Optional[str]=None,
                 study_name: str='jaxplan') -> None:
        '''Creates a new instance for tuning hyper-parameters for Jax planners
        on the given RDDL domain and instance.

        :param env: the RDDLEnv describing the MDP to optimize
        :param config_template: base configuration file content to tune, in which the
        tags of the hyper-parameters are replaced by their values
        :param hyperparams: list of hyper-parameters to replace in the config template
        :param online: whether the planner is optimized online or offline
        :param eval_trials: how many independent training runs to average to estimate
        the return of each configuration
        :param rollouts_per_trial: how many rollouts to perform during evaluation
        at the end of each training run (only applies when online=False)
        :param verbose: whether to print intermediate results of tuning
        :param timeout_tuning: no new configurations are started after this many seconds
        :param pool_context: context for multiprocessing pool (default "spawn")
        :param num_workers: how many configurations to evaluate in parallel
        :param num_configs: how many configurations to evaluate in total
        :param sampler: Optuna sampler suggesting configurations (default GPSampler if
        torch is installed, otherwise multivariate TPESampler)
        :param pruner: optional Optuna pruner to stop a configuration after any of its
        training runs, e.g. optuna.pruners.WilcoxonPruner (default no pruning)
        :param storage: optional journal file to save the study, and resume it if exists
        :param study_name: name of the study in the storage
        '''
        self.env = env
        self.config_template = config_template
        self.hyperparams_dict = {param.tag: param for param in hyperparams
                                 if _tag_pattern(param.tag).search(config_template)}
        self.online = online
        self.eval_trials = eval_trials
        self.rollouts_per_trial = rollouts_per_trial
        self.verbose = verbose
        self.timeout_tuning = timeout_tuning
        self.pool_context = pool_context
        self.num_workers = num_workers
        self.num_configs = num_configs
        self.sampler = sampler
        self.pruner = pruner
        self.storage = storage
        self.study_name = study_name

    def summarize_hyperparameters(self) -> str:
        '''Returns a summary string of the hyper-parameters being tuned and their bounds,
        as well as the tuning parameters for the optimizer and meta-objective.'''
        hyperparams_table = '\n'.join(f'        {param}'
                                      for param in self.hyperparams_dict.values())
        if self.sampler is not None:
            sampler_name = type(self.sampler).__name__
        elif importlib.util.find_spec('torch') is not None:
            sampler_name = 'GPSampler'
        else:
            sampler_name = 'TPESampler'
        return (f'hyperparameter optimizer parameters:\n'
                f'    tuned_hyper_parameters    =\n{hyperparams_table}\n'
                f'    sampler                   ={sampler_name}\n'
                f'    pruner                    ={type(self.pruner).__name__}\n'
                f'    num_configs               ={self.num_configs}\n'
                f'    tuning_timeout            ={self.timeout_tuning}\n'
                f'    num_workers               ={self.num_workers}\n'
                f'    mp_pool_context_type      ={self.pool_context}\n'
                f'    storage                   ={self.storage}\n'
                f'meta-objective parameters:\n'
                f'    planning_trials_per_iter  ={self.eval_trials}\n'
                f'    rollouts_per_trial        ={self.rollouts_per_trial}')

    @staticmethod
    def default_sampler(seed: int) -> optuna.samplers.BaseSampler:
        '''Returns the GP sampler, which requires torch, or else the TPE sampler: both
        account for configurations still running when suggesting new ones.'''
        if importlib.util.find_spec('torch') is not None:
            return optuna.samplers.GPSampler(seed=seed)
        raise_warning('[WARN] torch is not installed, so tuning uses TPESampler instead '
                      'of GPSampler: install torch to use the Gaussian process sampler.',
                      'yellow')
        return optuna.samplers.TPESampler(multivariate=True, constant_liar=True, seed=seed)

    def search_to_config_params(self, params: ParameterValues) -> ParameterValues:
        '''Transforms the hyper-parameter values suggested by the optimizer to the
        corresponding config parameter values.'''
        return {tag: param.to_config(params[tag])
                for (tag, param) in self.hyperparams_dict.items()}

    def config_from_template(self, config_params: ParameterValues) -> str:
        '''Replaces all parameter tags in the config template with the corresponding values.'''
        config_string = self.config_template
        for (tag, value) in config_params.items():
            config_string = _tag_pattern(tag).sub(lambda _: str(value), config_string)
        return config_string

    @property
    def best_config(self) -> str:
        '''Returns the concrete config by replacing all abstract parameters with
        the best values found during tuning.'''
        return self.config_from_template(self.best_params)

    def build_best_controller(self) -> Union[JaxOfflineController, JaxOnlineController]:
        '''Creates the concrete controller instance with the best hyper-parameters.'''
        planner_args, _, train_args = load_config_from_string(self.best_config)
        planner = create_planner(rddl=self.env.model, **planner_args)
        class_ = JaxOnlineController if self.online else JaxOfflineController
        controller = class_(planner, **train_args)
        return controller

    @staticmethod
    def launch_study_dashboard(storage: optuna.storages.BaseStorage, port: int=8080) -> None:
        '''Launches optuna-dashboard for the study in a browser window, if installed.'''
        try:
            import optuna_dashboard
        except ImportError:
            raise_warning('[WARN] optuna-dashboard is not installed, so the study is not '
                          'shown: install it with pip install optuna-dashboard.', 'yellow')
            return

        # serve without logging every request of the dashboard to the console
        class QuietHandler(WSGIRequestHandler):
            def log_message(self, *args):
                pass

        server = make_server('localhost', port, optuna_dashboard.wsgi(storage),
                             handler_class=QuietHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        webbrowser.open_new(f'http://localhost:{port}/')

    @staticmethod
    def queue_listener(queue, dashboard):
        '''Listens to the given queue and updates the dashboard with new experiment results.'''
        while True:
            args = queue.get()
            if args is None:
                break
            elif len(args) == 2:
                dashboard.update_experiment(*args)
            else:
                dashboard.register_experiment(*args)

    @staticmethod
    def offline_trial(env, planner, train_args, seed, number, rollouts_per_trial,
                      experiment_id, viz, queue):
        '''Trains the planner offline, then returns the average return of its policy.'''
        key = jax.random.PRNGKey(seed)

        # for the dashboard
        if queue is not None and JaxPlannerDashboard is not None:
            queue.put((experiment_id, JaxPlannerDashboard.get_planner_info(planner),
                       seed, viz))

        # train the policy
        callback = None
        for callback in planner.optimize_generator(key=key, **train_args):
            if queue is not None and queue.empty():
                queue.put((experiment_id, callback))
        best_params = None if callback is None else callback['best_params']
        hyperparams = None if callback is None else callback['policy_hyperparams']

        # evaluate the policy in the real environment with its training hyper-parameters
        policy = JaxOfflineController(
            planner=planner, key=key, tqdm_position=number, params=best_params,
            eval_hyperparams=hyperparams, train_on_reset=False)
        total_reward = policy.evaluate(env, episodes=rollouts_per_trial, seed=seed)['mean']
        iters = None if callback is None else callback['iteration']
        return total_reward, iters

    @staticmethod
    def online_trial(env, planner, train_args, seed, number, experiment_id, viz, queue):
        '''Runs the planner online for one episode, then returns its return.'''
        key = jax.random.PRNGKey(seed)

        # for the dashboard
        if queue is not None and JaxPlannerDashboard is not None:
            queue.put((experiment_id, JaxPlannerDashboard.get_planner_info(planner),
                       seed, viz))

        # initialize the online policy
        policy = JaxOnlineController(
            planner=planner, key=key, tqdm_position=number, **train_args)

        # evaluate the policy in the real environment
        total_reward = 0.0
        callback = None
        state, _ = env.reset(seed=seed)
        elapsed_time = 0.0
        for step in range(env.horizon):
            action = policy.sample_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            callback = policy.callback
            elapsed_time += callback['elapsed_time']
            callback['iteration'] = step
            callback['progress'] = int(100 * (step + 1.) / env.horizon)
            callback['elapsed_time'] = elapsed_time
            if queue is not None and queue.empty():
                queue.put((experiment_id, callback))
            if terminated or truncated:
                break
        iters = None if callback is None else callback['iteration']
        return total_reward, iters

    @staticmethod
    def objective_function(config_string: str, number: int, trial_id: Optional[int],
                           seeds: List[int], kwargs: Kwargs, queue: object) -> List[float]:
        '''A pickleable objective function that evaluates a single configuration in a
        worker process: returns the return of each training run, which are reported to
        the pruner if any (the list is cut short if the configuration is pruned).'''
        online = kwargs['online']
        verbose = kwargs['verbose']
        planner_args, _, train_args = load_config_from_string(config_string)

        # the dashboard requires the planner to cache test rollouts, but is updated
        # only through the queue
        planner_args['dashboard'] = True if queue is not None else None
        planner_args.pop('parallel_updates', None)

        # initialize env for evaluation (need fresh copy to avoid concurrency)
        env = RDDLEnv(kwargs['domain'], kwargs['instance'],
                      vectorized=True, enforce_action_constraints=False)
        planner = create_planner(rddl=env.model, **planner_args)
        planner.dashboard = None

        # the trial is accessed through the storage only for reporting to the pruner
        if trial_id is None:
            trial = None
        else:
            study = optuna.load_study(
                study_name=kwargs['study_name'], storage=kwargs['storage'],
                sampler=optuna.samplers.RandomSampler(), pruner=kwargs['pruner'])
            trial = optuna.trial.Trial(study, trial_id)

        rewards = []
        for (i, seed) in enumerate(seeds):
            experiment_id = f'config={number}, trial={i}'
            if online:
                reward, iters = JaxParameterTuning.online_trial(
                    env, planner, train_args, seed, number, experiment_id,
                    kwargs['viz'], queue)
            else:
                reward, iters = JaxParameterTuning.offline_trial(
                    env, planner, train_args, seed, number, kwargs['rollouts_per_trial'],
                    experiment_id, kwargs['viz'], queue)
            rewards.append(reward)
            if verbose:
                print(f'    [{number}] trial {i + 1}, seed={seed}, '
                      f'reward={reward:.6f}, iters={iters}', flush=True)

            # stop early if the pruner decides this configuration is not promising
            if trial is not None:
                trial.report(reward, step=i)
                if trial.should_prune():
                    break
        return rewards

    def tune(self, key: int,
             log_file: Optional[str]=None,
             show_dashboard: bool=False,
             print_hyperparams: bool=False) -> ParameterValues:
        '''Tunes the hyper-parameters for Jax planner, returns the best found.

        :param key: RNG seed of the sampler and of the training runs
        :param log_file: optional path to file where tuning progress will be saved
        :param show_dashboard: whether to display the study in optuna-dashboard and the
        training runs in the JaxPlan dashboard (each if installed)
        :param print_hyperparams: whether to print a hyper-parameter summary of the
        optimizer
        '''
        if self.verbose:
            print(JaxPlanner.summarize_system())
        if print_hyperparams:
            print(self.summarize_hyperparameters())

        # clear and prepare output file
        if log_file is not None:
            with open(log_file, 'w', newline='') as file:
                csv.writer(file).writerow(COLUMNS + list(self.hyperparams_dict.keys()))

        # workers report to the pruner through a shared storage: a temporary one is
        # kept until exit, since the study and its dashboard are still read after tuning
        storage_path = self.storage
        if storage_path is None and self.pruner is not None:
            temp_dir = tempfile.mkdtemp()
            atexit.register(shutil.rmtree, temp_dir, ignore_errors=True)
            storage_path = os.path.join(temp_dir, 'study.log')
        if storage_path is None:
            storage = optuna.storages.InMemoryStorage()
        else:
            storage = JournalStorage(JournalFileBackend(
                storage_path, lock_obj=JournalFileOpenLock(storage_path)))

        sampler = self.sampler
        if sampler is None:
            sampler = self.default_sampler(key)
        self.study = study = optuna.create_study(
            study_name=self.study_name, storage=storage, sampler=sampler,
            pruner=self.pruner, direction='maximize', load_if_exists=True)

        # configurations left running by an interrupted run never finish
        for trial in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
            study.tell(trial.number, state=TrialState.FAIL)

        # dash-boards for the study and the training runs, each shown if installed
        dashboard = None
        if show_dashboard:
            self.launch_study_dashboard(storage)
            if JaxPlannerDashboard is not None:
                dashboard = JaxPlannerDashboard()
                dashboard.launch()

        obj_kwargs = {
            'online': self.online,
            'domain': self.env.domain_text,
            'instance': self.env.instance_text,
            'rollouts_per_trial': self.rollouts_per_trial,
            'viz': self.env._visualizer,
            'verbose': self.verbose,
            'study_name': self.study_name,
            'storage': storage if self.pruner is not None else None,
            'pruner': self.pruner
        }
        seeds = [int(seed) for seed in
                 np.random.SeedSequence(key).generate_state(self.eval_trials)]
        distributions = {tag: param.distribution
                         for (tag, param) in self.hyperparams_dict.items()}
        self._run(study, distributions, seeds, obj_kwargs, dashboard, log_file)

        # print summary of results
        states = [trial.state for trial in study.trials]
        counts = ', '.join(f'{state.name.lower()}={states.count(state)}'
                           for state in TrialState if state in states)
        completed = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
        if completed:
            self.best_params = self.search_to_config_params(study.best_params)
            best_target = study.best_value
        else:
            self.best_params, best_target = None, None
        elapsed = time.time() - self._start_time
        print(f'Summary of hyper-parameter optimization:\n'
              f'    time_elapsed         ={datetime.timedelta(seconds=elapsed)}\n'
              f'    configurations       ={counts}\n'
              f'    best_hyper_parameters={self.best_params}\n'
              f'    best_meta_objective  ={best_target}\n')
        self.log_file = log_file
        return self.best_params

    def _run(self, study, distributions, seeds, obj_kwargs, dashboard, log_file) -> None:
        '''Evaluates configurations asynchronously: a new configuration is suggested
        as soon as a worker is free, accounting for the ones still running.'''
        self._start_time = start_time = time.time()
        pruning = obj_kwargs['pruner'] is not None
        done = Queue()
        with multiprocessing.Manager() as manager:

            # queue and parallel thread for handing render events
            if dashboard is not None:
                render_queue = manager.Queue()
                dashboard_thread = threading.Thread(
                    target=JaxParameterTuning.queue_listener, args=(render_queue, dashboard))
                dashboard_thread.start()
            else:
                render_queue = None

            # each configuration runs in a fresh process to release its memory after
            with multiprocessing.get_context(self.pool_context).Pool(
                    processes=self.num_workers, maxtasksperchild=1) as pool:
                started, running = 0, {}
                while True:

                    # fill free workers with new configurations
                    while (len(running) < self.num_workers and started < self.num_configs
                           and time.time() - start_time < self.timeout_tuning):
                        trial = study.ask(distributions)
                        config_params = self.search_to_config_params(trial.params)
                        if self.verbose:
                            elapsed = datetime.timedelta(seconds=time.time() - start_time)
                            config_str = ', '.join(f'{k}={v}' for (k, v) in config_params.items())
                            print(f'[{elapsed}] [{trial.number}] {config_str}', flush=True)
                        pool.apply_async(
                            JaxParameterTuning.objective_function,
                            (self.config_from_template(config_params), trial.number,
                             trial._trial_id if pruning else None, seeds, obj_kwargs,
                             render_queue),
                            callback=lambda result, n=trial.number: done.put((n, result)),
                            error_callback=lambda error, n=trial.number: done.put((n, error)))
                        running[trial.number] = config_params
                        started += 1
                    if not running:
                        break

                    # record the next configuration to finish
                    number, result = done.get()
                    config_params = running.pop(number)
                    if isinstance(result, BaseException):
                        print(f'[{number}] failed:\n' + ''.join(
                            traceback.format_exception(result)), flush=True)
                        frozen = study.tell(number, state=TrialState.FAIL)
                    elif len(result) < len(seeds):
                        frozen = study.tell(number, state=TrialState.PRUNED)
                    else:
                        frozen = study.tell(number, float(np.mean(result)))
                    self._log(study, frozen, config_params, log_file)

            # stop the queue listener thread
            if dashboard is not None:
                render_queue.put(None)
                dashboard_thread.join()

        if started < self.num_configs:
            print(f'[INFO] Global time limit reached after {started} configurations.')

    def _log(self, study, frozen, config_params, log_file) -> None:
        '''Prints and saves the result of a finished configuration.'''
        try:
            best = study.best_trial
        except ValueError:
            best = None
        if best is not None and best.number == frozen.number:
            print(f'[INFO] Found new best average reward {frozen.value:.6f}.', flush=True)
        if log_file is not None:
            seconds = (frozen.datetime_complete - frozen.datetime_start).total_seconds()
            row = [frozen.number, frozen.state.name, frozen.value,
                   None if best is None else best.value, seconds]
            with open(log_file, 'a', newline='') as file:
                csv.writer(file).writerow(row + list(config_params.values()))
