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


from ast import literal_eval
import atexit
import configparser
import csv
import datetime
import importlib.util
import multiprocessing
import multiprocessing.connection
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import traceback
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import webbrowser
from socketserver import ThreadingMixIn
from wsgiref.simple_server import make_server, WSGIRequestHandler, WSGIServer

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
from pyRDDLGym_jax.core.progress import JaxTuningProgress, format_value

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

    def describe(self) -> Tuple[str, str, str, str]:
        '''Returns the tag, range, scale and range of config values for display.'''
        dist = self.distribution
        if isinstance(dist, CategoricalDistribution):
            values = dist.choices
            space = '{' + ', '.join(format_value(value) for value in values) + '}'
            config = '{' + ', '.join(format_value(self.to_config(value))
                                     for value in values) + '}'
            scale = 'categorical'
        else:
            space = f'[{format_value(dist.low)}, {format_value(dist.high)}]'
            config = (f'[{format_value(self.to_config(dist.low))}, '
                      f'{format_value(self.to_config(dist.high))}]')
            if isinstance(dist, IntDistribution):
                scale = 'log int' if dist.log else 'int'
                if dist.step != 1:
                    scale += f' step {dist.step}'
            else:
                scale = 'log' if dist.log else 'linear'
                if dist.step is not None:
                    scale += f' step {format_value(dist.step)}'
        return self.tag, space, scale, ('' if self.transform is None else config)

    def __str__(self) -> str:
        (tag, space, scale, config) = self.describe()
        return f'{tag}: {space} {scale}' + (f' -> {config}' if config else '')


Kwargs = Dict[str, Any]
ParameterValues = Dict[str, Any]
Hyperparameters = Iterable[Hyperparameter]

COLUMNS = ['number', 'state', 'target', 'best_target', 'seconds', 'std']


def _tag_pattern(tag: str) -> re.Pattern:
    return re.compile(r'\b' + re.escape(tag) + r'\b')


class _ConfigProcess:
    '''A configuration being evaluated in its own worker process, which sends the return
    of each training run and then its outcome through a pipe.'''

    def __init__(self, trial, process, reader, config_params: ParameterValues) -> None:
        self.trial = trial
        self.process = process
        self.reader = reader
        self.config_params = config_params
        self.start = time.time()
        self.outcome = None

    def receive(self, number: int, progress: JaxTuningProgress) -> None:
        '''Records the messages sent by the worker so far.'''
        while self.reader is not None:
            try:
                if not self.reader.poll():
                    return
                message = self.reader.recv()
            except (EOFError, OSError):
                self.reader.close()
                self.reader = None
                return
            if message[0] == 'seed':
                progress.seed_finished(number, message[1], message[2])
            elif message[0] == 'done':
                self.outcome = message
            else:
                self.outcome = ('error', JaxParameterTuning._describe_error(*message[1:]))

    def stop(self, grace: float) -> None:
        '''Waits up to grace seconds for the worker to exit, then kills it.'''
        self.process.join(grace)
        if self.process.is_alive():
            self.process.kill()
            self.process.join()
        self.process.close()
        if self.reader is not None:
            self.reader.close()


class JaxParameterTuning:
    '''A general-purpose class for tuning a Jax planner with Optuna. Each configuration
    is evaluated in a fresh process by averaging the return of several independent
    training runs; the training seeds are shared across configurations (common random
    numbers), which makes the comparison between configurations paired.'''

    def __init__(self, env: RDDLEnv,
                 config_template: str,
                 hyperparams: Hyperparameters,
                 online: bool,
                 num_seeds: int=5,
                 rollouts_per_seed: int=1,
                 verbose: bool=True,
                 timeout_tuning: float=np.inf,
                 timeout_config: float=np.inf,
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
        :param num_seeds: how many independent training runs (seeds) to average to estimate
        the return of each configuration
        :param rollouts_per_seed: how many rollouts to perform during evaluation
        at the end of each training run (only applies when online=False)
        :param verbose: whether to print intermediate results of tuning
        :param timeout_tuning: no new configurations are started after this many seconds
        :param timeout_config: a configuration is stopped and fails after this many seconds
        :param pool_context: context for the worker processes (default "spawn")
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
        self.hyperparams_dict = {}
        for param in hyperparams:
            if _tag_pattern(param.tag).search(config_template):
                self.hyperparams_dict[param.tag] = param
            else:
                raise_warning(f'[WARN] hyper-parameter {param.tag} does not appear in the '
                              f'config template, so it is not tuned.', 'yellow')
        self.online = online
        self.num_seeds = num_seeds
        self.rollouts_per_seed = rollouts_per_seed
        self.verbose = verbose
        self.timeout_tuning = timeout_tuning
        self.timeout_config = timeout_config
        self.pool_context = pool_context
        self.num_workers = num_workers
        self.num_configs = num_configs
        self.sampler = sampler
        self.pruner = pruner
        self.storage = storage
        self.study_name = study_name
        self._check_template()

    def _check_template(self) -> None:
        '''Parses the config at the lowest values of the hyper-parameters, so that an
        invalid template fails here rather than in every worker.'''
        params = {}
        for (tag, param) in self.hyperparams_dict.items():
            dist = param.distribution
            if isinstance(dist, CategoricalDistribution):
                params[tag] = dist.choices[0]
            else:
                params[tag] = dist.low
        config_string = self.config_from_template(self.search_to_config_params(params))
        try:
            planner_args, _, _ = load_config_from_string(config_string)
        except Exception as error:
            entry = self._invalid_entry(config_string)
            where = '' if entry is None else f' in {entry}'
            raise ValueError(f'Config template is invalid{where} after replacing the '
                             f'hyper-parameters by values {params}: {error}') from error

        # each configuration is evaluated with a single optimizer
        parallel_updates = planner_args.get('parallel_updates', 1)
        if parallel_updates > 1:
            raise_warning(f'[WARN] parallel_updates={parallel_updates} in the config template '
                          f'is ignored during tuning', 'yellow')

    @staticmethod
    def _invalid_entry(config_string: str) -> Optional[str]:
        '''Returns the first entry of the config whose value is not a Python literal.'''
        parser = configparser.RawConfigParser()
        parser.optionxform = str
        parser.read_string(config_string)
        for section in parser.sections():
            for (key, value) in parser.items(section):
                try:
                    literal_eval(value)
                except (ValueError, SyntaxError):
                    return f'[{section}] {key}={value}'
        return None

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
    def launch_study_dashboard(storage: optuna.storages.BaseStorage,
                               port: int=8080) -> Optional[str]:
        '''Launches optuna-dashboard for the study in a browser window if installed, and
        returns its address.'''
        try:
            import optuna_dashboard
        except ImportError:
            raise_warning('[WARN] optuna-dashboard is not installed, so the study is not '
                          'shown: install it with pip install optuna-dashboard.', 'yellow')
            return None

        # serve without logging every request of the dashboard to the console
        class QuietHandler(WSGIRequestHandler):
            def log_message(self, *args):
                pass

        # requests are handled concurrently: a single-threaded server blocks on idle
        # connections that browsers open in advance, so the page never loads
        class ThreadedServer(ThreadingMixIn, WSGIServer):
            daemon_threads = True

        server = make_server('127.0.0.1', port, optuna_dashboard.wsgi(storage),
                             server_class=ThreadedServer, handler_class=QuietHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        url = f'http://127.0.0.1:{port}/'
        webbrowser.open_new(url)
        return url

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
    def offline_trial(env, planner, train_args, seed, number, rollouts_per_seed,
                      experiment_id, viz, queue):
        '''Trains the planner offline, then returns the average return of its policy and
        the number of training iterations.'''
        key = jax.random.PRNGKey(seed)

        # for the dashboard
        if queue is not None and JaxPlannerDashboard is not None:
            queue.put((experiment_id, JaxPlannerDashboard.get_planner_info(planner),
                       seed, viz))

        # train the policy
        callback = None
        for callback in planner.optimize_generator(key=key, **train_args):
            if queue is not None and queue.empty():
                queue.put((experiment_id,
                           JaxPlannerDashboard.compact_callback(callback, planner.rddl)))
        best_params = None if callback is None else callback['best_params']
        hyperparams = None if callback is None else callback['policy_hyperparams']

        # evaluate the policy in the real environment with its training hyper-parameters
        policy = JaxOfflineController(
            planner=planner, key=key, tqdm_position=number, params=best_params,
            eval_hyperparams=hyperparams, train_on_reset=False)
        total_reward = policy.evaluate(env, episodes=rollouts_per_seed, seed=seed)['mean']
        iters = 0 if callback is None else int(callback['iteration']) + 1
        return total_reward, iters

    @staticmethod
    def online_trial(env, planner, train_args, seed, number, experiment_id, viz, queue):
        '''Runs the planner online for one episode, then returns its return and the
        average number of training iterations per decision step.'''
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
        state, _ = env.reset(seed=seed)
        elapsed_time, total_iters, steps = 0.0, 0, 0
        for step in range(env.horizon):
            action = policy.sample_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            callback = policy.callback
            total_iters += int(callback['iteration']) + 1
            steps += 1
            elapsed_time += callback['elapsed_time']
            callback['iteration'] = step
            callback['progress'] = int(100 * (step + 1.) / env.horizon)
            callback['elapsed_time'] = elapsed_time
            if queue is not None and queue.empty():
                queue.put((experiment_id,
                           JaxPlannerDashboard.compact_callback(callback, planner.rddl)))
            if terminated or truncated:
                break
        return total_reward, total_iters / max(1, steps)

    @staticmethod
    def _silence_output() -> None:
        '''Discards all output of this process.'''
        for stream in (sys.stdout, sys.stderr):
            if stream is not None:
                stream.flush()
        devnull = open(os.devnull, 'w')
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)
        sys.stdout = sys.stderr = devnull

    @staticmethod
    def _worker(conn, config_string: str, number: int, trial_id: Optional[int],
                seeds: List[int], kwargs: Kwargs, queue: object) -> None:
        '''Evaluates a single configuration in a worker process, sending the return of each
        training run and then the outcome through the given connection.'''
        JaxParameterTuning._silence_output()
        iterations = []

        def on_seed(i, reward, iters):
            iterations.append(iters)
            conn.send(('seed', i, reward))

        try:
            rewards = JaxParameterTuning.objective_function(
                config_string, number, trial_id, seeds, kwargs, queue, on_seed=on_seed)
            message = ('done', rewards, iterations)
        except BaseException as error:
            message = ('error', type(error).__name__, str(error), traceback.format_exc())
        conn.send(message)
        conn.close()

    @staticmethod
    def objective_function(config_string: str, number: int, trial_id: Optional[int],
                           seeds: List[int], kwargs: Kwargs, queue: object,
                           on_seed: Optional[Callable]=None) -> List[float]:
        '''An objective function that evaluates a single configuration: returns the return
        of each training run, which are reported to the pruner if any (the list is cut
        short if the configuration is pruned), and to on_seed(index, return, iterations) if
        given.'''
        online = kwargs['online']
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
            experiment_id = f'config={number}, seed={i}'
            if online:
                reward, iters = JaxParameterTuning.online_trial(
                    env, planner, train_args, seed, number, experiment_id,
                    kwargs['viz'], queue)
            else:
                reward, iters = JaxParameterTuning.offline_trial(
                    env, planner, train_args, seed, number, kwargs['rollouts_per_seed'],
                    experiment_id, kwargs['viz'], queue)
            rewards.append(reward)
            if on_seed is not None:
                on_seed(i, reward, iters)

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
        :param print_hyperparams: whether to print the settings and search space even if
        not verbose
        '''
        if self.verbose:
            print(JaxPlanner.summarize_system())

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

        # Optuna's log of the study and each trial is replaced by the table of results
        verbosity = optuna.logging.get_verbosity()
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        try:
            self.study = study = optuna.create_study(
                study_name=self.study_name, storage=storage, sampler=sampler,
                pruner=self.pruner, direction='maximize', load_if_exists=True)
        finally:
            optuna.logging.set_verbosity(verbosity)

        # configurations left running by an interrupted run never finish
        for trial in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
            study.tell(trial.number, state=TrialState.FAIL)

        # the output file starts with the configurations of a resumed study
        previous = study.get_trials(
            deepcopy=False, states=(TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL))
        if log_file is not None:
            with open(log_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(self._log_columns())
                best_value = None
                for frozen in previous:
                    if (frozen.state == TrialState.COMPLETE
                            and (best_value is None or frozen.value > best_value)):
                        best_value = frozen.value
                    writer.writerow(self._log_row(frozen, best_value))

        # dash-boards for the study and the training runs, each shown if installed
        dashboard, dashboard_urls = None, []
        if show_dashboard:
            url = self.launch_study_dashboard(storage)
            if url is not None:
                dashboard_urls.append(('study', url))
            if JaxPlannerDashboard is not None:
                dashboard = JaxPlannerDashboard()
                dashboard.launch()
                dashboard_urls.append(('runs', 'http://127.0.0.1:1222/'))

        # print the settings and search space above the table of results
        progress = JaxTuningProgress(
            self.hyperparams_dict.keys(), self.num_configs, self.num_seeds,
            show_seeds=self.pruner is not None)
        if self.verbose or print_hyperparams:
            rddl = self.env.model
            settings = [
                [('configs', self.num_configs), ('seeds', self.num_seeds),
                 ('workers', self.num_workers),
                 ('timeout', 'none' if self.timeout_tuning == np.inf
                  else datetime.timedelta(seconds=self.timeout_tuning)),
                 ('config timeout', 'none' if self.timeout_config == np.inf
                  else datetime.timedelta(seconds=self.timeout_config))],
                [('sampler', type(sampler).__name__),
                 ('pruner', 'none' if self.pruner is None else type(self.pruner).__name__),
                 ('storage', storage_path or 'in memory')],
                [('results', log_file or 'not saved')]
            ]
            if dashboard_urls:
                settings.append([(f'{name} dashboard', url) for (name, url) in dashboard_urls])
            if previous:
                resumed = [('resumed', f'{len(previous)} configs')]
                best = self._best_trial(study)
                if best is not None:
                    resumed.append(('best', f'#{best.number} return {best.value:.3f}'))
                settings.append(resumed)
            progress.print_header(
                'JaxPlan tuning',
                f'{rddl.domain_name} / {rddl.instance_name}  '
                f'{"online" if self.online else "offline"}',
                settings, [param.describe() for param in self.hyperparams_dict.values()])

        obj_kwargs = {
            'online': self.online,
            'domain': self.env.domain_text,
            'instance': self.env.instance_text,
            'rollouts_per_seed': self.rollouts_per_seed,
            'viz': self.env._visualizer,
            'study_name': self.study_name,
            'storage': storage if self.pruner is not None else None,
            'pruner': self.pruner
        }
        seeds = [int(seed) for seed in
                 np.random.SeedSequence(key).generate_state(self.num_seeds)]
        distributions = {tag: param.distribution
                         for (tag, param) in self.hyperparams_dict.items()}

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        try:
            started = self._run(study, distributions, seeds, obj_kwargs, dashboard,
                                log_file, progress)
        finally:
            optuna.logging.set_verbosity(verbosity)

        # print summary of results
        names = {TrialState.COMPLETE: 'complete', TrialState.PRUNED: 'pruned',
                 TrialState.FAIL: 'failed'}
        states = [trial.state for trial in study.trials]
        counts = {names[state]: states.count(state) for state in names if state in states}
        completed = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
        if completed:
            self.best_params = self.search_to_config_params(study.best_params)
            best = (study.best_trial.number, study.best_value, self.best_params)
        else:
            self.best_params, best = None, None
        message = None
        if started < self.num_configs:
            message = f'time limit reached after {started} configurations'
        progress.print_summary(time.time() - self._start_time, counts, best, message)
        self.log_file = log_file
        return self.best_params

    @staticmethod
    def _describe_error(name: str, message: str, remote_traceback: str) -> str:
        '''Describes an error raised in a worker in one line, with the place it was raised
        from the traceback that the worker sends with it.'''
        lines = message.strip().splitlines()
        description = f'{name}: {lines[0] if lines else ""}'
        frames = re.findall(r'File "([^"]+)", line (\d+), in (\S+)', remote_traceback)
        if frames:
            (file, line, function) = frames[-1]
            description += f'  ({os.path.basename(file)}:{line} in {function})'
        return description

    @staticmethod
    def _describe_exit(exitcode: int) -> str:
        '''Describes the exit of a worker that died without sending a result.'''
        if exitcode < 0:
            return f'worker was killed by signal {-exitcode}'
        code = f'{exitcode:#x}' if exitcode > 255 else str(exitcode)
        return f'worker exited with code {code} without a result'

    def _run(self, study, distributions, seeds, obj_kwargs, dashboard, log_file,
             progress) -> int:
        '''Evaluates configurations asynchronously: a new configuration is suggested
        as soon as a worker is free, accounting for the ones still running. A configuration
        fails if its worker exits without a result or exceeds the time limit.'''
        self._start_time = start_time = time.time()
        pruning = obj_kwargs['pruner'] is not None
        context = multiprocessing.get_context(self.pool_context)

        # queue and parallel thread for handing render events
        if dashboard is not None:
            render_queue = context.Queue()
            dashboard_thread = threading.Thread(
                target=JaxParameterTuning.queue_listener, args=(render_queue, dashboard),
                daemon=True)
            dashboard_thread.start()
        else:
            render_queue = None

        # each configuration runs in a fresh process to release its memory after
        started, running = 0, {}
        if self.verbose:
            progress.start()
        try:
            while True:

                # fill free workers with new configurations
                while (len(running) < self.num_workers
                       and started < self.num_configs
                       and time.time() - start_time < self.timeout_tuning):
                    trial = study.ask(distributions)
                    config_params = self.search_to_config_params(trial.params)
                    reader, writer = context.Pipe(duplex=False)
                    process = context.Process(
                        target=JaxParameterTuning._worker,
                        args=(writer, self.config_from_template(config_params),
                              trial.number, trial._trial_id if pruning else None, seeds,
                              obj_kwargs, render_queue),
                        daemon=True)
                    process.start()
                    writer.close()
                    running[trial.number] = _ConfigProcess(
                        trial, process, reader, config_params)
                    progress.config_started(trial.number)
                    started += 1
                if not running:
                    break

                # wait for messages and exits of workers: the timeout checks the time
                # limit, and keeps the wait interruptible on all platforms
                handles = [config.process.sentinel for config in running.values()]
                handles += [config.reader for config in running.values()
                            if config.reader is not None]
                multiprocessing.connection.wait(handles, timeout=1.)

                # record the configurations that finished, died or ran out of time:
                # liveness is checked first, since a worker exits right after its result
                for (number, config) in list(running.items()):
                    alive = config.process.is_alive()
                    config.receive(number, progress)
                    if config.outcome is None and not alive:
                        config.outcome = ('error', self._describe_exit(config.process.exitcode))
                    elif (config.outcome is None
                          and time.time() - config.start > self.timeout_config):
                        config.outcome = ('error', f'exceeded the time limit of '
                                                   f'{self.timeout_config:g} seconds')
                    if config.outcome is not None:
                        del running[number]
                        config.stop(grace=5. if config.outcome[0] == 'done' else 0.)
                        self._finish(study, config, seeds, log_file, progress)
        finally:
            for config in running.values():
                config.stop(grace=0.)
            progress.stop()

        # stop the queue listener thread
        if dashboard is not None:
            render_queue.put(None)
            dashboard_thread.join()
        return started

    def _finish(self, study, config, seeds, log_file, progress) -> None:
        '''Tells the study the outcome of a finished configuration, and records it: the
        return and training iterations of each run are kept as attributes of the trial.'''
        trial = config.trial
        error, rewards = None, None
        if config.outcome[0] == 'error':
            error = config.outcome[1]
        else:
            (rewards, iterations) = config.outcome[1:]
            # non-finite values are not valid JSON in the storage
            trial.set_user_attr('returns', [float(reward) if np.isfinite(reward)
                                            else str(float(reward)) for reward in rewards])
            trial.set_user_attr('iterations', iterations)
            if np.isnan(np.mean(rewards)):
                error = 'return is NaN'
        if error is not None:
            trial.set_user_attr('error', error)
            frozen = study.tell(trial, state=TrialState.FAIL)
        elif len(rewards) < len(seeds):
            frozen = study.tell(trial, state=TrialState.PRUNED)
        else:
            frozen = study.tell(trial, float(np.mean(rewards)))
        is_best = self._log(study, frozen, log_file)
        if self.verbose:
            seconds = (frozen.datetime_complete - frozen.datetime_start).total_seconds()
            state = {TrialState.COMPLETE: 'complete',
                     TrialState.PRUNED: 'pruned'}.get(frozen.state, 'failed')
            progress.config_finished(
                trial.number, state, rewards, seconds, list(config.config_params.values()),
                is_best, error)

    @staticmethod
    def _best_trial(study) -> Optional[optuna.trial.FrozenTrial]:
        try:
            return study.best_trial
        except ValueError:
            return None

    def _log_columns(self) -> List[str]:
        return (COLUMNS + [f'return_{i}' for i in range(self.num_seeds)] +
                [f'iterations_{i}' for i in range(self.num_seeds)] +
                list(self.hyperparams_dict.keys()) + ['error'])

    def _log_row(self, frozen, best_value) -> List[Any]:
        '''The row of the output file of a finished configuration, with a column for
        each training run.'''
        seconds = (frozen.datetime_complete - frozen.datetime_start).total_seconds()
        returns = frozen.user_attrs.get('returns', [])
        std = float(np.std(returns)) if len(returns) > 1 else None

        def per_seed(values):
            values = list(values)[:self.num_seeds]
            return values + [None] * (self.num_seeds - len(values))

        config_values = [param.to_config(frozen.params[tag]) if tag in frozen.params else None
                         for (tag, param) in self.hyperparams_dict.items()]
        return ([frozen.number, frozen.state.name, frozen.value, best_value, seconds, std] +
                per_seed(returns) + per_seed(frozen.user_attrs.get('iterations', [])) +
                config_values + [frozen.user_attrs.get('error')])

    def _log(self, study, frozen, log_file) -> bool:
        '''Saves the result of a finished configuration, and returns whether it is the
        best so far.'''
        best = self._best_trial(study)
        if log_file is not None:
            with open(log_file, 'a', newline='') as file:
                csv.writer(file).writerow(
                    self._log_row(frozen, None if best is None else best.value))
        return best is not None and best.number == frozen.number
