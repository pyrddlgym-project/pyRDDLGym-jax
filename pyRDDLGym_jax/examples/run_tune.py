'''This example runs hyper-parameter tuning on the Jax planner. The tuning
is performed using asynchronous parallel Bayesian optimization with Optuna.

The syntax is:

    python run_tune.py <domain> <instance> <method> [<seeds>] [<configs>] [<workers>] [<dashboard>] [<filepath>]
    
where:
    <domain> is the name of a domain located in the /Examples directory
    <instance> is the instance number
    <method> is either slp, drp or replan, as described in JaxExample.py
    <seeds> is the number of independent training runs averaged to evaluate each
    configuration (defaults to 3)
    <configs> is the number of hyper-parameter configurations to evaluate
    (defaults to 50)
    <workers> is the number of configurations evaluated in parallel, which must
    not exceed the number of cores available on the machine (defaults to 4)
    <dashboard> is whether the dashboard is displayed
    <filepath> is the path to save the best configuration (optional)
'''
import os
import sys

import pyRDDLGym

from pyRDDLGym_jax.core.tuning import JaxParameterTuning, Hyperparameter


def power_2(x):
    return int(2 ** x)


def main(domain: str, instance: str, method: str, 
         seeds: int=3, configs: int=50, workers: int=4, dashboard: bool=False, 
         filepath: str='') -> None:
    
    # set up the environment   
    env = pyRDDLGym.make(domain, instance, vectorized=True)
    
    # load the config file with planner settings
    abs_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(abs_path, 'configs', f'tuning_{method}.cfg')
    with open(config_path, 'r') as file:
        config_template = file.read() 
    
    # map parameters in the config that will be tuned
    hyperparams = [
        Hyperparameter.real('MODEL_WEIGHT_TUNE', 0.1, 1e4, log=True),
        Hyperparameter.real('LEARNING_RATE_TUNE', 1e-5, 1., log=True),
        Hyperparameter.integer('LAYER1_TUNE', 1, 8, transform=power_2),
        Hyperparameter.integer('LAYER2_TUNE', 1, 8, transform=power_2),
        Hyperparameter.integer('ROLLOUT_HORIZON_TUNE', 1, min(env.horizon, 100))
    ]
    
    # build the tuner and tune
    tuning = JaxParameterTuning(env=env,
                                config_template=config_template,
                                hyperparams=hyperparams,
                                online=(method == 'replan'),
                                num_seeds=seeds, num_workers=workers, num_configs=configs)
    tuning.tune(key=42, 
                log_file=f'gp_{method}_{domain}_{instance}.csv', 
                show_dashboard=dashboard)
    if filepath is not None and filepath:
        with open(filepath, 'w') as file:
            file.write(tuning.best_config)
    
    # evaluate the agent on the best parameters
    controller = tuning.build_best_controller()
    controller.evaluate(env, episodes=1, verbose=True, render=True)
    env.close()


def run_from_args(args):
    if len(args) < 3:
        print('python run_tune.py <domain> <instance> <method> [<seeds>] [<configs>] [<workers>] [<dashboard>] [<filepath>]')
        exit(1)
    if args[2] not in ['drp', 'slp', 'replan']:
        print('<method> in [drp, slp, replan]')
        exit(1)
    kwargs = {'domain': args[0], 'instance': args[1], 'method': args[2]}
    if len(args) >= 4: kwargs['seeds'] = int(args[3])
    if len(args) >= 5: kwargs['configs'] = int(args[4])
    if len(args) >= 6: kwargs['workers'] = int(args[5])
    if len(args) >= 7: kwargs['dashboard'] = str(args[6]).strip().lower() in {'true', '1'}
    if len(args) >= 8: kwargs['filepath'] = str(args[7])
    main(**kwargs) 


if __name__ == "__main__":
    run_from_args(sys.argv[1:])
