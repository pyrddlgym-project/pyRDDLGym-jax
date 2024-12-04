'''This example runs hyper-parameter tuning on the Jax planner. The tuning
is performed using a batched parallelized Bayesian optimization.

The syntax is:

    python run_tune.py <domain> <instance> <method> [<trials>] [<iters>] [<workers>]
    
where:
    <domain> is the name of a domain located in the /Examples directory
    <instance> is the instance number
    <method> is either slp, drp or replan, as described in JaxExample.py
    <trials> is the number of trials to simulate when estimating the meta loss
    (defaults to 5)
    <iters> is the number of iterations of Bayesian optimization to perform
    (defaults to 20)
    <workers> is the number of parallel workers (i.e. batch size), which must
    not exceed the number of cores available on the machine (defaults to 4)
'''
import os
import sys

import pyRDDLGym

from pyRDDLGym_jax.core.tuning import JaxParameterTuning, Hyperparameter
from pyRDDLGym_jax.core.planner import (
    load_config_from_string, JaxBackpropPlanner, 
    JaxOfflineController, JaxOnlineController
)

def power_2(x):
    return int(2 ** x)


def power_10(x):
    return 10.0 ** x


def main(domain, instance, method, trials=5, iters=20, workers=4):
    
    # set up the environment   
    env = pyRDDLGym.make(domain, instance, vectorized=True)
    
    # load the config file with planner settings
    abs_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(abs_path, 'configs', f'tuning_{method}.cfg')
    with open(config_path, 'r') as file:
        config_template = file.read() 
    
    # map parameters in the config that will be tuned
    hyperparams = [
        Hyperparameter('MODEL_WEIGHT_TUNE', -1., 5., power_10),
        Hyperparameter('POLICY_WEIGHT_TUNE', -2., 2., power_10),
        Hyperparameter('LEARNING_RATE_TUNE', -5., 1., power_10),
        Hyperparameter('LAYER1_TUNE', 1, 8, power_2),
        Hyperparameter('LAYER2_TUNE', 1, 8, power_2),
        Hyperparameter('ROLLOUT_HORIZON_TUNE', 1, min(env.horizon, 100), int)       
    ]
    
    # build the tuner and tune
    tuning = JaxParameterTuning(env=env,
                                config_template=config_template,
                                hyperparams=hyperparams,
                                online=method == 'replan',
                                eval_trials=trials,
                                num_workers=workers,
                                gp_iters=iters)
    tuning.tune(key=42, log_file=f'gp_{method}_{domain}_{instance}.csv')
    tuning.save_plot(plot_file=f'{method}_{domain}_{instance}.pdf')
    
    # evaluate the agent on the best parameters
    planner_args, _, train_args = load_config_from_string(tuning.best_config)
    planner = JaxBackpropPlanner(rddl=env.model, **planner_args)
    klass = JaxOnlineController if method == 'replan' else JaxOfflineController
    controller = klass(planner, **train_args)
    controller.evaluate(env, episodes=1, verbose=True, render=True)
    env.close()


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 3:
        print('python run_tune.py <domain> <instance> <method> [<trials>] [<iters>] [<workers>]')
        exit(1)
    if args[2] not in ['drp', 'slp', 'replan']:
        print('<method> in [drp, slp, replan]')
        exit(1)
    kwargs = {'domain': args[0], 'instance': args[1], 'method': args[2]}
    if len(args) >= 4: kwargs['trials'] = int(args[3])
    if len(args) >= 5: kwargs['iters'] = int(args[4])
    if len(args) >= 6: kwargs['workers'] = int(args[5])
    main(**kwargs) 
    
