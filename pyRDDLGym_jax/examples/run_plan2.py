'''In this example, the user has the choice to run the Jax planner with three
different options:
    
    1. slp runs the straight-line planner offline, which trains an open-loop plan
    2. drp runs the deep reactive policy, which trains a policy network
    3. replan runs the straight-line planner online, at every decision epoch
    
The syntax for running this example is:

    python run_plan2.py <domain> <instance> <method> [<episodes>]
    
where:
    <domain> is the name of a domain located in the /Examples directory
    <instance> is the instance number
    <method> is either slp, drp, or replan
'''
import os
import sys

import pyRDDLGym
from pyRDDLGym.core.debug.exception import raise_warning

from pyRDDLGym_jax.core.planner import (
    load_config, JaxBackpropPlannerWithQ, JaxOfflineController, JaxOnlineController,
    JaxQValueFunction,
)

    
def main(domain, instance, method, episodes=1):
    
    # set up the environment
    env = pyRDDLGym.make(domain, instance, vectorized=True, enforce_action_constraints=False)
    
    # load the config file with planner settings
    abs_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(abs_path, 'configs', f'{domain}_{method}.cfg') 
    if not os.path.isfile(config_path):
        raise_warning(f'Config file {domain}_{method}.cfg was not found, '
                      f'using default config (parameters could be suboptimal).', 
                      'red')
        config_path = os.path.join(abs_path, 'configs', f'default_{method}.cfg') 
    planner_args, _, train_args = load_config(config_path)
    
    # create the planning algorithm
    q = JaxQValueFunction([], [], [64, 32])
    planner = JaxBackpropPlannerWithQ(rddl=env.model, q=q, **planner_args)
    
    # create the controller   
    if method == 'replan':
        controller = JaxOnlineController(planner, **train_args)
    else:
        controller = JaxOfflineController(planner, **train_args)
    
    controller.evaluate(env, episodes=episodes, verbose=True, render=True)
    
    env.close()
        
        
if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 3:
        print('python run_plan2.py <domain> <instance> <method> [<episodes>]')
        exit(1)
    if args[2] not in ['drp', 'slp', 'replan']:
        print('<method> in [drp, slp, replan]')
        exit(1)
    kwargs = {'domain': args[0], 'instance': args[1], 'method': args[2]}
    if len(args) >= 4: kwargs['episodes'] = int(args[3])
    main(**kwargs)
    
