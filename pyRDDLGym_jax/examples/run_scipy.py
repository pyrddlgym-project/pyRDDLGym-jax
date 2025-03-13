'''In this example, the user has the choice to run the Jax planner using an
optimizer from scipy.minimize.

The syntax for running this example is:

    python run_scipy.py <domain> <instance> <method> [<episodes>]
    
where:
    <domain> is the name of a domain located in the /Examples directory
    <instance> is the instance number
    <method> is the name of a method provided to scipy.optimize.minimize()
    <episodes> is the optional number of evaluation rollouts
'''
import os
import sys
import jax
from scipy.optimize import minimize
    
import pyRDDLGym
from pyRDDLGym.core.debug.exception import raise_warning

from pyRDDLGym_jax.core.planner import load_config, JaxBackpropPlanner, JaxOfflineController

    
def main(domain, instance, method, episodes=1):
    
    # set up the environment
    env = pyRDDLGym.make(domain, instance, vectorized=True)
    
    # load the config file with planner settings
    abs_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(abs_path, 'configs', f'{domain}_slp.cfg') 
    if not os.path.isfile(config_path):
        raise_warning(f'[WARN] Config file {config_path} was not found, '
                      f'using default_slp.cfg.', 'yellow')
        config_path = os.path.join(abs_path, 'configs', 'default_slp.cfg') 
    planner_args, _, train_args = load_config(config_path)

    # create the planning algorithm
    planner = JaxBackpropPlanner(rddl=env.model, **planner_args)
    
    # find the optimal plan
    loss_fn, grad_fn, guess, unravel_fn = planner.as_optimization_problem()
    opt = minimize(loss_fn, jac=grad_fn, x0=guess, method=method, options={'disp': True})
    params = unravel_fn(opt.x)
    
    # evaluate the optimal plan
    controller = JaxOfflineController(planner, params=params, **train_args)
    controller.evaluate(env, episodes=episodes, verbose=True, render=True)    
    env.close()
        
        
if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 3:
        print('python run_scipy.py <domain> <instance> <method> [<episodes>]')
        exit(1)
    kwargs = {'domain': args[0], 'instance': args[1], 'method': args[2]}
    if len(args) >= 4: kwargs['episodes'] = int(args[3])
    main(**kwargs)
    
