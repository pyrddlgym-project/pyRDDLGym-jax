import os
import sys
import jax
import numpy as np

import pyRDDLGym
from pyRDDLGym_jax.core.planner import load_config, JaxBackpropPlanner


NUM_TRIALS = 20

    
def main(instance):
    PATH = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(PATH, 'Reservoir_jaxplan.cfg') 
    planner_args, _, train_args = load_config(config_path)
    domain_path = os.path.join(PATH, 'domain.rddl')
    instance_path = os.path.join(PATH, f'instance{instance}.rddl')
    env = pyRDDLGym.make(domain_path, instance_path, vectorized=True)
    planner = JaxBackpropPlanner(rddl=env.model, **planner_args)
    
    # create the planning algorithm
    all_returns = []
    for it in range(NUM_TRIALS):
        train_args['key'] = jax.random.PRNGKey(it)
        returns = []
        for callback in planner.optimize_generator(**train_args):
            returns.append(callback['best_return'])
        all_returns.append(returns)
    all_returns = np.asarray(all_returns).T
    np.savetxt(f'jaxplan_instance{instance}.csv', all_returns)

    env.close()
        

if __name__ == "__main__":
    main(sys.argv[1])
