import os
import sys
import numpy as np

import pyRDDLGym
from pyRDDLGym_gurobi.core.planner import GurobiOnlineController, load_config


NUM_TRIALS = 5

    
def main(instance):
    PATH = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(PATH, 'Reservoir_gurobiplan.cfg') 
    controller_kwargs = load_config(config_path)  
    domain_path = os.path.join(PATH, 'domain.rddl')
    instance_path = os.path.join(PATH, f'instance{instance}.rddl')
    env = pyRDDLGym.make(domain_path, instance_path)
    controller = GurobiOnlineController(rddl=env.model, **controller_kwargs)
    
    # create the planning algorithm
    all_returns = []
    for it in range(NUM_TRIALS):
        state, _ = env.reset(seed=it)
        returns = []
        cuml_return = 0.0
        for _ in range(env.horizon):
            action = controller.sample_action(state)
            state, reward, *_ = env.step(action)
            cuml_return += reward
            returns.append(cuml_return)
            print(cuml_return)
        all_returns.append(returns)
    all_returns = np.asarray(all_returns).T
    np.savetxt(f'gurobiplan_instance{instance}.csv', all_returns)

    env.close()
        

if __name__ == "__main__":
    main(sys.argv[1])
