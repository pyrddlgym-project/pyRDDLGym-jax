import os
import sys
import jax
import numpy as np
import time

import pyRDDLGym
from pyRDDLGym_jax.core.planner import load_config, JaxBackpropPlanner, JaxOfflineController


NUM_TRIALS = 20


def measure_evaluation_time(planner, policy_params, env):
    controller = JaxOfflineController(planner, params=policy_params)
    start_time = time.time()
    controller.evaluate(env, episodes=100)
    end_time = time.time()
    print(f'eval time = {(end_time - start_time) / 100}')


def main(instance):
    PATH = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(PATH, 'Reservoir_jaxplan.cfg') 
    planner_args, _, train_args = load_config(config_path)
    domain_path = os.path.join(PATH, 'domain.rddl')
    instance_path = os.path.join(PATH, f'instance{instance}.rddl')
    env = pyRDDLGym.make(domain_path, instance_path, vectorized=True)
    planner = JaxBackpropPlanner(rddl=env.model, **planner_args)
    
    # create the planning algorithm
    all_returns, all_times = [], []
    for it in range(NUM_TRIALS):
        train_args['key'] = jax.random.PRNGKey(it)
        returns, times = [], []
        for callback in planner.optimize_generator(**train_args):
            returns.append(callback['best_return'])
            times.append(callback['elapsed_time'])
        all_returns.append(returns)
        all_times.append(times)
        time_to_train = callback['elapsed_time'] / callback['iteration']
        print(time_to_train)
        print(measure_evaluation_time(planner, callback['best_params'], env))
    all_returns = np.asarray(all_returns).T
    all_times = np.asarray(all_times).T
    np.savetxt(f'jaxplan_instance{instance}_return_slp.csv', all_returns)
    np.savetxt(f'jaxplan_instance{instance}_time_slp.csv', all_times)

    env.close()
        

if __name__ == "__main__":
    main(sys.argv[1])
