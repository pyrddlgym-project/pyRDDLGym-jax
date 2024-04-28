'''In this simple example, gradient of the return for a simple MDP is computed.

Setting:
    The policy is linear in the state:
        action = p * state
        
    The state evolves as state' = state + action + 1, so for 3-step problem:
        s0 = 0
        s1 = s0 + p * s0 + 1 = 1
        s2 = s1 + p * s1 + 1 = 2 + p
        s3 = s2 + p * s2 + 1 = (1 + p) * (2 + p) + 1 = 3 + 3 * p + p ^ 2
    
    The total return is:
        return = 1 + 2 + p + 3 + 3 * p + p ^ 2 
                = 6 + 4 * p + p ^ 2
    
    The gradient of the return is:
        gradient = 4 + 2 * p
        
    The example given uses p = 2, so it should be:
        return = 18, gradient = 8
    
    For p = 3, it should be:
        return = 27, gradient = 10
'''

import os
import sys
import jax

import pyRDDLGym

from pyRDDLGym_jax.core.planner import JaxRDDLCompilerWithGrad

# a simple domain with state' = state + action
DOMAIN = """
domain test {
    pvariables {
        nf : { non-fluent, real, default = 1.0 };
        state : { state-fluent, real, default = 0.0 };
        action : { action-fluent, real, default = 0.0 };
    };
    cpfs {
        state' = state + action + nf;
    };
    reward = state';
}
"""

INSTANCE = """
non-fluents test_nf {
    domain = test;
}
instance inst_test {
    domain = test;
    non-fluents = test_nf;
    max-nondef-actions = pos-inf;
    horizon  = 5;
    discount = 1.0;
}
"""


def main():
    
    # create the environment
    abs_path = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(abs_path, 'domain.rddl'), 'w') as dom_file:
        dom_file.write(DOMAIN)
    with open(os.path.join(abs_path, 'instance.rddl'), 'w') as inst_file:
        inst_file.write(INSTANCE)
    
    env = pyRDDLGym.make(os.path.join(abs_path, 'domain.rddl'),
                         os.path.join(abs_path, 'instance.rddl'))
    
    # policy is slope * state
    def policy(key, policy_params, hyperparams, step, states):
        return {'action': policy_params['slope'] * states['state']}
    
    # compile the return objective
    compiler = JaxRDDLCompilerWithGrad(env.model)
    compiler.compile()
    step_fn = compiler.compile_rollouts(policy, 3, 1)
    
    def sum_of_rewards(*args):
        return jax.numpy.sum(step_fn(*args)['reward'])
    
    # prepare the arguments (note that batching requires new axis at index 0)
    subs = {k: v[None, ...] for (k, v) in compiler.init_values.items()}
    params = {'slope': 2.0}
    my_args = [jax.random.PRNGKey(42), params, None, subs, compiler.model_params]
    
    # print the fluents over the trajectory, return and gradient
    print(step_fn(*my_args)['fluents'])
    print(sum_of_rewards(*my_args))
    print(jax.grad(sum_of_rewards, argnums=1)(*my_args))
    
    env.close()


if __name__ == "__main__":
    main()
