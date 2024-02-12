'''In this example, the gradient of the step function is computed.

The syntax for running this example is:

    python run_gradient.py <domain> <instance>
    
where:
    <domain> is the name of a domain located in the /Examples directory
    <instance> is the instance number
'''
import os
import sys
import jax

import pyRDDLGym

from pyRDDLGym_jax.core.planner import JaxRDDLCompilerWithGrad

    
def main(domain, instance):
    
    # set up the environment
    env = pyRDDLGym.make(domain, instance, vectorized=True)
    
    # create the step function
    compiled = JaxRDDLCompilerWithGrad(rddl=env.model)
    compiled.compile()
    step_fn = compiled.compile_transition()
    
    # gradient of the reward with respect to actions
    def rewards(*args):
        return step_fn(*args)['reward']
    
    grad_fn = jax.grad(rewards, argnums=1)
    print(grad_fn(jax.random.PRNGKey(42), compiled.init_values,
                  compiled.init_values, compiled.model_params))
        
        
if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 2:
        print('python run_gradient.py <domain> <instance>')
        exit(1)
    main(args[0], args[1])
    
