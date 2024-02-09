# pyRDDLGym-jax

Author: [Mike Gimelfarb](https://mike-gimelfarb.github.io)

This directory provides:
1. automated translation and compilation of RDDL description files into the JAX auto-diff library, which allows any RDDL domain to be converted to a differentiable simulator!
2. a powerful gradient-based planning algorithm, with extendible and flexible policy class representations, automatic discrete model relaxations, and much more!

## Installation

To use the compiler or planner without the automated hyper-parameter tuning, you will need the following packages installed: 
- ``pyRDDLGym>=2.0.1``
- ``tqdm>=4.66``
- ``jax>=0.4.12``
- ``optax>=0.1.9``
- ``dm-haiku>=0.0.10``
- ``tensorflow>=2.13.0``
- ``tensorflow-probability>=0.21.0``

Additionally, if you wish to run the automated tuning optimization, you will also need the ``bayesian-optimization>=1.4.3`` package.

You can install this package, together with all of its requirements as follows

```shell
# Create a new conda environment
conda create -n jaxplan
conda activate jaxplan

# Manually install pyRDDLGym >= 2.0.1
cd ~/path/to/pyRDDLGym
git checkout pyRDDLGym-v2-branch
pip install -e .

# Manually install rddlrepository >= 2
cd ~/path/to/rddlrepository
git checkout rddlrepository-v2-branch
pip install -e .

# Install pyRDDLGym-jax
cd ~/path/to/pyRDDLGym-jax
pip install -e .
```

A pip installer will be coming soon.

## Running the Basic Examples

A basic run script is provided to run the Jax Planner on any domain in ``rddlrepository``, provided a config file is available (currently, only a limited subset of configs are provided as examples).
The example can be run as follows in a standard shell:

```shell
python -m pyRDDLGym_jax.examples.run_plan <domain> <instance> <method> [<episodes>]
```

where:
- ``domain`` is the domain identifier as specified in rddlrepository (i.e. Wildfire_MDP_ippc2014)
- ``instance`` is the instance identifier (i.e. 1, 2, ... 10)
- ``method`` is the planning method to use (i.e. drp, slp, replan)
- ``episodes`` is the number of episodes to evaluate the learned policy

A basic run script is also provided to run the automatic hyper-parameter tuning. The structure of this stript is similar to the one above

```shell
python -m pyRDDLGym_jax.examples.run_tune <domain> <instance> <method> [<trials>] [<iters>] [<workers>]
```

where:
- ``domain`` is the domain identifier as specified in rddlrepository (i.e. Wildfire_MDP_ippc2014)
- ``instance`` is the instance identifier (i.e. 1, 2, ... 10)
- ``method`` is the planning method to use (i.e. drp, slp, replan)
- ``trials`` is the number of trials/episodes to average in evaluating each hyper-parameter setting
- ``iters`` is the maximum number of iterations/evaluations of Bayesian optimization to perform
- ``workers`` is the number of parallel evaluations to be done at each iteration, e.g. the total evaluations = ``iters * workers``

For example, copy and pasting the following will train the Jax Planner on the Quadcopter domain with 4 drones:

```shell
python -m pyRDDLGym_jax.examples.run_plan Quadcopter 1 slp
```

After several minutes of optimization, you should get a visualization as follows:

<p align="center">
<img src="Images/quadcopter.gif" width="400" height="400" margin=1/>
</p>

## Writing a Configuration File

The simplest way to interface with the Planner is to write a configuration file with all the necessary hyper-parameters.
The basic structure of a configuration file is provided below:

```ini
[Model]
logic='FuzzyLogic'
logic_kwargs={'weight': 20}
tnorm='ProductTNorm'
tnorm_kwargs={}

[Optimizer]
method='JaxStraightLinePlan'
method_kwargs={}
optimizer='rmsprop'
optimizer_kwargs={'learning_rate': 0.001}
batch_size_train=1
batch_size_test=1

[Training]
key=42
epochs=5000
train_seconds=30
```

The configuration file contains three sections:
- ``[Model]`` specifies the fuzzy logic operations used to relax discrete operations to differentiable approximations, the ``weight`` parameter for example dictates how well the approximation will fit to the true operation,
and ``tnorm`` specifies the type of [fuzzy logic](https://en.wikipedia.org/wiki/T-norm_fuzzy_logics) for relacing logical operations in RDDL (e.g. ``ProductTNorm``, ``GodelTNorm``, ``LukasiewiczTNorm``)
- ``[Optimizer]`` generally specify the optimizer and plan settings, the ``method`` specifies the plan/policy representation (e.g. ``JaxStraightLinePlan``, ``JaxDeepReactivePolicy``), the SGD optimizer to use from optax, learning rate, batch size, etc.
- ``[Training]`` specifies how long training should proceed, the ``epochs`` limits the total number of iterations, while ``train_seconds`` limits total training time

The configuration file can then be passed to the planner during initialization. For example, the following is a complete worked example for running any domain and user provided config file:

```python
import pyRDDLGym
from pyRDDLGym_jax.core.planner import load_config, JaxBackpropPlanner, JaxOfflineController

# set up the environment (NOTE: vectorized must be True for jax planner)
env = pyRDDLGym.make("domain", "instance", vectorized=True)
    
# load the config file with planner settings
abs_path = os.path.dirname(os.path.abspath(__file__))
planner_args, _, train_args = load_config("/path/to/config")
    
# create the planning algorithm
planner = JaxBackpropPlanner(rddl=env.model, **planner_args)
    
# create the controller   
controller = JaxOfflineController(planner, **train_args)
controller.evaluate(env, verbose=True, render=True)
```

## Using the JAX Compiler as a Simulator

The JAX compiler can be used as a backend for simulating and evaluating RDDL environments, instead of the usual pyRDDLGym one:

```python
import pyRDDLGym
from pyRDDLGym.core.policy import RandomAgent
from pyRDDLGym_jax.core.simulator import JaxRDDLSimulator

# create the environment
env = pyRDDLGym.make("domain", "instance", backend=JaxRDDLSimulator)

# evaluate the random policy
agent = RandomAgent(action_space=env.action_space,
                    num_actions=env.max_allowed_actions)
agent.evaluate(env, verbose=True, render=True)
```

For some domains, the JAX backend could perform better than the numpy-based one, due to various compiler optimizations. 
In any event, the simulation results using the JAX backend should match exactly those of the numpy-based backend.

