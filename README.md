# pyRDDLGym-jax

Author: [Mike Gimelfarb](https://mike-gimelfarb.github.io)

This directory provides:
1. automated translation and compilation of RDDL description files into the [JAX](https://github.com/google/jax) auto-diff library, which allows any RDDL domain to be converted to a differentiable simulator!
2. powerful, fast, and very scalable gradient-based planning algorithms, with extendible and flexible policy class representations, automatic model relaxations for working in discrete and hybrid domains, and much more!

> [!NOTE]  
> While Jax planners can support some discrete state/action problems through model relaxations, on some discrete problems it can perform poorly (though there is an ongoing effort to remedy this!).
> If you find it is not making sufficient progress, check out the [PROST planner](https://github.com/pyrddlgym-project/pyRDDLGym-prost) (for discrete spaces) or the [deep reinforcement learning wrappers](https://github.com/pyrddlgym-project/pyRDDLGym-rl).

## Contents

- [Installation](#installation)
- [Running the Basic Examples](#running-the-basic-examples)
- [Running from the Python API](#running-from-the-python-api)
- [Writing a Configuration File for a Custom Domain](#writing-a-configuration-file-for-a-custom-domain)
- [Changing the pyRDDLGym Simulation Backend to JAX](#changing-the-pyrddlgym-simulation-backend-to-jax)
- [Computing the Model Gradients Manually](#computing-the-model-gradients-manually)
- [Citing pyRDDLGym-jax](#citing-pyrddlgym-jax)
  
## Installation

To use the compiler or planner without the automated hyper-parameter tuning, you will need the following packages installed: 
- ``pyRDDLGym>=2.0``
- ``tqdm>=4.66``
- ``jax>=0.4.12``
- ``optax>=0.1.9``
- ``dm-haiku>=0.0.10``
- ``tensorflow>=2.13.0``
- ``tensorflow-probability>=0.21.0``

Additionally, if you wish to run the examples, you need ``rddlrepository>=2``, and run the automated tuning optimization, you will also need ``bayesian-optimization>=1.4.3``.

You can install this package, together with all of its requirements via pip:

```shell
pip install rddlrepository pyRDDLGym-jax
```

## Running the Basic Examples

A basic run script is provided to run the Jax Planner on any domain in ``rddlrepository``, provided a config file is available (currently, only a limited subset of configs are provided as examples).
The example can be run as follows in a standard shell, from the install directory of pyRDDLGym-jax:

```shell
python -m pyRDDLGym_jax.examples.run_plan <domain> <instance> <method> <episodes>
```

where:
- ``domain`` is the domain identifier as specified in rddlrepository (i.e. Wildfire_MDP_ippc2014), or a path pointing to a valid ``domain.rddl`` file
- ``instance`` is the instance identifier (i.e. 1, 2, ... 10), or a path pointing to a valid ``instance.rddl`` file
- ``method`` is the planning method to use (i.e. drp, slp, replan)
- ``episodes`` is the (optional) number of episodes to evaluate the learned policy

The ``method`` parameter warrants further explanation. Currently we support three possible modes:
- ``slp`` is the basic straight line planner described [in this paper](https://proceedings.neurips.cc/paper_files/paper/2017/file/98b17f068d5d9b7668e19fb8ae470841-Paper.pdf)
- ``drp`` is the deep reactive policy network described [in this paper](https://ojs.aaai.org/index.php/AAAI/article/view/4744)
- ``replan`` is the same as ``slp`` except the plan is recalculated at every decision time step.
   
A basic run script is also provided to run the automatic hyper-parameter tuning. The structure of this stript is similar to the one above

```shell
python -m pyRDDLGym_jax.examples.run_tune <domain> <instance> <method> <trials> <iters> <workers>
```

where:
- ``domain`` is the domain identifier as specified in rddlrepository (i.e. Wildfire_MDP_ippc2014)
- ``instance`` is the instance identifier (i.e. 1, 2, ... 10)
- ``method`` is the planning method to use (i.e. drp, slp, replan)
- ``trials`` is the (optional) number of trials/episodes to average in evaluating each hyper-parameter setting
- ``iters`` is the (optional) maximum number of iterations/evaluations of Bayesian optimization to perform
- ``workers`` is the (optional) number of parallel evaluations to be done at each iteration, e.g. the total evaluations = ``iters * workers``

For example, copy and pasting the following will train the Jax Planner on the Quadcopter domain with 4 drones:

```shell
python -m pyRDDLGym_jax.examples.run_plan Quadcopter 1 slp
```

After several minutes of optimization, you should get a visualization as follows:

<p align="center">
<img src="Images/quadcopter.gif" width="400" height="400" margin=1/>
</p>

## Running from the Python API

You can write a simple script to instantiate and run the planner yourself, if you wish:

```python
import pyRDDLGym
from pyRDDLGym_jax.core.planner import JaxBackpropPlanner, JaxOfflineController

# set up the environment (note the vectorized option must be True)
env = pyRDDLGym.make("domain", "instance", vectorized=True)

# create the planning algorithm
planner = JaxBackpropPlanner(rddl=env.model, **planner_args)
controller = JaxOfflineController(planner, **train_args)

# evaluate the planner
controller.evaluate(env, episodes=1, verbose=True, render=True)

env.close()
```

Here, we have used the straight-line (offline) controller, although you can configure the combination of planner and policy representation. 
All controllers are instances of pyRDDLGym's ``BaseAgent`` class, so they support the ``evaluate()`` function to streamline interaction with the environment.
The ``**planner_args`` and ``**train_args`` are keyword argument parameters to pass during initialization, but we strongly recommend creating and loading a config file as discussed in the next section.

## Writing a Configuration File for a Custom Domain

The simplest way to interface with the Planner for solving a custom problem is to write a configuration file with all the necessary hyper-parameters.
The basic structure of a configuration file is provided below for a straight line planning/MPC style planner:

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

For a policy network approach, simply change the configuration file like so:

```ini
...
[Optimizer]
method='JaxDeepReactivePolicy'
method_kwargs={'topology': [128, 64]}
...
```

which would create a policy network with two hidden layers and ReLU activations.

The configuration file can then be passed to the planner during initialization. For example, the [previous script here](#running-from-the-python-api) can be modified to set parameters from a config file as follows:

```python
from pyRDDLGym_jax.core.planner import load_config

# load the config file with planner settings
planner_args, _, train_args = load_config("/path/to/config.cfg")
    
# create the planning algorithm
planner = JaxBackpropPlanner(rddl=env.model, **planner_args)
controller = JaxOfflineController(planner, **train_args)
...
```

## Changing the pyRDDLGym Simulation Backend to JAX

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

## Computing the Model Gradients Manually

For custom applications, it is desirable to compute gradients of the model that can be optimized downstream. Fortunately, we provide a very convenient function for compiling the transition/step function ``P(s, a, s')`` of the environment into JAX.

```python
import pyRDDLGym
from pyRDDLGym_jax.core.planner import JaxRDDLCompilerWithGrad

# set up the environment
env = pyRDDLGym.make("domain", "instance", vectorized=True)

# create the step function
compiled = JaxRDDLCompilerWithGrad(rddl=env.model)
compiled.compile()
step_fn = compiled.compile_transition()
```

This will return a JAX compiled (pure) function that requires 4 arguments:
- ``key`` is the ``jax.random.PRNGKey`` key for reproducible randomness
- ``actions`` is the dictionary of action fluent JAX tensors
- ``subs`` is the dictionary of state-fluent and non-fluent JAX tensors
- ``model_params`` are the parameters of the differentiable relaxations, such as ``weight``
- 
The function returns a dictionary containing a variety of variables, such as updated pvariables including next-state fluents (``pvar``), reward obtained (``reward``), error codes (``error``).
It is thus possible to apply any JAX transformation to the output of the function, such as computing gradient using ``jax.grad()`` or batched simulation using ``jax.vmap()``.

Compilation of entire rollouts is possible by calling the ``compile_rollouts`` function, and providing a policy implementation that maps states (jax tensors) and tunable policy parameters to actions.
An [example is provided to illustrate how you can define your own policy class and compute the return gradient manually](https://github.com/pyrddlgym-project/pyRDDLGym-jax/blob/main/pyRDDLGym_jax/examples/run_gradient.py).

## Citing pyRDDLGym-jax

The following citation describes the main ideas of the framework. Please cite it if you found it useful:

```
@inproceedings{gimelfarb2024jaxplan,
    title={JaxPlan and GurobiPlan: Optimization Baselines for Replanning in Discrete and Mixed Discrete and Continuous Probabilistic Domains},
    author={Michael Gimelfarb and Ayal Taitler and Scott Sanner},
    booktitle={34th International Conference on Automated Planning and Scheduling},
    year={2024},
    url={https://openreview.net/forum?id=7IKtmUpLEH}
}
```

The utility optimization is discussed in this paper:

```
@inproceedings{patton2022distributional,
    title={A distributional framework for risk-sensitive end-to-end planning in continuous mdps},
    author={Patton, Noah and Jeong, Jihwan and Gimelfarb, Mike and Sanner, Scott},
    booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
    volume={36},
    number={9},
    pages={9894--9901},
    year={2022}
}
```

Some of the implementation details derive from the following literature, which you may wish to also cite in your research papers:
- [Deep reactive policies for planning in stochastic nonlinear domains, AAAI 2019](https://ojs.aaai.org/index.php/AAAI/article/view/4744)
- [Scalable planning with tensorflow for hybrid nonlinear domains, NeurIPS 2017](https://proceedings.neurips.cc/paper/2017/file/98b17f068d5d9b7668e19fb8ae470841-Paper.pdf)

