# pyRDDLGym-jax

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
[![PyPI Version](https://img.shields.io/pypi/v/pyRDDLGym-jax.svg)](https://pypi.org/project/pyRDDLGym-jax/)
[![Documentation Status](https://readthedocs.org/projects/pyrddlgym/badge/?version=latest)](https://pyrddlgym.readthedocs.io/en/latest/jax.html)
![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
[![Cumulative PyPI Downloads](https://img.shields.io/pypi/dm/pyrddlgym-jax)](https://pypistats.org/packages/pyrddlgym-jax)

[Installation](#installation) | [Run cmd](#running-from-the-command-line) | [Run python](#running-from-another-python-application) | [Configuration](#configuring-the-planner) | [Dashboard](#jaxplan-dashboard) | [Tuning](#tuning-the-planner) | [Simulation](#simulation) | [Citing](#citing-jaxplan)

**pyRDDLGym-jax (known in the literature as JaxPlan) is an efficient gradient-based/differentiable planning algorithm in JAX.** 

Purpose:

1. automatic translation of RDDL description files into differentiable JAX simulators
2. implementation of (highly configurable) operator relaxations for working in discrete and hybrid domains
3. flexible policy representations and automated Bayesian hyper-parameter tuning
4. interactive dashboard for dyanmic visualization and debugging
5. hybridization with parameter-exploring policy gradients.

Some demos of solved problems by JaxPlan:

<p align="middle">
<img src="https://github.com/pyrddlgym-project/pyRDDLGym-jax/blob/main/Images/intruders.gif" width="120" height="120" margin=0/>
<img src="https://github.com/pyrddlgym-project/pyRDDLGym-jax/blob/main/Images/marsrover.gif" width="120" height="120" margin=0/>
<img src="https://github.com/pyrddlgym-project/pyRDDLGym-jax/blob/main/Images/pong.gif" width="120" height="120" margin=0/>
<img src="https://github.com/pyrddlgym-project/pyRDDLGym-jax/blob/main/Images/quadcopter.gif" width="120" height="120" margin=0/>
<img src="https://github.com/pyrddlgym-project/pyRDDLGym-jax/blob/main/Images/reacher.gif" width="120" height="120" margin=0/>
<img src="https://github.com/pyrddlgym-project/pyRDDLGym-jax/blob/main/Images/reservoir.gif" width="120" height="120" margin=0/>
</p>

> [!WARNING]  
> Starting in version 1.0 (major release), the ``weight`` parameter in the config file was removed, 
and was moved to the individual logic components which have their own unique weight parameter assigned.
> Furthermore, the tuning module has been redesigned from the ground up, and supports tuning arbitrary hyper-parameters via config templates!
> Finally, the terrible visualizer for the planner was removed and replaced with an interactive real-time dashboard (similar to tensorboard, but custom designed for the planner)!

> [!NOTE]  
> While JaxPlan can support some discrete state/action problems through model relaxations, on some discrete problems it can perform poorly (though there is an ongoing effort to remedy this!).
> If you find it is not making sufficient progress, check out the [PROST planner](https://github.com/pyrddlgym-project/pyRDDLGym-prost) (for discrete spaces) or the [deep reinforcement learning wrappers](https://github.com/pyrddlgym-project/pyRDDLGym-rl).
  
## Installation

To install the bare-bones version of JaxPlan with **minimum installation requirements**:

```shell
pip install pyRDDLGym-jax
```

To install JaxPlan with the **automatic hyper-parameter tuning** and rddlrepository:

```shell
pip install pyRDDLGym-jax[extra]
```

(Since version 1.0) To install JaxPlan with the **visualization dashboard**:

```shell
pip install pyRDDLGym-jax[dashboard]
```

(Since version 1.0) To install JaxPlan with **all options**:

```shell
pip install pyRDDLGym-jax[extra,dashboard]
```

## Running from the Command Line

A basic run script is provided to train JaxPlan on any RDDL problem:

```shell
jaxplan plan <domain> <instance> <method> <episodes>
```

where:
- ``domain`` is the domain identifier as specified in rddlrepository (i.e. Wildfire_MDP_ippc2014), or a path pointing to a valid ``domain.rddl`` file
- ``instance`` is the instance identifier (i.e. 1, 2, ... 10), or a path pointing to a valid ``instance.rddl`` file
- ``method`` is the planning method to use (i.e. drp, slp, replan) or a path to a valid .cfg file (see section below)
- ``episodes`` is the (optional) number of episodes to evaluate the learned policy.

The ``method`` parameter supports four possible modes:
- ``slp`` is the basic straight line planner described [in this paper](https://proceedings.neurips.cc/paper_files/paper/2017/file/98b17f068d5d9b7668e19fb8ae470841-Paper.pdf)
- ``drp`` is the deep reactive policy network described [in this paper](https://ojs.aaai.org/index.php/AAAI/article/view/4744)
- ``replan`` is the same as ``slp`` except the plan is recalculated at every decision time step
- any other argument is interpreted as a file path to a valid configuration file.
   
For example, the following will train JaxPlan on the Quadcopter domain with 4 drones (with default config):

```shell
jaxplan plan Quadcopter 1 slp
```

## Running from Another Python Application

To run JaxPlan from within a Python application, refer to the following example:

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

Here, we have used the straight-line controller, although you can configure the combination of planner and policy representation if you wish. 
All controllers are instances of pyRDDLGym's ``BaseAgent`` class, so they provide the ``evaluate()`` function to streamline interaction with the environment.
The ``**planner_args`` and ``**train_args`` are keyword argument parameters to pass during initialization, but we strongly recommend creating and loading a config file as discussed in the next section.

## Configuring the Planner

The simplest way to configure the planner is to write and pass a configuration file with the necessary [hyper-parameters](https://pyrddlgym.readthedocs.io/en/latest/jax.html#configuring-pyrddlgym-jax).
The basic structure of a configuration file is provided below for a straight-line planner:

```ini
[Model]
logic='FuzzyLogic'
comparison_kwargs={'weight': 20}
rounding_kwargs={'weight': 20}
control_kwargs={'weight': 20}

[Optimizer]
method='JaxStraightLinePlan'
method_kwargs={}
optimizer='rmsprop'
optimizer_kwargs={'learning_rate': 0.001}

[Training]
key=42
epochs=5000
train_seconds=30
```

The configuration file contains three sections:
- ``[Model]`` specifies the fuzzy logic operations used to relax discrete operations to differentiable approximations; the ``weight`` dictates the quality of the approximation,
and ``tnorm`` specifies the type of [fuzzy logic](https://en.wikipedia.org/wiki/T-norm_fuzzy_logics) for relacing logical operations in RDDL (e.g. ``ProductTNorm``, ``GodelTNorm``, ``LukasiewiczTNorm``)
- ``[Optimizer]`` generally specify the optimizer and plan settings; the ``method`` specifies the plan/policy representation (e.g. ``JaxStraightLinePlan``, ``JaxDeepReactivePolicy``), the gradient descent settings, learning rate, batch size, etc.
- ``[Training]`` specifies computation limits, such as total training time and number of iterations, and options for printing or visualizing information from the planner.

For a policy network approach, simply change the ``[Optimizer]`` settings like so:

```ini
...
[Optimizer]
method='JaxDeepReactivePolicy'
method_kwargs={'topology': [128, 64], 'activation': 'tanh'}
...
```

The configuration file must then be passed to the planner during initialization. 
For example, the [previous script here](#running-from-another-python-application) can be modified to set parameters from a config file:

```python
from pyRDDLGym_jax.core.planner import load_config

# load the config file with planner settings
planner_args, _, train_args = load_config("/path/to/config.cfg")
    
# create the planning algorithm
planner = JaxBackpropPlanner(rddl=env.model, **planner_args)
controller = JaxOfflineController(planner, **train_args)
...
```

## JaxPlan Dashboard

Since version 1.0, JaxPlan has an optional dashboard that allows keeping track of the planner performance across multiple runs, 
and visualization of the policy or model, and other useful debugging features.

<p align="middle">
<img src="https://github.com/pyrddlgym-project/pyRDDLGym-jax/blob/main/Images/dashboard.png" width="480" height="248" margin=0/>
</p>

To run the dashboard, add the following entry to your config file:

```ini
...
[Training]
dashboard=True
...
```

More documentation about this and other new features will be coming soon.

## Tuning the Planner

A basic run script is provided to run automatic Bayesian hyper-parameter tuning for the most sensitive parameters of JaxPlan:

```shell
jaxplan tune <domain> <instance> <method> <trials> <iters> <workers> <dashboard> <filepath>
```

where:
- ``domain`` is the domain identifier as specified in rddlrepository
- ``instance`` is the instance identifier
- ``method`` is the planning method to use (i.e. drp, slp, replan)
- ``trials`` is the (optional) number of trials/episodes to average in evaluating each hyper-parameter setting
- ``iters`` is the (optional) maximum number of iterations/evaluations of Bayesian optimization to perform
- ``workers`` is the (optional) number of parallel evaluations to be done at each iteration, e.g. the total evaluations = ``iters * workers``
- ``dashboard`` is whether the optimizations are tracked in the dashboard application
- ``filepath`` is the optional file path where a config file with the best hyper-parameter setting will be saved.

It is easy to tune a custom range of the planner's hyper-parameters efficiently. 
First create a config file template with patterns replacing concrete parameter values that you want to tune, e.g.:

```ini
[Model]
logic='FuzzyLogic'
comparison_kwargs={'weight': TUNABLE_WEIGHT}
rounding_kwargs={'weight': TUNABLE_WEIGHT}
control_kwargs={'weight': TUNABLE_WEIGHT}

[Optimizer]
method='JaxStraightLinePlan'
method_kwargs={}
optimizer='rmsprop'
optimizer_kwargs={'learning_rate': TUNABLE_LEARNING_RATE}

[Training]
train_seconds=30
print_summary=False
print_progress=False
train_on_reset=True
```

would allow to tune the sharpness of model relaxations, and the learning rate of the optimizer.

Next, you must link the patterns in the config with concrete hyper-parameter ranges the tuner will understand, and run the optimizer:

```python
import pyRDDLGym
from pyRDDLGym_jax.core.tuning import JaxParameterTuning, Hyperparameter

# set up the environment   
env = pyRDDLGym.make(domain, instance, vectorized=True)
    
# load the config file template with planner settings
with open('path/to/config.cfg', 'r') as file: 
    config_template = file.read() 
    
# tune weight from 10^-1 ... 10^5 and lr from 10^-5 ... 10^1
def power_10(x):
    return 10.0 ** x    

hyperparams = [Hyperparameter('TUNABLE_WEIGHT', -1., 5., power_10),
               Hyperparameter('TUNABLE_LEARNING_RATE', -5., 1., power_10)]
    
# build the tuner and tune
tuning = JaxParameterTuning(env=env,
                            config_template=config_template, hyperparams=hyperparams,
                            online=False, eval_trials=trials, num_workers=workers, gp_iters=iters)
tuning.tune(key=42, log_file='path/to/log.csv')
```
  

## Simulation

The JAX compiler can be used as a backend for simulating and evaluating RDDL environments:

```python
import pyRDDLGym
from pyRDDLGym.core.policy import RandomAgent
from pyRDDLGym_jax.core.simulator import JaxRDDLSimulator

# create the environment
env = pyRDDLGym.make("domain", "instance", backend=JaxRDDLSimulator)

# evaluate the random policy
agent = RandomAgent(action_space=env.action_space, num_actions=env.max_allowed_actions)
agent.evaluate(env, verbose=True, render=True)
```

For some domains, the JAX backend could perform better than the numpy-based one, due to various compiler optimizations. 
In any event, the simulation results using the JAX backend should (almost) always match the numpy backend.


## Citing JaxPlan

The [following citation](https://ojs.aaai.org/index.php/ICAPS/article/view/31480) describes the main ideas of JaxPlan. Please cite it if you found it useful:

```
@inproceedings{gimelfarb2024jaxplan,
    title={JaxPlan and GurobiPlan: Optimization Baselines for Replanning in Discrete and Mixed Discrete and Continuous Probabilistic Domains},
    author={Michael Gimelfarb and Ayal Taitler and Scott Sanner},
    booktitle={34th International Conference on Automated Planning and Scheduling},
    year={2024},
    url={https://openreview.net/forum?id=7IKtmUpLEH}
}
```

Some of the implementation details derive from the following literature, which you may wish to also cite in your research papers:
- [A Distributional Framework for Risk-Sensitive End-to-End Planning in Continuous MDPs, AAAI 2022](https://ojs.aaai.org/index.php/AAAI/article/view/21226)
- [Deep reactive policies for planning in stochastic nonlinear domains, AAAI 2019](https://ojs.aaai.org/index.php/AAAI/article/view/4744)
- [Stochastic Planning with Lifted Symbolic Trajectory Optimization, AAAI 2019](https://ojs.aaai.org/index.php/ICAPS/article/view/3467/3335)
- [Scalable planning with tensorflow for hybrid nonlinear domains, NeurIPS 2017](https://proceedings.neurips.cc/paper/2017/file/98b17f068d5d9b7668e19fb8ae470841-Paper.pdf)
- [Baseline-Free Sampling in Parameter Exploring Policy Gradients: Super Symmetric PGPE, ANN 2015](https://link.springer.com/chapter/10.1007/978-3-319-09903-3_13)

The model relaxations in JaxPlan are based on the following works:
- [Poisson Variational Autoencoder, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2024/file/4f3cb9576dc99d62b80726690453716f-Paper-Conference.pdf)
- [Analyzing Differentiable Fuzzy Logic Operators, AI 2022](https://www.sciencedirect.com/science/article/pii/S0004370221001533)
- [Learning with algorithmic supervision via continuous relaxations, NeurIPS 2021](https://proceedings.neurips.cc/paper_files/paper/2021/file/89ae0fe22c47d374bc9350ef99e01685-Paper.pdf)
- [Universally quantized neural compression, NeurIPS 2020](https://papers.nips.cc/paper_files/paper/2020/file/92049debbe566ca5782a3045cf300a3c-Paper.pdf)
- [Generalized Gumbel-Softmax Gradient Estimator for Generic Discrete Random Variables, 2020](https://arxiv.org/pdf/2003.01847)
- [Categorical Reparametrization with Gumbel-Softmax, ICLR 2017](https://openreview.net/pdf?id=rkE3y85ee)
