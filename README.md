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

```
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

## Running the basic examples

A basic run script is provided to run the Jax Planner on any domain in ``rddlrepository``, provided a config file is available (currently, only a limited subset of configs are provided as examples).
The example can be run as follows in a standard shell:

```
python -m pyRDDLGym_jax.examples.run_plan <domain> <instance> <method> [<episodes>]
```

where:
- ``domain`` is the domain identifier as specified in rddlrepository (i.e. Wildfire_MDP_ippc2014)
- ``instance`` is the instance identifier (i.e. 1, 2, ... 10)
- ``method`` is the planning method to use (i.e. drp, slp, replan)
- ``episodes`` is the number of episodes to evaluate the learned policy

A basic run script is also provided to run the automatic hyper-parameter tuning. The structure of this stript is similar to the one above

```
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

```
python -m pyRDDLGym_jax.examples.run_plan Quadcopter 1 slp
```

After several minutes of optimization, you should get a visualization as follows:

<p align="center">
<img src="Images/quadcopter.gif" width="150" height="150" margin=0/>
</p>
