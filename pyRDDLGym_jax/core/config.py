# ***********************************************************************
# JAXPLAN
# 
# Author: Michael Gimelfarb
#
# CONFIG FILE MANAGEMENT:
# - read config files from file path
# - extract experiment settings
# - instantiate planner
#
# ***********************************************************************


from ast import literal_eval
import configparser
import os
import traceback
from typing import Any, Dict, Tuple

import jax
import jax.nn.initializers as initializers
import jax.random as random
import flax.linen as nn
import optax

from pyRDDLGym.core.debug.exception import raise_warning

from pyRDDLGym_jax.core import logic, plan, planner

# try to load the dash board
try:
    from pyRDDLGym_jax.core.visualization import JaxPlannerDashboard
except Exception:
    raise_warning('Failed to load the dashboard visualization tool: '
                  'ensure all prerequisite packages are installed.', 'red')
    traceback.print_exc()
    JaxPlannerDashboard = None

Kwargs = Dict[str, Any]


def _parse_config_file(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f'File {path} does not exist.')
    config = configparser.RawConfigParser()
    config.optionxform = str 
    config.read(path)
    args = {section: {k: literal_eval(v) for (k, v) in config.items(section)}
            for section in config.sections()}
    return config, args


def _parse_config_string(value: str):
    config = configparser.RawConfigParser()
    config.optionxform = str 
    config.read_string(value)
    args = {section: {k: literal_eval(v) for (k, v) in config.items(section)}
            for section in config.sections()}
    return config, args


def _getattr_any(packages, item):
    for package in packages:
        loaded = getattr(package, item, None)
        if loaded is not None:
            return loaded
    return None


def _load_config(config, args):
    compiler_kwargs = {k: args['Compiler'][k] for (k, _) in config.items('Compiler')}
    planner_args = {k: args['Planner'][k] for (k, _) in config.items('Planner')}
    train_args = {k: args['Optimize'][k] for (k, _) in config.items('Optimize')}    
    
    # read the compiler settings
    compiler_name = compiler_kwargs.pop('method', 'DefaultJaxRDDLCompilerWithGrad')
    planner_args['compiler'] = getattr(logic, compiler_name)
    planner_args['compiler_kwargs'] = compiler_kwargs

    # read the policy settings
    plan_method = planner_args.pop('method')
    plan_kwargs = planner_args.pop('method_kwargs', {})  
    
    # policy initialization
    plan_initializer = plan_kwargs.get('initializer', None)
    if plan_initializer is not None:
        initializer = _getattr_any(
            packages=[initializers, nn.initializers], item=plan_initializer)
        if initializer is None:
            raise ValueError(f'Invalid initializer <{plan_initializer}>.')
        else:
            init_kwargs = plan_kwargs.pop('initializer_kwargs', {})
            try: 
                plan_kwargs['initializer'] = initializer(**init_kwargs)
            except Exception as _:
                raise ValueError(f'Invalid initializer kwargs <{init_kwargs}>.')
    
    # policy activation
    plan_activation = plan_kwargs.get('activation', None)
    if plan_activation is not None:
        activation = _getattr_any(packages=[jax.nn, jax.numpy], item=plan_activation)
        if activation is None:
            raise ValueError(f'Invalid activation <{plan_activation}>.')
        else:
            plan_kwargs['activation'] = activation
    
    # policy embedding
    plan_time_embed = plan_kwargs.get('time_embedding', None)
    if plan_time_embed is not None:
        plan_kwargs['time_embedding'] = getattr(plan, plan_time_embed)

    # read the planner settings
    planner_args['plan'] = getattr(plan, plan_method)(**plan_kwargs)

    # planner class (see create_planner)
    planner_method = planner_args.get('planner', None)
    if planner_method is not None:
        planner_class = getattr(planner, planner_method, None)
        if not (isinstance(planner_class, type) and issubclass(planner_class, planner.JaxPlanner)):
            raise ValueError(f'Invalid planner <{planner_method}>.')
        planner_args['planner'] = planner_class

    # planner optimizer
    planner_optimizer = planner_args.get('optimizer', None)
    if planner_optimizer is not None:
        optimizer = _getattr_any(packages=[optax], item=planner_optimizer)
        if optimizer is None:
            raise ValueError(f'Invalid optimizer <{planner_optimizer}>.')
        else:
            planner_args['optimizer'] = optimizer

    # auxiliary planner: if not specified the default of the planner class is used
    aux_kwargs = planner_args.pop('auxiliary_kwargs', {})
    if 'auxiliary' not in planner_args:
        if aux_kwargs:
            raise ValueError('auxiliary_kwargs was provided without auxiliary.')
    elif planner_args['auxiliary'] is not None:
        aux_method = planner_args['auxiliary']
        if 'optimizer' in aux_kwargs:
            aux_optimizer = _getattr_any(packages=[optax], item=aux_kwargs['optimizer'])
            if aux_optimizer is None:
                raise ValueError(f'Invalid optimizer <{aux_kwargs["optimizer"]}>.')
            else:
                aux_kwargs['optimizer'] = aux_optimizer
        planner_args['auxiliary'] = getattr(planner, aux_method)(**aux_kwargs)

    # preprocessor settings
    preproc_method = planner_args.get('preprocessor', None)
    preproc_kwargs = planner_args.pop('preprocessor_kwargs', {})
    if preproc_method is not None:
        planner_args['preprocessor'] = getattr(
            plan, preproc_method)(**preproc_kwargs)

    # optimize call RNG key
    planner_key = train_args.get('key', None)
    if planner_key is not None:
        train_args['key'] = random.PRNGKey(planner_key)
    
    # dashboard
    dashboard_key = planner_args.get('dashboard', None)
    if dashboard_key is not None and dashboard_key and JaxPlannerDashboard is not None:
        planner_args['dashboard'] = JaxPlannerDashboard()
    elif dashboard_key is not None:
        del planner_args['dashboard']        
    
    # optimize call stopping rule
    stopping_rule = train_args.get('stopping_rule', None)
    if stopping_rule is not None:
        stopping_rule_kwargs = train_args.pop('stopping_rule_kwargs', {})
        train_args['stopping_rule'] = getattr(
            planner, stopping_rule)(**stopping_rule_kwargs)
    
    return planner_args, plan_kwargs, train_args


def load_config(path: str) -> Tuple[Kwargs, ...]:
    '''Loads a config file at the specified file path.
    
    :param path: the path of the config file to load and parse
    '''
    config, args = _parse_config_file(path)
    return _load_config(config, args)


def load_config_from_string(value: str) -> Tuple[Kwargs, ...]:
    '''Loads config file contents specified explicitly as a string value.
    
    :param value: the string in json format containing the config contents to parse
    '''
    config, args = _parse_config_string(value)
    return _load_config(config, args)
