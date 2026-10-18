# ***********************************************************************
# JAXPLAN
#
# Author: Michael Gimelfarb
#
# REFERENCES:
#
# [1] Patton, Noah, Jihwan Jeong, Mike Gimelfarb, and Scott Sanner. "A Distributional
# Framework for Risk-Sensitive End-to-End Planning in Continuous MDPs." In Proceedings of
# the AAAI Conference on Artificial Intelligence, vol. 36, no. 9, pp. 9894-9901. 2022.
#
# ***********************************************************************


import traceback
from typing import Callable
from pyRDDLGym.core.debug.exception import RDDLNotImplementedError, raise_warning

import jax
import jax.numpy as jnp
try:
    import softjax as sj
except Exception:
    raise_warning('Failed to import softjax: defaulting to jax.numpy.', 'red')
    traceback.print_exc()
    sj = jnp


@jax.jit
def entropic_utility(returns: jnp.ndarray, beta: float) -> float:
    return -jnp.divide(
        jax.scipy.special.logsumexp(-beta * returns, b=1. / returns.size), beta)


@jax.jit
def mean_variance_utility(returns: jnp.ndarray, beta: float) -> float:
    return jnp.mean(returns) - 0.5 * beta * jnp.var(returns)


@jax.jit
def mean_deviation_utility(returns: jnp.ndarray, beta: float) -> float:
    return jnp.mean(returns) - 0.5 * beta * jnp.std(returns)


@jax.jit
def mean_semideviation_utility(returns: jnp.ndarray, beta: float) -> float:
    mu = jnp.mean(returns)
    msd = sj.sqrt(jnp.mean(jnp.square(jnp.minimum(0.0, returns - mu))))
    return mu - 0.5 * beta * msd


@jax.jit
def mean_semivariance_utility(returns: jnp.ndarray, beta: float) -> float:
    mu = jnp.mean(returns)
    msv = jnp.mean(jnp.square(jnp.minimum(0.0, returns - mu)))
    return mu - 0.5 * beta * msv


@jax.jit
def sharpe_utility(returns: jnp.ndarray, risk_free: float=0.0, eps: float=1e-12) -> float:
    return jnp.divide(jnp.mean(returns) - risk_free, jnp.std(returns) + eps)


@jax.jit
def var_utility(returns: jnp.ndarray, alpha: float, *args, **kwargs) -> float:
    return sj.percentile(returns, 100 * alpha, *args, **kwargs)


@jax.jit
def cvar_utility(returns: jnp.ndarray, alpha: float, *args, **kwargs) -> float:
    var = sj.percentile(returns, 100 * alpha, *args, **kwargs)
    return var - jnp.divide(jnp.mean(jax.nn.relu(var - returns)), alpha)


@jax.jit
def cvar_ste_utility(returns: jnp.ndarray, alpha: float) -> float:
    mask = returns <= jnp.percentile(returns, 100 * alpha)
    return jnp.divide(jnp.sum(mask * returns), jnp.maximum(1, jnp.count_nonzero(mask)))


@jax.jit(static_argnames=['max_iter'])
def expectile_utility(returns: jnp.ndarray, alpha: float, max_iter: int=30) -> float:
    def scan_body(current_val, _):
        weights = jnp.where(returns > current_val, alpha, 1.0 - alpha)
        next_val = jnp.sum(weights * returns) / jnp.sum(weights)
        return next_val, None
    final_val, _ = jax.lax.scan(scan_body, jnp.mean(returns), xs=None, length=max_iter)
    return final_val


@jax.jit
def gini_utility(returns: jnp.ndarray, gamma: float) -> float:
    mean_return = jnp.mean(returns)
    diffs = returns[:, jnp.newaxis] - returns[jnp.newaxis, :]
    downside_diffs = jax.nn.relu(-diffs)
    gini_dispersion = 2.0 * jnp.mean(downside_diffs)
    return mean_return - gamma * gini_dispersion


UTILITY_LOOKUP = {
    'mean': jnp.mean,
    'mean_var': mean_variance_utility,
    'mean_std': mean_deviation_utility,
    'mean_semivar': mean_semivariance_utility,
    'mean_semidev': mean_semideviation_utility,
    'sharpe': sharpe_utility,
    'entropic': entropic_utility,
    'exponential': entropic_utility,
    'var': var_utility,
    'cvar': cvar_utility,
    'cvar_ste': cvar_ste_utility,
    'expectile': expectile_utility,
    'gini': gini_utility,
}


def create_utility(name: str) -> Callable:
    '''Returns the built-in utility function with the given name (case insensitive).'''
    utility_fn = UTILITY_LOOKUP.get(name.lower(), None)
    if utility_fn is None:
        raise RDDLNotImplementedError(
            f'Utility function <{name}> is not supported, '
            f'must be one of {list(UTILITY_LOOKUP.keys())}.'
        )
    return utility_fn
