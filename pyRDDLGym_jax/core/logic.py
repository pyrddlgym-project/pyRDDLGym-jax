# ***********************************************************************
# JAXPLAN
# 
# Author: Michael Gimelfarb
#
# REFERENCES:
#
# [1] Gimelfarb, Michael, Ayal Taitler, and Scott Sanner. "JaxPlan and GurobiPlan: 
# Optimization Baselines for Replanning in Discrete and Mixed Discrete-Continuous 
# Probabilistic Domains." Proceedings of the International Conference on Automated 
# Planning and Scheduling. Vol. 34. 2024.
#
# [2] Petersen, Felix, Christian Borgelt, Hilde Kuehne, and Oliver Deussen. "Learning with 
# algorithmic supervision via continuous relaxations." Advances in Neural Information 
# Processing Systems 34 (2021): 16520-16531.
#
# [3] Agustsson, Eirikur, and Lucas Theis. "Universally quantized neural compression." 
# Advances in neural information processing systems 33 (2020): 12367-12376.
#
# [4] Gupta, Madan M., and J11043360726 Qi. "Theory of T-norms and fuzzy inference 
# methods." Fuzzy sets and systems 40, no. 3 (1991): 431-450.
#
# [5] Jang, Eric, Shixiang Gu, and Ben Poole. "Categorical Reparametrization with 
# Gumble-Softmax." In International Conference on Learning Representations (ICLR 2017). 
# OpenReview. net, 2017.
#
# [6] Vafaii, H., Galor, D., & Yates, J. (2025). Poisson Variational Autoencoder. 
# Advances in Neural Information Processing Systems, 37, 44871-44906.
#
# ***********************************************************************


from abc import ABCMeta, abstractmethod
import traceback
from typing import Callable, Dict, Tuple, Union

import jax
import jax.numpy as jnp
import jax.random as random
import jax.scipy as scipy 

from pyRDDLGym.core.debug.exception import raise_warning

# more robust approach - if user does not have this or broken try to continue
try:
    from tensorflow_probability.substrates import jax as tfp
except Exception:
    raise_warning('Failed to import tensorflow-probability: '
                  'compilation of some probability distributions will fail.', 'red')
    traceback.print_exc()
    tfp = None


def enumerate_literals(shape: Tuple[int, ...], axis: int, dtype: type=jnp.int32) -> jnp.ndarray:
    literals = jnp.arange(shape[axis], dtype=dtype)
    literals = literals[(...,) + (jnp.newaxis,) * (len(shape) - 1)]
    literals = jnp.moveaxis(literals, source=0, destination=axis)
    literals = jnp.broadcast_to(literals, shape=shape)
    return literals


# ===========================================================================
# RELATIONAL OPERATIONS
# - abstract class
# - sigmoid comparison
#
# ===========================================================================

class Comparison(metaclass=ABCMeta):
    '''Base class for approximate comparison operations.'''
    
    @abstractmethod
    def greater_equal(self, id, init_params):
        pass
    
    @abstractmethod
    def greater(self, id, init_params):
        pass
    
    @abstractmethod
    def equal(self, id, init_params):
        pass
    
    @abstractmethod
    def sgn(self, id, init_params):
        pass
    
    @abstractmethod
    def argmax(self, id, init_params):
        pass
    

class SigmoidComparison(Comparison):
    '''Comparison operations approximated using sigmoid functions.'''
    
    def __init__(self, weight: float=10.0) -> None:
        self.weight = float(weight)
        
    # https://arxiv.org/abs/2110.05651
    def greater_equal(self, id, init_params):
        id_ = str(id)
        init_params[id_] = self.weight
        def _jax_wrapped_calc_greater_equal_approx(x, y, params):
            gre_eq = jax.nn.sigmoid(params[id_] * (x - y))
            return gre_eq, params
        return _jax_wrapped_calc_greater_equal_approx
    
    def greater(self, id, init_params):
        return self.greater_equal(id, init_params)
    
    def equal(self, id, init_params):
        id_ = str(id)
        init_params[id_] = self.weight
        def _jax_wrapped_calc_equal_approx(x, y, params):
            equal = 1.0 - jnp.square(jnp.tanh(params[id_] * (y - x)))
            return equal, params
        return _jax_wrapped_calc_equal_approx
    
    def sgn(self, id, init_params):
        id_ = str(id)
        init_params[id_] = self.weight
        def _jax_wrapped_calc_sgn_approx(x, params):
            sgn = jnp.tanh(params[id_] * x)
            return sgn, params
        return _jax_wrapped_calc_sgn_approx
    
    # https://arxiv.org/abs/2110.05651
    def argmax(self, id, init_params):
        id_ = str(id)
        init_params[id_] = self.weight
        def _jax_wrapped_calc_argmax_approx(x, axis, params):
            literals = enumerate_literals(jnp.shape(x), axis=axis)
            softmax = jax.nn.softmax(params[id_] * x, axis=axis)
            sample = jnp.sum(literals * softmax, axis=axis)
            return sample, params
        return _jax_wrapped_calc_argmax_approx
    
    def __str__(self) -> str:
        return f'Sigmoid comparison with weight {self.weight}'


# ===========================================================================
# ROUNDING OPERATIONS
# - abstract class
# - soft rounding
#
# ===========================================================================

class Rounding(metaclass=ABCMeta):
    '''Base class for approximate rounding operations.'''
    
    @abstractmethod
    def floor(self, id, init_params):
        pass
    
    @abstractmethod
    def round(self, id, init_params):
        pass


class SoftRounding(Rounding):
    '''Rounding operations approximated using soft operations.'''
    
    def __init__(self, weight: float=10.0) -> None:
        self.weight = float(weight)
        
    # https://www.tensorflow.org/probability/api_docs/python/tfp/substrates/jax/bijectors/Softfloor
    def floor(self, id, init_params):
        id_ = str(id)
        init_params[id_] = self.weight
        def _jax_wrapped_calc_floor_approx(x, params):
            param = params[id_]
            denom = jnp.tanh(param / 4.0)
            floor = (jax.nn.sigmoid(param * (x - jnp.floor(x) - 1.0)) - 
                     jax.nn.sigmoid(-param / 2.0)) / denom + jnp.floor(x)
            return floor, params
        return _jax_wrapped_calc_floor_approx
    
    # https://arxiv.org/abs/2006.09952
    def round(self, id, init_params):
        id_ = str(id)
        init_params[id_] = self.weight
        def _jax_wrapped_calc_round_approx(x, params):
            param = params[id_]
            m = jnp.floor(x) + 0.5
            rounded = m + 0.5 * jnp.tanh(param * (x - m)) / jnp.tanh(param / 2.0)
            return rounded, params
        return _jax_wrapped_calc_round_approx

    def __str__(self) -> str:
        return f'SoftFloor and SoftRound with weight {self.weight}'


# ===========================================================================
# LOGICAL COMPLEMENT
# - abstract class
# - standard complement
#
# ===========================================================================

class Complement(metaclass=ABCMeta):
    '''Base class for approximate logical complement operations.'''
    
    @abstractmethod
    def __call__(self, id, init_params):
        pass


class StandardComplement(Complement):
    '''The standard approximate logical complement given by x -> 1 - x.'''
    
    # https://www.sciencedirect.com/science/article/abs/pii/016501149190171L
    @staticmethod
    def _jax_wrapped_calc_not_approx(x, params):
        return 1.0 - x, params   
    
    def __call__(self, id, init_params):
        return self._jax_wrapped_calc_not_approx
    
    def __str__(self) -> str:
        return 'Standard complement'
    

# ===========================================================================
# TNORMS
# - abstract tnorm
# - product tnorm
# - Godel tnorm
# - Lukasiewicz tnorm
# - Yager(p) tnorm
#
# https://www.sciencedirect.com/science/article/abs/pii/016501149190171L
# ===========================================================================

class TNorm(metaclass=ABCMeta):
    '''Base class for fuzzy differentiable t-norms.'''
    
    @abstractmethod
    def norm(self, id, init_params):
        '''Elementwise t-norm of x and y.'''
        pass
    
    @abstractmethod
    def norms(self, id, init_params):
        '''T-norm computed for tensor x along axis.'''
        pass
    

class ProductTNorm(TNorm):
    '''Product t-norm given by the expression (x, y) -> x * y.'''
    
    @staticmethod
    def _jax_wrapped_calc_and_approx(x, y, params):
        return x * y, params    
        
    def norm(self, id, init_params):        
        return self._jax_wrapped_calc_and_approx
    
    @staticmethod
    def _jax_wrapped_calc_forall_approx(x, axis, params):
        return jnp.prod(x, axis=axis), params   
        
    def norms(self, id, init_params):        
        return self._jax_wrapped_calc_forall_approx

    def __str__(self) -> str:
        return 'Product t-norm'
    

class GodelTNorm(TNorm):
    '''Godel t-norm given by the expression (x, y) -> min(x, y).'''
    
    @staticmethod
    def _jax_wrapped_calc_and_approx(x, y, params):
        return jnp.minimum(x, y), params
    
    def norm(self, id, init_params):        
        return self._jax_wrapped_calc_and_approx
    
    @staticmethod
    def _jax_wrapped_calc_forall_approx(x, axis, params):
        return jnp.min(x, axis=axis), params   
        
    def norms(self, id, init_params):        
        return self._jax_wrapped_calc_forall_approx
    
    def __str__(self) -> str:
        return 'Godel t-norm'
    

class LukasiewiczTNorm(TNorm):
    '''Lukasiewicz t-norm given by the expression (x, y) -> max(x + y - 1, 0).'''
    
    @staticmethod
    def _jax_wrapped_calc_and_approx(x, y, params):
        land = jax.nn.relu(x + y - 1.0)
        return land, params
        
    def norm(self, id, init_params):
        return self._jax_wrapped_calc_and_approx
    
    @staticmethod
    def _jax_wrapped_calc_forall_approx(x, axis, params):
        forall = jax.nn.relu(jnp.sum(x - 1.0, axis=axis) + 1.0)
        return forall, params
        
    def norms(self, id, init_params):
        return self._jax_wrapped_calc_forall_approx
    
    def __str__(self) -> str:
        return 'Lukasiewicz t-norm'


class YagerTNorm(TNorm):
    '''Yager t-norm given by the expression 
    (x, y) -> max(1 - ((1 - x)^p + (1 - y)^p)^(1/p)).'''
    
    def __init__(self, p: float=2.0) -> None:
        self.p = float(p)
    
    def norm(self, id, init_params):
        id_ = str(id)
        init_params[id_] = self.p
        def _jax_wrapped_calc_and_approx(x, y, params):
            base = jax.nn.relu(1.0 - jnp.stack([x, y], axis=0))
            arg = jnp.linalg.norm(base, ord=params[id_], axis=0)
            land = jax.nn.relu(1.0 - arg)
            return land, params
        return _jax_wrapped_calc_and_approx
    
    def norms(self, id, init_params):
        id_ = str(id)
        init_params[id_] = self.p
        def _jax_wrapped_calc_forall_approx(x, axis, params):
            arg = jax.nn.relu(1.0 - x)
            for ax in sorted(axis, reverse=True):
                arg = jnp.linalg.norm(arg, ord=params[id_], axis=ax)
            forall = jax.nn.relu(1.0 - arg)
            return forall, params
        return _jax_wrapped_calc_forall_approx
    
    def __str__(self) -> str:
        return f'Yager({self.p}) t-norm'
    

# ===========================================================================
# RANDOM SAMPLING
# - abstract sampler
# - Gumbel-softmax sampler
# - determinization
#
# ===========================================================================

class RandomSampling(metaclass=ABCMeta):
    '''Describes how non-reparameterizable random variables are sampled.'''
    
    @abstractmethod
    def discrete(self, id, init_params, logic):
        pass
    
    @abstractmethod
    def poisson(self, id, init_params, logic):
        pass
    
    @abstractmethod
    def binomial(self, id, init_params, logic):
        pass
    
    @abstractmethod
    def negative_binomial(self, id, init_params, logic):
        pass
    
    @abstractmethod
    def geometric(self, id, init_params, logic):
        pass
    
    @abstractmethod
    def bernoulli(self, id, init_params, logic):
        pass
    
    def __str__(self) -> str:
        return 'RandomSampling'


class SoftRandomSampling(RandomSampling):
    '''Random sampling of discrete variables using Gumbel-softmax trick.'''
    
    def __init__(self, poisson_max_bins: int=100, 
                 poisson_min_cdf: float=0.999,
                 poisson_exp_sampling: bool=True,
                 binomial_max_bins: int=100,
                 bernoulli_gumbel_softmax: bool=False) -> None:
        '''Creates a new instance of soft random sampling.

        :param poisson_max_bins: maximum bins to use for Poisson distribution relaxation
        :param poisson_min_cdf: minimum cdf value of Poisson within truncated region
        in order to use Poisson relaxation
        :param poisson_exp_sampling: whether to use Poisson process sampling method
        instead of truncated Gumbel-Softmax
        :param binomial_max_bins: maximum bins to use for Binomial distribution relaxation
        :param bernoulli_gumbel_softmax: whether to use Gumbel-Softmax to approximate
        Bernoulli samples, or the standard uniform reparameterization instead
        '''
        self.poisson_bins = poisson_max_bins
        self.poisson_min_cdf = poisson_min_cdf
        self.poisson_exp_method = poisson_exp_sampling
        self.binomial_bins = binomial_max_bins
        self.bernoulli_gumbel_softmax = bernoulli_gumbel_softmax

    # https://arxiv.org/pdf/1611.01144
    def discrete(self, id, init_params, logic):
        argmax_approx = logic.argmax(id, init_params)
        def _jax_wrapped_calc_discrete_gumbel_softmax(key, prob, params):
            Gumbel01 = random.gumbel(key=key, shape=jnp.shape(prob), dtype=logic.REAL)
            sample = Gumbel01 + jnp.log(prob + logic.eps)
            return argmax_approx(sample, axis=-1, params=params)
        return _jax_wrapped_calc_discrete_gumbel_softmax
    
    def _poisson_gumbel_softmax(self, id, init_params, logic):        
        argmax_approx = logic.argmax(id, init_params)
        def _jax_wrapped_calc_poisson_gumbel_softmax(key, rate, params):
            ks = jnp.arange(self.poisson_bins)[(jnp.newaxis,) * jnp.ndim(rate) + (...,)]
            rate = rate[..., jnp.newaxis]
            log_prob = ks * jnp.log(rate + logic.eps) - rate - scipy.special.gammaln(ks + 1)
            Gumbel01 = random.gumbel(key=key, shape=jnp.shape(log_prob), dtype=logic.REAL)
            sample = Gumbel01 + log_prob
            return argmax_approx(sample, axis=-1, params=params)
        return _jax_wrapped_calc_poisson_gumbel_softmax
    
    # https://arxiv.org/abs/2405.14473
    def _poisson_exponential(self, id, init_params, logic):
        less_approx = logic.less(id, init_params)
        def _jax_wrapped_calc_poisson_exponential(key, rate, params):
            Exp1 = random.exponential(
                key=key,  shape=(self.poisson_bins,) + jnp.shape(rate), dtype=logic.REAL)
            delta_t = Exp1 / rate[jnp.newaxis, ...]
            times = jnp.cumsum(delta_t, axis=0)
            indicator, params = less_approx(times, 1.0, params)
            sample = jnp.sum(indicator, axis=0)
            return sample, params
        return _jax_wrapped_calc_poisson_exponential

    # normal approximation to Poisson: Poisson(rate) -> Normal(rate, rate)
    def _poisson_normal_approx(self, logic):
        def _jax_wrapped_calc_poisson_normal_approx(key, rate, params):
            normal = random.normal(key=key, shape=jnp.shape(rate), dtype=logic.REAL)
            sample = rate + jnp.sqrt(rate) * normal
            return sample, params    
        return _jax_wrapped_calc_poisson_normal_approx
    
    def poisson(self, id, init_params, logic):
        if self.poisson_exp_method:
            _jax_wrapped_calc_poisson_diff = self._poisson_exponential(
                id, init_params, logic)
        else:
            _jax_wrapped_calc_poisson_diff = self._poisson_gumbel_softmax(
                id, init_params, logic)
        _jax_wrapped_calc_poisson_normal = self._poisson_normal_approx(logic)
        
        # for small rate use the Poisson process or gumbel-softmax reparameterization
        # for large rate use the normal approximation
        def _jax_wrapped_calc_poisson_approx(key, rate, params):
            if self.poisson_bins > 0:
                cuml_prob = scipy.stats.poisson.cdf(self.poisson_bins, rate)
                small_rate = jax.lax.stop_gradient(cuml_prob >= self.poisson_min_cdf)
                small_sample, params = _jax_wrapped_calc_poisson_diff(key, rate, params)
                large_sample, params = _jax_wrapped_calc_poisson_normal(key, rate, params)
                sample = jnp.where(small_rate, small_sample, large_sample)
                return sample, params
            else:
                return _jax_wrapped_calc_poisson_normal(key, rate, params)
        return _jax_wrapped_calc_poisson_approx        
    
    # normal approximation to Binomial: Bin(n, p) -> Normal(np, np(1-p))
    def _binomial_normal_approx(self, logic):
        def _jax_wrapped_calc_binomial_normal_approx(key, trials, prob, params):
            normal = random.normal(key=key, shape=jnp.shape(trials), dtype=logic.REAL)
            mean = trials * prob
            std = jnp.sqrt(trials * prob * (1.0 - prob))
            sample = mean + std * normal
            return sample, params    
        return _jax_wrapped_calc_binomial_normal_approx
        
    def _binomial_gumbel_softmax(self, id, init_params, logic):
        argmax_approx = logic.argmax(id, init_params)
        def _jax_wrapped_calc_binomial_gumbel_softmax(key, trials, prob, params):
            ks = jnp.arange(self.binomial_bins)[(jnp.newaxis,) * jnp.ndim(trials) + (...,)]
            trials = trials[..., jnp.newaxis]
            prob = prob[..., jnp.newaxis]
            in_support = ks <= trials
            ks = jnp.minimum(ks, trials)
            log_prob = ((scipy.special.gammaln(trials + 1) - 
                         scipy.special.gammaln(ks + 1) - 
                         scipy.special.gammaln(trials - ks + 1)) +
                        ks * jnp.log(prob + logic.eps) + 
                        (trials - ks) * jnp.log1p(-prob + logic.eps))
            log_prob = jnp.where(in_support, log_prob, jnp.log(logic.eps))
            Gumbel01 = random.gumbel(key=key, shape=jnp.shape(log_prob), dtype=logic.REAL)
            sample = Gumbel01 + log_prob
            return argmax_approx(sample, axis=-1, params=params)
        return _jax_wrapped_calc_binomial_gumbel_softmax
        
    def binomial(self, id, init_params, logic):
        _jax_wrapped_calc_binomial_normal = self._binomial_normal_approx(logic)  
        _jax_wrapped_calc_binomial_gs = self._binomial_gumbel_softmax(id, init_params, logic)
        
        # for small trials use the Bernoulli relaxation
        # for large trials use the normal approximation
        def _jax_wrapped_calc_binomial_approx(key, trials, prob, params):
            small_trials = jax.lax.stop_gradient(trials < self.binomial_bins)
            small_sample, params = _jax_wrapped_calc_binomial_gs(key, trials, prob, params)
            large_sample, params = _jax_wrapped_calc_binomial_normal(key, trials, prob, params)
            sample = jnp.where(small_trials, small_sample, large_sample)
            return sample, params
        return _jax_wrapped_calc_binomial_approx
    
    # https://en.wikipedia.org/wiki/Negative_binomial_distribution#Gamma%E2%80%93Poisson_mixture
    def negative_binomial(self, id, init_params, logic):
        poisson_approx = self.poisson(id, init_params, logic)
        def _jax_wrapped_calc_negative_binomial_approx(key, trials, prob, params):
            key, subkey = random.split(key)
            trials = jnp.asarray(trials, dtype=logic.REAL)
            Gamma = random.gamma(key=key, a=trials, dtype=logic.REAL)
            scale = (1.0 - prob) / prob
            poisson_rate = scale * Gamma
            return poisson_approx(subkey, poisson_rate, params)
        return _jax_wrapped_calc_negative_binomial_approx

    def geometric(self, id, init_params, logic):
        approx_floor = logic.floor(id, init_params)
        def _jax_wrapped_calc_geometric_approx(key, prob, params):
            U = random.uniform(key=key, shape=jnp.shape(prob), dtype=logic.REAL)
            floor, params = approx_floor(
                jnp.log1p(-U) / jnp.log1p(-prob + logic.eps), params)
            sample = floor + 1
            return sample, params
        return _jax_wrapped_calc_geometric_approx
    
    def _bernoulli_uniform(self, id, init_params, logic):
        less_approx = logic.less(id, init_params)  
        def _jax_wrapped_calc_bernoulli_uniform(key, prob, params):
            U = random.uniform(key=key, shape=jnp.shape(prob), dtype=logic.REAL)
            return less_approx(U, prob, params)   
        return _jax_wrapped_calc_bernoulli_uniform
    
    def _bernoulli_gumbel_softmax(self, id, init_params, logic):
        discrete_approx = self.discrete(id, init_params, logic)        
        def _jax_wrapped_calc_bernoulli_gumbel_softmax(key, prob, params):
            prob = jnp.stack([1.0 - prob, prob], axis=-1)
            return discrete_approx(key, prob, params)        
        return _jax_wrapped_calc_bernoulli_gumbel_softmax
    
    def bernoulli(self, id, init_params, logic):
        if self.bernoulli_gumbel_softmax:
            return self._bernoulli_gumbel_softmax(id, init_params, logic)
        else:
            return self._bernoulli_uniform(id, init_params, logic)

    def __str__(self) -> str:
        return 'SoftRandomSampling'
    

class Determinization(RandomSampling):
    '''Random sampling of variables using their deterministic mean estimate.'''

    @staticmethod
    def _jax_wrapped_calc_discrete_determinized(key, prob, params):
        literals = enumerate_literals(jnp.shape(prob), axis=-1)
        sample = jnp.sum(literals * prob, axis=-1)
        return sample, params
    
    def discrete(self, id, init_params, logic):
        return self._jax_wrapped_calc_discrete_determinized
    
    @staticmethod
    def _jax_wrapped_calc_poisson_determinized(key, rate, params):
        return rate, params       

    def poisson(self, id, init_params, logic):
        return self._jax_wrapped_calc_poisson_determinized
    
    @staticmethod
    def _jax_wrapped_calc_binomial_determinized(key, trials, prob, params):
        sample = trials * prob
        return sample, params
    
    def binomial(self, id, init_params, logic):
        return self._jax_wrapped_calc_binomial_determinized
    
    @staticmethod
    def _jax_wrapped_calc_negative_binomial_determinized(key, trials, prob, params):
        sample = trials * ((1.0 / prob) - 1.0)
        return sample, params

    def negative_binomial(self, id, init_params, logic):
        return self._jax_wrapped_calc_negative_binomial_determinized
    
    @staticmethod
    def _jax_wrapped_calc_geometric_determinized(key, prob, params):
        sample = 1.0 / prob
        return sample, params   
    
    def geometric(self, id, init_params, logic):
        return self._jax_wrapped_calc_geometric_determinized
    
    @staticmethod
    def _jax_wrapped_calc_bernoulli_determinized(key, prob, params):
        sample = prob
        return sample, params
    
    def bernoulli(self, id, init_params, logic):
        return self._jax_wrapped_calc_bernoulli_determinized

    def __str__(self) -> str:
        return 'Deterministic'
    

# ===========================================================================
# CONTROL FLOW
# - soft flow
#
# ===========================================================================

class ControlFlow(metaclass=ABCMeta):
    '''A base class for control flow, including if and switch statements.'''
    
    @abstractmethod
    def if_then_else(self, id, init_params):
        pass
    
    @abstractmethod
    def switch(self, id, init_params):
        pass


class SoftControlFlow(ControlFlow):
    '''Soft control flow using a probabilistic interpretation.'''
    
    def __init__(self, weight: float=10.0) -> None:
        self.weight = float(weight)
        
    @staticmethod
    def _jax_wrapped_calc_if_then_else_soft(c, a, b, params):
        sample = c * a + (1.0 - c) * b
        return sample, params
    
    def if_then_else(self, id, init_params):
        return self._jax_wrapped_calc_if_then_else_soft
    
    def switch(self, id, init_params):
        id_ = str(id)
        init_params[id_] = self.weight
        def _jax_wrapped_calc_switch_soft(pred, cases, params):
            literals = enumerate_literals(jnp.shape(cases), axis=0)
            pred = jnp.broadcast_to(pred[jnp.newaxis, ...], shape=jnp.shape(cases))
            proximity = -jnp.square(pred - literals)
            softcase = jax.nn.softmax(params[id_] * proximity, axis=0)
            sample = jnp.sum(cases * softcase, axis=0)
            return sample, params
        return _jax_wrapped_calc_switch_soft
    
    def __str__(self) -> str:
        return f'Soft control flow with weight {self.weight}'
    
    
# ===========================================================================
# LOGIC
# - exact logic
# - fuzzy logic
#
# ===========================================================================


class Logic(metaclass=ABCMeta):
    '''A base class for representing logic computations in JAX.'''
    
    def __init__(self, use64bit: bool=False) -> None:
        self.set_use64bit(use64bit)
    
    def summarize_hyperparameters(self) -> str:
        return (f'model relaxation:\n'
                f'    use_64_bit    ={self.use64bit}')
    
    def set_use64bit(self, use64bit: bool) -> None:
        '''Toggles whether or not the JAX system will use 64 bit precision.'''
        self.use64bit = use64bit
        if use64bit:
            self.REAL = jnp.float64
            self.INT = jnp.int64
            jax.config.update('jax_enable_x64', True)
        else:
            self.REAL = jnp.float32
            self.INT = jnp.int32
            jax.config.update('jax_enable_x64', False)
    
    @staticmethod
    def wrap_logic(func):
        def exact_func(id, init_params):
            return func
        return exact_func
        
    def get_operator_dicts(self) -> Dict[str, Union[Callable, Dict[str, Callable]]]:
        '''Returns a dictionary of all operators in the current logic.'''
        return {
            'negative': self.wrap_logic(ExactLogic.exact_unary_function(jnp.negative)),
            'arithmetic': {
                '+': self.wrap_logic(ExactLogic.exact_binary_function(jnp.add)),
                '-': self.wrap_logic(ExactLogic.exact_binary_function(jnp.subtract)),
                '*': self.wrap_logic(ExactLogic.exact_binary_function(jnp.multiply)),
                '/': self.wrap_logic(ExactLogic.exact_binary_function(jnp.divide))
            },
            'relational': {
                '>=': self.greater_equal,
                '<=': self.less_equal,
                '<': self.less,
                '>': self.greater,
                '==': self.equal,
                '~=': self.not_equal
            },
            'logical_not': self.logical_not,
            'logical': {
                '^': self.logical_and,
                '&': self.logical_and,
                '|': self.logical_or,
                '~': self.xor,
                '=>': self.implies,
                '<=>': self.equiv
            },
            'aggregation': {
                'sum': self.wrap_logic(ExactLogic.exact_aggregation(jnp.sum)),
                'avg': self.wrap_logic(ExactLogic.exact_aggregation(jnp.mean)),
                'prod': self.wrap_logic(ExactLogic.exact_aggregation(jnp.prod)),
                'minimum': self.wrap_logic(ExactLogic.exact_aggregation(jnp.min)),
                'maximum': self.wrap_logic(ExactLogic.exact_aggregation(jnp.max)),
                'forall': self.forall,
                'exists': self.exists,
                'argmin': self.argmin,
                'argmax': self.argmax
            },
            'unary': {
                'abs': self.wrap_logic(ExactLogic.exact_unary_function(jnp.abs)),
                'sgn': self.sgn,
                'round': self.round,
                'floor': self.floor,
                'ceil': self.ceil,
                'cos': self.wrap_logic(ExactLogic.exact_unary_function(jnp.cos)),
                'sin': self.wrap_logic(ExactLogic.exact_unary_function(jnp.sin)),
                'tan': self.wrap_logic(ExactLogic.exact_unary_function(jnp.tan)),
                'acos': self.wrap_logic(ExactLogic.exact_unary_function(jnp.arccos)),
                'asin': self.wrap_logic(ExactLogic.exact_unary_function(jnp.arcsin)),
                'atan': self.wrap_logic(ExactLogic.exact_unary_function(jnp.arctan)),
                'cosh': self.wrap_logic(ExactLogic.exact_unary_function(jnp.cosh)),
                'sinh': self.wrap_logic(ExactLogic.exact_unary_function(jnp.sinh)),
                'tanh': self.wrap_logic(ExactLogic.exact_unary_function(jnp.tanh)),
                'exp': self.wrap_logic(ExactLogic.exact_unary_function(jnp.exp)),
                'ln': self.wrap_logic(ExactLogic.exact_unary_function(jnp.log)),
                'sqrt': self.sqrt,
                'lngamma': self.wrap_logic(ExactLogic.exact_unary_function(scipy.special.gammaln)),
                'gamma': self.wrap_logic(ExactLogic.exact_unary_function(scipy.special.gamma))
            },
            'binary': {
                'div': self.div,
                'mod': self.mod,
                'fmod': self.mod,
                'min': self.wrap_logic(ExactLogic.exact_binary_function(jnp.minimum)),
                'max': self.wrap_logic(ExactLogic.exact_binary_function(jnp.maximum)),
                'pow': self.wrap_logic(ExactLogic.exact_binary_function(jnp.power)),
                'log': self.wrap_logic(ExactLogic.exact_binary_log),
                'hypot': self.wrap_logic(ExactLogic.exact_binary_function(jnp.hypot)),
            },
            'control': {
                'if': self.control_if,
                'switch': self.control_switch
            },
            'sampling': {
                'Bernoulli': self.bernoulli,
                'Discrete': self.discrete,
                'Poisson': self.poisson,
                'Geometric': self.geometric,
                'Binomial': self.binomial,
                'NegativeBinomial': self.negative_binomial
            }
        }

    # ===========================================================================
    # logical operators
    # ===========================================================================
    
    @abstractmethod
    def logical_and(self, id, init_params):
        pass
    
    @abstractmethod
    def logical_not(self, id, init_params):
        pass
    
    @abstractmethod
    def logical_or(self, id, init_params):
        pass
    
    @abstractmethod
    def xor(self, id, init_params):
        pass
    
    @abstractmethod
    def implies(self, id, init_params):
        pass
    
    @abstractmethod
    def equiv(self, id, init_params):
        pass
    
    @abstractmethod
    def forall(self, id, init_params):
        pass
    
    @abstractmethod
    def exists(self, id, init_params):    
        pass
    
    # ===========================================================================
    # comparison operators
    # ===========================================================================
    
    @abstractmethod
    def greater_equal(self, id, init_params):
        pass
    
    @abstractmethod
    def greater(self, id, init_params):
        pass
    
    @abstractmethod
    def less_equal(self, id, init_params):
        pass
    
    @abstractmethod
    def less(self, id, init_params):
        pass
    
    @abstractmethod
    def equal(self, id, init_params):
        pass
    
    @abstractmethod
    def not_equal(self, id, init_params):
        pass
    
    # ===========================================================================
    # special functions
    # ===========================================================================
     
    @abstractmethod
    def sgn(self, id, init_params):
        pass
    
    @abstractmethod
    def floor(self, id, init_params):
        pass
    
    @abstractmethod
    def round(self, id, init_params):
        pass
    
    @abstractmethod
    def ceil(self, id, init_params):
        pass 
    
    @abstractmethod
    def div(self, id, init_params):
        pass 
    
    @abstractmethod
    def mod(self, id, init_params):
        pass
    
    @abstractmethod
    def sqrt(self, id, init_params):
        pass
    
    # ===========================================================================
    # indexing
    # ===========================================================================
    
    @abstractmethod
    def argmax(self, id, init_params):   
        pass
    
    @abstractmethod
    def argmin(self, id, init_params):   
        pass
    
    # ===========================================================================
    # control flow
    # ===========================================================================
     
    @abstractmethod
    def control_if(self, id, init_params):
        pass
        
    @abstractmethod
    def control_switch(self, id, init_params):
        pass
    
    # ===========================================================================
    # random variables
    # ===========================================================================
     
    @abstractmethod
    def discrete(self, id, init_params):
        pass
    
    @abstractmethod
    def bernoulli(self, id, init_params):
        pass
    
    @abstractmethod
    def poisson(self, id, init_params):
        pass
    
    @abstractmethod
    def geometric(self, id, init_params):
        pass
    
    @abstractmethod
    def binomial(self, id, init_params):
        pass

    @abstractmethod
    def negative_binomial(self, id, init_params):
        pass


class ExactLogic(Logic):
    '''A class representing exact logic in JAX.'''
    
    @staticmethod
    def exact_unary_function(op):
        def _jax_wrapped_calc_unary_function_exact(x, params):
            return op(x), params
        return _jax_wrapped_calc_unary_function_exact
    
    @staticmethod
    def exact_binary_function(op):
        def _jax_wrapped_calc_binary_function_exact(x, y, params):
            return op(x, y), params
        return _jax_wrapped_calc_binary_function_exact
    
    @staticmethod
    def exact_aggregation(op):        
        def _jax_wrapped_calc_aggregation_exact(x, axis, params):
            return op(x, axis=axis), params   
        return _jax_wrapped_calc_aggregation_exact

    # ===========================================================================
    # logical operators
    # ===========================================================================
     
    def logical_and(self, id, init_params):
        return self.exact_binary_function(jnp.logical_and)
    
    def logical_not(self, id, init_params):
        return self.exact_unary_function(jnp.logical_not)
    
    def logical_or(self, id, init_params):
        return self.exact_binary_function(jnp.logical_or)
    
    def xor(self, id, init_params):
        return self.exact_binary_function(jnp.logical_xor)
    
    @staticmethod
    def _jax_wrapped_calc_implies_exact(x, y, params):
        return jnp.logical_or(jnp.logical_not(x), y), params     
    
    def implies(self, id, init_params):
        return self._jax_wrapped_calc_implies_exact
    
    def equiv(self, id, init_params):
        return self.exact_binary_function(jnp.equal)
    
    def forall(self, id, init_params):
        return self.exact_aggregation(jnp.all)
    
    def exists(self, id, init_params):
        return self.exact_aggregation(jnp.any)
    
    # ===========================================================================
    # comparison operators
    # ===========================================================================
    
    def greater_equal(self, id, init_params):
        return self.exact_binary_function(jnp.greater_equal)
    
    def greater(self, id, init_params):
        return self.exact_binary_function(jnp.greater)
    
    def less_equal(self, id, init_params):
        return self.exact_binary_function(jnp.less_equal)
    
    def less(self, id, init_params):
        return self.exact_binary_function(jnp.less)
    
    def equal(self, id, init_params):
        return self.exact_binary_function(jnp.equal)
    
    def not_equal(self, id, init_params):
        return self.exact_binary_function(jnp.not_equal)
    
    # ===========================================================================
    # special functions
    # ===========================================================================
     
    @staticmethod
    def exact_binary_log(x, y, params):
        return jnp.log(x) / jnp.log(y), params
    
    def sgn(self, id, init_params):
        return self.exact_unary_function(jnp.sign)
    
    def floor(self, id, init_params):
        return self.exact_unary_function(jnp.floor)
    
    def round(self, id, init_params):
        return self.exact_unary_function(jnp.round)
    
    def ceil(self, id, init_params):
        return self.exact_unary_function(jnp.ceil)
    
    def div(self, id, init_params):
        return self.exact_binary_function(jnp.floor_divide)
    
    def mod(self, id, init_params):
        return self.exact_binary_function(jnp.mod)
    
    def sqrt(self, id, init_params):
        return self.exact_unary_function(jnp.sqrt)
    
    # ===========================================================================
    # indexing
    # ===========================================================================
     
    def argmax(self, id, init_params):   
        return self.exact_aggregation(jnp.argmax)
    
    def argmin(self, id, init_params):   
        return self.exact_aggregation(jnp.argmin)
    
    # ===========================================================================
    # control flow
    # ===========================================================================
    
    @staticmethod
    def _jax_wrapped_calc_if_then_else_exact(c, a, b, params):
        return jnp.where(c > 0.5, a, b), params
         
    def control_if(self, id, init_params):
        return self._jax_wrapped_calc_if_then_else_exact
    
    def control_switch(self, id, init_params):
        def _jax_wrapped_calc_switch_exact(pred, cases, params):
            pred = jnp.asarray(pred[jnp.newaxis, ...], dtype=self.INT)
            sample = jnp.take_along_axis(cases, pred, axis=0)
            assert sample.shape[0] == 1
            return sample[0, ...], params
        return _jax_wrapped_calc_switch_exact
    
    # ===========================================================================
    # random variables
    # ===========================================================================
    
    @staticmethod
    def _jax_wrapped_calc_discrete_exact(key, prob, params):
        sample = random.categorical(key=key, logits=jnp.log(prob), axis=-1)
        return sample, params  
    
    def discrete(self, id, init_params):
        return self._jax_wrapped_calc_discrete_exact
    
    @staticmethod
    def _jax_wrapped_calc_bernoulli_exact(key, prob, params):
        return random.bernoulli(key, prob), params
    
    def bernoulli(self, id, init_params):
        return self._jax_wrapped_calc_bernoulli_exact
    
    def poisson(self, id, init_params):
        def _jax_wrapped_calc_poisson_exact(key, rate, params):
            sample = random.poisson(key=key, lam=rate, dtype=self.INT)
            return sample, params
        return _jax_wrapped_calc_poisson_exact
    
    def geometric(self, id, init_params):
        def _jax_wrapped_calc_geometric_exact(key, prob, params):
            sample = random.geometric(key=key, p=prob, dtype=self.INT)
            return sample, params
        return _jax_wrapped_calc_geometric_exact
    
    def binomial(self, id, init_params):
        def _jax_wrapped_calc_binomial_exact(key, trials, prob, params):
            trials = jnp.asarray(trials, dtype=self.REAL)
            prob = jnp.asarray(prob, dtype=self.REAL)
            sample = random.binomial(key=key, n=trials, p=prob, dtype=self.REAL)
            sample = jnp.asarray(sample, dtype=self.INT)
            return sample, params
        return _jax_wrapped_calc_binomial_exact
    
    # note: for some reason tfp defines it as number of successes before trials failures
    # I will define it as the number of failures before trials successes
    def negative_binomial(self, id, init_params):
        def _jax_wrapped_calc_negative_binomial_exact(key, trials, prob, params):
            trials = jnp.asarray(trials, dtype=self.REAL)
            prob = jnp.asarray(prob, dtype=self.REAL)
            dist = tfp.distributions.NegativeBinomial(total_count=trials, probs=1.0 - prob)
            sample = jnp.asarray(dist.sample(seed=key), dtype=self.INT)
            return sample, params
        return _jax_wrapped_calc_negative_binomial_exact

    
class FuzzyLogic(Logic):
    '''A class representing fuzzy logic in JAX.'''
    
    def __init__(self, tnorm: TNorm=ProductTNorm(),
                 complement: Complement=StandardComplement(),
                 comparison: Comparison=SigmoidComparison(),
                 sampling: RandomSampling=SoftRandomSampling(),
                 rounding: Rounding=SoftRounding(),
                 control: ControlFlow=SoftControlFlow(),
                 eps: float=1e-15,
                 use64bit: bool=False) -> None:
        '''Creates a new fuzzy logic in Jax.
        
        :param tnorm: fuzzy operator for logical AND
        :param complement: fuzzy operator for logical NOT
        :param comparison: fuzzy operator for comparisons (>, >=, <, ==, ~=, ...)
        :param sampling: random sampling of non-reparameterizable distributions
        :param rounding: rounding floating values to integers
        :param control: if and switch control structures
        :param eps: small positive float to mitigate underflow
        :param use64bit: whether to perform arithmetic in 64 bit
        '''
        super().__init__(use64bit=use64bit)
        self.tnorm = tnorm
        self.complement = complement
        self.comparison = comparison
        self.sampling = sampling
        self.rounding = rounding
        self.control = control
        self.eps = eps
    
    def __str__(self) -> str:
        return (f'model relaxation:\n'
                f'    tnorm        ={str(self.tnorm)}\n'
                f'    complement   ={str(self.complement)}\n'
                f'    comparison   ={str(self.comparison)}\n'
                f'    sampling     ={str(self.sampling)}\n'
                f'    rounding     ={str(self.rounding)}\n'
                f'    control      ={str(self.control)}\n'
                f'    underflow_tol={self.eps}\n'
                f'    use_64_bit   ={self.use64bit}\n')

    def summarize_hyperparameters(self) -> str:
        return self.__str__()
        
    # ===========================================================================
    # logical operators
    # ===========================================================================
     
    def logical_and(self, id, init_params):
        return self.tnorm.norm(id, init_params)
    
    def logical_not(self, id, init_params):
        return self.complement(id, init_params)
    
    def logical_or(self, id, init_params):
        _not1 = self.complement(f'{id}_~1', init_params)
        _not2 = self.complement(f'{id}_~2', init_params)
        _and = self.tnorm.norm(f'{id}_^', init_params)
        _not = self.complement(f'{id}_~', init_params)
        
        def _jax_wrapped_calc_or_approx(x, y, params):
            not_x, params = _not1(x, params)
            not_y, params = _not2(y, params)
            not_x_and_not_y, params = _and(not_x, not_y, params)
            return _not(not_x_and_not_y, params)        
        return _jax_wrapped_calc_or_approx

    def xor(self, id, init_params):
        _not = self.complement(f'{id}_~', init_params)
        _and1 = self.tnorm.norm(f'{id}_^1', init_params)
        _and2 = self.tnorm.norm(f'{id}_^2', init_params)
        _or = self.logical_or(f'{id}_|', init_params)
        
        def _jax_wrapped_calc_xor_approx(x, y, params):
            x_and_y, params = _and1(x, y, params)
            not_x_and_y, params = _not(x_and_y, params)
            x_or_y, params = _or(x, y, params)
            return _and2(x_or_y, not_x_and_y, params)        
        return _jax_wrapped_calc_xor_approx
        
    def implies(self, id, init_params):
        _not = self.complement(f'{id}_~', init_params)
        _or = self.logical_or(f'{id}_|', init_params)
        
        def _jax_wrapped_calc_implies_approx(x, y, params):
            not_x, params = _not(x, params)
            return _or(not_x, y, params)        
        return _jax_wrapped_calc_implies_approx
       
    def equiv(self, id, init_params):
        _implies1 = self.implies(f'{id}_=>1', init_params)
        _implies2 = self.implies(f'{id}_=>2', init_params)
        _and = self.tnorm.norm(f'{id}_^', init_params)
           
        def _jax_wrapped_calc_equiv_approx(x, y, params):
            x_implies_y, params = _implies1(x, y, params)
            y_implies_x, params = _implies2(y, x, params)
            return _and(x_implies_y, y_implies_x, params)        
        return _jax_wrapped_calc_equiv_approx
    
    def forall(self, id, init_params):
        return self.tnorm.norms(id, init_params)
    
    def exists(self, id, init_params):
        _not1 = self.complement(f'{id}_~1', init_params)
        _not2 = self.complement(f'{id}_~2', init_params)
        _forall = self.forall(f'{id}_forall', init_params)
        
        def _jax_wrapped_calc_exists_approx(x, axis, params):
            not_x, params = _not1(x, params)
            forall_not_x, params = _forall(not_x, axis, params)
            return _not2(forall_not_x, params)        
        return _jax_wrapped_calc_exists_approx
    
    # ===========================================================================
    # comparison operators
    # ===========================================================================
    
    def greater_equal(self, id, init_params):
        return self.comparison.greater_equal(id, init_params)
    
    def greater(self, id, init_params):
        return self.comparison.greater(id, init_params)
    
    def less_equal(self, id, init_params):
        _greater_eq = self.greater_equal(id, init_params)        
        def _jax_wrapped_calc_leq_approx(x, y, params):
            return _greater_eq(-x, -y, params)        
        return _jax_wrapped_calc_leq_approx
    
    def less(self, id, init_params):
        _greater = self.greater(id, init_params)  
        def _jax_wrapped_calc_less_approx(x, y, params):
            return _greater(-x, -y, params)
        return _jax_wrapped_calc_less_approx

    def equal(self, id, init_params):
        return self.comparison.equal(id, init_params)
    
    def not_equal(self, id, init_params):
        _not = self.complement(f'{id}_~', init_params)
        _equal = self.comparison.equal(f'{id}_==', init_params)
        def _jax_wrapped_calc_neq_approx(x, y, params):
            equal, params = _equal(x, y, params)
            return _not(equal, params)
        return _jax_wrapped_calc_neq_approx
        
    # ===========================================================================
    # special functions
    # ===========================================================================
    
    def sgn(self, id, init_params):
        return self.comparison.sgn(id, init_params)
    
    def floor(self, id, init_params):
        return self.rounding.floor(id, init_params)
        
    def round(self, id, init_params):
        return self.rounding.round(id, init_params)
    
    def ceil(self, id, init_params):
        _floor = self.rounding.floor(id, init_params)
        def _jax_wrapped_calc_ceil_approx(x, params):
            neg_floor, params = _floor(-x, params) 
            return -neg_floor, params        
        return _jax_wrapped_calc_ceil_approx
    
    def div(self, id, init_params):
        _floor = self.rounding.floor(id, init_params)        
        def _jax_wrapped_calc_div_approx(x, y, params):
            return _floor(x / y, params)
        return _jax_wrapped_calc_div_approx
    
    def mod(self, id, init_params):
        _div = self.div(id, init_params)        
        def _jax_wrapped_calc_mod_approx(x, y, params):
            div, params = _div(x, y, params)
            return x - y * div, params        
        return _jax_wrapped_calc_mod_approx
    
    def sqrt(self, id, init_params):      
        def _jax_wrapped_calc_sqrt_approx(x, params):
            return jnp.sqrt(x + self.eps), params   
        return _jax_wrapped_calc_sqrt_approx
    
    # ===========================================================================
    # indexing
    # ===========================================================================
     
    def argmax(self, id, init_params): 
        return self.comparison.argmax(id, init_params)
    
    def argmin(self, id, init_params):
        _argmax = self.argmax(id, init_params)
        def _jax_wrapped_calc_argmin_approx(x, axis, param):
            return _argmax(-x, axis, param)        
        return _jax_wrapped_calc_argmin_approx
    
    # ===========================================================================
    # control flow
    # ===========================================================================
     
    def control_if(self, id, init_params):
        return self.control.if_then_else(id, init_params)
    
    def control_switch(self, id, init_params):
        return self.control.switch(id, init_params)
    
    # ===========================================================================
    # random variables
    # ===========================================================================
     
    def discrete(self, id, init_params):
        return self.sampling.discrete(id, init_params, self)
    
    def bernoulli(self, id, init_params):
        return self.sampling.bernoulli(id, init_params, self)
    
    def poisson(self, id, init_params):
        return self.sampling.poisson(id, init_params, self)
    
    def geometric(self, id, init_params):
        return self.sampling.geometric(id, init_params, self)
    
    def binomial(self, id, init_params):
        return self.sampling.binomial(id, init_params, self)
    
    def negative_binomial(self, id, init_params):
        return self.sampling.negative_binomial(id, init_params, self)


# ===========================================================================
# UNIT TESTS
#
# ===========================================================================

logic = FuzzyLogic(comparison=SigmoidComparison(10000.0),
                   rounding=SoftRounding(10000.0),
                   control=SoftControlFlow(10000.0))


def _test_logical():
    print('testing logical')
    init_params = {}
    _and = logic.logical_and(0, init_params)
    _not = logic.logical_not(1, init_params)
    _gre = logic.greater(2, init_params)
    _or = logic.logical_or(3, init_params)
    _if = logic.control_if(4, init_params)
    print(init_params)

    # https://towardsdatascience.com/emulating-logical-gates-with-a-neural-network-75c229ec4cc9
    def test_logic(x1, x2, w):
        q1, w = _gre(x1, 0, w)
        q2, w = _gre(x2, 0, w)
        q3, w = _and(q1, q2, w)
        q4, w = _not(q1, w)
        q5, w = _not(q2, w)
        q6, w = _and(q4, q5, w)        
        cond, w = _or(q3, q6, w)
        pred, w = _if(cond, +1, -1, w)
        return pred
    
    x1 = jnp.asarray([1, 1, -1, -1, 0.1, 15, -0.5], dtype=float)
    x2 = jnp.asarray([1, -1, 1, -1, 10, -30, 6], dtype=float)
    print(test_logic(x1, x2, init_params))    


def _test_indexing():
    print('testing indexing')
    init_params = {}
    _argmax = logic.argmax(0, init_params)
    _argmin = logic.argmin(1, init_params)
    print(init_params)

    def argmaxmin(x, w):
        amax, w = _argmax(x, 0, w)
        amin, w = _argmin(x, 0, w)
        return amax, amin
        
    values = jnp.asarray([2., 3., 5., 4.9, 4., 1., -1., -2.])
    amax, amin = argmaxmin(values, init_params)
    print(amax)
    print(amin)


def _test_control():
    print('testing control')
    init_params = {}
    _switch = logic.control_switch(0, init_params)
    print(init_params)
    
    pred = jnp.asarray(jnp.linspace(0, 2, 10))
    case1 = jnp.asarray([-10.] * 10)
    case2 = jnp.asarray([1.5] * 10)
    case3 = jnp.asarray([10.] * 10)
    cases = jnp.asarray([case1, case2, case3])
    switch, _ = _switch(pred, cases, init_params)
    print(switch)


def _test_random():
    print('testing random')
    key = random.PRNGKey(42)
    init_params = {}
    _bernoulli = logic.bernoulli(0, init_params)
    _discrete = logic.discrete(1, init_params)
    _geometric = logic.geometric(2, init_params)
    print(init_params)
    
    def bern(n, w):
        prob = jnp.asarray([0.3] * n)
        sample, _ = _bernoulli(key, prob, w)
        return sample
    
    samples = bern(50000, init_params)
    print(jnp.mean(samples))
    
    def disc(n, w):
        prob = jnp.asarray([0.1, 0.4, 0.5])
        prob = jnp.tile(prob, (n, 1))
        sample, _ = _discrete(key, prob, w)
        return sample
        
    samples = disc(50000, init_params)
    samples = jnp.round(samples)
    print([jnp.mean(samples == i) for i in range(3)])
    
    def geom(n, w):
        prob = jnp.asarray([0.3] * n)
        sample, _ = _geometric(key, prob, w)
        return sample
    
    samples = geom(50000, init_params)
    print(jnp.mean(samples))
    

def _test_rounding():
    print('testing rounding')
    init_params = {}
    _floor = logic.floor(0, init_params)
    _ceil = logic.ceil(1, init_params)
    _round = logic.round(2, init_params)
    _mod = logic.mod(3, init_params)
    print(init_params)
    
    x = jnp.asarray([2.1, 0.6, 1.99, -2.01, -3.2, -0.1, -1.01, 23.01, -101.99, 200.01])
    print(_floor(x, init_params)[0])
    print(_ceil(x, init_params)[0])
    print(_round(x, init_params)[0])
    print(_mod(x, 2.0, init_params)[0])


if __name__ == '__main__':
    _test_logical()
    _test_indexing()
    _test_control()
    _test_random()
    _test_rounding()
    
