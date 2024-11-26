from typing import Optional, Set

import jax
import jax.numpy as jnp
import jax.random as random


def enumerate_literals(shape, axis, dtype=jnp.int64):
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

class Comparison:
    '''Base class for approximate comparison operations.'''
    
    def greater_equal(self, id, init_params):
        raise NotImplementedError
    
    def greater(self, id, init_params):
        raise NotImplementedError
    
    def equal(self, id, init_params):
        raise NotImplementedError
    
    def sgn(self, id, init_params):
        raise NotImplementedError
    
    def argmax(self, id, init_params):
        raise NotImplementedError
    

class SigmoidComparison(Comparison):
    '''Comparison operations approximated using sigmoid functions.'''
    
    def __init__(self, weight: float=10.0):
        self.weight = weight
        
    # https://arxiv.org/abs/2110.05651
    def greater_equal(self, id, init_params):
        init_params[id] = self.weight
        def _jax_wrapped_calc_greater_equal_approx(x, y, params):
            gre_eq = jax.nn.sigmoid(params[id] * (x - y))
            return gre_eq, params
        return _jax_wrapped_calc_greater_equal_approx
    
    def greater(self, id, init_params):
        return self.greater_equal(id, init_params)
    
    def sgn(self, id, init_params):
        init_params[id] = self.weight
        def _jax_wrapped_calc_sgn_approx(x, params):
            sgn = jnp.tanh(params[id] * x)
            return sgn, params
        return _jax_wrapped_calc_sgn_approx
    
    # https://arxiv.org/abs/2110.05651
    def argmax(self, id, init_params):
        init_params[id] = self.weight
        def _jax_wrapped_calc_argmax_approx(x, axis, params):
            literals = enumerate_literals(x.shape, axis=axis)
            softmax = jax.nn.softmax(params[id] * x, axis=axis)
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

class Rounding:
    '''Base class for approximate rounding operations.'''
    
    def floor(self, id, init_params):
        raise NotImplementedError
    
    def round(self, id, init_params):
        raise NotImplementedError


class SoftRounding(Rounding):
    '''Rounding operations approximated using soft operations.'''
    
    def __init__(self, weight: float=10.0):
        self.weight = weight
        
    # https://www.tensorflow.org/probability/api_docs/python/tfp/substrates/jax/bijectors/Softfloor
    def floor(self, id, init_params):
        init_params[id] = self.weight
        def _jax_wrapped_calc_floor_approx(x, params):
            param = params[id]
            denom = jnp.tanh(param / 4.0)
            floor = (jax.nn.sigmoid(param * (x - jnp.floor(x) - 1.0)) - 
                     jax.nn.sigmoid(-param / 2.0)) / denom + jnp.floor(x)
            return floor, params
        return _jax_wrapped_calc_floor_approx
    
    # https://arxiv.org/abs/2006.09952
    def round(self, id, init_params):
        init_params[id] = self.weight
        def _jax_wrapped_calc_round_approx(x, params):
            param = params[id]
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

class Complement:
    '''Base class for approximate logical complement operations.'''
    
    def __call__(self, id, init_params):
        raise NotImplementedError


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

class TNorm:
    '''Base class for fuzzy differentiable t-norms.'''
    
    def norm(self, id, init_params):
        '''Elementwise t-norm of x and y.'''
        raise NotImplementedError
    
    def norms(self, id, init_params):
        '''T-norm computed for tensor x along axis.'''
        raise NotImplementedError
    

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
    
    def __init__(self, p=2.0):
        self.p = float(p)
    
    def norm(self, id, init_params):
        init_params[id] = self.p
        def _jax_wrapped_calc_and_approx(x, y, params):
            base = jax.nn.relu(1.0 - jnp.stack([x, y], axis=0))
            arg = jnp.linalg.norm(base, ord=params[id], axis=0)
            land = jax.nn.relu(1.0 - arg)
            return land, params
        return _jax_wrapped_calc_and_approx
    
    def norms(self, id, init_params):
        init_params[id] = self.p
        def _jax_wrapped_calc_forall_approx(x, axis, params):
            arg = jax.nn.relu(1.0 - x)
            for ax in sorted(axis, reverse=True):
                arg = jnp.linalg.norm(arg, ord=params[id], axis=ax)
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

class RandomSampling:
    '''An abstract class that describes how discrete and non-reparameterizable 
    random variables are sampled.'''
    
    def discrete(self, id, init_params, logic):
        raise NotImplementedError
    
    def bernoulli(self, id, init_params, logic):
        discrete_approx = self.discrete(id, init_params, logic)        
        def _jax_wrapped_calc_bernoulli_approx(key, prob, params):
            prob = jnp.stack([1.0 - prob, prob], axis=-1)
            return discrete_approx(key, prob, params)        
        return _jax_wrapped_calc_bernoulli_approx
    
    @staticmethod
    def _jax_wrapped_calc_poisson_exact(key, rate, params):
        sample = random.poisson(key=key, lam=rate, dtype=logic.INT)
        return sample, params
        
    def poisson(self, id, init_params, logic):
        return self._jax_wrapped_calc_poisson_exact
    
    def geometric(self, id, init_params, logic):
        approx_floor = logic.floor(id, init_params)
        def _jax_wrapped_calc_geometric_approx(key, prob, params):
            U = random.uniform(key=key, shape=jnp.shape(prob), dtype=logic.REAL)
            floor, params = approx_floor(jnp.log(U) / jnp.log(1.0 - prob), params)
            sample = floor + 1
            return sample, params
        return _jax_wrapped_calc_geometric_approx


class GumbelSoftmax(RandomSampling):
    '''Random sampling of discrete variables using Gumbel-softmax trick.'''
    
    # https://arxiv.org/pdf/1611.01144
    def discrete(self, id, init_params, logic):
        argmax_approx = logic.argmax(id, init_params)
        def _jax_wrapped_calc_discrete_gumbel_softmax(key, prob, params):
            Gumbel01 = random.gumbel(key=key, shape=prob.shape, dtype=logic.REAL)
            sample = Gumbel01 + jnp.log(prob + logic.eps)
            return argmax_approx(sample, axis=-1, params=params)
        return _jax_wrapped_calc_discrete_gumbel_softmax
    
    def __str__(self) -> str:
        return 'Gumbel-Softmax'
    

class Determinization(RandomSampling):
    '''Random sampling of variables using their deterministic mean estimate.'''
    
    @staticmethod
    def _jax_wrapped_calc_discrete_determinized(key, prob, params):
        literals = enumerate_literals(prob.shape, axis=-1)
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
    def _jax_wrapped_calc_geometric_determinized(key, prob, params):
        sample = 1.0 / prob
        return sample, params   
    
    def geometric(self, id, init_params, logic):
        return self._jax_wrapped_calc_geometric_determinized
    
    def __str__(self) -> str:
        return 'Deterministic'
    

# ===========================================================================
# CONTROL FLOW
# - soft flow
#
# ===========================================================================

class ControlFlow:
    '''A base class for control flow, including if and switch statements.'''
    
    def if_then_else(self, id, init_params):
        raise NotImplementedError
    
    def switch(self, id, init_params):
        raise NotImplementedError


class SoftControlFlow(ControlFlow):
    '''Soft control flow using a probabilistic interpretation.'''
    
    def __init__(self, weight: float=10.0) -> None:
        self.weight = weight
        
    @staticmethod
    def _jax_wrapped_calc_if_then_else_soft(c, a, b, params):
        sample = c * a + (1.0 - c) * b
        return sample, params
    
    def if_then_else(self, id, init_params):
        return self._jax_wrapped_calc_if_then_else_soft
    
    def switch(self, id, init_params):
        init_params[id] = self.weight
        def _jax_wrapped_calc_switch_soft(pred, cases, params):
            literals = enumerate_literals(cases.shape, axis=0)
            pred = jnp.broadcast_to(pred[jnp.newaxis, ...], shape=cases.shape)
            proximity = -jnp.square(pred - literals)
            softcase = jax.nn.softmax(params[id] * proximity, axis=0)
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


class Logic:
    '''A base class for representing logic computations in JAX.'''
    
    def __init__(self, use64bit: bool=False) -> None:
        self.set_use64bit(use64bit)
    
    def summarize_hyperparameters(self) -> None:
        print(f'model relaxation:\n'
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
    
    # ===========================================================================
    # logical operators
    # ===========================================================================
     
    def logical_and(self, id, init_params):
        raise NotImplementedError
    
    def logical_not(self, id, init_params):
        raise NotImplementedError
    
    def logical_or(self, id, init_params):
        raise NotImplementedError
    
    def xor(self, id, init_params):
        raise NotImplementedError
    
    def implies(self, id, init_params):
        raise NotImplementedError
    
    def equiv(self, id, init_params):
        raise NotImplementedError
    
    def forall(self, id, init_params):
        raise NotImplementedError
    
    def exists(self, id, init_params):    
        raise NotImplementedError 
    
    # ===========================================================================
    # comparison operators
    # ===========================================================================
    
    def greater_equal(self, id, init_params):
        raise NotImplementedError 
    
    def greater(self, id, init_params):
        raise NotImplementedError 
    
    def less_equal(self, id, init_params):
        raise NotImplementedError 
    
    def less(self, id, init_params):
        raise NotImplementedError 
    
    def equal(self, id, init_params):
        raise NotImplementedError 
    
    def not_equal(self, id, init_params):
        raise NotImplementedError 
    
    # ===========================================================================
    # special functions
    # ===========================================================================
     
    def sgn(self, id, init_params):
        raise NotImplementedError 
    
    def floor(self, id, init_params):
        raise NotImplementedError 
    
    def round(self, id, init_params):
        raise NotImplementedError 
    
    def ceil(self, id, init_params):
        raise NotImplementedError 
    
    def div(self, id, init_params):
        raise NotImplementedError 
    
    def mod(self, id, init_params):
        raise NotImplementedError 
    
    def sqrt(self, id, init_params):
        raise NotImplementedError 
    
    # ===========================================================================
    # indexing
    # ===========================================================================
     
    def argmax(self, id, init_params):   
        raise NotImplementedError  
    
    def argmin(self, id, init_params):   
        raise NotImplementedError     
    
    # ===========================================================================
    # control flow
    # ===========================================================================
     
    def control_if(self, id, init_params):
        raise NotImplementedError
        
    def control_switch(self, id, init_params):
        raise NotImplementedError
    
    # ===========================================================================
    # random variables
    # ===========================================================================
     
    def discrete(self, id, init_params):
        raise NotImplementedError
    
    def bernoulli(self, id, init_params):
        raise NotImplementedError
    
    def poisson(self, id, init_params):
        raise NotImplementedError
    
    def geometric(self, id, init_params):
        raise NotImplementedError


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
    def exact_binary_implies(x, y, params):
        return jnp.logical_or(jnp.logical_not(x), y), params     
    
    def implies(self, id, init_params):
        return self.exact_binary_implies
    
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
    def exact_if_then_else(c, a, b, params):
        return jnp.where(c > 0.5, a, b), params
         
    def control_if(self, id, init_params):
        return self.exact_if_then_else
    
    @staticmethod
    def exact_switch(pred, cases, params):
        pred = pred[jnp.newaxis, ...]
        sample = jnp.take_along_axis(cases, pred, axis=0)
        assert sample.shape[0] == 1
        return sample[0, ...], params
    
    def control_switch(self, id, init_params):
        return self.exact_switch
    
    # ===========================================================================
    # random variables
    # ===========================================================================
    
    @staticmethod
    def exact_discrete(key, prob, params):
        return random.categorical(key=key, logits=jnp.log(prob), axis=-1), params  
    
    def discrete(self, id, init_params):
        return self.exact_discrete
    
    @staticmethod
    def exact_bernoulli(key, prob, params):
        return random.bernoulli(key, prob), params
    
    def bernoulli(self, id, init_params):
        return self.exact_bernoulli
    
    @staticmethod
    def exact_poisson(key, rate, params):
        return random.poisson(key=key, lam=rate, dtype=self.INT), params   
    
    def poisson(self, id, init_params):
        return self.exact_poisson
    
    @staticmethod
    def exact_geometric(key, prob, params):
        return random.geometric(key=key, p=prob, dtype=self.INT), params
    
    def geometric(self, id, init_params):
        return self.exact_geometric
    
    
class FuzzyLogic(Logic):
    '''A class representing fuzzy logic in JAX.'''
    
    def __init__(self, tnorm: TNorm=ProductTNorm(),
                 complement: Complement=StandardComplement(),
                 comparison: Comparison=SigmoidComparison(),
                 sampling: RandomSampling=GumbelSoftmax(),
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
    
    def summarize_hyperparameters(self) -> None:
        print(f'model relaxation:\n'
              f'    tnorm         ={str(self.tnorm)}\n'
              f'    complement    ={str(self.complement)}\n'
              f'    comparison    ={str(self.comparison)}\n'
              f'    sampling      ={str(self.sampling)}\n'
              f'    rounding      ={str(self.rounding)}\n'
              f'    control       ={str(self.control)}\n'
              f'    underflow_tol ={self.eps}\n'
              f'    use_64_bit    ={self.use64bit}')
        
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
            x_implies_y, params = _implies(x, y, params)
            y_implies_x, params = _implies(y, x, params)
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
    
    x1 = jnp.asarray([1, 1, -1, -1, 0.1, 15, -0.5]).astype(float)
    x2 = jnp.asarray([1, -1, 1, -1, 10, -30, 6]).astype(float)
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
    
