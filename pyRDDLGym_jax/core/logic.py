from typing import Optional, Set

import jax
import jax.numpy as jnp
import jax.random as random

from pyRDDLGym.core.debug.exception import raise_warning


# ===========================================================================
# LOGICAL COMPLEMENT
# - abstract class
# - standard complement
#
# ===========================================================================

class Complement:
    '''Base class for approximate logical complement operations.'''
    
    def __call__(self, x):
        raise NotImplementedError


class StandardComplement(Complement):
    '''The standard approximate logical complement given by x -> 1 - x.'''
    
    def __call__(self, x):
        return 1.0 - x


# ===========================================================================
# RELATIONAL OPERATIONS
# - abstract class
# - sigmoid comparison
#
# ===========================================================================

class Comparison:
    '''Base class for approximate comparison operations.'''
    
    def greater_equal(self, x, y, param):
        raise NotImplementedError
    
    def greater(self, x, y, param):
        raise NotImplementedError
    
    def equal(self, x, y, param):
        raise NotImplementedError
    

class SigmoidComparison(Comparison):
    '''Comparison operations approximated using sigmoid functions.'''
    
    def greater_equal(self, x, y, param):
        return jax.nn.sigmoid(param * (x - y))
    
    def greater(self, x, y, param):
        return jax.nn.sigmoid(param * (x - y))
    
    def equal(self, x, y, param):
        return 1.0 - jnp.square(jnp.tanh(param * (y - x)))
 
        
# ===========================================================================
# TNORMS
# - abstract tnorm
# - product tnorm
# - Godel tnorm
# - Lukasiewicz tnorm
# - Yager(p) tnorm
#
# ===========================================================================

class TNorm:
    '''Base class for fuzzy differentiable t-norms.'''
    
    def norm(self, x, y):
        '''Elementwise t-norm of x and y.'''
        raise NotImplementedError
    
    def norms(self, x, axis):
        '''T-norm computed for tensor x along axis.'''
        raise NotImplementedError
        

class ProductTNorm(TNorm):
    '''Product t-norm given by the expression (x, y) -> x * y.'''
    
    def norm(self, x, y):
        return x * y
    
    def norms(self, x, axis):
        return jnp.prod(x, axis=axis)


class GodelTNorm(TNorm):
    '''Godel t-norm given by the expression (x, y) -> min(x, y).'''
    
    def norm(self, x, y):
        return jnp.minimum(x, y)
    
    def norms(self, x, axis):
        return jnp.min(x, axis=axis)


class LukasiewiczTNorm(TNorm):
    '''Lukasiewicz t-norm given by the expression (x, y) -> max(x + y - 1, 0).'''
    
    def norm(self, x, y):
        return jax.nn.relu(x + y - 1.0)
    
    def norms(self, x, axis):
        return jax.nn.relu(jnp.sum(x - 1.0, axis=axis) + 1.0)


class YagerTNorm(TNorm):
    '''Yager t-norm given by the expression 
    (x, y) -> max(1 - ((1 - x)^p + (1 - y)^p)^(1/p)).'''
    
    def __init__(self, p=2.0):
        self.p = p
    
    def norm(self, x, y):
        base_x = jax.nn.relu(1.0 - x)
        base_y = jax.nn.relu(1.0 - y)
        arg = jnp.power(base_x ** self.p + base_y ** self.p, 1.0 / self.p)
        return jax.nn.relu(1.0 - arg)
    
    def norms(self, x, axis):
        base = jax.nn.relu(1.0 - x)
        arg = jnp.power(jnp.sum(base ** self.p, axis=axis), 1.0 / self.p)
        return jax.nn.relu(1.0 - arg)
        

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
    
    def discrete(self, logic):
        raise NotImplementedError
    
    def bernoulli(self, logic):
        jax_discrete, jax_param = self.discrete(logic)
        
        def _jax_wrapped_calc_bernoulli_approx(key, prob, param):
            prob = jnp.stack([1.0 - prob, prob], axis=-1)
            sample = jax_discrete(key, prob, param)
            return sample
        
        return _jax_wrapped_calc_bernoulli_approx, jax_param
    
    def poisson(self, logic):
        
        def _jax_wrapped_calc_poisson_exact(key, rate, param):
            return random.poisson(key=key, lam=rate, dtype=logic.INT)
        
        return _jax_wrapped_calc_poisson_exact, None
    
    def geometric(self, logic):
        if logic.verbose:
            raise_warning('Using the replacement rule: '
                          'Geometric(p) --> floor(log(U) / log(1 - p)) + 1')
            
        jax_floor, jax_param = logic.floor()
            
        def _jax_wrapped_calc_geometric_approx(key, prob, param):
            U = random.uniform(key=key, shape=jnp.shape(prob), dtype=logic.REAL)
            sample = jax_floor(jnp.log(U) / jnp.log(1.0 - prob), param) + 1
            return sample
        
        return _jax_wrapped_calc_geometric_approx, jax_param


class GumbelSoftmax(RandomSampling):
    '''Random sampling of discrete variables using Gumbel-softmax trick.'''
    
    def discrete(self, logic):
        if logic.verbose:
            raise_warning('Using the replacement rule: '
                          'Discrete(p) --> Gumbel-softmax(p)')
        
        jax_argmax, jax_param = logic.argmax()
        
        def _jax_wrapped_calc_discrete_gumbel_softmax(key, prob, param):
            Gumbel01 = random.gumbel(key=key, shape=prob.shape, dtype=logic.REAL)
            sample = Gumbel01 + jnp.log(prob + logic.eps)
            sample = jax_argmax(sample, axis=-1, param=param)
            return sample
        
        return _jax_wrapped_calc_discrete_gumbel_softmax, jax_param
    

class Determinization(RandomSampling):
    '''Random sampling of variables using their deterministic mean estimate.'''
    
    def discrete(self, logic):
        if logic.verbose:
            raise_warning('Using the replacement rule: '
                          'Discrete(p) --> sum(i * p[i])')
        
        def _jax_wrapped_calc_discrete_determinized(key, prob, param):
            literals = FuzzyLogic.enumerate_literals(prob.shape, axis=-1)
            sample = jnp.sum(literals * prob, axis=-1)
            return sample
        
        return _jax_wrapped_calc_discrete_determinized, None
    
    def poisson(self, logic):
        if logic.verbose:
            raise_warning('Using the replacement rule: Poisson(rate) --> rate')
            
        def _jax_wrapped_calc_poisson_determinized(key, rate, param):
            return rate
        
        return _jax_wrapped_calc_poisson_determinized, None
    
    def geometric(self, logic):
        if logic.verbose:
            raise_warning('Using the replacement rule: Geometric(p) --> 1 / p')
            
        def _jax_wrapped_calc_geometric_determinized(key, prob, param):
            sample = 1.0 / prob
            return sample
        
        return _jax_wrapped_calc_geometric_determinized, None


# ===========================================================================
# FUZZY LOGIC
#
# ===========================================================================

class FuzzyLogic:
    '''A class representing fuzzy logic in JAX.
    
    Functionality can be customized by either providing a tnorm as parameters, 
    or by overriding its methods.
    '''
    
    def __init__(self, tnorm: TNorm=ProductTNorm(),
                 complement: Complement=StandardComplement(),
                 comparison: Comparison=SigmoidComparison(),
                 sampling: RandomSampling=GumbelSoftmax(),
                 weight: float=10.0,
                 debias: Optional[Set[str]]=None,
                 eps: float=1e-15,
                 verbose: bool=False,
                 use64bit: bool=False) -> None:
        '''Creates a new fuzzy logic in Jax.
        
        :param tnorm: fuzzy operator for logical AND
        :param complement: fuzzy operator for logical NOT
        :param comparison: fuzzy operator for comparisons (>, >=, <, ==, ~=, ...)
        :param sampling: random sampling of non-reparameterizable distributions
        :param weight: a sharpness parameter for sigmoid and softmax activations
        :param debias: which functions to de-bias approximate on forward pass
        :param eps: small positive float to mitigate underflow
        :param verbose: whether to dump replacements and other info to console
        :param use64bit: whether to perform arithmetic in 64 bit
        '''
        self.tnorm = tnorm
        self.complement = complement
        self.comparison = comparison
        self.sampling = sampling
        self.weight = float(weight)
        if debias is None:
            debias = set()
        self.debias = debias
        self.eps = eps
        self.verbose = verbose
        self.set_use64bit(use64bit)
    
    def set_use64bit(self, use64bit: bool) -> None:
        self.use64bit = use64bit
        if use64bit:
            self.REAL = jnp.float64
            self.INT = jnp.int64
            jax.config.update('jax_enable_x64', True)
        else:
            self.REAL = jnp.float32
            self.INT = jnp.int32
            jax.config.update('jax_enable_x64', False)
        
    def summarize_hyperparameters(self) -> None:
        print(f'model relaxation:\n'
              f'    tnorm         ={type(self.tnorm).__name__}\n'
              f'    complement    ={type(self.complement).__name__}\n'
              f'    comparison    ={type(self.comparison).__name__}\n'
              f'    sampling      ={type(self.sampling).__name__}\n'
              f'    sigmoid_weight={self.weight}\n'
              f'    cpfs_to_debias={self.debias}\n'
              f'    underflow_tol ={self.eps}\n'
              f'    use_64_bit    ={self.use64bit}')
        
    # ===========================================================================
    # logical operators
    # ===========================================================================
     
    def logical_and(self):
        if self.verbose:
            raise_warning('Using the replacement rule: a ^ b --> tnorm(a, b).')
        
        _and = self.tnorm.norm
        
        def _jax_wrapped_calc_and_approx(a, b, param):
            return _and(a, b)
        
        return _jax_wrapped_calc_and_approx, None
    
    def logical_not(self):
        if self.verbose:
            raise_warning('Using the replacement rule: ~a --> complement(a)')
        
        _not = self.complement
        
        def _jax_wrapped_calc_not_approx(x, param):
            return _not(x)
        
        return _jax_wrapped_calc_not_approx, None
    
    def logical_or(self):
        if self.verbose:
            raise_warning('Using the replacement rule: a | b --> tconorm(a, b).')
            
        _not = self.complement
        _and = self.tnorm.norm
        
        def _jax_wrapped_calc_or_approx(a, b, param):
            return _not(_and(_not(a), _not(b)))
        
        return _jax_wrapped_calc_or_approx, None

    def xor(self):
        if self.verbose:
            raise_warning('Using the replacement rule: '
                          'a ~ b --> (a | b) ^ (a ^ b).')
        
        _not = self.complement
        _and = self.tnorm.norm
        
        def _jax_wrapped_calc_xor_approx(a, b, param):
            _or = _not(_and(_not(a), _not(b)))
            return _and(_or(a, b), _not(_and(a, b)))
        
        return _jax_wrapped_calc_xor_approx, None
        
    def implies(self):
        if self.verbose:
            raise_warning('Using the replacement rule: a => b --> ~a ^ b')
        
        _not = self.complement
        _and = self.tnorm.norm
        
        def _jax_wrapped_calc_implies_approx(a, b, param):
            return _not(_and(a, _not(b)))
        
        return _jax_wrapped_calc_implies_approx, None
    
    def equiv(self):
        if self.verbose:
            raise_warning('Using the replacement rule: '
                          'a <=> b --> (a => b) ^ (b => a)')
        
        _not = self.complement
        _and = self.tnorm.norm
        
        def _jax_wrapped_calc_equiv_approx(a, b, param):
            atob = _not(_and(a, _not(b)))
            btoa = _not(_and(b, _not(a)))
            return _and(atob, btoa)
        
        return _jax_wrapped_calc_equiv_approx, None
    
    def forall(self):
        if self.verbose:
            raise_warning('Using the replacement rule: '
                          'forall(a) --> a[1] ^ a[2] ^ ...')
        
        _forall = self.tnorm.norms
        
        def _jax_wrapped_calc_forall_approx(x, axis, param):
            return _forall(x, axis=axis)
        
        return _jax_wrapped_calc_forall_approx, None
    
    def exists(self):
        _not = self.complement
        jax_forall, jax_param = self.forall()
        
        def _jax_wrapped_calc_exists_approx(x, axis, param):
            return _not(jax_forall(_not(x), axis, param))
        
        return _jax_wrapped_calc_exists_approx, jax_param
    
    # ===========================================================================
    # comparison operators
    # ===========================================================================
     
    def greater_equal(self):
        if self.verbose:
            raise_warning('Using the replacement rule: '
                          'a >= b --> comparison.greater_equal(a, b)')
        
        greater_equal_op = self.comparison.greater_equal
        debias = 'greater_equal' in self.debias
        
        def _jax_wrapped_calc_geq_approx(a, b, param):
            sample = greater_equal_op(a, b, param)
            if debias:
                hard_sample = jnp.greater_equal(a, b)
                sample += jax.lax.stop_gradient(hard_sample - sample)
            return sample
        
        tags = ('weight', 'greater_equal')
        new_param = (tags, self.weight)
        return _jax_wrapped_calc_geq_approx, new_param
    
    def greater(self):
        if self.verbose:
            raise_warning('Using the replacement rule: '
                          'a > b --> comparison.greater(a, b)')
        
        greater_op = self.comparison.greater
        debias = 'greater' in self.debias
        
        def _jax_wrapped_calc_gre_approx(a, b, param):
            sample = greater_op(a, b, param)
            if debias:
                hard_sample = jnp.greater(a, b)
                sample += jax.lax.stop_gradient(hard_sample - sample)
            return sample
        
        tags = ('weight', 'greater')
        new_param = (tags, self.weight)
        return _jax_wrapped_calc_gre_approx, new_param
    
    def less_equal(self):
        jax_geq, jax_param = self.greater_equal()
        
        def _jax_wrapped_calc_leq_approx(a, b, param):
            return jax_geq(-a, -b, param)
        
        return _jax_wrapped_calc_leq_approx, jax_param
    
    def less(self):
        jax_gre, jax_param = self.greater()

        def _jax_wrapped_calc_less_approx(a, b, param):
            return jax_gre(-a, -b, param)
        
        return _jax_wrapped_calc_less_approx, jax_param

    def equal(self):
        if self.verbose:
            raise_warning('Using the replacement rule: '
                          'a == b --> comparison.equal(a, b)')
        
        equal_op = self.comparison.equal
        debias = 'equal' in self.debias
        
        def _jax_wrapped_calc_equal_approx(a, b, param):
            sample = equal_op(a, b, param)
            if debias:
                hard_sample = jnp.equal(a, b)
                sample += jax.lax.stop_gradient(hard_sample - sample)
            return sample
        
        tags = ('weight', 'equal')
        new_param = (tags, self.weight)
        return _jax_wrapped_calc_equal_approx, new_param
    
    def not_equal(self):
        _not = self.complement
        jax_eq, jax_param = self.equal()
        
        def _jax_wrapped_calc_neq_approx(a, b, param):
            return _not(jax_eq(a, b, param))

        return _jax_wrapped_calc_neq_approx, jax_param
        
    # ===========================================================================
    # special functions
    # ===========================================================================
     
    def sgn(self):
        if self.verbose:
            raise_warning('Using the replacement rule: sgn(x) --> tanh(x)')
            
        debias = 'sgn' in self.debias
        
        def _jax_wrapped_calc_sgn_approx(x, param):
            sample = jnp.tanh(param * x)
            if debias:
                hard_sample = jnp.sign(x)
                sample += jax.lax.stop_gradient(hard_sample - sample)
            return sample
        
        tags = ('weight', 'sgn')
        new_param = (tags, self.weight)
        return _jax_wrapped_calc_sgn_approx, new_param
    
    def floor(self):
        if self.verbose:
            raise_warning('Using the replacement rule: '
                          'floor(x) --> x - atan(-1.0 / tan(pi * x)) / pi - 0.5')
        
        def _jax_wrapped_calc_floor_approx(x, param):
            sawtooth_part = jnp.arctan(-1.0 / jnp.tan(x * jnp.pi)) / jnp.pi + 0.5
            sample = x - jax.lax.stop_gradient(sawtooth_part)
            return sample
        
        return _jax_wrapped_calc_floor_approx, None
        
    def ceil(self):
        jax_floor, jax_param = self.floor()
        
        def _jax_wrapped_calc_ceil_approx(x, param):
            return -jax_floor(-x, param) 
        
        return _jax_wrapped_calc_ceil_approx, jax_param
    
    def round(self):
        if self.verbose:
            raise_warning('Using the replacement rule: round(x) --> soft_round(x)')
        
        debias = 'round' in self.debias
        
        def _jax_wrapped_calc_round_approx(x, param):
            m = jnp.floor(x) + 0.5
            sample = m + 0.5 * jnp.tanh(param * (x - m)) / jnp.tanh(param / 2.0)    
            if debias:
                hard_sample = jnp.round(x)
                sample += jax.lax.stop_gradient(hard_sample - sample)
            return sample
        
        tags = ('weight', 'round')
        new_param = (tags, self.weight)
        return _jax_wrapped_calc_round_approx, new_param
    
    def mod(self):
        jax_floor, jax_param = self.floor()
        
        def _jax_wrapped_calc_mod_approx(x, y, param):
            return x - y * jax_floor(x / y, param)
        
        return _jax_wrapped_calc_mod_approx, jax_param
    
    def div(self):
        jax_floor, jax_param = self.floor()
        
        def _jax_wrapped_calc_div_approx(x, y, param):
            return jax_floor(x / y, param)
        
        return _jax_wrapped_calc_div_approx, jax_param
    
    def sqrt(self):
        if self.verbose:
            raise_warning('Using the replacement rule: sqrt(x) --> sqrt(x + eps)')
        
        def _jax_wrapped_calc_sqrt_approx(x, param):
            return jnp.sqrt(x + self.eps)
        
        return _jax_wrapped_calc_sqrt_approx, None
    
    # ===========================================================================
    # indexing
    # ===========================================================================
     
    @staticmethod
    def enumerate_literals(shape, axis):
        literals = jnp.arange(shape[axis])
        literals = literals[(...,) + (jnp.newaxis,) * (len(shape) - 1)]
        literals = jnp.moveaxis(literals, source=0, destination=axis)
        literals = jnp.broadcast_to(literals, shape=shape)
        return literals
    
    def argmax(self):
        if self.verbose:
            raise_warning('Using the replacement rule: '
                          'argmax(x) --> sum(i * softmax(x[i]))')
            
        debias = 'argmax' in self.debias
        
        def _jax_wrapped_calc_argmax_approx(x, axis, param):
            literals = FuzzyLogic.enumerate_literals(x.shape, axis=axis)
            soft_max = jax.nn.softmax(param * x, axis=axis)
            sample = jnp.sum(literals * soft_max, axis=axis)
            if debias:
                hard_sample = jnp.argmax(x, axis=axis)
                sample += jax.lax.stop_gradient(hard_sample - sample)
            return sample
        
        tags = ('weight', 'argmax')
        new_param = (tags, self.weight)
        return _jax_wrapped_calc_argmax_approx, new_param
    
    def argmin(self):
        jax_argmax, jax_param = self.argmax()
        
        def _jax_wrapped_calc_argmin_approx(x, axis, param):
            return jax_argmax(-x, axis, param)
        
        return _jax_wrapped_calc_argmin_approx, jax_param
    
    # ===========================================================================
    # control flow
    # ===========================================================================
     
    def control_if(self):
        if self.verbose:
            raise_warning('Using the replacement rule: '
                          'if c then a else b --> c * a + (1 - c) * b')
        
        debias = 'if' in self.debias
        
        def _jax_wrapped_calc_if_approx(c, a, b, param):
            sample = c * a + (1.0 - c) * b
            if debias:
                hard_sample = jnp.where(c > 0.5, a, b)
                sample += jax.lax.stop_gradient(hard_sample - sample)
            return sample
        
        return _jax_wrapped_calc_if_approx, None
    
    def control_switch(self):
        if self.verbose:
            raise_warning('Using the replacement rule: '
                          'switch(pred) { cases } --> '
                          'sum(cases[i] * softmax(-abs(pred - i)))')   
            
        debias = 'switch' in self.debias
        
        def _jax_wrapped_calc_switch_approx(pred, cases, param):
            literals = FuzzyLogic.enumerate_literals(cases.shape, axis=0)
            pred = jnp.broadcast_to(pred[jnp.newaxis, ...], shape=cases.shape)
            proximity = -jnp.square(pred - literals)
            soft_case = jax.nn.softmax(param * proximity, axis=0)
            sample = jnp.sum(cases * soft_case, axis=0)
            if debias:
                hard_case = jnp.argmax(proximity, axis=0)[jnp.newaxis, ...]      
                hard_sample = jnp.take_along_axis(cases, hard_case, axis=0)[0, ...]
                sample += jax.lax.stop_gradient(hard_sample - sample)
            return sample
        
        tags = ('weight', 'switch')
        new_param = (tags, self.weight)
        return _jax_wrapped_calc_switch_approx, new_param
    
    # ===========================================================================
    # random variables
    # ===========================================================================
     
    def discrete(self):
        return self.sampling.discrete(self)
    
    def bernoulli(self):
        return self.sampling.bernoulli(self)
    
    def poisson(self):
        return self.sampling.poisson(self)
    
    def geometric(self):
        return self.sampling.geometric(self)


# ===========================================================================
# UNIT TESTS
#
# ===========================================================================

logic = FuzzyLogic()
w = 100.0


def _test_logical():
    print('testing logical')
    _and, _ = logic.logical_and()
    _not, _ = logic.logical_not()
    _gre, _ = logic.greater()
    _or, _ = logic.logical_or()
    _if, _ = logic.control_if()
    
    # https://towardsdatascience.com/emulating-logical-gates-with-a-neural-network-75c229ec4cc9
    def test_logic(x1, x2):
        q1 = _and(_gre(x1, 0, w), _gre(x2, 0, w), w)
        q2 = _and(_not(_gre(x1, 0, w), w), _not(_gre(x2, 0, w), w), w)
        cond = _or(q1, q2, w)
        pred = _if(cond, +1, -1, w)
        return pred
    
    x1 = jnp.asarray([1, 1, -1, -1, 0.1, 15, -0.5]).astype(float)
    x2 = jnp.asarray([1, -1, 1, -1, 10, -30, 6]).astype(float)
    print(test_logic(x1, x2))


def _test_indexing():
    print('testing indexing')
    _argmax, _ = logic.argmax()
    _argmin, _ = logic.argmin()

    def argmaxmin(x):
        amax = _argmax(x, 0, w)
        amin = _argmin(x, 0, w)
        return amax, amin
        
    values = jnp.asarray([2., 3., 5., 4.9, 4., 1., -1., -2.])
    amax, amin = argmaxmin(values)
    print(amax)
    print(amin)


def _test_control():
    print('testing control')
    _switch, _ = logic.control_switch()
    
    pred = jnp.asarray(jnp.linspace(0, 2, 10))
    case1 = jnp.asarray([-10.] * 10)
    case2 = jnp.asarray([1.5] * 10)
    case3 = jnp.asarray([10.] * 10)
    cases = jnp.asarray([case1, case2, case3])
    print(_switch(pred, cases, w))


def _test_random():
    print('testing random')
    key = random.PRNGKey(42)
    _bernoulli, _ = logic.bernoulli()
    _discrete, _ = logic.discrete()
    _geometric, _ = logic.geometric()
    
    def bern(n):
        prob = jnp.asarray([0.3] * n)
        sample = _bernoulli(key, prob, w)
        return sample
    
    samples = bern(50000)
    print(jnp.mean(samples))
    
    def disc(n):
        prob = jnp.asarray([0.1, 0.4, 0.5])
        prob = jnp.tile(prob, (n, 1))
        sample = _discrete(key, prob, w)
        return sample
        
    samples = disc(50000)
    samples = jnp.round(samples)
    print([jnp.mean(samples == i) for i in range(3)])
    
    def geom(n):
        prob = jnp.asarray([0.3] * n)
        sample = _geometric(key, prob, w)
        return sample
    
    samples = geom(50000)
    print(jnp.mean(samples))
    

def _test_rounding():
    print('testing rounding')
    _floor, _ = logic.floor()
    _ceil, _ = logic.ceil()
    _round, _ = logic.round()
    _mod, _ = logic.mod()
    
    x = jnp.asarray([2.1, 0.6, 1.99, -2.01, -3.2, -0.1, -1.01, 23.01, -101.99, 200.01])
    print(_floor(x, w))
    print(_ceil(x, w))
    print(_round(x, w))
    print(_mod(x, 2.0, w))


if __name__ == '__main__':
    _test_logical()
    _test_indexing()
    _test_control()
    _test_random()
    _test_rounding()
    
