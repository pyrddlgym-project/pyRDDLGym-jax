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

import termcolor
from typing import Any, Dict, Optional, Set, Tuple, Union

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random
import jax.scipy as scipy 

from pyRDDLGym_jax.core.compiler import JaxRDDLCompiler


def enumerate_literals(shape: Tuple[int, ...], axis: int, dtype: type=jnp.int32) -> jnp.ndarray:
    literals = jnp.arange(shape[axis], dtype=dtype)
    literals = literals[(...,) + (jnp.newaxis,) * (len(shape) - 1)]
    literals = jnp.moveaxis(literals, source=0, destination=axis)
    literals = jnp.broadcast_to(literals, shape=shape)
    return literals


# branching sigmoid to help reduce numerical issues
def stable_sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    value = jax.nn.sigmoid(x)
    return value


# branching tanh to help reduce numerical issues
@jax.custom_jvp
def stable_tanh(x: jnp.ndarray) -> jnp.ndarray:
    ax = jnp.abs(x)
    small = jnp.where(
        ax < 20.,
        jnp.expm1(2. * ax) / (jnp.expm1(2. * ax) + 2.),
        1. - 2. * jnp.exp(-2. * ax)
    )
    return jnp.sign(x) * small


@stable_tanh.defjvp
def stable_tanh_jvp(primals, tangents):
    (x,), (x_dot,) = primals, tangents
    t = stable_tanh(x)
    tangent_out = x_dot * (1. - t * t)
    return t, tangent_out


# it seems JAX uses the stability trick already
def stable_softmax_weight_sum(logits: jnp.ndarray, values: jnp.ndarray, axis: int) -> jnp.ndarray:
    return jnp.sum(values * jax.nn.softmax(logits), axis=axis)


class JaxRDDLCompilerWithGrad(JaxRDDLCompiler):
    '''Compiles a RDDL AST representation to an equivalent JAX representation. 
    Unlike its parent class, this class treats all fluents as real-valued, and
    replaces all mathematical operations by equivalent ones with a well defined 
    (e.g. non-zero) gradient where appropriate. 
    '''
    
    def __init__(self, *args,
                 cpfs_without_grad: Optional[Set[str]]=None,
                 print_warnings: bool=True, 
                 **kwargs) -> None:
        '''Creates a new RDDL to Jax compiler, where operations that are not
        differentiable are converted to approximate forms that have defined gradients.
        
        :param *args: arguments to pass to base compiler
        :param cpfs_without_grad: which CPFs do not have gradients (use straight
        through gradient trick)
        :param print_warnings: whether to print warnings
        :param *kwargs: keyword arguments to pass to base compiler
        '''
        super(JaxRDDLCompilerWithGrad, self).__init__(*args, **kwargs)
        
        if cpfs_without_grad is None:
            cpfs_without_grad = set()
        self.cpfs_without_grad = cpfs_without_grad
        self.print_warnings = print_warnings
        
        # actions and CPFs must be continuous
        pvars_cast = set()
        for (var, values) in self.init_values.items():
            self.init_values[var] = np.asarray(values, dtype=self.REAL) 
            if not np.issubdtype(np.result_type(values), np.floating):
                pvars_cast.add(var)
        if self.print_warnings and pvars_cast:
            print(termcolor.colored(
                f'[INFO] Compiler will cast pvars {pvars_cast} to float.', 'dark_grey'))
        
    def get_kwargs(self) -> Dict[str, Any]:
        kwargs = super().get_kwargs()
        kwargs['cpfs_without_grad'] = self.cpfs_without_grad
        kwargs['print_warnings'] = self.print_warnings
        return kwargs

    def _jax_stop_grad(self, jax_expr):        
        def _jax_wrapped_stop_grad(fls, nfls, params, key):
            sample, key, error, params = jax_expr(fls, nfls, params, key)
            sample = jax.lax.stop_gradient(sample)
            return sample, key, error, params
        return _jax_wrapped_stop_grad
        
    def _compile_cpfs(self, aux):

        # cpfs will all be cast to float
        cpfs_cast = set()   
        jax_cpfs = {}
        for (_, cpfs) in self.levels.items():
            for cpf in cpfs:
                _, expr = self.rddl.cpfs[cpf]
                jax_cpfs[cpf] = self._jax(expr, aux, dtype=self.REAL)
                if self.rddl.variable_ranges[cpf] != 'real':
                    cpfs_cast.add(cpf)
                if cpf in self.cpfs_without_grad:
                    jax_cpfs[cpf] = self._jax_stop_grad(jax_cpfs[cpf])
                    
        if self.print_warnings and cpfs_cast:
            print(termcolor.colored(
                f'[INFO] Compiler will cast CPFs {cpfs_cast} to float.', 'dark_grey'))
        if self.print_warnings and self.cpfs_without_grad:
            print(termcolor.colored(
                f'[INFO] Gradient disabled for CPFs {self.cpfs_without_grad}.', 'dark_grey'))
 
        return jax_cpfs
    
    def _jax_unary_with_param(self, jax_expr, jax_op):
        def _jax_wrapped_unary_op_with_param(fls, nfls, params, key):
            sample, key, err, params = jax_expr(fls, nfls, params, key)
            sample = self.ONE * sample
            sample, params = jax_op(sample, params)
            return sample, key, err, params
        return _jax_wrapped_unary_op_with_param
    
    def _jax_binary_with_param(self, jax_lhs, jax_rhs, jax_op):
        def _jax_wrapped_binary_op_with_param(fls, nfls, params, key):
            sample1, key, err1, params = jax_lhs(fls, nfls, params, key)
            sample2, key, err2, params = jax_rhs(fls, nfls, params, key)
            sample1 = self.ONE * sample1
            sample2 = self.ONE * sample2
            sample, params = jax_op(sample1, sample2, params)
            err = err1 | err2
            return sample, key, err, params
        return _jax_wrapped_binary_op_with_param
    
    def _jax_unary_helper_with_param(self, expr, aux, jax_op):
        JaxRDDLCompilerWithGrad._check_num_args(expr, 1)
        arg, = expr.args
        jax_arg = self._jax(arg, aux)
        return self._jax_unary_with_param(jax_arg, jax_op)
    
    def _jax_binary_helper_with_param(self, expr, aux, jax_op):
        JaxRDDLCompilerWithGrad._check_num_args(expr, 2)
        lhs, rhs = expr.args
        jax_lhs = self._jax(lhs, aux)
        jax_rhs = self._jax(rhs, aux)
        return self._jax_binary_with_param(jax_lhs, jax_rhs, jax_op)

    def _jax_kron(self, expr, aux):
        aux['overriden'][expr.id] = __class__.__name__
        arg, = expr.args
        arg = self._jax(arg, aux)
        return arg


# ===============================================================================
# relational relaxations
# ===============================================================================

# https://arxiv.org/abs/2110.05651
class SigmoidRelational(JaxRDDLCompilerWithGrad):
    '''Comparison operations approximated using sigmoid functions.'''
    
    def __init__(self, *args, sigmoid_weight: float=10., 
                 use_sigmoid_ste: bool=True, use_tanh_ste: bool=True, 
                 **kwargs) -> None:
        super(SigmoidRelational, self).__init__(*args, **kwargs)
        self.sigmoid_weight = float(sigmoid_weight)
        self.use_sigmoid_ste = use_sigmoid_ste
        self.use_tanh_ste = use_tanh_ste
    
    def get_kwargs(self) -> Dict[str, Any]:
        kwargs = super().get_kwargs()
        kwargs['sigmoid_weight'] = self.sigmoid_weight
        kwargs['use_sigmoid_ste'] = self.use_sigmoid_ste
        kwargs['use_tanh_ste'] = self.use_tanh_ste
        return kwargs

    def _jax_greater(self, expr, aux):
        if not self.traced.cached_is_fluent(expr):
            return JaxRDDLCompilerWithGrad._jax_greater(self, expr, aux)
        id_ = expr.id
        aux['params'][id_] = self.sigmoid_weight
        aux['overriden'][id_] = __class__.__name__
        def greater_op(x, y, params):
            sample = stable_sigmoid(params[id_] * (x - y))
            if self.use_sigmoid_ste:
                sample = sample + jax.lax.stop_gradient(jnp.greater(x, y) - sample)
            return sample, params
        return self._jax_binary_helper_with_param(expr, aux, greater_op)
    
    def _jax_greater_equal(self, expr, aux):
        if not self.traced.cached_is_fluent(expr):
            return JaxRDDLCompilerWithGrad._jax_greater_equal(self, expr, aux)
        id_ = expr.id
        aux['params'][id_] = self.sigmoid_weight
        aux['overriden'][id_] = __class__.__name__
        def greater_equal_op(x, y, params):
            sample = stable_sigmoid(params[id_] * (x - y))
            if self.use_sigmoid_ste:
                sample = sample + jax.lax.stop_gradient(jnp.greater_equal(x, y) - sample)
            return sample, params
        return self._jax_binary_helper_with_param(expr, aux, greater_equal_op)
    
    def _jax_less(self, expr, aux):
        if not self.traced.cached_is_fluent(expr):
            return JaxRDDLCompilerWithGrad._jax_less(self, expr, aux)
        id_ = expr.id
        aux['params'][id_] = self.sigmoid_weight
        aux['overriden'][id_] = __class__.__name__
        def less_op(x, y, params):
            sample = stable_sigmoid(params[id_] * (y - x))
            if self.use_sigmoid_ste:
                sample = sample + jax.lax.stop_gradient(jnp.less(x, y) - sample)
            return sample, params
        return self._jax_binary_helper_with_param(expr, aux, less_op)
    
    def _jax_less_equal(self, expr, aux):
        if not self.traced.cached_is_fluent(expr):
            return JaxRDDLCompilerWithGrad._jax_less_equal(self, expr, aux)
        id_ = expr.id
        aux['params'][id_] = self.sigmoid_weight
        aux['overriden'][id_] = __class__.__name__
        def less_equal_op(x, y, params):
            sample = stable_sigmoid(params[id_] * (y - x))
            if self.use_sigmoid_ste:
                sample = sample + jax.lax.stop_gradient(jnp.less_equal(x, y) - sample)
            return sample, params
        return self._jax_binary_helper_with_param(expr, aux, less_equal_op)
    
    def _jax_equal(self, expr, aux):
        if not self.traced.cached_is_fluent(expr):
            return JaxRDDLCompilerWithGrad._jax_equal(self, expr, aux)
        id_ = expr.id
        aux['params'][id_] = self.sigmoid_weight
        aux['overriden'][id_] = __class__.__name__
        def equal_op(x, y, params):
            sample = 1. - jnp.square(stable_tanh(params[id_] * (y - x)))
            if self.use_tanh_ste:
                sample = sample + jax.lax.stop_gradient(jnp.equal(x, y) - sample)
            return sample, params
        return self._jax_binary_helper_with_param(expr, aux, equal_op)
    
    def _jax_not_equal(self, expr, aux):
        if not self.traced.cached_is_fluent(expr):
            return JaxRDDLCompilerWithGrad._jax_not_equal(self, expr, aux)
        id_ = expr.id
        aux['params'][id_] = self.sigmoid_weight
        aux['overriden'][id_] = __class__.__name__
        def not_equal_op(x, y, params):
            sample = jnp.square(stable_tanh(params[id_] * (y - x)))
            if self.use_tanh_ste:
                sample = sample + jax.lax.stop_gradient(jnp.not_equal(x, y) - sample)
            return sample, params
        return self._jax_binary_helper_with_param(expr, aux, not_equal_op)
    
    def _jax_sgn(self, expr, aux):
        if not self.traced.cached_is_fluent(expr):
            return JaxRDDLCompilerWithGrad._jax_sgn(self, expr, aux)
        id_ = expr.id
        aux['params'][id_] = self.sigmoid_weight
        aux['overriden'][id_] = __class__.__name__
        def sgn_op(x, params):
            sample = stable_tanh(params[id_] * x)
            if self.use_tanh_ste:
                sample = sample + jax.lax.stop_gradient(jnp.sign(x) - sample)
            return sample, params
        return self._jax_unary_helper_with_param(expr, aux, sgn_op)
    

class SoftmaxArgmax(JaxRDDLCompilerWithGrad):
    '''Argmin/argmax operations approximated using softmax functions.'''

    def __init__(self, *args, argmax_weight: float=10., 
                 use_argmax_ste: bool=True, **kwargs) -> None:
        super(SoftmaxArgmax, self).__init__(*args, **kwargs)
        self.argmax_weight = float(argmax_weight)
        self.use_argmax_ste = use_argmax_ste
    
    def get_kwargs(self) -> Dict[str, Any]:
        kwargs = super().get_kwargs()
        kwargs['argmax_weight'] = self.argmax_weight
        kwargs['use_argmax_ste'] = self.use_argmax_ste
        return kwargs

    @staticmethod
    def soft_argmax(x: jnp.ndarray, w: float, axis: int, dtype: type=jnp.int32) -> jnp.ndarray:
        literals = enumerate_literals(jnp.shape(x), axis=axis, dtype=dtype)
        return stable_softmax_weight_sum(w * x, literals, axis=axis)

    def _jax_argmax(self, expr, aux):
        if not self.traced.cached_is_fluent(expr):
            return JaxRDDLCompilerWithGrad._jax_argmax(self, expr, aux)
        id_ = expr.id
        aux['params'][id_] = self.argmax_weight
        aux['overriden'][id_] = __class__.__name__
        arg = expr.args[-1]
        _, axis = self.traced.cached_sim_info(expr)   
        jax_expr = self._jax(arg, aux) 
        def argmax_op(x, params):
            sample = self.soft_argmax(x, w=params[id_], axis=axis, dtype=self.INT)
            if self.use_argmax_ste:
                hard_sample = jnp.argmax(x, axis=axis)
                sample = sample + jax.lax.stop_gradient(hard_sample - sample)
            return sample, params
        return self._jax_unary_with_param(jax_expr, argmax_op)

    def _jax_argmin(self, expr, aux):
        if not self.traced.cached_is_fluent(expr):
            return JaxRDDLCompilerWithGrad._jax_argmin(self, expr, aux)
        id_ = expr.id
        aux['params'][id_] = self.argmax_weight
        aux['overriden'][id_] = __class__.__name__
        arg = expr.args[-1]
        _, axis = self.traced.cached_sim_info(expr)   
        jax_expr = self._jax(arg, aux) 
        def argmin_op(x, params):
            sample = self.soft_argmax(-x, w=params[id_], axis=axis, dtype=self.INT)
            if self.use_argmax_ste:
                hard_sample = jnp.argmax(-x, axis=axis)
                sample = sample + jax.lax.stop_gradient(hard_sample - sample)
            return sample, params
        return self._jax_unary_with_param(jax_expr, argmin_op)
        

# ===============================================================================
# logical relaxations
# ===============================================================================

class ProductNormLogical(JaxRDDLCompilerWithGrad):
    '''Product t-norm given by the expression (x, y) -> x * y.'''
    
    def __init__(self, *args, use_logic_ste: bool=False, **kwargs) -> None:
        super(ProductNormLogical, self).__init__(*args, **kwargs)
        self.use_logic_ste = use_logic_ste

    def get_kwargs(self) -> Dict[str, Any]:
        kwargs = super().get_kwargs()
        kwargs['use_logic_ste'] = self.use_logic_ste
        return kwargs

    def _jax_not(self, expr, aux):
        if not self.traced.cached_is_fluent(expr):
            return JaxRDDLCompilerWithGrad._jax_not(self, expr, aux)
        aux['overriden'][expr.id] = __class__.__name__
        def not_op(x):
            sample = 1. - x
            if self.use_logic_ste:
                hard_sample = jnp.asarray(x <= 0.5, dtype=self.REAL)
                sample = sample + jax.lax.stop_gradient(hard_sample - sample)
            return sample
        return self._jax_unary_helper(expr, aux, not_op)

    def _jax_and(self, expr, aux):
        if not self.traced.cached_is_fluent(expr):
            return JaxRDDLCompilerWithGrad._jax_and(self, expr, aux)
        aux['overriden'][expr.id] = __class__.__name__
        def and_op(x, y):
            sample = jnp.multiply(x, y)
            if self.use_logic_ste:
                hard_sample = jnp.asarray(jnp.logical_and(x > 0.5, y > 0.5), dtype=self.REAL)
                sample = sample + jax.lax.stop_gradient(hard_sample - sample)
            return sample
        return self._jax_nary_helper(expr, aux, and_op)

    def _jax_or(self, expr, aux):
        if not self.traced.cached_is_fluent(expr):
            return JaxRDDLCompilerWithGrad._jax_or(self, expr, aux)
        aux['overriden'][expr.id] = __class__.__name__
        def or_op(x, y):
            sample = 1. - (1. - x) * (1. - y)
            if self.use_logic_ste:
                hard_sample = jnp.asarray(jnp.logical_or(x > 0.5, y > 0.5), dtype=self.REAL)
                sample = sample + jax.lax.stop_gradient(hard_sample - sample)
            return sample
        return self._jax_nary_helper(expr, aux, or_op)

    def _jax_xor(self, expr, aux):
        if not self.traced.cached_is_fluent(expr):
            return JaxRDDLCompilerWithGrad._jax_xor(self, expr, aux)
        aux['overriden'][expr.id] = __class__.__name__
        def xor_op(x, y):
            sample = (1. - (1. - x) * (1. - y)) * (1. - x * y)
            if self.use_logic_ste:
                hard_sample = jnp.asarray(jnp.logical_xor(x > 0.5, y > 0.5), dtype=self.REAL)
                sample = sample + jax.lax.stop_gradient(hard_sample - sample)
            return sample
        return self._jax_binary_helper(expr, aux, xor_op)

    def _jax_implies(self, expr, aux):
        if not self.traced.cached_is_fluent(expr):
            return JaxRDDLCompilerWithGrad._jax_implies(self, expr, aux)
        aux['overriden'][expr.id] = __class__.__name__
        def implies_op(x, y):
            sample = 1. - x * (1. - y)
            if self.use_logic_ste:
                hard_sample = jnp.asarray(jnp.logical_or(x <= 0.5, y > 0.5), dtype=self.REAL)
                sample = sample + jax.lax.stop_gradient(hard_sample - sample)
            return sample
        return self._jax_binary_helper(expr, aux, implies_op)

    def _jax_equiv(self, expr, aux):
        if not self.traced.cached_is_fluent(expr):
            return JaxRDDLCompilerWithGrad._jax_equiv(self, expr, aux)
        aux['overriden'][expr.id] = __class__.__name__
        def equiv_op(x, y):
            sample = (1. - x * (1. - y)) * (1. - y * (1. - x))
            if self.use_logic_ste:
                hard_sample = jnp.logical_and(
                    jnp.logical_or(x <= 0.5, y > 0.5), jnp.logical_or(y <= 0.5, x > 0.5))
                hard_sample = jnp.asarray(hard_sample, dtype=self.REAL)
                sample = sample + jax.lax.stop_gradient(hard_sample - sample)
            return sample
        return self._jax_binary_helper(expr, aux, equiv_op)
    
    def _jax_forall(self, expr, aux):
        if not self.traced.cached_is_fluent(expr):
            return JaxRDDLCompilerWithGrad._jax_forall(self, expr, aux)
        aux['overriden'][expr.id] = __class__.__name__
        def forall_op(x, axis):
            sample = jnp.prod(x, axis=axis)
            if self.use_logic_ste:
                hard_sample = jnp.all(x > 0.5, axis=axis)
                sample = sample + jax.lax.stop_gradient(hard_sample - sample)
            return sample
        return self._jax_aggregation_helper(expr, aux, forall_op)
    
    def _jax_exists(self, expr, aux):
        if not self.traced.cached_is_fluent(expr):
            return JaxRDDLCompilerWithGrad._jax_exists(self, expr, aux)
        aux['overriden'][expr.id] = __class__.__name__
        def exists_op(x, axis):
            sample = 1. - jnp.prod(1. - x, axis=axis)
            if self.use_logic_ste:
                hard_sample = jnp.any(x > 0.5, axis=axis)
                sample = sample + jax.lax.stop_gradient(hard_sample - sample)
            return sample
        return self._jax_aggregation_helper(expr, aux, exists_op)
        

class GodelNormLogical(JaxRDDLCompilerWithGrad):
    '''Godel t-norm given by the expression (x, y) -> min(x, y).'''
    
    def __init__(self, *args, use_logic_ste: bool=False, **kwargs) -> None:
        super(GodelNormLogical, self).__init__(*args, **kwargs)
        self.use_logic_ste = use_logic_ste
        
    def get_kwargs(self) -> Dict[str, Any]:
        kwargs = super().get_kwargs()
        kwargs['use_logic_ste'] = self.use_logic_ste
        return kwargs

    def _jax_not(self, expr, aux):
        if not self.traced.cached_is_fluent(expr):
            return JaxRDDLCompilerWithGrad._jax_not(self, expr, aux)
        aux['overriden'][expr.id] = __class__.__name__
        def not_op(x):
            sample = 1. - x
            if self.use_logic_ste:
                hard_sample = jnp.asarray(x <= 0.5, dtype=self.REAL)
                sample = sample + jax.lax.stop_gradient(hard_sample - sample)
            return sample
        return self._jax_unary_helper(expr, aux, not_op)
    
    def _jax_and(self, expr, aux):
        if not self.traced.cached_is_fluent(expr):
            return JaxRDDLCompilerWithGrad._jax_and(self, expr, aux)
        aux['overriden'][expr.id] = __class__.__name__
        def and_op(x, y):
            sample = jnp.minimum(x, y)
            if self.use_logic_ste:
                hard_sample = jnp.asarray(jnp.logical_and(x > 0.5, y > 0.5), dtype=self.REAL)
                sample = sample + jax.lax.stop_gradient(hard_sample - sample)
            return sample
        return self._jax_nary_helper(expr, aux, and_op)

    def _jax_or(self, expr, aux):
        if not self.traced.cached_is_fluent(expr):
            return JaxRDDLCompilerWithGrad._jax_or(self, expr, aux)
        aux['overriden'][expr.id] = __class__.__name__
        def or_op(x, y):
            sample = jnp.maximum(x, y)
            if self.use_logic_ste:
                hard_sample = jnp.asarray(jnp.logical_or(x > 0.5, y > 0.5), dtype=self.REAL)
                sample = sample + jax.lax.stop_gradient(hard_sample - sample)
            return sample
        return self._jax_nary_helper(expr, aux, or_op)

    def _jax_xor(self, expr, aux):
        if not self.traced.cached_is_fluent(expr):
            return JaxRDDLCompilerWithGrad._jax_xor(self, expr, aux)
        aux['overriden'][expr.id] = __class__.__name__
        def xor_op(x, y):
            sample = jnp.minimum(jnp.maximum(x, y), 1. - jnp.minimum(x, y))
            if self.use_logic_ste:
                hard_sample = jnp.asarray(jnp.logical_xor(x > 0.5, y > 0.5), dtype=self.REAL)
                sample = sample + jax.lax.stop_gradient(hard_sample - sample)
            return sample
        return self._jax_binary_helper(expr, aux, xor_op)

    def _jax_implies(self, expr, aux):
        if not self.traced.cached_is_fluent(expr):
            return JaxRDDLCompilerWithGrad._jax_implies(self, expr, aux)
        aux['overriden'][expr.id] = __class__.__name__
        def implies_op(x, y):
            sample = jnp.maximum(1. - x, y)
            if self.use_logic_ste:
                hard_sample = jnp.asarray(jnp.logical_or(x <= 0.5, y > 0.5), dtype=self.REAL)
                sample = sample + jax.lax.stop_gradient(hard_sample - sample)
            return sample
        return self._jax_binary_helper(expr, aux, implies_op)

    def _jax_equiv(self, expr, aux):
        if not self.traced.cached_is_fluent(expr):
            return JaxRDDLCompilerWithGrad._jax_equiv(self, expr, aux)
        aux['overriden'][expr.id] = __class__.__name__
        def equiv_op(x, y):
            sample = jnp.minimum(jnp.maximum(1. - x, y), jnp.maximum(1. - y, x))
            if self.use_logic_ste:
                hard_sample = jnp.logical_and(
                    jnp.logical_or(x <= 0.5, y > 0.5), jnp.logical_or(y <= 0.5, x > 0.5))
                hard_sample = jnp.asarray(hard_sample, dtype=self.REAL)
                sample = sample + jax.lax.stop_gradient(hard_sample - sample)
            return sample
        return self._jax_binary_helper(expr, aux, equiv_op)
    
    def _jax_forall(self, expr, aux):
        if not self.traced.cached_is_fluent(expr):
            return JaxRDDLCompilerWithGrad._jax_forall(self, expr, aux)
        aux['overriden'][expr.id] = __class__.__name__
        def all_op(x, axis):
            sample = jnp.min(x, axis=axis)
            if self.use_logic_ste:
                hard_sample = jnp.all(x > 0.5, axis=axis)
                sample = sample + jax.lax.stop_gradient(hard_sample - sample)
            return sample
        return self._jax_aggregation_helper(expr, aux, all_op)
    
    def _jax_exists(self, expr, aux):
        if not self.traced.cached_is_fluent(expr):
            return JaxRDDLCompilerWithGrad._jax_exists(self, expr, aux)
        aux['overriden'][expr.id] = __class__.__name__
        def exists_op(x, axis):
            sample = jnp.max(x, axis=axis)
            if self.use_logic_ste:
                hard_sample = jnp.any(x > 0.5, axis=axis)
                sample = sample + jax.lax.stop_gradient(hard_sample - sample)
            return sample
        return self._jax_aggregation_helper(expr, aux, exists_op)
        

class LukasiewiczNormLogical(JaxRDDLCompilerWithGrad):
    '''Lukasiewicz t-norm given by the expression (x, y) -> max(x + y - 1, 0).'''
    
    def __init__(self, *args, use_logic_ste: bool=False, **kwargs) -> None:
        super(LukasiewiczNormLogical, self).__init__(*args, **kwargs)
        self.use_logic_ste = use_logic_ste

    def get_kwargs(self) -> Dict[str, Any]:
        kwargs = super().get_kwargs()
        kwargs['use_logic_ste'] = self.use_logic_ste
        return kwargs

    def _jax_not(self, expr, aux):
        if not self.traced.cached_is_fluent(expr):
            return JaxRDDLCompilerWithGrad._jax_not(self, expr, aux)
        aux['overriden'][expr.id] = __class__.__name__
        def not_op(x):
            sample = 1. - x
            if self.use_logic_ste:
                hard_sample = jnp.asarray(x <= 0.5, dtype=self.REAL)
                sample = sample + jax.lax.stop_gradient(hard_sample - sample)
            return sample
        return self._jax_unary_helper(expr, aux, not_op)
    
    def _jax_and(self, expr, aux):
        if not self.traced.cached_is_fluent(expr):
            return JaxRDDLCompilerWithGrad._jax_and(self, expr, aux)
        aux['overriden'][expr.id] = __class__.__name__
        def and_op(x, y):
            sample = jax.nn.relu(x + y - 1.)
            if self.use_logic_ste:
                hard_sample = jnp.asarray(jnp.logical_and(x > 0.5, y > 0.5), dtype=self.REAL)
                sample = sample + jax.lax.stop_gradient(hard_sample - sample)
            return sample
        return self._jax_nary_helper(expr, aux, and_op)

    def _jax_or(self, expr, aux):
        if not self.traced.cached_is_fluent(expr):
            return JaxRDDLCompilerWithGrad._jax_or(self, expr, aux)
        aux['overriden'][expr.id] = __class__.__name__
        def or_op(x, y):
            sample = 1. - jax.nn.relu(1. - x - y)
            if self.use_logic_ste:
                hard_sample = jnp.asarray(jnp.logical_or(x > 0.5, y > 0.5), dtype=self.REAL)
                sample = sample + jax.lax.stop_gradient(hard_sample - sample)
            return sample
        return self._jax_nary_helper(expr, aux, or_op)

    def _jax_xor(self, expr, aux):
        if not self.traced.cached_is_fluent(expr):
            return JaxRDDLCompilerWithGrad._jax_xor(self, expr, aux)
        aux['overriden'][expr.id] = __class__.__name__
        def xor_op(x, y):
            sample = jax.nn.relu(1. - jnp.abs(1. - x - y))
            if self.use_logic_ste:
                hard_sample = jnp.asarray(jnp.logical_xor(x > 0.5, y > 0.5), dtype=self.REAL)
                sample = sample + jax.lax.stop_gradient(hard_sample - sample)
            return sample
        return self._jax_binary_helper(expr, aux, xor_op)

    def _jax_implies(self, expr, aux):
        if not self.traced.cached_is_fluent(expr):
            return JaxRDDLCompilerWithGrad._jax_implies(self, expr, aux)
        aux['overriden'][expr.id] = __class__.__name__
        def implies_op(x, y):
            sample = 1. - jax.nn.relu(x - y)
            if self.use_logic_ste:
                hard_sample = jnp.asarray(jnp.logical_or(x <= 0.5, y > 0.5), dtype=self.REAL)
                sample = sample + jax.lax.stop_gradient(hard_sample - sample)
            return sample
        return self._jax_binary_helper(expr, aux, implies_op)

    def _jax_equiv(self, expr, aux):
        if not self.traced.cached_is_fluent(expr):
            return JaxRDDLCompilerWithGrad._jax_equiv(self, expr, aux)
        aux['overriden'][expr.id] = __class__.__name__
        def equiv_op(x, y):
            sample = jax.nn.relu(1. - jnp.abs(x - y))
            if self.use_logic_ste:
                hard_sample = jnp.logical_and(
                    jnp.logical_or(x <= 0.5, y > 0.5), jnp.logical_or(y <= 0.5, x > 0.5))
                hard_sample = jnp.asarray(hard_sample, dtype=self.REAL)
                sample = sample + jax.lax.stop_gradient(hard_sample - sample)
            return sample
        return self._jax_binary_helper(expr, aux, equiv_op)
    
    def _jax_forall(self, expr, aux):
        if not self.traced.cached_is_fluent(expr):
            return JaxRDDLCompilerWithGrad._jax_forall(self, expr, aux)
        aux['overriden'][expr.id] = __class__.__name__
        def forall_op(x, axis):
            sample = jax.nn.relu(jnp.sum(x - 1., axis=axis) + 1.)
            if self.use_logic_ste:
                hard_sample = jnp.all(x > 0.5, axis=axis)
                sample = sample + jax.lax.stop_gradient(hard_sample - sample)
            return sample
        return self._jax_aggregation_helper(expr, aux, forall_op)
    
    def _jax_exists(self, expr, aux):
        if not self.traced.cached_is_fluent(expr):
            return JaxRDDLCompilerWithGrad._jax_exists(self, expr, aux)
        aux['overriden'][expr.id] = __class__.__name__
        def exists_op(x, axis):
            sample = 1. - jax.nn.relu(jnp.sum(-x, axis=axis) + 1.)
            if self.use_logic_ste:
                hard_sample = jnp.any(x > 0.5, axis=axis)
                sample = sample + jax.lax.stop_gradient(hard_sample - sample)
            return sample
        return self._jax_aggregation_helper(expr, aux, exists_op)


# ===============================================================================
# function relaxations
# ===============================================================================

class SafeSqrt(JaxRDDLCompilerWithGrad):
    '''Sqrt operation without negative underflow.'''

    def __init__(self, *args, sqrt_eps: float=1e-14, **kwargs) -> None:
        super(SafeSqrt, self).__init__(*args, **kwargs)
        self.sqrt_eps = float(sqrt_eps)

    def get_kwargs(self) -> Dict[str, Any]:
        kwargs = super().get_kwargs()
        kwargs['sqrt_eps'] = self.sqrt_eps
        return kwargs

    def _jax_sqrt(self, expr, aux):
        aux['overriden'][expr.id] = __class__.__name__
        def safe_sqrt_op(x):
            return jnp.sqrt(x + self.sqrt_eps)
        return self._jax_unary_helper(expr, aux, safe_sqrt_op, at_least_int=True)


class SoftFloor(JaxRDDLCompilerWithGrad):
    '''Floor and ceil operations approximated using soft operations.'''
    
    def __init__(self, *args, floor_weight: float=10., 
                 use_floor_ste: bool=True, **kwargs) -> None:
        super(SoftFloor, self).__init__(*args, **kwargs)
        self.floor_weight = float(floor_weight)
        self.use_floor_ste = use_floor_ste
    
    def get_kwargs(self) -> Dict[str, Any]:
        kwargs = super().get_kwargs()
        kwargs['floor_weight'] = self.floor_weight
        kwargs['use_floor_ste'] = self.use_floor_ste
        return kwargs

    @staticmethod
    def soft_floor(x: jnp.ndarray, w: float) -> jnp.ndarray:
        s = x - jnp.floor(x)
        return jnp.floor(x) + 0.5 * (
            1. + stable_tanh(w * (s - 1.) / 2.) / stable_tanh(w / 4.))

    def _jax_floor(self, expr, aux):
        if not self.traced.cached_is_fluent(expr):
            return JaxRDDLCompilerWithGrad._jax_floor(self, expr, aux)
        id_ = expr.id
        aux['params'][id_] = self.floor_weight
        aux['overriden'][id_] = __class__.__name__
        def floor_op(x, params):
            sample = self.soft_floor(x, params[id_])
            if self.use_floor_ste:
                sample = sample + jax.lax.stop_gradient(jnp.floor(x) - sample)
            return sample, params
        return self._jax_unary_helper_with_param(expr, aux, floor_op)

    def _jax_ceil(self, expr, aux):
        if not self.traced.cached_is_fluent(expr):
            return JaxRDDLCompilerWithGrad._jax_ceil(self, expr, aux)
        id_ = expr.id
        aux['params'][id_] = self.floor_weight
        aux['overriden'][id_] = __class__.__name__
        def ceil_op(x, params):
            sample = -self.soft_floor(-x, params[id_])
            if self.use_floor_ste:
                sample = sample + jax.lax.stop_gradient(jnp.ceil(x) - sample)
            return sample, params
        return self._jax_unary_helper_with_param(expr, aux, ceil_op)

    def _jax_div(self, expr, aux):
        if not self.traced.cached_is_fluent(expr):
            return JaxRDDLCompilerWithGrad._jax_div(self, expr, aux)
        id_ = expr.id
        aux['params'][id_] = self.floor_weight
        aux['overriden'][id_] = __class__.__name__
        def div_op(x, y, params):
            sample = self.soft_floor(x / y, params[id_])
            if self.use_floor_ste:
                sample = sample + jax.lax.stop_gradient(jnp.floor_divide(x, y) - sample)
            return sample, params
        return self._jax_binary_helper_with_param(expr, aux, div_op)

    def _jax_mod(self, expr, aux):
        if not self.traced.cached_is_fluent(expr):
            return JaxRDDLCompilerWithGrad._jax_mod(self, expr, aux)
        id_ = expr.id
        aux['params'][id_] = self.floor_weight
        aux['overriden'][id_] = __class__.__name__
        def mod_op(x, y, params):
            div = self.soft_floor(x / y, params[id_])
            if self.use_floor_ste:
                div = div + jax.lax.stop_gradient(jnp.floor_divide(x, y) - div)
            sample = x - y * div
            return sample, params
        return self._jax_binary_helper_with_param(expr, aux, mod_op)


class SoftRound(JaxRDDLCompilerWithGrad):
    '''Round operations approximated using soft operations.'''
    
    def __init__(self, *args, round_weight: float=10., 
                 use_round_ste: bool=True, **kwargs) -> None:
        super(SoftRound, self).__init__(*args, **kwargs)
        self.round_weight = float(round_weight)
        self.use_round_ste = use_round_ste
    
    def get_kwargs(self) -> Dict[str, Any]:
        kwargs = super().get_kwargs()
        kwargs['round_weight'] = self.round_weight
        kwargs['use_round_ste'] = self.use_round_ste
        return kwargs

    @staticmethod
    def soft_round(x: jnp.ndarray, w: float) -> jnp.ndarray:
        m = jnp.floor(x) + 0.5
        return m + 0.5 * stable_tanh(w * (x - m)) / stable_tanh(w / 2.)

    def _jax_round(self, expr, aux):
        if not self.traced.cached_is_fluent(expr):
            return JaxRDDLCompilerWithGrad._jax_round(self, expr, aux)
        id_ = expr.id
        aux['params'][id_] = self.round_weight
        aux['overriden'][id_] = __class__.__name__
        def round_op(x, params):
            sample = self.soft_round(x, params[id_])
            if self.use_round_ste:
                sample = sample + jax.lax.stop_gradient(jnp.round(x) - sample)
            return sample, params
        return self._jax_unary_helper_with_param(expr, aux, round_op)


# ===============================================================================
# control flow relaxations
# ===============================================================================

class LinearIfElse(JaxRDDLCompilerWithGrad):
    '''Approximate if else statement as a linear combination.'''

    def __init__(self, *args, use_if_else_ste: bool=True, **kwargs) -> None:
        super(LinearIfElse, self).__init__(*args, **kwargs)
        self.use_if_else_ste = use_if_else_ste

    def get_kwargs(self) -> Dict[str, Any]:
        kwargs = super().get_kwargs()
        kwargs['use_if_else_ste'] = self.use_if_else_ste
        return kwargs

    def _jax_if(self, expr, aux):
        JaxRDDLCompilerWithGrad._check_num_args(expr, 3)
        pred, if_true, if_false = expr.args   

        # if predicate is non-fluent, always use the exact operation
        if not self.traced.cached_is_fluent(pred):
            return JaxRDDLCompilerWithGrad._jax_if(self, expr, aux)  
        
        # recursively compile arguments   
        aux['overriden'][expr.id] = __class__.__name__
        jax_pred = self._jax(pred, aux)
        jax_true = self._jax(if_true, aux)
        jax_false = self._jax(if_false, aux)
        
        def _jax_wrapped_if_then_else_linear(fls, nfls, params, key):
            sample_pred, key, err1, params = jax_pred(fls, nfls, params, key)
            sample_true, key, err2, params = jax_true(fls, nfls, params, key)
            sample_false, key, err3, params = jax_false(fls, nfls, params, key)
            if self.use_if_else_ste:
                hard_pred = jnp.asarray(sample_pred > 0.5, dtype=sample_pred.dtype)
                sample_pred = sample_pred + jax.lax.stop_gradient(hard_pred - sample_pred)
            sample = sample_pred * sample_true + (1 - sample_pred) * sample_false
            err = err1 | err2 | err3
            return sample, key, err, params
        return _jax_wrapped_if_then_else_linear


class TriangleKernelSwitch(JaxRDDLCompilerWithGrad):
    '''Switch control flow using a traingular kernel.'''
    
    def __init__(self, *args, switch_weight: float=1., 
                 switch_eps: float=1e-12, 
                 use_switch_ste: bool=True, **kwargs) -> None:
        super(TriangleKernelSwitch, self).__init__(*args, **kwargs)
        self.switch_weight = float(switch_weight)
        self.switch_eps = switch_eps
        self.use_switch_ste = use_switch_ste
        
    def get_kwargs(self) -> Dict[str, Any]:
        kwargs = super().get_kwargs()
        kwargs['switch_weight'] = self.switch_weight
        kwargs['switch_eps'] = self.switch_eps
        kwargs['use_switch_ste'] = self.use_switch_ste
        return kwargs

    def _jax_switch(self, expr, aux):

        # if predicate is non-fluent, always use the exact operation
        # case conditions are currently only literals so they are non-fluent
        pred = expr.args[0]
        if not self.traced.cached_is_fluent(pred):
            return JaxRDDLCompilerWithGrad._jax_switch(self, expr, aux)  
        
        id_ = expr.id
        aux['params'][id_] = (self.switch_weight, self.switch_eps)
        aux['overriden'][id_] = __class__.__name__
        
        # recursively compile predicate
        jax_pred = self._jax(pred, aux)
        
        # recursively compile cases
        cases, default = self.traced.cached_sim_info(expr) 
        jax_default = None if default is None else self._jax(default, aux)
        jax_cases = [
            (jax_default if _case is None else self._jax(_case, aux))
            for _case in cases
        ]
                    
        def _jax_wrapped_switch_softmax(fls, nfls, params, key):
            
            # sample predicate
            sample_pred, key, err, params = jax_pred(fls, nfls, params, key) 
            
            # sample cases
            sample_cases = []
            for jax_case in jax_cases:
                sample, key, err_case, params = jax_case(fls, nfls, params, key)
                sample_cases.append(sample)
                err = err | err_case      
            sample_cases = jnp.asarray(sample_cases)          
            sample_cases = jnp.asarray(sample_cases, dtype=self._fix_dtype(sample_cases))
            
            # replace integer indexing with weighted triangular kernel
            sample_pred_soft = jnp.broadcast_to(
                sample_pred[jnp.newaxis, ...], shape=jnp.shape(sample_cases))
            literals = enumerate_literals(jnp.shape(sample_cases), axis=0, dtype=self.INT)
            strength, eps = params[id_]
            weight = jax.nn.relu(1. - strength * jnp.abs(sample_pred_soft - literals))
            weight = weight / (jnp.sum(weight, axis=0) + eps)
            sample = jnp.sum(weight * sample_cases, axis=0)

            # straight through estimator
            if self.use_switch_ste:
                sample_pred_hard = jnp.asarray(sample_pred[jnp.newaxis, ...], dtype=self.INT)
                hard_sample = jnp.take_along_axis(sample_cases, sample_pred_hard, axis=0)
                assert jnp.shape(hard_sample)[0] == 1
                hard_sample = hard_sample[0, ...]
                sample = sample + jax.lax.stop_gradient(hard_sample - sample)
            return sample, key, err, params
        return _jax_wrapped_switch_softmax
    
    
# ===============================================================================
# distribution relaxations - Geometric
# ===============================================================================

class ReparameterizedGeometric(JaxRDDLCompilerWithGrad):

    def __init__(self, *args, 
                 geometric_floor_weight: float=10., 
                 geometric_eps: float=1e-14, **kwargs) -> None:
        super(ReparameterizedGeometric, self).__init__(*args, **kwargs)
        self.geometric_floor_weight = float(geometric_floor_weight)
        self.geometric_eps = float(geometric_eps)
        
    def get_kwargs(self) -> Dict[str, Any]:
        kwargs = super().get_kwargs()
        kwargs['geometric_floor_weight'] = self.geometric_floor_weight
        kwargs['geometric_eps'] = self.geometric_eps
        return kwargs

    def _jax_geometric(self, expr, aux):
        ERR = JaxRDDLCompilerWithGrad.ERROR_CODES['INVALID_PARAM_GEOMETRIC']
        JaxRDDLCompilerWithGrad._check_num_args(expr, 1)        
        arg_prob, = expr.args

        # if prob is non-fluent, always use the exact operation
        if not self.traced.cached_is_fluent(arg_prob):
            return JaxRDDLCompilerWithGrad._jax_geometric(self, expr, aux)  
          
        id_ = expr.id
        aux['params'][id_] = (self.geometric_floor_weight, self.geometric_eps)
        aux['overriden'][id_] = __class__.__name__

        jax_prob = self._jax(arg_prob, aux)
        
        def _jax_wrapped_distribution_geometric_reparam(fls, nfls, params, key):
            w, eps = params[id_]
            prob, key, err, params = jax_prob(fls, nfls, params, key)
            key, subkey = random.split(key)
            U = random.uniform(key=subkey, shape=jnp.shape(prob), dtype=self.REAL)
            sample = 1. + SoftFloor.soft_floor(jnp.log1p(-U) / jnp.log1p(-prob + eps), w=w)
            out_of_bounds = jnp.logical_not(jnp.all(jnp.logical_and(prob >= 0, prob <= 1)))
            err = err | (out_of_bounds * ERR)
            return sample, key, err, params
        return _jax_wrapped_distribution_geometric_reparam


class DeterminizedGeometric(JaxRDDLCompilerWithGrad):

    def __init__(self, *args, **kwargs) -> None:
        super(DeterminizedGeometric, self).__init__(*args, **kwargs)

    def get_kwargs(self) -> Dict[str, Any]:
        return super().get_kwargs()

    def _jax_geometric(self, expr, aux):
        ERR = JaxRDDLCompilerWithGrad.ERROR_CODES['INVALID_PARAM_GEOMETRIC']
        JaxRDDLCompilerWithGrad._check_num_args(expr, 1)        
        arg_prob, = expr.args
        
        # if prob is non-fluent, always use the exact operation
        if not self.traced.cached_is_fluent(arg_prob):
            return JaxRDDLCompilerWithGrad._jax_geometric(self, expr, aux)
          
        aux['overriden'][expr.id] = __class__.__name__

        jax_prob = self._jax(arg_prob, aux)
        
        def _jax_wrapped_distribution_geometric_determinized(fls, nfls, params, key):
            prob, key, err, params = jax_prob(fls, nfls, params, key)
            sample = 1. / prob
            out_of_bounds = jnp.logical_not(jnp.all(jnp.logical_and(prob >= 0, prob <= 1)))
            err = err | (out_of_bounds * ERR)
            return sample, key, err, params
        return _jax_wrapped_distribution_geometric_determinized


# ===============================================================================
# distribution relaxations - Bernoulli
# ===============================================================================

class ReparameterizedSigmoidBernoulli(JaxRDDLCompilerWithGrad):

    def __init__(self, *args, 
                 bernoulli_sigmoid_weight: float=10., 
                 use_ste_bernoulli: bool=False, **kwargs) -> None:
        super(ReparameterizedSigmoidBernoulli, self).__init__(*args, **kwargs)
        self.bernoulli_sigmoid_weight = float(bernoulli_sigmoid_weight)
        self.use_ste_bernoulli = use_ste_bernoulli
        
    def get_kwargs(self) -> Dict[str, Any]:
        kwargs = super().get_kwargs()
        kwargs['bernoulli_sigmoid_weight'] = self.bernoulli_sigmoid_weight
        kwargs['use_ste_bernoulli'] = self.use_ste_bernoulli
        return kwargs

    def _jax_bernoulli(self, expr, aux):
        ERR = JaxRDDLCompilerWithGrad.ERROR_CODES['INVALID_PARAM_BERNOULLI']
        JaxRDDLCompilerWithGrad._check_num_args(expr, 1)
        arg_prob, = expr.args
        
        # if prob is non-fluent, always use the exact operation
        if not self.traced.cached_is_fluent(arg_prob):
            return JaxRDDLCompilerWithGrad._jax_bernoulli(self, expr, aux)  
        
        id_ = expr.id
        aux['params'][id_] = self.bernoulli_sigmoid_weight
        aux['overriden'][id_] = __class__.__name__

        jax_prob = self._jax(arg_prob, aux)
        
        def _jax_wrapped_distribution_bernoulli_reparam(fls, nfls, params, key):
            prob, key, err, params = jax_prob(fls, nfls, params, key)
            key, subkey = random.split(key)
            U = random.uniform(key=subkey, shape=jnp.shape(prob), dtype=self.REAL)
            sample = stable_sigmoid(params[id_] * (prob - U))
            if self.use_ste_bernoulli:
                hard_sample = jnp.asarray(prob > U, dtype=sample.dtype)
                sample = sample + jax.lax.stop_gradient(hard_sample - sample)
            out_of_bounds = jnp.logical_not(jnp.all(jnp.logical_and(prob >= 0, prob <= 1)))
            err = err | (out_of_bounds * ERR)
            return sample, key, err, params
        return _jax_wrapped_distribution_bernoulli_reparam


class GumbelSigmoidBernoulli(JaxRDDLCompilerWithGrad):
    
    def __init__(self, *args, 
                 bernoulli_sigmoid_weight: float=10., 
                 use_ste_bernoulli: bool=False, **kwargs) -> None:
        super(GumbelSigmoidBernoulli, self).__init__(*args, **kwargs)
        self.bernoulli_sigmoid_weight = float(bernoulli_sigmoid_weight)
        self.use_ste_bernoulli = use_ste_bernoulli
    
    def get_kwargs(self) -> Dict[str, Any]:
        kwargs = super().get_kwargs()
        kwargs['bernoulli_sigmoid_weight'] = self.bernoulli_sigmoid_weight
        kwargs['use_ste_bernoulli'] = self.use_ste_bernoulli
        return kwargs

    def _jax_bernoulli(self, expr, aux):
        ERR = JaxRDDLCompilerWithGrad.ERROR_CODES['INVALID_PARAM_BERNOULLI']
        JaxRDDLCompilerWithGrad._check_num_args(expr, 1)
        arg_prob, = expr.args
        
        # if prob is non-fluent, always use the exact operation
        if not self.traced.cached_is_fluent(arg_prob):
            return JaxRDDLCompilerWithGrad._jax_bernoulli(self, expr, aux)  
        
        id_ = expr.id
        aux['params'][id_] = self.bernoulli_sigmoid_weight
        aux['overriden'][id_] = __class__.__name__

        jax_prob = self._jax(arg_prob, aux)
        
        def _jax_wrapped_distribution_bernoulli_gumbel_sigmoid(fls, nfls, params, key):
            w = params[id_]
            prob, key, err, params = jax_prob(fls, nfls, params, key)
            key, subkey = random.split(key)
            noise = random.logistic(key=subkey, shape=jnp.shape(prob), dtype=self.REAL)
            logit = scipy.special.logit(prob) + noise          
            sample = stable_sigmoid(w * logit)
            if self.use_ste_bernoulli:
                hard_sample = jnp.asarray(logit > 0, dtype=logit.dtype)
                sample = sample + jax.lax.stop_gradient(hard_sample - sample)
            out_of_bounds = jnp.logical_not(jnp.all(jnp.logical_and(prob >= 0, prob <= 1)))
            err = err | (out_of_bounds * ERR)
            return sample, key, err, params
        return _jax_wrapped_distribution_bernoulli_gumbel_sigmoid
    

class DeterminizedBernoulli(JaxRDDLCompilerWithGrad):

    def __init__(self, *args, **kwargs) -> None:
        super(DeterminizedBernoulli, self).__init__(*args, **kwargs)

    def get_kwargs(self) -> Dict[str, Any]:
        return super().get_kwargs()

    def _jax_bernoulli(self, expr, aux):
        ERR = JaxRDDLCompilerWithGrad.ERROR_CODES['INVALID_PARAM_BERNOULLI']
        JaxRDDLCompilerWithGrad._check_num_args(expr, 1)
        arg_prob, = expr.args
        
        # if prob is non-fluent, always use the exact operation
        if not self.traced.cached_is_fluent(arg_prob):
            return JaxRDDLCompilerWithGrad._jax_bernoulli(self, expr, aux)  
        
        aux['overriden'][expr.id] = __class__.__name__

        jax_prob = self._jax(arg_prob, aux)
        
        def _jax_wrapped_distribution_bernoulli_determinized(fls, nfls, params, key):
            prob, key, err, params = jax_prob(fls, nfls, params, key)
            sample = prob
            out_of_bounds = jnp.logical_not(jnp.all(jnp.logical_and(prob >= 0, prob <= 1)))
            err = err | (out_of_bounds * ERR)
            return sample, key, err, params
        return _jax_wrapped_distribution_bernoulli_determinized
    

# ===============================================================================
# distribution relaxations - Discrete
# ===============================================================================

# https://arxiv.org/pdf/1611.01144
class GumbelSoftmaxDiscrete(JaxRDDLCompilerWithGrad):
    
    def __init__(self, *args, 
                 discrete_softmax_weight: float=10., 
                 discrete_eps: float=1e-14, **kwargs) -> None:
        super(GumbelSoftmaxDiscrete, self).__init__(*args, **kwargs)
        self.discrete_softmax_weight = float(discrete_softmax_weight)
        self.discrete_eps = float(discrete_eps)
        
    def get_kwargs(self) -> Dict[str, Any]:
        kwargs = super().get_kwargs()
        kwargs['discrete_softmax_weight'] = self.discrete_softmax_weight
        kwargs['discrete_eps'] = self.discrete_eps
        return kwargs

    def _jax_discrete(self, expr, aux, unnorm):

        # if all probabilities are non-fluent, then always sample exact
        ordered_args = self.traced.cached_sim_info(expr)
        if not any(self.traced.cached_is_fluent(arg) for arg in ordered_args):
            return JaxRDDLCompilerWithGrad._jax_discrete(self, expr, aux, unnorm) 
        
        id_ = expr.id
        aux['params'][id_] = (self.discrete_softmax_weight, self.discrete_eps)
        aux['overriden'][id_] = __class__.__name__

        jax_probs = [self._jax(arg, aux) for arg in ordered_args]
        prob_fn = self._jax_discrete_prob(jax_probs, unnorm)
        
        def _jax_wrapped_distribution_discrete_gumbel_softmax(fls, nfls, params, key):
            w, eps = params[id_]
            prob, key, err, params = prob_fn(fls, nfls, params, key)
            key, subkey = random.split(key)
            g = random.gumbel(key=subkey, shape=jnp.shape(prob), dtype=self.REAL)
            logits = g + jnp.log(prob + eps)
            sample = SoftmaxArgmax.soft_argmax(logits, w=w, axis=-1, dtype=self.INT)
            err = JaxRDDLCompilerWithGrad._jax_update_discrete_oob_error(err, prob)
            return sample, key, err, params
        return _jax_wrapped_distribution_discrete_gumbel_softmax
    
    def _jax_discrete_pvar(self, expr, aux, unnorm):
        JaxRDDLCompilerWithGrad._check_num_args(expr, 2)
        _, args = expr.args
        arg, = args

        # if all probabilities are non-fluent, then always sample exact
        if not self.traced.cached_is_fluent(arg):
            return JaxRDDLCompilerWithGrad._jax_discrete_pvar(self, expr, aux, unnorm) 
        
        id_ = expr.id
        aux['params'][id_] = (self.discrete_softmax_weight, self.discrete_eps)
        aux['overriden'][id_] = __class__.__name__
        
        jax_probs = self._jax(arg, aux)
        prob_fn = self._jax_discrete_pvar_prob(jax_probs, unnorm)

        def _jax_wrapped_distribution_discrete_pvar_gumbel_softmax(fls, nfls, params, key):
            w, eps = params[id_]
            prob, key, err, params = prob_fn(fls, nfls, params, key)
            key, subkey = random.split(key)
            g = random.gumbel(key=subkey, shape=jnp.shape(prob), dtype=self.REAL)
            logits = g + jnp.log(prob + eps)
            sample = SoftmaxArgmax.soft_argmax(logits, w=w, axis=-1, dtype=self.INT)
            err = JaxRDDLCompilerWithGrad._jax_update_discrete_oob_error(err, prob)
            return sample, key, err, params
        return _jax_wrapped_distribution_discrete_pvar_gumbel_softmax


class DeterminizedDiscrete(JaxRDDLCompilerWithGrad):
    
    def __init__(self, *args, **kwargs) -> None:
        super(DeterminizedDiscrete, self).__init__(*args, **kwargs)

    def get_kwargs(self) -> Dict[str, Any]:
        return super().get_kwargs()

    def _jax_discrete(self, expr, aux, unnorm):
        
        # if all probabilities are non-fluent, then always sample exact
        ordered_args = self.traced.cached_sim_info(expr)
        if not any(self.traced.cached_is_fluent(arg) for arg in ordered_args):
            return JaxRDDLCompilerWithGrad._jax_discrete(self, expr, aux, unnorm) 
        
        aux['overriden'][expr.id] = __class__.__name__
        
        jax_probs = [self._jax(arg, aux) for arg in ordered_args]
        prob_fn = self._jax_discrete_prob(jax_probs, unnorm)
        
        def _jax_wrapped_distribution_discrete_determinized(fls, nfls, params, key):
            prob, key, err, params = prob_fn(fls, nfls, params, key)
            literals = enumerate_literals(jnp.shape(prob), axis=-1, dtype=self.INT)
            sample = jnp.sum(literals * prob, axis=-1)
            err = JaxRDDLCompilerWithGrad._jax_update_discrete_oob_error(err, prob)
            return sample, key, err, params
        return _jax_wrapped_distribution_discrete_determinized
    
    def _jax_discrete_pvar(self, expr, aux, unnorm):
        JaxRDDLCompilerWithGrad._check_num_args(expr, 2)
        _, args = expr.args
        arg, = args

        # if all probabilities are non-fluent, then always sample exact
        if not self.traced.cached_is_fluent(arg):
            return JaxRDDLCompilerWithGrad._jax_discrete_pvar(self, expr, aux, unnorm) 

        aux['overriden'][expr.id] = __class__.__name__
        
        jax_probs = self._jax(arg, aux)
        prob_fn = self._jax_discrete_pvar_prob(jax_probs, unnorm)

        def _jax_wrapped_distribution_discrete_pvar_determinized(fls, nfls, params, key):
            prob, key, err, params = prob_fn(fls, nfls, params, key)
            literals = enumerate_literals(jnp.shape(prob), axis=-1, dtype=self.INT)
            sample = jnp.sum(literals * prob, axis=-1)
            err = JaxRDDLCompilerWithGrad._jax_update_discrete_oob_error(err, prob)
            return sample, key, err, params
        return _jax_wrapped_distribution_discrete_pvar_determinized


# ===============================================================================
# distribution relaxations - Binomial
# ===============================================================================

class GumbelSoftmaxBinomial(JaxRDDLCompilerWithGrad):

    def __init__(self, *args, 
                 binomial_nbins: int=100, 
                 binomial_softmax_weight: float=10., 
                 binomial_eps: float=1e-14, **kwargs) -> None:
        super(GumbelSoftmaxBinomial, self).__init__(*args, **kwargs)
        self.binomial_nbins = binomial_nbins
        self.binomial_softmax_weight = float(binomial_softmax_weight)
        self.binomial_eps = float(binomial_eps)
        
    def get_kwargs(self) -> Dict[str, Any]:
        kwargs = super().get_kwargs()
        kwargs['binomial_nbins'] = self.binomial_nbins
        kwargs['binomial_softmax_weight'] = self.binomial_softmax_weight
        kwargs['binomial_eps'] = self.binomial_eps
        return kwargs

    # normal approximation to Binomial: Bin(n, p) -> Normal(np, np(1-p))
    @staticmethod
    def normal_approx_to_binomial(key: random.PRNGKey, 
                                  trials: jnp.ndarray, prob: jnp.ndarray) -> jnp.ndarray:
        normal = random.normal(key=key, shape=jnp.shape(trials), dtype=prob.dtype)
        mean = trials * prob
        std = jnp.sqrt(trials * jnp.clip(prob * (1. - prob), 0., 1.))
        return mean + std * normal
    
    def gumbel_softmax_approx_to_binomial(self, key: random.PRNGKey, 
                                          trials: jnp.ndarray, prob: jnp.ndarray, 
                                          w: float, eps: float):
        ks = jnp.arange(self.binomial_nbins)[(jnp.newaxis,) * jnp.ndim(trials) + (...,)]
        trials = trials[..., jnp.newaxis]
        prob = prob[..., jnp.newaxis]
        in_support = ks <= trials
        ks = jnp.minimum(ks, trials)
        log_prob = ((scipy.special.gammaln(trials + 1) - 
                     scipy.special.gammaln(ks + 1) - 
                     scipy.special.gammaln(trials - ks + 1)) +
                    ks * jnp.log(prob + eps) + 
                    (trials - ks) * jnp.log1p(-prob + eps))
        log_prob = jnp.where(in_support, log_prob, jnp.log(eps))
        g = random.gumbel(key=key, shape=jnp.shape(log_prob), dtype=prob.dtype)
        logits = g + log_prob
        return SoftmaxArgmax.soft_argmax(logits, w=w, axis=-1, dtype=self.INT)
        
    def _jax_binomial(self, expr, aux):
        ERR = JaxRDDLCompilerWithGrad.ERROR_CODES['INVALID_PARAM_BINOMIAL']
        JaxRDDLCompilerWithGrad._check_num_args(expr, 2)
        arg_trials, arg_prob = expr.args

        # if prob is non-fluent, always use the exact operation
        if (not self.traced.cached_is_fluent(arg_trials) and 
            not self.traced.cached_is_fluent(arg_prob)):
            return JaxRDDLCompilerWithGrad._jax_binomial(self, expr, aux)
        
        id_ = expr.id
        aux['params'][id_] = (self.binomial_softmax_weight, self.binomial_eps)
        aux['overriden'][id_] = __class__.__name__
        
        # recursively compile arguments
        jax_trials = self._jax(arg_trials, aux)
        jax_prob = self._jax(arg_prob, aux)

        def _jax_wrapped_distribution_binomial_gumbel_softmax(fls, nfls, params, key):
            trials, key, err2, params = jax_trials(fls, nfls, params, key)       
            prob, key, err1, params = jax_prob(fls, nfls, params, key)
            key, subkey = random.split(key)
            trials = jnp.asarray(trials, dtype=self.REAL)
            prob = jnp.asarray(prob, dtype=self.REAL)

            # use the gumbel-softmax trick for small population size
            # use the normal approximation for large population size
            sample = jnp.where(
                jax.lax.stop_gradient(trials < self.binomial_nbins), 
                self.gumbel_softmax_approx_to_binomial(subkey, trials, prob, *params[id_]), 
                self.normal_approx_to_binomial(subkey, trials, prob)
            )

            out_of_bounds = jnp.logical_not(jnp.all(
                jnp.logical_and(jnp.logical_and(prob >= 0, prob <= 1), trials >= 0)))
            err = err1 | err2 | (out_of_bounds * ERR)
            return sample, key, err, params
        return _jax_wrapped_distribution_binomial_gumbel_softmax
    

class DeterminizedBinomial(JaxRDDLCompilerWithGrad):

    def __init__(self, *args, **kwargs) -> None:
        super(DeterminizedBinomial, self).__init__(*args, **kwargs)

    def get_kwargs(self) -> Dict[str, Any]:
        return super().get_kwargs()

    def _jax_binomial(self, expr, aux):
        ERR = JaxRDDLCompilerWithGrad.ERROR_CODES['INVALID_PARAM_BINOMIAL']
        JaxRDDLCompilerWithGrad._check_num_args(expr, 2)
        arg_trials, arg_prob = expr.args

        # if prob is non-fluent, always use the exact operation
        if (not self.traced.cached_is_fluent(arg_trials) and 
            not self.traced.cached_is_fluent(arg_prob)):
            return JaxRDDLCompilerWithGrad._jax_binomial(self, expr, aux)
        
        aux['overriden'][expr.id] = __class__.__name__
        
        jax_trials = self._jax(arg_trials, aux)
        jax_prob = self._jax(arg_prob, aux)

        def _jax_wrapped_distribution_binomial_determinized(fls, nfls, params, key):
            trials, key, err2, params = jax_trials(fls, nfls, params, key)       
            prob, key, err1, params = jax_prob(fls, nfls, params, key)
            trials = jnp.asarray(trials, dtype=self.REAL)
            prob = jnp.asarray(prob, dtype=self.REAL)
            sample = trials * prob
            out_of_bounds = jnp.logical_not(jnp.all(
                jnp.logical_and(jnp.logical_and(prob >= 0, prob <= 1), trials >= 0)))
            err = err1 | err2 | (out_of_bounds * ERR)
            return sample, key, err, params
        return _jax_wrapped_distribution_binomial_determinized    


# ===============================================================================
# distribution relaxations - Poisson and NegativeBinomial
# ===============================================================================

class ExponentialPoisson(JaxRDDLCompilerWithGrad):
    
    def __init__(self, *args, 
                 poisson_nbins: int=100, 
                 poisson_comparison_weight: float=10., 
                 poisson_min_cdf: float=0.999, **kwargs) -> None:
        super(ExponentialPoisson, self).__init__(*args, **kwargs)
        self.poisson_nbins = poisson_nbins
        self.poisson_comparison_weight = float(poisson_comparison_weight)
        self.poisson_min_cdf = float(poisson_min_cdf)
        
    def get_kwargs(self) -> Dict[str, Any]:
        kwargs = super().get_kwargs()
        kwargs['poisson_nbins'] = self.poisson_nbins
        kwargs['poisson_comparison_weight'] = self.poisson_comparison_weight
        kwargs['poisson_min_cdf'] = self.poisson_min_cdf
        return kwargs

    def exponential_approx_to_poisson(self, key: random.PRNGKey, 
                                      rate: jnp.ndarray, w: float) -> jnp.ndarray:
        exp = random.exponential(
            key=key, shape=(self.poisson_nbins,) + jnp.shape(rate), dtype=rate.dtype)
        delta_t = exp / rate[jnp.newaxis, ...]
        times = jnp.cumsum(delta_t, axis=0)
        indicator = stable_sigmoid(w * (1. - times))
        return jnp.sum(indicator, axis=0)

    def branched_approx_to_poisson(self, key: random.PRNGKey, 
                                   rate: jnp.ndarray, w: float, min_cdf: float) -> jnp.ndarray:
        cuml_prob = scipy.stats.poisson.cdf(self.poisson_nbins, rate)
        z = random.normal(key=key, shape=jnp.shape(rate), dtype=rate.dtype)
        return jnp.where(
            jax.lax.stop_gradient(cuml_prob >= min_cdf), 
            self.exponential_approx_to_poisson(key, rate, w), 
            rate + jnp.sqrt(rate) * z
        )

    def _jax_poisson(self, expr, aux):
        ERR = JaxRDDLCompilerWithGrad.ERROR_CODES['INVALID_PARAM_POISSON']
        JaxRDDLCompilerWithGrad._check_num_args(expr, 1)
        arg_rate, = expr.args
        
        # if rate is non-fluent, always use the exact operation
        if not self.traced.cached_is_fluent(arg_rate):
            return JaxRDDLCompilerWithGrad._jax_poisson(self, expr, aux)
        
        id_ = expr.id
        aux['params'][id_] = (self.poisson_comparison_weight, self.poisson_min_cdf)
        aux['overriden'][id_] = __class__.__name__
        
        jax_rate = self._jax(arg_rate, aux)
        
        # use the exponential/Poisson process trick for small rate
        # use the normal approximation for large rate
        def _jax_wrapped_distribution_poisson_exponential(fls, nfls, params, key):
            rate, key, err, params = jax_rate(fls, nfls, params, key)
            key, subkey = random.split(key)
            sample = self.branched_approx_to_poisson(subkey, rate, *params[id_])            
            out_of_bounds = jnp.logical_not(jnp.all(rate >= 0))
            err = err | (out_of_bounds * ERR)
            return sample, key, err, params
        return _jax_wrapped_distribution_poisson_exponential

    def _jax_negative_binomial(self, expr, aux):
        ERR = JaxRDDLCompilerWithGrad.ERROR_CODES['INVALID_PARAM_NEGATIVE_BINOMIAL']
        JaxRDDLCompilerWithGrad._check_num_args(expr, 2)
        arg_trials, arg_prob = expr.args

        # if prob and trials is non-fluent, always use the exact operation
        if (not self.traced.cached_is_fluent(arg_trials) and 
            not self.traced.cached_is_fluent(arg_prob)):
            return JaxRDDLCompilerWithGrad._jax_negative_binomial(self, expr, aux)
        
        id_ = expr.id
        aux['params'][id_] = (self.poisson_comparison_weight, self.poisson_min_cdf)
        aux['overriden'][id_] = __class__.__name__

        jax_trials = self._jax(arg_trials, aux)
        jax_prob = self._jax(arg_prob, aux)
        
        # https://en.wikipedia.org/wiki/Negative_binomial_distribution#Gamma%E2%80%93Poisson_mixture
        def _jax_wrapped_distribution_negative_binomial_exponential(fls, nfls, params, key):
            trials, key, err2, params = jax_trials(fls, nfls, params, key)       
            prob, key, err1, params = jax_prob(fls, nfls, params, key)
            key, subkey = random.split(key)
            trials = jnp.asarray(trials, dtype=self.REAL)
            prob = jnp.asarray(prob, dtype=self.REAL)
            gamma = random.gamma(key=subkey, a=trials, dtype=self.REAL)
            rate = (1. / prob - 1.) * gamma
            sample = self.branched_approx_to_poisson(subkey, rate, *params[id_])   
            out_of_bounds = jnp.logical_not(jnp.all(
                jnp.logical_and(jnp.logical_and(prob >= 0, prob <= 1), trials > 0)))
            err = err1 | err2 | (out_of_bounds * ERR)
            return sample, key, err, params
        return _jax_wrapped_distribution_negative_binomial_exponential 


class GumbelSoftmaxPoisson(JaxRDDLCompilerWithGrad):

    def __init__(self, *args, 
                 poisson_nbins: int=100, 
                 poisson_softmax_weight: float=10., 
                 poisson_min_cdf: float=0.999, 
                 poisson_eps: float=1e-14, **kwargs) -> None:
        super(GumbelSoftmaxPoisson, self).__init__(*args, **kwargs)
        self.poisson_nbins = poisson_nbins
        self.poisson_softmax_weight = float(poisson_softmax_weight)
        self.poisson_min_cdf = float(poisson_min_cdf)
        self.poisson_eps = float(poisson_eps)
    
    def get_kwargs(self) -> Dict[str, Any]:
        kwargs = super().get_kwargs()
        kwargs['poisson_nbins'] = self.poisson_nbins
        kwargs['poisson_softmax_weight'] = self.poisson_softmax_weight
        kwargs['poisson_min_cdf'] = self.poisson_min_cdf
        kwargs['poisson_eps'] = self.poisson_eps
        return kwargs

    def gumbel_softmax_poisson(self, key: random.PRNGKey, 
                               rate: jnp.ndarray, w: float, eps: float) -> jnp.ndarray:
        ks = jnp.arange(self.poisson_nbins)[(jnp.newaxis,) * jnp.ndim(rate) + (...,)]
        rate = rate[..., jnp.newaxis]
        log_prob = ks * jnp.log(rate + eps) - rate - scipy.special.gammaln(ks + 1)
        g = random.gumbel(key=key, shape=jnp.shape(log_prob), dtype=rate.dtype)
        logits = g + log_prob
        return SoftmaxArgmax.soft_argmax(logits, w=w, axis=-1, dtype=self.INT)
    
    def branched_approx_to_poisson(self, key: random.PRNGKey, 
                                   rate: jnp.ndarray, 
                                   w: float, min_cdf: float, eps: float) -> jnp.ndarray:
        cuml_prob = scipy.stats.poisson.cdf(self.poisson_nbins, rate)
        z = random.normal(key=key, shape=jnp.shape(rate), dtype=rate.dtype)
        return jnp.where(
            jax.lax.stop_gradient(cuml_prob >= min_cdf), 
            self.gumbel_softmax_poisson(key, rate, w, eps), 
            rate + jnp.sqrt(rate) * z
        )
        
    def _jax_poisson(self, expr, aux):
        ERR = JaxRDDLCompilerWithGrad.ERROR_CODES['INVALID_PARAM_POISSON']
        JaxRDDLCompilerWithGrad._check_num_args(expr, 1)
        arg_rate, = expr.args
        
        # if rate is non-fluent, always use the exact operation
        if not self.traced.cached_is_fluent(arg_rate):
            return JaxRDDLCompilerWithGrad._jax_poisson(self, expr, aux)
        
        id_ = expr.id
        aux['params'][id_] = (self.poisson_softmax_weight, self.poisson_min_cdf, self.poisson_eps)
        aux['overriden'][id_] = __class__.__name__

        jax_rate = self._jax(arg_rate, aux)
        
        # use the gumbel-softmax and truncation trick for small rate
        # use the normal approximation for large rate
        def _jax_wrapped_distribution_poisson_gumbel_softmax(fls, nfls, params, key):
            rate, key, err, params = jax_rate(fls, nfls, params, key)
            key, subkey = random.split(key)
            sample = self.branched_approx_to_poisson(subkey, rate, *params[id_])
            out_of_bounds = jnp.logical_not(jnp.all(rate >= 0))
            err = err | (out_of_bounds * ERR)
            return sample, key, err, params
        return _jax_wrapped_distribution_poisson_gumbel_softmax
    
    def _jax_negative_binomial(self, expr, aux):
        ERR = JaxRDDLCompilerWithGrad.ERROR_CODES['INVALID_PARAM_NEGATIVE_BINOMIAL']
        JaxRDDLCompilerWithGrad._check_num_args(expr, 2)
        arg_trials, arg_prob = expr.args

        # if prob and trials is non-fluent, always use the exact operation
        if (not self.traced.cached_is_fluent(arg_trials) and 
            not self.traced.cached_is_fluent(arg_prob)):
            return JaxRDDLCompilerWithGrad._jax_negative_binomial(self, expr, aux)
        
        id_ = expr.id
        aux['params'][id_] = (self.poisson_softmax_weight, self.poisson_min_cdf, self.poisson_eps)
        aux['overriden'][id_] = __class__.__name__
        
        jax_trials = self._jax(arg_trials, aux)
        jax_prob = self._jax(arg_prob, aux)

        # https://en.wikipedia.org/wiki/Negative_binomial_distribution#Gamma%E2%80%93Poisson_mixture
        def _jax_wrapped_distribution_negative_binomial_gumbel_softmax(fls, nfls, params, key):
            trials, key, err2, params = jax_trials(fls, nfls, params, key)       
            prob, key, err1, params = jax_prob(fls, nfls, params, key)
            key, subkey = random.split(key)
            trials = jnp.asarray(trials, dtype=self.REAL)
            prob = jnp.asarray(prob, dtype=self.REAL)
            gamma = random.gamma(key=subkey, a=trials, dtype=self.REAL)
            rate = (1. / prob - 1.) * gamma
            sample = self.branched_approx_to_poisson(subkey, rate, *params[id_])   
            out_of_bounds = jnp.logical_not(jnp.all(
                jnp.logical_and(jnp.logical_and(prob >= 0, prob <= 1), trials > 0)))
            err = err1 | err2 | (out_of_bounds * ERR)
            return sample, key, err, params
        return _jax_wrapped_distribution_negative_binomial_gumbel_softmax 


class DeterminizedPoisson(JaxRDDLCompilerWithGrad):

    def __init__(self, *args, **kwargs) -> None:
        super(DeterminizedPoisson, self).__init__(*args, **kwargs)

    def get_kwargs(self) -> Dict[str, Any]:
        return super().get_kwargs()

    def _jax_poisson(self, expr, aux):
        ERR = JaxRDDLCompilerWithGrad.ERROR_CODES['INVALID_PARAM_POISSON']
        JaxRDDLCompilerWithGrad._check_num_args(expr, 1)
        arg_rate, = expr.args
        
        # if rate is non-fluent, always use the exact operation
        if not self.traced.cached_is_fluent(arg_rate):
            return JaxRDDLCompilerWithGrad._jax_poisson(self, expr, aux)
        
        aux['overriden'][expr.id] = __class__.__name__

        jax_rate = self._jax(arg_rate, aux)
        
        def _jax_wrapped_distribution_poisson_determinized(fls, nfls, params, key):
            rate, key, err, params = jax_rate(fls, nfls, params, key)
            sample = rate
            out_of_bounds = jnp.logical_not(jnp.all(rate >= 0))
            err = err | (out_of_bounds * ERR)
            return sample, key, err, params
        return _jax_wrapped_distribution_poisson_determinized
    
    def _jax_negative_binomial(self, expr, aux):
        ERR = JaxRDDLCompilerWithGrad.ERROR_CODES['INVALID_PARAM_NEGATIVE_BINOMIAL']
        JaxRDDLCompilerWithGrad._check_num_args(expr, 2)
        arg_trials, arg_prob = expr.args

        # if prob and trials is non-fluent, always use the exact operation
        if (not self.traced.cached_is_fluent(arg_trials) and 
            not self.traced.cached_is_fluent(arg_prob)):
            return JaxRDDLCompilerWithGrad._jax_negative_binomial(self, expr, aux)
        
        aux['overriden'][expr.id] = __class__.__name__
        
        jax_trials = self._jax(arg_trials, aux)
        jax_prob = self._jax(arg_prob, aux)
        
        def _jax_wrapped_distribution_negative_binomial_determinized(fls, nfls, params, key):
            trials, key, err2, params = jax_trials(fls, nfls, params, key)       
            prob, key, err1, params = jax_prob(fls, nfls, params, key)
            trials = jnp.asarray(trials, dtype=self.REAL)
            prob = jnp.asarray(prob, dtype=self.REAL)
            sample = (1. / prob - 1.) * trials
            out_of_bounds = jnp.logical_not(jnp.all(
                jnp.logical_and(jnp.logical_and(prob >= 0, prob <= 1), trials > 0)))
            err = err1 | err2 | (out_of_bounds * ERR)
            return sample, key, err, params
        return _jax_wrapped_distribution_negative_binomial_determinized    
        

class DefaultJaxRDDLCompilerWithGrad(SigmoidRelational, SoftmaxArgmax, 
                                     ProductNormLogical, 
                                     SafeSqrt, SoftFloor, SoftRound, 
                                     LinearIfElse, TriangleKernelSwitch,
                                     ReparameterizedGeometric, 
                                     ReparameterizedSigmoidBernoulli,
                                     GumbelSoftmaxDiscrete, GumbelSoftmaxBinomial,
                                     ExponentialPoisson):
    
    def __init__(self, *args, **kwargs) -> None:
        super(DefaultJaxRDDLCompilerWithGrad, self).__init__(*args, **kwargs)
   
    def get_kwargs(self) -> Dict[str, Any]:
        kwargs = {}
        for base in type(self).__bases__:
            if base.__name__ != 'object':
                kwargs = {**kwargs, **base.get_kwargs(self)}
        return kwargs
