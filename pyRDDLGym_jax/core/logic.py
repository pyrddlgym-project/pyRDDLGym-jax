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
@jax.custom_jvp
def stable_sigmoid(x):
    return jnp.where(x >= 0, 1.0 / (1.0 + jnp.exp(-x)), jnp.exp(x) / (1.0 + jnp.exp(x)))


@stable_sigmoid.defjvp
def stable_sigmoid_jvp(primals, tangents):
    (x,), (x_dot,) = primals, tangents
    s = stable_sigmoid(x)
    primal_out = s
    tangent_out = x_dot * s * (1.0 - s)
    return primal_out, tangent_out


# branching tanh to help reduce numerical issues
@jax.custom_jvp
def stable_tanh(x):
    ax = jnp.abs(x)
    small = jnp.where(
        ax < 20.0,
        jnp.expm1(2.0 * ax) / (jnp.expm1(2.0 * ax) + 2.0),
        1.0 - 2.0 * jnp.exp(-2.0 * ax)
    )
    return jnp.sign(x) * small


@stable_tanh.defjvp
def stable_tanh_jvp(primals, tangents):
    (x,), (x_dot,) = primals, tangents
    t = stable_tanh(x)
    tangent_out = x_dot * (1.0 - t * t)
    return t, tangent_out


# it seems JAX uses the stability trick already
def stable_softmax_weight_sum(logits, values, axis):
    probs = jax.nn.softmax(logits)
    return jnp.sum(values * probs, axis=axis)


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
                f'[INFO] JAX gradient compiler will cast pvars {pvars_cast} to float.', 
                'dark_grey'
            ))
        
    def get_kwargs(self) -> Dict[str, Any]:
        kwargs = super().get_kwargs()
        kwargs['cpfs_without_grad'] = self.cpfs_without_grad
        kwargs['print_warnings'] = self.print_warnings
        return kwargs

    def _jax_stop_grad(self, jax_expr):        
        def _jax_wrapped_stop_grad(x, params, key):
            sample, key, error, params = jax_expr(x, params, key)
            sample = jax.lax.stop_gradient(sample)
            return sample, key, error, params
        return _jax_wrapped_stop_grad
        
    def _compile_cpfs(self, init_params):

        # cpfs will all be cast to float
        cpfs_cast = set()   
        jax_cpfs = {}
        for (_, cpfs) in self.levels.items():
            for cpf in cpfs:
                _, expr = self.rddl.cpfs[cpf]
                jax_cpfs[cpf] = self._jax(expr, init_params, dtype=self.REAL)
                if self.rddl.variable_ranges[cpf] != 'real':
                    cpfs_cast.add(cpf)
                if cpf in self.cpfs_without_grad:
                    jax_cpfs[cpf] = self._jax_stop_grad(jax_cpfs[cpf])
                    
        if self.print_warnings and cpfs_cast:
            print(termcolor.colored(
                f'[INFO] JAX gradient compiler will cast CPFs {cpfs_cast} to float.', 
                'dark_grey'
            ))
        if self.print_warnings and self.cpfs_without_grad:
            print(termcolor.colored(
                f'[INFO] Gradients will not flow through CPFs {self.cpfs_without_grad}.', 
                'dark_grey'
            ))
 
        return jax_cpfs
    
    def _jax_unary_with_param(self, jax_expr, jax_op):
        def _jax_wrapped_unary_op_with_param(x, params, key):
            sample, key, err, params = jax_expr(x, params, key)
            sample = self.ONE * sample
            sample, params = jax_op(sample, params)
            return sample, key, err, params
        return _jax_wrapped_unary_op_with_param
    
    def _jax_binary_with_param(self, jax_lhs, jax_rhs, jax_op):
        def _jax_wrapped_binary_op_with_param(x, params, key):
            sample1, key, err1, params = jax_lhs(x, params, key)
            sample2, key, err2, params = jax_rhs(x, params, key)
            sample1 = self.ONE * sample1
            sample2 = self.ONE * sample2
            sample, params = jax_op(sample1, sample2, params)
            err = err1 | err2
            return sample, key, err, params
        return _jax_wrapped_binary_op_with_param
    
    def _jax_unary_helper_with_param(self, expr, init_params, jax_op):
        JaxRDDLCompilerWithGrad._check_num_args(expr, 1)
        arg, = expr.args
        jax_arg = self._jax(arg, init_params)
        return self._jax_unary_with_param(jax_arg, jax_op)
    
    def _jax_binary_helper_with_param(self, expr, init_params, jax_op):
        JaxRDDLCompilerWithGrad._check_num_args(expr, 2)
        lhs, rhs = expr.args
        jax_lhs = self._jax(lhs, init_params)
        jax_rhs = self._jax(rhs, init_params)
        return self._jax_binary_with_param(jax_lhs, jax_rhs, jax_op)

    def _jax_kron(self, expr, init_params):       
        arg, = expr.args
        arg = self._jax(arg, init_params)
        return arg


# ===============================================================================
# relational relaxations
# ===============================================================================

# https://arxiv.org/abs/2110.05651
class SigmoidRelational(JaxRDDLCompilerWithGrad):
    '''Comparison operations approximated using sigmoid functions.'''
    
    def __init__(self, *args, sigmoid_weight: float=10., **kwargs) -> None:
        super(SigmoidRelational, self).__init__(*args, **kwargs)
        self.sigmoid_weight = float(sigmoid_weight)
    
    def get_kwargs(self) -> Dict[str, Any]:
        kwargs = super().get_kwargs()
        kwargs['sigmoid_weight'] = self.sigmoid_weight
        return kwargs

    def _jax_greater(self, expr, init_params):
        if not self.traced.cached_is_fluent(expr):
            return super()._jax_greater(expr, init_params)
        id_ = str(expr.id)
        init_params[id_] = self.sigmoid_weight
        self.overriden_ops[expr.id] = __class__.__name__
        def greater_op(x, y, params):
            sample = stable_sigmoid(params[id_] * (x - y))
            return sample, params
        return self._jax_binary_helper_with_param(expr, init_params, greater_op)
    
    def _jax_greater_equal(self, expr, init_params):
        return self._jax_greater(expr, init_params)
    
    def _jax_less(self, expr, init_params):
        if not self.traced.cached_is_fluent(expr):
            return super()._jax_less(expr, init_params)
        id_ = str(expr.id)
        init_params[id_] = self.sigmoid_weight
        self.overriden_ops[expr.id] = __class__.__name__
        def less_op(x, y, params):
            sample = stable_sigmoid(params[id_] * (y - x))
            return sample, params
        return self._jax_binary_helper_with_param(expr, init_params, less_op)
    
    def _jax_less_equal(self, expr, init_params):
        return self._jax_less(expr, init_params)
    
    def _jax_equal(self, expr, init_params):
        if not self.traced.cached_is_fluent(expr):
            return super()._jax_equal(expr, init_params)
        id_ = str(expr.id)
        init_params[id_] = self.sigmoid_weight
        self.overriden_ops[expr.id] = __class__.__name__
        def equal_op(x, y, params):
            sample = 1. - jnp.square(stable_tanh(params[id_] * (y - x)))
            return sample, params
        return self._jax_binary_helper_with_param(expr, init_params, equal_op)
    
    def _jax_not_equal(self, expr, init_params):
        if not self.traced.cached_is_fluent(expr):
            return super()._jax_not_equal(expr, init_params)
        id_ = str(expr.id)
        init_params[id_] = self.sigmoid_weight
        self.overriden_ops[expr.id] = __class__.__name__
        def not_equal_op(x, y, params):
            sample = jnp.square(stable_tanh(params[id_] * (y - x)))
            return sample, params
        return self._jax_binary_helper_with_param(expr, init_params, not_equal_op)
    
    def _jax_sgn(self, expr, init_params):
        if not self.traced.cached_is_fluent(expr):
            return super()._jax_sgn(expr, init_params)
        id_ = str(expr.id)
        init_params[id_] = self.sigmoid_weight
        self.overriden_ops[expr.id] = __class__.__name__
        def sgn_op(x, params):
            sample = stable_tanh(params[id_] * x)
            return sample, params
        return self._jax_unary_helper_with_param(expr, init_params, sgn_op)
    

class SoftmaxArgmax(JaxRDDLCompilerWithGrad):
    '''Argmin/argmax operations approximated using softmax functions.'''

    def __init__(self, *args, argmax_weight: float=10., **kwargs) -> None:
        super(SoftmaxArgmax, self).__init__(*args, **kwargs)
        self.argmax_weight = float(argmax_weight)
    
    def get_kwargs(self) -> Dict[str, Any]:
        kwargs = super().get_kwargs()
        kwargs['argmax_weight'] = self.argmax_weight
        return kwargs

    @staticmethod
    def soft_argmax(x: jnp.ndarray, w: float, axes: Union[int, Tuple[int, ...]]) -> jnp.ndarray:
        literals = enumerate_literals(jnp.shape(x), axis=axes)
        sample = stable_softmax_weight_sum(w * x, literals, axis=axes)
        return sample

    def _jax_argmax(self, expr, init_params):
        if not self.traced.cached_is_fluent(expr):
            return super()._jax_argmax(expr, init_params)
        id_ = str(expr.id)
        init_params[id_] = self.argmax_weight
        self.overriden_ops[expr.id] = __class__.__name__
        * _, arg = expr.args
        _, axes = self.traced.cached_sim_info(expr)   
        jax_expr = self._jax(arg, init_params) 
        def argmax_op(x, params):
            sample = self.soft_argmax(x, params[id_], axes)
            return sample, params
        return self._jax_unary_with_param(jax_expr, argmax_op)

    def _jax_argmin(self, expr, init_params):
        if not self.traced.cached_is_fluent(expr):
            return super()._jax_argmin(expr, init_params)
        id_ = str(expr.id)
        init_params[id_] = self.argmax_weight
        self.overriden_ops[expr.id] = __class__.__name__
        * _, arg = expr.args
        _, axes = self.traced.cached_sim_info(expr)   
        jax_expr = self._jax(arg, init_params) 
        def argmin_op(x, params):
            sample = self.soft_argmax(-x, params[id_], axes)
            return sample, params
        return self._jax_unary_with_param(jax_expr, argmin_op)
        

# ===============================================================================
# logical relaxations
# ===============================================================================

class ProductNormLogical(JaxRDDLCompilerWithGrad):
    '''Product t-norm given by the expression (x, y) -> x * y.'''
    
    def __init__(self, *args, **kwargs) -> None:
        super(ProductNormLogical, self).__init__(*args, **kwargs)

    def get_kwargs(self) -> Dict[str, Any]:
        kwargs = super().get_kwargs()
        return kwargs

    def _jax_not(self, expr, init_params):
        if not self.traced.cached_is_fluent(expr):
            return super()._jax_not(expr, init_params)
        self.overriden_ops[expr.id] = __class__.__name__
        def not_op(x):
            return 1. - x
        return self._jax_unary_helper(expr, init_params, not_op)

    def _jax_and(self, expr, init_params):
        if not self.traced.cached_is_fluent(expr):
            return super()._jax_and(expr, init_params)
        self.overriden_ops[expr.id] = __class__.__name__
        return self._jax_nary_helper(expr, init_params, jnp.multiply)

    def _jax_or(self, expr, init_params):
        if not self.traced.cached_is_fluent(expr):
            return super()._jax_or(expr, init_params)
        self.overriden_ops[expr.id] = __class__.__name__
        def or_op(x, y):
            return 1. - (1. - x) * (1. - y)
        return self._jax_nary_helper(expr, init_params, or_op)

    def _jax_xor(self, expr, init_params):
        if not self.traced.cached_is_fluent(expr):
            return super()._jax_xor(expr, init_params)
        self.overriden_ops[expr.id] = __class__.__name__
        def xor_op(x, y):
            return (1. - (1. - x) * (1. - y)) * (1. - x * y)
        return self._jax_binary_helper(expr, init_params, xor_op)

    def _jax_implies(self, expr, init_params):
        if not self.traced.cached_is_fluent(expr):
            return super()._jax_implies(expr, init_params)
        self.overriden_ops[expr.id] = __class__.__name__
        def implies_op(x, y):
            return 1. - x * (1. - y)
        return self._jax_binary_helper(expr, init_params, implies_op)

    def _jax_equiv(self, expr, init_params):
        if not self.traced.cached_is_fluent(expr):
            return super()._jax_equiv(expr, init_params)
        self.overriden_ops[expr.id] = __class__.__name__
        def equiv_op(x, y):
            return (1. - x * (1. - y)) * (1. - y * (1. - x))
        return self._jax_binary_helper(expr, init_params, equiv_op)
    
    def _jax_forall(self, expr, init_params):
        if not self.traced.cached_is_fluent(expr):
            return super()._jax_forall(expr, init_params)
        self.overriden_ops[expr.id] = __class__.__name__
        return self._jax_aggregation_helper(expr, init_params, jnp.prod)
    
    def _jax_exists(self, expr, init_params):
        if not self.traced.cached_is_fluent(expr):
            return super()._jax_exists(expr, init_params)
        self.overriden_ops[expr.id] = __class__.__name__
        def exists_op(x, axis):
            return 1. - jnp.prod(1. - x, axis=axis)
        return self._jax_aggregation_helper(expr, init_params, exists_op)
        

class GodelNormLogical(JaxRDDLCompilerWithGrad):
    '''Godel t-norm given by the expression (x, y) -> min(x, y).'''
    
    def __init__(self, *args, **kwargs) -> None:
        super(GodelNormLogical, self).__init__(*args, **kwargs)
        
    def get_kwargs(self) -> Dict[str, Any]:
        kwargs = super().get_kwargs()
        return kwargs

    def _jax_not(self, expr, init_params):
        if not self.traced.cached_is_fluent(expr):
            return super()._jax_not(expr, init_params)
        self.overriden_ops[expr.id] = __class__.__name__
        def not_op(x):
            return 1. - x
        return self._jax_unary_helper(expr, init_params, not_op)
    
    def _jax_and(self, expr, init_params):
        if not self.traced.cached_is_fluent(expr):
            return super()._jax_and(expr, init_params)
        self.overriden_ops[expr.id] = __class__.__name__
        return self._jax_nary_helper(expr, init_params, jnp.minimum)

    def _jax_or(self, expr, init_params):
        if not self.traced.cached_is_fluent(expr):
            return super()._jax_or(expr, init_params)
        self.overriden_ops[expr.id] = __class__.__name__
        return self._jax_nary_helper(expr, init_params, jnp.maximum)

    def _jax_xor(self, expr, init_params):
        if not self.traced.cached_is_fluent(expr):
            return super()._jax_xor(expr, init_params)
        self.overriden_ops[expr.id] = __class__.__name__
        def xor_op(x, y):
            return jnp.minimum(jnp.maximum(x, y), 1. - jnp.minimum(x, y))
        return self._jax_binary_helper(expr, init_params, xor_op)

    def _jax_implies(self, expr, init_params):
        if not self.traced.cached_is_fluent(expr):
            return super()._jax_implies(expr, init_params)
        self.overriden_ops[expr.id] = __class__.__name__
        def implies_op(x, y):
            return jnp.maximum(1. - x, y)
        return self._jax_binary_helper(expr, init_params, implies_op)

    def _jax_equiv(self, expr, init_params):
        if not self.traced.cached_is_fluent(expr):
            return super()._jax_equiv(expr, init_params)
        self.overriden_ops[expr.id] = __class__.__name__
        def equiv_op(x, y):
            return jnp.minimum(jnp.maximum(1. - x, y), jnp.maximum(1. - y, x))
        return self._jax_binary_helper(expr, init_params, equiv_op)
    
    def _jax_forall(self, expr, init_params):
        if not self.traced.cached_is_fluent(expr):
            return super()._jax_forall(expr, init_params)
        self.overriden_ops[expr.id] = __class__.__name__
        return self._jax_aggregation_helper(expr, init_params, jnp.min)
    
    def _jax_exists(self, expr, init_params):
        if not self.traced.cached_is_fluent(expr):
            return super()._jax_exists(expr, init_params)
        self.overriden_ops[expr.id] = __class__.__name__
        return self._jax_aggregation_helper(expr, init_params, jnp.max)
        

class LukasiewiczNormLogical(JaxRDDLCompilerWithGrad):
    '''Lukasiewicz t-norm given by the expression (x, y) -> max(x + y - 1, 0).'''
    
    def __init__(self, *args, **kwargs) -> None:
        super(LukasiewiczNormLogical, self).__init__(*args, **kwargs)

    def get_kwargs(self) -> Dict[str, Any]:
        kwargs = super().get_kwargs()
        return kwargs

    def _jax_not(self, expr, init_params):
        if not self.traced.cached_is_fluent(expr):
            return super()._jax_not(expr, init_params)
        self.overriden_ops[expr.id] = __class__.__name__
        def not_op(x):
            return 1. - x
        return self._jax_unary_helper(expr, init_params, not_op)
    
    def _jax_and(self, expr, init_params):
        if not self.traced.cached_is_fluent(expr):
            return super()._jax_and(expr, init_params)
        self.overriden_ops[expr.id] = __class__.__name__
        def and_op(x, y):
            return jax.nn.relu(x + y - 1.)
        return self._jax_nary_helper(expr, init_params, and_op)

    def _jax_or(self, expr, init_params):
        if not self.traced.cached_is_fluent(expr):
            return super()._jax_or(expr, init_params)
        self.overriden_ops[expr.id] = __class__.__name__
        def or_op(x, y):
            return 1. - jax.nn.relu(1. - x - y)
        return self._jax_nary_helper(expr, init_params, or_op)

    def _jax_xor(self, expr, init_params):
        if not self.traced.cached_is_fluent(expr):
            return super()._jax_xor(expr, init_params)
        self.overriden_ops[expr.id] = __class__.__name__
        def xor_op(x, y):
            return jax.nn.relu(1. - jnp.abs(1. - x - y))
        return self._jax_binary_helper(expr, init_params, xor_op)

    def _jax_implies(self, expr, init_params):
        if not self.traced.cached_is_fluent(expr):
            return super()._jax_implies(expr, init_params)
        self.overriden_ops[expr.id] = __class__.__name__
        def implies_op(x, y):
            return 1. - jax.nn.relu(x - y)
        return self._jax_binary_helper(expr, init_params, implies_op)

    def _jax_equiv(self, expr, init_params):
        if not self.traced.cached_is_fluent(expr):
            return super()._jax_equiv(expr, init_params)
        self.overriden_ops[expr.id] = __class__.__name__
        def equiv_op(x, y):
            return jax.nn.relu(1. - jnp.abs(x - y))
        return self._jax_binary_helper(expr, init_params, equiv_op)
    
    def _jax_forall(self, expr, init_params):
        if not self.traced.cached_is_fluent(expr):
            return super()._jax_forall(expr, init_params)
        self.overriden_ops[expr.id] = __class__.__name__
        def forall_op(x, axis):
            return jax.nn.relu(jnp.sum(x - 1., axis=axis) + 1.)
        return self._jax_aggregation_helper(expr, init_params, forall_op)
       

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

    def _jax_sqrt(self, expr, init_params):
        self.overriden_ops[expr.id] = __class__.__name__
        def safe_sqrt_op(x):
            return jnp.sqrt(x + self.sqrt_eps)
        return self._jax_unary_helper(expr, init_params, safe_sqrt_op, at_least_int=True)


class SoftFloor(JaxRDDLCompilerWithGrad):
    '''Floor and ceil operations approximated using soft operations.'''
    
    def __init__(self, *args, floor_weight: float=10., **kwargs) -> None:
        super(SoftFloor, self).__init__(*args, **kwargs)
        self.floor_weight = float(floor_weight)
    
    def get_kwargs(self) -> Dict[str, Any]:
        kwargs = super().get_kwargs()
        kwargs['floor_weight'] = self.floor_weight
        return kwargs

    @staticmethod
    def soft_floor(x: jnp.ndarray, w: float) -> jnp.ndarray:
        s = x - jnp.floor(x)
        return jnp.floor(x) + 0.5 * (
            1. + stable_tanh(w * (s - 1.) / 2.) / stable_tanh(w / 4.))

    def _jax_floor(self, expr, init_params):
        if not self.traced.cached_is_fluent(expr):
            return super()._jax_floor(expr, init_params)
        id_ = str(expr.id)
        init_params[id_] = self.floor_weight
        self.overriden_ops[expr.id] = __class__.__name__
        def floor_op(x, params):
            sample = self.soft_floor(x, params[id_])
            return sample, params
        return self._jax_unary_helper_with_param(expr, init_params, floor_op)

    def _jax_ceil(self, expr, init_params):
        if not self.traced.cached_is_fluent(expr):
            return super()._jax_ceil(expr, init_params)
        id_ = str(expr.id)
        init_params[id_] = self.floor_weight
        self.overriden_ops[expr.id] = __class__.__name__
        def ceil_op(x, params):
            sample = -self.soft_floor(-x, params[id_])
            return sample, params
        return self._jax_unary_helper_with_param(expr, init_params, ceil_op)

    def _jax_div(self, expr, init_params):
        if not self.traced.cached_is_fluent(expr):
            return super()._jax_div(expr, init_params)
        id_ = str(expr.id)
        init_params[id_] = self.floor_weight
        self.overriden_ops[expr.id] = __class__.__name__
        def div_op(x, y, params):
            sample = self.soft_floor(x / y, params[id_])
            return sample, params
        return self._jax_binary_helper_with_param(expr, init_params, div_op)

    def _jax_mod(self, expr, init_params):
        if not self.traced.cached_is_fluent(expr):
            return super()._jax_mod(expr, init_params)
        id_ = str(expr.id)
        init_params[id_] = self.floor_weight
        self.overriden_ops[expr.id] = __class__.__name__
        def mod_op(x, y, params):
            div = self.soft_floor(x / y, params[id_])
            sample = x - y * div
            return sample, params
        return self._jax_binary_helper_with_param(expr, init_params, mod_op)


class SoftRound(JaxRDDLCompilerWithGrad):
    '''Round operations approximated using soft operations.'''
    
    def __init__(self, *args, round_weight: float=10., **kwargs) -> None:
        super(SoftRound, self).__init__(*args, **kwargs)
        self.round_weight = float(round_weight)
    
    def get_kwargs(self) -> Dict[str, Any]:
        kwargs = super().get_kwargs()
        kwargs['round_weight'] = self.round_weight
        return kwargs

    def _jax_round(self, expr, init_params):
        if not self.traced.cached_is_fluent(expr):
            return super()._jax_round(expr, init_params)
        id_ = str(expr.id)
        init_params[id_] = self.round_weight
        self.overriden_ops[expr.id] = __class__.__name__
        def round_op(x, params):
            param = params[id_]
            m = jnp.floor(x) + 0.5
            sample = m + 0.5 * stable_tanh(param * (x - m)) / stable_tanh(param / 2.)
            return sample, params
        return self._jax_unary_helper_with_param(expr, init_params, round_op)


# ===============================================================================
# control flow relaxations
# ===============================================================================

class LinearIfElse(JaxRDDLCompilerWithGrad):
    '''Approximate if else statement as a linear combination.'''

    def __init__(self, *args, **kwargs) -> None:
        super(LinearIfElse, self).__init__(*args, **kwargs)

    def get_kwargs(self) -> Dict[str, Any]:
        kwargs = super().get_kwargs()
        return kwargs

    def _jax_if(self, expr, init_params):
        JaxRDDLCompilerWithGrad._check_num_args(expr, 3)
        pred, if_true, if_false = expr.args   

        # if predicate is non-fluent, always use the exact operation
        if not self.traced.cached_is_fluent(pred):
            return super()._jax_if(expr, init_params)  
        
        # recursively compile arguments   
        self.overriden_ops[expr.id] = __class__.__name__
        jax_pred = self._jax(pred, init_params)
        jax_true = self._jax(if_true, init_params)
        jax_false = self._jax(if_false, init_params)
        
        def _jax_wrapped_if_then_else_linear(x, params, key):
            sample_pred, key, err1, params = jax_pred(x, params, key)
            sample_true, key, err2, params = jax_true(x, params, key)
            sample_false, key, err3, params = jax_false(x, params, key)
            sample = sample_pred * sample_true + (1. - sample_pred) * sample_false
            err = err1 | err2 | err3
            return sample, key, err, params
        return _jax_wrapped_if_then_else_linear


class SoftmaxSwitch(JaxRDDLCompilerWithGrad):
    '''Softmax switch control flow using a probabilistic interpretation.'''
    
    def __init__(self, *args, switch_weight: float=10., **kwargs) -> None:
        super(SoftmaxSwitch, self).__init__(*args, **kwargs)
        self.switch_weight = float(switch_weight)
        
    def get_kwargs(self) -> Dict[str, Any]:
        kwargs = super().get_kwargs()
        kwargs['switch_weight'] = self.switch_weight
        return kwargs

    def _jax_switch(self, expr, init_params):

         # if predicate is non-fluent, always use the exact operation
        # case conditions are currently only literals so they are non-fluent
        pred, *_ = expr.args
        if not self.traced.cached_is_fluent(pred):
            return super()._jax_switch(expr, init_params)  
        
        id_ = str(expr.id)
        init_params[id_] = self.switch_weight
        self.overriden_ops[expr.id] = __class__.__name__
        
        # recursively compile predicate
        jax_pred = self._jax(pred, init_params)
        
        # recursively compile cases
        cases, default = self.traced.cached_sim_info(expr) 
        jax_default = None if default is None else self._jax(default, init_params)
        jax_cases = [(jax_default if _case is None else self._jax(_case, init_params))
                     for _case in cases]
                    
        def _jax_wrapped_switch_softmax(x, params, key):
            
            # sample predicate
            sample_pred, key, err, params = jax_pred(x, params, key) 
            
            # sample cases
            sample_cases = [None] * len(jax_cases)
            for (i, jax_case) in enumerate(jax_cases):
                sample_cases[i], key, err_case, params = jax_case(x, params, key)
                err |= err_case      
            sample_cases = jnp.asarray(sample_cases)          
            sample_cases = jnp.asarray(sample_cases, dtype=self._fix_dtype(sample_cases))
            
            # replace integer indexing with softmax
            sample_pred = jnp.broadcast_to(
                sample_pred[jnp.newaxis, ...], shape=jnp.shape(sample_cases))
            literals = enumerate_literals(jnp.shape(sample_cases), axis=0)
            proximity = -jnp.square(sample_pred - literals)
            logits = params[id_] * proximity
            sample = stable_softmax_weight_sum(logits, sample_cases, axis=0)
            return sample, key, err, params
        return _jax_wrapped_switch_softmax
    
    
# ===============================================================================
# distribution relaxations - Geometric
# ===============================================================================

class ReparameterizedGeometric(JaxRDDLCompilerWithGrad):

    def __init__(self, *args, geometric_floor_weight: float=10., 
                 geometric_eps: float=1e-14, **kwargs) -> None:
        super(ReparameterizedGeometric, self).__init__(*args, **kwargs)
        self.geometric_floor_weight = float(geometric_floor_weight)
        self.geometric_eps = float(geometric_eps)
        
    def get_kwargs(self) -> Dict[str, Any]:
        kwargs = super().get_kwargs()
        kwargs['geometric_floor_weight'] = self.geometric_floor_weight
        kwargs['geometric_eps'] = self.geometric_eps
        return kwargs

    def _jax_geometric(self, expr, init_params):
        ERR = JaxRDDLCompilerWithGrad.ERROR_CODES['INVALID_PARAM_GEOMETRIC']
        JaxRDDLCompilerWithGrad._check_num_args(expr, 1)        
        arg_prob, = expr.args

        # if prob is non-fluent, always use the exact operation
        if not self.traced.cached_is_fluent(arg_prob):
            return super()._jax_geometric(expr, init_params)  
          
        id_ = str(expr.id)
        init_params[id_] = (self.geometric_floor_weight, self.geometric_eps)
        self.overriden_ops[expr.id] = __class__.__name__

        jax_prob = self._jax(arg_prob, init_params)
        
        def _jax_wrapped_distribution_geometric_reparam(x, params, key):
            w, eps = params[id_]
            prob, key, err, params = jax_prob(x, params, key)
            key, subkey = random.split(key)
            U = random.uniform(key=subkey, shape=jnp.shape(prob), dtype=self.REAL)
            sample = 1. + SoftFloor.soft_floor(jnp.log1p(-U) / jnp.log1p(-prob + eps), w=w)
            out_of_bounds = jnp.logical_not(jnp.all((prob >= 0) & (prob <= 1)))
            err |= (out_of_bounds * ERR)
            return sample, key, err, params
        return _jax_wrapped_distribution_geometric_reparam


class DeterminizedGeometric(JaxRDDLCompilerWithGrad):

    def __init__(self, *args, **kwargs) -> None:
        super(DeterminizedGeometric, self).__init__(*args, **kwargs)

    def get_kwargs(self) -> Dict[str, Any]:
        kwargs = super().get_kwargs()
        return kwargs

    def _jax_geometric(self, expr, init_params):
        ERR = JaxRDDLCompilerWithGrad.ERROR_CODES['INVALID_PARAM_GEOMETRIC']
        JaxRDDLCompilerWithGrad._check_num_args(expr, 1)        
        arg_prob, = expr.args
        
        # if prob is non-fluent, always use the exact operation
        if not self.traced.cached_is_fluent(arg_prob):
            return super()._jax_geometric(expr, init_params)
          
        self.overriden_ops[expr.id] = __class__.__name__

        jax_prob = self._jax(arg_prob, init_params)
        
        def _jax_wrapped_distribution_geometric_determinized(x, params, key):
            prob, key, err, params = jax_prob(x, params, key)
            sample = 1. / prob
            out_of_bounds = jnp.logical_not(jnp.all((prob >= 0) & (prob <= 1)))
            err |= (out_of_bounds * ERR)
            return sample, key, err, params
        return _jax_wrapped_distribution_geometric_determinized


# ===============================================================================
# distribution relaxations - Bernoulli
# ===============================================================================

class ReparameterizedSigmoidBernoulli(JaxRDDLCompilerWithGrad):

    def __init__(self, *args, bernoulli_sigmoid_weight: float=10., **kwargs) -> None:
        super(ReparameterizedSigmoidBernoulli, self).__init__(*args, **kwargs)
        self.bernoulli_sigmoid_weight = float(bernoulli_sigmoid_weight)
        
    def get_kwargs(self) -> Dict[str, Any]:
        kwargs = super().get_kwargs()
        kwargs['bernoulli_sigmoid_weight'] = self.bernoulli_sigmoid_weight
        return kwargs

    def _jax_bernoulli(self, expr, init_params):
        ERR = JaxRDDLCompilerWithGrad.ERROR_CODES['INVALID_PARAM_BERNOULLI']
        JaxRDDLCompilerWithGrad._check_num_args(expr, 1)
        arg_prob, = expr.args
        
        # if prob is non-fluent, always use the exact operation
        if not self.traced.cached_is_fluent(arg_prob):
            return super()._jax_bernoulli(expr, init_params)  
        
        id_ = str(expr.id)
        init_params[id_] = self.bernoulli_sigmoid_weight
        self.overriden_ops[expr.id] = __class__.__name__

        jax_prob = self._jax(arg_prob, init_params)
        
        def _jax_wrapped_distribution_bernoulli_reparam(x, params, key):
            prob, key, err, params = jax_prob(x, params, key)
            key, subkey = random.split(key)
            U = random.uniform(key=subkey, shape=jnp.shape(prob), dtype=self.REAL)
            sample = stable_sigmoid(params[id_] * (prob - U))
            out_of_bounds = jnp.logical_not(jnp.all((prob >= 0) & (prob <= 1)))
            err |= (out_of_bounds * ERR)
            return sample, key, err, params
        return _jax_wrapped_distribution_bernoulli_reparam


class GumbelSoftmaxBernoulli(JaxRDDLCompilerWithGrad):
    
    def __init__(self, *args, bernoulli_softmax_weight: float=10., 
                 bernoulli_eps: float=1e-14, **kwargs) -> None:
        super(GumbelSoftmaxBernoulli, self).__init__(*args, **kwargs)
        self.bernoulli_softmax_weight = float(bernoulli_softmax_weight)
        self.bernoulli_eps = float(bernoulli_eps)
    
    def get_kwargs(self) -> Dict[str, Any]:
        kwargs = super().get_kwargs()
        kwargs['bernoulli_softmax_weight'] = self.bernoulli_softmax_weight
        kwargs['bernoulli_eps'] = self.bernoulli_eps
        return kwargs

    def _jax_bernoulli(self, expr, init_params):
        ERR = JaxRDDLCompilerWithGrad.ERROR_CODES['INVALID_PARAM_BERNOULLI']
        JaxRDDLCompilerWithGrad._check_num_args(expr, 1)
        arg_prob, = expr.args
        
        # if prob is non-fluent, always use the exact operation
        if not self.traced.cached_is_fluent(arg_prob):
            return super()._jax_bernoulli(expr, init_params)  
        
        id_ = str(expr.id)
        init_params[id_] = (self.bernoulli_softmax_weight, self.bernoulli_eps)
        self.overriden_ops[expr.id] = __class__.__name__

        jax_prob = self._jax(arg_prob, init_params)
        
        def _jax_wrapped_distribution_bernoulli_gumbel_softmax(x, params, key):
            w, eps = params[id_]
            prob, key, err, params = jax_prob(x, params, key)
            probs = jnp.stack([1. - prob, prob], axis=-1)
            key, subkey = random.split(key)
            Gumbel01 = random.gumbel(key=subkey, shape=jnp.shape(probs), dtype=self.REAL)
            samples = Gumbel01 + jnp.log(probs + eps)
            sample = SoftmaxArgmax.soft_argmax(samples, w=w, axes=-1)
            out_of_bounds = jnp.logical_not(jnp.all((prob >= 0) & (prob <= 1)))
            err |= (out_of_bounds * ERR)
            return sample, key, err, params
        return _jax_wrapped_distribution_bernoulli_gumbel_softmax
    

class DeterminizedBernoulli(JaxRDDLCompilerWithGrad):

    def __init__(self, *args, **kwargs) -> None:
        super(DeterminizedBernoulli, self).__init__(*args, **kwargs)

    def get_kwargs(self) -> Dict[str, Any]:
        kwargs = super().get_kwargs()
        return kwargs

    def _jax_bernoulli(self, expr, init_params):
        ERR = JaxRDDLCompilerWithGrad.ERROR_CODES['INVALID_PARAM_BERNOULLI']
        JaxRDDLCompilerWithGrad._check_num_args(expr, 1)
        arg_prob, = expr.args
        
        # if prob is non-fluent, always use the exact operation
        if not self.traced.cached_is_fluent(arg_prob):
            return super()._jax_bernoulli(expr, init_params)  
        
        self.overriden_ops[expr.id] = __class__.__name__

        jax_prob = self._jax(arg_prob, init_params)
        
        def _jax_wrapped_distribution_bernoulli_determinized(x, params, key):
            prob, key, err, params = jax_prob(x, params, key)
            sample = prob
            out_of_bounds = jnp.logical_not(jnp.all((prob >= 0) & (prob <= 1)))
            err |= (out_of_bounds * ERR)
            return sample, key, err, params
        return _jax_wrapped_distribution_bernoulli_determinized
    

# ===============================================================================
# distribution relaxations - Discrete
# ===============================================================================

# https://arxiv.org/pdf/1611.01144
class GumbelSoftmaxDiscrete(JaxRDDLCompilerWithGrad):
    
    def __init__(self, *args, discrete_softmax_weight: float=10., 
                 discrete_eps: float=1e-14, **kwargs) -> None:
        super(GumbelSoftmaxDiscrete, self).__init__(*args, **kwargs)
        self.discrete_softmax_weight = float(discrete_softmax_weight)
        self.discrete_eps = float(discrete_eps)
        
    def get_kwargs(self) -> Dict[str, Any]:
        kwargs = super().get_kwargs()
        kwargs['discrete_softmax_weight'] = self.discrete_softmax_weight
        kwargs['discrete_eps'] = self.discrete_eps
        return kwargs

    def _jax_discrete(self, expr, init_params, unnorm):

        # if all probabilities are non-fluent, then always sample exact
        ordered_args = self.traced.cached_sim_info(expr)
        if not any(self.traced.cached_is_fluent(arg) for arg in ordered_args):
            return super()._jax_discrete(expr, init_params) 
        
        id_ = str(expr.id)
        init_params[id_] = (self.discrete_softmax_weight, self.discrete_eps)
        self.overriden_ops[expr.id] = __class__.__name__

        jax_probs = [self._jax(arg, init_params) for arg in ordered_args]
        prob_fn = self._jax_discrete_prob(jax_probs, unnorm)
        
        def _jax_wrapped_distribution_discrete_gumbel_softmax(x, params, key):
            w, eps = params[id_]
            prob, key, err, params = prob_fn(x, params, key)
            key, subkey = random.split(key)
            Gumbel01 = random.gumbel(key=subkey, shape=jnp.shape(prob), dtype=self.REAL)
            sample = Gumbel01 + jnp.log(prob + eps)
            sample = SoftmaxArgmax.soft_argmax(sample, w=w, axes=-1)
            err = JaxRDDLCompilerWithGrad._jax_update_discrete_oob_error(err, prob)
            return sample, key, err, params
        return _jax_wrapped_distribution_discrete_gumbel_softmax
    
    def _jax_discrete_pvar(self, expr, init_params, unnorm):
        JaxRDDLCompilerWithGrad._check_num_args(expr, 2)
        _, args = expr.args
        arg, = args

        # if all probabilities are non-fluent, then always sample exact
        if not self.traced.cached_is_fluent(arg):
            return super()._jax_discrete_pvar(expr, init_params) 
        
        id_ = str(expr.id)
        init_params[id_] = (self.discrete_softmax_weight, self.discrete_eps)
        self.overriden_ops[expr.id] = __class__.__name__
        
        jax_probs = self._jax(arg, init_params)
        prob_fn = self._jax_discrete_pvar_prob(jax_probs, unnorm)

        def _jax_wrapped_distribution_discrete_pvar_gumbel_softmax(x, params, key):
            w, eps = params[id_]
            prob, key, err, params = prob_fn(x, params, key)
            key, subkey = random.split(key)
            Gumbel01 = random.gumbel(key=subkey, shape=jnp.shape(prob), dtype=self.REAL)
            sample = Gumbel01 + jnp.log(prob + eps)
            sample = SoftmaxArgmax.soft_argmax(sample, w=w, axes=-1)
            err = JaxRDDLCompilerWithGrad._jax_update_discrete_oob_error(err, prob)
            return sample, key, err, params
        return _jax_wrapped_distribution_discrete_pvar_gumbel_softmax


class DeterminizedDiscrete(JaxRDDLCompilerWithGrad):
    
    def __init__(self, *args, **kwargs) -> None:
        super(DeterminizedDiscrete, self).__init__(*args, **kwargs)

    def get_kwargs(self) -> Dict[str, Any]:
        kwargs = super().get_kwargs()
        return kwargs

    def _jax_discrete(self, expr, init_params, unnorm):
        
        # if all probabilities are non-fluent, then always sample exact
        ordered_args = self.traced.cached_sim_info(expr)
        if not any(self.traced.cached_is_fluent(arg) for arg in ordered_args):
            return super()._jax_discrete(expr, init_params) 
        
        self.overriden_ops[expr.id] = __class__.__name__
        
        jax_probs = [self._jax(arg, init_params) for arg in ordered_args]
        prob_fn = self._jax_discrete_prob(jax_probs, unnorm)
        
        def _jax_wrapped_distribution_discrete_determinized(x, params, key):
            prob, key, err, params = prob_fn(x, params, key)
            literals = enumerate_literals(jnp.shape(prob), axis=-1)
            sample = jnp.sum(literals * prob, axis=-1)
            err = JaxRDDLCompilerWithGrad._jax_update_discrete_oob_error(err, prob)
            return sample, key, err, params
        return _jax_wrapped_distribution_discrete_determinized
    
    def _jax_discrete_pvar(self, expr, init_params, unnorm):
        JaxRDDLCompilerWithGrad._check_num_args(expr, 2)
        _, args = expr.args
        arg, = args

        # if all probabilities are non-fluent, then always sample exact
        if not self.traced.cached_is_fluent(arg):
            return super()._jax_discrete_pvar(expr, init_params) 

        self.overriden_ops[expr.id] = __class__.__name__
        
        jax_probs = self._jax(arg, init_params)
        prob_fn = self._jax_discrete_pvar_prob(jax_probs, unnorm)

        def _jax_wrapped_distribution_discrete_pvar_determinized(x, params, key):
            prob, key, err, params = prob_fn(x, params, key)
            literals = enumerate_literals(jnp.shape(prob), axis=-1)
            sample = jnp.sum(literals * prob, axis=-1)
            err = JaxRDDLCompilerWithGrad._jax_update_discrete_oob_error(err, prob)
            return sample, key, err, params
        return _jax_wrapped_distribution_discrete_pvar_determinized


# ===============================================================================
# distribution relaxations - Binomial
# ===============================================================================

class GumbelSoftmaxBinomial(JaxRDDLCompilerWithGrad):

    def __init__(self, *args, binomial_nbins: int=100, 
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
        std = jnp.sqrt(trials * prob * (1.0 - prob))
        sample = mean + std * normal
        return sample
    
    @staticmethod
    def gumbel_softmax_approx_to_binomial(key: random.PRNGKey, 
                                          trials: jnp.ndarray, prob: jnp.ndarray, 
                                          bins: int, w: float, eps: float):
        ks = jnp.arange(bins)[(jnp.newaxis,) * jnp.ndim(trials) + (...,)]
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
        Gumbel01 = random.gumbel(key=key, shape=jnp.shape(log_prob), dtype=prob.dtype)
        sample = Gumbel01 + log_prob
        sample = SoftmaxArgmax.soft_argmax(sample, w=w, axes=-1)
        return sample    
        
    def _jax_binomial(self, expr, init_params):
        ERR = JaxRDDLCompilerWithGrad.ERROR_CODES['INVALID_PARAM_BINOMIAL']
        JaxRDDLCompilerWithGrad._check_num_args(expr, 2)
        arg_trials, arg_prob = expr.args

        # if prob is non-fluent, always use the exact operation
        if not self.traced.cached_is_fluent(arg_trials) \
            and not self.traced.cached_is_fluent(arg_prob):
            return super()._jax_binomial(expr, init_params)
        
        id_ = str(expr.id)
        init_params[id_] = (
            self.binomial_nbins, self.binomial_softmax_weight, self.binomial_eps)
        self.overriden_ops[expr.id] = __class__.__name__
        
        # recursively compile arguments
        jax_trials = self._jax(arg_trials, init_params)
        jax_prob = self._jax(arg_prob, init_params)

        def _jax_wrapped_distribution_binomial_gumbel_softmax(x, params, key):
            trials, key, err2, params = jax_trials(x, params, key)       
            prob, key, err1, params = jax_prob(x, params, key)
            key, subkey = random.split(key)
            trials = jnp.asarray(trials, dtype=self.REAL)
            prob = jnp.asarray(prob, dtype=self.REAL)

            # use the gumbel-softmax trick for small population size
            # use the normal approximation for large population size
            bins, w, eps = params[id_]
            small_trials = jax.lax.stop_gradient(trials < bins)
            small_sample = self.gumbel_softmax_approx_to_binomial(
                subkey, trials, prob, bins, w, eps)
            large_sample = self.normal_approx_to_binomial(subkey, trials, prob)
            sample = jnp.where(small_trials, small_sample, large_sample)

            out_of_bounds = jnp.logical_not(jnp.all(
                (prob >= 0) & (prob <= 1) & (trials >= 0)))
            err = err1 | err2 | (out_of_bounds * ERR)
            return sample, key, err, params
        return _jax_wrapped_distribution_binomial_gumbel_softmax
    

class DeterminizedBinomial(JaxRDDLCompilerWithGrad):

    def __init__(self, *args, **kwargs) -> None:
        super(DeterminizedBinomial, self).__init__(*args, **kwargs)

    def get_kwargs(self) -> Dict[str, Any]:
        kwargs = super().get_kwargs()
        return kwargs

    def _jax_binomial(self, expr, init_params):
        ERR = JaxRDDLCompilerWithGrad.ERROR_CODES['INVALID_PARAM_BINOMIAL']
        JaxRDDLCompilerWithGrad._check_num_args(expr, 2)
        arg_trials, arg_prob = expr.args

        # if prob is non-fluent, always use the exact operation
        if not self.traced.cached_is_fluent(arg_trials) \
            and not self.traced.cached_is_fluent(arg_prob):
            return super()._jax_binomial(expr, init_params)
        
        self.overriden_ops[expr.id] = __class__.__name__
        
        jax_trials = self._jax(arg_trials, init_params)
        jax_prob = self._jax(arg_prob, init_params)

        def _jax_wrapped_distribution_binomial_determinized(x, params, key):
            trials, key, err2, params = jax_trials(x, params, key)       
            prob, key, err1, params = jax_prob(x, params, key)
            trials = jnp.asarray(trials, dtype=self.REAL)
            prob = jnp.asarray(prob, dtype=self.REAL)
            sample = trials * prob
            out_of_bounds = jnp.logical_not(jnp.all(
                (prob >= 0) & (prob <= 1) & (trials >= 0)))
            err = err1 | err2 | (out_of_bounds * ERR)
            return sample, key, err, params
        return _jax_wrapped_distribution_binomial_determinized    


# ===============================================================================
# distribution relaxations - Poisson and NegativeBinomial
# ===============================================================================

class ExponentialPoisson(JaxRDDLCompilerWithGrad):
    
    def __init__(self, *args, poisson_nbins: int=100, 
                 poisson_comparison_weight: float=10., 
                 poisson_min_cdf: float=0.999, 
                 **kwargs) -> None:
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

    @staticmethod
    def exponential_approx_to_poisson(key: random.PRNGKey, rate: jnp.ndarray, 
                                      bins: int, w: float) -> jnp.ndarray:
        Exp1 = random.exponential(
            key=key,  shape=(bins,) + jnp.shape(rate), dtype=rate.dtype)
        delta_t = Exp1 / rate[jnp.newaxis, ...]
        times = jnp.cumsum(delta_t, axis=0)
        indicator = stable_sigmoid(w * (1. - times))
        sample = jnp.sum(indicator, axis=0)
        return sample

    @staticmethod
    def branched_approx_to_poisson(key: random.PRNGKey, rate: jnp.ndarray, 
                                   bins: int, w: float, min_cdf: float) -> jnp.ndarray:
        cuml_prob = scipy.stats.poisson.cdf(bins, rate)
        small_rate = jax.lax.stop_gradient(cuml_prob >= min_cdf)
        small_sample = ExponentialPoisson.exponential_approx_to_poisson(key, rate, bins, w)
        normal = random.normal(key=key, shape=jnp.shape(rate), dtype=rate.dtype)
        large_sample = rate + jnp.sqrt(rate) * normal
        sample = jnp.where(small_rate, small_sample, large_sample)
        return sample

    def _jax_poisson(self, expr, init_params):
        ERR = JaxRDDLCompilerWithGrad.ERROR_CODES['INVALID_PARAM_POISSON']
        JaxRDDLCompilerWithGrad._check_num_args(expr, 1)
        arg_rate, = expr.args
        
        # if rate is non-fluent, always use the exact operation
        if not self.traced.cached_is_fluent(arg_rate):
            return super()._jax_poisson(expr, init_params)
        
        id_ = str(expr.id)
        init_params[id_] = (
            self.poisson_nbins, self.poisson_comparison_weight, self.poisson_min_cdf)
        self.overriden_ops[expr.id] = __class__.__name__
        
        jax_rate = self._jax(arg_rate, init_params)
        
        # use the exponential/Poisson process trick for small rate
        # use the normal approximation for large rate
        def _jax_wrapped_distribution_poisson_exponential(x, params, key):
            rate, key, err, params = jax_rate(x, params, key)
            key, subkey = random.split(key)
            sample = self.branched_approx_to_poisson(subkey, rate, *params[id_])            
            out_of_bounds = jnp.logical_not(jnp.all(rate >= 0))
            err |= (out_of_bounds * ERR)
            return sample, key, err, params
        return _jax_wrapped_distribution_poisson_exponential

    def _jax_negative_binomial(self, expr, init_params):
        ERR = JaxRDDLCompilerWithGrad.ERROR_CODES['INVALID_PARAM_NEGATIVE_BINOMIAL']
        JaxRDDLCompilerWithGrad._check_num_args(expr, 2)
        arg_trials, arg_prob = expr.args

        # if prob and trials is non-fluent, always use the exact operation
        if not self.traced.cached_is_fluent(arg_trials) \
        and not self.traced.cached_is_fluent(arg_prob):
            return super()._jax_negative_binomial(expr, init_params)
        
        id_ = str(expr.id)
        init_params[id_] = (
            self.poisson_nbins, self.poisson_comparison_weight, self.poisson_min_cdf)
        self.overriden_ops[expr.id] = __class__.__name__

        jax_trials = self._jax(arg_trials, init_params)
        jax_prob = self._jax(arg_prob, init_params)
        
        # https://en.wikipedia.org/wiki/Negative_binomial_distribution#Gamma%E2%80%93Poisson_mixture
        def _jax_wrapped_distribution_negative_binomial_exponential(x, params, key):
            trials, key, err2, params = jax_trials(x, params, key)       
            prob, key, err1, params = jax_prob(x, params, key)
            key, subkey = random.split(key)
            trials = jnp.asarray(trials, dtype=self.REAL)
            prob = jnp.asarray(prob, dtype=self.REAL)
            Gamma = random.gamma(key=subkey, a=trials, dtype=self.REAL)
            rate = ((1.0 - prob) / prob) * Gamma
            sample = self.branched_approx_to_poisson(subkey, rate, *params[id_])   
            out_of_bounds = jnp.logical_not(jnp.all(
                (prob >= 0) & (prob <= 1) & (trials > 0)))
            err = err1 | err2 | (out_of_bounds * ERR)
            return sample, key, err, params
        return _jax_wrapped_distribution_negative_binomial_exponential 


class GumbelSoftmaxPoisson(JaxRDDLCompilerWithGrad):

    def __init__(self, *args, poisson_nbins: int=100, 
                 poisson_softmax_weight: float=10., 
                 poisson_min_cdf: float=0.999, 
                 poisson_eps: float=1e-14,
                 **kwargs) -> None:
        super(GumbelSoftmaxPoisson, self).__init__(*args, **kwargs)
        self.poisson_nbins = poisson_nbins
        self.poisson_softmax_weight = float(poisson_softmax_weight)
        self.poisson_min_cdf = float(poisson_min_cdf)
        self.poisson_eps = float(poisson_eps)
    
    def get_kwargs(self) -> Dict[str, Any]:
        kwargs = super().get_kwargs()
        kwargs['poisson_softmax_weight'] = self.poisson_softmax_weight
        kwargs['poisson_min_cdf'] = self.poisson_min_cdf
        kwargs['poisson_eps'] = self.poisson_eps
        return kwargs

    @staticmethod
    def gumbel_softmax_poisson(key: random.PRNGKey, rate: jnp.ndarray, 
                               bins: int, w: float, eps: float) -> jnp.ndarray:
        ks = jnp.arange(bins)[(jnp.newaxis,) * jnp.ndim(rate) + (...,)]
        rate = rate[..., jnp.newaxis]
        log_prob = ks * jnp.log(rate + eps) - rate - scipy.special.gammaln(ks + 1)
        Gumbel01 = random.gumbel(key=key, shape=jnp.shape(log_prob), dtype=rate.dtype)
        sample = Gumbel01 + log_prob
        sample = SoftmaxArgmax.soft_argmax(sample, w=w, axes=-1)
        return sample
    
    @staticmethod
    def branched_approx_to_poisson(key: random.PRNGKey, rate: jnp.ndarray, 
                                   bins: int, w: float, min_cdf: float, eps: float) -> jnp.ndarray:
        cuml_prob = scipy.stats.poisson.cdf(bins, rate)
        small_rate = jax.lax.stop_gradient(cuml_prob >= min_cdf)
        small_sample = GumbelSoftmaxPoisson.gumbel_softmax_poisson(key, rate, bins, w, eps)
        normal = random.normal(key=key, shape=jnp.shape(rate), dtype=rate.dtype)
        large_sample = rate + jnp.sqrt(rate) * normal
        sample = jnp.where(small_rate, small_sample, large_sample)
        return sample
        
    def _jax_poisson(self, expr, init_params):
        ERR = JaxRDDLCompilerWithGrad.ERROR_CODES['INVALID_PARAM_POISSON']
        JaxRDDLCompilerWithGrad._check_num_args(expr, 1)
        arg_rate, = expr.args
        
        # if rate is non-fluent, always use the exact operation
        if not self.traced.cached_is_fluent(arg_rate):
            return super()._jax_poisson(expr, init_params)
        
        id_ = str(expr.id)
        init_params[id_] = (
            self.poisson_nbins, self.poisson_softmax_weight, self.poisson_min_cdf,
            self.poisson_eps)
        self.overriden_ops[expr.id] = __class__.__name__

        jax_rate = self._jax(arg_rate, init_params)
        
        # use the gumbel-softmax and truncation trick for small rate
        # use the normal approximation for large rate
        def _jax_wrapped_distribution_poisson_gumbel_softmax(x, params, key):
            rate, key, err, params = jax_rate(x, params, key)
            key, subkey = random.split(key)
            sample = self.branched_approx_to_poisson(subkey, rate, *params[id_])
            out_of_bounds = jnp.logical_not(jnp.all(rate >= 0))
            err |= (out_of_bounds * ERR)
            return sample, key, err, params
        return _jax_wrapped_distribution_poisson_gumbel_softmax
    
    def _jax_negative_binomial(self, expr, init_params):
        ERR = JaxRDDLCompilerWithGrad.ERROR_CODES['INVALID_PARAM_NEGATIVE_BINOMIAL']
        JaxRDDLCompilerWithGrad._check_num_args(expr, 2)
        arg_trials, arg_prob = expr.args

        # if prob and trials is non-fluent, always use the exact operation
        if not self.traced.cached_is_fluent(arg_trials) \
        and not self.traced.cached_is_fluent(arg_prob):
            return super()._jax_negative_binomial(expr, init_params)
        
        id_ = str(expr.id)
        init_params[id_] = (
            self.poisson_nbins, self.poisson_softmax_weight, self.poisson_min_cdf,
            self.poisson_eps)
        self.overriden_ops[expr.id] = __class__.__name__
        
        jax_trials = self._jax(arg_trials, init_params)
        jax_prob = self._jax(arg_prob, init_params)

        # https://en.wikipedia.org/wiki/Negative_binomial_distribution#Gamma%E2%80%93Poisson_mixture
        def _jax_wrapped_distribution_negative_binomial_gumbel_softmax(x, params, key):
            trials, key, err2, params = jax_trials(x, params, key)       
            prob, key, err1, params = jax_prob(x, params, key)
            key, subkey = random.split(key)
            trials = jnp.asarray(trials, dtype=self.REAL)
            prob = jnp.asarray(prob, dtype=self.REAL)
            Gamma = random.gamma(key=subkey, a=trials, dtype=self.REAL)
            rate = ((1.0 - prob) / prob) * Gamma
            sample = self.branched_approx_to_poisson(subkey, rate, *params[id_])   
            out_of_bounds = jnp.logical_not(jnp.all(
                (prob >= 0) & (prob <= 1) & (trials > 0)))
            err = err1 | err2 | (out_of_bounds * ERR)
            return sample, key, err, params
        return _jax_wrapped_distribution_negative_binomial_gumbel_softmax 


class DeterminizedPoisson(JaxRDDLCompilerWithGrad):

    def __init__(self, *args, **kwargs) -> None:
        super(DeterminizedPoisson, self).__init__(*args, **kwargs)

    def get_kwargs(self) -> Dict[str, Any]:
        kwargs = super().get_kwargs()
        return kwargs

    def _jax_poisson(self, expr, init_params):
        ERR = JaxRDDLCompilerWithGrad.ERROR_CODES['INVALID_PARAM_POISSON']
        JaxRDDLCompilerWithGrad._check_num_args(expr, 1)
        arg_rate, = expr.args
        
        # if rate is non-fluent, always use the exact operation
        if not self.traced.cached_is_fluent(arg_rate):
            return super()._jax_poisson(expr, init_params)
        
        self.overriden_ops[expr.id] = __class__.__name__

        jax_rate = self._jax(arg_rate, init_params)
        
        def _jax_wrapped_distribution_poisson_determinized(x, params, key):
            rate, key, err, params = jax_rate(x, params, key)
            sample = rate
            out_of_bounds = jnp.logical_not(jnp.all(rate >= 0))
            err |= (out_of_bounds * ERR)
            return sample, key, err, params
        return _jax_wrapped_distribution_poisson_determinized
    
    def _jax_negative_binomial(self, expr, init_params):
        ERR = JaxRDDLCompilerWithGrad.ERROR_CODES['INVALID_PARAM_NEGATIVE_BINOMIAL']
        JaxRDDLCompilerWithGrad._check_num_args(expr, 2)
        arg_trials, arg_prob = expr.args

        # if prob and trials is non-fluent, always use the exact operation
        if not self.traced.cached_is_fluent(arg_trials) \
        and not self.traced.cached_is_fluent(arg_prob):
            return super()._jax_negative_binomial(expr, init_params)
        
        self.overriden_ops[expr.id] = __class__.__name__
        
        jax_trials = self._jax(arg_trials, init_params)
        jax_prob = self._jax(arg_prob, init_params)
        
        def _jax_wrapped_distribution_negative_binomial_determinized(x, params, key):
            trials, key, err2, params = jax_trials(x, params, key)       
            prob, key, err1, params = jax_prob(x, params, key)
            trials = jnp.asarray(trials, dtype=self.REAL)
            prob = jnp.asarray(prob, dtype=self.REAL)
            sample = ((1.0 - prob) / prob) * trials
            out_of_bounds = jnp.logical_not(jnp.all(
                (prob >= 0) & (prob <= 1) & (trials > 0)))
            err = err1 | err2 | (out_of_bounds * ERR)
            return sample, key, err, params
        return _jax_wrapped_distribution_negative_binomial_determinized    
        

class DefaultJaxRDDLCompilerWithGrad(SigmoidRelational, SoftmaxArgmax, 
                                     ProductNormLogical, 
                                     SafeSqrt, SoftFloor, SoftRound, 
                                     LinearIfElse, SoftmaxSwitch,
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
