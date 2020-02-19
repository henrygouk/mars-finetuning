from keras.constraints import Constraint
from keras.layers import Conv2D, Dense, BatchNormalization
from keras.models import Model
import keras.backend as K
import tensorflow as tf
import sys

def _update_constraint(param, constraint):
    try:
        param.constraint = constraint
    except AttributeError:
        param._constraint = constraint

class DummyConstraint(Constraint):
    def __call__(self, x):
        return x

    def get_config(self):
        return {}

def remove_constraints(model):
    for layer in model.layers:
        for param in model.weights:
            _update_constraint(param, DummyConstraint())

def add_constraints(model, norm, lambda_dense=float("inf"), lambda_conv=float("inf"), lambda_bn=float("inf"), verbose=False, zeros=None):

    if not zeros:
        zeros = [None] * len(model.layers)

    if isinstance(model, Model):
        layers = model.layers
    else:
        layers = model

    for layer, zero in zip(model.layers, zeros):
        if isinstance(layer, Conv2D) and lambda_conv != float("inf"):
            add_conv_constraint(layer, norm, lambda_conv, zero=zero)
        elif isinstance(layer, Dense) and lambda_dense != float("inf"):
            add_dense_constraint(layer, norm, lambda_dense, zero=zero)
        elif isinstance(layer, BatchNormalization) and lambda_bn != float("inf"):
            add_bn_constraint(layer, norm, lambda_bn, zero=zero)
        elif verbose:
            sys.stderr.write("Warning: no Lipschitz constraint added for layer of type " + type(layer).__name__ + "\n")

def add_dense_constraint(layer, norm, _lambda, zero=None):

    weights = layer.weights[0]

    if zero:
        zero = zero.weights[0]

    if norm == "inf-op":
        constraint = LInfLipschitzConstraint(_lambda, zero=zero)
    elif norm == "frob":
        constraint = FrobeniusConstraint(_lambda, zero=zero)

    _update_constraint(weights, constraint)

def add_conv_constraint(layer, norm, _lambda, zero=None):

    weights = layer.weights[0]

    if zero:
        zero = zero.weights[0]

    if norm == "inf-op":
        constraint = LInfLipschitzConstraint(_lambda, zero=zero)
    elif norm == "frob":
        constraint = FrobeniusConstraint(_lambda, zero=zero)

    _update_constraint(weights, constraint)

def add_bn_constraint(layer, norm, _lambda, zero=None):

    if _lambda < 0.0:
        raise Error("Lambda hyperparameters cannot be negative")

    constraint = BatchNormLipschitzConstraint(_lambda, layer.moving_variance, zero=zero)

    _update_constraint(layer.gamma, constraint)

class LInfLipschitzConstraint(Constraint):

    def __init__(self, max_k, zero=None):
        self.max_k = max_k
        self.zero = zero

    def __call__(self, w):
        if self.zero is not None:
            t = w - self.zero
        else:
            t = w

        axes=0

        if len(w.shape) == 4:
            axes=[0, 1, 2]

        norms = K.sum(K.abs(t), axis=axes, keepdims=True)
        v = t * (1.0 / K.maximum(1.0, norms / self.max_k))

        if self.zero is not None:
            return self.zero + v
        else:
            return v

    def get_config(self):
        return {"max_k": self.max_k}

class BatchNormLipschitzConstraint(Constraint):

    def __init__(self, max_k, variance, zero=None):
        self.max_k = max_k
        self.variance = variance
        self.zero = zero

    def __call__(self, w):
        diag = w / K.sqrt(self.variance + 1e-6)

        if self.zero is not None:
            zero_diag = (self.zero.gamma / K.sqrt(self.zero.moving_variance  + 1e-6))
            t = diag - zero_diag
        else:
            t = diag

        v = t * (1.0 / K.maximum(1.0, K.abs(t) / self.max_k))

        if self.zero is not None:
            return (v + zero_diag) * K.sqrt(self.variance + 1e-6)
        else:
            return v * K.sqrt(self.variance + 1e-6)

    def get_config(self):
        return {"max_k": self.max_k}

class FrobeniusConstraint(Constraint):

    def __init__(self, max_k, zero=None):
        self.max_k = max_k
        self.zero = zero

    def __call__(self, w):
        if self.zero is not None:
            t = w - self.zero
        else:
            t = w

        norm = K.sqrt(_frob_norm(t))
        v = t * (1.0 / K.maximum(1.0, norm / self.max_k))

        if self.zero is not None:
            return self.zero + v
        else:
            return v

    def get_config(self):
        return {"max_k": self.max_k}

def add_penalties(model, norm, lambda_dense=0.0, lambda_conv=0.0, lambda_bn=0.0, verbose=False, zeros=None):

    if not zeros:
        zeros = [None] * len(model.layers)

    layers = model.layers

    for layer, zero in zip(layers, zeros):
        if isinstance(layer, Conv2D) and lambda_conv != 0.0:
            add_conv_penalty(model, layer, norm, lambda_conv, zero=zero)
        elif isinstance(layer, Dense) and lambda_dense != 0.0:
            add_dense_penalty(model, layer, norm, lambda_dense, zero=zero)
        elif isinstance(layer, BatchNormalization) and lambda_bn != 0.0:
            add_bn_penalty(model, layer, norm, lambda_bn, zero=zero)
        elif verbose:
            sys.stderr.write("Warning: no Lipschitz penalty added for layer of type " + type(layer).__name__ + "\n")

def _create_penalty(_lambda, dist_func, zero):
    def penalty(w):
        if zero is not None:
            return _lambda * dist_func(w - zero)
        else:
            return _lambda * dist_func(w)

    return penalty

def add_dense_penalty(model, layer, norm, _lambda, zero=None):
    _add_penalty(model, layer, norm, _lambda, zero)

def add_conv_penalty(model, layer, norm, _lambda, zero=None):
    _add_penalty(model, layer, norm, _lambda, zero)

def add_bn_penalty(model, layer, norm, _lambda, zero=None):
    _add_penalty(model, layer, norm, _lambda, zero)

def _add_penalty(model, layer, norm, _lambda, zero=None):

    if zero:
        zero = zero.weights[0]

    if norm == "inf-op":
        penalty = _create_penalty(_lambda, _linf_norm, zero)
    elif norm == "frob":
        penalty = _create_penalty(_lambda, _frob_norm, zero)

    layer.kernel_regularizer = penalty
    model.add_loss(penalty(layer.weights[0]))

def _linf_norm(w):
    axes=0

    if len(w.shape) == 4:
        axes=[0, 1, 2]

    norm = K.max(K.sum(K.abs(w), axis=axes, keepdims=False))
    return norm

def _batchnorm_norm(diag):
    return K.max(K.abs(diag))

def _frob_norm(w):
    return K.sum(K.pow(w, 2.0), keepdims=False)

