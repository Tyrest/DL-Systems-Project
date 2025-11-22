"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype))
        if bias:
            self.bias = Parameter(init.kaiming_uniform(out_features, 1, device=device, dtype=dtype).reshape((1, out_features)))
        else:
            self.bias = None
            
        # Quantization support
        from needle import quantization
        self.observer = quantization.MinMaxObserver()
        self.quantized_weight = None
        self.weight_scale = None
        self.weight_zp = None
        self.input_scale = None
        self.input_zp = None

    def quantize_weights(self):
        from needle import quantization
        w_data = self.weight.detach().numpy()
        scale, zp = quantization.compute_scale_zero_point(w_data.min(), w_data.max())
        self.weight_scale = scale
        self.weight_zp = zp
        self.quantized_weight = self.weight.quantize_int8(scale, zp)

    def quantize_activations_static(self, x):
        if self.input_scale is None:
            scale, zp = self.observer.get_qparams()
            self.input_scale = scale
            self.input_zp = zp
        return x.quantize_int8(self.input_scale, self.input_zp)

    def forward(self, X: Tensor) -> Tensor:
        # Calibration Mode
        if hasattr(self, 'calibration_mode') and self.calibration_mode:
            self.observer.update(X)
            return self._float_forward(X)
            
        # Quantized Mode check
        if self.quantized_weight is not None:
            from needle import quantization
            # Check if we can do static quantization on input
            if self.input_scale is not None or (hasattr(self.observer, 'min_val') and self.observer.min_val != float('inf')):
                X_quant = self.quantize_activations_static(X)
            else:
                # Dynamic quantization
                x_data = X.numpy()
                scale, zp = quantization.compute_scale_zero_point(x_data.min(), x_data.max())
                X_quant = X.quantize_int8(scale, zp)
                
            X_shape = X.shape
            if len(X_shape) > 2:
                X_quant = X_quant.reshape((np.prod(X_shape[:-1]), self.in_features))
                
            out = X_quant @ self.quantized_weight
            
            if self.bias is not None:
                out = out + ops.broadcast_to(self.bias, out.shape)
                
            return out.reshape((*X_shape[:-1], self.out_features))
            
        else:
            return self._float_forward(X)

    def _float_forward(self, X):
        X_shape = X.shape
        assert X_shape[-1] == self.in_features
        X = X.reshape((np.prod(X_shape[:-1]), self.in_features))
        X = ops.matmul(X, self.weight)
        if self.bias is not None:
            X = X + ops.broadcast_to(self.bias, X.shape)
        return X.reshape((*X_shape[:-1], self.out_features))


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        return ops.reshape(X, (X.shape[0], np.prod(X.shape[1:])))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        log_sum_exp = ops.logsumexp(logits, axes=(1,))
        y_one_hot = init.one_hot(logits.shape[1], y, device=logits.device, dtype=logits.dtype)
        correct_class_scores = ops.summation(logits * y_one_hot, axes=(1,))
        error = log_sum_exp - correct_class_scores
        return ops.summation(error) / logits.shape[0]
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            mean = ops.summation(x, axes=(0,)) / x.shape[0]
            mean_broadcasted = ops.restoredims(mean, x.shape, axes=(0,))
            var = ops.summation((x - mean_broadcasted) ** 2, axes=(0,)) / x.shape[0]
            var_broadcasted = ops.restoredims(var, x.shape, axes=(0,))
            x = (x - mean_broadcasted) / (var_broadcasted + self.eps) ** 0.5
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            running_mean = ops.restoredims(self.running_mean, x.shape, axes=(0,))
            running_var = ops.restoredims(self.running_var, x.shape, axes=(0,))
            x = (x - running_mean) / (running_var + self.eps) ** 0.5
        weight = ops.restoredims(self.weight, x.shape, axes=(0,))
        bias = ops.restoredims(self.bias, x.shape, axes=(0,))
        return x * weight + bias
        ### END YOUR SOLUTION

class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.scale = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        axes = (len(x.shape) - 1,)
        mean = ops.summation(x, axes=axes, keepdims=True) / self.dim
        var = ops.summation((x - mean) ** 2, axes=axes, keepdims=True) / self.dim
        x = (x - mean) / (var + self.eps) ** 0.5
        restore_axes = tuple(range(len(x.shape) - 1))
        scale = ops.restoredims(self.scale, x.shape, axes=restore_axes)
        bias = ops.restoredims(self.bias, x.shape, axes=restore_axes)
        return x * scale + bias
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            mask = init.randb(*x.shape, p=1 - self.p, device=x.device, dtype=x.dtype)
            return x * mask / (1 - self.p)
        else:
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION
