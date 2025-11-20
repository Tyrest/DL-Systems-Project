"""The module.
"""
from typing import Any
import os
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from needle import quantization


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> list[Tensor]:
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


def _child_modules(value: object) -> list["Module"]:
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
    def __init__(self) -> None:
        self.training = True

    def parameters(self) -> list[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> list["Module"]:
        return _child_modules(self.__dict__)

    def eval(self) -> None:
        self.training = False
        for m in self._children():
            m.training = False
        return self

    def train(self) -> None:
        self.training = True
        for m in self._children():
            m.training = True
        return self

    def quantize(self, **kwargs) -> "Module":
        """Recursively quantize child modules that expose enable_quantization."""
        if hasattr(self, "enable_quantization"):
            self.enable_quantization(**kwargs)
        for m in self._children():
            if hasattr(m, "quantize"):
                m.quantize(**kwargs)
        return self

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


class Linear(Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Any | None = None,
        dtype: str = "float32",
        use_int8: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_int8 = use_int8
        self._weight_q = None
        self._weight_deq = None
        self._quant_axis = 1

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype))
        if bias:
            self.bias = Parameter(init.kaiming_uniform(out_features, 1, device=device, dtype=dtype).reshape((1, out_features)))
        ### END YOUR SOLUTION

    def enable_quantization(self, *, axis: int | None = None, symmetric: bool = True) -> quantization.QuantizedTensor:
        """Quantize weights for inference; keeps float32 weights for training."""
        axis = self._quant_axis if axis is None else axis
        self._weight_q = quantization.quantize_int8(self.weight.detach(), axis=axis, symmetric=symmetric)
        self.use_int8 = True
        # Cache dequantized float weight to avoid per-call reconstruction overhead
        self._weight_deq = self._weight_q.dequantize()
        return self._weight_q

    def disable_quantization(self) -> None:
        self._weight_q = None
        self._weight_deq = None
        self.use_int8 = False

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.use_int8 and not self.training and self._weight_q is not None:
            use_int8_kernel = os.environ.get("NEEDLE_USE_INT8_MATMUL", "0") == "1"
            if use_int8_kernel:
                try:
                    out = quantization.quantized_matmul_int8(X, self._weight_q, getattr(self, "bias", None))
                    return out
                except Exception:
                    pass
            weight = self._weight_deq if self._weight_deq is not None else self._weight_q.dequantize()
            out = ops.matmul(X, weight)
            if hasattr(self, "bias"):
                out = out + ops.broadcast_to(self.bias, out.shape)
        else:
            out = ops.matmul(X, self.weight)
            if hasattr(self, "bias"):
                out = out + ops.broadcast_to(self.bias, out.shape)
        return out
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.reshape(X, (X.shape[0], -1))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION

class Sequential(Module):
    def __init__(self, *modules: Module) -> None:
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        log_sum_exp = ops.logsumexp(logits, axes=(1,))
        y_one_hot = init.one_hot(logits.shape[1], y, device=logits.device, dtype=logits.dtype)
        correct_class_scores = ops.summation(logits * y_one_hot, axes=(1,))
        error = log_sum_exp - correct_class_scores
        return ops.summation(error) / logits.shape[0]
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1, device: Any | None = None, dtype: str = "float32") -> None:
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



class LayerNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, device: Any | None = None, dtype: str = "float32") -> None:
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
    def __init__(self, p: float = 0.5) -> None:
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
    def __init__(self, fn: Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION
