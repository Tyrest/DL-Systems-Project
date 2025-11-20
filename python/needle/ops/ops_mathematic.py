"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a**b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        return out_grad * rhs * (lhs ** (rhs - 1)), out_grad * (lhs**rhs) * log(lhs)
        ### END YOUR SOLUTION


def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a**self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * self.scalar * (node.inputs[0] ** (self.scalar - 1))
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        return out_grad / rhs, out_grad * (-lhs / (rhs**2))
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        assert axes is None or len(axes) == 2, "Axes must be a tuple of two integers."
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        new_axes = list(range(len(a.shape)))
        if self.axes is not None:
            new_axes[self.axes[0]], new_axes[self.axes[1]] = (
                new_axes[self.axes[1]],
                new_axes[self.axes[0]],
            )
        else:
            new_axes[-1], new_axes[-2] = new_axes[-2], new_axes[-1]
        return a.permute(tuple(new_axes))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return transpose(out_grad, self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a.compact(), self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return reshape(out_grad, node.inputs[0].shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.broadcast_to(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        in_shape = list(node.inputs[0].shape)
        axes = []
        for i in range(len(out_grad.shape) - len(in_shape)):
            axes.append(i)
        for i in range(len(in_shape)):
            if in_shape[i] == 1 and out_grad.shape[i] != 1:
                axes.append(i)
        return summation(out_grad, tuple(axes)).reshape(in_shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        if axes is not None and type(axes) is int:
            axes = (axes,)
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.axes is None:
            return array_api.sum(a)
        other_axes = [i for i in range(len(a.shape)) if i not in self.axes]
        shape_prod = numpy.prod([a.shape[axis] for axis in self.axes])
        new_shape = [a.shape[axis] for axis in other_axes]
        new_shape += [shape_prod]
        a = a.permute(tuple(other_axes + list(self.axes)))
        a = a.compact()
        a = a.reshape(tuple(new_shape))
        return array_api.sum(a, axis=a.ndim - 1)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        in_shape = list(node.inputs[0].shape)
        out_grad = reshape(
            out_grad,
            [
                1 if self.axes is None or i in self.axes else in_shape[i]
                for i in range(len(in_shape))
            ],
        )
        return broadcast_to(out_grad, in_shape)
        ### END YOUR SOLUTION


def restoredims(a, original_shape, axes=None):
    if axes is None:
        reshape_shape = (1,) * len(a.shape)
    elif isinstance(axes, int):
        reshape_shape = [
            1 if i == axes or i - len(original_shape) == axes else original_shape[i]
            for i in range(len(original_shape))
        ]
    else:
        reshape_shape = [1 if i in axes or i - len(original_shape) in axes else original_shape[i] for i in range(len(original_shape))]
    return broadcast_to(reshape(a, reshape_shape), original_shape)

def summation(a, axes=None, keepdims=False):
    if keepdims:
        return restoredims(Summation(axes)(a), a.shape, axes)
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        lhs_grad = matmul(out_grad, transpose(rhs))
        rhs_grad = matmul(transpose(lhs), out_grad)
        if len(lhs.shape) < len(lhs_grad.shape):
            lhs_grad = summation(
                lhs_grad, tuple(range(len(out_grad.shape) - len(lhs.shape)))
            )
        if len(rhs.shape) < len(rhs_grad.shape):
            rhs_grad = summation(
                rhs_grad, tuple(range(len(out_grad.shape) - len(rhs.shape)))
            )
        return lhs_grad, rhs_grad
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / node.inputs[0]
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * exp(node.inputs[0])
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * Tensor(node.inputs[0].realize_cached_data() > 0, device=out_grad.device, dtype=out_grad.dtype)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return (array_api.exp(a) - array_api.exp(-a)) / (array_api.exp(a) + array_api.exp(-a))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * (-tanh(node.inputs[0]) ** 2 + 1)
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        ### BEGIN YOUR SOLUTION
        n = len(args)
        shape = list(args[0].shape)
        new_shape = shape[:self.axis] + [n] + shape[self.axis:]

        result = array_api.empty(tuple(new_shape), device=args[0].device)
        for i, arr in enumerate(args):
            slices = [slice(None)] * len(new_shape)
            slices[self.axis] = i
            result[tuple(slices)] = arr

        return result
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        n = A.shape[self.axis]
        new_shape = list(A.shape)
        new_shape.pop(self.axis)
        
        result = []
        for i in range(n):
            slices = [slice(None)] * len(A.shape)
            slices[self.axis] = i
            result.append(A[tuple(slices)].compact().reshape(tuple(new_shape)))
        
        return tuple(result)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack(list(out_grad), self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.flip(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad, self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        new_shape = list(a.shape)
        for axis in self.axes:
            if axis < len(new_shape):
                new_shape[axis] *= self.dilation + 1
        result = array_api.empty(tuple(new_shape), device=a.device)
        result.fill(0)
        slices = [slice(None)] * len(new_shape)
        for axis in self.axes:
            if axis < len(new_shape):
                slices[axis] = slice(0, new_shape[axis], self.dilation + 1)
        result[tuple(slices)] = a
        return result
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        slices = [slice(None)] * len(a.shape)
        for axis in self.axes:
            if axis < len(a.shape):
                slices[axis] = slice(0, a.shape[axis], self.dilation + 1)
        return a[tuple(slices)]
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        N, H, W, C_in = A.shape
        K_h, K_w, _, C_out = B.shape
        out_h = (H + 2 * self.padding - K_h) // self.stride + 1
        out_w = (W + 2 * self.padding - K_w) // self.stride + 1
        if self.padding > 0:
            A = A.pad(axes=((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)))

        im2col_shape = (N, out_h, out_w, K_h, K_w, C_in)
        im2col_result = array_api.empty(im2col_shape, device=A.device)
        for i in range(K_h):
            for j in range(K_w):
                row_slice = slice(i, i + self.stride * out_h, self.stride)
                col_slice = slice(j, j + self.stride * out_w, self.stride)
                im2col_result[:, :, :, i, j, :] = A[:, row_slice, col_slice, :]

        im2col_reshaped = im2col_result.reshape((N * out_h * out_w, K_h * K_w * C_in))
        B = B.compact()
        B_reshaped = B.reshape((K_h * K_w * C_in, C_out))
        conv_result = im2col_reshaped @ B_reshaped
        return conv_result.reshape((N, out_h, out_w, C_out))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        A, B = node.inputs
        N, H, W, C_in = A.shape
        K_h, K_w, _, C_out = B.shape

        assert K_h == K_w, "Only supports square kernels"
        padding = K_h - 1 - self.padding
        B = flip(B, (0, 1)).transpose((2, 3))
        dilated_out_grad = dilate(out_grad, axes=(1, 2), dilation=self.stride - 1)
        A_grad = conv(dilated_out_grad, B, padding=padding)

        A_CHWN = A.transpose((0, 3))
        out_grad_HWNC = out_grad.transpose((0, 1)).transpose((1, 2))
        out_grad_HWNC = dilate(out_grad_HWNC, axes=(0, 1), dilation=self.stride - 1)
        B_grad = conv(A_CHWN, out_grad_HWNC, padding=self.padding)
        B_grad = B_grad.transpose((0, 1)).transpose((1, 2))

        return A_grad, B_grad
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)


