from typing import Optional, Any, Union
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        max_z = array_api.max(Z, axis=-1, keepdims=True)
        sum_exp = array_api.sum(array_api.exp(Z - max_z), axis=-1, keepdims=True)
        return Z - max_z - array_api.log(sum_exp)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        softmax = exp(Z - restoredims(logsumexp(Z, axes=(-1,)), Z.shape, axes=(-1,)))
        return out_grad - softmax * summation(out_grad, axes=(-1,), keepdims=True)
        ### END YOUR SOLUTION


def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        if axes is not None and type(axes) is int:
            axes = (axes,)
        self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        max_z = Z.max(axis=self.axes, keepdims=True)
        reshape_shape = [
            1 if self.axes is None or i in self.axes else Z.shape[i]
            for i in range(len(Z.shape))
        ]
        max_z_broadcasted = max_z.reshape(reshape_shape).broadcast_to(Z.shape)
        sum_exp = array_api.sum(array_api.exp(Z - max_z_broadcasted), axis=self.axes)
        return array_api.log(sum_exp) + max_z.reshape(sum_exp.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        softmax = exp(Z - restoredims(logsumexp(Z, axes=self.axes), Z.shape, self.axes))
        return softmax * restoredims(out_grad, Z.shape, self.axes)
        ### END YOUR SOLUTION


def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)
