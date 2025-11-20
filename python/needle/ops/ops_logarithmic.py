from typing import Optional, Any, Union
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

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
        self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        max_z = array_api.max(Z, axis=self.axes, keepdims=True)
        sum_exp = array_api.sum(array_api.exp(Z - max_z), axis=self.axes)
        return array_api.log(sum_exp) + array_api.squeeze(max_z, axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        softmax = exp(Z - restoredims(logsumexp(Z, axes=self.axes), Z.shape, self.axes))
        return softmax * restoredims(out_grad, Z.shape, self.axes)
        ### END YOUR SOLUTION


def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)