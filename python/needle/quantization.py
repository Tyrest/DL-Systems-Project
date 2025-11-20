"""Utilities for post-training int8 quantization.

The helpers here implement a simple symmetric int8 scheme inspired by
LLM.int8(): weights (or activations) are mapped to int8 with a scale factor.
If per-axis quantization is requested we keep axis dimension for easy
broadcasting when dequantizing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, Union

import numpy as np

from .autograd import Tensor

ArrayLike = Union[np.ndarray, Tensor, Iterable[float]]


def _to_numpy(arr: ArrayLike) -> np.ndarray:
    if isinstance(arr, Tensor):
        return np.array(arr.realize_cached_data(), dtype=np.float32)
    return np.array(arr, dtype=np.float32)


def _reduction_axes(arr: np.ndarray, axis: Optional[int]) -> Optional[tuple[int, ...]]:
    if axis is None:
        return None
    if axis < 0:
        axis = arr.ndim + axis
    return tuple(i for i in range(arr.ndim) if i != axis)


def _symmetric_scale(arr: np.ndarray, axis: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (scale, zero_point) for symmetric int8 quantization."""
    reduce_axes = _reduction_axes(arr, axis)
    max_abs = np.max(np.abs(arr), axis=reduce_axes, keepdims=True)
    max_abs = np.where(max_abs == 0, 1.0, max_abs)
    scale = max_abs / 127.0
    zero_point = np.zeros_like(scale, dtype=np.int8)
    return scale.astype(np.float32), zero_point


def _asymmetric_scale(arr: np.ndarray, axis: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (scale, zero_point) for an asymmetric int8 quantization."""
    reduce_axes = _reduction_axes(arr, axis)
    arr_min = np.min(arr, axis=reduce_axes, keepdims=True)
    arr_max = np.max(arr, axis=reduce_axes, keepdims=True)
    span = np.where(arr_max - arr_min == 0, 1.0, arr_max - arr_min)
    scale = span / 255.0
    zp = np.round(-arr_min / scale) - 128.0
    zp = np.clip(zp, -128, 127).astype(np.int8)
    return scale.astype(np.float32), zp


@dataclass
class QuantizedTensor:
    data: np.ndarray
    scale: np.ndarray
    zero_point: np.ndarray
    cached_dequantized: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        assert self.data.dtype == np.int8, "Quantized data must be int8"
        if self.scale.shape != self.zero_point.shape:
            raise ValueError("scale and zero_point shape mismatch")

    def dequantize(self, device=None) -> Tensor:
        """Dequantize into a float32 Tensor detached from autograd."""
        if self.cached_dequantized is None:
            scale = self.scale
            zero_point = self.zero_point.astype(np.int32)
            float_data = (self.data.astype(np.int32) - zero_point) * scale
            self.cached_dequantized = float_data.astype(np.float32)
        return Tensor.make_const(self.cached_dequantized, requires_grad=False)

    def memory_bytes(self) -> int:
        return int(self.data.nbytes + self.scale.nbytes + self.zero_point.nbytes)


def quantize_int8(arr: ArrayLike, *, axis: Optional[int] = None, symmetric: bool = True) -> QuantizedTensor:
    """Quantize a tensor/array into int8 representation.

    Args:
        arr: array-like or Tensor to quantize.
        axis: optional axis for per-channel quantization.
        symmetric: use symmetric zero-point (default) or asymmetric scheme.
    """
    np_arr = _to_numpy(arr)
    scale, zero_point = (_symmetric_scale(np_arr, axis) if symmetric else _asymmetric_scale(np_arr, axis))
    scaled = np.round(np_arr / scale + zero_point).astype(np.int32)
    clipped = np.clip(scaled, -128, 127).astype(np.int8)
    return QuantizedTensor(clipped, scale, zero_point)


def dequantize_int8(q: QuantizedTensor, device=None) -> Tensor:
    """Dequantize a QuantizedTensor back into a float32 Tensor."""
    return q.dequantize(device=device)


def quantized_matmul(lhs: Tensor, rhs: QuantizedTensor, bias: Optional[Tensor] = None) -> Tensor:
    """Matrix multiply using cached dequantized weights to avoid per-call overhead."""
    weight = rhs.dequantize()
    out = lhs @ weight
    if bias is not None:
        out = out + bias
    return out
