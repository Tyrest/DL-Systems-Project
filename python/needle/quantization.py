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
        data = arr.realize_cached_data()
        if hasattr(data, "numpy"):
            return data.numpy().astype(np.float32)
        return np.array(data, dtype=np.float32)
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
    cached_dequantized: Union[Tensor, None] = None

    def __post_init__(self) -> None:
        assert self.data.dtype == np.int8, "Quantized data must be int8"
        if self.scale.shape != self.zero_point.shape:
            raise ValueError("scale and zero_point shape mismatch")

    def dequantize(self, device=None) -> Tensor:
        """Dequantize into a float32 Tensor detached from autograd."""
        if self.cached_dequantized is not None:
            if device is None or self.cached_dequantized.device == device:
                return self.cached_dequantized.detach()
            # If device mismatch, we re-compute (or could move, but re-compute from int8 is safer/simpler)

        scale = self.scale
        zero_point = self.zero_point.astype(np.int32)
        float_data = (self.data.astype(np.int32) - zero_point) * scale
        float_data = float_data.astype(np.float32)
        
        self.cached_dequantized = Tensor(float_data, device=device, requires_grad=False)
        return self.cached_dequantized.detach()

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
    weight = rhs.dequantize(device=lhs.device)
    out = lhs @ weight
    if bias is not None:
        out = out + bias
    return out


def quantized_matmul_int8(lhs: Tensor, rhs: QuantizedTensor, bias: Optional[Tensor] = None) -> Tensor:
    """Int8×int8→int32 matmul via CPU backend; quantizes activations per-call.

    Note: rhs.scale is reduced to a scalar (mean) if per-axis; this keeps API simple.
    """
    lhs_np = lhs.numpy().astype(np.float32)
    # activation quantization
    a_scale = np.max(np.abs(lhs_np))
    if a_scale == 0:
        a_scale = 1.0
    a_scale = a_scale / 127.0
    a_int8 = np.clip(np.round(lhs_np / a_scale), -128, 127).astype(np.int8)

    # weight use existing quantization (may be per-axis); collapse scale to scalar for now
    w_int8 = rhs.data
    w_scale = rhs.scale.mean().item() if rhs.scale.size > 1 else float(rhs.scale.item())

    out = lhs.device.matmul_int8(a_int8, w_int8, float(a_scale), float(w_scale))
        
    if bias is not None:
        out += bias.numpy()
    
    # The output of matmul_int8 is a numpy array (float32)
    # We need to convert it back to a Tensor on the correct device
    return Tensor(out, device=lhs.device, requires_grad=False)
