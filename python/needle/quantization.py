"""Quantization utilities for needle deep learning framework.

This module provides utilities for quantizing floating-point tensors to uint8
format using affine quantization with scale and zero-point parameters.
"""

import numpy as np
from typing import Tuple, Optional, NamedTuple


class QuantParams(NamedTuple):
    """Quantization parameters for affine quantization.
    
    Attributes:
        scale: Scaling factor for quantization
        zero_point: Zero-point offset for quantization
        original_dtype: Original data type before quantization (e.g., 'float32')
    """
    scale: float
    zero_point: int
    original_dtype: str


def compute_quantization_params(
    data: np.ndarray,
    dtype: str = "uint8",
    symmetric: bool = False,
    per_channel: bool = False,
    axis: Optional[int] = None
) -> QuantParams:
    """Compute quantization parameters from data.
    
    Args:
        data: Input data to compute quantization parameters from
        dtype: Target quantization dtype ('uint8' supported)
        symmetric: If True, use symmetric quantization (zero_point=0 for uint8, 128 adjusted)
        per_channel: If True, compute per-channel quantization parameters
        axis: Axis for per-channel quantization (required if per_channel=True)
    
    Returns:
        QuantParams with scale and zero_point
    """
    if dtype != "uint8":
        raise ValueError(f"Unsupported quantization dtype: {dtype}")
    
    # For per-channel, we'll return a single QuantParams but with arrays
    # For now, implement per-tensor quantization
    if per_channel:
        if axis is None:
            raise ValueError("axis must be specified for per-channel quantization")
        # Move axis to last position for easier computation
        data_moved = np.moveaxis(data, axis, -1)
        original_shape = data_moved.shape
        # Reshape to (num_elements, num_channels)
        data_reshaped = data_moved.reshape(-1, original_shape[-1])
        
        min_vals = np.min(data_reshaped, axis=0)
        max_vals = np.max(data_reshaped, axis=0)
    else:
        min_vals = float(np.min(data))
        max_vals = float(np.max(data))
    
    # uint8 range: [0, 255]
    qmin, qmax = 0, 255
    
    if symmetric:
        # Symmetric quantization: map [-max_abs, max_abs] to [0, 255]
        # with zero_point at 128
        if per_channel:
            max_abs = np.maximum(np.abs(min_vals), np.abs(max_vals))
            scale = 2 * max_abs / (qmax - qmin)
            # Avoid division by zero
            scale = np.where(scale == 0, 1.0, scale)
            zero_point = np.full_like(scale, 128, dtype=np.int32)
        else:
            max_abs = max(abs(min_vals), abs(max_vals))
            scale = 2 * max_abs / (qmax - qmin) if max_abs > 0 else 1.0
            zero_point = 128
    else:
        # Asymmetric quantization
        if per_channel:
            scale = (max_vals - min_vals) / (qmax - qmin)
            # Avoid division by zero
            scale = np.where(scale == 0, 1.0, scale)
            zero_point = np.round(qmin - min_vals / scale).astype(np.int32)
            zero_point = np.clip(zero_point, qmin, qmax)
        else:
            scale = (max_vals - min_vals) / (qmax - qmin) if max_vals != min_vals else 1.0
            zero_point = int(np.round(qmin - min_vals / scale))
            zero_point = np.clip(zero_point, qmin, qmax)
    
    if per_channel:
        # Return arrays as scale and zero_point
        return QuantParams(scale=scale, zero_point=zero_point, original_dtype="float32")
    else:
        return QuantParams(scale=float(scale), zero_point=int(zero_point), original_dtype="float32")


def quantize_numpy(
    data: np.ndarray,
    scale: float,
    zero_point: int,
    dtype: str = "uint8"
) -> np.ndarray:
    """Quantize floating-point data to uint8.
    
    Formula: q = clip(round(x / scale) + zero_point, 0, 255)
    
    Args:
        data: Input floating-point data
        scale: Quantization scale
        zero_point: Quantization zero-point
        dtype: Target dtype ('uint8')
    
    Returns:
        Quantized uint8 array
    """
    if dtype != "uint8":
        raise ValueError(f"Unsupported quantization dtype: {dtype}")
    
    quantized = np.round(data / scale) + zero_point
    quantized = np.clip(quantized, 0, 255)
    return quantized.astype(np.uint8)


def dequantize_numpy(
    data: np.ndarray,
    scale: float,
    zero_point: int,
    output_dtype: str = "float32"
) -> np.ndarray:
    """Dequantize uint8 data back to floating-point.
    
    Formula: x = scale * (q - zero_point)
    
    Args:
        data: Quantized uint8 data
        scale: Quantization scale
        zero_point: Quantization zero-point
        output_dtype: Output dtype ('float32')
    
    Returns:
        Dequantized floating-point array
    """
    dequantized = scale * (data.astype(np.float32) - zero_point)
    if output_dtype == "float32":
        return dequantized.astype(np.float32)
    else:
        return dequantized


def validate_quant_params(quant_params: Optional[QuantParams]) -> bool:
    """Validate quantization parameters.
    
    Args:
        quant_params: Quantization parameters to validate
    
    Returns:
        True if valid, False otherwise
    """
    if quant_params is None:
        return False
    
    if not isinstance(quant_params, QuantParams):
        return False
    
    # Check that scale and zero_point are valid
    if isinstance(quant_params.scale, (int, float)):
        if quant_params.scale <= 0:
            return False
        if not isinstance(quant_params.zero_point, (int, np.integer)):
            return False
        if quant_params.zero_point < 0 or quant_params.zero_point > 255:
            return False
    elif isinstance(quant_params.scale, np.ndarray):
        # Per-channel quantization
        if np.any(quant_params.scale <= 0):
            return False
        if not isinstance(quant_params.zero_point, np.ndarray):
            return False
        if np.any(quant_params.zero_point < 0) or np.any(quant_params.zero_point > 255):
            return False
    else:
        return False
    
    return True
