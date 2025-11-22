# DL-Systems-Project

## Quantization Support

Needle implements post-training int8 quantization to enable low-precision inference. The implementation spans both the Python high-level API and C++/CUDA backends.

### Python API (`python/needle/quantization.py`)

The core abstraction is the `QuantizedTensor` dataclass, which stores:
- `data`: The quantized int8 data (numpy array).
- `scale`: Float32 scaling factor.
- `zero_point`: Int8 zero point (for asymmetric quantization).
- `cached_dequantized`: Cached float32 `Tensor` to avoid re-dequantization overhead during inference.

Key functions include:
- `quantize_int8(arr, axis=None, symmetric=True)`: Quantizes a float array/tensor to int8. Supports both symmetric (scale only) and asymmetric (scale + zero_point) schemes.
- `dequantize_int8(q_tensor, device=None)`: Converts a `QuantizedTensor` back to a float32 `Tensor` on the specified device.
- `quantized_matmul_int8(lhs, rhs, bias)`: Performs a quantized matrix multiplication.
    - **Activations (`lhs`)**: Quantized on-the-fly using symmetric quantization.
    - **Weights (`rhs`)**: Expected to be a pre-quantized `QuantizedTensor`.
    - **Computation**: Calls the backend's `matmul_int8` function.
    - **Output**: Returns a float32 `Tensor` (dequantized result).

### NN Module Integration (`python/needle/nn/nn_basic.py`)

The `Linear` layer supports post-training quantization directly:
- `enable_quantization(axis=1, symmetric=True)`: Converts the layer's weights to int8 and enables quantized inference.
- `forward()`: Automatically handles quantized execution.
    - By default, it dequantizes weights and runs float32 matmul (simulated quantization).
    - If `NEEDLE_USE_INT8_MATMUL=1` is set, it attempts to use the optimized int8 kernel (`quantized_matmul_int8`).

### Backend Implementation

The actual heavy lifting is done in the backend extensions, which expose a `matmul_int8` function.

#### CPU Backend (`src/ndarray_backend_cpu.cc`)
The CPU backend implements an optimized int8 matrix multiplication:
- **Input**: Takes two int8 numpy arrays and their corresponding float scales.
- **Optimization**:
    - **Transposition**: The second matrix (B) is transposed to ensure contiguous memory access during the inner loop.
    - **Accumulation**: Intermediate results are accumulated in `int32` to prevent overflow before being scaled back to `float32`.

#### CUDA Backend (`src/ndarray_backend_cuda.cu`)
The CUDA backend provides GPU acceleration for quantized operations:
- **Kernel**: `MatmulInt8Kernel` performs the matrix multiplication on the GPU.
- **Memory Management**: Handles allocation and data transfer (Host <-> Device) for inputs and outputs.
- **Computation**: Similar to the CPU, it accumulates products in `int32` and scales the final result to `float32`.