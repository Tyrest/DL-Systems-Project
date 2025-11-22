# Implementation Plan: Needle `int8` Quantization Support

This plan outlines the steps to introduce `int8` data types and quantization support across the Needle library, enabling low-precision inference and operations.

## 1. Core Data Structures & Plumbing

### `python/needle/backend_ndarray/ndarray.py`
**Goal**: Enable `NDArray` to store and manage `int8` data and quantization metadata.

*   **Data Storage**:
    *   Add `_dtype` attribute to `NDArray`.
    *   Add `_quant_params` attribute (optional) to store `scale`, `zero_point`, and `original_dtype`.
*   **Construction**:
    *   Update `NDArray.make()`, `BackendDevice.empty()`, and `BackendDevice.full()` to accept a `dtype` argument (defaulting to `float32`).
    *   Expose a `dtype` property on `NDArray`.
*   **New Methods**:
    *   `astype(dtype)`: Casts array to specified type.
    *   `quantize_int8(scale, zero_point)`: Returns a new `int8` `NDArray` with attached `_quant_params`.
    *   `dequantize()`: Converts a `int8` array back to `float32` using stored `_quant_params`.
*   **Operator Dispatch**:
    *   Update `__matmul__`:
        *   Check if both operands are `int8` and have valid `_quant_params`.
        *   If yes, dispatch to the new `matmul_int8` backend kernel.
        *   If no, fall back to standard `float32` path (implicitly dequantizing if needed).

### `python/needle/autograd.py`
**Goal**: Expose quantization capabilities at the `Tensor` level.

*   **Tensor Metadata**:
    *   Ensure `Tensor` construction defaults to `float32`.
    *   Update `Tensor.realize_cached_data()` to preserve the underlying `NDArray`'s `dtype` and `_quant_params`.
*   **Methods**:
    *   Implement `Tensor.astype(dtype)` and `Tensor.quantize_int8(...)` which delegate to the underlying `NDArray`.
*   **Automatic Dequantization**:
    *   Ensure that operations *not* marked as quantizable automatically dequantize `int8` inputs to `float32` before execution to maintain correctness.

### `python/needle/quantization.py` (New File)
**Goal**: Centralize quantization logic.

*   Implement utilities for:
    *   Computing `scale` and `zero_point` from min/max values.
    *   Per-axis scaling logic.
    *   Validation of quantization parameters.
*   **Calibration Infrastructure**:
    *   `Observer` classes (e.g., `MinMaxObserver`, `MovingAverageMinMaxObserver`) to collect statistics on activation distributions during calibration.
    *   `calibrate(model, dataloader)` function to run the model in calibration mode and populate observers.
*   This module will be used by both `NDArray` and NN modules.

## 2. Backend Implementation

### `python/needle/backend_ndarray/ndarray_backend_numpy.py`
**Goal**: Reference implementation using NumPy.

*   **Array Wrapper**: Update `Array` class to wrap `np.ndarray` with dynamic `dtype` and `itemsize`.
*   **Kernels**:
    *   Update elementwise kernels to handle `int8` inputs (upcast to `float32` for computation, then cast back).
    *   Implement `quantize_int8(src, dst, scale, zero_point)`.
    *   Implement `matmul_int8(a, b, out, sa, za, sb, zb, sout)`.

### `src/ndarray_backend_cpu.cc`
**Goal**: High-performance CPU implementation.

*   **AlignedArray**:
    *   Refactor to store `void* data` instead of `scalar_t*`.
    *   Add `DType` enum/struct and `elem_size` field.
*   **Kernels**:
    *   Keep existing float kernels (using typed pointer casts).
    *   Implement `QuantizeInt8`: Packs `float32` -> `int8`.
    *   Implement `DequantizeInt8`: Unpacks `int8` -> `float32`.
    *   Implement `MatmulInt8`:
        *   Inputs: `int8` matrices A and B, quantization params.
        *   Logic: Accumulate in `int32`, rescale results to `float32` output.
*   **PyBind**: Expose new kernels and dtype-aware constructors.

### `src/ndarray_backend_cuda.cu`
**Goal**: CUDA acceleration for quantized operations.

*   Mirror changes from CPU backend:
    *   Generalize `CudaArray` to support `void*` and dtypes.
    *   Implement CUDA kernels for `QuantizeInt8`, `DequantizeInt8`, and `MatmulInt8`.

## 3. Operator & Neural Network Support

### `python/needle/ops/ops_mathematic.py`
**Goal**: Enable quantized execution for specific operators.

*   **Quantizable Flag**: Add a mechanism to mark ops as "quantizable" (e.g., `MatMul`).
*   **MatMul Op**:
    *   Update `compute()` to check for `int8` inputs.
    *   If inputs are `int8`, invoke the backend's `matmul_int8`.
    *   Otherwise, proceed with standard `float32` execution.

### `python/needle/nn/nn_basic.py`
**Goal**: Quantization support for Linear layers.

*   **Linear Layer**:
    *   Store weights in `float32` by default.
    *   Add `observer` attribute (optional) to track input activation statistics.
    *   Implement `quantize_weights()`:
        *   Computes scales/zero-points for weights (likely per-output-channel).
        *   Creates and caches a `int8` version of the weights.
    *   Implement `quantize_activations_static()`:
        *   Uses calibrated scales/zero-points from the observer to quantize inputs.
    *   Update `forward()`:
        *   **Calibration Mode**: If enabled, update the observer with input statistics and run standard float forward.
        *   **Quantized Mode**:
            *   If static quantization is ready (observer has data), use `quantize_activations_static`.
            *   Else (dynamic), compute scale/zero-point on the fly.
            *   Perform `matmul` using quantized input and cached quantized weights.

## 4. Testing & Validation

### Unit Tests
*   **`tests/hw3/test_quantization.py`** (New):
    *   **Round-trip Accuracy**: Test `x.quantize_int8().dequantize()` vs `x` for random inputs. Assert error is within expected quantization noise bounds.
    *   **Backends**: Run on CPU, CUDA, and NumPy.
    *   **Metadata**: Verify `_quant_params` persist through slicing and reshaping.
    *   **Calibration**: Test that `MinMaxObserver` correctly captures the range of input data over multiple batches.

### Integration Tests
*   **`tests/hw3/test_ndarray.py`**:
    *   Extend to test creation of explicit `float32` vs `int8` arrays.
    *   Test broadcasting and reductions on `float32` arrays (regression testing).
*   **Mixed Precision**:
    *   Test `int8` @ `int8` (should use new kernel).
    *   Test `int8` @ `float32` (should fallback to float).
    *   Test unsupported ops (e.g., `ReLU(int8_tensor)`) to ensure they correctly dequantize and run.
*   **Static Quantization Workflow**:
    *   Train a small model (e.g., MLP on MNIST).
    *   Run calibration.
    *   Switch to quantized mode.
    *   Verify accuracy remains acceptable compared to float32 baseline.

### Benchmarking
*   Profile `MatMul` performance (Float32 vs Int8) on CPU and CUDA to verify speedups.

## 5. Demonstration Notebook

### `quantization_demo.ipynb` (New File)
**Goal**: Showcase the end-to-end quantization workflow and performance benefits.

*   **Introduction**: Briefly explain the concepts of Dynamic vs. Static Quantization and the benefits of `int8` inference.
*   **Setup**: Import Needle and necessary modules.
*   **Basic Usage**:
    *   Create a `float32` tensor.
    *   Demonstrate `quantize_int8()` and `dequantize()`.
    *   Show the underlying `int8` data and quantization parameters.
    *   Verify round-trip accuracy.
*   **Operator Demo**:
    *   Perform matrix multiplication with `int8` tensors.
    *   Compare the result with standard `float32` matmul.
    *   Show automatic fallback for unsupported ops (e.g., `ReLU` on `int8` input).
*   **Neural Network Demo (Dynamic & Static Quantization)**:
    *   **Architectures**:
        *   **MLP**: Simple fully connected network on MNIST.
        *   **CNN**: ResNet9 on CIFAR-10.
        *   **RNN/LSTM**: Language Model on Penn Treebank.
        *   **Transformer**: Sequence-to-sequence or language model task.
    *   For each architecture:
        *   Load pre-trained weights.
        *   Run **Dynamic Quantization** inference and measure accuracy/latency.
        *   Run **Calibration** for Static Quantization.
        *   Run **Static Quantization** inference and measure accuracy/latency.
        *   Compare results (Accuracy vs. Latency vs. Memory) against Float32 baseline.
*   **Performance Benchmark**:
    *   Measure and plot the inference latency (time per batch) for:
        *   Float32 Model
        *   Dynamic Quantized Model
        *   Static Quantized Model
    *   Highlight the speedup achieved on CPU/CUDA.
    *   **Memory Usage**:
        *   Measure peak GPU memory usage for Float32 vs. Quantized models.
        *   Demonstrate the reduction in memory footprint for weights and activations.
