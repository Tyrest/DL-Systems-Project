"""Tests for quantization functionality."""
import sys
sys.path.append('./python')

import numpy as np
import needle as ndl
from needle import backend_ndarray as nd


def test_ndarray_quantization_basic():
    """Test basic NDArray quantization and dequantization."""
    print("Testing NDArray quantization...")
    
    # Create a simple array
    np_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    arr = nd.NDArray(np_data)
    
    print(f"Original array:\n{arr.numpy()}")
    print(f"Original dtype: {arr.dtype}")
    
    # Quantize
    arr_quant = arr.quantize_uint8()
    print(f"\nQuantized array:\n{arr_quant.numpy()}")
    print(f"Quantized dtype: {arr_quant.dtype}")
    print(f"Quantization params: scale={arr_quant.quant_params.scale}, zero_point={arr_quant.quant_params.zero_point}")
    
    # Dequantize
    arr_dequant = arr_quant.dequantize()
    print(f"\nDequantized array:\n{arr_dequant.numpy()}")
    print(f"Dequantized dtype: {arr_dequant.dtype}")
    
    # Check error
    error = np.abs(arr.numpy() - arr_dequant.numpy()).max()
    print(f"\nMax error after quantization roundtrip: {error}")
    print(f"Test passed: {error < 0.5}")  # Allow some quantization error
    print("-" * 60)


def test_quantized_matmul():
    """Test quantized matrix multiplication."""
    print("\nTesting quantized matmul...")
    
    # Create two matrices
    np.random.seed(0)
    a_np = np.random.randn(4, 3).astype(np.float32)
    b_np = np.random.randn(3, 5).astype(np.float32)
    
    a = nd.NDArray(a_np)
    b = nd.NDArray(b_np)
    
    # Float32 matmul
    c_float = a @ b
    print(f"Float32 matmul result shape: {c_float.shape}")
    print(f"Float32 result sample:\n{c_float.numpy()[:2, :3]}")
    
    # Quantize inputs
    a_quant = a.quantize_uint8()
    b_quant = b.quantize_uint8()
    
    # Quantized matmul (should automatically use uint8 kernel)
    c_quant = a_quant @ b_quant
    print(f"\nQuantized matmul result shape: {c_quant.shape}")
    print(f"Quantized result sample:\n{c_quant.numpy()[:2, :3]}")
    
    # Compare
    error = np.abs(c_float.numpy() - c_quant.numpy()).max()
    print(f"\nMax error between float32 and quantized matmul: {error}")
    print(f"Test passed: {error < 1.0}")  # Allow quantization error
    print("-" * 60)


def test_tensor_quantization():
    """Test Tensor quantization API."""
    print("\nTesting Tensor quantization...")
    
    # Create a tensor
    np_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    tensor = ndl.Tensor(np_data, dtype="float32")
    
    print(f"Original tensor:\n{tensor.numpy()}")
    print(f"Original dtype: {tensor.dtype}")
    
    # Quantize
    tensor_quant = tensor.quantize_uint8()
    print(f"\nQuantized tensor:\n{tensor_quant.numpy()}")
    print(f"Quantized dtype: {tensor_quant.dtype}")
    
    # Dequantize
    tensor_dequant = tensor_quant.dequantize()
    print(f"\nDequantized tensor:\n{tensor_dequant.numpy()}")
    
    error = np.abs(tensor.numpy() - tensor_dequant.numpy()).max()
    print(f"\nMax error: {error}")
    print(f"Test passed: {error < 0.5}")
    print("-" * 60)


def test_linear_layer_quantization():
    """Test quantized Linear layer."""
    print("\nTesting Linear layer quantization...")
    
    # Create a simple linear layer
    np.random.seed(42)
    layer = ndl.nn.Linear(4, 3, bias=True, device=ndl.cpu_numpy(), dtype="float32")
    
    # Create input
    x_np = np.random.randn(2, 4).astype(np.float32)
    x = ndl.Tensor(x_np, dtype="float32")
    
    # Forward pass in training mode (float32)
    layer.train()
    y_train = layer(x)
    print(f"Training mode output shape: {y_train.shape}")
    print(f"Training output:\n{y_train.numpy()}")
    
    # Quantize weights and switch to eval mode
    layer.eval()
    layer.quantize_weights()
    
    # Forward pass in eval mode (quantized)
    y_eval_quant = layer(x)
    print(f"\nEval mode (quantized) output shape: {y_eval_quant.shape}")
    print(f"Quantized output:\n{y_eval_quant.numpy()}")
    
    # Compare
    error = np.abs(y_train.numpy() - y_eval_quant.numpy()).max()
    print(f"\nMax error between float32 and quantized: {error}")
    print(f"Test passed: {error < 1.0}")
    print("-" * 60)


if __name__ == "__main__":
    print("=" * 60)
    print("QUANTIZATION TESTS")
    print("=" * 60)
    
    test_ndarray_quantization_basic()
    test_quantized_matmul()
    test_tensor_quantization()
    test_linear_layer_quantization()
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)
