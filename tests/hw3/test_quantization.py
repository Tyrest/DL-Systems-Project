import numpy as np
import needle as ndl


def test_quantize_round_trip():
    arr = np.random.uniform(-2.5, 2.5, size=(4, 5)).astype(np.float32)
    qt = ndl.quantization.quantize_int8(arr)
    recovered = ndl.quantization.dequantize_int8(qt).numpy()
    np.testing.assert_allclose(recovered, arr, rtol=1e-2, atol=5e-2)


def test_per_channel_quantization_shapes():
    arr = np.random.uniform(-4, 4, size=(3, 6)).astype(np.float32)
    qt = ndl.quantization.quantize_int8(arr, axis=1)
    assert qt.scale.shape == (1, 6)
    assert qt.zero_point.shape == (1, 6)
    recovered = ndl.quantization.dequantize_int8(qt).numpy()
    np.testing.assert_allclose(recovered, arr, rtol=1e-2, atol=5e-2)


def test_linear_quantized_matches_float():
    np.random.seed(0)
    lin = ndl.nn.Linear(8, 5, bias=True)
    x = ndl.Tensor(np.random.randn(4, 8).astype(np.float32), requires_grad=False)
    float_out = lin(x).numpy()

    lin.eval()
    lin.enable_quantization(axis=1)
    quant_out = lin(x).numpy()
    np.testing.assert_allclose(float_out, quant_out, rtol=1e-1, atol=1e-1)

    lin.train()
    train_out = lin(x).numpy()
    np.testing.assert_allclose(train_out, float_out, rtol=1e-6, atol=1e-6)


def test_quantize_zero_tensor():
    arr = np.zeros((2, 3), dtype=np.float32)
    qt = ndl.quantization.quantize_int8(arr)
    assert np.all(qt.data == 0)
    assert np.all(qt.scale > 0)
    assert qt.zero_point.shape == (1, 1)


def test_quantization_per_axis_negative_axis():
    arr = np.random.uniform(-3, 3, size=(3, 4)).astype(np.float32)
    qt = ndl.quantization.quantize_int8(arr, axis=-1)
    assert qt.scale.shape == (1, 4)
    assert qt.zero_point.shape == (1, 4)
    recovered = ndl.quantization.dequantize_int8(qt).numpy()
    np.testing.assert_allclose(recovered, arr, rtol=1e-2, atol=5e-2)


def test_quantized_tensor_cache_reuse():
    arr = np.random.randn(2, 2).astype(np.float32)
    qt = ndl.quantization.quantize_int8(arr)
    first = qt.dequantize().numpy()
    assert qt.cached_dequantized is not None
    second = qt.dequantize().numpy()
    np.testing.assert_array_equal(first, second)


def test_quantized_matmul_shape():
    a = ndl.Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=False)
    w = np.random.randn(4, 5).astype(np.float32)
    qt = ndl.quantization.quantize_int8(w, axis=0)
    out = ndl.quantization.quantized_matmul(a, qt)
    assert out.shape == (3, 5)


def test_linear_disable_quantization():
    lin = ndl.nn.Linear(4, 3, bias=False)
    lin.enable_quantization(axis=1)
    assert lin.use_int8
    lin.disable_quantization()
    assert not lin.use_int8
    assert lin._weight_q is None


def test_quantization_bias_effect():
    lin = ndl.nn.Linear(3, 2, bias=True)
    x = ndl.Tensor(np.ones((1, 3), dtype=np.float32), requires_grad=False)
    float_out = lin(x).numpy()
    lin.eval()
    lin.enable_quantization(axis=1)
    quant_out = lin(x).numpy()
    np.testing.assert_allclose(float_out, quant_out, rtol=1e-1, atol=1e-1)


def test_quantization_memory_bytes_smaller():
    arr = np.random.randn(16, 16).astype(np.float32)
    qt = ndl.quantization.quantize_int8(arr)
    float_bytes = arr.size * 4
    assert qt.memory_bytes() < float_bytes


def test_linear_multiple_forward_reuses_quantized():
    lin = ndl.nn.Linear(6, 4, bias=True)
    x = ndl.Tensor(np.random.randn(5, 6).astype(np.float32), requires_grad=False)
    lin.eval()
    lin.enable_quantization(axis=1)
    out1 = lin(x).numpy()
    out2 = lin(x).numpy()
    np.testing.assert_allclose(out1, out2, rtol=1e-6, atol=1e-6)


def test_quantize_asymmetric_roundtrip():
    arr = np.linspace(-1.2, 0.8, 12, dtype=np.float32).reshape(3, 4)
    qt = ndl.quantization.quantize_int8(arr, symmetric=False)
    recovered = ndl.quantization.dequantize_int8(qt).numpy()
    np.testing.assert_allclose(recovered, arr, rtol=1e-2, atol=5e-2)


def test_zero_point_clipped_to_range():
    arr = np.full((2, 2), 10.0, dtype=np.float32)
    qt = ndl.quantization.quantize_int8(arr, symmetric=False)
    assert np.all(qt.zero_point >= -128)
    assert np.all(qt.zero_point <= 127)


def test_scale_positive():
    arr = np.random.randn(4, 4).astype(np.float32)
    qt = ndl.quantization.quantize_int8(arr, axis=1)
    assert np.all(qt.scale > 0)


def test_quantized_matmul_matches_float():
    a = np.random.randn(2, 3).astype(np.float32)
    b = np.random.randn(3, 4).astype(np.float32)
    qt = ndl.quantization.quantize_int8(b, axis=0)
    out = ndl.quantization.quantized_matmul(ndl.Tensor(a, requires_grad=False), qt).numpy()
    np.testing.assert_allclose(out, a @ b, rtol=1e-1, atol=1e-1)


def test_memory_bytes_includes_scale_overhead():
    arr = np.random.randn(8, 8).astype(np.float32)
    qt = ndl.quantization.quantize_int8(arr)
    assert qt.memory_bytes() > qt.data.nbytes


def test_linear_train_mode_after_quantization_uses_float():
    lin = ndl.nn.Linear(5, 4, bias=True)
    x = ndl.Tensor(np.random.randn(3, 5).astype(np.float32), requires_grad=False)
    ref = lin(x).numpy()
    lin.enable_quantization(axis=1)
    lin.train()
    out = lin(x).numpy()
    np.testing.assert_allclose(out, ref, rtol=1e-6, atol=1e-6)


def test_linear_eval_quantization_caches_weight():
    lin = ndl.nn.Linear(4, 3, bias=False)
    lin.eval()
    lin.enable_quantization(axis=1)
    assert lin.use_int8
    assert lin._weight_q is not None
    assert lin._weight_deq is not None


def test_quantize_small_range():
    arr = np.full((3, 3), 1e-4, dtype=np.float32)
    qt = ndl.quantization.quantize_int8(arr)
    rec = ndl.quantization.dequantize_int8(qt).numpy()
    np.testing.assert_allclose(rec, arr, rtol=1e-2, atol=1e-3)


def test_quantization_per_axis_shape_three_dim():
    arr = np.random.randn(2, 5, 7).astype(np.float32)
    qt = ndl.quantization.quantize_int8(arr, axis=2)
    assert qt.scale.shape == (1, 1, 7)
    assert qt.zero_point.shape == (1, 1, 7)


def test_dequantize_reuses_cached_array():
    arr = np.random.randn(2, 2).astype(np.float32)
    qt = ndl.quantization.quantize_int8(arr)
    t1 = qt.dequantize()
    t2 = qt.dequantize()
    assert t1.realize_cached_data() is t2.realize_cached_data()
