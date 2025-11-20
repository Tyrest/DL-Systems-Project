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
