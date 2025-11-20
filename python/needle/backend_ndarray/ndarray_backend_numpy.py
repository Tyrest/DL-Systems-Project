import numpy as np


__device_name__ = "numpy"


def _dtype_to_numpy(dtype: str):
    """Convert dtype string to numpy dtype."""
    if dtype == "float32":
        return np.float32
    elif dtype == "uint8":
        return np.uint8
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


class Array:
    def __init__(self, size, dtype="float32"):
        np_dtype = _dtype_to_numpy(dtype)
        self.array = np.empty(size, dtype=np_dtype)
        self.dtype = dtype

    @property
    def size(self):
        return self.array.size


def to_numpy(a, shape, strides, offset):
    itemsize = a.array.itemsize
    return np.lib.stride_tricks.as_strided(
        a.array[offset:], shape, tuple([s * itemsize for s in strides])
    )


def from_numpy(a, out):
    out.array[:] = a.flatten()


def fill(out, val):
    out.array.fill(val)


def compact(a, out, shape, strides, offset):
    out.array[:] = to_numpy(a, shape, strides, offset).flatten()


def ewise_setitem(a, out, shape, strides, offset):
    to_numpy(out, shape, strides, offset)[:] = a.array.reshape(shape)


def scalar_setitem(size, val, out, shape, strides, offset):
    to_numpy(out, shape, strides, offset)[:] = val


def ewise_add(a, b, out):
    # Upcast uint8 to float32 for computation
    a_data = a.array.astype(np.float32) if a.array.dtype == np.uint8 else a.array
    b_data = b.array.astype(np.float32) if b.array.dtype == np.uint8 else b.array
    result = a_data + b_data
    # Cast back to output dtype
    out.array[:] = result.astype(out.array.dtype)


def scalar_add(a, val, out):
    a_data = a.array.astype(np.float32) if a.array.dtype == np.uint8 else a.array
    result = a_data + val
    out.array[:] = result.astype(out.array.dtype)


def ewise_mul(a, b, out):
    a_data = a.array.astype(np.float32) if a.array.dtype == np.uint8 else a.array
    b_data = b.array.astype(np.float32) if b.array.dtype == np.uint8 else b.array
    result = a_data * b_data
    out.array[:] = result.astype(out.array.dtype)


def scalar_mul(a, val, out):
    a_data = a.array.astype(np.float32) if a.array.dtype == np.uint8 else a.array
    result = a_data * val
    out.array[:] = result.astype(out.array.dtype)


def ewise_div(a, b, out):
    a_data = a.array.astype(np.float32) if a.array.dtype == np.uint8 else a.array
    b_data = b.array.astype(np.float32) if b.array.dtype == np.uint8 else b.array
    result = a_data / b_data
    out.array[:] = result.astype(out.array.dtype)


def scalar_div(a, val, out):
    a_data = a.array.astype(np.float32) if a.array.dtype == np.uint8 else a.array
    result = a_data / val
    out.array[:] = result.astype(out.array.dtype)


def scalar_power(a, val, out):
    out.array[:] = a.array**val


def ewise_maximum(a, b, out):
    out.array[:] = np.maximum(a.array, b.array)


def scalar_maximum(a, val, out):
    out.array[:] = np.maximum(a.array, val)


def ewise_eq(a, b, out):
    out.array[:] = (a.array == b.array).astype(np.float32)


def scalar_eq(a, val, out):
    out.array[:] = (a.array == val).astype(np.float32)


def ewise_ge(a, b, out):
    out.array[:] = (a.array >= b.array).astype(np.float32)


def scalar_ge(a, val, out):
    out.array[:] = (a.array >= val).astype(np.float32)


def ewise_log(a, out):
    out.array[:] = np.log(a.array)


def ewise_exp(a, out):
    out.array[:] = np.exp(a.array)


def ewise_tanh(a, out):
    out.array[:] = np.tanh(a.array)


def matmul(a, b, out, m, n, p):
    out.array[:] = (a.array.reshape(m, n) @ b.array.reshape(n, p)).reshape(-1)


def reduce_max(a, out, reduce_size):
    out.array[:] = a.array[:].reshape(-1, reduce_size).max(axis=1)


def reduce_sum(a, out, reduce_size):
    out.array[:] = a.array[:].reshape(-1, reduce_size).sum(axis=1)


def quantize_uint8(a, out, scale, zero_point):
    """Quantize float32 array to uint8."""
    quantized = np.round(a.array / scale) + zero_point
    quantized = np.clip(quantized, 0, 255)
    out.array[:] = quantized.astype(np.uint8)


def dequantize_uint8(a, out, scale, zero_point):
    """Dequantize uint8 array to float32."""
    dequantized = scale * (a.array.astype(np.float32) - zero_point)
    out.array[:] = dequantized


def matmul_uint8(a, b, out, m, n, p, scale_a, zero_a, scale_b, zero_b):
    """Quantized matrix multiplication.
    
    Performs integer matrix multiplication and rescales the result.
    Formula: (scale_a * scale_b) * ((a_int - zero_a) @ (b_int - zero_b))
    """
    # Convert to int32 for accumulation
    a_int = a.array.reshape(m, n).astype(np.int32) - zero_a
    b_int = b.array.reshape(n, p).astype(np.int32) - zero_b
    
    # Integer matmul
    result_int = a_int @ b_int
    
    # Rescale to float32
    result_float = (scale_a * scale_b) * result_int.astype(np.float32)
    out.array[:] = result_float.reshape(-1)
