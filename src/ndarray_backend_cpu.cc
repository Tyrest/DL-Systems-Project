#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace needle {
namespace cpu {

#define ALIGNMENT 256
#define TILE 8
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);


/**
 * This is a utility structure for maintaining an array aligned to ALIGNMENT boundaries in
 * memory.  This alignment should be at least TILE * ELEM_SIZE, though we make it even larger
 * here by default.
 */
struct AlignedArray {
  AlignedArray(const size_t size, int8_t dtype = 0) {
    this->size = size;
    this->dtype = dtype;
    size_t elem_size = (dtype == 0) ? sizeof(scalar_t) : sizeof(int8_t);
    int ret = posix_memalign(&ptr, ALIGNMENT, size * elem_size);
    if (ret != 0) throw std::bad_alloc();
  }
  ~AlignedArray() { free(ptr); }
  
  template<typename T>
  T* as() const { return (T*)ptr; }
  
  size_t ptr_as_int() {return (size_t)ptr; }
  void* ptr;
  size_t size;
  int8_t dtype; // 0: float32, 1: int8
};



void Fill(AlignedArray* out, scalar_t val) {
  /**
   * Fill the values of an aligned array with val
   */
  scalar_t* out_ptr = out->as<scalar_t>();
  for (int i = 0; i < out->size; i++) {
    out_ptr[i] = val;
  }
}



template<typename Func>
void IndexIterator(std::vector<int32_t> shape, std::vector<int32_t> strides, 
                   size_t dim, size_t idx, Func fn, size_t& count) {
  if (dim >= shape.size()) {
    fn(idx, count++);
    return;
  }
  for (size_t i = 0; i < shape[dim]; i++) {
    IndexIterator(shape, strides, dim + 1, idx + strides[dim] * i, fn, count);
  }
}

void Compact(const AlignedArray& a, AlignedArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory
   */
  size_t count = 0;
  if (a.dtype == 1) { // int8
      const int8_t* a_ptr = a.as<int8_t>();
      int8_t* out_ptr = out->as<int8_t>();
      IndexIterator(shape, strides, 0, offset, 
                    [&](size_t idx, size_t compact_idx) {
                      out_ptr[compact_idx] = a_ptr[idx];
                    }, count);
  } else {
      const scalar_t* a_ptr = a.as<scalar_t>();
      scalar_t* out_ptr = out->as<scalar_t>();
      IndexIterator(shape, strides, 0, offset, 
                    [&](size_t idx, size_t compact_idx) {
                      out_ptr[compact_idx] = a_ptr[idx];
                    }, count);
  }
}

void EwiseSetitem(const AlignedArray& a, AlignedArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  size_t count = 0;
  if (a.dtype == 1) {
      const int8_t* a_ptr = a.as<int8_t>();
      int8_t* out_ptr = out->as<int8_t>();
      IndexIterator(shape, strides, 0, offset,
                    [&](size_t idx, size_t compact_idx) {
                      out_ptr[idx] = a_ptr[compact_idx];
                    }, count);
  } else {
      const scalar_t* a_ptr = a.as<scalar_t>();
      scalar_t* out_ptr = out->as<scalar_t>();
      IndexIterator(shape, strides, 0, offset,
                    [&](size_t idx, size_t compact_idx) {
                      out_ptr[idx] = a_ptr[compact_idx];
                    }, count);
  }
}

void ScalarSetitem(const size_t size, scalar_t val, AlignedArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  size_t count = 0;
  scalar_t* out_ptr = out->as<scalar_t>();
  IndexIterator(shape, strides, 0, offset,
                [&](size_t idx, size_t compact_idx) {
                  out_ptr[idx] = val;
                }, count);
}

void EwiseAdd(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  const scalar_t* a_ptr = a.as<scalar_t>();
  const scalar_t* b_ptr = b.as<scalar_t>();
  scalar_t* out_ptr = out->as<scalar_t>();
  for (size_t i = 0; i < a.size; i++) {
    out_ptr[i] = a_ptr[i] + b_ptr[i];
  }
}

void ScalarAdd(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  const scalar_t* a_ptr = a.as<scalar_t>();
  scalar_t* out_ptr = out->as<scalar_t>();
  for (size_t i = 0; i < a.size; i++) {
    out_ptr[i] = a_ptr[i] + val;
  }
}

#define EWISE_BINARY_OP(name, expr) \
void Ewise##name(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) { \
  const scalar_t* a_ptr = a.as<scalar_t>(); \
  const scalar_t* b_ptr = b.as<scalar_t>(); \
  scalar_t* out_ptr = out->as<scalar_t>(); \
  for (size_t i = 0; i < a.size; i++) { \
    out_ptr[i] = expr(a_ptr[i], b_ptr[i]); \
  } \
}

#define SCALAR_BINARY_OP(name, expr) \
void Scalar##name(const AlignedArray& a, scalar_t val, AlignedArray* out) { \
  const scalar_t* a_ptr = a.as<scalar_t>(); \
  scalar_t* out_ptr = out->as<scalar_t>(); \
  for (size_t i = 0; i < a.size; i++) { \
    out_ptr[i] = expr(a_ptr[i], val); \
  } \
}

#define EWISE_UNARY_OP(name, func) \
void Ewise##name(const AlignedArray& a, AlignedArray* out) { \
  const scalar_t* a_ptr = a.as<scalar_t>(); \
  scalar_t* out_ptr = out->as<scalar_t>(); \
  for (size_t i = 0; i < a.size; i++) { \
    out_ptr[i] = func(a_ptr[i]); \
  } \
}

#define OP_MUL(a, b) ((a) * (b))
#define OP_DIV(a, b) ((a) / (b))
#define OP_POW(a, b) (std::pow(a, b))
#define OP_MAX(a, b) (std::max(a, b))
#define OP_EQ(a, b) ((a) == (b))
#define OP_GE(a, b) ((a) >= (b))
#define OP_LOG(a) (std::log(a))
#define OP_EXP(a) (std::exp(a))
#define OP_TANH(a) (std::tanh(a))

EWISE_BINARY_OP(Mul, OP_MUL)
SCALAR_BINARY_OP(Mul, OP_MUL)

EWISE_BINARY_OP(Div, OP_DIV)
SCALAR_BINARY_OP(Div, OP_DIV)

SCALAR_BINARY_OP(Power, OP_POW)

EWISE_BINARY_OP(Maximum, OP_MAX)
SCALAR_BINARY_OP(Maximum, OP_MAX)

EWISE_BINARY_OP(Eq, OP_EQ)
SCALAR_BINARY_OP(Eq, OP_EQ)

EWISE_BINARY_OP(Ge, OP_GE)
SCALAR_BINARY_OP(Ge, OP_GE)

EWISE_UNARY_OP(Log, OP_LOG)
EWISE_UNARY_OP(Exp, OP_EXP)
EWISE_UNARY_OP(Tanh, OP_TANH)

void Matmul(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t m, uint32_t n,
            uint32_t p) {
  const scalar_t* a_ptr = a.as<scalar_t>();
  const scalar_t* b_ptr = b.as<scalar_t>();
  scalar_t* out_ptr = out->as<scalar_t>();
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      out_ptr[i * p + j] = 0;
      for (size_t k = 0; k < n; k++) {
        out_ptr[i * p + j] += a_ptr[i * n + k] * b_ptr[k * p + j];
      }
    }
  }
}

inline void AlignedDot(const float* __restrict__ a,
                       const float* __restrict__ b,
                       float* __restrict__ out) {
  a = (const float*)__builtin_assume_aligned(a, TILE * ELEM_SIZE);
  b = (const float*)__builtin_assume_aligned(b, TILE * ELEM_SIZE);
  out = (float*)__builtin_assume_aligned(out, TILE * ELEM_SIZE);

  for (size_t i = 0; i < TILE; i++) {
    for (size_t j = 0; j < TILE; j++) {
      for (size_t k = 0; k < TILE; k++) {
        out[i * TILE + j] += a[i * TILE + k] * b[k * TILE + j];
      }
    }
  }
}

void MatmulTiled(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t m,
                 uint32_t n, uint32_t p) {
  const scalar_t* a_ptr = a.as<scalar_t>();
  const scalar_t* b_ptr = b.as<scalar_t>();
  scalar_t* out_ptr = out->as<scalar_t>();
  
  for (size_t i = 0; i < m / TILE; i++) {
    for (size_t j = 0; j < p / TILE; j++) {
      for (size_t k = 0; k < TILE; k++) {
        for (size_t l = 0; l < TILE; l++) {
          out_ptr[(i * (p / TILE) + j) * TILE * TILE + k * TILE + l] = 0;
        }
      }
      for (size_t k = 0; k < n / TILE; k++) {
        AlignedDot(&a_ptr[(i * (n / TILE) + k) * TILE * TILE],
                   &b_ptr[(k * (p / TILE) + j) * TILE * TILE],
                   &out_ptr[(i * (p / TILE) + j) * TILE * TILE]);
      }
    }
  }
}

void ReduceMax(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
  const scalar_t* a_ptr = a.as<scalar_t>();
  scalar_t* out_ptr = out->as<scalar_t>();
  for (size_t i = 0; i < out->size; i++) {
    scalar_t max_val = a_ptr[i * reduce_size];
    for (size_t j = 1; j < reduce_size; j++) {
      max_val = std::max(max_val, a_ptr[i * reduce_size + j]);
    }
    out_ptr[i] = max_val;
  }
}

void ReduceSum(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
  const scalar_t* a_ptr = a.as<scalar_t>();
  scalar_t* out_ptr = out->as<scalar_t>();
  for (size_t i = 0; i < out->size; i++) {
    scalar_t max_val = a_ptr[i * reduce_size];
    for (size_t j = 1; j < reduce_size; j++) {
      max_val += a_ptr[i * reduce_size + j];
    }
    out_ptr[i] = max_val;
  }
}

void QuantizeInt8(const AlignedArray& src, AlignedArray* dst, float scale, int8_t zero_point) {
  const scalar_t* src_ptr = src.as<scalar_t>();
  int8_t* dst_ptr = dst->as<int8_t>();
  for (size_t i = 0; i < src.size; i++) {
    float val = round(src_ptr[i] / scale + zero_point);
    val = std::max(-128.0f, std::min(127.0f, val));
    dst_ptr[i] = (int8_t)val;
  }
}

void DequantizeInt8(const AlignedArray& src, AlignedArray* dst, float scale, int8_t zero_point) {
  const int8_t* src_ptr = src.as<int8_t>();
  scalar_t* dst_ptr = dst->as<scalar_t>();
  for (size_t i = 0; i < src.size; i++) {
    dst_ptr[i] = (scalar_t)(src_ptr[i] - zero_point) * scale;
  }
}

void MatmulInt8(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, 
                uint32_t m, uint32_t n, uint32_t p,
                float sa, int8_t za, float sb, int8_t zb) {
  const int8_t* a_ptr = a.as<int8_t>();
  const int8_t* b_ptr = b.as<int8_t>();
  scalar_t* out_ptr = out->as<scalar_t>();
  
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      float sum = 0;
      for (size_t k = 0; k < n; k++) {
        float val_a = (float)(a_ptr[i * n + k] - za);
        float val_b = (float)(b_ptr[k * p + j] - zb);
        sum += val_a * val_b;
      }
      out_ptr[i * p + j] = sum * sa * sb;
    }
  }
}

}  // namespace cpu
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cpu, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cpu;

  m.attr("__device_name__") = "cpu";
  m.attr("__tile_size__") = TILE;

  py::class_<AlignedArray>(m, "Array")
      .def(py::init<size_t, int8_t>(), py::arg("size"), py::arg("dtype") = 0, py::return_value_policy::take_ownership)
      .def("ptr", &AlignedArray::ptr_as_int)
      .def_readonly("size", &AlignedArray::size);

  m.def("to_numpy", [](const AlignedArray& a, std::vector<size_t> shape,
                       std::vector<size_t> strides, size_t offset) -> py::object {
    std::vector<size_t> numpy_strides = strides;
    size_t elem_size = (a.dtype == 0) ? ELEM_SIZE : 1;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [&](size_t& c) { return c * elem_size; });
    
    if (a.dtype == 0) {
        return py::array_t<scalar_t>(shape, numpy_strides, a.as<scalar_t>() + offset);
    } else {
        return py::array_t<int8_t>(shape, numpy_strides, a.as<int8_t>() + offset);
    }
  });

  m.def("from_numpy", [](py::array a, AlignedArray* out) {
    size_t elem_size = (out->dtype == 0) ? ELEM_SIZE : 1;
    std::memcpy(out->ptr, a.request().ptr, out->size * elem_size);
  });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_power", ScalarPower);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);

  m.def("matmul", Matmul);
  m.def("matmul_tiled", MatmulTiled);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
  
  m.def("quantize_int8", QuantizeInt8);
  m.def("dequantize_int8", DequantizeInt8);
  m.def("matmul_int8", MatmulInt8);
}
