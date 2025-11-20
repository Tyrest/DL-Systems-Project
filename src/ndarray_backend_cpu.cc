#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>
#ifdef __AVX2__
#include <immintrin.h>
#endif

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
  AlignedArray(const size_t size) {
    int ret = posix_memalign((void**)&ptr, ALIGNMENT, size * ELEM_SIZE);
    if (ret != 0) throw std::bad_alloc();
    this->size = size;
  }
  ~AlignedArray() { free(ptr); }
  size_t ptr_as_int() {return (size_t)ptr; }
  scalar_t* ptr;
  size_t size;
};



void Fill(AlignedArray* out, scalar_t val) {
  /**
   * Fill the values of an aligned array with val
   */
  for (int i = 0; i < out->size; i++) {
    out->ptr[i] = val;
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
   *
   * Args:
   *   a: non-compact representation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   *
   * Returns:
   *  void (you need to modify out directly, rather than returning anything; this is true for all the
   *  function will implement here, so we won't repeat this note.)
   */
  /// BEGIN SOLUTION
  size_t count = 0;
  IndexIterator(shape, strides, 0, offset, 
                [&](size_t idx, size_t compact_idx) {
                  out->ptr[compact_idx] = a.ptr[idx];
                }, count);
  /// END SOLUTION
}

void EwiseSetitem(const AlignedArray& a, AlignedArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array
   *
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  /// BEGIN SOLUTION
  size_t count = 0;
  IndexIterator(shape, strides, 0, offset,
                [&](size_t idx, size_t compact_idx) {
                  out->ptr[idx] = a.ptr[compact_idx];
                }, count);
  /// END SOLUTION
}

void ScalarSetitem(const size_t size, scalar_t val, AlignedArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   *
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the
   *         product of items in shape, but convenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */

  /// BEGIN SOLUTION
  size_t count = 0;
  IndexIterator(shape, strides, 0, offset,
                [&](size_t idx, size_t compact_idx) {
                  out->ptr[idx] = val;
                }, count);
  /// END SOLUTION
}

void EwiseAdd(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of correspondings entires in a and b.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] + b.ptr[i];
  }
}

void ScalarAdd(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of corresponding entry in a plus the scalar val.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] + val;
  }
}


/**
 * In the code the follows, use the above template to create analogous element-wise
 * and and scalar operators for the following functions.  See the numpy backend for
 * examples of how they should work.
 *   - EwiseMul, ScalarMul
 *   - EwiseDiv, ScalarDiv
 *   - ScalarPower
 *   - EwiseMaximum, ScalarMaximum
 *   - EwiseEq, ScalarEq
 *   - EwiseGe, ScalarGe
 *   - EwiseLog
 *   - EwiseExp
 *   - EwiseTanh
 *
 * If you implement all these naively, there will be a lot of repeated code, so
 * you are welcome (but not required), to use macros or templates to define these
 * functions (however you want to do so, as long as the functions match the proper)
 * signatures above.
 */

#define EWISE_BINARY_OP(name, expr) \
void Ewise##name(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) { \
  for (size_t i = 0; i < a.size; i++) { \
    out->ptr[i] = expr(a.ptr[i], b.ptr[i]); \
  } \
}

#define SCALAR_BINARY_OP(name, expr) \
void Scalar##name(const AlignedArray& a, scalar_t val, AlignedArray* out) { \
  for (size_t i = 0; i < a.size; i++) { \
    out->ptr[i] = expr(a.ptr[i], val); \
  } \
}

#define EWISE_UNARY_OP(name, func) \
void Ewise##name(const AlignedArray& a, AlignedArray* out) { \
  for (size_t i = 0; i < a.size; i++) { \
    out->ptr[i] = func(a.ptr[i]); \
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
  /**
   * Multiply two (compact) matrices into an output (also compact) matrix.  For this implementation
   * you can use the "naive" three-loop algorithm.
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: compact 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   m: rows of a / out
   *   n: columns of a / rows of b
   *   p: columns of b / out
   */

  /// BEGIN SOLUTION
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      out->ptr[i * p + j] = 0;
      for (size_t k = 0; k < n; k++) {
        out->ptr[i * p + j] += a.ptr[i * n + k] * b.ptr[k * p + j];
      }
    }
  }
  /// END SOLUTION
}

inline void AlignedDot(const float* __restrict__ a,
                       const float* __restrict__ b,
                       float* __restrict__ out) {

  /**
   * Multiply together two TILE x TILE matrices, and _add _the result to out (it is important to add
   * the result to the existing out, which you should not set to zero beforehand).  We are including
   * the compiler flags here that enable the compile to properly use vector operators to implement
   * this function.  Specifically, the __restrict__ keyword indicates to the compile that a, b, and
   * out don't have any overlapping memory (which is necessary in order for vector operations to be
   * equivalent to their non-vectorized counterparts (imagine what could happen otherwise if a, b,
   * and out had overlapping memory).  Similarly the __builtin_assume_aligned keyword tells the
   * compiler that the input array will be aligned to the appropriate blocks in memory, which also
   * helps the compiler vectorize the code.
   *
   * Args:
   *   a: compact 2D array of size TILE x TILE
   *   b: compact 2D array of size TILE x TILE
   *   out: compact 2D array of size TILE x TILE to write to
   */

  a = (const float*)__builtin_assume_aligned(a, TILE * ELEM_SIZE);
  b = (const float*)__builtin_assume_aligned(b, TILE * ELEM_SIZE);
  out = (float*)__builtin_assume_aligned(out, TILE * ELEM_SIZE);

  /// BEGIN SOLUTION
  for (size_t i = 0; i < TILE; i++) {
    for (size_t j = 0; j < TILE; j++) {
      for (size_t k = 0; k < TILE; k++) {
        out[i * TILE + j] += a[i * TILE + k] * b[k * TILE + j];
      }
    }
  }
  /// END SOLUTION
}

void MatmulTiled(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t m,
                 uint32_t n, uint32_t p) {
  /**
   * Matrix multiplication on tiled representations of array.  In this setting, a, b, and out
   * are all *4D* compact arrays of the appropriate size, e.g. a is an array of size
   *   a[m/TILE][n/TILE][TILE][TILE]
   * You should do the multiplication tile-by-tile to improve performance of the array (i.e., this
   * function should call `AlignedDot()` implemented above).
   *
   * Note that this function will only be called when m, n, p are all multiples of TILE, so you can
   * assume that this division happens without any remainder.
   *
   * Args:
   *   a: compact 4D array of size m/TILE x n/TILE x TILE x TILE
   *   b: compact 4D array of size n/TILE x p/TILE x TILE x TILE
   *   out: compact 4D array of size m/TILE x p/TILE x TILE x TILE to write to
   *   m: rows of a / out
   *   n: columns of a / rows of b
   *   p: columns of b / out
   *
   */
  /// BEGIN SOLUTION
  for (size_t i = 0; i < m / TILE; i++) {
    for (size_t j = 0; j < p / TILE; j++) {
      for (size_t k = 0; k < TILE; k++) {
        for (size_t l = 0; l < TILE; l++) {
          out->ptr[(i * (p / TILE) + j) * TILE * TILE + k * TILE + l] = 0;
        }
      }
      for (size_t k = 0; k < n / TILE; k++) {
        AlignedDot(&a.ptr[(i * (n / TILE) + k) * TILE * TILE],
                   &b.ptr[(k * (p / TILE) + j) * TILE * TILE],
                   &out->ptr[(i * (p / TILE) + j) * TILE * TILE]);
      }
    }
  }
  /// END SOLUTION
}

void ReduceMax(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   reduce_size: size of the dimension to reduce over
   */

  /// BEGIN SOLUTION
  for (size_t i = 0; i < out->size; i++) {
    scalar_t max_val = a.ptr[i * reduce_size];
    for (size_t j = 1; j < reduce_size; j++) {
      max_val = std::max(max_val, a.ptr[i * reduce_size + j]);
    }
    out->ptr[i] = max_val;
  }
  /// END SOLUTION
}

void ReduceSum(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
  /**
   * Reduce by taking sum over `reduce_size` contiguous blocks.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   reduce_size: size of the dimension to reduce over
   */

  /// BEGIN SOLUTION
  for (size_t i = 0; i < out->size; i++) {
    scalar_t max_val = a.ptr[i * reduce_size];
    for (size_t j = 1; j < reduce_size; j++) {
      max_val += a.ptr[i * reduce_size + j];
    }
    out->ptr[i] = max_val;
  }
  /// END SOLUTION
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
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def("ptr", &AlignedArray::ptr_as_int)
      .def_readonly("size", &AlignedArray::size);

  // return numpy array (with copying for simplicity, otherwise garbage
  // collection is a pain)
  m.def("to_numpy", [](const AlignedArray& a, std::vector<size_t> shape,
                       std::vector<size_t> strides, size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });
    return py::array_t<scalar_t>(shape, numpy_strides, a.ptr + offset);
  });

  // convert from numpy (with copying)
  m.def("from_numpy", [](py::array_t<scalar_t> a, AlignedArray* out) {
    std::memcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE);
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

  // Simple int8 matmul for benchmark/inference use.
  // Inputs are numpy int8 arrays with shapes (m,k) and (k,n), plus scales.
  // Produces float32 numpy array.
  m.def(
      "matmul_int8",
      [](py::array_t<int8_t, py::array::c_style | py::array::forcecast> a,
         py::array_t<int8_t, py::array::c_style | py::array::forcecast> b,
         float a_scale, float b_scale) {
        auto a_buf = a.request();
        auto b_buf = b.request();
        if (a_buf.ndim != 2 || b_buf.ndim != 2) {
          throw std::runtime_error("int8 matmul expects 2D inputs");
        }
        auto m = a_buf.shape[0];
        auto k_a = a_buf.shape[1];
        auto k_b = b_buf.shape[0];
        auto n = b_buf.shape[1];
        if (k_a != k_b) {
          throw std::runtime_error("Inner dims must match");
        }
        // Transpose B to make columns contiguous for vectorized dot products.
        std::vector<int8_t> bT(n * k_a);
        {
          int8_t* b_ptr = static_cast<int8_t*>(b_buf.ptr);
          for (ssize_t kk = 0; kk < k_a; kk++) {
            for (ssize_t j = 0; j < n; j++) {
              bT[j * k_a + kk] = b_ptr[kk * n + j];
            }
          }
        }
        py::array_t<float> out({m, n});
        auto out_buf = out.request();
        int8_t* a_ptr = static_cast<int8_t*>(a_buf.ptr);
        float* o_ptr = static_cast<float*>(out_buf.ptr);
#ifdef __AVX2__
        __m256i ones = _mm256_set1_epi16(1);
#endif
        for (ssize_t i = 0; i < m; i++) {
          int8_t* ai = a_ptr + i * k_a;
          for (ssize_t j = 0; j < n; j++) {
            int32_t acc = 0;
#ifdef __AVX2__
            int kk = 0;
            __m256i vacc0 = _mm256_setzero_si256();
            __m256i vacc1 = _mm256_setzero_si256();
            for (; kk + 32 <= k_a; kk += 32) {
              __m256i va0 = _mm256_cvtepi8_epi16(
                  _mm_loadu_si128(reinterpret_cast<const __m128i*>(ai + kk)));
              __m256i vb0 = _mm256_cvtepi8_epi16(
                  _mm_loadu_si128(reinterpret_cast<const __m128i*>(bT.data() + j * k_a + kk)));
              __m256i prod0 = _mm256_mullo_epi16(va0, vb0);
              vacc0 = _mm256_add_epi32(vacc0, _mm256_madd_epi16(prod0, ones));

              __m256i va1 = _mm256_cvtepi8_epi16(
                  _mm_loadu_si128(reinterpret_cast<const __m128i*>(ai + kk + 16)));
              __m256i vb1 = _mm256_cvtepi8_epi16(
                  _mm_loadu_si128(reinterpret_cast<const __m128i*>(bT.data() + j * k_a + kk + 16)));
              __m256i prod1 = _mm256_mullo_epi16(va1, vb1);
              vacc1 = _mm256_add_epi32(vacc1, _mm256_madd_epi16(prod1, ones));
            }
            // horizontal add accumulators
            alignas(32) int32_t tmp[8];
            _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), vacc0);
            for (int t = 0; t < 8; t++) acc += tmp[t];
            _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), vacc1);
            for (int t = 0; t < 8; t++) acc += tmp[t];
            for (; kk < k_a; kk++) {
              acc += static_cast<int32_t>(ai[kk]) * static_cast<int32_t>(bT[j * k_a + kk]);
            }
#else
            for (ssize_t kk = 0; kk < k_a; kk++) {
              acc += static_cast<int32_t>(ai[kk]) *
                     static_cast<int32_t>(bT[j * k_a + kk]);
            }
#endif
            o_ptr[i * n + j] = static_cast<float>(acc) * a_scale * b_scale;
          }
        }
        return out;
      },
      py::arg("a"), py::arg("b"), py::arg("a_scale"), py::arg("b_scale"),
      "Int8 matmul (a: m x k, b: k x n) returning float32.");
}
