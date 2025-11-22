#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>

namespace needle {
namespace cuda {

#define BASE_THREAD_NUM 256

#define TILE 4
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);

struct CudaArray {
  CudaArray(const size_t size, int8_t dtype = 0) {
    this->size = size;
    this->dtype = dtype;
    size_t elem_size = (dtype == 0) ? sizeof(scalar_t) : sizeof(int8_t);
    cudaError_t err = cudaMalloc(&ptr, size * elem_size);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  }
  ~CudaArray() { cudaFree(ptr); }
  size_t ptr_as_int() { return (size_t)ptr; }
  
  template<typename T>
  T* as() const { return (T*)ptr; }
  
  void* ptr;
  size_t size;
  int8_t dtype;
};

struct CudaDims {
  dim3 block, grid;
};

CudaDims CudaOneDim(size_t size) {
  CudaDims dim;
  size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
  dim.block = dim3(BASE_THREAD_NUM, 1, 1);
  dim.grid = dim3(num_blocks, 1, 1);
  return dim;
}

#define MAX_VEC_SIZE 8
struct CudaVec {
  uint32_t size;
  int32_t data[MAX_VEC_SIZE];
};

CudaVec VecToCuda(const std::vector<int32_t>& x) {
  CudaVec shape;
  if (x.size() > MAX_VEC_SIZE) throw std::runtime_error("Exceeded CUDA supported max dimesions");
  shape.size = x.size();
  for (size_t i = 0; i < x.size(); i++) {
    shape.data[i] = x[i];
  }
  return shape;
}

__global__ void FillKernel(scalar_t* out, scalar_t val, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = val;
}

void Fill(CudaArray* out, scalar_t val) {
  CudaDims dim = CudaOneDim(out->size);
  FillKernel<<<dim.grid, dim.block>>>(out->as<scalar_t>(), val, out->size);
}

__device__ size_t GetMemoryLocation(const CudaVec& shape, const CudaVec& strides, size_t i) {
  size_t idx = 0;
  for (size_t j = 0; j < strides.size; j++) {
    size_t dim = shape.size - 1 - j;
    idx += (i % shape.data[dim]) * strides.data[dim];
    i /= shape.data[dim];
  }
  return idx;
}

template <typename T>
__global__ void CompactKernel(const T* a, T* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    out[gid] = a[GetMemoryLocation(shape, strides, gid) + offset];
  }
}

void Compact(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  CudaDims dim = CudaOneDim(out->size);
  if (a.dtype == 0) {
      CompactKernel<<<dim.grid, dim.block>>>(a.as<scalar_t>(), out->as<scalar_t>(), out->size, VecToCuda(shape),
                                             VecToCuda(strides), offset);
  } else {
      CompactKernel<<<dim.grid, dim.block>>>(a.as<int8_t>(), out->as<int8_t>(), out->size, VecToCuda(shape),
                                             VecToCuda(strides), offset);
  }
}

template <typename T>
__global__ void EwiseSetitemKernel(const T* a, T* out, size_t size, CudaVec shape,
                                   CudaVec strides, size_t offset) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    out[GetMemoryLocation(shape, strides, gid) + offset] = a[gid];
  }
}

void EwiseSetitem(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  CudaDims dim = CudaOneDim(a.size);
  if (a.dtype == 0) {
      EwiseSetitemKernel<<<dim.grid, dim.block>>>(a.as<scalar_t>(), out->as<scalar_t>(), a.size, VecToCuda(shape),
                                                  VecToCuda(strides), offset);
  } else {
      EwiseSetitemKernel<<<dim.grid, dim.block>>>(a.as<int8_t>(), out->as<int8_t>(), a.size, VecToCuda(shape),
                                                  VecToCuda(strides), offset);
  }
}

__global__ void ScalarSetitemKernel(size_t size, scalar_t val, scalar_t* out, CudaVec shape,
                                    CudaVec strides, size_t offset) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    out[GetMemoryLocation(shape, strides, gid) + offset] = val;
  }
}

void ScalarSetitem(size_t size, scalar_t val, CudaArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarSetitemKernel<<<dim.grid, dim.block>>>(size, val, out->as<scalar_t>(), VecToCuda(shape),
                                               VecToCuda(strides), offset);
}

__global__ void EwiseAddKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + b[gid];
}

void EwiseAdd(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseAddKernel<<<dim.grid, dim.block>>>(a.as<scalar_t>(), b.as<scalar_t>(), out->as<scalar_t>(), out->size);
}

__global__ void ScalarAddKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + val;
}

void ScalarAdd(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarAddKernel<<<dim.grid, dim.block>>>(a.as<scalar_t>(), val, out->as<scalar_t>(), out->size);
}

#define EWISE_BINARY_OP(name, expr) \
__global__ void Ewise##name##Kernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) { \
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x; \
  if (gid < size) out[gid] = expr(a[gid], b[gid]); \
} \
void Ewise##name(const CudaArray& a, const CudaArray& b, CudaArray* out) { \
  CudaDims dim = CudaOneDim(out->size); \
  Ewise##name##Kernel<<<dim.grid, dim.block>>>(a.as<scalar_t>(), b.as<scalar_t>(), out->as<scalar_t>(), out->size); \
}

#define SCALAR_BINARY_OP(name, expr) \
__global__ void Scalar##name##Kernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) { \
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x; \
  if (gid < size) out[gid] = expr(a[gid], val); \
} \
void Scalar##name(const CudaArray& a, scalar_t val, CudaArray* out) { \
  CudaDims dim = CudaOneDim(out->size); \
  Scalar##name##Kernel<<<dim.grid, dim.block>>>(a.as<scalar_t>(), val, out->as<scalar_t>(), out->size); \
}

#define EWISE_UNARY_OP(name, expr) \
__global__ void Ewise##name##Kernel(const scalar_t* a, scalar_t* out, size_t size) { \
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x; \
  if (gid < size) out[gid] = expr(a[gid]); \
} \
void Ewise##name(const CudaArray& a, CudaArray* out) { \
  CudaDims dim = CudaOneDim(out->size); \
  Ewise##name##Kernel<<<dim.grid, dim.block>>>(a.as<scalar_t>(), out->as<scalar_t>(), out->size); \
}

#define OP_MUL(a, b) ((a) * (b))
#define OP_DIV(a, b) ((a) / (b))
#define OP_POW(a, b) (pow(a, b))
#define OP_MAX(a, b) (max(a, b))
#define OP_EQ(a, b) ((a) == (b))
#define OP_GE(a, b) ((a) >= (b))
#define OP_LOG(a) (log(a))
#define OP_EXP(a) (exp(a))
#define OP_TANH(a) (tanh(a))

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


#define MATMUL_S 8
#define MATMUL_L 8
#define MATMUL_V 2

__device__ size_t idx2d(size_t y, size_t x, size_t cols) {
  return y * cols + x;
}

__device__ void LoadTileToShared(const scalar_t* A, const scalar_t* B, scalar_t* sA, scalar_t* sB,
                                 size_t yblock, size_t xblock, size_t k, size_t tid, size_t nthreads,
                                 uint32_t M, uint32_t N, uint32_t P) {
  for(int j = 0; j < MATMUL_L * MATMUL_S / nthreads; ++j) {
    size_t y = (j * nthreads + tid) / MATMUL_L;
    size_t x = (j * nthreads + tid) % MATMUL_L;
    
    size_t row_a = yblock * MATMUL_L + y;
    size_t col_a = k + x;
    sA[idx2d(y, x, MATMUL_L)] = (row_a < M && col_a < N) ? A[idx2d(row_a, col_a, N)] : 0;
    
    size_t row_b = k + y;
    size_t col_b = xblock * MATMUL_L + x;
    sB[idx2d(y, x, MATMUL_L)] = (row_b < N && col_b < P) ? B[idx2d(row_b, col_b, P)] : 0;
  }
}

__device__ void LoadRegistersFromShared(const scalar_t* sA, const scalar_t* sB, scalar_t* a, scalar_t* b,
                                        size_t ki) {
  for (size_t y = 0; y < MATMUL_V; ++y) {
    a[y] = sA[idx2d(threadIdx.y * MATMUL_V + y, ki, MATMUL_L)];
    b[y] = sB[idx2d(ki, threadIdx.x * MATMUL_V + y, MATMUL_L)];
  }
}

__device__ void ComputeOuterProduct(const scalar_t* a, const scalar_t* b, scalar_t* c) {
  for (size_t y = 0; y < MATMUL_V; ++y) {
    for (size_t x = 0; x < MATMUL_V; ++x) {
      c[idx2d(y, x, MATMUL_V)] += a[y] * b[x];
    }
  }
}

__device__ void WriteResultstoOut(scalar_t* out, const scalar_t* c, size_t yblock, size_t xblock,
                                  uint32_t M, uint32_t P) {
  size_t ybase = yblock * MATMUL_L + threadIdx.y * MATMUL_V;
  size_t xbase = xblock * MATMUL_L + threadIdx.x * MATMUL_V;
  
  for (size_t y = 0; y < MATMUL_V; ++y) {
    for (size_t x = 0; x < MATMUL_V; ++x) {
      size_t row = ybase + y;
      size_t col = xbase + x;
      if (row < M && col < P) {
        out[idx2d(row, col, P)] = c[idx2d(y, x, MATMUL_V)];
      }
    }
  }
}

__global__ void MatmulKernel(const scalar_t* A, const scalar_t* B, scalar_t* out, uint32_t M,
                             uint32_t N, uint32_t P) {
  __shared__ scalar_t sA[MATMUL_S * MATMUL_L], sB[MATMUL_S * MATMUL_L];
  scalar_t c[MATMUL_V * MATMUL_V] = {0};
  scalar_t a[MATMUL_V], b[MATMUL_V];
  
  size_t yblock = blockIdx.y;
  size_t xblock = blockIdx.x;
  size_t nthreads = blockDim.y * blockDim.x;
  size_t tid = threadIdx.y * blockDim.x + threadIdx.x;

  for (size_t k = 0; k < N; k += MATMUL_S) {
    __syncthreads();
    LoadTileToShared(A, B, sA, sB, yblock, xblock, k, tid, nthreads, M, N, P);
    __syncthreads();
    for (size_t ki = 0; ki < MATMUL_S; ++ki) {
      LoadRegistersFromShared(sA, sB, a, b, ki);
      ComputeOuterProduct(a, b, c);
    }
  }
  WriteResultstoOut(out, c, yblock, xblock, M, P);
}

void Matmul(const CudaArray& a, const CudaArray& b, CudaArray* out, uint32_t M, uint32_t N,
            uint32_t P) {
  dim3 blockDim(MATMUL_L / MATMUL_V, MATMUL_L / MATMUL_V);
  dim3 gridDim((P + MATMUL_L - 1) / MATMUL_L, (M + MATMUL_L - 1) / MATMUL_L);
  MatmulKernel<<<gridDim, blockDim>>>(a.as<scalar_t>(), b.as<scalar_t>(), out->as<scalar_t>(), M, N, P);
}

__global__ void ReduceMaxKernel(const scalar_t* a, scalar_t* out, size_t size, size_t reduce_size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    out[gid] = a[gid * reduce_size];
    for (size_t i = 1; i < reduce_size; i++) {
     out[gid] = max(out[gid], a[gid * reduce_size + i]);
    }
  }
}

void ReduceMax(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  CudaDims dim = CudaOneDim(out->size);
  ReduceMaxKernel<<<dim.grid, dim.block>>>(a.as<scalar_t>(), out->as<scalar_t>(), out->size, reduce_size);
}

__global__ void ReduceSumKernel(const scalar_t* a, scalar_t* out, size_t size, size_t reduce_size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    out[gid] = 0;
    for (size_t i = 0; i < reduce_size; i++) {
     out[gid] += a[gid * reduce_size + i];
    }
  }
}

void ReduceSum(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  CudaDims dim = CudaOneDim(out->size);
  ReduceSumKernel<<<dim.grid, dim.block>>>(a.as<scalar_t>(), out->as<scalar_t>(), out->size, reduce_size);
}

__global__ void QuantizeInt8Kernel(const scalar_t* src, int8_t* dst, size_t size, float scale, int8_t zero_point) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    float val = round(src[gid] / scale + zero_point);
    val = max(-128.0f, min(127.0f, val));
    dst[gid] = (int8_t)val;
  }
}

void QuantizeInt8(const CudaArray& src, CudaArray* dst, float scale, int8_t zero_point) {
  CudaDims dim = CudaOneDim(src.size);
  QuantizeInt8Kernel<<<dim.grid, dim.block>>>(src.as<scalar_t>(), dst->as<int8_t>(), src.size, scale, zero_point);
}

__global__ void DequantizeInt8Kernel(const int8_t* src, scalar_t* dst, size_t size, float scale, int8_t zero_point) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    dst[gid] = (scalar_t)(src[gid] - zero_point) * scale;
  }
}

void DequantizeInt8(const CudaArray& src, CudaArray* dst, float scale, int8_t zero_point) {
  CudaDims dim = CudaOneDim(src.size);
  DequantizeInt8Kernel<<<dim.grid, dim.block>>>(src.as<int8_t>(), dst->as<scalar_t>(), src.size, scale, zero_point);
}

__global__ void MatmulInt8Kernel(const int8_t* A, const int8_t* B, scalar_t* out, uint32_t M, uint32_t N, uint32_t P,
                                 float sa, int8_t za, float sb, int8_t zb) {
  size_t i = blockIdx.y * blockDim.y + threadIdx.y;
  size_t j = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (i < M && j < P) {
    float sum = 0;
    for (size_t k = 0; k < N; k++) {
      float val_a = (float)(A[i * N + k] - za);
      float val_b = (float)(B[k * P + j] - zb);
      sum += val_a * val_b;
    }
    out[i * P + j] = sum * sa * sb;
  }
}

void MatmulInt8(const CudaArray& a, const CudaArray& b, CudaArray* out, 
                uint32_t m, uint32_t n, uint32_t p,
                float sa, int8_t za, float sb, int8_t zb) {
  dim3 blockDim(16, 16);
  dim3 gridDim((p + 15) / 16, (m + 15) / 16);
  MatmulInt8Kernel<<<gridDim, blockDim>>>(a.as<int8_t>(), b.as<int8_t>(), out->as<scalar_t>(), m, n, p, sa, za, sb, zb);
}

}  // namespace cuda
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cuda, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cuda;

  m.attr("__device_name__") = "cuda";
  m.attr("__tile_size__") = TILE;

  py::class_<CudaArray>(m, "Array")
      .def(py::init<size_t, int8_t>(), py::arg("size"), py::arg("dtype") = 0, py::return_value_policy::take_ownership)
      .def_readonly("size", &CudaArray::size)
      .def("ptr", &CudaArray::ptr_as_int);

  m.def("to_numpy", [](const CudaArray& a, std::vector<size_t> shape, std::vector<size_t> strides,
                       size_t offset) -> py::object {
    std::vector<size_t> numpy_strides = strides;
    size_t elem_size = (a.dtype == 0) ? ELEM_SIZE : 1;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [&](size_t& c) { return c * elem_size; });

    if (a.dtype == 0) {
        scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * ELEM_SIZE);
        if (host_ptr == 0) throw std::bad_alloc();
        cudaError_t err = cudaMemcpy(host_ptr, a.ptr, a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
        py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
        return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset, deallocate_buffer);
    } else {
        int8_t* host_ptr = (int8_t*)std::malloc(a.size * 1);
        if (host_ptr == 0) throw std::bad_alloc();
        cudaError_t err = cudaMemcpy(host_ptr, a.ptr, a.size * 1, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
        py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
        return py::array_t<int8_t>(shape, numpy_strides, host_ptr + offset, deallocate_buffer);
    }
  });

  m.def("from_numpy", [](py::array a, CudaArray* out) {
    size_t elem_size = (out->dtype == 0) ? ELEM_SIZE : 1;
    cudaError_t err =
        cudaMemcpy(out->ptr, a.request().ptr, out->size * elem_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
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

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
  
  m.def("quantize_int8", QuantizeInt8);
  m.def("dequantize_int8", DequantizeInt8);
  m.def("matmul_int8", MatmulInt8);
}
