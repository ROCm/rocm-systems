/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

// CE AllReduce local-reduction kernel — vectorised 16B load/store edition.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <stdint.h>
#include <algorithm>

#include "nccl.h"

// *********************************************************************************
// VecTrait<T>
//   LoadT : 16-byte HIP vector used for a single memory transaction
//   W     : number of T elements packed into one 16-byte load/store
// *********************************************************************************
template<typename T> struct VecTrait {
  using LoadT = uint4;
  static constexpr int W = (int)(16 / sizeof(T));
};
// Native vector types allow the compiler to emit packed arithmetic as well.
template<> struct VecTrait<float>    { using LoadT = float4;     static constexpr int W = 4; };
template<> struct VecTrait<double>   { using LoadT = double2;    static constexpr int W = 2; };
template<> struct VecTrait<int32_t>  { using LoadT = int4;       static constexpr int W = 4; };
template<> struct VecTrait<uint32_t> { using LoadT = uint4;      static constexpr int W = 4; };
template<> struct VecTrait<int64_t>  { using LoadT = longlong2;  static constexpr int W = 2; };
template<> struct VecTrait<uint64_t> { using LoadT = ulonglong2; static constexpr int W = 2; };
// 16-bit types: 8 elements per 16B load (use uint4 carrier).
template<> struct VecTrait<__half>       { using LoadT = uint4; static constexpr int W = 8;  };
template<> struct VecTrait<hip_bfloat16> { using LoadT = uint4; static constexpr int W = 8;  };
// 8-bit types: 16 elements per 16B load (use uint4 carrier).
template<> struct VecTrait<int8_t>  { using LoadT = uint4; static constexpr int W = 16; };
template<> struct VecTrait<uint8_t> { using LoadT = uint4; static constexpr int W = 16; };

// *********************************************************************************
// ReduceOp<T, RedOp> — compile-time reduction functor, eliminates the runtime
// switch from every inner-loop iteration.
// 0=Sum  1=Prod  2=Min  3=Max
// *********************************************************************************
template<typename T, int RedOp> struct ReduceOp;
template<typename T> struct ReduceOp<T, 0> {
  __device__ __forceinline__ static T apply(T a, T b) { return a + b; }
};
template<typename T> struct ReduceOp<T, 1> {
  __device__ __forceinline__ static T apply(T a, T b) { return a * b; }
};
template<typename T> struct ReduceOp<T, 2> {
  __device__ __forceinline__ static T apply(T a, T b) { return a < b ? a : b; }
};
template<typename T> struct ReduceOp<T, 3> {
  __device__ __forceinline__ static T apply(T a, T b) { return a > b ? a : b; }
};

// *********************************************************************************
// ncclCeLocalReduceKernelVec<T, RedOp>
//
// scatterBuf layout after CE scatter phase:
//   in[k * chunkElems + i]  =  element i contributed by rank k
//
// Each thread processes W = 16/sizeof(T) elements per pass using a single
// 16-byte global load/store (global_load_dwordx4 on CDNA).
// RedOp is a compile-time constant so no branch exists in the inner loop.
// *********************************************************************************
template<typename T, int RedOp>
__global__ __launch_bounds__(256)
void ncclCeLocalReduceKernelVec(const T* __restrict__ in, T* __restrict__ out,
                                 int nRanks, size_t chunkElems) {

  using LT = typename VecTrait<T>::LoadT;
  constexpr int W = VecTrait<T>::W;

  // Union allows a single 16B load into the VecTrait::LoadT hardware register
  // followed by element-wise scalar arithmetic on T v[W].
  union alignas(16) Pack { LT vec; T v[W]; };

  const size_t nVec   = chunkElems / W;
  const size_t tid    = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = (size_t)blockDim.x * gridDim.x;

  //vectorised main loop (16 bytes / load) 
  for (size_t vi = tid; vi < nVec; vi += stride) {
    Pack acc;
    // 16B load: rank 0's W elements
    acc.vec = *reinterpret_cast<const LT*>(in + vi * W);

    for (int r = 1; r < nRanks; r++) {
      Pack tmp;
      // 16B load: rank r's W elements for the same output position
      tmp.vec = *reinterpret_cast<const LT*>(
                    in + (size_t)r * chunkElems + vi * W);
      #pragma unroll
      for (int k = 0; k < W; k++)
        acc.v[k] = ReduceOp<T, RedOp>::apply(acc.v[k], tmp.v[k]);
    }
    // 16B store
    *reinterpret_cast<LT*>(out + vi * W) = acc.vec;
  }

  // scalar tail (< W elements when chunkElems % W != 0) 
  const size_t tailBase = nVec * W;
  const size_t tailLen  = chunkElems - tailBase;
  for (size_t i = tid; i < tailLen; i += stride) {
    T a = in[tailBase + i];
    for (int r = 1; r < nRanks; r++) {
      T v = in[(size_t)r * chunkElems + tailBase + i];
      a = ReduceOp<T, RedOp>::apply(a, v);
    }
    out[tailBase + i] = a;
  }
}

// *********************************************************************************
// ncclCeLaunchLocalReduce — host launcher
// *********************************************************************************
ncclResult_t ncclCeLaunchLocalReduce(
    const void* tmpBuf, void* output,
    int nRanks, size_t chunkElems,
    ncclDataType_t datatype, ncclRedOp_t op,
    hipStream_t stream) {

  if (chunkElems == 0) return ncclSuccess;

  // Map ncclRedOp_t → compile-time index used by ReduceOp<>.
  int redOp;
  switch (op) {
    case ncclSum:  redOp = 0; break;
    case ncclProd: redOp = 1; break;
    case ncclMin:  redOp = 2; break;
    case ncclMax:  redOp = 3; break;
    default: return ncclInvalidArgument;
  }

  const int threads = 256;
  // Block count: cover nVec work items (each thread handles W elements).
  // Cap at 112 CUs so the reduce kernel leaves headroom for CE DMA engines.
#define NCCL_CE_VEC_BLOCKS(T)                                               \
  (int)std::min((size_t)112,                                                \
    (chunkElems / VecTrait<T>::W + threads - 1) / threads)

  // Launch the vectorised kernel with a compile-time RedOp to eliminate the
  // branch from every inner-loop iteration.
#define NCCL_CE_LAUNCH_VEC(T)                                               \
  do {                                                                       \
    const int _blocks = NCCL_CE_VEC_BLOCKS(T);                              \
    (void)hipGetLastError();                                                 \
    switch (redOp) {                                                         \
      case 0: hipLaunchKernelGGL(                                            \
          (ncclCeLocalReduceKernelVec<T, 0>),                               \
          dim3(_blocks), dim3(threads), 0, stream,                           \
          (const T*)tmpBuf, (T*)output, nRanks, chunkElems); break;         \
      case 1: hipLaunchKernelGGL(                                            \
          (ncclCeLocalReduceKernelVec<T, 1>),                               \
          dim3(_blocks), dim3(threads), 0, stream,                           \
          (const T*)tmpBuf, (T*)output, nRanks, chunkElems); break;         \
      case 2: hipLaunchKernelGGL(                                            \
          (ncclCeLocalReduceKernelVec<T, 2>),                               \
          dim3(_blocks), dim3(threads), 0, stream,                           \
          (const T*)tmpBuf, (T*)output, nRanks, chunkElems); break;         \
      case 3: hipLaunchKernelGGL(                                            \
          (ncclCeLocalReduceKernelVec<T, 3>),                               \
          dim3(_blocks), dim3(threads), 0, stream,                           \
          (const T*)tmpBuf, (T*)output, nRanks, chunkElems); break;         \
    }                                                                        \
    hipError_t _e = hipGetLastError();                                       \
    if (_e != hipSuccess) {                                                  \
      printf("[CE reduce] launch failed: %d (%s)\n",                        \
             (int)_e, hipGetErrorString(_e));                               \
      return ncclUnhandledCudaError;                                         \
    }                                                                        \
  } while (0)

  switch (datatype) {
    case ncclFloat32:  NCCL_CE_LAUNCH_VEC(float);       break;
    case ncclFloat64:  NCCL_CE_LAUNCH_VEC(double);      break;
    case ncclFloat16:  NCCL_CE_LAUNCH_VEC(__half);      break;
    case ncclBfloat16: NCCL_CE_LAUNCH_VEC(hip_bfloat16); break;
    case ncclInt32:    NCCL_CE_LAUNCH_VEC(int32_t);     break;
    case ncclUint32:   NCCL_CE_LAUNCH_VEC(uint32_t);    break;
    case ncclInt64:    NCCL_CE_LAUNCH_VEC(int64_t);     break;
    case ncclUint64:   NCCL_CE_LAUNCH_VEC(uint64_t);    break;
    case ncclInt8:     NCCL_CE_LAUNCH_VEC(int8_t);      break;
    case ncclUint8:    NCCL_CE_LAUNCH_VEC(uint8_t);     break;
    default: return ncclInvalidArgument;
  }

#undef NCCL_CE_LAUNCH_VEC
#undef NCCL_CE_VEC_BLOCKS

  return ncclSuccess;
}
