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

// CE AllReduce local-reduction kernel — vectorized 16B load/store edition.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <stdint.h>
#include <algorithm>

#include "nccl.h"

// *********************************************************************************
// VecTrait<T>
//   LoadT     : 16-byte HIP vector used for a single memory transaction
//   W         : number of T elements packed into one 16-byte load/store
//   GpuUnroll : compile-time unroll factor (U) to balance ILP vs VGPRs
// *********************************************************************************
template<typename T> struct VecTrait {
  using LoadT = uint4;
  static constexpr int W = (int)(16 / sizeof(T));
  static constexpr int GpuUnroll = 4;
};

template<> struct VecTrait<float>    { using LoadT = float4;     static constexpr int W = 4;  static constexpr int GpuUnroll = 4; };
template<> struct VecTrait<int32_t>  { using LoadT = int4;       static constexpr int W = 4;  static constexpr int GpuUnroll = 4; };
template<> struct VecTrait<uint32_t> { using LoadT = uint4;      static constexpr int W = 4;  static constexpr int GpuUnroll = 4; };

template<> struct VecTrait<double>   { using LoadT = double2;    static constexpr int W = 2;  static constexpr int GpuUnroll = 2; };
template<> struct VecTrait<int64_t>  { using LoadT = longlong2;  static constexpr int W = 2;  static constexpr int GpuUnroll = 2; };
template<> struct VecTrait<uint64_t> { using LoadT = ulonglong2; static constexpr int W = 2;  static constexpr int GpuUnroll = 2; };

template<> struct VecTrait<__half>       { using LoadT = uint4; static constexpr int W = 8;  static constexpr int GpuUnroll = 4; };
template<> struct VecTrait<hip_bfloat16> { using LoadT = uint4; static constexpr int W = 8;  static constexpr int GpuUnroll = 4; };

template<> struct VecTrait<int8_t>  { using LoadT = uint4; static constexpr int W = 16; static constexpr int GpuUnroll = 4; };
template<> struct VecTrait<uint8_t> { using LoadT = uint4; static constexpr int W = 16; static constexpr int GpuUnroll = 4; };

// *********************************************************************************
// ReduceOp<T, RedOp> — compile-time reduction functor
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
// ncclCeLocalReduceKernelVec<T, RedOp, U>
//
// scatterBuf layout after CE scatter phase:
//   in[k * chunkElems + i]  =  element i contributed by rank k
// *********************************************************************************
template<typename T, int RedOp, int U>
__global__ __launch_bounds__(256)
void ncclCeLocalReduceKernelVec(const T* __restrict__ in, T* __restrict__ out,
                                int nRanks, size_t chunkElems) {

  using LT = typename VecTrait<T>::LoadT;
  constexpr int W = VecTrait<T>::W;

  union alignas(16) Pack { LT vec; T v[W]; };

  const size_t nVec   = chunkElems / W;
  const size_t tid    = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = (size_t)blockDim.x * gridDim.x;

  size_t vi = tid;

  // Strided unroll: thread tid handles vector indices
  //   vi, vi + stride, ..., vi + (U-1)*stride per outer step.
  for (; vi + (U-1) * stride < nVec; vi += stride * U) {
    Pack acc[U];
    #pragma unroll
    for (int i = 0; i < U; i++) {
      size_t elementOffset = (vi + (size_t)i * stride) * W;
      acc[i].vec = *reinterpret_cast<const LT*>(in + elementOffset);
    }

    for (int r = 1; r < nRanks; r++) {
      Pack tmp[U];
      #pragma unroll
      for (int i = 0; i < U; i++) {
        size_t rankOffset = (size_t)r * chunkElems
                          + (vi + (size_t)i * stride) * W;
        tmp[i].vec = *reinterpret_cast<const LT*>(in + rankOffset);
      }
      #pragma unroll
      for (int i = 0; i < U; i++) {
        #pragma unroll
        for (int k = 0; k < W; k++) {
          acc[i].v[k] = ReduceOp<T, RedOp>::apply(acc[i].v[k], tmp[i].v[k]);
        }
      }
    }
    #pragma unroll
    for (int i = 0; i < U; i++) {
      size_t outOffset = (vi + (size_t)i * stride) * W;
      *reinterpret_cast<LT*>(out + outOffset) = acc[i].vec;
    }
  }

  // Drain remaining vectorized elements (one vector per thread per step).
  for (; vi < nVec; vi += stride) {
    Pack acc;
    size_t elementOffset = vi * W;
    acc.vec = *reinterpret_cast<const LT*>(in + elementOffset);

    for (int r = 1; r < nRanks; r++) {
      Pack tmp;
      size_t rankOffset = (size_t)r * chunkElems + vi * W;
      tmp.vec = *reinterpret_cast<const LT*>(in + rankOffset);

      #pragma unroll
      for (int k = 0; k < W; k++) {
        acc.v[k] = ReduceOp<T, RedOp>::apply(acc.v[k], tmp.v[k]);
      }
    }
    *reinterpret_cast<LT*>(out + elementOffset) = acc.vec;
  }

  // Scalar tail (chunkElems % W != 0).
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

  int redOp;
  switch (op) {
    case ncclSum:  redOp = 0; break;
    case ncclProd: redOp = 1; break;
    case ncclMin:  redOp = 2; break;
    case ncclMax:  redOp = 3; break;
    default: return ncclInvalidArgument;
  }

  const int threads = 256;

  // Each thread covers GpuUnroll vectors per outer step; cap blocks for CE DMA headroom.
#define NCCL_CE_VEC_BLOCKS(T)                                                \
  (int)std::min((size_t)46,                                                  \
    (chunkElems / VecTrait<T>::W                                              \
     + (size_t)threads * VecTrait<T>::GpuUnroll - 1)                          \
     / ((size_t)threads * VecTrait<T>::GpuUnroll))

#define NCCL_CE_LAUNCH_VEC(T)                                                \
  do {                                                                       \
    constexpr int UnrollFactor = VecTrait<T>::GpuUnroll;                     \
    const int _blocks = NCCL_CE_VEC_BLOCKS(T);                               \
    (void)hipGetLastError();                                                 \
    switch (redOp) {                                                         \
      case 0: hipLaunchKernelGGL(                                            \
          (ncclCeLocalReduceKernelVec<T, 0, UnrollFactor>),                  \
          dim3(_blocks), dim3(threads), 0, stream,                           \
          (const T*)tmpBuf, (T*)output, nRanks, chunkElems); break;          \
      case 1: hipLaunchKernelGGL(                                            \
          (ncclCeLocalReduceKernelVec<T, 1, UnrollFactor>),                  \
          dim3(_blocks), dim3(threads), 0, stream,                           \
          (const T*)tmpBuf, (T*)output, nRanks, chunkElems); break;          \
      case 2: hipLaunchKernelGGL(                                            \
          (ncclCeLocalReduceKernelVec<T, 2, UnrollFactor>),                  \
          dim3(_blocks), dim3(threads), 0, stream,                           \
          (const T*)tmpBuf, (T*)output, nRanks, chunkElems); break;          \
      case 3: hipLaunchKernelGGL(                                            \
          (ncclCeLocalReduceKernelVec<T, 3, UnrollFactor>),                  \
          dim3(_blocks), dim3(threads), 0, stream,                           \
          (const T*)tmpBuf, (T*)output, nRanks, chunkElems); break;          \
    }                                                                        \
    hipError_t _e = hipGetLastError();                                       \
    if (_e != hipSuccess) {                                                  \
      printf("[CE reduce] launch failed: %d (%s)\n",                         \
             (int)_e, hipGetErrorString(_e));                                \
      return ncclUnhandledCudaError;                                         \
    }                                                                        \
  } while (0)

  switch (datatype) {
    case ncclFloat32:  NCCL_CE_LAUNCH_VEC(float);        break;
    case ncclFloat64:  NCCL_CE_LAUNCH_VEC(double);       break;
    case ncclFloat16:  NCCL_CE_LAUNCH_VEC(__half);       break;
    case ncclBfloat16: NCCL_CE_LAUNCH_VEC(hip_bfloat16); break;
    case ncclInt32:    NCCL_CE_LAUNCH_VEC(int32_t);      break;
    case ncclUint32:   NCCL_CE_LAUNCH_VEC(uint32_t);     break;
    case ncclInt64:    NCCL_CE_LAUNCH_VEC(int64_t);      break;
    case ncclUint64:   NCCL_CE_LAUNCH_VEC(uint64_t);     break;
    case ncclInt8:     NCCL_CE_LAUNCH_VEC(int8_t);       break;
    case ncclUint8:    NCCL_CE_LAUNCH_VEC(uint8_t);     break;
    default: return ncclInvalidArgument;
  }

#undef NCCL_CE_LAUNCH_VEC
#undef NCCL_CE_VEC_BLOCKS

  return ncclSuccess;
}
