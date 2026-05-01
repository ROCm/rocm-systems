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

// Minimal device-only file for the CE AllReduce local-reduction kernel.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <stdint.h>
#include <algorithm>

// Pull in NCCL public types (ncclResult_t, ncclDataType_t, ncclRedOp_t, etc.)
#include "nccl.h"

// ---------------------------------------------------------------------------
// CE AllReduce local-reduction kernel
//
// After the CE scatter phase, rank r's ceARTmpBuf holds:
//   slot [k * chunkElems .. (k+1)*chunkElems)  =  rank k's contribution to shard r
// for k = 0 .. nRanks-1.
//
// This kernel reduces all nRanks slices and writes the single fully-reduced
// shard into 'out' (the reduceScratch region at tmpBuf + nRanks*chunkBytes).
//
// redOp encoding: 0=Sum  1=Prod  2=Min  3=Max
// ---------------------------------------------------------------------------
template<typename T>
__global__ __launch_bounds__(256)
void ncclCeLocalReduceKernel(const T* __restrict__ in, T* __restrict__ out,
                              int nRanks, size_t chunkElems, int redOp) {
  const size_t idx    = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = (size_t)blockDim.x * gridDim.x;
  for (size_t i = idx; i < chunkElems; i += stride) {
    T acc = in[i];
    for (int r = 1; r < nRanks; r++) {
      T v = in[(size_t)r * chunkElems + i];
      switch (redOp) {
        case 0: acc = acc + v;         break;
        case 1: acc = acc * v;         break;
        case 2: if (v < acc) acc = v;  break;
        case 3: if (v > acc) acc = v;  break;
      }
    }
    out[i] = acc;
  }
}

// ---------------------------------------------------------------------------
// Host-callable launcher (non-static so ce_coll.cc can call it after
// forward-declaring it; the symbol is exported by the device-linker fat obj).
// ---------------------------------------------------------------------------
ncclResult_t ncclCeLaunchLocalReduce(
    const void* tmpBuf, void* output,
    int nRanks, size_t chunkElems,
    ncclDataType_t datatype, ncclRedOp_t op,
    hipStream_t stream) {

  int redOp;
  switch (op) {
    case ncclSum:  redOp = 0; break;
    case ncclProd: redOp = 1; break;
    case ncclMin:  redOp = 2; break;
    case ncclMax:  redOp = 3; break;
    default: return ncclInvalidArgument;
  }

  if (chunkElems == 0) return ncclSuccess;

  const int threads = 256;
  const int blocks  = (int)std::min((size_t)1024,
                                    (chunkElems + threads - 1) / threads);

#define NCCL_CE_LAUNCH_REDUCE(T)                                          \
  do {                                                                    \
    (void)hipGetLastError();                                              \
    hipLaunchKernelGGL((ncclCeLocalReduceKernel<T>),                      \
        dim3(blocks), dim3(threads), 0, stream,                           \
        (const T*)tmpBuf, (T*)output, nRanks, chunkElems, redOp);         \
    hipError_t _e = hipGetLastError();                                    \
    if (_e != hipSuccess) {                                               \
      printf("[CE reduce] launch failed: %d (%s)\n",                      \
             (int)_e, hipGetErrorString(_e));                             \
      return ncclUnhandledCudaError;                                      \
    }                                                                     \
  } while (0);                                                            \
  break

  switch (datatype) {
    case ncclFloat32:  NCCL_CE_LAUNCH_REDUCE(float);
    case ncclFloat64:  NCCL_CE_LAUNCH_REDUCE(double);
    case ncclFloat16:  NCCL_CE_LAUNCH_REDUCE(__half);
    case ncclBfloat16: NCCL_CE_LAUNCH_REDUCE(hip_bfloat16);
    case ncclInt32:    NCCL_CE_LAUNCH_REDUCE(int32_t);
    case ncclUint32:   NCCL_CE_LAUNCH_REDUCE(uint32_t);
    case ncclInt64:    NCCL_CE_LAUNCH_REDUCE(int64_t);
    case ncclUint64:   NCCL_CE_LAUNCH_REDUCE(uint64_t);
    case ncclInt8:     NCCL_CE_LAUNCH_REDUCE(int8_t);
    case ncclUint8:    NCCL_CE_LAUNCH_REDUCE(uint8_t);
    default: return ncclInvalidArgument;
  }
#undef NCCL_CE_LAUNCH_REDUCE

  return ncclSuccess;
}
