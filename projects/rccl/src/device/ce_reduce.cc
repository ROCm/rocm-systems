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
#include <hip/hip_cooperative_groups.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <stdint.h>
#include <algorithm>
#include "nccl.h"

#ifndef NCCL_CE_REDUCE_MAX_BLOCKS
#define NCCL_CE_REDUCE_MAX_BLOCKS 46
#endif

#ifndef NCCL_CE_NUM_SLOTS
#define NCCL_CE_NUM_SLOTS 2
#endif

// Global atomic-counter barrier for regular launch (avoids CTA scheduling deadlock).
__device__ __forceinline__ void ncclCeGlobalBlockBarrier(
    uint32_t* d_barrierSync, int numBlocks, uint32_t gen) {
  if (numBlocks <= 1) {
    __syncthreads();
    return;
  }
  uint32_t* arrival = d_barrierSync + 0;
  uint32_t* doneGen = d_barrierSync + 1;
  __syncthreads();
  if (threadIdx.x == 0) {
    uint32_t prev = __hip_atomic_fetch_add(arrival, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    if (prev + 1 == (uint32_t)numBlocks) {
      __hip_atomic_store(arrival, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      __hip_atomic_store(doneGen, gen, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    } else {
      while (__hip_atomic_load(doneGen, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) != gen) {
        __builtin_amdgcn_s_sleep(1);
      }
    }
  }
  __syncthreads();
}

// One barrier generation: thread 0 bumps s_gen, all threads enter the barrier with s_val.
__device__ __forceinline__ void ncclCeBlockBarrier(
    int blockId, uint64_t* s_gen, uint32_t* d_barrierSync) {
  __shared__ uint64_t s_val;
  __syncthreads();
  if (threadIdx.x == 0) s_val = (*s_gen)++;
  __syncthreads();
  ncclCeGlobalBlockBarrier(d_barrierSync, gridDim.x, (uint32_t)s_val);
}

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
//   in[slot * slotChunkElems * nRanks + r * slotChunkElems + i] = element i from rank r
// *********************************************************************************
template<typename T, int RedOp, int U>
__global__ __launch_bounds__(256)
void ncclCeLocalReduceKernelVec(
   const T* __restrict__ in,
   T* __restrict__ out, // recvbuff + rank*shardBytes (final AllReduce shard region)
   int nRanks,
   size_t baseChunkElems,
   size_t tailChunkElems,
   size_t chunksPerShard,
   size_t slotChunkElems,
   volatile uint32_t* signalBuffer,
   volatile uint32_t* d_kernelReady,
   uint32_t readyValue,  
   size_t totalSteps,
   uint64_t* d_barrierGenBase,
   uint32_t* d_barrierSync)
{

   if (blockIdx.x >= NCCL_CE_REDUCE_MAX_BLOCKS) return;

   __shared__ uint64_t s_gen;
   if (threadIdx.x == 0) s_gen = *d_barrierGenBase;
   __syncthreads();

   using LT = typename VecTrait<T>::LoadT;
   constexpr int W = VecTrait<T>::W;
   union alignas(16) Pack { LT vec; T v[W]; };

   const size_t tid    = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
   const size_t stride = (size_t)blockDim.x * gridDim.x;

   uint32_t step = 0;
   if (blockIdx.x == 0 && threadIdx.x == 0) {
    __hip_atomic_store((uint32_t*)d_kernelReady, readyValue,
                       __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
  }

   while (step < totalSteps) {
       const int    slot         = (int)(step % NCCL_CE_NUM_SLOTS);
       const size_t chunkInShard = step % chunksPerShard;

       const bool isTailChunk = (chunkInShard == chunksPerShard - 1) && (tailChunkElems > 0);
       const size_t currentChunkElems = isTailChunk ? tailChunkElems : baseChunkElems;
       const size_t nVec = currentChunkElems / W;

       if (blockIdx.x == 0 && threadIdx.x == 0) {
           while (true) {
               int ready = 0;
               for (int r = 0; r < nRanks; r++) {
                   uint32_t sv = __hip_atomic_load(
                       (uint32_t*)&signalBuffer[(size_t)slot * nRanks + r],
                       __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
                   if (sv == 1) ready++;
               }
               if (ready == nRanks) break;
               __builtin_amdgcn_s_sleep(1);
           }
       }
       ncclCeBlockBarrier(blockIdx.x, &s_gen, d_barrierSync);

       const size_t slotStrideElems = slotChunkElems * (size_t)nRanks;
       const T* __restrict__ slotIn = in + ((size_t)slot * slotStrideElems);
       T* __restrict__ chunkOut = out + ((size_t)chunkInShard * baseChunkElems);

       size_t vi = tid;
       for (; vi + (U-1) * stride < nVec; vi += stride * U) {
           Pack acc[U];
           #pragma unroll
           for (int i = 0; i < U; i++) {
               acc[i].vec = *reinterpret_cast<const LT*>(slotIn + (vi + (size_t)i*stride)*W);
           }
           for (int r = 1; r < nRanks; r++) {
               Pack tmp[U];
               #pragma unroll
               for (int i = 0; i < U; i++) {
                   size_t rankOff = (size_t)r*slotChunkElems + (vi + (size_t)i*stride)*W;
                   tmp[i].vec = *reinterpret_cast<const LT*>(slotIn + rankOff);
               }
               #pragma unroll
               for (int i = 0; i < U; i++) {
                   #pragma unroll
                   for (int k = 0; k < W; k++)
                       acc[i].v[k] = ReduceOp<T, RedOp>::apply(acc[i].v[k], tmp[i].v[k]);
               }
           }
           #pragma unroll
           for (int i = 0; i < U; i++) {
               *reinterpret_cast<LT*>(chunkOut + (vi + (size_t)i*stride)*W) = acc[i].vec;
           }
       }

       size_t vectorTailStart = nVec - (nVec % (stride * U));
       if (nVec > 0 && nVec % (stride * U) != 0) {
           for (size_t vIdx = vectorTailStart + tid; vIdx < nVec; vIdx += stride) {
               Pack acc;
               acc.vec = *reinterpret_cast<const LT*>(slotIn + vIdx*W);
               for (int r = 1; r < nRanks; r++) {
                   Pack tmp;
                   tmp.vec = *reinterpret_cast<const LT*>(slotIn + (size_t)r*slotChunkElems + vIdx*W);
                   #pragma unroll
                   for (int k = 0; k < W; k++)
                       acc.v[k] = ReduceOp<T, RedOp>::apply(acc.v[k], tmp.v[k]);
               }
               *reinterpret_cast<LT*>(chunkOut + vIdx*W) = acc.vec;
           }
       }

       const size_t tailBase = nVec * W;
       if (tailBase < currentChunkElems) {
           for (size_t i = tid; tailBase + i < currentChunkElems; i += stride) {
               size_t globalIdx = tailBase + i;
               T a = slotIn[globalIdx];
               for (int r = 1; r < nRanks; r++)
                   a = ReduceOp<T, RedOp>::apply(a, slotIn[(size_t)r*slotChunkElems + globalIdx]);
               chunkOut[globalIdx] = a;
           }
       }

       __threadfence_system();
       ncclCeBlockBarrier(blockIdx.x, &s_gen, d_barrierSync);

       if (blockIdx.x == 0) {
           for (int r = (int)threadIdx.x; r < nRanks; r += (int)blockDim.x) {
               __hip_atomic_store((uint32_t*)&signalBuffer[(size_t)slot * nRanks + r],
                                  0, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
           }
       }
       step++;
  }
}
// *********************************************************************************
// ncclCeLaunchReduceTyped — per-type launch dispatch (coop or GGL).
// *********************************************************************************
template<typename T>
static ncclResult_t ncclCeLaunchReduceTyped(
  const void* in, void* out, int nRanks,
  size_t baseChunkElems, size_t tailChunkElems, size_t chunksPerShard, size_t slotChunkElems,
  uint32_t* signalBuffer, uint32_t* d_kernelReady, uint32_t readyValue,
  size_t totalSteps,
  uint64_t* d_barrierGenBase, uint32_t* d_barrierSync,
  int redOp, int threads, void** kernelArgs, hipStream_t stream) {

  constexpr int U = VecTrait<T>::GpuUnroll;
  const int blocks = (int)std::clamp<size_t>(
    (baseChunkElems / VecTrait<T>::W + (size_t)threads * VecTrait<T>::GpuUnroll - 1)
      / ((size_t)threads * VecTrait<T>::GpuUnroll), (size_t)1, (size_t)46);

  (void)hipGetLastError();
    switch (redOp) {
      case 0: {
        auto kernelFn = ncclCeLocalReduceKernelVec<T, 0, U>;
        hipLaunchKernelGGL(
          kernelFn,
          dim3(blocks), dim3(threads), 0, stream,
          static_cast<const T*>(in), static_cast<T*>(out), nRanks,
          baseChunkElems, tailChunkElems, chunksPerShard, slotChunkElems,
          signalBuffer, d_kernelReady,
          readyValue, totalSteps, d_barrierGenBase, d_barrierSync);
        break;
        } case 1: {
        auto kernelFn = ncclCeLocalReduceKernelVec<T, 1, U>;
        hipLaunchKernelGGL(
          kernelFn,
          dim3(blocks), dim3(threads), 0, stream,
          static_cast<const T*>(in), static_cast<T*>(out), nRanks,
          baseChunkElems, tailChunkElems, chunksPerShard, slotChunkElems,
          signalBuffer, d_kernelReady,
          readyValue, totalSteps, d_barrierGenBase, d_barrierSync);
        break;
      } case 2: {
        auto kernelFn = ncclCeLocalReduceKernelVec<T, 2, U>;
        hipLaunchKernelGGL(
          kernelFn,
          dim3(blocks), dim3(threads), 0, stream,
          static_cast<const T*>(in), static_cast<T*>(out), nRanks,
          baseChunkElems, tailChunkElems, chunksPerShard, slotChunkElems,
          signalBuffer, d_kernelReady,
          readyValue, totalSteps, d_barrierGenBase, d_barrierSync);
        break;
      } case 3: {
        auto kernelFn = ncclCeLocalReduceKernelVec<T, 3, U>;
        hipLaunchKernelGGL(
          kernelFn,
          dim3(blocks), dim3(threads), 0, stream,
          static_cast<const T*>(in), static_cast<T*>(out), nRanks,
          baseChunkElems, tailChunkElems, chunksPerShard, slotChunkElems,
          signalBuffer, d_kernelReady,
          readyValue, totalSteps, d_barrierGenBase, d_barrierSync);
        break;
      }
    }
    hipError_t e = hipGetLastError();
    if (e != hipSuccess) return ncclUnhandledCudaError;
  return ncclSuccess;
}

// *********************************************************************************
// ncclCeLaunchPersistentReduce — cooperative-launch dispatch over datatype & op.
// Same kernelArgs work for any T because all args are raw pointers / scalars;
// only the instantiated function pointer and the block count depend on T.
// *********************************************************************************
ncclResult_t ncclCeLaunchPersistentReduce(
  const void* in, void* out, int nRanks,
  size_t baseChunkElems, size_t tailChunkElems, size_t chunksPerShard,
   size_t slotChunkElems, uint32_t* signalBuffer, uint32_t* d_kernelReady,
   uint32_t readyValue, size_t totalSteps,
  uint64_t* d_barrierGenBase, uint32_t* d_barrierSync,
  ncclDataType_t datatype, ncclRedOp_t op, hipStream_t stream, int coopLaunch) {

int redOp;
switch (op) {
  case ncclSum:  redOp = 0; break;
  case ncclProd: redOp = 1; break;
  case ncclMin:  redOp = 2; break;
  case ncclMax:  redOp = 3; break;
  default: return ncclInvalidArgument;
}

const int threads = 256;

void* kernelArgs[] = {
  &in, &out, &nRanks, &baseChunkElems, &tailChunkElems, &chunksPerShard, &slotChunkElems,
  &signalBuffer, &d_kernelReady, 
  &readyValue, &totalSteps, &d_barrierGenBase, &d_barrierSync
};

switch (datatype) {
  case ncclFloat32:
    return ncclCeLaunchReduceTyped<float>(
      in, out, nRanks, baseChunkElems, tailChunkElems, chunksPerShard, slotChunkElems,
      signalBuffer, d_kernelReady, readyValue, totalSteps, d_barrierGenBase, d_barrierSync,
      redOp, threads, kernelArgs, stream);
  case ncclFloat64:
    return ncclCeLaunchReduceTyped<double>(
      in, out, nRanks, baseChunkElems, tailChunkElems, chunksPerShard, slotChunkElems,
      signalBuffer, d_kernelReady, readyValue, totalSteps, d_barrierGenBase, d_barrierSync,
      redOp, threads, kernelArgs, stream);
  case ncclFloat16:
    return ncclCeLaunchReduceTyped<__half>(
      in, out, nRanks, baseChunkElems, tailChunkElems, chunksPerShard, slotChunkElems,
      signalBuffer, d_kernelReady, readyValue, totalSteps, d_barrierGenBase, d_barrierSync,
      redOp, threads, kernelArgs, stream);
  case ncclBfloat16:
    return ncclCeLaunchReduceTyped<hip_bfloat16>(
      in, out, nRanks, baseChunkElems, tailChunkElems, chunksPerShard, slotChunkElems,
      signalBuffer, d_kernelReady, readyValue, totalSteps, d_barrierGenBase, d_barrierSync,
      redOp, threads, kernelArgs, stream);
  case ncclInt32:
    return ncclCeLaunchReduceTyped<int32_t>(
      in, out, nRanks, baseChunkElems, tailChunkElems, chunksPerShard, slotChunkElems,
      signalBuffer, d_kernelReady, readyValue, totalSteps, d_barrierGenBase, d_barrierSync,
      redOp, threads, kernelArgs, stream);
  case ncclUint32:
    return ncclCeLaunchReduceTyped<uint32_t>(
      in, out, nRanks, baseChunkElems, tailChunkElems, chunksPerShard, slotChunkElems,
      signalBuffer, d_kernelReady, readyValue, totalSteps, d_barrierGenBase, d_barrierSync,
      redOp, threads, kernelArgs, stream);
  case ncclInt64:
    return ncclCeLaunchReduceTyped<int64_t>(
      in, out, nRanks, baseChunkElems, tailChunkElems, chunksPerShard, slotChunkElems,
      signalBuffer, d_kernelReady, readyValue, totalSteps, d_barrierGenBase, d_barrierSync,
      redOp, threads, kernelArgs, stream);
  case ncclUint64:
    return ncclCeLaunchReduceTyped<uint64_t>(
      in, out, nRanks, baseChunkElems, tailChunkElems, chunksPerShard, slotChunkElems,
      signalBuffer, d_kernelReady, readyValue, totalSteps, d_barrierGenBase, d_barrierSync,
      redOp, threads, kernelArgs, stream);
  case ncclInt8:
    return ncclCeLaunchReduceTyped<int8_t>(
      in, out, nRanks, baseChunkElems, tailChunkElems, chunksPerShard, slotChunkElems,
      signalBuffer, d_kernelReady, readyValue, totalSteps, d_barrierGenBase, d_barrierSync,
      redOp, threads, kernelArgs, stream);
  case ncclUint8:
    return ncclCeLaunchReduceTyped<uint8_t>(
      in, out, nRanks, baseChunkElems, tailChunkElems, chunksPerShard, slotChunkElems,
      signalBuffer, d_kernelReady, readyValue, totalSteps, d_barrierGenBase, d_barrierSync,
      redOp, threads, kernelArgs, stream);
  default: return ncclInvalidArgument;
}

return ncclSuccess;
}
