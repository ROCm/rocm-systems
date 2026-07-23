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

#ifndef NCCL_CE_REDUCE_MAX_BLOCKS
#define NCCL_CE_REDUCE_MAX_BLOCKS 46
#endif

#ifndef NCCL_CE_NUM_SLOTS
#define NCCL_CE_NUM_SLOTS 2
#endif

#define NCCL_CE_WORK_NEXT   0
#define NCCL_CE_WORK_TOTAL  1
#define NCCL_CE_WORK_DONE   2
#define NCCL_CE_WORK_READY  3
#define NCCL_CE_WORK_WORDS_PER_SLOT 4

// Persistent path: 2x max blocks keeps all resident CTAs fed via work stealing.
#ifndef NCCL_CE_REDUCE_PARTS_FACTOR
#define NCCL_CE_REDUCE_PARTS_FACTOR 2
#endif

// NCCL_CE_REDUCE_ALL_OPS=0 (default): Sum-only reduce kernels.
// NCCL_CE_REDUCE_ALL_OPS=1 (BUILD_CE_REDUCE_ALL_OPS=ON): also Prod/Min/Max.
#ifndef NCCL_CE_REDUCE_ALL_OPS
#define NCCL_CE_REDUCE_ALL_OPS 0
#endif

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
// Static chunk reduce — grid-stride (totalSteps == 1 path).
// *********************************************************************************
template<typename T, int RedOp, int U, int W, typename LT>
__device__ __forceinline__ void ncclCeReduceChunkStatic(
    const T* __restrict__ slotIn, T* __restrict__ chunkOut, size_t slotChunkElems, int nRanks,
    size_t tid, size_t stride, size_t nVec, size_t currentChunkElems) {
  union alignas(16) Pack { LT vec; T v[W]; };

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
}

// *********************************************************************************
// Partition reduce — block-local stride (persistent totalSteps > 1 path).
// vStart/vEnd are vector indices; isLastPart also handles chunk scalar tail.
// *********************************************************************************
template<typename T, int RedOp, int U, int W, typename LT>
__device__ __forceinline__ void ncclCeReduceChunkPartition(
    const T* __restrict__ slotIn, T* __restrict__ chunkOut, size_t slotChunkElems, int nRanks,
    size_t vStart, size_t vEnd, size_t nVec, size_t currentChunkElems, bool isLastPart) {
  union alignas(16) Pack { LT vec; T v[W]; };
  const size_t localTid = (size_t)threadIdx.x;
  const size_t localStride = (size_t)blockDim.x;
  const size_t partNVec = (vEnd > vStart) ? (vEnd - vStart) : 0;

  size_t vi = localTid;
  for (; vi + (U-1) * localStride < partNVec; vi += localStride * U) {
    Pack acc[U];
    #pragma unroll
    for (int i = 0; i < U; i++) {
      acc[i].vec = *reinterpret_cast<const LT*>(slotIn + (vStart + vi + (size_t)i*localStride)*W);
    }
    for (int r = 1; r < nRanks; r++) {
      Pack tmp[U];
      #pragma unroll
      for (int i = 0; i < U; i++) {
        size_t rankOff = (size_t)r*slotChunkElems + (vStart + vi + (size_t)i*localStride)*W;
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
      *reinterpret_cast<LT*>(chunkOut + (vStart + vi + (size_t)i*localStride)*W) = acc[i].vec;
    }
  }

  if (partNVec > 0) {
    const size_t vectorTailStart = partNVec - (partNVec % (localStride * U));
    if (partNVec % (localStride * U) != 0) {
      for (size_t vIdx = vStart + vectorTailStart + localTid; vIdx < vEnd; vIdx += localStride) {
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
  }

  if (!isLastPart) return;

  const size_t tailBase = nVec * W;
  if (tailBase < currentChunkElems) {
    for (size_t i = localTid; tailBase + i < currentChunkElems; i += localStride) {
      size_t globalIdx = tailBase + i;
      T a = slotIn[globalIdx];
      for (int r = 1; r < nRanks; r++)
        a = ReduceOp<T, RedOp>::apply(a, slotIn[(size_t)r*slotChunkElems + globalIdx]);
      chunkOut[globalIdx] = a;
    }
  }
}

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
  size_t totalSteps,
  uint32_t* d_barrierSync) {

  if (blockIdx.x >= NCCL_CE_REDUCE_MAX_BLOCKS) return;

  using LT = typename VecTrait<T>::LoadT;
  constexpr int W = VecTrait<T>::W;

  if (totalSteps > 1) {
    // -------------------------------------------------------------------------
    // Persistent path: dynamic partitions, always launched with MAX blocks.
    // Block 0 waits on signalBuffer; workers steal partitions via workNext.
    // -------------------------------------------------------------------------
    uint32_t step = 0;
    while (step < totalSteps) {
      const int    slot         = (int)(step % NCCL_CE_NUM_SLOTS);
      const size_t chunkInShard = step % chunksPerShard;

      const bool isTailChunk = (chunkInShard == chunksPerShard - 1) && (tailChunkElems > 0);
      const size_t currentChunkElems = isTailChunk ? tailChunkElems : baseChunkElems;
      const size_t nVec = currentChunkElems / W;

      uint32_t* ws = d_barrierSync + (size_t)slot * NCCL_CE_WORK_WORDS_PER_SLOT;
      uint32_t* workNext  = ws + NCCL_CE_WORK_NEXT;
      uint32_t* workTotal = ws + NCCL_CE_WORK_TOTAL;
      uint32_t* workDone  = ws + NCCL_CE_WORK_DONE;
      uint32_t* slotReady = ws + NCCL_CE_WORK_READY;

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

        size_t totalParts = 1;
        if (nVec > 0) {
          const size_t maxParts = (size_t)NCCL_CE_REDUCE_MAX_BLOCKS * NCCL_CE_REDUCE_PARTS_FACTOR;
          totalParts = nVec < maxParts ? nVec : maxParts;
        }
        *workNext  = 0;
        *workDone  = 0;
        *workTotal = (uint32_t)totalParts;
        __hip_atomic_store(slotReady, 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
      }

      if (threadIdx.x == 0) {
        while (__hip_atomic_load(slotReady, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) != 1) {
          __builtin_amdgcn_s_sleep(1);
        }
      }
      __syncthreads();

      const size_t slotStrideElems = slotChunkElems * (size_t)nRanks;
      const T* __restrict__ slotIn = in + ((size_t)slot * slotStrideElems);
      T* __restrict__ chunkOut = out + ((size_t)chunkInShard * baseChunkElems);

      const uint32_t totalPartsVal = __hip_atomic_load(workTotal, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      const size_t partVecs = (nVec + (size_t)totalPartsVal - 1) / (size_t)totalPartsVal;

      __shared__ uint32_t s_partId;
      while (true) {
        if (threadIdx.x == 0) {
          s_partId = __hip_atomic_fetch_add(workNext, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
        __syncthreads();
        const uint32_t partId = s_partId;
        if (partId >= totalPartsVal) break;

        const size_t vStart = (size_t)partId * partVecs;
        const size_t vEnd = vStart + partVecs < nVec ? vStart + partVecs : nVec;
        const bool isLastPart = (partId + 1 == totalPartsVal);

        ncclCeReduceChunkPartition<T, RedOp, U, W, LT>(
            slotIn, chunkOut, slotChunkElems, nRanks,
            vStart, vEnd, nVec, currentChunkElems, isLastPart);

        if (threadIdx.x == 0) {
          uint32_t prev = __hip_atomic_fetch_add(workDone, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
          if (prev + 1 == totalPartsVal) {
            for (int r = 0; r < nRanks; r++) {
              __hip_atomic_store((uint32_t*)&signalBuffer[(size_t)slot * nRanks + r],
                                  0, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
            }
            __hip_atomic_store(slotReady, 0, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
          }
        }
        __syncthreads();
      }

      step++;
    }
  } else {
    // -------------------------------------------------------------------------
    // Single-step path: static grid-stride (unchanged behavior).
    // -------------------------------------------------------------------------
    const size_t tid    = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)blockDim.x * gridDim.x;

    const int    slot         = 0;
    const size_t chunkInShard = 0;
    const bool isTailChunk = (chunkInShard == chunksPerShard - 1) && (tailChunkElems > 0);
    const size_t currentChunkElems = isTailChunk ? tailChunkElems : baseChunkElems;
    const size_t nVec = currentChunkElems / W;

    const size_t slotStrideElems = slotChunkElems * (size_t)nRanks;
    const T* __restrict__ slotIn = in + ((size_t)slot * slotStrideElems);
    T* __restrict__ chunkOut = out + ((size_t)chunkInShard * baseChunkElems);

    ncclCeReduceChunkStatic<T, RedOp, U, W, LT>(
        slotIn, chunkOut, slotChunkElems, nRanks,
        tid, stride, nVec, currentChunkElems);
  }
}

// *********************************************************************************
// ncclCeLaunchReduceTyped — per-type launch dispatch.
// *********************************************************************************
template<typename T>
static ncclResult_t ncclCeLaunchReduceTyped(
  const void* in, void* out, int nRanks,
  size_t baseChunkElems, size_t tailChunkElems, size_t chunksPerShard, size_t slotChunkElems,
  uint32_t* signalBuffer, size_t totalSteps, uint32_t* d_barrierSync, int redOp, int threads,
  hipStream_t stream, int /*coopLaunch*/) {

  constexpr int U = VecTrait<T>::GpuUnroll;
  const int blocks = (totalSteps > 1)
    ? NCCL_CE_REDUCE_MAX_BLOCKS
    : (int)std::clamp<size_t>(
        (baseChunkElems / VecTrait<T>::W + (size_t)threads * VecTrait<T>::GpuUnroll - 1)
          / ((size_t)threads * VecTrait<T>::GpuUnroll), (size_t)1, (size_t)NCCL_CE_REDUCE_MAX_BLOCKS);

  (void)hipGetLastError();
  switch (redOp) {
    case ncclSum: {
      auto kernelFn = ncclCeLocalReduceKernelVec<T, 0, U>;
      hipLaunchKernelGGL(
        kernelFn,
        dim3(blocks), dim3(threads), 0, stream,
        static_cast<const T*>(in), static_cast<T*>(out), nRanks,
        baseChunkElems, tailChunkElems, chunksPerShard, slotChunkElems,
        signalBuffer, totalSteps, d_barrierSync);
      break;
    }
#if NCCL_CE_REDUCE_ALL_OPS
    case ncclProd: {
      auto kernelFn = ncclCeLocalReduceKernelVec<T, 1, U>;
      hipLaunchKernelGGL(
        kernelFn,
        dim3(blocks), dim3(threads), 0, stream,
        static_cast<const T*>(in), static_cast<T*>(out), nRanks,
        baseChunkElems, tailChunkElems, chunksPerShard, slotChunkElems,
        signalBuffer, totalSteps, d_barrierSync);
      break;
    }
    case ncclMin: {
      auto kernelFn = ncclCeLocalReduceKernelVec<T, 2, U>;
      hipLaunchKernelGGL(
        kernelFn,
        dim3(blocks), dim3(threads), 0, stream,
        static_cast<const T*>(in), static_cast<T*>(out), nRanks,
        baseChunkElems, tailChunkElems, chunksPerShard, slotChunkElems,
        signalBuffer, totalSteps, d_barrierSync);
      break;
    }
    case ncclMax: {
      auto kernelFn = ncclCeLocalReduceKernelVec<T, 3, U>;
      hipLaunchKernelGGL(
        kernelFn,
        dim3(blocks), dim3(threads), 0, stream,
        static_cast<const T*>(in), static_cast<T*>(out), nRanks,
        baseChunkElems, tailChunkElems, chunksPerShard, slotChunkElems,
        signalBuffer, totalSteps, d_barrierSync);
      break;
    }
#endif
    default:
      return ncclInvalidArgument;
  }
  hipError_t e = hipGetLastError();
  if (e != hipSuccess) return ncclUnhandledCudaError;
  return ncclSuccess;
}

// *********************************************************************************
// ncclCeLaunchPersistentReduce — host launcher
// *********************************************************************************
ncclResult_t ncclCeLaunchPersistentReduce(
  const void* in, void* out, int nRanks,
  size_t baseChunkElems, size_t tailChunkElems, size_t chunksPerShard,
  size_t slotChunkElems, uint32_t* signalBuffer, size_t totalSteps, uint32_t* d_barrierSync,
  ncclDataType_t datatype, ncclRedOp_t op, hipStream_t stream, int coopLaunch) {

  int redOp;
  switch (op) {
    case ncclSum:  redOp = 0; break;
#if NCCL_CE_REDUCE_ALL_OPS
    case ncclProd: redOp = 1; break;
    case ncclMin:  redOp = 2; break;
    case ncclMax:  redOp = 3; break;
#endif
    default: return ncclInvalidArgument;
  }

  const int threads = 256;

  switch (datatype) {
    case ncclFloat32:
      return ncclCeLaunchReduceTyped<float>(
        in, out, nRanks, baseChunkElems, tailChunkElems, chunksPerShard, slotChunkElems,
        signalBuffer, totalSteps, d_barrierSync, redOp, threads, stream, coopLaunch);
    case ncclFloat64:
      return ncclCeLaunchReduceTyped<double>(
        in, out, nRanks, baseChunkElems, tailChunkElems, chunksPerShard, slotChunkElems,
        signalBuffer, totalSteps, d_barrierSync, redOp, threads, stream, coopLaunch);
    case ncclFloat16:
      return ncclCeLaunchReduceTyped<__half>(
        in, out, nRanks, baseChunkElems, tailChunkElems, chunksPerShard, slotChunkElems,
        signalBuffer, totalSteps, d_barrierSync, redOp, threads, stream, coopLaunch);
    case ncclBfloat16:
      return ncclCeLaunchReduceTyped<hip_bfloat16>(
        in, out, nRanks, baseChunkElems, tailChunkElems, chunksPerShard, slotChunkElems,
        signalBuffer, totalSteps, d_barrierSync, redOp, threads, stream, coopLaunch);
    case ncclInt32:
      return ncclCeLaunchReduceTyped<int32_t>(
        in, out, nRanks, baseChunkElems, tailChunkElems, chunksPerShard, slotChunkElems,
        signalBuffer, totalSteps, d_barrierSync, redOp, threads, stream, coopLaunch);
    case ncclUint32:
      return ncclCeLaunchReduceTyped<uint32_t>(
        in, out, nRanks, baseChunkElems, tailChunkElems, chunksPerShard, slotChunkElems,
        signalBuffer, totalSteps, d_barrierSync, redOp, threads, stream, coopLaunch);
    case ncclInt64:
      return ncclCeLaunchReduceTyped<int64_t>(
        in, out, nRanks, baseChunkElems, tailChunkElems, chunksPerShard, slotChunkElems,
        signalBuffer, totalSteps, d_barrierSync, redOp, threads, stream, coopLaunch);
    case ncclUint64:
      return ncclCeLaunchReduceTyped<uint64_t>(
        in, out, nRanks, baseChunkElems, tailChunkElems, chunksPerShard, slotChunkElems,
        signalBuffer, totalSteps, d_barrierSync, redOp, threads, stream, coopLaunch);
    case ncclInt8:
      return ncclCeLaunchReduceTyped<int8_t>(
        in, out, nRanks, baseChunkElems, tailChunkElems, chunksPerShard, slotChunkElems,
        signalBuffer, totalSteps, d_barrierSync, redOp, threads, stream, coopLaunch);
    case ncclUint8:
      return ncclCeLaunchReduceTyped<uint8_t>(
        in, out, nRanks, baseChunkElems, tailChunkElems, chunksPerShard, slotChunkElems,
        signalBuffer, totalSteps, d_barrierSync, redOp, threads, stream, coopLaunch);
    default: return ncclInvalidArgument;
  }

  return ncclSuccess;
}
