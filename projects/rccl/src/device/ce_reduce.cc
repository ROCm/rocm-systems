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
namespace cg = cooperative_groups;
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
void ncclCeLocalReduceKernelVec(
   const T* __restrict__ in,
   T* __restrict__ out, // recvbuff + rank*shardBytes (final AllReduce shard region)
   int nRanks,
   size_t baseChunkElems,
   size_t tailChunkElems,
   size_t chunksPerShard,
   volatile uint32_t* signalBuffer,
   volatile uint32_t* d_quitFlag,
   volatile uint32_t* d_kernelReady,
   volatile uint32_t* d_reduceDone,   // published as (step+1) after each chunk
   uint32_t readyValue,  
   size_t totalSteps)
{
   cg::grid_group grid = cg::this_grid();
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
  grid.sync();


   // Persistent execution loop tracking total step transitions
   while (step < totalSteps) {
       const int    slot         = (int)(step % 2); // NUM_SLOTS = 2
       const size_t chunkInShard = step % chunksPerShard;

       const bool isTailChunk = (chunkInShard == chunksPerShard - 1) && (tailChunkElems > 0);
       const size_t currentChunkElems = isTailChunk ? tailChunkElems : baseChunkElems;
       const size_t nVec = currentChunkElems / W;
        
       // Block 0, thread 0: Poll on the inbound slots until all doorbells arrive
       if (blockIdx.x == 0 && threadIdx.x == 0) {
           while (true) {
               // Optional: Check quit flag if supporting early truncation/fault recovery
               /* uint32_t qf = __hip_atomic_load((uint32_t*)d_quitFlag, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
                  if (qf) break; */

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
       grid.sync();

       // --- Core Computational Reduction Phase ---
       const size_t slotStrideElems = baseChunkElems * (size_t)nRanks;
       const T* __restrict__ slotIn = in + ((size_t)slot * slotStrideElems);
       
       // CRITICAL UPDATE: Divert target writes to the rotating double-buffer scratch segment
       T* __restrict__ chunkOut = out + ((size_t)chunkInShard * baseChunkElems);
       // Vectorized Unrolled Path
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
                   size_t rankOff = (size_t)r*baseChunkElems + (vi + (size_t)i*stride)*W;
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

       // Vectorized Tail Path
       size_t vectorTailStart = nVec - (nVec % (stride * U));
       if (nVec > 0 && nVec % (stride * U) != 0) {
           for (size_t vIdx = vectorTailStart + tid; vIdx < nVec; vIdx += stride) {
               Pack acc;
               acc.vec = *reinterpret_cast<const LT*>(slotIn + vIdx*W);
               for (int r = 1; r < nRanks; r++) {
                   Pack tmp;
                   tmp.vec = *reinterpret_cast<const LT*>(slotIn + (size_t)r*baseChunkElems + vIdx*W);
                   #pragma unroll
                   for (int k = 0; k < W; k++)
                       acc.v[k] = ReduceOp<T, RedOp>::apply(acc.v[k], tmp.v[k]);
               }
               *reinterpret_cast<LT*>(chunkOut + vIdx*W) = acc.vec;
           }
       }

       // Elemental Scalar Tail Path
       const size_t tailBase = nVec * W;
       if (tailBase < currentChunkElems) {
           for (size_t i = tid; tailBase + i < currentChunkElems; i += stride) {
               size_t globalIdx = tailBase + i;
               T a = slotIn[globalIdx];
               for (int r = 1; r < nRanks; r++)
                   a = ReduceOp<T, RedOp>::apply(a, slotIn[(size_t)r*baseChunkElems + globalIdx]);
               chunkOut[globalIdx] = a;
           }
       }

       // Ensure calculations completely flush to cache line bounds before clearing flags
       grid.sync();
       if (blockIdx.x == 0 && threadIdx.x == 0) {
        __hip_atomic_store((uint32_t*)d_reduceDone, step + 1,
                           __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
       }

       // Release-mark step: Clear inbound signals back to 0 to inform the host loop
       if (blockIdx.x == 0) {
           for (int r = (int)threadIdx.x; r < nRanks; r += (int)blockDim.x) {
               __hip_atomic_store((uint32_t*)&signalBuffer[(size_t)slot * nRanks + r],
                                  0, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
           }
       }
       
       grid.sync();
       step++;
   }
}
// *********************************************************************************
// ncclCeLaunchPersistentReduce — cooperative-launch dispatch over datatype & op.
// Same kernelArgs work for any T because all args are raw pointers / scalars;
// only the instantiated function pointer and the block count depend on T.
// *********************************************************************************
ncclResult_t ncclCeLaunchPersistentReduce(
  const void* in, void* out, int nRanks,
  size_t baseChunkElems, size_t tailChunkElems, size_t chunksPerShard,
  uint32_t* signalBuffer, uint32_t* d_quitFlag, uint32_t* d_kernelReady,
  uint32_t* d_reduceDone, uint32_t readyValue, size_t totalSteps,
  ncclDataType_t datatype, ncclRedOp_t op, hipStream_t stream) {

int redOp;
switch (op) {
  case ncclSum:  redOp = 0; break;
  case ncclProd: redOp = 1; break;
  case ncclMin:  redOp = 2; break;
  case ncclMax:  redOp = 3; break;
  default: return ncclInvalidArgument;
}

const int threads = 256;

// hipLaunchCooperativeKernel copies sizeof(pointer)/sizeof(scalar) bytes from
// each address; identical layout for every T, so this array is type-agnostic.
void* kernelArgs[] = {
  &in, &out, &nRanks, &baseChunkElems, &tailChunkElems, &chunksPerShard,
  &signalBuffer, &d_quitFlag, &d_kernelReady, &d_reduceDone,
  &readyValue, &totalSteps
};

#define NCCL_CE_COOP_BLOCKS(T)                                                 \
(int)std::clamp<size_t>(                                                     \
  (baseChunkElems / VecTrait<T>::W + (size_t)threads*VecTrait<T>::GpuUnroll - 1) \
    / ((size_t)threads * VecTrait<T>::GpuUnroll), (size_t)1, (size_t)46)

#define NCCL_CE_COOP_LAUNCH(T)                                                 \
do {                                                                         \
  constexpr int U = VecTrait<T>::GpuUnroll;                                  \
  const int _blocks = NCCL_CE_COOP_BLOCKS(T);                               \
  const void* fn = nullptr;                                                  \
  switch (redOp) {                                                           \
    case 0: fn = (const void*)ncclCeLocalReduceKernelVec<T,0,U>; break;      \
    case 1: fn = (const void*)ncclCeLocalReduceKernelVec<T,1,U>; break;      \
    case 2: fn = (const void*)ncclCeLocalReduceKernelVec<T,2,U>; break;      \
    case 3: fn = (const void*)ncclCeLocalReduceKernelVec<T,3,U>; break;      \
  }                                                                         \
  (void)hipGetLastError();                                                   \
  hipError_t _e = hipLaunchCooperativeKernel(                               \
      fn, dim3(_blocks), dim3(threads), kernelArgs, 0, stream);             \
  if (_e != hipSuccess) {                                                    \
    printf("[CE persistent reduce] launch failed: %d (%s)\n",              \
           (int)_e, hipGetErrorString(_e));                                 \
    return ncclUnhandledCudaError;                                          \
  }                                                                         \
} while (0)

switch (datatype) {
  case ncclFloat32:  NCCL_CE_COOP_LAUNCH(float);        break;
  case ncclFloat64:  NCCL_CE_COOP_LAUNCH(double);       break;
  case ncclFloat16:  NCCL_CE_COOP_LAUNCH(__half);       break;
  case ncclBfloat16: NCCL_CE_COOP_LAUNCH(hip_bfloat16); break;
  case ncclInt32:    NCCL_CE_COOP_LAUNCH(int32_t);      break;
  case ncclUint32:   NCCL_CE_COOP_LAUNCH(uint32_t);     break;
  case ncclInt64:    NCCL_CE_COOP_LAUNCH(int64_t);      break;
  case ncclUint64:   NCCL_CE_COOP_LAUNCH(uint64_t);     break;
  case ncclInt8:     NCCL_CE_COOP_LAUNCH(int8_t);       break;
  case ncclUint8:    NCCL_CE_COOP_LAUNCH(uint8_t);      break;
  default: return ncclInvalidArgument;
}

#undef NCCL_CE_COOP_LAUNCH
#undef NCCL_CE_COOP_BLOCKS
return ncclSuccess;
}
