/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

// CE DDA AllGather local-copy kernel -- vectorized load/store edition.
//
// Self-contained device-only TU: compiled with full HIP (host+device) by the
// device linker pipeline (see cmake/DeviceLinker.cmake) so the fat binary is
// embedded in ce_local_copy.o. ce_coll.cc (main target, --offload-host-only)
// only forward-declares and calls ncclCeLaunchLocalCopyKernel() below, so it
// has no __global__ call site of its own and produces no undefined
// __hip_fatbin_<hash> reference.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include "nccl.h"

#ifndef NCCL_CE_NUM_SLOTS
#define NCCL_CE_NUM_SLOTS 2
#endif

// Kept in sync with src/include/ce_coll.h (same value CE AllReduce's local
// reduce kernel clamps to): the grid has to leave CU/queue headroom for the
// concurrent CE/SDMA traffic feeding the other pipeline slot.
#ifndef NCCL_CE_REDUCE_MAX_BLOCKS
#define NCCL_CE_REDUCE_MAX_BLOCKS 46
#endif

// Pure byte copy: no reduction is involved, so the vector width only needs to
// respect src/dst/n alignment, not the collective's logical datatype. Same
// int4-vectorized-with-scalar-tail idiom as hierarchicalAGShuffle() in
// hierarchical_ag_shuffle.h.

#define MAX_CE_RANKS 32
struct CeCopyBatchParams {
 const uint8_t* srcs[MAX_CE_RANKS];
 uint8_t*       dsts[MAX_CE_RANKS];
 size_t         n;
 int            vecShift;
};

template <typename Vec, unsigned int UNROLL = 4>
__device__ __forceinline__ void ceVectorCopyBody(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst,
                                                  size_t n) {
  size_t nVec = n / sizeof(Vec);
  const Vec* srcV = reinterpret_cast<const Vec*>(src);
  Vec* dstV = reinterpret_cast<Vec*>(dst);
  size_t totalThreads = (size_t) gridDim.x * blockDim.x;
  size_t tid    = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = totalThreads * UNROLL;
  #pragma unroll
  for (size_t i = tid; i < nVec; i += stride) 
  {
    Vec reg[UNROLL];
    #pragma unroll
    for (unsigned int j = 0; j < UNROLL; j++)
    {
      reg[j] = srcV[i + j];
    }
    #pragma unroll
    for (unsigned int j = 0; j < UNROLL; j++)
    {
      dstV[i + j] = reg[j];
    }
  }
  // strided Tail loop for remaining bytes that don't fill a full sizeof(Vec) lane.
  for (size_t remaining = tid + (tid % UNROLL); remaining < nVec; remaining += totalThreads)
  {
    dstV[remaining] = srcV[remaining];
  }
  // Tail bytes that don't fill a full sizeof(Vec) lane.
  size_t copied = nVec * sizeof(Vec);
  size_t tailBytes = n - copied;
  if (tailBytes > 0 && blockIdx.x == 0) {
    for (size_t i = threadIdx.x; i < tailBytes; i += blockDim.x) {
      dst[copied + i] = src[copied + i];
    }
  }
}

template <typename Vec, unsigned int UNROLL = 4>
__device__ __forceinline__ void cePersistentVectorCopyBody(const uint8_t* __restrict__ src,
                                                            uint8_t* __restrict__ dst, size_t n, int blockInRank,
                                                            int blocksPerRank) {
  const Vec* srcV = reinterpret_cast<const Vec*>(src);
  Vec* dstV = reinterpret_cast<Vec*>(dst);
  const size_t nVec = n / sizeof(Vec);
  const size_t tid = (size_t)blockInRank * blockDim.x + threadIdx.x;
  const size_t stride = (size_t)blocksPerRank * blockDim.x;

  for (size_t i = tid * UNROLL; i < nVec; i += stride * UNROLL) {
#pragma unroll
    for (unsigned int j = 0; j < UNROLL; j++) {
      if (i + j < nVec) dstV[i + j] = srcV[i + j];
    }
  }

  const size_t copied = nVec * sizeof(Vec);
  if (blockInRank == 0) {
    for (size_t i = threadIdx.x; copied + i < n; i += blockDim.x) dst[copied + i] = src[copied + i];
  }
}

// One launch consumes every oversized-scratch pipeline step. Synchronization is
// a generation-numbered, barrier-free handshake modeled on CE AllReduce:
//
//   signalBuffer layout (per rank, symmetric window):
//     arm[slot, r]   = signalBuffer[slot*nRanks + r]
//                      producer r bumps this to generation g (monotonic) once
//                      its SDMA writes into our scratch slot are globally
//                      visible. A block waits until every remote producer has
//                      armed generation g before touching the slot.
//     consumed[slot] = signalBuffer[NCCL_CE_NUM_SLOTS*nRanks + slot]
//                      the last block to finish a generation publishes g here
//                      so producers (draining via LSA) may refill the slot.
//
// localSrcBase selects the own-rank source:
//   AllGather / Gather: 0           (sendBuff is a single contribution)
//   AlltoAll:           myRank*totalBytes (sendBuff is nRanks contributions)
//
// Monotonic generation numbers make each block's polling ABA-safe, so blocks
// progress independently with no grid-wide barrier. doneCounter[slot] is a
// device-local per-slot arrival counter; the last arrival advances consumed.
//
// Generations are only safe within one launch: they run 1..totalSteps/NUM_SLOTS
// and rely on the host clearing signalBuffer before the launch, so a replay that
// restarts at g=1 against a window still holding the previous run's generations
// would fall straight through both waits. Callers gate this path off under graph
// capture.
__global__ __launch_bounds__(256) void ncclCePersistentGatherCopyKernel(
  const uint8_t* sendBuff, const uint8_t* scratch, uint8_t* userRecv, size_t sub, size_t totalBytes, int nRanks,
  int myRank, size_t localSrcBase, volatile uint32_t* signalBuffer, size_t totalSteps, uint32_t* doneCounter,
  int blocksPerRank) {
  volatile uint32_t* arm = signalBuffer;
  volatile uint32_t* consumed = signalBuffer + (size_t)NCCL_CE_NUM_SLOTS * nRanks;
  // Lanes are carved out of the grid, so anything past blocksPerRank*nRanks has
  // no lane to work on. Park those blocks and keep them out of the arrival count
  // rather than letting them index a rank that does not exist.
  const uint32_t numBlocks = (uint32_t)blocksPerRank * nRanks;
  if (blockIdx.x >= numBlocks) return;
  const int rank = (int)blockIdx.x / blocksPerRank;
  const int blockInRank = (int)blockIdx.x % blocksPerRank;

  for (size_t step = 0; step < totalSteps; step++) {
    const int slot = (int)(step % NCCL_CE_NUM_SLOTS);
    const uint32_t g = (uint32_t)(step / NCCL_CE_NUM_SLOTS) + 1;
    const size_t off = step * sub;
    const size_t remaining = totalBytes - off;
    const size_t n = sub < remaining ? sub : remaining;

    // Slot gate: wait until the previous occupant of this slot has been fully
    // consumed, and until every remote producer has delivered generation g.
    if (threadIdx.x == 0) {
      while (__hip_atomic_load((uint32_t*)&consumed[slot], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) < g - 1) {
        __builtin_amdgcn_s_sleep(1);
      }
      for (int r = 0; r < nRanks; r++) {
        if (r == myRank) continue;
        volatile uint32_t* db = arm + (size_t)slot * nRanks + r;
        while (__hip_atomic_load((uint32_t*)db, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < g) {
          __builtin_amdgcn_s_sleep(1);
        }
      }
    }
    __syncthreads();

    const size_t slotOffset = (size_t)slot * nRanks * sub;
    const uint8_t* src =
      rank == myRank ? sendBuff + localSrcBase + off : scratch + slotOffset + (size_t)rank * sub;
    uint8_t* dst = userRecv + (size_t)rank * totalBytes + off;
    // In-place AllGather / AlltoAll aliases the local slice (src == dst).
    if (src != dst) {
      const uintptr_t alignment = (uintptr_t)src | (uintptr_t)dst | n;
      if ((alignment & 15) == 0) {
        cePersistentVectorCopyBody<int4>(src, dst, n, blockInRank, blocksPerRank);
      } else if ((alignment & 7) == 0) {
        cePersistentVectorCopyBody<int2>(src, dst, n, blockInRank, blocksPerRank);
      } else if ((alignment & 3) == 0) {
        cePersistentVectorCopyBody<int>(src, dst, n, blockInRank, blocksPerRank);
      } else {
        cePersistentVectorCopyBody<uint8_t>(src, dst, n, blockInRank, blocksPerRank);
      }
    }

    // Publish consumption once every block has finished generation g.
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
      uint32_t prior = __hip_atomic_fetch_add(&doneCounter[slot], 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
      if (prior + 1 == numBlocks) {
        __hip_atomic_store(&doneCounter[slot], 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        __hip_atomic_store((uint32_t*)&consumed[slot], g, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
      }
    }
  }
}

// Scatter consumer: each rank holds a single lane per slot (not nRanks lanes).
// Root is the sole producer and rings arm[slot, rootRank]; every rank copies
// scratch[slot*sub ..] -> userRecv[off ..].
__global__ __launch_bounds__(256) void ncclCePersistentScatterCopyKernel(
  const uint8_t* scratch, uint8_t* userRecv, size_t sub, size_t totalBytes, int nRanks, int rootRank,
  volatile uint32_t* signalBuffer, size_t totalSteps, uint32_t* doneCounter) {
  volatile uint32_t* arm = signalBuffer;
  volatile uint32_t* consumed = signalBuffer + (size_t)NCCL_CE_NUM_SLOTS * nRanks;
  const int blocksPerRank = (int)gridDim.x; // entire grid works one lane
  const uint32_t numBlocks = gridDim.x;

  for (size_t step = 0; step < totalSteps; step++) {
    const int slot = (int)(step % NCCL_CE_NUM_SLOTS);
    const uint32_t g = (uint32_t)(step / NCCL_CE_NUM_SLOTS) + 1;
    const size_t off = step * sub;
    const size_t remaining = totalBytes - off;
    const size_t n = sub < remaining ? sub : remaining;

    if (threadIdx.x == 0) {
      while (__hip_atomic_load((uint32_t*)&consumed[slot], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) < g - 1) {
        __builtin_amdgcn_s_sleep(1);
      }
      volatile uint32_t* db = arm + (size_t)slot * nRanks + rootRank;
      while (__hip_atomic_load((uint32_t*)db, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < g) {
        __builtin_amdgcn_s_sleep(1);
      }
    }
    __syncthreads();

    const uint8_t* src = scratch + (size_t)slot * sub;
    uint8_t* dst = userRecv + off;
    if (src != dst) {
      const uintptr_t alignment = (uintptr_t)src | (uintptr_t)dst | n;
      if ((alignment & 15) == 0) {
        cePersistentVectorCopyBody<int4>(src, dst, n, (int)blockIdx.x, blocksPerRank);
      } else if ((alignment & 7) == 0) {
        cePersistentVectorCopyBody<int2>(src, dst, n, (int)blockIdx.x, blocksPerRank);
      } else if ((alignment & 3) == 0) {
        cePersistentVectorCopyBody<int>(src, dst, n, (int)blockIdx.x, blocksPerRank);
      } else {
        cePersistentVectorCopyBody<uint8_t>(src, dst, n, (int)blockIdx.x, blocksPerRank);
      }
    }

    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
      uint32_t prior = __hip_atomic_fetch_add(&doneCounter[slot], 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
      if (prior + 1 == numBlocks) {
        __hip_atomic_store(&doneCounter[slot], 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        __hip_atomic_store((uint32_t*)&consumed[slot], g, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
      }
    }
  }
}

__global__ __launch_bounds__(256)
void ncclCeMultiSlotGatherCopyKernel(CeCopyBatchParams params) {
 // blockIdx.y determines which rank slot this thread block processes
 int slot = blockIdx.y;
 const uint8_t* __restrict__ src = params.srcs[slot];
 uint8_t*       __restrict__ dst = params.dsts[slot];
 size_t n                        = params.n;
 // Execute unrolled copy body for this rank
 switch (params.vecShift) {
   case 4: ceVectorCopyBody<int4, 4>(src, dst, n); break;
   case 3: ceVectorCopyBody<int2, 4>(src, dst, n); break;
   case 2: ceVectorCopyBody<int,  4>(src, dst, n); break;
   default: ceVectorCopyBody<uint8_t, 4>(src, dst, n); break;
 }
}

// vecShift selects the widest pack size that src, dst, and n all share:
// 4->int4 (16B), 3->int2 (8B), 2->int (4B), else scalar byte fallback.
__global__ __launch_bounds__(256)
void ncclCeLocalCopyKernel(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst, size_t n, int vecShift) {
  switch (vecShift) {
  case 4: ceVectorCopyBody<int4, 4>(src, dst, n); break;
  case 3: ceVectorCopyBody<int2, 4>(src, dst, n); break;
  case 2: ceVectorCopyBody<int,  4>(src, dst, n); break;
  default: ceVectorCopyBody<uint8_t, 4>(src, dst, n); break;
  }
}

/*ncclResult_t ncclCeLaunchMultiSlotCopyKernel(const CeCopyBatchParams& batch, int nRanks, hipStream_t stream) {
  if (batch.n == 0 || nRanks == 0) return ncclSuccess;
  const int threadsPerBlock = 256;
  static const int vecBytes[] = {1, 4, 8, 16};
  size_t nVec = batch.n / vecBytes[batch.vecShift - 1];
  // Calculate blocks required per rank
  int blocksX = (int)std::min<size_t>(128, (nVec + threadsPerBlock - 1) / threadsPerBlock);
  blocksX = std::max(blocksX, 1);
  // 2D Grid: X = blocks working on 1 rank, Y = total ranks
  dim3 grid(blocksX, nRanks);
  (void)hipGetLastError();
  ncclCeMultiSlotGatherCopyKernel<<<grid, threadsPerBlock, 0, stream>>>(batch);
  return (hipGetLastError() == hipSuccess) ? ncclSuccess : ncclUnhandledCudaError;
}*/

ncclResult_t ncclCeLaunchPersistentGatherCopy(const void* sendBuff, const void* scratch, void* userRecv, size_t sub,
                                               size_t totalBytes, int nRanks, int myRank, size_t localSrcBase,
                                               uint32_t* signalBuffer, size_t totalSteps, uint32_t* doneCounter,
                                               hipStream_t stream) {
  if (totalSteps == 0 || totalBytes == 0 || sub == 0 || nRanks <= 0) return ncclSuccess;

  const int threads = 256;
  // The kernel derives its rank lane as blockIdx.x / blocksPerRank, so the grid
  // has to stay a multiple of nRanks -- always size it from blocksPerRank rather
  // than setting the block count directly.
  const int targetBlocks = 128;
  const int blocksPerRank = std::max(1, targetBlocks / nRanks);
  const int blocks = blocksPerRank * nRanks;
  (void)hipGetLastError();
  ncclCePersistentGatherCopyKernel<<<blocks, threads, 0, stream>>>(
    (const uint8_t*)sendBuff, (const uint8_t*)scratch, (uint8_t*)userRecv, sub, totalBytes, nRanks, myRank,
    localSrcBase, signalBuffer, totalSteps, doneCounter, blocksPerRank);
  return hipGetLastError() == hipSuccess ? ncclSuccess : ncclUnhandledCudaError;
}

ncclResult_t ncclCeLaunchPersistentScatterCopy(const void* scratch, void* userRecv, size_t sub, size_t totalBytes,
                                                int nRanks, int rootRank, uint32_t* signalBuffer, size_t totalSteps,
                                                uint32_t* doneCounter, hipStream_t stream) {
  if (totalSteps == 0 || totalBytes == 0 || sub == 0 || nRanks <= 0) return ncclSuccess;

  const int threads = 256;
  const int blocks = NCCL_CE_REDUCE_MAX_BLOCKS; // single lane, whole grid on it
  (void)hipGetLastError();
  ncclCePersistentScatterCopyKernel<<<blocks, threads, 0, stream>>>(
    (const uint8_t*)scratch, (uint8_t*)userRecv, sub, totalBytes, nRanks, rootRank, signalBuffer, totalSteps,
    doneCounter);
  return hipGetLastError() == hipSuccess ? ncclSuccess : ncclUnhandledCudaError;
}

// Host launcher -- external linkage, forward-declared and called from
// ce_coll.cc (ncclCeAllGatherPipelined's copy-back loop).
ncclResult_t ncclCeLaunchLocalCopyKernel(const void* src, void* dst, size_t n, hipStream_t stream) {
  if (n == 0) return ncclSuccess;

  uintptr_t s = (uintptr_t)src, d = (uintptr_t)dst;
  int vecShift = ((s | d | n) % 16 == 0) ? 4
               : ((s | d | n) % 8  == 0) ? 3
               : ((s | d | n) % 4  == 0) ? 2 : 1;
  static const int vecBytes[] = {1, 4, 8, 16};
  const int threadsPerBlock = 256;
  size_t nVec = n / vecBytes[vecShift - 1];
  int blocks = (int)std::min<size_t>(32, (nVec + threadsPerBlock - 1) / threadsPerBlock);
  blocks = std::max(blocks, 1);

  (void)hipGetLastError(); // clear any stale error before checking this launch
  ncclCeLocalCopyKernel<<<blocks, threadsPerBlock, 0, stream>>>((const uint8_t*)src, (uint8_t*)dst, n, vecShift);
  hipError_t e = hipGetLastError();
  if (e != hipSuccess) return ncclUnhandledCudaError;
  return ncclSuccess;
}
