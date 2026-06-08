/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * DDA all-reduce kernels for the fabric/VMM path, using FabricGpuBarrier.
 *
 * The kernels are templated on a compile-time rank count NRANKS_CT:
 *   - NRANKS_CT > 0  : specialized for that clique size; uses the unrolled
 *                      CollCommon reduceScatter/allGather (matching the IPC
 *                      fast path). The host launcher instantiates this for the
 *                      common sizes (e.g. 4, 8).
 *   - NRANKS_CT == 0 : runtime fallback; takes the rank count as the nRanks
 *                      argument and uses reduceScatterRuntime/allGatherRuntime,
 *                      so a single instantiation covers any other clique size
 *                      up to kDdaMaxNranks.
 * See LICENSE.txt for license information.
 ************************************************************************/

#pragma once

#include "algorithms/CollCommon.h"
#include "fabric_gpu_barrier.h"

namespace meta::comms {

// Runtime-nRanks reduce. pattern==2: one-shot full reduce into destbuff;
// pattern==1: two-shot reduce-scatter (this rank's shard into destbuff).
template <typename T, bool hasAcc>
static inline __device__ void reduceScatterRuntime(
    T* const* __restrict__ ipcbuffs,
    T* __restrict__ destbuff,
    const T* __restrict__ acc,
    int selfRank,
    int nRanks,
    const size_t idxStart,
    const size_t idxEnd,
    const size_t idxStride,
    int pattern) {
  static_assert(is_supported_type_v<T>, "dda: unsupported element type");
  for (size_t idx = idxStart; idx < idxEnd; idx += idxStride) {
    const size_t srcIdx = (pattern == 2) ? idx : (idx + selfRank * idxEnd);
    const size_t destIdx = (pattern == 1) ? (idx + selfRank * idxEnd) : idx;

    uint4 sum{0, 0, 0, 0};
    if constexpr (hasAcc) {
      sum = reinterpret_cast<const uint4*>(&acc[srcIdx])[0];
    }

    uint4 srcVals[2];
    srcVals[0] = reinterpret_cast<const uint4*>(&ipcbuffs[0][srcIdx])[0];
    for (int r = 0; r < nRanks - 1; ++r) {
      srcVals[(r + 1) & 1] =
          reinterpret_cast<const uint4*>(&ipcbuffs[r + 1][srcIdx])[0];
      sum = vecElementAdd<T>(sum, srcVals[r & 1]);
    }
    sum = vecElementAdd<T>(sum, srcVals[(nRanks - 1) & 1]);

    *reinterpret_cast<uint4*>(&destbuff[destIdx]) =
        *reinterpret_cast<const uint4*>(&sum);
  }
}

template <typename T>
static inline __device__ void allGatherRuntime(
    T* const* __restrict__ ipcbuffs,
    T* __restrict__ destbuff,
    int selfRank,
    int nRanks,
    const size_t idxStart,
    const size_t idxEnd,
    const size_t idxStride,
    bool enable_offset) {
  static_assert(is_supported_type_v<T>, "dda: unsupported element type");
  for (size_t idx = idxStart; idx < idxEnd; idx += idxStride) {
    for (int r = 0; r < nRanks; ++r) {
      const int srcRank = (selfRank + r) % nRanks;
      const int destIdx = idx + srcRank * idxEnd;
      const int srcIdx = enable_offset ? destIdx : static_cast<int>(idx);
      *reinterpret_cast<uint4*>(&destbuff[destIdx]) =
          reinterpret_cast<const uint4*>(&ipcbuffs[srcRank][srcIdx])[0];
    }
  }
}

template <typename T, int NRANKS_CT, bool hasAcc>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
__global__ void ddaAllReduceFlatFabric(
    T* const* __restrict__ ipcbuffs,
    T* __restrict__ recvbuff,
    size_t count,
    const T* __restrict__ sendbuff,
    int selfRank,
    int nRanks,
    FabricGpuBarrier barrier,
    const T* __restrict__ acc) {
  constexpr auto countPerThread = sizeof(uint4) / sizeof(T);
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  const auto idxStart = gtIdx * countPerThread;
  const auto idxEnd = count;
  const auto idxStride = gridDim.x * blockDim.x * countPerThread;

  copyFromSrcToDest<T>(
      sendbuff, ipcbuffs[selfRank], idxStart, idxEnd, idxStride);

  barrier.syncOnSameBlockIdx<
      true /* hasPreviousMemAccess */,
      true /* hasSubsequentMemAccess */>();

  if constexpr (NRANKS_CT > 0) {
    // Compile-time rank count: unrolled CollCommon reduce (matches IPC path).
    reduceScatter<T, NRANKS_CT, hasAcc>(
        ipcbuffs, recvbuff, acc, selfRank, idxStart, idxEnd, idxStride, 2);
  } else {
    reduceScatterRuntime<T, hasAcc>(
        ipcbuffs, recvbuff, acc, selfRank, nRanks, idxStart, idxEnd, idxStride,
        2);
  }

  barrier.syncOnSameBlockIdx<
      true /* hasPreviousMemAccess */,
      false /* hasSubsequentMemAccess */>();
}

template <typename T, int NRANKS_CT, bool hasAcc>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
__global__ void ddaAllReduceTreeFabric(
    T* const* __restrict__ ipcbuffs,
    T* __restrict__ recvbuff,
    size_t count,
    const T* __restrict__ sendbuff,
    int selfRank,
    int nRanks,
    FabricGpuBarrier barrier,
    const T* __restrict__ acc) {
  barrier.syncOnSameBlockIdx<
      false /* hasPreviousMemAccess */,
      true /* hasSubsequentMemAccess */>();

  // Use the compile-time rank count as the divisor when specialized.
  const int nRanksEff = (NRANKS_CT > 0) ? NRANKS_CT : nRanks;
  const size_t countPerRank = count / nRanksEff;
  constexpr auto countPerThread = sizeof(uint4) / sizeof(T);
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  const auto idxStart = gtIdx * countPerThread;
  const auto idxEnd = countPerRank;
  const size_t idxStride = gridDim.x * blockDim.x * countPerThread;

  if constexpr (NRANKS_CT > 0) {
    reduceScatter<T, NRANKS_CT, hasAcc>(
        ipcbuffs, ipcbuffs[selfRank], acc, selfRank, idxStart, idxEnd, idxStride,
        1);

    barrier.syncOnSameBlockIdx<
        true /* hasPreviousMemAccess */,
        true /* hasSubsequentMemAccess */>();

    allGather<T, NRANKS_CT>(
        ipcbuffs, recvbuff, selfRank, idxStart, idxEnd, idxStride, true);
  } else {
    reduceScatterRuntime<T, hasAcc>(
        ipcbuffs, ipcbuffs[selfRank], acc, selfRank, nRanks, idxStart, idxEnd,
        idxStride, 1);

    barrier.syncOnSameBlockIdx<
        true /* hasPreviousMemAccess */,
        true /* hasSubsequentMemAccess */>();

    allGatherRuntime<T>(
        ipcbuffs, recvbuff, selfRank, nRanks, idxStart, idxEnd, idxStride, true);
  }

  barrier.syncOnSameBlockIdx<
      true /* hasPreviousMemAccess */,
      false /* hasSubsequentMemAccess */>();
}

} // namespace meta::comms
