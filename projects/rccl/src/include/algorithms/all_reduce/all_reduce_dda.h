/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Derived from Meta torchcomms comms/common/algorithms/all_reduce/all_reduce_dda.cuh.
 * Includes use *.h names so RCCL hipify output (src/include/...) resolves correctly.
 * See LICENSE.txt for license information.
 ************************************************************************/

#pragma once

#include "ipc_gpu_barrier.h"
#include "algorithms/CollCommon.h"

namespace meta::comms {

template <typename T, int NRANKS, bool hasAcc>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
__global__ void ddaAllReduceFlatIpc(
    T* const* __restrict__ ipcbuffs,
    T* __restrict__ recvbuff,
    size_t count,
    const T* __restrict__ sendbuff,
    int selfRank,
    IpcGpuBarrier barrier,
    const T* __restrict__ acc) {
  constexpr auto countPerThread = sizeof(uint4) / sizeof(T);
  const size_t countAligned = (count / countPerThread) * countPerThread;
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  const auto idxStart = gtIdx * countPerThread;
  const auto idxEnd = countAligned;
  const auto idxStride = gridDim.x * blockDim.x * countPerThread;

  copyFromSrcToDest<T>(
      sendbuff, ipcbuffs[selfRank], idxStart, idxEnd, idxStride);

  // copy the remaining data (block 0 only)
  if (blockIdx.x == 0) {
    for (size_t idx = countAligned + threadIdx.x; idx < count; idx += blockDim.x) {
      ipcbuffs[selfRank][idx] = sendbuff[idx];
    }
  }

  barrier.syncOnSameBlockIdx<
      true /* hasPreviousMemAccess */,
      true /* hasSubsequentMemAccess */>();

  // pattern=2: full reduce into recvbuff (one-shot, not scatter)
  // count is unused for pattern=2
  reduceScatter<T, NRANKS, hasAcc>(
      ipcbuffs, recvbuff, acc, selfRank, idxStart, idxEnd, idxStride, 2, count);

  // reduce the remaining data (block 0 only)
  if (blockIdx.x == 0) {
    for (size_t idx = countAligned + threadIdx.x; idx < count; idx += blockDim.x) {
        T s = hasAcc ? acc[idx] : T(0);
        #pragma unroll
        for (int r = 0; r < NRANKS; ++r) {
            s = s + ipcbuffs[r][idx];
        }
        recvbuff[idx] = s;
    }
  }

  barrier.syncOnSameBlockIdx<
      true /* hasPreviousMemAccess */,
      false /* hasSubsequentMemAccess */>();
}

template <typename T, int NRANKS, bool hasAcc>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
__global__ void ddaAllReduceTreeIpc(
    T* const* __restrict__ ipcbuffs,
    T* __restrict__ recvbuff,
    size_t count,
    const T* __restrict__ sendbuff,
    int selfRank,
    IpcGpuBarrier barrier,
    const T* __restrict__ acc) {
  barrier.syncOnSameBlockIdx<
      false /* hasPreviousMemAccess */,
      true /* hasSubsequentMemAccess */>();

  const size_t countPerRank = count / NRANKS;
  constexpr auto countPerThread = sizeof(uint4) / sizeof(T);
  const size_t countAlignedPerRank = (countPerRank / countPerThread) * countPerThread;
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  const auto idxStart = gtIdx * countPerThread;
  const auto idxEnd = countAlignedPerRank;
  const size_t idxStride = gridDim.x * blockDim.x * countPerThread;

  reduceScatter<T, NRANKS, hasAcc>(
      ipcbuffs,
      ipcbuffs[selfRank],
      acc,
      selfRank,
      idxStart,
      idxEnd,
      idxStride,
      1,
      countPerRank);

  // tail reduce-scatter (block 0)
  if (blockIdx.x == 0) {
    for (size_t idx = countAlignedPerRank + threadIdx.x; idx < countPerRank; idx += blockDim.x) {
        const size_t srcIdx = idx + selfRank * countPerRank;
        T s = hasAcc ? acc[srcIdx] : T(0);
        #pragma unroll
        for (int r = 0; r < NRANKS; ++r) {
            s = s + ipcbuffs[r][srcIdx];
        }
        ipcbuffs[selfRank][srcIdx] = s;
    }
  }
  barrier.syncOnSameBlockIdx<
      true /* hasPreviousMemAccess */,
      true /* hasSubsequentMemAccess */>();

  allGather<T, NRANKS>(
      ipcbuffs, recvbuff, selfRank, idxStart, idxEnd, idxStride, true, countPerRank);

  // tail all-gather (block 0)
  if (blockIdx.x == 0) {
    for (size_t idx = countAlignedPerRank + threadIdx.x; idx < countPerRank; idx += blockDim.x) {
        #pragma unroll
        for (int r = 0; r < NRANKS; ++r) {
            const size_t srcIdx = idx + r * countPerRank;
            recvbuff[srcIdx] = ipcbuffs[r][srcIdx];
        }
    }
  }

  barrier.syncOnSameBlockIdx<
      true /* hasPreviousMemAccess */,
      false /* hasSubsequentMemAccess */>();
}

} // namespace meta::comms
