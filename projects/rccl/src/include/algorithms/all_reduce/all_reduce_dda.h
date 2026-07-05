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
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  const auto idxStart = gtIdx * countPerThread;
  const auto idxEnd = count;
  const auto idxStride = gridDim.x * blockDim.x * countPerThread;

  copyFromSrcToDest<T>(
      sendbuff, ipcbuffs[selfRank], idxStart, idxEnd, idxStride);

  barrier.syncOnSameBlockIdx<
      true /* hasPreviousMemAccess */,
      true /* hasSubsequentMemAccess */>();

  // pattern=2: full reduce into recvbuff (one-shot, not scatter)
  reduceScatter<T, NRANKS, hasAcc>(
      ipcbuffs, recvbuff, acc, selfRank, idxStart, idxEnd, idxStride, 2);

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
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  const auto idxStart = gtIdx * countPerThread;
  const auto idxEnd = countPerRank;
  const size_t idxStride = gridDim.x * blockDim.x * countPerThread;

  reduceScatter<T, NRANKS, hasAcc>(
      ipcbuffs,
      ipcbuffs[selfRank],
      acc,
      selfRank,
      idxStart,
      idxEnd,
      idxStride,
      1);

  barrier.syncOnSameBlockIdx<
      true /* hasPreviousMemAccess */,
      true /* hasSubsequentMemAccess */>();

  allGather<T, NRANKS>(
      ipcbuffs, recvbuff, selfRank, idxStart, idxEnd, idxStride, true);

  barrier.syncOnSameBlockIdx<
      true /* hasPreviousMemAccess */,
      false /* hasSubsequentMemAccess */>();
}

} // namespace meta::comms
