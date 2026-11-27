/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Derived from Meta torchcomms comms/common/algorithms/all_reduce/all_reduce_dda.cuh.
 * Includes use *.h names so RCCL hipify output (src/include/...) resolves correctly.
 * See LICENSE.txt for license information.
 ************************************************************************/

#pragma once

#include "algorithms/dda/ipc/ipc_gpu_barrier.h"
#include "algorithms/dda/device/CollCommon.h"

namespace dda::common {

// kBanked: stage in this call's staging bank, no trailing barrier; ipcbuffs then holds both banks' peer pointers.
template <typename T, int NRANKS, bool hasAcc, bool kBanked = false>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
  __global__ void ddaAllReduceFlatIpc(T* const* __restrict__ ipcbuffs, T* __restrict__ recvbuff, size_t count,
                                      const T* __restrict__ sendbuff, int selfRank, const IpcGpuBarrier barrier,
                                      const T* __restrict__ acc) {
  bool barrierAborted = false; // sticky abort state, see IpcGpuBarrier::syncOnSameBlockIdx
  T* const* bufs = barrier.stagingBuffers<kBanked>(ipcbuffs);
  constexpr auto countPerThread = sizeof(uint4) / sizeof(T);
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  const auto idxStart = gtIdx * countPerThread;
  const auto idxEnd = count;
  const auto idxStride = gridDim.x * blockDim.x * countPerThread;

  copyFromSrcToDest<T>(sendbuff, bufs[selfRank], idxStart, idxEnd, idxStride);

  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */, kBanked>(
    barrierAborted);

  // pattern=2: full reduce into recvbuff (one-shot, not scatter)
  reduceScatter<T, NRANKS, hasAcc>(bufs, recvbuff, acc, selfRank, NRANKS, idxStart, idxEnd, idxStride, 2);

  // Peers finish reading this rank's staging buffer before its next call reuses it; a banked call skips it.
  barrier.syncAfterPeerReads<kBanked>(barrierAborted);
}

template <typename T, int NRANKS, bool hasAcc>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
  __global__ void ddaAllReduceTreeIpc(T* const* __restrict__ ipcbuffs, T* __restrict__ recvbuff, size_t count,
                                      const T* __restrict__ sendbuff, int selfRank, const IpcGpuBarrier barrier,
                                      const T* __restrict__ acc) {
  bool barrierAborted = false; // sticky abort state, see IpcGpuBarrier::syncOnSameBlockIdx
  barrier.syncOnSameBlockIdx<false /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */>(barrierAborted);

  const size_t countPerRank = count / NRANKS;
  constexpr auto countPerThread = sizeof(uint4) / sizeof(T);
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  const auto idxStart = gtIdx * countPerThread;
  const auto idxEnd = countPerRank;
  const size_t idxStride = gridDim.x * blockDim.x * countPerThread;

  reduceScatter<T, NRANKS, hasAcc>(ipcbuffs, ipcbuffs[selfRank], acc, selfRank, NRANKS, idxStart, idxEnd, idxStride, 1);

  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */>(barrierAborted);

  allGather<T, NRANKS>(ipcbuffs, recvbuff, selfRank, NRANKS, idxStart, idxEnd, idxStride, true);

  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, false /* hasSubsequentMemAccess */>(barrierAborted);
}

} // namespace dda::common
