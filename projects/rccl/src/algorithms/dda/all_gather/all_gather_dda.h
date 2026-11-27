/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Derived from Meta torchcomms comms/common/algorithms/all_reduce/all_reduce_dda.cuh.
 * Adapted for allgather collective operation.
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
  __global__ void ddaAllGatherIpc(T* const* __restrict__ ipcbuffs, T* __restrict__ recvbuff, size_t count,
                                  const T* __restrict__ sendbuff, int selfRank, const IpcGpuBarrier barrier) {
  bool barrierAborted = false; // sticky abort state, see IpcGpuBarrier::syncOnSameBlockIdx
  T* const* bufs = barrier.stagingBuffers<kBanked>(ipcbuffs);

  const size_t countPerRank = count;
  constexpr auto countPerThread = sizeof(uint4) / sizeof(T);
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  const auto idxStart = gtIdx * countPerThread;
  const auto idxEnd = countPerRank;
  const auto idxStride = gridDim.x * blockDim.x * countPerThread;

  // It is expensive to launch hipMemcpyAsync on ROCm
  // Move data copy here. Each block copies part of sendbuff data
  copyFromSrcToDest<T>(sendbuff, bufs[selfRank], idxStart, idxEnd, idxStride);

  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */, kBanked>(
    barrierAborted);

  allGather<T, NRANKS>(bufs, recvbuff, selfRank, NRANKS, idxStart, idxEnd, idxStride, false);

  // Peers finish reading this rank's staging buffer before its next call reuses it; a banked call skips it.
  barrier.syncAfterPeerReads<kBanked>(barrierAborted);
}

} // namespace dda::common
