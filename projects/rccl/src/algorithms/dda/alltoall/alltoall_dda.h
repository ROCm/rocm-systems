/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Derived from Meta torchcomms comms/common/algorithms/all_reduce/all_reduce_dda.cuh.
 * Adapted for alltoall collective operation.
 * Includes use *.h names so RCCL hipify output (src/include/...) resolves correctly.
 * See LICENSE.txt for license information.
 ************************************************************************/

#pragma once

#include "algorithms/dda/ipc/ipc_gpu_barrier.h"
#include "algorithms/dda/device/CollCommon.h"

namespace dda::common {

// kBanked (kStagingCopyInKernel only): stage in this call's staging bank, no trailing barrier; ipcbuffs then holds
// both banks' peer pointers.
template <typename T, int NRANKS, bool hasAcc, bool kStagingCopyInKernel = false, bool kBanked = false>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
  __global__ void ddaAllToAllIpc(T* const* __restrict__ ipcbuffs, T* __restrict__ recvbuff, size_t count,
                                 const T* __restrict__ sendbuff, int selfRank, const IpcGpuBarrier barrier) {
  static_assert(kStagingCopyInKernel || !kBanked, "a copy before the launch cannot follow the device-side bank");
  bool barrierAborted = false; // sticky abort state, see IpcGpuBarrier::syncOnSameBlockIdx
  T* const* bufs = barrier.stagingBuffers<kBanked>(ipcbuffs);
  // use uint4 to do 16-byte loads to maximize memory efficiency
  // We assume that count % countPerThread == 0. This assumption is enforced
  // before kernel launch
  // TODO: we should be able to deal with left over as well
  const size_t countPerRank = count;
  constexpr auto countPerThread = sizeof(uint4) / sizeof(T);
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  const auto idxStart = gtIdx * countPerThread;
  const auto idxEnd = countPerRank;
  const auto idxStride = gridDim.x * blockDim.x * countPerThread;

  if constexpr (kStagingCopyInKernel) {
    // Small messages: fuse sendbuff -> scratch copy into the kernel to avoid
    // cudaMemcpyAsync launch overhead on ROCm.
    const size_t copyCount = count * NRANKS;
    copyFromSrcToDest<T>(sendbuff, bufs[selfRank], idxStart, copyCount, idxStride);
    barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */, kBanked>(
      barrierAborted);
  } else {
    // Large messages: host enqueues cudaMemcpyAsync into ddaScratch before launch.
    barrier.syncOnSameBlockIdx<false /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */>(barrierAborted);
  }

  for (size_t idx = idxStart; idx < idxEnd; idx += idxStride) {
#pragma unroll NRANKS
    for (int r = 0; r < NRANKS; ++r) {
      int srcRank = r;
      size_t srcIdx = idx + selfRank * idxEnd;
      size_t destIdx = idx + r * idxEnd;
      *reinterpret_cast<uint4*>(&recvbuff[destIdx]) = reinterpret_cast<const uint4*>(&bufs[srcRank][srcIdx])[0];
    }
  }

  // Peers finish reading this rank's staging buffer before its next call reuses it; a banked call skips it.
  barrier.syncAfterPeerReads<kBanked>(barrierAborted);
}

} // namespace dda::common
