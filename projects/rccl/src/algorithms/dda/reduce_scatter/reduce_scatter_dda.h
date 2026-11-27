/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Derived from Meta torchcomms comms/common/algorithms/reduce_scatter/reduce_scatter_dda.cuh.
 * Includes use *.h names so RCCL hipify output (src/include/...) resolves correctly.
 * See LICENSE.txt for license information.
 ************************************************************************/

#pragma once

#include "algorithms/dda/ipc/ipc_gpu_barrier.h"
#include "algorithms/dda/device/CollCommon.h"

namespace dda::common {

// kBanked (kCopyInKernel only): stage in this call's staging bank, no trailing barrier; ipcbuffs then holds both
// banks' peer pointers.
template <typename T, int NRANKS, bool hasAcc, bool kCopyInKernel = false, bool kBanked = false>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
  __global__ void ddaReduceScatterIpc(T* const* __restrict__ ipcbuffs, T* __restrict__ recvbuff, size_t count,
                                      const T* __restrict__ sendbuff, int selfRank, const IpcGpuBarrier barrier) {
  static_assert(kCopyInKernel || !kBanked, "a copy before the launch cannot follow the device-side bank");
  bool barrierAborted = false; // sticky abort state, see IpcGpuBarrier::syncOnSameBlockIdx
  T* const* bufs = barrier.stagingBuffers<kBanked>(ipcbuffs);
  constexpr auto countPerThread = sizeof(uint4) / sizeof(T);
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  const auto idxStart = gtIdx * countPerThread;
  const auto idxEnd = count;
  const auto idxStride = gridDim.x * blockDim.x * countPerThread;

  // kCopyInKernel is the host's only staging decision (ddaReduceScatterIpcStaging).
  // The per-shard loop is the same for a one-block message and a multi-block one,
  // so there is no fused whole-buffer copy and no second size cutoff in here.
  if constexpr (kCopyInKernel) {
#pragma unroll NRANKS
    for (int s = 0; s < NRANKS; ++s) {
      const size_t off = static_cast<size_t>(s) * count;
      copyFromSrcToDest<T>(sendbuff + off, bufs[selfRank] + off, idxStart, idxEnd, idxStride);
    }
    barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */, kBanked>(
      barrierAborted);
  } else {
    // Host enqueued cudaMemcpyAsync into ddaScratch before launch.
    barrier.syncOnSameBlockIdx<false /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */>(barrierAborted);
  }

  reduceScatter<T, NRANKS, hasAcc>(bufs, recvbuff, nullptr, selfRank, NRANKS, idxStart, idxEnd, idxStride, 0);

  // Peers finish reading this rank's staging buffer before its next call reuses it; a banked call skips it.
  barrier.syncAfterPeerReads<kBanked>(barrierAborted);
}

} // namespace dda::common
