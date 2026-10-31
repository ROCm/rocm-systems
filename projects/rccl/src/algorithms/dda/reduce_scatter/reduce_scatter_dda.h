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

// NRANKS semantics match the CollCommon helpers:
//   - NRANKS  > 0 : compile-time clique size; the peer loop is fully unrolled and
//                   nRanksRuntime is ignored.
//   - NRANKS == 0 : runtime fallback; the clique size comes from nRanksRuntime and
//                   the peer loop is partially unrolled 8-wide, so one instantiation
//                   covers every supported clique size.
template <typename T, int NRANKS, bool hasAcc, bool kCopyInKernel = false>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
  __global__ void ddaReduceScatterIpc(T* const* __restrict__ ipcbuffs, T* __restrict__ recvbuff, size_t count,
                                      const T* __restrict__ sendbuff, int selfRank, int nRanksRuntime,
                                      IpcGpuBarrier barrier) {
  const int nRanks = (NRANKS > 0) ? NRANKS : nRanksRuntime;
  constexpr int kUnroll = (NRANKS > 0) ? NRANKS : 8;
  constexpr auto countPerThread = sizeof(uint4) / sizeof(T);
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  const auto idxStart = gtIdx * countPerThread;
  const auto idxEnd = count;
  const auto idxStride = gridDim.x * blockDim.x * countPerThread;

  // kCopyInKernel is the host's only staging decision (ddaReduceScatterIpcStaging).
  // The per-shard loop is the same for a one-block message and a multi-block one,
  // so there is no fused whole-buffer copy and no second size cutoff in here.
  if constexpr (kCopyInKernel) {
#pragma unroll kUnroll
    for (int s = 0; s < nRanks; ++s) {
      const size_t off = static_cast<size_t>(s) * count;
      copyFromSrcToDest<T>(sendbuff + off, ipcbuffs[selfRank] + off, idxStart, idxEnd, idxStride);
    }
    barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */>();
  } else {
    // Host enqueued cudaMemcpyAsync into ddaScratch before launch.
    barrier.syncOnSameBlockIdx<false /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */>();
  }

  reduceScatter<T, NRANKS, hasAcc>(ipcbuffs, recvbuff, nullptr, selfRank, nRanks, idxStart, idxEnd, idxStride, 0);

  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, false /* hasSubsequentMemAccess */>();
}

} // namespace dda::common
