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

// NRANKS semantics match the CollCommon helpers:
//   - NRANKS  > 0 : compile-time clique size; the peer loop is fully unrolled and
//                   nRanksRuntime is ignored.
//   - NRANKS == 0 : runtime fallback; the clique size comes from nRanksRuntime and
//                   the peer loop is partially unrolled 8-wide, so one instantiation
//                   covers every supported clique size.
template <typename T, int NRANKS, bool hasAcc, bool kStagingCopyInKernel = false>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
  __global__ void ddaAllToAllIpc(T* const* __restrict__ ipcbuffs, T* __restrict__ recvbuff, size_t count,
                                 const T* __restrict__ sendbuff, int selfRank, int nRanksRuntime,
                                 IpcGpuBarrier barrier) {
  // use uint4 to do 16-byte loads to maximize memory efficiency
  // We assume that count % countPerThread == 0. This assumption is enforced
  // before kernel launch
  // TODO: we should be able to deal with left over as well
  const int nRanks = (NRANKS > 0) ? NRANKS : nRanksRuntime;
  constexpr int kUnroll = (NRANKS > 0) ? NRANKS : 8;
  const size_t countPerRank = count;
  constexpr auto countPerThread = sizeof(uint4) / sizeof(T);
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  const auto idxStart = gtIdx * countPerThread;
  const auto idxEnd = countPerRank;
  const auto idxStride = gridDim.x * blockDim.x * countPerThread;

  if constexpr (kStagingCopyInKernel) {
    // Small messages: fuse sendbuff -> scratch copy into the kernel to avoid
    // cudaMemcpyAsync launch overhead on ROCm.
    const size_t copyCount = count * nRanks;
    copyFromSrcToDest<T>(sendbuff, ipcbuffs[selfRank], idxStart, copyCount, idxStride);
    barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */>();
  } else {
    // Large messages: host enqueues cudaMemcpyAsync into ddaScratch before launch.
    barrier.syncOnSameBlockIdx<false /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */>();
  }

  for (size_t idx = idxStart; idx < idxEnd; idx += idxStride) {
#pragma unroll kUnroll
    for (int r = 0; r < nRanks; ++r) {
      int srcRank = r;
      int srcIdx = idx + selfRank * idxEnd;
      int destIdx = idx + r * idxEnd;
      *reinterpret_cast<uint4*>(&recvbuff[destIdx]) = reinterpret_cast<const uint4*>(&ipcbuffs[srcRank][srcIdx])[0];
    }
  }

  // barrier to ensure remote ranks won't free their buffers until I'm done
  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, false /* hasSubsequentMemAccess */>();
}

} // namespace dda::common
