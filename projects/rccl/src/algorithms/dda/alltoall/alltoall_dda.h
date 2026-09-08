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

// Templated on a compile-time rank count NRANKS_CT (matching the fabric path):
//   - NRANKS_CT > 0  : specialized for that clique size; the peer loop below is
//                      fully unrolled.
//   - NRANKS_CT == 0 : runtime fallback; the rank count comes from nRanks and
//                      the peer loop is partially unrolled 8-wide.
template <typename T, int NRANKS_CT, bool hasAcc, bool kStagingCopyInKernel = false>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
  __global__ void ddaAllToAllIpc(T* const* __restrict__ ipcbuffs, T* __restrict__ recvbuff, size_t count,
                                 const T* __restrict__ sendbuff, int selfRank, int nRanks, IpcGpuBarrier barrier) {
  // use uint4 to do 16-byte loads to maximize memory efficiency
  // We assume that count % countPerThread == 0. This assumption is enforced
  // before kernel launch
  // TODO: we should be able to deal with left over as well
  const size_t countPerRank = count;
  const int nRanksEff = (NRANKS_CT > 0) ? NRANKS_CT : nRanks;
  constexpr int kUnroll = (NRANKS_CT > 0) ? NRANKS_CT : 8;
  constexpr auto countPerThread = sizeof(uint4) / sizeof(T);
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  const auto idxStart = gtIdx * countPerThread;
  const auto idxEnd = countPerRank;
  const auto idxStride = gridDim.x * blockDim.x * countPerThread;

  if constexpr (kStagingCopyInKernel) {
    // Small messages: fuse sendbuff -> scratch copy into the kernel to avoid
    // cudaMemcpyAsync launch overhead on ROCm.
    const size_t copyCount = count * nRanksEff;
    copyFromSrcToDest<T>(sendbuff, ipcbuffs[selfRank], idxStart, copyCount, idxStride);
    barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */>();
  } else {
    // Large messages: host enqueues cudaMemcpyAsync into ddaScratch before launch.
    barrier.syncOnSameBlockIdx<false /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */>();
  }

  for (size_t idx = idxStart; idx < idxEnd; idx += idxStride) {
#pragma unroll kUnroll
    for (int r = 0; r < nRanksEff; ++r) {
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
