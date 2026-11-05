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

template <typename T, int NRANKS, bool hasAcc, bool kStagingCopyInKernel = false>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
  __global__ void ddaAllToAllIpc(T* const* __restrict__ ipcbuffs, T* __restrict__ recvbuff, size_t count,
                                 const T* __restrict__ sendbuff, int selfRank, IpcGpuBarrier barrier) {
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
    copyFromSrcToDest<T>(sendbuff, ipcbuffs[selfRank], idxStart, copyCount, idxStride);
    barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */>();
  } else {
    // Large messages: host enqueues cudaMemcpyAsync into ddaScratch before launch.
    barrier.syncOnSameBlockIdx<false /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */>();
  }

  for (size_t idx = idxStart; idx < idxEnd; idx += idxStride) {
#pragma unroll NRANKS
    for (int r = 0; r < NRANKS; ++r) {
      int srcRank = r;
      size_t srcIdx = idx + selfRank * idxEnd;
      size_t destIdx = idx + r * idxEnd;
      *reinterpret_cast<uint4*>(&recvbuff[destIdx]) = reinterpret_cast<const uint4*>(&ipcbuffs[srcRank][srcIdx])[0];
    }
  }

  // barrier to ensure remote ranks won't free their buffers until I'm done
  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, false /* hasSubsequentMemAccess */>();
}

} // namespace dda::common
