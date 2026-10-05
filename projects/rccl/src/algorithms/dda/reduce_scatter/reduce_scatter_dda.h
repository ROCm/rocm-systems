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

template <typename T, int NRANKS, bool hasAcc, bool kStagingCopyInKernel = false>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
  __global__ void ddaReduceScatterIpc(T* const* __restrict__ ipcbuffs, T* __restrict__ recvbuff, size_t count,
                                      const T* __restrict__ sendbuff, int selfRank, IpcGpuBarrier barrier) {
  constexpr auto countPerThread = sizeof(uint4) / sizeof(T);
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  const auto idxStart = gtIdx * countPerThread;
  const auto idxEnd = count;
  const auto idxStride = gridDim.x * blockDim.x * countPerThread;

  if constexpr (kStagingCopyInKernel) {
    // Small messages: fuse sendbuff -> scratch copy into the kernel to avoid
    // cudaMemcpyAsync launch overhead on ROCm.
    const size_t copyCount = count * NRANKS;
    copyFromSrcToDest<T>(sendbuff, ipcbuffs[selfRank], idxStart, copyCount, idxStride);
    barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */>();
  } else {
    if (count * sizeof(T) <= 4194304) {
#pragma unroll NRANKS
      for (int s = 0; s < NRANKS; ++s) {
        const size_t off = static_cast<size_t>(s) * count;
        copyFromSrcToDest<T>(sendbuff + off, ipcbuffs[selfRank] + off, idxStart, idxEnd, idxStride);
      }
      barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */>();	
    } else {
      // Large messages: host enqueues cudaMemcpyAsync into ddaScratch before launch.
      barrier.syncOnSameBlockIdx<false /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */>();
    }
  }

  reduceScatter<T, NRANKS, hasAcc>(ipcbuffs, recvbuff, nullptr, selfRank, NRANKS, idxStart, idxEnd, idxStride, 0);

  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, false /* hasSubsequentMemAccess */>();
}

} // namespace dda::common
