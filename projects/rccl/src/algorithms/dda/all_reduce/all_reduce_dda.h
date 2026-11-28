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

// kInPlace: sendbuff == recvbuff. The kernel then reads its input through recvbuff and never dereferences sendbuff, so
// every access to the user buffer is based on recvbuff and __restrict__ on both pointers holds.
template <typename T, int NRANKS, bool hasAcc, bool kInPlace = false>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
  __global__ void ddaAllReduceFlatIpc(T* const* __restrict__ ipcbuffs, T* __restrict__ recvbuff, size_t count,
                                      const T* __restrict__ sendbuff, int selfRank, const IpcGpuBarrier barrier,
                                      const T* __restrict__ acc) {
  bool barrierAborted = false; // sticky abort state, see IpcGpuBarrier::syncOnSameBlockIdx
  constexpr auto countPerThread = sizeof(uint4) / sizeof(T);
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  const auto idxStart = gtIdx * countPerThread;
  const auto idxEnd = count;
  const auto idxStride = gridDim.x * blockDim.x * countPerThread;

  const T* const src = kInPlace ? recvbuff : sendbuff;
  copyFromSrcToDest<T>(src, ipcbuffs[selfRank], idxStart, idxEnd, idxStride);

  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */>(barrierAborted);

  // pattern=2: full reduce into recvbuff (one-shot, not scatter)
  reduceScatter<T, NRANKS, hasAcc>(ipcbuffs, recvbuff, acc, selfRank, NRANKS, idxStart, idxEnd, idxStride, 2);

  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, false /* hasSubsequentMemAccess */>(barrierAborted);
}

// kCopyInKernel: the kernel copies sendbuff into its scratch first.
// Otherwise the host copied it before the launch and barrier 1 only acquires.
// kInPlace as in ddaAllReduceFlatIpc. Without kCopyInKernel the kernel never reads sendbuff and needs no kInPlace form.
template <typename T, int NRANKS, bool hasAcc, bool kCopyInKernel = false, bool kInPlace = false>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
  __global__ void ddaAllReduceTreeIpc(T* const* __restrict__ ipcbuffs, T* __restrict__ recvbuff, size_t count,
                                      const T* __restrict__ sendbuff, int selfRank, const IpcGpuBarrier barrier,
                                      const T* __restrict__ acc) {
  static_assert(kCopyInKernel || !kInPlace, "only the kernel-copy tree reads sendbuff");
  bool barrierAborted = false; // sticky abort state, see IpcGpuBarrier::syncOnSameBlockIdx
  const size_t countPerRank = count / NRANKS;
  constexpr auto countPerThread = sizeof(uint4) / sizeof(T);
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  const auto idxStart = gtIdx * countPerThread;
  const auto idxEnd = countPerRank;
  const size_t idxStride = gridDim.x * blockDim.x * countPerThread;

  if constexpr (kCopyInKernel) {
    const T* const src = kInPlace ? recvbuff : sendbuff;
    // Every rank launches the same grid, so block b copies exactly the slots block b of every rank reads below.
    for (size_t idx = idxStart; idx < idxEnd; idx += idxStride) {
#pragma unroll NRANKS
      for (int s = 0; s < NRANKS; ++s) {
        const size_t slotIdx = idx + static_cast<size_t>(s) * countPerRank;
        *reinterpret_cast<uint4*>(&ipcbuffs[selfRank][slotIdx]) = reinterpret_cast<const uint4*>(&src[slotIdx])[0];
      }
    }
    barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */>(barrierAborted);
  } else {
    barrier.syncOnSameBlockIdx<false /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */>(barrierAborted);
  }

  reduceScatter<T, NRANKS, hasAcc>(ipcbuffs, ipcbuffs[selfRank], acc, selfRank, NRANKS, idxStart, idxEnd, idxStride, 1);

  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */>(barrierAborted);

  allGather<T, NRANKS>(ipcbuffs, recvbuff, selfRank, NRANKS, idxStart, idxEnd, idxStride, true);

  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, false /* hasSubsequentMemAccess */>(barrierAborted);
}

} // namespace dda::common
