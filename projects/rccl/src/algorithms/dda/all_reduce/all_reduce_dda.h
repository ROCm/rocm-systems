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

// Templated on a compile-time rank count NRANKS_CT (matching the fabric path):
//   - NRANKS_CT > 0  : specialized for that clique size; CollCommon's helpers
//                      fold nRanks to a constant and fully unroll the peer loop.
//   - NRANKS_CT == 0 : runtime fallback; the rank count comes from nRanks.
template <typename T, int NRANKS_CT, bool hasAcc>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
  __global__ void ddaAllReduceFlatIpc(T* const* __restrict__ ipcbuffs, T* __restrict__ recvbuff, size_t count,
                                      const T* __restrict__ sendbuff, int selfRank, int nRanks, IpcGpuBarrier barrier,
                                      const T* __restrict__ acc) {
  constexpr auto countPerThread = sizeof(uint4) / sizeof(T);
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  const auto idxStart = gtIdx * countPerThread;
  const auto idxEnd = count;
  const auto idxStride = gridDim.x * blockDim.x * countPerThread;

  copyFromSrcToDest<T>(sendbuff, ipcbuffs[selfRank], idxStart, idxEnd, idxStride);

  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */>();

  // pattern=2: full reduce into recvbuff (one-shot, not scatter)
  reduceScatter<T, NRANKS_CT, hasAcc>(ipcbuffs, recvbuff, acc, selfRank, nRanks, idxStart, idxEnd, idxStride, 2);

  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, false /* hasSubsequentMemAccess */>();
}

// See ddaAllReduceFlatIpc above for the NRANKS_CT / nRanks semantics.
template <typename T, int NRANKS_CT, bool hasAcc>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
  __global__ void ddaAllReduceTreeIpc(T* const* __restrict__ ipcbuffs, T* __restrict__ recvbuff, size_t count,
                                      const T* __restrict__ sendbuff, int selfRank, int nRanks, IpcGpuBarrier barrier,
                                      const T* __restrict__ acc) {
  barrier.syncOnSameBlockIdx<false /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */>();

  const size_t countPerRank = count / ((NRANKS_CT > 0) ? NRANKS_CT : nRanks);
  constexpr auto countPerThread = sizeof(uint4) / sizeof(T);
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  const auto idxStart = gtIdx * countPerThread;
  const auto idxEnd = countPerRank;
  const size_t idxStride = gridDim.x * blockDim.x * countPerThread;

  reduceScatter<T, NRANKS_CT, hasAcc>(ipcbuffs, ipcbuffs[selfRank], acc, selfRank, nRanks, idxStart, idxEnd, idxStride,
                                      1);

  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */>();

  allGather<T, NRANKS_CT>(ipcbuffs, recvbuff, selfRank, nRanks, idxStart, idxEnd, idxStride, true);

  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, false /* hasSubsequentMemAccess */>();
}

} // namespace dda::common
