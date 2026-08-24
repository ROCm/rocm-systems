/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * DDA all-gather kernel for the fabric/VMM path, using FabricGpuBarrier.
 *
 * Templated on a compile-time rank count NRANKS_CT (matching the all-reduce
 * fabric design):
 *   - NRANKS_CT > 0  : specialized for that clique size; the unified CollCommon
 *                      allGather fully unrolls the peer loop.
 *   - NRANKS_CT == 0 : runtime fallback; the rank count is passed via nRanks and
 *                      the unified helper partially unrolls 8-wide, so a single
 *                      instantiation covers any other clique size up to
 *                      kDdaMaxNranks.
 * See LICENSE.txt for license information.
 ************************************************************************/

#pragma once

#include "algorithms/CollCommon.h"
#include "fabric_gpu_barrier.h"

namespace meta::comms {

template <typename T, int NRANKS_CT>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
  __global__
  void ddaAllGatherFabric(T* const* __restrict__ ipcbuffs, T* __restrict__ recvbuff, size_t count,
                          const T* __restrict__ sendbuff, int selfRank, int nRanks, FabricGpuBarrier barrier) {

  const size_t countPerRank = count;
  constexpr auto countPerThread = sizeof(uint4) / sizeof(T);
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  const auto idxStart = gtIdx * countPerThread;
  const auto idxEnd = countPerRank;
  const auto idxStride = gridDim.x * blockDim.x * countPerThread;

  // It is expensive to launch hipMemcpyAsync on ROCm: each block copies part
  // of sendbuff into this rank's scratch buffer.
  copyFromSrcToDest<T>(sendbuff, ipcbuffs[selfRank], idxStart, idxEnd, idxStride);

  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */>();

  allGather<T, NRANKS_CT>(ipcbuffs, recvbuff, selfRank, nRanks, idxStart, idxEnd, idxStride, false);

  // barrier to ensure remote ranks won't free their buffers until I'm done
  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, false /* hasSubsequentMemAccess */>();
}

} // namespace meta::comms
