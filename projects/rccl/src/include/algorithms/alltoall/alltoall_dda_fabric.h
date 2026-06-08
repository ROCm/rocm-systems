/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * DDA alltoall kernel for the fabric/VMM path, using FabricGpuBarrier.
 *
 * Templated on a compile-time rank count NRANKS_CT (matching the all-reduce
 * fabric design):
 *   - NRANKS_CT > 0  : specialized for that clique size (unrolled peer loop).
 *   - NRANKS_CT == 0 : runtime fallback driven by the nRanks argument, so a
 *                      single instantiation covers any other clique size up to
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
__global__ void ddaAllToAllFabric(
    T* const* __restrict__ ipcbuffs,
    T* __restrict__ recvbuff,
    size_t count,
    const T* __restrict__ sendbuff,
    int selfRank,
    int nRanks,
    FabricGpuBarrier barrier) {
  // use uint4 to do 16-byte loads to maximize memory efficiency. We assume
  // that count % countPerThread == 0, enforced before kernel launch.
  const int nRanksEff = (NRANKS_CT > 0) ? NRANKS_CT : nRanks;
  // Fully unroll when the clique size is a compile-time constant; otherwise
  // partially unroll 8-wide for the runtime fallback (Option A).
  constexpr int kUnroll = (NRANKS_CT > 0) ? NRANKS_CT : 8;
  const size_t countPerRank = count;
  constexpr auto countPerThread = sizeof(uint4) / sizeof(T);
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  const auto idxStart = gtIdx * countPerThread;
  const auto idxEnd = countPerRank;
  const size_t copyCount = count * nRanksEff;
  const auto idxStride = gridDim.x * blockDim.x * countPerThread;

  // It is expensive to launch hipMemcpyAsync on ROCm: each block copies part
  // of sendbuff into this rank's scratch buffer.
  copyFromSrcToDest<T>(
      sendbuff, ipcbuffs[selfRank], idxStart, copyCount, idxStride);

  barrier.syncOnSameBlockIdx<
      true /* hasPreviousMemAccess */,
      true /* hasSubsequentMemAccess */>();

  for (size_t idx = idxStart; idx < idxEnd; idx += idxStride) {
#pragma unroll kUnroll
    for (int r = 0; r < nRanksEff; ++r) {
      int srcRank = r;
      int srcIdx = idx + selfRank * idxEnd;
      int destIdx = idx + r * idxEnd;
      *reinterpret_cast<uint4*>(&recvbuff[destIdx]) =
          reinterpret_cast<const uint4*>(&ipcbuffs[srcRank][srcIdx])[0];
    }
  }

  // barrier to ensure remote ranks won't free their buffers until I'm done
  barrier.syncOnSameBlockIdx<
      true /* hasPreviousMemAccess */,
      false /* hasSubsequentMemAccess */>();
}

} // namespace meta::comms
