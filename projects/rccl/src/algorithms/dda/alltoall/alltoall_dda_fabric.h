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

#include "algorithms/dda/device/CollCommon.h"
#include "algorithms/dda/fabric/fabric_gpu_barrier.h"

namespace dda::common {

template <typename T, int NRANKS_CT>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
  __global__ void ddaAllToAllFabric(T* const* __restrict__ ipcbuffs, T* __restrict__ recvbuff, size_t count,
                                    int selfRank, int nRanks, FabricGpuBarrier barrier) {
  // Release-acquire barrier ensures the stream-ordered scratch write is visible to peers.
  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */>();

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
  const auto idxStride = gridDim.x * blockDim.x * countPerThread;

  for (size_t idx = idxStart; idx < idxEnd; idx += idxStride) {
#pragma unroll kUnroll
    for (int r = 0; r < nRanksEff; ++r) {
      int srcRank = r;
      size_t srcIdx = idx + selfRank * idxEnd;
      size_t destIdx = idx + r * idxEnd;
      *reinterpret_cast<uint4*>(&recvbuff[destIdx]) = reinterpret_cast<const uint4*>(&ipcbuffs[srcRank][srcIdx])[0];
    }
  }

  // barrier to ensure remote ranks won't free their buffers until I'm done
  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, false /* hasSubsequentMemAccess */>();
}

// Push alltoall: each rank writes its send chunk for peer r into peer r's scratch at this rank's slot,
// then drains its own scratch into recvbuff. The per-block barrier pairs those writes with the drain, so
// every rank must launch the same grid. Scratch for peer r, slot s, holds the chunk rank s sends to r.
template <typename T, int NRANKS_CT>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
  __global__ void ddaAllToAllFabricPush(T* const* __restrict__ ipcbuffs, T* __restrict__ recvbuff, size_t count,
                                        const T* __restrict__ sendbuff, int selfRank, int nRanks,
                                        FabricGpuBarrier barrier) {
  // The previous collective's readers must be done with every scratch slot this block is about to overwrite.
  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */>();

  const int nRanksEff = (NRANKS_CT > 0) ? NRANKS_CT : nRanks;
  constexpr int kUnroll = (NRANKS_CT > 0) ? NRANKS_CT : 8;
  const size_t countPerRank = count;
  constexpr auto countPerThread = sizeof(uint4) / sizeof(T);
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  const auto idxStart = gtIdx * countPerThread;
  const auto idxEnd = countPerRank;
  const auto idxStride = gridDim.x * blockDim.x * countPerThread;

  // sendbuff[r] is the chunk destined for rank r. Load a whole peer group into registers before issuing its
  // remote stores, so those loads can be in flight together. Block i on every rank owns the same idx set.
  const size_t destIdx = selfRank * idxEnd;
  for (size_t idx = idxStart; idx < idxEnd; idx += idxStride) {
    int r0 = 0;
    for (; r0 + kUnroll <= nRanksEff; r0 += kUnroll) {
      uint4 v[kUnroll];
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) {
        v[u] = reinterpret_cast<const uint4*>(&sendbuff[idx + (r0 + u) * idxEnd])[0];
      }
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) {
        *reinterpret_cast<uint4*>(&ipcbuffs[r0 + u][destIdx + idx]) = v[u];
      }
    }
    for (int r = r0; r < nRanksEff; ++r) {
      *reinterpret_cast<uint4*>(&ipcbuffs[r][destIdx + idx]) =
        reinterpret_cast<const uint4*>(&sendbuff[idx + r * idxEnd])[0];
    }
  }

  // Publish the remote writes before any rank reads its scratch.
  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */>();

  T* selfScratch = ipcbuffs[selfRank];
  for (size_t idx = idxStart; idx < idxEnd; idx += idxStride) {
#pragma unroll kUnroll
    for (int r = 0; r < nRanksEff; ++r) {
      size_t slotIdx = idx + r * idxEnd;
      *reinterpret_cast<uint4*>(&recvbuff[slotIdx]) = reinterpret_cast<const uint4*>(&selfScratch[slotIdx])[0];
    }
  }
}

} // namespace dda::common
