/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * DDA reduce-scatter kernel for the fabric/VMM path, using FabricGpuBarrier.
 *
 * Templated on a compile-time rank count NRANKS_CT (matching the all-reduce
 * fabric design):
 *   - NRANKS_CT > 0  : specialized for that clique size; the unified CollCommon
 *                      reduceScatter (pattern 0, one-shot) fully unrolls the
 *                      peer loop.
 *   - NRANKS_CT == 0 : runtime fallback; the rank count is passed via nRanks and
 *                      the unified helper partially unrolls 8-wide, so a single
 *                      instantiation covers any other clique size up to
 *                      kDdaMaxNranks.
 *
 * The host launcher copies the full sendbuff into this rank's scratch buffer
 * before launch; the kernel reduces this rank's shard across all peers.
 * See LICENSE.txt for license information.
 ************************************************************************/

#pragma once

#include "algorithms/CollCommon.h"
#include "algorithms/CollCommonTdm.h"
#include "fabric_gpu_barrier.h"

namespace meta::comms {

template <typename T, int NRANKS_CT, bool hasAcc>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
  __global__ void ddaReduceScatterFabric(T* const* __restrict__ ipcbuffs, T* __restrict__ recvbuff, size_t count,
                                         const T* __restrict__ sendbuff, int selfRank, int nRanks,
                                         FabricGpuBarrier barrier, const T* __restrict__ acc) {

  barrier.syncOnSameBlockIdx<false /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */>();

  constexpr auto countPerThread = sizeof(uint4) / sizeof(T);
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  const auto idxStart = gtIdx * countPerThread;
  const auto idxEnd = count;
  const auto idxStride = gridDim.x * blockDim.x * countPerThread;

  reduceScatter<T, NRANKS_CT, hasAcc>(ipcbuffs, recvbuff, acc, selfRank, nRanks, idxStart, idxEnd, idxStride, 0);

  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, false /* hasSubsequentMemAccess */>();
}

// Tensor-data-mover variant: same phases and barrier placement as the kernel
// above. The host has already staged the full sendbuff into this rank's
// scratch, so the kernel only folds this rank's shard across the peers.
template <typename T, int NRANKS_CT, bool hasAcc>
__launch_bounds__(kTdmThreadsPerBlock) __global__
  void ddaReduceScatterFabricTdm(T* const* __restrict__ ipcbuffs, T* __restrict__ recvbuff, size_t count,
                                 const T* __restrict__ sendbuff, int selfRank, int nRanks, FabricGpuBarrier barrier,
                                 const T* __restrict__ acc) {
  __shared__ __align__(kTdmRowBytes) uint8_t lds[kTdmLdsBytes];
  const TdmWarpTile tile = tdmWarpTile();
  uint8_t* window0 = tdmWindow(lds, 0);
  uint8_t* window1 = tdmWindow(lds, 1);

  barrier.syncOnSameBlockIdx<false /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */>();

  uint8_t* const* peers = reinterpret_cast<uint8_t* const*>(ipcbuffs);
  const size_t shardBytes = count * sizeof(T);

  tdmReduceRange<T, NRANKS_CT, hasAcc>(peers, reinterpret_cast<uint8_t*>(recvbuff),
                                       reinterpret_cast<const uint8_t*>(acc), nRanks,
                                       /*srcOff=*/(size_t)selfRank * shardBytes, /*dstOff=*/0, shardBytes, window0,
                                       window1, tile);

  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, false /* hasSubsequentMemAccess */>();
}

} // namespace meta::comms
