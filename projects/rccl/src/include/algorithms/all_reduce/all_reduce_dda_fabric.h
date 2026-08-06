/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * DDA all-reduce kernels for the fabric/VMM path, using FabricGpuBarrier.
 *
 * The kernels are templated on a compile-time rank count NRANKS_CT:
 *   - NRANKS_CT > 0  : specialized for that clique size; the unified CollCommon
 *                      reduceScatter/allGather fully unroll the peer loop
 *                      (matching the IPC fast path). The host launcher
 *                      instantiates this for the common sizes (e.g. 4, 8).
 *   - NRANKS_CT == 0 : runtime fallback; the rank count is passed via the nRanks
 *                      argument and the unified helpers partially unroll 8-wide,
 *                      so a single instantiation covers any other clique size
 *                      up to kDdaMaxNranks.
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
  __global__ void ddaAllReduceFlatFabric(T* const* __restrict__ ipcbuffs, T* __restrict__ recvbuff, size_t count,
                                         const T* __restrict__ sendbuff, int selfRank, int nRanks,
                                         FabricGpuBarrier barrier, const T* __restrict__ acc) {
  constexpr auto countPerThread = sizeof(uint4) / sizeof(T);
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  const auto idxStart = gtIdx * countPerThread;
  const auto idxEnd = count;
  const auto idxStride = gridDim.x * blockDim.x * countPerThread;

  copyFromSrcToDest<T>(sendbuff, ipcbuffs[selfRank], idxStart, idxEnd, idxStride);

  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */>();

  // pattern=2: full reduce into recvbuff (one-shot). The unified helper folds
  // nRanks to NRANKS_CT (full unroll) when specialized, else uses the runtime
  // nRanks with an 8-wide partial unroll.
  reduceScatter<T, NRANKS_CT, hasAcc>(ipcbuffs, recvbuff, acc, selfRank, nRanks, idxStart, idxEnd, idxStride, 2);

  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, false /* hasSubsequentMemAccess */>();
}

template <typename T, int NRANKS_CT, bool hasAcc>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
  __global__ void ddaAllReduceTreeFabric(T* const* __restrict__ ipcbuffs, T* __restrict__ recvbuff, size_t count,
                                         const T* __restrict__ sendbuff, int selfRank, int nRanks,
                                         FabricGpuBarrier barrier, const T* __restrict__ acc) {
  barrier.syncOnSameBlockIdx<false /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */>();

  // Use the compile-time rank count as the divisor when specialized.
  const int nRanksEff = (NRANKS_CT > 0) ? NRANKS_CT : nRanks;
  const size_t countPerRank = count / nRanksEff;
  constexpr auto countPerThread = sizeof(uint4) / sizeof(T);
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  const auto idxStart = gtIdx * countPerThread;
  const auto idxEnd = countPerRank;
  const size_t idxStride = gridDim.x * blockDim.x * countPerThread;

  // Two-shot: reduce-scatter this rank's shard, then all-gather. The unified
  // helpers fold nRanks to NRANKS_CT (full unroll) when specialized, else use
  // the runtime nRanks with an 8-wide partial unroll.
  reduceScatter<T, NRANKS_CT, hasAcc>(ipcbuffs, ipcbuffs[selfRank], acc, selfRank, nRanks, idxStart, idxEnd, idxStride,
                                      1);

  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */>();

  allGather<T, NRANKS_CT>(ipcbuffs, recvbuff, selfRank, nRanks, idxStart, idxEnd, idxStride, true);

  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, false /* hasSubsequentMemAccess */>();
}

// ---------------------------------------------------------------------------
// Tensor-data-mover variants
//
// Same phases and the same barrier placement as the two kernels above; only the
// bulk movement differs, so the two families are interchangeable at launch. The
// grid shape differs (see getTdmGridAndBlockDims), which is fine because every
// rank picks the same one.
// ---------------------------------------------------------------------------

template <typename T, int NRANKS_CT, bool hasAcc>
__launch_bounds__(kTdmThreadsPerBlock) __global__
  void ddaAllReduceFlatFabricTdm(T* const* __restrict__ ipcbuffs, T* __restrict__ recvbuff, size_t count,
                                 const T* __restrict__ sendbuff, int selfRank, int nRanks, FabricGpuBarrier barrier,
                                 const T* __restrict__ acc) {
  __shared__ __align__(kTdmRowBytes) uint8_t lds[kTdmLdsBytes];
  const TdmWarpTile tile = tdmWarpTile();
  uint8_t* window0 = tdmWindow(lds, 0);
  uint8_t* window1 = tdmWindow(lds, 1);

  uint8_t* const* peers = reinterpret_cast<uint8_t* const*>(ipcbuffs);
  const size_t nbytes = count * sizeof(T);

  tdmCopyRange(reinterpret_cast<const uint8_t*>(sendbuff), peers[selfRank], nbytes, window0, window1, tile);

  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */>();

  tdmReduceRange<T, NRANKS_CT, hasAcc>(peers, reinterpret_cast<uint8_t*>(recvbuff),
                                       reinterpret_cast<const uint8_t*>(acc), nRanks, /*srcOff=*/0, /*dstOff=*/0,
                                       nbytes, window0, window1, tile);

  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, false /* hasSubsequentMemAccess */>();
}

template <typename T, int NRANKS_CT, bool hasAcc>
__launch_bounds__(kTdmThreadsPerBlock) __global__
  void ddaAllReduceTreeFabricTdm(T* const* __restrict__ ipcbuffs, T* __restrict__ recvbuff, size_t count,
                                 const T* __restrict__ sendbuff, int selfRank, int nRanks, FabricGpuBarrier barrier,
                                 const T* __restrict__ acc) {
  __shared__ __align__(kTdmRowBytes) uint8_t lds[kTdmLdsBytes];
  const TdmWarpTile tile = tdmWarpTile();
  uint8_t* window0 = tdmWindow(lds, 0);
  uint8_t* window1 = tdmWindow(lds, 1);

  barrier.syncOnSameBlockIdx<false /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */>();

  uint8_t* const* peers = reinterpret_cast<uint8_t* const*>(ipcbuffs);
  const int nRanksEff = (NRANKS_CT > 0) ? NRANKS_CT : nRanks;
  const size_t shardBytes = (count / nRanksEff) * sizeof(T);
  const size_t shardOff = (size_t)selfRank * shardBytes;

  // Two-shot: reduce this rank's shard across all peers into its own scratch
  // slot, then gather every rank's reduced shard.
  tdmReduceRange<T, NRANKS_CT, hasAcc>(peers, peers[selfRank], reinterpret_cast<const uint8_t*>(acc), nRanks, shardOff,
                                       shardOff, shardBytes, window0, window1, tile);

  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */>();

  tdmAllGather<NRANKS_CT>(peers, reinterpret_cast<uint8_t*>(recvbuff), selfRank, nRanks, shardBytes,
                          /*enableOffset=*/true, window0, window1, tile);

  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, false /* hasSubsequentMemAccess */>();
}

} // namespace meta::comms
