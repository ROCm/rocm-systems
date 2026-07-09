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
#include "fabric_gpu_barrier.h"
#include "ll_fabric.h"

namespace meta::comms {

// DDA all-reduce, LL (low-latency) protocol, REMOTE-WRITE one-shot path.
//
// Each rank pushes its input as LL {data,flag} packets into every peer's
// per-source slot (remote writes), then polls its own nRanks slots locally.
// A packet is consumed once both flag halves equal the current epoch, so no
// cross-GPU barrier is needed. The reduction is folded on the consumer side as
// slots arrive, reusing vecElementAdd<T> (handles fp32/fp16/bf16 lanes per 4B
// data word). recvbuff receives the full reduced result (every rank computes
// the whole reduction: one-shot / flat).
//
// Buffer layout (per rank, in the dedicated LL recv buffer):
//   [ bank 0: nRanks slots ][ bank 1: nRanks slots ]
// each slot is slotStridePkts packets; the active bank is bankOffset packets in.
template <typename T, int NRANKS_CT>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
__global__ void ddaAllReduceFlatFabricLL(
    LLPacket16* const* __restrict__ peerLLbufs,
    T* __restrict__ recvbuff,
    size_t count,
    const T* __restrict__ sendbuff,
    int selfRank,
    int nRanks,
    size_t numPackets,
    size_t slotStridePkts,
    size_t bankStridePkts,
    uint32_t flagVal) {
  const int nR = (NRANKS_CT > 0) ? NRANKS_CT : nRanks;
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;
  const auto stride = gridDim.x * blockDim.x;

  // The host-incremented epoch is the packet flag and selects the double-buffer
  // bank, so a lagging peer's read of op N doesn't collide with op N+1.
  const size_t bankOffset =
      static_cast<size_t>((flagVal - 1) & 1) * bankStridePkts;
  const uint32_t* __restrict__ sendU32 =
      reinterpret_cast<const uint32_t*>(sendbuff);
  uint32_t* __restrict__ recvU32 = reinterpret_cast<uint32_t*>(recvbuff);

  // 1. PUSH: pack my payload into every peer's slot for my rank (remote write).
#pragma unroll(NRANKS_CT > 0 ? NRANKS_CT : 1)
  for (int p = 0; p < nR; ++p) {
    if (p == selfRank) {
      continue;
    }
    LLPacket16* __restrict__ dst =
        peerLLbufs[p] + static_cast<size_t>(selfRank) * slotStridePkts +
        bankOffset;
    for (size_t pkt = gtIdx; pkt < numPackets; pkt += stride) {
      const uint32_t* s = sendU32 + pkt * 2;
      llStoreLine(
          reinterpret_cast<uint32_t*>(&dst[pkt]), s[0], flagVal, s[1], flagVal);
    }
  }

  // 2. POLL + REDUCE: local reads of my nRanks slots, fused sum -> recvbuff.
  LLPacket16* __restrict__ myBuf = peerLLbufs[selfRank] + bankOffset;
  for (size_t pkt = gtIdx; pkt < numPackets; pkt += stride) {
    // Seed with my own contribution (already local; no packetization).
    uint32_t acc0 = sendU32[pkt * 2 + 0];
    uint32_t acc1 = sendU32[pkt * 2 + 1];
#pragma unroll(NRANKS_CT > 0 ? NRANKS_CT : 1)
    for (int p = 0; p < nR; ++p) {
      if (p == selfRank) {
        continue;
      }
      volatile LLPacket16* slot =
          myBuf + static_cast<size_t>(p) * slotStridePkts;
      uint32_t d0, f0, d1, f1;
      do {
        llLoadLine(
            reinterpret_cast<const uint32_t*>(
                const_cast<LLPacket16*>(&slot[pkt])),
            d0, f0, d1, f1);
      } while (f0 != flagVal || f1 != flagVal);
      acc0 = vecElementAdd<T>(acc0, d0);
      acc1 = vecElementAdd<T>(acc1, d1);
    }
    recvU32[pkt * 2 + 0] = acc0;
    recvU32[pkt * 2 + 1] = acc1;
  }
}

template <typename T, int NRANKS_CT, bool hasAcc>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
__global__ void ddaAllReduceFlatFabric(
    T* const* __restrict__ ipcbuffs,
    T* __restrict__ recvbuff,
    size_t count,
    const T* __restrict__ sendbuff,
    int selfRank,
    int nRanks,
    FabricGpuBarrier barrier,
    const T* __restrict__ acc) {
  constexpr auto countPerThread = sizeof(uint4) / sizeof(T);
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  const auto idxStart = gtIdx * countPerThread;
  const auto idxEnd = count;
  const auto idxStride = gridDim.x * blockDim.x * countPerThread;

  copyFromSrcToDest<T>(
      sendbuff, ipcbuffs[selfRank], idxStart, idxEnd, idxStride);

  barrier.syncOnSameBlockIdx<
      true /* hasPreviousMemAccess */,
      true /* hasSubsequentMemAccess */>();

  // pattern=2: full reduce into recvbuff (one-shot). The unified helper folds
  // nRanks to NRANKS_CT (full unroll) when specialized, else uses the runtime
  // nRanks with an 8-wide partial unroll.
  reduceScatter<T, NRANKS_CT, hasAcc>(
      ipcbuffs, recvbuff, acc, selfRank, nRanks, idxStart, idxEnd, idxStride, 2);

  barrier.syncOnSameBlockIdx<
      true /* hasPreviousMemAccess */,
      false /* hasSubsequentMemAccess */>();
}

template <typename T, int NRANKS_CT, bool hasAcc>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
__global__ void ddaAllReduceTreeFabric(
    T* const* __restrict__ ipcbuffs,
    T* __restrict__ recvbuff,
    size_t count,
    const T* __restrict__ sendbuff,
    int selfRank,
    int nRanks,
    FabricGpuBarrier barrier,
    const T* __restrict__ acc) {
  barrier.syncOnSameBlockIdx<
      false /* hasPreviousMemAccess */,
      true /* hasSubsequentMemAccess */>();

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
  reduceScatter<T, NRANKS_CT, hasAcc>(
      ipcbuffs, ipcbuffs[selfRank], acc, selfRank, nRanks, idxStart, idxEnd,
      idxStride, 1);

  barrier.syncOnSameBlockIdx<
      true /* hasPreviousMemAccess */,
      true /* hasSubsequentMemAccess */>();

  allGather<T, NRANKS_CT>(
      ipcbuffs, recvbuff, selfRank, nRanks, idxStart, idxEnd, idxStride, true);

  barrier.syncOnSameBlockIdx<
      true /* hasPreviousMemAccess */,
      false /* hasSubsequentMemAccess */>();
}

} // namespace meta::comms
