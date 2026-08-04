/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * LL-protocol all-gather device kernel for the DDA path. Each 16-byte line holds
 * 8 bytes of payload and two 4-byte flags; the flags carry cross-rank sync, so
 * no GPU barrier is needed. Staging uses the DDA scratch (comm->ddaScratch,
 * reachable via comm->ddaPeerPtrsDev).
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

#include "algorithms/CollCommon.h"

namespace meta::comms {

// Per-rank scratch slot capacity and hard per-rank cap (enforced in the
// eligibility check), fixing the slot stride at compile time so the
// double-buffered layout is identical on every rank and call. The size gate is on
// the total gathered size (DDA_ALLGATHER_LL_THRESHOLD in collectives.cc), so the
// per-rank payload is <= total / nRanks and usually well under this cap. Footprint
// = 2 banks * nRanks * (kDdaLLAgMaxPerRankBytes * 2) for the 8B->16B expansion:
// 36 MiB at 128 KiB / 72 ranks, within the 64 MiB DDA scratch.
constexpr size_t kDdaLLAgMaxPerRankBytes = 131072;                     // 128 KiB
constexpr size_t kDdaLLAgSlotStridePkts  = kDdaLLAgMaxPerRankBytes / 8; // 16384

// LL scratch: 2 banks * nRanks slots * kDdaLLAgSlotStridePkts * 16B.
inline size_t ddaLLAgScratchSize(int nRanks) {
  return (size_t)2 * (size_t)nRanks * kDdaLLAgSlotStridePkts * sizeof(LLPacket16);
}

// LL all-gather kernel on a 2D grid: grid.x == nRanks selects the peer a column
// owns, grid.y == blocksPerPeer splits that peer's packets into contiguous chunks.
//
// The self column copies sendbuff -> recvbuff[self] locally; other columns scatter
// this rank's chunk into peer b's slot, then poll their own slot b for peer b's
// chunk, with threads splitting the packet range.
template <typename T, int NRANKS_CT>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
__global__ void ddaAllGatherFabricLL(
    T* const* __restrict__ peerScratch,    // ddaPeerPtrsDev: nRanks scratch bases
    T* __restrict__ recvbuff,              // local user output
    const T* __restrict__ sendbuff,        // local user input
    size_t perRankBytes,                   // per-rank payload; multiple of 16
    int selfRank,
    int nRanksRt,
    uint32_t* __restrict__ epochDev,       // per-block LL epoch cells
    int epochLen) {                        // number of cells in epochDev

  const int nRanks = NRANKS_CT ? NRANKS_CT : nRanksRt;
  const int peer = blockIdx.x;             // grid.x == nRanks: one column/peer
  const int chunk = blockIdx.y;            // grid.y == blocksPerPeer
  const int nChunks = gridDim.y;           // >= 1
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  const size_t nPk = perRankBytes >> 3;    // 8 payload bytes per packet
  const size_t slot = kDdaLLAgSlotStridePkts;

  const int flatBlockId = blockIdx.x * gridDim.y + blockIdx.y;
  const int total = gridDim.x * gridDim.y;
  uint32_t flag = epochDev[flatBlockId] + 1u;
  if (flag == 0u) flag = 2u;               // skip 0 sentinel; keep bank parity
  const size_t bankOffsetPkts = (size_t)(flag & 1u) * (size_t)nRanks * slot;

  // This block's packet range [pkBegin, pkEnd); [0, nPk) when nChunks == 1.
  const size_t pkPerChunk = (nPk + (size_t)nChunks - 1) / (size_t)nChunks;
  const size_t pkBegin = (size_t)chunk * pkPerChunk;
  size_t pkEnd = pkBegin + pkPerChunk;
  if (pkEnd > nPk) pkEnd = nPk;

  if (peer == selfRank) {
    // self column: local copy sendbuff -> recvbuff[self].
    const uint4* s4 = reinterpret_cast<const uint4*>(sendbuff);
    uint4* d4 = reinterpret_cast<uint4*>(
        reinterpret_cast<char*>(recvbuff) + (size_t)selfRank * perRankBytes);
    const size_t nVec = perRankBytes >> 4; // number of 16B chunks
    // split the copy across this peer's blocks too.
    const size_t vecPerChunk = (nVec + (size_t)nChunks - 1) / (size_t)nChunks;
    const size_t vBegin = (size_t)chunk * vecPerChunk;
    size_t vEnd = vBegin + vecPerChunk;
    if (vEnd > nVec) vEnd = nVec;
    for (size_t i = vBegin + tid; i < vEnd; i += nthreads) {
      const uint4* p = &s4[i];
      uint4 v;
      v.x = __builtin_nontemporal_load(&p->x);
      v.y = __builtin_nontemporal_load(&p->y);
      v.z = __builtin_nontemporal_load(&p->z);
      v.w = __builtin_nontemporal_load(&p->w);
      uint4* q = &d4[i];
      __builtin_nontemporal_store(v.x, &q->x);
      __builtin_nontemporal_store(v.y, &q->y);
      __builtin_nontemporal_store(v.z, &q->z);
      __builtin_nontemporal_store(v.w, &q->w);
    }
  } else {
    // scatter: write my payload into peer's slot (== selfRank).
    const uint32_t* sw = reinterpret_cast<const uint32_t*>(sendbuff);
    LLPacket16* dst = reinterpret_cast<LLPacket16*>(peerScratch[peer]) +
        (size_t)selfRank * slot + bankOffsetPkts;
    for (size_t pk = pkBegin + tid; pk < pkEnd; pk += nthreads) {
      ddaLLStoreLineB128(
          reinterpret_cast<uint32_t*>(&dst[pk]),
          sw[2 * pk], flag, sw[2 * pk + 1], flag);
    }

    // gather: poll my slot for peer, unpack into recvbuff[peer].
    volatile LLPacket16* src = reinterpret_cast<LLPacket16*>(peerScratch[selfRank]) +
        bankOffsetPkts + (size_t)peer * slot;
    uint32_t* out = reinterpret_cast<uint32_t*>(
        reinterpret_cast<char*>(recvbuff) + (size_t)peer * perRankBytes);
    for (size_t pk = pkBegin + tid; pk < pkEnd; pk += nthreads) {
      uint32_t d0, f0, d1, f1;
      do {
        ddaLLLoadLineB128(
            reinterpret_cast<const uint32_t*>(const_cast<LLPacket16*>(&src[pk])),
            d0, f0, d1, f1);
      } while (f0 != flag || f1 != flag);
      out[2 * pk] = d0;
      out[2 * pk + 1] = d1;
    }
  }
  __syncthreads();
  for (int e = flatBlockId + tid * total; e < epochLen; e += total * nthreads) {
    epochDev[e] = flag;
  }
}

} // namespace meta::comms
