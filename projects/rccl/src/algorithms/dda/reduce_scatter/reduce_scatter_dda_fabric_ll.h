/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * LL-protocol reduce-scatter device kernel for the DDA fabric path. Each 16-byte
 * line holds 8 bytes of payload and two 4-byte flags; the flags carry cross-rank
 * sync, so no GPU barrier is needed. Staging uses the DDA scratch
 * (comm->ddaScratch, reachable via comm->ddaPeerPtrsDev).
 *
 * Reduce-scatter combines the personalized scatter of all-to-all with the fold
 * of all-reduce: rank selfRank owns output shard selfRank, so recvbuff[i] =
 * sum over ranks r of sendbuff_r[selfRank*recvcount + i]. Phase 1 scatters this
 * rank's per-peer chunk (sendbuff[peer]) into peer's slot; phase 2 folds every
 * incoming chunk (plus this rank's own self-chunk) into recvbuff. Only the
 * per-peer publish source and the reduce seed differ from all_reduce_dda_ll.h.
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

#include "algorithms/dda/device/CollCommon.h"

namespace dda::common {

// Per-shard staging slot capacity and hard per-shard cap (enforced in the
// eligibility check). Fixes the slot stride at compile time so the
// double-buffered layout is identical on every rank and call.
// Footprint = 2 banks * nRanks * (kDdaLLRsMaxBytes * 2) for the 8B->16B
// expansion; 16 MiB at 1 MiB/rank * 4 ranks * 2 banks * 2 (LL overhead),
// within the 10 GiB DDA fabric scratch on gfx1250.
constexpr size_t kDdaLLRsMaxBytes = 1ULL*1024*1024;         // 1 MiB per-rank (= 4 MiB total at 4 ranks)
constexpr size_t kDdaLLRsSlotStridePkts = kDdaLLRsMaxBytes / 8;   // 131072

// LL reduce-scatter kernel. 1D grid over packets (8B payload each) of the
// per-rank shard (recvcount elements).
//
// Phase 1 (scatter): rank selfRank writes its chunk-for-peer (sendbuff[peer])
// into peer's scratch at slot selfRank, as LL lines carrying the epoch flag.
// Phase 2 (reduce): rank selfRank seeds the accumulator with its own self-chunk
// (sendbuff[selfRank]), polls its own scratch slot for each other rank (waiting
// on the flag), and sums those into recvbuff. Flag polling provides cross-rank
// ordering, so no GPU barrier is used. Scratch is double-buffered: bank = flag & 1.
template <typename T, int NRANKS_CT>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
  __global__ void ddaReduceScatterFabricLL(T* const* __restrict__ peerScratch, // ddaPeerPtrsDev: nRanks scratch bases
                                           T* __restrict__ recvbuff, // local user output (recvcount elems)
                                           const T* __restrict__ sendbuff, // local user input (recvcount*nRanks)
                                           size_t recvcount, // per-rank shard element count
                                           int selfRank, int nRanksRt,
                                           uint32_t* __restrict__ epochDev, // per-block LL epoch cells
                                           int epochLen) { // number of cells in epochDev

  const int nRanks = NRANKS_CT ? NRANKS_CT : nRanksRt;
  const size_t bytes = recvcount * sizeof(T);
  const size_t nPk = bytes >> 3; // 8 payload bytes per packet
  const size_t slot = kDdaLLRsSlotStridePkts;
  const size_t chunkWords = nPk * 2; // uint32 words per shard chunk

  const uint32_t flag = ddaGetLLEpochInc(epochDev, blockIdx.x, 1);
  const size_t bankOffsetPkts = (size_t)(flag & 1u) * (size_t)nRanks * slot;

  const size_t gtid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = (size_t)gridDim.x * blockDim.x;

  const uint32_t* in = reinterpret_cast<const uint32_t*>(sendbuff);
  uint32_t* out = reinterpret_cast<uint32_t*>(recvbuff);

  // Phase 1: scatter my chunk-for-peer (sendbuff[peer]) into peer's slot[self].
  for (size_t pk = gtid; pk < nPk; pk += stride) {
#pragma unroll
    for (int r = 1; r < nRanks; ++r) {
      const int peer = (selfRank + r) % nRanks;
      const uint32_t* inPeer = in + (size_t)peer * chunkWords;
      const uint32_t d0 = inPeer[2 * pk];
      const uint32_t d1 = inPeer[2 * pk + 1];
      LLPacket16* dst = reinterpret_cast<LLPacket16*>(peerScratch[peer]) + bankOffsetPkts + (size_t)selfRank * slot;
      ddaLLStoreLineB128(reinterpret_cast<uint32_t*>(&dst[pk]), d0, flag, d1, flag);
    }
  }

  // Phase 2: seed with my self-chunk, poll my slots for the others, reduce.
  const uint32_t* inSelf = in + (size_t)selfRank * chunkWords;
  LLPacket16* myBase = reinterpret_cast<LLPacket16*>(peerScratch[selfRank]) + bankOffsetPkts;
  for (size_t pk = gtid; pk < nPk; pk += stride) {
    uint32_t acc0 = inSelf[2 * pk];
    uint32_t acc1 = inSelf[2 * pk + 1];
    for (int r = 1; r < nRanks; ++r) {
      const int peer = (selfRank + r) % nRanks;
      volatile LLPacket16* src = myBase + (size_t)peer * slot;
      uint32_t d0, f0, d1, f1;
      do {
        ddaLLLoadLineB128(reinterpret_cast<const uint32_t*>(const_cast<LLPacket16*>(&src[pk])), d0, f0, d1, f1);
      } while (f0 != flag || f1 != flag);
      acc0 = vecElementAdd<T>(acc0, d0);
      acc1 = vecElementAdd<T>(acc1, d1);
    }
    out[2 * pk] = acc0;
    out[2 * pk + 1] = acc1;
  }

  ddaSetLLEpoch(epochDev, epochLen, blockIdx.x, gridDim.x, flag);
}

} // namespace dda::common
