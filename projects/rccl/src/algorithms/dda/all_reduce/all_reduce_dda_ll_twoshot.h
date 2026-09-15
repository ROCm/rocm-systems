/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Two-shot tier of the LL-protocol all-reduce for the DDA path.
 *
 * Each rank owns one shard of count/nRanks elements and only ever transports
 * that shard, so every rank moves nRanks-1 shards instead of nRanks-1 whole
 * messages the way the one-shot tier does. Three phases, all flag-ordered like
 * the one-shot, so still no GPU barrier:
 *
 *   1. publish: send each peer the shard that peer owns,
 *   2. reduce:  sum the copies of my own shard that arrived, write it to my part
 *               of recvbuff, and publish the reduced shard back to every peer,
 *   3. writeback: collect the peers' reduced shards into the rest of recvbuff.
 *
 * It shares comm->ddaScratch and the LL epoch counter with the one-shot tier
 * (see the constants below), so the two stay interchangeable on the same comm.
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
// Staging layout is shared with the one-shot tier, not redefined here.
#include "algorithms/dda/all_reduce/all_reduce_dda_ll.h"

namespace dda::common {

// Both tiers share one scratch and epoch, so bank 1 has to start at the same
// offset for both: a slot here holds only a shard, and the bank spans two of them
// (publish + write-back). dda_all_reduce_fabric_ll.cu static_asserts the halving.
constexpr size_t kDdaLLArTwoShotSlotStridePkts = kDdaLLArSlotStridePkts / 2;

// Fixed-width peer staging for phase 1: a runtime-sized array, or a runtime index
// into one, is placed in scratch. 8 covers rank counts 4 and 8 in a single pass.
// TODO: should this be moved to common place so other kernels can do the same
constexpr int kDdaLLArTwoShotPeerBatch = 8;

// LL two-shot all-reduce kernel. 1D grid over one shard's packets (8B payload
// each), so the launcher sizes the grid on count/nRanks, not on count.
//
// Phase 1 (publish): send peer p the shard p owns, into p's scratch at slot
// selfRank, as LL lines carrying the epoch flag.
// Phase 2 (reduce and publish): poll my slot for each peer's copy of my shard,
// sum with my own into my part of recvbuff, then publish the result to every peer.
// Phase 3 (gather): poll for the reduced shards the peers published and write each
// into that peer's part of recvbuff.
//
// Flag polling provides all the cross-rank ordering, so no GPU barrier is used.
// Scratch is double-buffered: bank = epoch & 1, selected via bankOffsetPkts, with
// bankOffsetPkts_next naming the second stage inside that bank. Self does not
// round-trip through scratch; its contribution is read directly from sendbuff in
// the reduce.
//
// Threads own disjoint packet indices in every phase, so an in-place call (where
// sendbuff aliases recvbuff) never has one thread's phase-3 store land on a
// packet another thread still has to read in phase 1.
template <typename T, int NRANKS_CT>
#if defined(USE_ROCM)
__launch_bounds__(1024)
#endif
  __global__ void ddaAllReduceTwoShotLL(T* const* __restrict__ peerScratch, // ddaPeerPtrsDev: nRanks scratch bases
                                        T* __restrict__ recvbuff,           // local user output
                                        const T* __restrict__ sendbuff,     // local user input
                                        size_t count,                       // full-message element count
                                        int selfRank, int nRanksRt,
                                        uint32_t* __restrict__ epochDev,    // per-block LL epoch cells (shared AG+AR)
                                        int epochLen) {                     // number of cells in epochDev

  const int nRanks = NRANKS_CT ? NRANKS_CT : nRanksRt;
  const size_t bytes = count * sizeof(T);
  const size_t nPk = bytes >> 3;           // 8 payload bytes per packet
  const size_t nPk_rank = nPk / nRanks;
  const size_t slot = kDdaLLArTwoShotSlotStridePkts;

  const uint32_t flag = ddaGetLLEpochInc(epochDev, blockIdx.x, 1);
  const size_t bankOffsetPkts = (size_t)(flag & 1u) * (size_t)nRanks * slot * 2;
  const size_t bankOffsetPkts_next = bankOffsetPkts + (size_t)nRanks * slot;

  const size_t gtid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = (size_t)gridDim.x * blockDim.x;

  const uint32_t* in = reinterpret_cast<const uint32_t*>(sendbuff) + selfRank * nPk_rank * 2;
  uint32_t* out = reinterpret_cast<uint32_t*>(recvbuff) + selfRank * nPk_rank * 2;

  // Phase 1: publish my payload into every peer's slot[selfRank]. The gather is
  // kept out of the store loop on purpose -- interleaved, each store waits on the
  // next load and the per-peer streams stop overlapping, which costs about 2x. The
  // fixed trip count with the rank test as a predicate is what unrolls both loops
  // and keeps d0/d1 in registers rather than scratch.
  for (size_t pk = gtid; pk < nPk_rank; pk += stride) {
    for (int base = 1; base < nRanks; base += kDdaLLArTwoShotPeerBatch) {
      uint32_t d0[kDdaLLArTwoShotPeerBatch], d1[kDdaLLArTwoShotPeerBatch];
#pragma unroll
      for (int i = 0; i < kDdaLLArTwoShotPeerBatch; ++i) {
        const int r = base + i;
        if (r < nRanks) {
          const int peer = (selfRank + r) % nRanks;
          const uint32_t* peer_in = reinterpret_cast<const uint32_t*>(sendbuff) + peer * nPk_rank * 2;
          d0[i] = peer_in[2 * pk];
          d1[i] = peer_in[2 * pk + 1];
        }
      }
#pragma unroll
      for (int i = 0; i < kDdaLLArTwoShotPeerBatch; ++i) {
        const int r = base + i;
        if (r < nRanks) {
          const int peer = (selfRank + r) % nRanks;
          LLPacket16* dst = reinterpret_cast<LLPacket16*>(peerScratch[peer]) + bankOffsetPkts + (size_t)selfRank * slot;
          ddaLLStoreLineB128(reinterpret_cast<uint32_t*>(&dst[pk]), d0[i], flag, d1[i], flag);
        }
      }
    }
  }

  // Phase 2: poll my slots for the other ranks, reduce with my own data, then
  // publish the reduced shard to every peer's second stage.
  LLPacket16* myBase = reinterpret_cast<LLPacket16*>(peerScratch[selfRank]) + bankOffsetPkts;
  for (size_t pk = gtid; pk < nPk_rank; pk += stride) {
    uint32_t acc0 = in[2 * pk];
    uint32_t acc1 = in[2 * pk + 1];
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

    #pragma unroll
    for (int r = 1; r < nRanks; ++r) {
      const int peer = (selfRank + r) % nRanks;
      LLPacket16* dst = reinterpret_cast<LLPacket16*>(peerScratch[peer]) + bankOffsetPkts_next + (size_t)selfRank * slot;
      ddaLLStoreLineB128(reinterpret_cast<uint32_t*>(&dst[pk]), acc0, flag, acc1, flag);
    }
  }

  // Phase 3: poll my slots for the other ranks, read data and write to output.
  myBase = reinterpret_cast<LLPacket16*>(peerScratch[selfRank]) + bankOffsetPkts_next;
  for (size_t pk = gtid; pk < nPk_rank; pk += stride) {
    #pragma unroll
    for (int r = 1; r < nRanks; ++r) {
      const int peer = (selfRank + r) % nRanks;
      uint32_t *peer_out = reinterpret_cast<uint32_t*>(recvbuff) + peer * nPk_rank * 2;
      volatile LLPacket16* src = myBase + (size_t)peer * slot;
      uint32_t d0, f0, d1, f1;
      do {
        ddaLLLoadLineB128(reinterpret_cast<const uint32_t*>(const_cast<LLPacket16*>(&src[pk])), d0, f0, d1, f1);
      } while (f0 != flag || f1 != flag);
      peer_out[2 * pk] = d0;
      peer_out[2 * pk + 1] = d1;
    }
  }

  ddaSetLLEpoch(epochDev, epochLen, blockIdx.x, gridDim.x, flag);
}

} // namespace dda::common
