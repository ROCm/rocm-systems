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

#include "algorithms/CollCommon.h"
// Staging layout is shared with the one-shot tier, not redefined here.
#include "algorithms/all_reduce/all_reduce_dda_ll.h"

namespace meta::comms {

// Both tiers stage through comm->ddaScratch under one epoch counter, so bank =
// flag & 1 has to land where the other tier expects it. A slot here only holds a
// shard, so it is half the one-shot slot; the bank then spans two of them (the
// publish stage and the write-back stage), which puts bank 1 at the same offset
// the one-shot tier uses. Derived from the one-shot constants rather than spelled
// out so that pairing cannot drift silently -- dda_all_reduce_fabric_ll.cu
// static_asserts the halving.
constexpr size_t kDdaLLArTwoShotMaxBytes = kDdaLLArMaxBytes;
constexpr size_t kDdaLLArTwoShotSlotStridePkts = kDdaLLArSlotStridePkts / 2;

// LL two-shot all-reduce kernel. 1D grid over one shard's packets (8B payload
// each), so the launcher sizes the grid on count/nRanks, not on count.
//
// Phase 1 (publish): rank selfRank sends peer p the part of sendbuff that p
// owns, into p's scratch at slot selfRank, as LL lines carrying the epoch flag.
// The shard reads are hoisted out of the store loop on purpose: leaving them
// interleaved makes each store wait on the next load and the per-peer streams
// stop overlapping, which costs about 2x.
// Phase 2 (reduce): rank selfRank polls its own scratch slot for each other
// rank (waiting on the flag), sums those with its own shard of sendbuff into
// recvbuff, then publishes that reduced shard back to every peer at the
// write-back stage of the bank.
// Phase 3 (write-back): poll each peer's reduced shard and drop it into that
// peer's part of recvbuff. Flag polling provides all the cross-rank ordering, so
// no GPU barrier is used.
//
// Scratch is double-buffered: bank = epoch & 1, selected via bankOffsetPkts,
// with bankOffsetPkts_next naming the write-back stage inside that bank. Self
// does not round-trip through scratch; its contribution is read directly from
// sendbuff in the reduce.
//
// Threads own disjoint packet indices in every phase, so an in-place call (where
// sendbuff aliases recvbuff) never has one thread's phase-3 store land on a
// packet another thread still has to read in phase 1.
template <typename T, int NRANKS_CT>
#if defined(USE_ROCM)
__launch_bounds__(512)
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

  // Flat block id + total launched blocks. tid 0 reads our own epoch cell (all
  // cells hold the same value) and derives this launch's flag on the device, so
  // nothing is baked into a HIP graph capture. bank = flag & 1.
  const int flatBlockId = blockIdx.x;
  const int total = gridDim.x;
  __shared__ uint32_t s_flag;
  if (threadIdx.x == 0) {
    uint32_t f = epochDev[flatBlockId] + 1u;
    if (f == 0u) f = 2u;                   // skip 0 sentinel; keep bank parity
    s_flag = f;
  }
  __syncthreads();
  const uint32_t flag = s_flag;
  const size_t bankOffsetPkts = (size_t)(flag & 1u) * (size_t)nRanks * slot * 2;
  const size_t bankOffsetPkts_next = bankOffsetPkts + (size_t)nRanks * slot;

  const size_t gtid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = (size_t)gridDim.x * blockDim.x;

  const uint32_t* in = reinterpret_cast<const uint32_t*>(sendbuff) + selfRank * nPk_rank * 2;
  uint32_t* out = reinterpret_cast<uint32_t*>(recvbuff) + selfRank * nPk_rank * 2;

  // Phase 1: publish my payload into every peer's slot[selfRank].
  for (size_t pk = gtid; pk < nPk_rank; pk += stride) {
    uint32_t d0[nRanks], d1[nRanks];
#pragma unroll
    for (int r = 1; r < nRanks; ++r) {
      const int peer = (selfRank + r) % nRanks;
      const uint32_t* peer_in = reinterpret_cast<const uint32_t*>(sendbuff) + peer * nPk_rank * 2;
      d0[r] = peer_in[2 * pk];
      d1[r] = peer_in[2 * pk + 1];
    }
#pragma unroll
    for (int r = 1; r < nRanks; ++r) {
      const int peer = (selfRank + r) % nRanks;
      LLPacket16* dst = reinterpret_cast<LLPacket16*>(peerScratch[peer]) + bankOffsetPkts + (size_t)selfRank * slot;
      ddaLLStoreLineB128(reinterpret_cast<uint32_t*>(&dst[pk]), d0[r], flag, d1[r], flag);
    }
  }

  // Phase 2: poll my slots for the other ranks, reduce with my own data.
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

  if (threadIdx.x == 0) {
    for (int e = flatBlockId; e < epochLen; e += total) {
      epochDev[e] = flag;
    }
  }
}

} // namespace meta::comms
