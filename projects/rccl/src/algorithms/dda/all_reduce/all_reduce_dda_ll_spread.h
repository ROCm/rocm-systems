/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Spread two-shot tier of the LL-protocol all-reduce for the DDA path.
 *
 * Runs on the one-shot tier's grid and moves the two-shot tier's bytes. The
 * one-shot puts one packet per thread on the wire to every peer, so one block
 * carries blockDim x 16B x (nRanks-1); here each block owns the same slice of
 * every rank's shard on all ranks, and its threads share (peer, packet) units,
 * so a block carries about 2 x 16B x blockDim x (nRanks-1) / nRanks:
 *
 *   1. publish:   send each peer my copy of that peer's shard slice,
 *   2. reduce:    poll the copies of my shard slice into LDS, fold ranks 0..n-1
 *                 in rank order (the one-shot's fold, so the result is bitwise
 *                 the one-shot's), write it out and publish it to every peer,
 *   3. writeback: poll the peers' reduced slices into recvbuff.
 *
 * Scratch layout, slot geometry and the LL epoch are the two-shot tier's.
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

#include "algorithms/dda/all_reduce/all_reduce_dda_ll_twoshot.h"

namespace dda::common {

// Threads per block: the one-shot tier's, so the spread kernel runs on exactly its grid.
constexpr int kDdaLLSpreadThreads = 512;

// Payload words (8B each) one reduce pass stages in LDS: the slice length times nRanks.
constexpr int kDdaLLSpreadLdsPkts = 2048;

// Longest slice one reduce pass covers (the slice length at 2 ranks).
constexpr int kDdaLLSpreadMaxPassPkts = kDdaLLSpreadLdsPkts / 2;

// Ranks the LDS peer table holds; the launcher asserts it covers every comm the fabric path admits.
constexpr int kDdaLLSpreadMaxRanks = 144;

// Units (one LL line each) a thread keeps in flight per sweep.
constexpr int kDdaLLSpreadBatch = 4;

// A pass polls (nRanks - 1) * slice < kDdaLLSpreadLdsPkts lines, so one sweep of a full block covers them.
static_assert(kDdaLLSpreadLdsPkts <= kDdaLLSpreadThreads * kDdaLLSpreadBatch, "a reduce pass needs one poll sweep");

// Packets of one shard a single reduce pass covers for nRanks ranks.
__host__ __device__ constexpr size_t ddaLLSpreadPassPkts(int nRanks) {
  return (size_t)kDdaLLSpreadLdsPkts / (size_t)nRanks;
}

// Packets of each rank's shard of nPk packets; kernel, selection rule and tests all size the shards with this.
__host__ __device__ constexpr size_t ddaLLSpreadShardPkts(size_t nPk, int nRanks) {
  return (nPk + (size_t)nRanks - 1) / (size_t)nRanks;
}

// Packets rank owns in shards of shardPkts: the last shards are shorter, or empty, when nRanks does not divide nPk.
__host__ __device__ constexpr size_t ddaLLSpreadShardLen(size_t nPk, size_t shardPkts, int rank) {
  const size_t first = (size_t)rank * shardPkts;
  return first < nPk ? (nPk - first < shardPkts ? nPk - first : shardPkts) : (size_t)0;
}

// Workgroup barrier for LDS only: on gfx1250 __syncthreads() would also wait for every remote store in flight.
__device__ __forceinline__ void ddaLLSpreadLdsBarrier() {
#if defined(__gfx1250__) || defined(__gfx1250_strict__)
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
#else
  __syncthreads();
#endif
}

// Polls the lines of up to kDdaLLSpreadBatch units this thread owns (mask bit i = unit i valid); false if aborted.
__device__ __forceinline__ bool ddaLLSpreadPoll(const uint32_t* const (&line)[kDdaLLSpreadBatch], uint32_t mask,
                                                uint32_t flag, DdaKernelParams params,
                                                uint32_t (&d0)[kDdaLLSpreadBatch], uint32_t (&d1)[kDdaLLSpreadBatch]) {
  return ddaLLWaitLineFlagsBatched<kDdaLLSpreadBatch>(line, mask, flag, params, d0, d1);
}

// LL spread two-shot all-reduce kernel; launched on the one-shot tier's grid (kDdaLLSpreadThreads per block).
template <typename T, int NRANKS_CT>
#if defined(USE_ROCM)
__launch_bounds__(kDdaLLSpreadThreads)
#endif
  __global__ void ddaAllReduceSpreadTwoShotLL(T* const* __restrict__ peerScratch,    // ddaPeerPtrsDev: scratch bases
                                              T* __restrict__ recvbuff,           // local user output
                                              const T* __restrict__ sendbuff,     // local user input
                                              size_t count,                       // full-message element count
                                              int selfRank, int nRanksRt,
                                              uint32_t* __restrict__ epochDev,    // per-block LL epoch cells
                                              int epochLen,                       // number of cells in epochDev
                                              size_t bankSize,                    // scratch bank size (from host)
                                              DdaKernelParams params) {
  __shared__ uint2 ldsIn[kDdaLLSpreadLdsPkts];
  __shared__ uint2 ldsAcc[kDdaLLSpreadMaxPassPkts];
  __shared__ LLPacket16* ldsPeer[kDdaLLSpreadMaxRanks];
  // Set by any thread whose wait aborted: the fold reads LDS rows other threads polled, so an abort is block-wide.
  __shared__ int ldsAborted;

  const int nRanks = NRANKS_CT ? NRANKS_CT : nRanksRt;
  const size_t nPk = (count * sizeof(T)) >> 3;
  const size_t nPkRank = ddaLLSpreadShardPkts(nPk, nRanks);
  auto shardLen = [&](int rank) { return ddaLLSpreadShardLen(nPk, nPkRank, rank); };
  const size_t slot = ddaLLArTwoShotSlotPkts(bankSize, nRanks);

  // This block's slice of every shard: the same packet range on every rank, so its LL epoch stays in lockstep.
  const size_t chunk = (nPkRank + gridDim.x - 1) / gridDim.x;
  const size_t p0 = (size_t)blockIdx.x * chunk < nPkRank ? (size_t)blockIdx.x * chunk : nPkRank;
  const size_t p1 = p0 + chunk < nPkRank ? p0 + chunk : nPkRank;
  const size_t len = p1 - p0;
  const size_t peerUnits = (size_t)(nRanks - 1) * len;
  // Unit counts stay far below 2^32 (the slot caps a shard), so unit index math runs in 32 bits.
  const uint32_t len32 = len > 0 ? (uint32_t)len : 1u;
  const size_t passPkts = ddaLLSpreadPassPkts(nRanks);

  const uint32_t* in = reinterpret_cast<const uint32_t*>(sendbuff);
  uint32_t* out = reinterpret_cast<uint32_t*>(recvbuff);
  const size_t tid = threadIdx.x;
  const size_t nthr = blockDim.x;

  // Peer of unit u in runs of run32 packets per peer, counted from selfRank + 1.
  auto unitPeer = [&](size_t u, uint32_t run32) { return (selfRank + 1 + (int)((uint32_t)u / run32)) % nRanks; };

  // Unit u of phase 1 (peer, packet pk): peer and packet, and its payload loaded from sendbuff.
  auto sendUnit = [&](size_t u, int& peer, size_t& pk, uint32_t& d0, uint32_t& d1) {
    peer = -1;
    if (u < peerUnits) {
      peer = unitPeer(u, len32);
      pk = p0 + (uint32_t)u % len32;
      if (pk >= shardLen(peer)) {
        peer = -1;
        return;
      }
      const size_t w = ((size_t)peer * nPkRank + pk) * 2;
      d0 = in[w];
      d1 = in[w + 1];
    }
  };

  // Loaded ahead of the epoch barrier: the peer table, this thread's first phase-1 batch, its first self packet.
  for (int r = (int)tid; r < nRanks; r += (int)nthr) {
    ldsPeer[r] = reinterpret_cast<LLPacket16*>(peerScratch[r]);
  }
  if (tid == 0) {
    ldsAborted = 0;
  }
  uint32_t sendD0[kDdaLLSpreadBatch], sendD1[kDdaLLSpreadBatch];
  int sendPeer[kDdaLLSpreadBatch];
  size_t sendPk[kDdaLLSpreadBatch];
#pragma unroll
  for (int i = 0; i < kDdaLLSpreadBatch; ++i) {
    sendUnit(tid + (size_t)i * nthr, sendPeer[i], sendPk[i], sendD0[i], sendD1[i]);
  }
  // My shard's part of this block's range: the reduce covers only packets every rank sends.
  const size_t mine = shardLen(selfRank);
  const size_t p1Mine = p1 < mine ? p1 : (p0 < mine ? mine : p0);
  const size_t plen0 = p1Mine - p0 < passPkts ? p1Mine - p0 : passPkts;
  uint2 self0 = make_uint2(0, 0);
  if (tid < plen0) {
    const size_t w = ((size_t)selfRank * nPkRank + p0 + tid) * 2;
    self0 = make_uint2(in[w], in[w + 1]);
  }

  const uint32_t flag = ddaGetLLEpochInc(epochDev, blockIdx.x, 1);
  const size_t bankOffsetPkts = (size_t)(flag & 1u) * (bankSize / sizeof(LLPacket16));
  const size_t bankOffsetPktsNext = bankOffsetPkts + (size_t)nRanks * slot;
  const LLPacket16* myBase = ldsPeer[selfRank];

  // Phase 1: a batch of loads goes out before its stores; the first batch was loaded above.
  for (size_t base = tid; base < peerUnits; base += nthr * kDdaLLSpreadBatch) {
    if (base != tid) {
#pragma unroll
      for (int i = 0; i < kDdaLLSpreadBatch; ++i) {
        sendUnit(base + (size_t)i * nthr, sendPeer[i], sendPk[i], sendD0[i], sendD1[i]);
      }
    }
#pragma unroll
    for (int i = 0; i < kDdaLLSpreadBatch; ++i) {
      if (sendPeer[i] >= 0) {
        LLPacket16* dst = ldsPeer[sendPeer[i]] + bankOffsetPkts + (size_t)selfRank * slot + sendPk[i];
        ddaLLStoreLineB128(reinterpret_cast<uint32_t*>(dst), sendD0[i], flag, sendD1[i], flag);
      }
    }
  }

  // Phase 2 per LDS pass: stage each rank's copy, fold in rank order, publish; an abort leaves at the staging barrier.
  bool aborted = false;
  for (size_t q0 = p0; q0 < p1Mine; q0 += passPkts) {
    const size_t plen = p1Mine - q0 < passPkts ? p1Mine - q0 : passPkts;
    const size_t pollUnits = (size_t)(nRanks - 1) * plen;
    const uint32_t plen32 = (uint32_t)plen;
    for (size_t base = tid; base < pollUnits; base += nthr * kDdaLLSpreadBatch) {
      const uint32_t* line[kDdaLLSpreadBatch];
      int peer[kDdaLLSpreadBatch];
      size_t k[kDdaLLSpreadBatch];
      uint32_t mask = 0;
#pragma unroll
      for (int i = 0; i < kDdaLLSpreadBatch; ++i) {
        const size_t u = base + (size_t)i * nthr;
        peer[i] = 0;
        k[i] = 0;
        line[i] = nullptr;
        if (u < pollUnits) {
          peer[i] = unitPeer(u, plen32);
          k[i] = (uint32_t)u % plen32;
          line[i] = reinterpret_cast<const uint32_t*>(myBase + bankOffsetPkts + (size_t)peer[i] * slot + q0 + k[i]);
          mask |= 1u << i;
        }
      }
      uint32_t d0[kDdaLLSpreadBatch], d1[kDdaLLSpreadBatch];
      if (!aborted) {
        aborted = !ddaLLSpreadPoll(line, mask, flag, params, d0, d1);
      }
#pragma unroll
      for (int i = 0; i < kDdaLLSpreadBatch; ++i) {
        if (((mask >> i) & 1u) != 0) {
          ldsIn[(size_t)peer[i] * plen + k[i]] = make_uint2(d0[i], d1[i]);
        }
      }
    }
    for (size_t k = tid; k < plen; k += nthr) {
      if (q0 == p0 && k == tid) {
        ldsIn[(size_t)selfRank * plen + k] = self0;
      } else {
        const size_t w = ((size_t)selfRank * nPkRank + q0 + k) * 2;
        ldsIn[(size_t)selfRank * plen + k] = make_uint2(in[w], in[w + 1]);
      }
    }
    if (aborted) {
      ldsAborted = 1;
    }
    ddaLLSpreadLdsBarrier();
    aborted = aborted || ldsAborted != 0;
    // Partial sums: skip the fold, my recvbuff shard and the publish. aborted is uniform here, so no barrier is split.
    if (aborted) {
      break;
    }

    for (size_t k = tid; k < plen; k += nthr) {
      uint32_t acc0 = ldsIn[k].x;
      uint32_t acc1 = ldsIn[k].y;
      // A batch of LDS reads is issued before its adds, which still run in rank order.
      for (int r0 = 1; r0 < nRanks; r0 += kDdaLLPeerBatch) {
        uint2 e[kDdaLLPeerBatch];
#pragma unroll
        for (int i = 0; i < kDdaLLPeerBatch; ++i) {
          e[i] = r0 + i < nRanks ? ldsIn[(size_t)(r0 + i) * plen + k] : make_uint2(0, 0);
        }
#pragma unroll
        for (int i = 0; i < kDdaLLPeerBatch; ++i) {
          if (r0 + i < nRanks) {
            ddaLLAddPayload<T>(acc0, acc1, e[i].x, e[i].y);
          }
        }
      }
      const size_t w = ((size_t)selfRank * nPkRank + q0 + k) * 2;
      out[w] = acc0;
      out[w + 1] = acc1;
      ldsAcc[k] = make_uint2(acc0, acc1);
    }
    ddaLLSpreadLdsBarrier();

    for (size_t u = tid; u < pollUnits; u += nthr) {
      const int peer = unitPeer(u, plen32);
      const size_t k = (uint32_t)u % plen32;
      const uint2 acc = ldsAcc[k];
      LLPacket16* dst = ldsPeer[peer] + bankOffsetPktsNext + (size_t)selfRank * slot + q0 + k;
      ddaLLStoreLineB128(reinterpret_cast<uint32_t*>(dst), acc.x, flag, acc.y, flag);
    }
  }

  // Phase 3: unit u = (peer, packet pk): poll that peer's reduced packet and write it to its part of recvbuff.
  for (size_t base = tid; base < peerUnits; base += nthr * kDdaLLSpreadBatch) {
    const uint32_t* line[kDdaLLSpreadBatch];
    size_t w[kDdaLLSpreadBatch];
    uint32_t mask = 0;
#pragma unroll
    for (int i = 0; i < kDdaLLSpreadBatch; ++i) {
      const size_t u = base + (size_t)i * nthr;
      line[i] = nullptr;
      w[i] = 0;
      if (u < peerUnits) {
        const int peer = unitPeer(u, len32);
        const size_t pk = p0 + (uint32_t)u % len32;
        if (pk < shardLen(peer)) {
          line[i] = reinterpret_cast<const uint32_t*>(myBase + bankOffsetPktsNext + (size_t)peer * slot + pk);
          w[i] = ((size_t)peer * nPkRank + pk) * 2;
          mask |= 1u << i;
        }
      }
    }
    uint32_t d0[kDdaLLSpreadBatch], d1[kDdaLLSpreadBatch];
    if (aborted) {
      continue;
    }
    aborted = !ddaLLSpreadPoll(line, mask, flag, params, d0, d1);
#pragma unroll
    for (int i = 0; i < kDdaLLSpreadBatch; ++i) {
      if (((mask >> i) & 1u) != 0 && !aborted) {
        out[w[i]] = d0[i];
        out[w[i] + 1] = d1[i];
      }
    }
  }

  ddaSetLLEpoch(epochDev, epochLen, blockIdx.x, gridDim.x, flag);
}

} // namespace dda::common
