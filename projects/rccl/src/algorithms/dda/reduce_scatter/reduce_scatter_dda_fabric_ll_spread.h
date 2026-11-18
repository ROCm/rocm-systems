/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Spread tier of the LL-protocol reduce-scatter for the DDA fabric path: the
 * shard kernel's scatter and fold on a grid over the whole message.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#pragma once

#include <cstddef>
#include <cstdint>

#include <hip/hip_runtime.h>

#include "algorithms/dda/device/CollCommon.h"
#include "algorithms/dda/reduce_scatter/reduce_scatter_dda_fabric_ll.h"

namespace dda::common {

// Threads per block: the shard kernel's.
constexpr int kDdaLLRsSpreadThreads = 256;

// Ranks from which the launcher runs the spread kernel instead of the shard kernel.
constexpr int kDdaLLRsSpreadMinRanks = 16;

// Phase-1 units a thread loads before storing any: the shard kernel's send batch.
constexpr int kDdaLLRsSpreadSendBatch = kDdaLLPeerBatch;

// Phase-2 lines a thread keeps in flight per poll sweep.
constexpr int kDdaLLRsSpreadPollBatch = 4;

// Payload words (8B each) one reduce pass stages in LDS, sized so one poll sweep of a full block covers a pass.
constexpr int kDdaLLRsSpreadLdsPkts = kDdaLLRsSpreadThreads * kDdaLLRsSpreadPollBatch;

// Packets in one 128B line of LL wire. Slices are whole lines, so no two blocks' remote stores share a line.
constexpr size_t kDdaLLRsSpreadLinePkts = 128 / sizeof(LLPacket16);

// Packets of each shard one reduce pass covers for nRanks ranks.
__host__ __device__ constexpr size_t ddaLLRsSpreadPassPkts(int nRanks) {
  return (size_t)kDdaLLRsSpreadLdsPkts / (size_t)nRanks;
}

// Packets of each shard a block owns on a grid of nBlocks, rounded up to whole lines. The launcher sizes the grid with
// it too.
__host__ __device__ constexpr size_t ddaLLRsSpreadSlicePkts(size_t nPk, unsigned nBlocks) {
  const size_t pkts = (nPk + nBlocks - 1) / nBlocks;
  return (pkts + kDdaLLRsSpreadLinePkts - 1) / kDdaLLRsSpreadLinePkts * kDdaLLRsSpreadLinePkts;
}

// LL reduce-scatter over the whole message. Each block's threads split the (peer, packet) units of its slice of
// every shard.
template <typename T, int NRANKS_CT>
__launch_bounds__(kDdaLLRsSpreadThreads)
  __global__ void ddaReduceScatterFabricSpreadLL(T* const* __restrict__ peerScratch, // ddaPeerPtrsDev: scratch bases
                                                 T* __restrict__ recvbuff, // local user output (recvcount elems)
                                                 const T* __restrict__ sendbuff, // local user input (recvcount*nRanks)
                                                 size_t recvcount, // per-rank shard element count
                                                 int selfRank, int nRanksRt,
                                                 uint32_t* __restrict__ epochDev, // per-block LL epoch cells
                                                 int epochLen, // number of cells in epochDev
                                                 size_t bankSize, // scratch bank size (from host)
                                                 DdaKernelParams params) {
  __shared__ uint2 ldsIn[kDdaLLRsSpreadLdsPkts];
  // Set by any thread whose wait aborted: the fold reads LDS rows other threads polled, so an abort is block-wide.
  __shared__ int ldsAborted;

  const int nRanks = NRANKS_CT ? NRANKS_CT : nRanksRt;
  const size_t nPk = (recvcount * sizeof(T)) >> 3; // 8 payload bytes per packet
  const size_t slot = kDdaLLRsSlotStridePkts;

  // This block's slice of every shard: the same packet range on every rank, so its LL epoch stays in lockstep.
  const size_t chunk = ddaLLRsSpreadSlicePkts(nPk, gridDim.x);
  const size_t p0 = (size_t)blockIdx.x * chunk < nPk ? (size_t)blockIdx.x * chunk : nPk;
  const size_t p1 = p0 + chunk < nPk ? p0 + chunk : nPk;
  const size_t len = p1 - p0;
  const size_t peerUnits = (size_t)(nRanks - 1) * len;
  // Unit counts stay far below 2^32 (the slot caps a shard), so unit index math runs in 32 bits.
  const uint32_t len32 = len > 0 ? (uint32_t)len : 1u;
  const size_t passPkts = ddaLLRsSpreadPassPkts(nRanks);

  const uint32_t* in = reinterpret_cast<const uint32_t*>(sendbuff);
  uint32_t* out = reinterpret_cast<uint32_t*>(recvbuff);
  const size_t tid = threadIdx.x;
  const size_t nthr = blockDim.x;

  // Rank offset of unit u in runs of run32 packets per rank, counted from selfRank + 1.
  auto unitOff = [&](size_t u, uint32_t run32) { return 1 + (int)((uint32_t)u / run32); };

  // Phase-1 unit u: destination before the bank offset, and payload. A unit past peerUnits repeats unit 0, so
  // lanes load and store without a branch.
  auto sendUnit = [&](size_t u, bool& valid, LLPacket16*& dst, uint32_t& d0, uint32_t& d1) {
    valid = u < peerUnits;
    const uint32_t v = valid ? (uint32_t)u : 0u;
    // unitOff stays below nRanks, so one conditional subtract wraps the rank.
    const int off = selfRank + unitOff(v, len32);
    const int peer = off < nRanks ? off : off - nRanks;
    const size_t pk = len > 0 ? p0 + v % len32 : 0;
    const size_t w = ((size_t)peer * nPk + pk) * 2;
    d0 = in[w];
    d1 = in[w + 1];
    dst = reinterpret_cast<LLPacket16*>(peerScratch[peer]) + (size_t)selfRank * slot + pk;
  };

  // This thread's first phase-1 batch, loaded before the epoch read so those loads overlap it.
  bool sendValid[kDdaLLRsSpreadSendBatch];
  LLPacket16* sendDst[kDdaLLRsSpreadSendBatch];
  uint32_t sendD0[kDdaLLRsSpreadSendBatch], sendD1[kDdaLLRsSpreadSendBatch];
#pragma unroll
  for (int i = 0; i < kDdaLLRsSpreadSendBatch; ++i) {
    sendUnit(tid + (size_t)i * nthr, sendValid[i], sendDst[i], sendD0[i], sendD1[i]);
  }
  if (tid == 0) {
    ldsAborted = 0;
  }

  // No barrier before phase 1's stores. The one after phase 1 orders this read before ddaSetLLEpoch.
  const uint32_t flag = ddaNextLLEpoch(epochDev, blockIdx.x, 1);
  const size_t bankOffsetPkts = ddaLLRsBankOffsetPkts(flag, bankSize);
  const LLPacket16* myBase = reinterpret_cast<const LLPacket16*>(peerScratch[selfRank]) + bankOffsetPkts;

  // Phase 1: a batch's loads go out before its stores. The first batch was loaded above.
  for (size_t b0 = 0; b0 < peerUnits; b0 += nthr * kDdaLLRsSpreadSendBatch) {
    if (b0 != 0) {
#pragma unroll
      for (int i = 0; i < kDdaLLRsSpreadSendBatch; ++i) {
        sendUnit(b0 + tid + (size_t)i * nthr, sendValid[i], sendDst[i], sendD0[i], sendD1[i]);
      }
    }
    // A slot stores if any lane of the wave has a unit there, and then every lane stores, so no per-lane branch splits
    // the batch's stores.
#pragma unroll
    for (int i = 0; i < kDdaLLRsSpreadSendBatch; ++i) {
      if (__any(sendValid[i])) {
        ddaLLStoreLineB128(reinterpret_cast<uint32_t*>(sendDst[i] + bankOffsetPkts), sendD0[i], flag, sendD1[i], flag);
      }
    }
  }
  // Orders epoch reads before ddaSetLLEpoch and the ldsAborted reset before phase 2. The LDS-only barrier ignores
  // global loads, hence the wait.
#if defined(__gfx1250__) || defined(__gfx1250_strict__)
  asm volatile("s_wait_loadcnt 0x0\n\ts_wait_kmcnt 0x0" ::: "memory");
#endif
  ddaLLSpreadLdsBarrier();

  // Phase 2 per LDS pass: row r holds the copy from selfRank + r, row 0 my own. Folding rows in order gives the shard
  // kernel's sums bit for bit.
  bool aborted = false;
  for (size_t q0 = p0; q0 < p1; q0 += passPkts) {
    const size_t plen = p1 - q0 < passPkts ? p1 - q0 : passPkts;
    const size_t pollUnits = (size_t)(nRanks - 1) * plen;
    const uint32_t plen32 = (uint32_t)plen;
    // This thread's packet of my own row, loaded before the polls so it shares their wait. A thread past the pass
    // reads packet q0 and stages nothing.
    const size_t ws = ((size_t)selfRank * nPk + q0 + (tid < plen ? tid : 0)) * 2;
    const uint2 self = make_uint2(in[ws], in[ws + 1]);
    for (size_t base = tid; base < pollUnits; base += nthr * kDdaLLRsSpreadPollBatch) {
      const uint32_t* line[kDdaLLRsSpreadPollBatch];
      int row[kDdaLLRsSpreadPollBatch];
      size_t k[kDdaLLRsSpreadPollBatch];
      uint32_t mask = 0;
#pragma unroll
      for (int i = 0; i < kDdaLLRsSpreadPollBatch; ++i) {
        const size_t u = base + (size_t)i * nthr;
        row[i] = 0;
        k[i] = 0;
        line[i] = nullptr;
        if (u < pollUnits) {
          row[i] = unitOff(u, plen32);
          k[i] = (uint32_t)u % plen32;
          const int peer = (selfRank + row[i]) % nRanks;
          line[i] = reinterpret_cast<const uint32_t*>(myBase + (size_t)peer * slot + q0 + k[i]);
          mask |= 1u << i;
        }
      }
      uint32_t d0[kDdaLLRsSpreadPollBatch], d1[kDdaLLRsSpreadPollBatch];
      if (!aborted) {
        aborted = !ddaLLWaitLineFlagsBatched<kDdaLLRsSpreadPollBatch>(line, mask, flag, params, d0, d1);
      }
#pragma unroll
      for (int i = 0; i < kDdaLLRsSpreadPollBatch; ++i) {
        if (((mask >> i) & 1u) != 0) {
          ldsIn[(size_t)row[i] * plen + k[i]] = make_uint2(d0[i], d1[i]);
        }
      }
    }
    if (tid < plen) {
      ldsIn[tid] = self;
    }
    // A pass longer than the block, at 2 or 3 ranks, stages the rest of my row here.
    for (size_t k = tid + nthr; k < plen; k += nthr) {
      const size_t w = ((size_t)selfRank * nPk + q0 + k) * 2;
      ldsIn[k] = make_uint2(in[w], in[w + 1]);
    }
    if (aborted) {
      atomicExch(&ldsAborted, 1);
    }
    ddaLLSpreadLdsBarrier();
    aborted = aborted || ldsAborted != 0;
    // Partial sums: skip the fold and recvbuff. aborted is uniform here, so no barrier is split.
    if (aborted) {
      break;
    }

    for (size_t k = tid; k < plen; k += nthr) {
      uint32_t acc0 = ldsIn[k].x;
      uint32_t acc1 = ldsIn[k].y;
      // A batch of LDS reads is issued before its adds, which still run in row order.
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
      out[2 * (q0 + k)] = acc0;
      out[2 * (q0 + k) + 1] = acc1;
    }
    // The next pass restages ldsIn.
    if (q0 + passPkts < p1) {
      ddaLLSpreadLdsBarrier();
    }
  }

  ddaSetLLEpoch(epochDev, epochLen, blockIdx.x, gridDim.x, flag);
}

} // namespace dda::common
