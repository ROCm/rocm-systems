/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Two-shot tier of the LL128-protocol all-reduce for the DDA path.
 *
 * Same relation to all_reduce_dda_ll128.h as all_reduce_dda_ll_twoshot.h has to
 * all_reduce_dda_ll.h: each rank owns one shard of count/nRanks elements and only
 * ever transports that shard, so every rank moves nRanks-1 shards instead of
 * nRanks-1 whole messages. Three phases, all flag-ordered like the one-shot, so
 * still no GPU barrier:
 *
 *   1. publish: send each peer the shard that peer owns,
 *   2. reduce:  sum the copies of my own shard that arrived, publish the reduced
 *               shard back to every peer, and write it to my part of recvbuff,
 *   3. writeback: collect the peers' reduced shards into the rest of recvbuff.
 *
 * The wire format, slice geometry and pack/poll/unpack helpers are the one-shot
 * tier's, unchanged; only the unit of work (a shard rather than the whole
 * message) and the extra staging area differ.
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
#include "algorithms/dda/device/CollCommon_ll128.h"
// Slice/wire geometry is shared with the one-shot tier, not redefined here.
#include "algorithms/dda/all_reduce/all_reduce_dda_ll128.h"

namespace dda::common {

// Slot geometry for the two-shot tier. A bank has to hold two staging areas
// (publish + write-back) of nRanks slots each, so a slot is half the one-shot's
// and the bank stride is unchanged. Floored to a whole number of slices for the
// same reason as the one-shot: an unrounded stride can come out an odd number of
// 8B words and misalign every slot but rank 0's.
constexpr size_t ddaLL128ArTwoShotSlotWords(size_t bankSize, int nRanks) {
  return ddaLL128SlotWords(bankSize, nRanks, 2);
}

// LL128 two-shot all-reduce kernel. One warp owns one slice of a shard at a time,
// so the launcher sizes the grid on the shard's slices, not the message's.
//
// Phase 1 (publish): send peer p the shard p owns, into p's scratch at slot
// selfRank in the first staging area, as LL128 lines carrying the epoch flag.
// Phase 2 (reduce and publish): poll my slot for each peer's copy of my shard,
// sum with my own, publish the result to every peer's second staging area, and
// unpack it into my part of recvbuff.
// Phase 3 (gather): poll for the reduced shards the peers published and unpack
// each into that peer's part of recvbuff.
//
// Flag polling provides all the cross-rank ordering, so no GPU barrier is used.
// Scratch is double buffered: bank = flag & 1, with bankWordsNext naming the
// second staging area inside that bank. Self does not round-trip through scratch;
// its contribution seeds the accumulator straight from sendbuff.
//
// A warp owns the same slice index in every phase, and phase 1 only ever reads
// peer shards while phase 3 only ever touches the self shard, so an in-place call
// (sendbuff aliasing recvbuff) never has one warp's write land on a range another
// warp still has to read.
//
// WIRE (DdaLL128Wire) picks the phase-1 and phase-2 writer; staged, it takes ddaLL128StageBytes(blockDim.x) of LDS.
template <typename T, int NRANKS_CT, int WIRE = kDdaLL128WireCoop>
__launch_bounds__(kDdaLL128ArThreads)
  __global__ void ddaAllReduceTwoShotLL128(T* const* __restrict__ peerScratch, // ddaPeerPtrsDev: nRanks scratch bases
                                           T* __restrict__ recvbuff, // local user output
                                           const T* __restrict__ sendbuff, // local user input
                                           size_t shardBytes, // per-rank shard payload; multiple of 16
                                           int selfRank, int nRanksRt,
                                           uint32_t* __restrict__ epochDev, // per-block LL epoch cells
                                           int epochLen, // number of cells in epochDev
                                           size_t slicesTotal, // slices per shard this call uses
                                           size_t bankSize, // scratch bank size (from host)
                                           DdaKernelParams params) {

  const int nRanks = NRANKS_CT ? NRANKS_CT : nRanksRt;

  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  const int lane = tid % kDdaLL128Warp;
  const int warp = tid / kDdaLL128Warp;
  const int nwarps = nthreads / kDdaLL128Warp;
  const bool flagLane = ddaLL128IsFlagLane(lane);

  const uint32_t flag32 = ddaGetLLEpochInc(epochDev, blockIdx.x, 1);
  const uint64_t flag = ((uint64_t)flag32 << 32) | (uint64_t)flag32;

  // Two staging areas per bank; the second starts one full fan-out in.
  const size_t slotWords = ddaLL128ArTwoShotSlotWords(bankSize, nRanks);
  const uint64_t bankWords = (uint64_t)(flag32 & 1u) * (uint64_t)(bankSize / sizeof(uint64_t));
  const uint64_t bankWordsNext = bankWords + (uint64_t)nRanks * (uint64_t)slotWords;

  // Slices stride by warp across the whole grid. The bound has to be the slice
  // count, not the warp count: the grid is capped at ddaFabricMaxBlocks, so a
  // large shard has more slices than warps and each warp must take several.
  const size_t gwarp = (size_t)blockIdx.x * (size_t)nwarps + (size_t)warp;
  const size_t wstride = (size_t)gridDim.x * (size_t)nwarps;

  const int8_t* srcBytes = reinterpret_cast<const int8_t*>(sendbuff);
  int8_t* dstBytes = reinterpret_cast<int8_t*>(recvbuff);
  const uint64_t* selfBase = reinterpret_cast<const uint64_t*>(peerScratch[selfRank]);

  DdaLL128Stage stage{};
  if constexpr (ddaLL128Staged<WIRE>()) {
    extern __shared__ __attribute__((aligned(128))) uint8_t ddaLL128ArStageLds[];
    stage = ddaLL128StageInit(ddaLL128ArStageLds, warp);
  }

  // Start of the wire slice holding wireOff in my slot of the peer's staging area `area` words into its scratch.
  auto scatterSlice = [&](int peer, uint64_t area, size_t wireOff) {
    return reinterpret_cast<uint64_t*>(peerScratch[peer]) + area + (uint64_t)selfRank * (uint64_t)slotWords +
           (wireOff - 2 * lane);
  };

  // Phase 1: send each peer the shard that peer owns, kBatch peers per round.
  // Staged: a push returns once issued, so loading one peer ahead hides the load; a batch of 8 would spill.
  // Coop: load a batch of 8 peers, then store it.
  constexpr int kAhead = ddaLL128Staged<WIRE>() ? 1 : 0;
  constexpr int kBatch = kAhead ? 1 : kDdaLLPeerBatch;
  for (size_t s = gwarp; s < slicesTotal; s += wstride) {
    const size_t sliceByte = s * (size_t)kDdaLL128DataBytesPerSlice;
    const size_t rem = shardBytes - sliceByte;
    const int eltInSlice =
      rem < (size_t)kDdaLL128DataBytesPerSlice ? (int)rem : kDdaLL128DataBytesPerSlice;
    const size_t wireOff = s * (size_t)kDdaLL128WireWordsPerSlice + 2 * lane;

    // With look-ahead, the round's batch was loaded by the round before.
    uint64_t cur[kBatch][kDdaLL128WordsPerThread] = {};
    if constexpr (kAhead) {
      ddaLL128LoadRegs<int8_t>(cur[0], srcBytes + (size_t)((selfRank + 1) % nRanks) * shardBytes + sliceByte,
                               eltInSlice, lane, flagLane);
    }
#pragma unroll 1
    for (int base = 1; base < nRanks; base += kBatch) {
      uint64_t next[kBatch][kDdaLL128WordsPerThread] = {};
      // do-while: a batch of one leaves no inner loop, which keeps the staged code as tight as a plain peer loop.
      int i = 0;
#pragma unroll
      do {
        const int r = base + i;
        if (r + kAhead < nRanks) {
          const int peer = (selfRank + r) % nRanks;
          const int src = kAhead ? (peer + 1) % nRanks : peer;
          ddaLL128LoadRegs<int8_t>(next[i], srcBytes + (size_t)src * shardBytes + sliceByte, eltInSlice, lane,
                                   flagLane);
        }
      } while (++i < kBatch);
      uint64_t (&send)[kBatch][kDdaLL128WordsPerThread] = kAhead ? cur : next;
      i = 0;
#pragma unroll
      do {
        const int r = base + i;
        if (r < nRanks) {
          const int peer = (selfRank + r) % nRanks;
          ddaLL128StageFill<WIRE>(stage, send[i], flag, flagLane, lane);
          ddaLL128StagePush<WIRE>(stage, scatterSlice(peer, bankWords, wireOff), send[i], flag, flagLane, lane);
        }
      } while (++i < kBatch);
      if constexpr (kAhead) {
#pragma unroll
        for (int u = 0; u < kDdaLL128WordsPerThread; ++u) {
          cur[0][u] = next[0][u];
        }
      }
    }
  }

  // Phase 2: fold every copy of my shard, hand the result back out, then keep it.
  // The staged kernel folds over the runtime count: an unrolled fold on top of the staging state raises register use.
  const int foldRanks = ddaLL128Staged<WIRE>() ? nRanksRt : nRanks;
  for (size_t s = gwarp; s < slicesTotal; s += wstride) {
    const size_t sliceByte = s * (size_t)kDdaLL128DataBytesPerSlice;
    const size_t rem = shardBytes - sliceByte;
    const int eltInSlice =
      rem < (size_t)kDdaLL128DataBytesPerSlice ? (int)rem : kDdaLL128DataBytesPerSlice;
    const size_t wireOff = s * (size_t)kDdaLL128WireWordsPerSlice + 2 * lane;

    // Seed with my own shard; peers are folded on top.
    uint64_t acc[kDdaLL128WordsPerThread] = {};
    ddaLL128LoadRegs<int8_t>(acc, srcBytes + (size_t)selfRank * shardBytes + sliceByte, eltInSlice, lane, flagLane);

    for (int r = 1; r < foldRanks; ++r) {
      const int peer = (selfRank + r) % nRanks;
      const uint64_t* gatherSlot = selfBase + bankWords + (uint64_t)peer * (uint64_t)slotWords;
      uint64_t vr[kDdaLL128WordsPerThread];
      if (!ddaLL128PollWire(gatherSlot + wireOff, vr, flag, lane, params)) {
        ddaLL128StageDrain<WIRE>();
        ddaSetLLEpoch(epochDev, epochLen, blockIdx.x, gridDim.x, flag32);
        return;
      }
      // On a flag lane the odd words hold flags rather than payload, so the sums
      // landing there are meaningless -- ddaLL128StoreRegs re-derives those slots
      // from the even ones and never writes them out, so folding blind is cheaper
      // than predicating the loop.
#pragma unroll
      for (int u = 0; u < kDdaLL128WordsPerThread; ++u) {
        acc[u] = ddaLL128AddWord<T>(acc[u], vr[u]);
      }
    }

    // Publish before the local unpack: it is what the peers are spinning on, and
    // ddaLL128StoreRegs rewrites acc in place when it reverses the flag-lane shuffle.
    if constexpr (ddaLL128Staged<WIRE>()) {
      // The abort return leaves lastOps in a VGPR; an EXEC-masked wait ladder rung would wait to 0, so keep it scalar.
      stage.lastOps = __builtin_amdgcn_readfirstlane(stage.lastOps);
    }
    ddaLL128StageFill<WIRE>(stage, acc, flag, flagLane, lane);
#pragma unroll
    for (int r = 1; r < nRanks; ++r) {
      const int peer = (selfRank + r) % nRanks;
      ddaLL128StagePush<WIRE>(stage, scatterSlice(peer, bankWordsNext, wireOff), acc, flag, flagLane, lane);
    }

    ddaLL128StoreRegs<int8_t>(dstBytes + (size_t)selfRank * shardBytes + sliceByte, acc, eltInSlice, lane, flagLane);
  }

  // Phase 3: collect the peers' reduced shards into the rest of recvbuff.
  for (size_t s = gwarp; s < slicesTotal; s += wstride) {
    const size_t sliceByte = s * (size_t)kDdaLL128DataBytesPerSlice;
    const size_t rem = shardBytes - sliceByte;
    const int eltInSlice =
      rem < (size_t)kDdaLL128DataBytesPerSlice ? (int)rem : kDdaLL128DataBytesPerSlice;
    const size_t wireOff = s * (size_t)kDdaLL128WireWordsPerSlice + 2 * lane;

    for (int r = 1; r < nRanks; ++r) {
      const int peer = (selfRank + r) % nRanks;
      const uint64_t* gatherSlot = selfBase + bankWordsNext + (uint64_t)peer * (uint64_t)slotWords;
      uint64_t vr[kDdaLL128WordsPerThread];
      if (!ddaLL128PollWire(gatherSlot + wireOff, vr, flag, lane, params)) {
        ddaLL128StageDrain<WIRE>();
        ddaSetLLEpoch(epochDev, epochLen, blockIdx.x, gridDim.x, flag32);
        return;
      }
      ddaLL128StoreRegs<int8_t>(dstBytes + (size_t)peer * shardBytes + sliceByte, vr, eltInSlice, lane, flagLane);
    }
  }

#if defined(__gfx1250__) || defined(__gfx1250_strict__)
  asm volatile("s_wait_storecnt 0x0" ::: "memory");
#endif
  ddaLL128StageDrain<WIRE>();

  ddaSetLLEpoch(epochDev, epochLen, blockIdx.x, gridDim.x, flag32);
}

// Slices phases B and D poll per round trip (ddaLL128PollWireN): the slices' loads share each wait.
constexpr int kDdaLL128SplitPollSlices = 2;
static_assert(kDdaLL128SplitPollSlices == 2, "phase D polls a unit and the unit one warp stride on");

// The peer scratch pointer table that follows the staged windows in dynamic LDS rounds up to 16 bytes.
constexpr size_t kDdaLL128ArSplitPeerTblAlign = 16;

// Dynamic LDS of ddaAllReduceTwoShotLL128Split: the staged writer's windows, then the peer scratch pointers.
constexpr size_t ddaLL128ArSplitSharedBytes(unsigned threads, int nRanks) {
  return ddaLL128StageBytes(threads) +
         (((size_t)nRanks * sizeof(void*) + kDdaLL128ArSplitPeerTblAlign - 1) & ~(kDdaLL128ArSplitPeerTblAlign - 1));
}

// A block's (slice, peer) units, slice major: unit k is slice firstSlice + k / (n - 1)
// to peer selfRank + 1 + k % (n - 1).
struct DdaLL128SplitUnit {
  uint32_t unit;       // index in the block's units
  uint32_t slice;      // slice of the shard
  uint32_t peerIndex;  // the peer is selfRank + 1 + peerIndex, mod nRanks
};

// Peer-split LL128 two-shot: the two-shot's layout, wire and fold order over (slice, peer) units.
// Block b owns the same slices of every shard on every rank and waits only on block b, here or on a peer (a block is
// resident whole), so with in-order dispatch the lowest unfinished block always runs and a partly resident grid drains.
// An aborted wait leaves only its warp, before anything that depends on it is stored.
// In place is safe: A reads only the peers' shards and D writes (peer, slice) only after the peer folded my copy of it.
// The rank count is runtime only: the peer loops do not unroll, and a compile-time count adds no resident block.
template <typename T>
__launch_bounds__(kDdaLL128ArThreads) __global__
  void ddaAllReduceTwoShotLL128Split(T* const* __restrict__ peerScratch, T* __restrict__ recvbuff,
                                     const T* __restrict__ sendbuff, size_t shardBytes, int selfRank, int nRanks,
                                     uint32_t* __restrict__ epochDev, int epochLen, size_t slicesTotal, size_t bankSize,
                                     DdaKernelParams params) {
  constexpr int kWire = kDdaLL128WireAsync;
  constexpr uint32_t kPairMask = (1u << kDdaLL128SplitPollSlices) - 1;  // both slices of a poll pair
  const int lane = threadIdx.x % kDdaLL128Warp;
  // Wave uniform where an LL128 warp is a wave (wave32), so the unit bookkeeping below stays in scalar registers.
  const uint32_t warp = __builtin_amdgcn_wavefrontsize() == kDdaLL128Warp ?
                          __builtin_amdgcn_readfirstlane(threadIdx.x / kDdaLL128Warp) :
                          threadIdx.x / kDdaLL128Warp;
  const uint32_t nwarps = blockDim.x / kDdaLL128Warp;
  const bool flagLane = ddaLL128IsFlagLane(lane);

  // The peer table goes in before the epoch read, whose barrier publishes it. The staged windows start line aligned.
  extern __shared__ __attribute__((aligned(kDdaLL128LineBytes))) uint8_t ddaLL128ArStageLds[];
  uint64_t** peerTbl = reinterpret_cast<uint64_t**>(ddaLL128ArStageLds + ddaLL128StageBytes(blockDim.x));
  for (int rank = threadIdx.x; rank < nRanks; rank += blockDim.x) {
    peerTbl[rank] = reinterpret_cast<uint64_t*>(peerScratch[rank]);
  }
  const uint32_t flag32 = ddaGetLLEpochInc(epochDev, blockIdx.x, 1);
  const uint64_t flag = ((uint64_t)flag32 << 32) | (uint64_t)flag32;
  const size_t slotWords = ddaLL128ArTwoShotSlotWords(bankSize, nRanks);
  const uint64_t bankWords = (uint64_t)(flag32 & 1u) * (uint64_t)(bankSize / sizeof(uint64_t));
  const uint64_t bankWordsNext = bankWords + (uint64_t)nRanks * (uint64_t)slotWords;
  DdaLL128Stage stage = ddaLL128StageInit(ddaLL128ArStageLds, warp);

  // A shard fits one scratch slot, so slice and unit counts stay far below 2^32 and their math runs in 32 bits.
  const uint32_t peers = (uint32_t)(nRanks - 1);
  const uint32_t perBlock = (uint32_t)slicesTotal / gridDim.x;
  const uint32_t extra = (uint32_t)slicesTotal % gridDim.x;
  const uint32_t firstSlice = blockIdx.x * perBlock + (blockIdx.x < extra ? blockIdx.x : extra);
  const uint32_t nSlices = perBlock + (blockIdx.x < extra ? 1u : 0u);
  const uint32_t units = nSlices * peers;

  const int8_t* srcBytes = reinterpret_cast<const int8_t*>(sendbuff);
  int8_t* dstBytes = reinterpret_cast<int8_t*>(recvbuff);
  uint64_t* selfBase = peerTbl[selfRank];

  auto unitAt = [&](uint32_t index) {
    return DdaLL128SplitUnit{index, firstSlice + index / peers, index % peers};
  };
  // One warp stride on, without a divide.
  const uint32_t sliceStride = nwarps / peers;
  const uint32_t peerIndexStride = nwarps % peers;
  auto next = [&](DdaLL128SplitUnit item) {
    item.unit += nwarps;
    item.slice += sliceStride;
    item.peerIndex += peerIndexStride;
    if (item.peerIndex >= peers) {
      item.peerIndex -= peers;
      ++item.slice;
    }
    return item;
  };
  auto peerOf = [&](uint32_t peerIndex) {
    const int peer = selfRank + 1 + (int)peerIndex;
    return peer < nRanks ? peer : peer - nRanks;
  };
  auto eltIn = [&](uint32_t slice) { return ddaLL128SliceBytes(shardBytes, slice); };
  auto shardAt = [&](auto* base, int rank, uint32_t slice) {
    return base + (size_t)rank * shardBytes + (size_t)slice * (size_t)kDdaLL128DataBytesPerSlice;
  };
  // Slice `slice` of rank's slot in the staging area `area` words into the scratch at base.
  auto slotAt = [&](uint64_t* base, uint64_t area, int rank, uint32_t slice) {
    return base + area + (uint64_t)rank * (uint64_t)slotWords + (uint64_t)slice * (uint64_t)kDdaLL128WireWordsPerSlice;
  };
  // This lane's words of slice `slice` of rank's slot in my own scratch.
  auto wireAt = [&](uint64_t area, int rank, uint32_t slice) {
    return slotAt(selfBase, area, rank, slice) + 2 * lane;
  };
  auto quit = [&]() {
    ddaLL128StageDrain<kWire>();
    ddaSetLLEpoch(epochDev, epochLen, blockIdx.x, gridDim.x, flag32);
  };

  // A. scatter, units striding by warp, two loads ahead: send each unit's slice of the peer's shard to its first area,
  // slot[me].
  auto load = [&](DdaLL128SplitUnit item, DdaLL128Raw& raw) {
    ddaLL128LoadRaw<int8_t>(raw, shardAt(srcBytes, peerOf(item.peerIndex), item.slice), eltIn(item.slice), lane,
                            flagLane);
  };
  auto send = [&](DdaLL128SplitUnit item, const DdaLL128Raw& raw) {
    uint64_t regs[kDdaLL128WordsPerThread];
    ddaLL128RawToRegs<int8_t>(raw, regs, eltIn(item.slice), lane, flagLane);
    ddaLL128StageFill<kWire>(stage, regs, flag, flagLane, lane);
    ddaLL128StagePush<kWire>(stage, slotAt(peerTbl[peerOf(item.peerIndex)], bankWords, selfRank, item.slice), regs,
                             flag, flagLane, lane);
  };
  if (warp < units) {
    // Look-ahead loads past the last unit reload the one just sent, so every wait count stays exact.
    DdaLL128SplitUnit first = unitAt(warp);
    DdaLL128SplitUnit second = next(first);
    DdaLL128Raw firstRaw;
    DdaLL128Raw secondRaw;
    load(first, firstRaw);
    load(second.unit < units ? second : first, secondRaw);
#pragma unroll 1
    for (;;) {
      send(first, firstRaw);
      const DdaLL128SplitUnit nextFirst = next(second);
      load(nextFirst.unit < units ? nextFirst : first, firstRaw);
      if (second.unit >= units) {
        break;
      }
      send(second, secondRaw);
      const DdaLL128SplitUnit nextSecond = next(nextFirst);
      load(nextSecond.unit < units ? nextSecond : second, secondRaw);
      if (nextFirst.unit >= units) {
        break;
      }
      first = nextFirst;
      second = nextSecond;
    }
  }

  // B. reduce my slices in peer order from me + 1, to recvbuff and to my own second area slot[me] (no peer writes it).
#pragma unroll 1
  for (uint32_t slice = firstSlice + warp; slice < firstSlice + nSlices; slice += nwarps) {
    uint64_t acc[kDdaLL128WordsPerThread] = {};
    ddaLL128LoadRegs<int8_t>(acc, shardAt(srcBytes, selfRank, slice), eltIn(slice), lane, flagLane);
#pragma unroll 1
    for (uint32_t peerIndex = 0; peerIndex < peers; peerIndex += kDdaLL128SplitPollSlices) {
      const uint32_t mask = peerIndex + 1 < peers ? kPairMask : 1u;
      const uint64_t* wires[kDdaLL128SplitPollSlices] = {wireAt(bankWords, peerOf(peerIndex), slice),
                                                         wireAt(bankWords, peerOf(peerIndex + 1), slice)};
      uint64_t polled[kDdaLL128SplitPollSlices][kDdaLL128WordsPerThread];
      if (!ddaLL128PollWireN(wires, mask, polled, flag, lane, params)) {
        quit();
        return;
      }
#pragma unroll
      for (int i = 0; i < kDdaLL128SplitPollSlices; ++i) {
        if ((mask >> i) & 1u) {
#pragma unroll
          for (int k = 0; k < kDdaLL128WordsPerThread; ++k) {
            acc[k] = ddaLL128AddWord<T>(acc[k], polled[i][k]);
          }
        }
      }
    }
    ddaLL128StoreWire(wireAt(bankWordsNext, selfRank, slice), acc, flag, flagLane);
    ddaLL128StoreRegs<int8_t>(shardAt(dstBytes, selfRank, slice), acc, eltIn(slice), lane, flagLane);
  }

  // C. push my reduced slices to every peer, each warp an equal run of the units, polling and filling each slice once.
  const uint32_t runLen = units / nwarps;
  const uint32_t runExtra = units % nwarps;
  const uint32_t runBegin = warp * runLen + (warp < runExtra ? warp : runExtra);
  const uint32_t runEnd = runBegin + runLen + (warp < runExtra ? 1u : 0u);
  if (runBegin < runEnd) {
    DdaLL128SplitUnit item = unitAt(runBegin);
    uint64_t cur[1][kDdaLL128WordsPerThread];
#pragma unroll 1
    for (; item.unit < runEnd; ++item.unit) {
      if (item.unit == runBegin || item.peerIndex == 0) {
        const uint64_t* mine[1] = {wireAt(bankWordsNext, selfRank, item.slice)};
        if (!ddaLL128PollWireN(mine, 1u, cur, flag, lane, params)) {
          quit();
          return;
        }
        // lastOps must stay scalar: an EXEC-masked rung of the wait ladder would wait to 0.
        stage.lastOps = __builtin_amdgcn_readfirstlane(stage.lastOps);
        ddaLL128StageFill<kWire>(stage, cur[0], flag, flagLane, lane);
      }
      ddaLL128StagePush<kWire>(stage, slotAt(peerTbl[peerOf(item.peerIndex)], bankWordsNext, selfRank, item.slice),
                               cur[0], flag, flagLane, lane);
      if (++item.peerIndex == peers) {
        item.peerIndex = 0;
        ++item.slice;
      }
    }
  }

  // D. gather the peers' reduced slices, two units one warp stride apart per round trip.
  if (warp < units) {
#pragma unroll 1
    for (DdaLL128SplitUnit first = unitAt(warp); first.unit < units;) {
      const DdaLL128SplitUnit second = next(first);
      const uint32_t mask = second.unit < units ? kPairMask : 1u;
      const uint64_t* wires[kDdaLL128SplitPollSlices] = {wireAt(bankWordsNext, peerOf(first.peerIndex), first.slice),
                                                         wireAt(bankWordsNext, peerOf(second.peerIndex), second.slice)};
      uint64_t polled[kDdaLL128SplitPollSlices][kDdaLL128WordsPerThread];
      if (!ddaLL128PollWireN(wires, mask, polled, flag, lane, params)) {
        quit();
        return;
      }
      ddaLL128StoreRegs<int8_t>(shardAt(dstBytes, peerOf(first.peerIndex), first.slice), polled[0], eltIn(first.slice),
                                lane, flagLane);
      if ((mask >> 1) & 1u) {
        ddaLL128StoreRegs<int8_t>(shardAt(dstBytes, peerOf(second.peerIndex), second.slice), polled[1],
                                  eltIn(second.slice), lane, flagLane);
      }
      first = next(second);
    }
  }

#if defined(__gfx1250__) || defined(__gfx1250_strict__)
  asm volatile("s_wait_storecnt 0x0" ::: "memory");
#endif
  quit();
}

} // namespace dda::common
