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
static_assert(kDdaLL128SplitPollSlices == 2, "phase D polls unit u and the unit one warp stride on");

// B -> C hand-off: B leaves each reduced wire slice of its block in LDS for C to push, a ready word per slice after.
constexpr size_t kDdaArSplitHandoffImageBytes = 64 * 1024;
constexpr size_t kDdaArSplitHandoffSlices = kDdaArSplitHandoffImageBytes / kDdaLL128WireBytesPerSlice;
constexpr size_t kDdaArSplitHandoffAreaBytes =
  kDdaArSplitHandoffImageBytes + kDdaArSplitHandoffSlices * sizeof(uint32_t);

// Wide B: the block's warps poll a round's (slice, copy) units into this stage, then each slice's owner folds them.
// 48 slice copies: 6 slices of 8 copies.
constexpr size_t kDdaArSplitWideStageBytes = 96 * 1024;
// Chunked wide B stages this many copies of a slice at a time.
constexpr uint32_t kDdaArSplitWideChunkCopies = 8;
// B's forms: pairwise (two copies a poll), wide (all copies of a round's slices staged), or wide in chunks of copies.
enum DdaArSplitB : int { kDdaArSplitBPairwise, kDdaArSplitBWide, kDdaArSplitBWideChunked };

// Slices a wide B round stages: n - 1 copies each (0 when one slice's copies overflow the stage).
constexpr size_t ddaArSplitWideRoundSlices(int nRanks) {
  return kDdaArSplitWideStageBytes / ((size_t)(nRanks - 1) * kDdaLL128WireBytesPerSlice);
}

// Slices a chunked wide B round stages, a chunk of copies each: at most one a warp.
constexpr size_t ddaArSplitWideChunkRoundSlices(unsigned warps) {
  const size_t fit = kDdaArSplitWideStageBytes / (kDdaArSplitWideChunkCopies * kDdaLL128WireBytesPerSlice);
  return fit < warps ? fit : warps;
}

// LDS B form b stages its copies in.
constexpr size_t ddaArSplitBStageBytes(DdaArSplitB b) {
  return b == kDdaArSplitBWide || b == kDdaArSplitBWideChunked ? kDdaArSplitWideStageBytes : 0;
}

// Dynamic LDS of ddaAllReduceTwoShotLL128Split: the staged writer's windows, the hand-off area, B form b's stage, the
// peer table.
constexpr size_t ddaLL128ArSplitSharedBytes(unsigned threads, int nRanks, bool handoff = false,
                                            DdaArSplitB b = kDdaArSplitBPairwise) {
  return ddaLL128StageBytes(threads) + (handoff ? kDdaArSplitHandoffAreaBytes : 0) + ddaArSplitBStageBytes(b) +
         (((size_t)nRanks * sizeof(void*) + 15) & ~(size_t)15);
}

// C's push of a wire slice image in LDS to `wireSlice`: async LDS stores where built, else this lane's 16 B stores.
__device__ __forceinline__ void ddaArSplitHandoffPush(const uint8_t* image, uint64_t* wireSlice, int lane) {
#if ASYNC_COPY_SUPPORTED
  // Every image stays in LDS to the end, so only the async counter's range bounds the pushes in flight.
  asyncWait<kDdaLL128StagePushOps * ((1 << kDdaLL128StageWaitRungs) - 1)>();
  asyncStoreFromLDS<SyncPolicy::Async, kDdaLL128StagePolicy, true>(image, reinterpret_cast<uint8_t*>(wireSlice),
                                                                   kDdaLL128WireBytesPerSlice);
#else
  const uint64_t* pairs = reinterpret_cast<const uint64_t*>(image) + 2 * lane;
#pragma unroll
  for (int u = 0; u < kDdaLL128WordsPerThread; u += 2) {
    ddaLL128StoreAtomic16B(wireSlice + 2 * lane + u * kDdaLL128Warp, pairs[u * kDdaLL128Warp],
                           pairs[u * kDdaLL128Warp + 1]);
  }
#endif
}

// Unit u of a block's (slice, peer) units, slice major: slice s0 + u / (n - 1) to peer selfRank + 1 + u % (n - 1).
struct DdaLL128SplitUnit {
  uint32_t u;   // index in the block's units
  uint32_t s;   // slice of the shard
  uint32_t pi;  // peer index: the peer is selfRank + 1 + pi, mod nRanks
};

// Peer-split LL128 two-shot: the two-shot's layout, wire and fold order over (slice, peer) units.
// Block b owns the same slices of every shard on every rank and waits only on block b, here or on a peer (a block is
// resident whole), so with in-order dispatch the lowest unfinished block always runs and a partly resident grid drains.
// An aborted wait leaves only its warp (in wide B the whole block, at the stage's barrier), before anything that
// depends on it is stored.
// In place is safe: A reads only the peers' shards and D writes (peer, slice) only after the peer folded my copy of it.
// The rank count is runtime only: the peer loops do not unroll, and a compile-time count adds no resident block.
// kDdaLL128FlagBytesNarrow runs the narrow flag: slicesTotal counts 1984 B slices; units and fold order hold.
// kHandoff: B hands each reduced slice to C through LDS; a block owns at most kDdaArSplitHandoffSlices slices.
// kB (with the hand-off): wide B polls a round's copies on every warp into LDS, then each slice's owner adds them in
// peer order from me + 1, as pairwise B does; chunked, kDdaArSplitWideChunkCopies copies at a time. Wide B needs
// ddaArSplitWideRoundSlices(nRanks) >= 1, which the launcher's copies rule keeps.
template <typename T, int kFlagBytes = kDdaLL128FlagBytesWide, bool kHandoff = false,
          DdaArSplitB kB = kDdaArSplitBPairwise>
__launch_bounds__(kDdaLL128ArThreads) __global__
  void ddaAllReduceTwoShotLL128Split(T* const* __restrict__ peerScratch, T* __restrict__ recvbuff,
                                     const T* __restrict__ sendbuff, size_t shardBytes, int selfRank, int nRanks,
                                     uint32_t* __restrict__ epochDev, int epochLen, size_t slicesTotal, size_t bankSize,
                                     DdaKernelParams params) {
  constexpr int kWire = kDdaLL128WireAsync;
  constexpr size_t kSliceData = (size_t)ddaLL128DataBytesPerSliceOf<kFlagBytes>();
  static_assert(kFlagBytes == kDdaLL128FlagBytesWide || sizeof(T) <= 4, "the narrow layout moves whole dwords");
  static_assert(kB == kDdaArSplitBPairwise || kHandoff, "wide B stages after the hand-off area");
  constexpr uint32_t kPairMask = (1u << kDdaLL128SplitPollSlices) - 1;  // both slices of a poll pair
  const int lane = threadIdx.x % kDdaLL128Warp;
  // Wave uniform where an LL128 warp is a wave (wave32), so the unit bookkeeping below stays in scalar registers.
  const uint32_t warp = __builtin_amdgcn_wavefrontsize() == kDdaLL128Warp ?
                          __builtin_amdgcn_readfirstlane(threadIdx.x / kDdaLL128Warp) :
                          threadIdx.x / kDdaLL128Warp;
  const uint32_t nwarps = blockDim.x / kDdaLL128Warp;
  const bool flagLane = ddaLL128IsFlagLane(lane);

  // The peer table goes in before the epoch read, whose barrier publishes it.
  extern __shared__ __attribute__((aligned(128))) uint8_t ddaLL128ArStageLds[];
  uint8_t* const handoffLds = ddaLL128ArStageLds + ddaLL128StageBytes(blockDim.x);
  uint32_t* const handoffReady = reinterpret_cast<uint32_t*>(handoffLds + kDdaArSplitHandoffImageBytes);
  uint8_t* const wideLds = handoffLds + (kHandoff ? kDdaArSplitHandoffAreaBytes : 0);
  uint64_t** peerTbl = reinterpret_cast<uint64_t**>(wideLds + ddaArSplitBStageBytes(kB));
  for (int r = threadIdx.x; r < nRanks; r += blockDim.x) {
    peerTbl[r] = reinterpret_cast<uint64_t*>(peerScratch[r]);
  }
  if constexpr (kHandoff) {
    for (int i = threadIdx.x; i < (int)kDdaArSplitHandoffSlices; i += blockDim.x) {
      handoffReady[i] = 0;
    }
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
  const uint32_t s0 = blockIdx.x * perBlock + (blockIdx.x < extra ? blockIdx.x : extra);
  const uint32_t nSlices = perBlock + (blockIdx.x < extra ? 1u : 0u);
  const uint32_t units = nSlices * peers;

  const int8_t* srcBytes = reinterpret_cast<const int8_t*>(sendbuff);
  int8_t* dstBytes = reinterpret_cast<int8_t*>(recvbuff);
  uint64_t* selfBase = peerTbl[selfRank];

  auto unitAt = [&](uint32_t u) { return DdaLL128SplitUnit{u, s0 + u / peers, u % peers}; };
  // One warp stride on, without a divide.
  const uint32_t strideS = nwarps / peers;
  const uint32_t stridePi = nwarps % peers;
  auto next = [&](DdaLL128SplitUnit x) {
    x.u += nwarps;
    x.s += strideS;
    x.pi += stridePi;
    if (x.pi >= peers) {
      x.pi -= peers;
      ++x.s;
    }
    return x;
  };
  auto peerOf = [&](uint32_t pi) {
    const int p = selfRank + 1 + (int)pi;
    return p < nRanks ? p : p - nRanks;
  };
  auto eltIn = [&](uint32_t s) { return ddaLL128SliceBytesOf<kFlagBytes>(shardBytes, s); };
  auto shardAt = [&](auto* base, int rank, uint32_t s) {
    return base + (size_t)rank * shardBytes + (size_t)s * kSliceData;
  };
  // Slice s of rank's slot in the staging area `area` words into the scratch at base.
  auto slotAt = [&](uint64_t* base, uint64_t area, int rank, uint32_t s) {
    return base + area + (uint64_t)rank * (uint64_t)slotWords + (uint64_t)s * (uint64_t)kDdaLL128WireWordsPerSlice;
  };
  // This lane's words of slice s of rank's slot in my own scratch.
  auto wireAt = [&](uint64_t area, int rank, uint32_t s) { return slotAt(selfBase, area, rank, s) + 2 * lane; };
  auto quit = [&]() {
    ddaLL128StageDrain<kWire>();
    ddaSetLLEpoch(epochDev, epochLen, blockIdx.x, gridDim.x, flag32);
  };

  // A. scatter, units striding by warp, two loads ahead: send slice s of the peer's shard to its first area, slot[me].
  auto load = [&](DdaLL128SplitUnit x, DdaLL128Raw& raw) {
    ddaLL128LoadRaw<int8_t, kFlagBytes>(raw, shardAt(srcBytes, peerOf(x.pi), x.s), eltIn(x.s), lane, flagLane);
  };
  auto send = [&](DdaLL128SplitUnit x, const DdaLL128Raw& raw) {
    uint64_t regs[kDdaLL128WordsPerThread];
    ddaLL128RawToRegs<int8_t, kFlagBytes>(raw, regs, eltIn(x.s), lane, flagLane);
    ddaLL128StageFill<kWire, kFlagBytes>(stage, regs, flag, flagLane, lane);
    ddaLL128StagePush<kWire, kFlagBytes>(stage, slotAt(peerTbl[peerOf(x.pi)], bankWords, selfRank, x.s), regs, flag,
                                         flagLane, lane);
  };
  if (warp < units) {
    // Look-ahead loads past the last unit reload the one just sent, so every wait count stays exact.
    DdaLL128SplitUnit a = unitAt(warp);
    DdaLL128SplitUnit b = next(a);
    DdaLL128Raw ra;
    DdaLL128Raw rb;
    load(a, ra);
    load(b.u < units ? b : a, rb);
#pragma unroll 1
    for (;;) {
      send(a, ra);
      const DdaLL128SplitUnit a2 = next(b);
      load(a2.u < units ? a2 : a, ra);
      if (b.u >= units) {
        break;
      }
      send(b, rb);
      const DdaLL128SplitUnit b2 = next(a2);
      load(b2.u < units ? b2 : b, rb);
      if (a2.u >= units) {
        break;
      }
      a = a2;
      b = b2;
    }
  }

  // B. reduce my slices in peer order from me + 1, to recvbuff and to my own second area slot[me] (no peer writes
  // it), or to the hand-off area.
  if constexpr (kB != kDdaArSplitBPairwise) {
    // B's last step for reduced slice s: its wire image to the hand-off area, marked ready, then to recvbuff.
    auto finish = [&](uint32_t s, uint64_t (&acc)[kDdaLL128WordsPerThread]) {
      uint8_t* const mine = handoffLds + (size_t)(s - s0) * kDdaLL128WireBytesPerSlice + (size_t)lane * 16;
      ddaLL128ForEachWirePair<kFlagBytes>(acc, flag, flagLane, [mine](int word, uint64_t lo, uint64_t hi) {
        uint64_t* pair = reinterpret_cast<uint64_t*>(mine) + word;
        pair[0] = lo;
        pair[1] = hi;
      });
      // The release waits for this wave's image writes (dscnt) before lane 0 marks the slice ready.
      __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
      if (lane == 0) {
        __scoped_atomic_store_n(&handoffReady[s - s0], 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_WRKGRP);
      }
      ddaLL128StoreRegs<int8_t, kFlagBytes>(shardAt(dstBytes, selfRank, s), acc, eltIn(s), lane, flagLane);
    };
    // This lane's words of staged copy u.
    auto copyAt = [&](uint32_t u) {
      return reinterpret_cast<uint64_t*>(wideLds + (size_t)u * kDdaLL128WireBytesPerSlice) +
             (size_t)lane * kDdaLL128WordsPerThread;
    };
    // Polls copy k (from peer me + 1 + k) of slice s into staged copy u; an aborted poll sets `aborted`.
    bool aborted = false;
    auto stageCopy = [&](uint32_t u, uint32_t s, uint32_t k) {
      const uint64_t* at[1] = {wireAt(bankWords, peerOf(k), s)};
      uint64_t v[1][kDdaLL128WordsPerThread];
      if (aborted || !ddaLL128PollWireN<1, kFlagBytes>(at, 1u, v, flag, lane, params)) {
        aborted = true;
        return;
      }
      uint64_t* const dst = copyAt(u);
#pragma unroll
      for (int w = 0; w < kDdaLL128WordsPerThread; ++w) {
        dst[w] = v[0][w];
      }
    };
    // Adds staged copies u0 .. u0 + count - 1 to acc in order.
    auto addStaged = [&](uint64_t (&acc)[kDdaLL128WordsPerThread], uint32_t u0, uint32_t count) {
#pragma unroll 1
      for (uint32_t k = 0; k < count; ++k) {
        const uint64_t* const src = copyAt(u0 + k);
#pragma unroll
        for (int w = 0; w < kDdaLL128WordsPerThread; ++w) {
          acc[w] = ddaLL128AddWord<T>(acc[w], src[w]);
        }
      }
    };
    // Wide: per round every warp stages (slice, copy) units; chunked: a round (a slice a warp at most) stages a chunk
    // of copies of its slices at a time. Each slice's owner adds own + copies in order. Every stage is block wide, so
    // an aborted poll anywhere stops the block before anything that depends on it is stored.
    constexpr bool kChunked = kB == kDdaArSplitBWideChunked;
    const uint32_t roundSlices =
      (uint32_t)(kChunked ? ddaArSplitWideChunkRoundSlices(nwarps) : ddaArSplitWideRoundSlices(nRanks));
    const uint32_t chunk = kChunked ? kDdaArSplitWideChunkCopies : peers;
#pragma unroll 1
    for (uint32_t r0 = 0; r0 < nSlices; r0 += roundSlices) {
      // The round's slices r0 .. r0 + rs - 1; chunked, warp i owns slice r0 + i through the round's chunks of copies.
      const uint32_t rs = nSlices - r0 < roundSlices ? nSlices - r0 : roundSlices;
      uint64_t acc[kDdaLL128WordsPerThread] = {};
      if constexpr (kChunked) {
        if (warp < rs) {
          ddaLL128LoadRegs<int8_t, kFlagBytes>(acc, shardAt(srcBytes, selfRank, s0 + r0 + warp),
                                               eltIn(s0 + r0 + warp), lane, flagLane);
        }
      }
#pragma unroll 1
      for (uint32_t k0 = 0; k0 < peers; k0 += chunk) {
        const uint32_t kc = peers - k0 < chunk ? peers - k0 : chunk;
#pragma unroll 1
        for (uint32_t u = warp; u < rs * kc; u += nwarps) {
          stageCopy(u, s0 + r0 + u / kc, k0 + u % kc);
        }
        if (__syncthreads_or(aborted)) {
          quit();
          return;
        }
        if constexpr (kChunked) {
          if (warp < rs) {
            addStaged(acc, warp * kc, kc);
          }
        } else {
#pragma unroll 1
          for (uint32_t i = warp; i < rs; i += nwarps) {
            const uint32_t s = s0 + r0 + i;
            uint64_t own[kDdaLL128WordsPerThread] = {};
            ddaLL128LoadRegs<int8_t, kFlagBytes>(own, shardAt(srcBytes, selfRank, s), eltIn(s), lane, flagLane);
            addStaged(own, i * kc, kc);
            finish(s, own);
          }
        }
        // The next stage overwrites this one.
        __syncthreads();
      }
      if constexpr (kChunked) {
        if (warp < rs) {
          finish(s0 + r0 + warp, acc);
        }
      }
    }
  } else {
#pragma unroll 1
    for (uint32_t s = s0 + warp; s < s0 + nSlices; s += nwarps) {
      uint64_t acc[kDdaLL128WordsPerThread] = {};
      ddaLL128LoadRegs<int8_t, kFlagBytes>(acc, shardAt(srcBytes, selfRank, s), eltIn(s), lane, flagLane);
#pragma unroll 1
      for (uint32_t pi = 0; pi < peers; pi += kDdaLL128SplitPollSlices) {
        const uint32_t mask = pi + 1 < peers ? kPairMask : 1u;
        const uint64_t* at[kDdaLL128SplitPollSlices] = {wireAt(bankWords, peerOf(pi), s),
                                                        wireAt(bankWords, peerOf(pi + 1), s)};
        uint64_t v[kDdaLL128SplitPollSlices][kDdaLL128WordsPerThread];
        if (!ddaLL128PollWireN<kDdaLL128SplitPollSlices, kFlagBytes>(at, mask, v, flag, lane, params)) {
          quit();
          return;
        }
#pragma unroll
        for (int i = 0; i < kDdaLL128SplitPollSlices; ++i) {
          if ((mask >> i) & 1u) {
#pragma unroll
            for (int k = 0; k < kDdaLL128WordsPerThread; ++k) {
              acc[k] = ddaLL128AddWord<T>(acc[k], v[i][k]);
            }
          }
        }
      }
      if constexpr (kHandoff) {
        // The wire image C would read back from my own slot, written once to the hand-off area, then marked ready.
        uint8_t* const mine = handoffLds + (size_t)(s - s0) * kDdaLL128WireBytesPerSlice + (size_t)lane * 16;
        ddaLL128ForEachWirePair<kFlagBytes>(acc, flag, flagLane, [mine](int word, uint64_t lo, uint64_t hi) {
          uint64_t* pair = reinterpret_cast<uint64_t*>(mine) + word;
          pair[0] = lo;
          pair[1] = hi;
        });
        // The release waits for this wave's image writes (dscnt) before lane 0 marks the slice ready.
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
        if (lane == 0) {
          __scoped_atomic_store_n(&handoffReady[s - s0], 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_WRKGRP);
        }
      } else {
        ddaLL128StoreWire<kFlagBytes>(wireAt(bankWordsNext, selfRank, s), acc, flag, flagLane);
      }
      ddaLL128StoreRegs<int8_t, kFlagBytes>(shardAt(dstBytes, selfRank, s), acc, eltIn(s), lane, flagLane);
    }
  }

  // C. push my reduced slices to every peer, each warp an equal run of the units, getting each slice once (polled,
  // or ready in the hand-off area).
  const uint32_t runLen = units / nwarps;
  const uint32_t runExtra = units % nwarps;
  const uint32_t lo = warp * runLen + (warp < runExtra ? warp : runExtra);
  const uint32_t hi = lo + runLen + (warp < runExtra ? 1u : 0u);
  if constexpr (kHandoff) {
    // Each slice's image once B marked it ready (an abort leaves the wait), then straight from LDS to every peer.
    const uint8_t* image = handoffLds;
#pragma unroll 1
    for (DdaLL128SplitUnit x = unitAt(lo); x.u < hi; ++x.u) {
      if (x.u == lo || x.pi == 0) {
        int spins = 0;
        while (__scoped_atomic_load_n(&handoffReady[x.s - s0], __ATOMIC_RELAXED, __MEMORY_SCOPE_WRKGRP) == 0) {
          if (params.abortSpinTick(spins)) {
            quit();
            return;
          }
        }
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
        image = handoffLds + (size_t)(x.s - s0) * kDdaLL128WireBytesPerSlice;
      }
      ddaArSplitHandoffPush(image, slotAt(peerTbl[peerOf(x.pi)], bankWordsNext, selfRank, x.s), lane);
      if (++x.pi == peers) {
        x.pi = 0;
        ++x.s;
      }
    }
  } else if (lo < hi) {
    DdaLL128SplitUnit x = unitAt(lo);
    uint64_t cur[1][kDdaLL128WordsPerThread];
#pragma unroll 1
    for (; x.u < hi; ++x.u) {
      if (x.u == lo || x.pi == 0) {
        const uint64_t* mine[1] = {wireAt(bankWordsNext, selfRank, x.s)};
        if (!ddaLL128PollWireN<1, kFlagBytes>(mine, 1u, cur, flag, lane, params)) {
          quit();
          return;
        }
        // lastOps must stay scalar: an EXEC-masked rung of the wait ladder would wait to 0.
        stage.lastOps = __builtin_amdgcn_readfirstlane(stage.lastOps);
        ddaLL128StageFill<kWire, kFlagBytes>(stage, cur[0], flag, flagLane, lane);
      }
      ddaLL128StagePush<kWire, kFlagBytes>(stage, slotAt(peerTbl[peerOf(x.pi)], bankWordsNext, selfRank, x.s), cur[0],
                                           flag, flagLane, lane);
      if (++x.pi == peers) {
        x.pi = 0;
        ++x.s;
      }
    }
  }

  // D. gather the peers' reduced slices, two units one warp stride apart per round trip.
  if (warp < units) {
#pragma unroll 1
    for (DdaLL128SplitUnit a = unitAt(warp); a.u < units;) {
      const DdaLL128SplitUnit b = next(a);
      const uint32_t mask = b.u < units ? kPairMask : 1u;
      const uint64_t* at[kDdaLL128SplitPollSlices] = {wireAt(bankWordsNext, peerOf(a.pi), a.s),
                                                      wireAt(bankWordsNext, peerOf(b.pi), b.s)};
      uint64_t v[kDdaLL128SplitPollSlices][kDdaLL128WordsPerThread];
      if (!ddaLL128PollWireN<kDdaLL128SplitPollSlices, kFlagBytes>(at, mask, v, flag, lane, params)) {
        quit();
        return;
      }
      ddaLL128StoreRegs<int8_t, kFlagBytes>(shardAt(dstBytes, peerOf(a.pi), a.s), v[0], eltIn(a.s), lane, flagLane);
      if ((mask >> 1) & 1u) {
        ddaLL128StoreRegs<int8_t, kFlagBytes>(shardAt(dstBytes, peerOf(b.pi), b.s), v[1], eltIn(b.s), lane, flagLane);
      }
      a = next(b);
    }
  }

#if defined(__gfx1250__) || defined(__gfx1250_strict__)
  asm volatile("s_wait_storecnt 0x0" ::: "memory");
#endif
  quit();
}

} // namespace dda::common
