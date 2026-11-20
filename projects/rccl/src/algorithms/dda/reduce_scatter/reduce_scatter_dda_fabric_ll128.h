/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * LL128-protocol reduce-scatter device kernel for the DDA fabric path (gfx1250).
 * Reduce-scatter counterpart of all_reduce_dda_ll128.h: same slice / wire
 * geometry and the same CollCommon_ll128.h pack/unpack helpers, but each
 * peer receives a different shard (the personalized scatter of two-shot
 * all-reduce phase 1) and there is no write-back gather. Rank selfRank owns
 * output shard selfRank, so recvbuff[i] = sum over ranks r of
 * sendbuff_r[selfRank*recvcount + i].
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
#include "device/uniform.h"

namespace dda::common {

// Slot geometry, derived from the scratch bank the host picked. The bank splits
// evenly across ranks and the per-rank stride is floored to a whole number of
// slices, so a slot always begins on a slice boundary and the 16B wire accesses
// stay aligned however nRanks divides the bank. One staging area (publish then
// reduce), matching the all-reduce one-shot layout rather than the two-shot's
// halved slots.
constexpr size_t ddaLL128RsSlotWords(size_t bankSize, int nRanks) {
  return ddaLL128SlotWords(bankSize, nRanks, 1);
}

// Block size the kernel is compiled for (its __launch_bounds__), so the launcher's default and its cap.
constexpr unsigned kDdaLL128RsMaxThreads = 512;

// Phase-2 poll batch: peers read and re-read together. fp8 takes half, its per-word upconverts need the registers.
template <typename T>
inline constexpr int kDdaLL128RsPollBatch = is_fp8_v<T> ? 4 : 8;

// (rank + 1) % nRanks without the divide.
constexpr int ddaLL128RsNextRank(int rank, int nRanks) {
  return rank + 1 == nRanks ? 0 : rank + 1;
}

// Fill a batch from walk index base; entries past nRanks repeat the next peer. Returns the next batch's first peer.
template <int kBatch>
__host__ __device__ __forceinline__ constexpr int ddaLL128RsBatchPeers(int peer, int base, int nRanks,
                                                                       int (&peers)[kBatch], bool (&valid)[kBatch]) {
#pragma unroll
  for (int k = 0; k < kBatch; ++k) {
    valid[k] = base + k < nRanks;
    peers[k] = peer;
    if (valid[k]) {
      peer = ddaLL128RsNextRank(peer, nRanks);
    }
  }
  return peer;
}

// Wait until every valid flag in the batch is current; vr then holds the landed payload. Returns false on abort.
// Every line is read, and all the batch's loads issue before its first flag check.
template <int kBatch>
__device__ __forceinline__ bool ddaLL128RsWaitBatch(const uint64_t* const (&lines)[kBatch], const bool (&valid)[kBatch],
                                                    int laneWord, uint64_t flag, bool flagLane, DdaKernelParams params,
                                                    uint64_t (&vr)[kBatch][kDdaLL128WordsPerThread]) {
  int spins = 0;
  while (true) {
#pragma unroll
    for (int k = 0; k < kBatch; ++k) {
#pragma unroll
      for (int u = 0; u < kDdaLL128WordsPerThread; u += 2) {
        ddaLL128LoadAtomic16B(lines[k] + laneWord + u * kDdaLL128Warp, vr[k][u], vr[k][u + 1]);
      }
    }
    bool stale = false;
#pragma unroll
    for (int k = 0; k < kBatch; ++k) {
#pragma unroll
      for (int u = 0; u < kDdaLL128WordsPerThread; u += 2) {
        stale |= valid[k] && flagLane && vr[k][u + 1] != flag;
      }
    }
    if (!__any(stale)) {
#if !RCCL_HAVE_COOPERATIVE_ATOMIC_BUILTINS
#pragma unroll
      for (int k = 0; k < kBatch; ++k) {
#pragma unroll
        for (int u = 0; u < kDdaLL128WordsPerThread; u += 2) {
          ddaLL128LoadAtomic16B(lines[k] + laneWord + u * kDdaLL128Warp, vr[k][u], vr[k][u + 1]);
        }
      }
#endif
      return true;
    }
    if (params.abortSpinTick(spins)) {
      return false;
    }
  }
}

// LL128 reduce-scatter kernel. One warp owns one slice of a shard at a time, so
// the launcher sizes the grid on the shard's slices, not the message's.
//
// Phase 1 (scatter): send peer p the shard p owns, into p's scratch at slot
// selfRank, as LL128 lines carrying the epoch flag.
// Phase 2 (reduce): poll my slots for the peers' copies of my shard in batches,
// fold them onto my own in rank order, and unpack the result into recvbuff.
//
// Flag polling provides all the cross-rank ordering, so no GPU barrier is used.
// Scratch is double buffered: bank = flag & 1. Self does not round-trip through
// scratch; its contribution seeds the accumulator straight from sendbuff.
//
// A warp owns the same slice index in both phases, and phase 1 only ever reads
// peer shards while phase 2 only ever writes recvbuff (this rank's shard), so an
// in-place call (recvbuff aliasing sendbuff[selfRank]) never has one warp's
// write land on a range another warp still has to read.
template <typename T, int NRANKS_CT>
__launch_bounds__(kDdaLL128RsMaxThreads)
  __global__ void ddaReduceScatterFabricLL128(T* const* __restrict__ peerScratch, // ddaPeerPtrsDev: scratch bases
                                              T* __restrict__ recvbuff, // local user output (one shard)
                                              const T* __restrict__ sendbuff, // local user input (nRanks shards)
                                              size_t shardBytes, // per-rank shard payload; multiple of 16
                                              int selfRank, int nRanksRt,
                                              uint32_t* __restrict__ epochDev, // per-block LL epoch cells
                                              int epochLen, // number of cells in epochDev
                                              size_t slicesTotal, // slices per shard this call uses
                                              size_t bankSize, // scratch bank size (from host)
                                              DdaKernelParams params) {

  const int nRanks = NRANKS_CT ? NRANKS_CT : nRanksRt;

  const int tid = threadIdx.x;
  const int lane = tid % kDdaLL128Warp;
#if defined(__gfx1250__) || defined(__gfx1250_strict__)
  // Wave32: a warp is one wave, so its index is wave-uniform and slice and scratch line addresses stay scalar.
  const int warp = __builtin_amdgcn_readfirstlane(tid / kDdaLL128Warp);
#else
  const int warp = tid / kDdaLL128Warp;
#endif
  const int nwarps = blockDim.x / kDdaLL128Warp;
  const bool flagLane = ddaLL128IsFlagLane(lane);

  const uint32_t flag32 = ddaGetLLEpochInc(epochDev, blockIdx.x, 1);
  const uint64_t flag = ((uint64_t)flag32 << 32) | (uint64_t)flag32;
  // ddaBankSize() floors the bank to a multiple of 128B, so it is always a whole
  // number of 8B words and the slot stride below is a whole number of slices.
  const size_t slotWords = ddaLL128RsSlotWords(bankSize, nRanks);
  const uint64_t bankWords = (uint64_t)(flag32 & 1u) * (uint64_t)(bankSize / sizeof(uint64_t));

  // Slices stride by warp across the whole grid. The bound has to be the slice
  // count, not the warp count: the grid is capped at ddaFabricMaxBlocks, so a
  // large shard has more slices than warps and each warp must take several.
  const size_t gwarp = (size_t)blockIdx.x * (size_t)nwarps + (size_t)warp;
  const size_t wstride = (size_t)gridDim.x * (size_t)nwarps;

  const int8_t* srcBytes = reinterpret_cast<const int8_t*>(sendbuff);
  int8_t* dstBytes = reinterpret_cast<int8_t*>(recvbuff);
  const uint64_t* gatherBase = reinterpret_cast<const uint64_t*>(peerScratch[selfRank]) + bankWords;

  // Phase 1: send each peer the shard that peer owns.
  for (size_t s = gwarp; s < slicesTotal; s += wstride) {
    const size_t sliceByte = s * (size_t)kDdaLL128DataBytesPerSlice;
    const size_t rem = shardBytes - sliceByte;
    const int eltInSlice = rem < (size_t)kDdaLL128DataBytesPerSlice ? (int)rem : kDdaLL128DataBytesPerSlice;
    const size_t sliceWire = s * (size_t)kDdaLL128WireWordsPerSlice;

    // 8 peers per batch fits the register budget of the 512-thread launch bound.
    constexpr int kSendBatch = kDdaLLPeerBatch;
    for (int base = 1; base < nRanks; base += kSendBatch) {
      uint64_t regs[kSendBatch][kDdaLL128WordsPerThread];
#pragma unroll
      for (int i = 0; i < kSendBatch; ++i) {
        const int r = base + i;
        // The runtime kernel also loads past the last peer, a shard the modulo wraps to and nothing sends, so no
        // branch splits its batch.
        if (NRANKS_CT == 0 || r < nRanks) {
          const int peer = (selfRank + r) % nRanks;
          ddaLL128LoadRegsBegin<int8_t>(regs[i], srcBytes + (size_t)peer * shardBytes + sliceByte, eltInSlice, lane,
                                        flagLane);
        }
      }
      // Keeps every flag-lane shuffle below the batch's loads, so no peer's loads wait on another peer's.
      __builtin_amdgcn_sched_barrier(0);
#pragma unroll
      for (int i = 0; i < kSendBatch; ++i) {
        const int r = base + i;
        if (r < nRanks) {
          ddaLL128LoadRegsFinish(regs[i], flagLane);
          // peer is wave-uniform; readfirstlane makes that explicit so the scratch base is a scalar load.
          const int peer = __builtin_amdgcn_readfirstlane((selfRank + r) % nRanks);
          // A scalar slot base keeps the lane offset a 32-bit vector offset.
          uint64_t* scatterSlot = wave::uniform(reinterpret_cast<uint64_t*>(peerScratch[peer]) + bankWords +
            (uint64_t)selfRank * (uint64_t)slotWords + sliceWire);
          ddaLL128StoreWire(scatterSlot + 2 * lane, regs[i], flag, flagLane);
        }
      }
    }
  }

  // Phase 2: fold every copy of my shard into recvbuff, peers selfRank + 1, selfRank + 2, ... in poll batches.
  const int laneWord = 2 * lane;
  // A fixed-rank instance polls at most its peers in one batch.
  constexpr int kPollBatch =
    NRANKS_CT && NRANKS_CT - 1 < kDdaLL128RsPollBatch<T> ? NRANKS_CT - 1 : kDdaLL128RsPollBatch<T>;
  const int firstRecvPeer = ddaLL128RsNextRank(selfRank, nRanks);
  // Abort exits via the common tail: a second ddaSetLLEpoch call would keep its epoch cell index live and spill it.
  bool aborted = false;
  for (size_t s = gwarp; s < slicesTotal; s += wstride) {
    const size_t sliceByte = s * (size_t)kDdaLL128DataBytesPerSlice;
    const size_t rem = shardBytes - sliceByte;
    const int eltInSlice = rem < (size_t)kDdaLL128DataBytesPerSlice ? (int)rem : kDdaLL128DataBytesPerSlice;
    const size_t sliceWire = s * (size_t)kDdaLL128WireWordsPerSlice;

    // Seed with my own shard; peers are folded on top.
    // Its loads stay in flight through the first poll batch, which is when it is finished.
    uint64_t acc[kDdaLL128WordsPerThread];
    ddaLL128LoadRegsBegin<int8_t>(acc, srcBytes + (size_t)selfRank * shardBytes + sliceByte, eltInSlice, lane,
                                  flagLane);

    int peer = firstRecvPeer;
    for (int base = 1; base < nRanks; base += kPollBatch) {
      int peers[kPollBatch];
      bool valid[kPollBatch];
      peer = ddaLL128RsBatchPeers(peer, base, nRanks, peers, valid);
      const uint64_t* lines[kPollBatch];
#pragma unroll
      for (int k = 0; k < kPollBatch; ++k) {
        // Every line is read to stay branch-free, and a past-the-end one is my own unused slot, ignored.
        lines[k] = gatherBase + (uint64_t)peers[k] * (uint64_t)slotWords + sliceWire;
      }
      uint64_t vr[kPollBatch][kDdaLL128WordsPerThread];
      if (!ddaLL128RsWaitBatch(lines, valid, laneWord, flag, flagLane, params, vr)) {
        aborted = true;
        break;
      }
      if (base == 1) {
        ddaLL128LoadRegsFinish(acc, flagLane);
      }
      // Fold word by word, each word taking the batch's peers in rank order, so every element sums in the same order.
      // On a flag lane the odd words hold flags rather than payload, so the sums
      // landing there are meaningless; ddaLL128StoreRegs re-derives those slots
      // from the even ones and never writes them out, so folding blind is cheaper
      // than predicating the loop.
#pragma unroll
      for (int u = 0; u < kDdaLL128WordsPerThread; ++u) {
#pragma unroll
        for (int k = 0; k < kPollBatch; ++k) {
          // A select rather than a branch, so the batch's reads stay in flight together.
          const uint64_t sum = ddaLL128AddWord<T>(acc[u], vr[k][u]);
          acc[u] = valid[k] ? sum : acc[u];
        }
        if constexpr (is_fp8_v<T>) {
          // Fold fp8 per word: hoisted pk8 upconverts take 8 VGPRs each and spill; mask 0x20 lets VMEM reads cross.
          __builtin_amdgcn_sched_barrier(0x20);
        }
      }
    }
    if (aborted) {
      break;
    }

    ddaLL128StoreRegs<int8_t>(dstBytes + sliceByte, acc, eltInSlice, lane, flagLane);
  }

#if defined(__gfx1250__) || defined(__gfx1250_strict__)
  asm volatile("s_wait_storecnt 0x0" ::: "memory");
#endif

  ddaSetLLEpoch(epochDev, epochLen, blockIdx.x, gridDim.x, flag32);
}

} // namespace dda::common
