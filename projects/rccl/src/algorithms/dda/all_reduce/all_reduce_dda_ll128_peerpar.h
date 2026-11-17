/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Peer-parallel LL128 one-shot and two-shot all-reduce kernels for the DDA
 * fabric path.
 *
 * The flat tiers (all_reduce_dda_ll128.h, all_reduce_dda_ll128_twoshot.h) give
 * a slice to one warp, which then walks every peer in turn: nRanks - 1 stores
 * and, worse, nRanks - 1 flag polls back to back, each a full fabric round trip.
 * Their per-slice latency therefore grows linearly with nRanks, while all-gather
 * and all-to-all, which give each peer its own warps, stay flat.
 *
 * Here a slice gets a group of P = ddaLL128ArPeerWarps() warps instead and the
 * peer loops are split across the group: warp pw sends to, and polls, the ranks
 * congruent to pw mod P, so a group pays one poll round trip per P ranks. The
 * polled copies meet in LDS and are folded there, still in rank order
 * 0..nRanks-1, so every rank rounds identically.
 *
 * Wire format, slot layout, epoch/bank handling and the pack/poll/unpack helpers
 * are the flat tiers', unchanged, so these kernels are drop-in replacements.
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
// Slot geometry is shared with the flat tiers, not redefined here.
#include "algorithms/dda/all_reduce/all_reduce_dda_ll128.h"
#include "algorithms/dda/all_reduce/all_reduce_dda_ll128_twoshot.h"

namespace dda::common {

// Widest block the peer-parallel kernels take: it sizes their static LDS.
constexpr int kDdaLL128ArPeerMaxThreads = 512;
constexpr int kDdaLL128ArPeerMaxWarps = kDdaLL128ArPeerMaxThreads / kDdaLL128Warp;
// One 16B wire pair (lo, hi); 16B-aligned so LDS accesses are 16B wide.
struct alignas(16) DdaLL128Pair {
  uint64_t lo;
  uint64_t hi;
};
// 16B pairs of one slice.
constexpr int kDdaLL128PairsPerSlice = kDdaLL128WireWordsPerSlice / 2;
// Pairs of a slice one thread folds. Blocks come in 64-thread granules and nRanks >= 2, so a group is at least two
// warps.
constexpr int kDdaLL128ArPeerFoldPairs = kDdaLL128PairsPerSlice / (2 * kDdaLL128Warp);

// Warps per slice group: one per rank, up to the whole block.
__host__ __device__ constexpr int ddaLL128ArPeerWarps(int nRanks, int nwarps) {
  return nRanks < nwarps ? nRanks : nwarps;
}

// Packed add of two 16B pairs, word by word.
template <typename T>
__device__ __forceinline__ DdaLL128Pair ddaLL128ArAddPair(const DdaLL128Pair& a, const DdaLL128Pair& b) {
  return {ddaLL128AddWord<T>(a.lo, b.lo), ddaLL128AddWord<T>(a.hi, b.hi)};
}

// Poll one slice. With cooperative loads, all of the slice's loads issue before any flag is checked and the payload
// lands with its flag, so there is no re-read; without them the plain poll re-reads once the flags are current.
__device__ __forceinline__ bool ddaLL128ArPeerPoll(const uint64_t* wire, uint64_t (&vr)[kDdaLL128WordsPerThread],
                                                   uint64_t flag, int lane, DdaKernelParams params) {
#if RCCL_HAVE_COOPERATIVE_ATOMIC_BUILTINS
  return ddaLL128PollWireBatched(wire, vr, flag, lane, params);
#else
  return ddaLL128PollWire(wire, vr, flag, lane, params);
#endif
}

// Rank-order fold of one slice across a warp group, through LDS. In rounds of P
// ranks, warp pw parks rank base + pw's copy of the slice (its own from `self`, a
// peer's polled from that peer's slot under `gather`) in its LDS slice, and after
// a barrier the group's threads fold their pairs of the parked copies in rank
// order, so each rank rounds the same partial sums. The sum lands in the store
// warp's LDS slice, lds[warp - pw + (nRanks - 1) % P], until that warp parks
// again.
//
// The barriers are block-wide, so every warp of the block calls this the same
// number of times; `live` is false for a warp with no slice. Returns false, on
// every warp alike, if a poll saw the abort flag.
template <typename T, int NRANKS_CT>
__device__ __forceinline__ bool ddaLL128ArPeerFold(DdaLL128Pair (*lds)[kDdaLL128PairsPerSlice], int& aborted,
                                                   bool live, int nRanks, int selfRank, int P, int warp,
                                                   const int8_t* self, int eltInSlice, const uint64_t* gather,
                                                   size_t slotWords, uint64_t flag, DdaKernelParams params) {
  const int lane = threadIdx.x % kDdaLL128Warp;
  const bool flagLane = ddaLL128IsFlagLane(lane);
  const int group = warp / P;
  const int pw = warp % P;
  const int storeWarp = (nRanks - 1) % P;
  // This thread's place among the group's threads.
  const int gt = pw * kDdaLL128Warp + lane;
  const int gThreads = P * kDdaLL128Warp;

  DdaLL128Pair acc[kDdaLL128ArPeerFoldPairs] = {};
  for (int base = 0; base < nRanks; base += P) {
    const int rank = base + pw;
    if (live && rank < nRanks) {
      uint64_t vr[kDdaLL128WordsPerThread] = {};
      if (rank == selfRank) {
        ddaLL128LoadRegsBatched<int8_t>(vr, self, eltInSlice, lane, flagLane);
      } else if (!ddaLL128ArPeerPoll(gather + (uint64_t)rank * (uint64_t)slotWords + 2 * lane, vr, flag, lane,
                                     params)) {
        aborted = 1;
      }
#pragma unroll
      for (int u = 0; u < kDdaLL128Pairs; ++u) {
        lds[warp][u * kDdaLL128Warp + lane] = {vr[2 * u], vr[2 * u + 1]};
      }
    }
    __syncthreads();
    if (aborted) {
      return false;
    }

    // On a flag lane the copies hold a flag in each pair's hi word, not payload; that
    // sum is never stored, since ddaLL128StoreRegs re-derives those slots.
    if (live) {
      const int nIn = nRanks - base < P ? nRanks - base : P;
      constexpr int kRoundMax =
        NRANKS_CT != 0 && NRANKS_CT < kDdaLL128ArPeerMaxWarps ? NRANKS_CT : kDdaLL128ArPeerMaxWarps;
#pragma unroll
      for (int k = 0; k < kDdaLL128ArPeerFoldPairs; ++k) {
        const int q = gt + k * gThreads;
        if (q < kDdaLL128PairsPerSlice) {
#pragma unroll
          for (int j = 0; j < kRoundMax; ++j) {
            if (j < nIn) {
              const DdaLL128Pair v = lds[group * P + j][q];
              acc[k] = (base == 0 && j == 0) ? v : ddaLL128ArAddPair<T>(acc[k], v);
            }
          }
          // Only this thread reads pair q, and the store warp parks nothing more until it has read the sum.
          if (base + P >= nRanks) {
            lds[group * P + storeWarp][q] = acc[k];
          }
        }
      }
    }
    // Ends the round: the next one overwrites lds, and the caller reads the sum.
    __syncthreads();
  }
  return true;
}

// LL128 peer-parallel one-shot all-reduce kernel. Same contract and scratch
// layout as ddaAllReduceFlatLL128, but a group of P warps owns one slice at a
// time; the grid is 1D over slice groups.
//
// Phase 1 (publish): pack the slice into registers once; warp pw of the group
// pushes it to peers selfRank + 1 + pw, selfRank + 1 + pw + P, ...
// Phase 2 (reduce): in rounds of P ranks, warp pw parks rank base + pw's copy
// of the slice in its LDS slice (its own from sendbuff, a peer's by polling our
// scratch), and after a barrier the group's threads fold their pairs of the
// parked copies in rank order. The sum lands in the store warp's LDS slice,
// which that warp unpacks into recvbuff.
//
// The barriers are block-wide, so every warp runs the same number of phase-2
// iterations. A slice belongs to one group in both phases, and only that group
// writes it, after all of its reads, so an in-place call is safe.
template <typename T, int NRANKS_CT>
__launch_bounds__(kDdaLL128ArPeerMaxThreads)
  __global__ void ddaAllReduceFlatLL128PeerPar(T* const* __restrict__ peerScratch, // ddaPeerPtrsDev: scratch bases
                                               T* __restrict__ recvbuff, // local user output
                                               const T* __restrict__ sendbuff, // local user input
                                               size_t bytes, // full-message payload; multiple of 16
                                               int selfRank, int nRanksRt,
                                               uint32_t* __restrict__ epochDev, // per-block LL epoch cells
                                               int epochLen, // number of cells in epochDev
                                               size_t slicesTotal, // slices this call actually uses
                                               size_t bankSize, // scratch bank size (from host)
                                               DdaKernelParams params) {
  // One wire slice per warp: the copy it parked this round.
  __shared__ DdaLL128Pair lds[kDdaLL128ArPeerMaxWarps][kDdaLL128PairsPerSlice];
  __shared__ int aborted;

  const int nRanks = NRANKS_CT ? NRANKS_CT : nRanksRt;

  const int tid = threadIdx.x;
  const int lane = tid % kDdaLL128Warp;
#if defined(__gfx1250__) || defined(__gfx1250_strict__)
  // Wave32: a warp is one wave, so its index is wave-uniform and the scratch addresses stay scalar.
  const int warp = __builtin_amdgcn_readfirstlane(tid / kDdaLL128Warp);
#else
  const int warp = tid / kDdaLL128Warp;
#endif
  const int nwarps = blockDim.x / kDdaLL128Warp;
  const bool flagLane = ddaLL128IsFlagLane(lane);

  // Warp pw of group `group` handles ranks congruent to pw mod P. Leftover warps, when P does not divide the block,
  // only join the barriers.
  const int P = ddaLL128ArPeerWarps(nRanks, nwarps);
  const int groups = nwarps / P;
  const int group = warp / P;
  const int pw = warp % P;
  const bool inGroup = group < groups;
  // The warp that unpacks the group's sum: the one with the fewest peers to publish to, so its recvbuff stores do
  // not queue behind phase 1's.
  const int storeWarp = (nRanks - 1) % P;

  if (tid == 0) {
    aborted = 0;
  }
  // Its barrier also publishes `aborted`.
  const uint32_t flag32 = ddaGetLLEpochInc(epochDev, blockIdx.x, 1);
  const uint64_t flag = ((uint64_t)flag32 << 32) | (uint64_t)flag32;
  const size_t slotWords = ddaLL128ArSlotWords(bankSize, nRanks);
  const uint64_t bankWords = (uint64_t)(flag32 & 1u) * (uint64_t)(bankSize / sizeof(uint64_t));
  const size_t sliceGroups = (slicesTotal + groups - 1) / groups;

  const int8_t* srcBytes = reinterpret_cast<const int8_t*>(sendbuff);
  int8_t* dstBytes = reinterpret_cast<int8_t*>(recvbuff);
  const uint64_t* gatherBase = reinterpret_cast<const uint64_t*>(peerScratch[selfRank]) + bankWords;

  // Phase 1: pack each slice once, then push it to this warp's share of the peers' slot[selfRank].
  for (size_t g = blockIdx.x; g < sliceGroups; g += gridDim.x) {
    const size_t s = g * (size_t)groups + (size_t)group;
    if (!inGroup || s >= slicesTotal || 1 + pw >= nRanks) {
      break;
    }
    const size_t dataByte = s * (size_t)kDdaLL128DataBytesPerSlice;
    const size_t rem = bytes - dataByte;
    const int eltInSlice = rem < (size_t)kDdaLL128DataBytesPerSlice ? (int)rem : kDdaLL128DataBytesPerSlice;
    uint64_t regs[kDdaLL128WordsPerThread] = {};
    ddaLL128LoadRegsBatched<int8_t>(regs, srcBytes + dataByte, eltInSlice, lane, flagLane);

    for (int r = 1 + pw; r < nRanks; r += P) {
      const int peer = (selfRank + r) % nRanks;
      uint64_t* scatterSlot =
        reinterpret_cast<uint64_t*>(peerScratch[peer]) + bankWords + (uint64_t)selfRank * (uint64_t)slotWords;
      ddaLL128StoreWire(scatterSlot + s * (size_t)kDdaLL128WireWordsPerSlice + 2 * lane, regs, flag, flagLane);
    }
  }

  // Phase 2: gather every rank's copy of the slice P at a time and fold them in.
  for (size_t g = blockIdx.x; g < sliceGroups; g += gridDim.x) {
    const size_t s = g * (size_t)groups + (size_t)group;
    const bool live = inGroup && s < slicesTotal;
    const size_t dataByte = s * (size_t)kDdaLL128DataBytesPerSlice;
    const size_t rem = live ? bytes - dataByte : 0;
    const int eltInSlice = rem < (size_t)kDdaLL128DataBytesPerSlice ? (int)rem : kDdaLL128DataBytesPerSlice;

    if (!ddaLL128ArPeerFold<T, NRANKS_CT>(lds, aborted, live, nRanks, selfRank, P, warp, srcBytes + dataByte,
                                          eltInSlice, gatherBase + s * (size_t)kDdaLL128WireWordsPerSlice, slotWords,
                                          flag, params)) {
      ddaSetLLEpoch(epochDev, epochLen, blockIdx.x, gridDim.x, flag32);
      return;
    }

    if (live && pw == storeWarp) {
      uint64_t out[kDdaLL128WordsPerThread];
#pragma unroll
      for (int u = 0; u < kDdaLL128Pairs; ++u) {
        const DdaLL128Pair v = lds[warp][u * kDdaLL128Warp + lane];
        out[2 * u] = v.lo;
        out[2 * u + 1] = v.hi;
      }
      ddaLL128StoreRegs<int8_t>(dstBytes + dataByte, out, eltInSlice, lane, flagLane);
    }
  }

#if defined(__gfx1250__) || defined(__gfx1250_strict__)
  asm volatile("s_wait_storecnt 0x0" ::: "memory");
#endif

  ddaSetLLEpoch(epochDev, epochLen, blockIdx.x, gridDim.x, flag32);
}

// LL128 peer-parallel two-shot all-reduce kernel. Same contract and scratch
// layout as ddaAllReduceTwoShotLL128 (cooperative writer).
//
// Phase 1 (publish): one warp per (slice, peer) item across the whole grid sends
// the peer that slice of the shard it owns.
// Phase 2 (reduce and publish): a group of P warps per slice folds every rank's
// copy of my shard's slice through LDS in rank order (ddaLL128ArPeerFold); each
// warp hands the sum to its share of the peers, and the store warp unpacks it
// into my part of recvbuff.
// Phase 3 (gather): the phase-1 items again, now polling the peer's reduced
// shard and unpacking it.
//
// In-place safety: a (slice, peer) item maps to the same warp in phases 1 and 3,
// so sendbuff shard `peer` is read before recvbuff shard `peer` is written, in
// program order; my own shard is read before phase 2's fold barrier and written
// after it.
template <typename T, int NRANKS_CT>
__launch_bounds__(kDdaLL128ArPeerMaxThreads)
  __global__ void ddaAllReduceTwoShotLL128PeerPar(T* const* __restrict__ peerScratch, // ddaPeerPtrsDev: scratch bases
                                                  T* __restrict__ recvbuff, // local user output
                                                  const T* __restrict__ sendbuff, // local user input
                                                  size_t shardBytes, // per-rank shard payload; multiple of 16
                                                  int selfRank, int nRanksRt,
                                                  uint32_t* __restrict__ epochDev, // per-block LL epoch cells
                                                  int epochLen, // number of cells in epochDev
                                                  size_t slicesTotal, // slices per shard this call uses
                                                  size_t bankSize, // scratch bank size (from host)
                                                  DdaKernelParams params) {
  // One wire slice per warp: the copy it parked this round.
  __shared__ DdaLL128Pair lds[kDdaLL128ArPeerMaxWarps][kDdaLL128PairsPerSlice];
  __shared__ int aborted;

  const int nRanks = NRANKS_CT ? NRANKS_CT : nRanksRt;

  const int tid = threadIdx.x;
  const int lane = tid % kDdaLL128Warp;
#if defined(__gfx1250__) || defined(__gfx1250_strict__)
  // Wave32: a warp is one wave, so its index is wave-uniform and the scratch addresses stay scalar.
  const int warp = __builtin_amdgcn_readfirstlane(tid / kDdaLL128Warp);
#else
  const int warp = tid / kDdaLL128Warp;
#endif
  const int nwarps = blockDim.x / kDdaLL128Warp;
  const bool flagLane = ddaLL128IsFlagLane(lane);

  // Phase 2 layout, as in the one-shot: warp pw of group `group` handles ranks congruent to pw mod P.
  const int P = ddaLL128ArPeerWarps(nRanks, nwarps);
  const int groups = nwarps / P;
  const int group = warp / P;
  const int pw = warp % P;
  const bool inGroup = group < groups;
  const int storeWarp = (nRanks - 1) % P;

  if (tid == 0) {
    aborted = 0;
  }
  // Its barrier also publishes `aborted`.
  const uint32_t flag32 = ddaGetLLEpochInc(epochDev, blockIdx.x, 1);
  const uint64_t flag = ((uint64_t)flag32 << 32) | (uint64_t)flag32;
  // Two staging areas per bank; the second starts one full fan-out in.
  const size_t slotWords = ddaLL128ArTwoShotSlotWords(bankSize, nRanks);
  const uint64_t bankWords = (uint64_t)(flag32 & 1u) * (uint64_t)(bankSize / sizeof(uint64_t));
  const uint64_t bankWordsNext = bankWords + (uint64_t)nRanks * (uint64_t)slotWords;
  const size_t sliceGroups = (slicesTotal + groups - 1) / groups;

  const int8_t* srcBytes = reinterpret_cast<const int8_t*>(sendbuff);
  int8_t* dstBytes = reinterpret_cast<int8_t*>(recvbuff);
  const uint64_t* selfBase = reinterpret_cast<const uint64_t*>(peerScratch[selfRank]);

  // Phases 1 and 3 need no reduction, so they skip the groups and spread their (slice, peer) items over every warp:
  // item i is slice i / (nRanks - 1) for peer selfRank + 1 + i % (nRanks - 1).
  const size_t items = slicesTotal * (size_t)(nRanks - 1);
  const size_t gwarp = (size_t)blockIdx.x * (size_t)nwarps + (size_t)warp;
  const size_t wstride = (size_t)gridDim.x * (size_t)nwarps;

  // Phase 1: send each item's peer the slice of the shard that peer owns.
  for (size_t i = gwarp; i < items; i += wstride) {
    const size_t s = i / (size_t)(nRanks - 1);
    const int peer = (selfRank + 1 + (int)(i - s * (size_t)(nRanks - 1))) % nRanks;
    const size_t sliceByte = s * (size_t)kDdaLL128DataBytesPerSlice;
    const size_t rem = shardBytes - sliceByte;
    const int eltInSlice = rem < (size_t)kDdaLL128DataBytesPerSlice ? (int)rem : kDdaLL128DataBytesPerSlice;
    uint64_t regs[kDdaLL128WordsPerThread] = {};
    ddaLL128LoadRegsBatched<int8_t>(regs, srcBytes + (size_t)peer * shardBytes + sliceByte, eltInSlice, lane,
                                    flagLane);
    uint64_t* scatterSlot =
      reinterpret_cast<uint64_t*>(peerScratch[peer]) + bankWords + (uint64_t)selfRank * (uint64_t)slotWords;
    ddaLL128StoreWire(scatterSlot + s * (size_t)kDdaLL128WireWordsPerSlice + 2 * lane, regs, flag, flagLane);
  }

  // Phase 2: fold every copy of my shard's slice, hand the sum back out, then keep it.
  for (size_t g = blockIdx.x; g < sliceGroups; g += gridDim.x) {
    const size_t s = g * (size_t)groups + (size_t)group;
    const bool live = inGroup && s < slicesTotal;
    const size_t sliceByte = s * (size_t)kDdaLL128DataBytesPerSlice;
    const size_t rem = live ? shardBytes - sliceByte : 0;
    const int eltInSlice = rem < (size_t)kDdaLL128DataBytesPerSlice ? (int)rem : kDdaLL128DataBytesPerSlice;

    if (!ddaLL128ArPeerFold<T, NRANKS_CT>(lds, aborted, live, nRanks, selfRank, P, warp,
                                          srcBytes + (size_t)selfRank * shardBytes + sliceByte, eltInSlice,
                                          selfBase + bankWords + s * (size_t)kDdaLL128WireWordsPerSlice, slotWords,
                                          flag, params)) {
      ddaSetLLEpoch(epochDev, epochLen, blockIdx.x, gridDim.x, flag32);
      return;
    }
    uint64_t acc[kDdaLL128WordsPerThread];
    if (live) {
#pragma unroll
      for (int u = 0; u < kDdaLL128Pairs; ++u) {
        const DdaLL128Pair v = lds[group * P + storeWarp][u * kDdaLL128Warp + lane];
        acc[2 * u] = v.lo;
        acc[2 * u + 1] = v.hi;
      }
    }
    // Every warp of the group has read the sum before the store warp parks its next copy over it.
    __syncthreads();
    if (live) {
      // Publish before the local unpack: it is what the peers are spinning on, and
      // ddaLL128StoreRegs rewrites acc in place when it reverses the flag-lane shuffle.
      for (int r = 1 + pw; r < nRanks; r += P) {
        const int peer = (selfRank + r) % nRanks;
        uint64_t* scatterSlot =
          reinterpret_cast<uint64_t*>(peerScratch[peer]) + bankWordsNext + (uint64_t)selfRank * (uint64_t)slotWords;
        ddaLL128StoreWire(scatterSlot + s * (size_t)kDdaLL128WireWordsPerSlice + 2 * lane, acc, flag, flagLane);
      }
      if (pw == storeWarp) {
        ddaLL128StoreRegs<int8_t>(dstBytes + (size_t)selfRank * shardBytes + sliceByte, acc, eltInSlice, lane,
                                  flagLane);
      }
    }
  }

  // Phase 3: collect each item's peer's reduced shard slice into the rest of recvbuff.
  for (size_t i = gwarp; i < items; i += wstride) {
    const size_t s = i / (size_t)(nRanks - 1);
    const int peer = (selfRank + 1 + (int)(i - s * (size_t)(nRanks - 1))) % nRanks;
    const size_t sliceByte = s * (size_t)kDdaLL128DataBytesPerSlice;
    const size_t rem = shardBytes - sliceByte;
    const int eltInSlice = rem < (size_t)kDdaLL128DataBytesPerSlice ? (int)rem : kDdaLL128DataBytesPerSlice;
    const uint64_t* gatherSlot = selfBase + bankWordsNext + (uint64_t)peer * (uint64_t)slotWords;
    uint64_t vr[kDdaLL128WordsPerThread];
    if (!ddaLL128ArPeerPoll(gatherSlot + s * (size_t)kDdaLL128WireWordsPerSlice + 2 * lane, vr, flag, lane,
                            params)) {
      break;
    }
    ddaLL128StoreRegs<int8_t>(dstBytes + (size_t)peer * shardBytes + sliceByte, vr, eltInSlice, lane, flagLane);
  }

#if defined(__gfx1250__) || defined(__gfx1250_strict__)
  asm volatile("s_wait_storecnt 0x0" ::: "memory");
#endif

  ddaSetLLEpoch(epochDev, epochLen, blockIdx.x, gridDim.x, flag32);
}

} // namespace dda::common
