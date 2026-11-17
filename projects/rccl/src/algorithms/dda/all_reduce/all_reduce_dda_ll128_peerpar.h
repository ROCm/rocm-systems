/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Peer-parallel LL128 one-shot all-reduce kernel for the DDA fabric path.
 *
 * The flat one-shot (all_reduce_dda_ll128.h) gives a slice to one warp, which
 * then walks every peer in turn: nRanks - 1 stores and, worse, nRanks - 1 flag
 * polls back to back, each a full fabric round trip. Its per-slice latency
 * therefore grows linearly with nRanks, while all-gather
 * and all-to-all, which give each peer its own warps, stay flat.
 *
 * Here a slice gets a group of P = ddaLL128ArPeerWarps() warps instead and the
 * peer loops are split across the group: warp pw sends to, and polls, the ranks
 * congruent to pw mod P, so a group pays one poll round trip per P ranks. The
 * polled copies meet in LDS and are folded there, still in rank order
 * 0..nRanks-1, so every rank rounds identically.
 *
 * Wire format, slot layout, epoch/bank handling and the pack/poll/unpack helpers
 * are the flat one-shot's, unchanged, so this kernel is a drop-in replacement.
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
// Slot geometry is shared with the flat one-shot, not redefined here.
#include "algorithms/dda/all_reduce/all_reduce_dda_ll128.h"

namespace dda::common {

// Widest block the peer-parallel kernel takes: it sizes its static LDS.
constexpr int kDdaLL128ArPeerMaxThreads = 512;
constexpr int kDdaLL128ArPeerMaxWarps = kDdaLL128ArPeerMaxThreads / kDdaLL128Warp;
// Blocks come in 64-thread granules and nRanks >= 2, so a group is at least two warps.
constexpr int kDdaLL128ArPeerMaxGroups = kDdaLL128ArPeerMaxWarps / 2;
// One 16B wire pair (lo, hi), the unit a lane moves on the wire; 16B-aligned so LDS accesses are 16B wide.
struct alignas(16) DdaLL128Pair {
  uint64_t lo;
  uint64_t hi;
};
// 16B pairs of one slice.
constexpr int kDdaLL128PairsPerSlice = kDdaLL128WireWordsPerSlice / 2;
// Pairs of a slice one thread of a group folds, at the two-warp minimum group.
constexpr int kDdaLL128ArPeerFoldPairs = kDdaLL128PairsPerSlice / (2 * kDdaLL128Warp);

// Warps per slice group: one per rank, up to the whole block.
__host__ __device__ constexpr int ddaLL128ArPeerWarps(int nRanks, int nwarps) {
  return nRanks < nwarps ? nRanks : nwarps;
}

// Wire pair of register pair g on lane `lane`: the 16B-pair layout ddaLL128PollWire reads.
__device__ __forceinline__ int ddaLL128ArWirePair(int g, int lane) {
  return g * kDdaLL128Warp + lane;
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

// Per-block LDS: one wire slice per warp for the copies it polled, one per group for the folded result.
struct DdaLL128ArPeerLds {
  DdaLL128Pair in[kDdaLL128ArPeerMaxWarps][kDdaLL128PairsPerSlice];
  DdaLL128Pair red[kDdaLL128ArPeerMaxGroups][kDdaLL128PairsPerSlice];
  int aborted;
};

// Where a warp sits in the peer-parallel block layout.
struct DdaLL128ArPeerPos {
  int lane;
  int warp;
  int P;     // warps per group
  int groups; // groups per block
  int group;
  int pw;    // warp index within its group
  int storeWarp; // pw that unpacks a group's sum: the one with the fewest peers to send to
  bool flagLane;
  bool inGroup; // false for the leftover warps when P does not divide the block
};

__device__ __forceinline__ DdaLL128ArPeerPos ddaLL128ArPeerPos(int nRanks) {
  DdaLL128ArPeerPos p;
  const int tid = threadIdx.x;
  p.lane = tid % kDdaLL128Warp;
#if defined(__gfx1250__) || defined(__gfx1250_strict__)
  // Wave32: a warp is one wave, so its index is wave-uniform and the scratch addresses stay scalar.
  p.warp = __builtin_amdgcn_readfirstlane(tid / kDdaLL128Warp);
#else
  p.warp = tid / kDdaLL128Warp;
#endif
  const int nwarps = blockDim.x / kDdaLL128Warp;
  p.P = ddaLL128ArPeerWarps(nRanks, nwarps);
  p.groups = nwarps / p.P;
  p.group = p.warp / p.P;
  p.pw = p.warp % p.P;
  p.storeWarp = (nRanks - 1) % p.P;
  p.flagLane = ddaLL128IsFlagLane(p.lane);
  p.inGroup = p.group < p.groups;
  return p;
}

// Rank-order fold of one slice across a warp group, in the register layout the wire delivers.
//
// Rounds of P ranks: warp pw loads rank base + pw (its own payload from `self`,
// a peer's by polling that peer's slot under `gather`) into lds.in, and after a
// barrier the group's threads fold their 16B pairs of the parked copies in rank
// order. The last round leaves the sum in lds.red[group], readable on return.
//
// The barriers are block-wide, so every warp of the block calls this once per
// slice group; `live` is false for a warp with no slice. Returns false, uniformly
// across the block, if any poll saw the abort flag. With NRANKS_CT set, a round's
// fold is unrolled to its largest possible size, so its LDS reads issue together.
template <typename T, int NRANKS_CT>
__device__ __forceinline__ bool ddaLL128ArPeerFold(DdaLL128ArPeerLds& lds, const DdaLL128ArPeerPos& p, bool live,
                                                   int nRanks, int selfRank, const int8_t* self, int eltInSlice,
                                                   const uint64_t* gather, size_t slotWords, uint64_t flag,
                                                   DdaKernelParams params) {
  const int gt = p.pw * kDdaLL128Warp + p.lane;
  const int gThreads = p.P * kDdaLL128Warp;
  DdaLL128Pair acc[kDdaLL128ArPeerFoldPairs] = {};

  for (int base = 0; base < nRanks; base += p.P) {
    const int rank = base + p.pw;
    if (live && rank < nRanks) {
      uint64_t vr[kDdaLL128WordsPerThread] = {};
      if (rank == selfRank) {
        ddaLL128LoadRegsBatched<int8_t>(vr, self, eltInSlice, p.lane, p.flagLane);
      } else if (!ddaLL128ArPeerPoll(gather + (uint64_t)rank * (uint64_t)slotWords + 2 * p.lane, vr, flag, p.lane,
                                     params)) {
        lds.aborted = 1;
      }
#pragma unroll
      for (int g = 0; g < kDdaLL128Pairs; ++g) {
        lds.in[p.warp][ddaLL128ArWirePair(g, p.lane)] = {vr[2 * g], vr[2 * g + 1]};
      }
    }
    __syncthreads();
    if (lds.aborted) {
      return false;
    }
    if (live) {
      const int nIn = nRanks - base < p.P ? nRanks - base : p.P;
      const bool last = base + p.P >= nRanks;
#pragma unroll
      for (int k = 0; k < kDdaLL128ArPeerFoldPairs; ++k) {
        const int q = gt + k * gThreads;
        if (q < kDdaLL128PairsPerSlice) {
          // On a flag lane the copies hold a flag in each pair's hi word, not payload; that sum
          // is never stored, since ddaLL128StoreRegs re-derives those slots.
          const auto fold = [&](int j) {
            const DdaLL128Pair v = lds.in[p.group * p.P + j][q];
            acc[k] = (base == 0 && j == 0) ? v : ddaLL128ArAddPair<T>(acc[k], v);
          };
          if constexpr (NRANKS_CT != 0) {
            constexpr int kRoundMax = NRANKS_CT < kDdaLL128ArPeerMaxWarps ? NRANKS_CT : kDdaLL128ArPeerMaxWarps;
#pragma unroll
            for (int j = 0; j < kRoundMax; ++j) {
              if (j < nIn) {
                fold(j);
              }
            }
          } else {
            for (int j = 0; j < nIn; ++j) {
              fold(j);
            }
          }
          if (last) {
            lds.red[p.group][q] = acc[k];
          }
        }
      }
    }
    // Ends the round: the next one overwrites lds.in, and the caller reads lds.red.
    __syncthreads();
  }
  return true;
}

// The group's folded slice, back in this lane's register layout.
__device__ __forceinline__ void ddaLL128ArPeerLoadRed(const DdaLL128ArPeerLds& lds, const DdaLL128ArPeerPos& p,
                                                      uint64_t (&acc)[kDdaLL128WordsPerThread]) {
#pragma unroll
  for (int g = 0; g < kDdaLL128Pairs; ++g) {
    const DdaLL128Pair v = lds.red[p.group][ddaLL128ArWirePair(g, p.lane)];
    acc[2 * g] = v.lo;
    acc[2 * g + 1] = v.hi;
  }
}

// Peer-parallel LL128 one-shot all-reduce. Same contract and scratch layout as
// ddaAllReduceFlatLL128; the grid is 1D over slice groups.
//
// Phase 1 (publish): warp pw of a slice's group pushes the slice to peers
// selfRank + 1 + pw, selfRank + 1 + pw + P, ...
// Phase 2 (reduce): the group polls its slice from every peer P at a time and
// folds the copies through LDS in rank order; the group's storeWarp unpacks the
// sum into recvbuff.
//
// A slice belongs to one group in both phases, and only that group's phase 2
// writes it, after all of its reads, so an in-place call is safe.
template <typename T, int NRANKS_CT>
__launch_bounds__(kDdaLL128ArPeerMaxThreads)
  __global__ void ddaAllReduceFlatLL128PeerPar(T* const* __restrict__ peerScratch, // ddaPeerPtrsDev: nRanks scratch bases
                                               T* __restrict__ recvbuff, // local user output
                                               const T* __restrict__ sendbuff, // local user input
                                               size_t bytes, // full-message payload; multiple of 16
                                               int selfRank, int nRanksRt,
                                               uint32_t* __restrict__ epochDev, // per-block LL epoch cells
                                               int epochLen, // number of cells in epochDev
                                               size_t slicesTotal, // slices this call actually uses
                                               size_t bankSize, // scratch bank size (from host)
                                               DdaKernelParams params) {
  __shared__ DdaLL128ArPeerLds lds;
  const int nRanks = NRANKS_CT ? NRANKS_CT : nRanksRt;
  const DdaLL128ArPeerPos p = ddaLL128ArPeerPos(nRanks);

  if (threadIdx.x == 0) {
    lds.aborted = 0;
  }
  // Its barrier also publishes lds.aborted.
  const uint32_t flag32 = ddaGetLLEpochInc(epochDev, blockIdx.x, 1);
  const uint64_t flag = ((uint64_t)flag32 << 32) | (uint64_t)flag32;
  const size_t slotWords = ddaLL128ArSlotWords(bankSize, nRanks);
  const uint64_t bankWords = (uint64_t)(flag32 & 1u) * (uint64_t)(bankSize / sizeof(uint64_t));
  const size_t sliceGroups = (slicesTotal + p.groups - 1) / p.groups;

  const int8_t* srcBytes = reinterpret_cast<const int8_t*>(sendbuff);
  int8_t* dstBytes = reinterpret_cast<int8_t*>(recvbuff);
  const uint64_t* gatherBase = reinterpret_cast<const uint64_t*>(peerScratch[selfRank]) + bankWords;

  // Phase 1: pack the slice once, push it to this warp's share of the peers.
  for (size_t g = blockIdx.x; g < sliceGroups; g += gridDim.x) {
    const size_t s = g * (size_t)p.groups + (size_t)p.group;
    if (!p.inGroup || s >= slicesTotal || 1 + p.pw >= nRanks) {
      break;
    }
    const size_t dataByte = s * (size_t)kDdaLL128DataBytesPerSlice;
    const size_t rem = bytes - dataByte;
    const int eltInSlice = rem < (size_t)kDdaLL128DataBytesPerSlice ? (int)rem : kDdaLL128DataBytesPerSlice;
    uint64_t regs[kDdaLL128WordsPerThread] = {};
    ddaLL128LoadRegsBatched<int8_t>(regs, srcBytes + dataByte, eltInSlice, p.lane, p.flagLane);
    for (int r = 1 + p.pw; r < nRanks; r += p.P) {
      const int peer = (selfRank + r) % nRanks;
      uint64_t* scatterSlot =
        reinterpret_cast<uint64_t*>(peerScratch[peer]) + bankWords + (uint64_t)selfRank * (uint64_t)slotWords;
      ddaLL128StoreWire(scatterSlot + s * (size_t)kDdaLL128WireWordsPerSlice + 2 * p.lane, regs, flag, p.flagLane);
    }
  }

  // Phase 2: fold every rank's copy of the slice; the group's storeWarp writes it out.
  for (size_t g = blockIdx.x; g < sliceGroups; g += gridDim.x) {
    const size_t s = g * (size_t)p.groups + (size_t)p.group;
    const bool live = p.inGroup && s < slicesTotal;
    const size_t dataByte = s * (size_t)kDdaLL128DataBytesPerSlice;
    const int eltInSlice =
      !live ? 0 : (bytes - dataByte < (size_t)kDdaLL128DataBytesPerSlice ? (int)(bytes - dataByte)
                                                                         : kDdaLL128DataBytesPerSlice);
    if (!ddaLL128ArPeerFold<T, NRANKS_CT>(lds, p, live, nRanks, selfRank, srcBytes + dataByte, eltInSlice,
                               gatherBase + s * (size_t)kDdaLL128WireWordsPerSlice, slotWords, flag, params)) {
      break;
    }
    if (live && p.pw == p.storeWarp) {
      uint64_t acc[kDdaLL128WordsPerThread];
      ddaLL128ArPeerLoadRed(lds, p, acc);
      ddaLL128StoreRegs<int8_t>(dstBytes + dataByte, acc, eltInSlice, p.lane, p.flagLane);
    }
  }

#if defined(__gfx1250__) || defined(__gfx1250_strict__)
  asm volatile("s_wait_storecnt 0x0" ::: "memory");
#endif

  ddaSetLLEpoch(epochDev, epochLen, blockIdx.x, gridDim.x, flag32);
}

} // namespace dda::common
