/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Peer-parallel LL128 reduce-scatter kernel for the DDA fabric path. A slice
 * gets a group of up to one warp per rank, which polls its copies P ranks at a
 * time and folds them through LDS in rank order, so the per-slice latency stops
 * growing with nRanks. Wire format, slot layout and epoch/bank handling are the
 * batched kernel's (reduce_scatter_dda_fabric_ll128.h).
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#pragma once

#include <cstddef>
#include <cstdint>

#include <hip/hip_runtime.h>

#include "algorithms/dda/device/CollCommon.h"
#include "algorithms/dda/device/CollCommon_ll128.h"
#include "algorithms/dda/reduce_scatter/reduce_scatter_dda_fabric_ll128.h"

namespace dda::common {

// Widest block the peer-parallel kernel takes: it sizes the kernel's static LDS.
constexpr int kDdaLL128RsPeerMaxThreads = 512;
constexpr int kDdaLL128RsPeerMaxWarps = kDdaLL128RsPeerMaxThreads / kDdaLL128Warp;
// nRanks >= 2, so a group is at least two warps.
constexpr int kDdaLL128RsPeerMaxGroups = kDdaLL128RsPeerMaxWarps / 2;
// Wire words of a slice one thread of a group folds, at the two-warp minimum group.
constexpr int kDdaLL128RsPeerFoldWords = kDdaLL128WireWordsPerSlice / (2 * kDdaLL128Warp);

// Warps per slice group: one per rank, up to the whole block.
__host__ __device__ constexpr int ddaLL128RsPeerWarps(int nRanks, int nwarps) {
  return nRanks < nwarps ? nRanks : nwarps;
}

// Wire word of register u on lane `lane`: the 16B-pair layout ddaLL128PollWire reads.
__device__ __forceinline__ int ddaLL128RsWireWord(int u, int lane) {
  return (u & ~1) * kDdaLL128Warp + 2 * lane + (u & 1);
}

// Poll one slice. Cooperative loads land the payload with its flag, so the batched poll needs no re-read. Without
// them the plain poll re-reads once the flags are current.
__device__ __forceinline__ bool ddaLL128RsPeerPoll(const uint64_t* wire, uint64_t (&vr)[kDdaLL128WordsPerThread],
                                                   uint64_t flag, int lane, DdaKernelParams params) {
#if RCCL_HAVE_COOPERATIVE_ATOMIC_BUILTINS
  return ddaLL128PollWireBatched(wire, vr, flag, lane, params);
#else
  return ddaLL128PollWire(wire, vr, flag, lane, params);
#endif
}

// Per-block LDS: one wire slice per warp for the copy it read, one per group for the folded result.
struct DdaLL128RsPeerLds {
  uint64_t in[kDdaLL128RsPeerMaxWarps][kDdaLL128WireWordsPerSlice];
  uint64_t red[kDdaLL128RsPeerMaxGroups][kDdaLL128WireWordsPerSlice];
  int aborted;
};

// Where a warp sits in the peer-parallel block layout.
struct DdaLL128RsPeerPos {
  int lane;
  int warp;
  int P;        // warps per group
  int groups;   // groups per block
  int group;
  int pw;       // warp index within its group
  bool flagLane;
  bool inGroup; // false for the leftover warps when P does not divide the block
};

__device__ __forceinline__ DdaLL128RsPeerPos ddaLL128RsPeerPos(int nRanks) {
  DdaLL128RsPeerPos p;
  const int tid = threadIdx.x;
  p.lane = tid % kDdaLL128Warp;
#if defined(__gfx1250__) || defined(__gfx1250_strict__)
  // Wave32: a warp is one wave, so its index is wave-uniform and the scratch addresses stay scalar.
  p.warp = __builtin_amdgcn_readfirstlane(tid / kDdaLL128Warp);
#else
  p.warp = tid / kDdaLL128Warp;
#endif
  const int nwarps = blockDim.x / kDdaLL128Warp;
  p.P = ddaLL128RsPeerWarps(nRanks, nwarps);
  p.groups = nwarps / p.P;
  p.group = p.warp / p.P;
  p.pw = p.warp % p.P;
  p.flagLane = ddaLL128IsFlagLane(p.lane);
  p.inGroup = p.group < p.groups;
  return p;
}

// Rank-order fold of one slice across a warp group: rounds of P ranks parked in lds.in and folded after a block-wide
// barrier, the sum left in lds.red. Every warp calls it once per slice group. Returns false, block-uniform, on abort.
template <typename T, int NRANKS_CT>
__device__ __forceinline__ bool ddaLL128RsPeerFold(DdaLL128RsPeerLds& lds, const DdaLL128RsPeerPos& p, bool live,
                                                   int nRanks, int selfRank, const int8_t* self, int eltInSlice,
                                                   const uint64_t* gather, size_t slotWords, uint64_t flag,
                                                   DdaKernelParams params) {
  const int gt = p.pw * kDdaLL128Warp + p.lane;
  const int gThreads = p.P * kDdaLL128Warp;
  uint64_t acc[kDdaLL128RsPeerFoldWords] = {};

  for (int base = 0; base < nRanks; base += p.P) {
    const int rank = base + p.pw;
    if (live && rank < nRanks) {
      uint64_t vr[kDdaLL128WordsPerThread] = {};
      if (rank == selfRank) {
        ddaLL128LoadRegsBatched<int8_t>(vr, self, eltInSlice, p.lane, p.flagLane);
      } else if (!ddaLL128RsPeerPoll(gather + (uint64_t)rank * (uint64_t)slotWords + 2 * p.lane, vr, flag, p.lane,
                                     params)) {
        atomicExch(&lds.aborted, 1);
      }
#pragma unroll
      for (int u = 0; u < kDdaLL128WordsPerThread; ++u) {
        lds.in[p.warp][ddaLL128RsWireWord(u, p.lane)] = vr[u];
      }
    }
    // LDS only: every copy, my own slice included, is in lds.in by now, so stores still in flight need not land.
    ddaLLSpreadLdsBarrier();
    if (lds.aborted) {
      return false;
    }
    if (live) {
      const int nIn = nRanks - base < p.P ? nRanks - base : p.P;
      const bool last = base + p.P >= nRanks;
#pragma unroll
      for (int k = 0; k < kDdaLL128RsPeerFoldWords; ++k) {
        const int w = gt + k * gThreads;
        if (w < kDdaLL128WireWordsPerSlice) {
          // A flag lane's odd words hold flags in the copies, and ddaLL128StoreRegs never stores their sum.
          const auto fold = [&](int j) {
            const uint64_t v = lds.in[p.group * p.P + j][w];
            acc[k] = (base == 0 && j == 0) ? v : ddaLL128AddWord<T>(acc[k], v);
          };
          if constexpr (NRANKS_CT != 0) {
            constexpr int kRoundMax = NRANKS_CT < kDdaLL128RsPeerMaxWarps ? NRANKS_CT : kDdaLL128RsPeerMaxWarps;
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
            lds.red[p.group][w] = acc[k];
          }
        }
      }
    }
    // Ends the round: the next one overwrites lds.in, and the caller reads lds.red.
    ddaLLSpreadLdsBarrier();
  }
  return true;
}

// The group's folded slice, back in this lane's register layout.
__device__ __forceinline__ void ddaLL128RsPeerLoadRed(const DdaLL128RsPeerLds& lds, const DdaLL128RsPeerPos& p,
                                                      uint64_t (&acc)[kDdaLL128WordsPerThread]) {
#pragma unroll
  for (int u = 0; u < kDdaLL128WordsPerThread; ++u) {
    acc[u] = lds.red[p.group][ddaLL128RsWireWord(u, p.lane)];
  }
}

// Peer-parallel LL128 reduce-scatter with ddaReduceScatterFabricLL128's contract and scratch layout. Phase 1 spreads
// (slice, peer) sends over every warp of the grid, and phase 2 folds each slice of my shard across a warp group.
template <typename T, int NRANKS_CT>
__launch_bounds__(kDdaLL128RsPeerMaxThreads)
  __global__ void ddaReduceScatterFabricLL128PeerPar(T* const* __restrict__ peerScratch, // ddaPeerPtrsDev
                                                     T* __restrict__ recvbuff, // local user output (one shard)
                                                     const T* __restrict__ sendbuff, // local input (nRanks shards)
                                                     size_t shardBytes, // per-rank shard payload, a multiple of 16
                                                     int selfRank, int nRanksRt,
                                                     uint32_t* __restrict__ epochDev, // per-block LL epoch cells
                                                     int epochLen, // number of cells in epochDev
                                                     size_t slicesTotal, // slices per shard this call uses
                                                     size_t bankSize, // scratch bank size (from host)
                                                     DdaKernelParams params) {
  __shared__ DdaLL128RsPeerLds lds;
  const int nRanks = NRANKS_CT ? NRANKS_CT : nRanksRt;
  const DdaLL128RsPeerPos p = ddaLL128RsPeerPos(nRanks);

  if (threadIdx.x == 0) {
    lds.aborted = 0;
  }
  // Its barrier also publishes lds.aborted.
  const uint32_t flag32 = ddaGetLLEpochInc(epochDev, blockIdx.x, 1);
  const uint64_t flag = ((uint64_t)flag32 << 32) | (uint64_t)flag32;
  const size_t slotWords = ddaLL128RsSlotWords(bankSize, nRanks);
  const uint64_t bankWords = (uint64_t)(flag32 & 1u) * (uint64_t)(bankSize / sizeof(uint64_t));
  const size_t sliceGroups = (slicesTotal + p.groups - 1) / p.groups;

  const int8_t* srcBytes = reinterpret_cast<const int8_t*>(sendbuff);
  int8_t* dstBytes = reinterpret_cast<int8_t*>(recvbuff);
  const uint64_t* gatherBase = reinterpret_cast<const uint64_t*>(peerScratch[selfRank]) + bankWords;

  // Phase 1: item i sends slice i / (nRanks - 1) of its shard to peer selfRank + 1 + i % (nRanks - 1).
  const size_t items = slicesTotal * (size_t)(nRanks - 1);
  const size_t nwarps = blockDim.x / kDdaLL128Warp;
  for (size_t i = (size_t)blockIdx.x * nwarps + (size_t)p.warp; i < items; i += (size_t)gridDim.x * nwarps) {
    const size_t s = i / (size_t)(nRanks - 1);
    const int peer = (selfRank + 1 + (int)(i - s * (size_t)(nRanks - 1))) % nRanks;
    const size_t sliceByte = s * (size_t)kDdaLL128DataBytesPerSlice;
    const size_t rem = shardBytes - sliceByte;
    const int eltInSlice = rem < (size_t)kDdaLL128DataBytesPerSlice ? (int)rem : kDdaLL128DataBytesPerSlice;
    uint64_t regs[kDdaLL128WordsPerThread] = {};
    ddaLL128LoadRegsBatched<int8_t>(regs, srcBytes + (size_t)peer * shardBytes + sliceByte, eltInSlice, p.lane,
                                    p.flagLane);
    uint64_t* scatterSlot =
      reinterpret_cast<uint64_t*>(peerScratch[peer]) + bankWords + (uint64_t)selfRank * (uint64_t)slotWords;
    ddaLL128StoreWire(scatterSlot + s * (size_t)kDdaLL128WireWordsPerSlice + 2 * p.lane, regs, flag, p.flagLane);
  }

  // Phase 2: fold every rank's copy of my shard's slice, then warp 0 of the group unpacks it. My own slice is read
  // before the fold's first barrier and written after its last, so an in-place call is safe.
  for (size_t g = blockIdx.x; g < sliceGroups; g += gridDim.x) {
    const size_t s = g * (size_t)p.groups + (size_t)p.group;
    const bool live = p.inGroup && s < slicesTotal;
    const size_t sliceByte = s * (size_t)kDdaLL128DataBytesPerSlice;
    const int eltInSlice =
      !live ? 0 : (shardBytes - sliceByte < (size_t)kDdaLL128DataBytesPerSlice ? (int)(shardBytes - sliceByte)
                                                                               : kDdaLL128DataBytesPerSlice);
    if (!ddaLL128RsPeerFold<T, NRANKS_CT>(lds, p, live, nRanks, selfRank,
                                          srcBytes + (size_t)selfRank * shardBytes + sliceByte, eltInSlice,
                                          gatherBase + s * (size_t)kDdaLL128WireWordsPerSlice, slotWords, flag,
                                          params)) {
      break;
    }
    if (live && p.pw == 0) {
      uint64_t acc[kDdaLL128WordsPerThread];
      ddaLL128RsPeerLoadRed(lds, p, acc);
      ddaLL128StoreRegs<int8_t>(dstBytes + sliceByte, acc, eltInSlice, p.lane, p.flagLane);
    }
  }

  ddaLL128WaitStores();

  ddaSetLLEpoch(epochDev, epochLen, blockIdx.x, gridDim.x, flag32);
}

} // namespace dda::common
