/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * LL128 scatter-allgather broadcast for the DDA path: the root sends chunk q of nRanks-1 to
 * non-root (root+1+q) % nRanks, which forwards it to the other non-roots. Start tokens and
 * bank layout are the one-shot kernel's, so empty chunks need no padding.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#pragma once

#include "algorithms/dda/broadcast/broadcast_dda_fabric_ll128.h"

namespace dda::common {

// Slices per chunk; chunk q starts at slice q * cs.
constexpr size_t ddaLL128BcChunkSlices(size_t slices, int nRanks) {
  return (slices + nRanks - 2) / (nRanks - 1);
}

// Warp tasks of kU slices per chunk.
constexpr size_t ddaLL128BcChunkTasks(size_t cs, int kU) {
  return (cs + kU - 1) / kU;
}

// Slices in task i of a chunk, from slice s0: kU, or fewer at the chunk's or payload's end.
constexpr size_t ddaLL128BcTaskSlices(size_t cs, size_t i, size_t s0, size_t slices, int kU) {
  const size_t n = cs - i * kU < (size_t)kU ? cs - i * kU : (size_t)kU;
  return slices - s0 < n ? slices - s0 : n;
}

// Root: push kW/8 consecutive slices from s0 and, out of place, write them to dst too.
template <int kW, bool kAligned>
__device__ __forceinline__ void ddaLL128BcRootUnit(uint64_t* ownerData, const int8_t* src, int8_t* dst, size_t s0,
                                                   size_t bytes, uint64_t flag, int lane, bool flagLane, bool copy) {
  constexpr int kLine = kDdaLL128ArchLineElems;
  constexpr int kWarp = kDdaLL128ArchWarp;
  constexpr int kUnitBytes = kDdaLL128DataBytesPerSliceL<kLine, kWarp> * (kW / kDdaLL128WordsPerThread);
  constexpr size_t kSliceWords = kDdaLL128WireWordsPerSliceW<kWarp>;
  const size_t off = s0 * kDdaLL128DataBytesPerSliceL<kLine, kWarp>;
  const size_t rem = bytes - off;
  const int eltN = rem < (size_t)kUnitBytes ? (int)rem : kUnitBytes;
  uint64_t regs[kW] = {};
  ddaLL128LoadRegs<int8_t, kLine, kAligned, kWarp, kW>(regs, src + off, eltN, lane, flagLane);
  ddaLL128StoreWire<kWarp, kW>(ownerData + s0 * kSliceWords + 2 * lane, regs, flag, flagLane);
  if (copy) ddaLL128StoreRegs<int8_t, kLine, kAligned, kWarp, kW>(dst + off, regs, eltN, lane, flagLane);
}

// Non-root: poll kW/8 consecutive slices from s0, forward them if own, unpack, clear.
template <int kW, bool kAligned>
__device__ __forceinline__ bool ddaLL128BcPeerUnit(char* const* peerScratch, uint64_t* selfData, size_t dataWords,
                                                   int8_t* dst, size_t s0, size_t bytes, uint64_t flag, int lane,
                                                   bool flagLane, bool own, int selfRank, int nRanks, int root,
                                                   DdaKernelParams params) {
  constexpr int kLine = kDdaLL128ArchLineElems;
  constexpr int kWarp = kDdaLL128ArchWarp;
  constexpr int kUnitBytes = kDdaLL128DataBytesPerSliceL<kLine, kWarp> * (kW / kDdaLL128WordsPerThread);
  constexpr size_t kSliceWords = kDdaLL128WireWordsPerSliceW<kWarp>;
  uint64_t* wire = selfData + s0 * kSliceWords + 2 * lane;
  uint64_t vr[kW];
  if (!ddaLL128PollWire<kLine, kWarp, kW>(wire, vr, flag, lane, params)) return false;
  if (own) {
    // vr is still in wire format with the current flag on the flag lanes, so
    // it is re-stored as is. Must precede StoreRegs, which unshuffles vr.
    for (int r = 1; r < nRanks; r++) {
      const int p = (selfRank + r) % nRanks;
      if (p == root) continue;
      ddaLL128StoreWire<kWarp, kW>(
        reinterpret_cast<uint64_t*>(peerScratch[p]) + dataWords + s0 * kSliceWords + 2 * lane, vr, flag, flagLane);
    }
  }
  const size_t off = s0 * kDdaLL128DataBytesPerSliceL<kLine, kWarp>;
  const size_t rem = bytes - off;
  const int eltN = rem < (size_t)kUnitBytes ? (int)rem : kUnitBytes;
  ddaLL128StoreRegs<int8_t, kLine, kAligned, kWarp, kW>(dst + off, vr, eltN, lane, flagLane);
  ddaLL128ClearWire<kWarp, kW>(wire);
  return true;
}

// A warp task is kU consecutive slices of one chunk; a short last task runs slice by slice.
template <bool kAligned, int kU>
#if defined(__HIP_PLATFORM_AMD__)
__launch_bounds__(512)
#endif
  __global__ void ddaBroadcastFabricLL128TwoShot(char* const* __restrict__ peerScratch, // nRanks scratch bases
                                                 char* __restrict__ recvbuff, const char* __restrict__ sendbuff,
                                                 size_t bytes, // multiple of 16
                                                 int selfRank, int nRanks, int root,
                                                 uint32_t* __restrict__ epochDev, int epochLen, size_t bankSize,
                                                 DdaKernelParams params) {
  constexpr int kLine = kDdaLL128ArchLineElems;
  constexpr int kWarp = kDdaLL128ArchWarp;
  constexpr int kW1 = kDdaLL128WordsPerThread;
  constexpr int kWU = kW1 * kU;
  constexpr size_t kSliceWords = kDdaLL128WireWordsPerSliceW<kWarp>;

  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  const int lane = tid % kWarp;
  const int warp = tid / kWarp;
  const int nwarps = nthreads / kWarp;
  const bool flagLane = ddaLL128IsFlagLane<kLine>(lane);
  const int flatBlockId = blockIdx.x * gridDim.y + blockIdx.y;
  const int total = gridDim.x * gridDim.y;

  const uint32_t flag32 = ddaGetLLEpochInc(epochDev, flatBlockId, 1);
  const uint64_t flag = ((uint64_t)flag32 << 32) | (uint64_t)flag32;
  const size_t bankWords = (size_t)(flag32 & 1u) * (bankSize / sizeof(uint64_t));
  const size_t dataWords = bankWords + ddaLL128BcTokenWords(nRanks, kSliceWords);
  const size_t slices = ddaLL128SlicesL<kLine, kWarp>(bytes);
  const int nPeers = nRanks - 1;
  const size_t cs = ddaLL128BcChunkSlices(slices, nRanks);
  const size_t ng = ddaLL128BcChunkTasks(cs, kU);

  uint64_t* selfBase = reinterpret_cast<uint64_t*>(peerScratch[selfRank]);
  const LLPacket16* selfTok = reinterpret_cast<const LLPacket16*>(selfBase + bankWords);
  const int8_t* src = reinterpret_cast<const int8_t*>(sendbuff);
  int8_t* dst = reinterpret_cast<int8_t*>(recvbuff);

  if (selfRank == root) {
    const int q = (int)blockIdx.x;
    const int owner = (root + 1 + q) % nRanks;
    uint64_t* ownerData = reinterpret_cast<uint64_t*>(peerScratch[owner]) + dataWords;
    const bool copy = sendbuff != recvbuff;
    for (size_t i = (size_t)blockIdx.y * nwarps + warp; i < ng; i += (size_t)gridDim.y * nwarps) {
      const size_t s0 = (size_t)q * cs + i * kU;
      if (s0 >= slices) break;
      const size_t n = ddaLL128BcTaskSlices(cs, i, s0, slices, kU);
      if (n == (size_t)kU) {
        ddaLL128BcRootUnit<kWU, kAligned>(ownerData, src, dst, s0, bytes, flag, lane, flagLane, copy);
      } else {
        for (size_t j = 0; j < n; j++) {
          ddaLL128BcRootUnit<kW1, kAligned>(ownerData, src, dst, s0 + j, bytes, flag, lane, flagLane, copy);
        }
      }
    }

    if (blockIdx.y == 0 && tid == 0) {
      uint32_t d0, f0, d1, f1;
      (void)ddaLLWaitLineFlags(&selfTok[owner], flag32, params, d0, f0, d1, f1);
    }
  } else {
    if (flatBlockId == 0) {
      for (int p = tid; p < nRanks; p += nthreads) {
        if (p == selfRank) continue;
        LLPacket16* tok = reinterpret_cast<LLPacket16*>(reinterpret_cast<uint64_t*>(peerScratch[p]) + bankWords) +
          selfRank;
        ddaLLStoreLineB128(reinterpret_cast<uint32_t*>(tok), 0u, flag32, 0u, flag32);
      }
    }

    // Tasks [0, ng): own chunk (poll, forward, unpack). [ng, nPeers*ng): the
    // other chunks in order, skipping our own.
    const int myQ = (selfRank - root - 1 + nRanks) % nRanks;
    const size_t nTasks = (size_t)nPeers * ng;
    uint64_t* selfData = selfBase + dataWords;
    for (size_t t = (size_t)flatBlockId * nwarps + warp; t < nTasks; t += (size_t)total * nwarps) {
      const bool own = t < ng;
      size_t q = (size_t)myQ, i = t;
      if (!own) {
        const size_t k = (t - ng) / ng;
        q = k < (size_t)myQ ? k : k + 1;
        i = (t - ng) % ng;
      }
      const size_t s0 = q * cs + i * kU;
      if (s0 >= slices) continue;
      const size_t n = ddaLL128BcTaskSlices(cs, i, s0, slices, kU);
      bool ok = true;
      if (n == (size_t)kU) {
        ok = ddaLL128BcPeerUnit<kWU, kAligned>(peerScratch, selfData, dataWords, dst, s0, bytes, flag, lane, flagLane,
                                               own, selfRank, nRanks, root, params);
      } else {
        for (size_t j = 0; j < n && ok; j++) {
          ok = ddaLL128BcPeerUnit<kW1, kAligned>(peerScratch, selfData, dataWords, dst, s0 + j, bytes, flag, lane,
                                                 flagLane, own, selfRank, nRanks, root, params);
        }
      }
      if (!ok) break;
    }

    if (flatBlockId == 0) {
      for (int p = tid; p < nRanks; p += nthreads) {
        if (p == selfRank || p == root) continue;
        uint32_t d0, f0, d1, f1;
        if (!ddaLLWaitLineFlags(&selfTok[p], flag32, params, d0, f0, d1, f1)) break;
      }
    }
  }

#if (defined(__gfx1250__) || defined(__gfx1250_strict__))
  asm volatile("s_wait_storecnt 0x0" ::: "memory");
#endif
  __syncthreads();
  ddaSetLLEpoch(epochDev, epochLen, flatBlockId, total, flag32);
}

} // namespace dda::common
