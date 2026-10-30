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

// Root grid: (nRanks-1) chunk columns x blocks per chunk. Non-root grid: blocks x 1
// over warp tasks, own-chunk forwards first so they never wait behind a gather.
template <bool kAligned>
#if defined(USE_ROCM)
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
  constexpr int kSliceBytes = kDdaLL128DataBytesPerSliceL<kLine, kWarp>;
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
  const size_t cs = (slices + nPeers - 1) / nPeers;

  uint64_t* selfBase = reinterpret_cast<uint64_t*>(peerScratch[selfRank]);
  const LLPacket16* selfTok = reinterpret_cast<const LLPacket16*>(selfBase + bankWords);
  const int8_t* src = reinterpret_cast<const int8_t*>(sendbuff);
  int8_t* dst = reinterpret_cast<int8_t*>(recvbuff);

  if (selfRank == root) {
    const int q = (int)blockIdx.x;
    const int owner = (root + 1 + q) % nRanks;
    uint64_t* ownerBase = reinterpret_cast<uint64_t*>(peerScratch[owner]) + dataWords;
    for (size_t i = (size_t)blockIdx.y * nwarps + warp; i < cs; i += (size_t)gridDim.y * nwarps) {
      const size_t s = (size_t)q * cs + i;
      if (s >= slices) break;
      const size_t rem = bytes - s * kSliceBytes;
      uint64_t regs[kDdaLL128WordsPerThread] = {};
      ddaLL128LoadRegs<int8_t, kLine, kAligned, kWarp>(regs, src + s * kSliceBytes,
                                                      rem < kSliceBytes ? (int)rem : kSliceBytes, lane, flagLane);
      ddaLL128StoreWire<kWarp>(ownerBase + s * kSliceWords + 2 * lane, regs, flag, flagLane);
    }

    if (sendbuff != recvbuff) {
      for (size_t i = (size_t)flatBlockId * nthreads + tid; i < (bytes >> 4); i += (size_t)total * nthreads) {
        uint64_t lo, hi;
        ddaLL128UserLoad16B<kAligned>(src + 16 * i, lo, hi);
        ddaLL128UserStore16B<kAligned>(dst + 16 * i, lo, hi);
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

    // Tasks [0, cs): own chunk (poll, forward, unpack). [cs, nPeers*cs): the
    // other chunks in order, skipping our own.
    const int myQ = (selfRank - root - 1 + nRanks) % nRanks;
    const size_t nTasks = (size_t)nPeers * cs;
    for (size_t t = (size_t)flatBlockId * nwarps + warp; t < nTasks; t += (size_t)total * nwarps) {
      const bool own = t < cs;
      size_t q = (size_t)myQ, i = t;
      if (!own) {
        const size_t k = (t - cs) / cs;
        q = k < (size_t)myQ ? k : k + 1;
        i = (t - cs) % cs;
      }
      const size_t s = q * cs + i;
      if (s >= slices) continue;
      uint64_t* wire = selfBase + dataWords + s * kSliceWords + 2 * lane;
      uint64_t vr[kDdaLL128WordsPerThread];
      if (!ddaLL128PollWire<kLine, kWarp>(wire, vr, flag, lane, params)) break;
      if (own) {
        // vr is still in wire format with the current flag on the flag lanes, so
        // it is re-stored as is. Must precede StoreRegs, which unshuffles vr.
        for (int r = 1; r < nRanks; r++) {
          const int p = (selfRank + r) % nRanks;
          if (p == root) continue;
          ddaLL128StoreWire<kWarp>(reinterpret_cast<uint64_t*>(peerScratch[p]) + dataWords + s * kSliceWords + 2 * lane,
                                   vr, flag, flagLane);
        }
      }
      const size_t rem = bytes - s * kSliceBytes;
      ddaLL128StoreRegs<int8_t, kLine, kAligned, kWarp>(dst + s * kSliceBytes, vr,
                                                       rem < kSliceBytes ? (int)rem : kSliceBytes, lane, flagLane);
      ddaLL128ClearWire<kWarp>(wire);
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
