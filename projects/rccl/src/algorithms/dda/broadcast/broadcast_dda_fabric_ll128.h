/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * LL128-protocol one-shot broadcast for the DDA path. Same protocol as the LL
 * tier (root push, all-to-all start tokens), with the payload in register-packed
 * LL128 slices. Line length follows the arch (kDdaLL128ArchLineElems).
 *
 * Bank layout (8B words): [nRanks LL token lines, padded to whole slices][data slices].
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

namespace dda::common {

// Token region in 8B words: one 16B LL line per rank, padded to whole slices.
constexpr size_t ddaLL128BcTokenWords(int nRanks, size_t sliceWords) {
  return (2 * (size_t)nRanks + sliceWords - 1) / sliceWords * sliceWords;
}

// Data slices one bank holds after the token region.
constexpr size_t ddaLL128BcSlotSlices(size_t bankSize, int nRanks, size_t sliceWords) {
  const size_t words = bankSize / sizeof(uint64_t);
  const size_t tok = ddaLL128BcTokenWords(nRanks, sliceWords);
  return words > tok ? (words - tok) / sliceWords : 0;
}

// A block's warps take consecutive slices (a "group"). Root grid: (nRanks-1)
// peer columns x blocks per peer, as in AllGather LL128. Non-root grid: blocks x 1.
template <bool kAligned>
__launch_bounds__(512)
  __global__ void ddaBroadcastFabricLL128(char* const* __restrict__ peerScratch, // nRanks scratch bases
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
  const size_t nGroups = (slices + nwarps - 1) / nwarps;

  uint64_t* selfBase = reinterpret_cast<uint64_t*>(peerScratch[selfRank]);
  const LLPacket16* selfTok = reinterpret_cast<const LLPacket16*>(selfBase + bankWords);
  const int8_t* src = reinterpret_cast<const int8_t*>(sendbuff);
  int8_t* dst = reinterpret_cast<int8_t*>(recvbuff);

  // A wait saw the abort flag: skip the remaining waits, still publish the epoch.
  bool aborted = false;

  if (selfRank == root) {
    const int peer = (root + 1 + (int)blockIdx.x) % nRanks;
    for (size_t g = blockIdx.y; g < nGroups; g += gridDim.y) {
      const size_t s = g * nwarps + warp;
      if (s >= slices) continue;
      const size_t rem = bytes - s * kSliceBytes;
      uint64_t regs[kDdaLL128WordsPerThread] = {};
      ddaLL128LoadRegsBatched<int8_t, kLine, kAligned, kWarp>(regs, src + s * kSliceBytes,
                                                      rem < kSliceBytes ? (int)rem : kSliceBytes, lane, flagLane);
      uint64_t* wire = reinterpret_cast<uint64_t*>(peerScratch[peer]) + dataWords + s * kSliceWords + 2 * lane;
      ddaLL128StoreWire<kWarp>(wire, regs, flag, flagLane);
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
      aborted = !ddaLLWaitLineFlags(&selfTok[peer], flag32, params, d0, f0, d1, f1);
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

    for (size_t g = flatBlockId; g < nGroups; g += total) {
      const size_t s = g * nwarps + warp;
      if (s >= slices) continue;
      uint64_t* wire = selfBase + dataWords + s * kSliceWords + 2 * lane;
      uint64_t vr[kDdaLL128WordsPerThread];
#if RCCL_HAVE_COOPERATIVE_ATOMIC_BUILTINS
      const bool landed = ddaLL128PollWireBatched<kLine, kWarp>(wire, vr, flag, lane, params);
#else
      const bool landed = ddaLL128PollWire<kLine, kWarp>(wire, vr, flag, lane, params);
#endif
      if (!landed) {
        aborted = true;
        break;
      }
      const size_t rem = bytes - s * kSliceBytes;
      ddaLL128StoreRegs<int8_t, kLine, kAligned, kWarp>(dst + s * kSliceBytes, vr,
                                                       rem < kSliceBytes ? (int)rem : kSliceBytes, lane, flagLane);
      ddaLL128ClearWire<kWarp>(wire);
    }

    if (!aborted && flatBlockId == 0) {
      for (int p = tid; p < nRanks; p += nthreads) {
        if (p == selfRank || p == root) continue;
        uint32_t d0, f0, d1, f1;
        if (!ddaLLWaitLineFlags(&selfTok[p], flag32, params, d0, f0, d1, f1)) {
          aborted = true;
          break;
        }
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
