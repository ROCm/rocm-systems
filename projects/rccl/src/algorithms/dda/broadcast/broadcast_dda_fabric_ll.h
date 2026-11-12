/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * LL-protocol one-shot broadcast for the DDA path. The root pushes the payload
 * to every non-root; every non-root writes a start token to every other rank.
 * A rank exits only after hearing from every rank, which keeps the two-bank
 * scratch safe for the next DDA call.
 *
 * Bank layout (16B lines): [nRanks token lines][bytes/8 data lines].
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

namespace dda::common {

// Data lines one bank holds after the token lines.
constexpr size_t ddaLLBcSlotPkts(size_t bankSize, int nRanks) {
  const size_t lines = bankSize / sizeof(LLPacket16);
  return lines > (size_t)nRanks ? lines - (size_t)nRanks : 0;
}

template <bool kAligned>
__device__ __forceinline__ void ddaBcLoad8(const char* p, uint32_t& lo, uint32_t& hi) {
  if constexpr (kAligned) {
    const uint32_t* w = reinterpret_cast<const uint32_t*>(p);
    lo = w[0];
    hi = w[1];
  } else {
    uint32_t w[2];
    __builtin_memcpy(w, p, sizeof(w));
    lo = w[0];
    hi = w[1];
  }
}

template <bool kAligned>
__device__ __forceinline__ void ddaBcStore8(char* p, uint32_t lo, uint32_t hi) {
  if constexpr (kAligned) {
    uint32_t* w = reinterpret_cast<uint32_t*>(p);
    w[0] = lo;
    w[1] = hi;
  } else {
    const uint32_t w[2] = {lo, hi};
    __builtin_memcpy(p, w, sizeof(w));
  }
}

// Splits [0, nPk) into nChunks contiguous ranges.
__device__ __forceinline__ void ddaBcChunkRange(size_t nPk, int chunk, int nChunks, size_t& begin, size_t& end) {
  const size_t per = (nPk + (size_t)nChunks - 1) / (size_t)nChunks;
  begin = (size_t)chunk * per;
  end = begin + per < nPk ? begin + per : nPk;
}

// Root grid: (nRanks-1) peer columns x chunks, as in AllGather LL128. Non-root
// grid: chunks x 1. Each block owns one contiguous packet range.
// kAligned: this rank's user buffers are 4B aligned (per rank, wire unaffected).
template <bool kAligned>
__launch_bounds__(512) __global__
  void ddaBroadcastFabricLL(char* const* __restrict__ peerScratch, // nRanks scratch bases
                            char* __restrict__ recvbuff, const char* __restrict__ sendbuff,
                            size_t bytes, // multiple of 16
                            int selfRank, int nRanks, int root, uint32_t* __restrict__ epochDev, int epochLen,
                            size_t bankSize, DdaKernelParams params) {
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  const int flatBlockId = blockIdx.x * gridDim.y + blockIdx.y;
  const int total = gridDim.x * gridDim.y;
  const uint32_t flag = ddaGetLLEpochInc(epochDev, flatBlockId, 1);
  const size_t bankLines = (size_t)(flag & 1u) * (bankSize / sizeof(LLPacket16));
  const size_t nPk = bytes >> 3;
  const bool isRoot = selfRank == root;

  const int chunk = isRoot ? (int)blockIdx.y : (int)blockIdx.x;
  const int nChunks = isRoot ? (int)gridDim.y : (int)gridDim.x;
  size_t pkBegin, pkEnd;
  ddaBcChunkRange(nPk, chunk, nChunks, pkBegin, pkEnd);

  LLPacket16* self = reinterpret_cast<LLPacket16*>(peerScratch[selfRank]) + bankLines;

  if (isRoot) {
    const int peer = (root + 1 + (int)blockIdx.x) % nRanks;
    LLPacket16* dst = reinterpret_cast<LLPacket16*>(peerScratch[peer]) + bankLines + nRanks;
    for (size_t pk = pkBegin + tid; pk < pkEnd; pk += nthreads) {
      uint32_t d0, d1;
      ddaBcLoad8<kAligned>(sendbuff + 8 * pk, d0, d1);
      ddaLLStoreLineB128(reinterpret_cast<uint32_t*>(&dst[pk]), d0, flag, d1, flag);
    }

    if (sendbuff != recvbuff) {
      for (size_t i = (size_t)flatBlockId * nthreads + tid; i < nPk; i += (size_t)total * nthreads) {
        uint32_t d0, d1;
        ddaBcLoad8<kAligned>(sendbuff + 8 * i, d0, d1);
        ddaBcStore8<kAligned>(recvbuff + 8 * i, d0, d1);
      }
    }

    if (blockIdx.y == 0 && tid == 0) {
      uint32_t d0, f0, d1, f1;
      (void)ddaLLWaitLineFlags(&self[peer], flag, params, d0, f0, d1, f1);
    }
  } else {
    if (flatBlockId == 0) {
      for (int p = tid; p < nRanks; p += nthreads) {
        if (p == selfRank) continue;
        LLPacket16* tok = reinterpret_cast<LLPacket16*>(peerScratch[p]) + bankLines + selfRank;
        ddaLLStoreLineB128(reinterpret_cast<uint32_t*>(tok), 0u, flag, 0u, flag);
      }
    }

    const LLPacket16* src = self + nRanks;
    for (size_t pk = pkBegin + tid; pk < pkEnd; pk += nthreads) {
      uint32_t d0, f0, d1, f1;
      if (!ddaLLWaitLineFlags(&src[pk], flag, params, d0, f0, d1, f1)) break;
      ddaBcStore8<kAligned>(recvbuff + 8 * pk, d0, d1);
    }

    if (flatBlockId == 0) {
      for (int p = tid; p < nRanks; p += nthreads) {
        if (p == selfRank || p == root) continue;
        uint32_t d0, f0, d1, f1;
        if (!ddaLLWaitLineFlags(&self[p], flag, params, d0, f0, d1, f1)) break;
      }
    }
  }

  ddaSetLLEpoch(epochDev, epochLen, flatBlockId, total, flag);
}

} // namespace dda::common
