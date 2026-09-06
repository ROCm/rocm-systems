/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * LL128-protocol reduce-scatter device kernel for the DDA fabric path (gfx1250).
 * Each 128B line holds 120B of payload (15 x uint64) + a trailing flag word;
 * 16 lanes cooperatively write one coalesced line, flag-last and unfenced (see
 * CollCommon_ll128.h). No GPU barrier; staging uses comm->ddaScratch reached via
 * comm->ddaPeerPtrsDev.
 *
 * Reduce-scatter combines the personalized scatter of all-to-all with the fold
 * of all-reduce: rank selfRank owns output shard selfRank, so recvbuff[i] =
 * sum over ranks r of sendbuff_r[selfRank*recvcount + i]. Only the per-peer
 * publish source and the reduce seed differ from all_reduce_dda_fabric_ll128.h.
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

// Per-shard hard cap and the resulting fixed slot stride in 128B lines
// (compile-time, so the double-buffered layout is identical on every rank and
// call). The effective size gate is the runtime LL128 threshold; this cap
// bounds the scratch footprint.
constexpr size_t kDdaLL128RsMaxBytes = 524288;                       // 512 KiB
constexpr size_t kDdaLL128RsSlotStrideLines =
  (kDdaLL128RsMaxBytes / 8 + (size_t)kDdaLL128DataElems - 1) / (size_t)kDdaLL128DataElems; // ceil(nWords/15)

// LL128 reduce-scatter kernel. 1D grid over 128B lines of the per-rank shard;
// within a block the threads split into 16-lane groups, each owning one line.
//
// Phase 1 (scatter): rank selfRank writes its chunk-for-peer (sendbuff[peer])
// into peer's scratch at slot selfRank as LL128 lines (flag-last).
// Phase 2 (reduce): rank selfRank seeds the accumulator with its own self-chunk
// (sendbuff[selfRank]), polls its own scratch slots for the other ranks (word 15
// == flag), and folds them into recvbuff. Scratch is double-buffered: bank =
// flag & 1.
template <typename T, int NRANKS_CT>
#if defined(USE_ROCM)
__launch_bounds__(1024)
#endif
  __global__
  void ddaReduceScatterFabricLL128(T* const* __restrict__ peerScratch, // ddaPeerPtrsDev: nRanks scratch bases
                                   T* __restrict__ recvbuff, // local user output (recvcount elems)
                                   const T* __restrict__ sendbuff, // local user input (recvcount*nRanks)
                                   size_t recvcount, // per-rank shard element count
                                   int selfRank, int nRanksRt,
                                   uint32_t* __restrict__ epochDev, // per-block LL epoch cells
                                   int epochLen, // number of cells in epochDev
                                   const uint32_t* __restrict__ abortFlag) { // comm->abortFlagDev

  const int nRanks = NRANKS_CT ? NRANKS_CT : nRanksRt;
  const size_t bytes = recvcount * sizeof(T);
  const size_t nWords = bytes >> 3; // 8B payload words
  const size_t numLines = ddaLL128NumLines(nWords); // 128B lines this size
  const size_t slot = kDdaLL128RsSlotStrideLines; // lines per slot

  // On-device, graph-safe flag/bank derivation (1D grid: flatBlockId=blockIdx.x).
  const int flatBlockId = blockIdx.x;
  const int total = gridDim.x;
  __shared__ uint32_t s_flag;
  const uint32_t flag = ddaLLEpochBegin(epochDev, flatBlockId, s_flag);
  const size_t bankOffsetLines = (size_t)(flag & 1u) * (size_t)nRanks * slot;

  // 16 lanes cooperate on one 128B line; grid-stride over line-groups.
  const int group = threadIdx.x / kDdaLL128Lanes;
  const int lane = threadIdx.x % kDdaLL128Lanes;
  const int groups = blockDim.x / kDdaLL128Lanes;
  const size_t groupBase = (size_t)blockIdx.x * (size_t)groups + (size_t)group;
  const size_t groupStride = (size_t)gridDim.x * (size_t)groups;

  const uint64_t* sw = reinterpret_cast<const uint64_t*>(sendbuff);

  // Phase 1: scatter my chunk-for-peer (sendbuff[peer]) into peer's slot[self].
  for (size_t ln = groupBase; ln < numLines; ln += groupStride) {
    const size_t base = ln * (size_t)kDdaLL128DataElems;
#pragma unroll
    for (int r = 1; r < nRanks; ++r) {
      const int peer = (selfRank + r) % nRanks;
      const uint64_t* swPeer = sw + (size_t)peer * nWords;
      LLLine128* dst = reinterpret_cast<LLLine128*>(peerScratch[peer]) + bankOffsetLines + (size_t)selfRank * slot;
      if (lane < kDdaLL128DataElems) {
        const size_t e = base + (size_t)lane;
        const uint64_t v = (e < nWords) ? swPeer[e] : 0ull;
        ddaLL128StoreWord(&dst[ln].w[lane], v);
      }
      // Unfenced: the payload store above precedes this flag store in warp
      // program order; gfx1250 preserves the visibility order.
      if (lane == kDdaLL128FlagElem) {
        ddaLL128StoreWord(&dst[ln].w[kDdaLL128FlagElem], (uint64_t)flag);
      }
    }
  }

  // Phase 2: seed with my self-chunk, poll my slots for the others, fold.
  const uint64_t* swSelf = sw + (size_t)selfRank * nWords;
  LLLine128* myBase = reinterpret_cast<LLLine128*>(peerScratch[selfRank]) + bankOffsetLines;
  uint64_t* out = reinterpret_cast<uint64_t*>(recvbuff);
  for (size_t ln = groupBase; ln < numLines; ln += groupStride) {
    const size_t base = ln * (size_t)kDdaLL128DataElems;
    const size_t e = base + (size_t)lane;
    const bool hasWord = (lane < kDdaLL128DataElems) && (e < nWords);
    uint64_t acc = hasWord ? swSelf[e] : 0ull;
    for (int r = 1; r < nRanks; ++r) {
      const int peer = (selfRank + r) % nRanks;
      LLLine128* src = myBase + (size_t)peer * slot;
      // All 16 lanes poll the shared flag word (broadcast); unfenced.
      if (!ddaLL128WaitFlag(&src[ln].w[kDdaLL128FlagElem], (uint64_t)flag, abortFlag)) {
        ddaLLEpochEnd(epochDev, flatBlockId, total, epochLen, flag);
        return;
      }
      if (hasWord) {
        const uint64_t d = ddaLL128LoadWord(&src[ln].w[lane]);
        acc = ddaLL128AddWord<T>(acc, d);
      }
    }
    if (hasWord) {
      out[e] = acc;
    }
  }

  ddaLLEpochEnd(epochDev, flatBlockId, total, epochLen, flag);
}

} // namespace dda::common
