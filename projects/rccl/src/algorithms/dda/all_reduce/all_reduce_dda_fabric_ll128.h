/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * LL128-protocol all-reduce device kernel for the DDA fabric path (gfx1250).
 * Each 128B line holds 120B of payload (15 x uint64) + a trailing flag word;
 * 16 lanes cooperatively write one coalesced line, flag-last and unfenced (see
 * CollCommon_ll128.h). No GPU barrier; staging uses comm->ddaScratch reached via
 * comm->ddaPeerPtrsDev.
 *
 * All-reduce analogue of all_gather_dda_fabric_ll128.h and the 128B counterpart
 * of all_reduce_dda_ll.h: publish this rank's payload to every peer's slot, then
 * poll our own slots for the other ranks and fold them into recvbuff.
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

// LL128 flat all-reduce kernel. 1D grid over 128B lines; within a block the
// threads split into 16-lane groups, each owning one line at a time.
//
// Phase 1 (publish): rank selfRank writes its full sendbuff into every peer's
// scratch at slot selfRank, as LL128 lines carrying the epoch flag (flag-last).
// Phase 2 (reduce): rank selfRank polls its own scratch slot for each other
// rank (waiting on word 15 == flag), and folds those words with its own
// sendbuff into recvbuff. Flag polling provides the cross-rank ordering, so no
// GPU barrier is used. Scratch is double-buffered: bank = flag & 1.
template <typename T, int NRANKS_CT>
#if defined(USE_ROCM)
__launch_bounds__(1024)
#endif
  __global__ void ddaAllReduceFlatLL128(T* const* __restrict__ peerScratch,    // ddaPeerPtrsDev: nRanks scratch bases
                                        T* __restrict__ recvbuff,              // local user output
                                        const T* __restrict__ sendbuff,        // local user input
                                        size_t count,                          // full-message element count
                                        int selfRank, int nRanksRt,
                                        uint32_t* __restrict__ epochDev, // per-block LL epoch cells (shared AG+AR)
                                        int epochLen, // number of cells in epochDev
                                        size_t slotStrideLines) { // per-call lines/slot (from host)

  const int nRanks = NRANKS_CT ? NRANKS_CT : nRanksRt;
  const size_t bytes = count * sizeof(T);
  const size_t nWords = bytes >> 3; // 8B payload words
  const size_t numLines = ddaLL128NumLines(nWords); // 128B lines this size
  const size_t slot = slotStrideLines; // lines per slot (per-call)

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

  // Phase 1: publish my payload into every peer's slot[selfRank], flag-last.
  for (size_t ln = groupBase; ln < numLines; ln += groupStride) {
    const size_t base = ln * (size_t)kDdaLL128DataElems;
    uint64_t v = 0ull;
    if (lane < kDdaLL128DataElems) {
      const size_t e = base + (size_t)lane;
      v = (e < nWords) ? sw[e] : 0ull;
    }
#pragma unroll
    for (int r = 1; r < nRanks; ++r) {
      const int peer = (selfRank + r) % nRanks;
      LLLine128* dst = reinterpret_cast<LLLine128*>(peerScratch[peer]) + bankOffsetLines + (size_t)selfRank * slot;
      if (lane < kDdaLL128DataElems) {
        ddaLL128StoreWord(&dst[ln].w[lane], v);
      }
      // Unfenced: the payload store above precedes this flag store in warp
      // program order; gfx1250 preserves the visibility order.
      if (lane == kDdaLL128FlagElem) {
        ddaLL128StoreWord(&dst[ln].w[kDdaLL128FlagElem], (uint64_t)flag);
      }
    }
  }

  // Phase 2: poll my slots for the other ranks, fold with my own data.
  LLLine128* myBase = reinterpret_cast<LLLine128*>(peerScratch[selfRank]) + bankOffsetLines;
  uint64_t* out = reinterpret_cast<uint64_t*>(recvbuff);
  for (size_t ln = groupBase; ln < numLines; ln += groupStride) {
    const size_t base = ln * (size_t)kDdaLL128DataElems;
    const size_t e = base + (size_t)lane;
    const bool hasWord = (lane < kDdaLL128DataElems) && (e < nWords);
    uint64_t acc = hasWord ? sw[e] : 0ull;
    for (int r = 1; r < nRanks; ++r) {
      const int peer = (selfRank + r) % nRanks;
      LLLine128* src = myBase + (size_t)peer * slot;
      // All 16 lanes poll the shared flag word (broadcast); unfenced.
      while (ddaLL128LoadWord(&src[ln].w[kDdaLL128FlagElem]) != (uint64_t)flag) {
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
