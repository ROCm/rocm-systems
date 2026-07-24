/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Variant of all_reduce_dda_fabric_ll128.h that carries payload in ALL 16 words
 * of each 128B line (no in-line flag word), and synchronizes phases with a
 * separate arrival barrier instead of per-line flags:
 *
 *  - Data staging starts 1 MiB into each peer's scratch (peerScratch +
 *    kDdaLL128ArReserveBytes). Lines are double-buffered (bank = flag & 1).
 *  - The first 1 MiB is repurposed as the barrier flag region: each rank owns a
 *    per-rank sub-region, indexed by global warp id.
 *
 * Sync (between phase 1 and phase 2): every warp publishes the current epoch
 * flag to each peer's flag slot, then waits until every peer has published to
 * it (an all-to-all, per-global-warp arrival barrier). Because the line->warp
 * mapping is identical on every rank and in both phases, peer p's warp w wrote
 * exactly the lines self's warp w reads, so a per-warp barrier is sufficient.
 * Unlike the data region, the flag region is NOT double-buffered, so the wait
 * uses a monotonic ">= flag" compare to tolerate a peer racing one call ahead.
 *
 * Ordering: like the base LL128 kernel this is unfenced and relies on gfx1250
 * preserving wave program-order remote-write visibility -- but here the flag and
 * payload live in separate scratch regions, a stronger assumption than the base
 * kernel's same-line flag. Validate on hardware, or add __threadfence_system()
 * before the flag store to be conservative.
 *
 * The kernel is codepath-agnostic (it only needs the peer scratch table), so the
 * same kernel can back an IPC or a fabric launcher.
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

#include "algorithms/CollCommon.h"
#include "algorithms/CollCommon_ll128.h"
#include "algorithms/all_reduce/all_reduce_dda_fabric_ll128.h" // kDdaLL128ArMaxBytes

namespace meta::comms {

// Front region of every peer's scratch reserved for the barrier flag exchange.
// Data staging begins at peerScratch + kDdaLL128ArReserveBytes. The per-message
// cap (kDdaLL128ArMaxBytes) is shared with the base LL128 all-reduce.
constexpr size_t kDdaLL128ArReserveBytes = 1048576;                 // 1 MiB

// LL128 flat all-reduce kernel (full-line + barrier variant). 1D grid over 128B
// lines; within a block the threads split into 16-lane groups, each owning one
// line at a time.
//
// Phase 1 (publish): rank selfRank writes its full sendbuff into every peer's
// data slot selfRank; all 16 words of each line are payload (no flag word).
// Barrier: all-to-all, per-global-warp arrival barrier via the reserved flag
// region (see file header) -- publish this epoch's flag to every peer, then wait
// (monotonic ">= flag") until every peer has published.
// Phase 2 (reduce): rank selfRank reads its own data slot for each other rank
// and folds those words with its own sendbuff into recvbuff. The barrier (not a
// per-line flag) provides the cross-rank ordering. Data is double-buffered:
// bank = flag & 1. Data slot addressing is relative to peerScratch +
// kDdaLL128ArReserveBytes.
template <typename T, int NRANKS_CT>
#if defined(USE_ROCM)
__launch_bounds__(1024)
#endif
__global__ void ddaAllReduceFlatLL128_warpsync(
    T* const* __restrict__ peerScratch,    // ddaPeerPtrsDev: nRanks scratch bases
    T* __restrict__ recvbuff,              // local user output
    const T* __restrict__ sendbuff,        // local user input
    size_t count,                          // full-message element count
    int selfRank,
    int nRanksRt,
    uint32_t* __restrict__ epochDev,       // per-block LL epoch cells
    int epochLen,                          // number of cells in epochDev
    size_t slotStrideLines) {              // per-call lines/slot (from host)

  const int nRanks = NRANKS_CT ? NRANKS_CT : nRanksRt;
  const size_t bytes = count * sizeof(T);
  const size_t nWords = bytes >> 3;                        // 8B payload words
  // All 16 words of a line are payload here (no in-line flag word), so a line
  // carries kDdaLL128LineElems words = 128B of data.
  const size_t numLines =
      (nWords + (size_t)kDdaLL128LineElems - 1) / (size_t)kDdaLL128LineElems;
  const size_t slot = slotStrideLines;                     // lines per slot (per-call)

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

  // Phase 1: publish my payload into every peer's slot[selfRank]. No in-line
  // flag word: all 16 lanes carry payload.
  // Staging base is offset by kDdaLL128ArReserveBytes (first 1 MiB reserved).
  for (size_t ln = groupBase; ln < numLines; ln += groupStride) {
    const size_t base = ln * (size_t)kDdaLL128LineElems;
    uint64_t v = 0ull;
    if (lane < kDdaLL128LineElems) {
      const size_t e = base + (size_t)lane;
      v = (e < nWords) ? sw[e] : 0ull;
    }
    #pragma unroll
    for (int r = 1; r < nRanks; ++r) {
      const int peer = (selfRank + r) % nRanks;
      LLLine128* dst = reinterpret_cast<LLLine128*>(
          reinterpret_cast<char*>(peerScratch[peer]) + kDdaLL128ArReserveBytes) +
          bankOffsetLines + (size_t)selfRank * slot;
      if (lane < kDdaLL128LineElems) {
        ddaLL128StoreWord(&dst[ln].w[lane], v);
      }
    }
  }

  // Arrival barrier (between phase 1 and phase 2). The reserved 1 MiB front
  // region is split into per-rank sub-regions (flag_perRankBytes each), indexed
  // by global warp id. Each warp's lanes [0, nRanks) publish this epoch's flag
  // to every peer (lane r -> peer r, into peer's sub-region[selfRank]), then the
  // same lanes wait until every peer has published to this warp's slot.
  size_t flag_perRankBytes = kDdaLL128ArReserveBytes / nRanks;
  size_t flag_selfRankOffset = flag_perRankBytes * selfRank;
  const size_t gtid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  int gwarpid = gtid / (int)warpSize;
  int lwarpid = gtid % (int)warpSize;
  #pragma unroll
  for (int r = lwarpid; r < nRanks; r+=(int)warpSize) {
    uint64_t *dst_flag = reinterpret_cast<uint64_t*>(reinterpret_cast<char*>(peerScratch[r]) + flag_selfRankOffset);
    ddaLL128StoreWord(dst_flag + gwarpid, flag);
  }

  #pragma unroll
  for (int r = lwarpid; r < nRanks; r+=(int)warpSize) {
      const uint64_t *src_flag = reinterpret_cast<size_t*>(reinterpret_cast<char*>(peerScratch[selfRank]) + flag_perRankBytes * r);
      // Monotonic compare (not ==): epochs strictly increase, so ">= flag" means
      // "peer arrived this call or later". This tolerates a peer racing one call
      // ahead and overwriting its flag slot, which is otherwise unprotected
      // because the flag region (unlike the data region) is not double-buffered.
      while (ddaLL128LoadWord(src_flag + gwarpid) < (uint64_t)flag) {}
  }

  __syncwarp();

  // Phase 2: read my slots for the other ranks, fold with my own data. No flag
  // word to poll; all 16 lanes carry payload.
  LLLine128* myBase = reinterpret_cast<LLLine128*>(
      reinterpret_cast<char*>(peerScratch[selfRank]) + kDdaLL128ArReserveBytes) +
      bankOffsetLines;
  uint64_t* out = reinterpret_cast<uint64_t*>(recvbuff);
  for (size_t ln = groupBase; ln < numLines; ln += groupStride) {
    const size_t base = ln * (size_t)kDdaLL128LineElems;
    const size_t e = base + (size_t)lane;
    const bool hasWord = (lane < kDdaLL128LineElems) && (e < nWords);
    uint64_t acc = hasWord ? sw[e] : 0ull;
    for (int r = 1; r < nRanks; ++r) {
      const int peer = (selfRank + r) % nRanks;
      LLLine128* src = myBase + (size_t)peer * slot;
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

} // namespace meta::comms
