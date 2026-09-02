/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * LL128-protocol all-to-all device kernel for the DDA fabric path (gfx1250).
 * Each 128B line holds 120B of payload (15 x uint64) + a trailing flag word;
 * 16 lanes cooperatively write one coalesced line, flag-last and unfenced (see
 * CollCommon_ll128.h). No GPU barrier; staging uses comm->ddaScratch reached via
 * comm->ddaPeerPtrsDev.
 *
 * All-to-all is the "personalized" analogue of all-gather: each rank sends a
 * distinct chunk (sendbuff[peer]) to each peer, and recvbuff[src] receives the
 * chunk sent by rank src. Only the scatter source and self-copy source differ
 * from all_gather_dda_fabric_ll128.h (per-peer offsets into sendbuff).
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

// Per-peer chunk hard cap and the resulting fixed slot stride in 128B lines
// (compile-time, so the double-buffered layout is identical on every rank and
// call). The effective size gate is the runtime LL128 threshold; this cap
// bounds the scratch footprint.
constexpr size_t kDdaLL128A2AMaxPerChunkBytes = 524288;              // 512 KiB
constexpr size_t kDdaLL128A2ASlotStrideLines =
  (kDdaLL128A2AMaxPerChunkBytes / 8 + (size_t)kDdaLL128DataElems - 1) / (size_t)kDdaLL128DataElems; // ceil(nWords/15)

// LL128 all-to-all kernel. 2D grid: grid.x == nRanks selects the peer column;
// grid.y == blocksPerPeer splits that peer's line range into gridDim.y chunks.
// The self column copies sendbuff[self] -> recvbuff[self] locally; other columns
// scatter this rank's chunk-for-peer-b (sendbuff[b]) into peer b's slot (warp-
// cooperative 128B writes), then poll their own slot b for peer b's chunk and
// unpack into recvbuff[b].
template <typename T, int NRANKS_CT>
#if defined(USE_ROCM)
__launch_bounds__(1024)
#endif
  __global__ void ddaAllToAllFabricLL128(T* const* __restrict__ peerScratch, // ddaPeerPtrsDev: nRanks scratch bases
                                         T* __restrict__ recvbuff, // local user output (nRanks chunks)
                                         const T* __restrict__ sendbuff, // local user input (nRanks chunks)
                                         size_t perChunkBytes, // per-peer chunk payload; multiple of 16
                                         int selfRank, int nRanksRt,
                                         uint32_t* __restrict__ epochDev, // per-block LL epoch cells
                                         int epochLen) { // number of cells in epochDev

  const int nRanks = NRANKS_CT ? NRANKS_CT : nRanksRt;
  const int peer = blockIdx.x; // grid.x == nRanks: one column/peer
  if (peer >= nRanks) return; // safety if grid.x > nRanks
  const int chunk = blockIdx.y; // grid.y == blocksPerPeer
  const int nChunks = gridDim.y; // >= 1

  const size_t nWords = perChunkBytes >> 3; // 8B payload words
  const size_t numLines = ddaLL128NumLines(nWords); // 128B lines this size
  const size_t slot = kDdaLL128A2ASlotStrideLines; // lines per slot

  // On-device, graph-safe flag/bank derivation.
  const int flatBlockId = blockIdx.x * gridDim.y + blockIdx.y;
  const int total = gridDim.x * gridDim.y;
  __shared__ uint32_t s_flag;
  const uint32_t flag = ddaLLEpochBegin(epochDev, flatBlockId, s_flag);
  const size_t bankOffsetLines = (size_t)(flag & 1u) * (size_t)nRanks * slot;

  // This block's line range [lnBegin, lnEnd); [0, numLines) when nChunks == 1.
  const size_t lnPerChunk = (numLines + (size_t)nChunks - 1) / (size_t)nChunks;
  const size_t lnBegin = (size_t)chunk * lnPerChunk;
  size_t lnEnd = lnBegin + lnPerChunk;
  if (lnEnd > numLines) lnEnd = numLines;

  if (peer == selfRank) {
    // self column: local copy sendbuff[self] -> recvbuff[self] (16B nontemporal).
    const uint4* s4 =
      reinterpret_cast<const uint4*>(reinterpret_cast<const char*>(sendbuff) + (size_t)selfRank * perChunkBytes);
    uint4* d4 = reinterpret_cast<uint4*>(reinterpret_cast<char*>(recvbuff) + (size_t)selfRank * perChunkBytes);
    const size_t nVec = perChunkBytes >> 4; // number of 16B chunks
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const size_t vecPerChunk = (nVec + (size_t)nChunks - 1) / (size_t)nChunks;
    const size_t vBegin = (size_t)chunk * vecPerChunk;
    size_t vEnd = vBegin + vecPerChunk;
    if (vEnd > nVec) vEnd = nVec;
    for (size_t i = vBegin + tid; i < vEnd; i += nthreads) {
      const uint4* p = &s4[i];
      uint4 v;
      v.x = __builtin_nontemporal_load(&p->x);
      v.y = __builtin_nontemporal_load(&p->y);
      v.z = __builtin_nontemporal_load(&p->z);
      v.w = __builtin_nontemporal_load(&p->w);
      uint4* q = &d4[i];
      __builtin_nontemporal_store(v.x, &q->x);
      __builtin_nontemporal_store(v.y, &q->y);
      __builtin_nontemporal_store(v.z, &q->z);
      __builtin_nontemporal_store(v.w, &q->w);
    }
  } else {
    // 16 lanes cooperate on one 128B line.
    const int group = threadIdx.x / kDdaLL128Lanes;
    const int lane = threadIdx.x % kDdaLL128Lanes;
    const int groups = blockDim.x / kDdaLL128Lanes;
    const uint64_t* sw =
      reinterpret_cast<const uint64_t*>(reinterpret_cast<const char*>(sendbuff) + (size_t)peer * perChunkBytes);

    // scatter: write my chunk-for-peer into peer's slot (== selfRank), flag-last.
    LLLine128* dst = reinterpret_cast<LLLine128*>(peerScratch[peer]) + (size_t)selfRank * slot + bankOffsetLines;
    for (size_t ln = lnBegin + group; ln < lnEnd; ln += groups) {
      const size_t base = ln * (size_t)kDdaLL128DataElems;
      if (lane < kDdaLL128DataElems) {
        const size_t e = base + (size_t)lane;
        const uint64_t v = (e < nWords) ? sw[e] : 0ull;
        ddaLL128StoreWord(&dst[ln].w[lane], v);
      }
      // Unfenced: the payload store above precedes this flag store in warp
      // program order; gfx1250 preserves the visibility order.
      if (lane == kDdaLL128FlagElem) {
        ddaLL128StoreWord(&dst[ln].w[kDdaLL128FlagElem], (uint64_t)flag);
      }
    }

    // gather: poll my slot for peer, unpack into recvbuff[peer].
    LLLine128* src = reinterpret_cast<LLLine128*>(peerScratch[selfRank]) + bankOffsetLines + (size_t)peer * slot;
    uint64_t* out = reinterpret_cast<uint64_t*>(reinterpret_cast<char*>(recvbuff) + (size_t)peer * perChunkBytes);
    for (size_t ln = lnBegin + group; ln < lnEnd; ln += groups) {
      const size_t base = ln * (size_t)kDdaLL128DataElems;
      // all 16 lanes poll the shared flag word (broadcast); unfenced.
      while (ddaLL128LoadWord(&src[ln].w[kDdaLL128FlagElem]) != (uint64_t)flag) {
      }
      if (lane < kDdaLL128DataElems) {
        const size_t e = base + (size_t)lane;
        const uint64_t v = ddaLL128LoadWord(&src[ln].w[lane]);
        if (e < nWords) out[e] = v;
      }
    }
  }

  ddaLLEpochEnd(epochDev, flatBlockId, total, epochLen, flag);
}

} // namespace dda::common
