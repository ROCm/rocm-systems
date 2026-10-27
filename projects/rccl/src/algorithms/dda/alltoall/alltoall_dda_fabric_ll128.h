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

// ---- register-packed (16B/lane, warp-per-2KiB-slice) A2A, 8-lane cooperative atomic
// 128B lines, built on the same slice helpers as all_gather_dda_fabric_ll128.h. One
// atomic 16B load per line both polls the flag and carries the payload (no dependent
// second load); a slice's sendbuff loads are all issued before its first wire store;
// peer columns are rotated (peer = self+x+1) so the grid does not hammer one
// destination at a time; the self copy is spread over the whole grid; slots are carved
// from the shared bank layout (ddaBankSize), so the per-chunk cap is the slot size.
// The user buffers take cached access, as this kernel's own helpers did.

constexpr size_t ddaLL128A2ASlotWords(size_t bankSize, int nRanks) {
  return ddaLL128SlotWords(bankSize, nRanks, 1);
}

template <typename T>
#if defined(USE_ROCM)
__launch_bounds__(1024)
#endif
  __global__ void ddaAllToAllFabricLL128V2(T* const* __restrict__ peerScratch, T* __restrict__ recvbuff,
                                           const T* __restrict__ sendbuff,
                                           size_t perChunkBytes, // per-peer payload; multiple of 16
                                           int selfRank, int nRanks, uint32_t* __restrict__ epochDev, int epochLen,
                                           size_t slicesTotal, // slices per peer chunk
                                           size_t bankSize, DdaKernelParams params) {
  constexpr DdaLL128UserAccess kUser = DdaLL128UserAccess::Cached;
  const int peer = (selfRank + (int)blockIdx.x + 1) % nRanks;

  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  const int lane = tid % kDdaLL128Warp;
  const int warp = tid / kDdaLL128Warp;
  const int nwarps = nthreads / kDdaLL128Warp;
  const bool flagLane = ddaLL128IsFlagLane(lane);

  const int flatBlockId = (int)(blockIdx.x * gridDim.y + blockIdx.y);
  const int total = (int)(gridDim.x * gridDim.y);
  const uint32_t flag32 = ddaLLEpochLoad(epochDev, flatBlockId, 1u);
  const uint64_t flag = ((uint64_t)flag32 << 32) | (uint64_t)flag32;
  const size_t slotWords = ddaLL128A2ASlotWords(bankSize, nRanks);
  const uint64_t bankWords = (uint64_t)(flag32 & 1u) * (uint64_t)(bankSize / sizeof(uint64_t));

  const size_t gwarp = (size_t)blockIdx.y * (size_t)nwarps + (size_t)warp;
  const size_t wstride = (size_t)gridDim.y * (size_t)nwarps;

  const int8_t* srcBytes = reinterpret_cast<const int8_t*>(sendbuff) + (size_t)peer * perChunkBytes;
  uint64_t* scatterSlot =
    reinterpret_cast<uint64_t*>(peerScratch[peer]) + bankWords + (uint64_t)selfRank * (uint64_t)slotWords + 2 * lane;
  const uint64_t* gatherSlot = reinterpret_cast<const uint64_t*>(peerScratch[selfRank]) + bankWords +
                               (uint64_t)peer * (uint64_t)slotWords + 2 * lane;
  int8_t* dstBytes = reinterpret_cast<int8_t*>(recvbuff) + (size_t)peer * perChunkBytes;

  for (size_t s = gwarp; s < slicesTotal; s += wstride)
    ddaLL128SendSlice<kUser>(srcBytes + s * (size_t)kDdaLL128DataBytesPerSlice, ddaLL128SliceBytes(perChunkBytes, s),
                             scatterSlot + s * (size_t)kDdaLL128WireWordsPerSlice, flag, lane, flagLane);

  ddaGridCopy16B(reinterpret_cast<const int8_t*>(sendbuff) + (size_t)selfRank * perChunkBytes,
                 reinterpret_cast<int8_t*>(recvbuff) + (size_t)selfRank * perChunkBytes, perChunkBytes,
                 (size_t)flatBlockId * (size_t)nthreads + (size_t)tid, (size_t)total * (size_t)nthreads);

  for (size_t s = gwarp; s < slicesTotal; s += wstride) {
    if (!ddaLL128RecvSlice<kUser>(gatherSlot + s * (size_t)kDdaLL128WireWordsPerSlice,
                                  dstBytes + s * (size_t)kDdaLL128DataBytesPerSlice,
                                  ddaLL128SliceBytes(perChunkBytes, s), flag, lane, flagLane, params))
      break; // aborted; the epoch commit below still runs
  }
  ddaLLEpochCommit(epochDev, epochLen, flatBlockId, total, flag32);
}

} // namespace dda::common
