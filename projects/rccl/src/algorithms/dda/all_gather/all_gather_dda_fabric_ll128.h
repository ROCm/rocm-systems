/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * LL128-protocol all-gather device kernel for the DDA fabric path (gfx1250).
 * A slice is 2 KiB, eight uint64 registers per lane; a warp moves one at a
 * time. No GPU barrier; staging uses comm->ddaScratch reached via
 * comm->ddaPeerPtrsDev.
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

// Slot geometry, derived from the scratch bank the host picked. The bank splits
// evenly across ranks and the per-rank stride is floored to a whole number of
// slices, so a slot always begins on a slice boundary and the 16B wire accesses
// stay aligned however nRanks divides the bank.
constexpr size_t ddaLL128AgSlotWords(size_t bankSize, int nRanks) {
  return ddaLL128SlotWords(bankSize, nRanks, 1);
}

// LL128 all-gather. 2D grid: grid.x == nRanks - 1 places one column per remote
// peer; grid.y splits that peer's slices across blocks, and a warp takes kUnroll
// (1 or 4) of its slices per pass. Each column packs this rank's payload into its
// peer's slot, then polls its own slot for that peer's payload and unpacks it.
template <typename T, int kUnroll>
#if defined(USE_ROCM)
__launch_bounds__(1024)
#endif
  __global__ void ddaAllGatherFabricLL128(T* const* __restrict__ peerScratch, // ddaPeerPtrsDev: nRanks scratch bases
                                          T* __restrict__ recvbuff, // local user output
                                          const T* __restrict__ sendbuff, // local user input
                                          size_t perRankBytes, // per-rank payload; multiple of 16
                                          int selfRank, int nRanks,
                                          uint32_t* __restrict__ epochDev, // per-block LL epoch cells
                                          int epochLen, // number of cells in epochDev
                                          size_t slicesTotal, // slices this call uses
                                          size_t bankSize, // scratch bank size (from host)
                                          DdaKernelParams params) {

  const int nPeers = (int)blockIdx.x;
  const int peer = (selfRank + nPeers + 1) % nRanks;

  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  const int lane = tid % kDdaLL128Warp;
  const int warp = tid / kDdaLL128Warp;
  const int nwarps = nthreads / kDdaLL128Warp;
  const bool flagLane = ddaLL128IsFlagLane(lane);

  const int flatBlockId = (int)(blockIdx.x * gridDim.y + blockIdx.y);
  const int total = (int)(gridDim.x * gridDim.y);
  const uint32_t flag32 = ddaGetLLEpochInc(epochDev, flatBlockId, 1);
  const uint64_t flag = ((uint64_t)flag32 << 32) | (uint64_t)flag32;
  const size_t slotWords = ddaLL128AgSlotWords(bankSize, nRanks);
  // Bank 1 sits at bankSize rather than right after the slots, so every DDA LL
  // tier sharing this scratch and epoch counter banks at the same byte offsets.
  const uint64_t bankWords = (uint64_t)(flag32 & 1u) * (uint64_t)(bankSize / sizeof(uint64_t));

  // Slices stride by warp within this peer's column only.
  const size_t gwarp = (size_t)blockIdx.y * (size_t)nwarps + (size_t)warp;
  const size_t wstride = (size_t)gridDim.y * (size_t)nwarps;

  const int8_t* srcBytes = reinterpret_cast<const int8_t*>(sendbuff);
  uint64_t* scatterSlot = reinterpret_cast<uint64_t*>(peerScratch[peer]) + bankWords +
    (uint64_t)selfRank * (uint64_t)slotWords + 2 * lane;
  const uint64_t* gatherSlot = reinterpret_cast<const uint64_t*>(peerScratch[selfRank]) + bankWords +
    (uint64_t)peer * (uint64_t)slotWords + 2 * lane;
  int8_t* dstBytes = reinterpret_cast<int8_t*>(recvbuff) + (size_t)peer * perRankBytes;

  // Phase 1: pack and push this column's slices to the one peer it owns.
  const size_t pstride = (size_t)kUnroll * wstride;
  for (size_t s0 = gwarp; s0 < slicesTotal; s0 += pstride) {
    ddaLL128SlicePass<kUnroll>(s0, wstride, slicesTotal, [&](auto slices) {
      ddaLL128SendSlices<slices>(srcBytes, perRankBytes, scatterSlot, s0, wstride, flag, lane, flagLane);
    });
  }

  // Local copy sendbuff -> recvbuff[selfRank].
  ddaGridCopy16B<kUnroll>(sendbuff, reinterpret_cast<char*>(recvbuff) + (size_t)selfRank * perRankBytes,
                          perRankBytes, (size_t)flatBlockId * (size_t)nthreads + (size_t)tid,
                          (size_t)total * (size_t)nthreads);

  // Phase 2: poll the same slices in that peer's slot and unpack.
  for (size_t s0 = gwarp; s0 < slicesTotal; s0 += pstride) {
    bool done = true;
    ddaLL128SlicePass<kUnroll>(s0, wstride, slicesTotal, [&](auto slices) {
      done = ddaLL128RecvSlices<slices>(gatherSlot, dstBytes, perRankBytes, s0, wstride, flag, lane, flagLane, params);
    });
    if (!done) {
      break; // not return: every thread must reach the __syncthreads below
    }
  }
#if (defined(__gfx1250__) || defined(__gfx1250_strict__))
  asm volatile("s_wait_storecnt 0x0" ::: "memory");
#endif
  __syncthreads();
  ddaSetLLEpoch(epochDev, epochLen, flatBlockId, total, flag32);
}

} // namespace dda::common
