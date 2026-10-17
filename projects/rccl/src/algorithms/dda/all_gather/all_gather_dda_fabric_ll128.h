/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * LL128-protocol all-gather device kernel for the DDA fabric path (gfx1250).
 * A warp owns one 2 KiB slice: eight uint64 registers per lane. 
 * No GPU barrier; staging uses comm->ddaScratch reached via comm->ddaPeerPtrsDev.
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

// How many of its own pushes a warp keeps ahead of its unpacks.
constexpr size_t kDdaLL128AGUnpackLag = 2;

// LL128 all-gather. 2D grid: grid.x == nRanks - 1 places one column per remote
// peer; grid.y splits that peer's slices across blocks, one
// warp per slice. Each column packs this rank's payload into its peer's slot
// and, a few slices behind, polls its own slot for that peer's payload and
// unpacks it.
template <typename T, int NRANKS_CT>
#if defined(USE_ROCM)
__launch_bounds__(1024)
#endif
  __global__ void ddaAllGatherFabricLL128(T* const* __restrict__ peerScratch, // ddaPeerPtrsDev: nRanks scratch bases
                                          T* __restrict__ recvbuff, // local user output
                                          const T* __restrict__ sendbuff, // local user input
                                          size_t perRankBytes, // per-rank payload; multiple of 16
                                          int selfRank, int nRanksRt,
                                          uint32_t* __restrict__ epochDev, // per-block LL epoch cells
                                          int epochLen, // number of cells in epochDev
                                          size_t slicesTotal, // slices this call uses
                                          size_t bankSize) { // scratch bank size (from host)

  const int nRanks = NRANKS_CT ? NRANKS_CT : nRanksRt;

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
    (uint64_t)selfRank * (uint64_t)slotWords;
  const uint64_t* gatherSlot = reinterpret_cast<const uint64_t*>(peerScratch[selfRank]) + bankWords +
    (uint64_t)peer * (uint64_t)slotWords;
  int8_t* dstBytes = reinterpret_cast<int8_t*>(recvbuff) + (size_t)peer * perRankBytes;

  auto sliceElts = [&](size_t s) {
    const size_t rem = perRankBytes - s * (size_t)kDdaLL128DataBytesPerSlice;
    return rem < (size_t)kDdaLL128DataBytesPerSlice ? (int)rem : kDdaLL128DataBytesPerSlice;
  };
  // Pack slice s and push it to the one peer this column owns.
  auto push = [&](size_t s) {
    uint64_t regs[kDdaLL128WordsPerThread];
    ddaLL128LoadRegs<int8_t>(
      regs, srcBytes + s * (size_t)kDdaLL128DataBytesPerSlice, sliceElts(s), lane, flagLane);
    ddaLL128StoreWire(
      scatterSlot + s * (size_t)kDdaLL128WireWordsPerSlice + 2 * lane, regs, flag, flagLane);
  };
  // Poll slice s of that peer's payload in this rank's slot and unpack it.
  auto unpack = [&](size_t s) {
    uint64_t vr[kDdaLL128WordsPerThread];
    ddaLL128PollWire(gatherSlot + s * (size_t)kDdaLL128WireWordsPerSlice + 2 * lane, vr, flag, lane);
    ddaLL128StoreRegs<int8_t>(
      dstBytes + s * (size_t)kDdaLL128DataBytesPerSlice, vr, sliceElts(s), lane, flagLane);
  };

  // Push this warp's i-th slice and unpack its (i - kDdaLL128AGUnpackLag)-th in the
  // same pass, so the unpack overlaps the pushes still in flight instead of trailing
  // all of them. Every warp pushes its slices in order and only waits on one it is
  // already past, so the waits always resolve.
  const size_t mine = gwarp < slicesTotal ? (slicesTotal - gwarp + wstride - 1) / wstride : 0;
  for (size_t i = 0; i < mine + kDdaLL128AGUnpackLag; ++i) {
    if (i < mine) push(gwarp + i * wstride);
    if (i >= kDdaLL128AGUnpackLag) unpack(gwarp + (i - kDdaLL128AGUnpackLag) * wstride);
  }

  // Local copy sendbuff -> recvbuff[selfRank].
  {
    v4u_gptr s4 = (v4u_gptr)sendbuff;
    v4u_gptr d4 = (v4u_gptr)(reinterpret_cast<char*>(recvbuff) + (size_t)selfRank * perRankBytes);
    const size_t nVec = perRankBytes >> 4; // 16B chunks
    const size_t gtid = (size_t)flatBlockId * (size_t)nthreads + (size_t)tid;
    const size_t stride = (size_t)total * (size_t)nthreads;
    for (size_t i = gtid; i < nVec; i += stride) {
      d4[i] = s4[i];
    }
  }
#if (defined(__gfx1250__) || defined(__gfx1250_strict__))
  asm volatile("s_wait_storecnt 0x0" ::: "memory");
#endif
  __syncthreads();
  ddaSetLLEpoch(epochDev, epochLen, flatBlockId, total, flag32);
}

} // namespace dda::common
