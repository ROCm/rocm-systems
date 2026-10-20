/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * LL128-protocol all-gather device kernel for the DDA fabric path (gfx1250).
 * A slice is 2 KiB, eight uint64 registers per lane; a warp moves up to
 * kDdaLL128AgUnroll of them at a time. No GPU barrier; staging uses
 * comm->ddaScratch reached via comm->ddaPeerPtrsDev.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

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

// Slices a warp keeps in flight: each pass packs or unpacks this many of its slices
// together, so their memory round trips overlap.
constexpr int kDdaLL128AgUnroll = 4;

// Runs body(std::integral_constant<int, L>{}) for the warp's pass that starts at
// slice s0, where L is how many slices the pass holds: kDdaLL128AgUnroll, or fewer
// on the warp's last pass. L is a compile-time constant so the pass body is
// straight-line.
template <typename Body>
__device__ __forceinline__ void ddaLL128AgPass(size_t s0, size_t wstride, size_t slicesTotal, Body&& body) {
  static_assert(kDdaLL128AgUnroll >= 1 && kDdaLL128AgUnroll <= 4, "the switch below covers up to 4 slices");
  int slices = 1;
#pragma unroll
  for (int u = 1; u < kDdaLL128AgUnroll; ++u) {
    if (s0 + (size_t)u * wstride < slicesTotal) slices = u + 1;
  }
  switch (slices) {
  case 4:
    body(std::integral_constant<int, 4>{});
    break;
  case 3:
    body(std::integral_constant<int, 3>{});
    break;
  case 2:
    body(std::integral_constant<int, 2>{});
    break;
  default:
    body(std::integral_constant<int, 1>{});
    break;
  }
}

// Packs this warp's L slices s0 + u * wstride (u < L) and pushes them to the
// peer's slot. Kept free of branches between the loads and the stores: any branch
// there makes each slice's loads land before the next slice's are issued.
template <int L>
__device__ __forceinline__ void ddaLL128AgSendSlices(const int8_t* srcBytes, uint64_t* scatterSlot,
                                                     size_t perRankBytes, size_t s0, size_t wstride,
                                                     uint64_t flag, int lane, bool flagLane) {
  uint64_t regs[L][kDdaLL128WordsPerThread];
#pragma unroll
  for (int u = 0; u < L; ++u) {
    const size_t dataByte = (s0 + (size_t)u * wstride) * (size_t)kDdaLL128DataBytesPerSlice;
    const size_t rem = perRankBytes - dataByte;
    const int eltInSlice = rem < (size_t)kDdaLL128DataBytesPerSlice ? (int)rem : kDdaLL128DataBytesPerSlice;
    ddaLL128LoadRegsBegin<int8_t>(regs[u], srcBytes + dataByte, eltInSlice, lane, flagLane);
  }
#if defined(__HIP_DEVICE_COMPILE__) && __has_builtin(__builtin_amdgcn_sched_barrier)
  // Without it the scheduler moves the flag-lane shuffle up between the loads, and
  // the loads after the shuffle wait for the ones before it.
  __builtin_amdgcn_sched_barrier(0);
#endif
#pragma unroll
  for (int u = 0; u < L; ++u) {
    ddaLL128LoadRegsFinish(regs[u], flagLane);
    ddaLL128StoreWire(scatterSlot + (s0 + (size_t)u * wstride) * (size_t)kDdaLL128WireWordsPerSlice + 2 * lane,
                      regs[u], flag, flagLane);
  }
}

// Polls this warp's L slices s0 + u * wstride (u < L) in the peer's slot and
// unpacks them. Straight-line like ddaLL128AgSendSlices, so the polls of all L
// slices are in flight together. Returns false if the comm abort flag is set.
template <int L>
__device__ __forceinline__ bool ddaLL128AgRecvSlices(const uint64_t* gatherSlot, int8_t* dstBytes,
                                                     size_t perRankBytes, size_t s0, size_t wstride,
                                                     uint64_t flag, int lane, bool flagLane,
                                                     DdaKernelParams params) {
  const uint64_t* wire[L];
#pragma unroll
  for (int u = 0; u < L; ++u) {
    wire[u] = gatherSlot + (s0 + (size_t)u * wstride) * (size_t)kDdaLL128WireWordsPerSlice + 2 * lane;
  }
  uint64_t vr[L][kDdaLL128WordsPerThread];
  if (!ddaLL128PollWireN<L>(wire, vr, flag, lane, params)) return false;
#pragma unroll
  for (int u = 0; u < L; ++u) {
    const size_t dataByte = (s0 + (size_t)u * wstride) * (size_t)kDdaLL128DataBytesPerSlice;
    const size_t rem = perRankBytes - dataByte;
    const int eltInSlice = rem < (size_t)kDdaLL128DataBytesPerSlice ? (int)rem : kDdaLL128DataBytesPerSlice;
    ddaLL128StoreRegs<int8_t>(dstBytes + dataByte, vr[u], eltInSlice, lane, flagLane);
  }
  return true;
}

// LL128 all-gather. 2D grid: grid.x == nRanks - 1 places one column per remote
// peer; grid.y splits that peer's slices across blocks, and a warp takes
// kDdaLL128AgUnroll of its slices per pass so that many are in flight at once.
// Each column packs this rank's payload into its peer's slot, then polls its own
// slot for that peer's payload and unpacks it.
template <typename T>
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

  // Slices stride by warp within this peer's column only. A pass takes the
  // warp's next kDdaLL128AgUnroll slices, so that many are in flight at once.
  const size_t gwarp = (size_t)blockIdx.y * (size_t)nwarps + (size_t)warp;
  const size_t wstride = (size_t)gridDim.y * (size_t)nwarps;
  const size_t pstride = (size_t)kDdaLL128AgUnroll * wstride;

  const int8_t* srcBytes = reinterpret_cast<const int8_t*>(sendbuff);
  uint64_t* scatterSlot = reinterpret_cast<uint64_t*>(peerScratch[peer]) + bankWords +
    (uint64_t)selfRank * (uint64_t)slotWords;
  const uint64_t* gatherSlot = reinterpret_cast<const uint64_t*>(peerScratch[selfRank]) + bankWords +
    (uint64_t)peer * (uint64_t)slotWords;
  int8_t* dstBytes = reinterpret_cast<int8_t*>(recvbuff) + (size_t)peer * perRankBytes;

  // Phase 1: pack and push this column's slices to the one peer it owns.
  for (size_t s0 = gwarp; s0 < slicesTotal; s0 += pstride) {
    ddaLL128AgPass(s0, wstride, slicesTotal, [&](auto slices) {
      ddaLL128AgSendSlices<slices>(srcBytes, scatterSlot, perRankBytes, s0, wstride, flag, lane, flagLane);
    });
  }

  // Local copy sendbuff -> recvbuff[selfRank].
  {
    v4u_gptr s4 = (v4u_gptr)sendbuff;
    v4u_gptr d4 = (v4u_gptr)(reinterpret_cast<char*>(recvbuff) + (size_t)selfRank * perRankBytes);
    const size_t nVec = perRankBytes >> 4; // 16B chunks
    const size_t gtid = (size_t)flatBlockId * (size_t)nthreads + (size_t)tid;
    const size_t stride = (size_t)total * (size_t)nthreads;
    for (size_t i0 = gtid; i0 < nVec; i0 += (size_t)kDdaLL128AgUnroll * stride) {
      v4u v[kDdaLL128AgUnroll];
#pragma unroll
      for (int u = 0; u < kDdaLL128AgUnroll; ++u) {
        const size_t i = i0 + (size_t)u * stride;
        if (i < nVec) v[u] = s4[i];
      }
#pragma unroll
      for (int u = 0; u < kDdaLL128AgUnroll; ++u) {
        const size_t i = i0 + (size_t)u * stride;
        if (i < nVec) d4[i] = v[u];
      }
    }
  }

  // Phase 2: poll the same slices in that peer's slot and unpack.
  for (size_t s0 = gwarp; s0 < slicesTotal; s0 += pstride) {
    bool done = true;
    ddaLL128AgPass(s0, wstride, slicesTotal, [&](auto slices) {
      done = ddaLL128AgRecvSlices<slices>(gatherSlot, dstBytes, perRankBytes, s0, wstride, flag, lane, flagLane,
                                          params);
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
