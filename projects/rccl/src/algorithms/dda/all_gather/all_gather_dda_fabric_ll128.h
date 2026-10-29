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

// Rank of the peer peerOff (0 .. nRanks - 2) places after selfRank.
__device__ __forceinline__ int ddaLL128AgPeer(int selfRank, int peerOff, int nRanks) {
  const int peer = selfRank + peerOff + 1;
  return peer < nRanks ? peer : peer - nRanks;
}

// Runs body(s, peerOff) on items first, first + step, ... below end, stopping early if
// body returns false. Items are the (slice, peer) pairs of a group in peer-major
// order, item = (peerOff - group0) * slicesTotal + s; the slice and the peer step
// without a division per item.
template <typename Body>
__device__ __forceinline__ void ddaLL128AgForEachItem(size_t slicesTotal, int group0, size_t first, size_t end,
                                                      size_t step, Body&& body) {
  size_t s = first % slicesTotal;
  int j = (int)(first / slicesTotal);
  const size_t ds = step % slicesTotal;
  const int dj = (int)(step / slicesTotal);
  for (size_t it = first; it < end; it += step) {
    if (!body(s, group0 + j)) return;
    s += ds;
    j += dj;
    if (s >= slicesTotal) {
      s -= slicesTotal;
      ++j;
    }
  }
}

// LL128 all-gather. 2D grid: grid.x places one column per group of up to
// peersPerBlock consecutive peers. Each block of a column takes an equal contiguous
// range of the group's (slice, peer) items in peer-major order, so a block works for
// one peer at a time while the column's blocks spread over all of the group's peers;
// a warp takes one item at a time. With peersPerBlock == 1 a column serves a single
// peer. Each item packs a slice of this rank's payload into that peer's slot; the
// same items then poll this rank's slot for the peer's payload and unpack it.
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
                                          int peersPerBlock, // peers one column serves
                                          DdaKernelParams params) {

  const int nPeers = nRanks - 1;
  const int group0 = (int)blockIdx.x * peersPerBlock;
  const int groupPeers = peersPerBlock < nPeers - group0 ? peersPerBlock : nPeers - group0;

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

  // This block's range of the group's items; its warps take every nwarps-th item.
  const size_t items = slicesTotal * (size_t)groupPeers;
  const size_t perBlock = (items + gridDim.y - 1) / gridDim.y;
  const size_t rangeEnd = ((size_t)blockIdx.y + 1) * perBlock;
  const size_t itemEnd = rangeEnd < items ? rangeEnd : items;
  const size_t itemFirst = (size_t)blockIdx.y * perBlock + (size_t)warp;

  const int8_t* srcBytes = reinterpret_cast<const int8_t*>(sendbuff);
  const uint64_t selfSlot = bankWords + (uint64_t)selfRank * (uint64_t)slotWords;
  const uint64_t* gatherBase = reinterpret_cast<const uint64_t*>(peerScratch[selfRank]) + bankWords;

  // Phase 1: pack and push the group's slices to its peers.
  ddaLL128AgForEachItem(slicesTotal, group0, itemFirst, itemEnd, nwarps, [&](size_t s, int peerOff) {
    const int peer = ddaLL128AgPeer(selfRank, peerOff, nRanks);
    uint64_t* scatterSlot = reinterpret_cast<uint64_t*>(peerScratch[peer]) + selfSlot;
    ddaLL128AgSendSlices<1>(srcBytes, scatterSlot, perRankBytes, s, 0, flag, lane, flagLane);
    return true;
  });

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

  // Phase 2: poll the same items in this rank's slots and unpack. An abort ends the
  // loop, not the kernel: every thread must reach the __syncthreads below.
  ddaLL128AgForEachItem(slicesTotal, group0, itemFirst, itemEnd, nwarps, [&](size_t s, int peerOff) {
    const int peer = ddaLL128AgPeer(selfRank, peerOff, nRanks);
    const uint64_t* gatherSlot = gatherBase + (uint64_t)peer * (uint64_t)slotWords;
    int8_t* dstBytes = reinterpret_cast<int8_t*>(recvbuff) + (size_t)peer * perRankBytes;
    return ddaLL128AgRecvSlices<1>(gatherSlot, dstBytes, perRankBytes, s, 0, flag, lane, flagLane, params);
  });
#if (defined(__gfx1250__) || defined(__gfx1250_strict__))
  asm volatile("s_wait_storecnt 0x0" ::: "memory");
#endif
  __syncthreads();
  ddaSetLLEpoch(epochDev, epochLen, flatBlockId, total, flag32);
}

} // namespace dda::common
