/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * LL128 two-shot (scatter-allgather) broadcast kernel for the DDA fabric
 * path (gfx1250). The root scatters the message into nRanks chunks; all
 * ranks then run an allgather of the chunks. The allgather phase is
 * symmetric, so it restores the completion invariant without start tokens.
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
#include "device/uniform.h"

namespace dda::common {

constexpr size_t ddaLL128BcTwoShotChunkSlices(size_t totalSlices, int nRanks) {
  // chunk into nPeers = nRanks-1 so root has no separate chunk.
  const int nPeers = nRanks - 1;
  size_t cs = ddaCeilDiv(totalSlices, (size_t)nPeers);
  return cs < 1 ? 1 : cs;
}

constexpr size_t ddaLL128BcTwoShotSlotSlices(size_t bankSize, int nRanks, size_t totalSlices) {
  const size_t cs = ddaLL128BcTwoShotChunkSlices(totalSlices, nRanks);
  const size_t totalSlotsNeeded = (size_t)(nRanks + 1);
  const size_t bankSlices = bankSize / (size_t)kDdaLL128WireBytesPerSlice;
  return (bankSlices / totalSlotsNeeded >= cs) ? cs : 0;
}

// Root phase 1: load chunk[col] from sendbuff, store wire to peer's scatter slot,
// and fuse recvbuff write for the in-place case.
__device__ __forceinline__ void ddaBcLL128RootScatter(
    int8_t* const* __restrict__ peerScratch,
    int8_t* __restrict__ recvbuff,
    const int8_t* __restrict__ sendbuff,
    size_t totalBytes,
    size_t chunkSlices,
    int col,
    int peer,
    uint64_t bankWords,
    size_t gwarp,
    size_t wstride,
    int lane,
    bool flagLane,
    uint64_t flag) {
  uint64_t* scatterDst = reinterpret_cast<uint64_t*>(peerScratch[peer]) + bankWords;
  for (size_t s = gwarp; s < chunkSlices; s += wstride) {
    const size_t wireOff = s * (size_t)kDdaLL128WireWordsPerSlice + 2 * lane;
    const size_t dOff = (size_t)col * chunkSlices * (size_t)kDdaLL128DataBytesPerSlice
                       + s * (size_t)kDdaLL128DataBytesPerSlice;
    const size_t rem = totalBytes > dOff ? totalBytes - dOff : 0;
    const int elt = rem < (size_t)kDdaLL128DataBytesPerSlice ? (int)rem : kDdaLL128DataBytesPerSlice;
    uint64_t regs[kDdaLL128WordsPerThread];
    if (elt > 0)
      ddaLL128LoadRegsBatched<int8_t>(regs, sendbuff + dOff, elt, lane, flagLane);
    else
      for (int u = 0; u < kDdaLL128WordsPerThread; ++u) regs[u] = 0;
    ddaLL128StoreWire(scatterDst + wireOff, regs, flag, flagLane);
    if (elt > 0 && sendbuff != reinterpret_cast<const int8_t*>(recvbuff))
      ddaLL128StoreRegs<int8_t>(recvbuff + dOff, regs, elt, lane, flagLane);
  }
}

// Root phase 2: poll the 1-slice completion token that each non-root forwards,
// then clear the gather slot. Only warp 0 participates.
__device__ __forceinline__ void ddaBcLL128RootWait(
    int8_t* const* __restrict__ peerScratch,
    int selfRank,
    int peer,
    uint64_t gatherOff,
    size_t slotWireWords,
    size_t gwarp,
    int lane,
    uint64_t flag,
    DdaKernelParams params) {
  const uint64_t* myGather = reinterpret_cast<const uint64_t*>(peerScratch[selfRank])
                             + gatherOff + (uint64_t)peer * slotWireWords;
  if (gwarp == 0) {
    uint64_t vr[kDdaLL128WordsPerThread];
#if RCCL_HAVE_COOPERATIVE_ATOMIC_BUILTINS
    if (ddaLL128PollWireBatched(myGather + 2 * lane, vr, flag, lane, params))
#else
    if (ddaLL128PollWire(myGather + 2 * lane, vr, flag, lane, params))
#endif
      ddaLL128ClearWire(const_cast<uint64_t*>(myGather) + 2 * lane, lane);
  }
}

// Non-root phase 1: poll own scatter slot, forward slices to gather slots on all
// target peers (fan-out limited), and fuse recvbuff write for the own chunk.
__device__ __forceinline__ void ddaBcLL128NonRootForward(
    int8_t* const* __restrict__ peerScratch,
    int8_t* __restrict__ recvbuff,
    size_t totalBytes,
    size_t chunkSlices,
    int selfRank,
    int col,
    int root,
    int nPeers,
    uint64_t bankWords,
    uint64_t gatherOff,
    size_t slotWireWords,
    int kFanOut,
    int nFanRounds,
    int nwarps,
    int warp,
    int lane,
    bool flagLane,
    uint64_t flag,
    DdaKernelParams params) {
  const uint64_t* myScatter = reinterpret_cast<const uint64_t*>(peerScratch[selfRank]) + bankWords;
  const int selfCol = selfRank < root ? selfRank : selfRank - 1;
  bool aborted = false;
  for (int fRound = 0; fRound < nFanRounds && !aborted; ++fRound) {
    const int kActive = kFanOut < (nPeers - fRound * kFanOut)
                      ? kFanOut : (nPeers - fRound * kFanOut);
    if (kActive <= 0) break;
    const int peerGroup = col % kActive;
    const int tgtCol = fRound * kFanOut + peerGroup;
    if (tgtCol >= nPeers) continue;
    const int tgtPeer = tgtCol < selfRank ? tgtCol : tgtCol + 1;
    const int localCol = col / kActive;
    const int myColCount = (int)ddaCeilDiv((size_t)(nPeers - peerGroup), (size_t)kActive);
    if (localCol >= myColCount) continue;
    const size_t gwarp_r = ((size_t)localCol * (size_t)gridDim.y + (size_t)blockIdx.y)
                           * (size_t)nwarps + (size_t)warp;
    const size_t wstride_r = (size_t)gridDim.y * (size_t)myColCount * (size_t)nwarps;
    uint64_t* tgtGather = reinterpret_cast<uint64_t*>(peerScratch[tgtPeer]) + gatherOff
                        + (uint64_t)selfRank * slotWireWords;
    for (size_t s = gwarp_r; s < chunkSlices; s += wstride_r) {
      uint64_t vr[kDdaLL128WordsPerThread];
#if RCCL_HAVE_COOPERATIVE_ATOMIC_BUILTINS
      if (!ddaLL128PollWireBatched(myScatter + s * (size_t)kDdaLL128WireWordsPerSlice + 2 * lane,
                            vr, flag, lane, params)) {
#else
      if (!ddaLL128PollWire(myScatter + s * (size_t)kDdaLL128WireWordsPerSlice + 2 * lane,
                            vr, flag, lane, params)) {
#endif
        aborted = true;
        break;
      }
      // for the root peer, only forward 1 padding slice (s==0).
      if (tgtPeer != root || s == 0) {
        ddaLL128StoreWire(tgtGather + s * (size_t)kDdaLL128WireWordsPerSlice + 2 * lane,
                          vr, flag, flagLane);
      }
      if (nFanRounds > 1 ? (fRound == 0) : (col == (int)(s % (size_t)nPeers))) {
        const size_t dOff = (size_t)selfCol * chunkSlices * (size_t)kDdaLL128DataBytesPerSlice
                           + s * (size_t)kDdaLL128DataBytesPerSlice;
        const size_t rem = totalBytes > dOff ? totalBytes - dOff : 0;
        const int elt = rem < (size_t)kDdaLL128DataBytesPerSlice ? (int)rem : kDdaLL128DataBytesPerSlice;
        if (elt > 0)
          ddaLL128StoreRegs<int8_t>(recvbuff + dOff, vr, elt, lane, flagLane);
      }
    }
  }
}

// Non-root phase 2: poll own gather slot for peer's chunk, unpack into recvbuff,
// and clear each wire slice after consumption.
// Caller must guard: only called when peer != root.
__device__ __forceinline__ void ddaBcLL128NonRootGather(
    int8_t* const* __restrict__ peerScratch,
    int8_t* __restrict__ recvbuff,
    size_t totalBytes,
    size_t chunkSlices,
    int selfRank,
    int peer,
    int root,
    uint64_t gatherOff,
    size_t slotWireWords,
    size_t gwarp,
    size_t wstride,
    int lane,
    bool flagLane,
    uint64_t flag,
    DdaKernelParams params) {
  const uint64_t* myGather = reinterpret_cast<const uint64_t*>(peerScratch[selfRank])
                             + gatherOff + (uint64_t)peer * slotWireWords;
  for (size_t s = gwarp; s < chunkSlices; s += wstride) {
    const int peerCol = peer < root ? peer : peer - 1;
    const size_t dOff = (size_t)peerCol * chunkSlices * (size_t)kDdaLL128DataBytesPerSlice
                       + s * (size_t)kDdaLL128DataBytesPerSlice;
    const size_t rem = totalBytes > dOff ? totalBytes - dOff : 0;
    const int elt = rem < (size_t)kDdaLL128DataBytesPerSlice ? (int)rem : kDdaLL128DataBytesPerSlice;
    uint64_t vr[kDdaLL128WordsPerThread];
#if RCCL_HAVE_COOPERATIVE_ATOMIC_BUILTINS
    if (!ddaLL128PollWireBatched(myGather + s * (size_t)kDdaLL128WireWordsPerSlice + 2 * lane,
                          vr, flag, lane, params))
#else
    if (!ddaLL128PollWire(myGather + s * (size_t)kDdaLL128WireWordsPerSlice + 2 * lane,
                          vr, flag, lane, params))
#endif
      break;
    if (elt > 0)
      ddaLL128StoreRegs<int8_t>(recvbuff + dOff, vr, elt, lane, flagLane);
    ddaLL128ClearWire(const_cast<uint64_t*>(myGather) +
      s * (size_t)kDdaLL128WireWordsPerSlice + 2 * lane, lane);
  }
}

template <int NRANKS_CT>
__launch_bounds__(512)
__global__ void ddaBroadcastFabricLL128TwoShot(
    int8_t* const* __restrict__ peerScratch,
    int8_t* __restrict__ recvbuff,
    const int8_t* __restrict__ sendbuff,
    size_t totalBytes,
    int selfRank, int nRanksRt, int root,
    uint32_t* __restrict__ epochDev,
    int epochLen,
    size_t totalSlices,
    size_t chunkSlices,
    size_t bankSize,
    DdaKernelParams params,
    int fanOut) {

  const int nRanks = NRANKS_CT ? NRANKS_CT : nRanksRt;
  const int nPeers = nRanks - 1;
  const bool isRoot = (selfRank == root);

  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  const int lane = tid % kDdaLL128Warp;
  const int warp = tid / kDdaLL128Warp;
  const int nwarps = nthreads / kDdaLL128Warp;
  const bool flagLane = ddaLL128IsFlagLane(lane);

  const int flatBlockId = (int)(blockIdx.x * gridDim.y + blockIdx.y);
  const int totalBlk = (int)(gridDim.x * gridDim.y);
  const uint32_t flag32 = ddaGetLLEpochInc(epochDev, flatBlockId, 1);
  const uint64_t flag = ((uint64_t)flag32 << 32) | (uint64_t)flag32;

  const uint64_t bankWords = wave::uniform((uint64_t)(flag32 & 1u) * (uint64_t)(bankSize / sizeof(uint64_t)));
  const size_t slotWireWords = wave::uniform(chunkSlices * (size_t)kDdaLL128WireWordsPerSlice);
  const uint64_t gatherOff = wave::uniform(bankWords + slotWireWords);

  const size_t gwarp = (size_t)blockIdx.y * (size_t)nwarps + (size_t)warp;
  const size_t wstride = (size_t)gridDim.y * (size_t)nwarps;

  const int col = (int)blockIdx.x;
  const int peer = col < selfRank ? col : col + 1;

  // Fan-out: 0 or >= nPeers means full fan-out
  const int kFanOut = (fanOut <= 0 || fanOut >= nPeers) ? nPeers : fanOut;
  const int nFanRounds = (int)ddaCeilDiv((size_t)nPeers, (size_t)kFanOut);

  if (isRoot) {
    ddaBcLL128RootScatter(peerScratch, recvbuff, sendbuff, totalBytes, chunkSlices,
                          col, peer, bankWords, gwarp, wstride, lane, flagLane, flag);
    ddaBcLL128RootWait(peerScratch, selfRank, peer, gatherOff, slotWireWords,
                       gwarp, lane, flag, params);
  } else {
    ddaBcLL128NonRootForward(peerScratch, recvbuff, totalBytes, chunkSlices,
                             selfRank, col, root, nPeers,
                             bankWords, gatherOff, slotWireWords,
                             kFanOut, nFanRounds, nwarps, warp,
                             lane, flagLane, flag, params);
    if (peer != root)
      ddaBcLL128NonRootGather(peerScratch, recvbuff, totalBytes, chunkSlices,
                              selfRank, peer, root, gatherOff, slotWireWords,
                              gwarp, wstride, lane, flagLane, flag, params);
  }

#if (defined(__gfx1250__) || defined(__gfx1250_strict__))
  asm volatile("s_wait_storecnt 0x0" ::: "memory");
#endif
  __syncthreads();
  ddaSetLLEpoch(epochDev, epochLen, flatBlockId, totalBlk, flag32);
}

} // namespace dda::common
