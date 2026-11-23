/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Warp-specialised LL128 two-shot broadcast kernel for the DDA fabric
 * path (gfx1250).  Based on ddaBroadcastFabricLL128TwoShot but allows
 * multiple peers to share a single block by partitioning the block's
 * warps across peers.
 *
 * Parameter kWarpsPerPeer controls how many warps each peer gets within
 * a block.  With nwarps=8 and kWarpsPerPeer=2, each block services 4
 * peers simultaneously.  grid.x = ceil(nPeers / peersPerBlock).
 *
 * kWarpsPerPeer=nwarps (e.g. 8) recovers the base kernel's 1-peer-per-
 * block geometry.  kWarpsPerPeer=1 packs all peers into one block.
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

template <int NRANKS_CT>
__launch_bounds__(512)
__global__ void ddaBroadcastFabricLL128TwoShotWarpSpec(
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
    int fanOut,
    int kWarpsPerPeer) {

  const int nRanks = NRANKS_CT ? NRANKS_CT : nRanksRt;
  const int nPeers = nRanks - 1;
  const bool isRoot = (selfRank == root);

  const int tid      = threadIdx.x;
  const int nthreads = blockDim.x;
  const int lane     = tid % kDdaLL128Warp;
  const int warp     = tid / kDdaLL128Warp;
  const int nwarps   = nthreads / kDdaLL128Warp;
  const bool flagLane = ddaLL128IsFlagLane(lane);

  // --- Warp-group computation ---
  const int peersPerBlock = nwarps / kWarpsPerPeer;

  // How many peers does THIS block actually service?
  const int globalPeerStart = (int)blockIdx.x * peersPerBlock;
  int peersHere = nPeers - globalPeerStart;
  if (peersHere > peersPerBlock) peersHere = peersPerBlock;
  if (peersHere < 0) peersHere = 0;

  // Assign warp -> peer-slot.
  // First peersHere * kWarpsPerPeer warps form full groups.
  // Remaining warps are redistributed round-robin.
  int myPeerSlot;
  int localWarp;
  int k_eff;

  const int assignedWarps = peersHere * kWarpsPerPeer;

  if (warp < assignedWarps) {
    myPeerSlot = warp / kWarpsPerPeer;
    localWarp  = warp % kWarpsPerPeer;
    // Count extra warps this slot receives from redistribution.
    const int remWarps = nwarps - assignedWarps;
    k_eff = kWarpsPerPeer + (myPeerSlot < remWarps ? 1 : 0);
  } else if (peersHere > 0) {
    // Remainder warp: distribute round-robin across active peer-slots.
    const int remIdx = warp - assignedWarps;
    myPeerSlot = remIdx % peersHere;
    // Compute k_eff for this slot.
    const int remWarps = nwarps - assignedWarps;
    k_eff = kWarpsPerPeer + (myPeerSlot < remWarps ? 1 : 0);
    // Position of this warp within its adopted group: after the base k warps,
    // count how many previous remainder warps went to the same slot.
    int prevRem = 0;
    for (int r = 0; r < remIdx; ++r) {
      if ((r % peersHere) == myPeerSlot) ++prevRem;
    }
    localWarp = kWarpsPerPeer + prevRem;
  } else {
    myPeerSlot = -1;
    localWarp  = 0;
    k_eff      = 1;
  }

  const bool active = (myPeerSlot >= 0 && myPeerSlot < peersHere);

  // Global peer index -> rank mapping (skip self).
  const int peerIdx = active ? (globalPeerStart + myPeerSlot) : -1;
  const int col     = peerIdx;
  const int peer    = active ? (peerIdx < selfRank ? peerIdx : peerIdx + 1) : -1;

  // --- Epoch ---
  const int flatBlockId = (int)(blockIdx.x * gridDim.y + blockIdx.y);
  const int totalBlk    = (int)(gridDim.x * gridDim.y);
  const uint32_t flag32 = ddaGetLLEpochInc(epochDev, flatBlockId, 1);
  const uint64_t flag   = ((uint64_t)flag32 << 32) | (uint64_t)flag32;

  // --- Scratch layout (identical to base two-shot) ---
  const uint64_t bankWords   = wave::uniform((uint64_t)(flag32 & 1u) * (uint64_t)(bankSize / sizeof(uint64_t)));
  const size_t slotWireWords = wave::uniform(chunkSlices * (size_t)kDdaLL128WireWordsPerSlice);
  const uint64_t gatherOff   = wave::uniform(bankWords + slotWireWords);

  // --- Slice iteration stride (per warp-group) ---
  const size_t gwarp   = (size_t)blockIdx.y * (size_t)k_eff + (size_t)localWarp;
  const size_t wstride = (size_t)gridDim.y * (size_t)k_eff;

  // Fan-out parameters.
  const int kFanOut    = (fanOut <= 0 || fanOut >= nPeers) ? nPeers : fanOut;
  const int nFanRounds = (int)ddaCeilDiv((size_t)nPeers, (size_t)kFanOut);

  // =====================================================================
  // ROOT
  // =====================================================================
  if (isRoot && active) {
    // --- Scatter: load chunk[col] from sendbuff, push to peer scatter slot ---
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

    // --- Wait: poll completion token (localWarp==0, blockIdx.y==0 only) ---
    if (localWarp == 0 && blockIdx.y == 0) {
      const uint64_t* myGather = reinterpret_cast<const uint64_t*>(peerScratch[selfRank])
                                 + gatherOff + (uint64_t)peer * slotWireWords;
      uint64_t vr[kDdaLL128WordsPerThread];
#if RCCL_HAVE_COOPERATIVE_ATOMIC_BUILTINS
      if (ddaLL128PollWireBatched(myGather + 2 * lane, vr, flag, lane, params))
#else
      if (ddaLL128PollWire(myGather + 2 * lane, vr, flag, lane, params))
#endif
        ddaLL128ClearWire(const_cast<uint64_t*>(myGather) + 2 * lane, lane);
    }
  }

  // =====================================================================
  // NON-ROOT
  // =====================================================================
  else if (!isRoot && active) {
    const int selfCol = selfRank < root ? selfRank : selfRank - 1;

    // --- Phase 1: forward (scatter receive + fan-out) ---
    const uint64_t* myScatter = reinterpret_cast<const uint64_t*>(peerScratch[selfRank]) + bankWords;

    bool aborted = false;
    for (int fRound = 0; fRound < nFanRounds && !aborted; ++fRound) {
      const int kActive = kFanOut < (nPeers - fRound * kFanOut)
                        ? kFanOut : (nPeers - fRound * kFanOut);
      if (kActive <= 0) break;
      const int peerGroup = col % kActive;
      const int tgtCol    = fRound * kFanOut + peerGroup;
      if (tgtCol >= nPeers) continue;
      const int tgtPeer = tgtCol < selfRank ? tgtCol : tgtCol + 1;
      const int localCol   = col / kActive;
      const int myColCount = (int)ddaCeilDiv((size_t)(nPeers - peerGroup), (size_t)kActive);
      if (localCol >= myColCount) continue;

      const size_t gwarp_r  = ((size_t)localCol * (size_t)gridDim.y + (size_t)blockIdx.y)
                              * (size_t)k_eff + (size_t)localWarp;
      const size_t wstride_r = (size_t)gridDim.y * (size_t)myColCount * (size_t)k_eff;

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

    // --- Phase 2: gather (poll other peers' chunks, unpack to recvbuff) ---
    if (peer != root) {
      const int peerCol = peer < root ? peer : peer - 1;
      const uint64_t* myGather = reinterpret_cast<const uint64_t*>(peerScratch[selfRank])
                                 + gatherOff + (uint64_t)peer * slotWireWords;
      for (size_t s = gwarp; s < chunkSlices; s += wstride) {
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
  }

#if (defined(__gfx1250__) || defined(__gfx1250_strict__))
  asm volatile("s_wait_storecnt 0x0" ::: "memory");
#endif
  __syncthreads();
  ddaSetLLEpoch(epochDev, epochLen, flatBlockId, totalBlk, flag32);
}

} // namespace dda::common
