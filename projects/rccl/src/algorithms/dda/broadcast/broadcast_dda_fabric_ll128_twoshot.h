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

namespace dda::common {

constexpr size_t ddaLL128BcTwoShotChunkSlices(size_t totalSlices, int nRanks) {
  // O8: chunk into nPeers = nRanks-1 so root has no separate chunk.
  const int nPeers = nRanks - 1;
  size_t cs = (totalSlices + (size_t)nPeers - 1) / (size_t)nPeers;
  return cs < 1 ? 1 : cs;
}

constexpr size_t ddaLL128BcTwoShotSlotSlices(size_t bankSize, int nRanks, size_t totalSlices) {
  const size_t cs = ddaLL128BcTwoShotChunkSlices(totalSlices, nRanks);
  const size_t totalSlotsNeeded = (size_t)(nRanks + 1);
  const size_t bankSlices = bankSize / (size_t)kDdaLL128WireBytesPerSlice;
  return (bankSlices / totalSlotsNeeded >= cs) ? cs : 0;
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

  const uint64_t bankWords = (uint64_t)(flag32 & 1u) * (uint64_t)(bankSize / sizeof(uint64_t));
  const size_t slotWireWords = chunkSlices * (size_t)kDdaLL128WireWordsPerSlice;
  const uint64_t gatherOff = bankWords + slotWireWords;

  const size_t gwarp = (size_t)blockIdx.y * (size_t)nwarps + (size_t)warp;
  const size_t wstride = (size_t)gridDim.y * (size_t)nwarps;

  const int col = (int)blockIdx.x;
  const int peer = col < selfRank ? col : col + 1;

  auto chunkOff = [&](int r) -> size_t {
    return (size_t)r * chunkSlices * (size_t)kDdaLL128DataBytesPerSlice;
  };

  // Fan-out: 0 or >= nPeers means full fan-out
  const int kFanOut = (fanOut <= 0 || fanOut >= nPeers) ? nPeers : fanOut;
  const int nFanRounds = (nPeers + kFanOut - 1) / kFanOut;

  if (isRoot) {
    // scatter chunk[col] to peer (nPeers chunks, no root chunk).
    // no gather push needed.
    // Recvbuff write fused: each column writes chunk[col] to recvbuff.
    uint64_t* scatterDst = reinterpret_cast<uint64_t*>(peerScratch[peer]) + bankWords;
    for (size_t s = gwarp; s < chunkSlices; s += wstride) {
      const size_t wireOff = s * (size_t)kDdaLL128WireWordsPerSlice + 2 * lane;
      const size_t dOff = chunkOff(col) + s * (size_t)kDdaLL128DataBytesPerSlice;
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



    // wait for every peer's allgather contribution.
    // non-roots forward only 1 padding slice to the root (see Phase 1).
    // The root uses its own sendbuff, so only 1 slice per peer is needed here.
    const uint64_t* myScratch = reinterpret_cast<const uint64_t*>(peerScratch[selfRank]);
    const uint64_t* myGather = myScratch + gatherOff + (uint64_t)peer * slotWireWords;
    if (gwarp == 0) {
      uint64_t vr[kDdaLL128WordsPerThread];
      if (ddaLL128PollWireBatched(myGather + 2 * lane, vr, flag, lane, params))
        ddaLL128ClearWire(const_cast<uint64_t*>(myGather) + 2 * lane, lane);
    }
  } else {
    // --- NON-ROOT ---

    // Phase 1: poll scatter slot, forward to gather slots -- fan-out limited.
    // recvbuff write happens in round 0 only.
    const uint64_t* myScatter = reinterpret_cast<const uint64_t*>(peerScratch[selfRank]) + bankWords;
    // chunk index for this non-root (0..nPeers-1, skipping root)
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
      const int myColCount = (nPeers - peerGroup + kActive - 1) / kActive;
      if (localCol >= myColCount) continue;
      const size_t gwarp_r = ((size_t)localCol * (size_t)gridDim.y + (size_t)blockIdx.y)
                             * (size_t)nwarps + (size_t)warp;
      const size_t wstride_r = (size_t)gridDim.y * (size_t)myColCount * (size_t)nwarps;
      uint64_t* tgtGather = reinterpret_cast<uint64_t*>(peerScratch[tgtPeer]) + gatherOff +
        (uint64_t)selfRank * slotWireWords;
      for (size_t s = gwarp_r; s < chunkSlices; s += wstride_r) {
        uint64_t vr[kDdaLL128WordsPerThread];
        if (!ddaLL128PollWireBatched(myScatter + s * (size_t)kDdaLL128WireWordsPerSlice + 2 * lane,
                              vr, flag, lane, params)) {
          aborted = true;
          break;
        }
        // for the root peer, only forward 1 padding slice (s==0).
        if (tgtPeer != root || s == 0) {
          ddaLL128StoreWire(tgtGather + s * (size_t)kDdaLL128WireWordsPerSlice + 2 * lane,
                            vr, flag, flagLane);
        }
        if (nFanRounds > 1 ? (fRound == 0) : (col == (int)(s % (size_t)nPeers))) {
          const size_t dOff = chunkOff(selfCol) + s * (size_t)kDdaLL128DataBytesPerSlice;
          const size_t rem = totalBytes > dOff ? totalBytes - dOff : 0;
          const int elt = rem < (size_t)kDdaLL128DataBytesPerSlice ? (int)rem : kDdaLL128DataBytesPerSlice;
          if (elt > 0)
            ddaLL128StoreRegs<int8_t>(recvbuff + dOff, vr, elt, lane, flagLane);
        }
      }
    }

    // Phase 2: gather -- poll own gather slot[peer], unpack into recvbuff.
    // skip for root column (root has no chunk in the allgather).
    if (peer != root) {
    const uint64_t* myGather = reinterpret_cast<const uint64_t*>(peerScratch[selfRank]) +
      gatherOff + (uint64_t)peer * slotWireWords;
    for (size_t s = gwarp; s < chunkSlices; s += wstride) {
      const int peerCol = peer < root ? peer : peer - 1; // O8: rank to chunk index
      const size_t dOff = chunkOff(peerCol) + s * (size_t)kDdaLL128DataBytesPerSlice;
      const size_t rem = totalBytes > dOff ? totalBytes - dOff : 0;
      const int elt = rem < (size_t)kDdaLL128DataBytesPerSlice ? (int)rem : kDdaLL128DataBytesPerSlice;
      uint64_t vr[kDdaLL128WordsPerThread];
      if (!ddaLL128PollWireBatched(myGather + s * (size_t)kDdaLL128WireWordsPerSlice + 2 * lane,
                            vr, flag, lane, params))
        break;
      if (elt > 0)
        ddaLL128StoreRegs<int8_t>(recvbuff + dOff, vr, elt, lane, flagLane);
      ddaLL128ClearWire(const_cast<uint64_t*>(myGather) +
        s * (size_t)kDdaLL128WireWordsPerSlice + 2 * lane, lane);
    }
    } // if (peer != root) -- O8 guard
  }

#if (defined(__gfx1250__) || defined(__gfx1250_strict__))
  asm volatile("s_wait_storecnt 0x0" ::: "memory");
#endif
  __syncthreads();
  ddaSetLLEpoch(epochDev, epochLen, flatBlockId, totalBlk, flag32);
}

} // namespace dda::common
