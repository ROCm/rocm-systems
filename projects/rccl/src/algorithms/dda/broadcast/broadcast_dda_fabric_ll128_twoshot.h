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
  size_t cs = (totalSlices + (size_t)nRanks - 1) / (size_t)nRanks;
  return cs < 1 ? 1 : cs;
}

constexpr size_t ddaLL128BcTwoShotSlotSlices(size_t bankSize, int nRanks, size_t totalSlices) {
  const size_t cs = ddaLL128BcTwoShotChunkSlices(totalSlices, nRanks);
  const size_t totalSlotsNeeded = (size_t)(nRanks + 1);
  const size_t bankSlices = bankSize / (size_t)kDdaLL128WireBytesPerSlice;
  return (bankSlices / totalSlotsNeeded >= cs) ? cs : 0;
}

template <int NRANKS_CT>
#if defined(USE_ROCM)
__launch_bounds__(1024)
#endif
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

  // Fan-out: 0 or >= nPeers means full fan-out (original behavior).
  const int kFanOut = (fanOut <= 0 || fanOut >= nPeers) ? nPeers : fanOut;
  const int nFanRounds = (nPeers + kFanOut - 1) / kFanOut;

  if (isRoot) {
    // Fused Phase 1+2 (O4): scatter chunk[peer] to scatter slot and push
    // chunk[root] to gather slot[root] in a single pass.
    uint64_t* scatterDst = reinterpret_cast<uint64_t*>(peerScratch[peer]) + bankWords;
    uint64_t* gatherDst = reinterpret_cast<uint64_t*>(peerScratch[peer]) + gatherOff +
      (uint64_t)root * slotWireWords;
    for (size_t s = gwarp; s < chunkSlices; s += wstride) {
      const size_t wireOff = s * (size_t)kDdaLL128WireWordsPerSlice + 2 * lane;

      // Scatter: chunk[peer] -> scatter slot
      {
        const size_t dOff = chunkOff(peer) + s * (size_t)kDdaLL128DataBytesPerSlice;
        const size_t rem = totalBytes > dOff ? totalBytes - dOff : 0;
        const int elt = rem < (size_t)kDdaLL128DataBytesPerSlice ? (int)rem : kDdaLL128DataBytesPerSlice;
        uint64_t regs[kDdaLL128WordsPerThread];
        if (elt > 0)
          ddaLL128LoadRegs<int8_t>(regs, sendbuff + dOff, elt, lane, flagLane);
        else
          for (int u = 0; u < kDdaLL128WordsPerThread; ++u) regs[u] = 0;
        ddaLL128StoreWire(scatterDst + wireOff, regs, flag, flagLane);
      }

      // Gather push: chunk[root] -> gather slot[root]
      {
        const size_t dOff = chunkOff(root) + s * (size_t)kDdaLL128DataBytesPerSlice;
        const size_t rem = totalBytes > dOff ? totalBytes - dOff : 0;
        const int elt = rem < (size_t)kDdaLL128DataBytesPerSlice ? (int)rem : kDdaLL128DataBytesPerSlice;
        uint64_t regs[kDdaLL128WordsPerThread];
        if (elt > 0)
          ddaLL128LoadRegs<int8_t>(regs, sendbuff + dOff, elt, lane, flagLane);
        else
          for (int u = 0; u < kDdaLL128WordsPerThread; ++u) regs[u] = 0;
        ddaLL128StoreWire(gatherDst + wireOff, regs, flag, flagLane);
      }
    }

    // Phase 3: self-copy (out-of-place only)
    if (sendbuff != reinterpret_cast<const int8_t*>(recvbuff)) {
      const v4u* s4 = reinterpret_cast<const v4u*>(sendbuff);
      v4u* d4 = reinterpret_cast<v4u*>(recvbuff);
      const size_t nVec = totalBytes >> 4;
      const size_t gtid = (size_t)flatBlockId * (size_t)nthreads + (size_t)tid;
      const size_t stride = (size_t)totalBlk * (size_t)nthreads;
      for (size_t i = gtid; i < nVec; i += stride) d4[i] = s4[i];
    }

        // Phase 4: wait for every peer's allgather contribution.
    // Q5: non-roots forward only 1 padding slice to the root (see Phase 1).
    // The root uses its own sendbuff, so only 1 slice per peer is needed here.
    const uint64_t* myScratch = reinterpret_cast<const uint64_t*>(peerScratch[selfRank]);
    const uint64_t* myGather = myScratch + gatherOff + (uint64_t)peer * slotWireWords;
    if (gwarp == 0) {
      uint64_t vr[kDdaLL128WordsPerThread];
      if (ddaLL128PollWire(myGather + 2 * lane, vr, flag, lane, params))
        ddaLL128ClearWire(const_cast<uint64_t*>(myGather) + 2 * lane, lane);
    }
  } else {
    // --- NON-ROOT ---

    // Phase 1: poll scatter slot, forward to gather slots -- fan-out limited.
    // D6 recvbuff write happens in round 0 only.
    const uint64_t* myScatter = reinterpret_cast<const uint64_t*>(peerScratch[selfRank]) + bankWords;
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
        if (!ddaLL128PollWire(myScatter + s * (size_t)kDdaLL128WireWordsPerSlice + 2 * lane,
                              vr, flag, lane, params)) {
          aborted = true;
          break;
        }
        // Q5/O1: for the root peer, only forward 1 padding slice (s==0).
        if (tgtPeer != root || s == 0) {
          ddaLL128StoreWire(tgtGather + s * (size_t)kDdaLL128WireWordsPerSlice + 2 * lane,
                            vr, flag, flagLane);
        }
        if (nFanRounds > 1 ? (fRound == 0) : (col == (int)(s % (size_t)nPeers))) {
          const size_t dOff = chunkOff(selfRank) + s * (size_t)kDdaLL128DataBytesPerSlice;
          const size_t rem = totalBytes > dOff ? totalBytes - dOff : 0;
          const int elt = rem < (size_t)kDdaLL128DataBytesPerSlice ? (int)rem : kDdaLL128DataBytesPerSlice;
          if (elt > 0)
            ddaLL128StoreRegs<int8_t>(recvbuff + dOff, vr, elt, lane, flagLane);
        }
      }
    }

    // Phase 2: gather -- poll own gather slot[peer], unpack into recvbuff (unchanged)
    const uint64_t* myGather = reinterpret_cast<const uint64_t*>(peerScratch[selfRank]) +
      gatherOff + (uint64_t)peer * slotWireWords;
    for (size_t s = gwarp; s < chunkSlices; s += wstride) {
      const size_t dOff = chunkOff(peer) + s * (size_t)kDdaLL128DataBytesPerSlice;
      const size_t rem = totalBytes > dOff ? totalBytes - dOff : 0;
      const int elt = rem < (size_t)kDdaLL128DataBytesPerSlice ? (int)rem : kDdaLL128DataBytesPerSlice;
      uint64_t vr[kDdaLL128WordsPerThread];
      if (!ddaLL128PollWire(myGather + s * (size_t)kDdaLL128WireWordsPerSlice + 2 * lane,
                            vr, flag, lane, params))
        break;
      if (elt > 0)
        ddaLL128StoreRegs<int8_t>(recvbuff + dOff, vr, elt, lane, flagLane);
      //ddaLL128ClearWire(const_cast<uint64_t*>(myGather) +
      //  s * (size_t)kDdaLL128WireWordsPerSlice + 2 * lane, lane);
    }
  }

#if (defined(__gfx1250__) || defined(__gfx1250_strict__))
  asm volatile("s_wait_storecnt 0x0" ::: "memory");
#endif
  __syncthreads();
  ddaSetLLEpoch(epochDev, epochLen, flatBlockId, totalBlk, flag32);
}

} // namespace dda::common
