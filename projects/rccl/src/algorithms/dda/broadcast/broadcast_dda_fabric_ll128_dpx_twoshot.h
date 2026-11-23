/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * DPX-aware LL128 two-shot broadcast kernel for the DDA fabric path
 * (gfx1250 partition mode, hasMloPart == true, scaleInSize == 2).
 *
 * Each physical MI450 package is split into two logical ranks.  Running
 * scale-in and scale-up traffic concurrently degrades bandwidth (~58%
 * scale-up, ~70% scale-in remain).  This kernel minimises concurrent link
 * usage by separating sibling traffic from non-root peer traffic within
 * each rank's code path.
 *
 * col = blockIdx.x identifies which of the N-2 non-root peers this block
 * services.  col % 2 == rootLocal determines column ownership:
 *   colIsRoot  == true  → root scatters this chunk directly to the peer
 *                          (scale-up); rootSib receives this chunk from
 *                          non-root all-gather fan-out (scale-up).
 *   colIsRoot  == false → root stages this chunk to rootSib scratch
 *                          (scale-in); rootSib cut-through forwards it
 *                          to the peer scatter slot (scale-up).
 *
 * Root path (two separate loops per block):
 *   Sibling loop   (!colIsRoot): stage chunk[col] → rootSib staging slot
 *                                 over the in-package (scale-in) link.
 *   Non-root loop  ( colIsRoot): scatter chunk[col] → peer scatter slot
 *                                 over the scale-up fabric.
 *   Completion poll: one padding slice per non-root from own gather slot.
 *
 * rootSib path (two separate loops per block):
 *   Sibling loop   (!colIsRoot): poll staging slot written by root, cut-
 *                                 through forward to peer scatter slot
 *                                 (scale-up); clear staging; write recvbuff.
 *   Non-root loop  ( colIsRoot): poll own gather slot written by non-root
 *                                 all-gather fan-out; write recvbuff; clear.
 *
 * Non-root path: standard SPX all-gather with DPX column mapping.
 *
 * Chunk sizing: cs = ceil(totalSlices / (N-2))  (excludes both co-roots).
 * Grid:         dim3(N-2, bpp) — blockIdx.x == col (0..N-3).
 * col → rank:   peer(col) = col < min(root, rootSib) ? col : col + 2
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

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

constexpr size_t ddaLL128BcDpxChunkSlices(size_t totalSlices, int nRanks) {
  // DPX: N-2 non-root peers share the message (both co-roots excluded).
  const int nNonRoot = nRanks - 2;
  if (nNonRoot < 1) return totalSlices;
  size_t cs = (totalSlices + (size_t)nNonRoot - 1) / (size_t)nNonRoot;
  return cs < 1 ? 1 : cs;
}

constexpr size_t ddaLL128BcDpxSlotSlices(size_t bankSize, int nRanks, size_t totalSlices) {
  const size_t cs = ddaLL128BcDpxChunkSlices(totalSlices, nRanks);
  // Scratch layout identical to SPX: (nRanks+1) slots of cs slices each.
  const size_t bankSlices = bankSize / (size_t)kDdaLL128WireBytesPerSlice;
  return (bankSlices / (size_t)(nRanks + 1) >= cs) ? cs : 0;
}

------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

template <int NRANKS_CT>
__launch_bounds__(512)
__global__ void ddaBroadcastFabricLL128DpxTwoShot(
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

  const int nRanks   = NRANKS_CT ? NRANKS_CT : nRanksRt;
  const int nNonRoot = nRanks - 2;   // number of non-root peers = grid width

  // Co-root identities.
  // DPX topology invariant (enforced by scaleInSize validation in fabric_init.cu):
  // sibling ranks are always assigned as consecutive even-odd pairs —
  // (0,1), (2,3), (4,5), ... — so rank % 2 gives the intra-package position
  // and rank ^ 1 gives the sibling.  root % 2 is therefore well-defined even
  // though root can be any rank; it only carries this meaning because the
  // eligibility check has already confirmed scaleInSize == 2 with that ordering.
  const int rootLocal = root % 2;          // 0 if root is even, 1 if odd
  const int rootSib   = root ^ 1;          // the other rank on root's package

  const bool isRoot    = (selfRank == root);
  const bool isRootSib = (selfRank == rootSib);

  // Thread/warp layout
  const int tid      = threadIdx.x;
  const int nthreads = blockDim.x;
  const int lane     = tid % kDdaLL128Warp;
  const int warp     = tid / kDdaLL128Warp;
  const int nwarps   = nthreads / kDdaLL128Warp;
  const bool flagLane = ddaLL128IsFlagLane(lane);

  const int flatBlockId = (int)(blockIdx.x * gridDim.y + blockIdx.y);
  const int totalBlk    = (int)(gridDim.x * gridDim.y);

  const uint32_t flag32 = ddaGetLLEpochInc(epochDev, flatBlockId, 1);
  const uint64_t flag   = ((uint64_t)flag32 << 32) | (uint64_t)flag32;

  // Scratch bank (alternates each call via flag parity)
  const uint64_t bankWords   = (uint64_t)(flag32 & 1u) * (uint64_t)(bankSize / sizeof(uint64_t));
  const size_t slotWireWords = chunkSlices * (size_t)kDdaLL128WireWordsPerSlice;
  const uint64_t gatherOff   = bankWords + slotWireWords;

  // col == blockIdx.x: which of the N-2 non-root peers this block services.
  // peer(col): skip both root and rootSib (always adjacent).
  const int col       = (int)blockIdx.x;
  const int minCoRoot = root < rootSib ? root : rootSib;
  const int peer      = col < minCoRoot ? col : col + 2;   // non-root rank

  // Column ownership: the N-2 non-root columns are split evenly between the two
  // co-roots.  Columns with the same parity as rootLocal belong to root; the
  // rest belong to rootSib.  This ensures each co-root handles exactly half the
  // scale-up transfers so neither link is idle while the other is saturated.
  const bool colIsRoot = (col % 2 == rootLocal);

  // Data offset for col (0-based index into the N-2 chunk layout)
  auto chunkOff = [&](int c) -> size_t {
    return (size_t)c * chunkSlices * (size_t)kDdaLL128DataBytesPerSlice;
  };

  // Warp stripe for slice iteration within this block
  const size_t gwarp   = (size_t)blockIdx.y * (size_t)nwarps + (size_t)warp;
  const size_t wstride = (size_t)gridDim.y * (size_t)nwarps;

  // Fan-out: 0 or >= nNonRoot means full fan-out (same semantics as SPX)
  const int kFanOut   = (fanOut <= 0 || fanOut >= nNonRoot) ? nNonRoot : fanOut;
  const int nFanRounds = (nNonRoot + kFanOut - 1) / kFanOut;

  // =========================================================================
  // ROOT role
  // =========================================================================
  if (isRoot) {
    // ---- All-CU two-phase root scatter ----
    // All N-2 blocks are active in both phases.  Columns are paired:
    //   chunkPair = col / 2  — which of the (N-2)/2 chunk pairs to service
    //   subBlk    = col % 2  — selects interleaved warp start within the pair
    //
    // Two blocks cooperate to cover all chunkSlices slices of the same chunk
    // via a doubled warp stride (wstride2). Root sends to colIsRoot ranks only
    // (dirPeer, one per non-root package); rootSib sends to !colIsRoot ranks
    // only (sibPeer) via staging cut-through. The two co-roots never share a
    // scatter slot.
    //
    // Phase 1 (scale-in):  both blocks stage chunk[sibColIdx] to rootSib's
    //                      staging slot for sibPeer, interleaved via wstride2.
    // Phase 2 (scale-up):  both blocks scatter chunk[dirColIdx] directly to
    //                      dirPeer's scatter slot, interleaved via wstride2.
    // Within each block phase 1 completes before phase 2: per-block temporal
    // separation of scale-in and scale-up traffic.

    const int chunkPair = col / 2;
    const int subBlk    = col % 2;

    // !colIsRoot column for this pair -> rootSib staging target in phase 1
    const int sibColIdx = chunkPair * 2 + (1 - rootLocal);
    const int sibPeer   = sibColIdx < minCoRoot ? sibColIdx : sibColIdx + 2;

    // colIsRoot column for this pair -> direct scatter target in phase 2
    const int dirColIdx = chunkPair * 2 + rootLocal;
    const int dirPeer   = dirColIdx < minCoRoot ? dirColIdx : dirColIdx + 2;

    // Two blocks cover the same chunk; each takes alternate warps.
    const size_t wstart   = (size_t)subBlk * (size_t)gridDim.y * (size_t)nwarps
                            + (size_t)blockIdx.y * (size_t)nwarps + (size_t)warp;
    //each pair of blocks works on a contigious chunk (set of slices)
    const size_t wstride2 = 2 * (size_t)gridDim.y * (size_t)nwarps;

    // ---- Phase 1: stage chunk[sibColIdx] to rootSib staging (scale-in) ----
    {
      uint64_t* stagingDst =
        reinterpret_cast<uint64_t*>(peerScratch[rootSib]) + gatherOff +
        (uint64_t)sibPeer * slotWireWords;
      for (size_t s = wstart; s < chunkSlices; s += wstride2) {
        const size_t wireOff = s * (size_t)kDdaLL128WireWordsPerSlice + 2 * lane;
        const size_t dOff    = chunkOff(sibColIdx) + s * (size_t)kDdaLL128DataBytesPerSlice;
        const size_t rem     = totalBytes > dOff ? totalBytes - dOff : 0;
        const int    elt     = rem < (size_t)kDdaLL128DataBytesPerSlice
                                 ? (int)rem : kDdaLL128DataBytesPerSlice;
        uint64_t regs[kDdaLL128WordsPerThread];
        if (elt > 0)
          ddaLL128LoadRegsBatched<int8_t>(regs, sendbuff + dOff, elt, lane, flagLane);
        else
          for (int u = 0; u < kDdaLL128WordsPerThread; ++u) regs[u] = 0;
        ddaLL128StoreWire(stagingDst + wireOff, regs, flag, flagLane);
        if (elt > 0 && sendbuff != reinterpret_cast<const int8_t*>(recvbuff))
          ddaLL128StoreRegs<int8_t>(recvbuff + dOff, regs, elt, lane, flagLane);
      }
    }

    // ---- Phase 2: scatter chunk[dirColIdx] to dirPeer (scale-up) ----
    {
      uint64_t* scatterDst = reinterpret_cast<uint64_t*>(peerScratch[dirPeer]) + bankWords;
      for (size_t s = wstart; s < chunkSlices; s += wstride2) {
        const size_t wireOff = s * (size_t)kDdaLL128WireWordsPerSlice + 2 * lane;
        const size_t dOff    = chunkOff(dirColIdx) + s * (size_t)kDdaLL128DataBytesPerSlice;
        const size_t rem     = totalBytes > dOff ? totalBytes - dOff : 0;
        const int    elt     = rem < (size_t)kDdaLL128DataBytesPerSlice
                                 ? (int)rem : kDdaLL128DataBytesPerSlice;
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

    // ---- Completion poll: one padding slice per non-root ----
    // root already holds all data from sendbuff; this poll gates exit only.
    const uint64_t* myGather =
      reinterpret_cast<const uint64_t*>(peerScratch[selfRank]) + gatherOff +
      (uint64_t)peer * slotWireWords;
    if (gwarp == 0) {
      uint64_t vr[kDdaLL128WordsPerThread];
      if (ddaLL128PollWireBatched(myGather + 2 * lane, vr, flag, lane, params))
        ddaLL128ClearWire(const_cast<uint64_t*>(myGather) + 2 * lane, lane);
    }
  }
  // ROOTSIB role
  // =========================================================================
  else if (isRootSib) {
    // ---- Sibling cut-through: !colIsRoot blocks only ----
    // rootSib polls peerScratch[rootSib].gather[peer] (staging slot written by
    // root over scale-in), forwards wire words to peerScratch[peer].scatter_slot
    // (scale-up), clears staging, and fuses the recvbuff write.
    if (!colIsRoot) {
      const uint64_t* stagingSrc =
        reinterpret_cast<const uint64_t*>(peerScratch[rootSib]) + gatherOff +
        (uint64_t)peer * slotWireWords;
      uint64_t* scatterDst = reinterpret_cast<uint64_t*>(peerScratch[peer]) + bankWords;
      for (size_t s = gwarp; s < chunkSlices; s += wstride) {
        const size_t wireOff = s * (size_t)kDdaLL128WireWordsPerSlice + 2 * lane;
        const size_t dOff    = chunkOff(col) + s * (size_t)kDdaLL128DataBytesPerSlice;
        const size_t rem     = totalBytes > dOff ? totalBytes - dOff : 0;
        const int    elt     = rem < (size_t)kDdaLL128DataBytesPerSlice
                                 ? (int)rem : kDdaLL128DataBytesPerSlice;
        uint64_t vr[kDdaLL128WordsPerThread];
        if (!ddaLL128PollWireBatched(stagingSrc + wireOff, vr, flag, lane, params))
          break;
        ddaLL128StoreWire(scatterDst + wireOff, vr, flag, flagLane);
        ddaLL128ClearWire(const_cast<uint64_t*>(stagingSrc) + wireOff, lane);
        if (elt > 0)
          ddaLL128StoreRegs<int8_t>(recvbuff + dOff, vr, elt, lane, flagLane);
      }
    }

    // ---- Allgather gather: colIsRoot blocks only ----
    // Non-roots fan out to rootSib's gather[peer] during the allgather phase.
    // rootSib polls, writes recvbuff, and clears.
    if (colIsRoot) {
      const uint64_t* myGather =
        reinterpret_cast<const uint64_t*>(peerScratch[selfRank]) + gatherOff +
        (uint64_t)peer * slotWireWords;
      for (size_t s = gwarp; s < chunkSlices; s += wstride) {
        const size_t wireOff = s * (size_t)kDdaLL128WireWordsPerSlice + 2 * lane;
        const size_t dOff    = chunkOff(col) + s * (size_t)kDdaLL128DataBytesPerSlice;
        const size_t rem     = totalBytes > dOff ? totalBytes - dOff : 0;
        const int    elt     = rem < (size_t)kDdaLL128DataBytesPerSlice
                                 ? (int)rem : kDdaLL128DataBytesPerSlice;
        uint64_t vr[kDdaLL128WordsPerThread];
        if (!ddaLL128PollWireBatched(myGather + wireOff, vr, flag, lane, params))
          break;
        if (elt > 0)
          ddaLL128StoreRegs<int8_t>(recvbuff + dOff, vr, elt, lane, flagLane);
        ddaLL128ClearWire(const_cast<uint64_t*>(myGather) + wireOff, lane);
      }
    }
  }
  // NON-ROOT role — standard SPX all-gather with DPX column mapping
  // =========================================================================
  else {
    // selfCol: this rank's own column index (0..N-3, skipping both co-roots)
    const int selfCol = selfRank < minCoRoot ? selfRank : selfRank - 2;

    // Phase 1 (scatter receive) + fan-out to all gather slots.
    // Identical to SPX non-root phase 1, with nNonRoot substituted for nPeers
    // and the DPX gather-slot targets including rootSib (but not root, which
    // only needs 1 padding slice).
    const uint64_t* myScatter =
      reinterpret_cast<const uint64_t*>(peerScratch[selfRank]) + bankWords;

    bool aborted = false;
    for (int fRound = 0; fRound < nFanRounds && !aborted; ++fRound) {
      const int kActive = kFanOut < (nNonRoot - fRound * kFanOut)
                        ? kFanOut : (nNonRoot - fRound * kFanOut);
      if (kActive <= 0) break;
      const int peerGroup = col % kActive;
      const int tgtCol    = fRound * kFanOut + peerGroup;
      if (tgtCol >= nNonRoot) continue;

      // tgtCol → tgtPeer: map back through DPX column mapping (skip both co-roots)
      const int tgtPeer = tgtCol < minCoRoot ? tgtCol : tgtCol + 2;

      const int localCol   = col / kActive;
      const int myColCount = (nNonRoot - peerGroup + kActive - 1) / kActive;
      if (localCol >= myColCount) continue;

      const size_t gwarp_r  = ((size_t)localCol * (size_t)gridDim.y + (size_t)blockIdx.y)
                              * (size_t)nwarps + (size_t)warp;
      const size_t wstride_r = (size_t)gridDim.y * (size_t)myColCount * (size_t)nwarps;

      // Write gather slot on tgtPeer.  For root: only 1 padding slice (s==0).
      // For rootSib: write full chunk so rootSib can assemble recvbuff.
      uint64_t* tgtGather =
        reinterpret_cast<uint64_t*>(peerScratch[tgtPeer]) + gatherOff +
        (uint64_t)selfRank * slotWireWords;

      for (size_t s = gwarp_r; s < chunkSlices; s += wstride_r) {
        uint64_t vr[kDdaLL128WordsPerThread];
        if (!ddaLL128PollWireBatched(
              myScatter + s * (size_t)kDdaLL128WireWordsPerSlice + 2 * lane,
              vr, flag, lane, params)) {
          aborted = true;
          break;
        }
        // Fan-out: skip non-padding slices when target is root.
        if (tgtPeer != root || s == 0) {
          ddaLL128StoreWire(tgtGather + s * (size_t)kDdaLL128WireWordsPerSlice + 2 * lane,
                            vr, flag, flagLane);
        }
        // Fuse recvbuff write in the round that covers our own scatter slot.
        if (nFanRounds > 1 ? (fRound == 0) : (col == (int)(s % (size_t)nNonRoot))) {
          const size_t dOff = chunkOff(selfCol) + s * (size_t)kDdaLL128DataBytesPerSlice;
          const size_t rem  = totalBytes > dOff ? totalBytes - dOff : 0;
          const int    elt  = rem < (size_t)kDdaLL128DataBytesPerSlice
                                ? (int)rem : kDdaLL128DataBytesPerSlice;
          if (elt > 0)
            ddaLL128StoreRegs<int8_t>(recvbuff + dOff, vr, elt, lane, flagLane);
        }
      }
    }

    // Phase 2 (gather): poll own gather slot[peer] for every other non-root chunk.
    // Skip the root column (root has no chunk; it only received a padding slice).
    // rootSib column is included: rootSib wrote the full chunk to our gather slot.
    if (peer != root) {
      const uint64_t* myGather =
        reinterpret_cast<const uint64_t*>(peerScratch[selfRank]) + gatherOff +
        (uint64_t)peer * slotWireWords;
      // col index for this peer (needed for recvbuff offset)
      const int peerCol = peer < minCoRoot ? peer : peer - 2;
      for (size_t s = gwarp; s < chunkSlices; s += wstride) {
        const size_t dOff = chunkOff(peerCol) + s * (size_t)kDdaLL128DataBytesPerSlice;
        const size_t rem  = totalBytes > dOff ? totalBytes - dOff : 0;
        const int    elt  = rem < (size_t)kDdaLL128DataBytesPerSlice
                              ? (int)rem : kDdaLL128DataBytesPerSlice;
        uint64_t vr[kDdaLL128WordsPerThread];
        if (!ddaLL128PollWireBatched(
              myGather + s * (size_t)kDdaLL128WireWordsPerSlice + 2 * lane,
              vr, flag, lane, params))
          break;
        if (elt > 0)
          ddaLL128StoreRegs<int8_t>(recvbuff + dOff, vr, elt, lane, flagLane);
        ddaLL128ClearWire(
          const_cast<uint64_t*>(myGather) + s * (size_t)kDdaLL128WireWordsPerSlice + 2 * lane,
          lane);
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
