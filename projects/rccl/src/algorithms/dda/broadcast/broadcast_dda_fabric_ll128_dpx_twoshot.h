/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * DPX-aware LL128 two-shot broadcast kernel for the DDA fabric path
 * (gfx1250 partition mode, hasMloPart == true, scaleInSize == 2).
 *
 * Warp-specialized design: within each block, warps are split into a
 * scale-up group (round-robin across same-parity peers) and a scale-in
 * group (targeting the in-package sibling).  Two tunables control the
 * split: warpsPerPeer (SU warps per target) and scaleInPct (% of warps
 * doing SI).  Grid width is decoupled from peer count and derived from
 * the tunables + CU budget.
 *
 * See dpx_twoshot_warp_specialized_design.md + Addendum A for the full
 * design specification.
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
// Geometry helpers (DPX variant — N-2 chunks, excluding both co-roots)
// ---------------------------------------------------------------------------

constexpr size_t ddaLL128BcDpxChunkSlices(size_t totalSlices, int nRanks) {
  const int nNonRoot = nRanks - 2;
  if (nNonRoot < 1) return totalSlices;
  size_t cs = (totalSlices + (size_t)nNonRoot - 1) / (size_t)nNonRoot;
  return cs < 1 ? 1 : cs;
}

constexpr size_t ddaLL128BcDpxSlotSlices(size_t bankSize, int nRanks, size_t totalSlices) {
  const size_t cs = ddaLL128BcDpxChunkSlices(totalSlices, nRanks);
  const size_t bankSlices = bankSize / (size_t)kDdaLL128WireBytesPerSlice;
  return (bankSlices / (size_t)(nRanks + 1) >= cs) ? cs : 0;
}

// ---------------------------------------------------------------------------
// Host-side geometry derivation (called from the .cu launcher)
// ---------------------------------------------------------------------------

struct DpxWsGeom {
  int suWarps;        // scale-up warps per block
  int siWarps;        // scale-in warps per block
  int peersPerBlock;  // same-parity peers served by SU warps per block
  int gridWidth;      // gridDim.x
  int bpp;            // gridDim.y (blocks per peer-set, for CU filling)
};

inline DpxWsGeom ddaDpxWsGeom(int nwarps, int warpsPerPeer, int scaleInPct,
                               int nSameParityPeers, int budget) {
  int siRaw  = nwarps * scaleInPct / 100;
  if (siRaw < 1) siRaw = 1;
  int suRaw  = nwarps - siRaw;
  int ppb    = suRaw / warpsPerPeer;
  if (ppb < 1) ppb = 1;
  int suWarps = ppb * warpsPerPeer;
  int siWarps = nwarps - suWarps;
  if (siWarps < 1) { siWarps = 1; suWarps = nwarps - 1; ppb = suWarps / warpsPerPeer; if (ppb < 1) ppb = 1; suWarps = ppb * warpsPerPeer; siWarps = nwarps - suWarps; }

  int gw = (nSameParityPeers + ppb - 1) / ppb;
  if (gw < 1) gw = 1;
  int bpp = budget / gw;
  if (bpp < 1) bpp = 1;

  return {suWarps, siWarps, ppb, gw, bpp};
}

// ---------------------------------------------------------------------------
// Parity helpers
// ---------------------------------------------------------------------------

// Map a 0-based index within a parity group back to a non-root rank.
// Parity-p ranks form the sequence p, p+2, p+4, ... and the co-root of
// that parity sits at position minCoRoot/2 in the sequence.  Skip it.
__device__ __forceinline__ int dpxParityPeerRank(int idx, int parity, int minCoRoot, int /*nRanks*/) {
  const int skipIdx = minCoRoot / 2;
  return parity + 2 * (idx < skipIdx ? idx : idx + 1);
}

// Count non-root ranks of given parity.
// Co-roots are always an adjacent even-odd pair, so removing them takes
// exactly one from each parity group: result is nRanks/2 - 1 for either parity.
__device__ __forceinline__ int dpxParityPeerCount(int /*parity*/, int /*minCoRoot*/, int nRanks) {
  return nRanks / 2 - 1;
}

// Map a non-root rank to its column index (0-based, skipping both co-roots).
__device__ __forceinline__ int dpxRankToCol(int rank, int minCoRoot) {
  return rank < minCoRoot ? rank : rank - 2;
}

// ---------------------------------------------------------------------------
// Shared derived state passed to extracted per-role device functions.
// ---------------------------------------------------------------------------

struct DpxKernelState {
  int rootSib;
  int minCoRoot;
  int warp;
  int lane;
  bool flagLane;
  uint64_t flag;
  uint64_t bankWords;
  uint64_t gatherOff;
  size_t slotWireWords;
};

// ---------------------------------------------------------------------------
// Root scale-up: scatter to same-parity non-root peers + stage to rootSib.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void ddaDpxRootScatterSU(
    int8_t* const* __restrict__ peerScratch,
    int8_t* __restrict__ recvbuff,
    const int8_t* __restrict__ sendbuff,
    size_t totalBytes,
    int nRanks, int root,
    size_t chunkSlices,
    int warpsPerPeer,
    int peersPerBlock,
    const DpxKernelState& st) {

  const int suParity = root % 2;
  const int suWarp     = st.warp;
  const int peerIdx    = suWarp / warpsPerPeer;
  const int warpInPeer = suWarp % warpsPerPeer;

  const int globalPeerIdx = (int)blockIdx.x * peersPerBlock + peerIdx;
  const int nSuPeers = dpxParityPeerCount(suParity, st.minCoRoot, nRanks);
  if (globalPeerIdx >= nSuPeers) return;

  const int peer     = dpxParityPeerRank(globalPeerIdx, suParity, st.minCoRoot, nRanks);
  const int chunkIdx = dpxRankToCol(peer, st.minCoRoot);

  uint64_t* scatterDst = reinterpret_cast<uint64_t*>(peerScratch[peer]) + st.bankWords;
  uint64_t* rootSibGather =
    reinterpret_cast<uint64_t*>(peerScratch[st.rootSib]) + st.gatherOff +
    (uint64_t)peer * st.slotWireWords;

  const size_t gwarp_su   = (size_t)blockIdx.y * (size_t)warpsPerPeer + (size_t)warpInPeer;
  const size_t wstride_su = (size_t)gridDim.y * (size_t)warpsPerPeer;

  for (size_t s = gwarp_su; s < chunkSlices; s += wstride_su) {
    const size_t wireOff = s * (size_t)kDdaLL128WireWordsPerSlice + 2 * st.lane;
    const size_t dOff    = ((size_t)chunkIdx * chunkSlices + s) * (size_t)kDdaLL128DataBytesPerSlice;
    const size_t rem     = totalBytes > dOff ? totalBytes - dOff : 0;
    const int    elt     = rem < (size_t)kDdaLL128DataBytesPerSlice
                             ? (int)rem : kDdaLL128DataBytesPerSlice;
    uint64_t regs[kDdaLL128WordsPerThread];
    if (elt > 0)
      ddaLL128LoadRegsBatched<int8_t>(regs, sendbuff + dOff, elt, st.lane, st.flagLane);
    else
      for (int u = 0; u < kDdaLL128WordsPerThread; ++u) regs[u] = 0;
    ddaLL128StoreWire(scatterDst + wireOff, regs, st.flag, st.flagLane);
    ddaLL128StoreWire(rootSibGather + wireOff, regs, st.flag, st.flagLane);
    if (elt > 0 && sendbuff != reinterpret_cast<const int8_t*>(recvbuff))
      ddaLL128StoreRegs<int8_t>(recvbuff + dOff, regs, elt, st.lane, st.flagLane);
  }
}

// ---------------------------------------------------------------------------
// Root scatter: stage rootSib-parity chunks to sibling co-root (flat grid stride).
// ---------------------------------------------------------------------------

__device__ __forceinline__ void ddaDpxRootStageToSib(
    int8_t* const* __restrict__ peerScratch,
    int8_t* __restrict__ recvbuff,
    const int8_t* __restrict__ sendbuff,
    size_t totalBytes,
    int nRanks, int root,
    size_t chunkSlices,
    int suWarpsPerBlock,
    int siWarpsPerBlock,
    const DpxKernelState& st) {

  const int siParity = (root % 2) ^ 1;
  const int nSiPeers = dpxParityPeerCount(siParity, st.minCoRoot, nRanks);
  const int siWarp = st.warp - suWarpsPerBlock;

  const int totalSiInRow = (int)gridDim.x * siWarpsPerBlock;
  int siWarpsPerPeer = totalSiInRow > nSiPeers ? totalSiInRow / nSiPeers : 1;
  if (siWarpsPerPeer > siWarpsPerBlock) siWarpsPerPeer = siWarpsPerBlock;
  const int siPeersPerBlock = siWarpsPerBlock / siWarpsPerPeer;
  const int peerIdx    = siWarp / siWarpsPerPeer;
  const int warpInPeer = siWarp % siWarpsPerPeer;
  const int totalSiPeersPerRow = (int)gridDim.x * siPeersPerBlock;

  for (int pi = (int)blockIdx.x * siPeersPerBlock + peerIdx;
       pi < nSiPeers; pi += totalSiPeersPerRow) {
    const int peer = dpxParityPeerRank(pi, siParity, st.minCoRoot, nRanks);
    const int chunkIdx = dpxRankToCol(peer, st.minCoRoot);

    uint64_t* stagingDst =
      reinterpret_cast<uint64_t*>(peerScratch[st.rootSib]) + st.gatherOff +
      (uint64_t)peer * st.slotWireWords;

    const size_t gwarp_si   = (size_t)blockIdx.y * (size_t)siWarpsPerPeer + (size_t)warpInPeer;
    const size_t wstride_si = (size_t)gridDim.y * (size_t)siWarpsPerPeer;
    for (size_t s = gwarp_si; s < chunkSlices; s += wstride_si) {
      const size_t wireOff = s * (size_t)kDdaLL128WireWordsPerSlice + 2 * st.lane;
      const size_t dOff    = ((size_t)chunkIdx * chunkSlices + s) * (size_t)kDdaLL128DataBytesPerSlice;
      const size_t rem     = totalBytes > dOff ? totalBytes - dOff : 0;
      const int    elt     = rem < (size_t)kDdaLL128DataBytesPerSlice
                               ? (int)rem : kDdaLL128DataBytesPerSlice;
      uint64_t regs[kDdaLL128WordsPerThread];
      if (elt > 0)
        ddaLL128LoadRegsBatched<int8_t>(regs, sendbuff + dOff, elt, st.lane, st.flagLane);
      else
        for (int u = 0; u < kDdaLL128WordsPerThread; ++u) regs[u] = 0;
      ddaLL128StoreWire(stagingDst + wireOff, regs, st.flag, st.flagLane);
      if (elt > 0 && sendbuff != reinterpret_cast<const int8_t*>(recvbuff))
        ddaLL128StoreRegs<int8_t>(recvbuff + dOff, regs, elt, st.lane, st.flagLane);
    }
  }
}

// ---------------------------------------------------------------------------
// Co-root completion poll: one padding slice per non-root (flat stride).
// ---------------------------------------------------------------------------

__device__ __forceinline__ void ddaDpxCoRootCompletionPoll(
    int8_t* const* __restrict__ peerScratch,
    int selfRank, int nNonRoot, int minCoRoot,
    int nwarps,
    size_t chunkSlices,
    DdaKernelParams params,
    const DpxKernelState& st) {

  const int flatBlockId = (int)(blockIdx.x * gridDim.y + blockIdx.y);
  const int totalBlk    = (int)(gridDim.x * gridDim.y);
  const size_t gwarp_c   = (size_t)flatBlockId * (size_t)nwarps + (size_t)st.warp;
  const size_t wstride_c = (size_t)totalBlk * (size_t)nwarps;

  for (size_t pi = gwarp_c; pi < (size_t)nNonRoot; pi += wstride_c) {
    const int peer = (int)pi < minCoRoot ? (int)pi : (int)pi + 2;
    const uint64_t* myGather =
      reinterpret_cast<const uint64_t*>(peerScratch[selfRank]) + st.gatherOff +
      (uint64_t)peer * st.slotWireWords;
    uint64_t vr[kDdaLL128WordsPerThread];
    if (ddaLL128PollWireBatched(myGather + 2 * st.lane, vr, st.flag, st.lane, params))
      ddaLL128ClearWire(const_cast<uint64_t*>(myGather) + 2 * st.lane, st.lane);
  }
}

// ---------------------------------------------------------------------------
// RootSib scale-up: cut-through forward staged chunks to same-parity peers.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void ddaDpxRootSibForwardSU(
    int8_t* const* __restrict__ peerScratch,
    int8_t* __restrict__ recvbuff,
    size_t totalBytes,
    int selfRank, int nRanks, int rootSibParity,
    size_t chunkSlices,
    int warpsPerPeer,
    int peersPerBlock,
    DdaKernelParams params,
    const DpxKernelState& st) {

  const int suWarp     = st.warp;
  const int peerIdx    = suWarp / warpsPerPeer;
  const int warpInPeer = suWarp % warpsPerPeer;

  const int globalPeerIdx = (int)blockIdx.x * peersPerBlock + peerIdx;
  const int nSuPeers = dpxParityPeerCount(rootSibParity, st.minCoRoot, nRanks);
  if (globalPeerIdx >= nSuPeers) return;

  const int peer = dpxParityPeerRank(globalPeerIdx, rootSibParity, st.minCoRoot, nRanks);
  const int chunkIdx = dpxRankToCol(peer, st.minCoRoot);

  const uint64_t* stagingSrc =
    reinterpret_cast<const uint64_t*>(peerScratch[selfRank]) + st.gatherOff +
    (uint64_t)peer * st.slotWireWords;
  uint64_t* scatterDst = reinterpret_cast<uint64_t*>(peerScratch[peer]) + st.bankWords;

  const size_t gwarp_su   = (size_t)blockIdx.y * (size_t)warpsPerPeer + (size_t)warpInPeer;
  const size_t wstride_su = (size_t)gridDim.y * (size_t)warpsPerPeer;

  for (size_t s = gwarp_su; s < chunkSlices; s += wstride_su) {
    const size_t wireOff = s * (size_t)kDdaLL128WireWordsPerSlice + 2 * st.lane;
    uint64_t vr[kDdaLL128WordsPerThread];
    if (!ddaLL128PollWireBatched(stagingSrc + wireOff, vr, st.flag, st.lane, params))
      break;
    ddaLL128StoreWire(scatterDst + wireOff, vr, st.flag, st.flagLane);
    ddaLL128ClearWire(const_cast<uint64_t*>(stagingSrc) + wireOff, st.lane);

    const size_t dOff = ((size_t)chunkIdx * chunkSlices + s) * (size_t)kDdaLL128DataBytesPerSlice;
    const size_t rem  = totalBytes > dOff ? totalBytes - dOff : 0;
    const int    elt  = rem < (size_t)kDdaLL128DataBytesPerSlice
                          ? (int)rem : kDdaLL128DataBytesPerSlice;
    if (elt > 0)
      ddaLL128StoreRegs<int8_t>(recvbuff + dOff, vr, elt, st.lane, st.flagLane);
  }
}

// ---------------------------------------------------------------------------
// RootSib scale-in: receive root-parity chunks from root's SU staging.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void ddaDpxRootSibReceiveFromRoot(
    int8_t* const* __restrict__ peerScratch,
    int8_t* __restrict__ recvbuff,
    size_t totalBytes,
    int selfRank, int nRanks, int root,
    size_t chunkSlices,
    int suWarpsPerBlock,
    int siWarpsPerBlock,
    DdaKernelParams params,
    const DpxKernelState& st) {

  const int rootParity = root % 2;
  const int nRootPeers = dpxParityPeerCount(rootParity, st.minCoRoot, nRanks);
  const int siWarp = st.warp - suWarpsPerBlock;

  const int totalSiInRow = (int)gridDim.x * siWarpsPerBlock;
  int siWarpsPerPeer = totalSiInRow > nRootPeers ? totalSiInRow / nRootPeers : 1;
  if (siWarpsPerPeer > siWarpsPerBlock) siWarpsPerPeer = siWarpsPerBlock;
  const int siPeersPerBlock = siWarpsPerBlock / siWarpsPerPeer;
  const int peerIdx    = siWarp / siWarpsPerPeer;
  const int warpInPeer = siWarp % siWarpsPerPeer;
  const int totalSiPeersPerRow = (int)gridDim.x * siPeersPerBlock;

  for (int pi = (int)blockIdx.x * siPeersPerBlock + peerIdx;
       pi < nRootPeers; pi += totalSiPeersPerRow) {
    const int peer = dpxParityPeerRank(pi, rootParity, st.minCoRoot, nRanks);
    const int chunkIdx = dpxRankToCol(peer, st.minCoRoot);
    const uint64_t* stagingSrc =
      reinterpret_cast<const uint64_t*>(peerScratch[selfRank]) + st.gatherOff +
      (uint64_t)peer * st.slotWireWords;

    const size_t gwarp_si   = (size_t)blockIdx.y * (size_t)siWarpsPerPeer + (size_t)warpInPeer;
    const size_t wstride_si = (size_t)gridDim.y * (size_t)siWarpsPerPeer;
    for (size_t s = gwarp_si; s < chunkSlices; s += wstride_si) {
      const size_t wireOff = s * (size_t)kDdaLL128WireWordsPerSlice + 2 * st.lane;
      const size_t dOff    = ((size_t)chunkIdx * chunkSlices + s) * (size_t)kDdaLL128DataBytesPerSlice;
      const size_t rem     = totalBytes > dOff ? totalBytes - dOff : 0;
      const int    elt     = rem < (size_t)kDdaLL128DataBytesPerSlice
                               ? (int)rem : kDdaLL128DataBytesPerSlice;
      uint64_t vr[kDdaLL128WordsPerThread];
      if (!ddaLL128PollWireBatched(stagingSrc + wireOff, vr, st.flag, st.lane, params))
        break;
      if (elt > 0)
        ddaLL128StoreRegs<int8_t>(recvbuff + dOff, vr, elt, st.lane, st.flagLane);
      ddaLL128ClearWire(const_cast<uint64_t*>(stagingSrc) + wireOff, st.lane);
    }
  }
}

// ---------------------------------------------------------------------------
// Non-root scale-up: fan out own chunk to same-parity peers + sibling.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void ddaDpxNonRootFanOutSU(
    int8_t* const* __restrict__ peerScratch,
    int8_t* __restrict__ recvbuff,
    size_t totalBytes,
    const uint64_t* __restrict__ myScatter,
    int selfRank, int sibling, int selfChunkIdx, int selfParity,
    int nRanks,
    size_t chunkSlices,
    int warpsPerPeer,
    int peersPerBlock,
    DdaKernelParams params,
    const DpxKernelState& st) {

  const int suWarp     = st.warp;
  const int peerIdx    = suWarp / warpsPerPeer;
  const int warpInPeer = suWarp % warpsPerPeer;

  const int globalPeerIdx = (int)blockIdx.x * peersPerBlock + peerIdx;
  const int suTargetCount = nRanks / 2 - 2;

  if (globalPeerIdx >= suTargetCount) return;

  const int selfIdxInParity = selfRank < st.minCoRoot ? selfRank / 2
                                                       : selfRank / 2 - 1;
  const int adjustedIdx = globalPeerIdx < selfIdxInParity
                            ? globalPeerIdx : globalPeerIdx + 1;
  const int tgtRank = dpxParityPeerRank(adjustedIdx, selfParity, st.minCoRoot, nRanks);

  uint64_t* tgtGather =
    reinterpret_cast<uint64_t*>(peerScratch[tgtRank]) + st.gatherOff +
    (uint64_t)selfRank * st.slotWireWords;
  uint64_t* sibGather =
    reinterpret_cast<uint64_t*>(peerScratch[sibling]) + st.gatherOff +
    (uint64_t)selfRank * st.slotWireWords;

  const size_t gwarp_su   = (size_t)blockIdx.y * (size_t)warpsPerPeer + (size_t)warpInPeer;
  const size_t wstride_su = (size_t)gridDim.y * (size_t)warpsPerPeer;

  for (size_t s = gwarp_su; s < chunkSlices; s += wstride_su) {
    const size_t wireOff = s * (size_t)kDdaLL128WireWordsPerSlice + 2 * st.lane;
    uint64_t vr[kDdaLL128WordsPerThread];
    if (!ddaLL128PollWireBatched(myScatter + wireOff, vr, st.flag, st.lane, params))
      break;
    ddaLL128StoreWire(tgtGather + wireOff, vr, st.flag, st.flagLane);
    ddaLL128StoreWire(sibGather + wireOff, vr, st.flag, st.flagLane);

    const size_t dOff = ((size_t)selfChunkIdx * chunkSlices + s) * (size_t)kDdaLL128DataBytesPerSlice;
    const size_t rem  = totalBytes > dOff ? totalBytes - dOff : 0;
    const int    elt  = rem < (size_t)kDdaLL128DataBytesPerSlice
                          ? (int)rem : kDdaLL128DataBytesPerSlice;
    if (elt > 0)
      ddaLL128StoreRegs<int8_t>(recvbuff + dOff, vr, elt, st.lane, st.flagLane);
  }
}

// ---------------------------------------------------------------------------
// Non-root scale-in: relay own chunk to in-package sibling.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void ddaDpxNonRootRelaySI(
    int8_t* const* __restrict__ peerScratch,
    int8_t* __restrict__ recvbuff,
    size_t totalBytes,
    const uint64_t* __restrict__ myScatter,
    int selfRank, int sibling, int selfChunkIdx,
    size_t chunkSlices,
    int suWarpsPerBlock,
    int siWarpsPerBlock,
    DdaKernelParams params,
    const DpxKernelState& st) {

  const int siWarp = st.warp - suWarpsPerBlock;

  uint64_t* sibGather =
    reinterpret_cast<uint64_t*>(peerScratch[sibling]) + st.gatherOff +
    (uint64_t)selfRank * st.slotWireWords;

  const size_t gwarp_si   = (size_t)blockIdx.y * (size_t)siWarpsPerBlock + (size_t)siWarp;
  const size_t wstride_si = (size_t)gridDim.y * (size_t)siWarpsPerBlock;

  for (size_t s = gwarp_si; s < chunkSlices; s += wstride_si) {
    const size_t wireOff = s * (size_t)kDdaLL128WireWordsPerSlice + 2 * st.lane;
    uint64_t vr[kDdaLL128WordsPerThread];
    if (!ddaLL128PollWireBatched(myScatter + wireOff, vr, st.flag, st.lane, params))
      break;
    ddaLL128StoreWire(sibGather + wireOff, vr, st.flag, st.flagLane);

    const size_t dOff = ((size_t)selfChunkIdx * chunkSlices + s) * (size_t)kDdaLL128DataBytesPerSlice;
    const size_t rem  = totalBytes > dOff ? totalBytes - dOff : 0;
    const int    elt  = rem < (size_t)kDdaLL128DataBytesPerSlice
                          ? (int)rem : kDdaLL128DataBytesPerSlice;
    if (elt > 0)
      ddaLL128StoreRegs<int8_t>(recvbuff + dOff, vr, elt, st.lane, st.flagLane);
  }
}

// ---------------------------------------------------------------------------
// Non-root gather: poll gather slots from all peers, write recvbuff,
// relay same-parity data to sibling. Peers via x-dim, slices via y-dim.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void ddaDpxNonRootGather(
    int8_t* const* __restrict__ peerScratch,
    int8_t* __restrict__ recvbuff,
    size_t totalBytes,
    int selfRank, int sibling, int selfParity, int selfChunkIdx,
    int nRanks, int nwarps,
    size_t chunkSlices,
    DdaKernelParams params,
    const DpxKernelState& st) {

  const int nGatherPeers = nRanks - 2 - 1;
  const size_t peerWarp       = (size_t)blockIdx.x * (size_t)nwarps + (size_t)st.warp;
  const size_t totalPeerWarps = (size_t)gridDim.x * (size_t)nwarps;

  for (int pi = (int)peerWarp; pi < nGatherPeers; pi += (int)totalPeerWarps) {
    const int adjustedPi = pi < selfChunkIdx ? pi : pi + 1;
    const int r = adjustedPi < st.minCoRoot ? adjustedPi : adjustedPi + 2;

    const int peerChunkIdx = dpxRankToCol(r, st.minCoRoot);
    const bool isSameParity = ((r % 2) == selfParity);

    const uint64_t* myGather =
      reinterpret_cast<const uint64_t*>(peerScratch[selfRank]) + st.gatherOff +
      (uint64_t)r * st.slotWireWords;

    uint64_t* sibRelay = nullptr;
    if (isSameParity) {
      sibRelay = reinterpret_cast<uint64_t*>(peerScratch[sibling]) + st.gatherOff +
                 (uint64_t)r * st.slotWireWords;
    }

    for (size_t s = (size_t)blockIdx.y; s < chunkSlices; s += (size_t)gridDim.y) {
      const size_t wireOff = s * (size_t)kDdaLL128WireWordsPerSlice + 2 * st.lane;
      const size_t dOff    = ((size_t)peerChunkIdx * chunkSlices + s) * (size_t)kDdaLL128DataBytesPerSlice;
      const size_t rem     = totalBytes > dOff ? totalBytes - dOff : 0;
      const int    elt     = rem < (size_t)kDdaLL128DataBytesPerSlice
                               ? (int)rem : kDdaLL128DataBytesPerSlice;
      uint64_t vr[kDdaLL128WordsPerThread];
      if (!ddaLL128PollWireBatched(myGather + wireOff, vr, st.flag, st.lane, params))
        break;
      if (elt > 0)
        ddaLL128StoreRegs<int8_t>(recvbuff + dOff, vr, elt, st.lane, st.flagLane);
      if (sibRelay)
        ddaLL128StoreWire(sibRelay + wireOff, vr, st.flag, st.flagLane);
      ddaLL128ClearWire(const_cast<uint64_t*>(myGather) + wireOff, st.lane);
    }
  }
}

// ---------------------------------------------------------------------------
// Non-root completion: send one padding slice to each co-root.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void ddaDpxNonRootCompletion(
    int8_t* const* __restrict__ peerScratch,
    int selfRank, int root,
    const DpxKernelState& st) {

  const int flatBlockId = (int)(blockIdx.x * gridDim.y + blockIdx.y);
  const int nwarps = (int)(blockDim.x / kDdaLL128Warp);
  const size_t gwarp_p = (size_t)flatBlockId * (size_t)nwarps + (size_t)st.warp;
  if (gwarp_p != 0) return;

  uint64_t padding[kDdaLL128WordsPerThread];
  for (int u = 0; u < kDdaLL128WordsPerThread; ++u) padding[u] = 0;

  uint64_t* rootGather =
    reinterpret_cast<uint64_t*>(peerScratch[root]) + st.gatherOff +
    (uint64_t)selfRank * st.slotWireWords;
  ddaLL128StoreWire(rootGather + 2 * st.lane, padding, st.flag, st.flagLane);

  uint64_t* sibGather =
    reinterpret_cast<uint64_t*>(peerScratch[st.rootSib]) + st.gatherOff +
    (uint64_t)selfRank * st.slotWireWords;
  ddaLL128StoreWire(sibGather + 2 * st.lane, padding, st.flag, st.flagLane);
}

// ---------------------------------------------------------------------------
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
    int warpsPerPeer,
    int suWarpsPerBlock,
    int siWarpsPerBlock,
    int peersPerBlock) {

  const int nRanks   = NRANKS_CT ? NRANKS_CT : nRanksRt;
  const int nNonRoot = nRanks - 2;

  // Co-root identities
  const int rootLocal = root % 2;
  const int rootSib   = root - rootLocal + (1 - rootLocal);
  const int minCoRoot = root < rootSib ? root : rootSib;

  const bool isRoot    = (selfRank == root);
  const bool isRootSib = (selfRank == rootSib);
  const bool isCoRoot  = isRoot || isRootSib;

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

  // Scratch bank
  const uint64_t bankWords   = (uint64_t)(flag32 & 1u) * (uint64_t)(bankSize / sizeof(uint64_t));
  const size_t slotWireWords = chunkSlices * (size_t)kDdaLL128WireWordsPerSlice;
  const uint64_t gatherOff   = bankWords + slotWireWords;

  // Warp specialization: determine if this warp is scale-up or scale-in
  const bool isSU = (warp < suWarpsPerBlock);

  const DpxKernelState st = {rootSib, minCoRoot, warp, lane, flagLane,
                             flag, bankWords, gatherOff, slotWireWords};

  // Parity groups
  const int selfParity  = selfRank % 2;
  const int sibParity   = selfParity ^ 1;
  const int sibling     = selfRank ^ 1; // in-package sibling

  // =========================================================================
  // ROOT path
  // =========================================================================
  if (isRoot) {
    const int suParity = root % 2;      // root's same-parity peers
    const int siParity = suParity ^ 1;  // rootSib's parity peers (staged via SI)

    if (isSU) {
      ddaDpxRootScatterSU(peerScratch, recvbuff, sendbuff, totalBytes,
                          nRanks, root, chunkSlices, warpsPerPeer, peersPerBlock, st);
    } else {
      ddaDpxRootStageToSib(peerScratch, recvbuff, sendbuff, totalBytes,
                        nRanks, root, chunkSlices, suWarpsPerBlock, siWarpsPerBlock, st);
    }

    ddaDpxCoRootCompletionPoll(peerScratch, selfRank, nNonRoot, minCoRoot,
                              nwarps, chunkSlices, params, st);
  }

  // =========================================================================
  // ROOTSIB path
  // =========================================================================
  else if (isRootSib) {
    const int suParity = rootSib % 2;   // rootSib's same-parity peers (to forward)

    if (isSU) {
      ddaDpxRootSibForwardSU(peerScratch, recvbuff, totalBytes,
                             selfRank, nRanks, suParity, chunkSlices,
                             warpsPerPeer, peersPerBlock, params, st);
    } else {
      ddaDpxRootSibReceiveFromRoot(peerScratch, recvbuff, totalBytes,
                                   selfRank, nRanks, root, chunkSlices,
                                   suWarpsPerBlock, siWarpsPerBlock, params, st);
    }

    ddaDpxCoRootCompletionPoll(peerScratch, selfRank, nNonRoot, minCoRoot,
                              nwarps, chunkSlices, params, st);
  }

  // =========================================================================
  // NON-ROOT path
  // =========================================================================
  else {
    const int selfChunkIdx = dpxRankToCol(selfRank, minCoRoot);

    // --- Phase 1: Scatter receive ---
    // Poll own scatter slot for the full chunk (written by one co-root).
    const uint64_t* myScatter =
      reinterpret_cast<const uint64_t*>(peerScratch[selfRank]) + bankWords;

    // --- Phase 2: Fan-out with warp specialization ---
    // SU warps: send own chunk to same-parity non-root peers' gather slots.
    // SI warps: send own chunk to in-package sibling's gather slot.
    //
    // Both warp groups read from the scatter slot (wire format, no re-encode).

    if (isSU) {
      ddaDpxNonRootFanOutSU(peerScratch, recvbuff, totalBytes, myScatter,
                            selfRank, sibling, selfChunkIdx, selfParity,
                            nRanks, chunkSlices, warpsPerPeer, peersPerBlock,
                            params, st);
    } else {
      ddaDpxNonRootRelaySI(peerScratch, recvbuff, totalBytes, myScatter,
                           selfRank, sibling, selfChunkIdx, chunkSlices,
                           suWarpsPerBlock, siWarpsPerBlock, params, st);
    }

    ddaDpxNonRootGather(peerScratch, recvbuff, totalBytes,
                        selfRank, sibling, selfParity, selfChunkIdx,
                        nRanks, nwarps, chunkSlices, params, st);

    ddaDpxNonRootCompletion(peerScratch, selfRank, root, st);
  }

#if (defined(__gfx1250__) || defined(__gfx1250_strict__))
  asm volatile("s_wait_storecnt 0x0" ::: "memory");
#endif
  __syncthreads();
  ddaSetLLEpoch(epochDev, epochLen, flatBlockId, totalBlk, flag32);
}

} // namespace dda::common
