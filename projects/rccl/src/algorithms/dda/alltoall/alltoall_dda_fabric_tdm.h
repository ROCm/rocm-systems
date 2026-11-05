/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#pragma once

#include "algorithms/dda/fabric/fabric_gpu_barrier.h"
#include "tdm/tdmCopy.h"

namespace dda::common {

// gfx1250 is wave32. One wave owns a 16 KiB LDS window and fills it with UnrollPeers tiles before the
// stores go out, so 16 waves of a 512-thread block use 256 KiB of dynamic LDS.
constexpr int kDdaA2ATdmWave = 32;
constexpr int kDdaA2ATdmMaxWarpsPerBlock = 16;
constexpr int kDdaA2ATdmWindowBytes = 16 << 10;
constexpr int kDdaA2ATdmMaxSmemBytes = kDdaA2ATdmMaxWarpsPerBlock * kDdaA2ATdmWindowBytes;
constexpr size_t kDdaA2ATdmAlign = 256;
constexpr CachePolicy kDdaA2ATdmSysPolicy = createCachePolicy(TemporalHint::NT, MemScope::SYS);
constexpr CachePolicy kDdaA2ATdmDevPolicy = createCachePolicy(TemporalHint::NT, MemScope::DEV);

// Push the same exchange as ddaAllToAllFabricPush, but each tile moves through TDM: global -> this wave's
// LDS window -> the peer's scratch slot. Items are numbered item = tileIdx * nRanks + k and item k goes to
// peer (selfRank + k) % nRanks. A rank that drains slot s therefore sees the item index its writer used, and
// since the grid depends only on bytesPerRank the same block owns it on both ends, so the per-block barrier
// pairs the write with the read.
// Out of place only: in place, the drain of item (tileIdx, nRanks - k) overwrites the sendbuff tile that item
// (tileIdx, k) reads, and a tile's nRanks items can sit in different blocks, which that barrier does not order.
template <int TileBytes, int UnrollPeers>
__global__ void __launch_bounds__(kDdaA2ATdmMaxWarpsPerBlock* kDdaA2ATdmWave)
ddaAllToAllFabricTdmPush(uint8_t* const* __restrict__ peerScratch, uint8_t* __restrict__ recvbuff,
                          const uint8_t* __restrict__ sendbuff, size_t bytesPerRank, int selfRank, int nRanks,
                          FabricGpuBarrier barrier) {
  static_assert(TileBytes * UnrollPeers == kDdaA2ATdmWindowBytes, "tiles must fill one wave window");
  static_assert(TileBytes % kDdaA2ATdmAlign == 0, "tiles must be whole 256 B rows");
  static_assert(UnrollPeers < 10, "tdm unroll must be less than 10");
#if TDM_SUPPORTED
  extern __shared__ __attribute__((aligned(128))) uint8_t ddaA2ATdmSmem[];
  // TDM descriptors live in SGPRs, so the wave id that selects the LDS window must be wave-uniform.
  const int warp = __builtin_amdgcn_readfirstlane(static_cast<int>(threadIdx.x / warpSize));
  const size_t warps = blockDim.x / warpSize;
  uint8_t* window = ddaA2ATdmSmem + static_cast<size_t>(warp) * kDdaA2ATdmWindowBytes;

  const size_t tilesPerPeer = (bytesPerRank + TileBytes - 1) / TileBytes;
  const size_t nItems = tilesPerPeer * static_cast<size_t>(nRanks);
  const size_t stride = static_cast<size_t>(gridDim.x) * warps * UnrollPeers;
  const size_t first = (static_cast<size_t>(blockIdx.x) * warps + warp) * UnrollPeers;
  const size_t selfSlot = static_cast<size_t>(selfRank) * bytesPerRank;

  // The previous collective's readers must be done with every scratch slot this block is about to overwrite.
  barrier.syncOnSameBlockIdx<false /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */>();

  for (size_t base = first; base < nItems; base += stride) {
#pragma unroll
    for (int u = 0; u < UnrollPeers; ++u) {
      const size_t item = base + u;
      if (item < nItems) {
        const int peer = (selfRank + static_cast<int>(item % nRanks)) % nRanks;
        const size_t off = (item / nRanks) * TileBytes;
        const size_t n = bytesPerRank - off < TileBytes ? bytesPerRank - off : TileBytes;
        tdm::asyncLoadToLDS<SyncPolicy::Async, kDdaA2ATdmDevPolicy, /*Aligned=*/false>(
            sendbuff + static_cast<size_t>(peer) * bytesPerRank + off, window + u * TileBytes, n);
      }
    }
    tdm::tdmWait();
#pragma unroll
    for (int u = 0; u < UnrollPeers; ++u) {
      const size_t item = base + u;
      if (item < nItems) {
        const int peer = (selfRank + static_cast<int>(item % nRanks)) % nRanks;
        const size_t off = (item / nRanks) * TileBytes;
        const size_t n = bytesPerRank - off < TileBytes ? bytesPerRank - off : TileBytes;
        tdm::asyncStoreFromLDS<SyncPolicy::Async, kDdaA2ATdmSysPolicy, /*Aligned=*/false>(
            window + u * TileBytes, peerScratch[peer] + selfSlot + off, n);
      }
    }
    tdm::tdmWait();
  }

  // tdmWait is per wave. Drain the whole block's stores before any peer is told to read its scratch.
  barrier.orderDestWrites();
  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */>();

  const uint8_t* selfScratch = peerScratch[selfRank];
  for (size_t base = first; base < nItems; base += stride) {
#pragma unroll
    for (int u = 0; u < UnrollPeers; ++u) {
      const size_t item = base + u;
      if (item < nItems) {
        const int src = (selfRank - static_cast<int>(item % nRanks) + nRanks) % nRanks;
        const size_t off = (item / nRanks) * TileBytes;
        const size_t n = bytesPerRank - off < TileBytes ? bytesPerRank - off : TileBytes;
        tdm::asyncLoadToLDS<SyncPolicy::Async, kDdaA2ATdmSysPolicy, /*Aligned=*/false>(
            selfScratch + static_cast<size_t>(src) * bytesPerRank + off, window + u * TileBytes, n);
      }
    }
    tdm::tdmWait();
#pragma unroll
    for (int u = 0; u < UnrollPeers; ++u) {
      const size_t item = base + u;
      if (item < nItems) {
        const int src = (selfRank - static_cast<int>(item % nRanks) + nRanks) % nRanks;
        const size_t off = (item / nRanks) * TileBytes;
        const size_t n = bytesPerRank - off < TileBytes ? bytesPerRank - off : TileBytes;
        tdm::asyncStoreFromLDS<SyncPolicy::Async, kDdaA2ATdmDevPolicy, /*Aligned=*/false>(
            window + u * TileBytes, recvbuff + static_cast<size_t>(src) * bytesPerRank + off, n);
      }
    }
    tdm::tdmWait();
  }

  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, false /* hasSubsequentMemAccess */>();
#else
  // Only dispatched on gfx1250; every other arch still has to compile this kernel.
  (void)peerScratch;
  (void)recvbuff;
  (void)sendbuff;
  (void)bytesPerRank;
  (void)selfRank;
  (void)nRanks;
  (void)barrier;
  __builtin_trap();
#endif
}

#if TDM_SUPPORTED
// Copies, from src to dst, the tile groups that start at begin, begin + stride, ... below limit (UnrollPeers tiles per
// group, clipped to nTiles), staging each group through this wave's LDS window. Load and Store are the cache policies
// of the TDM load and store.
template <CachePolicy Load, CachePolicy Store, int TileBytes, int UnrollPeers>
__device__ __forceinline__ void ddaA2ATdmPeerCopy(uint8_t* window, uint8_t* dst, const uint8_t* src, size_t begin,
                                                  size_t limit, size_t stride, size_t nTiles, size_t bytesPerRank) {
  for (size_t base = begin; base < nTiles && base < limit; base += stride) {
#pragma unroll
    for (int u = 0; u < UnrollPeers; ++u) {
      const size_t t = base + u;
      if (t < nTiles) {
        const size_t off = t * TileBytes;
        const size_t n = bytesPerRank - off < TileBytes ? bytesPerRank - off : TileBytes;
        tdm::asyncLoadToLDS<SyncPolicy::Async, Load, /*Aligned=*/false>(src + off, window + u * TileBytes, n);
      }
    }
    tdm::tdmWait();
#pragma unroll
    for (int u = 0; u < UnrollPeers; ++u) {
      const size_t t = base + u;
      if (t < nTiles) {
        const size_t off = t * TileBytes;
        const size_t n = bytesPerRank - off < TileBytes ? bytesPerRank - off : TileBytes;
        tdm::asyncStoreFromLDS<SyncPolicy::Async, Store, /*Aligned=*/false>(window + u * TileBytes, dst + off, n);
      }
    }
    tdm::tdmWait();
  }
}
#endif

// Peer-major TDM push: nRanks * blocksPerPeer blocks; block b pushes only to peer (selfRank + k) % nRanks with
// k = b / blocksPerPeer, its waves taking UnrollPeers consecutive tiles round-robin over the chunk, and drains the slot
// of rank (selfRank - k) % nRanks, which that rank's block b wrote with the same tiles. Out of place only.
// The share is cut into pieces (>= 1) of rounds: step p pushes piece p, drains the source's piece p - 1 (postPeer /
// waitPeer of FabricGpuBarrier), then posts p. With drain > 0, drain * (nRanks - 1) extra blocks do the draining (see
// ddaAllToAllFabricPushPeer). Push blocks start with a two-rank handshake with their peer; at exit every block syncs
// with the same block index on all ranks, so the scratch is not reused (e.g. by LL/LL128) before every rank is done.
template <int TileBytes, int UnrollPeers>
__global__ void __launch_bounds__(kDdaA2ATdmMaxWarpsPerBlock* kDdaA2ATdmWave)
ddaAllToAllFabricTdmPushPeer(uint8_t* const* __restrict__ peerScratch, uint8_t* __restrict__ recvbuff,
                              const uint8_t* __restrict__ sendbuff, size_t bytesPerRank, int selfRank, int nRanks,
                              int blocksPerPeer, unsigned pieces, unsigned drain, FabricGpuBarrier barrier) {
  static_assert(TileBytes * UnrollPeers == kDdaA2ATdmWindowBytes, "tiles must fill one wave window");
  static_assert(TileBytes % kDdaA2ATdmAlign == 0, "tiles must be whole 256 B rows");
  static_assert(UnrollPeers < 10, "tdm unroll must be less than 10");
#if TDM_SUPPORTED
  extern __shared__ __attribute__((aligned(128))) uint8_t ddaA2ATdmSmem[];
  // TDM descriptors live in SGPRs, so the wave id that selects the LDS window must be wave-uniform.
  const int warp = __builtin_amdgcn_readfirstlane(static_cast<int>(threadIdx.x / warpSize));
  const size_t warps = blockDim.x / warpSize;
  uint8_t* window = ddaA2ATdmSmem + static_cast<size_t>(warp) * kDdaA2ATdmWindowBytes;

  const size_t nTiles = (bytesPerRank + TileBytes - 1) / TileBytes;
  const size_t stride = static_cast<size_t>(blocksPerPeer) * warps * UnrollPeers;
  const unsigned pushBlocks = static_cast<unsigned>(nRanks * blocksPerPeer);

  // Rounds a wave goes through and the pieces they are cut into. Every wave of every rank derives the same piece
  // count, since it depends only on bytesPerRank, blocksPerPeer and blockDim.
  const size_t nRounds = (nTiles + stride - 1) / stride;
  const size_t perPiece = (nRounds + pieces - 1) / pieces;
  const size_t nPieces = (nRounds + perPiece - 1) / perPiece;

  // Start of this wave's rounds [p * perPiece, (p + 1) * perPiece), counted from tile `first`.
  auto pieceBegin = [&](size_t first, size_t p) { return first + p * perPiece * stride; };

  const uint32_t epoch = barrier.epochBegin();
  if (blockIdx.x >= pushBlocks) {
    // Drain block: waits for the push blocks it serves and copies their pieces out of this rank's scratch.
    const unsigned idx = blockIdx.x - pushBlocks;
    const int r = 1 + static_cast<int>(idx / drain);
    const int fromRank = (selfRank - r + nRanks) % nRanks;
    const size_t fromOff = static_cast<size_t>(fromRank) * bytesPerRank;
    for (size_t p = 0; p < nPieces; ++p) {
      for (unsigned j = idx % drain; j < static_cast<unsigned>(blocksPerPeer); j += drain) {
        barrier.waitPeer(fromRank, epoch, static_cast<unsigned>(p), r * blocksPerPeer + static_cast<int>(j));
        const size_t jFirst = (static_cast<size_t>(j) * warps + warp) * UnrollPeers;
        ddaA2ATdmPeerCopy<kDdaA2ATdmSysPolicy, kDdaA2ATdmDevPolicy, TileBytes, UnrollPeers>(
            window, recvbuff + fromOff, peerScratch[selfRank] + fromOff, pieceBegin(jFirst, p),
            pieceBegin(jFirst, p + 1), stride, nTiles, bytesPerRank);
      }
    }
    barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, false /* hasSubsequentMemAccess */>();
    barrier.epochEnd(epoch, pieces);
    return;
  }

  const int k = static_cast<int>(blockIdx.x) / blocksPerPeer;
  const size_t slice = static_cast<size_t>(blockIdx.x) % blocksPerPeer;
  const size_t first = (slice * warps + warp) * UnrollPeers;

  const int peer = (selfRank + k) % nRanks;
  const int from = (selfRank - k + nRanks) % nRanks;

  // Entry handshake instead of a full barrier: only peer reads the slot this block writes, so wait for peer alone.
  if (k != 0) barrier.signalWait(from, peer, epoch);

  uint8_t* const dst = peerScratch[peer] + static_cast<size_t>(selfRank) * bytesPerRank;
  const uint8_t* const src = sendbuff + static_cast<size_t>(peer) * bytesPerRank;
  const uint8_t* const slot = peerScratch[selfRank] + static_cast<size_t>(from) * bytesPerRank;
  uint8_t* const out = recvbuff + static_cast<size_t>(from) * bytesPerRank;
  // With drain blocks the push blocks (k != 0) only push and post. Group 0 pushes its own chunk into its own scratch
  // and still drains it, since no drain block serves the own slot.
  const bool drainHere = drain == 0 || k == 0;
  // Every TDM store of a piece has drained (tdmWait) before epochPostPiece, which also waits for the wave's stores and
  // syncs the block before thread 0 posts. Every wave polls the source's flag itself.
  for (size_t p = 0; p <= nPieces; ++p) {
    if (p < nPieces) {
      ddaA2ATdmPeerCopy<kDdaA2ATdmDevPolicy, kDdaA2ATdmSysPolicy, TileBytes, UnrollPeers>(
          window, dst, src, pieceBegin(first, p), pieceBegin(first, p + 1), stride, nTiles, bytesPerRank);
    }
    if (p > 0 && drainHere) {
      barrier.waitPeer(from, epoch, static_cast<unsigned>(p - 1));
      ddaA2ATdmPeerCopy<kDdaA2ATdmSysPolicy, kDdaA2ATdmDevPolicy, TileBytes, UnrollPeers>(
          window, out, slot, pieceBegin(first, p - 1), pieceBegin(first, p), stride, nTiles, bytesPerRank);
    }
    if (p < nPieces) barrier.postPeer(peer, epoch, static_cast<unsigned>(p));
  }

  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, false /* hasSubsequentMemAccess */>();
  barrier.epochEnd(epoch, pieces);
#else
  // Only dispatched on gfx1250; every other arch still has to compile this kernel.
  (void)peerScratch;
  (void)recvbuff;
  (void)sendbuff;
  (void)bytesPerRank;
  (void)selfRank;
  (void)nRanks;
  (void)blocksPerPeer;
  (void)pieces;
  (void)drain;
  (void)barrier;
  __builtin_trap();
#endif
}

} // namespace dda::common
