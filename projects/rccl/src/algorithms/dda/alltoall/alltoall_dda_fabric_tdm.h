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
constexpr CachePolicy kDdaA2ATdmPolicy = createCachePolicy(TemporalHint::NT, MemScope::SYS);

// Push the same exchange as ddaAllToAllFabricPush, but each tile moves through TDM: global -> this wave's
// LDS window -> the peer's scratch slot. Items are numbered item = tileIdx * nRanks + k and item k goes to
// peer (selfRank + k) % nRanks. A rank that drains slot s therefore sees the item index its writer used, and
// since the grid depends only on bytesPerRank the same block owns it on both ends, so the per-block barrier
// pairs the write with the read. There is no closing barrier: the drain is local and the next kernel's
// opening barrier covers scratch reuse.
template <int TileBytes, int UnrollPeers>
__global__ void __launch_bounds__(kDdaA2ATdmMaxWarpsPerBlock* kDdaA2ATdmWave)
ddaAllToAllFabricTdmPush(uint8_t* const* __restrict__ peerScratch, uint8_t* __restrict__ recvbuff,
                          const uint8_t* __restrict__ sendbuff, size_t bytesPerRank, int selfRank, int nRanks,
                          FabricGpuBarrier barrier) {
  static_assert(TileBytes * UnrollPeers == kDdaA2ATdmWindowBytes, "tiles must fill one wave window");
  static_assert(TileBytes % kDdaA2ATdmAlign == 0, "tiles must be whole 256 B rows");
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
  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */>();

  for (size_t base = first; base < nItems; base += stride) {
#pragma unroll
    for (int u = 0; u < UnrollPeers; ++u) {
      const size_t item = base + u;
      if (item < nItems) {
        const int peer = (selfRank + static_cast<int>(item % nRanks)) % nRanks;
        const size_t off = (item / nRanks) * TileBytes;
        const size_t n = bytesPerRank - off < TileBytes ? bytesPerRank - off : TileBytes;
        tdm::asyncLoadToLDS<SyncPolicy::Async, kDdaA2ATdmPolicy, /*Aligned=*/true>(
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
        tdm::asyncStoreFromLDS<SyncPolicy::Async, kDdaA2ATdmPolicy, /*Aligned=*/true>(
            window + u * TileBytes, peerScratch[peer] + selfSlot + off, n);
      }
    }
    tdm::tdmWait();
  }

  // tdmWait is per wave. Drain the whole block's stores before any peer is told to read its scratch.
  __syncthreads();
  asm volatile("s_wait_loadcnt 0x0\n\ts_wait_storecnt 0x0" ::: "memory");
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
        tdm::asyncLoadToLDS<SyncPolicy::Async, kDdaA2ATdmPolicy, /*Aligned=*/true>(
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
        tdm::asyncStoreFromLDS<SyncPolicy::Async, kDdaA2ATdmPolicy, /*Aligned=*/true>(
            window + u * TileBytes, recvbuff + static_cast<size_t>(src) * bytesPerRank + off, n);
      }
    }
    tdm::tdmWait();
  }
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

// Peer-major TDM push: the grid is nRanks * blocksPerPeer blocks, and block b works only on peer
// k = b / blocksPerPeer (rank (selfRank + k) % nRanks), so its tiles stream to one remote GPU. Slice
// j = b % blocksPerPeer of the group interleaves over the peer's chunk: the group's waves take UnrollPeers
// consecutive tiles at a time, round-robin. The block drains the slot of rank (selfRank - k) % nRanks, which that
// rank's block b wrote with the same tiles, so the two-rank handshake between block b on the destination and on the
// source pairs the write with the read, as in ddaAllToAllFabricPushPeer. Out of place only: a block drains into
// recvbuff slots that other blocks of this rank may still be reading from sendbuff.
template <int TileBytes, int UnrollPeers>
__global__ void __launch_bounds__(kDdaA2ATdmMaxWarpsPerBlock* kDdaA2ATdmWave)
ddaAllToAllFabricTdmPushPeer(uint8_t* const* __restrict__ peerScratch, uint8_t* __restrict__ recvbuff,
                              const uint8_t* __restrict__ sendbuff, size_t bytesPerRank, int selfRank, int nRanks,
                              int blocksPerPeer, FabricGpuBarrier barrier) {
  static_assert(TileBytes * UnrollPeers == kDdaA2ATdmWindowBytes, "tiles must fill one wave window");
  static_assert(TileBytes % kDdaA2ATdmAlign == 0, "tiles must be whole 256 B rows");
#if TDM_SUPPORTED
  extern __shared__ __attribute__((aligned(128))) uint8_t ddaA2ATdmSmem[];
  // TDM descriptors live in SGPRs, so the wave id that selects the LDS window must be wave-uniform.
  const int warp = __builtin_amdgcn_readfirstlane(static_cast<int>(threadIdx.x / warpSize));
  const size_t warps = blockDim.x / warpSize;
  uint8_t* window = ddaA2ATdmSmem + static_cast<size_t>(warp) * kDdaA2ATdmWindowBytes;

  const int k = static_cast<int>(blockIdx.x) / blocksPerPeer;
  const size_t slice = static_cast<size_t>(blockIdx.x) % blocksPerPeer;
  const size_t nTiles = (bytesPerRank + TileBytes - 1) / TileBytes;
  const size_t first = (slice * warps + warp) * UnrollPeers;
  const size_t stride = static_cast<size_t>(blocksPerPeer) * warps * UnrollPeers;

  const int peer = (selfRank + k) % nRanks;
  const int from = (selfRank - k + nRanks) % nRanks;

  // The previous collective's readers must be done with every scratch slot this block is about to overwrite.
  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */>();

  // Copies this wave's tiles of a bytesPerRank chunk from src to dst through its LDS window.
  auto pass = [&](const uint8_t* src, uint8_t* dst) {
    for (size_t base = first; base < nTiles; base += stride) {
#pragma unroll
      for (int u = 0; u < UnrollPeers; ++u) {
        const size_t t = base + u;
        if (t < nTiles) {
          const size_t off = t * TileBytes;
          const size_t n = bytesPerRank - off < TileBytes ? bytesPerRank - off : TileBytes;
          tdm::asyncLoadToLDS<SyncPolicy::Async, kDdaA2ATdmPolicy, /*Aligned=*/true>(src + off,
                                                                                      window + u * TileBytes, n);
        }
      }
      tdm::tdmWait();
#pragma unroll
      for (int u = 0; u < UnrollPeers; ++u) {
        const size_t t = base + u;
        if (t < nTiles) {
          const size_t off = t * TileBytes;
          const size_t n = bytesPerRank - off < TileBytes ? bytesPerRank - off : TileBytes;
          tdm::asyncStoreFromLDS<SyncPolicy::Async, kDdaA2ATdmPolicy, /*Aligned=*/true>(window + u * TileBytes,
                                                                                         dst + off, n);
        }
      }
      tdm::tdmWait();
    }
  };

  pass(sendbuff + static_cast<size_t>(peer) * bytesPerRank, peerScratch[peer] + static_cast<size_t>(selfRank) * bytesPerRank);

  // Every TDM store above has drained (tdmWait); signalWaitPeer also waits for the wave's stores and syncs the block
  // before thread 0 posts.
  barrier.signalWaitPeer(peer, from, blockIdx.x);

  pass(peerScratch[selfRank] + static_cast<size_t>(from) * bytesPerRank, recvbuff + static_cast<size_t>(from) * bytesPerRank);
#else
  // Only dispatched on gfx1250; every other arch still has to compile this kernel.
  (void)peerScratch;
  (void)recvbuff;
  (void)sendbuff;
  (void)bytesPerRank;
  (void)selfRank;
  (void)nRanks;
  (void)blocksPerPeer;
  (void)barrier;
  __builtin_trap();
#endif
}

} // namespace dda::common
