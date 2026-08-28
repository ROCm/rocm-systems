/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "sym_kernels.h"
#if defined(__HIP_PLATFORM_AMD__)
#include "symmetric/kernel.h"
#include "symmetric/primitives.h"
#include "symmetric/gin_scratch__types.h"
#else
#include "kernel.cuh"
#include "primitives.cuh"
#include "gin_scratch__types.h"
#endif

// Replay a register block of packs into every LSA peer's output slot. `getDst` is
// a hoisted pointer getter, so a peer costs address arithmetic rather than a
// reload of the window metadata. Peers are visited starting from `rank` so the
// node's GPUs don't all target the same peer at once.
template <int UnrollPacks, int UnrollPeers, typename Pack, typename GetDst>
static __device__ __forceinline__ void bcastPacksToLsa(GetDst const& getDst, intptr_t cursor, int packStride,
                                                       Pack const (&tmp)[UnrollPacks], int nRanks, int rank,
                                                       int selfSkip) {
  int dr = selfSkip;
  int r = rank + dr;
  if (nRanks <= r) r -= nRanks;
  NVCC_PRAGMA_UNROLL_DISABLED
  for (; dr + UnrollPeers <= nRanks; dr += UnrollPeers) {
    NVCC_PRAGMA_UNROLL(UnrollPeers)
    for (int up = 0; up < UnrollPeers; up++) {
      Pack* dst = getDst(r) + cursor;
      NVCC_PRAGMA_UNROLL(UnrollPacks)
      for (int u = 0; u < UnrollPacks; u++) dst[u * packStride] = tmp[u];
      if (++r == nRanks) r = 0;
    }
  }
  NVCC_PRAGMA_UNROLL(UnrollPeers)
  for (int up = 0; up < UnrollPeers; up++) {
    if (dr + up == nRanks) break;
    Pack* dst = getDst(r) + cursor;
    NVCC_PRAGMA_UNROLL(UnrollPacks)
    for (int u = 0; u < UnrollPacks; u++) dst[u * packStride] = tmp[u];
    if (++r == nRanks) r = 0;
  }
}

// Warp-blocked broadcast over whole tiles of UnrollPacks*WARP_SIZE packs. The
// source reads are batched into registers so they overlap, then replayed to all
// peers; the next tile's loads are issued before the back edge.
template <int BytePerPack, int UnrollPacks, int UnrollPeers>
static __device__ void bcastLsaDeep(int tn, int t, ncclSymPtr<char> input, ncclSymPtr<char> output, ncclTeam lsa,
                                    int selfSkip, int nIters) {
  using Pack = BytePack<BytePerPack>;
  int wn = tn / WARP_SIZE;
  int w = t / WARP_SIZE;
  int lane = t % WARP_SIZE;

  Pack const* inpPacks = (Pack const*)input.localPtr() + intptr_t(w) * UnrollPacks * WARP_SIZE + lane;
  ncclSymPtr<Pack> outPacks = (ncclSymPtr<Pack>)output + intptr_t(w) * UnrollPacks * WARP_SIZE + lane;
  ncclLsaPointerGetter<Pack> getDst{outPacks};
  intptr_t cursor = 0;
  Pack tmp[UnrollPacks];

  nIters -= w;
  if (0 < nIters) {
    NVCC_PRAGMA_UNROLL_AUTO
    for (int u = 0; u < UnrollPacks; u++) tmp[u] = inpPacks[u * WARP_SIZE];

    while (true) {
      bcastPacksToLsa<UnrollPacks, UnrollPeers>(getDst, cursor, WARP_SIZE, tmp, lsa.nRanks, lsa.rank, selfSkip);
      inpPacks += intptr_t(wn) * UnrollPacks * WARP_SIZE;
      cursor += intptr_t(wn) * UnrollPacks * WARP_SIZE;
      nIters -= wn;
      if (nIters <= 0) break;
      NVCC_PRAGMA_UNROLL_AUTO
      for (int u = 0; u < UnrollPacks; u++) tmp[u] = inpPacks[u * WARP_SIZE];
    }
  }
}

// Whole packs left over once the tiled loop can no longer fill a warp tile.
template <int BytePerPack, int UnrollPeers>
static __device__ void bcastLsaPacks(int tn, int t, ncclSymPtr<char> input, ncclSymPtr<char> output, ncclTeam lsa,
                                     int selfSkip, size_t nPacks) {
  using Pack = BytePack<BytePerPack>;
  Pack const* inpPacks = (Pack const*)input.localPtr();
  ncclLsaPointerGetter<Pack> getDst{(ncclSymPtr<Pack>)output};
  NVCC_PRAGMA_UNROLL_DISABLED
  for (size_t i = t; i < nPacks; i += tn) {
    Pack tmp[1];
    tmp[0] = inpPacks[i];
    bcastPacksToLsa<1, UnrollPeers>(getDst, (intptr_t)i, 1, tmp, lsa.nRanks, lsa.rank, selfSkip);
  }
}

// Unaligned head and ragged tail, one byte at a time.
template <int UnrollPeers>
static __device__ void bcastLsaEnds(int tn, int t, ncclSymPtr<char> input, ncclSymPtr<char> output, ncclTeam lsa,
                                    int selfSkip, size_t nBytes, uint32_t nPreBytes, size_t nSufBytes) {
  using Pack = BytePack<1>;
  Pack const* inpPacks = (Pack const*)input.localPtr();
  ncclLsaPointerGetter<Pack> getDst{(ncclSymPtr<Pack>)output};
  NVCC_PRAGMA_UNROLL_DISABLED
  for (size_t i = t; i < nPreBytes + nSufBytes; i += tn) {
    size_t elt = i < nPreBytes ? i : nBytes - nPreBytes - nSufBytes + i;
    Pack tmp[1];
    tmp[0] = inpPacks[elt];
    bcastPacksToLsa<1, UnrollPeers>(getDst, (intptr_t)elt, 1, tmp, lsa.nRanks, lsa.rank, selfSkip);
  }
}

template <typename T>
static __device__ void bcastLsa(ncclSymkArgsHandler& handler, int tn, int t, ncclSymPtr<T> input,
                                ncclSymPtr<T> output, size_t nElts, BoolTag</*multimem=*/true>) {
  bcastMultimem(handler, tn, t, input, output, nElts);
}

template <typename T>
static __device__ void bcastLsa(ncclSymkArgsHandler& handler, int tn, int t, ncclSymPtr<T> input,
                                ncclSymPtr<T> output, size_t nElts, BoolTag</*multimem=*/false>) {
  static_assert(sizeof(T) == 1, "The GIN AllGather drives bcastLsa with byte elements.");
  ncclTeam lsa = ncclTeamLsa(handler.comm);
    // When the chunk already landed locally the self store is redundant, and it
    // races with the ring warp relaying that same chunk over GIN.
  int selfSkip = (input == output) ? 1 : 0;
  size_t nBytes = nElts;

    // Both sides advance together, so one alignment value governs the pack width
    // usable for the whole chunk. The relayed path has input == output, hence 0.
  uint32_t alignment = uint32_t(input.offset - output.offset);
  uint32_t nPreBytes = (16 - input.offset) % 16;
  nPreBytes = min((size_t)nPreBytes, nBytes);
  uintptr_t cursor = nPreBytes;

  if (alignment % 16 == 0) {
    constexpr int BytePerPack = 16, UnrollPacks = 4, UnrollPeers = 2;
    constexpr int BytePerTile = UnrollPacks * WARP_SIZE * BytePerPack;
      // A zero-warp span would never retire a tile, so leave the bytes to the
      // pack loop below.
    size_t tiles = (tn < WARP_SIZE) ? 0 : (nBytes - cursor) / BytePerTile;
    if (tiles != 0) {
      bcastLsaDeep<BytePerPack, UnrollPacks, UnrollPeers>(tn, t, (ncclSymPtr<char>)input + cursor,
                                                          (ncclSymPtr<char>)output + cursor, lsa, selfSkip,
                                                          (int)tiles);
      cursor += tiles * BytePerTile;
    }
    size_t packs = (nBytes - cursor) / BytePerPack;
    if (packs != 0) {
      bcastLsaPacks<BytePerPack, /*UnrollPeers=*/4>(tn, t, (ncclSymPtr<char>)input + cursor,
                                                    (ncclSymPtr<char>)output + cursor, lsa, selfSkip, packs);
      cursor += packs * BytePerPack;
    }
  }

  if (alignment % 4 == 0) {
    constexpr int BytePerPack = 4;
    size_t packs = (nBytes - cursor) / BytePerPack;
    if (packs != 0) {
      bcastLsaPacks<BytePerPack, /*UnrollPeers=*/4>(tn, t, (ncclSymPtr<char>)input + cursor,
                                                    (ncclSymPtr<char>)output + cursor, lsa, selfSkip, packs);
      cursor += packs * BytePerPack;
    }
  }

  bcastLsaEnds</*UnrollPeers=*/8>(tn, t, (ncclSymPtr<char>)input, (ncclSymPtr<char>)output, lsa, selfSkip, nBytes,
                                  nPreBytes, nBytes - cursor);
}

template <bool multimem>
static __device__ void agAlgoHier(ncclSymkDevWorkArgs const* args, BoolTag<multimem> multimemTag) {
  ncclCoopCta cta;
  ncclSymkArgsHandler handler(args);
  ncclTeam rail = ncclTeamRail(handler.comm);
  ncclGin gin(handler.comm, (int)(blockIdx.x % handler.comm.ginContextCount));
  constexpr int chunkSize = ncclSymkAllGather_RailRing_ChunkSize;
  ncclGinSignal_t railSignals = handler.ginSyncHandle.railSignals + blockIdx.x * rail.nRanks;
  ncclBarrierSession<ncclCoopCta> bar(cta, ncclTeamTagWorld(), gin, blockIdx.x, multimem);
  int nextPeer = (rail.rank + 1) % rail.nRanks;
  int prevPeer = (rail.rank + rail.nRanks - 1) % rail.nRanks;
  uint64_t* localSignalPtr = gin.getSignalShadowPtr(railSignals + prevPeer);
  uint64_t localSignalValue = *localSignalPtr;
  const int ringThreads = WARP_SIZE;

    // Zero the AMD software warp-span barrier slots before any coop sync (no-op on NVIDIA).
  ncclCoopNamedBarrierInit();

  bar.sync(cta, cuda::memory_order_acquire, ncclGinFenceLevel::None);

  handler.template forEachWorkNoFusion<uint8_t>([&] __device__(size_t nElts, size_t nAllElts, ncclSymPtr<uint8_t> input,
                                                               ncclSymPtr<uint8_t> output) {
    if (threadIdx.x < ringThreads) {
      ncclCoopWarpSpan warps(0, 1, 0);
      for (int step = 0; step < rail.nRanks - 1; step++) {
        int dataPeer = (rail.rank - step + rail.nRanks) % rail.nRanks;
        int dgrank = ncclTeamRankToWorld(handler.comm, rail, dataPeer);
        size_t remainingElts = nElts;
        size_t offset = 0;
        if (dataPeer == rail.rank) {
          while (remainingElts) {
            size_t chunkElts = min(remainingElts, size_t(chunkSize));
              // Send data chunk to next peer in ring
            gin.put(rail, nextPeer, output + dgrank * nAllElts + offset, input + offset, chunkElts,
                    ncclGin_SignalInc{railSignals + rail.rank}, ncclGin_None{}, warps);
            offset += chunkElts;
            remainingElts -= chunkElts;
          }
        } else {
          while (remainingElts) {
            size_t chunkElts = min(remainingElts, size_t(chunkSize));
              // Wait for ready signal from next peer before sending
            gin.waitSignal(warps, railSignals + prevPeer, localSignalValue + 1, 32);
              // Send data chunk to next peer in ring
            gin.put(rail, nextPeer, output + dgrank * nAllElts + offset, output + dgrank * nAllElts + offset, chunkElts,
                    ncclGin_SignalInc{railSignals + rail.rank}, ncclGin_None{}, warps);
            offset += chunkElts;
            remainingElts -= chunkElts;
            localSignalValue++;
          }
        }
      }
      gin.flush(warps);
    } else {
      ncclCoopWarpSpan warps(1, blockDim.x / WARP_SIZE - 1, 1);
        // Loop through rail ranks starting from itself
      for (int step = 0; step < rail.nRanks; step++) {
        int dataPeer = (rail.rank - step + rail.nRanks) % rail.nRanks;
        int dgrank = ncclTeamRankToWorld(handler.comm, rail, dataPeer);
        size_t remainingElts = nElts;
        size_t offset = 0;
        if (dataPeer == rail.rank) {
          while (remainingElts) {
            size_t chunkElts = min(remainingElts, size_t(chunkSize));
              // Put self rank's data
            bcastLsa(handler, warps.num_threads(), warps.thread_rank(), input + offset,
                     output + dgrank * nAllElts + offset, chunkElts, multimemTag);
            offset += chunkElts;
            remainingElts -= chunkElts;
          }
        } else {
          while (remainingElts) {
            size_t chunkElts = min(remainingElts, size_t(chunkSize));
              // Wait for signal from other peers before putting their data
            gin.waitSignal(warps, railSignals + prevPeer, localSignalValue + 1, 32);
            bcastLsa(handler, warps.num_threads(), warps.thread_rank(), output + dgrank * nAllElts + offset,
                     output + dgrank * nAllElts + offset, chunkElts, multimemTag);
            offset += chunkElts;
            remainingElts -= chunkElts;
            localSignalValue++;
          }
        }
      }
    }
  });

  // update the shadow signal value
  if (threadIdx.x == ringThreads) {
    *localSignalPtr = localSignalValue;
  }
  bar.sync(cta, cuda::memory_order_release, ncclGinFenceLevel::None);
}

__device__ __forceinline__ void ncclSymkRun_AllGather_RailRing_LsaST(struct ncclSymkDevWorkArgs const* args) {
  agAlgoHier(args, /*multimem=*/BoolTag<false>{});
}

__device__ __forceinline__ void ncclSymkRun_AllGather_RailRing_LsaSTMC(struct ncclSymkDevWorkArgs const* args) {
  agAlgoHier(args, /*multimem=*/BoolTag<true>{});
}
