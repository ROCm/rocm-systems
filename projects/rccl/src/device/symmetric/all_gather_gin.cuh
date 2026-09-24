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

// Resolve the source of a broadcast. `srcSlot` is the LSA slot to read from, or
// -1 for the local buffer, which is also the only form that works when the input
// window is unregistered.
static __device__ __forceinline__ char const* bcastLsaSrc(ncclSymPtr<char> input, int srcSlot) {
  return srcSlot < 0 ? input.localPtr() : input.lsaPtr(srcSlot);
}

// Warp-blocked broadcast over whole tiles of UnrollPacks*WARP_SIZE packs. The
// source reads are batched into registers so they overlap, then replayed to all
// peers; the next tile's loads are issued before the back edge.
template <int BytePerPack, int UnrollPacks, int UnrollPeers>
static __device__ void bcastLsaDeep(int tn, int t, ncclSymPtr<char> input, ncclSymPtr<char> output, ncclTeam team,
                                    int srcSlot, int slot0, int selfSkip, int nIters) {
  using Pack = BytePack<BytePerPack>;
  int wn = tn / WARP_SIZE;
  int w = t / WARP_SIZE;
  int lane = t % WARP_SIZE;

  Pack const* inpPacks = (Pack const*)bcastLsaSrc(input, srcSlot) + intptr_t(w) * UnrollPacks * WARP_SIZE + lane;
  ncclSymPtr<Pack> outPacks = (ncclSymPtr<Pack>)output + intptr_t(w) * UnrollPacks * WARP_SIZE + lane;
  ncclLsaPointerGetter<Pack> getDst{outPacks, slot0};
  intptr_t cursor = 0;
  Pack tmp[UnrollPacks];

  nIters -= w;
  if (0 < nIters) {
    NVCC_PRAGMA_UNROLL_AUTO
    for (int u = 0; u < UnrollPacks; u++) tmp[u] = inpPacks[u * WARP_SIZE];

    while (true) {
      bcastPacksToLsa<UnrollPacks, UnrollPeers>(getDst, cursor, WARP_SIZE, tmp, team.nRanks, team.rank, selfSkip);
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
static __device__ void bcastLsaPacks(int tn, int t, ncclSymPtr<char> input, ncclSymPtr<char> output, ncclTeam team,
                                     int srcSlot, int slot0, int selfSkip, size_t nPacks) {
  using Pack = BytePack<BytePerPack>;
  Pack const* inpPacks = (Pack const*)bcastLsaSrc(input, srcSlot);
  ncclLsaPointerGetter<Pack> getDst{(ncclSymPtr<Pack>)output, slot0};
  NVCC_PRAGMA_UNROLL_DISABLED
  for (size_t i = t; i < nPacks; i += tn) {
    Pack tmp[1];
    tmp[0] = inpPacks[i];
    bcastPacksToLsa<1, UnrollPeers>(getDst, (intptr_t)i, 1, tmp, team.nRanks, team.rank, selfSkip);
  }
}

// Unaligned head and ragged tail, one byte at a time.
template <int UnrollPeers>
static __device__ void bcastLsaEnds(int tn, int t, ncclSymPtr<char> input, ncclSymPtr<char> output, ncclTeam team,
                                    int srcSlot, int slot0, int selfSkip, size_t nBytes, uint32_t nPreBytes,
                                    size_t nSufBytes) {
  using Pack = BytePack<1>;
  Pack const* inpPacks = (Pack const*)bcastLsaSrc(input, srcSlot);
  ncclLsaPointerGetter<Pack> getDst{(ncclSymPtr<Pack>)output, slot0};
  NVCC_PRAGMA_UNROLL_DISABLED
  for (size_t i = t; i < nPreBytes + nSufBytes; i += tn) {
    size_t elt = i < nPreBytes ? i : nBytes - nPreBytes - nSufBytes + i;
    Pack tmp[1];
    tmp[0] = inpPacks[elt];
    bcastPacksToLsa<1, UnrollPeers>(getDst, (intptr_t)elt, 1, tmp, team.nRanks, team.rank, selfSkip);
  }
}

template <typename T>
static __device__ void bcastLsa(ncclSymkArgsHandler& handler, int tn, int t, ncclSymPtr<T> input,
                                ncclSymPtr<T> output, size_t nElts, BoolTag</*multimem=*/true>, ncclTeam, int, int) {
  bcastMultimem(handler, tn, t, input, output, nElts);
}

// Replay `input` into `output` on every rank of `team`, which must be a unit-stride
// sub-team of the LSA team based at LSA slot `slot0`. `srcSlot` selects the LSA slot
// the source is read from, or -1 for the local buffer.
template <typename T>
static __device__ void bcastLsa(ncclSymkArgsHandler& handler, int tn, int t, ncclSymPtr<T> input,
                                ncclSymPtr<T> output, size_t nElts, BoolTag</*multimem=*/false>, ncclTeam team,
                                int srcSlot, int slot0) {
    // When the chunk already landed locally the self store is redundant, and it
    // races with the ring warp relaying that same chunk over GIN. Pulling from a
    // peer slot is never redundant, even though source and destination name the
    // same offset, because the source is another rank's copy of it.
  int selfSkip = (srcSlot < 0 && input == output) ? 1 : 0;
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
                                                          (ncclSymPtr<char>)output + cursor, team, srcSlot, slot0,
                                                          selfSkip, (int)tiles);
      cursor += tiles * BytePerTile;
    }
    size_t packs = (nBytes - cursor) / BytePerPack;
    if (packs != 0) {
      bcastLsaPacks<BytePerPack, /*UnrollPeers=*/4>(tn, t, (ncclSymPtr<char>)input + cursor,
                                                    (ncclSymPtr<char>)output + cursor, team, srcSlot, slot0, selfSkip,
                                                    packs);
      cursor += packs * BytePerPack;
    }
  }

  if (alignment % 4 == 0) {
    constexpr int BytePerPack = 4;
    size_t packs = (nBytes - cursor) / BytePerPack;
    if (packs != 0) {
      bcastLsaPacks<BytePerPack, /*UnrollPeers=*/4>(tn, t, (ncclSymPtr<char>)input + cursor,
                                                    (ncclSymPtr<char>)output + cursor, team, srcSlot, slot0, selfSkip,
                                                    packs);
      cursor += packs * BytePerPack;
    }
  }

  bcastLsaEnds</*UnrollPeers=*/8>(tn, t, (ncclSymPtr<char>)input, (ncclSymPtr<char>)output, team, srcSlot, slot0,
                                  selfSkip, nBytes, nPreBytes, nBytes - cursor);
}

template <bool multimem>
static __device__ void agAlgoHier(ncclSymkDevWorkArgs const* args, BoolTag<multimem> multimemTag) {
  ncclCoopCta cta;
  ncclSymkArgsHandler handler(args);
  ncclTeam rail = ncclTeamRail(handler.comm);
  ncclTeam lsa = ncclTeamLsa(handler.comm);
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
                     output + dgrank * nAllElts + offset, chunkElts, multimemTag, lsa, /*srcSlot=*/-1, /*slot0=*/0);
            offset += chunkElts;
            remainingElts -= chunkElts;
          }
        } else {
          while (remainingElts) {
            size_t chunkElts = min(remainingElts, size_t(chunkSize));
              // Wait for signal from other peers before putting their data
            gin.waitSignal(warps, railSignals + prevPeer, localSignalValue + 1, 32);
            bcastLsa(handler, warps.num_threads(), warps.thread_rank(), output + dgrank * nAllElts + offset,
                     output + dgrank * nAllElts + offset, chunkElts, multimemTag, lsa, /*srcSlot=*/-1, /*slot0=*/0);
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

// Two-level load/store AllGather for an LSA team spanning several packages joined by
// narrow links, e.g. MI300X in CPX mode where each run of S slots is the XCDs of one
// card. The team factors into a scale-in team (peers reachable over the on-package
// fabric) and a scale-up team (the matching slot on each other package).
//
// Scale-in first: each rank stores its own contribution into its own output slot on
// every same-package peer, so afterwards a package holds the slots of all its own
// ranks. Scale-up second: each rank pulls one slot per remote package -- always the
// one at its own lane -- and replays each to its same-package peers. Lanes cover
// disjoint slots, so a package collects every remote slot exactly once and the links
// carry one slot per rank instead of one per cross-package pair, which is what makes
// the flat kernel collapse.
//
// Scale-up has to read the partner's *output* slot rather than its input: input
// offsets are not symmetric across ranks, since in-place AllGather puts each rank's
// input at rank*nAllElts inside the output window. That is what the middle barrier
// pays for -- it publishes the scale-in phase before anyone reads it.
__device__ __forceinline__ void ncclSymkRun_AllGather_HierLsa(ncclSymkDevWorkArgs const* args) {
  ncclSymkArgsHandler handler{args};
  ncclLsaBarrierSession<ncclCoopCta> bar{ncclCoopCta(), handler.comm, ncclTeamTagLsa(), blockIdx.x};

  ncclTeam lsa = ncclTeamLsa(handler.comm);
  ncclTeam scaleIn = ncclTeamInnerFactor(lsa, handler.hierScaleInSize);
  ncclTeam scaleUp = ncclTeamOuterFactor(lsa, handler.hierScaleInSize);
  int slot0 = ncclTeamRankToLsa(handler.comm, scaleIn, 0);
  int const& rank = handler.comm.rank;

  bar.sync(ncclCoopCta(), cuda::memory_order_acquire);

  handler.forEachWork<char>([&] __device__(int block, int nBlocks, size_t nElts, size_t nAllElts,
                                           ncclSymPtr<char> input, ncclSymPtr<char> output) {
        // Threads numbered over rank.
    int t =
      flattenIx(threadIdx.x % WARP_SIZE, WARP_SIZE, block, nBlocks, threadIdx.x / WARP_SIZE, blockDim.x / WARP_SIZE);
    int tn = nBlocks * blockDim.x;
    bcastLsa(handler, tn, t, input, output + rank * nAllElts, nElts, /*multimem=*/BoolTag<false>{}, scaleIn,
             /*srcSlot=*/-1, slot0);
  });

  bar.sync(ncclCoopCta(), cuda::memory_order_acq_rel);

  handler.forEachWork<char>([&] __device__(int block, int nBlocks, size_t nElts, size_t nAllElts,
                                           ncclSymPtr<char> input, ncclSymPtr<char> output) {
    int t =
      flattenIx(threadIdx.x % WARP_SIZE, WARP_SIZE, block, nBlocks, threadIdx.x / WARP_SIZE, blockDim.x / WARP_SIZE);
    int tn = nBlocks * blockDim.x;
    for (int su = 0; su < scaleUp.nRanks; su++) {
      if (su == scaleUp.rank) continue; // own package landed in the scale-in phase
      int srcSlot = ncclTeamRankToLsa(handler.comm, scaleUp, su);
      ncclSymPtr<char> slot = output + ncclTeamRankToWorld(handler.comm, scaleUp, su) * nAllElts;
      bcastLsa(handler, tn, t, slot, slot, nElts, /*multimem=*/BoolTag<false>{}, scaleIn, srcSlot, slot0);
    }
  });

  bar.sync(ncclCoopCta(), cuda::memory_order_release);
}
