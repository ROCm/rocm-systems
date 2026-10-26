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
#else
#include "kernel.cuh"
#include "primitives.cuh"
#endif

// TileAligned: the caller's tier guarantees every tile address is ncclSymkTileLine-aligned, which
// lets the staging wrappers drop TDM's head peel.
template <int BytePerPack, int UnrollPacks, int UnrollPeers, bool EnableTma, bool TileAligned = false>
static __device__ void bcastDeep(ncclSymkArgsHandler const& handler, int tn, int t, bool waitNeeded,
                                 ncclLsaBarrierSession<ncclCoopCta>& bar, ncclSymPtr<char> input,
                                 ncclSymPtr<char> output, bool inPlace, int nIters, ncclTeam team, int slot0) {
  using Pack = BytePack<BytePerPack>;
  int wn = tn / WARP_SIZE;
  int w = t / WARP_SIZE;
  int lane = t % WARP_SIZE;
  int const& rank = team.rank;
  int const& nRanks = team.nRanks;

  Pack* inpPacks = (Pack*)input.localPtr() + intptr_t(w) * UnrollPacks * WARP_SIZE +
                   (
#if NCCL_SYMK_ASYNC_TILE
                     EnableTma ? 0 :
#endif
                                 lane);

  ncclSymPtr<Pack> outPacks = (ncclSymPtr<Pack>)output + intptr_t(w) * UnrollPacks * WARP_SIZE +
                              (
#if NCCL_SYMK_ASYNC_TILE
                                EnableTma ? 0 :
#endif
                                            lane);
  // Read the window once here: lsaPtr() inside the loop re-reads it after every store.
  ncclLsaPointerGetter<Pack> outLsa(outPacks);

  Pack tmp[UnrollPacks];

#if NCCL_SYMK_ASYNC_TILE
  int lw = threadIdx.x / WARP_SIZE;
  using tmaSmemStruct_t = tmaSmemStruct<Pack, UnrollPacks>;
  tmaSmemStruct_t* tmaSmem = ncclSymkTileSmem<tmaSmemStruct_t>(lw);
  constexpr size_t tileSize = UnrollPacks * WARP_SIZE * BytePerPack;
#endif
  bool skip = false; // all lanes issue loads/stores

#if NCCL_SYMK_ASYNC_TILE
  size_t pending = 0;
  if NCCL_IF_CONSTEXPR (EnableTma) {
    ncclSymkTileBarInit(&tmaSmem->bar, /*arrivers=*/1, lane);
    // TMA issues from lane 0, so the rest of the warp has nothing left to do.
    // TDM needs the whole warp to build the descriptor, so every lane stays in.
    skip = NCCL_SYMK_TILE_TMA && lane != 0;
  }
#endif

  nIters -= w;
  if (0 < nIters) {
#if NCCL_SYMK_ASYNC_TILE
    if NCCL_IF_CONSTEXPR (EnableTma) {
      ncclSymkTileLoad<TileAligned>(tmaSmem->buff[0], inpPacks, tileSize, tmaSmem->bar, pending, lane);
      ncclSymkTileLoadWait</*Arrivers=*/1>(tmaSmem->bar, pending, lane);
    } else
#endif
    {
      NVCC_PRAGMA_UNROLL_AUTO
      for (int u = 0; u < UnrollPacks; u++) {
        tmp[u] = inpPacks[u * WARP_SIZE];
      }
    }
  }

  if (waitNeeded) bar.wait(ncclCoopCta(), cuda::memory_order_acquire);

  if (0 < nIters) {
    while (true) {
      int dr = inPlace ? 1 : 0;
      int r = rank + dr;
      if (r == nRanks) r = 0;
      NVCC_PRAGMA_UNROLL(2)
      for (int partial = 0; partial <= 1 && !skip; partial++) {
        NVCC_PRAGMA_UNROLL_DISABLED
        for (int i = 0; partial ? i < 1 : (dr + UnrollPeers <= nRanks); partial ? i++ : (dr += UnrollPeers)) {
          NVCC_PRAGMA_UNROLL_AUTO
          for (int ur = 0; ur < UnrollPeers - partial; ur++) {
            if (partial && dr + ur == nRanks) break;
#if NCCL_SYMK_ASYNC_TILE
            if NCCL_IF_CONSTEXPR (EnableTma) {
              ncclSymkTileStore<TileAligned>(outLsa(slot0 + r), tmaSmem->buff[0], tileSize, lane);
            } else
#endif
            {
              NVCC_PRAGMA_UNROLL(UnrollPacks)
              for (int u = 0; u < UnrollPacks; u++) {
                outLsa(slot0 + r)[u * WARP_SIZE] = tmp[u];
              }
            }
            if (++r == nRanks) r = 0;
          }
#if NCCL_SYMK_ASYNC_TILE
          if NCCL_IF_CONSTEXPR (EnableTma) {
            ncclSymkTileStoreWait(lane);
          }
#endif
        }
      }
      inpPacks += intptr_t(wn) * UnrollPacks * WARP_SIZE;
      outLsa.base = (Pack*)outLsa.base + intptr_t(wn) * UnrollPacks * WARP_SIZE;
      nIters -= wn;
      if (nIters <= 0) break;
#if NCCL_SYMK_ASYNC_TILE
      if NCCL_IF_CONSTEXPR (EnableTma) {
        ncclSymkTileLoad<TileAligned>(tmaSmem->buff[0], inpPacks, tileSize, tmaSmem->bar, pending, lane);
        ncclSymkTileLoadWait</*Arrivers=*/1>(tmaSmem->bar, pending, lane);
      } else
#endif
      {
        NVCC_PRAGMA_UNROLL_AUTO
        for (int u = 0; u < UnrollPacks; u++) {
          tmp[u] = inpPacks[u * WARP_SIZE];
        }
      }
    }
  }
}

template <int UnrollPeers, typename T>
static __device__ void bcastEnds(ncclSymkArgsHandler const& handler, int tn, int t, ncclSymPtr<T> input,
                                 ncclSymPtr<T> output, bool inPlace, size_t nElts, uint32_t nPreElts, size_t nSufElts,
                                 ncclTeam team, int slot0) {
  int const& rank = team.rank;
  int const& nRanks = team.nRanks;
  BytePack<sizeof(T)>* inpPacks = (BytePack<sizeof(T)>*)input.localPtr();
  ncclLsaPointerGetter<BytePack<sizeof(T)>> outLsa((ncclSymPtr<BytePack<sizeof(T)>>)output);
  NVCC_PRAGMA_UNROLL_DISABLED
  for (size_t i = t; i < nPreElts + nSufElts; i += tn) {
    size_t elt = i < nPreElts ? i : nElts - nPreElts - nSufElts + i;
    BytePack<sizeof(T)> tmp = inpPacks[elt];
    int dr = inPlace ? 1 : 0;
    int r = rank + dr;
    if (r == nRanks) r = 0;
    NVCC_PRAGMA_UNROLL_DISABLED
    for (; dr + UnrollPeers <= nRanks; dr += UnrollPeers) {
      NVCC_PRAGMA_UNROLL(UnrollPeers)
      for (int u = 0; u < UnrollPeers; u++) {
        outLsa(slot0 + r)[elt] = tmp;
        if (++r == nRanks) r = 0;
      }
    }
    NVCC_PRAGMA_UNROLL(UnrollPeers)
    for (int u = 0; u < UnrollPeers; u++) {
      if (dr + u == nRanks) break;
      outLsa(slot0 + r)[elt] = tmp;
      if (++r == nRanks) r = 0;
    }
  }
}

// Copy `input` into `output` on every rank of `team`, which must be a unit-stride sub-team of the
// LSA team, or only on LSA slot `dstSlot` when one is given.
template <typename T, bool EnableTma>
static __device__ void bcast(ncclSymkArgsHandler const& handler, int tn, int t, int nBlocks, bool waitNeeded,
                             ncclLsaBarrierSession<ncclCoopCta>& bar, ncclSymPtr<T> input, ncclSymPtr<T> output,
                             size_t nElts, ncclTeam team, int dstSlot = -1) {
  // A single destination is always written, even when input and output name the same offset.
  bool inPlace = dstSlot < 0 && input == output;
  int slot0 = ncclTeamRankToLsa(handler.comm, team, 0);
  if (dstSlot >= 0) {
    team = ncclTeam{/*nRanks=*/1, /*rank=*/0, /*stride=*/1};
    slot0 = dstSlot;
  }
  size_t nBytes = nElts * sizeof(T);

#if defined(__gfx950__)
  // Engage on a floor instead of trimming, so the partial final wave is kept rather than handed to
  // the per-byte tail. The floor is the old trim modulus, so the deep path engages where it did.
  uint32_t const chunkFloor = uint32_t(nBlocks);
#else
  uint32_t nBlocks_rcp32 = nccl::utility::idivRcp32_upto64(nBlocks);
  uint32_t const chunkFloor = 1;
#endif

  uint32_t alignment = uint32_t(input.offset - output.offset);
  uint32_t nPreBytes =
#if NCCL_SYMK_ASYNC_TILE
    (EnableTma && alignment % 256 == 0) ? (256 - input.offset) % 256 :
#endif
                                          (16 - input.offset) % 16;

  nPreBytes = min((size_t)nPreBytes, nBytes);
  uintptr_t cursor = nPreBytes;

#if NCCL_SYMK_ASYNC_TILE
  if NCCL_IF_CONSTEXPR (EnableTma) {
    if (alignment % 256 == 0) {
      constexpr int BytePerPack = ncclSymkBytePerPack, UnrollPacks = ncclSymkAlign256BDeepUnrollPacks, UnrollPeers = 2;
      constexpr int BytePerChunk = ncclSymkAlign256BDeepBytePerChunk;
      uint32_t chunks = (nBytes - cursor) / BytePerChunk;
      chunks -= imodFast32(chunks, nBlocks, nBlocks_rcp32);
      if (chunks != 0) {
        uintptr_t cursorAfter = cursor + uintptr_t(chunks) * BytePerChunk;
        // Both sides start on a 256 B boundary here (nPreBytes peels input to one, alignment % 256
        // carries output with it, window bases are page-aligned) and every tile advances by a
        // multiple of 256, so the tiles are ncclSymkTileLine-aligned and the peel can go.
        bcastDeep<BytePerPack, UnrollPacks, UnrollPeers, EnableTma, /*TileAligned=*/true>(
          handler, tn, t, waitNeeded, bar, (ncclSymPtr<char>)input + cursor,
          (ncclSymPtr<char>)output + cursor, inPlace, chunks * ncclSymkMinWarpsPerBlock, team, slot0);
        cursor = cursorAfter;
        waitNeeded = false;
      }
    }
  }
#endif

  if (alignment % 16 == 0) {
#if defined(__gfx950__)
    // Dropping to one pack cuts BytePerChunk to a quarter, so mid sizes reach this tier's floor instead
    // of falling to the 4-byte and per-byte paths. One pack per peer lets UnrollPeers batch them all.
    constexpr int UnrollPacks = 1, UnrollPeers = 8;
#else
    constexpr int UnrollPacks = ncclSymkUnrollPacks, UnrollPeers = 2;
#endif
    constexpr int BytePerPack = ncclSymkBytePerPack;
    constexpr int BytePerChunk = ncclSymkGetBytesPerChunk(ncclSymkMinWarpsPerBlock, UnrollPacks);
    uint32_t chunks = (nBytes - cursor) / BytePerChunk;
#if !defined(__gfx950__)
    chunks -= imodFast32(chunks, nBlocks, nBlocks_rcp32);
#endif
    if (chunks >= chunkFloor) {
      uintptr_t cursorAfter = cursor + uintptr_t(chunks) * BytePerChunk;
      bcastDeep<BytePerPack, UnrollPacks, UnrollPeers, EnableTma>(handler, tn, t, waitNeeded, bar,
                                                                  (ncclSymPtr<char>)input + cursor,
                                                                  (ncclSymPtr<char>)output + cursor, inPlace,
                                                                  chunks * ncclSymkMinWarpsPerBlock, team, slot0);
      cursor = cursorAfter;
      waitNeeded = false;
    }
  }

  if (sizeof(T) == 4 || (sizeof(T) < 4 && alignment % 4 == 0)) {
#if defined(__gfx950__)
    // Only reached by 16-byte misaligned buffers, since the tier above shares this chunk size.
    constexpr int UnrollPeers = 8;
#else
    constexpr int UnrollPeers = 4;
#endif
    constexpr int BytePerPack = 4, UnrollPacks = 4;
    constexpr int BytePerChunk = ncclSymkMinWarpsPerBlock * UnrollPacks * WARP_SIZE * BytePerPack;
    uint32_t chunks = (nBytes - cursor) / BytePerChunk;
#if !defined(__gfx950__)
    chunks -= imodFast32(chunks, nBlocks, nBlocks_rcp32);
#endif
    if (chunks >= chunkFloor) {
      uintptr_t cursorAfter = cursor + uintptr_t(chunks) * BytePerChunk;
      bcastDeep<(sizeof(T) <= BytePerPack ? BytePerPack : 0), UnrollPacks, UnrollPeers, false>(
        handler, tn, t, waitNeeded, bar, (ncclSymPtr<char>)input + cursor, (ncclSymPtr<char>)output + cursor, inPlace,
        chunks * ncclSymkMinWarpsPerBlock, team, slot0);
      cursor = cursorAfter;
      waitNeeded = false;
    }
  }

  if (waitNeeded) bar.wait(ncclCoopCta(), cuda::memory_order_acquire);

  constexpr int UnrollPeers = 8;
  size_t nSufElts = (nBytes - cursor) / sizeof(T);
  bcastEnds<UnrollPeers>(handler, tn, t, input, output, inPlace, nElts, nPreBytes / sizeof(T), nSufElts, team, slot0);
}

template <bool EnableProfiler, bool EnableTma>
__device__ __forceinline__ void ncclSymkRun_AllGather_ST_impl(ncclSymkDevWorkArgs const* args) {
  ncclSymkArgsHandler handler{args};
  ncclLsaBarrierSession<ncclCoopCta> bar{ncclCoopCta(), handler.comm, ncclTeamTagLsa(), blockIdx.x};
  int const& rank = handler.comm.rank;

  bar.arrive(ncclCoopCta(), cuda::memory_order_relaxed);
  if NCCL_IF_CONSTEXPR (EnableProfiler) {
    // Finish the opening barrier here so AFTER_OPEN marks the end of the peer sync.
    // Same barrier ops as the default variant (which fuses the wait into bcast), so
    // the two stay barrier-compatible when peers disagree on profiling.
    bar.wait(ncclCoopCta(), cuda::memory_order_acquire);
    ncclSymkProfilerPhase(args, NCCL_KERNEL_PHASE_AFTER_OPEN);
  }

  bool waitNeeded = !EnableProfiler;
  handler.forEachWork<char>([&] __device__(int block, int nBlocks, size_t nElts, size_t nAllElts,
                                           ncclSymPtr<char> input, ncclSymPtr<char> output) {
        // Threads numbered over rank.
    int bt =
      flattenIx(threadIdx.x % WARP_SIZE, WARP_SIZE, block, nBlocks, threadIdx.x / WARP_SIZE, blockDim.x / WARP_SIZE);
    int btn = nBlocks * blockDim.x;
    bcast<char, EnableTma>(handler, btn, bt, nBlocks, waitNeeded, bar, input, output + rank * nAllElts, nElts,
                           ncclTeamLsa(handler.comm));
    waitNeeded = false;
  });

  if NCCL_IF_CONSTEXPR (EnableProfiler) ncclSymkProfilerPhase(args, NCCL_KERNEL_PHASE_BEFORE_CLOSE);
  bar.sync(ncclCoopCta(), cuda::memory_order_release);
}

template <bool EnableProfiler>
__device__ __forceinline__ void ncclSymkRun_AllGather_ST(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_AllGather_ST_impl<EnableProfiler, /*EnableTma=*/false>(args);
}

template <bool EnableProfiler>
__device__ __forceinline__ void ncclSymkRun_AllGather_TmaST(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_AllGather_ST_impl<EnableProfiler, /*EnableTma=*/true>(args);
}

// Largest chunk a block pushes.
static constexpr size_t ncclSymkAgHierMaxChunkBytes = 256 << 10;

static __device__ void agHierPush(ncclSymkArgsHandler const& handler, ncclLsaBarrierSession<ncclCoopCta>& bar,
                                  int block, int nBlocks, ncclTeam scaleIn, ncclTeam scaleUp,
                                  ncclDevResourceHandle flagsBuf, uint64_t& seq, size_t nElts, size_t nAllElts,
                                  ncclSymPtr<char> input, ncclSymPtr<char> output) {
  int const& rank = handler.comm.rank;
  int lane = threadIdx.x % WARP_SIZE;
  int warp = threadIdx.x / WARP_SIZE;
  int nBlockWarps = blockDim.x / WARP_SIZE;
  int nGroups = scaleUp.nRanks;
  int nG0Blocks = ncclSymkAgHierG0Blocks(nBlocks, nGroups);
  int nGroupBlocks = ncclSymkAgHierGroupBlocks(nBlocks, nGroups);
  // Units below nRows hold a block of every group, the rest a block of each pushing group.
  int nRows = nGroups * nG0Blocks;
  int nUnits = nG0Blocks + (nGroups - 1) * nGroupBlocks;
  ncclSymPtr<char> mySlot = output + rank * nAllElts;

  // Half of each block pushes and half forwards, threads numbered over the half.
  int nHalfWarps = nBlockWarps / 2;
  bool pusher = warp < nHalfWarps;
  bool forwarder = !pusher && warp < 2 * nHalfWarps;
  int t = (pusher ? warp : warp - nHalfWarps) * WARP_SIZE + lane;
  int tn = nHalfWarps * WARP_SIZE;
  ncclCoopWarpSpan half{/*warp0=*/pusher ? 0 : nHalfWarps, /*nWarps=*/nHalfWarps, /*id=*/pusher ? 0 : 1};

  // Every block of a group pushes the same number of chunks of up to ncclSymkAgHierMaxChunkBytes, each a
  // multiple of ncclSymkBytePerChunk per warp, so bcast()'s 16-byte tier copies it all, split evenly.
  size_t chunkAlign = size_t(nHalfWarps) * ncclSymkBytePerChunk;
  size_t nBlockChunksMax = max(size_t(1), divUp(divUp(nElts, size_t(nGroupBlocks)), ncclSymkAgHierMaxChunkBytes));
  size_t chunkBytes = roundUp(divUp(nElts, nBlockChunksMax * nGroupBlocks), chunkAlign);
  size_t nChunks = chunkBytes == 0 ? 0 : divUp(nElts, chunkBytes);
  size_t flagRow = size_t(blockIdx.x) * nGroups;

  for (int unit = block; unit < nUnits; unit += nBlocks) {
    int group = unit < nRows ? unit % nGroups : 1 + (unit - nRows) % (nGroups - 1);
    int groupBlock = unit < nRows ? unit / nGroups : nG0Blocks + (unit - nRows) / (nGroups - 1);
    if (group == 0) {
      int bt = flattenIx(lane, WARP_SIZE, groupBlock, nG0Blocks, warp, nBlockWarps);
      bcast<char, /*EnableTma=*/false>(handler, nG0Blocks * blockDim.x, bt, /*nBlocks=*/1, /*waitNeeded=*/false, bar,
                                       input, mySlot, nElts, scaleIn);
      continue;
    }

    int to = scaleUp.rank + group;
    if (to >= nGroups) to -= nGroups;
    int from = scaleUp.rank - group;
    if (from < 0) from += nGroups;
    int toSlot = ncclTeamRankToLsa(handler.comm, scaleUp, to);
    ncclSymPtr<char> fromSlot = output + ncclTeamRankToWorld(handler.comm, scaleUp, from) * nAllElts;

    // The block's chunk i is chunk i * nGroupBlocks + groupBlock of the slot.
    int nBlockChunks = groupBlock < nChunks ? int(divUp(nChunks - groupBlock, size_t(nGroupBlocks))) : 0;
    if (pusher) {
      uint64_t* flags = (uint64_t*)ncclGetResourceBufferLsaPointer(handler.comm, flagsBuf, toSlot);
      cuda::atomic_ref<uint64_t> flag(flags[flagRow + group]);
      for (int i = 0; i < nBlockChunks; i++) {
        size_t lo = (size_t(i) * nGroupBlocks + groupBlock) * chunkBytes;
        bcast<char, false>(handler, tn, t, 1, false, bar, input + lo, mySlot + lo, min(chunkBytes, nElts - lo), scaleIn,
                           toSlot);
        // half.sync() orders every pusher warp's stores before this system-scope fence
        half.sync();
        if (t == 0) {
          cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_system);
          flag.store(seq + i + 1, cuda::memory_order_relaxed);
        }
      }
    } else if (forwarder) {
      uint64_t* flags = (uint64_t*)ncclGetResourceBufferLocalPointer(handler.comm, flagsBuf);
      cuda::atomic_ref<uint64_t> flag(flags[flagRow + group]);
      for (int i = 0; i < nBlockChunks; i++) {
        if (t == 0) {
          uint32_t steps = 0;
          while (flag.load(cuda::memory_order_relaxed) < seq + i + 1) {
            if (nccl::utility::testAbort(handler.comm.abortFlag, steps)) break;
          }
          cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_system);
        }
        half.sync();
        size_t lo = (size_t(i) * nGroupBlocks + groupBlock) * chunkBytes;
        bcast<char, false>(handler, tn, t, 1, false, bar, fromSlot + lo, fromSlot + lo, min(chunkBytes, nElts - lo),
                           scaleIn);
      }
    }
    seq += nBlockChunks;
  }
}

// Two-level load/store AllGather for an LSA team over packages of hierScaleInSize consecutive ranks.
// Each rank pushes its slot to its counterpart on every other package, which forwards it to its
// package peers, so each remote slot reaches a package once.
template <bool EnableProfiler>
__device__ __forceinline__ void ncclSymkRun_AllGather_HierLsaST(ncclSymkDevWorkArgs const* args) {
  ncclSymkArgsHandler handler{args};
  ncclLsaBarrierSession<ncclCoopCta> bar{ncclCoopCta(), handler.comm, ncclTeamTagLsa(), blockIdx.x};

  ncclTeam lsa = ncclTeamLsa(handler.comm);
  ncclTeam scaleIn = ncclTeamInnerFactor(lsa, args->kcomm.hierScaleInSize);
  ncclTeam scaleUp = ncclTeamOuterFactor(lsa, args->kcomm.hierScaleInSize);
  // Group 0 raises no flag, so its slot in our row keeps the block's chunk count across launches.
  uint64_t* seqSlot = (uint64_t*)ncclGetResourceBufferLocalPointer(handler.comm, args->kcomm.hierPushFlags) +
                      size_t(blockIdx.x) * scaleUp.nRanks;
  uint64_t seq = *seqSlot;

  ncclCoopNamedBarrierInit();
  bar.sync(ncclCoopCta(), cuda::memory_order_acquire);
  if NCCL_IF_CONSTEXPR (EnableProfiler) ncclSymkProfilerPhase(args, NCCL_KERNEL_PHASE_AFTER_OPEN);

  handler.forEachWork<char>([&] __device__(int block, int nBlocks, size_t nElts, size_t nAllElts,
                                           ncclSymPtr<char> input, ncclSymPtr<char> output) {
    agHierPush(handler, bar, block, nBlocks, scaleIn, scaleUp, args->kcomm.hierPushFlags, seq, nElts, nAllElts, input,
               output);
  });
  if (threadIdx.x == 0) *seqSlot = seq;

  if NCCL_IF_CONSTEXPR (EnableProfiler) ncclSymkProfilerPhase(args, NCCL_KERNEL_PHASE_BEFORE_CLOSE);
  bar.sync(ncclCoopCta(), cuda::memory_order_release);
}

template <bool EnableProfiler, bool EnableTma>
__device__ __forceinline__ void ncclSymkRun_AllGather_STMC_impl(ncclSymkDevWorkArgs const* args) {
  ncclSymkArgsHandler handler{args};
  ncclLsaBarrierSession<ncclCoopCta> bar(ncclCoopCta(), handler.comm, ncclTeamTagLsa(), blockIdx.x, /*multimem=*/true);
  int const& rank = handler.comm.rank;

  bar.sync(ncclCoopCta(), cuda::memory_order_acquire);
  if NCCL_IF_CONSTEXPR (EnableProfiler) ncclSymkProfilerPhase(args, NCCL_KERNEL_PHASE_AFTER_OPEN);

  handler.forEachWork<char>([&] __device__(int block, int nBlocks, size_t nElts, size_t nAllElts,
                                           ncclSymPtr<char> input, ncclSymPtr<char> output) {
    // Round robin memory to blocks.
    int t =
      flattenIx(threadIdx.x % WARP_SIZE, WARP_SIZE, block, nBlocks, threadIdx.x / WARP_SIZE, blockDim.x / WARP_SIZE);
    int tn = nBlocks * blockDim.x;
    bcastMultimem<char, EnableTma>(handler, tn, t, input, output + rank * nAllElts, nElts, rank);
  });

  if NCCL_IF_CONSTEXPR (EnableProfiler) ncclSymkProfilerPhase(args, NCCL_KERNEL_PHASE_BEFORE_CLOSE);
  bar.sync(ncclCoopCta(), cuda::memory_order_release);
}

template <bool EnableProfiler>
__device__ __forceinline__ void ncclSymkRun_AllGather_STMC(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_AllGather_STMC_impl<EnableProfiler, /*EnableTma=*/false>(args);
}

template <bool EnableProfiler>
__device__ __forceinline__ void ncclSymkRun_AllGather_TmaSTMC(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_AllGather_STMC_impl<EnableProfiler, /*EnableTma=*/true>(args);
}

template <bool EnableProfiler, typename EltType>
static __device__ void allgather_LL_body(ncclSymkDevWorkArgs const* args, ncclSymkArgsHandler& handler,
                                         ncclLLA2ASession<ncclCoopCta>& lla2a, EltType* input, EltType* output,
                                         int nElts, int nPacks, int nStrideElts) {
  using Pack = BytePack<8>;
  constexpr int EltPerPack = 8 / sizeof(EltType);
  int const& rank = handler.comm.rank;
  int const& nRanks = handler.comm.nRanks;
  int t = threadIdx.x;
  // The round-downs below mask with -(Unroll * tn), so the launch width must be a power of two.
  int tn = blockDim.x;

  // LL fuses the peer sync into the first epoch, so AFTER_OPEN is stamped once, at the
  // first endEpoch below (see ncclDevProfilerPhases in device.h); BEGIN marks the start.
  [[maybe_unused]] bool profilerPhase1Done = false;
  NVCC_PRAGMA_UNROLL_DISABLED
  while (0 < nElts) {
    int nIterPacks = min(nPacks, tn);
    if (t < nIterPacks) {
      Pack x = loadPack<Pack>(input, t * EltPerPack, nElts);
      lla2a.bcast(/*slot=*/nIterPacks * rank + t, x);
    }

    int tn_div_nPacks = tn / nIterPacks;
    int tn_mod_nPacks = tn % nIterPacks;
    int peer = t / nIterPacks;
    int pack = t % nIterPacks;
#if 1
    // NOTE: Unrolling speedup on eos nranks=8 size=64K: 5.7us vs 6.7us
    constexpr int Unroll = 4;
    NVCC_PRAGMA_UNROLL_DISABLED
    for (int i = t; i < (nRanks * nIterPacks & -(Unroll * tn)); i += Unroll * tn) {
      Pack got[Unroll];
      lla2a.template recvUnrolled<Unroll, Unroll>(i, Unroll, tn, /*&*/ got);
      NVCC_PRAGMA_UNROLL_AUTO
      for (int u = 0; u < Unroll; u++) {
        storePack<Pack>(output + peer * nStrideElts, pack * EltPerPack, nElts, got[u]);
        peer += tn_div_nPacks;
        pack += tn_mod_nPacks;
        if (nIterPacks <= pack) {
          peer += 1;
          pack -= nIterPacks;
        }
      }
    }

    int i = (nRanks * nIterPacks & -(Unroll * tn)) + t;
    int n = (nRanks * nIterPacks) / tn % Unroll;
    if (i + n * tn < nRanks * nIterPacks) n += 1;
    if (n != 0) {
      Pack got[Unroll];
      lla2a.template recvUnrolled<1, Unroll>(i, n, tn, /*&*/ got);
      NVCC_PRAGMA_UNROLL_AUTO
      for (int u = 0; u < Unroll; u++) {
        if (u != 0 && u == n) break;
        storePack(output + peer * nStrideElts, pack * EltPerPack, nElts, got[u]);
        peer += tn_div_nPacks;
        pack += tn_mod_nPacks;
        if (nIterPacks <= pack) {
          peer += 1;
          pack -= nIterPacks;
        }
      }
    }
#else
    // The non-unrolled but "obviously correct" implementation for reference.
    NVCC_PRAGMA_UNROLL_DISABLED
    for (int i = t; i < nRanks * nIterPacks; i += tn) {
      Pack got = lla2a.template recv<Pack>(i);
      storePack(output + peer * nStrideElts, pack * EltPerPack, nElts, got);
      peer += tn_div_nPacks;
      pack += tn_mod_nPacks;
      if (nIterPacks <= pack) {
        peer += 1;
        pack -= nIterPacks;
      }
    }
#endif

    lla2a.endEpoch(ncclCoopCta());
    if NCCL_IF_CONSTEXPR (EnableProfiler) {
      if (!profilerPhase1Done) {
        ncclSymkProfilerPhase(args, NCCL_KERNEL_PHASE_AFTER_OPEN);
        profilerPhase1Done = true;
      }
    }

    input += tn * EltPerPack;
    output += tn * EltPerPack;
    nElts -= tn * EltPerPack;
    nPacks -= tn;
  }
  if NCCL_IF_CONSTEXPR (EnableProfiler) ncclSymkProfilerPhase(args, NCCL_KERNEL_PHASE_BEFORE_CLOSE);
}

template <bool EnableProfiler>
static __device__ void ncclSymkRun_AllGather_LL_impl(ncclSymkDevWorkArgs const* args, bool multimem) {
  ncclSymkArgsHandler handler{args};
  ncclLLA2ASession<ncclCoopCta> lla2a(ncclCoopCta(), handler.comm, ncclTeamLsa(handler.comm), handler.lsaLLA2A,
                                      blockIdx.x, /*maxElts=*/ncclSymkMaxThreads, multimem, handler.comm.lsaMultimem);

  using Pack = BytePack<8>;
  constexpr int BytePerPack = 8;

  handler.singleWork<char>([&] __device__(int nElts, int nAllElts, ncclSymPtr<char> input, ncclSymPtr<char> output) {
    int nPacks = divUp(nElts, BytePerPack);

    char* blockInput = input.localPtr();
    char* blockOutput = output.localPtr();

    uint32_t lowBits = nAllElts;
    lowBits |= (uintptr_t)blockInput;
    lowBits |= (uintptr_t)blockOutput;
    if (__builtin_expect(lowBits % 8 == 0, true)) {
          // NOTE: Specializing for 8-byte alignment in one case help at size=65K: 8.9us vs 5.6us
      allgather_LL_body<EnableProfiler>(args, handler, lla2a, (BytePack<8>*)blockInput, (BytePack<8>*)blockOutput,
                                        nElts / 8, nPacks, nAllElts / 8);
    } else {
      allgather_LL_body<EnableProfiler>(args, handler, lla2a, blockInput, blockOutput, nElts, nPacks, nAllElts);
    }
  });
}

template <bool EnableProfiler>
__device__ __forceinline__ void ncclSymkRun_AllGather_LL(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_AllGather_LL_impl<EnableProfiler>(args, /*multimem=*/false);
}

template <bool EnableProfiler>
__device__ __forceinline__ void ncclSymkRun_AllGather_LLMC(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_AllGather_LL_impl<EnableProfiler>(args, /*multimem=*/true);
}
