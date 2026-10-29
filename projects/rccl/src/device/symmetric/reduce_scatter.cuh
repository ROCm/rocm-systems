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

template <int BytePerPack, int UnrollPacks, int UnrollPeers, typename T, bool EnableTma, typename Red>
static __device__ void reduceDeep(ncclSymkArgsHandler const& handler, int tn, int t, bool waitNeeded,
                                  ncclLsaBarrierSession<ncclCoopCta>& bar, Red red, ncclSymPtr<char> input,
                                  ncclSymPtr<char> output, int32_t nIters) {
  using Pack = BytePack<BytePerPack>;
  using Acc = typename Red::EltType;
  using AccPack = BytePack<BytePerPack * sizeof(Acc) / sizeof(T)>;

  ncclTeam world = ncclTeamWorld(handler.comm);
  int wn = tn / WARP_SIZE;
  int w = t / WARP_SIZE;
  int lane = t % WARP_SIZE;
  int const& rank = handler.comm.rank;
  int const& nRanks = handler.comm.nRanks;
  ncclSymPtr<Pack> inpPacks = (ncclSymPtr<Pack>)input + intptr_t(w) * UnrollPacks * WARP_SIZE +
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

  Pack acc0[UnrollPacks];

#if NCCL_SYMK_ASYNC_TILE
  int lw = threadIdx.x / WARP_SIZE;
  using tmaSmemStruct_t = tmaSmemStruct<Pack, UnrollPacks, UnrollPeers>;
  tmaSmemStruct_t* tmaSmem = ncclSymkTileSmem<tmaSmemStruct_t>(lw);
  constexpr size_t tilePack = UnrollPacks * WARP_SIZE;
  constexpr size_t tileSize = tilePack * BytePerPack;
  size_t tmaSize = 0;

  if NCCL_IF_CONSTEXPR (EnableTma) {
    ncclSymkTileBarInit(&tmaSmem->bar, /*arrivers=*/WARP_SIZE, lane);
  }
#endif

  nIters -= w;
  if (0 < nIters) {
#if NCCL_SYMK_ASYNC_TILE
    if NCCL_IF_CONSTEXPR (EnableTma) {
      ncclSymkTileLoad(tmaSmem->buff[0], inpPacks.peerPtr(world, rank), tileSize, tmaSmem->bar, tmaSize, lane);
    } else
#endif
    {
      NVCC_PRAGMA_UNROLL_AUTO
      for (int u = 0; u < UnrollPacks; u++) {
        acc0[u] = inpPacks.peerPtr(world, rank)[u * WARP_SIZE];
      }
    }
  }

  if (waitNeeded) bar.wait(ncclCoopCta(), cuda::memory_order_acquire);

  if (0 < nIters) {
    while (true) {
      AccPack acc1[UnrollPacks];
      int r = rank + 1;
      if (r == nRanks) r = 0;
      {
        Pack tmp1[UnrollPacks];
#if NCCL_SYMK_ASYNC_TILE
        if NCCL_IF_CONSTEXPR (EnableTma) {
          ncclSymkTileLoad(tmaSmem->buff[1], inpPacks.peerPtr(world, r), tileSize, tmaSmem->bar, tmaSize, lane);
          ncclSymkTileLoadWait</*Arrivers=*/WARP_SIZE>(tmaSmem->bar, tmaSize, lane);
        } else
#endif
        {
          NVCC_PRAGMA_UNROLL_AUTO
          for (int u = 0; u < UnrollPacks; u++) {
            tmp1[u] = inpPacks.peerPtr(world, r)[u * WARP_SIZE];
          }
        }
        NVCC_PRAGMA_UNROLL_AUTO
        for (int u = 0; u < UnrollPacks; u++) {
#if NCCL_SYMK_ASYNC_TILE
          if NCCL_IF_CONSTEXPR (EnableTma) {
            acc0[u] = tmaSmem->buff[0][lane + WARP_SIZE * u];
            tmp1[u] = tmaSmem->buff[1][lane + WARP_SIZE * u];
          }
#endif
          acc1[u] = applyReduce(red, applyCast<T, Acc>(acc0[u]), applyCast<T, Acc>(tmp1[u]));
        }
      }

      r += 1;
      if (r == nRanks) r = 0;

      int dr = 2;
      NVCC_PRAGMA_UNROLL(2)
      for (int partial = 0; partial <= 1; partial++) {
        NVCC_PRAGMA_UNROLL_DISABLED
        for (int i = 0; partial ? i < 1 : (dr + UnrollPeers <= nRanks); partial ? i++ : (dr += UnrollPeers)) {
          if (partial && dr == nRanks) break;

          Pack tmp1[UnrollPeers][UnrollPacks];
#if NCCL_SYMK_ASYNC_TILE
          if NCCL_IF_CONSTEXPR (EnableTma) {
            // The next batch reloads these same tiles, so hold until every lane has
            // finished reducing out of them.
            __syncwarp();
          }
#endif

          NVCC_PRAGMA_UNROLL_AUTO
          for (int ur = 0; ur < UnrollPeers - partial; ur++) {
            if (partial && ur != 0 && dr + ur == nRanks) break;
#if NCCL_SYMK_ASYNC_TILE
            if NCCL_IF_CONSTEXPR (EnableTma) {
              ncclSymkTileLoad(tmaSmem->buff[ur], inpPacks.peerPtr(world, r), tileSize, tmaSmem->bar, tmaSize, lane);
            } else
#endif
            {
              NVCC_PRAGMA_UNROLL(UnrollPacks)
              for (int u = 0; u < UnrollPacks; u++) {
                tmp1[ur][u] = inpPacks.peerPtr(world, r)[u * WARP_SIZE];
              }
            }
            r += 1;
            if (r == nRanks) r = 0;
          }
#if NCCL_SYMK_ASYNC_TILE
          if NCCL_IF_CONSTEXPR (EnableTma) {
            ncclSymkTileLoadWait</*Arrivers=*/WARP_SIZE>(tmaSmem->bar, tmaSize, lane);
          }
#endif
          NVCC_PRAGMA_UNROLL_AUTO
          for (int ur = 0; ur < UnrollPeers - partial; ur++) {
            if (partial && ur != 0 && dr + ur == nRanks) break;
            NVCC_PRAGMA_UNROLL(UnrollPacks)
            for (int u = 0; u < UnrollPacks; u++) {
#if NCCL_SYMK_ASYNC_TILE
              if NCCL_IF_CONSTEXPR (EnableTma) {
                tmp1[ur][u] = tmaSmem->buff[ur][lane + WARP_SIZE * u];
              }
#endif
              acc1[u] = applyReduce(red, acc1[u], applyCast<T, Acc>(tmp1[ur][u]));
            }
          }
        }
      }

      NVCC_PRAGMA_UNROLL_AUTO
      for (int u = 0; u < UnrollPacks; u++) {
        acc1[u] = applyPostOp(red, acc1[u]);
      }

      NVCC_PRAGMA_UNROLL_AUTO
      for (int u = 0; u < UnrollPacks; u++) {
#if NCCL_SYMK_ASYNC_TILE
        if NCCL_IF_CONSTEXPR (EnableTma) {
          tmaSmem->buff[0][lane + WARP_SIZE * u] = applyCast<Acc, T>(acc1[u]);
        } else
#endif
        {
          acc0[u] = applyCast<Acc, T>(acc1[u]);
        }
      }
#if NCCL_SYMK_ASYNC_TILE
      if NCCL_IF_CONSTEXPR (EnableTma) {
        // Publish the lanes' reduced tile to the engine that is about to read it.
        ncclSymkTileFenceSmem();
        ncclSymkTileStore(outPacks.localPtr(), tmaSmem->buff[0], tileSize, lane);
        // Drain before the next iteration refills the tile.
        ncclSymkTileStoreWait(lane);
        __syncwarp();
      } else
#endif
      {
        NVCC_PRAGMA_UNROLL(UnrollPacks)
        for (int u = 0; u < UnrollPacks; u++) outPacks.localPtr()[u * WARP_SIZE] = acc0[u];
      }

      inpPacks += intptr_t(wn) * UnrollPacks * WARP_SIZE;
      outPacks += intptr_t(wn) * UnrollPacks * WARP_SIZE;
      nIters -= wn;
      if (nIters <= 0) break;

#if NCCL_SYMK_ASYNC_TILE
      if NCCL_IF_CONSTEXPR (EnableTma) {
        ncclSymkTileLoad(tmaSmem->buff[0], inpPacks.peerPtr(world, rank), tileSize, tmaSmem->bar, tmaSize, lane);
      } else
#endif
      {
        // Load data for next iteration.
        NVCC_PRAGMA_UNROLL_AUTO
        for (int u = 0; u < UnrollPacks; u++) {
          acc0[u] = inpPacks.peerPtr(world, rank)[u * WARP_SIZE];
        }
      }
    }
  }
}

template <int UnrollPeers, typename Red, typename T>
static __device__ void reduceEnds(ncclSymkArgsHandler const& handler, int tn, int t, Red red, ncclSymPtr<T> input,
                                  ncclSymPtr<T> output, size_t nElts, uint32_t nPreElts, size_t nSufElts) {
  using Acc = typename Red::EltType;

  ncclTeam world = ncclTeamWorld(handler.comm);
  int const& rank = handler.comm.rank;
  int const& nRanks = handler.comm.nRanks;

  ncclSymPtr<BytePack<sizeof(T)>> inpPacks = (ncclSymPtr<BytePack<sizeof(T)>>)input;
  ncclSymPtr<BytePack<sizeof(T)>> outPacks = (ncclSymPtr<BytePack<sizeof(T)>>)output;
  NVCC_PRAGMA_UNROLL_DISABLED
  for (size_t i = t; i < nPreElts + nSufElts; i += tn) {
    size_t elt = i < nPreElts ? i : nElts - nSufElts - nPreElts + i;
    BytePack<sizeof(T)> acc0 = inpPacks.peerPtr(world, rank)[elt];
    BytePack<sizeof(Acc)> acc1;
    BytePack<sizeof(T)> tmp[UnrollPeers];
    int dr = 1;
    int r = rank + 1;
    if (nRanks == r) r = 0;
    bool first = true;

    NVCC_PRAGMA_UNROLL(2)
    for (int partial = 0; partial <= 1; partial++) {
      NVCC_PRAGMA_UNROLL_DISABLED
      for (int j = 0; partial ? j < 1 : (dr + UnrollPeers <= nRanks); partial ? j++ : (dr += UnrollPeers)) {
        if (partial && dr == nRanks) break;

        NVCC_PRAGMA_UNROLL_AUTO
        for (int u = 0; u < UnrollPeers - partial; u++) {
          if (partial && u != 0 && dr + u == nRanks) break;
          tmp[u] = inpPacks.peerPtr(world, r)[elt];
          r += 1;
          if (r == nRanks) r = 0;
        }
        if (first) {
          first = false;
          acc1 = applyCast<T, Acc>(acc0);
        }
        NVCC_PRAGMA_UNROLL_AUTO
        for (int u = 0; u < UnrollPeers - partial; u++) {
          if (partial && u != 0 && dr + u == nRanks) break;
          acc1 = applyReduce(red, acc1, applyCast<T, Acc>(tmp[u]));
        }
      }
    }

    acc1 = applyPostOp(red, acc1);
    acc0 = applyCast<Acc, T>(acc1);
    outPacks.localPtr()[elt] = acc0;
  }
}

template <typename Red, typename T, bool EnableTma>
static __device__ void reduce(ncclSymkArgsHandler const& handler, int tn, int t, int nBlocks, bool waitNeeded,
                              ncclLsaBarrierSession<ncclCoopCta>& bar, Red red, ncclSymPtr<T> input,
                              ncclSymPtr<T> output, size_t nElts) {
  int const& nRanks = handler.comm.nRanks;

#if defined(__gfx950__)
  // Engage on a floor instead of trimming, so the partial final wave is kept rather than handed to
  // the per-element tail. The floor is the old trim modulus, so the deep path engages where it did.
  uint32_t const chunkFloor = uint32_t(nRanks * nBlocks);
#else
  int const& nRanks_rcp32 = handler.nRanks_rcp32;
  uint32_t nBlocks_rcp32 = nccl::utility::idivRcp32_upto64(nBlocks);
  uint32_t nRanks_nBlocks_rcp32 = nccl::utility::imulRcp32(nRanks, nRanks_rcp32, nBlocks, nBlocks_rcp32);
  uint32_t const chunkFloor = 1;
#endif

  // True only where the deep loop really stages through a TDM engine.
  constexpr bool AsyncTile = ncclSymkAsyncTile && EnableTma;

  uint32_t alignment = uint32_t(input.offset - output.offset);
  size_t nBytes = nElts * sizeof(T);

  uint32_t nPreBytes = (16u - input.offset) % 16u;
  nPreBytes = min((size_t)nPreBytes, nBytes);
  uintptr_t cursor = nPreBytes;

  if (alignment % 16 == 0) {
#if defined(__gfx950__)
    // Dropping to one pack cuts BytePerChunk to a quarter, which brings 16 MB within this tier's
    // floor instead of falling through to 4-byte packs, and doubles iterations per warp at 32 MB.
    constexpr int UnrollPacksPlain = 1, UnrollPeers = 8;
#else
    constexpr int UnrollPacksPlain = ncclSymkUnrollPacks, UnrollPeers = 2;
#endif
    constexpr int BytePerPack = ncclSymkBytePerPack,
                  UnrollPacks = AsyncTile ? ncclSymkDeepUnrollPacks(sizeof(T)) : UnrollPacksPlain;

    // Derived from UnrollPacks so the two cannot disagree: a chunk wider than what reduceDeep()
    // reduces would leave the difference unreduced.
    constexpr int BytePerChunk = ncclSymkGetBytesPerChunk(ncclSymkMinWarpsPerBlock, UnrollPacks);
    uint32_t chunks = (nBytes - cursor) / BytePerChunk;
#if !defined(__gfx950__)
    chunks -= imodFast32(chunks, nRanks * nBlocks, nRanks_nBlocks_rcp32);
#endif
    if (chunks >= chunkFloor) {
      uintptr_t cursorAfter = cursor + uintptr_t(chunks) * BytePerChunk;
      reduceDeep<BytePerPack, UnrollPacks, UnrollPeers, T, EnableTma>(handler, tn, t, waitNeeded, bar, red,
                                                                      (ncclSymPtr<char>)input + cursor,
                                                                      (ncclSymPtr<char>)output + cursor,
                                                                      chunks * ncclSymkMinWarpsPerBlock);
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
    chunks -= imodFast32(chunks, nRanks * nBlocks, nRanks_nBlocks_rcp32);
#endif
    if (chunks >= chunkFloor) {
      uintptr_t cursorAfter = cursor + uintptr_t(chunks) * BytePerChunk;
      reduceDeep<(sizeof(T) <= BytePerPack ? BytePerPack : 0), UnrollPacks, UnrollPeers, T, false>(
        handler, tn, t, waitNeeded, bar, red, (ncclSymPtr<char>)input + cursor, (ncclSymPtr<char>)output + cursor,
        chunks * ncclSymkMinWarpsPerBlock);
      cursor = cursorAfter;
      waitNeeded = false;
    }
  }

  if (waitNeeded) bar.wait(ncclCoopCta(), cuda::memory_order_acquire);

  constexpr int UnrollPeers = 8;
  size_t nSufElts = (nBytes - cursor) / sizeof(T);
  reduceEnds<UnrollPeers>(handler, tn, t, red, input, output, nElts, nPreBytes / sizeof(T), nSufElts);
}

template <bool EnableProfiler, template <typename> typename Red, typename T, bool EnableTma>
__device__ __forceinline__ void ncclSymkRun_ReduceScatter_LD_impl(ncclSymkDevWorkArgs const* args) {
  ncclSymkArgsHandler handler{args};
  ncclLsaBarrierSession<ncclCoopCta> bar{ncclCoopCta(), handler.comm, ncclTeamTagLsa(), blockIdx.x};
  Red<typename ncclSymkAccumType<Red, T, /*nvls=*/false>::Type> red(handler.devWork->redOpArg);
  int const& rank = handler.comm.rank;

  bar.arrive(ncclCoopCta(), cuda::memory_order_relaxed);
  if NCCL_IF_CONSTEXPR (EnableProfiler) {
    // Finish the opening barrier here so AFTER_OPEN marks the end of the peer sync.
    // Same barrier ops as the default variant (which fuses the wait into reduce),
    // so the two stay barrier-compatible when peers disagree on profiling.
    bar.wait(ncclCoopCta(), cuda::memory_order_acquire);
    ncclSymkProfilerPhase(args, NCCL_KERNEL_PHASE_AFTER_OPEN);
  }

  bool waitNeeded = !EnableProfiler;
  handler.forEachWork<T>([&] __device__(int block, int nBlocks, size_t nElts, size_t nAllElts, ncclSymPtr<T> input,
                                        ncclSymPtr<T> output) {
        // Round robin warps over blocks.
    int t =
      flattenIx(threadIdx.x % WARP_SIZE, WARP_SIZE, block, nBlocks, threadIdx.x / WARP_SIZE, blockDim.x / WARP_SIZE);
    int tn = nBlocks * blockDim.x;

    reduce<decltype(red), T, EnableTma>(handler, tn, t, nBlocks, waitNeeded, bar, red, input + rank * nAllElts, output,
                                        nElts);

    waitNeeded = false;
  });

  if NCCL_IF_CONSTEXPR (EnableProfiler) ncclSymkProfilerPhase(args, NCCL_KERNEL_PHASE_BEFORE_CLOSE);
  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed);
}

template <bool EnableProfiler, template <typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_ReduceScatter_LD(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_ReduceScatter_LD_impl<EnableProfiler, Red, T, /*EnableTma=*/false>(args);
}

template <bool EnableProfiler, template <typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_ReduceScatter_TmaLD(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_ReduceScatter_LD_impl<EnableProfiler, Red, T, /*EnableTma=*/true>(args);
}

// Pack loads and stores through the global aperture, so p must point to device memory. Plain accesses
// through generic pointers compile to flat instructions, which the compiler must assume may reach LDS.
template <int Size>
static __device__ __forceinline__ BytePack<Size> rsHierLoad(BytePack<Size> const* p) {
  BytePack<Size> v;
#if defined(__HIP_PLATFORM_AMD__)
  if constexpr (Size == 16) {
    v.v4u = *(__attribute__((address_space(1))) v4u const*)p;
  } else {
    v.native = *(__attribute__((address_space(1))) decltype(v.native) const*)p;
  }
#else
  v = *p;
#endif
  return v;
}

template <int Size>
static __device__ __forceinline__ void rsHierStore(BytePack<Size>* p, BytePack<Size> v) {
#if defined(__HIP_PLATFORM_AMD__)
  if constexpr (Size == 16) {
    *(__attribute__((address_space(1))) v4u*)p = v.v4u;
  } else {
    *(__attribute__((address_space(1))) decltype(v.native)*)p = v.native;
  }
#else
  *p = v;
#endif
}

// LSA rank of the package rank i places up from ours, so copy 0 is our own.
static __device__ __forceinline__ int rsHierCopyLsa(ncclDevComm const& comm, ncclTeam scaleIn, int i) {
  int r = scaleIn.rank + i;
  if (r >= scaleIn.nRanks) r -= scaleIn.nRanks;
  return ncclTeamRankToLsa(comm, scaleIn, r);
}

// Adds N of a warp tile's sources, from source first on, into acc, issuing all of their loads before the
// first add so that they overlap. src(i) points at source i's first pack for this lane, and pack u of it
// is u * WARP_SIZE packs further. Init starts acc from source first instead. Only the first nLeft packs
// are in range unless Full.
template <int N, bool Init, bool Full, int UnrollPacks, typename T, typename Pack, typename Red, typename Src,
          typename AccPack>
static __device__ __forceinline__ void rsHierSumGroup(Red red, Src const& src, int first, int nLeft,
                                                      AccPack (&acc)[UnrollPacks]) {
  using Acc = typename Red::EltType;
  Pack tmp[N][UnrollPacks];
  NVCC_PRAGMA_UNROLL_AUTO
  for (int p = 0; p < N; p++) {
    Pack const* ptr = src(first + p);
    NVCC_PRAGMA_UNROLL_AUTO
    for (int u = 0; u < UnrollPacks; u++) {
      if (Full || u * WARP_SIZE < nLeft) tmp[p][u] = rsHierLoad(ptr + u * WARP_SIZE);
    }
  }
#if defined(__HIP_PLATFORM_AMD__)
  // Otherwise the scheduler starts adding before the group's last loads issue and hands those loads the
  // registers the first ones land in, which waits for the first loads before the group is in flight.
  __builtin_amdgcn_sched_barrier(0);
#endif
  NVCC_PRAGMA_UNROLL_AUTO
  for (int p = 0; p < N; p++) {
    NVCC_PRAGMA_UNROLL_AUTO
    for (int u = 0; u < UnrollPacks; u++) {
      AccPack x = applyCast<T, Acc>(tmp[p][u]);
      acc[u] = Init && p == 0 ? x : applyReduce(red, acc[u], x);
    }
  }
}

// Adds a warp tile's sources first through n - 1 into acc in order, with their loads in flight UnrollPeers
// sources at a time and the rest a pair, then one, at a time. Which groups run depends on n, so callers
// start acc with a fixed group: one chosen at run time would have the compiler hoist the loads the
// choices share above the branch and split them from the rest of their group.
template <int UnrollPeers, bool Full, int UnrollPacks, typename T, typename Pack, typename Red, typename Src,
          typename AccPack>
static __device__ __forceinline__ void rsHierSumRest(Red red, Src const& src, int first, int n, int nLeft,
                                                     AccPack (&acc)[UnrollPacks]) {
  static_assert(UnrollPeers >= 2, "Required");
  int i = first;
  NVCC_PRAGMA_UNROLL_DISABLED
  for (; i + UnrollPeers <= n; i += UnrollPeers) {
    rsHierSumGroup<UnrollPeers, /*Init=*/false, Full, UnrollPacks, T, Pack>(red, src, i, nLeft, acc);
  }
  NVCC_PRAGMA_UNROLL_DISABLED
  for (; UnrollPeers > 2 && i + 2 <= n; i += 2) {
    rsHierSumGroup<2, /*Init=*/false, Full, UnrollPacks, T, Pack>(red, src, i, nLeft, acc);
  }
  if (i < n) rsHierSumGroup<1, /*Init=*/false, Full, UnrollPacks, T, Pack>(red, src, i, nLeft, acc);
}

// This block's slots for a round on the given rank of scaleUp, one per other package. Rounds alternate
// between two sets so senders can store a round while the owners' finishers still read the one before.
static __device__ char* rsHierSlots(ncclDevComm const& comm, ncclTeam scaleUp, ncclDevResourceHandle slotsBuf, int peer,
                                    uint64_t round) {
  size_t roundBytes = size_t(scaleUp.nRanks - 1) * ncclSymkRsHierChunkBytes;
  return (char*)ncclGetResourceBufferLsaPointer(comm, slotsBuf, ncclTeamRankToLsa(comm, scaleUp, peer)) +
         (2 * blockIdx.x + (round & 1)) * roundBytes;
}

// One warp tile of a sender's round: for each other package, sums our package's copies of the tile of
// the block its counterpart of us owns and stores that into our slot there. Slot s takes the sums from
// the package s + 1 behind. pack is this lane's first pack of the tile, and nLeft is as for
// rsHierSumGroup. inputLsa is a pointer getter so the window is read before the loops: ncclSymPtr
// re-reads it after every store.
template <bool Full, int UnrollPeers, int UnrollPacks, typename T, typename Pack, typename Red>
static __device__ void rsHierSendTile(ncclDevComm const& comm, ncclTeam scaleIn, ncclTeam scaleUp, Red red,
                                      ncclLsaPointerGetter<Pack> const& inputLsa, size_t nAllElts,
                                      ncclDevResourceHandle slotsBuf, uint64_t round, int pack, int nLeft) {
  using Acc = typename Red::EltType;
  using AccPack = BytePack<sizeof(Pack) * sizeof(Acc) / sizeof(T)>;
  NVCC_PRAGMA_UNROLL_DISABLED
  for (int s = 0; s < scaleUp.nRanks - 1; s++) {
    int to = scaleUp.rank + 1 + s;
    if (to >= scaleUp.nRanks) to -= scaleUp.nRanks;
    ncclLsaPointerGetter<Pack> copies = inputLsa;
    copies.base = (Pack*)((T*)inputLsa.base + ncclTeamRankToWorld(comm, scaleUp, to) * nAllElts) + pack;
    auto copy = [&] __device__(int i) -> Pack const* { return copies(rsHierCopyLsa(comm, scaleIn, i)); };
    AccPack acc[UnrollPacks];
    rsHierSumGroup<2, /*Init=*/true, Full, UnrollPacks, T, Pack>(red, copy, 0, nLeft, acc);
    rsHierSumRest<UnrollPeers, Full, UnrollPacks, T, Pack>(red, copy, 2, scaleIn.nRanks, nLeft, acc);
    Pack* dst = (Pack*)(rsHierSlots(comm, scaleUp, slotsBuf, to, round) + s * ncclSymkRsHierChunkBytes) + pack;
    NVCC_PRAGMA_UNROLL_AUTO
    for (int u = 0; u < UnrollPacks; u++) {
      if (Full || u * WARP_SIZE < nLeft) rsHierStore(dst + u * WARP_SIZE, applyCast<Acc, T>(acc[u]));
    }
  }
}

// A sender's part of a round. Warp tiles go round-robin to the nWarps sending warps, of which ours is
// warp.
template <int BytePerPack, int UnrollPeers, int UnrollPacks, typename T, typename Red>
static __device__ void rsHierSend(ncclDevComm const& comm, ncclTeam scaleIn, ncclTeam scaleUp, Red red, int warp,
                                  int nWarps, ncclSymPtr<T> input, size_t nAllElts, ncclDevResourceHandle slotsBuf,
                                  uint64_t round, int nPacks) {
  using Pack = BytePack<BytePerPack>;
  constexpr int TilePacks = UnrollPacks * WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  ncclLsaPointerGetter<Pack> inputLsa((ncclSymPtr<Pack>)input);
  int tile = warp * TilePacks;
  NVCC_PRAGMA_UNROLL_DISABLED
  for (; tile + TilePacks <= nPacks; tile += nWarps * TilePacks) {
    rsHierSendTile</*Full=*/true, UnrollPeers, UnrollPacks, T>(comm, scaleIn, scaleUp, red, inputLsa, nAllElts,
                                                               slotsBuf, round, tile + lane, 0);
  }
  // Only the last tile of a work range can be partial, which is not worth the code of bigger groups.
  if (tile < nPacks) {
    rsHierSendTile</*Full=*/false, /*UnrollPeers=*/2, UnrollPacks, T>(
      comm, scaleIn, scaleUp, red, inputLsa, nAllElts, slotsBuf, round, tile + lane, nPacks - tile - lane);
  }
}

// One warp tile of a finisher's round: sums our package's copies of the tile of our own block with the
// sums the other packages stored into our slots, and writes that to output.
template <bool Full, int UnrollPeers, int UnrollPacks, typename T, typename Pack, typename Red>
static __device__ void rsHierFinishTile(ncclDevComm const& comm, ncclTeam scaleIn, ncclTeam scaleUp, Red red,
                                        ncclLsaPointerGetter<Pack> const& blockLsa, char const* slots, T* output,
                                        int pack, int nLeft) {
  using Acc = typename Red::EltType;
  using AccPack = BytePack<sizeof(Pack) * sizeof(Acc) / sizeof(T)>;
  constexpr int SlotPacks = ncclSymkRsHierChunkBytes / sizeof(Pack);
  ncclLsaPointerGetter<Pack> copies = blockLsa;
  copies.base = (Pack*)blockLsa.base + pack;
  Pack const* slot0 = (Pack const*)slots + pack;
  auto copy = [&] __device__(int i) -> Pack const* { return copies(rsHierCopyLsa(comm, scaleIn, i)); };
  auto slot = [&] __device__(int s) -> Pack const* { return slot0 + s * SlotPacks; };
  AccPack acc[UnrollPacks];
  // Every package has two ranks or more and there are two packages or more, so every tile has our own
  // copy, one more, and a slot.
  rsHierSumGroup<3, /*Init=*/true, Full, UnrollPacks, T, Pack>(
    red, [&] __device__(int i) -> Pack const* { return i < 2 ? copy(i) : slot(0); }, 0, nLeft, acc);
  rsHierSumRest<UnrollPeers, Full, UnrollPacks, T, Pack>(red, copy, 2, scaleIn.nRanks, nLeft, acc);
  rsHierSumRest<UnrollPeers, Full, UnrollPacks, T, Pack>(red, slot, 1, scaleUp.nRanks - 1, nLeft, acc);
  Pack* dst = (Pack*)output + pack;
  NVCC_PRAGMA_UNROLL_AUTO
  for (int u = 0; u < UnrollPacks; u++) {
    if (Full || u * WARP_SIZE < nLeft) rsHierStore(dst + u * WARP_SIZE, applyCast<Acc, T>(applyPostOp(red, acc[u])));
  }
}

// A finisher's part of a round, dealing warp tiles the same way.
template <int BytePerPack, int UnrollPeers, int UnrollPacks, typename T, typename Red>
static __device__ void rsHierFinish(ncclDevComm const& comm, ncclTeam scaleIn, ncclTeam scaleUp, Red red, int warp,
                                    int nWarps, ncclSymPtr<T> input, size_t nAllElts, char const* slots, T* output,
                                    int nPacks) {
  using Pack = BytePack<BytePerPack>;
  constexpr int TilePacks = UnrollPacks * WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  ncclLsaPointerGetter<Pack> blockLsa((ncclSymPtr<Pack>)(input + comm.rank * nAllElts));
  int tile = warp * TilePacks;
  NVCC_PRAGMA_UNROLL_DISABLED
  for (; tile + TilePacks <= nPacks; tile += nWarps * TilePacks) {
    rsHierFinishTile</*Full=*/true, UnrollPeers, UnrollPacks, T>(comm, scaleIn, scaleUp, red, blockLsa, slots, output,
                                                                 tile + lane, 0);
  }
  if (tile < nPacks) {
    rsHierFinishTile</*Full=*/false, /*UnrollPeers=*/2, UnrollPacks, T>(comm, scaleIn, scaleUp, red, blockLsa, slots,
                                                                        output, tile + lane, nPacks - tile - lane);
  }
}

// This block's row of rsHierFlags on the given rank of scaleUp.
static __device__ uint64_t* rsHierFlagRow(ncclDevComm const& comm, ncclTeam scaleUp, ncclDevResourceHandle flagsBuf,
                                          int peer) {
  return (uint64_t*)ncclGetResourceBufferLsaPointer(comm, flagsBuf, ncclTeamRankToLsa(comm, scaleUp, peer)) +
         blockIdx.x * ncclSymkRsHierFlagsPerBlock(scaleUp.nRanks);
}

// Spins until each of the n flags starting at flags reaches target, then orders the caller's later
// accesses after what their raisers did before raising them.
static __device__ void rsHierWaitFlags(ncclDevComm const& comm, uint64_t* flags, int n, uint64_t target) {
  uint32_t steps = 0;
  NVCC_PRAGMA_UNROLL_DISABLED
  for (int s = 0; s < n; s++) {
    cuda::atomic_ref<uint64_t> flag(flags[s]);
    while (flag.load(cuda::memory_order_relaxed) < target) {
      if (nccl::utility::testAbort(comm.abortFlag, steps)) return;
    }
  }
  cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_system);
}

// Two-level load ReduceScatter for an LSA team over packages of hierScaleInSize consecutive ranks. Each
// rank sums its package's copies of the block its counterpart on every other package owns and stores
// that one sum there, so a block crosses packages once per package rather than once per rank. A block's
// warps split into senders, which do that, and finishers, which add the sums that landed to our
// package's copies of our own block, so one round's sends overlap the last round's finish. Senders raise
// a ready flag on each owner once a round's sums are stored, and finishers raise a credit flag on each
// sender once they have read them. Every rank splits the work the same way, so peers agree on the
// rounds and their slots.
template <bool EnableProfiler, template <typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_ReduceScatter_HierLsaLD(ncclSymkDevWorkArgs const* args) {
  // 8-bit floats sum as 32-bit ones, which leaves room to hold only pairs of their sources at once.
  constexpr int UnrollPacks = ncclSymkUnrollPacks, UnrollPeers = sizeof(T) == 1 ? 2 : 4;
  constexpr int EltPerPack = ncclSymkBytePerPack / sizeof(T);
  constexpr size_t RoundElts = ncclSymkRsHierChunkBytes / sizeof(T);
  ncclSymkArgsHandler handler{args};
  ncclDevComm const& comm = handler.comm;
  ncclLsaBarrierSession<ncclCoopCta> bar{ncclCoopCta(), comm, ncclTeamTagLsa(), blockIdx.x};
  Red<typename ncclSymkAccumType<Red, T, /*nvls=*/false>::Type> red(handler.devWork->redOpArg);

  ncclTeam lsa = ncclTeamLsa(comm);
  ncclTeam scaleIn = ncclTeamInnerFactor(lsa, args->kcomm.hierScaleInSize);
  ncclTeam scaleUp = ncclTeamOuterFactor(lsa, args->kcomm.hierScaleInSize);
  int nPeers = scaleUp.nRanks - 1;
  ncclDevResourceHandle slotsBuf = args->kcomm.rsHierSlots;
  ncclDevResourceHandle flagsBuf = args->kcomm.rsHierFlags;
  uint64_t* flags = (uint64_t*)ncclGetResourceBufferLocalPointer(comm, flagsBuf) +
                    blockIdx.x * ncclSymkRsHierFlagsPerBlock(scaleUp.nRanks);
  uint64_t* ready = flags;
  uint64_t* credit = flags + scaleUp.nRanks;
  // Peers never write the round count, so it carries over from this block's last launch.
  uint64_t seq = flags[nPeers];

  // Per element, a sender reads every package rank's copy of the block for each other package, and a
  // finisher reads every package rank's copy of ours plus a slot per other package. Split the warps in
  // that proportion, keeping at least one of each.
  int nWarps = blockDim.x / WARP_SIZE;
  int sendReads = scaleIn.nRanks * nPeers;
  int allReads = sendReads + scaleIn.nRanks + nPeers;
  int nSendWarps = min(max((2 * nWarps * sendReads + allReads) / (2 * allReads), 1), nWarps - 1);
  int warp = threadIdx.x / WARP_SIZE;
  bool sender = warp < nSendWarps;
  int roleWarp = sender ? warp : warp - nSendWarps;
  ncclCoopWarpSpan role(sender ? 0 : nSendWarps, sender ? nSendWarps : nWarps - nSendWarps, sender ? 0 : 1);

  ncclCoopNamedBarrierInit();
  bar.sync(ncclCoopCta(), cuda::memory_order_acquire);
  if NCCL_IF_CONSTEXPR (EnableProfiler) ncclSymkProfilerPhase(args, NCCL_KERNEL_PHASE_AFTER_OPEN);

  uint16_t workLo, workHi;
  size_t indexLo, indexHi;
  handler.getWorkRange<T>(blockIdx.x, workLo, indexLo, workHi, indexHi);
  uint64_t round = 0;
  NVCC_PRAGMA_UNROLL_DISABLED
  for (int w = workLo; w <= workHi; w++) {
    ncclSymkDevWork const& dw = handler.devWork[w];
    ncclSymPtr<T> input(dw.inputWin, dw.inputOff);
    T* output = ncclSymPtr<T>(dw.outputWin, dw.outputOff).localPtr();
    size_t lo = w > workLo ? 0 : indexLo;
    size_t hi = w < workHi ? dw.nElts : indexHi;
    // 16-byte packs need every rank's block of the input aligned, not just ours.
    bool aligned = (dw.inputOff | dw.outputOff | dw.nElts * sizeof(T)) % ncclSymkBytePerPack == 0;
    NVCC_PRAGMA_UNROLL_DISABLED
    for (; lo < hi; lo += RoundElts, round++) {
      int n = int(min(RoundElts, hi - lo));
      if (sender) {
        // The opening barrier frees the slots for this launch's first two rounds. After that, a round's
        // slots last held the round two back, which each owner credits once it has read it.
        if (round >= 2) {
          if (role.thread_rank() == 0) rsHierWaitFlags(comm, credit, nPeers, seq + round - 1);
          role.sync();
        }
        if (aligned) {
          rsHierSend<ncclSymkBytePerPack, UnrollPeers, UnrollPacks, T>(
            comm, scaleIn, scaleUp, red, roleWarp, role.nWarps, input + lo, dw.nElts, slotsBuf, round, n / EltPerPack);
        } else {
          rsHierSend<sizeof(T), UnrollPeers, UnrollPacks, T>(comm, scaleIn, scaleUp, red, roleWarp, role.nWarps,
                                                             input + lo, dw.nElts, slotsBuf, round, n);
        }
        role.sync();
        if (role.thread_rank() == 0) {
          cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_system);
          NVCC_PRAGMA_UNROLL_DISABLED
          for (int s = 0; s < nPeers; s++) {
            int to = scaleUp.rank + 1 + s;
            if (to >= scaleUp.nRanks) to -= scaleUp.nRanks;
            cuda::atomic_ref<uint64_t>(rsHierFlagRow(comm, scaleUp, flagsBuf, to)[s])
              .store(seq + round + 1, cuda::memory_order_relaxed);
          }
        }
      } else {
        char const* slots = rsHierSlots(comm, scaleUp, slotsBuf, scaleUp.rank, round);
        if (role.thread_rank() == 0) rsHierWaitFlags(comm, ready, nPeers, seq + round + 1);
        role.sync();
        if (aligned) {
          rsHierFinish<ncclSymkBytePerPack, UnrollPeers, UnrollPacks, T>(comm, scaleIn, scaleUp, red, roleWarp,
                                                                         role.nWarps, input + lo, dw.nElts, slots,
                                                                         output + lo, n / EltPerPack);
        } else {
          rsHierFinish<sizeof(T), UnrollPeers, UnrollPacks, T>(comm, scaleIn, scaleUp, red, roleWarp, role.nWarps,
                                                               input + lo, dw.nElts, slots, output + lo, n);
        }
        role.sync();
        if (role.thread_rank() == 0) {
          cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_system);
          NVCC_PRAGMA_UNROLL_DISABLED
          for (int s = 0; s < nPeers; s++) {
            int from = scaleUp.rank - 1 - s;
            if (from < 0) from += scaleUp.nRanks;
            cuda::atomic_ref<uint64_t>(rsHierFlagRow(comm, scaleUp, flagsBuf, from)[scaleUp.nRanks + s])
              .store(seq + round + 1, cuda::memory_order_relaxed);
          }
        }
      }
    }
  }
  if (threadIdx.x == 0) flags[nPeers] = seq + round;

  if NCCL_IF_CONSTEXPR (EnableProfiler) ncclSymkProfilerPhase(args, NCCL_KERNEL_PHASE_BEFORE_CLOSE);
  bar.sync(ncclCoopCta(), cuda::memory_order_release);
}

template <typename Red, typename T>
static __device__ void reduceMultimem(int tn, int t, Red red, T* input, T* output, size_t nElts, uint32_t rank = 0) {
  uintptr_t inputUptr = reinterpret_cast<uintptr_t>(input);
  uintptr_t outputUptr = reinterpret_cast<uintptr_t>(output);
  size_t nBytes = nElts * sizeof(T);

  constexpr int BytePerPack = LoadMultimem_BigPackSize<Red>::BigPackSize;
  uint32_t nPreBytes = (BytePerPack - inputUptr) % BytePerPack;
  nPreBytes = min((size_t)nPreBytes, nBytes);
  uintptr_t nSufBytes;

  if (sizeof(T) == BytePerPack || (inputUptr - outputUptr) % BytePerPack == 0) {
    constexpr int UnrollPacks = 8 * (16 / BytePerPack);
    constexpr int BytePerChunk = UnrollPacks * WARP_SIZE * BytePerPack;
    static_assert(ncclSymkMcPerRankOffsetBytes % BytePerChunk == 0);
    uint32_t nChunks = (nBytes - nPreBytes) / BytePerChunk;
    uintptr_t cursorAfter = nPreBytes + uintptr_t(nChunks) * BytePerChunk;
    nSufBytes = nBytes - cursorAfter;
    size_t nMainBytes = nBytes - nPreBytes - nSufBytes;

    // Ranks use offset multipliers 0 1 3 2 4 5 7 6 etc. This ensures proper address distribution even if
    // each rank has only 64MiB to process.
    uint32_t startOffsetMultiplier = rank ^ ((rank >> 1) & 1);
    size_t startBytes = nMainBytes > 0 ? (size_t)startOffsetMultiplier * ncclSymkMcPerRankOffsetBytes % nMainBytes : 0;
    uintptr_t wrapAt = cursorAfter + (t % WARP_SIZE) * BytePerPack;
    uintptr_t cursor = nPreBytes + uintptr_t(startBytes / BytePerChunk) * BytePerChunk;
    cursor += (t / WARP_SIZE) * UnrollPacks * WARP_SIZE * BytePerPack;
    cursor += (t % WARP_SIZE) * BytePerPack;
    int nIters = (int)nChunks - (int)(t / WARP_SIZE);
    NVCC_PRAGMA_UNROLL_DISABLED
    while (0 < nIters) {
      if (cursor >= wrapAt) cursor -= nMainBytes;
      BytePack<BytePerPack> tmp[UnrollPacks];
      NVCC_PRAGMA_UNROLL_AUTO
      for (int u = 0; u < UnrollPacks; u++) {
        tmp[u] =
          applyPostOp(red, applyLoadMultimem<Red, BytePerPack>(red, inputUptr + cursor + u * WARP_SIZE * BytePerPack));
      }
      NVCC_PRAGMA_UNROLL_AUTO
      for (int u = 0; u < UnrollPacks; u++) {
        *reinterpret_cast<BytePack<BytePerPack>*>(outputUptr + cursor + u * WARP_SIZE * BytePerPack) = tmp[u];
      }
      cursor += tn * UnrollPacks * BytePerPack;
      nIters -= tn / WARP_SIZE;
    }
  } else {
    nPreBytes = 0;
    nSufBytes = nBytes;
  }

  // Get the prefix+suffix element one at a time.
  NVCC_PRAGMA_UNROLL(4)
  for (uintptr_t i = t * sizeof(T); i < nPreBytes + nSufBytes; i += tn * sizeof(T)) {
    uintptr_t cursor = i < nPreBytes ? i : nBytes - nSufBytes + (i - nPreBytes);
    BytePack<sizeof(T)> val = applyPostOp(red, applyLoadMultimem<Red, sizeof(T)>(red, inputUptr + cursor));
    *reinterpret_cast<BytePack<sizeof(T)>*>(outputUptr + cursor) = val;
  }
}

template <bool EnableProfiler, template <typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_ReduceScatter_LDMC(ncclSymkDevWorkArgs const* args) {
  ncclSymkArgsHandler handler{args};
  ncclLsaBarrierSession<ncclCoopCta> bar{ncclCoopCta(), handler.comm, ncclTeamTagLsa(), blockIdx.x, /*multimem=*/true};
  Red<typename ncclSymkAccumType<Red, T, /*nvls=*/true>::Type> red(handler.devWork->redOpArg);

  int const& rank = handler.comm.rank;
  auto const& multimem = handler.comm.lsaMultimem;

  bar.sync(ncclCoopCta(), cuda::memory_order_acquire);
  if NCCL_IF_CONSTEXPR (EnableProfiler) ncclSymkProfilerPhase(args, NCCL_KERNEL_PHASE_AFTER_OPEN);

  handler.forEachWork<T>([&] __device__(int block, int nBlocks, size_t nElts, size_t nAllElts, ncclSymPtr<T> input,
                                        ncclSymPtr<T> output) {
        // Round robin warps over blocks.
    int t =
      flattenIx(threadIdx.x % WARP_SIZE, WARP_SIZE, block, nBlocks, threadIdx.x / WARP_SIZE, blockDim.x / WARP_SIZE);
    int tn = nBlocks * blockDim.x;

    reduceMultimem(tn, t, red, input.multimemPtr(multimem) + rank * nAllElts, output.localPtr(), nElts, rank);
  });

  if NCCL_IF_CONSTEXPR (EnableProfiler) ncclSymkProfilerPhase(args, NCCL_KERNEL_PHASE_BEFORE_CLOSE);
  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed);
}

// T is user type, EltType is the most aligned type
template <bool EnableProfiler, typename T, typename Red, typename EltType>
__device__ __forceinline__ void ncclSymkRun_ReduceScatter_LL_body(
  ncclSymkDevWorkArgs const* args, ncclSymkArgsHandler& handler, ncclLLA2ASession<ncclCoopCta>& lla2a, Red red,
  EltType* input, EltType* output, int nElts, int nPacks, int nStrideElts) {
  using Pack = BytePack<8>;
  using Acc = typename Red::EltType;
  using AccPack = BytePack<8 * sizeof(Acc) / sizeof(T)>;
  constexpr int EltPerPack = 8 / sizeof(EltType);

  int const& nRanks = handler.comm.nRanks;
  int const& rank = handler.comm.rank;
  int t = threadIdx.x;
  int tn = blockDim.x;
  ncclCoopCta cta;
  // LL fuses the peer sync into the first epoch, so AFTER_OPEN is stamped once, at the
  // first endEpoch below (see ncclDevProfilerPhases in device.h); BEGIN marks the start.
  [[maybe_unused]] bool profilerPhase1Done = false;

  NVCC_PRAGMA_UNROLL_DISABLED
  while (0 < nElts) {
    int nIterPacks = min(nPacks, tn);
    int tn_div_nPacks = tn / nIterPacks;
    int tn_mod_nPacks = tn % nIterPacks;
    int peer = t / nIterPacks;
    int pack = t % nIterPacks;

    NVCC_PRAGMA_UNROLL_DISABLED
    for (int i = t; i < nRanks * nIterPacks; i += tn) {
      Pack got = loadPack<Pack>(input + peer * nStrideElts, pack * EltPerPack, nElts);
      lla2a.send(peer, rank * nIterPacks + pack, got);
      peer += tn_div_nPacks;
      pack += tn_mod_nPacks;
      if (nIterPacks <= pack) {
        peer += 1;
        pack -= nIterPacks;
      }
    }

    if (t < nIterPacks) {
      AccPack got = lla2a.template recvReduce</*Unroll=*/8, Pack>(
        /*slotStart=*/t, /*slotCount=*/nRanks, /*slotStride=*/nIterPacks,
        /*eltToAcc=*/[&] __device__(Pack x) -> AccPack { return applyCast<T, Acc>(x); },
        /*reduce=*/[&] __device__(AccPack a, AccPack b) -> AccPack { return applyReduce(red, a, b); });
      got = applyPostOp(red, got);
      storePack(output, t * EltPerPack, nElts, applyCast<Acc, T>(got));
    }
    lla2a.endEpoch(cta);
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

template <bool EnableProfiler, template <typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_ReduceScatter_LL(ncclSymkDevWorkArgs const* args) {
  ncclSymkArgsHandler handler{args};
  ncclLLA2ASession<ncclCoopCta> lla2a(ncclCoopCta(), handler.comm, ncclTeamLsa(handler.comm), handler.lsaLLA2A,
                                      blockIdx.x, ncclSymkMaxThreads);
  Red<typename ncclSymkAccumType<Red, T, /*nvls=*/false>::Type> red(handler.devWork->redOpArg);
  using Pack = BytePack<8>;
  constexpr int EltPerPack = 8 / sizeof(T);

  handler.singleWork<T>([&] __device__(int nElts, int nAllElts, ncclSymPtr<T> inputPtr, ncclSymPtr<T> outputPtr) {
    int nPacks = divUp(nElts, EltPerPack);

    T* input = (T*)inputPtr.localPtr();
    T* output = (T*)outputPtr.localPtr();

    uint32_t lowBits = nAllElts * sizeof(T);
    lowBits |= (uintptr_t)input;
    lowBits |= (uintptr_t)output;
    if (__builtin_expect(lowBits % 8 == 0, true)) {
      ncclSymkRun_ReduceScatter_LL_body<EnableProfiler, T>(args, handler, lla2a, red, (Pack*)input, (Pack*)output,
                                                           nPacks, nPacks, divUp(nAllElts, EltPerPack));
    } else {
      ncclSymkRun_ReduceScatter_LL_body<EnableProfiler, T>(args, handler, lla2a, red, input, output, nElts, nPacks,
                                                           nAllElts);
    }
  });
}
