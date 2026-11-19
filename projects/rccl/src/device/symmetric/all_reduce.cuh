/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "sym_kernels.h"
#include "nccl_device.h"
#if defined(__HIP_PLATFORM_AMD__)
#include "symmetric/kernel.h"
#include "symmetric/primitives.h"
#include "symmetric/data_ops.h"
#include "symmetric/hier.h"
#else
#include "kernel.cuh"
#include "primitives.cuh"
#include "data_ops.cuh"
#include "hier.cuh"
#endif
#include <stdio.h>

template <int BytePerPack, int UnrollPacks, int UnrollPeers, typename T, bool EnableTma, typename Red>
static __device__ __forceinline__ void allreduceDeep(ncclSymkArgsHandler const& handler, int tn, int t, bool waitNeeded,
                                                     ncclLsaBarrierSession<ncclCoopCta>& bar, Red red,
                                                     ncclSymPtr<char> input, ncclSymPtr<char> output, int32_t nIters) {
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
      int r = rank;
      if (++r == nRanks) r = 0;
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

      if (++r == nRanks) r = 0;

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
            if (++r == nRanks) r = 0;
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
      }
#endif

      dr = 0;
      r = rank;
      NVCC_PRAGMA_UNROLL(2)
      for (int partial = 0; partial <= 1; partial++) {
        NVCC_PRAGMA_UNROLL_DISABLED
        for (int i = 0; partial ? i < 1 : (dr + UnrollPeers <= nRanks); partial ? i++ : (dr += UnrollPeers)) {
          NVCC_PRAGMA_UNROLL_AUTO
          for (int ur = 0; ur < UnrollPeers - partial; ur++) {
            if (partial && dr + ur == nRanks) break;
#if NCCL_SYMK_ASYNC_TILE
            if NCCL_IF_CONSTEXPR (EnableTma) {
              ncclSymkTileStore(outPacks.peerPtr(world, r), tmaSmem->buff[0], tileSize, lane);
            } else
#endif
            {
              NVCC_PRAGMA_UNROLL(UnrollPacks)
              for (int u = 0; u < UnrollPacks; u++) {
                outPacks.peerPtr(world, r)[u * WARP_SIZE] = acc0[u];
              }
            }
            if (++r == nRanks) r = 0;
          }
        }
      }
#if NCCL_SYMK_ASYNC_TILE
      if NCCL_IF_CONSTEXPR (EnableTma) {
        // Drain every peer's store before the next iteration refills the tile.
        ncclSymkTileStoreWait(lane);
        __syncwarp();
      }
#endif

      inpPacks += intptr_t(wn) * UnrollPacks * WARP_SIZE;
      outPacks += intptr_t(wn) * UnrollPacks * WARP_SIZE;
      nIters -= wn;
      if (nIters <= 0) break;

#if NCCL_SYMK_ASYNC_TILE
      // Load data for next iteration.
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
  }
}

template <int UnrollPeers, typename Red, typename T>
static __device__ __forceinline__ void allreduceEnds(ncclSymkArgsHandler const& handler, int tn, int t, Red red,
                                                     ncclSymPtr<T> input, ncclSymPtr<T> output, size_t nElts,
                                                     uint32_t nPreElts, size_t nSufElts) {
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

    acc0 = applyCast<Acc, T>(acc1);
    dr = 0;
    r = rank;
    NVCC_PRAGMA_UNROLL(2)
    for (int partial = 0; partial <= 1; partial++) {
      NVCC_PRAGMA_UNROLL_DISABLED
      for (int j = 0; partial ? j < 1 : (dr + UnrollPeers <= nRanks); partial ? j++ : (dr += UnrollPeers)) {
        NVCC_PRAGMA_UNROLL_AUTO
        for (int u = 0; u < UnrollPeers - partial; u++) {
          if (partial && dr + u == nRanks) break;
          outPacks.peerPtr(world, r)[elt] = acc0;
          r += 1;
          if (r == nRanks) r = 0;
        }
      }
    }
  }
}

template <typename Red, typename T, bool EnableTma>
static __device__ void allreduce(ncclSymkArgsHandler const& handler, int tn, int t, int nBlocks, bool waitNeeded,
                                 ncclLsaBarrierSession<ncclCoopCta>& bar, Red red, ncclSymPtr<T> input,
                                 ncclSymPtr<T> output, size_t nElts) {
  int const& nRanks = handler.comm.nRanks;
  size_t nBytes = nElts * sizeof(T);

#if defined(__gfx950__)
  // Engage on a floor instead of trimming, so the partial final wave is kept rather than handed to
  // the per-element tail. A floor of one warp per block cost 8% at 512 KB, two gained 8% at 1 MB.
  constexpr int MinEngagedWarpPerBlock = 2;
  uint32_t const chunkFloor = uint32_t(MinEngagedWarpPerBlock * nRanks * nBlocks) / ncclSymkMinWarpsPerBlock;
#else
  int const& nRanks_rcp32 = handler.nRanks_rcp32;
  uint32_t nBlocks_rcp32 = nccl::utility::idivRcp32_upto64(nBlocks);
  uint32_t nRanks_nBlocks_rcp32 = nccl::utility::imulRcp32(nRanks, nRanks_rcp32, nBlocks, nBlocks_rcp32);
  uint32_t const chunkFloor = 1;
#endif

  // True only where the deep loop really stages through a TDM engine.
  constexpr bool AsyncTile = ncclSymkAsyncTile && EnableTma;

  uint32_t nPreBytes = (16u - input.offset) % 16u;
  nPreBytes = min((size_t)nPreBytes, nBytes);
  uintptr_t cursor = nPreBytes;

  if ((input.offset - output.offset) % 16 == 0) {
#if defined(__gfx950__)
    // Dropping to one pack cuts BytePerChunk to a quarter, which a 64-wide wavefront needs to keep
    // iterations per warp in range at 8 MB. One pack per peer also lets UnrollPeers batch them all.
    constexpr int UnrollPacksPlain = 1, UnrollPeers = 8;
#else
    constexpr int UnrollPacksPlain = ncclSymkUnrollPacks, UnrollPeers = 2;
#endif
    constexpr int BytePerPack = ncclSymkBytePerPack,
                  UnrollPacks = AsyncTile ? ncclSymkDeepUnrollPacks(sizeof(T)) : UnrollPacksPlain;

    // Derived from UnrollPacks so the two cannot disagree: a chunk wider than what allreduceDeep()
    // reduces would leave the difference unreduced.
    constexpr int BytePerChunk = ncclSymkGetBytesPerChunk(ncclSymkMinWarpsPerBlock, UnrollPacks);
    uint32_t chunks = (nBytes - cursor) / BytePerChunk;
#if !defined(__gfx950__)
    chunks -= imodFast32(chunks, nRanks * nBlocks, nRanks_nBlocks_rcp32);
#endif
    if (chunks >= chunkFloor) {
      uintptr_t cursorAfter = cursor + uintptr_t(chunks) * BytePerChunk;
      allreduceDeep<BytePerPack, UnrollPacks, UnrollPeers, T, EnableTma>(handler, tn, t, waitNeeded, bar, red,
                                                                         (ncclSymPtr<char>)input + cursor,
                                                                         (ncclSymPtr<char>)output + cursor,
                                                                         chunks * ncclSymkMinWarpsPerBlock);
      cursor = cursorAfter;
      waitNeeded = false;
    }
  }

  if (sizeof(T) == 4 || (sizeof(T) < 4 && (input.offset - output.offset) % 4 == 0)) {
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
      allreduceDeep<(sizeof(T) <= BytePerPack ? BytePerPack : 0), UnrollPacks, UnrollPeers, T, /*EnableTma*/ false>(
        handler, tn, t, waitNeeded, bar, red, (ncclSymPtr<char>)input + cursor, (ncclSymPtr<char>)output + cursor,
        chunks * ncclSymkMinWarpsPerBlock);
      cursor = cursorAfter;
      waitNeeded = false;
    }
  }

  if (waitNeeded) bar.wait(ncclCoopCta(), cuda::memory_order_acquire);

  constexpr int UnrollPeers = 8;
  size_t nSufElts = (nBytes - cursor) / sizeof(T);
  allreduceEnds<UnrollPeers>(handler, tn, t, red, input, output, nElts, nPreBytes / sizeof(T), nSufElts);
}

template <bool EnableProfiler, template <typename> typename Red, typename T, bool EnableTma>
__device__ __forceinline__ void ncclSymkRun_AllReduce_RSxLD_AGxST_impl(ncclSymkDevWorkArgs const* args) {
  ncclSymkArgsHandler handler{args};
  ncclLsaBarrierSession<ncclCoopCta> bar{ncclCoopCta(), handler.comm, ncclTeamTagLsa(), blockIdx.x};

  Red<typename ncclSymkAccumType<Red, T, /*nvls=*/false>::Type> red(handler.devWork->redOpArg);

  int const& rank = handler.comm.rank;
  int const& nRanks = handler.comm.nRanks;

  bar.arrive(ncclCoopCta(), cuda::memory_order_relaxed);
  if NCCL_IF_CONSTEXPR (EnableProfiler) {
    // Finish the opening barrier here so AFTER_OPEN marks the end of the peer sync.
    // Same barrier ops as the default variant (which fuses the wait into allreduce),
    // so the two stay barrier-compatible when peers disagree on profiling.
    bar.wait(ncclCoopCta(), cuda::memory_order_acquire);
    ncclSymkProfilerPhase(args, NCCL_KERNEL_PHASE_AFTER_OPEN);
  }

  bool waitNeeded = !EnableProfiler;
  handler.forEachWork<T>([&] __device__(int block, int nBlocks, size_t nElts, size_t nAllElts, ncclSymPtr<T> input,
                                        ncclSymPtr<T> output) {
        // Threads numbered globally such that we round robin warps by rank then block.
    int gt = flattenIx(threadIdx.x % WARP_SIZE, WARP_SIZE, rank, nRanks, block, nBlocks, threadIdx.x / WARP_SIZE,
                       blockDim.x / WARP_SIZE);
    int gtn = nRanks * nBlocks * blockDim.x;

    allreduce<decltype(red), T, EnableTma>(handler, gtn, gt, nBlocks, waitNeeded, bar, red, input, output, nElts);

    waitNeeded = false;
  });

  if NCCL_IF_CONSTEXPR (EnableProfiler) ncclSymkProfilerPhase(args, NCCL_KERNEL_PHASE_BEFORE_CLOSE);
  bar.sync(ncclCoopCta(), cuda::memory_order_release);
}

template <bool EnableProfiler, template <typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_RSxLD_AGxST(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_AllReduce_RSxLD_AGxST_impl<EnableProfiler, Red, T, /*EnableTma=*/false>(args);
}

template <bool EnableProfiler, template <typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_RSxTmaLD_AGxTmaST(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_AllReduce_RSxLD_AGxST_impl<EnableProfiler, Red, T, /*EnableTma=*/true>(args);
}

// Batched copy: a one-source reduceBatch, exact through Red's accumulator.
template <typename T, typename Red, typename GetDst, typename GetSrc>
static __device__ void arHierCopyBatch(ncclCoopWarpSpan role, Red red, int nBatch, int nElts, bool aligned,
                                       GetDst getDst, GetSrc getSrc) {
  reduceBatch</*PostOp=*/false, hierBatchMode>(
    role, /*inPlace=*/false, nBatch, nElts, GMemApertureTag(), /*dstAlignMin=*/16, getDst, red, /*nSrcs=*/1,
    /*getSrc=*/[&] __device__(int b, int) -> T* { return getSrc(b); },
    /*srcPtrCommonMask=*/aligned ? 15u : 0u, /*getSrcPtrMasked=*/[&] __device__(int) -> unsigned { return 0u; });
}

// Two-level AllReduce: the hierarchical ReduceScatter's send and finish over one piece per rank, then
// ST each reduced piece to the other packages and forward it within the package. When pipelined, sender,
// finisher and forwarder warps run concurrently, each on its own round.
template <bool EnableProfiler, template <typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_HierLsaLDST(ncclSymkDevWorkArgs const* args) {
  size_t const chunkBytes = args->kcomm.rsHierChunkSize;
  size_t const RoundElts = chunkBytes / sizeof(T);
  constexpr size_t PackElts = ncclSymkBytePerPack / sizeof(T);
  ncclSymkArgsHandler handler{args};
  ncclDevComm const& comm = handler.comm;
  ncclLsaBarrierSession<ncclCoopCta> bar{ncclCoopCta(), comm, ncclTeamTagLsa(), blockIdx.x};
  Red<typename ncclSymkAccumType<Red, T, /*nvls=*/false>::Type> red(handler.devWork->redOpArg);

  int const& rank = comm.rank;
  int const& nRanks = comm.nRanks;
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
  uint64_t* agReady = flags + 2 * scaleUp.nRanks;
  // Shared with the hierarchical ReduceScatter.
  uint64_t seq = flags[nPeers];

  int t = threadIdx.x;
  int tn = blockDim.x;
  int nFwd = scaleIn.nRanks - 1;

  // Without the pipeline every warp takes the three roles in turn.
  bool pipeline = args->kcomm.arHierPipeline;
  int nWarps = blockDim.x / WARP_SIZE;
  int roleId = 0;
  ncclCoopWarpSpan role(0, nWarps, 0);
  if (pipeline) {
    // Split the warps by the loads and stores each role does per element of a piece, at least one each.
    int sendW = (scaleIn.nRanks + 1) * nPeers;
    int finishW = 2 * (scaleIn.nRanks + scaleUp.nRanks) - 1;
    int fwdW = scaleIn.nRanks * nPeers;
    int allW = sendW + finishW + fwdW;
    int nSendWarps = min(max((2 * nWarps * sendW + allW) / (2 * allW), 1), nWarps - 2);
    int nFwdWarps = min(max((2 * nWarps * fwdW + allW) / (2 * allW), 1), nWarps - nSendWarps - 1);
    int nFinishWarps = nWarps - nSendWarps - nFwdWarps;
    int warp = threadIdx.x / WARP_SIZE;
    roleId = warp < nSendWarps ? 0 : warp < nSendWarps + nFinishWarps ? 1 : 2;
    role = ncclCoopWarpSpan(roleId == 0 ? 0 : roleId == 1 ? nSendWarps : nSendWarps + nFinishWarps,
                            roleId == 0 ? nSendWarps : roleId == 1 ? nFinishWarps : nFwdWarps, roleId);
  }
  bool leader = role.thread_rank() == 0;

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
    ncclSymPtr<T> output(dw.outputWin, dw.outputOff);
    T* outLocal = output.localPtr();
    ncclLsaPointerGetter<T> outLsa(output);
    size_t const nElts = dw.nElts;
    size_t const pieceElts = nElts / nRanks / PackElts * PackElts;
    // The block's share of the message maps to the same share of every piece.
    size_t lo = w > workLo ? 0 : indexLo;
    size_t hi = w < workHi ? nElts : indexHi;
    bool ownsEnd = hi == nElts;
    lo = min(lo / nRanks / PackElts * PackElts, pieceElts);
    hi = min(hi / nRanks / PackElts * PackElts, pieceElts);
    bool aligned = (dw.inputOff | dw.outputOff) % ncclSymkBytePerPack == 0;

    NVCC_PRAGMA_UNROLL_DISABLED
    for (; lo < hi; lo += RoundElts, round++) {
      int n = int(min(RoundElts, hi - lo));
      uint64_t const done = seq + round + 1;
      size_t const myOff = rank * pieceElts + lo;
      auto runRole = [&] __device__(int id) {
        if (id == 0) {
          // Scale-in reduce-scatter into the owners' slots. A slot set is reused every other round.
          if (round >= 2) {
            if (leader) hierWaitFlags(comm, credit, nPeers, seq + round - 1);
            role.sync();
          }
          hierReduceSend<T>(comm, scaleIn, scaleUp, red, role, input + lo, pieceElts, slotsBuf, round, n, chunkBytes,
                            aligned);
          role.sync();
          if (leader) hierRaiseUp(comm, scaleUp, flagsBuf, /*base=*/0, done);
        } else if (id == 1) {
          // Scale-up reduce-scatter into our output, then ST our piece to our counterparts and package peers.
          char const* slots = hierSlots(comm, scaleUp, slotsBuf, scaleUp.rank, round, chunkBytes);
          if (leader) hierWaitFlags(comm, ready, nPeers, done);
          role.sync();
          hierReduceFinish<T>(comm, scaleIn, scaleUp, red, role, input + lo, pieceElts, slots, outLocal + myOff, n,
                              chunkBytes, aligned);
          role.sync();
          if (leader) hierRaiseDown(comm, scaleUp, flagsBuf, /*base=*/scaleUp.nRanks, done);
          arHierCopyBatch<T>(
            role, red, /*nBatch=*/nPeers, n, aligned,
            /*getDst=*/
            [&] __device__(int s) -> T* {
              return outLsa(ncclTeamRankToLsa(comm, scaleUp, hierPeerUp(scaleUp, s))) + myOff;
            },
            /*getSrc=*/[&] __device__(int) -> T* { return outLocal + myOff; });
          role.sync();
          if (leader) hierRaiseUp(comm, scaleUp, flagsBuf, /*base=*/2 * scaleUp.nRanks, done);
          arHierCopyBatch<T>(
            role, red, /*nBatch=*/nFwd, n, aligned,
            /*getDst=*/
            [&] __device__(int b) -> T* { return outLsa(hierCopyLsa(comm, scaleIn, 1 + b)) + myOff; },
            /*getSrc=*/[&] __device__(int) -> T* { return outLocal + myOff; });
        } else {
          // Scale-in allgather of our counterparts' pieces once they land.
          if (leader) hierWaitFlags(comm, agReady, nPeers, done);
          role.sync();
          auto pieceOff = [&] __device__(int b) -> size_t {
            return ncclTeamRankToWorld(comm, scaleUp, hierPeerDown(scaleUp, b / nFwd)) * pieceElts + lo;
          };
          arHierCopyBatch<T>(
            role, red, /*nBatch=*/nPeers * nFwd, n, aligned,
            /*getDst=*/
            [&] __device__(int b) -> T* {
              return outLsa(hierCopyLsa(comm, scaleIn, 1 + b % nFwd)) + pieceOff(b);
            },
            /*getSrc=*/[&] __device__(int b) -> T* { return outLocal + pieceOff(b); });
        }
      };
      if (pipeline) {
        runRole(roleId);
      } else {
        NVCC_PRAGMA_UNROLL_DISABLED
        for (int id = 0; id < 3; id++) runRole(id);
      }
    }

    // Each tail element is read and written by one rank only, so this is safe in place.
    if (ownsEnd) {
      allreduceEnds</*UnrollPeers=*/8>(handler, nRanks * tn, rank * tn + t, red, input, output, nElts,
                                       /*nPreElts=*/0, /*nSufElts=*/nElts - nRanks * pieceElts);
    }
  }
  if (threadIdx.x == 0) flags[nPeers] = seq + round;

  if NCCL_IF_CONSTEXPR (EnableProfiler) ncclSymkProfilerPhase(args, NCCL_KERNEL_PHASE_BEFORE_CLOSE);
  bar.sync(ncclCoopCta(), cuda::memory_order_release);
}

template <typename Red, typename T>
static __device__ void allreduceMultimem(int tn, int t, Red red, T* input, T* output, size_t nElts) {
  uintptr_t inputUptr = reinterpret_cast<uintptr_t>(input);
  uintptr_t outputUptr = reinterpret_cast<uintptr_t>(output);
  size_t nBytes = nElts * sizeof(T);

  constexpr int BytePerPack = LoadMultimem_BigPackSize<Red>::BigPackSize;
  uint32_t nPreBytes = (BytePerPack - inputUptr) % BytePerPack;
  nPreBytes = min((size_t)nPreBytes, nBytes);
  uintptr_t nSufBytes;

  if (alignof(T) == BytePerPack || (inputUptr - outputUptr) % BytePerPack == 0) {
    constexpr int UnrollPacks = 16 * 8 / BytePerPack;
    constexpr int BytePerChunk = UnrollPacks * WARP_SIZE * BytePerPack;
    uintptr_t cursor = nPreBytes;
    int nChunks = (nBytes - cursor) / BytePerChunk;
    uintptr_t cursorAfter = cursor + uintptr_t(nChunks) * BytePerChunk;
    nSufBytes = nBytes - cursorAfter;
    cursor += (t / WARP_SIZE) * UnrollPacks * WARP_SIZE * BytePerPack;
    cursor += (t % WARP_SIZE) * BytePerPack;
    int nIters = nChunks - t / WARP_SIZE;
    NVCC_PRAGMA_UNROLL_DISABLED
    while (0 < nIters) {
      BytePack<BytePerPack> tmp[UnrollPacks];
      NVCC_PRAGMA_UNROLL_AUTO
      for (int u = 0; u < UnrollPacks; u++) {
        tmp[u] = applyLoadMultimem<Red, BytePerPack>(red, inputUptr + cursor + u * WARP_SIZE * BytePerPack);
      }
      NVCC_PRAGMA_UNROLL_AUTO
      for (int u = 0; u < UnrollPacks; u++) {
        multimem_st_global(outputUptr + cursor + u * WARP_SIZE * BytePerPack, tmp[u]);
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
    BytePack<sizeof(T)> val = applyLoadMultimem<Red, sizeof(T)>(red, inputUptr + cursor);
    multimem_st_global(outputUptr + cursor, val);
  }
}

template <bool EnableProfiler, template <typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_RSxLDMC_AGxSTMC(ncclSymkDevWorkArgs const* args) {
  ncclSymkArgsHandler handler{args};
  ncclLsaBarrierSession<ncclCoopCta> bar{ncclCoopCta(), handler.comm, ncclTeamTagLsa(), blockIdx.x, /*multimem=*/true};

  Red<typename ncclSymkAccumType<Red, T, /*nvls=*/true>::Type> red(handler.devWork->redOpArg);

  int const& rank = handler.comm.rank;
  int const& nRanks = handler.comm.nRanks;
  auto const& multimem = handler.comm.lsaMultimem;

  bar.sync(ncclCoopCta(), cuda::memory_order_acquire);
  if NCCL_IF_CONSTEXPR (EnableProfiler) ncclSymkProfilerPhase(args, NCCL_KERNEL_PHASE_AFTER_OPEN);

  handler.forEachWork<T>([&] __device__(int block, int nBlocks, size_t nElts, size_t nAllElts, ncclSymPtr<T> input,
                                        ncclSymPtr<T> output) {
        // Threads numbered globally such that we round robin warps by rank then block.
    int gt = flattenIx(threadIdx.x % WARP_SIZE, WARP_SIZE, rank, nRanks, block, nBlocks, threadIdx.x / WARP_SIZE,
                       blockDim.x / WARP_SIZE);
    int gtn = nRanks * nBlocks * blockDim.x;

    allreduceMultimem(gtn, gt, red, input.multimemPtr(multimem), output.multimemPtr(multimem), nElts);
  });

  if NCCL_IF_CONSTEXPR (EnableProfiler) ncclSymkProfilerPhase(args, NCCL_KERNEL_PHASE_BEFORE_CLOSE);
  bar.sync(ncclCoopCta(), cuda::memory_order_release);
}

template <bool EnableProfiler, template <typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_AGxLL_R_impl(ncclSymkDevWorkArgs const* args, bool multimem) {
  ncclSymkArgsHandler handler{args};
  ncclLLA2ASession<ncclCoopCta> lla2a(ncclCoopCta(), handler.comm, ncclTeamLsa(handler.comm), handler.lsaLLA2A,
                                      blockIdx.x, ncclSymkMaxThreads, multimem, handler.comm.lsaMultimem);

  int const& rank = handler.comm.rank;
  int const& nRanks = handler.comm.nRanks;
  using Acc = typename ncclSymkAccumType<Red, T, /*nvls=*/false>::Type;
  Red<Acc> red(handler.devWork->redOpArg);

  using Pack = BytePack<8>;
  using AccPack = BytePack<8 * sizeof(Acc) / sizeof(T)>;
  constexpr int EltPerPack = 8 / sizeof(T);

  handler.singleWork<T>([&] __device__(int nElts, int nAllElts, ncclSymPtr<T> inputPtr, ncclSymPtr<T> outputPtr) {
    int nPacks = divUp(nElts, EltPerPack);

    T* input = (T*)inputPtr.localPtr();
    T* output = (T*)outputPtr.localPtr();

    bool packAligned = 8 <= alignof(T) || (nElts * sizeof(T) | (uintptr_t)input | (uintptr_t)output) % 8 == 0;

    ncclCoopCta cta;
    int t = threadIdx.x;
    int tn = blockDim.x;
    // LL fuses the peer sync into the first epoch, so AFTER_OPEN is stamped once, at the
    // first endEpoch below (see ncclDevProfilerPhases in device.h); BEGIN marks the start.
    [[maybe_unused]] bool profilerPhase1Done = false;

    if (__builtin_expect(packAligned, true)) {
      NVCC_PRAGMA_UNROLL_DISABLED
      while (0 < nPacks) {
        if (t < nPacks) {
          int nIterPacks = min(nPacks, tn);
          Pack inp = loadPack<Pack>((Pack*)input, t, nPacks);
          lla2a.bcast(/*slot=*/nIterPacks * rank + t, inp);
          AccPack out = lla2a.template recvReduce</*Unroll=*/8, Pack>(
            /*slotStart=*/t, /*slotCount=*/nRanks, /*slotStride=*/nIterPacks,
            /*eltToAcc=*/[&] __device__(Pack x) -> AccPack { return applyCast<T, Acc>(x); },
            /*reduce=*/[&] __device__(AccPack a, AccPack b) -> AccPack { return applyReduce(red, a, b); });
          storePack((Pack*)output, t, nPacks, applyCast<Acc, T>(out));
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
        nPacks -= tn;
      }
    } else {
      NVCC_PRAGMA_UNROLL_DISABLED
      while (0 < nElts) {
        if (t * EltPerPack < nElts) {
          int nIterPacks = min(nPacks, tn);
          Pack inp = loadPack<Pack>(input, t * EltPerPack, nElts);
          lla2a.bcast(/*slot=*/nIterPacks * rank + t, inp);
          AccPack out = lla2a.template recvReduce</*Unroll=*/8, Pack>(
            /*slotStart=*/t, /*slotCount=*/nRanks, /*slotStride=*/nIterPacks,
            /*eltToAcc=*/[&] __device__(Pack x) -> AccPack { return applyCast<T, Acc>(x); },
            /*reduce=*/[&] __device__(AccPack a, AccPack b) -> AccPack { return applyReduce(red, a, b); });
          storePack(output, t * EltPerPack, nElts, applyCast<Acc, T>(out));
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
    }
    if NCCL_IF_CONSTEXPR (EnableProfiler) ncclSymkProfilerPhase(args, NCCL_KERNEL_PHASE_BEFORE_CLOSE);
  });
}

template <bool EnableProfiler, template <typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_AGxLL_R(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_AllReduce_AGxLL_R_impl<EnableProfiler, Red, T>(args, /*multimem=*/false);
}

template <bool EnableProfiler, template <typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_AGxLLMC_R(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_AllReduce_AGxLL_R_impl<EnableProfiler, Red, T>(args, /*multimem=*/true);
}
