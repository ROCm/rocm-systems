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

template <typename T>
static __device__ void bcastLsa(ncclSymkArgsHandler& handler, int tn, int t, ncclSymPtr<T> input,
                                ncclSymPtr<T> output, size_t nElts, BoolTag</*multimem=*/true>) {
  bcastMultimem(handler, tn, t, input, output, nElts);
}

template <typename T>
static __device__ void bcastLsa(ncclSymkArgsHandler& handler, int tn, int t, ncclSymPtr<T> input,
                                ncclSymPtr<T> output, size_t nElts, BoolTag</*multimem=*/false>) {
  ncclTeam lsa = ncclTeamLsa(handler.comm);
  T const* src = input.localPtr();
    // When the chunk already landed locally the self store is redundant, and it
    // races with the ring warp relaying that same chunk over GIN.
  int selfSkip = (input == output) ? 1 : 0;
  for (size_t i = t; i < nElts; i += tn) {
    T v = src[i];
      // Stagger the first destination by LSA rank so the node's GPUs don't all
      // target the same peer at once.
    for (int s = selfSkip; s < lsa.nRanks; s++) {
      int r = (lsa.rank + s) % lsa.nRanks;
      output.lsaPtr(r)[i] = v;
    }
  }
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
