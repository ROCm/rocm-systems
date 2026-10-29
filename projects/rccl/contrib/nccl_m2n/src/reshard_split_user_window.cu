/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

/*************************************************************************
 * Tensor Reshard — Split-Comm RING Kernel (QP-scalability path)
 *
 * Dual-DevComm variant of reshardKernelUserWindow used by the PACK
 * copy mode when the split path is active (RING + NODE_AWARE, generator
 * spanning >= 2 NVL domains).  It replaces the single full all-to-all
 * DevComm with two narrower ones:
 *
 *   commA (FULL): trainer ranks + the FIRST generator NVL domain.  Carries
 *                 the source -> first-domain-leader GIN puts.
 *   commB (RAIL): all generator ranks.  Carries the cross-NVL ring forward
 *                 and the intra-domain LSA fan-out.
 *
 * Because PACK always stages contiguously (Stage G in
 * reshardCopyPackNormalized sets isContiguous=true on every target/source),
 * this kernel only implements the contiguous branches of the original
 * RING kernel — there is no strided plan path here.
 *
 * Phasing (per rank, in program order):
 *   Phase A (members of commA): trainer puts; first-domain leaders wait
 *     for their data on commA.  Bracketed by commA initial/final barriers.
 *   Phase B (members of commB): downstream leaders wait for the ring
 *     forward on commB; ALL leaders ring-forward + LSA-fan-out on commB.
 *     Bracketed by commB initial/final barriers.
 *
 * A first-domain leader is a member of BOTH comms: it receives in Phase A
 * (commA) and forwards/replicates in Phase B (commB).  Phase A fully
 * precedes Phase B in program order, so its commA wait is satisfied before
 * any commB forward it issues.  See reshard_split.h for the signal-index
 * scheme (commA wait keyed by trainer commA-rank; commB ring keyed by
 * srcShardIdx, consistent across every domain's matching rep slot).
 ************************************************************************/

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "cuda_runtime.h"
#include "nccl.h"
#include "nccl_device.h"

#include "nccl_m2n.h"
#include "reshard_types.h"
#include "m2n_checks.h"
#include "m2n_log.h"
#include "reshard_internal.h"
#include "reshard_kernels.cuh"
#include "reshard_split.h"


#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 30, 5)
#define NCCLM2N_RESHARD_GIN_FINAL_FENCE (ncclGinFenceLevel::Put | ncclGinFenceLevel::Get)
#else
#define NCCLM2N_RESHARD_GIN_FINAL_FENCE ncclGinFenceLevel::Relaxed
#endif

// ============================================================================
// Dual-DevComm RING kernel (contiguous PACK staging only).
// ============================================================================
__global__ __launch_bounds__(DEFAULT_KERNEL_MAX_NTHREADS, 1) void reshardKernelUserWindowSplit(
  ncclReshardParamsSplit params, struct ncclDevComm devCommA, struct ncclDevComm devCommB) {
  const bool inA = (params.myRankInA >= 0);
  const bool inB = (params.myRankInB >= 0);

  const int warpId = threadIdx.x / 32;
  const int laneId = threadIdx.x % 32;

  // GIN context index.  commA (per-parent, FULL) maps this CTA over the
  // whole devCommA context pool.  commB is the SHARED DevComm: it carries
  // maxConcurrency slots, so this reshard maps its CTAs over its own slot
  // of ctxPerSlotB contexts, then offsets by slotIdx*ctxPerSlotB.
  int ginContextA = 0;
  if (inA) {
    int ctxCountA = (devCommA.ginContextCount > 0) ? (int)devCommA.ginContextCount : 1;
    ginContextA = reshardMapCtaToGinContext((int)blockIdx.x, (int)gridDim.x, ctxCountA);
  }

  int ginContextB = 0;
  if (inB) {
    int ctxPerSlot = (params.ctxPerSlotB > 0) ? params.ctxPerSlotB : 1;
    int localCtxB = reshardMapCtaToGinContext((int)blockIdx.x, (int)gridDim.x, ctxPerSlot);
    ginContextB = params.slotIdx * ctxPerSlot + localCtxB;
  }

  // Per-bucket offset keeps concurrent staging classes disjoint on commB.
  const unsigned int signalSlotOffsetB = (unsigned int)(params.slotIdx * params.signalsPerSlotB);

  __shared__ uint64_t initialSignals[MAX_SOURCES];
  static_assert(sizeof(initialSignals) / sizeof(initialSignals[0]) >= MAX_SOURCES,
                "initialSignals[] must be sized at least MAX_SOURCES");

  // ----------------------------------------------------------------------
  // Phase A — commA (trainer puts; first-domain leaders receive).
  // ----------------------------------------------------------------------
  if (inA) {
    ncclGin ginA{devCommA, ginContextA};
    ncclTeam worldA = ncclTeamWorld(devCommA);

    if (params.kernelTrace && blockIdx.x == 0 && threadIdx.x == 0) {
      printf("TRACE[A-enter] rankA=%d rankB=%d isSource=%d isDest=%d recvViaA=%d numSrc=%d numTgt=%d\n",
             params.myRankInA, params.myRankInB, (int)params.isSource, (int)params.isDest, (int)params.destRecvViaCommA,
             params.numSources, params.numTargets);
    }

    // First-domain leaders snapshot their commA baseline signals.
    if (params.isDest && params.destRecvViaCommA && params.numSources > 0) {
      if (threadIdx.x < params.numSources) {
        unsigned int signalIdx = params.sources[threadIdx.x].signalBase + blockIdx.x;
        initialSignals[threadIdx.x] = ginA.readSignal(signalIdx);
      }
    }
    __syncthreads();

    ncclBarrierSession<ncclCoopCta> barA{ncclCoopCta(), ncclTeamTagWorld(), ginA, blockIdx.x};
    barA.sync(ncclCoopCta(), cuda::memory_order_relaxed, ncclGinFenceLevel::Relaxed);

    // SOURCE (trainer): contiguous puts to the first-domain leaders.
    if (params.isSource) {
      int activeSrcWarps = min(MAX_SRC_WARPS, (int)(blockDim.x / 32));
      if (warpId < activeSrcWarps) {
        for (int t = warpId; t < params.numTargets; t += activeSrcWarps) {
          ncclReshardTargetInfo& target = params.targets[t];
          ncclReshardTransferPlan& plan = target.plan;
          if (plan.totalInnerTransfers == 0) {
            continue;
          }

          unsigned int signalIdx = (unsigned int)params.myRankInA * params.totalCtas + blockIdx.x;

          size_t totalSize = target.totalBytes;
          size_t bytesPerCta = (totalSize + params.totalCtas - 1) / params.totalCtas;
          size_t myStart = blockIdx.x * bytesPerCta;
          size_t myEnd = min(myStart + bytesPerCta, totalSize);

          if (laneId == 0) {
            if (myStart < totalSize) {
              size_t dstOff = target.windowOffset + plan.dstBaseOffset + myStart;
              size_t srcOff = params.myWindowOffset + plan.srcBaseOffset + myStart;
              size_t len = myEnd - myStart;
              // kernelTrace-gated bounds guard: if either side overflows the
              // staging window, log it and SKIP the put so the kernel does not
              // IMA (the device printf buffer then flushes on normal sync).
              // Validation will fail but the offending offsets are captured.
              if (params.kernelTrace &&
                  (srcOff + len > params.windowCapacityBytes || dstOff + len > params.windowCapacityBytes)) {
                printf("TRACE[A-put-OOB] rankA=%d t=%d to=%d srcOff=%llu dstOff=%llu len=%llu cap=%llu\n",
                       params.myRankInA, t, target.dstWorldRank, (unsigned long long)srcOff, (unsigned long long)dstOff,
                       (unsigned long long)len, (unsigned long long)params.windowCapacityBytes);
                ginA.signal(worldA, target.dstWorldRank, ncclGin_SignalInc{signalIdx});
              } else {
                ginA.put(worldA, target.dstWorldRank, params.windowA, dstOff, params.windowA, srcOff, len,
                         ncclGin_SignalInc{signalIdx});
              }
            } else {
              ginA.signal(worldA, target.dstWorldRank, ncclGin_SignalInc{signalIdx});
            }
          }
        }
      }
    }

    // DEST (first-domain leaders): wait for the trainer data on commA.
    // Ring forward + LSA fan-out happen later, in Phase B (commB).
    if (params.isDest && params.destRecvViaCommA && params.numSources > 0) {
      int warpsPerCta = blockDim.x / 32;
      int activeSources = min(params.numSources, min(warpsPerCta, (int)MAX_WARP_GROUPS));
      int warpsPerSource = warpsPerCta / activeSources;
      if (warpsPerSource < 1) {
        warpsPerSource = 1;
      }
      int mySourceGroup = warpId / warpsPerSource;
      int warpInGroup = warpId % warpsPerSource;
      int groupStartWarp = mySourceGroup * warpsPerSource;
      bool isActive = (mySourceGroup < activeSources);

      for (int srcOffset = mySourceGroup; srcOffset < params.numSources && isActive; srcOffset += activeSources) {
        ncclReshardSourceInfo& source = params.sources[srcOffset];
        int barrierId = mySourceGroup;
        ncclCoopWarpSpan warps(groupStartWarp, warpsPerSource, barrierId);

        unsigned int signalIdx = source.signalBase + blockIdx.x;
        if (warpInGroup == 0) {
          if (params.kernelTrace && blockIdx.x == 0 && laneId == 0) {
            printf("TRACE[A-wait-pre] rankA=%d srcOff=%d sigIdx=%u expect=%llu\n", params.myRankInA, srcOffset,
                   signalIdx, (unsigned long long)(initialSignals[srcOffset] + 1));
          }
          ginA.waitSignal(ncclCoopWarp(), signalIdx, initialSignals[srcOffset] + 1);
          if (params.kernelTrace && blockIdx.x == 0 && laneId == 0) {
            printf("TRACE[A-wait-done] rankA=%d srcOff=%d\n", params.myRankInA, srcOffset);
          }
        }
        warps.sync();
      }
    }

    __threadfence_system();
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 30, 5)
    __syncthreads();
#else
    ginA.flush(ncclCoopCta());
#endif
    barA.sync(ncclCoopCta(), cuda::memory_order_acquire, NCCLM2N_RESHARD_GIN_FINAL_FENCE);
    if (params.kernelTrace && blockIdx.x == 0 && threadIdx.x == 0) {
      printf("TRACE[A-done] rankA=%d\n", params.myRankInA);
    }
  }

  // ----------------------------------------------------------------------
  // Phase B — commB (ring forward + LSA fan-out; downstream receive).
  // ----------------------------------------------------------------------
  if (inB) {
    ncclGin ginB{devCommB, ginContextB};
    ncclTeam worldB = ncclTeamWorld(devCommB);
    ncclTeam lsaB = ncclTeamLsa(devCommB);

    if (params.kernelTrace && blockIdx.x == 0 && threadIdx.x == 0) {
      printf("TRACE[B-enter] rankB=%d slot=%d ctxB=%d numSrc=%d isDest=%d recvViaA=%d ringLast=%d ringNext=%d\n",
             params.myRankInB, params.slotIdx, ginContextB, params.numSources, (int)params.isDest,
             (int)params.destRecvViaCommA, (int)params.isRingLast, params.ringNextRankB);
    }

    // Downstream leaders (not on the first domain) snapshot their commB
    // baseline signals; first-domain leaders already received via commA.
    if (params.isDest && !params.destRecvViaCommA && params.numSources > 0) {
      if (threadIdx.x < params.numSources) {
        unsigned int signalIdx = params.sources[threadIdx.x].signalBase + signalSlotOffsetB + blockIdx.x;
        initialSignals[threadIdx.x] = ginB.readSignal(signalIdx);
      }
    }
    __syncthreads();

    // commB is the SHARED DevComm: concurrent parent PP comms partition not
    // only its GIN contexts/signals but ALSO its barrier pool.  Offset the
    // world-barrier id by slotIdx*totalCtas so slot 0 and slot 1 never alias
    // on the same barrier object (the DevComm is sized for
    // totalCtas*maxConcurrency barriers in reshardSplitEnsureResources).
    const int barIdB = params.slotIdx * params.totalCtas + (int)blockIdx.x;
    if (params.kernelTrace && blockIdx.x == 0 && threadIdx.x == 0) {
      printf("TRACE[B-bar0-pre] rankB=%d barId=%d\n", params.myRankInB, barIdB);
    }
    ncclBarrierSession<ncclCoopCta> barB{ncclCoopCta(), ncclTeamTagWorld(), ginB, (unsigned int)barIdB};
    barB.sync(ncclCoopCta(), cuda::memory_order_relaxed, ncclGinFenceLevel::Relaxed);
    if (params.kernelTrace && blockIdx.x == 0 && threadIdx.x == 0) {
      printf("TRACE[B-bar0-done] rankB=%d\n", params.myRankInB);
    }

    if (params.isDest && params.numSources > 0) {
      int warpsPerCta = blockDim.x / 32;
      int activeSources = min(params.numSources, min(warpsPerCta, (int)MAX_WARP_GROUPS));
      int warpsPerSource = warpsPerCta / activeSources;
      if (warpsPerSource < 1) {
        warpsPerSource = 1;
      }
      int mySourceGroup = warpId / warpsPerSource;
      int warpInGroup = warpId % warpsPerSource;
      int groupStartWarp = mySourceGroup * warpsPerSource;
      bool isActive = (mySourceGroup < activeSources);

      const int lsaStartRank = worldB.rank - lsaB.rank;
      char* localBuffer = (char*)ncclGetLocalPointer(params.windowB, params.myWindowOffset);

      for (int srcOffset = mySourceGroup; srcOffset < params.numSources && isActive; srcOffset += activeSources) {
        ncclReshardSourceInfo& source = params.sources[srcOffset];
        ncclReshardTransferPlan& plan = source.plan;
        int barrierId = mySourceGroup;
        ncclCoopWarpSpan warps(groupStartWarp, warpsPerSource, barrierId);

        unsigned int waitIdx = source.signalBase + signalSlotOffsetB + blockIdx.x;     // commB receive index
        unsigned int fwdIdx =
          params.sourceSignalBaseB[srcOffset] + signalSlotOffsetB + blockIdx.x;        // commB ring-forward index

        size_t totalSize = source.totalBytes;
        size_t bytesPerCta = (totalSize + params.totalCtas - 1) / params.totalCtas;
        size_t myStart = blockIdx.x * bytesPerCta;
        size_t myEnd = min(myStart + bytesPerCta, totalSize);

        // Downstream leaders wait for the upstream forward; first-domain
        // leaders already have their data from Phase A (commA).
        if (!params.destRecvViaCommA && warpInGroup == 0) {
          if (params.kernelTrace && blockIdx.x == 0 && laneId == 0) {
            printf("TRACE[B-wait-pre] rankB=%d srcOff=%d waitIdx=%u expect=%llu\n", params.myRankInB, srcOffset,
                   waitIdx, (unsigned long long)(initialSignals[srcOffset] + 1));
          }
          ginB.waitSignal(ncclCoopWarp(), waitIdx, initialSignals[srcOffset] + 1);
          if (params.kernelTrace && blockIdx.x == 0 && laneId == 0) {
            printf("TRACE[B-wait-done] rankB=%d srcOff=%d waitIdx=%u\n", params.myRankInB, srcOffset, waitIdx);
          }
        }
        warps.sync();

        // Ring forward to the next NVL domain's matching leader.
        if (!params.isRingLast && warpInGroup == 0 && laneId == 0) {
          if (params.kernelTrace && blockIdx.x == 0) {
            printf("TRACE[B-fwd] rankB=%d srcOff=%d fwdIdx=%u to=%d active=%d\n", params.myRankInB, srcOffset, fwdIdx,
                   params.ringNextRankB, (int)(myStart < totalSize));
          }
          if (myStart < totalSize) {
            size_t dstOffB = params.ringNextWindowOffset + plan.dstBaseOffset + myStart;
            size_t srcOffB = params.myWindowOffset + plan.dstBaseOffset + myStart;
            size_t lenB = myEnd - myStart;
            if (params.kernelTrace &&
                (srcOffB + lenB > params.windowCapacityBytes || dstOffB + lenB > params.windowCapacityBytes)) {
              printf("TRACE[B-fwd-OOB] rankB=%d srcOff=%d to=%d srcOffB=%llu dstOffB=%llu len=%llu cap=%llu\n",
                     params.myRankInB, srcOffset, params.ringNextRankB, (unsigned long long)srcOffB,
                     (unsigned long long)dstOffB, (unsigned long long)lenB,
                     (unsigned long long)params.windowCapacityBytes);
              ginB.signal(worldB, params.ringNextRankB, ncclGin_SignalInc{fwdIdx});
            } else {
              ginB.put(worldB, params.ringNextRankB, params.windowB, dstOffB, params.windowB, srcOffB, lenB,
                       ncclGin_SignalInc{fwdIdx});
            }
          } else {
            ginB.signal(worldB, params.ringNextRankB, ncclGin_SignalInc{fwdIdx});
          }
        }

        // LSA fan-out within this leader's NVL domain (commB LSA team).
        if (params.numLocalFollowers > 0 && myStart < totalSize) {
          int threadsInGroup = warpsPerSource * 32;
          int threadInGroup = warpInGroup * 32 + laneId;

          char* srcPtr = localBuffer + plan.dstBaseOffset + myStart;
          size_t chunkSize = myEnd - myStart;
          size_t dstByteOffset = plan.dstBaseOffset + myStart;

          lsaReplicateChunk(srcPtr, chunkSize, dstByteOffset, /*fallbackWindow=*/params.windowB,
                            params.localFollowerRanksB, params.localFollowerWindowOffsets, lsaStartRank,
                            params.numLocalFollowers, threadsInGroup, threadInGroup);
        }

        warps.sync();
      }
    }

    __threadfence_system();
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 30, 5)
    __syncthreads();
#else
    ginB.flush(ncclCoopCta());
#endif
    barB.sync(ncclCoopCta(), cuda::memory_order_acquire, NCCLM2N_RESHARD_GIN_FINAL_FENCE);
    if (params.kernelTrace && blockIdx.x == 0 && threadIdx.x == 0) {
      printf("TRACE[B-done] rankB=%d\n", params.myRankInB);
    }
  }
}

// ============================================================================
// Host: PACK split launch (split comms already formed + active).
// ============================================================================
ncclResult_t reshardLaunchPackSplit(const ReshardSplitComms* sc, void* stagingBuffer, size_t stagingCapacity,
                                    const ncclReshardParams* baseParams, int numCtas, cudaStream_t stream) {
  NCCL_M2N_CHECK_ARG(sc != nullptr && baseParams != nullptr && sc->active, -1,
                     "reshardLaunchPackSplit: active split state and base parameters are required");

  // Signal-space sizing:
  //   commA: trainers send to first-domain leaders, index = trainerRankInA
  //          * numCtas, trainerRankInA in [0, srcMeshSize).  Per-parent.
  //   commB: ring forward / receive index = srcShardIdx * numCtas,
  //          srcShardIdx in [0, srcShardCount).  commB partitions its signal
  //          space into maxConcurrency staging-bucket slots, each
  //          of stride signalsPerSlotB.  srcShardCount <= srcMeshSize, so
  //          srcMeshSize * numCtas is a safe per-slot upper bound (matches
  //          the commA sizing) and is independent of the per-tensor shard
  //          count — letting one DevComm serve every reshard.
  const int maxConcurrency = sc->slotCount;
  const int ginSignalCountA = sc->srcMeshSize * numCtas;
  const int signalsPerSlotB = sc->srcMeshSize * numCtas;
  const int ctxPerSlotB = reshardGetGinContextCount();

  // Defensive: the per-tensor commB signal footprint must fit one slot.
  if (sc->inB && baseParams->srcShardCount * numCtas > signalsPerSlotB) {
    NCCL_M2N_FAIL(ncclInvalidArgument, sc->parentRank,
                  "reshardLaunchPackSplit: commB signal overflow: srcShardCount=%d numCtas=%d exceeds "
                  "signalsPerSlotB=%d; reduce NCCL_RESHARD_NUM_CTAS or increase the signal allocation",
                  baseParams->srcShardCount, numCtas, signalsPerSlotB);
  }

  ncclWindow_t windowA = nullptr, windowB = nullptr;
  ncclDevComm devCommA, devCommB;
  ReshardDevCommUse devCommAUse, devCommBUse;
  ReshardCommBResourcePlan resourcesB{};
  NCCL_M2N_CHECK_ARG(reshardCheckedResourceProduct(signalsPerSlotB, maxConcurrency, &resourcesB.signals) &&
                       reshardCheckedResourceProduct(ctxPerSlotB, maxConcurrency, &resourcesB.contexts) &&
                       reshardCheckedResourceProduct(numCtas, maxConcurrency, &resourcesB.barriers),
                     sc->parentRank, "reshardLaunchPackSplit: commB resource count overflow");
  NCCL_M2N_CHECK(reshardSplitEnsureResources(
    sc, stagingBuffer, stagingCapacity, numCtas, RESHARD_DEVCOMM_BARRIER_HYBRID, ginSignalCountA,
    /*ginCounterCountA=*/0, resourcesB, stream, &windowA,
    &windowB, &devCommA, &devCommAUse, &devCommB, &devCommBUse));

  ncclReshardParamsSplit sp;
  buildSplitReshardParams(baseParams, sc, numCtas, windowA, windowB, &sp);

  // commB slot partition: this parent comm's reshards use GIN contexts
  // [slotIdx*ctxPerSlotB, +ctxPerSlotB) and add slotIdx*signalsPerSlotB to
  // every commB signal index (see the kernel).  commA is per-parent and
  // is not slotted.
  sp.slotIdx = sc->slotIdx;
  sp.ctxPerSlotB = ctxPerSlotB;
  sp.signalsPerSlotB = signalsPerSlotB;

  // Registered staging-window size: bounds the src-read and dst-write
  // offsets of every put (windowA/windowB are symmetric, same capacity).
  sp.windowCapacityBytes = stagingCapacity;

  // Debug hang-localization: NCCL_RESHARD_SPLIT_KERNEL_TRACE=1 makes the
  // split kernel printf at each commB (Phase B) barrier/waitSignal boundary.
  {
    const char* kt = getenv("NCCL_RESHARD_SPLIT_KERNEL_TRACE");
    sp.kernelTrace = (kt != nullptr && atoi(kt) != 0) ? 1 : 0;
  }

  // Host-side staging-window bounds check (kernelTrace-gated): before the
  // kernel launches, flag any source-side put whose dst-write or src-read
  // extent overflows the staging window.  This fires on the host so the
  // diagnostic is guaranteed even if the kernel later faults (IMA).  Each
  // target's full transfer spans windowOffset + plan.{dst,src}BaseOffset ..
  // + totalBytes (the kernel partitions totalBytes across CTAs, so the last
  // CTA reaches that extent).
  if (sp.kernelTrace && sp.isSource) {
    for (int t = 0; t < sp.numTargets; t++) {
      const ncclReshardTargetInfo& tgt = sp.targets[t];
      if (tgt.plan.totalInnerTransfers == 0) continue;
      size_t dstEnd = tgt.windowOffset + tgt.plan.dstBaseOffset + tgt.totalBytes;
      size_t srcEnd = sp.myWindowOffset + tgt.plan.srcBaseOffset + tgt.totalBytes;
      if (dstEnd > stagingCapacity || srcEnd > stagingCapacity) {
        RESHARD_WARN(sc->parentRank,
                     "split-bounds OOB: rankInA=%d target=%d dstWorldRankA=%d windowOffset=%zu dstBaseOffset=%zu "
                     "srcBaseOffset=%zu totalBytes=%zu myWindowOffset=%zu => dstEnd=%zu srcEnd=%zu capacity=%zu "
                     "(%s overflow)",
                     sp.myRankInA, t, tgt.dstWorldRank, tgt.windowOffset, tgt.plan.dstBaseOffset,
                     tgt.plan.srcBaseOffset, tgt.totalBytes, sp.myWindowOffset, dstEnd, srcEnd, stagingCapacity,
                     (dstEnd > stagingCapacity && srcEnd > stagingCapacity) ?
                       "dst+src" :
                       (dstEnd > stagingCapacity ? "dst-write" : "src-read"));
      }
    }
  }
  // Host-side dump of the commB ring plan (one INFO per reshard) so the device
  // trace can be cross-checked against the host-computed forward/wait indices.
  if (sp.kernelTrace) {
    RESHARD_INFO(sc->parentRank,
                 "split-ringplan: inB=%d isDest=%d destRecvViaCommA=%d isRingLast=%d ringNextRankB=%d numSources=%d "
                 "slot=%d sigPerSlotB=%d src0_signalBase=%u src0_fwdBaseB=%u",
                 (int)sc->inB, (int)sp.isDest, (int)sp.destRecvViaCommA, (int)sp.isRingLast, sp.ringNextRankB,
                 sp.numSources, sp.slotIdx, sp.signalsPerSlotB, (sp.numSources > 0 ? sp.sources[0].signalBase : 0u),
                 (sp.numSources > 0 ? sp.sourceSignalBaseB[0] : 0u));
  }

  const int threadsPerCta = DEFAULT_KERNEL_MAX_NTHREADS;
  reshardKernelUserWindowSplit<<<numCtas, threadsPerCta, 0, stream>>>(sp, devCommA, devCommB);
  cudaError_t launchErr = cudaGetLastError();
  if (launchErr != cudaSuccess) {
    NCCL_M2N_FAIL(ncclSystemError, sc->parentRank, "reshardLaunchPackSplit kernel launch failed: %s [numCtas=%d]",
                  cudaGetErrorString(launchErr), numCtas);
  }
  NCCL_M2N_CHECK(reshardRecordDevCommUse(&devCommAUse, stream));
  NCCL_M2N_CHECK(reshardRecordDevCommUse(&devCommBUse, stream));

  RESHARD_INFO(sc->parentRank,
               "split-launch: inA=%d inB=%d rankInA=%d rankInB=%d srcLsa=%d genLsa=%d numGenDomains=%d "
               "sigA=%d sigPerSlotB=%d slot=%d/%d ctxPerSlotB=%d numCtas=%d",
               (int)sc->inA, (int)sc->inB, sc->rankInA, sc->rankInB, sc->srcLsaSize, sc->lsaSize, sc->numGenDomains,
               ginSignalCountA, signalsPerSlotB, sc->slotIdx, maxConcurrency, ctxPerSlotB, numCtas);

  return ncclSuccess;
}
