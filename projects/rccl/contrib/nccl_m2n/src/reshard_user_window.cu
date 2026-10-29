/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

/*************************************************************************
 * Tensor Reshard — PACK Path
 *
 * PACK stages data in a library-managed NCCL window, transfers it
 * with the hierarchical kernel below, then unpacks it into the destination.
 * ncclReshardWithWindow is retained as a compatibility alias for ncclReshard;
 * the caller-provided window is no longer used for transport.
 ************************************************************************/

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <vector>

#include "cuda_runtime.h"
#include "nccl.h"
#include "nccl_device.h"

#include "nccl_m2n.h"
#include "reshard_types.h"
#include "m2n_checks.h"
#include "m2n_checked_math.h"
#include "m2n_log.h"
#include "reshard_call_setup.h"
#include "reshard_internal.h"
#include "reshard_kernels.cuh"
#include "reshard_split.h"

#ifdef NCCL_M2N_TESTING
static thread_local size_t gFusedSubmissionCount = 0;

void reshardResetFusedSubmissionCountForTest() {
  gFusedSubmissionCount = 0;
}

size_t reshardGetFusedSubmissionCountForTest() {
  return gFusedSubmissionCount;
}
#endif

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 30, 5)
#define NCCL_RESHARD_GIN_FINAL_FENCE (ncclGinFenceLevel::Put | ncclGinFenceLevel::Get)
#else
#define NCCL_RESHARD_GIN_FINAL_FENCE ncclGinFenceLevel::Relaxed
#endif

/* The cooperative-group types below (ncclCoopWarp, ncclCoopWarpSpan) account for
 * threads in units of the target's native warp/wavefront: 64 lanes on AMD GFX9,
 * 32 on NVIDIA. The warp partitioning in these kernels must use that same unit.
 * A ncclCoopWarpSpan built from 32-thread warps on a 64-lane wave claims twice
 * as many warps as the span holds leaders for, and its sync() then waits for
 * arrivals that never happen -> device-side deadlock on AMD. Upstream hardcodes
 * 32; key off the target's real warp size (WARP_SIZE from nccl_device/coop.h). */
#if defined(WARP_SIZE)
#define NCCL_RESHARD_WARP_SIZE WARP_SIZE
#else
#define NCCL_RESHARD_WARP_SIZE 32
#endif

// ============================================================================
// RING (hierarchical) kernel used by PACK
//
// Uses the GLOBAL window for local-buffer access and LSA fan-out, resolving
// peer pointers via world-rank arithmetic.
// ============================================================================

__global__ __launch_bounds__(DEFAULT_KERNEL_MAX_NTHREADS, 1) void reshardKernelUserWindow(ncclReshardParams params,
                                                                                          struct ncclDevComm devComm) {
  int ginContext = reshardMapCtaToGinContext((int)blockIdx.x, (int)gridDim.x, (int)devComm.ginContextCount);
  ncclGin gin{devComm, ginContext};

  ncclTeam world = ncclTeamWorld(devComm);
  ncclTeam lsa = ncclTeamLsa(devComm);

  int warpId = threadIdx.x / NCCL_RESHARD_WARP_SIZE;
  int laneId = threadIdx.x % NCCL_RESHARD_WARP_SIZE;

  /* Required before the first ncclCoopWarpSpan::sync() below: where the target
   * has no hardware named barriers the span is emulated with a shared
   * sense-reversing barrier whose slots must be zeroed first. No-op on targets
   * that do have named barriers. */
  ncclCoopNamedBarrierInit();

  // [USER-WINDOW] Local pointer comes from the GLOBAL window.  Offset is
  // zero by contract but
  // we still pass params.myWindowOffset for symmetry.
  char* localBuffer = (char*)ncclGetLocalPointer(params.window, params.myWindowOffset);

  __shared__ uint64_t initialSignals[MAX_SOURCES];
  // Compile-time guard: shared-array dim must be at least the prep-side cap
  // on params.numSources.
  static_assert(sizeof(initialSignals) / sizeof(initialSignals[0]) >= MAX_SOURCES,
                "initialSignals[] must be sized at least MAX_SOURCES — "
                "kernel reads initialSignals[i] for i < params.numSources, "
                "and prepareReshardParams caps params.numSources at MAX_SOURCES");

  // Read initial signals (dest ranks only)
  if (params.isDest && params.numSources > 0) {
    if (threadIdx.x < params.numSources) {
      unsigned int signalIdx = params.sources[threadIdx.x].signalBase + blockIdx.x;
      initialSignals[threadIdx.x] = gin.readSignal(signalIdx);
    }
  }
  __syncthreads();

  // Initial barrier
  ncclBarrierSession<ncclCoopCta> bar{ncclCoopCta(), ncclTeamTagWorld(), gin, blockIdx.x};
  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed, ncclGinFenceLevel::Relaxed);

  // SOURCE: send data to targets
  if (params.isSource) {
    int activeSrcWarps = min(MAX_SRC_WARPS, (int)(blockDim.x / NCCL_RESHARD_WARP_SIZE));
    if (warpId < activeSrcWarps) {
      for (int t = warpId; t < params.numTargets; t += activeSrcWarps) {
        ncclReshardTargetInfo& target = params.targets[t];
        ncclReshardTransferPlan& plan = target.plan;

        if (plan.totalInnerTransfers == 0) continue;

        unsigned int signalIdx = params.mySignalBase + blockIdx.x;

        if (target.isContiguous) {
          size_t totalSize = target.totalBytes;
          size_t bytesPerCta = (totalSize + params.totalCtas - 1) / params.totalCtas;
          size_t myStart = blockIdx.x * bytesPerCta;
          size_t myEnd = min(myStart + bytesPerCta, totalSize);

          if (laneId == 0) {
            if (myStart < totalSize) {
              gin.put(world, target.dstWorldRank, params.window, target.windowOffset + plan.dstBaseOffset + myStart,
                      params.window, params.myWindowOffset + plan.srcBaseOffset + myStart, myEnd - myStart,
                      ncclGin_SignalInc{signalIdx});
            } else {
              gin.signal(world, target.dstWorldRank, ncclGin_SignalInc{signalIdx});
            }
          }
        } else {
          size_t itersPerCta = (plan.totalInnerTransfers + params.totalCtas - 1) / params.totalCtas;
          size_t myIterStart = blockIdx.x * itersPerCta;
          size_t myIterEnd = min(myIterStart + itersPerCta, plan.totalInnerTransfers);
          size_t myTotalBytes = (myIterEnd - myIterStart) * plan.innerSize;
          const size_t csb = params.chunkSizeBytes;
          size_t numChunks = (myTotalBytes + csb - 1) / csb;

          for (size_t chunk = 0; chunk < numChunks; chunk++) {
            size_t byteStart = chunk * csb;
            size_t remaining = myTotalBytes - byteStart;
            size_t thisBytes = (csb < remaining) ? csb : remaining;

            if (laneId == 0) {
              emitStridedChunkPuts(gin, world, target.dstWorldRank, params.window, plan, params.ndims, myIterStart,
                                   byteStart, thisBytes, signalIdx,
                                   /*useDstAsSrc=*/false, params.myWindowOffset, target.windowOffset);
            }
          }
        }
      }
    }
  }

  // DEST: receive and replicate
  if (params.isDest && params.numSources > 0) {
    int warpsPerCta = blockDim.x / NCCL_RESHARD_WARP_SIZE;
    int activeSources = min(params.numSources, min(warpsPerCta, (int)MAX_WARP_GROUPS));
    int warpsPerSource = warpsPerCta / activeSources;
    if (warpsPerSource < 1) warpsPerSource = 1;

    int mySourceGroup = warpId / warpsPerSource;
    int warpInGroup = warpId % warpsPerSource;
    int groupStartWarp = mySourceGroup * warpsPerSource;
    bool isActive = (mySourceGroup < activeSources);

    // [USER-WINDOW] lsaStartRank = world rank of LSA-rank-0 on this
    // rank's LSA team within the input comm.  Used to translate world
    // ranks to LSA ranks for ncclGetLsaPointer on the global window.
    const int lsaStartRank = world.rank - lsa.rank;

    for (int srcOffset = mySourceGroup; srcOffset < params.numSources && isActive; srcOffset += activeSources) {
      ncclReshardSourceInfo& source = params.sources[srcOffset];
      ncclReshardTransferPlan& plan = source.plan;

      int barrierId = mySourceGroup;
      ncclCoopWarpSpan warps(groupStartWarp, warpsPerSource, barrierId);

      unsigned int signalIdx = source.signalBase + blockIdx.x;

      if (source.isContiguous) {
        size_t totalSize = source.totalBytes;
        size_t bytesPerCta = (totalSize + params.totalCtas - 1) / params.totalCtas;
        size_t myStart = blockIdx.x * bytesPerCta;
        size_t myEnd = min(myStart + bytesPerCta, totalSize);

        if (warpInGroup == 0) gin.waitSignal(ncclCoopWarp(), signalIdx, initialSignals[srcOffset] + 1);
        warps.sync();

        // Ring forward
        if (!params.isRingLast && warpInGroup == 0 && laneId == 0) {
          if (myStart < totalSize) {
            gin.put(world, params.ringNextWorldRank, params.window,
                    params.ringNextWindowOffset + plan.dstBaseOffset + myStart, params.window,
                    params.myWindowOffset + plan.dstBaseOffset + myStart, myEnd - myStart,
                    ncclGin_SignalInc{signalIdx});
          } else {
            gin.signal(world, params.ringNextWorldRank, ncclGin_SignalInc{signalIdx});
          }
        }

        // [USER-WINDOW] LSA fan-out via the global window keyed by
        // world-rank arithmetic.
        if (params.numLocalFollowers > 0 && myStart < totalSize) {
          int threadsInGroup = warpsPerSource * NCCL_RESHARD_WARP_SIZE;
          int threadInGroup = warpInGroup * NCCL_RESHARD_WARP_SIZE + laneId;

          char* srcPtr = localBuffer + plan.dstBaseOffset + myStart;
          size_t chunkSize = myEnd - myStart;
          size_t dstByteOffset = plan.dstBaseOffset + myStart;

          lsaReplicateChunk(srcPtr, chunkSize, dstByteOffset,
                            /*fallbackWindow=*/params.window, params.localFollowerWorldRanks,
                            params.localFollowerWindowOffsets, lsaStartRank, params.numLocalFollowers, threadsInGroup,
                            threadInGroup);
        }

        warps.sync();
      } else {
        size_t itersPerCta = (plan.totalInnerTransfers + params.totalCtas - 1) / params.totalCtas;
        size_t myIterStart = blockIdx.x * itersPerCta;
        size_t myIterEnd = min(myIterStart + itersPerCta, plan.totalInnerTransfers);
        size_t myTotalBytes = (myIterEnd - myIterStart) * plan.innerSize;
        const size_t csb = params.chunkSizeBytes;
        size_t numChunks = (myTotalBytes + csb - 1) / csb;

        for (size_t chunk = 0; chunk < numChunks; chunk++) {
          size_t byteStart = chunk * csb;
          size_t remaining = myTotalBytes - byteStart;
          size_t thisBytes = (csb < remaining) ? csb : remaining;

          if (warpInGroup == 0) gin.waitSignal(ncclCoopWarp(), signalIdx, initialSignals[srcOffset] + chunk + 1);
          warps.sync();

          // Ring forward this chunk
          if (!params.isRingLast && warpInGroup == 0 && laneId == 0) {
            emitStridedChunkPuts(gin, world, params.ringNextWorldRank, params.window, plan, params.ndims, myIterStart,
                                 byteStart, thisBytes, signalIdx,
                                 /*useDstAsSrc=*/true, params.myWindowOffset, params.ringNextWindowOffset);
          }

          // [USER-WINDOW] LSA fan-out via the global window for
          // strided chunks.
          if (params.numLocalFollowers > 0 && thisBytes > 0) {
            int threadsInGroup = warpsPerSource * NCCL_RESHARD_WARP_SIZE;
            int threadInGroup = warpInGroup * NCCL_RESHARD_WARP_SIZE + laneId;

            const size_t inner = plan.innerSize;
            size_t lsaIter = byteStart / inner;
            size_t lsaOffInIter = byteStart % inner;
            size_t lsaRemaining = thisBytes;

            while (lsaRemaining > 0) {
              size_t avail = inner - lsaOffInIter;
              size_t piece = (avail < lsaRemaining) ? avail : lsaRemaining;

              size_t srcOff, dstOff;
              computeTransferOffset(plan, myIterStart + lsaIter, params.ndims, &srcOff, &dstOff);

              char* piecePtr = localBuffer + dstOff + lsaOffInIter;
              size_t dstByteOffset = dstOff + lsaOffInIter;

              lsaReplicateChunk(piecePtr, piece, dstByteOffset,
                                /*fallbackWindow=*/params.window, params.localFollowerWorldRanks,
                                params.localFollowerWindowOffsets, lsaStartRank, params.numLocalFollowers,
                                threadsInGroup, threadInGroup);

              lsaRemaining -= piece;
              lsaIter++;
              lsaOffInIter = 0;
            }
          }
        }
      }
    }
  }

  __threadfence_system();
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 30, 5)
  __syncthreads();
#else
  // Older GIN barriers do not drain puts; flush also synchronizes the CTA.
  gin.flush(ncclCoopCta());
#endif

  // Final barrier
  bar.sync(ncclCoopCta(), cuda::memory_order_acquire, NCCL_RESHARD_GIN_FINAL_FENCE);
}

// ============================================================================
// Host: ncclReshardWithWindow
// ============================================================================

struct PackStagingEventGuard {
  ncclComm_t comm = nullptr;
  cudaStream_t stream = nullptr;

  ~PackStagingEventGuard() {
    if (comm != nullptr) {
      NCCL_M2N_CHECK_WARN(packStagingRecordEvent(comm, stream));
    }
  }

  void arm(ncclComm_t guardedComm, cudaStream_t guardedStream) {
    comm = guardedComm;
    stream = guardedStream;
  }

  ncclResult_t record() {
    ncclResult_t result = packStagingRecordEvent(comm, stream);
    if (result == ncclSuccess) {
      comm = nullptr;
    }
    return result;
  }
};

extern "C" ncclResult_t ncclReshardWithWindow(ncclM2nHandle_t handle, ncclComm_t comm, ncclWindow_t window,
                                               const ncclDistTensor_t* src, const ncclDistTensor_t* dst,
                                               cudaStream_t stream) {
  (void)window;
  return ncclReshard(handle, comm, src, dst, stream);
}

// ============================================================================
// PACK copy mode
//
// Pack each destination's bytes contiguously into the reused PACK staging
// buffer (CE cudaMemcpy3D), transfer with the hierarchical
// user-window kernel over contiguous plans, then unpack into the user dst
// buffer (CE cudaMemcpy3D).
//
// Two plans per (src,dst) pair:
//   * the original plan drives pack with source strides and unpack with
//     destination strides;
//   * a synthesized contiguous plan drives the transfer kernel.
//
// The dst-side contiguous offset (rx) is keyed on the source shard index via
// packPackedRxOffset() so it is rank-independent: the cross-NVL ring forwards
// and LSA fan-out all land a given source's data at the same staging offset
// on every replica of a dst shard (the kernel forwards at plan.dstBaseOffset).
// ============================================================================

static inline bool packPlanPairBytes(const ncclReshardTransferPlan& p, size_t* bytes) {
  return m2nCheckedMulSize(p.totalInnerTransfers, p.innerSize, bytes);
}

// Build one cudaMemcpy3D descriptor from a transfer plan.  Pass nullptr for
// whichever side is the contiguous staging region.  Caps at 3-D, i.e.
// numOuterLoops <= 2 (accepted limitation for now; higher-rank cross-dim
// plans are not yet handled here).
static cudaMemcpy3DParms packBuildCopy(void* srcPtr, void* dstPtr, const ncclReshardTransferPlan& plan,
                                     const size_t* srcStrides /* nullable: contig */,
                                     const size_t* dstStrides /* nullable: contig */) {
  const size_t width = plan.innerSize;
  size_t height = 1;
  size_t depth = 1;
  if (plan.numOuterLoops == 1) {
    height = plan.outerCounts[0];
  } else if (plan.numOuterLoops >= 2) {
    height = plan.outerCounts[1];
    depth = plan.outerCounts[0];
  }

  size_t srcPitch = width;
  size_t srcSliceHeight = height;
  if (srcStrides != nullptr) {
    srcPitch = (plan.numOuterLoops >= 1) ? srcStrides[plan.numOuterLoops - 1] : width;
    srcSliceHeight = (depth > 1 && srcStrides[1] > 0) ? srcStrides[0] / srcStrides[1] : height;
  }
  size_t dstPitch = width;
  size_t dstSliceHeight = height;
  if (dstStrides != nullptr) {
    dstPitch = (plan.numOuterLoops >= 1) ? dstStrides[plan.numOuterLoops - 1] : width;
    dstSliceHeight = (depth > 1 && dstStrides[1] > 0) ? dstStrides[0] / dstStrides[1] : height;
  }

  cudaMemcpy3DParms copy = {};
  copy.kind = cudaMemcpyDeviceToDevice;
  copy.srcPtr = make_cudaPitchedPtr(srcPtr, srcPitch, width, srcSliceHeight);
  copy.dstPtr = make_cudaPitchedPtr(dstPtr, dstPitch, width, dstSliceHeight);
  copy.extent = make_cudaExtent(width, height, depth);
  return copy;
}

template <typename Params>
static ncclResult_t packPackedRxOffset(const Params& p, int dstShard, int srcShardLimit, size_t rxBase, size_t* offset) {
  size_t off = rxBase;
  for (int s = 0; s < srcShardLimit; s++) {
    ncclReshardTransferPlan probe;
    NCCL_M2N_CHECK(computeTransferPlanChecked(p.srcDims, p.srcStrides, p.srcShardLayout, s, p.dstDims,
                                              p.dstStrides, p.dstShardLayout, dstShard, p.ndims,
                                              p.elementsPerChunk, &probe));
    size_t pairBytes = 0;
    NCCL_M2N_CHECK_ARG(packPlanPairBytes(probe, &pairBytes) && m2nCheckedAddSize(off, pairBytes, &off), -1,
                       "reshardCopyPack: packed receive offset overflow for dstShard=%d srcShard=%d",
                       dstShard, s);
  }
  *offset = off;
  return ncclSuccess;
}

/* NCCL lazily initializes the host-RMA CE path with a communicator collective.
 * A ring signal/wait makes every rank enter that initialization in lockstep,
 * including ranks with no payload, and leaves no unmatched signal count. */
static ncclResult_t packLsaHputWarmup(ncclComm_t comm, int worldRank, int worldSize, cudaStream_t stream) {
  if (worldSize < 2) {
    return ncclSuccess;
  }
  const int next = (worldRank + 1) % worldSize;
  const int previous = (worldRank - 1 + worldSize) % worldSize;
  M2nApiUnlock apiUnlock;
  NCCL_M2N_CHECK(ncclSignal(next, 0, 0, 0, comm, stream));
  NCCL_M2N_CHECK(m2nWaitCommReady(comm));
  ncclWaitSignalDesc_t wait = {1, previous, 0, 0};
  NCCL_M2N_CHECK(ncclWaitSignal(1, &wait, comm, stream));
  NCCL_M2N_CHECK(m2nWaitCommReady(comm));
  return ncclSuccess;
}

struct PackLsaHputProtocolGuard {
  ncclComm_t comm = nullptr;

  ~PackLsaHputProtocolGuard() {
    if (comm != nullptr) {
      reshardRequireResourceQuarantine();
      RESHARD_WARN(-1,
                   "PACK host-RMA protocol failed after entry for communicator %p; the communicator epoch "
                   "is not reusable",
                   (void*)comm);
    }
  }

  void arm(ncclComm_t guardedComm) {
    comm = guardedComm;
  }

  void complete() {
    comm = nullptr;
  }
};

static ncclResult_t packGetOrRegisterStagingWindow(ncclComm_t comm, void* staging, size_t capacity,
                                                 ncclWindow_t* outWindow) {
  NCCL_M2N_CHECK_ARG(outWindow != nullptr, -1, "PACK staging-window output must be non-null");
  ncclWindow_t* cachedWin =
    findCachedInternalWindowByPtr(comm, staging, capacity, RESHARD_INTERNAL_WINDOW_PACK);
  if (cachedWin != nullptr) {
    *outWindow = *cachedWin;
    return ncclSuccess;
  }

  ncclWindow_t window = nullptr;
  {
    M2nApiUnlock apiUnlock;
    NCCL_M2N_CHECK(ncclCommWindowRegister(comm, staging, capacity, &window, NCCL_WIN_COLL_SYMMETRIC));
    NCCL_M2N_CHECK(m2nWaitCommReady(comm));
  }
  NCCL_M2N_CHECK(cacheInternalWindow(comm, staging, capacity, RESHARD_INTERNAL_WINDOW_PACK, window));
  *outWindow = window;
  return ncclSuccess;
}

/* Entire-communicator LSA fast path: use the DIRECT ownership plan only to
 * enumerate final source/destination edges. Each unique destination shard is
 * packed once, then the same contiguous bytes are copied to every destination
 * replica assigned to this source replica. */
static ncclResult_t reshardCopyPackLsaHput(ncclComm_t comm, const ncclDistTensor_t* src,
                                                 const size_t srcDimsBytes[], const ncclDistTensor_t* dst,
                                                 const size_t dstDimsBytes[], void* staging, size_t stagingCapacity,
                                                 ncclWindow_t stagingWindow, size_t txLimit, size_t rxBase,
                                                 size_t elementsPerChunk, int worldRank, int worldSize,
                                                 cudaStream_t workStream) {
  std::vector<size_t> allWindowOffsets(worldSize, 0);
  ncclReshardDirectParams params{};
  NCCL_M2N_CHECK(validateReshardPlanLimits(worldRank, src, srcDimsBytes, dst, dstDimsBytes, elementsPerChunk,
                                           RESHARD_ALGO_DIRECT, worldSize));
  NCCL_M2N_CHECK(prepareDirectReshardParams(worldRank, src, srcDimsBytes, dst, dstDimsBytes, stagingWindow,
                                            elementsPerChunk, 1, 0, worldSize, allWindowOffsets.data(), &params));
  params.myWindowOffset = 0;

  ncclReshardTransferPlan targetPlans[MAX_DIRECT_TARGETS];
  size_t txOffset[MAX_DIRECT_TARGETS] = {0};
  size_t txBytes[MAX_DIRECT_TARGETS] = {0};
  size_t peerRx[MAX_DIRECT_TARGETS] = {0};
  size_t shardTxOffset[MAX_DIRECT_TARGETS];
  bool packTarget[MAX_DIRECT_TARGETS] = {false};
  ncclReshardTransferPlan unpackPlans[MAX_DIRECT_SOURCES] = {};
  size_t unpackBytes[MAX_DIRECT_SOURCES] = {0};
  size_t unpackRx[MAX_DIRECT_SOURCES] = {0};
  for (size_t& offset : shardTxOffset) offset = SIZE_MAX;

  if (params.isSource) {
    size_t cursor = 0;
    for (int t = 0; t < params.numTargets; t++) {
      const ncclReshardDirectTargetInfo& target = params.targets[t];
      NCCL_M2N_CHECK_ARG(target.dstShardIdx >= 0 && target.dstShardIdx < MAX_DIRECT_TARGETS, worldRank,
                         "PACK LSA target shard index %d exceeds capacity %d", target.dstShardIdx,
                         MAX_DIRECT_TARGETS);
      targetPlans[t] = target.plan;
      NCCL_M2N_CHECK_ARG(packPlanPairBytes(targetPlans[t], &txBytes[t]), worldRank,
                         "PACK LSA target %d byte count overflow", t);

      if (shardTxOffset[target.dstShardIdx] == SIZE_MAX) {
        shardTxOffset[target.dstShardIdx] = cursor;
        packTarget[t] = true;
        size_t txEnd = 0;
        NCCL_M2N_CHECK_ARG(m2nCheckedAddSize(cursor, txBytes[t], &txEnd) && txEnd <= txLimit, worldRank,
                           "PACK LSA packed source range exceeds %zu bytes", txLimit);
        cursor = txEnd;
      }
      txOffset[t] = shardTxOffset[target.dstShardIdx];
      NCCL_M2N_CHECK(
        packPackedRxOffset(params, target.dstShardIdx, params.mySrcShardIdx, rxBase, &peerRx[t]));
      size_t peerEnd = 0;
      NCCL_M2N_CHECK_ARG(m2nCheckedAddSize(peerRx[t], txBytes[t], &peerEnd) && peerEnd <= stagingCapacity, worldRank,
                         "PACK LSA target %d peer range exceeds %zu bytes", t, stagingCapacity);
    }
  }

  /* Complete deterministic destination validation before entering the signal
   * protocol. After GRANT/ARRIVAL work starts, an error is fail-stop for this
   * communicator epoch and the call must not be retried. */
  if (params.isDest) {
    for (int srcShard = 0; srcShard < params.srcShardCount; srcShard++) {
      ncclReshardTransferPlan& plan = unpackPlans[srcShard];
      NCCL_M2N_CHECK(computeTransferPlanChecked(params.srcDims, params.srcStrides, params.srcShardLayout,
                                                srcShard, params.dstDims, params.dstStrides,
                                                params.dstShardLayout, params.myDstShardIdx, params.ndims,
                                                params.elementsPerChunk, &plan));
      NCCL_M2N_CHECK_ARG(packPlanPairBytes(plan, &unpackBytes[srcShard]), worldRank,
                         "PACK LSA unpack shard %d byte count overflow", srcShard);
      if (unpackBytes[srcShard] == 0) {
        continue;
      }
      size_t rxEnd = 0;
      NCCL_M2N_CHECK(
        packPackedRxOffset(params, params.myDstShardIdx, srcShard, rxBase, &unpackRx[srcShard]));
      NCCL_M2N_CHECK_ARG(m2nCheckedAddSize(unpackRx[srcShard], unpackBytes[srcShard], &rxEnd) &&
                           rxEnd <= stagingCapacity,
                         worldRank, "PACK LSA unpack shard %d range exceeds %zu bytes", srcShard,
                         stagingCapacity);
    }
  }

  bool rmaWarmed = false;
  NCCL_M2N_CHECK(getPackRmaWarmed(comm, &rmaWarmed));

  PackLsaHputProtocolGuard protocolGuard;
  protocolGuard.arm(comm);

  if (!rmaWarmed) {
    NCCL_M2N_CHECK(packLsaHputWarmup(comm, worldRank, worldSize, workStream));
    NCCL_M2N_CHECK(setPackRmaWarmed(comm, /*warmed=*/true));
  }

  /* A grant is a one-call lease for this destination's physical staging slot.
   * ensurePackStagingBuffer() has already queued any wait for the previous local
   * user's completion event on workStream, so these signals cannot reach a
   * source until the destination's RX area is safe to overwrite. The grant
   * travels on the current communicator and therefore does not need to identify
   * which communicator previously occupied the physical slot. */
  if (params.isDest && params.numSources > 0) {
    M2nApiUnlock apiUnlock;
    NCCL_M2N_CHECK(ncclGroupStart());
    ncclResult_t signalResult = ncclSuccess;
    for (int sourceIdx = 0; sourceIdx < params.numSources; sourceIdx++) {
      const int sourceRank = src->mesh->startRank + (int)params.sources[sourceIdx].signalBase;
      signalResult = ncclSignal(sourceRank, 0, 0, 0, comm, workStream);
      if (signalResult != ncclSuccess && signalResult != ncclInProgress) break;
    }
    const ncclResult_t groupResult = ncclGroupEnd();
    NCCL_M2N_CHECK(signalResult);
    NCCL_M2N_CHECK(groupResult);
    NCCL_M2N_CHECK(m2nWaitCommReady(comm));
  }

  if (params.isSource && src->dataPtr != nullptr) {
    for (int t = 0; t < params.numTargets; t++) {
      if (!packTarget[t] || txBytes[t] == 0) {
        continue;
      }
      cudaMemcpy3DParms cp =
        packBuildCopy((char*)src->dataPtr + targetPlans[t].srcBaseOffset, (char*)staging + txOffset[t], targetPlans[t],
                    targetPlans[t].outerSrcStrides, /*dst contig=*/nullptr);
      NCCL_M2N_CUDACHECK(cudaMemcpy3DAsync(&cp, workStream));
    }
  }

  /* Packing may overlap the remote destination becoming available, but no put
   * can begin until every target grants access to its receive slot. Signal
   * index/context zero carries one balanced meaning per direction:
   * destination-to-source is GRANT; source-to-destination is ARRIVAL. */
  if (params.isSource && params.numTargets > 0) {
    ncclWaitSignalDesc_t grants[MAX_DIRECT_TARGETS];
    for (int targetIdx = 0; targetIdx < params.numTargets; targetIdx++) {
      grants[targetIdx] = {1, params.targets[targetIdx].dstWorldRank, 0, 0};
    }
    M2nApiUnlock apiUnlock;
    NCCL_M2N_CHECK(ncclWaitSignal(params.numTargets, grants, comm, workStream));
    NCCL_M2N_CHECK(m2nWaitCommReady(comm));
  }

  {
    M2nApiUnlock apiUnlock;
    NCCL_M2N_CHECK(ncclGroupStart());
    ncclResult_t putResult = ncclSuccess;
    if (params.isSource) {
      for (int targetIdx = 0; targetIdx < params.numTargets; targetIdx++) {
        if (txBytes[targetIdx] == 0) continue;
        putResult = ncclPutSignal((char*)staging + txOffset[targetIdx], txBytes[targetIdx], ncclUint8,
                                  params.targets[targetIdx].dstWorldRank, stagingWindow, peerRx[targetIdx], 0, 0, 0,
                                  comm, workStream);
        if (putResult != ncclSuccess && putResult != ncclInProgress) break;
      }
    }
    const ncclResult_t groupResult = ncclGroupEnd();
    NCCL_M2N_CHECK(putResult);
    NCCL_M2N_CHECK(groupResult);
    NCCL_M2N_CHECK(m2nWaitCommReady(comm));
  }

  if (params.isDest && params.numSources > 0) {
    ncclWaitSignalDesc_t arrivals[MAX_DIRECT_SOURCES];
    for (int sourceIdx = 0; sourceIdx < params.numSources; sourceIdx++) {
      const int sourceRank = src->mesh->startRank + (int)params.sources[sourceIdx].signalBase;
      arrivals[sourceIdx] = {1, sourceRank, 0, 0};
    }
    M2nApiUnlock apiUnlock;
    NCCL_M2N_CHECK(ncclWaitSignal(params.numSources, arrivals, comm, workStream));
    NCCL_M2N_CHECK(m2nWaitCommReady(comm));
  }

  if (params.isDest && dst->dataPtr != nullptr) {
    for (int srcShard = 0; srcShard < params.srcShardCount; srcShard++) {
      const ncclReshardTransferPlan& plan = unpackPlans[srcShard];
      if (unpackBytes[srcShard] == 0) {
        continue;
      }
      cudaMemcpy3DParms cp =
        packBuildCopy((char*)staging + unpackRx[srcShard], (char*)dst->dataPtr + plan.dstBaseOffset, plan,
                    /*src contig=*/nullptr, plan.outerDstStrides);
      NCCL_M2N_CUDACHECK(cudaMemcpy3DAsync(&cp, workStream));
    }
  }

  protocolGuard.complete();
  return ncclSuccess;
}

ncclResult_t reshardCopyPackNormalized(ncclComm_t comm, const ncclDistTensor_t* src,
                                             const ncclDistTensor_t* dst, cudaStream_t workStream) {
  const int ndims = src->ndims;
  void* srcBuffer = src->dataPtr;
  void* dstBuffer = dst->dataPtr;
  const size_t* srcTensorDims = src->localShape;
  const size_t* dstTensorDims = dst->localShape;
  const size_t elementSize = getNcclDtSize(src->dtype);
  const ncclDistTensor_t* srcTensor = src;
  const ncclDistTensor_t* dstTensor = dst;
  const ncclMesh_t* srcMesh = srcTensor->mesh;
  const ncclMesh_t* dstMesh = dstTensor->mesh;

  int worldRank = 0;
  int worldSize = 0;
  NCCL_M2N_CHECK(ncclCommUserRank(comm, &worldRank));
  NCCL_M2N_CHECK(ncclCommCount(comm, &worldSize));

  size_t srcDimsBytes[NCCL_RESHARD_MAX_TENSOR_DIMS] = {0};
  size_t dstDimsBytes[NCCL_RESHARD_MAX_TENSOR_DIMS] = {0};
  NCCL_M2N_CHECK(reshardDimsToBytes(worldRank, "reshardCopyPack:", ndims, elementSize, srcTensorDims,
                                    dstTensorDims, srcDimsBytes, dstDimsBytes));

  const int srcMeshSize = srcMesh->dims[0] * srcMesh->dims[1];
  const bool isSource = worldRank >= srcMesh->startRank && worldRank < srcMesh->startRank + srcMeshSize;
  ncclReshardMeshGroupInfo srcInfo{}, dstInfo{};
  computeMeshGroupInfo(srcTensor, srcMesh->startRank, &srcInfo);
  computeMeshGroupInfo(dstTensor, dstMesh->startRank, &dstInfo);
  const int srcShardCount = srcInfo.shardCount;
  const int dstShardCount = dstInfo.shardCount;
  size_t resolvedSrc[NCCL_RESHARD_MAX_TENSOR_DIMS] = {};
  size_t resolvedDst[NCCL_RESHARD_MAX_TENSOR_DIMS] = {};
  NCCL_M2N_CHECK(resolveReshardDims(worldRank, srcDimsBytes, srcInfo.shardLayout, dstDimsBytes,
                                    dstInfo.shardLayout, ndims, resolvedSrc, resolvedDst));
  size_t srcLocal = 1;
  size_t dstLocal = 1;
  for (int d = 0; d < ndims; d++) {
    NCCL_M2N_CHECK_ARG(m2nCheckedMulSize(srcLocal, resolvedSrc[d], &srcLocal) &&
                         m2nCheckedMulSize(dstLocal, resolvedDst[d], &dstLocal), worldRank,
                       "reshardCopyPack: staging size overflow at dim %d", d);
  }

  const size_t areaBytes = std::max(std::max(srcLocal, dstLocal), kMinPackStagingBytes);
  const int numCtas = pickNumCtas(areaBytes, RESHARD_ALGO_RING);
  const size_t elementsPerChunk = pickElementsPerChunk(areaBytes, RESHARD_ALGO_RING);
  int ginSignalCount = 0;
  NCCL_M2N_CHECK(computeReshardGinSignalCount(srcMesh, numCtas, worldRank, &ginSignalCount));
  unsigned int mySignalBase = 0;
  if (isSource) {
    NCCL_M2N_CHECK(computeReshardSignalBase(srcMesh, worldRank, numCtas, worldRank, &mySignalBase));
  }

  ncclCommProperties commProperties = NCCL_COMM_PROPERTIES_INITIALIZER;
  ncclResult_t queryResult = ncclSuccess;
  {
    M2nApiUnlock apiUnlock;
    queryResult = ncclCommQueryProperties(comm, &commProperties);
  }
  NCCL_M2N_CHECK(queryResult);
  int cudaDriverVersion = 0;
  NCCL_M2N_CUDACHECK(cudaDriverGetVersion(&cudaDriverVersion));
  int ncclVersion = 0;
  NCCL_M2N_CHECK(ncclGetVersion(&ncclVersion));
  const bool lsaCePlanFits =
    worldSize <= MAX_DIRECT_TARGETS && srcShardCount <= MAX_DIRECT_SOURCES && dstShardCount <= MAX_DIRECT_TARGETS;
  const bool hostRmaOrderingSupport = ncclVersion >= NCCL_VERSION(2, 30, 7);
  const bool forwardMeshOrder = srcMesh->startRank < dstMesh->startRank;
  const bool useLsaHput = commProperties.deviceApiSupport && commProperties.hostRmaSupport &&
                          commProperties.nLsaTeams == 1 && cudaDriverVersion >= 12050 && hostRmaOrderingSupport &&
                          forwardMeshOrder && lsaCePlanFits;

  NCCL_M2N_CHECK(ensurePackStagingBuffer(comm, areaBytes, workStream));
  PackStagingEventGuard stagingEvent;
  stagingEvent.arm(comm, workStream);
  void* staging = getPackStagingBuffer(comm);
  const size_t stagingCapacity = getPackStagingCapacity(comm);

  if (useLsaHput) {
    ncclWindow_t stagingWindow = nullptr;
    NCCL_M2N_CHECK(packGetOrRegisterStagingWindow(comm, staging, stagingCapacity, &stagingWindow));
    RESHARD_INFO(worldRank,
                 "pack-lsa-hput: ranks=%d srcShards=%d dstShards=%d areaBytes=%zu stagingCapacity=%zu "
                 "splitConfigured=%d",
                 worldSize, srcShardCount, dstShardCount, areaBytes, stagingCapacity,
                 (int)reshardGetSplitCommEnabled());
    NCCL_M2N_CHECK(reshardCopyPackLsaHput(comm, srcTensor, srcDimsBytes, dstTensor, dstDimsBytes, staging,
                                                stagingCapacity, stagingWindow, areaBytes, /*rxBase=*/0,
                                                elementsPerChunk, worldRank, worldSize, workStream));
    NCCL_M2N_CHECK(stagingEvent.record());
    return ncclSuccess;
  }

  if (commProperties.nLsaTeams == 1 && !hostRmaOrderingSupport) {
    RESHARD_INFO(worldRank,
                 "pack-lsa-hput: NCCL %d.%d.%d predates ordered host-RMA data/signal launch; using existing "
                 "kernel",
                 ncclVersion / 10000, (ncclVersion % 10000) / 100, ncclVersion % 100);
  } else if (commProperties.nLsaTeams == 1 && cudaDriverVersion < 12050) {
    RESHARD_INFO(worldRank,
                 "pack-lsa-hput: CUDA driver %d.%d is below the NCCL host-RMA minimum 12.5; using existing "
                 "kernel",
                 cudaDriverVersion / 1000, (cudaDriverVersion % 1000) / 10);
  } else if (commProperties.nLsaTeams == 1 && !forwardMeshOrder) {
    RESHARD_INFO(worldRank,
                 "pack-lsa-hput: reversed source/destination mesh ordering is not enabled; using existing "
                 "kernel");
  } else if (commProperties.nLsaTeams == 1 && !lsaCePlanFits) {
    RESHARD_INFO(worldRank,
                 "pack-lsa-hput: plan exceeds conservative limits (ranks=%d srcShards=%d dstShards=%d); "
                 "using existing kernel",
                 worldSize, srcShardCount, dstShardCount);
  }

  ReshardSplitComms splitComms;
  memset(&splitComms, 0, sizeof(splitComms));
  bool splitActive = false;
  if (reshardGetSplitCommEnabled() &&
      reshardEffectiveLbMode(srcTensor, dstTensor) == RESHARD_LB_NODE_AWARE) {
    /* Source / dest replication factors (placement-independent of lsaSize)
     * feed the injection-domain count.  dstRepCount lets the split decide
     * domainsPerRep = numGenDomains / dstRepCount, so commA holds one
     * complete destination replica even when a replica spans multiple NVL
     * domains (reps/domain < 1). */
    int srcRepCount = 1;
    int dstRepCount = 1;
    for (int i = 0; i < NCCL_RESHARD_MESH_NDIMS; i++) {
      if (src->placements[i] == NCCL_RESHARD_REPLICATE) srcRepCount = srcMesh->dims[i];
      if (dst->placements[i] == NCCL_RESHARD_REPLICATE) dstRepCount = dstMesh->dims[i];
    }
    const bool dstRepStrided =
      dst->placements[1] == NCCL_RESHARD_REPLICATE && dstMesh->dims[0] > 1 && dstMesh->dims[1] > 1;
    NCCL_M2N_CHECK(reshardGetOrCreateSplitComms(comm, srcMesh, dstMesh, srcRepCount, dstRepCount, dstRepStrided,
                                                numCtas, workStream, &splitComms));
    splitActive = splitComms.active;
    const int configuredDstDomain = reshardGetDstDomainSize();
    if (splitActive && configuredDstDomain > 0 && configuredDstDomain != splitComms.lsaSize) {
      RESHARD_INFO(worldRank,
                   "split-comm: NCCL_RESHARD_DST_DOMAIN_SIZE=%d differs from probed commB lsaSize=%d -> fallback",
                   configuredDstDomain, splitComms.lsaSize);
      splitActive = false;
    }
    if (splitActive && reshardPlanHasCrossRailRingHop(src, dst, splitComms.lsaSize, dstMesh->startRank,
                                                      splitComms.strided, splitComms.numInjectionDomains,
                                                      splitComms.domainsPerRep)) {
      RESHARD_INFO(worldRank,
                   "split-comm: rail-incompatible ring hop for this layout (commB is RAIL-only) -> fallback to "
                   "non-split path");
      splitActive = false;
    }
  }

  ncclWindow_t stagingWindow = nullptr;
  if (!splitActive) {
    NCCL_M2N_CHECK(packGetOrRegisterStagingWindow(comm, staging, stagingCapacity, &stagingWindow));
  }

  ncclDevComm activeDevComm;
  ReshardDevCommUse devCommUse;
  int srcGpusPerDomain = 0;
  int dstGpusPerDomain = 0;
  int srcLsaSize = 0;
  int dstLsaSize = 0;

  if (splitActive) {
    srcLsaSize = splitComms.srcLsaSize;
    dstLsaSize = splitComms.lsaSize;
  } else {
    NCCL_M2N_CHECK(reshardGetOrCreateDevComm(comm, numCtas, ginSignalCount, 0, RESHARD_DEVCOMM_BARRIER_HYBRID,
                                             reshardGetGinContextCount(), workStream, &activeDevComm, &devCommUse));
    srcLsaSize = activeDevComm.lsaSize > 0 ? activeDevComm.lsaSize : 0;
    dstLsaSize = srcLsaSize;
  }
  NCCL_M2N_CHECK(resolveReshardDomainSizes(worldRank, RESHARD_ALGO_RING, srcLsaSize, dstLsaSize, &srcGpusPerDomain,
                                           &dstGpusPerDomain));

  std::vector<size_t> allWindowOffsets(worldSize, 0);
  const bool splitStrided = splitActive && splitComms.strided;
  const int splitNumInjectionDomains = splitActive ? splitComms.numInjectionDomains : -1;
  const int splitDomainsPerRep = splitActive ? splitComms.domainsPerRep : 1;
  ncclReshardParams params{};
  NCCL_M2N_CHECK(validateReshardPlanLimits(worldRank, src, srcDimsBytes, dst, dstDimsBytes, elementsPerChunk,
                                           RESHARD_ALGO_RING, dstGpusPerDomain));
  NCCL_M2N_CHECK(prepareReshardParams(worldRank, src, srcDimsBytes, dst, dstDimsBytes, stagingWindow,
                                      elementsPerChunk, numCtas, mySignalBase, srcGpusPerDomain, dstGpusPerDomain,
                                      allWindowOffsets.data(), &params, splitStrided,
                                      /*nodeAnchorAtMeshStart=*/splitActive, splitNumInjectionDomains,
                                      splitDomainsPerRep));
  params.window = stagingWindow;
  params.myWindowOffset = 0;
  params.ringNextWindowOffset = 0;
  for (int follower = 0; follower < params.numLocalFollowers; follower++) {
    params.localFollowerWindowOffsets[follower] = 0;
  }

  RESHARD_INFO(worldRank,
	               "reshardCopyPack: isSource=%d isDest=%d numTargets=%d numSources=%d numCtas=%d "
	               "ginSignal=%d areaBytes=%zu stagingCapacity=%zu staging=%p lsa=%d",
	               (int)params.isSource, (int)params.isDest, params.numTargets, params.numSources, numCtas, ginSignalCount,
	               areaBytes, stagingCapacity, staging, dstLsaSize);

  ncclReshardTransferPlan origTargetPlan[MAX_TARGETS];
  size_t txOffset[MAX_TARGETS] = {0};
  size_t txBytes[MAX_TARGETS] = {0};
  size_t peerRx[MAX_TARGETS] = {0};
  size_t rxOffset[MAX_SOURCES] = {0};
  size_t rxBytes[MAX_SOURCES] = {0};

  if (params.isSource) {
    size_t cursor = 0;
    for (int t = 0; t < params.numTargets; t++) {
      origTargetPlan[t] = params.targets[t].plan;
      NCCL_M2N_CHECK_ARG(packPlanPairBytes(origTargetPlan[t], &txBytes[t]), worldRank,
                         "reshardCopyPack: target %d byte count overflow", t);
      txOffset[t] = cursor;
      NCCL_M2N_CHECK(packPackedRxOffset(params, params.targets[t].dstShardIdx, params.mySrcShardIdx, /*rxBase=*/0,
                                      &peerRx[t]));
      size_t txEnd = 0;
      size_t peerEnd = 0;
      NCCL_M2N_CHECK_ARG(m2nCheckedAddSize(cursor, txBytes[t], &txEnd) && txEnd <= areaBytes, worldRank,
                         "reshardCopyPack: target %d staging range exceeds %zu bytes", t, areaBytes);
      NCCL_M2N_CHECK_ARG(m2nCheckedAddSize(peerRx[t], txBytes[t], &peerEnd) && peerEnd <= stagingCapacity, worldRank,
                         "reshardCopyPack: target %d peer range exceeds %zu bytes", t, stagingCapacity);
      cursor = txEnd;
    }
  }

  if (params.isDest) {
    for (int s = 0; s < params.numSources; s++) {
      NCCL_M2N_CHECK_ARG(packPlanPairBytes(params.sources[s].plan, &rxBytes[s]), worldRank,
                         "reshardCopyPack: source %d byte count overflow", s);
      NCCL_M2N_CHECK(packPackedRxOffset(params, params.myDstShardIdx, params.sources[s].srcShardIdx, /*rxBase=*/0,
                                      &rxOffset[s]));
      size_t rxEnd = 0;
      NCCL_M2N_CHECK_ARG(m2nCheckedAddSize(rxOffset[s], rxBytes[s], &rxEnd) && rxEnd <= stagingCapacity, worldRank,
                         "reshardCopyPack: source %d staging range exceeds %zu bytes", s, stagingCapacity);
    }
  }

  if (params.isSource && srcBuffer != nullptr) {
    for (int t = 0; t < params.numTargets; t++) {
      if (txBytes[t] == 0) {
        continue;
      }
      cudaMemcpy3DParms cp = packBuildCopy((char*)srcBuffer + origTargetPlan[t].srcBaseOffset, (char*)staging + txOffset[t],
                                         origTargetPlan[t], origTargetPlan[t].outerSrcStrides, /*dst contig=*/nullptr);
      NCCL_M2N_CUDACHECK(cudaMemcpy3DAsync(&cp, workStream));
    }
  }

  if (params.isSource) {
    for (int targetIdx = 0; targetIdx < params.numTargets; targetIdx++) {
      ncclReshardTargetInfo& target = params.targets[targetIdx];
      target.windowOffset = 0;
      target.isContiguous = true;
      target.totalBytes = txBytes[targetIdx];
      target.plan.srcBaseOffset = txOffset[targetIdx];
      target.plan.dstBaseOffset = peerRx[targetIdx];
      if (target.plan.totalInnerTransfers == 0) target.plan.totalInnerTransfers = 1;
    }
  }
  if (params.isDest) {
    for (int sourceIdx = 0; sourceIdx < params.numSources; sourceIdx++) {
      ncclReshardSourceInfo& source = params.sources[sourceIdx];
      source.isContiguous = true;
      source.totalBytes = rxBytes[sourceIdx];
      source.plan.dstBaseOffset = rxOffset[sourceIdx];
      if (source.plan.totalInnerTransfers == 0) source.plan.totalInnerTransfers = 1;
    }
  }

  const int threadsPerCta = DEFAULT_KERNEL_MAX_NTHREADS;
  if (splitActive) {
    NCCL_M2N_CHECK(reshardLaunchPackSplit(&splitComms, staging, getPackStagingCapacity(comm), &params, numCtas,
                                              workStream));
  } else {
    reshardKernelUserWindow<<<numCtas, threadsPerCta, 0, workStream>>>(params, activeDevComm);
    cudaError_t launchErr = cudaGetLastError();
    if (launchErr != cudaSuccess) {
      NCCL_M2N_FAIL(ncclSystemError, worldRank, "reshardCopyPack kernel launch failed: %s [numCtas=%d]",
                    cudaGetErrorString(launchErr), numCtas);
    }
  }

  if (params.isDest && dstBuffer != nullptr) {
    for (int srcShard = 0; srcShard < params.srcShardCount; srcShard++) {
      ncclReshardTransferPlan plan;
      NCCL_M2N_CHECK(computeTransferPlanChecked(params.srcDims, params.srcStrides, params.srcShardLayout, srcShard,
                                                params.dstDims, params.dstStrides, params.dstShardLayout,
                                                params.myDstShardIdx, params.ndims, params.elementsPerChunk, &plan));
      size_t pairBytes = 0;
      NCCL_M2N_CHECK_ARG(packPlanPairBytes(plan, &pairBytes), worldRank,
                         "reshardCopyPack: unpack shard %d byte count overflow", srcShard);
      if (pairBytes == 0) {
        continue;  // src shard does not overlap my dst shard
      }
      size_t rxOff = 0;
      size_t rxEnd = 0;
      NCCL_M2N_CHECK(packPackedRxOffset(params, params.myDstShardIdx, srcShard, /*rxBase=*/0, &rxOff));
      NCCL_M2N_CHECK_ARG(m2nCheckedAddSize(rxOff, pairBytes, &rxEnd) && rxEnd <= stagingCapacity, worldRank,
                         "reshardCopyPack: unpack shard %d staging range exceeds %zu bytes", srcShard,
                         stagingCapacity);
      cudaMemcpy3DParms cp = packBuildCopy((char*)staging + rxOff, (char*)dstBuffer + plan.dstBaseOffset, plan,
                                         /*src contig=*/nullptr, plan.outerDstStrides);
      NCCL_M2N_CUDACHECK(cudaMemcpy3DAsync(&cp, workStream));
    }
  }

  if (!splitActive) {
    NCCL_M2N_CHECK(reshardRecordDevCommUse(&devCommUse, workStream));
  }
  NCCL_M2N_CHECK(stagingEvent.record());
  return ncclSuccess;
}

// ============================================================================
// Cross-tensor PACK group fusion.
//
// All entries share one normalized mesh/placement signature.  Each entry is
// prepared independently, then its per-(src shard, dst shard) fragment is
// packed into a source-major batch layout:
//
//   [src shard 0: tensor 0..N][src shard 1: tensor 0..N]...
//
// A source therefore sends one contiguous payload to each destination and the
// existing PACK kernel performs one barrier/signal epoch for the group.
// ============================================================================

static bool packGroupSameTopology(const ReshardTensorSetup& first, const ReshardTensorSetup& entry) {
  return m2nSameTensorTopology(first.srcTensor, entry.srcTensor) &&
         m2nSameTensorTopology(first.dstTensor, entry.dstTensor);
}

static ncclResult_t packGroupTensorBytes(int worldRank, const ReshardTensorSetup& setup, size_t* srcBytes,
                                       size_t* dstBytes) {
  size_t srcDims[NCCL_RESHARD_MAX_TENSOR_DIMS] = {};
  size_t dstDims[NCCL_RESHARD_MAX_TENSOR_DIMS] = {};
  NCCL_M2N_CHECK(reshardDimsToBytes(worldRank, "ncclM2nGroupEnd:", setup.ndims, setup.elementSize,
                                    setup.srcTensor.localShape, setup.dstTensor.localShape, srcDims, dstDims));
  ncclReshardMeshGroupInfo srcInfo{}, dstInfo{};
  computeMeshGroupInfo(&setup.srcTensor, setup.srcMesh.startRank, &srcInfo);
  computeMeshGroupInfo(&setup.dstTensor, setup.dstMesh.startRank, &dstInfo);
  NCCL_M2N_CHECK(resolveReshardDims(worldRank, srcDims, srcInfo.shardLayout, dstDims, dstInfo.shardLayout,
                                    setup.ndims, srcDims, dstDims));
  *srcBytes = 1;
  *dstBytes = 1;
  for (int d = 0; d < setup.ndims; d++) {
    NCCL_M2N_CHECK_ARG(m2nCheckedMulSize(*srcBytes, srcDims[d], srcBytes) &&
                         m2nCheckedMulSize(*dstBytes, dstDims[d], dstBytes), worldRank,
                       "ncclM2nGroupEnd: local byte size overflow at dim %d", d);
  }
  return ncclSuccess;
}

struct PackGroupBin {
  std::vector<size_t> entries;
  size_t srcBytes = 0;
  size_t dstBytes = 0;
  size_t areaBytes = 0;
};

static size_t groupOriginalIndex(const size_t* originalIndices, size_t entry) {
  return originalIndices == nullptr ? entry : originalIndices[entry];
}

static ncclResult_t packGroupPairOffset(const size_t* pairBytes, size_t count, int srcShardCount, int dstShardCount,
                                      int srcShard, int dstShard, size_t entry, size_t rxBase, size_t* offset) {
  size_t result = rxBase;
  for (int s = 0; s < srcShard; s++) {
    for (size_t e = 0; e < count; e++) {
      const size_t index = (e * (size_t)srcShardCount + (size_t)s) * (size_t)dstShardCount + (size_t)dstShard;
      NCCL_M2N_CHECK_ARG(m2nCheckedAddSize(result, pairBytes[index], &result), -1,
                         "ncclM2nGroupEnd: packed receive offset overflow");
    }
  }
  for (size_t e = 0; e < entry; e++) {
    const size_t index = (e * (size_t)srcShardCount + (size_t)srcShard) * (size_t)dstShardCount + (size_t)dstShard;
    NCCL_M2N_CHECK_ARG(m2nCheckedAddSize(result, pairBytes[index], &result), -1,
                       "ncclM2nGroupEnd: packed receive offset overflow");
  }
  *offset = result;
  return ncclSuccess;
}

static ncclResult_t reshardCopyPackGroupNormalized(ncclComm_t comm, const ReshardTensorSetup* setups,
                                                         const size_t* entries, const size_t* originalIndices,
                                                         size_t count, cudaStream_t workStream,
                                                         size_t* failedOriginalIndex) {
  *failedOriginalIndex = groupOriginalIndex(originalIndices, entries[0]);
  int worldRank = 0;
  int worldSize = 0;
  NCCL_M2N_CHECK(ncclCommUserRank(comm, &worldRank));
  NCCL_M2N_CHECK(ncclCommCount(comm, &worldSize));

  size_t srcBatchBytes = 0;
  size_t dstBatchBytes = 0;
  for (size_t e = 0; e < count; e++) {
    *failedOriginalIndex = groupOriginalIndex(originalIndices, entries[e]);
    const ReshardTensorSetup& setup = setups[entries[e]];
    size_t srcBytes = 0;
    size_t dstBytes = 0;
    NCCL_M2N_CHECK(packGroupTensorBytes(worldRank, setup, &srcBytes, &dstBytes));
    NCCL_M2N_CHECK_ARG(m2nCheckedAddSize(srcBatchBytes, srcBytes, &srcBatchBytes) &&
                         m2nCheckedAddSize(dstBatchBytes, dstBytes, &dstBatchBytes),
                       worldRank, "ncclM2nGroupEnd: aggregate local byte size overflow at entry %zu", entries[e]);
  }

  const size_t areaBytes = std::max(std::max(srcBatchBytes, dstBatchBytes), kMinPackStagingBytes);
  const int numCtas = pickNumCtas(areaBytes, RESHARD_ALGO_RING);
  const size_t elementsPerChunk = pickElementsPerChunk(areaBytes, RESHARD_ALGO_RING);

  const ncclMesh_t* srcMesh = &setups[entries[0]].srcMesh;
  const bool isSource = reshardRankInMesh(srcMesh, worldRank);
  int ginSignalCount = 0;
  NCCL_M2N_CHECK(computeReshardGinSignalCount(srcMesh, numCtas, worldRank, &ginSignalCount));
  unsigned int mySignalBase = 0;
  if (isSource) {
    NCCL_M2N_CHECK(computeReshardSignalBase(srcMesh, worldRank, numCtas, worldRank, &mySignalBase));
  }

  NCCL_M2N_CHECK(ensurePackStagingBuffer(comm, areaBytes, workStream));
  PackStagingEventGuard stagingEvent;
  stagingEvent.arm(comm, workStream);
  void* staging = getPackStagingBuffer(comm);
  const size_t stagingCapacity = getPackStagingCapacity(comm);

  ncclWindow_t stagingWindow = nullptr;
  NCCL_M2N_CHECK(packGetOrRegisterStagingWindow(comm, staging, stagingCapacity, &stagingWindow));

  ncclDevComm activeDevComm;
  ReshardDevCommUse devCommUse;
  NCCL_M2N_CHECK(reshardGetOrCreateDevComm(comm, numCtas, ginSignalCount, 0, RESHARD_DEVCOMM_BARRIER_HYBRID,
                                           reshardGetGinContextCount(), workStream, &activeDevComm, &devCommUse));
  const int lsaSize = activeDevComm.lsaSize > 0 ? activeDevComm.lsaSize : 0;
  int srcGpusPerDomain = 0;
  int dstGpusPerDomain = 0;
  NCCL_M2N_CHECK(resolveReshardDomainSizes(worldRank, RESHARD_ALGO_RING, lsaSize, lsaSize, &srcGpusPerDomain,
                                           &dstGpusPerDomain));

  size_t planSlots = 0;
  size_t pairSlots = 0;
  NCCL_M2N_CHECK_ARG(m2nCheckedMulSize(count, (size_t)MAX_TARGETS, &planSlots), worldRank,
                     "ncclM2nGroupEnd: target plan count overflow");

  std::unique_ptr<ncclReshardParams> aggregate(new (std::nothrow) ncclReshardParams);
  std::unique_ptr<ncclReshardParams> entryParams(new (std::nothrow) ncclReshardParams);
  std::unique_ptr<ncclReshardTransferPlan[]> targetPlans(
    new (std::nothrow) ncclReshardTransferPlan[planSlots]);
  std::unique_ptr<size_t[]> targetBytes(new (std::nothrow) size_t[planSlots]);
  if (aggregate == nullptr || entryParams == nullptr || targetPlans == nullptr || targetBytes == nullptr) {
    NCCL_M2N_FAIL(ncclSystemError, worldRank, "ncclM2nGroupEnd: failed to allocate host planning storage");
  }

  std::unique_ptr<size_t[]> allWindowOffsets(new (std::nothrow) size_t[(size_t)worldSize]());
  if (allWindowOffsets == nullptr) {
    NCCL_M2N_FAIL(ncclSystemError, worldRank, "ncclM2nGroupEnd: failed to allocate window-offset storage");
  }
  std::unique_ptr<ncclReshardTransferPlan[]> unpackPlans;
  std::unique_ptr<size_t[]> pairBytes;
  int srcShardCount = 0;
  int dstShardCount = 0;

  for (size_t e = 0; e < count; e++) {
    *failedOriginalIndex = groupOriginalIndex(originalIndices, entries[e]);
    const ReshardTensorSetup& setup = setups[entries[e]];
    size_t srcDimsBytes[NCCL_RESHARD_MAX_TENSOR_DIMS] = {0};
    size_t dstDimsBytes[NCCL_RESHARD_MAX_TENSOR_DIMS] = {0};
    NCCL_M2N_CHECK(reshardDimsToBytes(worldRank, "ncclM2nGroupEnd:", setup.ndims, setup.elementSize,
                                      setup.srcTensor.localShape, setup.dstTensor.localShape, srcDimsBytes,
                                      dstDimsBytes));
    NCCL_M2N_CHECK(validateReshardPlanLimits(worldRank, &setup.srcTensor, srcDimsBytes, &setup.dstTensor,
                                             dstDimsBytes, elementsPerChunk, RESHARD_ALGO_RING, dstGpusPerDomain));
    memset(entryParams.get(), 0, sizeof(*entryParams));
    NCCL_M2N_CHECK(prepareReshardParams(worldRank, &setup.srcTensor, srcDimsBytes, &setup.dstTensor, dstDimsBytes,
                                        stagingWindow, elementsPerChunk, numCtas, mySignalBase, srcGpusPerDomain,
                                        dstGpusPerDomain, allWindowOffsets.get(), entryParams.get()));

    if (e == 0) {
      *aggregate = *entryParams;
      srcShardCount = aggregate->srcShardCount;
      dstShardCount = aggregate->dstShardCount;
      size_t unpackSlots = 0;
      NCCL_M2N_CHECK_ARG(m2nCheckedMulSize(count, (size_t)srcShardCount, &unpackSlots) &&
                           m2nCheckedMulSize(unpackSlots, (size_t)dstShardCount, &pairSlots),
                         worldRank, "ncclM2nGroupEnd: source plan count overflow");
      unpackPlans.reset(new (std::nothrow) ncclReshardTransferPlan[unpackSlots]);
      pairBytes.reset(new (std::nothrow) size_t[pairSlots]);
      if (unpackPlans == nullptr || pairBytes == nullptr) {
        NCCL_M2N_FAIL(ncclSystemError, worldRank, "ncclM2nGroupEnd: failed to allocate host planning storage");
      }
    } else {
      NCCL_M2N_CHECK_ARG(entryParams->numTargets == aggregate->numTargets &&
                         entryParams->numSources == aggregate->numSources &&
                           entryParams->srcShardCount == srcShardCount && entryParams->dstShardCount == dstShardCount,
                         worldRank, "ncclM2nGroupEnd: entry %zu produced a different peer topology", entries[e]);
      for (int t = 0; t < aggregate->numTargets; t++) {
        NCCL_M2N_CHECK_ARG(entryParams->targets[t].dstWorldRank == aggregate->targets[t].dstWorldRank &&
                             entryParams->targets[t].dstShardIdx == aggregate->targets[t].dstShardIdx,
                           worldRank, "ncclM2nGroupEnd: entry %zu produced a different target order", entries[e]);
      }
      for (int s = 0; s < aggregate->numSources; s++) {
        NCCL_M2N_CHECK_ARG(entryParams->sources[s].srcShardIdx == aggregate->sources[s].srcShardIdx, worldRank,
                           "ncclM2nGroupEnd: entry %zu produced a different source order", entries[e]);
      }
    }

    for (int t = 0; t < aggregate->numTargets; t++) {
      const size_t index = e * (size_t)MAX_TARGETS + (size_t)t;
      targetPlans[index] = entryParams->targets[t].plan;
      NCCL_M2N_CHECK_ARG(packPlanPairBytes(targetPlans[index], &targetBytes[index]), worldRank,
                         "ncclM2nGroupEnd: target byte count overflow at entry %zu", entries[e]);
    }

    for (int s = 0; s < srcShardCount; s++) {
      ncclReshardTransferPlan& plan = unpackPlans[e * (size_t)srcShardCount + (size_t)s];
      memset(&plan, 0, sizeof(plan));
      if (entryParams->isDest) {
        NCCL_M2N_CHECK(computeTransferPlanChecked(entryParams->srcDims, entryParams->srcStrides,
                                                  entryParams->srcShardLayout, s, entryParams->dstDims,
                                                  entryParams->dstStrides, entryParams->dstShardLayout,
                                                  entryParams->myDstShardIdx, entryParams->ndims,
                                                  entryParams->elementsPerChunk, &plan));
      }
      for (int d = 0; d < dstShardCount; d++) {
        ncclReshardTransferPlan pairPlan;
        NCCL_M2N_CHECK(computeTransferPlanChecked(entryParams->srcDims, entryParams->srcStrides,
                                                  entryParams->srcShardLayout, s, entryParams->dstDims,
                                                  entryParams->dstStrides, entryParams->dstShardLayout, d,
                                                  entryParams->ndims, entryParams->elementsPerChunk, &pairPlan));
        const size_t index = (e * (size_t)srcShardCount + (size_t)s) * (size_t)dstShardCount + (size_t)d;
        NCCL_M2N_CHECK_ARG(packPlanPairBytes(pairPlan, &pairBytes[index]), worldRank,
                           "ncclM2nGroupEnd: pair byte count overflow at entry %zu", entries[e]);
      }
    }
  }

  aggregate->window = stagingWindow;
  aggregate->myWindowOffset = 0;
  aggregate->ringNextWindowOffset = 0;
  for (int f = 0; f < aggregate->numLocalFollowers; f++) {
    aggregate->localFollowerWindowOffsets[f] = 0;
  }

  size_t txCursor = 0;
  for (int t = 0; t < aggregate->numTargets; t++) {
    size_t total = 0;
    for (size_t e = 0; e < count; e++) {
      NCCL_M2N_CHECK_ARG(m2nCheckedAddSize(total, targetBytes[e * (size_t)MAX_TARGETS + (size_t)t], &total),
                         worldRank, "ncclM2nGroupEnd: target payload overflow");
    }
    size_t txEnd = 0;
    NCCL_M2N_CHECK_ARG(m2nCheckedAddSize(txCursor, total, &txEnd) && txEnd <= areaBytes, worldRank,
                       "ncclM2nGroupEnd: source staging range exceeds %zu bytes", areaBytes);
    size_t peerOffset = 0;
    NCCL_M2N_CHECK(packGroupPairOffset(pairBytes.get(), count, srcShardCount, dstShardCount,
                                     aggregate->mySrcShardIdx, aggregate->targets[t].dstShardIdx, 0, /*rxBase=*/0,
                                     &peerOffset));
    size_t peerEnd = 0;
    NCCL_M2N_CHECK_ARG(m2nCheckedAddSize(peerOffset, total, &peerEnd) && peerEnd <= stagingCapacity, worldRank,
                       "ncclM2nGroupEnd: destination staging range exceeds %zu bytes", stagingCapacity);

    size_t entryCursor = txCursor;
    if (aggregate->isSource) {
      for (size_t e = 0; e < count; e++) {
        *failedOriginalIndex = groupOriginalIndex(originalIndices, entries[e]);
        const size_t index = e * (size_t)MAX_TARGETS + (size_t)t;
        if (targetBytes[index] != 0) {
          const ncclReshardTransferPlan& plan = targetPlans[index];
          cudaMemcpy3DParms cp = packBuildCopy((char*)setups[entries[e]].srcTensor.dataPtr + plan.srcBaseOffset,
                                             (char*)staging + entryCursor, plan, plan.outerSrcStrides, nullptr);
          NCCL_M2N_CUDACHECK(cudaMemcpy3DAsync(&cp, workStream));
        }
        NCCL_M2N_CHECK_ARG(m2nCheckedAddSize(entryCursor, targetBytes[index], &entryCursor), worldRank,
                           "ncclM2nGroupEnd: source packing offset overflow");
      }
    }

    ncclReshardTargetInfo& target = aggregate->targets[t];
    target.windowOffset = 0;
    target.isContiguous = true;
    target.totalBytes = total;
    target.plan.srcBaseOffset = txCursor;
    target.plan.dstBaseOffset = peerOffset;
    target.plan.totalInnerTransfers = 1;
    txCursor = txEnd;
  }

  if (aggregate->isDest) {
    for (int i = 0; i < aggregate->numSources; i++) {
      ncclReshardSourceInfo& source = aggregate->sources[i];
      size_t total = 0;
      for (size_t e = 0; e < count; e++) {
        const size_t index = (e * (size_t)srcShardCount + (size_t)source.srcShardIdx) *
                               (size_t)dstShardCount +
                             (size_t)aggregate->myDstShardIdx;
        NCCL_M2N_CHECK_ARG(m2nCheckedAddSize(total, pairBytes[index], &total), worldRank,
                           "ncclM2nGroupEnd: source payload overflow");
      }
      size_t offset = 0;
      NCCL_M2N_CHECK(packGroupPairOffset(pairBytes.get(), count, srcShardCount, dstShardCount, source.srcShardIdx,
                                       aggregate->myDstShardIdx, 0, /*rxBase=*/0, &offset));
      source.isContiguous = true;
      source.totalBytes = total;
      source.plan.dstBaseOffset = offset;
      source.plan.totalInnerTransfers = 1;
    }
  }

  RESHARD_INFO(worldRank,
               "ncclM2nGroupEnd: count=%zu numTargets=%d numSources=%d numCtas=%d areaBytes=%zu staging=%p",
               count, aggregate->numTargets, aggregate->numSources, numCtas, areaBytes, staging);

  reshardKernelUserWindow<<<numCtas, DEFAULT_KERNEL_MAX_NTHREADS, 0, workStream>>>(*aggregate, activeDevComm);
  cudaError_t launchError = cudaGetLastError();
  if (launchError != cudaSuccess) {
    NCCL_M2N_FAIL(ncclSystemError, worldRank, "ncclM2nGroupEnd kernel launch failed: %s [numCtas=%d]",
                  cudaGetErrorString(launchError), numCtas);
  }
  if (aggregate->isDest) {
    for (int s = 0; s < srcShardCount; s++) {
      for (size_t e = 0; e < count; e++) {
        *failedOriginalIndex = groupOriginalIndex(originalIndices, entries[e]);
        const size_t pairIndex = (e * (size_t)srcShardCount + (size_t)s) * (size_t)dstShardCount +
                                 (size_t)aggregate->myDstShardIdx;
        if (pairBytes[pairIndex] == 0) {
          continue;
        }
        const ncclReshardTransferPlan& plan = unpackPlans[e * (size_t)srcShardCount + (size_t)s];
        size_t offset = 0;
        NCCL_M2N_CHECK(packGroupPairOffset(pairBytes.get(), count, srcShardCount, dstShardCount, s,
                                         aggregate->myDstShardIdx, e, /*rxBase=*/0, &offset));
        size_t end = 0;
        NCCL_M2N_CHECK_ARG(m2nCheckedAddSize(offset, pairBytes[pairIndex], &end) && end <= stagingCapacity, worldRank,
                           "ncclM2nGroupEnd: unpack range exceeds %zu bytes", stagingCapacity);
        cudaMemcpy3DParms cp =
          packBuildCopy((char*)staging + offset,
                      (char*)setups[entries[e]].dstTensor.dataPtr + plan.dstBaseOffset, plan, nullptr,
                      plan.outerDstStrides);
        NCCL_M2N_CUDACHECK(cudaMemcpy3DAsync(&cp, workStream));
      }
    }
  }

  NCCL_M2N_CHECK(reshardRecordDevCommUse(&devCommUse, workStream));
  NCCL_M2N_CHECK(stagingEvent.record());
  return ncclSuccess;
}

ncclResult_t reshardTryExecuteStagingGroup(ncclM2nHandle_t handle, ncclComm_t comm,
                                           const ncclDistTensor_t* srcs, const ncclDistTensor_t* dsts,
                                           const size_t* originalIndices, size_t count, cudaStream_t stream,
                                           bool* handled, size_t* failedOriginalIndex) {
  *handled = false;
  if (count < 2 || count > kM2nGroupMaxFusionEntries ||
      reshardGetCopyAlgorithm() != RESHARD_COPY_ALGO_PACK) {
    return ncclSuccess;
  }
  *failedOriginalIndex = groupOriginalIndex(originalIndices, 0);

  M2nApiLock apiLock;
  m2nClearLastError();
  NCCL_M2N_CHECK_ARG(comm != nullptr, -1, "ncclM2nGroupEnd: comm must be non-null");
  NCCL_M2N_CHECK_ARG(srcs != nullptr && dsts != nullptr, -1,
                     "ncclM2nGroupEnd: srcs and dsts must both be non-null");

  std::unique_ptr<ReshardTensorSetup[]> setups(new (std::nothrow) ReshardTensorSetup[count]);
  if (setups == nullptr) {
    NCCL_M2N_FAIL(ncclSystemError, -1, "ncclM2nGroupEnd: failed to allocate descriptor storage");
  }
  for (size_t e = 0; e < count; e++) {
    *failedOriginalIndex = groupOriginalIndex(originalIndices, e);
    NCCL_M2N_CHECK(reshardPrepareTensorSetup("ncclM2nGroupEnd", &srcs[e], &dsts[e], &setups[e]));
  }

  std::shared_ptr<ncclM2nHandleState> handleState;
  NCCL_M2N_CHECK(acquireM2nHandle(handle, &handleState));
  /* Grouped execution bypasses ncclReshard; ready the parent before metadata. */
  {
    M2nApiUnlock apiUnlock;
    NCCL_M2N_CHECK(m2nWaitCommReady(comm));
  }
  int parentCommSize = 0;
  NCCL_M2N_CHECK(ncclCommCount(comm, &parentCommSize));
  reshardResolveAdaptiveScaleConfig(parentCommSize, /*splitCapable=*/false);
  if (reshardGetSplitCommEnabled()) {
    return ncclSuccess;
  }

  int worldRank = 0;
  int worldSize = 0;
  NCCL_M2N_CHECK(ncclCommUserRank(comm, &worldRank));
  NCCL_M2N_CHECK(ncclCommCount(comm, &worldSize));
  for (size_t e = 0; e < count; e++) {
    *failedOriginalIndex = groupOriginalIndex(originalIndices, e);
    NCCL_M2N_CHECK(validateReshardMeshBounds(&setups[e].srcMesh, &setups[e].dstMesh, worldSize, worldRank));
    NCCL_M2N_CHECK(
      reshardValidateActiveBuffers("ncclM2nGroupEnd", worldRank, &setups[e].srcTensor, &setups[e].dstTensor));
  }

  int currentCudaDev = 0;
  ncclCommProperties commProps = NCCL_COMM_PROPERTIES_INITIALIZER;
  ncclResult_t propsResult = ncclSuccess;
  NCCL_M2N_CHECK(reshardMatchCommCudaDevice(comm, &currentCudaDev, &commProps, &propsResult));
  ncclResult_t captureResult = reshardRejectGraphCapture("ncclReshard", stream);
  if (captureResult == ncclInvalidUsage) {
    return ncclSuccess;
  }
  NCCL_M2N_CHECK(captureResult);

  std::unique_ptr<size_t[]> srcEntryBytes(new (std::nothrow) size_t[count]);
  std::unique_ptr<size_t[]> dstEntryBytes(new (std::nothrow) size_t[count]);
  if (srcEntryBytes == nullptr || dstEntryBytes == nullptr) {
    NCCL_M2N_FAIL(ncclSystemError, worldRank, "ncclM2nGroupEnd: failed to allocate bin sizing storage");
  }
  size_t largestSingleArea = 0;
  for (size_t e = 0; e < count; e++) {
    *failedOriginalIndex = groupOriginalIndex(originalIndices, e);
    NCCL_M2N_CHECK(packGroupTensorBytes(worldRank, setups[e], &srcEntryBytes[e], &dstEntryBytes[e]));
    largestSingleArea =
      std::max(largestSingleArea, std::max(std::max(srcEntryBytes[e], dstEntryBytes[e]), (size_t)2048));
  }

  struct GroupBufferRange {
    uintptr_t begin;
    uintptr_t end;
    size_t entry;
  };
  std::vector<GroupBufferRange> ranges;
  try {
    ranges.reserve(count * 2);
    for (size_t e = 0; e < count; e++) {
      *failedOriginalIndex = groupOriginalIndex(originalIndices, e);
      if (setups[e].srcTensor.dataPtr != nullptr && srcEntryBytes[e] > 0) {
        const uintptr_t begin = reinterpret_cast<uintptr_t>(setups[e].srcTensor.dataPtr);
        size_t end = 0;
        NCCL_M2N_CHECK_ARG(m2nCheckedAddSize(begin, srcEntryBytes[e], &end), worldRank,
                           "ncclM2nGroupEnd: source range overflow at entry %zu",
                           groupOriginalIndex(originalIndices, e));
        ranges.push_back({begin, end, groupOriginalIndex(originalIndices, e)});
      }
      if (setups[e].dstTensor.dataPtr != nullptr && dstEntryBytes[e] > 0) {
        const uintptr_t begin = reinterpret_cast<uintptr_t>(setups[e].dstTensor.dataPtr);
        size_t end = 0;
        NCCL_M2N_CHECK_ARG(m2nCheckedAddSize(begin, dstEntryBytes[e], &end), worldRank,
                           "ncclM2nGroupEnd: destination range overflow at entry %zu",
                           groupOriginalIndex(originalIndices, e));
        ranges.push_back({begin, end, groupOriginalIndex(originalIndices, e)});
      }
    }
    std::sort(ranges.begin(), ranges.end(), [](const GroupBufferRange& a, const GroupBufferRange& b) {
      return a.begin < b.begin;
    });
    for (size_t i = 0; i < ranges.size(); i++) {
      for (size_t j = i + 1; j < ranges.size() && ranges[j].begin < ranges[i].end; j++) {
        const size_t firstEntry = std::min(ranges[i].entry, ranges[j].entry);
        const size_t secondEntry = std::max(ranges[i].entry, ranges[j].entry);
        *failedOriginalIndex = secondEntry;
        NCCL_M2N_CHECK_ARG(ranges[i].entry == ranges[j].entry, worldRank,
                           "ncclM2nGroupEnd: grouped entries %zu and %zu have overlapping buffers",
                           firstEntry, secondEntry);
      }
    }
  } catch (const std::bad_alloc&) {
    NCCL_M2N_FAIL(ncclSystemError, worldRank, "ncclM2nGroupEnd: failed to allocate overlap validation storage");
  }

  size_t stagingBudget = gReshardStagingBuckets[gReshardStagingBucketCount - 1].size;
  for (int i = 0; i < gReshardStagingBucketCount; i++) {
    if (gReshardStagingBuckets[i].size >= largestSingleArea) {
      stagingBudget = gReshardStagingBuckets[i].size;
      break;
    }
  }

  std::vector<PackGroupBin> bins;
  try {
    for (size_t e = 0; e < count; e++) {
      *failedOriginalIndex = groupOriginalIndex(originalIndices, e);
      bool added = false;
      for (PackGroupBin& bin : bins) {
        if (!packGroupSameTopology(setups[bin.entries.front()], setups[e])) {
          continue;
        }
        size_t srcBytes = 0;
        size_t dstBytes = 0;
        if (!m2nCheckedAddSize(bin.srcBytes, srcEntryBytes[e], &srcBytes) ||
            !m2nCheckedAddSize(bin.dstBytes, dstEntryBytes[e], &dstBytes) ||
            std::max(std::max(srcBytes, dstBytes), kMinPackStagingBytes) > stagingBudget) {
          continue;
        }
        bin.entries.push_back(e);
        bin.srcBytes = srcBytes;
        bin.dstBytes = dstBytes;
        bin.areaBytes = std::max(std::max(srcBytes, dstBytes), kMinPackStagingBytes);
        added = true;
        break;
      }
      if (!added) {
        PackGroupBin bin;
        bin.entries.push_back(e);
        bin.srcBytes = srcEntryBytes[e];
        bin.dstBytes = dstEntryBytes[e];
        bin.areaBytes = std::max(std::max(bin.srcBytes, bin.dstBytes), kMinPackStagingBytes);
        bins.push_back(std::move(bin));
      }
    }
    std::stable_sort(bins.begin(), bins.end(),
                     [](const PackGroupBin& a, const PackGroupBin& b) { return a.areaBytes > b.areaBytes; });
  } catch (const std::bad_alloc&) {
    NCCL_M2N_FAIL(ncclSystemError, worldRank, "ncclM2nGroupEnd: failed to allocate fusion bins");
  }

  ReshardWorkStream work{};
  NCCL_M2N_CHECK(reshardSetupWorkStream(comm, stream, currentCudaDev, propsResult, &commProps, &work));
  ReshardWorkStreamCompletion workCompletion(stream, &work);
  size_t fusedBins = 0;
  size_t maxBinEntries = 0;
  for (const PackGroupBin& bin : bins) {
    *failedOriginalIndex = groupOriginalIndex(originalIndices, bin.entries.front());
    maxBinEntries = std::max(maxBinEntries, bin.entries.size());
    if (bin.entries.size() == 1) {
      const ReshardTensorSetup& setup = setups[bin.entries.front()];
      NCCL_M2N_CHECK(
        reshardCopyPackNormalized(comm, &setup.srcTensor, &setup.dstTensor, work.stream));
      continue;
    }
    NCCL_M2N_CHECK(reshardCopyPackGroupNormalized(comm, setups.get(), bin.entries.data(), originalIndices,
                                                        bin.entries.size(), work.stream, failedOriginalIndex));
    fusedBins++;
#ifdef NCCL_M2N_TESTING
    gFusedSubmissionCount++;
#endif
  }
  RESHARD_INFO(worldRank, "ncclM2nGroupEnd: entries=%zu bins=%zu fusedBins=%zu maxBinEntries=%zu budget=%zu", count,
               bins.size(), fusedBins, maxBinEntries, stagingBudget);
  *handled = true;
  return workCompletion.complete();
}
