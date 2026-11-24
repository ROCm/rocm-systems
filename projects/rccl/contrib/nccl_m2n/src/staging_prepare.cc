/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

/*************************************************************************
 * staging_prepare.cc — host-side descriptor builders for ncclReshard.
 *
 * Builds the per-rank staging transfer descriptors, delegating mesh and
 * load-balance analysis to the reshard_mesh.cc + reshard_loadbalance.cc
 * helpers.
 *
 * The initial implementation uses a pure RDMA all-to-all descriptor.
 ************************************************************************/

#include "staging_types.h"
#include "staging_buffer.h"

#include "reshard_internal.h"
#include "reshard_types.h"
#include "m2n_log.h"
#include "m2n_checks.h"
#include "m2n_checked_math.h"

#include <cstdio>
#include <algorithm>
#include <cstring>
#include "nccl.h"

/* ======================================================================
 * Helper: copy a window/staging plan from ncclReshardTransferPlan into the
 * staging path's StagingTransferPlan layout.
 * ====================================================================*/
static ncclResult_t fillStagingPlan(StagingTransferPlan* sp, const ncclReshardTransferPlan& plan, int ndims) {
  sp->numOuterLoops = plan.numOuterLoops;
  sp->srcBaseOffset = plan.srcBaseOffset;
  sp->dstBaseOffset = plan.dstBaseOffset;
  sp->innerSize = plan.innerSize;
  sp->totalInnerTransfers = plan.totalInnerTransfers;
  sp->isContiguous = (plan.totalInnerTransfers == 1);
  NCCL_M2N_CHECK_ARG(m2nCheckedMulSize(plan.totalInnerTransfers, plan.innerSize, &sp->totalBytes), -1,
                     "[STAGING] plan totalBytes overflow: transfers=%zu innerSize=%zu", plan.totalInnerTransfers,
                     plan.innerSize);
  for (int d = 0; d < ndims; d++) {
    sp->outerCounts[d] = plan.outerCounts[d];
    sp->outerSrcStrides[d] = plan.outerSrcStrides[d];
    sp->outerDstStrides[d] = plan.outerDstStrides[d];
  }
  return ncclSuccess;
}

static int pickLocalRepIndex(int sourcePosition, int numSources, int numLocalReps) {
  if (numSources <= 0 || numLocalReps <= 0) {
    return 0;
  }
  int sourcesPerRep = numSources / numLocalReps;
  int extra = numSources % numLocalReps;
  int threshold = extra * (sourcesPerRep + 1);
  if (sourcesPerRep == 0 || sourcePosition < threshold) {
    return std::min(sourcePosition / (sourcesPerRep + 1), numLocalReps - 1);
  }
  return std::min(extra + (sourcePosition - threshold) / sourcesPerRep, numLocalReps - 1);
}

/* Match pickLocalRepIndex() with the exact contiguous source range owned by
 * one local RDMA handler.  This gives GIN a compact RDMA-only ordinal while
 * the full source ordering remains available for staging-buffer offsets. */
static void computeBalancedSourceRange(int numSources, int numHandlers, int handlerIdx, int* start, int* end) {
  *start = 0;
  *end = 0;
  if (numSources <= 0 || numHandlers <= 0 || handlerIdx < 0 || handlerIdx >= numHandlers) {
    return;
  }
  const int sourcesPerHandler = numSources / numHandlers;
  const int extra = numSources % numHandlers;
  if (handlerIdx < extra) {
    *start = handlerIdx * (sourcesPerHandler + 1);
    *end = *start + sourcesPerHandler + 1;
  } else {
    *start = extra * (sourcesPerHandler + 1) + (handlerIdx - extra) * sourcesPerHandler;
    *end = *start + sourcesPerHandler;
  }
}

static int computeStagingPeerGroupSizeBound(const StagingTransferDescriptor* desc) {
  int bound = 1;
  if (desc == nullptr) {
    return bound;
  }
  bound = std::max(bound, desc->numSources);
  bound = std::max(bound, desc->numTargets);
  for (int t = 0; t < desc->numTargets; t++) {
    if (desc->destNumSources[t] > 0) {
      bound = std::max(bound, desc->destNumSources[t]);
    }
  }
  return bound;
}

static ncclResult_t computeMaxShardEdgeBytes(int worldRank, const size_t srcDims[], const size_t srcStrides[],
                                             const ncclReshardShardLayout_t& srcShardLayout, int srcShardCount, const size_t dstDims[],
                                             const size_t dstStrides[], const ncclReshardShardLayout_t& dstShardLayout, int dstShardCount, int ndims,
                                             size_t* maxEdgeBytes) {
  NCCL_M2N_CHECK_ARG(maxEdgeBytes != nullptr, worldRank, "computeMaxShardEdgeBytes: output must be non-null");
  *maxEdgeBytes = 0;
  for (int ss = 0; ss < srcShardCount; ss++) {
    for (int ds = 0; ds < dstShardCount; ds++) {
      ncclReshardTransferPlan plan;
      NCCL_M2N_CHECK(computeTransferPlanChecked(srcDims, srcStrides, srcShardLayout, ss, dstDims, dstStrides, dstShardLayout,
                                                ds, ndims, /*elementsPerChunk=*/1, &plan));
      if (plan.totalInnerTransfers == 0) {
        continue;
      }
      size_t totalBytes = 0;
      NCCL_M2N_CHECK_ARG(m2nCheckedMulSize(plan.totalInnerTransfers, plan.innerSize, &totalBytes), worldRank,
                         "computeMaxShardEdgeBytes: edge byte count overflows: transfers=%zu innerSize=%zu",
                         plan.totalInnerTransfers, plan.innerSize);
      *maxEdgeBytes = std::max(*maxEdgeBytes, totalBytes);
    }
  }
  return ncclSuccess;
}

static int countOverlappingDstShards(const size_t srcDims[], const size_t srcStrides[], const ncclReshardShardLayout_t& srcShardLayout,
                                     int srcShardIdx, const size_t dstDims[], const size_t dstStrides[],
                                     const ncclReshardShardLayout_t& dstShardLayout, int dstShardCount, int ndims) {
  int count = 0;
  for (int dstShard = 0; dstShard < dstShardCount; dstShard++) {
    ncclReshardTransferPlan plan;
    computeTransferPlan(srcDims, srcStrides, srcShardLayout, srcShardIdx, dstDims, dstStrides, dstShardLayout, dstShard,
                        ndims, /*elementsPerChunk=*/1, &plan);
    if (plan.totalInnerTransfers > 0) {
      count++;
    }
  }
  return count;
}

static int countOverlappingSrcShards(const size_t srcDims[], const size_t srcStrides[], const ncclReshardShardLayout_t& srcShardLayout,
                                     int srcShardCount, const size_t dstDims[], const size_t dstStrides[],
                                     const ncclReshardShardLayout_t& dstShardLayout, int dstShardIdx, int ndims) {
  int count = 0;
  for (int srcShard = 0; srcShard < srcShardCount; srcShard++) {
    ncclReshardTransferPlan plan;
    computeTransferPlan(srcDims, srcStrides, srcShardLayout, srcShard, dstDims, dstStrides, dstShardLayout, dstShardIdx,
                        ndims, /*elementsPerChunk=*/1, &plan);
    if (plan.totalInnerTransfers > 0) {
      count++;
    }
  }
  return count;
}

static ncclResult_t validateRankStagingCounts(
  int worldRank, int rank, const ncclDistTensor_t* srcTensor, const ncclReshardMeshGroupInfo* fullSrcInfo,
  const ncclDistTensor_t* dstTensor, const ncclReshardMeshGroupInfo* fullDstInfo, const size_t srcDims[],
  const size_t srcStrides[], const size_t dstDims[], const size_t dstStrides[], int ndims,
  const ncclReshardRepLoadBalancer* lb, ReshardCopyAlgorithm copyAlgo, int gpusPerDomain, int dstNodeAnchor,
  size_t* maxPeerGroupSize) {
  const bool isSource = reshardRankInMesh(srcTensor->mesh, rank);
  const bool isDest = reshardRankInMesh(dstTensor->mesh, rank);
  if (!isSource && !isDest) {
    return ncclSuccess;
  }

  size_t numTargets = 0;
  int numSources = 0;

  auto addTargets = [&](size_t count) -> ncclResult_t {
    size_t updated = 0;
    NCCL_M2N_CHECK_ARG(m2nCheckedAddSize(numTargets, count, &updated), worldRank,
                       "validateStagingPlanLimits: target count overflows size_t for rank %d", rank);
    numTargets = updated;
    return ncclSuccess;
  };

  if (isSource) {
    ncclReshardMeshGroupInfo srcInfo;
    computeMeshGroupInfo(srcTensor, rank, &srcInfo);
    int targetRepStart = 0;
    int targetRepEnd = 0;
    getTargetRepRange(lb, srcInfo.repIdx, &targetRepStart, &targetRepEnd);
    const int targetRepCount = targetRepEnd - targetRepStart;
    if (targetRepCount > 0) {
      const int overlapDstCount =
        countOverlappingDstShards(srcDims, srcStrides, fullSrcInfo->shardLayout, srcInfo.shardIdx, dstDims,
                                  dstStrides, fullDstInfo->shardLayout, fullDstInfo->shardCount, ndims);
      if (copyAlgo == RESHARD_COPY_ALGO_DIRECT) {
        size_t directTargets = 0;
        NCCL_M2N_CHECK_ARG(
          m2nCheckedMulSize(static_cast<size_t>(overlapDstCount), static_cast<size_t>(targetRepCount), &directTargets),
          worldRank, "validateStagingPlanLimits: direct target count overflows size_t for rank %d", rank);
        NCCL_M2N_CHECK(addTargets(directTargets));
      } else {
        NCCL_M2N_CHECK(addTargets(static_cast<size_t>(overlapDstCount)));
      }
    }
  }

  if (isDest) {
    ncclReshardMeshGroupInfo dstInfo;
    computeMeshGroupInfo(dstTensor, rank, &dstInfo);
    numSources = countOverlappingSrcShards(srcDims, srcStrides, fullSrcInfo->shardLayout, fullSrcInfo->shardCount,
                                           dstDims, dstStrides, fullDstInfo->shardLayout, dstInfo.shardIdx, ndims);

    if (copyAlgo != RESHARD_COPY_ALGO_DIRECT && numSources > 0) {
      int sourceRep = getSourceRepForDest(lb, dstInfo.repIdx);
      int targetRepStart = 0;
      int targetRepEnd = 0;
      getTargetRepRange(lb, sourceRep, &targetRepStart, &targetRepEnd);

      int numLocalReps = 0;
      int myLocalRepIdx = -1;
      const int myNode = (rank - dstNodeAnchor) / gpusPerDomain;
      for (int rep = targetRepStart; rep < targetRepEnd; rep++) {
        int repRank = getMeshRank(dstTensor, fullDstInfo, dstInfo.shardIdx, rep);
        if ((repRank - dstNodeAnchor) / gpusPerDomain != myNode) {
          continue;
        }
        if (rep == dstInfo.repIdx) {
          myLocalRepIdx = numLocalReps;
        }
        numLocalReps++;
      }

      NCCL_M2N_CHECK_ARG(numLocalReps <= MAX_LOCAL_FOLLOWERS + 1, worldRank,
                         "validateStagingPlanLimits: local destination fan-out exceeds capacity at rank %d, "
                         "dstShard %d, domain %d (localRepCount=%d > MAX_LOCAL_FOLLOWERS+1=%d)",
                         rank, dstInfo.shardIdx, myNode, numLocalReps, MAX_LOCAL_FOLLOWERS + 1);
      NCCL_M2N_CHECK_ARG(myLocalRepIdx >= 0, worldRank,
                         "validateStagingPlanLimits: destination rank %d has no local handler in target rep range "
                         "[%d, %d)",
                         rank, targetRepStart, targetRepEnd);

      int firstRepNodeDest = -1;
      int firstNodeLocalReps = 0;
      if (targetRepStart < targetRepEnd) {
        int firstRepRank = getMeshRank(dstTensor, fullDstInfo, dstInfo.shardIdx, targetRepStart);
        firstRepNodeDest = (firstRepRank - dstNodeAnchor) / gpusPerDomain;
        for (int rep = targetRepStart; rep < targetRepEnd; rep++) {
          int repRank = getMeshRank(dstTensor, fullDstInfo, dstInfo.shardIdx, rep);
          if ((repRank - dstNodeAnchor) / gpusPerDomain == firstRepNodeDest) {
            firstNodeLocalReps++;
          }
        }
      }

      int sourceRepSlots = firstNodeLocalReps > 0 ? firstNodeLocalReps : numLocalReps;
      int activeSourceSlots = sourceRepSlots;
      for (int node = firstRepNodeDest; node >= 0 && node <= myNode; node++) {
        int nodeLocalReps = 0;
        for (int rep = targetRepStart; rep < targetRepEnd; rep++) {
          int repRank = getMeshRank(dstTensor, fullDstInfo, dstInfo.shardIdx, rep);
          if ((repRank - dstNodeAnchor) / gpusPerDomain == node) {
            nodeLocalReps++;
          }
        }
        if (nodeLocalReps > 0 && nodeLocalReps < activeSourceSlots) {
          activeSourceSlots = nodeLocalReps;
        }
      }

      int mySourceStart = 0;
      int mySourceEnd = 0;
      int mySourceSlotStart = myLocalRepIdx;
      int mySourceSlotEnd = myLocalRepIdx + 1;
      if (activeSourceSlots > 0 && myLocalRepIdx == activeSourceSlots - 1 && sourceRepSlots > activeSourceSlots) {
        mySourceSlotEnd = sourceRepSlots;
      }
      if (sourceRepSlots > 0 && mySourceSlotStart < activeSourceSlots) {
        int sourcesPerRep = numSources / sourceRepSlots;
        int extra = numSources % sourceRepSlots;
        int threshold = extra * (sourcesPerRep + 1);
        mySourceStart = (mySourceSlotStart < extra) ? mySourceSlotStart * (sourcesPerRep + 1)
                                                   : threshold + (mySourceSlotStart - extra) * sourcesPerRep;
        if (mySourceSlotEnd >= sourceRepSlots) mySourceEnd = numSources;
        else if (mySourceSlotEnd < extra) mySourceEnd = mySourceSlotEnd * (sourcesPerRep + 1);
        else mySourceEnd = threshold + (mySourceSlotEnd - extra) * sourcesPerRep;
      }

      bool hasRingNext = false;
      for (int rep = targetRepStart; rep < targetRepEnd; rep++) {
        int repRank = getMeshRank(dstTensor, fullDstInfo, dstInfo.shardIdx, rep);
        if ((repRank - dstNodeAnchor) / gpusPerDomain > myNode) {
          hasRingNext = true;
          break;
        }
      }

      const int handledSources = mySourceEnd - mySourceStart;
      size_t localTargets = 0;
      NCCL_M2N_CHECK_ARG(m2nCheckedMulSize(static_cast<size_t>(handledSources),
                                           static_cast<size_t>(numLocalReps - 1), &localTargets),
                         worldRank, "validateStagingPlanLimits: local target count overflows size_t for rank %d", rank);
      NCCL_M2N_CHECK(addTargets(localTargets));
      if (hasRingNext) NCCL_M2N_CHECK(addTargets(static_cast<size_t>(handledSources)));
    }
  }

  if (numTargets > static_cast<size_t>(MAX_TARGETS)) {
    NCCL_M2N_FAIL(ncclInvalidArgument, worldRank,
                  "validateStagingPlanLimits: target list would exceed capacity for rank %d "
                  "(targets=%zu > MAX_TARGETS=%d). Fix: increase MAX_TARGETS in reshard_limits.h.",
                  rank, numTargets, MAX_TARGETS);
  }
  if (numSources > MAX_SOURCES) {
    NCCL_M2N_FAIL(ncclInvalidArgument, worldRank,
                  "validateStagingPlanLimits: source list would exceed capacity for rank %d "
                  "(sources=%d > MAX_SOURCES=%d). Fix: increase MAX_SOURCES in reshard_limits.h.",
                  rank, numSources, MAX_SOURCES);
  }
  if (maxPeerGroupSize != nullptr) {
    *maxPeerGroupSize = std::max(*maxPeerGroupSize, std::max(numTargets, static_cast<size_t>(numSources)));
  }
  return ncclSuccess;
}

ncclResult_t validateStagingPlanLimits(int worldRank, const ncclDistTensor_t* srcTensor, const size_t* srcTensorDims,
                                       const ncclDistTensor_t* dstTensor, const size_t* dstTensorDims,
                                       ReshardCopyAlgorithm copyAlgo, int gpusPerDomain, size_t* maxPeerGroupSize,
                                       bool splitStrided, int splitNumInjectionDomains, int splitDomainsPerRep,
                                       bool nodeAnchorAtMeshStart) {
  NCCL_M2N_CHECK_ARG(srcTensor != nullptr, worldRank, "validateStagingPlanLimits: src tensor must be non-null");
  NCCL_M2N_CHECK_ARG(dstTensor != nullptr, worldRank, "validateStagingPlanLimits: dst tensor must be non-null");
  NCCL_M2N_CHECK_ARG(srcTensor->mesh != nullptr, worldRank, "validateStagingPlanLimits: src mesh must be non-null");
  NCCL_M2N_CHECK_ARG(dstTensor->mesh != nullptr, worldRank, "validateStagingPlanLimits: dst mesh must be non-null");
  NCCL_M2N_CHECK_ARG(srcTensorDims != nullptr, worldRank,
                     "validateStagingPlanLimits: src shape array must be non-null");
  NCCL_M2N_CHECK_ARG(dstTensorDims != nullptr, worldRank,
                     "validateStagingPlanLimits: dst shape array must be non-null");
  NCCL_M2N_CHECK_ARG(srcTensor->ndims == dstTensor->ndims, worldRank,
                     "validateStagingPlanLimits: src and dst ndims must match");
  NCCL_M2N_CHECK_ARG(gpusPerDomain > 0, worldRank, "validateStagingPlanLimits: gpusPerDomain must be positive");

  const ncclMesh_t* srcMesh = srcTensor->mesh;
  const ncclMesh_t* dstMesh = dstTensor->mesh;
  ReshardMeshInterval srcInterval{};
  ReshardMeshInterval dstInterval{};
  NCCL_M2N_CHECK(computeReshardMeshInterval(srcMesh, worldRank, &srcInterval));
  NCCL_M2N_CHECK(computeReshardMeshInterval(dstMesh, worldRank, &dstInterval));
  const bool meshesOverlap = srcInterval.startRank < dstInterval.endRank && dstInterval.startRank < srcInterval.endRank;
  NCCL_M2N_CHECK_ARG(copyAlgo != RESHARD_COPY_ALGO_PIPE || !meshesOverlap, worldRank,
                     "validateStagingPlanLimits: PIPE requires disjoint source and destination rank intervals "
                     "(src=[%d,%d), dst=[%d,%d))",
                     srcInterval.startRank, srcInterval.endRank, dstInterval.startRank, dstInterval.endRank);
  ncclReshardMeshGroupInfo fullSrcInfo;
  ncclReshardMeshGroupInfo fullDstInfo;
  computeMeshGroupInfo(srcTensor, srcMesh->startRank, &fullSrcInfo);
  computeMeshGroupInfo(dstTensor, dstMesh->startRank, &fullDstInfo);

  size_t srcDims[NCCL_RESHARD_MAX_TENSOR_DIMS] = {0};
  size_t dstDims[NCCL_RESHARD_MAX_TENSOR_DIMS] = {0};
  size_t srcStrides[NCCL_RESHARD_MAX_TENSOR_DIMS] = {0};
  size_t dstStrides[NCCL_RESHARD_MAX_TENSOR_DIMS] = {0};
  NCCL_M2N_CHECK(resolveReshardDims(worldRank, srcTensorDims, fullSrcInfo.shardLayout, dstTensorDims,
                                    fullDstInfo.shardLayout, srcTensor->ndims, srcDims, dstDims));
  NCCL_M2N_CHECK(computeStridesChecked(srcDims, srcTensor->ndims, srcStrides));
  NCCL_M2N_CHECK(computeStridesChecked(dstDims, dstTensor->ndims, dstStrides));

  ncclReshardRepLoadBalancer lb = {};
  lb.srcRepCount = fullSrcInfo.repCount;
  lb.dstRepCount = fullDstInfo.repCount;
  lb.dstGpusPerDomain = gpusPerDomain;
  lb.dstRepStartRank = dstMesh->startRank;
  lb.dstRepStride = (fullDstInfo.repMeshDim == 0) ? dstMesh->dims[1] : 1;
  lb.mode = reshardEffectiveLbMode(srcTensor, dstTensor);
  lb.dstNodeAnchor = nodeAnchorAtMeshStart ? dstMesh->startRank : 0;
  if (splitStrided && lb.mode == RESHARD_LB_NODE_AWARE && splitNumInjectionDomains > 0) {
    lb.numInjectionDomains = splitNumInjectionDomains;
    lb.domainsPerRep = (splitDomainsPerRep > 0) ? splitDomainsPerRep : 1;
    lb.strided = true;
  }

  const int firstRank = std::min(srcInterval.startRank, dstInterval.startRank);
  const int lastRank = std::max(srcInterval.endRank, dstInterval.endRank);
  size_t maxPeerGroup = 1;
  for (int rank = firstRank; rank < lastRank; rank++) {
    NCCL_M2N_CHECK(validateRankStagingCounts(worldRank, rank, srcTensor, &fullSrcInfo, dstTensor, &fullDstInfo, srcDims,
                                             srcStrides, dstDims, dstStrides, srcTensor->ndims, &lb, copyAlgo,
                                             gpusPerDomain, lb.dstNodeAnchor, &maxPeerGroup));
  }
  if (maxPeerGroupSize != nullptr) {
    *maxPeerGroupSize = maxPeerGroup;
  }
  return ncclSuccess;
}

/* ======================================================================
 * buildStagingTransferDescriptor — default (hierarchical) variant.
 * ====================================================================*/
ncclResult_t buildStagingTransferDescriptor(
  ncclComm_t globalComm, void* srcBuffer, const size_t* srcTensorDims, int ndims, const ncclDistTensor_t* srcTensor,
  void* dstBuffer, const size_t* dstTensorDims, const ncclDistTensor_t* dstTensor, int srcGpusPerDomain,
  int dstGpusPerDomain, int nodeLocalRank, StagingTransferDescriptor* desc, bool splitStrided,
  int splitNumInjectionDomains, int splitDomainsPerRep, bool nodeAnchorAtMeshStart, bool physicalLsaRanks) {
  if (!desc) {
    NCCL_M2N_FAIL(ncclInvalidArgument, -1, "buildStagingTransferDescriptor: output descriptor must be non-null");
  }
  memset(desc, 0, sizeof(*desc));

  const ncclMesh_t* srcMesh = srcTensor->mesh;
  const ncclMesh_t* dstMesh = dstTensor->mesh;

  int worldRank = 0;
  NCCL_M2N_CHECK(ncclCommUserRank(globalComm, &worldRank));

  bool isSource = reshardRankInMesh(srcMesh, worldRank);
  bool isDest = reshardRankInMesh(dstMesh, worldRank);

  desc->myWorldRank = worldRank;
  desc->isSource = isSource;
  desc->isDest = isDest;
  desc->srcBuffer = srcBuffer;
  desc->dstBuffer = dstBuffer;
  desc->ndims = ndims;

  ncclReshardMeshGroupInfo fullSrcInfo, fullDstInfo;
  computeMeshGroupInfo(srcTensor, srcMesh->startRank, &fullSrcInfo);
  computeMeshGroupInfo(dstTensor, dstMesh->startRank, &fullDstInfo);

  const ncclReshardShardLayout_t& srcShardLayout = fullSrcInfo.shardLayout;
  const ncclReshardShardLayout_t& dstShardLayout = fullDstInfo.shardLayout;
  int srcShardCount = fullSrcInfo.shardCount;
  int dstShardCount = fullDstInfo.shardCount;

  size_t srcDims[NCCL_RESHARD_MAX_TENSOR_DIMS] = {0};
  size_t dstDims[NCCL_RESHARD_MAX_TENSOR_DIMS] = {0};
  size_t srcStrides[NCCL_RESHARD_MAX_TENSOR_DIMS] = {0};
  size_t dstStrides[NCCL_RESHARD_MAX_TENSOR_DIMS] = {0};
  NCCL_M2N_CHECK(resolveReshardDims(worldRank, srcTensorDims, srcShardLayout, dstTensorDims,
                                    dstShardLayout, ndims, srcDims, dstDims));
  NCCL_M2N_CHECK(computeStridesChecked(srcDims, ndims, srcStrides));
  NCCL_M2N_CHECK(computeStridesChecked(dstDims, ndims, dstStrides));

  ncclReshardMeshGroupInfo srcInfo{}, dstInfo{};
  if (isSource) {
    computeMeshGroupInfo(srcTensor, worldRank, &srcInfo);
  }
  if (isDest) {
    computeMeshGroupInfo(dstTensor, worldRank, &dstInfo);
  }

  for (int d = 0; d < ndims; d++) {
    desc->srcDims[d] = srcDims[d];
    desc->dstDims[d] = dstDims[d];
    desc->srcStrides[d] = srcStrides[d];
    desc->dstStrides[d] = dstStrides[d];
  }
  NCCL_M2N_CHECK(computeMaxShardEdgeBytes(worldRank, srcDims, srcStrides, srcShardLayout, srcShardCount, dstDims,
                                          dstStrides, dstShardLayout, dstShardCount, ndims, &desc->maxEdgeBytes));
  desc->hasLocalFanout = fullDstInfo.repCount > 1;

  const int srcGpd = std::max(1, srcGpusPerDomain);
  const int dstGpd = std::max(1, dstGpusPerDomain);
  const int srcStart = srcMesh->startRank;
  const int srcEnd = srcStart + srcMesh->dims[0] * srcMesh->dims[1];
  const int dstStart = dstMesh->startRank;
  const int dstEnd = dstStart + dstMesh->dims[0] * dstMesh->dims[1];
  const int dstNodeAnchor = nodeAnchorAtMeshStart ? dstStart : 0;
  auto rankNode = [&](int rank) -> int {
    if (rank >= dstStart && rank < dstEnd) {
      return (rank - dstNodeAnchor) / dstGpd;
    }
    if (rank >= srcStart && rank < srcEnd) {
      return (rank - srcStart) / srcGpd;
    }
    return rank / dstGpd;
  };
  const int lsaStartRank = worldRank - nodeLocalRank;
  auto rankLocal = [&](int rank) -> int {
    if (physicalLsaRanks) {
      return rank - lsaStartRank;
    }
    if (rank >= dstStart && rank < dstEnd) {
      return (rank - dstStart) % dstGpd;
    }
    if (rank >= srcStart && rank < srcEnd) {
      return (rank - srcStart) % srcGpd;
    }
    return rank % dstGpd;
  };
  desc->myLocalRank = rankLocal(worldRank);

  ncclReshardRepLoadBalancer lb = {};
  lb.srcRepCount = fullSrcInfo.repCount;
  lb.dstRepCount = fullDstInfo.repCount;
  lb.dstGpusPerDomain = dstGpd;
  lb.dstRepStartRank = dstMesh->startRank;
  lb.dstRepStride = (fullDstInfo.repMeshDim == 0) ? dstMesh->dims[1] : 1;
  lb.mode = reshardEffectiveLbMode(srcTensor, dstTensor);
  lb.dstNodeAnchor = dstNodeAnchor;
  if (splitStrided && lb.mode == RESHARD_LB_NODE_AWARE && splitNumInjectionDomains > 0) {
    lb.numInjectionDomains = splitNumInjectionDomains;
    lb.domainsPerRep = (splitDomainsPerRep > 0) ? splitDomainsPerRep : 1;
    lb.strided = true;
  }

  const size_t epc_dummy = 1;

  /* ============================================================
   * 5. SOURCE SIDE: populate targets[] (hierarchical leader pick).
   * ============================================================ */
  if (isSource) {
    int targetRepStart, targetRepEnd;
    getTargetRepRange(&lb, srcInfo.repIdx, &targetRepStart, &targetRepEnd);

    desc->numTargets = 0;
    bool targetsTruncated = false;

    for (int dstShard = 0; dstShard < dstShardCount; dstShard++) {
      ncclReshardTransferPlan plan;
      NCCL_M2N_CHECK(computeTransferPlanChecked(srcDims, srcStrides, srcShardLayout, srcInfo.shardIdx, dstDims, dstStrides,
                                                dstShardLayout, dstShard, ndims, epc_dummy, &plan));

      if (plan.totalInnerTransfers == 0) {
        continue;
      }
      if (desc->numTargets >= MAX_TARGETS) {
        targetsTruncated = true;
        break;
      }
      if (targetRepStart >= targetRepEnd) {
        continue;
      }

      int numSourcesToDstShard = 0;
      int myPosition = 0;
      for (int ss = 0; ss < srcShardCount; ss++) {
        ncclReshardTransferPlan check;
        NCCL_M2N_CHECK(computeTransferPlanChecked(srcDims, srcStrides, srcShardLayout, ss, dstDims, dstStrides,
                                                  dstShardLayout, dstShard, ndims, epc_dummy, &check));
        if (check.totalInnerTransfers > 0) {
          if (ss < srcInfo.shardIdx) {
            myPosition++;
          }
          numSourcesToDstShard++;
        }
      }

      int firstRepRank = getMeshRank(dstTensor, &fullDstInfo, dstShard, targetRepStart);
      int firstRepNode = rankNode(firstRepRank);

      int localRepsOnTargetNode[MAX_LOCAL_FOLLOWERS + 1];
      int numLocalRepsOnTargetNode = 0;
      for (int rep = targetRepStart; rep < targetRepEnd; rep++) {
        int rr = getMeshRank(dstTensor, &fullDstInfo, dstShard, rep);
        int rrNode = rankNode(rr);
        if (rrNode == firstRepNode) {
          if (numLocalRepsOnTargetNode < MAX_LOCAL_FOLLOWERS + 1) {
            localRepsOnTargetNode[numLocalRepsOnTargetNode++] = rep;
          }
        }
      }

      NCCL_M2N_CHECK_ARG(numLocalRepsOnTargetNode > 0, worldRank, "[STAGING] no local target reps for dst shard %d",
                         dstShard);
      int targetLocalRepIdx = pickLocalRepIndex(myPosition, numSourcesToDstShard, numLocalRepsOnTargetNode);
      int leaderSourceStart = 0;
      int leaderSourceEnd = 0;
      computeBalancedSourceRange(numSourcesToDstShard, numLocalRepsOnTargetNode, targetLocalRepIdx, &leaderSourceStart,
                                 &leaderSourceEnd);

      int leaderRep = localRepsOnTargetNode[targetLocalRepIdx];
      int leaderRank = getMeshRank(dstTensor, &fullDstInfo, dstShard, leaderRep);

      int ti = desc->numTargets++;
      StagingPeerDescriptor* td = &desc->targets[ti];
      td->peerWorldRank = leaderRank;
      td->peerShardIdx = dstShard;
      td->isRdma = true;
      td->peerLocalRank = -1;
      NCCL_M2N_CHECK(fillStagingPlan(&td->plan, plan, ndims));

      desc->destNumSources[ti] = numSourcesToDstShard;
      desc->sourceIndexOnDest[ti] = myPosition;
      desc->destNumRdmaSources[ti] = leaderSourceEnd - leaderSourceStart;
      desc->rdmaSourceIndexOnDest[ti] = myPosition - leaderSourceStart;

      RESHARD_TRACE(worldRank,
                    "  SRC target[%d]: dstShard=%d -> leader rank=%d "
                    "(isRdma=%d, local_rank=%d)",
                    ti, dstShard, leaderRank, td->isRdma, td->peerLocalRank);
    }

    if (targetsTruncated) {
      NCCL_M2N_FAIL(ncclInvalidArgument, worldRank,
                    "buildStagingTransferDescriptor: target list truncated at %d/%d entries; increase MAX_TARGETS",
                    desc->numTargets, MAX_TARGETS);
    }
  }

  /* ============================================================
   * 6. DEST SIDE: populate sources[] + LSA fan-out + ring fwd.
   * ============================================================ */
  if (isDest) {
    int sourceRep = getSourceRepForDest(&lb, dstInfo.repIdx);
    int targetRepStart, targetRepEnd;
    getTargetRepRange(&lb, sourceRep, &targetRepStart, &targetRepEnd);

    int myNode = rankNode(worldRank);

    int localReps[MAX_LOCAL_FOLLOWERS + 1];
    int numLocalReps = 0;
    int myLocalRepIdx = 0;

    for (int rep = targetRepStart; rep < targetRepEnd; rep++) {
      int rr = getMeshRank(dstTensor, &fullDstInfo, dstInfo.shardIdx, rep);
      int rrNode = rankNode(rr);
      if (rrNode == myNode) {
        if (rep == dstInfo.repIdx) {
          myLocalRepIdx = numLocalReps;
        }
        if (numLocalReps < MAX_LOCAL_FOLLOWERS + 1) {
          localReps[numLocalReps++] = rep;
        }
      }
    }

    RESHARD_TRACE(worldRank,
                  "  DST local_rep_discovery: numLocalReps=%d, "
                  "myLocalRepIdx=%d, myNode=%d",
                  numLocalReps, myLocalRepIdx, myNode);
    if (reshardGetLogLevel() >= RESHARD_LOG_TRACE) {
      for (int r = 0; r < numLocalReps; r++) {
        int rr = getMeshRank(dstTensor, &fullDstInfo, dstInfo.shardIdx, localReps[r]);
        RESHARD_TRACE(worldRank, "    local_rep[%d]: rep=%d, rank=%d, node=%d%s", r, localReps[r], rr, rankNode(rr),
                      (r == myLocalRepIdx) ? " <- ME" : "");
      }
    }

    int firstNodeLocalReps = 0;
    int firstRepNodeDest = -1;
    if (targetRepStart < targetRepEnd) {
      int frr = getMeshRank(dstTensor, &fullDstInfo, dstInfo.shardIdx, targetRepStart);
      firstRepNodeDest = rankNode(frr);
      for (int rep = targetRepStart; rep < targetRepEnd; rep++) {
        int rr = getMeshRank(dstTensor, &fullDstInfo, dstInfo.shardIdx, rep);
        if (rankNode(rr) == firstRepNodeDest) {
          firstNodeLocalReps++;
        }
      }
    }

    int allSourceShards[MAX_SOURCES];
    int numAllSources = 0;
    bool sourcesTruncated = false;
    for (int ss = 0; ss < srcShardCount; ss++) {
      ncclReshardTransferPlan check;
      NCCL_M2N_CHECK(computeTransferPlanChecked(srcDims, srcStrides, srcShardLayout, ss, dstDims, dstStrides, dstShardLayout,
                                                dstInfo.shardIdx, ndims, epc_dummy, &check));
      if (check.totalInnerTransfers > 0 && numAllSources < MAX_SOURCES) {
        allSourceShards[numAllSources++] = ss;
      } else if (check.totalInnerTransfers > 0) {
        sourcesTruncated = true;
      }
    }
    if (sourcesTruncated) {
      NCCL_M2N_FAIL(ncclInvalidArgument, worldRank,
                    "buildStagingTransferDescriptor: allSourceShards truncated (%d / MAX_SOURCES=%d); increase "
                    "MAX_SOURCES",
                    numAllSources, MAX_SOURCES);
    }

    int mySourceStart = 0, mySourceEnd = 0;
    int sourceRepSlots = firstNodeLocalReps > 0 ? firstNodeLocalReps : numLocalReps;
    int activeSourceSlots = sourceRepSlots;
    for (int node = firstRepNodeDest; node >= 0 && node <= myNode; node++) {
      int nodeLocalReps = 0;
      for (int rep = targetRepStart; rep < targetRepEnd; rep++) {
        int rr = getMeshRank(dstTensor, &fullDstInfo, dstInfo.shardIdx, rep);
        if (rankNode(rr) == node) {
          nodeLocalReps++;
        }
      }
      if (nodeLocalReps > 0 && nodeLocalReps < activeSourceSlots) {
        activeSourceSlots = nodeLocalReps;
      }
    }

    int mySourceSlotStart = myLocalRepIdx;
    int mySourceSlotEnd = myLocalRepIdx + 1;
    if (activeSourceSlots > 0 && myLocalRepIdx == activeSourceSlots - 1 && sourceRepSlots > activeSourceSlots) {
      mySourceSlotEnd = sourceRepSlots;
    }

    if (sourceRepSlots > 0 && mySourceSlotStart < activeSourceSlots && numAllSources > 0) {
      int spr = numAllSources / sourceRepSlots;
      int extra = numAllSources % sourceRepSlots;
      int threshold = extra * (spr + 1);

      if (mySourceSlotStart < extra) {
        mySourceStart = mySourceSlotStart * (spr + 1);
      } else {
        mySourceStart = threshold + (mySourceSlotStart - extra) * spr;
      }

      if (mySourceSlotEnd >= sourceRepSlots) {
        mySourceEnd = numAllSources;
      } else if (mySourceSlotEnd < extra) {
        mySourceEnd = mySourceSlotEnd * (spr + 1);
      } else {
        mySourceEnd = threshold + (mySourceSlotEnd - extra) * spr;
      }
    }

    int repSrcStart[MAX_LOCAL_FOLLOWERS + 1];
    int repSrcEnd[MAX_LOCAL_FOLLOWERS + 1];
    memset(repSrcStart, 0, sizeof(repSrcStart));
    memset(repSrcEnd, 0, sizeof(repSrcEnd));
    if (sourceRepSlots > 0 && numAllSources > 0) {
      int spr = numAllSources / sourceRepSlots;
      int extra = numAllSources % sourceRepSlots;
      int threshold = extra * (spr + 1);
      for (int r = 0; r < numLocalReps; r++) {
        int slotStart = r;
        int slotEnd = r + 1;
        if (activeSourceSlots > 0 && r == activeSourceSlots - 1 && sourceRepSlots > activeSourceSlots) {
          slotEnd = sourceRepSlots;
        }

        if (slotStart < extra) {
          repSrcStart[r] = slotStart * (spr + 1);
        } else {
          repSrcStart[r] = threshold + (slotStart - extra) * spr;
        }

        if (slotEnd >= sourceRepSlots) {
          repSrcEnd[r] = numAllSources;
        } else if (slotEnd < extra) {
          repSrcEnd[r] = slotEnd * (spr + 1);
        } else {
          repSrcEnd[r] = threshold + (slotEnd - extra) * spr;
        }
      }
    }

    /* ring_prev / ring_next */
    int ringPrevRank = -1;
    int ringNextRank = -1;
    int ringNextNumRdma = 0;
    int ringNextSrcStart = 0;

    int ringPrevNode = -1;
    for (int rep = targetRepStart; rep < targetRepEnd; rep++) {
      int rr = getMeshRank(dstTensor, &fullDstInfo, dstInfo.shardIdx, rep);
      int rrNode = rankNode(rr);
      if (rrNode < myNode && (ringPrevNode == -1 || rrNode > ringPrevNode)) {
        ringPrevNode = rrNode;
      }
    }
    if (ringPrevNode >= 0) {
      int prevLocalIdx = 0;
      for (int rep = targetRepStart; rep < targetRepEnd; rep++) {
        int rr = getMeshRank(dstTensor, &fullDstInfo, dstInfo.shardIdx, rep);
        int rrNode = rankNode(rr);
        if (rrNode == ringPrevNode) {
          if (prevLocalIdx == myLocalRepIdx) {
            ringPrevRank = rr;
            break;
          }
          prevLocalIdx++;
        }
      }
    }
    int ringNextNode = -1;
    for (int rep = targetRepStart; rep < targetRepEnd; rep++) {
      int rr = getMeshRank(dstTensor, &fullDstInfo, dstInfo.shardIdx, rep);
      int rrNode = rankNode(rr);
      if (rrNode > myNode && (ringNextNode == -1 || rrNode < ringNextNode)) {
        ringNextNode = rrNode;
      }
    }
    if (ringNextNode >= 0) {
      int ringNodeLocalReps = 0;
      for (int rep = targetRepStart; rep < targetRepEnd; rep++) {
        int rr = getMeshRank(dstTensor, &fullDstInfo, dstInfo.shardIdx, rep);
        if (rankNode(rr) == ringNextNode) {
          ringNodeLocalReps++;
        }
      }
      int targetRingLocalRepIdx = myLocalRepIdx;
      int nextActive = activeSourceSlots;
      if (ringNodeLocalReps > 0 && nextActive > ringNodeLocalReps) {
        nextActive = ringNodeLocalReps;
      }
      if (nextActive > 0 && targetRingLocalRepIdx >= nextActive) {
        targetRingLocalRepIdx = nextActive - 1;
      }

      /* Match ring_next's ringPrev calculation: its inbound RDMA ordinal is
       * partitioned by this sender node's local handlers, not by its own. */
      int ringNextSrcEnd = 0;
      computeBalancedSourceRange(numAllSources, numLocalReps, targetRingLocalRepIdx, &ringNextSrcStart,
                                 &ringNextSrcEnd);
      ringNextNumRdma = ringNextSrcEnd - ringNextSrcStart;

      int nextLocalIdx = 0;
      for (int rep = targetRepStart; rep < targetRepEnd; rep++) {
        int rr = getMeshRank(dstTensor, &fullDstInfo, dstInfo.shardIdx, rep);
        int rrNode = rankNode(rr);
        if (rrNode == ringNextNode) {
          if (nextLocalIdx == targetRingLocalRepIdx) {
            ringNextRank = rr;
            break;
          }
          nextLocalIdx++;
        }
      }
    }

    RESHARD_TRACE(worldRank, "  DST ring: prev_rank=%d (node=%d), next_rank=%d (node=%d)", ringPrevRank, ringPrevNode,
                  ringNextRank, ringNextNode);

    int ringPrevNumLocalReps = 0;
    int ringPrevNumRdma = 0;
    int ringPrevSrcStart = 0;
    if (ringPrevRank != -1 && ringPrevNode >= 0) {
      for (int rep = targetRepStart; rep < targetRepEnd; rep++) {
        int rr = getMeshRank(dstTensor, &fullDstInfo, dstInfo.shardIdx, rep);
        int rrNode = rankNode(rr);
        if (rrNode == ringPrevNode) {
          ringPrevNumLocalReps++;
        }
      }
      if (ringPrevNumLocalReps > 0 && numAllSources > 0) {
        int prevSpr = numAllSources / ringPrevNumLocalReps;
        int prevExtra = numAllSources % ringPrevNumLocalReps;
        int prevSrcEnd;
        if (myLocalRepIdx < prevExtra) {
          ringPrevSrcStart = myLocalRepIdx * (prevSpr + 1);
          prevSrcEnd = ringPrevSrcStart + prevSpr + 1;
        } else {
          ringPrevSrcStart = prevExtra * (prevSpr + 1) + (myLocalRepIdx - prevExtra) * prevSpr;
          prevSrcEnd = ringPrevSrcStart + prevSpr;
        }
        ringPrevNumRdma = prevSrcEnd - ringPrevSrcStart;
      }
    }

    auto sourceTargetIndexForDstShard = [&](int ss, int targetDstShard) -> int {
      int srcTargetCount = 0;
      for (int ds = 0; ds < dstShardCount; ds++) {
        ncclReshardTransferPlan dsPlan;
        computeTransferPlan(srcDims, srcStrides, srcShardLayout, ss, dstDims, dstStrides, dstShardLayout, ds, ndims,
                            epc_dummy, &dsPlan);
        if (dsPlan.totalInnerTransfers == 0) {
          continue;
        }
        if (ds == targetDstShard) {
          return srcTargetCount;
        }
        srcTargetCount++;
      }
      return 0;
    };

    /* Populate sources[] in shard order, classifying RDMA vs LSA. */
    desc->numSources = 0;
    for (int idx = 0; idx < numAllSources; idx++) {
      int ss = allSourceShards[idx];

      int handlerRepIdx = -1;
      for (int r = 0; r < numLocalReps; r++) {
        if (idx >= repSrcStart[r] && idx < repSrcEnd[r]) {
          handlerRepIdx = r;
          break;
        }
      }
      NCCL_M2N_CHECK_ARG(handlerRepIdx >= 0, worldRank, "[STAGING] no local handler for source index %d", idx);
      bool isMyRdma = (handlerRepIdx == myLocalRepIdx);

      ncclReshardTransferPlan plan;
      NCCL_M2N_CHECK(computeTransferPlanChecked(srcDims, srcStrides, srcShardLayout, ss, dstDims, dstStrides, dstShardLayout,
                                                dstInfo.shardIdx, ndims, epc_dummy, &plan));

      if (desc->numSources >= MAX_SOURCES) {
        break;
      }

      int si = desc->numSources++;
      StagingPeerDescriptor* sd = &desc->sources[si];

      if (isMyRdma) {
        if (ringPrevRank != -1) {
          sd->peerWorldRank = ringPrevRank;
          sd->peerShardIdx = ss;
          sd->isRdma = true;
          sd->peerLocalRank = -1;
        } else {
          int srcRank = getMeshRank(srcTensor, &fullSrcInfo, ss, sourceRep);
          sd->peerWorldRank = srcRank;
          sd->peerShardIdx = ss;
          sd->isRdma = true;
          sd->peerLocalRank = -1;
        }
      } else {
        int handlerRep = localReps[handlerRepIdx];
        int handlerRank = getMeshRank(dstTensor, &fullDstInfo, dstInfo.shardIdx, handlerRep);
        sd->peerWorldRank = handlerRank;
        sd->peerShardIdx = ss;
        sd->isRdma = false;
        sd->peerLocalRank = rankLocal(handlerRank);
      }
      NCCL_M2N_CHECK(fillStagingPlan(&sd->plan, plan, ndims));

      RESHARD_TRACE(worldRank,
                    "  DST source[%d]: shard=%d peer_rank=%d isRdma=%d "
                    "local_rank=%d (%s)",
                    si, ss, sd->peerWorldRank, sd->isRdma, sd->peerLocalRank,
                    isMyRdma ? (ringPrevRank != -1 ? "RDMA-via-ring" : "RDMA-direct") : "LSA-handler");

      /* Cross-indices for sources[si]. */
      if (isMyRdma) {
        if (ringPrevRank != -1) {
          int k = idx - ringPrevSrcStart;
          int ringPrevLsaCount = ringPrevNumRdma * (ringPrevNumLocalReps - 1);
          desc->targetIndexOnSource[si] = ringPrevLsaCount + k;
          desc->rdmaTargetIndexOnSource[si] = k;
          desc->channelTargetIndexOnSource[si] = sourceTargetIndexForDstShard(ss, dstInfo.shardIdx);
          desc->sourceNumTargets[si] = ringPrevNumRdma * ringPrevNumLocalReps;
          desc->sourceNumRdmaTargets[si] = ringPrevNumRdma;
        } else {
          int myTargetIdxOnSource = 0;
          int sourceNumTargetsCount = 0;
          int srcTargetCount = 0;
          bool found = false;

          int srcTrepStart, srcTrepEnd;
          getTargetRepRange(&lb, sourceRep, &srcTrepStart, &srcTrepEnd);

          for (int ds = 0; ds < dstShardCount; ds++) {
            ncclReshardTransferPlan dsPlan;
            NCCL_M2N_CHECK(computeTransferPlanChecked(srcDims, srcStrides, srcShardLayout, ss, dstDims, dstStrides,
                                                      dstShardLayout, ds, ndims, epc_dummy, &dsPlan));
            if (dsPlan.totalInnerTransfers == 0) {
              continue;
            }
            if (srcTrepStart >= srcTrepEnd) {
              continue;
            }

            int nSrcToDs = 0;
            int posSs = 0;
            for (int qq = 0; qq < srcShardCount; qq++) {
              ncclReshardTransferPlan q;
              NCCL_M2N_CHECK(computeTransferPlanChecked(srcDims, srcStrides, srcShardLayout, qq, dstDims, dstStrides,
                                                        dstShardLayout, ds, ndims, epc_dummy, &q));
              if (q.totalInnerTransfers > 0) {
                if (qq < ss) {
                  posSs++;
                }
                nSrcToDs++;
              }
            }

            int fr = getMeshRank(dstTensor, &fullDstInfo, ds, srcTrepStart);
            int frNode = rankNode(fr);

            int lrOnTn[MAX_LOCAL_FOLLOWERS + 1];
            int nlr = 0;
            for (int rr = srcTrepStart; rr < srcTrepEnd; rr++) {
              int rrk = getMeshRank(dstTensor, &fullDstInfo, ds, rr);
              int rrn = rankNode(rrk);
              if (rrn == frNode) {
                if (nlr < MAX_LOCAL_FOLLOWERS + 1) {
                  lrOnTn[nlr++] = rr;
                }
              }
            }

            NCCL_M2N_CHECK_ARG(nlr > 0, worldRank, "[STAGING] no local target reps for dst shard %d", ds);
            int tlri = pickLocalRepIndex(posSs, nSrcToDs, nlr);

            int lr = lrOnTn[tlri];
            int lrRank = getMeshRank(dstTensor, &fullDstInfo, ds, lr);

            if (lrRank == worldRank && !found) {
              myTargetIdxOnSource = srcTargetCount;
              found = true;
            }
            srcTargetCount++;
          }
          sourceNumTargetsCount = srcTargetCount;
          desc->targetIndexOnSource[si] = myTargetIdxOnSource;
          desc->rdmaTargetIndexOnSource[si] = myTargetIdxOnSource;
          desc->channelTargetIndexOnSource[si] = desc->targetIndexOnSource[si];
          desc->sourceNumTargets[si] = sourceNumTargetsCount;
          desc->sourceNumRdmaTargets[si] = sourceNumTargetsCount;
        }
      } else {
        int shardOffsetInRange = idx - repSrcStart[handlerRepIdx];
        int posInOthers;
        if (myLocalRepIdx < handlerRepIdx) {
          posInOthers = myLocalRepIdx;
        } else {
          posInOthers = myLocalRepIdx - 1;
        }

        int providerNumRdma = repSrcEnd[handlerRepIdx] - repSrcStart[handlerRepIdx];

        int lsaTargetIdxOnProvider = shardOffsetInRange * (numLocalReps - 1) + posInOthers;
        int handlerTargetIdxOnSource = 0;
        int srcTargetCount = 0;
        bool foundHandlerTarget = false;
        for (int ds = 0; ds < dstShardCount; ds++) {
          ncclReshardTransferPlan dsPlan;
          NCCL_M2N_CHECK(computeTransferPlanChecked(srcDims, srcStrides, srcShardLayout, ss, dstDims, dstStrides,
                                                    dstShardLayout, ds, ndims, epc_dummy, &dsPlan));
          if (dsPlan.totalInnerTransfers == 0) {
            continue;
          }
          if (targetRepStart >= targetRepEnd) {
            continue;
          }

          int nSrcToDs = 0;
          int posSs = 0;
          for (int qq = 0; qq < srcShardCount; qq++) {
            ncclReshardTransferPlan q;
            NCCL_M2N_CHECK(computeTransferPlanChecked(srcDims, srcStrides, srcShardLayout, qq, dstDims, dstStrides,
                                                      dstShardLayout, ds, ndims, epc_dummy, &q));
            if (q.totalInnerTransfers > 0) {
              if (qq < ss) {
                posSs++;
              }
              nSrcToDs++;
            }
          }

          int fr = getMeshRank(dstTensor, &fullDstInfo, ds, targetRepStart);
          int frNode = rankNode(fr);

          int lrOnTn[MAX_LOCAL_FOLLOWERS + 1];
          int nlr = 0;
          for (int rr = targetRepStart; rr < targetRepEnd; rr++) {
            int rrk = getMeshRank(dstTensor, &fullDstInfo, ds, rr);
            int rrn = rankNode(rrk);
            if (rrn == frNode) {
              if (nlr < MAX_LOCAL_FOLLOWERS + 1) {
                lrOnTn[nlr++] = rr;
              }
            }
          }

          NCCL_M2N_CHECK_ARG(nlr > 0, worldRank, "[STAGING] no local handler reps for dst shard %d", ds);
          int tlri = pickLocalRepIndex(posSs, nSrcToDs, nlr);

          int lr = lrOnTn[tlri];
          int lrRank = getMeshRank(dstTensor, &fullDstInfo, ds, lr);
          if (ds == dstInfo.shardIdx && lrRank == sd->peerWorldRank && !foundHandlerTarget) {
            handlerTargetIdxOnSource = srcTargetCount;
            foundHandlerTarget = true;
          }
          srcTargetCount++;
        }

        desc->targetIndexOnSource[si] = handlerTargetIdxOnSource;
        desc->channelTargetIndexOnSource[si] =
          (ringPrevRank != -1) ? sourceTargetIndexForDstShard(ss, dstInfo.shardIdx) : handlerTargetIdxOnSource;
        desc->sourceNumTargets[si] = providerNumRdma * std::max(0, numLocalReps - 1);
        desc->sourceLsaHeadIndexOnProvider[si] = lsaTargetIdxOnProvider;
      }
    }

    /* numLsaFollowers carries the per-source fan-out factor into
     * the kernel (lsaTargets[ch][s * numLsaFollowers + f]). */
    desc->numLsaFollowers = (numLocalReps > 0) ? (numLocalReps - 1) : 0;
    int numRdmaSources = 0;
    for (int si = 0; si < desc->numSources; si++) {
      if (desc->sources[si].isRdma) {
        numRdmaSources++;
      }
    }
    NCCL_M2N_CHECK_ARG(stagingLsaFollowersFitKernelCapacity(desc->numLsaFollowers), worldRank,
                       "[STAGING] LSA fan-out followers exceed kernel capacity: followers=%d max=%d",
                       desc->numLsaFollowers, STAGING_LSA_FANOUT_MAX_FOLLOWERS);
    NCCL_M2N_CHECK_ARG(stagingLsaFanoutFitsTargetCapacity(numRdmaSources, desc->numLsaFollowers), worldRank,
                       "[STAGING] LSA fan-out target count exceeds staging target capacity: rdmaSources=%d "
                       "followers=%d max=%d",
                       numRdmaSources, desc->numLsaFollowers, MAX_TARGETS);

    /* 6b. LSA fan-out targets (dest leader → followers). */
    for (int idx = mySourceStart; idx < mySourceEnd; idx++) {
      int ss = allSourceShards[idx];

      ncclReshardTransferPlan plan;
      NCCL_M2N_CHECK(computeTransferPlanChecked(srcDims, srcStrides, srcShardLayout, ss, dstDims, dstStrides, dstShardLayout,
                                                dstInfo.shardIdx, ndims, epc_dummy, &plan));

      for (int r = 0; r < numLocalReps; r++) {
        if (r == myLocalRepIdx) {
          continue;
        }
        if (desc->numTargets >= MAX_TARGETS) {
          NCCL_M2N_FAIL(ncclInvalidArgument, worldRank,
                        "buildStagingTransferDescriptor: LSA fan-out target list truncated (%d / MAX_TARGETS=%d); "
                        "increase MAX_TARGETS",
                        desc->numTargets, MAX_TARGETS);
        }
        int rep = localReps[r];
        int repRank = getMeshRank(dstTensor, &fullDstInfo, dstInfo.shardIdx, rep);

        int ti = desc->numTargets++;
        StagingPeerDescriptor* td = &desc->targets[ti];
        td->peerWorldRank = repRank;
        td->peerShardIdx = dstInfo.shardIdx;
        td->isRdma = false;
        td->peerLocalRank = rankLocal(repRank);
        NCCL_M2N_CHECK(fillStagingPlan(&td->plan, plan, ndims));

        desc->destNumSources[ti] = numAllSources;
        desc->sourceIndexOnDest[ti] = idx;

        RESHARD_TRACE(worldRank,
                      "  DST lsa_target[%d]: repRank=%d "
                      "(isRdma=%d, local_rank=%d) -> LSA follower",
                      ti, repRank, td->isRdma, td->peerLocalRank);
      }
    }

    /* 6c. Ring RDMA targets to ring_next. */
    desc->numRingTargets = 0;
    if (ringNextRank != -1) {
      for (int idx = mySourceStart; idx < mySourceEnd; idx++) {
        int ss = allSourceShards[idx];
        if (desc->numTargets >= MAX_TARGETS) {
          NCCL_M2N_FAIL(ncclInvalidArgument, worldRank,
                        "buildStagingTransferDescriptor: ring target list truncated (%d / MAX_TARGETS=%d); increase "
                        "MAX_TARGETS",
                        desc->numTargets, MAX_TARGETS);
        }
        ncclReshardTransferPlan plan;
        NCCL_M2N_CHECK(computeTransferPlanChecked(srcDims, srcStrides, srcShardLayout, ss, dstDims, dstStrides,
                                                  dstShardLayout, dstInfo.shardIdx, ndims, epc_dummy, &plan));

        int ti = desc->numTargets++;
        desc->numRingTargets++;

        StagingPeerDescriptor* td = &desc->targets[ti];
        td->peerWorldRank = ringNextRank;
        td->peerShardIdx = dstInfo.shardIdx;
        td->isRdma = true;
        td->peerLocalRank = -1;
        NCCL_M2N_CHECK(fillStagingPlan(&td->plan, plan, ndims));

        desc->destNumSources[ti] = numAllSources;
        desc->sourceIndexOnDest[ti] = idx;
        desc->destNumRdmaSources[ti] = ringNextNumRdma;
        desc->rdmaSourceIndexOnDest[ti] = idx - ringNextSrcStart;
        NCCL_M2N_CHECK_ARG(
          desc->rdmaSourceIndexOnDest[ti] >= 0 && desc->rdmaSourceIndexOnDest[ti] < desc->destNumRdmaSources[ti],
          worldRank, "[STAGING] ring RDMA source ordinal %d/%d is outside next handler range for source %d",
          desc->rdmaSourceIndexOnDest[ti], desc->destNumRdmaSources[ti], idx);

        RESHARD_TRACE(worldRank,
                      "  DST ring_target[%d]: next_rank=%d "
                      "(isRdma=%d) -> RING NEXT",
                      ti, ringNextRank, td->isRdma);
      }
    }
  }

  RESHARD_TRACE(worldRank,
                "  SUMMARY: numTargets=%d (ring=%d), numSources=%d, "
                "is_src=%d, is_dst=%d",
                desc->numTargets, desc->numRingTargets, desc->numSources, desc->isSource, desc->isDest);

  desc->peerGroupSizeBound = computeStagingPeerGroupSizeBound(desc);
  desc->ctaHeuristicPeerCount = std::max(desc->numTargets, desc->numSources);

  return ncclSuccess;
}

/* ======================================================================
 * buildStagingDirectTransferDescriptor — direct (no LSA, no ring).
 * ====================================================================*/
ncclResult_t buildStagingDirectTransferDescriptor(ncclComm_t globalComm, void* srcBuffer, const size_t* srcTensorDims,
                                                  int ndims, const ncclDistTensor_t* srcTensor, void* dstBuffer,
                                                  const size_t* dstTensorDims, const ncclDistTensor_t* dstTensor,
                                                  int gpusPerDomain, int nodeLocalRank,
                                                  StagingTransferDescriptor* desc) {
  if (!desc) {
    NCCL_M2N_FAIL(ncclInvalidArgument, -1, "buildStagingDirectTransferDescriptor: output descriptor must be non-null");
  }
  memset(desc, 0, sizeof(*desc));

  const ncclMesh_t* srcMesh = srcTensor->mesh;
  const ncclMesh_t* dstMesh = dstTensor->mesh;

  int worldRank = 0;
  NCCL_M2N_CHECK(ncclCommUserRank(globalComm, &worldRank));

  bool isSource = reshardRankInMesh(srcMesh, worldRank);
  bool isDest = reshardRankInMesh(dstMesh, worldRank);

  desc->myWorldRank = worldRank;
  desc->myLocalRank = nodeLocalRank;
  desc->isSource = isSource;
  desc->isDest = isDest;
  desc->srcBuffer = srcBuffer;
  desc->dstBuffer = dstBuffer;
  desc->ndims = ndims;

  ncclReshardMeshGroupInfo fullSrcInfo, fullDstInfo;
  computeMeshGroupInfo(srcTensor, srcMesh->startRank, &fullSrcInfo);
  computeMeshGroupInfo(dstTensor, dstMesh->startRank, &fullDstInfo);

  const auto& srcShardLayout = fullSrcInfo.shardLayout;
  const auto& dstShardLayout = fullDstInfo.shardLayout;
  int srcShardCount = fullSrcInfo.shardCount;
  int dstShardCount = fullDstInfo.shardCount;

  size_t srcDims[NCCL_RESHARD_MAX_TENSOR_DIMS] = {0};
  size_t dstDims[NCCL_RESHARD_MAX_TENSOR_DIMS] = {0};
  size_t srcStrides[NCCL_RESHARD_MAX_TENSOR_DIMS] = {0};
  size_t dstStrides[NCCL_RESHARD_MAX_TENSOR_DIMS] = {0};
  NCCL_M2N_CHECK(resolveReshardDims(worldRank, srcTensorDims, srcShardLayout, dstTensorDims,
                                    dstShardLayout, ndims, srcDims, dstDims));
  NCCL_M2N_CHECK(computeStridesChecked(srcDims, ndims, srcStrides));
  NCCL_M2N_CHECK(computeStridesChecked(dstDims, ndims, dstStrides));

  ncclReshardMeshGroupInfo srcInfo{}, dstInfo{};
  if (isSource) {
    computeMeshGroupInfo(srcTensor, worldRank, &srcInfo);
  }
  if (isDest) {
    computeMeshGroupInfo(dstTensor, worldRank, &dstInfo);
  }

  for (int d = 0; d < ndims; d++) {
    desc->srcDims[d] = srcDims[d];
    desc->dstDims[d] = dstDims[d];
    desc->srcStrides[d] = srcStrides[d];
    desc->dstStrides[d] = dstStrides[d];
  }

  ncclReshardRepLoadBalancer lb = {};
  lb.srcRepCount = fullSrcInfo.repCount;
  lb.dstRepCount = fullDstInfo.repCount;
  lb.dstGpusPerDomain = gpusPerDomain;
  lb.dstRepStartRank = dstMesh->startRank;
  lb.dstRepStride = (fullDstInfo.repMeshDim == 0) ? dstMesh->dims[1] : 1;
  lb.mode = reshardEffectiveLbMode(srcTensor, dstTensor);

  const size_t epc_dummy = 1;

  if (isSource) {
    int targetRepStart, targetRepEnd;
    getTargetRepRange(&lb, srcInfo.repIdx, &targetRepStart, &targetRepEnd);

    desc->numTargets = 0;
    bool targetsTruncated = false;

    for (int dstShard = 0; dstShard < dstShardCount; dstShard++) {
      ncclReshardTransferPlan plan;
      NCCL_M2N_CHECK(computeTransferPlanChecked(srcDims, srcStrides, srcShardLayout, srcInfo.shardIdx, dstDims, dstStrides,
                                                dstShardLayout, dstShard, ndims, epc_dummy, &plan));

      if (plan.totalInnerTransfers == 0) {
        continue;
      }
      if (targetRepStart >= targetRepEnd) {
        continue;
      }

      int numSourcesToDstShard = 0;
      int myPosition = 0;
      for (int ss = 0; ss < srcShardCount; ss++) {
        ncclReshardTransferPlan check;
        NCCL_M2N_CHECK(computeTransferPlanChecked(srcDims, srcStrides, srcShardLayout, ss, dstDims, dstStrides,
                                                  dstShardLayout, dstShard, ndims, epc_dummy, &check));
        if (check.totalInnerTransfers > 0) {
          if (ss < srcInfo.shardIdx) {
            myPosition++;
          }
          numSourcesToDstShard++;
        }
      }

      for (int dstRep = targetRepStart; dstRep < targetRepEnd; dstRep++) {
        if (desc->numTargets >= MAX_TARGETS) {
          targetsTruncated = true;
          break;
        }
        int dstRank = getMeshRank(dstTensor, &fullDstInfo, dstShard, dstRep);

        int ti = desc->numTargets++;
        StagingPeerDescriptor* td = &desc->targets[ti];
        td->peerWorldRank = dstRank;
        td->peerShardIdx = dstShard;
        td->isRdma = true;
        td->peerLocalRank = -1;
        NCCL_M2N_CHECK(fillStagingPlan(&td->plan, plan, ndims));

        desc->destNumSources[ti] = numSourcesToDstShard;
        desc->sourceIndexOnDest[ti] = myPosition;
        desc->destNumRdmaSources[ti] = numSourcesToDstShard;
        desc->rdmaSourceIndexOnDest[ti] = myPosition;
      }
      if (targetsTruncated) {
        break;
      }
    }

    if (targetsTruncated) {
      NCCL_M2N_FAIL(ncclInvalidArgument, worldRank,
                    "buildStagingDirectTransferDescriptor: target list truncated (%d / MAX_TARGETS=%d); increase "
                    "MAX_TARGETS",
                    desc->numTargets, MAX_TARGETS);
    }
  }

  if (isDest) {
    int sourceRep = getSourceRepForDest(&lb, dstInfo.repIdx);
    int targetRepStart, targetRepEnd;
    getTargetRepRange(&lb, sourceRep, &targetRepStart, &targetRepEnd);
    int numTargetReps = targetRepEnd - targetRepStart;

    desc->numSources = 0;

    for (int ss = 0; ss < srcShardCount; ss++) {
      ncclReshardTransferPlan plan;
      NCCL_M2N_CHECK(computeTransferPlanChecked(srcDims, srcStrides, srcShardLayout, ss, dstDims, dstStrides, dstShardLayout,
                                                dstInfo.shardIdx, ndims, epc_dummy, &plan));

      if (plan.totalInnerTransfers == 0) {
        continue;
      }
      if (desc->numSources >= MAX_SOURCES) {
        NCCL_M2N_FAIL(ncclInvalidArgument, worldRank,
                      "buildStagingDirectTransferDescriptor: source list truncated (%d / MAX_SOURCES=%d); increase "
                      "MAX_SOURCES",
                      desc->numSources, MAX_SOURCES);
      }

      int srcRank = getMeshRank(srcTensor, &fullSrcInfo, ss, sourceRep);

      int si = desc->numSources++;
      StagingPeerDescriptor* sd = &desc->sources[si];
      sd->peerWorldRank = srcRank;
      sd->peerShardIdx = ss;
      sd->isRdma = true;
      sd->peerLocalRank = -1;
      NCCL_M2N_CHECK(fillStagingPlan(&sd->plan, plan, ndims));

      int overlappingBeforeMe = 0;
      int totalOverlapping = 0;
      for (int ds = 0; ds < dstShardCount; ds++) {
        ncclReshardTransferPlan check;
        NCCL_M2N_CHECK(computeTransferPlanChecked(srcDims, srcStrides, srcShardLayout, ss, dstDims, dstStrides,
                                                  dstShardLayout, ds, ndims, epc_dummy, &check));
        if (check.totalInnerTransfers > 0) {
          if (ds < dstInfo.shardIdx) {
            overlappingBeforeMe++;
          }
          totalOverlapping++;
        }
      }

      desc->targetIndexOnSource[si] = overlappingBeforeMe * numTargetReps + (dstInfo.repIdx - targetRepStart);
      desc->rdmaTargetIndexOnSource[si] = desc->targetIndexOnSource[si];
      desc->channelTargetIndexOnSource[si] = desc->targetIndexOnSource[si];
      desc->sourceNumTargets[si] = totalOverlapping * numTargetReps;
      desc->sourceNumRdmaTargets[si] = desc->sourceNumTargets[si];
    }
  }

  desc->numRingTargets = 0;
  desc->peerGroupSizeBound = computeStagingPeerGroupSizeBound(desc);
  desc->ctaHeuristicPeerCount = std::max(desc->numTargets, desc->numSources);
  return ncclSuccess;
}
