/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

/*************************************************************************
 * Tensor Reshard — Copy/Staging Path
 *
 * Public C entry point `ncclReshard(handle, comm, src, dst, stream)` that
 * takes arbitrary device buffers (no symmetric-window contract) and runs
 * the staging-buffer-backed kernel.
 *
 * Implementation notes for ncclReshard:
 *   1. No NVL domain detection — gpus_per_domain derived from
 *      devComm->lsaSize, matching the window API.
 *   2. No separate devComm for the staging path — uses the main comm's
 *      devComm and caches via the existing findCachedDevComm pattern.
 *   3. Per-comm staging buffer pool (StagingBufferPoolEntry) with
 *      event-based cross-stream ordering matching the PACK staging pool.
 *
 * Algorithm selection: NCCL_RESHARD_COPY_ALGORITHM env var
 *   {default, direct, pipe}.
 ************************************************************************/

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <vector>

#include "cuda_runtime.h"
#include "nccl.h"
#include "nccl_device.h"

#include "nccl_m2n.h"
#include "m2n_checks.h"
#include "m2n_checked_math.h"
#include "reshard_call_setup.h"
#include "reshard_internal.h"
#include "m2n_log.h"
#include "reshard_split.h"
#include "reshard_types.h"
#include "staging_buffer.h"
#include "staging_profile.h"
#include "staging_types.h"

/* Assigns each reshard call a unique trace epoch. */
static std::atomic<uint64_t> gStagingEpoch{0};

#ifdef NCCL_M2N_TESTING
static std::atomic<ReshardCopyAlgorithm> gLastCompletedCopyAlgorithm{RESHARD_COPY_ALGO_DIRECT};

ReshardCopyAlgorithm reshardGetLastCompletedCopyAlgorithmForTest() {
  return gLastCompletedCopyAlgorithm.load(std::memory_order_relaxed);
}
#endif

static size_t parseEnvSize(const char* value) {
  if (value == nullptr || value[0] == '\0') {
    return 0;
  }
  char* end = nullptr;
  unsigned long long parsed = strtoull(value, &end, 0);
  if (parsed == 0) {
    return 0;
  }
  if (end != nullptr && end[0] != '\0') {
    bool kib = (end[0] == 'k' || end[0] == 'K') &&
               (end[1] == '\0' || ((end[1] == 'b' || end[1] == 'B') && end[2] == '\0') ||
                ((end[1] == 'i' || end[1] == 'I') && (end[2] == 'b' || end[2] == 'B') && end[3] == '\0'));
    bool mib = (end[0] == 'm' || end[0] == 'M') &&
               (end[1] == '\0' || ((end[1] == 'b' || end[1] == 'B') && end[2] == '\0') ||
                ((end[1] == 'i' || end[1] == 'I') && (end[2] == 'b' || end[2] == 'B') && end[3] == '\0'));
    if (kib) {
      parsed *= 1024ULL;
    } else if (mib) {
      parsed *= 1024ULL * 1024ULL;
    }
  }
  return (size_t)parsed;
}

static int getPipeTmaTileSize() {
  size_t tileSize = parseEnvSize(getenv("NCCL_RESHARD_PIPE_TMA_TILE_SIZE"));
  if (tileSize == 0) {
    return 64 * 1024;
  }
  switch (tileSize) {
  case 8 * 1024:
  case 16 * 1024:
  case 32 * 1024:
  case 64 * 1024:
    return (int)tileSize;
  default:
    RESHARD_WARN(-1, "unsupported NCCL_RESHARD_PIPE_TMA_TILE_SIZE=%zu; using 64KiB", tileSize);
    return 64 * 1024;
  }
}

static ReshardStagingTensorSignature stagingProcessTensorSignature(const ncclDistTensor_t* src,
                                                                   const ncclDistTensor_t* dst) {
  ReshardStagingTensorSignature signature{};
  signature.srcNdims = src->ndims;
  signature.srcDtype = src->dtype;
  signature.dstNdims = dst->ndims;
  signature.dstDtype = dst->dtype;
  for (int d = 0; d < NCCL_RESHARD_MAX_TENSOR_DIMS; d++) {
    signature.srcLocalShape[d] = (d < src->ndims) ? src->localShape[d] : 0;
    signature.dstLocalShape[d] = (d < dst->ndims) ? dst->localShape[d] : 0;
  }
  return signature;
}

static ReshardStagingPipeSignature stagingProcessPipeSignature(const ReshardStagingMeshSignature& meshSignature,
                                                               const ReshardStagingTensorSignature& tensorSignature) {
  ReshardStagingPipeSignature signature{};
  signature.meshSignature = meshSignature;
  signature.tensorSignature = tensorSignature;
  return signature;
}

static ReshardStagingMeshSignature stagingProcessMeshSignature(const ncclDistTensor_t* src, const ncclDistTensor_t* dst,
                                                               int srcGpusPerDomain, int dstGpusPerDomain,
                                                               bool splitStrided, int splitNumInjectionDomains,
                                                               int splitDomainsPerRep) {
  ReshardStagingMeshSignature signature{};
  signature.srcStartRank = src->mesh->startRank;
  signature.dstStartRank = dst->mesh->startRank;
  for (int d = 0; d < NCCL_RESHARD_MESH_NDIMS; d++) {
    signature.srcMeshDims[d] = src->mesh->dims[d];
    signature.srcPlacements[d] = src->placements[d];
    signature.dstMeshDims[d] = dst->mesh->dims[d];
    signature.dstPlacements[d] = dst->placements[d];
  }
  signature.srcGpusPerDomain = srcGpusPerDomain;
  signature.dstGpusPerDomain = dstGpusPerDomain;
  signature.loadBalanceMode = reshardEffectiveLbMode(src, dst);
  signature.splitStrided = splitStrided;
  signature.splitNumInjectionDomains = splitNumInjectionDomains;
  signature.splitDomainsPerRep = splitDomainsPerRep;
  return signature;
}

/* DIRECT has no persistent per-shape state in raw staging storage. Reuse a
 * buffer across shapes with the same channel capacity; its completion event
 * orders successive launches on different streams. */
static uint64_t stagingDirectReusablePoolKey(int capacityChannels) {
  return 0x73746167696e6744ULL ^ static_cast<uint64_t>(capacityChannels); /* "stagingD" */
}

static uint64_t stagingPipeReusablePoolKey() {
  return 0x73746167696e6750ULL; /* "stagingP" */
}

static int stagingComputeCtaHeuristicPeerCount(const StagingTransferDescriptor* desc, const ncclDistTensor_t* src,
                                               const ncclDistTensor_t* dst) {
  ncclReshardMeshGroupInfo srcInfo{}, dstInfo{};
  computeMeshGroupInfo(src, src->mesh->startRank, &srcInfo);
  computeMeshGroupInfo(dst, dst->mesh->startRank, &dstInfo);

  const int srcShardCount = std::max(1, srcInfo.shardCount);
  const int dstShardCount = std::max(1, dstInfo.shardCount);
  int maxActivePeers = 1;
  bool foundActive = false;
  const size_t elementsPerChunk = 1;

  for (int ss = 0; ss < srcShardCount; ss++) {
    int active = 0;
    for (int ds = 0; ds < dstShardCount; ds++) {
      ncclReshardTransferPlan plan{};
      computeTransferPlan(desc->srcDims, desc->srcStrides, srcInfo.shardLayout, ss, desc->dstDims, desc->dstStrides,
                          dstInfo.shardLayout, ds, desc->ndims, elementsPerChunk, &plan);
      if (plan.totalInnerTransfers > 0) {
        active++;
      }
    }
    if (active > 0) {
      maxActivePeers = std::max(maxActivePeers, active);
      foundActive = true;
    }
  }

  for (int ds = 0; ds < dstShardCount; ds++) {
    int active = 0;
    for (int ss = 0; ss < srcShardCount; ss++) {
      ncclReshardTransferPlan plan{};
      computeTransferPlan(desc->srcDims, desc->srcStrides, srcInfo.shardLayout, ss, desc->dstDims, desc->dstStrides,
                          dstInfo.shardLayout, ds, desc->ndims, elementsPerChunk, &plan);
      if (plan.totalInnerTransfers > 0) {
        active++;
      }
    }
    if (active > 0) {
      maxActivePeers = std::max(maxActivePeers, active);
      foundActive = true;
    }
  }

  return foundActive ? std::max(1, maxActivePeers) : 1;
}

static int stagingComputePeerGroupSizeBound(int activePeerCount, const ncclDistTensor_t* dst, int gpusPerDomain,
                                            ReshardCopyAlgorithm copyAlgo) {
  ncclReshardMeshGroupInfo dstInfo{};
  computeMeshGroupInfo(dst, dst->mesh->startRank, &dstInfo);

  int bound = std::max(1, activePeerCount);
  if (copyAlgo == RESHARD_COPY_ALGO_DIRECT) {
    bound *= std::max(1, dstInfo.repCount);
  } else {
    const int localDstReps = std::max(1, std::min(std::max(1, dstInfo.repCount), std::max(1, gpusPerDomain)));
    bound *= localDstReps;
  }
  return std::max(1, bound);
}

static int stagingComputeHierarchyPeerGroupSizeBound(int activePeerCount, const ncclDistTensor_t* dst,
                                                     int dstGpusPerDomain, int dstNodeAnchor) {
  ncclReshardMeshGroupInfo dstInfo{};
  computeMeshGroupInfo(dst, dst->mesh->startRank, &dstInfo);

  const int active = std::max(1, activePeerCount);
  const int dstShards = std::max(1, dstInfo.shardCount);
  const int dstReps = std::max(1, dstInfo.repCount);
  const int gpd = std::max(1, dstGpusPerDomain);
  auto rankNode = [&](int rank) -> int { return (rank - dstNodeAnchor) / gpd; };

  int maxLocalReps = 1;
  for (int shard = 0; shard < dstShards; shard++) {
    for (int rep = 0; rep < dstReps; rep++) {
      const int rank = getMeshRank(dst, &dstInfo, shard, rep);
      const int node = rankNode(rank);
      int localReps = 0;
      for (int other = 0; other < dstReps; other++) {
        const int otherRank = getMeshRank(dst, &dstInfo, shard, other);
        if (rankNode(otherRank) == node) {
          localReps++;
        }
      }
      maxLocalReps = std::max(maxLocalReps, localReps);
    }
  }

  const int maxAssignedSources = (active + maxLocalReps - 1) / maxLocalReps;
  const int maxHandlerTargets = maxAssignedSources * maxLocalReps;
  return std::max(active, maxHandlerTargets);
}

static int tensorRepCount(const ncclDistTensor_t* tensor) {
  ncclReshardMeshGroupInfo info{};
  computeMeshGroupInfo(tensor, tensor->mesh->startRank, &info);
  return std::max(1, info.repCount);
}

static int splitParentRankToA(const ReshardSplitComms* sc, int parentRank) {
  if (sc == nullptr) {
    return parentRank;
  }
  if (parentRank >= sc->srcStartRank && parentRank < sc->srcStartRank + sc->srcMeshSize) {
    return parentRank - sc->srcStartRank;
  }
  const int genLimit = sc->dstStartRank + sc->numInjectionDomains * sc->lsaSize;
  if (parentRank >= sc->dstStartRank && parentRank < genLimit) {
    return sc->srcMeshSize + (parentRank - sc->dstStartRank);
  }
  return -1;
}

static int splitParentRankToB(const ReshardSplitComms* sc, int parentRank) {
  if (sc == nullptr) {
    return parentRank;
  }
  if (parentRank >= sc->dstStartRank && parentRank < sc->dstStartRank + sc->dstMeshSize) {
    return parentRank - sc->dstStartRank;
  }
  return -1;
}

static bool splitParentRankIsSource(const ReshardSplitComms* sc, int parentRank) {
  return sc != nullptr && parentRank >= sc->srcStartRank && parentRank < sc->srcStartRank + sc->srcMeshSize;
}

static void offsetPipeFlowCtrl(StagingFlowCtrl* fc, int signalOffset, int counterOffset) {
  fc->localTailSignal += signalOffset;
  fc->localHeadSignal += signalOffset;
  fc->remoteTailSignal += signalOffset;
  fc->remoteHeadSignal += signalOffset;
  fc->localPutCounter += counterOffset;
}

static void applyPipeRdmaOffsets(StagingKernelParams* params, const ReshardSplitComms* splitComms, int signalOffsetA,
                                 int counterOffsetA, int signalOffsetB, int counterOffsetB) {
  const bool split = splitComms != nullptr;
  const bool thisRankIsSource = split && splitParentRankIsSource(splitComms, params->myRank);
  for (int ch = 0; ch < params->numChannels; ch++) {
    for (int t = 0; t < params->numRdmaTargets; t++) {
      StagingPeerInfo* target = &params->rdmaTargets[ch][t];
      if (!target->active) {
        continue;
      }
      const bool useCommB = split && !thisRankIsSource;
      const int signalOffset = useCommB ? signalOffsetB : signalOffsetA;
      const int counterOffset = useCommB ? counterOffsetB : counterOffsetA;
      offsetPipeFlowCtrl(&target->fc, signalOffset, counterOffset);
      offsetPipeFlowCtrl(&params->localRdmaFc[ch][t], signalOffset, counterOffset);
    }
    for (int s = 0; s < params->numRdmaSources; s++) {
      StagingPeerInfo* source = &params->rdmaSources[ch][s];
      if (source->active) {
        const bool useCommB = split && !splitParentRankIsSource(splitComms, source->peerWorldRank);
        offsetPipeFlowCtrl(&source->fc, useCommB ? signalOffsetB : signalOffsetA,
                           useCommB ? counterOffsetB : counterOffsetA);
      }
    }
  }
}

static ncclResult_t translateSplitPipePeer(StagingPeerInfo* peer, const ReshardSplitComms* sc, bool useCommB,
                                           int parentRankForLog) {
  if (peer == nullptr || !peer->active) {
    return ncclSuccess;
  }
  const int parentPeer = peer->peerWorldRank;
  if (useCommB) {
    const int rankB = splitParentRankToB(sc, parentPeer);
    if (rankB < 0) {
      NCCL_M2N_FAIL(ncclInvalidArgument, parentRankForLog, "PIPE split translate: parent peer %d is not in commB",
                    parentPeer);
    }
    peer->peerWorldRank = rankB;
    peer->fc.remoteRank = rankB;
    peer->rdmaTransport = STAGING_RDMA_TRANSPORT_SPLIT_B;
  } else {
    const int rankA = splitParentRankToA(sc, parentPeer);
    if (rankA < 0) {
      NCCL_M2N_FAIL(ncclInvalidArgument, parentRankForLog, "PIPE split translate: parent peer %d is not in commA",
                    parentPeer);
    }
    peer->peerWorldRank = rankA;
    peer->fc.remoteRank = rankA;
    peer->rdmaTransport = STAGING_RDMA_TRANSPORT_SPLIT_A;
  }
  return ncclSuccess;
}

static ncclResult_t translateSplitPipeLsaPeer(StagingPeerInfo* peer, const ReshardSplitComms* sc,
                                              int parentRankForLog) {
  if (peer == nullptr || !peer->active) {
    return ncclSuccess;
  }
  const int parentPeer = peer->peerWorldRank;
  const int rankB = splitParentRankToB(sc, parentPeer);
  if (rankB < 0) {
    NCCL_M2N_FAIL(ncclInvalidArgument, parentRankForLog, "PIPE split translate: LSA parent peer %d is not in commB",
                  parentPeer);
  }
  peer->peerWorldRank = rankB;
  peer->rdmaTransport = STAGING_RDMA_TRANSPORT_SPLIT_B;
  return ncclSuccess;
}

static ncclResult_t applySplitPipeParams(StagingKernelParams* params, const ReshardSplitComms* sc, ncclWindow_t windowA,
                                         ncclWindow_t windowB, int signalsPerControlSlot, int countersPerControlSlot,
                                         int signalsPerNamespaceB, int countersPerNamespaceB, int persistentControlSlot) {
  NCCL_M2N_CHECK_ARG(params != nullptr && sc != nullptr && sc->active, -1,
                     "applySplitPipeParams: active split state and params are required");
  params->rdmaWindow = windowA;
  params->rdmaWindowB = windowB;
  params->lsaWindow = windowB;

  const ncclWindow_t localRdmaWindow = (sc->inA && windowA != nullptr) ? windowA : windowB;
  for (int ch = 0; ch < params->numChannels; ch++) {
    params->rdmaRegions[ch].window = localRdmaWindow;
    params->lsaRegions[ch].window = windowB;
  }

  const int signalOffsetA = persistentControlSlot * signalsPerControlSlot;
  const int counterOffsetA = persistentControlSlot * countersPerControlSlot;
  NCCL_M2N_CHECK_ARG(sc->pipeNamespaceId >= 0 && sc->pipeNamespaceId < sc->pipeNamespaceCapacity,
                     params->myRank, "applySplitPipeParams: invalid persistent PIPE namespace %d/%d",
                     sc->pipeNamespaceId, sc->pipeNamespaceCapacity);
  const int signalOffsetB = sc->pipeNamespaceId * signalsPerNamespaceB + signalOffsetA;
  const int counterOffsetB = sc->pipeNamespaceId * countersPerNamespaceB + counterOffsetA;
  const bool thisRankIsSource = splitParentRankIsSource(sc, params->myRank);

  applyPipeRdmaOffsets(params, sc, signalOffsetA, counterOffsetA, signalOffsetB, counterOffsetB);

  /* stagingPrepareTransfer builds parent-rank peers; translate them after
   * assigning their graph-slot offsets. */
  for (int ch = 0; ch < params->numChannels; ch++) {
    for (int t = 0; t < params->numRdmaTargets; t++) {
      const bool useCommB = !thisRankIsSource;
      NCCL_M2N_CHECK(translateSplitPipePeer(&params->rdmaTargets[ch][t], sc, useCommB, params->myRank));
    }
    for (int s = 0; s < params->numRdmaSources; s++) {
      StagingPeerInfo* source = &params->rdmaSources[ch][s];
      const bool useCommB = source->active && !splitParentRankIsSource(sc, source->peerWorldRank);
      NCCL_M2N_CHECK(translateSplitPipePeer(source, sc, useCommB, params->myRank));
    }
    for (int t = 0; t < params->numLsaTargets; t++) {
      NCCL_M2N_CHECK(translateSplitPipeLsaPeer(&params->lsaTargets[ch][t], sc, params->myRank));
    }
    for (int s = 0; s < params->numLsaSources; s++) {
      NCCL_M2N_CHECK(translateSplitPipeLsaPeer(&params->lsaSources[ch][s], sc, params->myRank));
    }
  }
  return ncclSuccess;
}

static void stagingStripPipeTransferPlans(StagingKernelParams* params) {
  for (int ch = 0; ch < STAGING_MAX_CHANNELS; ch++) {
    for (int t = 0; t < MAX_TARGETS; t++) {
      memset(&params->rdmaTargets[ch][t].plan, 0, sizeof(params->rdmaTargets[ch][t].plan));
      memset(&params->lsaTargets[ch][t].plan, 0, sizeof(params->lsaTargets[ch][t].plan));
    }
    for (int s = 0; s < MAX_SOURCES; s++) {
      memset(&params->rdmaSources[ch][s].plan, 0, sizeof(params->rdmaSources[ch][s].plan));
      memset(&params->lsaSources[ch][s].plan, 0, sizeof(params->lsaSources[ch][s].plan));
    }
  }
}

static void stagingNormalizePipeStaticPlan(StagingKernelParams* params) {
  /* After stagingBuildPipeDevicePlan extracts edge/layout metadata, keep
   * only immutable launch state in cached StagingKernelParams.  Per-call
   * buffers, debug epoch, and full copy plans are supplied separately or via
   * the compact StagingPipeDevicePlan. */
  params->srcBuffer = nullptr;
  params->dstBuffer = nullptr;
  params->stagingBuffer = nullptr;
  params->ginSignalCount = 0;
  params->ginCounterCount = 0;
  params->epoch = 0;

  memset(params->srcDims, 0, sizeof(params->srcDims));
  memset(params->dstDims, 0, sizeof(params->dstDims));
  memset(params->srcStrides, 0, sizeof(params->srcStrides));
  memset(params->dstStrides, 0, sizeof(params->dstStrides));
  params->ndims = 0;

  stagingStripPipeTransferPlans(params);
}

static void stagingMakePipeCopyLayout(const StagingTransferPlan* plan, bool pack, StagingPipeCopyLayout* layout) {
  memset(layout, 0, sizeof(*layout));
  if (plan == nullptr) {
    return;
  }

  layout->numOuterLoops = plan->numOuterLoops;
  layout->baseOffset = pack ? plan->srcBaseOffset : plan->dstBaseOffset;
  layout->innerSize = plan->innerSize;
  layout->isContiguous = plan->isContiguous;
  for (int d = 0; d < NCCL_RESHARD_MAX_TENSOR_DIMS; d++) {
    layout->outerCounts[d] = plan->outerCounts[d];
    layout->outerStrides[d] = pack ? plan->outerSrcStrides[d] : plan->outerDstStrides[d];
  }
}

static void stagingMakePipePeerEdge(const StagingPeerInfo* peer, int copyLayoutIndex, StagingPipePeerEdge* edge) {
  memset(edge, 0, sizeof(*edge));
  if (peer == nullptr) {
    edge->peerWorldRank = -1;
    edge->peerLocalRank = -1;
    return;
  }

  edge->peerWorldRank = peer->peerWorldRank;
  edge->peerLocalRank = peer->peerLocalRank;
  edge->rdmaTransport = peer->rdmaTransport;
  edge->copyLayoutIndex = copyLayoutIndex;
  edge->active = peer->active && peer->channelCount > 0;
  edge->totalBytes = peer->totalBytes;
  edge->logicalChunkSize = peer->logicalChunkSize;
  edge->channelRank = peer->channelRank;
  edge->channelCount = peer->channelCount;
  edge->chunkStart = peer->chunkStart;
  edge->chunkEnd = peer->chunkEnd;
  edge->fc = peer->fc;
}

static void stagingBuildPipeDevicePlan(const StagingKernelParams* params, StagingPipeDevicePlan* plan) {
  memset(plan, 0, sizeof(*plan));
  for (int ch = 0; ch < STAGING_MAX_CHANNELS; ch++) {
    bool hasRdmaTarget = false;
    bool hasRdmaSource = false;
    bool hasLsaSource = false;
    for (int t = 0; t < MAX_TARGETS; t++) {
      const StagingPeerInfo* target = &params->rdmaTargets[ch][t];
      stagingMakePipePeerEdge(target, t, &plan->rdmaTargets[ch][t].peer);
      hasRdmaTarget = hasRdmaTarget || plan->rdmaTargets[ch][t].peer.active;
      plan->rdmaTargets[ch][t].localFc = params->localRdmaFc[ch][t];
      stagingMakePipeCopyLayout(&target->plan, true, &plan->rdmaTargetLayouts[ch][t]);

      stagingMakePipePeerEdge(&params->lsaTargets[ch][t], t, &plan->lsaTargets[ch][t]);
    }
    for (int s = 0; s < MAX_SOURCES; s++) {
      const StagingPeerInfo* rdmaSource = &params->rdmaSources[ch][s];
      stagingMakePipePeerEdge(rdmaSource, s, &plan->rdmaSources[ch][s]);
      hasRdmaSource = hasRdmaSource || plan->rdmaSources[ch][s].active;
      stagingMakePipeCopyLayout(&rdmaSource->plan, false, &plan->rdmaSourceLayouts[ch][s]);

      const StagingPeerInfo* lsaSource = &params->lsaSources[ch][s];
      stagingMakePipePeerEdge(lsaSource, s, &plan->lsaSources[ch][s]);
      hasLsaSource = hasLsaSource || plan->lsaSources[ch][s].active;
      stagingMakePipeCopyLayout(&lsaSource->plan, false, &plan->lsaSourceLayouts[ch][s]);
    }
    if (hasRdmaTarget && plan->numTrainerRdmaLaunchChannels < STAGING_MAX_CHANNELS) {
      plan->trainerRdmaLaunchChannels[plan->numTrainerRdmaLaunchChannels++] = ch;
    }
    if (hasRdmaSource && plan->numGeneratorRdmaLaunchChannels < STAGING_MAX_CHANNELS) {
      plan->generatorRdmaLaunchChannels[plan->numGeneratorRdmaLaunchChannels++] = ch;
    }
    if (hasLsaSource && plan->numGeneratorLsaLaunchChannels < STAGING_MAX_CHANNELS) {
      plan->generatorLsaLaunchChannels[plan->numGeneratorLsaLaunchChannels++] = ch;
    }
  }
}

static ncclResult_t stagingEnsurePipeHostPlanEntryStorage(StagingPipePlanCacheEntry* entry, int rank) {
  NCCL_M2N_CHECK_ARG(entry != nullptr, rank, "[STAGING] PIPE plan cache requested with null entry");
  if (entry->hostParams == nullptr) {
    entry->hostParams = new (std::nothrow) StagingKernelParams;
  }
  if (entry->hostPlan == nullptr) {
    entry->hostPlan = new (std::nothrow) StagingPipeDevicePlan;
  }
  if (entry->hostParams == nullptr || entry->hostPlan == nullptr) {
    NCCL_M2N_FAIL(ncclSystemError, rank, "[STAGING] failed to allocate cached PIPE host plan");
  }
  return ncclSuccess;
}

static bool stagingPlanMatches(const StagingPipePlanCacheEntry& entry, const ReshardStagingPipeSignature& signature) {
  return entry.hostValid && entry.signature == signature;
}

static void stagingInvalidatePipePlanEntry(StagingPipePlanCacheEntry* entry,
                                           const ReshardStagingPipeSignature& signature) {
  entry->signature = signature;
  entry->hostValid = false;
  entry->deviceValid = false;
}

static ncclResult_t stagingGetPipePlanEntry(StagingBufferState* staging,
                                            const ReshardStagingPipeSignature& signature, int preferredSlot, int rank,
                                            StagingPipePlanCacheEntry** outEntry, bool* cacheMiss) {
  NCCL_M2N_CHECK_ARG(staging != nullptr && outEntry != nullptr && cacheMiss != nullptr, rank,
                     "[STAGING] PIPE plan cache lookup requires staging, entry, and miss outputs");
  for (int i = 0; i < STAGING_PIPE_CONTROL_SLOTS; i++) {
    StagingPipePlanCacheEntry& entry = staging->pipePlanCache[i];
    if (stagingPlanMatches(entry, signature)) {
      *outEntry = &entry;
      *cacheMiss = false;
      return ncclSuccess;
    }
  }

  int slot = -1;
  if (preferredSlot >= 0 && preferredSlot < STAGING_PIPE_CONTROL_SLOTS &&
      !staging->pipePlanCache[preferredSlot].hostValid) {
    slot = preferredSlot;
  }
  if (slot < 0) {
    for (int i = 0; i < STAGING_PIPE_CONTROL_SLOTS; i++) {
      if (!staging->pipePlanCache[i].hostValid) {
        slot = i;
        break;
      }
    }
  }
  if (slot < 0) {
    slot = staging->pipePlanCacheNextVictim;
    staging->pipePlanCacheNextVictim = (staging->pipePlanCacheNextVictim + 1) % STAGING_PIPE_CONTROL_SLOTS;
  }

  StagingPipePlanCacheEntry& entry = staging->pipePlanCache[slot];
  NCCL_M2N_CHECK(stagingEnsurePipeHostPlanEntryStorage(&entry, rank));
  stagingInvalidatePipePlanEntry(&entry, signature);
  *outEntry = &entry;
  *cacheMiss = true;
  return ncclSuccess;
}

static ncclResult_t stagingFinalizePipeHostPlan(StagingPipePlanCacheEntry* entry,
                                                const ReshardStagingPipeSignature& signature, int rank) {
  NCCL_M2N_CHECK_ARG(entry != nullptr && entry->hostParams != nullptr && entry->hostPlan != nullptr, rank,
                     "[STAGING] cannot finalize missing PIPE host plan");

  /* Build the compact device plan before stripping bulky transfer plans from
   * hostParams; this keeps warm launches to two cached device pointers plus a
   * small by-value StagingPipeCallParams. */
  stagingBuildPipeDevicePlan(entry->hostParams, entry->hostPlan);
  stagingNormalizePipeStaticPlan(entry->hostParams);
  entry->signature = signature;
  entry->hostValid = true;
  entry->deviceValid = false;
  return ncclSuccess;
}

static ncclResult_t stagingEnsurePipeDevicePlan(StagingPipePlanCacheEntry* entry,
                                                const ReshardStagingPipeSignature& signature, cudaStream_t stream,
                                                int rank) {
  NCCL_M2N_CHECK_ARG(entry != nullptr && entry->hostParams != nullptr && entry->hostPlan != nullptr &&
                       entry->hostValid && entry->signature == signature,
                     rank, "[STAGING] cannot upload missing PIPE device plan");
  if (entry->deviceValid && entry->signature == signature) {
    return ncclSuccess;
  }

  if (entry->devParams == nullptr) {
    NCCL_M2N_CUDACHECK(cudaMalloc(&entry->devParams, sizeof(StagingKernelParams)));
  }
  if (entry->devPlan == nullptr) {
    NCCL_M2N_CUDACHECK(cudaMalloc(&entry->devPlan, sizeof(StagingPipeDevicePlan)));
  }
  /* This upload happens only on a shape-cache miss.  Warm calls pass mutable
   * src/dst pointers and toggles as kernel arguments and do not update this
   * cached device state. */
  NCCL_M2N_CUDACHECK(cudaMemcpyAsync(entry->devParams, entry->hostParams, sizeof(*entry->hostParams),
                                     cudaMemcpyHostToDevice, stream));
  NCCL_M2N_CUDACHECK(cudaMemcpyAsync(entry->devPlan, entry->hostPlan, sizeof(*entry->hostPlan), cudaMemcpyHostToDevice,
                                     stream));
  entry->deviceValid = true;
  return ncclSuccess;
}

static ncclResult_t stagingEnsureDirectDevParams(StagingBufferState* staging, int rank) {
  NCCL_M2N_CHECK_ARG(staging != nullptr, rank, "[STAGING] DIRECT requested with null staging buffer");
  if (staging->devParams == nullptr) {
    NCCL_M2N_CUDACHECK(cudaMalloc(&staging->devParams, sizeof(StagingKernelParams)));
  }
  return ncclSuccess;
}

static bool pipeHostEdgeActive(const StagingPipePeerEdge& edge) {
  return edge.active && edge.totalBytes > 0 && edge.channelRank >= 0 && edge.channelRank < edge.channelCount;
}

/* Host-RMA execution uses the same compact PIPE plan as the device kernels.
 * The helpers below translate plan edges into host NCCL one-sided operations
 * and CUDA CE copies; CUDA events provide intra-rank ordering while host RMA
 * wait/signal op counts provide inter-rank ordering. */
static thread_local int gPipeHostNcclGroupDepth = 0;

/* Edge transport tags are already assigned by the shared staging descriptor.
 * Host-RMA maps those tags to the communicator/rank namespace used by host
 * ncclPutSignal/ncclWaitSignal.  LSA pointer lookup still uses peerLocalRank
 * from the edge, not these communicator ranks. */
enum PipeHostCommId {
  PIPE_HOST_COMM_PARENT,
  PIPE_HOST_COMM_SPLIT_A,
  PIPE_HOST_COMM_SPLIT_B,
  PIPE_HOST_COMM_COUNT
};

static constexpr std::array<PipeHostCommId, PIPE_HOST_COMM_COUNT> kPipeHostCommIds = {
  PIPE_HOST_COMM_PARENT,
  PIPE_HOST_COMM_SPLIT_A,
  PIPE_HOST_COMM_SPLIT_B,
};

struct PipeHostComm {
  ncclComm_t comm;
  int rank;
  const char* label;
};

struct PipeHostRmaContext {
  std::array<PipeHostComm, PIPE_HOST_COMM_COUNT> comms;
  bool split;
};

static bool pipeHostCommActive(const PipeHostComm& comm) {
  return comm.comm != nullptr && comm.rank >= 0;
}

static PipeHostCommId pipeHostCommIdForTransport(int rdmaTransport) {
  switch (rdmaTransport) {
  case STAGING_RDMA_TRANSPORT_SPLIT_A:
    return PIPE_HOST_COMM_SPLIT_A;
  case STAGING_RDMA_TRANSPORT_SPLIT_B:
    return PIPE_HOST_COMM_SPLIT_B;
  case STAGING_RDMA_TRANSPORT_PARENT:
  default:
    return PIPE_HOST_COMM_PARENT;
  }
}

static const PipeHostComm* pipeHostCommForId(const PipeHostRmaContext* ctx, PipeHostCommId commId, int rank) {
  if (ctx == nullptr || commId < 0 || commId >= PIPE_HOST_COMM_COUNT) {
    return nullptr;
  }
  const PipeHostComm* comm = &ctx->comms[commId];
  if (!pipeHostCommActive(*comm)) {
    RESHARD_TRACE(rank, "PIPE host_rma: inactive comm id=%d label=%s", static_cast<int>(commId),
                  comm->label != nullptr ? comm->label : "unknown");
    return nullptr;
  }
  return comm;
}

static ncclResult_t pipeHostBuildRmaContext(ncclComm_t parentComm, const ReshardSplitComms* splitComms, int parentRank,
                                            PipeHostRmaContext* out) {
  NCCL_M2N_CHECK_ARG(out != nullptr, parentRank, "PIPE host_rma: null RMA context output");
  out->comms = {};
  out->split = splitComms != nullptr && splitComms->active;
  out->comms[PIPE_HOST_COMM_PARENT] = {out->split ? nullptr : parentComm, out->split ? -1 : parentRank, "parent"};
  out->comms[PIPE_HOST_COMM_SPLIT_A] = {nullptr, -1, "splitA"};
  out->comms[PIPE_HOST_COMM_SPLIT_B] = {nullptr, -1, "splitB"};

  if (!out->split) {
    return ncclSuccess;
  }

  NCCL_M2N_CHECK_ARG(splitComms->valid, parentRank, "PIPE host_rma: invalid split communicator state");
  if (splitComms->inA) {
    out->comms[PIPE_HOST_COMM_SPLIT_A] = {splitComms->commA, splitComms->rankInA, "splitA"};
  }
  if (splitComms->inB) {
    out->comms[PIPE_HOST_COMM_SPLIT_B] = {splitComms->commB, splitComms->rankInB, "splitB"};
  }
  return ncclSuccess;
}

static ncclResult_t pipeHostRdmaWindow(const StagingKernelParams* params, const StagingPipePeerEdge& edge, int rank,
                                       ncclWindow_t* outWindow) {
  NCCL_M2N_CHECK_ARG(params != nullptr && outWindow != nullptr, rank, "PIPE host_rma: invalid RDMA window arguments");
  *outWindow = (edge.rdmaTransport == STAGING_RDMA_TRANSPORT_SPLIT_B) ? params->rdmaWindowB : params->rdmaWindow;
  NCCL_M2N_CHECK_ARG(*outWindow != nullptr, rank, "PIPE host_rma: RDMA window is null for peer %d", edge.peerWorldRank);
  return ncclSuccess;
}

static bool pipeHostTraceSyncEnabled() {
  return reshardGetLogLevel() >= RESHARD_LOG_TRACE && gPipeHostNcclGroupDepth == 0;
}

static ncclResult_t pipeHostTraceSync(cudaStream_t stream, int rank, const char* tag, const char* op, int peer) {
  if (!pipeHostTraceSyncEnabled()) {
    return ncclSuccess;
  }
  RESHARD_TRACE(rank, "PIPE host_rma %s %s sync begin peer=%d", tag, op, peer);
  NCCL_M2N_CUDACHECK(cudaStreamSynchronize(stream));
  RESHARD_TRACE(rank, "PIPE host_rma %s %s sync end peer=%d", tag, op, peer);
  return ncclSuccess;
}

static ncclResult_t pipeHostSignal(ncclComm_t comm, int peer, cudaStream_t stream, int rank, const char* tag) {
  RESHARD_DEBUG(rank, "PIPE host_rma %s signal enqueue begin peer=%d sig=0 ctx=0", tag, peer);
  NCCL_M2N_CHECK(ncclSignal(peer, 0, 0, 0, comm, stream));
  RESHARD_DEBUG(rank, "PIPE host_rma %s signal enqueue end peer=%d sig=0 ctx=0", tag, peer);
  NCCL_M2N_CHECK(pipeHostTraceSync(stream, rank, tag, "signal", peer));
  return ncclSuccess;
}

/* One host wait phase may need to wait on several peers.  The wait list folds
 * repeated waits for the same peer/signal/context into a single descriptor by
 * increasing opCnt, which mirrors the persistent-counter credit semantics
 * without keeping GPU counter state for Host-RMA. */
struct PipeHostWaitList {
  static constexpr size_t kInlineCapacity = 64;

  bool empty() const {
    return size() == 0;
  }

  size_t size() const {
    return overflowDescs.empty() ? inlineSize : overflowDescs.size();
  }

  ncclWaitSignalDesc_t* data() {
    return overflowDescs.empty() ? inlineDescs.data() : overflowDescs.data();
  }

  ncclWaitSignalDesc_t& at(size_t index) {
    return overflowDescs.empty() ? inlineDescs[index] : overflowDescs[index];
  }

  ncclResult_t push(const ncclWaitSignalDesc_t& desc, int rank) {
    if (!overflowDescs.empty()) {
      overflowDescs.push_back(desc);
      return ncclSuccess;
    }
    if (inlineSize < inlineDescs.size()) {
      inlineDescs[inlineSize++] = desc;
      return ncclSuccess;
    }

    overflowDescs.reserve(inlineDescs.size() * 2);
    overflowDescs.insert(overflowDescs.end(), inlineDescs.begin(), inlineDescs.end());
    overflowDescs.push_back(desc);
    NCCL_M2N_CHECK_ARG(overflowDescs.size() > inlineDescs.size(), rank,
                       "PIPE host_rma: wait desc overflow promotion failed");
    return ncclSuccess;
  }

private:
  size_t inlineSize = 0;
  std::array<ncclWaitSignalDesc_t, kInlineCapacity> inlineDescs{};
  std::vector<ncclWaitSignalDesc_t> overflowDescs;
};

static ncclResult_t pipeHostWaitListAdd(PipeHostWaitList* waits, int peer, int sigIdx, int ctx, int opCnt, int rank) {
  NCCL_M2N_CHECK_ARG(waits != nullptr, rank, "PIPE host_rma: wait list is null");
  NCCL_M2N_CHECK_ARG(peer >= 0 && sigIdx >= 0 && ctx >= 0 && opCnt > 0, rank,
                     "PIPE host_rma: invalid wait desc peer=%d sig=%d ctx=%d opCnt=%d", peer, sigIdx, ctx, opCnt);
  for (size_t i = 0; i < waits->size(); i++) {
    ncclWaitSignalDesc_t& desc = waits->at(i);
    if (desc.peer == peer && desc.sigIdx == sigIdx && desc.ctx == ctx) {
      NCCL_M2N_CHECK_ARG(desc.opCnt <= std::numeric_limits<int>::max() - opCnt, rank,
                         "PIPE host_rma: wait opCnt overflow for peer=%d sig=%d ctx=%d", peer, sigIdx, ctx);
      desc.opCnt += opCnt;
      return ncclSuccess;
    }
  }
  ncclWaitSignalDesc_t desc{};
  desc.opCnt = opCnt;
  desc.peer = peer;
  desc.sigIdx = sigIdx;
  desc.ctx = ctx;
  NCCL_M2N_CHECK(waits->push(desc, rank));
  return ncclSuccess;
}

static ncclResult_t pipeHostWaitSignals(ncclComm_t comm, PipeHostWaitList* waits, cudaStream_t stream, int rank,
                                        const char* tag) {
  NCCL_M2N_CHECK_ARG(waits != nullptr && !waits->empty(), rank, "PIPE host_rma: wait phase %s has no descriptors", tag);
  NCCL_M2N_CHECK_ARG(waits->size() <= static_cast<size_t>(std::numeric_limits<int>::max()), rank,
                     "PIPE host_rma: wait desc count overflow");
  const int nDesc = static_cast<int>(waits->size());
  RESHARD_DEBUG(rank, "PIPE host_rma %s wait enqueue begin ndesc=%d", tag, nDesc);
  NCCL_M2N_CHECK(ncclWaitSignal(nDesc, waits->data(), comm, stream));
  NCCL_M2N_CHECK(m2nWaitCommReady(comm));
  RESHARD_DEBUG(rank, "PIPE host_rma %s wait enqueue end ndesc=%d", tag, nDesc);
  NCCL_M2N_CHECK(pipeHostTraceSync(stream, rank, tag, "wait", -1));
  return ncclSuccess;
}

static ncclResult_t pipeHostPutSignal(ncclComm_t comm, const void* localSrc, size_t bytes, int peer,
                                      ncclWindow_t peerWindow, size_t peerWindowOffset, cudaStream_t stream, int rank,
                                      const char* tag) {
  if (bytes == 0) {
    return pipeHostSignal(comm, peer, stream, rank, tag);
  }
  RESHARD_DEBUG(rank,
                "PIPE host_rma %s put_signal enqueue begin peer=%d bytes=%zu remote_offset=%zu sig=0 ctx=0 "
                "src=%p",
                tag, peer, bytes, peerWindowOffset, localSrc);
  NCCL_M2N_CHECK(ncclPutSignal(localSrc, bytes, ncclInt8, peer, peerWindow, peerWindowOffset, 0, 0, 0, comm, stream));
  RESHARD_DEBUG(rank, "PIPE host_rma %s put_signal enqueue end peer=%d bytes=%zu remote_offset=%zu sig=0 ctx=0 src=%p",
                tag, peer, bytes, peerWindowOffset, localSrc);
  NCCL_M2N_CHECK(pipeHostTraceSync(stream, rank, tag, "put_signal", peer));
  return ncclSuccess;
}

static ncclResult_t pipeHostRmaPhaseBegin(int rank, const char* phase) {
  RESHARD_DEBUG(rank, "PIPE host_rma phase %s group start begin", phase);
  NCCL_M2N_CHECK(ncclGroupStart());
  gPipeHostNcclGroupDepth++;
  RESHARD_DEBUG(rank, "PIPE host_rma phase %s group start end", phase);
  return ncclSuccess;
}

static ncclResult_t pipeHostRmaPhaseEnd(ncclComm_t comm, cudaStream_t stream, int rank, const char* phase,
                                        ncclResult_t scheduleResult) {
  NCCL_M2N_CHECK_ARG(gPipeHostNcclGroupDepth > 0, rank, "PIPE host_rma: phase %s ended without a matching group start",
                     phase);
  gPipeHostNcclGroupDepth--;
  RESHARD_DEBUG(rank, "PIPE host_rma phase %s group end begin schedule_result=%d", phase,
                static_cast<int>(scheduleResult));
  ncclResult_t groupEndResult = ncclGroupEnd();
  RESHARD_DEBUG(rank, "PIPE host_rma phase %s group end result=%d", phase, static_cast<int>(groupEndResult));
  if (scheduleResult != ncclSuccess) {
    return scheduleResult;
  }
  NCCL_M2N_CHECK(groupEndResult);
  /* Group completion can leave enqueue work pending even after a success return.
   * Finish it before the pipeline records events or submits a dependent phase. */
  NCCL_M2N_CHECK(m2nWaitCommReady(comm));
  NCCL_M2N_CHECK(pipeHostTraceSync(stream, rank, phase, "phase_end", -1));
  return ncclSuccess;
}

template <typename EnqueueFn>
static ncclResult_t pipeHostRunPutSignalPhase(const PipeHostComm& hostComm, int logRank, cudaStream_t stream,
                                              const char* phase, EnqueueFn enqueueFn) {
  if (!pipeHostCommActive(hostComm)) {
    return ncclSuccess;
  }
  NCCL_M2N_CHECK(pipeHostRmaPhaseBegin(logRank, phase));
  ncclResult_t scheduleResult = enqueueFn();
  return pipeHostRmaPhaseEnd(hostComm.comm, stream, logRank, phase, scheduleResult);
}

template <typename BuildWaitFn>
static ncclResult_t pipeHostRunWaitPhase(const PipeHostComm& hostComm, int logRank, cudaStream_t stream,
                                         const char* phase, BuildWaitFn buildWaits) {
  if (!pipeHostCommActive(hostComm)) {
    return ncclSuccess;
  }
  PipeHostWaitList waits;
  NCCL_M2N_CHECK(buildWaits(&waits));
  if (waits.empty()) {
    return ncclSuccess;
  }

  return pipeHostWaitSignals(hostComm.comm, &waits, stream, logRank, phase);
}

/* Operation-family streams mirror the device PIPE stages, but only active
 * rank roles allocate streams.  Single-round transfers collapse every stream
 * id onto the caller-ordered work stream to avoid event overhead. */
enum PipeHostStreamId {
  PIPE_HOST_STREAM_PACK,
  PIPE_HOST_STREAM_SOURCE_PUT,
  PIPE_HOST_STREAM_ROOT_COPY,
  PIPE_HOST_STREAM_RING_PUT,
  PIPE_HOST_STREAM_ROOT_SIGNAL,
  PIPE_HOST_STREAM_FOLLOWER_WAIT,
  PIPE_HOST_STREAM_FOLLOWER_UNPACK,
  PIPE_HOST_STREAM_FOLLOWER_SIGNAL,
  PIPE_HOST_STREAM_ROOT_WAIT,
  PIPE_HOST_STREAM_CREDIT_SIGNAL,
  PIPE_HOST_STREAM_CREDIT_WAIT,
  PIPE_HOST_STREAM_COUNT
};

enum PipeHostEventId {
  PIPE_HOST_EVENT_DEST_EDGE_READY,
  PIPE_HOST_EVENT_TARGET_PACK_DONE,
  PIPE_HOST_EVENT_TARGET_PUT_DONE,
  PIPE_HOST_EVENT_TARGET_CREDIT_DONE,
  PIPE_HOST_EVENT_ROOT_COPY_DONE,
  PIPE_HOST_EVENT_RING_PUT_DONE,
  PIPE_HOST_EVENT_ROOT_SIGNAL_DONE,
  PIPE_HOST_EVENT_ROOT_FOLLOWERS_DONE,
  PIPE_HOST_EVENT_FOLLOWER_READY,
  PIPE_HOST_EVENT_FOLLOWER_COPY_DONE,
  PIPE_HOST_EVENT_COUNT
};

static constexpr uint64_t pipeHostMaskBit(int id) {
  return 1ULL << static_cast<unsigned int>(id);
}

static bool pipeHostMaskContains(uint64_t mask, int id) {
  return (mask & pipeHostMaskBit(id)) != 0;
}

static ncclResult_t pipeHostRecordEvent(cudaEvent_t event, cudaStream_t stream, int rank, const char* tag) {
  NCCL_M2N_CHECK_ARG(event != nullptr, rank, "PIPE host_rma: null event for %s", tag);
  NCCL_M2N_CUDACHECK(cudaEventRecord(event, stream));
  return ncclSuccess;
}

static ncclResult_t pipeHostWaitEvent(cudaStream_t stream, cudaEvent_t event, int rank, const char* tag) {
  NCCL_M2N_CHECK_ARG(event != nullptr, rank, "PIPE host_rma: null dependency event for %s", tag);
  NCCL_M2N_CUDACHECK(cudaStreamWaitEvent(stream, event, 0));
  return ncclSuccess;
}

/* Optional CUDA stream/event graph for the host scheduler. In pipeline mode,
 * each stage records an event for its edge in each round, so a ready chunk
 * never waits for unrelated peer work. In single-stream mode, record/wait are
 * no-ops and stream order carries dependencies. */
class PipeHostPipeline {
public:
  PipeHostPipeline() = default;
  PipeHostPipeline(const PipeHostPipeline&) = delete;
  PipeHostPipeline& operator=(const PipeHostPipeline&) = delete;

  ~PipeHostPipeline() {
    destroy();
  }

  ncclResult_t ensure(size_t rounds, int numChannels, int numRdmaTargets, int numDestSources, int numLsaSources,
                      uint64_t streamMask, uint64_t eventMask, bool enablePipeline, bool channelSourcePipeline,
                      bool destWaitPipeline) {
    rounds_ = rounds;
    streamMask_ = streamMask;
    eventMask_ = eventMask;
    pipelineEnabled_ = enablePipeline;
    numChannels_ = numChannels;
    numRdmaTargets_ = numRdmaTargets;
    numDestSources_ = numDestSources;
    numLsaSources_ = numLsaSources;
    channelSourcePipeline_ = channelSourcePipeline;
    destWaitPipeline_ = destWaitPipeline;

    if (!pipelineEnabled_) {
      return ncclSuccess;
    }

    for (size_t i = 0; i < streams_.size(); i++) {
      if (pipeHostMaskContains(streamMask_, static_cast<int>(i))) {
        if (streams_[i] == nullptr) {
          NCCL_M2N_CUDACHECK(cudaStreamCreateWithFlags(&streams_[i], cudaStreamNonBlocking));
        }
        if (joinEvents_[i] == nullptr) {
          NCCL_M2N_CUDACHECK(cudaEventCreateWithFlags(&joinEvents_[i], cudaEventDisableTiming));
        }
      }
    }
    if (startEvent_ == nullptr) {
      NCCL_M2N_CUDACHECK(cudaEventCreateWithFlags(&startEvent_, cudaEventDisableTiming));
    }

    for (size_t i = 0; i < events_.size(); i++) {
      if (pipeHostMaskContains(eventMask_, static_cast<int>(i))) {
        size_t eventCount = 0;
        NCCL_M2N_CHECK(eventCountFor(static_cast<PipeHostEventId>(i), rounds_, &eventCount));
        NCCL_M2N_CHECK(ensureEventVector(&events_[i], eventCount));
      }
    }

    if (channelSourcePipeline_) {
      channelPackStreams_.resize(std::max(channelPackStreams_.size(), static_cast<size_t>(numChannels_)), nullptr);
      channelPackJoinEvents_.resize(std::max(channelPackJoinEvents_.size(), static_cast<size_t>(numChannels_)),
                                    nullptr);
      for (int ch = 0; ch < numChannels_; ch++) {
        if (channelPackStreams_[ch] == nullptr) {
          NCCL_M2N_CUDACHECK(cudaStreamCreateWithFlags(&channelPackStreams_[ch], cudaStreamNonBlocking));
        }
        if (channelPackJoinEvents_[ch] == nullptr) {
          NCCL_M2N_CUDACHECK(cudaEventCreateWithFlags(&channelPackJoinEvents_[ch], cudaEventDisableTiming));
        }
      }
    }
    return ncclSuccess;
  }

  ncclResult_t begin(cudaStream_t joinStream) {
    joinStream_ = joinStream;
    if (!pipelineEnabled_) {
      return ncclSuccess;
    }
    NCCL_M2N_CUDACHECK(cudaEventRecord(startEvent_, joinStream_));
    for (size_t i = 0; i < streams_.size(); i++) {
      if (pipeHostMaskContains(streamMask_, static_cast<int>(i)) && streams_[i] != nullptr) {
        NCCL_M2N_CUDACHECK(cudaStreamWaitEvent(streams_[i], startEvent_, 0));
      }
    }
    if (channelSourcePipeline_) {
      for (int ch = 0; ch < numChannels_; ch++) {
        NCCL_M2N_CUDACHECK(cudaStreamWaitEvent(channelPackStreams_[ch], startEvent_, 0));
      }
    }
    beginGeneration_++;
    return ncclSuccess;
  }

  ncclResult_t join() {
    if (!pipelineEnabled_) {
      return ncclSuccess;
    }

    for (size_t i = 0; i < streams_.size(); i++) {
      if (!pipeHostMaskContains(streamMask_, static_cast<int>(i)) || streams_[i] == nullptr) {
        continue;
      }
      NCCL_M2N_CHECK(joinOneStream(streams_[i], joinEvents_[i]));
    }
    if (channelSourcePipeline_) {
      for (int ch = 0; ch < numChannels_; ch++) {
        NCCL_M2N_CHECK(joinOneStream(channelPackStreams_[ch], channelPackJoinEvents_[ch]));
      }
    }
    for (const SourcePutStream& sourcePutStream : sourcePutStreams_) {
      if (sourcePutStream.beginGeneration == beginGeneration_) {
        NCCL_M2N_CHECK(joinOneStream(sourcePutStream.stream, sourcePutStream.joinEvent));
      }
    }
    for (const DestWaitStream& destWaitStream : destWaitStreams_) {
      if (destWaitStream.beginGeneration == beginGeneration_) {
        NCCL_M2N_CHECK(joinOneStream(destWaitStream.stream, destWaitStream.joinEvent));
      }
    }
    return ncclSuccess;
  }

  cudaStream_t stream(PipeHostStreamId id) const {
    if (!pipelineEnabled_) {
      return joinStream_;
    }
    return streams_[id];
  }

  cudaStream_t channelStream(PipeHostStreamId id, int channel) const {
    if (!pipelineEnabled_ || !channelSourcePipeline_) {
      return stream(id);
    }
    if (id == PIPE_HOST_STREAM_PACK) {
      return channelPackStreams_[channel];
    }
    return stream(id);
  }

  ncclResult_t sourcePutStream(ncclComm_t comm, int peer, int rank, cudaStream_t* out) {
    NCCL_M2N_CHECK_ARG(out != nullptr, rank, "PIPE host_rma: null source-put stream output");
    if (!pipelineEnabled_ || !channelSourcePipeline_) {
      *out = stream(PIPE_HOST_STREAM_SOURCE_PUT);
      return ncclSuccess;
    }
    // sigIdx=0 is one FIFO per (communicator, peer), even when more than one
    // channel maps to that peer.
    SourcePutStream* entry = nullptr;
    for (SourcePutStream& candidate : sourcePutStreams_) {
      if (candidate.comm == comm && candidate.peer == peer) {
        entry = &candidate;
        break;
      }
    }
    if (entry == nullptr) {
      sourcePutStreams_.push_back({comm, peer, nullptr, nullptr, 0});
      entry = &sourcePutStreams_.back();
    }
    if (entry->stream == nullptr) {
      NCCL_M2N_CUDACHECK(cudaStreamCreateWithFlags(&entry->stream, cudaStreamNonBlocking));
    }
    if (entry->joinEvent == nullptr) {
      NCCL_M2N_CUDACHECK(cudaEventCreateWithFlags(&entry->joinEvent, cudaEventDisableTiming));
    }
    if (entry->beginGeneration != beginGeneration_) {
      NCCL_M2N_CHECK(pipeHostWaitEvent(entry->stream, startEvent_, rank, "source_put.start"));
      entry->beginGeneration = beginGeneration_;
    }
    *out = entry->stream;
    return ncclSuccess;
  }

  ncclResult_t destWaitStream(int source, int rank, cudaStream_t* out) {
    NCCL_M2N_CHECK_ARG(out != nullptr, rank, "PIPE host_rma: null destination-wait stream output");
    if (!pipelineEnabled_ || !destWaitPipeline_) {
      *out = joinStream_;
      return ncclSuccess;
    }
    DestWaitStream* entry = nullptr;
    for (DestWaitStream& candidate : destWaitStreams_) {
      if (candidate.source == source) {
        entry = &candidate;
        break;
      }
    }
    if (entry == nullptr) {
      destWaitStreams_.push_back({source, nullptr, nullptr, 0});
      entry = &destWaitStreams_.back();
    }
    if (entry->stream == nullptr) {
      NCCL_M2N_CUDACHECK(cudaStreamCreateWithFlags(&entry->stream, cudaStreamNonBlocking));
    }
    if (entry->joinEvent == nullptr) {
      NCCL_M2N_CUDACHECK(cudaEventCreateWithFlags(&entry->joinEvent, cudaEventDisableTiming));
    }
    if (entry->beginGeneration != beginGeneration_) {
      NCCL_M2N_CHECK(pipeHostWaitEvent(entry->stream, startEvent_, rank, "dest_wait.start"));
      entry->beginGeneration = beginGeneration_;
    }
    *out = entry->stream;
    return ncclSuccess;
  }

  ncclResult_t recordTargetEdge(PipeHostEventId eventId, PipeHostStreamId streamId, size_t round, int channel,
                                int target, int rank, const char* tag) const {
    return recordEdge(eventId, streamId, round, channel, target, numRdmaTargets_, rank, tag);
  }

  ncclResult_t waitTargetEdge(PipeHostStreamId streamId, PipeHostEventId eventId, size_t round, int channel, int target,
                              int rank, const char* tag) const {
    return waitEdge(streamId, eventId, round, channel, target, numRdmaTargets_, rank, tag);
  }

  ncclResult_t recordTargetEdgeOnStream(PipeHostEventId eventId, cudaStream_t stream, size_t round, int channel,
                                        int target, int rank, const char* tag) const {
    return recordEdgeOnStream(eventId, stream, round, channel, target, numRdmaTargets_, rank, tag);
  }

  ncclResult_t waitTargetEdgeOnStream(cudaStream_t stream, PipeHostEventId eventId, size_t round, int channel,
                                      int target, int rank, const char* tag) const {
    return waitEdgeOnStream(stream, eventId, round, channel, target, numRdmaTargets_, rank, tag);
  }

  ncclResult_t recordDestEdge(PipeHostEventId eventId, PipeHostStreamId streamId, size_t round, int channel, int source,
                              int rank, const char* tag) const {
    return recordEdge(eventId, streamId, round, channel, source, numDestSources_, rank, tag);
  }

  ncclResult_t recordDestEdgeOnStream(PipeHostEventId eventId, cudaStream_t stream, size_t round, int channel,
                                      int source, int rank, const char* tag) const {
    return recordEdgeOnStream(eventId, stream, round, channel, source, numDestSources_, rank, tag);
  }

  ncclResult_t waitDestEdge(PipeHostStreamId streamId, PipeHostEventId eventId, size_t round, int channel, int source,
                            int rank, const char* tag) const {
    return waitEdge(streamId, eventId, round, channel, source, numDestSources_, rank, tag);
  }

  ncclResult_t recordFollowerEdge(PipeHostEventId eventId, PipeHostStreamId streamId, size_t round, int channel,
                                  int source, int rank, const char* tag) const {
    return recordEdge(eventId, streamId, round, channel, source, numLsaSources_, rank, tag);
  }

  ncclResult_t waitFollowerEdge(PipeHostStreamId streamId, PipeHostEventId eventId, size_t round, int channel,
                                int source, int rank, const char* tag) const {
    return waitEdge(streamId, eventId, round, channel, source, numLsaSources_, rank, tag);
  }

private:
  ncclResult_t eventCountFor(PipeHostEventId eventId, size_t rounds, size_t* eventCount) const {
    NCCL_M2N_CHECK_ARG(eventCount != nullptr, -1, "PIPE host_rma: null event count");
    int edgeCount = 0;
    switch (eventId) {
    case PIPE_HOST_EVENT_TARGET_PACK_DONE:
    case PIPE_HOST_EVENT_TARGET_PUT_DONE:
    case PIPE_HOST_EVENT_TARGET_CREDIT_DONE:
      edgeCount = numRdmaTargets_;
      break;
    case PIPE_HOST_EVENT_DEST_EDGE_READY:
    case PIPE_HOST_EVENT_ROOT_COPY_DONE:
    case PIPE_HOST_EVENT_RING_PUT_DONE:
    case PIPE_HOST_EVENT_ROOT_SIGNAL_DONE:
    case PIPE_HOST_EVENT_ROOT_FOLLOWERS_DONE:
      edgeCount = numDestSources_;
      break;
    case PIPE_HOST_EVENT_FOLLOWER_READY:
    case PIPE_HOST_EVENT_FOLLOWER_COPY_DONE:
      edgeCount = numLsaSources_;
      break;
    default:
      NCCL_M2N_FAIL(ncclInternalError, -1, "PIPE host_rma: invalid event id %d", (int)eventId);
    }
    NCCL_M2N_CHECK_ARG(edgeCount > 0 && m2nCheckedMulSize(rounds, static_cast<size_t>(numChannels_), eventCount) &&
                         m2nCheckedMulSize(*eventCount, static_cast<size_t>(edgeCount), eventCount),
                       -1, "PIPE host_rma: edge event count overflow");
    return ncclSuccess;
  }

  cudaEvent_t edgeEvent(PipeHostEventId eventId, size_t round, int channel, int edge, int edgeCount) const {
    const size_t index =
      (round * static_cast<size_t>(numChannels_) + static_cast<size_t>(channel)) * static_cast<size_t>(edgeCount) +
      static_cast<size_t>(edge);
    return events_[eventId][index];
  }

  ncclResult_t recordEdge(PipeHostEventId eventId, PipeHostStreamId streamId, size_t round, int channel, int edge,
                          int edgeCount, int rank, const char* tag) const {
    return recordEdgeOnStream(eventId, channelStream(streamId, channel), round, channel, edge, edgeCount, rank, tag);
  }

  ncclResult_t recordEdgeOnStream(PipeHostEventId eventId, cudaStream_t stream, size_t round, int channel, int edge,
                                  int edgeCount, int rank, const char* tag) const {
    if (!pipelineEnabled_) {
      return ncclSuccess;
    }
    NCCL_M2N_CHECK_ARG(channel >= 0 && channel < numChannels_ && edge >= 0 && edge < edgeCount, rank,
                       "PIPE host_rma: invalid edge channel=%d edge=%d/%d", channel, edge, edgeCount);
    return pipeHostRecordEvent(edgeEvent(eventId, round, channel, edge, edgeCount), stream, rank, tag);
  }

  ncclResult_t waitEdge(PipeHostStreamId streamId, PipeHostEventId eventId, size_t round, int channel, int edge,
                        int edgeCount, int rank, const char* tag) const {
    return waitEdgeOnStream(channelStream(streamId, channel), eventId, round, channel, edge, edgeCount, rank, tag);
  }

  ncclResult_t waitEdgeOnStream(cudaStream_t stream, PipeHostEventId eventId, size_t round, int channel, int edge,
                                int edgeCount, int rank, const char* tag) const {
    if (!pipelineEnabled_) {
      return ncclSuccess;
    }
    NCCL_M2N_CHECK_ARG(channel >= 0 && channel < numChannels_ && edge >= 0 && edge < edgeCount, rank,
                       "PIPE host_rma: invalid edge channel=%d edge=%d/%d", channel, edge, edgeCount);
    return pipeHostWaitEvent(stream, edgeEvent(eventId, round, channel, edge, edgeCount), rank, tag);
  }

  ncclResult_t ensureEventVector(std::vector<cudaEvent_t>* events, size_t eventCount) {
    NCCL_M2N_CHECK_ARG(events != nullptr, -1, "PIPE host_rma: null event vector");
    const size_t existingCount = events->size();
    events->resize(std::max(existingCount, eventCount), nullptr);
    for (size_t i = existingCount; i < eventCount; i++) {
      NCCL_M2N_CUDACHECK(cudaEventCreateWithFlags(&(*events)[i], cudaEventDisableTiming));
    }
    return ncclSuccess;
  }

  ncclResult_t joinOneStream(cudaStream_t pipelineStream, cudaEvent_t joinEvent) {
    if (pipelineStream == nullptr || joinEvent == nullptr) {
      return ncclSuccess;
    }
    ncclResult_t result = pipeHostRecordEvent(joinEvent, pipelineStream, -1, "pipeline_join");
    if (result == ncclSuccess) {
      result = pipeHostWaitEvent(joinStream_, joinEvent, -1, "pipeline_join");
    }
    return result;
  }

  void destroyEventVector(std::vector<cudaEvent_t>* events) {
    for (cudaEvent_t& event : *events) {
      if (event != nullptr) {
        (void)cudaEventDestroy(event);
        event = nullptr;
      }
    }
    events->clear();
  }

  void destroy() {
    for (auto& pipelineEvents : events_) {
      destroyEventVector(&pipelineEvents);
    }

    if (startEvent_ != nullptr) {
      (void)cudaEventDestroy(startEvent_);
      startEvent_ = nullptr;
    }
    for (cudaStream_t& pipelineStream : streams_) {
      destroyStream(&pipelineStream);
    }
    for (cudaEvent_t& joinEvent : joinEvents_) {
      destroyEvent(&joinEvent);
    }
    for (cudaStream_t& pipelineStream : channelPackStreams_) {
      destroyStream(&pipelineStream);
    }
    for (cudaEvent_t& joinEvent : channelPackJoinEvents_) {
      destroyEvent(&joinEvent);
    }
    for (SourcePutStream& sourcePutStream : sourcePutStreams_) {
      destroyStream(&sourcePutStream.stream);
      destroyEvent(&sourcePutStream.joinEvent);
    }
    for (DestWaitStream& destWaitStream : destWaitStreams_) {
      destroyStream(&destWaitStream.stream);
      destroyEvent(&destWaitStream.joinEvent);
    }
    channelPackStreams_.clear();
    channelPackJoinEvents_.clear();
    sourcePutStreams_.clear();
    destWaitStreams_.clear();
    rounds_ = 0;
    joinStream_ = nullptr;
    streamMask_ = 0;
    eventMask_ = 0;
    pipelineEnabled_ = false;
    numChannels_ = 0;
    numRdmaTargets_ = 0;
    numDestSources_ = 0;
    numLsaSources_ = 0;
    channelSourcePipeline_ = false;
    destWaitPipeline_ = false;
  }

  void destroyStream(cudaStream_t* stream) {
    if (*stream != nullptr) {
      (void)cudaStreamDestroy(*stream);
      *stream = nullptr;
    }
  }

  void destroyEvent(cudaEvent_t* event) {
    if (*event != nullptr) {
      (void)cudaEventDestroy(*event);
      *event = nullptr;
    }
  }

  cudaStream_t joinStream_ = nullptr;
  size_t rounds_ = 0;
  uint64_t streamMask_ = 0;
  uint64_t eventMask_ = 0;
  bool pipelineEnabled_ = false;
  int numChannels_ = 0;
  int numRdmaTargets_ = 0;
  bool channelSourcePipeline_ = false;
  bool destWaitPipeline_ = false;
  std::array<cudaStream_t, PIPE_HOST_STREAM_COUNT> streams_{};
  std::array<cudaEvent_t, PIPE_HOST_STREAM_COUNT> joinEvents_{};
  std::vector<cudaStream_t> channelPackStreams_;
  std::vector<cudaEvent_t> channelPackJoinEvents_;
  struct SourcePutStream {
    ncclComm_t comm;
    int peer;
    cudaStream_t stream;
    cudaEvent_t joinEvent;
    uint64_t beginGeneration;
  };
  std::vector<SourcePutStream> sourcePutStreams_;
  struct DestWaitStream {
    int source;
    cudaStream_t stream;
    cudaEvent_t joinEvent;
    uint64_t beginGeneration;
  };
  std::vector<DestWaitStream> destWaitStreams_;
  cudaEvent_t startEvent_ = nullptr;
  uint64_t beginGeneration_ = 0;
  int numDestSources_ = 0;
  int numLsaSources_ = 0;
  std::array<std::vector<cudaEvent_t>, PIPE_HOST_EVENT_COUNT> events_;
};

extern "C" void stagingPipeHostRmaPipelineDestroy(void* pipeline) {
  delete static_cast<PipeHostPipeline*>(pipeline);
}

static ncclResult_t pipeHostChunkInfo(const StagingPipePeerEdge& edge, size_t globalChunk, int rank, size_t* byteStart,
                                      size_t* bytes) {
  NCCL_M2N_CHECK_ARG(byteStart != nullptr && bytes != nullptr, rank, "PIPE host_rma: null chunk info output");
  const size_t chunkSize = edge.logicalChunkSize;
  NCCL_M2N_CHECK_ARG(chunkSize > 0, rank, "PIPE host_rma: invalid chunk size 0");
  size_t start = 0;
  NCCL_M2N_CHECK_ARG(m2nCheckedMulSize(globalChunk, chunkSize, &start), rank,
                     "PIPE host_rma: chunk offset overflow (chunk=%zu chunkSize=%zu)", globalChunk, chunkSize);
  NCCL_M2N_CHECK_ARG(start < edge.totalBytes, rank, "PIPE host_rma: chunk start %zu exceeds edge bytes %zu", start,
                     edge.totalBytes);
  const size_t remaining = edge.totalBytes - start;
  *byteStart = start;
  *bytes = std::min(chunkSize, remaining);
  return ncclSuccess;
}

struct PipeHostChunk {
  bool active;
  bool isBox;
  size_t byteStart;
  size_t bytes;
  size_t width;
  size_t height;
  size_t depth;
};

/* A cudaMemcpy3D descriptor covers one affine 1D/2D/3D box, not an
 * arbitrary flat byte range.  Host-RMA therefore partitions eligible layouts
 * into row- or plane-aligned boxes before assigning them to channels.  The
 * fixed-size FIFO slots and counter rounds remain unchanged; only the payload
 * bytes in the final box may be smaller than a slot. */
struct PipeHostChunkGeometry {
  size_t width;
  size_t height;
  size_t depth;
  size_t rowsPerTile;
  size_t depthsPerTile;
  size_t tilesPerDepth;
  size_t totalTiles;
};

static bool pipeHostBuildChunkGeometry(const StagingPipePeerEdge& edge, const StagingPipeCopyLayout& layout,
                                       size_t chunkSize, PipeHostChunkGeometry* geometry) {
  *geometry = {};
  if (!pipeHostEdgeActive(edge) || layout.isContiguous || layout.numOuterLoops < 1 || layout.numOuterLoops > 2 ||
      layout.innerSize == 0 || layout.innerSize > chunkSize) {
    return false;
  }

  const size_t width = layout.innerSize;
  const size_t height = layout.outerCounts[layout.numOuterLoops - 1];
  const size_t depth = layout.numOuterLoops == 2 ? layout.outerCounts[0] : 1;
  if (height == 0 || depth == 0) {
    return false;
  }

  size_t logicalBytes = 0;
  if (!m2nCheckedMulSize(width, height, &logicalBytes) || !m2nCheckedMulSize(logicalBytes, depth, &logicalBytes) ||
      logicalBytes != edge.totalBytes) {
    return false;
  }

  const size_t rowsPerTile = chunkSize / width;
  if (rowsPerTile == 0) {
    return false;
  }
  geometry->width = width;
  geometry->height = height;
  geometry->depth = depth;
  geometry->rowsPerTile = rowsPerTile;
  if (rowsPerTile >= height) {
    const size_t maxDepthsByCapacity = rowsPerTile / height;
    const size_t flatTileCount = 1 + (edge.totalBytes - 1) / chunkSize;
    /* Full planes can be much smaller than chunkSize.  Do not collapse the
     * transport schedule to fewer tiles than the established byte-chunk
     * schedule: that would idle channels and remove pipeline overlap. */
    const size_t maxDepthsByParallelism =
      flatTileCount > 1 ? std::max<size_t>(1, (depth - 1) / (flatTileCount - 1)) : depth;
    geometry->depthsPerTile = std::min(maxDepthsByCapacity, maxDepthsByParallelism);
    geometry->totalTiles = 1 + (depth - 1) / geometry->depthsPerTile;
  } else {
    geometry->tilesPerDepth = 1 + (height - 1) / rowsPerTile;
    if (!m2nCheckedMulSize(depth, geometry->tilesPerDepth, &geometry->totalTiles)) {
      *geometry = {};
      return false;
    }
  }
  return true;
}

static bool pipeHostLayoutSupportsBoxCopy(const StagingPipeCopyLayout& layout) {
  if (layout.numOuterLoops < 1 || layout.numOuterLoops > 2 || layout.innerSize == 0 ||
      layout.outerStrides[layout.numOuterLoops - 1] < layout.innerSize) {
    return false;
  }
  if (layout.numOuterLoops == 2) {
    const size_t pitch = layout.outerStrides[1];
    const size_t height = layout.outerCounts[1];
    if (pitch == 0 || layout.outerStrides[0] % pitch != 0 || layout.outerStrides[0] / pitch < height) {
      return false;
    }
  }
  return true;
}

static ncclResult_t pipeHostChunkTileRange(const StagingPipePeerEdge& edge, const PipeHostChunkGeometry& geometry,
                                           int rank, size_t* tileStart, size_t* tileEnd) {
  NCCL_M2N_CHECK_ARG(tileStart != nullptr && tileEnd != nullptr, rank,
                     "PIPE host_rma: null geometric tile range output");
  size_t startNumerator = 0;
  size_t endNumerator = 0;
  NCCL_M2N_CHECK_ARG(m2nCheckedMulSize(geometry.totalTiles, static_cast<size_t>(edge.channelRank), &startNumerator) &&
                       m2nCheckedMulSize(geometry.totalTiles, static_cast<size_t>(edge.channelRank + 1), &endNumerator),
                     rank, "PIPE host_rma: geometric tile range overflow");
  *tileStart = startNumerator / static_cast<size_t>(edge.channelCount);
  *tileEnd = endNumerator / static_cast<size_t>(edge.channelCount);
  return ncclSuccess;
}

static ncclResult_t pipeHostChunkForRound(const StagingPipePeerEdge& edge, const StagingPipeCopyLayout& layout,
                                          size_t round, int rank, PipeHostChunk* chunk) {
  NCCL_M2N_CHECK_ARG(chunk != nullptr, rank, "PIPE host_rma: null chunk output");
  *chunk = {};
  if (!pipeHostEdgeActive(edge)) {
    return ncclSuccess;
  }
  const size_t chunkSize = edge.logicalChunkSize;
  NCCL_M2N_CHECK_ARG(chunkSize > 0, rank, "PIPE host_rma: invalid edge chunk size 0");

  PipeHostChunkGeometry geometry;
  if (pipeHostBuildChunkGeometry(edge, layout, chunkSize, &geometry)) {
    size_t tileStart = 0;
    size_t tileEnd = 0;
    NCCL_M2N_CHECK(pipeHostChunkTileRange(edge, geometry, rank, &tileStart, &tileEnd));
    size_t tile = 0;
    NCCL_M2N_CHECK_ARG(m2nCheckedAddSize(tileStart, round, &tile), rank,
                       "PIPE host_rma: geometric tile round overflow");
    if (tile >= tileEnd) {
      return ncclSuccess;
    }

    size_t rowStart = 0;
    size_t tileHeight = geometry.height;
    size_t depthStart = 0;
    size_t tileDepth = 1;
    if (geometry.depthsPerTile > 0) {
      depthStart = tile * geometry.depthsPerTile;
      tileDepth = std::min(geometry.depthsPerTile, geometry.depth - depthStart);
    } else {
      depthStart = tile / geometry.tilesPerDepth;
      rowStart = (tile % geometry.tilesPerDepth) * geometry.rowsPerTile;
      tileHeight = std::min(geometry.rowsPerTile, geometry.height - rowStart);
    }

    size_t logicalRows = 0;
    size_t byteStart = 0;
    size_t bytes = 0;
    NCCL_M2N_CHECK_ARG(m2nCheckedMulSize(depthStart, geometry.height, &logicalRows) &&
                         m2nCheckedAddSize(logicalRows, rowStart, &logicalRows) &&
                         m2nCheckedMulSize(logicalRows, geometry.width, &byteStart) &&
                         m2nCheckedMulSize(geometry.width, tileHeight, &bytes) &&
                         m2nCheckedMulSize(bytes, tileDepth, &bytes) && byteStart < edge.totalBytes &&
                         bytes <= chunkSize && bytes <= edge.totalBytes - byteStart,
                       rank, "PIPE host_rma: invalid geometric tile");
    chunk->active = true;
    chunk->isBox = pipeHostLayoutSupportsBoxCopy(layout);
    chunk->byteStart = byteStart;
    chunk->bytes = bytes;
    chunk->width = geometry.width;
    chunk->height = tileHeight;
    chunk->depth = tileDepth;
    return ncclSuccess;
  }

  size_t globalChunk = 0;
  NCCL_M2N_CHECK_ARG(m2nCheckedAddSize(edge.chunkStart, round, &globalChunk), rank,
                     "PIPE host_rma: chunk round overflow");
  if (globalChunk >= edge.chunkEnd) {
    return ncclSuccess;
  }
  NCCL_M2N_CHECK(pipeHostChunkInfo(edge, globalChunk, rank, &chunk->byteStart, &chunk->bytes));
  chunk->active = true;
  return ncclSuccess;
}

static ncclResult_t pipeHostChunkCountForEdge(const StagingPipePeerEdge& edge, const StagingPipeCopyLayout& layout,
                                              int rank, size_t* count) {
  NCCL_M2N_CHECK_ARG(count != nullptr, rank, "PIPE host_rma: null chunk count output");
  *count = 0;
  if (!pipeHostEdgeActive(edge)) {
    return ncclSuccess;
  }
  const size_t chunkSize = edge.logicalChunkSize;
  NCCL_M2N_CHECK_ARG(chunkSize > 0, rank, "PIPE host_rma: invalid edge chunk size 0");
  PipeHostChunkGeometry geometry;
  if (pipeHostBuildChunkGeometry(edge, layout, chunkSize, &geometry)) {
    size_t tileStart = 0;
    size_t tileEnd = 0;
    NCCL_M2N_CHECK(pipeHostChunkTileRange(edge, geometry, rank, &tileStart, &tileEnd));
    *count = tileEnd - tileStart;
  } else {
    *count = edge.chunkEnd - edge.chunkStart;
  }
  return ncclSuccess;
}

static ncclResult_t pipeHostPlanMaxChunkRounds(const StagingKernelParams* params, const StagingPipeDevicePlan* plan,
                                               int rank, size_t* rounds) {
  NCCL_M2N_CHECK_ARG(params != nullptr && plan != nullptr && rounds != nullptr, rank,
                     "PIPE host_rma: null plan round-count input");
  *rounds = 0;
  auto update = [&](const StagingPipePeerEdge& edge, const StagingPipeCopyLayout& layout) {
    size_t edgeRounds = 0;
    ncclResult_t result = pipeHostChunkCountForEdge(edge, layout, rank, &edgeRounds);
    if (result == ncclSuccess) {
      *rounds = std::max(*rounds, edgeRounds);
    }
    return result;
  };
  for (int ch = 0; ch < params->numChannels; ch++) {
    for (int t = 0; t < params->numRdmaTargets; t++) {
      const StagingPipePeerEdge& edge = plan->rdmaTargets[ch][t].peer;
      NCCL_M2N_CHECK(update(edge, plan->rdmaTargetLayouts[ch][edge.copyLayoutIndex]));
    }
    for (int s = 0; s < params->numRdmaSources; s++) {
      NCCL_M2N_CHECK(update(plan->rdmaSources[ch][s], plan->rdmaSourceLayouts[ch][s]));
    }
    for (int s = 0; s < params->numLsaSources; s++) {
      NCCL_M2N_CHECK(update(plan->lsaSources[ch][s], plan->lsaSourceLayouts[ch][s]));
    }
  }
  return ncclSuccess;
}

static ncclResult_t pipeHostSlotOffset(const StagingFlowCtrl& fc, size_t slot, size_t bytes, int rank, size_t* offset) {
  NCCL_M2N_CHECK_ARG(offset != nullptr, rank, "PIPE host_rma: null slot offset output");
  NCCL_M2N_CHECK_ARG(fc.peerNumSlots > 0 && fc.peerChunkSize > 0, rank,
                     "PIPE host_rma: invalid peer slots=%d chunkSize=%zu", fc.peerNumSlots, fc.peerChunkSize);
  NCCL_M2N_CHECK_ARG(slot < static_cast<size_t>(fc.peerNumSlots), rank,
                     "PIPE host_rma: slot %zu exceeds peerNumSlots=%d", slot, fc.peerNumSlots);
  size_t slotBytes = 0;
  size_t slotOffset = 0;
  NCCL_M2N_CHECK_ARG(m2nCheckedMulSize(slot, fc.peerChunkSize, &slotBytes) &&
                       m2nCheckedAddSize(fc.peerDataOffset, slotBytes, &slotOffset),
                     rank, "PIPE host_rma: peer staging offset overflow");
  NCCL_M2N_CHECK_ARG(bytes <= fc.peerChunkSize, rank, "PIPE host_rma: chunk bytes %zu exceed peer chunk size %zu",
                     bytes, fc.peerChunkSize);
  *offset = slotOffset;
  return ncclSuccess;
}

static ncclResult_t pipeHostLayoutOffset(const StagingPipeCopyLayout& layout, size_t iter, int rank,
                                         size_t* outOffset) {
  NCCL_M2N_CHECK_ARG(outOffset != nullptr, rank, "PIPE host_rma: null layout offset output");
  size_t offset = layout.baseOffset;
  size_t tmp = iter;
  for (int d = layout.numOuterLoops - 1; d >= 0; d--) {
    const size_t count = layout.outerCounts[d];
    NCCL_M2N_CHECK_ARG(count > 0, rank, "PIPE host_rma: layout outerCounts[%d] is zero", d);
    const size_t idx = tmp % count;
    tmp /= count;
    size_t delta = 0;
    NCCL_M2N_CHECK_ARG(m2nCheckedMulSize(idx, layout.outerStrides[d], &delta) &&
                         m2nCheckedAddSize(offset, delta, &offset),
                       rank, "PIPE host_rma: layout offset overflow");
  }
  *outOffset = offset;
  return ncclSuccess;
}

static ncclResult_t pipeHostMemcpy1D(void* dst, const void* src, size_t bytes, cudaStream_t stream, int rank = -1,
                                     const char* tag = "memcpy1d") {
  if (bytes == 0) {
    return ncclSuccess;
  }
  RESHARD_DEBUG(rank, "PIPE host_rma %s memcpy1d enqueue begin bytes=%zu src=%p dst=%p", tag, bytes, src, dst);
  NCCL_M2N_CUDACHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream));
  RESHARD_DEBUG(rank, "PIPE host_rma %s memcpy1d enqueue end bytes=%zu src=%p dst=%p", tag, bytes, src, dst);
  NCCL_M2N_CHECK(pipeHostTraceSync(stream, rank, tag, "memcpy1d", -1));
  return ncclSuccess;
}

static ncclResult_t pipeHostMemcpyRows(void* dst, const void* src, size_t width, size_t rows, size_t dstPitch,
                                       size_t srcPitch, cudaStream_t stream, int rank = -1,
                                       const char* tag = "memcpy_rows") {
  if (width == 0 || rows == 0) {
    return ncclSuccess;
  }
  if (rows == 1 || (dstPitch == width && srcPitch == width)) {
    size_t bytes = 0;
    NCCL_M2N_CHECK_ARG(m2nCheckedMulSize(width, rows, &bytes), -1, "PIPE host_rma: row memcpy byte count overflow");
    return pipeHostMemcpy1D(dst, src, bytes, stream, rank, tag);
  }
  cudaMemcpy3DParms cp{};
  cp.kind = cudaMemcpyDeviceToDevice;
  cp.srcPtr = make_cudaPitchedPtr(const_cast<void*>(src), srcPitch, width, rows);
  cp.dstPtr = make_cudaPitchedPtr(dst, dstPitch, width, rows);
  cp.extent = make_cudaExtent(width, rows, 1);
  RESHARD_DEBUG(rank,
                "PIPE host_rma %s memcpy3d enqueue begin width=%zu rows=%zu src_pitch=%zu dst_pitch=%zu src=%p "
                "dst=%p",
                tag, width, rows, srcPitch, dstPitch, src, dst);
  NCCL_M2N_CUDACHECK(cudaMemcpy3DAsync(&cp, stream));
  RESHARD_DEBUG(rank,
                "PIPE host_rma %s memcpy3d enqueue end width=%zu rows=%zu src_pitch=%zu dst_pitch=%zu src=%p "
                "dst=%p",
                tag, width, rows, srcPitch, dstPitch, src, dst);
  NCCL_M2N_CHECK(pipeHostTraceSync(stream, rank, tag, "memcpy3d", -1));
  return ncclSuccess;
}

static ncclResult_t pipeHostMemcpyBox(void* dst, const void* src, size_t width, size_t height, size_t depth,
                                      size_t dstPitch, size_t dstSliceHeight, size_t srcPitch, size_t srcSliceHeight,
                                      cudaStream_t stream, int rank, const char* tag) {
  NCCL_M2N_CHECK_ARG(width > 0 && height > 0 && depth > 0 && dstPitch >= width && srcPitch >= width, rank,
                     "PIPE host_rma: invalid %s memcpy3d box", tag);
  if (depth == 1) {
    return pipeHostMemcpyRows(dst, src, width, height, dstPitch, srcPitch, stream, rank, tag);
  }
  NCCL_M2N_CHECK_ARG(dstSliceHeight >= height && srcSliceHeight >= height, rank,
                     "PIPE host_rma: invalid %s memcpy3d slice height", tag);
  cudaMemcpy3DParms cp{};
  cp.kind = cudaMemcpyDeviceToDevice;
  cp.srcPtr = make_cudaPitchedPtr(const_cast<void*>(src), srcPitch, width, srcSliceHeight);
  cp.dstPtr = make_cudaPitchedPtr(dst, dstPitch, width, dstSliceHeight);
  cp.extent = make_cudaExtent(width, height, depth);
  RESHARD_DEBUG(rank,
                "PIPE host_rma %s memcpy3d enqueue begin width=%zu height=%zu depth=%zu src_pitch=%zu "
                "dst_pitch=%zu src=%p dst=%p",
                tag, width, height, depth, srcPitch, dstPitch, src, dst);
  NCCL_M2N_CUDACHECK(cudaMemcpy3DAsync(&cp, stream));
  RESHARD_DEBUG(rank,
                "PIPE host_rma %s memcpy3d enqueue end width=%zu height=%zu depth=%zu src_pitch=%zu "
                "dst_pitch=%zu src=%p dst=%p",
                tag, width, height, depth, srcPitch, dstPitch, src, dst);
  NCCL_M2N_CHECK(pipeHostTraceSync(stream, rank, tag, "memcpy3d", -1));
  return ncclSuccess;
}

enum PipeHostLayoutCopyDir {
  PIPE_HOST_LAYOUT_TO_CONTIG,
  PIPE_HOST_CONTIG_TO_LAYOUT
};

/* CE pack/unpack counterpart to the device layout-copy helpers.  It emits a
 * single cudaMemcpyAsync for contiguous spans, cudaMemcpy3DAsync for maximal
 * fast-dimension row runs, and smaller 1D copies only for partial row edges. */
static ncclResult_t pipeHostEnqueueLayoutCopy(void* dst, const void* src, const StagingPipeCopyLayout& layout,
                                              size_t byteStart, size_t bytes, cudaStream_t stream, int rank,
                                              PipeHostLayoutCopyDir dir) {
  if (bytes == 0) {
    return ncclSuccess;
  }
  const bool pack = dir == PIPE_HOST_LAYOUT_TO_CONTIG;
  const char* const op = pack ? "pack" : "unpack";
  NCCL_M2N_CHECK_ARG(dst != nullptr && src != nullptr, rank, "PIPE host_rma: %s copy requires non-null pointers", op);
  NCCL_M2N_CHECK_ARG(layout.innerSize > 0, rank, "PIPE host_rma: %s layout has zero innerSize", op);
  if (layout.isContiguous || layout.numOuterLoops <= 0) {
    size_t layoutOffset = 0;
    NCCL_M2N_CHECK_ARG(m2nCheckedAddSize(layout.baseOffset, byteStart, &layoutOffset), rank,
                       "PIPE host_rma: contiguous %s offset overflow", op);
    char* const dstBytes = static_cast<char*>(dst);
    const char* const srcBytes = static_cast<const char*>(src);
    return pack ? pipeHostMemcpy1D(dstBytes, srcBytes + layoutOffset, bytes, stream, rank, op) :
                  pipeHostMemcpy1D(dstBytes + layoutOffset, srcBytes, bytes, stream, rank, op);
  }

  char* const dstBase = static_cast<char*>(dst);
  const char* const srcBase = static_cast<const char*>(src);
  const size_t inner = layout.innerSize;
  size_t iter = byteStart / inner;
  size_t firstOffset = byteStart % inner;
  size_t done = 0;
  size_t remaining = bytes;
  while (remaining > 0) {
    size_t layoutOffset = 0;
    NCCL_M2N_CHECK(pipeHostLayoutOffset(layout, iter, rank, &layoutOffset));
    if (firstOffset != 0 || remaining < inner) {
      const size_t copyStart = firstOffset;
      const size_t copyBytes = std::min(inner - copyStart, remaining);
      char* const copyDst = pack ? dstBase + done : dstBase + layoutOffset + copyStart;
      const char* const copySrc = pack ? srcBase + layoutOffset + copyStart : srcBase + done;
      NCCL_M2N_CHECK(pipeHostMemcpy1D(copyDst, copySrc, copyBytes, stream, rank, op));
      done += copyBytes;
      remaining -= copyBytes;
      iter++;
      firstOffset = 0;
      continue;
    }

    const size_t fullRows = remaining / inner;
    const int fastDim = layout.numOuterLoops - 1;
    const size_t fastCount = layout.outerCounts[fastDim];
    NCCL_M2N_CHECK_ARG(fastCount > 0, rank, "PIPE host_rma: layout outerCounts[%d] is zero", fastDim);
    NCCL_M2N_CHECK_ARG(layout.outerStrides[fastDim] >= inner, rank,
                       "PIPE host_rma: %s layout stride %zu is smaller than inner size %zu", op,
                       layout.outerStrides[fastDim], inner);
    const size_t fastIdx = iter % fastCount;
    const size_t rows = std::min(fullRows, fastCount - fastIdx);
    NCCL_M2N_CHECK_ARG(rows > 0, rank, "PIPE host_rma: computed zero-row %s span", op);
    size_t copyBytes = 0;
    NCCL_M2N_CHECK_ARG(m2nCheckedMulSize(rows, inner, &copyBytes), rank,
                       "PIPE host_rma: %s row span byte count overflow", op);
    void* const copyDst = pack ? static_cast<void*>(dstBase + done) : static_cast<void*>(dstBase + layoutOffset);
    const void* const copySrc =
      pack ? static_cast<const void*>(srcBase + layoutOffset) : static_cast<const void*>(srcBase + done);
    const size_t dstPitch = pack ? inner : layout.outerStrides[fastDim];
    const size_t srcPitch = pack ? layout.outerStrides[fastDim] : inner;
    NCCL_M2N_CHECK(pipeHostMemcpyRows(copyDst, copySrc, inner, rows, dstPitch, srcPitch, stream, rank, op));
    done += copyBytes;
    remaining -= copyBytes;
    iter += rows;
  }
  return ncclSuccess;
}

static ncclResult_t pipeHostEnqueueLayoutBoxCopy(void* dst, const void* src, const StagingPipeCopyLayout& layout,
                                                 const PipeHostChunk& chunk, cudaStream_t stream, int rank,
                                                 PipeHostLayoutCopyDir dir) {
  const bool pack = dir == PIPE_HOST_LAYOUT_TO_CONTIG;
  const char* const op = pack ? "pack" : "unpack";
  NCCL_M2N_CHECK_ARG(chunk.isBox && chunk.width == layout.innerSize && layout.numOuterLoops >= 1 &&
                       layout.numOuterLoops <= 2 && chunk.height > 0 && chunk.depth > 0,
                     rank, "PIPE host_rma: invalid %s memcpy3d tile", op);
  NCCL_M2N_CHECK_ARG(dst != nullptr && src != nullptr, rank, "PIPE host_rma: %s memcpy3d requires non-null pointers",
                     op);

  size_t iter = chunk.byteStart / chunk.width;
  size_t layoutOffset = 0;
  NCCL_M2N_CHECK(pipeHostLayoutOffset(layout, iter, rank, &layoutOffset));
  const size_t layoutPitch = layout.outerStrides[layout.numOuterLoops - 1];
  const size_t layoutSliceHeight = chunk.depth > 1 ? layout.outerStrides[0] / layoutPitch : chunk.height;
  char* const dstBytes = static_cast<char*>(dst);
  const char* const srcBytes = static_cast<const char*>(src);
  void* const copyDst = pack ? static_cast<void*>(dstBytes) : static_cast<void*>(dstBytes + layoutOffset);
  const void* const copySrc = pack ? static_cast<const void*>(srcBytes + layoutOffset) : srcBytes;
  const size_t dstPitch = pack ? chunk.width : layoutPitch;
  const size_t dstSliceHeight = pack ? chunk.height : layoutSliceHeight;
  const size_t srcPitch = pack ? layoutPitch : chunk.width;
  const size_t srcSliceHeight = pack ? layoutSliceHeight : chunk.height;
  return pipeHostMemcpyBox(copyDst, copySrc, chunk.width, chunk.height, chunk.depth, dstPitch, dstSliceHeight, srcPitch,
                           srcSliceHeight, stream, rank, op);
}

static ncclResult_t pipeHostEnqueueLayoutToContig(void* contigDst, const void* layoutSrc,
                                                  const StagingPipeCopyLayout& layout, size_t byteStart, size_t bytes,
                                                  cudaStream_t stream, int rank, const PipeHostChunk* chunk = nullptr) {
  if (chunk != nullptr && chunk->isBox) {
    return pipeHostEnqueueLayoutBoxCopy(contigDst, layoutSrc, layout, *chunk, stream, rank, PIPE_HOST_LAYOUT_TO_CONTIG);
  }
  return pipeHostEnqueueLayoutCopy(contigDst, layoutSrc, layout, byteStart, bytes, stream, rank,
                                   PIPE_HOST_LAYOUT_TO_CONTIG);
}

static ncclResult_t pipeHostEnqueueContigToLayout(void* layoutDst, const void* contigSrc,
                                                  const StagingPipeCopyLayout& layout, size_t byteStart, size_t bytes,
                                                  cudaStream_t stream, int rank, const PipeHostChunk* chunk = nullptr) {
  if (chunk != nullptr && chunk->isBox) {
    return pipeHostEnqueueLayoutBoxCopy(layoutDst, contigSrc, layout, *chunk, stream, rank, PIPE_HOST_CONTIG_TO_LAYOUT);
  }
  return pipeHostEnqueueLayoutCopy(layoutDst, contigSrc, layout, byteStart, bytes, stream, rank,
                                   PIPE_HOST_CONTIG_TO_LAYOUT);
}

static int pipeHostTrainerTargetEnd(const StagingKernelParams* params) {
  const int ringTargets = (params != nullptr && params->numRingTargets > 0) ? params->numRingTargets : 0;
  const int end = (params != nullptr) ? params->numRdmaTargets - ringTargets : 0;
  return std::max(0, end);
}

static int pipeHostRingTargetStart(const StagingKernelParams* params) {
  return pipeHostTrainerTargetEnd(params);
}

static bool pipeHostEdgeUsesComm(const StagingPipePeerEdge& edge, PipeHostCommId commId) {
  return pipeHostEdgeActive(edge) && pipeHostCommIdForTransport(edge.rdmaTransport) == commId;
}

/* ncclPutSignal enters a group; record the stream completion only after the
 * group closes and NCCL has enqueued the operation. */
struct PipeHostPutCompletion {
  cudaStream_t stream;
  int target;
};

struct PipeHostPutCompletionList {
  std::array<PipeHostPutCompletion, MAX_TARGETS> entries{};
  int count = 0;

  ncclResult_t add(cudaStream_t stream, int target, int rank) {
    NCCL_M2N_CHECK_ARG(count < MAX_TARGETS, rank, "PIPE host_rma: source put completion list overflow");
    entries[count++] = {stream, target};
    return ncclSuccess;
  }
};

static ncclResult_t pipeHostPhasedSourcePacks(const StagingKernelParams* params, const StagingPipeDevicePlan* plan,
                                              const StagingPipeCallParams* call, void* stagingBuffer,
                                              const PipeHostPipeline* pipeline, bool waitForSlotCredit,
                                              cudaStream_t stream, size_t round, int onlyChannel = -1) {
  if (!params->isSource || params->numRdmaTargets <= 0) {
    return ncclSuccess;
  }
  NCCL_M2N_CHECK_ARG(call->srcBuffer != nullptr, params->myRank, "PIPE host_rma: source rank has null srcBuffer");
  const int targetEnd = pipeHostTrainerTargetEnd(params);
  const int firstChannel = onlyChannel < 0 ? 0 : onlyChannel;
  const int lastChannel = onlyChannel < 0 ? params->numChannels : onlyChannel + 1;
  NCCL_M2N_CHECK_ARG(firstChannel >= 0 && lastChannel <= params->numChannels, params->myRank,
                     "PIPE host_rma: invalid source-pack channel %d/%d", onlyChannel, params->numChannels);
  for (int ch = firstChannel; ch < lastChannel; ch++) {
    for (int t = 0; t < targetEnd; t++) {
      const StagingPipeTrainerEdge& trainerEdge = plan->rdmaTargets[ch][t];
      const StagingPipePeerEdge& target = trainerEdge.peer;
      PipeHostChunk chunk;
      NCCL_M2N_CHECK(pipeHostChunkForRound(target, plan->rdmaTargetLayouts[ch][target.copyLayoutIndex], round,
                                           params->myRank, &chunk));
      if (!chunk.active) {
        continue;
      }
      const size_t slot = round % static_cast<size_t>(trainerEdge.localFc.peerNumSlots);
      size_t localOffset = 0;
      NCCL_M2N_CHECK(pipeHostSlotOffset(trainerEdge.localFc, slot, chunk.bytes, params->myRank, &localOffset));
      const size_t reuseDistance = static_cast<size_t>(trainerEdge.localFc.peerNumSlots);
      if (waitForSlotCredit && pipeline != nullptr && round >= reuseDistance) {
        NCCL_M2N_CHECK(pipeline->waitTargetEdge(PIPE_HOST_STREAM_PACK, PIPE_HOST_EVENT_TARGET_CREDIT_DONE,
                                                round - reuseDistance, ch, t, params->myRank,
                                                "source_pack.slot_credit"));
      }
      const StagingPipeCopyLayout& layout = plan->rdmaTargetLayouts[ch][target.copyLayoutIndex];
      NCCL_M2N_CHECK(pipeHostEnqueueLayoutToContig(static_cast<char*>(stagingBuffer) + localOffset, call->srcBuffer,
                                                   layout, chunk.byteStart, chunk.bytes, stream, params->myRank,
                                                   &chunk));
      NCCL_M2N_CHECK(pipeline->recordTargetEdge(PIPE_HOST_EVENT_TARGET_PACK_DONE, PIPE_HOST_STREAM_PACK, round, ch, t,
                                                params->myRank, "source_pack_done"));
    }
  }
  return ncclSuccess;
}

static ncclResult_t pipeHostPhasedSourcePuts(const StagingKernelParams* params, const StagingPipeDevicePlan* plan,
                                             void* stagingBuffer, const PipeHostRmaContext* rma,
                                             PipeHostPipeline* pipeline, PipeHostCommId commId, size_t round,
                                             PipeHostPutCompletionList* completions, int onlyChannel = -1) {
  if (!params->isSource || params->numRdmaTargets <= 0) {
    return ncclSuccess;
  }
  NCCL_M2N_CHECK_ARG(completions != nullptr, params->myRank, "PIPE host_rma: null source-put completions");
  const PipeHostComm* hostComm = pipeHostCommForId(rma, commId, params->myRank);
  if (hostComm == nullptr) {
    return ncclSuccess;
  }
  const int targetEnd = pipeHostTrainerTargetEnd(params);
  const int firstChannel = onlyChannel < 0 ? 0 : onlyChannel;
  const int lastChannel = onlyChannel < 0 ? params->numChannels : onlyChannel + 1;
  NCCL_M2N_CHECK_ARG(firstChannel >= 0 && lastChannel <= params->numChannels, params->myRank,
                     "PIPE host_rma: invalid source-put channel %d/%d", onlyChannel, params->numChannels);
  for (int ch = firstChannel; ch < lastChannel; ch++) {
    for (int t = 0; t < targetEnd; t++) {
      const StagingPipeTrainerEdge& trainerEdge = plan->rdmaTargets[ch][t];
      const StagingPipePeerEdge& target = trainerEdge.peer;
      if (!pipeHostEdgeUsesComm(target, commId)) {
        continue;
      }
      PipeHostChunk chunk;
      NCCL_M2N_CHECK(pipeHostChunkForRound(target, plan->rdmaTargetLayouts[ch][target.copyLayoutIndex], round,
                                           params->myRank, &chunk));
      if (!chunk.active) {
        continue;
      }
      cudaStream_t putStream = nullptr;
      NCCL_M2N_CHECK(pipeline->sourcePutStream(hostComm->comm, target.peerWorldRank, params->myRank, &putStream));
      NCCL_M2N_CHECK(pipeline->waitTargetEdgeOnStream(putStream, PIPE_HOST_EVENT_TARGET_PACK_DONE, round, ch, t,
                                                      params->myRank, "source_put.wait_pack"));
      ncclWindow_t edgeWindow = nullptr;
      NCCL_M2N_CHECK(pipeHostRdmaWindow(params, target, params->myRank, &edgeWindow));
      const size_t localSlot = round % static_cast<size_t>(trainerEdge.localFc.peerNumSlots);
      const size_t remoteSlot = round % static_cast<size_t>(target.fc.peerNumSlots);
      size_t localOffset = 0;
      size_t remoteOffset = 0;
      NCCL_M2N_CHECK(pipeHostSlotOffset(trainerEdge.localFc, localSlot, chunk.bytes, params->myRank, &localOffset));
      NCCL_M2N_CHECK(pipeHostSlotOffset(target.fc, remoteSlot, chunk.bytes, params->myRank, &remoteOffset));
      NCCL_M2N_CHECK(pipeHostPutSignal(hostComm->comm, static_cast<char*>(stagingBuffer) + localOffset, chunk.bytes,
                                       target.peerWorldRank, edgeWindow, remoteOffset, putStream, params->myRank,
                                       "source.put_data"));
      NCCL_M2N_CHECK(completions->add(putStream, t, params->myRank));
    }
  }
  return ncclSuccess;
}

static ncclResult_t pipeHostSourceCreditActive(const StagingKernelParams* params, const StagingPipeDevicePlan* plan,
                                               size_t round, int channel, int targetIndex, bool* active) {
  NCCL_M2N_CHECK_ARG(active != nullptr, params->myRank, "PIPE host_rma: null credit activity output");
  *active = false;
  if (params->numRdmaTargets <= 0) {
    return ncclSuccess;
  }
  if (channel < 0 || channel >= params->numChannels || targetIndex < 0 || targetIndex >= params->numRdmaTargets) {
    return ncclSuccess;
  }
  const int ringStart = pipeHostRingTargetStart(params);
  const bool trainerTarget = targetIndex < ringStart && params->isSource;
  const bool ringTarget = targetIndex >= ringStart && params->isDest && params->numRingTargets > 0;
  if (!trainerTarget && !ringTarget) {
    return ncclSuccess;
  }
  const StagingPipePeerEdge& target = plan->rdmaTargets[channel][targetIndex].peer;
  PipeHostChunk chunk;
  NCCL_M2N_CHECK(pipeHostChunkForRound(target, plan->rdmaTargetLayouts[channel][target.copyLayoutIndex], round,
                                       params->myRank, &chunk));
  if (!chunk.active) {
    return ncclSuccess;
  }
  *active = chunk.active;
  return ncclSuccess;
}

static ncclResult_t pipeHostPhasedSourceCreditWait(const StagingKernelParams* params, const StagingPipeDevicePlan* plan,
                                                   PipeHostCommId commId, PipeHostWaitList* waits, size_t round,
                                                   int channel, int targetIndex) {
  bool active = false;
  NCCL_M2N_CHECK(pipeHostSourceCreditActive(params, plan, round, channel, targetIndex, &active));
  if (!active) {
    return ncclSuccess;
  }
  const StagingPipePeerEdge& target = plan->rdmaTargets[channel][targetIndex].peer;
  if (pipeHostEdgeUsesComm(target, commId)) {
    NCCL_M2N_CHECK(pipeHostWaitListAdd(waits, target.peerWorldRank, 0, 0, 1, params->myRank));
  }
  return ncclSuccess;
}

static ncclResult_t pipeHostPhasedDestSourceWait(const StagingKernelParams* params, const StagingPipeDevicePlan* plan,
                                                 PipeHostCommId commId, PipeHostWaitList* waits, size_t round,
                                                 int channel, int sourceIndex) {
  if (!params->isDest || channel < 0 || channel >= params->numChannels || sourceIndex < 0 ||
      sourceIndex >= params->numRdmaSources) {
    return ncclSuccess;
  }
  const StagingPipePeerEdge& source = plan->rdmaSources[channel][sourceIndex];
  if (!pipeHostEdgeUsesComm(source, commId)) {
    return ncclSuccess;
  }
  PipeHostChunk chunk;
  NCCL_M2N_CHECK(pipeHostChunkForRound(source, plan->rdmaSourceLayouts[channel][sourceIndex], round, params->myRank,
                                       &chunk));
  if (chunk.active) {
    NCCL_M2N_CHECK(pipeHostWaitListAdd(waits, source.peerWorldRank, 0, 0, 1, params->myRank));
  }
  return ncclSuccess;
}

static ncclResult_t pipeHostPhasedDestRootCopy(const StagingKernelParams* params, const StagingPipeDevicePlan* plan,
                                               const StagingPipeCallParams* call, void* stagingBuffer,
                                               cudaStream_t stream, size_t round, int channel, int sourceIndex) {
  if (!params->isDest || channel < 0 || channel >= params->numChannels || sourceIndex < 0 ||
      sourceIndex >= params->numRdmaSources) {
    return ncclSuccess;
  }
  const StagingPipePeerEdge& source = plan->rdmaSources[channel][sourceIndex];
  PipeHostChunk chunk;
  NCCL_M2N_CHECK(pipeHostChunkForRound(source, plan->rdmaSourceLayouts[channel][sourceIndex], round, params->myRank,
                                       &chunk));
  if (!chunk.active) {
    return ncclSuccess;
  }
  const size_t slot = round % static_cast<size_t>(source.fc.peerNumSlots);
  size_t recvOffset = 0;
  NCCL_M2N_CHECK(pipeHostSlotOffset(source.fc, slot, chunk.bytes, params->myRank, &recvOffset));
  NCCL_M2N_CHECK(pipeHostEnqueueContigToLayout(call->dstBuffer, static_cast<char*>(stagingBuffer) + recvOffset,
                                               plan->rdmaSourceLayouts[channel][sourceIndex], chunk.byteStart,
                                               chunk.bytes, stream, params->myRank, &chunk));
  return ncclSuccess;
}

static ncclResult_t pipeHostPhasedRingForward(const StagingKernelParams* params, const StagingPipeDevicePlan* plan,
                                              void* stagingBuffer, const PipeHostPipeline* pipeline,
                                              bool waitForSlotCredit, const PipeHostRmaContext* rma,
                                              PipeHostCommId commId, cudaStream_t stream, size_t round, int channel,
                                              int sourceIndex) {
  if (!params->isDest || params->numRingTargets <= 0 || channel < 0 || channel >= params->numChannels ||
      sourceIndex < 0 || sourceIndex >= params->numRdmaSources) {
    return ncclSuccess;
  }
  const PipeHostComm* hostComm = pipeHostCommForId(rma, commId, params->myRank);
  if (hostComm == nullptr) {
    return ncclSuccess;
  }
  const int ringStart = pipeHostRingTargetStart(params);
  const StagingPipePeerEdge& source = plan->rdmaSources[channel][sourceIndex];
  PipeHostChunk sourceChunk;
  NCCL_M2N_CHECK(pipeHostChunkForRound(source, plan->rdmaSourceLayouts[channel][sourceIndex], round, params->myRank,
                                       &sourceChunk));
  if (!sourceChunk.active) {
    return ncclSuccess;
  }

  const int ringIdx = ringStart + sourceIndex;
  if (ringIdx < ringStart || ringIdx >= params->numRdmaTargets) {
    return ncclSuccess;
  }
  const StagingPipePeerEdge& target = plan->rdmaTargets[channel][ringIdx].peer;
  if (!pipeHostEdgeUsesComm(target, commId)) {
    return ncclSuccess;
  }
  PipeHostChunk targetChunk;
  NCCL_M2N_CHECK(pipeHostChunkForRound(target, plan->rdmaTargetLayouts[channel][target.copyLayoutIndex], round,
                                       params->myRank, &targetChunk));
  NCCL_M2N_CHECK_ARG(targetChunk.active && targetChunk.byteStart == sourceChunk.byteStart &&
                       targetChunk.bytes == sourceChunk.bytes,
                     params->myRank,
                     "PIPE host_rma: ring target/source chunk mismatch "
                     "(ch=%d source=%d target=%d source_active=%d target_active=%d source_start=%zu "
                     "target_start=%zu source_bytes=%zu target_bytes=%zu)",
                     channel, sourceIndex, ringIdx, (int)sourceChunk.active, (int)targetChunk.active,
                     sourceChunk.byteStart, targetChunk.byteStart, sourceChunk.bytes, targetChunk.bytes);

  const size_t localSlot = round % static_cast<size_t>(source.fc.peerNumSlots);
  const size_t remoteSlot = round % static_cast<size_t>(target.fc.peerNumSlots);
  size_t localOffset = 0;
  size_t remoteOffset = 0;
  NCCL_M2N_CHECK(pipeHostSlotOffset(source.fc, localSlot, sourceChunk.bytes, params->myRank, &localOffset));
  NCCL_M2N_CHECK(pipeHostSlotOffset(target.fc, remoteSlot, sourceChunk.bytes, params->myRank, &remoteOffset));
  const size_t reuseDistance = static_cast<size_t>(target.fc.peerNumSlots);
  if (waitForSlotCredit && pipeline != nullptr && round >= reuseDistance) {
    NCCL_M2N_CHECK(pipeline->waitTargetEdge(PIPE_HOST_STREAM_RING_PUT, PIPE_HOST_EVENT_TARGET_CREDIT_DONE,
                                            round - reuseDistance, channel, ringIdx, params->myRank,
                                            "ring_forward.slot_credit"));
  }

  ncclWindow_t targetWindow = nullptr;
  NCCL_M2N_CHECK(pipeHostRdmaWindow(params, target, params->myRank, &targetWindow));
  NCCL_M2N_CHECK(pipeHostPutSignal(hostComm->comm, static_cast<char*>(stagingBuffer) + localOffset, sourceChunk.bytes,
                                   target.peerWorldRank, targetWindow, remoteOffset, stream, params->myRank,
                                   "ring.put_data"));
  return ncclSuccess;
}

static ncclResult_t pipeHostPhasedRootSignalFollowers(const StagingKernelParams* params,
                                                      const StagingPipeDevicePlan* plan, const PipeHostRmaContext* rma,
                                                      PipeHostCommId commId, cudaStream_t stream, size_t round,
                                                      int channel, int sourceIndex) {
  if (!params->isDest || params->numLsaFollowers <= 0 || channel < 0 || channel >= params->numChannels ||
      sourceIndex < 0 || sourceIndex >= params->numRdmaSources) {
    return ncclSuccess;
  }
  const PipeHostComm* hostComm = pipeHostCommForId(rma, commId, params->myRank);
  if (hostComm == nullptr) {
    return ncclSuccess;
  }
  const StagingPipePeerEdge& source = plan->rdmaSources[channel][sourceIndex];
  PipeHostChunk chunk;
  NCCL_M2N_CHECK(pipeHostChunkForRound(source, plan->rdmaSourceLayouts[channel][sourceIndex], round, params->myRank,
                                       &chunk));
  if (!chunk.active) {
    return ncclSuccess;
  }
  const int fwdStart = sourceIndex * params->numLsaFollowers;
  for (int f = 0; f < params->numLsaFollowers; f++) {
    const int targetIdx = fwdStart + f;
    if (targetIdx >= params->numLsaTargets) {
      continue;
    }
    const StagingPipePeerEdge& target = plan->lsaTargets[channel][targetIdx];
    if (pipeHostEdgeUsesComm(target, commId)) {
      NCCL_M2N_CHECK(pipeHostSignal(hostComm->comm, target.peerWorldRank, stream, params->myRank,
                                    "root.signal_lsa_follower"));
    }
  }
  return ncclSuccess;
}

static ncclResult_t pipeHostPhasedFollowerWaitRoot(const StagingKernelParams* params, const StagingPipeDevicePlan* plan,
                                                   PipeHostCommId commId, PipeHostWaitList* waits, size_t round,
                                                   int channel, int sourceIndex) {
  if (!params->isDest || channel < 0 || channel >= params->numChannels || sourceIndex < 0 ||
      sourceIndex >= params->numLsaSources) {
    return ncclSuccess;
  }
  const StagingPipePeerEdge& source = plan->lsaSources[channel][sourceIndex];
  if (!pipeHostEdgeUsesComm(source, commId)) {
    return ncclSuccess;
  }
  PipeHostChunk chunk;
  NCCL_M2N_CHECK(pipeHostChunkForRound(source, plan->lsaSourceLayouts[channel][sourceIndex], round, params->myRank,
                                       &chunk));
  if (chunk.active) {
    NCCL_M2N_CHECK(pipeHostWaitListAdd(waits, source.peerWorldRank, 0, 0, 1, params->myRank));
  }
  return ncclSuccess;
}

static ncclResult_t pipeHostPhasedFollowerDirectCopy(const StagingKernelParams* params,
                                                     const StagingPipeDevicePlan* plan,
                                                     const StagingPipeCallParams* call, cudaStream_t stream,
                                                     size_t round, int channel, int sourceIndex) {
  if (!params->isDest || channel < 0 || channel >= params->numChannels || sourceIndex < 0 ||
      sourceIndex >= params->numLsaSources) {
    return ncclSuccess;
  }
  NCCL_M2N_CHECK_ARG(params->lsaWindow != nullptr, params->myRank, "PIPE host_rma: LSA window is null");
  const StagingPipePeerEdge& source = plan->lsaSources[channel][sourceIndex];
  PipeHostChunk chunk;
  NCCL_M2N_CHECK(pipeHostChunkForRound(source, plan->lsaSourceLayouts[channel][sourceIndex], round, params->myRank,
                                       &chunk));
  if (!chunk.active) {
    return ncclSuccess;
  }
  const size_t slot = round % static_cast<size_t>(source.fc.peerNumSlots);
  size_t remoteOffset = 0;
  NCCL_M2N_CHECK(pipeHostSlotOffset(source.fc, slot, chunk.bytes, params->myRank, &remoteOffset));
  void* remoteSrc = nullptr;
  NCCL_M2N_CHECK(ncclGetLsaDevicePointer(params->lsaWindow, remoteOffset, source.peerLocalRank, &remoteSrc));
  const StagingPipeCopyLayout& layout = plan->lsaSourceLayouts[channel][sourceIndex];
  return pipeHostEnqueueContigToLayout(call->dstBuffer, remoteSrc, layout, chunk.byteStart, chunk.bytes, stream,
                                       params->myRank, &chunk);
}

static ncclResult_t pipeHostPhasedFollowerDoneSignals(const StagingKernelParams* params,
                                                      const StagingPipeDevicePlan* plan, const PipeHostRmaContext* rma,
                                                      PipeHostCommId commId, cudaStream_t stream, size_t round,
                                                      int channel, int sourceIndex) {
  if (!params->isDest || channel < 0 || channel >= params->numChannels || sourceIndex < 0 ||
      sourceIndex >= params->numLsaSources) {
    return ncclSuccess;
  }
  const PipeHostComm* hostComm = pipeHostCommForId(rma, commId, params->myRank);
  if (hostComm == nullptr) {
    return ncclSuccess;
  }
  const StagingPipePeerEdge& source = plan->lsaSources[channel][sourceIndex];
  if (!pipeHostEdgeUsesComm(source, commId)) {
    return ncclSuccess;
  }
  PipeHostChunk chunk;
  NCCL_M2N_CHECK(pipeHostChunkForRound(source, plan->lsaSourceLayouts[channel][sourceIndex], round, params->myRank,
                                       &chunk));
  if (chunk.active) {
    NCCL_M2N_CHECK(pipeHostSignal(hostComm->comm, source.peerWorldRank, stream, params->myRank,
                                  "follower.signal_root"));
  }
  return ncclSuccess;
}

static ncclResult_t pipeHostPhasedRootFollowerWaits(const StagingKernelParams* params,
                                                    const StagingPipeDevicePlan* plan, PipeHostCommId commId,
                                                    PipeHostWaitList* waits, size_t round, int channel,
                                                    int sourceIndex) {
  if (!params->isDest || params->numLsaFollowers <= 0 || channel < 0 || channel >= params->numChannels ||
      sourceIndex < 0 || sourceIndex >= params->numRdmaSources) {
    return ncclSuccess;
  }
  const StagingPipePeerEdge& source = plan->rdmaSources[channel][sourceIndex];
  PipeHostChunk chunk;
  NCCL_M2N_CHECK(pipeHostChunkForRound(source, plan->rdmaSourceLayouts[channel][sourceIndex], round, params->myRank,
                                       &chunk));
  if (!chunk.active) {
    return ncclSuccess;
  }
  const int fwdStart = sourceIndex * params->numLsaFollowers;
  for (int f = 0; f < params->numLsaFollowers; f++) {
    const int targetIdx = fwdStart + f;
    if (targetIdx >= params->numLsaTargets) {
      continue;
    }
    const StagingPipePeerEdge& target = plan->lsaTargets[channel][targetIdx];
    if (pipeHostEdgeUsesComm(target, commId)) {
      NCCL_M2N_CHECK(pipeHostWaitListAdd(waits, target.peerWorldRank, 0, 0, 1, params->myRank));
    }
  }
  return ncclSuccess;
}

static ncclResult_t pipeHostPhasedDestCreditSignals(const StagingKernelParams* params,
                                                    const StagingPipeDevicePlan* plan, const PipeHostRmaContext* rma,
                                                    PipeHostCommId commId, cudaStream_t stream, size_t round,
                                                    int channel, int sourceIndex) {
  if (!params->isDest || channel < 0 || channel >= params->numChannels || sourceIndex < 0 ||
      sourceIndex >= params->numRdmaSources) {
    return ncclSuccess;
  }
  const PipeHostComm* hostComm = pipeHostCommForId(rma, commId, params->myRank);
  if (hostComm == nullptr) {
    return ncclSuccess;
  }
  const StagingPipePeerEdge& source = plan->rdmaSources[channel][sourceIndex];
  if (!pipeHostEdgeUsesComm(source, commId)) {
    return ncclSuccess;
  }
  PipeHostChunk chunk;
  NCCL_M2N_CHECK(pipeHostChunkForRound(source, plan->rdmaSourceLayouts[channel][sourceIndex], round, params->myRank,
                                       &chunk));
  if (chunk.active) {
    NCCL_M2N_CHECK(pipeHostSignal(hostComm->comm, source.peerWorldRank, stream, params->myRank,
                                  "root.signal_source_credit"));
  }
  return ncclSuccess;
}

static ncclResult_t launchStagingReshardPipeHostRma(
  const StagingKernelParams* hostParams, const StagingPipeDevicePlan* hostPlan, StagingBufferState* stagingState,
  const StagingPipeCallParams* call, void* stagingBuffer, ncclComm_t parentComm, const ReshardSplitComms* splitComms,
  cudaStream_t stream, bool verbose) {
  NCCL_M2N_CHECK_ARG(
    hostParams != nullptr && hostPlan != nullptr && stagingState != nullptr && call != nullptr &&
      stagingBuffer != nullptr,
    -1, "PIPE host_rma: host params, host plan, staging state, call params, and staging buffer are required");
  if (verbose) {
    printf("[STAGING_PIPE_HOST_RMA] rank=%d channels=%d source=%d dest=%d "
           "rdma_targets=%d rdma_sources=%d lsa_targets=%d lsa_sources=%d\n",
           hostParams->myRank, hostParams->numChannels, (int)hostParams->isSource, (int)hostParams->isDest,
           hostParams->numRdmaTargets, hostParams->numRdmaSources, hostParams->numLsaTargets,
           hostParams->numLsaSources);
    fflush(stdout);
  }

  PipeHostRmaContext rma;
  NCCL_M2N_CHECK(pipeHostBuildRmaContext(parentComm, splitComms, hostParams->myRank, &rma));

  if (call->maxChunkRounds == 0) {
    return ncclSuccess;
  }

  /* Stage enablement is role-derived from the shared PIPE plan.  Host-RMA
   * does not create a separate graph; it only substitutes host NCCL RMA and CE
   * copies for the device trainer/generator work at each active stage. */
  const bool sourceRdmaStage = hostParams->isSource && pipeHostTrainerTargetEnd(hostParams) > 0;
  const bool destRdmaStage = hostParams->isDest && hostParams->numRdmaSources > 0;
  const bool ringStage = destRdmaStage && hostParams->numRingTargets > 0;
  const bool rootFanoutStage =
    call->hasLocalFanout && hostParams->isDest && hostParams->numRdmaSources > 0 && hostParams->numLsaFollowers > 0;
  const bool followerStage = call->hasLocalFanout && hostParams->isDest && hostParams->numLsaSources > 0;
  const bool creditSignalStage = destRdmaStage;
  const bool creditWaitStage = sourceRdmaStage || ringStage;
  uint64_t streamMask = 0;
  uint64_t eventMask = 0;
  auto useStream = [&](PipeHostStreamId id) { streamMask |= pipeHostMaskBit(id); };
  auto useEvent = [&](PipeHostEventId id) { eventMask |= pipeHostMaskBit(id); };
  if (sourceRdmaStage) {
    useStream(PIPE_HOST_STREAM_PACK);
    useStream(PIPE_HOST_STREAM_SOURCE_PUT);
    useEvent(PIPE_HOST_EVENT_TARGET_PACK_DONE);
    useEvent(PIPE_HOST_EVENT_TARGET_PUT_DONE);
  }
  if (destRdmaStage) {
    useEvent(PIPE_HOST_EVENT_DEST_EDGE_READY);
    useStream(PIPE_HOST_STREAM_ROOT_COPY);
    useEvent(PIPE_HOST_EVENT_ROOT_COPY_DONE);
  }
  if (ringStage) {
    useStream(PIPE_HOST_STREAM_RING_PUT);
    useEvent(PIPE_HOST_EVENT_RING_PUT_DONE);
    useEvent(PIPE_HOST_EVENT_TARGET_PUT_DONE);
  }
  if (rootFanoutStage) {
    useStream(PIPE_HOST_STREAM_ROOT_SIGNAL);
    useStream(PIPE_HOST_STREAM_ROOT_WAIT);
    useEvent(PIPE_HOST_EVENT_ROOT_SIGNAL_DONE);
    useEvent(PIPE_HOST_EVENT_ROOT_FOLLOWERS_DONE);
  }
  if (followerStage) {
    useStream(PIPE_HOST_STREAM_FOLLOWER_WAIT);
    useStream(PIPE_HOST_STREAM_FOLLOWER_UNPACK);
    useStream(PIPE_HOST_STREAM_FOLLOWER_SIGNAL);
    useEvent(PIPE_HOST_EVENT_FOLLOWER_READY);
    useEvent(PIPE_HOST_EVENT_FOLLOWER_COPY_DONE);
  }
  if (creditSignalStage) {
    useStream(PIPE_HOST_STREAM_CREDIT_SIGNAL);
  }
  if (creditWaitStage) {
    useStream(PIPE_HOST_STREAM_CREDIT_WAIT);
    useEvent(PIPE_HOST_EVENT_TARGET_CREDIT_DONE);
  }
  if (streamMask == 0) {
    return ncclSuccess;
  }

  const bool multiRoundPipeline = call->maxChunkRounds > 1;
  /* A source lane owns one peer-group FIFO. Give each lane independent CE
   * pack/PutSignal streams so completed peers can inject before unrelated
   * peers finish packing. This also enables useful overlap for one-round
   * transfers that otherwise collapse onto the caller stream. */
  const bool channelSourcePipeline = sourceRdmaStage && hostParams->numChannels > 1;
  /* Host WaitSignal calls share one signal FIFO per source peer. Keep that
   * order, but allow independent inbound peers to progress on separate
   * persistent streams. */
  const bool destWaitPipeline = destRdmaStage && hostParams->numRdmaSources > 1;
  const bool ringEdgePipeline = ringStage && (hostParams->numChannels > 1 || hostParams->numRdmaSources > 1);
  const bool enablePipeline = multiRoundPipeline || channelSourcePipeline || destWaitPipeline || ringEdgePipeline;
  if (stagingState->hostRmaPipeline == nullptr) {
    stagingState->hostRmaPipeline = new (std::nothrow) PipeHostPipeline;
    NCCL_M2N_CHECK_ARG(stagingState->hostRmaPipeline != nullptr, hostParams->myRank,
                       "PIPE host_rma: failed to allocate pipeline cache");
  }
  PipeHostPipeline& pipeline = *static_cast<PipeHostPipeline*>(stagingState->hostRmaPipeline);
  NCCL_M2N_CHECK(pipeline.ensure(call->maxChunkRounds, hostParams->numChannels, hostParams->numRdmaTargets,
                                 hostParams->numRdmaSources, hostParams->numLsaSources, streamMask, eventMask,
                                 enablePipeline, channelSourcePipeline, destWaitPipeline));
  NCCL_M2N_CHECK(pipeline.begin(stream));

  const int rank = hostParams->myRank;
  auto streamFor = [&](PipeHostStreamId streamId) { return pipeline.stream(streamId); };
  auto streamForChannel = [&](PipeHostStreamId streamId, int channel) {
    return pipeline.channelStream(streamId, channel);
  };
  auto forwardRingEdge = [&](size_t round, int channel, int sourceIndex) {
    NCCL_M2N_CHECK(pipeline.waitDestEdge(PIPE_HOST_STREAM_RING_PUT, PIPE_HOST_EVENT_DEST_EDGE_READY, round, channel,
                                         sourceIndex, rank, "ring_put.wait_source_edge"));
    for (PipeHostCommId commId : kPipeHostCommIds) {
      NCCL_M2N_CHECK(pipeHostRunPutSignalPhase(
        rma.comms[commId], rank, streamFor(PIPE_HOST_STREAM_RING_PUT), "ring_data_put", [&]() {
          return pipeHostPhasedRingForward(hostParams, hostPlan, stagingBuffer, &pipeline, creditWaitStage, &rma,
                                           commId, streamFor(PIPE_HOST_STREAM_RING_PUT), round, channel, sourceIndex);
        }));
    }
    NCCL_M2N_CHECK(pipeline.recordDestEdge(PIPE_HOST_EVENT_RING_PUT_DONE, PIPE_HOST_STREAM_RING_PUT, round, channel,
                                           sourceIndex, rank, "ring_put_done"));
    const int targetIndex = pipeHostRingTargetStart(hostParams) + sourceIndex;
    if (targetIndex < hostParams->numRdmaTargets) {
      NCCL_M2N_CHECK(pipeline.recordTargetEdge(PIPE_HOST_EVENT_TARGET_PUT_DONE, PIPE_HOST_STREAM_RING_PUT, round,
                                               channel, targetIndex, rank, "ring_target_put_done"));
    }
    return ncclSuccess;
  };

  /* Main per-round schedule.  RMA phase order is expressed by ncclWaitSignal
   * op counts, and same-rank CE/RMA dependencies are expressed by CUDA events. */
  for (size_t round = 0; round < call->maxChunkRounds; round++) {
    if (sourceRdmaStage) {
      for (int ch = 0; ch < hostParams->numChannels; ch++) {
        NCCL_M2N_CHECK(pipeHostPhasedSourcePacks(hostParams, hostPlan, call, stagingBuffer, &pipeline, creditWaitStage,
                                                 streamForChannel(PIPE_HOST_STREAM_PACK, ch), round, ch));
        for (PipeHostCommId commId : kPipeHostCommIds) {
          PipeHostPutCompletionList completions;
          NCCL_M2N_CHECK(pipeHostRunPutSignalPhase(
            rma.comms[commId], rank, streamFor(PIPE_HOST_STREAM_SOURCE_PUT), "source_data_put", [&]() {
              return pipeHostPhasedSourcePuts(hostParams, hostPlan, stagingBuffer, &rma, &pipeline, commId, round,
                                              &completions, ch);
            }));
          for (int i = 0; i < completions.count; i++) {
            NCCL_M2N_CHECK(pipeline.recordTargetEdgeOnStream(PIPE_HOST_EVENT_TARGET_PUT_DONE,
                                                             completions.entries[i].stream, round, ch,
                                                             completions.entries[i].target, rank, "source_put_done"));
          }
        }
      }
    }

    if (destRdmaStage) {
      for (int ch = 0; ch < hostParams->numChannels; ch++) {
        for (int s = 0; s < hostParams->numRdmaSources; s++) {
          cudaStream_t destWaitStream = nullptr;
          NCCL_M2N_CHECK(pipeline.destWaitStream(s, rank, &destWaitStream));
          for (PipeHostCommId commId : kPipeHostCommIds) {
            NCCL_M2N_CHECK(pipeHostRunWaitPhase(
              rma.comms[commId], rank, destWaitStream, "dest_wait_source_edge", [&](PipeHostWaitList* waits) {
                return pipeHostPhasedDestSourceWait(hostParams, hostPlan, commId, waits, round, ch, s);
              }));
          }
          NCCL_M2N_CHECK(pipeline.recordDestEdgeOnStream(PIPE_HOST_EVENT_DEST_EDGE_READY, destWaitStream, round, ch, s,
                                                         rank, "dest_source_edge_ready"));
          if (ringStage) {
            NCCL_M2N_CHECK(forwardRingEdge(round, ch, s));
          }
        }
      }
    }

    if (destRdmaStage) {
      for (int ch = 0; ch < hostParams->numChannels; ch++) {
        for (int s = 0; s < hostParams->numRdmaSources; s++) {
          NCCL_M2N_CHECK(pipeline.waitDestEdge(PIPE_HOST_STREAM_ROOT_COPY, PIPE_HOST_EVENT_DEST_EDGE_READY, round, ch,
                                               s, rank, "root_copy.wait_source"));
          NCCL_M2N_CHECK(pipeHostPhasedDestRootCopy(hostParams, hostPlan, call, stagingBuffer,
                                                    streamFor(PIPE_HOST_STREAM_ROOT_COPY), round, ch, s));
          NCCL_M2N_CHECK(pipeline.recordDestEdge(PIPE_HOST_EVENT_ROOT_COPY_DONE, PIPE_HOST_STREAM_ROOT_COPY, round, ch,
                                                 s, rank, "root_copy_done"));
        }
      }
    }

    if (rootFanoutStage) {
      for (int ch = 0; ch < hostParams->numChannels; ch++) {
        for (int s = 0; s < hostParams->numRdmaSources; s++) {
          NCCL_M2N_CHECK(pipeline.waitDestEdge(PIPE_HOST_STREAM_ROOT_SIGNAL, PIPE_HOST_EVENT_DEST_EDGE_READY, round, ch,
                                               s, rank, "root_signal.wait_source"));
          for (PipeHostCommId commId : kPipeHostCommIds) {
            NCCL_M2N_CHECK(pipeHostRunPutSignalPhase(
              rma.comms[commId], rank, streamFor(PIPE_HOST_STREAM_ROOT_SIGNAL), "root_signal_followers", [&]() {
                return pipeHostPhasedRootSignalFollowers(hostParams, hostPlan, &rma, commId,
                                                         streamFor(PIPE_HOST_STREAM_ROOT_SIGNAL), round, ch, s);
              }));
          }
          NCCL_M2N_CHECK(pipeline.recordDestEdge(PIPE_HOST_EVENT_ROOT_SIGNAL_DONE, PIPE_HOST_STREAM_ROOT_SIGNAL, round,
                                                 ch, s, rank, "root_signal_done"));
        }
      }
    }

    if (followerStage) {
      for (int ch = 0; ch < hostParams->numChannels; ch++) {
        for (int s = 0; s < hostParams->numLsaSources; s++) {
          for (PipeHostCommId commId : kPipeHostCommIds) {
            NCCL_M2N_CHECK(pipeHostRunWaitPhase(rma.comms[commId], rank, streamFor(PIPE_HOST_STREAM_FOLLOWER_WAIT),
                                                "follower_wait_root", [&](PipeHostWaitList* waits) {
                                                  return pipeHostPhasedFollowerWaitRoot(hostParams, hostPlan, commId,
                                                                                        waits, round, ch, s);
                                                }));
          }
          NCCL_M2N_CHECK(pipeline.recordFollowerEdge(PIPE_HOST_EVENT_FOLLOWER_READY, PIPE_HOST_STREAM_FOLLOWER_WAIT,
                                                     round, ch, s, rank, "follower_ready"));
          NCCL_M2N_CHECK(pipeline.waitFollowerEdge(PIPE_HOST_STREAM_FOLLOWER_UNPACK, PIPE_HOST_EVENT_FOLLOWER_READY,
                                                   round, ch, s, rank, "follower_unpack.wait_ready"));
          NCCL_M2N_CHECK(pipeHostPhasedFollowerDirectCopy(hostParams, hostPlan, call,
                                                          streamFor(PIPE_HOST_STREAM_FOLLOWER_UNPACK), round, ch, s));
          NCCL_M2N_CHECK(pipeline.recordFollowerEdge(PIPE_HOST_EVENT_FOLLOWER_COPY_DONE,
                                                     PIPE_HOST_STREAM_FOLLOWER_UNPACK, round, ch, s, rank,
                                                     "follower_copy_done"));
          NCCL_M2N_CHECK(pipeline.waitFollowerEdge(PIPE_HOST_STREAM_FOLLOWER_SIGNAL, PIPE_HOST_EVENT_FOLLOWER_COPY_DONE,
                                                   round, ch, s, rank, "follower_signal.wait_copy"));
          for (PipeHostCommId commId : kPipeHostCommIds) {
            NCCL_M2N_CHECK(pipeHostRunPutSignalPhase(
              rma.comms[commId], rank, streamFor(PIPE_HOST_STREAM_FOLLOWER_SIGNAL), "follower_signal_root", [&]() {
                return pipeHostPhasedFollowerDoneSignals(hostParams, hostPlan, &rma, commId,
                                                         streamFor(PIPE_HOST_STREAM_FOLLOWER_SIGNAL), round, ch, s);
              }));
          }
        }
      }
    }

    if (rootFanoutStage) {
      for (int ch = 0; ch < hostParams->numChannels; ch++) {
        for (int s = 0; s < hostParams->numRdmaSources; s++) {
          NCCL_M2N_CHECK(pipeline.waitDestEdge(PIPE_HOST_STREAM_ROOT_WAIT, PIPE_HOST_EVENT_ROOT_SIGNAL_DONE, round, ch,
                                               s, rank, "root_wait_followers.wait_root_signal"));
          for (PipeHostCommId commId : kPipeHostCommIds) {
            NCCL_M2N_CHECK(pipeHostRunWaitPhase(rma.comms[commId], rank, streamFor(PIPE_HOST_STREAM_ROOT_WAIT),
                                                "root_wait_followers", [&](PipeHostWaitList* waits) {
                                                  return pipeHostPhasedRootFollowerWaits(hostParams, hostPlan, commId,
                                                                                         waits, round, ch, s);
                                                }));
          }
          NCCL_M2N_CHECK(pipeline.recordDestEdge(PIPE_HOST_EVENT_ROOT_FOLLOWERS_DONE, PIPE_HOST_STREAM_ROOT_WAIT, round,
                                                 ch, s, rank, "root_followers_done"));
        }
      }
    }

    if (creditSignalStage) {
      for (int ch = 0; ch < hostParams->numChannels; ch++) {
        for (int s = 0; s < hostParams->numRdmaSources; s++) {
          NCCL_M2N_CHECK(pipeline.waitDestEdge(PIPE_HOST_STREAM_CREDIT_SIGNAL, PIPE_HOST_EVENT_ROOT_COPY_DONE, round,
                                               ch, s, rank, "source_credit.wait_local"));
          if (ringStage) {
            NCCL_M2N_CHECK(pipeline.waitDestEdge(PIPE_HOST_STREAM_CREDIT_SIGNAL, PIPE_HOST_EVENT_RING_PUT_DONE, round,
                                                 ch, s, rank, "source_credit.wait_ring_put"));
          }
          if (rootFanoutStage) {
            NCCL_M2N_CHECK(pipeline.waitDestEdge(PIPE_HOST_STREAM_CREDIT_SIGNAL, PIPE_HOST_EVENT_ROOT_FOLLOWERS_DONE,
                                                 round, ch, s, rank, "source_credit.wait_followers"));
          }
          for (PipeHostCommId commId : kPipeHostCommIds) {
            NCCL_M2N_CHECK(pipeHostRunPutSignalPhase(
              rma.comms[commId], rank, streamFor(PIPE_HOST_STREAM_CREDIT_SIGNAL), "source_credit_signal", [&]() {
                return pipeHostPhasedDestCreditSignals(hostParams, hostPlan, &rma, commId,
                                                       streamFor(PIPE_HOST_STREAM_CREDIT_SIGNAL), round, ch, s);
              }));
          }
        }
      }
    }

    if (creditWaitStage) {
      for (int ch = 0; ch < hostParams->numChannels; ch++) {
        for (int t = 0; t < hostParams->numRdmaTargets; t++) {
          bool active = false;
          NCCL_M2N_CHECK(pipeHostSourceCreditActive(hostParams, hostPlan, round, ch, t, &active));
          if (!active) {
            continue;
          }
          NCCL_M2N_CHECK(pipeline.waitTargetEdge(PIPE_HOST_STREAM_CREDIT_WAIT, PIPE_HOST_EVENT_TARGET_PUT_DONE, round,
                                                 ch, t, rank, "source_wait_credit.wait_put"));
          for (PipeHostCommId commId : kPipeHostCommIds) {
            NCCL_M2N_CHECK(pipeHostRunWaitPhase(rma.comms[commId], rank, streamFor(PIPE_HOST_STREAM_CREDIT_WAIT),
                                                "source_wait_credit", [&](PipeHostWaitList* waits) {
                                                  return pipeHostPhasedSourceCreditWait(hostParams, hostPlan, commId,
                                                                                        waits, round, ch, t);
                                                }));
          }
          NCCL_M2N_CHECK(pipeline.recordTargetEdge(PIPE_HOST_EVENT_TARGET_CREDIT_DONE, PIPE_HOST_STREAM_CREDIT_WAIT,
                                                   round, ch, t, rank, "source_credit_done"));
        }
      }
    }
  }

  NCCL_M2N_CHECK(pipeline.join());
  return ncclSuccess;
}

class StagingBufferEventGuard {
public:
  StagingBufferEventGuard(ncclComm_t comm, uint64_t poolKey, cudaStream_t stream)
    : comm_(comm), poolKey_(poolKey), stream_(stream) {}
  StagingBufferEventGuard(const StagingBufferEventGuard&) = delete;
  StagingBufferEventGuard& operator=(const StagingBufferEventGuard&) = delete;
  ~StagingBufferEventGuard() {
    if (bActive_) {
      NCCL_M2N_CHECK_WARN(complete());
    }
  }

  ncclResult_t complete() {
    if (!bActive_) {
      return ncclSuccess;
    }
    bActive_ = false;
    return stagingBufferPoolRecordEvent(comm_, poolKey_, stream_);
  }

private:
  ncclComm_t comm_;
  uint64_t poolKey_;
  cudaStream_t stream_;
  bool bActive_ = true;
};

/* ======================================================================
 * ncclReshard — copy/staging-based public entry.
 * ====================================================================*/

extern "C" ncclResult_t ncclReshard(ncclM2nHandle_t handle, ncclComm_t comm, const ncclDistTensor_t* src,
                                    const ncclDistTensor_t* dst, cudaStream_t stream) {
  if (m2nGroupIsActive()) {
    return m2nGroupEnqueueReshard(handle, comm, src, dst, stream);
  }
  M2nApiLock apiLock;
  m2nClearLastError();
  NCCL_M2N_CHECK_ARG(comm != nullptr, -1, "ncclReshard: comm must be non-null");
  NCCL_M2N_CHECK_ARG(src != nullptr && dst != nullptr, -1,
                     "ncclReshard: src and dst tensor descriptors must both be non-null on every rank");
  ReshardTensorSetup tensorSetup;
  NCCL_M2N_CHECK(reshardPrepareTensorSetup("ncclReshard", src, dst, &tensorSetup));
  const int ndims = tensorSetup.ndims;
  const size_t element_size = tensorSetup.elementSize;
  void* const srcBuffer = tensorSetup.srcTensor.dataPtr;
  void* const dstBuffer = tensorSetup.dstTensor.dataPtr;
  const size_t* const src_tensor_dims = tensorSetup.srcTensor.localShape;
  const size_t* const dst_tensor_dims = tensorSetup.dstTensor.localShape;
  ncclDistTensor_t& src_local = tensorSetup.srcTensor;
  ncclDistTensor_t& dst_local = tensorSetup.dstTensor;
  const ncclMesh_t* const src_mesh = &tensorSetup.srcMesh;
  const ncclMesh_t* const dst_mesh = &tensorSetup.dstMesh;
  const ncclDistTensor_t* const srcTensor = &src_local;
  const ncclDistTensor_t* const dstTensor = &dst_local;
  const ncclMesh_t* const srcMesh = src_mesh;
  const ncclMesh_t* const dstMesh = dst_mesh;
  std::shared_ptr<ncclM2nHandleState> handleState;
  NCCL_M2N_CHECK(acquireM2nHandle(handle, &handleState));
  /* A caller comm created with ncclConfig_t.blocking=0 may still be
   * initializing; drive it to readiness before the first communicator query.
   * Run under M2nApiUnlock so a peer rank in this process can progress it. */
  {
    M2nApiUnlock apiUnlock;
    NCCL_M2N_CHECK(m2nWaitCommReady(comm));
  }
  const ReshardCopyAlgorithm copyAlgo = reshardGetCopyAlgorithm();
  int parentCommSize = 0;
  NCCL_M2N_CHECK(ncclCommCount(comm, &parentCommSize));
  const bool pipeSplitCandidate = copyAlgo == RESHARD_COPY_ALGO_PIPE && tensorRepCount(dstTensor) > 1;
  reshardResolveAdaptiveScaleConfig(parentCommSize, copyAlgo == RESHARD_COPY_ALGO_PACK || pipeSplitCandidate);

  int world_rank = 0, world_size = 0;
  NCCL_M2N_CHECK(ncclCommUserRank(comm, &world_rank));
  NCCL_M2N_CHECK(ncclCommCount(comm, &world_size));
  NCCL_M2N_CHECK(validateReshardMeshBounds(src_mesh, dst_mesh, world_size, world_rank));
  NCCL_M2N_CHECK(reshardValidateActiveBuffers("ncclReshard", world_rank, &src_local, &dst_local));

  auto check_shard_global_size = [&](const char* side, const ncclDistTensor_t* tensor,
                                     const size_t* dims) -> ncclResult_t {
    ncclReshardMeshGroupInfo info;
    computeMeshGroupInfo(tensor, tensor->mesh->startRank, &info);
    for (int d = 0; d < ndims; d++) {
      size_t globalDim = 0;
      NCCL_M2N_CHECK_ARG(m2nCheckedMulSize(dims[d], (size_t)info.shardLayout.shardCounts[d], &globalDim), world_rank,
                         "ncclReshard: %s shard dimension overflows global extent: dim=%d local=%zu shardCount=%d",
                         side, d, dims[d], info.shardLayout.shardCounts[d]);
    }
    return ncclSuccess;
  };
  NCCL_M2N_CHECK(check_shard_global_size("source", &src_local, src_tensor_dims));
  NCCL_M2N_CHECK(check_shard_global_size("destination", &dst_local, dst_tensor_dims));

  auto check_local_bytes = [&](const size_t* dims, const char* side) -> ncclResult_t {
    size_t total = element_size;
    for (int d = 0; d < ndims; d++) {
      NCCL_M2N_CHECK_ARG(m2nCheckedMulSize(total, dims[d], &total), world_rank,
                         "ncclReshard: %s local byte size overflow at dim %d: current=%zu dim=%zu", side, d, total,
                         dims[d]);
    }
    return ncclSuccess;
  };
  NCCL_M2N_CHECK(check_local_bytes(src_tensor_dims, "source"));
  NCCL_M2N_CHECK(check_local_bytes(dst_tensor_dims, "destination"));

  int currentCudaDev = 0;
  ncclCommProperties commProps = NCCL_COMM_PROPERTIES_INITIALIZER;
  ncclResult_t propsResult = ncclSuccess;
  NCCL_M2N_CHECK(reshardMatchCommCudaDevice(comm, &currentCudaDev, &commProps, &propsResult));
  NCCL_M2N_CHECK(reshardRejectGraphCapture("ncclReshard", stream));

  /* Stream pool for default-stream callers. */
  ReshardWorkStream work{};
  ncclResult_t setupResult = reshardSetupWorkStream(comm, stream, currentCudaDev, propsResult, &commProps, &work);
  if (setupResult != ncclSuccess) {
    return setupResult;
  }
  ReshardWorkStreamCompletion workCompletion(stream, &work);
  cudaStream_t workStream = work.stream;
  const uint64_t callEpoch = gStagingEpoch.fetch_add(1, std::memory_order_relaxed);
  const bool pipeHostRma =
    (copyAlgo == RESHARD_COPY_ALGO_PIPE) && (reshardGetPipeNetMode() == RESHARD_PIPE_NET_HOST_RMA);

  if (pipeHostRma) {
    if (!commProps.hostRmaSupport) {
      NCCL_M2N_FAIL(ncclInvalidUsage, world_rank,
                    "ncclReshard: PIPE host_rma requested but the parent communicator does not report host RMA "
                    "support");
    }
    /* Before 2.31.2, NCCL starts its collective host-RMA init on the first
     * put/signal; a rank with no PIPE edges never makes one, so peers hang. */
    int ncclVersion = 0;
    NCCL_M2N_CHECK(ncclGetVersion(&ncclVersion));
    if (ncclVersion < NCCL_VERSION(2, 31, 2)) {
      NCCL_M2N_FAIL(ncclInvalidUsage, world_rank,
                    "ncclReshard: PIPE host_rma requires NCCL 2.31.2 or newer; found %d.%d.%d",
                    ncclVersion / 10000, (ncclVersion % 10000) / 100, ncclVersion % 100);
    }
  }

  /* PACK: full per-dest-contiguous CE pack + hierarchical
   * user-window kernel + CE unpack.  Reuses the transpose/transfer
   * buffer and reshardKernelUserWindow; bypasses the chunk-ring
   * staging kernel below. */
  if (copyAlgo == RESHARD_COPY_ALGO_PACK) {
    NCCL_M2N_CHECK(reshardStartWorkStream(stream, &work));
    NCCL_M2N_CHECK(reshardCopyPackNormalized(comm, srcTensor, dstTensor, workStream));
    NCCL_M2N_CHECK(workCompletion.complete());
#ifdef NCCL_M2N_TESTING
    gLastCompletedCopyAlgorithm.store(RESHARD_COPY_ALGO_PACK, std::memory_order_relaxed);
#endif
    return ncclSuccess;
  }

  const bool debugLogging = reshardGetLogLevel() >= RESHARD_LOG_DEBUG;
  auto profile = debugLogging ? stagingProfileCreate() : std::unique_ptr<StagingProfile>{};

  /* Convert dims to bytes (last dim absorbs element size). */
  size_t src_dims_bytes[NCCL_RESHARD_MAX_TENSOR_DIMS] = {0};
  size_t dst_dims_bytes[NCCL_RESHARD_MAX_TENSOR_DIMS] = {0};
  NCCL_M2N_CHECK(reshardDimsToBytes(world_rank, "ncclReshard:", ndims, element_size, src_tensor_dims, dst_tensor_dims,
                                    src_dims_bytes, dst_dims_bytes));

  int lsa_size_from_comm = 0;
  const int gpus_per_node_cfg = reshardGetGpusPerNode();
  int src_gpus_per_domain = (gpus_per_node_cfg > 0) ? gpus_per_node_cfg : 1;
  int dst_gpus_per_domain = src_gpus_per_domain;
  int node_local_rank = world_rank % dst_gpus_per_domain;
  ncclDevComm probeDevComm{};
  ReshardDevCommUse probeDevCommUse;
  bool parentTopologyReady = false;
  auto ensureParentDomainTopology = [&]() -> ncclResult_t {
    if (parentTopologyReady) {
      return ncclSuccess;
    }
    /* The probe only reads the parent DevComm's LSA size/rank. Match split
     * setup's zero-resource RAIL probe instead of allocating a full-GIN
     * world barrier before the actual transfer requirements are known. */
    StagingProfileScope profileScope(profile.get(), STAGING_PROFILE_PROBE_DEV_COMM);
    NCCL_M2N_CHECK(reshardGetOrCreateDevCommWithRequirements(comm, 0, 0, 0, RESHARD_DEVCOMM_BARRIER_WORLD, 1,
                                                             NCCLM2N_GIN_RAIL_CONNECTION, workStream, &probeDevComm,
                                                             &probeDevCommUse));
    NCCL_M2N_CHECK(reshardRecordDevCommUse(&probeDevCommUse, workStream));
    lsa_size_from_comm = (probeDevComm.lsaSize > 0) ? probeDevComm.lsaSize : 0;
    NCCL_M2N_CHECK(resolveReshardDomainSizes(world_rank, RESHARD_ALGO_RING, lsa_size_from_comm, lsa_size_from_comm,
                                             &src_gpus_per_domain, &dst_gpus_per_domain));
    node_local_rank = (probeDevComm.lsaRank >= 0 && probeDevComm.lsaRank < dst_gpus_per_domain) ?
                        probeDevComm.lsaRank :
                        world_rank % dst_gpus_per_domain;
    parentTopologyReady = true;
    return ncclSuccess;
  };

  if (copyAlgo != RESHARD_COPY_ALGO_PIPE) {
    NCCL_M2N_CHECK(ensureParentDomainTopology());
  }

  StagingTransferDescriptor desc;
  size_t maxPeerGroupSize = 1;
  bool splitMetaAvailable = false;
  auto buildCurrentStagingDescriptor = [&](StagingTransferDescriptor* outDesc, bool splitStrided = false,
                                           int splitNumInjectionDomains = 0, int splitDomainsPerRep = 1,
                                           bool nodeAnchorAtMeshStart = false) -> ncclResult_t {
    size_t localMaxPeerGroupSize = 1;
    StagingProfileScope profileScope(profile.get(), STAGING_PROFILE_BUILD_DESCRIPTOR);
    NCCL_M2N_CHECK(validateStagingPlanLimits(world_rank, srcTensor, src_dims_bytes, dstTensor, dst_dims_bytes, copyAlgo,
                                             dst_gpus_per_domain, &localMaxPeerGroupSize, splitStrided,
                                             splitNumInjectionDomains, splitDomainsPerRep, nodeAnchorAtMeshStart));
    maxPeerGroupSize = localMaxPeerGroupSize;
    memset(outDesc, 0, sizeof(*outDesc));
    if (copyAlgo == RESHARD_COPY_ALGO_DIRECT) {
      NCCL_M2N_CHECK(buildStagingDirectTransferDescriptor(comm, srcBuffer, src_dims_bytes, ndims, srcTensor, dstBuffer,
                                                          dst_dims_bytes, dstTensor, dst_gpus_per_domain,
                                                          node_local_rank, outDesc));
    } else {
      NCCL_M2N_CHECK(buildStagingTransferDescriptor(comm, srcBuffer, src_dims_bytes, ndims, srcTensor, dstBuffer,
                                                    dst_dims_bytes, dstTensor, src_gpus_per_domain, dst_gpus_per_domain,
                                                    node_local_rank, outDesc, splitStrided, splitNumInjectionDomains,
                                                    splitDomainsPerRep, nodeAnchorAtMeshStart,
                                                    copyAlgo == RESHARD_COPY_ALGO_PIPE));
    }
    return ncclSuccess;
  };

  auto finalizeCurrentStagingDescriptor = [&]() -> int {
    int activePeerCount = stagingComputeCtaHeuristicPeerCount(&desc, srcTensor, dstTensor);
    desc.ctaHeuristicPeerCount = (copyAlgo == RESHARD_COPY_ALGO_PIPE) ? activePeerCount : 0;
    if (copyAlgo == RESHARD_COPY_ALGO_DIRECT) {
      desc.peerGroupSizeBound =
        stagingComputePeerGroupSizeBound(activePeerCount, dstTensor, dst_gpus_per_domain, copyAlgo);
    } else {
      const int dstNodeAnchor = splitMetaAvailable ? dstMesh->startRank : 0;
      desc.peerGroupSizeBound =
        stagingComputeHierarchyPeerGroupSizeBound(activePeerCount, dstTensor, dst_gpus_per_domain, dstNodeAnchor);
    }
    return stagingResolveNumChannelsForTransfer(&desc);
  };

  int requestedStagingNumCtas = stagingResolveNumChannelsForTransfer(nullptr);
  ReshardSplitComms splitComms{};
  const bool pipeMayAttemptSplit =
    copyAlgo == RESHARD_COPY_ALGO_PIPE && reshardShouldAttemptPipeSplitComms(srcTensor, dstTensor);
  if (pipeMayAttemptSplit) {
    const int srcRepCount = tensorRepCount(srcTensor);
    const int dstRepCount = tensorRepCount(dstTensor);
    const bool dstRepStrided =
      dstTensor->placements[1] == NCCL_RESHARD_REPLICATE && dstMesh->dims[0] > 1 && dstMesh->dims[1] > 1;
    NCCL_M2N_CHECK(reshardGetOrCreateSplitComms(comm, srcMesh, dstMesh, srcRepCount, dstRepCount, dstRepStrided,
                                                requestedStagingNumCtas, workStream, &splitComms));
    splitMetaAvailable = splitComms.valid && splitComms.active && splitComms.lsaSize > 0;

    if (splitMetaAvailable) {
      lsa_size_from_comm = splitComms.lsaSize;
      NCCL_M2N_CHECK(resolveReshardDomainSizes(world_rank, RESHARD_ALGO_RING, splitComms.srcLsaSize, splitComms.lsaSize,
                                               &src_gpus_per_domain, &dst_gpus_per_domain));
      if (reshardRankInMesh(dstMesh, world_rank)) {
        node_local_rank = (world_rank - dstMesh->startRank) % dst_gpus_per_domain;
      } else if (reshardRankInMesh(srcMesh, world_rank)) {
        node_local_rank = (world_rank - srcMesh->startRank) % src_gpus_per_domain;
      } else {
        node_local_rank = world_rank % dst_gpus_per_domain;
      }

      StagingTransferDescriptor splitDesc;
      NCCL_M2N_CHECK(buildCurrentStagingDescriptor(&splitDesc, splitComms.strided, splitComms.numInjectionDomains,
                                                   splitComms.domainsPerRep, true));
      desc = splitDesc;
      requestedStagingNumCtas = finalizeCurrentStagingDescriptor();
    }
  }

  if (copyAlgo == RESHARD_COPY_ALGO_PIPE && !splitMetaAvailable) {
    NCCL_M2N_CHECK(ensureParentDomainTopology());
    NCCL_M2N_CHECK(buildCurrentStagingDescriptor(&desc));
    requestedStagingNumCtas = finalizeCurrentStagingDescriptor();
  } else if (copyAlgo != RESHARD_COPY_ALGO_PIPE) {
    NCCL_M2N_CHECK(buildCurrentStagingDescriptor(&desc));
    requestedStagingNumCtas = finalizeCurrentStagingDescriptor();
  }

  const bool splitStagingActive = (copyAlgo == RESHARD_COPY_ALGO_PIPE) && splitMetaAvailable;
  const ReshardStagingMeshSignature stagingMeshSignature = stagingProcessMeshSignature(
    srcTensor, dstTensor, src_gpus_per_domain, dst_gpus_per_domain, splitStagingActive && splitComms.strided,
    splitStagingActive ? splitComms.numInjectionDomains : 0, splitStagingActive ? splitComms.domainsPerRep : 0);
  const ReshardStagingTensorSignature stagingTensorSignature = stagingProcessTensorSignature(srcTensor, dstTensor);
  const ReshardStagingPipeSignature stagingPipeSignature =
    stagingProcessPipeSignature(stagingMeshSignature, stagingTensorSignature);
  const int stagingControlSlotCount =
    (copyAlgo == RESHARD_COPY_ALGO_PIPE) ? STAGING_PIPE_CONTROL_SLOTS : STAGING_DEFAULT_CONTROL_SLOTS;
  int persistentControlSlot = 0;
  bool persistentControlSlotValid = false;
  if (copyAlgo == RESHARD_COPY_ALGO_PIPE) {
    NCCL_M2N_CHECK(reshardGetOrCreatePersistentControlSlot(comm, stagingPipeSignature, world_rank,
                                                           &persistentControlSlot));
    NCCL_M2N_CHECK_ARG(persistentControlSlot >= 0 && persistentControlSlot < stagingControlSlotCount, world_rank,
                       "PIPE persistent control slot %d exceeds control slot count %d", persistentControlSlot,
                       stagingControlSlotCount);
    persistentControlSlotValid = true;
    desc.controlSlot = persistentControlSlot;
  }
  StagingBufferState* staging = nullptr;
  const ReshardStagingRuntimeConfig& stagingConfig = reshardGetStagingRuntimeConfig();
  const int stagingPoolCapacityChannels =
    (pipeHostRma && !stagingConfig.numChannelsExplicit) ? STAGING_PIPE_HOST_RMA_DEFAULT_NUM_CHANNELS :
    (copyAlgo == RESHARD_COPY_ALGO_PIPE && !stagingConfig.numChannelsFixed) ? STAGING_MAX_CHANNELS :
                                                                              requestedStagingNumCtas;
  /* PIPE reuses one staging allocation per communicator. Its persistent
   * control slots are shared across split and parent launches so cursor
   * slices cannot alias when a model mixes the two paths. */
  const uint64_t stagingPoolKey = (copyAlgo == RESHARD_COPY_ALGO_PIPE) ?
                                    stagingPipeReusablePoolKey() :
                                    stagingDirectReusablePoolKey(stagingPoolCapacityChannels);
  {
    StagingProfileScope profileScope(profile.get(), STAGING_PROFILE_ENSURE_BUFFER);
    NCCL_M2N_CHECK(ensureStagingBufferPool(comm, stagingPoolKey, workStream, requestedStagingNumCtas,
                                           stagingPoolCapacityChannels, stagingControlSlotCount, &staging));
  }
  const int staging_num_ctas = staging->numChannels;
  StagingBufferEventGuard stagingEvent(comm, stagingPoolKey, workStream);

  if (copyAlgo == RESHARD_COPY_ALGO_PIPE) {
    desc.pipeGinPeerCapacity = reshardGetPipeGinPeersPerSlot();
    desc.pipeGinChannelsPerPeer = reshardGetPipeGinChannelsPerPeer();
  }

  ncclWindow_t staging_window = nullptr;
  {
    StagingProfileScope profileScope(profile.get(), STAGING_PROFILE_REGISTER_WINDOW);
    if (!splitStagingActive) {
      ncclWindow_t* cached_win =
        findCachedInternalWindowByPtr(comm, staging->buffer, staging->totalSize, RESHARD_INTERNAL_WINDOW_STAGING);
      if (cached_win != nullptr) {
        staging_window = *cached_win;
      } else {
        {
          M2nApiUnlock apiUnlock;
          NCCL_M2N_CHECK(ncclCommWindowRegister(comm, staging->buffer, staging->totalSize, &staging_window,
                                                NCCL_WIN_COLL_SYMMETRIC | NCCL_WIN_STRICT_ORDERING));
          NCCL_M2N_CHECK(m2nWaitCommReady(comm));
        }
        NCCL_M2N_CHECK(cacheInternalWindow(comm, staging->buffer, staging->totalSize, RESHARD_INTERNAL_WINDOW_STAGING,
                                           staging_window));
      }
    }
  }

  std::unique_ptr<StagingKernelParams> directParams;
  const int pipeTmaTileSize = (copyAlgo == RESHARD_COPY_ALGO_PIPE) ? getPipeTmaTileSize() : 0;
  StagingPipeCallParams pipeCall{};
  StagingPipePlanCacheEntry* pipePlanEntry = nullptr;
  const StagingKernelParams* pipeLaunchParams = nullptr;
  bool pipePlanCacheMiss = false;

  {
    StagingProfileScope profileScope(profile.get(), STAGING_PROFILE_PREPARE_PARAMS);
    if (copyAlgo == RESHARD_COPY_ALGO_PIPE) {
      NCCL_M2N_CHECK(stagingGetPipePlanEntry(staging, stagingPipeSignature, persistentControlSlot, world_rank,
                                             &pipePlanEntry, &pipePlanCacheMiss));
      if (pipePlanCacheMiss) {
        NCCL_M2N_CHECK(stagingPrepareTransfer(staging, &desc, staging_window, staging_window,
                                              pipePlanEntry->hostParams));
      }
      pipeLaunchParams = pipePlanEntry->hostParams;
      pipeCall.srcBuffer = srcBuffer;
      pipeCall.dstBuffer = dstBuffer;
      pipeCall.epoch = callEpoch;
      const size_t pipeChunkSize = pipeLaunchParams->chunkSize;
      pipeCall.maxChunkRounds =
        (desc.maxEdgeBytes > 0 && pipeChunkSize > 0) ? (1 + (desc.maxEdgeBytes - 1) / pipeChunkSize) : 0;
      pipeCall.splitComm = false;
      pipeCall.hasLocalFanout = desc.hasLocalFanout;
      pipeCall.splitCommBContextBase = 0;
      pipeCall.splitCommBContextCount = 0;
    } else {
      directParams.reset(new (std::nothrow) StagingKernelParams{});
      NCCL_M2N_CHECK_ARG(directParams != nullptr, world_rank, "ncclReshard: failed to allocate staging params");
      NCCL_M2N_CHECK(stagingPrepareTransfer(staging, &desc, staging_window, staging_window, directParams.get()));
      directParams->epoch = callEpoch;
    }
  }

  int resourceGinSignalCount = 0;
  int resourceGinCounterCount = 0;
  int activePipeRdmaPeers = 0;
  const int activeStagingChannels = (copyAlgo == RESHARD_COPY_ALGO_PIPE && pipeLaunchParams != nullptr) ?
                                      pipeLaunchParams->numChannels :
                                      directParams->numChannels;
  /* Buffer capacity is reusable allocation headroom. DevComm GIN resources
   * are collective and use the algorithm's rank-uniform resource plan. */
  const int resourceStagingChannels = activeStagingChannels;
  const int pipeGinPeersPerSlot = (copyAlgo == RESHARD_COPY_ALGO_PIPE) ? reshardGetPipeGinPeersPerSlot() : 0;
  const int pipeGinChannelsPerPeer = (copyAlgo == RESHARD_COPY_ALGO_PIPE) ? reshardGetPipeGinChannelsPerPeer() : 0;
  {
    StagingProfileScope profileScope(profile.get(), STAGING_PROFILE_RESOLVE_GIN_COUNTS);
    ReshardMeshInterval srcInterval{};
    ReshardMeshInterval dstInterval{};
    NCCL_M2N_CHECK(computeReshardMeshInterval(srcMesh, world_rank, &srcInterval));
    NCCL_M2N_CHECK(computeReshardMeshInterval(dstMesh, world_rank, &dstInterval));
    size_t maxPeersBound = std::max<size_t>(1, maxPeerGroupSize);
    maxPeersBound = std::max(maxPeersBound, static_cast<size_t>(desc.peerGroupSizeBound));
    NCCL_M2N_CHECK_ARG(static_cast<size_t>(desc.numTargets) <= maxPeersBound &&
                         static_cast<size_t>(desc.numSources) <= maxPeersBound,
                       world_rank,
                       "ncclReshard: local staging peer count exceeds rank-uniform staging bound "
                       "(targets=%d sources=%d bound=%zu)",
                       desc.numTargets, desc.numSources, maxPeersBound);
    if (copyAlgo == RESHARD_COPY_ALGO_PIPE) {
      activePipeRdmaPeers = std::max(pipeLaunchParams->numRdmaTargets, pipeLaunchParams->numRdmaSources);
      NCCL_M2N_CHECK_ARG(activePipeRdmaPeers <= pipeGinPeersPerSlot, world_rank,
                         "ncclReshard: PIPE RDMA peer count %d exceeds fixed per-slot capacity %d; "
                         "set NCCL_RESHARD_PIPE_GIN_PEERS_PER_SLOT to a sufficient bound",
                         activePipeRdmaPeers, pipeGinPeersPerSlot);
      NCCL_M2N_CHECK_ARG(pipeGinChannelsPerPeer > 0, world_rank,
                         "ncclReshard: PIPE GIN channels-per-peer capacity must be positive");
      NCCL_M2N_CHECK(reshardComputeStagingGinCounts(world_rank, pipeGinChannelsPerPeer, pipeGinPeersPerSlot,
                                                    &resourceGinSignalCount, &resourceGinCounterCount));
    } else {
      NCCL_M2N_CHECK(reshardComputeStagingGinCounts(world_rank, activeStagingChannels, maxPeersBound,
                                                    &resourceGinSignalCount, &resourceGinCounterCount));
      directParams->ginSignalCount = resourceGinSignalCount;
      directParams->ginCounterCount = resourceGinCounterCount;
    }
  }

  ncclDevComm activeDevComm{};
  ncclDevComm* devCommPtr = nullptr;
  ReshardDevCommUse devCommUse;
  ncclDevComm splitDevCommA{};
  ncclDevComm splitDevCommB{};
  ReshardDevCommUse splitDevCommAUse;
  ReshardDevCommUse splitDevCommBUse;
  ncclDevComm launchDevCommA{};
  ncclDevComm launchDevCommB{};
  const int configuredGinContextCount = std::max(1, reshardGetGinContextCount());
  const int stagingGinContextCount =
    (copyAlgo == RESHARD_COPY_ALGO_PIPE) ? configuredGinContextCount : staging_num_ctas;
  {
    StagingProfileScope profileScope(profile.get(), STAGING_PROFILE_GET_DEV_COMM);
    if (splitStagingActive) {
      const int maxConcurrency = splitComms.slotCount;
      const int persistentControlSlots = reshardGetPersistentControlSlotCount();
      const int allocatedPersistentControlSlots = persistentControlSlots;
      NCCL_M2N_CHECK_ARG(persistentControlSlotValid, world_rank,
                         "ncclReshard: split PIPE persistent control slot was not initialized");
      NCCL_M2N_CHECK_ARG(persistentControlSlots > 0 && persistentControlSlot >= 0 &&
                           persistentControlSlot < persistentControlSlots,
                         world_rank, "ncclReshard: invalid persistent control slot %d/%d", persistentControlSlot,
                         persistentControlSlots);
      NCCL_M2N_CHECK_ARG(allocatedPersistentControlSlots > 0 &&
                           allocatedPersistentControlSlots <= persistentControlSlots &&
                           persistentControlSlot < allocatedPersistentControlSlots,
                         world_rank, "ncclReshard: invalid allocated persistent control-slot count %d for slot %d/%d",
                         allocatedPersistentControlSlots, persistentControlSlot, persistentControlSlots);
      const int signalsPerControlSlot = resourceGinSignalCount;
      const int countersPerControlSlot = resourceGinCounterCount;
      NCCL_M2N_CHECK_ARG(signalsPerControlSlot <= std::numeric_limits<int>::max() / allocatedPersistentControlSlots &&
                           countersPerControlSlot <= std::numeric_limits<int>::max() / allocatedPersistentControlSlots,
                         world_rank,
                         "ncclReshard: split PIPE persistent-control GIN resource count overflows int "
                         "(signalStride=%d counterStride=%d slots=%d)",
                         signalsPerControlSlot, countersPerControlSlot, allocatedPersistentControlSlots);
      const int ginSignalCountA = signalsPerControlSlot * allocatedPersistentControlSlots;
      const int ginCounterCountA = countersPerControlSlot * allocatedPersistentControlSlots;
      const int signalsPerNamespaceB = signalsPerControlSlot * allocatedPersistentControlSlots;
      const int countersPerNamespaceB = countersPerControlSlot * allocatedPersistentControlSlots;
      const int ctxPerSlotB = configuredGinContextCount;
      ReshardCommBResourcePlan resourcesB{};
      NCCL_M2N_CHECK_ARG(reshardBuildPipeCommBResourcePlan(splitComms.pipeNamespaceCapacity, maxConcurrency,
                                                          allocatedPersistentControlSlots, signalsPerControlSlot,
                                                          countersPerControlSlot, ctxPerSlotB, &resourcesB),
                         world_rank, "ncclReshard: invalid or overflowing split PIPE commB resource plan");
      ncclWindow_t windowA = nullptr;
      ncclWindow_t windowB = nullptr;
      ncclDevComm* outSplitDevCommA = pipeHostRma ? nullptr : &splitDevCommA;
      ReshardDevCommUse* outSplitDevCommAUse = pipeHostRma ? nullptr : &splitDevCommAUse;
      ncclDevComm* outSplitDevCommB = pipeHostRma ? nullptr : &splitDevCommB;
      ReshardDevCommUse* outSplitDevCommBUse = pipeHostRma ? nullptr : &splitDevCommBUse;
      NCCL_M2N_CHECK(reshardSplitEnsureResources(
        &splitComms, staging->buffer, staging->totalSize, 0, RESHARD_DEVCOMM_BARRIER_NONE, ginSignalCountA,
        ginCounterCountA, resourcesB, workStream, &windowA,
        &windowB, outSplitDevCommA, outSplitDevCommAUse, outSplitDevCommB, outSplitDevCommBUse));
      if (pipePlanCacheMiss) {
        NCCL_M2N_CHECK(applySplitPipeParams(pipePlanEntry->hostParams, &splitComms, windowA, windowB,
                                            signalsPerControlSlot, countersPerControlSlot, signalsPerNamespaceB,
                                            countersPerNamespaceB, persistentControlSlot));
      }
      pipeCall.splitComm = true;
      pipeCall.splitCommBContextBase = splitComms.slotIdx * ctxPerSlotB;
      pipeCall.splitCommBContextCount = ctxPerSlotB;
      RESHARD_INFO(world_rank,
                   "PIPE commB-state: namespace=%d/%d controlSlot=%d lane=%d/%d "
                   "signalBaseB=%d counterBaseB=%d contextBaseB=%d totalSignalsB=%d totalCountersB=%d totalContextsB=%d",
                   splitComms.pipeNamespaceId, splitComms.pipeNamespaceCapacity, persistentControlSlot,
                   splitComms.slotIdx, splitComms.slotCount,
                   splitComms.pipeNamespaceId * signalsPerNamespaceB + persistentControlSlot * signalsPerControlSlot,
                   splitComms.pipeNamespaceId * countersPerNamespaceB + persistentControlSlot * countersPerControlSlot,
                   pipeCall.splitCommBContextBase, resourcesB.signals, resourcesB.counters, resourcesB.contexts);
      if (!pipeHostRma) {
        launchDevCommA = splitComms.inA ? splitDevCommA : splitDevCommB;
        launchDevCommB = splitComms.inB ? splitDevCommB : splitDevCommA;
      }
    } else if (copyAlgo == RESHARD_COPY_ALGO_PIPE) {
      const int persistentControlSlots = STAGING_PIPE_CONTROL_SLOTS;
      NCCL_M2N_CHECK_ARG(resourceGinSignalCount <= std::numeric_limits<int>::max() / persistentControlSlots &&
                           resourceGinCounterCount <= std::numeric_limits<int>::max() / persistentControlSlots,
                         world_rank,
                         "ncclReshard: non-split PIPE persistent-control GIN resource count overflows int "
                         "(signalStride=%d counterStride=%d slots=%d)",
                         resourceGinSignalCount, resourceGinCounterCount, persistentControlSlots);
      const int totalGinSignalCount = resourceGinSignalCount * persistentControlSlots;
      const int totalGinCounterCount = resourceGinCounterCount * persistentControlSlots;
      if (!pipeHostRma) {
        NCCL_M2N_CHECK(reshardGetOrCreateDevComm(comm, 0, totalGinSignalCount,
                                                 totalGinCounterCount, RESHARD_DEVCOMM_BARRIER_NONE,
                                                 stagingGinContextCount, workStream, &activeDevComm, &devCommUse));
        devCommPtr = &activeDevComm;
      }
      if (pipePlanCacheMiss) {
        applyPipeRdmaOffsets(pipePlanEntry->hostParams, nullptr, persistentControlSlot * resourceGinSignalCount,
                             persistentControlSlot * resourceGinCounterCount, 0, 0);
      }
    } else {
      NCCL_M2N_CHECK(reshardGetOrCreateDevComm(comm, resourceStagingChannels, resourceGinSignalCount,
                                               resourceGinCounterCount, RESHARD_DEVCOMM_BARRIER_WORLD,
                                               stagingGinContextCount, workStream, &activeDevComm, &devCommUse));
      devCommPtr = &activeDevComm;
    }

    if (copyAlgo == RESHARD_COPY_ALGO_PIPE && pipePlanCacheMiss) {
      NCCL_M2N_CHECK(stagingFinalizePipeHostPlan(pipePlanEntry, stagingPipeSignature, world_rank));
      pipeLaunchParams = pipePlanEntry->hostParams;
    }
  }

  if (pipeHostRma) {
    NCCL_M2N_CHECK(pipeHostPlanMaxChunkRounds(pipeLaunchParams, pipePlanEntry->hostPlan, world_rank,
                                              &pipeCall.maxChunkRounds));
  }

  RESHARD_INFO(world_rank,
               "ncclReshard: copy_algo=%d num_ctas=%d staging_channels=%d resource_channels=%d "
               "gin_signal=%d gin_counter=%d "
               "gpus_per_domain=%d lsa_size=%d split_pipe=%d pipe_tma_tile=%d "
               "pipe_net_mode=%d src_buf=%p dst_buf=%p staging=%p",
               (int)copyAlgo, staging_num_ctas, activeStagingChannels, resourceStagingChannels, resourceGinSignalCount,
               resourceGinCounterCount, dst_gpus_per_domain, lsa_size_from_comm, (int)splitStagingActive,
               pipeTmaTileSize, (int)reshardGetPipeNetMode(), srcBuffer, dstBuffer, staging->buffer);

  if (copyAlgo == RESHARD_COPY_ALGO_DIRECT) {
    NCCL_M2N_CHECK(stagingEnsureDirectDevParams(staging, world_rank));
    NCCL_M2N_CUDACHECK(cudaMemcpyAsync(staging->devParams, directParams.get(), sizeof(*directParams),
                                       cudaMemcpyHostToDevice, workStream));
  }
  NCCL_M2N_CHECK(reshardStartWorkStream(stream, &work));
  if (splitStagingActive) {
    NCCL_M2N_CHECK(reshardSplitPrepareCommBResourceLane(&splitComms, workStream));
  }

  ncclResult_t launchResult = ncclSuccess;
  auto launchPipe = [&](bool splitLaunch) -> ncclResult_t {
    ncclResult_t result = stagingEnsurePipeDevicePlan(pipePlanEntry, stagingPipeSignature, workStream, world_rank);
    if (result == ncclSuccess) {
      if (splitLaunch) {
        result = launchStagingReshardPipeSplit(pipeLaunchParams, &pipeCall, pipePlanEntry->devParams,
                                               pipePlanEntry->devPlan, &launchDevCommA, &launchDevCommB,
                                               staging_num_ctas, pipeTmaTileSize, workStream);
      } else {
        result = launchStagingReshardPipe(pipeLaunchParams, &pipeCall, pipePlanEntry->devParams, pipePlanEntry->devPlan,
                                          devCommPtr, staging_num_ctas, pipeTmaTileSize, workStream);
      }
    }
    return result;
  };
  auto launchPipeHostRmaWithTiming = [&](bool splitLaunch, bool verbose) -> ncclResult_t {
    /* Host-RMA keeps the cached host plan but replaces the device trainer /
     * generator kernels with host-orchestrated NCCL RMA and CUDA CE work.
     * There is no DevComm use to record for this path; split state is passed
     * only so host RMA can choose the same split A/B communication graph.
     * Release the API gate for every submission and readiness wait in the
     * pipeline so another local rank can enqueue its matching operations. */
    {
      M2nApiUnlock apiUnlock;
      return launchStagingReshardPipeHostRma(pipeLaunchParams, pipePlanEntry->hostPlan, staging, &pipeCall,
                                             staging->buffer, comm, splitLaunch ? &splitComms : nullptr, workStream,
                                             verbose);
    }
  };
  {
    StagingProfileScope profileScope(profile.get(), STAGING_PROFILE_LAUNCH_KERNEL);
    const bool verbose = debugLogging;
    if (copyAlgo == RESHARD_COPY_ALGO_DIRECT) {
      launchResult = launchStagingReshardDirect(staging->devParams, devCommPtr, staging_num_ctas, workStream, verbose);
    } else if (pipeHostRma) {
      launchResult = launchPipeHostRmaWithTiming(splitStagingActive, verbose);
    } else if (splitStagingActive) {
      launchResult = launchPipe(true);
    } else {
      launchResult = launchPipe(false);
    }
  }
  if (splitStagingActive && !pipeHostRma) {
    NCCL_M2N_CHECK(reshardRecordDevCommUse(&splitDevCommAUse, workStream));
    NCCL_M2N_CHECK(reshardRecordDevCommUse(&splitDevCommBUse, workStream));
  } else if (!pipeHostRma) {
    NCCL_M2N_CHECK(reshardRecordDevCommUse(&devCommUse, workStream));
  }
  if (splitStagingActive) {
    NCCL_M2N_CHECK(reshardSplitRecordCommBResourceLane(&splitComms, workStream));
  }

  if (launchResult != ncclSuccess) {
    NCCL_M2N_CHECK_WARN(stagingEvent.complete());
    NCCL_M2N_CHECK_WARN(workCompletion.complete());
    return launchResult;
  }

  if (profile != nullptr) {
    profile->log(world_rank);
  }

  /* Record event on the staging buffer for cross-stream ordering
   * (Change 3 — per-comm buffer with events). */
  NCCL_M2N_CHECK(stagingEvent.complete());

  NCCL_M2N_CHECK(workCompletion.complete());
#ifdef NCCL_M2N_TESTING
  gLastCompletedCopyAlgorithm.store(copyAlgo, std::memory_order_relaxed);
#endif

  return ncclSuccess;
}
