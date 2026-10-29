/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

/*************************************************************************
 * staging_buffer.cc — host-side implementation of the staging-buffer
 * lifecycle that backs ncclReshard.
 *
 * This TU has no CUDA-kernel code; cudaMemset / cudaMalloc are runtime
 * host APIs and link cleanly when compiled by the host C++ frontend.
 ************************************************************************/

#include "staging_buffer.h"
#include "m2n_checks.h"
#include "m2n_checked_math.h"
#include "reshard_internal.h"
#include "reshard_limits.h"
#include "m2n_log.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>

#include <cuda_runtime.h>
#include "nccl.h"

/* ======================================================================
 * Verbose-flag shim (preserves the public stagingSetVerbose API).  In
 * the new repo we mostly drive logging through reshardSetLogLevel, but
 * the staging-only knob is useful in the short-term while the kernel is
 * being stabilised.
 * ====================================================================*/
static bool gStagingVerbose = false;

void stagingSetVerbose(bool verbose) {
  gStagingVerbose = verbose;
  if (verbose && reshardGetLogLevel() < RESHARD_LOG_DEBUG) {
    reshardSetLogLevel(RESHARD_LOG_DEBUG);
  }
}

#define STAGING_LOG(rank, fmt, ...) \
  do { \
    if (gStagingVerbose || reshardGetLogLevel() >= RESHARD_LOG_DEBUG) { \
      RESHARD_DEBUG((rank), "[STAGING] " fmt, ##__VA_ARGS__); \
    } \
  } while (0)

#define STAGING_NCCLCHECK(cmd) \
  do { \
    ncclResult_t r = (cmd); \
    if (r != ncclSuccess) { \
      fprintf(stderr, "[STAGING] NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(r)); \
      return r; \
    } \
  } while (0)

#define STAGING_CUDACHECK(cmd) \
  do { \
    cudaError_t e = (cmd); \
    if (e != cudaSuccess) { \
      fprintf(stderr, "[STAGING] CUDA error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
      return ncclInternalError; \
    } \
  } while (0)

static bool stagingPipeUsesSharedDataRegion() {
  return reshardGetCopyAlgorithm() == RESHARD_COPY_ALGO_PIPE;
}

static int stagingDefaultTargetCtasForPeerGroups(int peerGroupCount) {
  if (peerGroupCount <= 0) {
    return 6;
  }
  int ctasPerPeerGroup = 1;
  if (peerGroupCount <= 8) {
    ctasPerPeerGroup = 4;
  } else if (peerGroupCount <= 16) {
    ctasPerPeerGroup = 2;
  }
  return std::max(6, peerGroupCount * ctasPerPeerGroup);
}

static int clampNumChannels(int numChannels) {
  if (numChannels <= 0) {
    numChannels = STAGING_DEFAULT_NUM_CHANNELS;
  }
  if (numChannels > STAGING_MAX_CHANNELS) {
    RESHARD_WARN(-1, "[STAGING] numChannels %d exceeds STAGING_MAX_CHANNELS %d, clamping", numChannels,
                 STAGING_MAX_CHANNELS);
    numChannels = STAGING_MAX_CHANNELS;
  }
  return numChannels;
}

static size_t stagingChannelControlBase(const StagingBufferState* state, int channel) {
  return state->adaptiveChannelLayout ? (size_t)channel * state->controlRegionSize :
                                        (size_t)channel * state->channelSize;
}

static size_t stagingChannelDataBase(const StagingBufferState* state, int channel) {
  if (state->adaptiveChannelLayout) {
    return (size_t)state->capacityChannels * state->controlRegionSize + (size_t)channel * state->channelDataSize;
  }
  return stagingChannelControlBase(state, channel) + state->controlRegionSize;
}

/* ======================================================================
 * stagingBufferInit
 * ====================================================================*/

static ncclResult_t stagingBufferInitInternal(StagingBufferState* state, int numChannelsOverride,
                                              int controlSlotCountOverride) {
  NCCL_M2N_CHECK_ARG(state != nullptr, -1, "[STAGING] stagingBufferInit called with null state");
  memset(state, 0, sizeof(*state));

  const ReshardStagingRuntimeConfig& config = reshardGetStagingRuntimeConfig();
  int numChannels = (numChannelsOverride > 0) ?
                      numChannelsOverride :
                      (config.numChannelsFixed ? config.numChannels : STAGING_DEFAULT_NUM_CHANNELS);
  numChannels = clampNumChannels(numChannels);
  int controlSlotCount = (controlSlotCountOverride > 0) ? controlSlotCountOverride : STAGING_DEFAULT_CONTROL_SLOTS;
  size_t channelDataSize = config.channelDataSize;
  size_t chunkSize = config.chunkSize;
  int peersPerChannel = config.peersPerChannel;
  const bool adaptiveChannelLayout =
    config.hostRmaDefault && !config.numChannelsExplicit && !config.channelDataSizeExplicit;

  STAGING_LOG(-1, "stagingBufferInit() ENTRY");
  STAGING_LOG(-1,
              "  numChannels=%d%s channelDataSize=%zu (%zuMB) controlSlots=%d chunkSize=%zu (%zuKB) "
              "peersPerChannel=%d",
              numChannels, config.numChannelsExplicit ? " explicit" : (config.numChannelsFixed ? " default" : ""),
              channelDataSize, channelDataSize / (1024 * 1024), controlSlotCount, chunkSize, chunkSize / 1024,
              peersPerChannel);

  /* The configured channel size is data capacity only. PIPE shares that
   * capacity between RDMA ingress and local LSA forwarding. */
  const bool sharedPipeDataRegion = stagingPipeUsesSharedDataRegion();
  size_t chunkPairSize = 0;
  NCCL_M2N_CHECK_ARG(m2nCheckedMulSize(2, chunkSize, &chunkPairSize), -1,
                     "[STAGING] chunkSize %zu overflows staging channel sizing", chunkSize);
  size_t ctrlRegionSize = 0;
  NCCL_M2N_CHECK_ARG(m2nCheckedMulSize((size_t)controlSlotCount, (size_t)STAGING_CTRL_REGION_SIZE, &ctrlRegionSize), -1,
                     "[STAGING] controlSlotCount %d overflows staging channel sizing", controlSlotCount);
  const size_t minChannelDataSize = sharedPipeDataRegion ? chunkSize : chunkPairSize;
  NCCL_M2N_CHECK_ARG(channelDataSize >= minChannelDataSize, -1,
                     "[STAGING] channel data size %zu too small (min %zu with chunkSize %zu)", channelDataSize,
                     minChannelDataSize, chunkSize);
  size_t channelSize = 0;
  NCCL_M2N_CHECK_ARG(m2nCheckedAddSize(ctrlRegionSize, channelDataSize, &channelSize), -1,
                     "[STAGING] channel data size %zu overflows with control region %zu", channelDataSize,
                     ctrlRegionSize);

  size_t dataCapacity = 0;
  NCCL_M2N_CHECK_ARG(m2nCheckedMulSize((size_t)numChannels, channelDataSize, &dataCapacity), -1,
                     "[STAGING] data capacity overflows: channels=%d channelDataSize=%zu", numChannels,
                     channelDataSize);
  size_t totalSize = 0;
  if (adaptiveChannelLayout) {
    size_t controlCapacity = 0;
    NCCL_M2N_CHECK_ARG(m2nCheckedMulSize((size_t)numChannels, ctrlRegionSize, &controlCapacity) &&
                         m2nCheckedAddSize(controlCapacity, dataCapacity, &totalSize),
                       -1, "[STAGING] adaptive staging size overflows: channels=%d", numChannels);
  } else {
    NCCL_M2N_CHECK_ARG(m2nCheckedMulSize((size_t)numChannels, channelSize, &totalSize), -1,
                       "[STAGING] total staging size overflows: channels=%d channelSize=%zu", numChannels, channelSize);
  }
  size_t dataRegionSize = sharedPipeDataRegion ? channelDataSize : channelDataSize / 2;
  size_t slotsPerRegionSize = dataRegionSize / chunkSize;
  NCCL_M2N_CHECK_ARG(slotsPerRegionSize <= (size_t)INT_MAX, -1, "[STAGING] slots/region %zu exceeds INT_MAX",
                     slotsPerRegionSize);
  int slotsPerRegion = (int)slotsPerRegionSize;

  STAGING_LOG(-1, "  per-channel: ctrl=%zuB data=%zuB shared_pipe_data=%d region=%zuB slots=%d", ctrlRegionSize,
              channelDataSize, (int)sharedPipeDataRegion, dataRegionSize, slotsPerRegion);
  STAGING_LOG(-1, "  total alloc=%zu bytes (%zuMB) adaptive=%d", totalSize, totalSize / (1024 * 1024),
              (int)adaptiveChannelLayout);

  void* buffer = nullptr;
  STAGING_NCCLCHECK(ncclMemAlloc(&buffer, totalSize));

  /* From here on, state->initialized is still false, so a later
   * stagingBufferFinalize would early-out without reclaiming `buffer`.
   * Free it explicitly on any failure between here and `initialized=true`. */
  cudaError_t memsetResult = cudaSuccess;
  if (adaptiveChannelLayout) {
    size_t controlCapacity = 0;
    if (!m2nCheckedMulSize((size_t)numChannels, ctrlRegionSize, &controlCapacity)) {
      ncclMemFree(buffer);
      return ncclInternalError;
    }
    memsetResult = cudaMemset(buffer, 0, controlCapacity);
  } else {
    memsetResult = cudaMemset2D(buffer, channelSize, 0, ctrlRegionSize, numChannels);
  }
  if (memsetResult != cudaSuccess) {
    fprintf(stderr, "[STAGING] CUDA error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(memsetResult));
    ncclMemFree(buffer);
    return ncclInternalError;
  }

  state->buffer = buffer;
  state->totalSize = totalSize;
  state->numChannels = numChannels;
  state->capacityChannels = numChannels;
  state->channelSize = channelSize;
  state->channelDataSize = channelDataSize;
  state->dataCapacity = dataCapacity;
  state->controlSlotCount = controlSlotCount;
  state->controlRegionSize = ctrlRegionSize;
  state->chunkSize = chunkSize;
  state->peersPerChannel = peersPerChannel;
  state->adaptiveChannelLayout = adaptiveChannelLayout;
  state->devParams = nullptr;
  state->pipePlanCacheNextVictim = 0;
  state->initialized = true;

  RESHARD_INFO(-1,
               "[STAGING] init complete: %d channels x %zuMB data (+%zuKB control) = %zuMB total, "
               "chunkSize=%zuKB controlSlots=%d slots/region=%d peersPerChannel=%d",
               numChannels, channelDataSize / (1024 * 1024), ctrlRegionSize / 1024, totalSize / (1024 * 1024),
               chunkSize / 1024, controlSlotCount, slotsPerRegion, peersPerChannel);
  return ncclSuccess;
}

ncclResult_t stagingBufferInit(StagingBufferState* state) {
  return stagingBufferInitInternal(state, 0, STAGING_DEFAULT_CONTROL_SLOTS);
}

ncclResult_t stagingBufferInitWithNumChannels(StagingBufferState* state, int numChannels) {
  return stagingBufferInitInternal(state, numChannels, STAGING_DEFAULT_CONTROL_SLOTS);
}

ncclResult_t stagingBufferInitWithNumChannelsAndControlSlots(StagingBufferState* state, int numChannels,
                                                             int controlSlotCount) {
  return stagingBufferInitInternal(state, numChannels, controlSlotCount);
}

ncclResult_t stagingBufferConfigureActiveChannels(StagingBufferState* state, int numChannels) {
  NCCL_M2N_CHECK_ARG(state != nullptr && state->initialized, -1,
                     "[STAGING] configure active channels requires an initialized state");
  NCCL_M2N_CHECK_ARG(numChannels > 0 && numChannels <= state->capacityChannels, -1,
                     "[STAGING] active channels %d exceed pool capacity %d", numChannels, state->capacityChannels);
  if (!state->adaptiveChannelLayout) {
    state->numChannels = numChannels;
    return ncclSuccess;
  }

  const bool sharedPipeDataRegion = stagingPipeUsesSharedDataRegion();
  size_t slotPairBytes = 0;
  NCCL_M2N_CHECK_ARG(m2nCheckedMulSize(2, state->chunkSize, &slotPairBytes), -1,
                     "[STAGING] chunk size %zu overflows adaptive channel sizing", state->chunkSize);
  size_t channelDataSize = state->dataCapacity / (size_t)numChannels;
  const size_t channelDataGranularity = sharedPipeDataRegion ? state->chunkSize : slotPairBytes;
  channelDataSize -= channelDataSize % channelDataGranularity;
  NCCL_M2N_CHECK_ARG(channelDataSize >= channelDataGranularity, -1,
                     "[STAGING] adaptive channel data size %zu is smaller than %zu bytes", channelDataSize,
                     channelDataGranularity);
  size_t channelSize = 0;
  NCCL_M2N_CHECK_ARG(m2nCheckedAddSize(state->controlRegionSize, channelDataSize, &channelSize), -1,
                     "[STAGING] adaptive channel size overflow");

  state->numChannels = numChannels;
  state->channelDataSize = channelDataSize;
  state->channelSize = channelSize;
  STAGING_LOG(-1, "adaptive Host-RMA layout: activeChannels=%d channelData=%zuMB shared_pipe_data=%d slots/region=%zu",
              numChannels, channelDataSize / (1024 * 1024), (int)sharedPipeDataRegion,
              (sharedPipeDataRegion ? channelDataSize : channelDataSize / 2) / state->chunkSize);
  return ncclSuccess;
}

/* ======================================================================
 * stagingPrepareTransfer — local helpers
 * ====================================================================*/

static void initFlowCtrl(StagingFlowCtrl* fc) {
  memset(fc, 0, sizeof(*fc));
  fc->remoteRank = -1;
  fc->isLocal = false;
  fc->useGinSignal = false;
}

static void setFcDataRegion(StagingFlowCtrl* fc, size_t peerDataOffset, int peerNumSlots, size_t peerChunkSize) {
  fc->peerDataOffset = peerDataOffset;
  fc->peerNumSlots = peerNumSlots;
  fc->peerChunkSize = peerChunkSize;
}

static void setFcLsaProducer(StagingFlowCtrl* fc, size_t channelBase, int myTargetIdx, int mySourceIdxOnDest) {
  fc->useGinSignal = false;
  fc->isLocal = false;
  fc->localHeadOffset = channelBase + (size_t)myTargetIdx * STAGING_CTRL_ENTRY_SIZE + CTRL_FIELD_LSA_HEAD;
  fc->remoteTailOffset = channelBase + (size_t)mySourceIdxOnDest * STAGING_CTRL_ENTRY_SIZE + CTRL_FIELD_LSA_TAIL;
  /* Persistent-counter PIPE keeps LSA bases at zero and tracks progress
   * with per-edge cursors. */
  fc->lsaTailBase = 0;
  fc->lsaHeadBase = 0;
}

static void setFcLsaConsumer(StagingFlowCtrl* fc, size_t channelBase, int mySourceIdx, int sourceTargetIdxForMe) {
  fc->useGinSignal = false;
  fc->isLocal = false;
  fc->localTailOffset = channelBase + (size_t)mySourceIdx * STAGING_CTRL_ENTRY_SIZE + CTRL_FIELD_LSA_TAIL;
  fc->remoteHeadOffset = channelBase + (size_t)sourceTargetIdxForMe * STAGING_CTRL_ENTRY_SIZE + CTRL_FIELD_LSA_HEAD;
  /* Persistent-counter PIPE keeps LSA bases at zero and tracks progress
   * with per-edge cursors. */
  fc->lsaTailBase = 0;
  fc->lsaHeadBase = 0;
}

struct StagingRdmaSignalLayout {
  bool dense;
  int channelsPerPeer;
};

static int stagingRdmaGinSlot(const StagingRdmaSignalLayout& layout, int channelId, int channelRank, int peerIndex,
                              int peerCount) {
  /* PIPE assigns each active edge to a stable (peer, channel-rank) GIN slot.
   * PACK retains the legacy channel-major map. */
  return layout.dense ? peerIndex * layout.channelsPerPeer + channelRank : channelId * peerCount + peerIndex;
}

static int stagingRdmaSignalId(const StagingRdmaSignalLayout& layout, int channelId, int channelRank, int peerIndex,
                               int peerCount) {
  return stagingRdmaGinSlot(layout, channelId, channelRank, peerIndex, peerCount) * 2;
}

static int stagingRdmaCounterId(const StagingRdmaSignalLayout& layout, int channelId, int channelRank, int peerIndex,
                                int peerCount) {
  return stagingRdmaGinSlot(layout, channelId, channelRank, peerIndex, peerCount);
}

static void setFcRdmaSignals(StagingFlowCtrl* fc, const StagingRdmaSignalLayout& layout, int channelId, int channelRank,
                             int numPeers, int myLocalPeerIdx, int myRemotePeerIdx, int remoteNumPeers) {
  fc->useGinSignal = true;
  fc->isLocal = false;

  int localTailId = stagingRdmaSignalId(layout, channelId, channelRank, myLocalPeerIdx, numPeers);
  int localHeadId = localTailId + 1;
  fc->localTailSignal = (ncclGinSignal_t)localTailId;
  fc->localHeadSignal = (ncclGinSignal_t)localHeadId;

  int remoteTailId = stagingRdmaSignalId(layout, channelId, channelRank, myRemotePeerIdx, remoteNumPeers);
  int remoteHeadId = remoteTailId + 1;
  fc->remoteTailSignal = (ncclGinSignal_t)remoteTailId;
  fc->remoteHeadSignal = (ncclGinSignal_t)remoteHeadId;

  fc->tailSignalBase = 0;
  fc->headSignalBase = 0;
}

static void setFcLocalPipeline(StagingFlowCtrl* fc, int myRank, size_t channelBase, int ctrlEntryIndex,
                               size_t peerDataOffset, int peerNumSlots, size_t chunkSize, int ctrlFieldTail,
                               int ctrlFieldHead) {
  fc->remoteRank = myRank;
  fc->isLocal = true;
  fc->useGinSignal = false;

  size_t entryBase = channelBase + (size_t)ctrlEntryIndex * STAGING_CTRL_ENTRY_SIZE;

  fc->localTailOffset = entryBase + ctrlFieldTail;
  fc->localHeadOffset = entryBase + ctrlFieldHead;
  fc->remoteTailOffset = fc->localTailOffset;
  fc->remoteHeadOffset = fc->localHeadOffset;

  setFcDataRegion(fc, peerDataOffset, peerNumSlots, chunkSize);
}

static int ceilDivInt(int numerator, int denominator) {
  if (denominator <= 0) {
    return 0;
  }
  return (numerator + denominator - 1) / denominator;
}

static int ceilDivSizeClamped(size_t numerator, size_t denominator, int clamp) {
  if (denominator == 0 || clamp <= 0) {
    return 0;
  }
  size_t quotient = numerator == 0 ? 0 : 1 + (numerator - 1) / denominator;
  if (quotient > (size_t)clamp) {
    return clamp;
  }
  return (int)quotient;
}

static void setPeerChunkRange(StagingPeerInfo* peer, size_t chunkSize) {
  if (peer == nullptr || chunkSize == 0 || peer->channelCount <= 0) {
    return;
  }
  const size_t totalChunks = (peer->plan.totalBytes + chunkSize - 1) / chunkSize;
  const size_t rank = (size_t)peer->channelRank;
  const size_t count = (size_t)peer->channelCount;
  peer->totalBytes = peer->plan.totalBytes;
  peer->logicalChunkSize = chunkSize;
  peer->chunkStart = (totalChunks * rank) / count;
  peer->chunkEnd = (totalChunks * (rank + 1)) / count;
}

static int positiveMod(int value, int modulus) {
  int result = value % modulus;
  return result < 0 ? result + modulus : result;
}

static int channelsInPeerGroup(int numChannels, int peerGroupCount, int group) {
  if (peerGroupCount <= 0 || group < 0 || group >= peerGroupCount || group >= numChannels) {
    return 0;
  }
  return ((numChannels - 1 - group) / peerGroupCount) + 1;
}

static int channelRankInPeerGroup(int channel, int peerGroupCount) {
  return peerGroupCount > 0 ? channel / peerGroupCount : channel;
}

static int channelPeerGroup(int channel, int peerGroupCount) {
  return peerGroupCount > 0 ? channel % peerGroupCount : 0;
}

static int edgePeerGroup(int sourceIdxOnDest, int targetIdxOnSource, int peerGroupCount) {
  if (peerGroupCount <= 1) {
    return 0;
  }
  return positiveMod(sourceIdxOnDest + targetIdxOnSource, peerGroupCount);
}

static bool channelHasEdge(int channel, int sourceIdxOnDest, int targetIdxOnSource, int peerGroupCount) {
  if (peerGroupCount <= 1) {
    return true;
  }
  return channelPeerGroup(channel, peerGroupCount) == edgePeerGroup(sourceIdxOnDest, targetIdxOnSource, peerGroupCount);
}

static int channelTargetIndexForSource(const StagingTransferDescriptor* desc, int sourceIdx) {
  if (sourceIdx >= 0 && sourceIdx < desc->numSources) {
    return desc->channelTargetIndexOnSource[sourceIdx];
  }
  return 0;
}

static int targetIndexForTarget(const StagingTransferDescriptor* desc, int targetIdx) {
  if (desc->isDest && !desc->isSource) {
    int sourceIdx = desc->sourceIndexOnDest[targetIdx];
    if (sourceIdx >= 0 && sourceIdx < desc->numSources) {
      return channelTargetIndexForSource(desc, sourceIdx);
    }
  }
  return targetIdx;
}

static void channelEdgeKeyForSource(const StagingTransferDescriptor* desc, int sourceIdx, int* sourceKey,
                                    int* targetKey) {
  if (sourceKey == nullptr || targetKey == nullptr) {
    return;
  }
  *sourceKey = sourceIdx;
  *targetKey = channelTargetIndexForSource(desc, sourceIdx);
}

static void channelEdgeKeyForTarget(const StagingTransferDescriptor* desc, int targetIdx, int* sourceKey,
                                    int* targetKey) {
  if (sourceKey == nullptr || targetKey == nullptr) {
    return;
  }
  *sourceKey = desc->sourceIndexOnDest[targetIdx];
  *targetKey = targetIndexForTarget(desc, targetIdx);
}

static int countSourcesForGroup(const StagingTransferDescriptor* desc, int group, int peerGroupCount) {
  int count = 0;
  for (int j = 0; j < desc->numSources; j++) {
    int sourceKey = 0;
    int targetKey = 0;
    channelEdgeKeyForSource(desc, j, &sourceKey, &targetKey);
    if (edgePeerGroup(sourceKey, targetKey, peerGroupCount) == group) {
      count++;
    }
  }
  return count;
}

static int countTargetsForGroup(const StagingTransferDescriptor* desc, int group, int peerGroupCount) {
  int count = 0;
  for (int j = 0; j < desc->numTargets; j++) {
    int sourceKey = 0;
    int targetKey = 0;
    channelEdgeKeyForTarget(desc, j, &sourceKey, &targetKey);
    if (edgePeerGroup(sourceKey, targetKey, peerGroupCount) == group) {
      count++;
    }
  }
  return count;
}

static size_t maxPeersInChannelGroup(const StagingTransferDescriptor* desc, int peerGroupCount) {
  size_t maxPeers = 1;
  for (int group = 0; group < peerGroupCount; group++) {
    maxPeers = std::max(maxPeers, (size_t)countSourcesForGroup(desc, group, peerGroupCount));
    maxPeers = std::max(maxPeers, (size_t)countTargetsForGroup(desc, group, peerGroupCount));
  }
  return maxPeers;
}

static int targetRankInGroup(const StagingTransferDescriptor* desc, int targetIdx, int group, int peerGroupCount) {
  int rank = 0;
  for (int j = 0; j < targetIdx; j++) {
    int sourceKey = 0;
    int targetKey = 0;
    channelEdgeKeyForTarget(desc, j, &sourceKey, &targetKey);
    if (edgePeerGroup(sourceKey, targetKey, peerGroupCount) == group) {
      rank++;
    }
  }
  return rank;
}

struct StagingReceiverSourceLayout {
  size_t dataOffset;
  size_t dataSize;
  int numSlots;
  int sourceCount;
  int sourceRank;
};

static StagingReceiverSourceLayout stagingResolveReceiverSourceLayout(size_t regionStart, size_t regionSize,
                                                                       size_t chunkSize, int receiverSourceCount,
                                                                       int sourceIndexOnReceiver, int targetIndexOnSource,
                                                                       int peerGroupCount) {
  const int group = edgePeerGroup(sourceIndexOnReceiver, targetIndexOnSource, peerGroupCount);
  int sourceCount = receiverSourceCount;
  int sourceRank = sourceIndexOnReceiver;
  if (peerGroupCount > 1) {
    /* Sources assigned to a group are one residue class modulo
     * peerGroupCount. Compute its size and this source's ordinal directly;
     * this helper is called for every populated edge. */
    const int firstSource = positiveMod(group - targetIndexOnSource, peerGroupCount);
    sourceCount = (receiverSourceCount > firstSource) ?
                    1 + (receiverSourceCount - 1 - firstSource) / peerGroupCount :
                    0;
    sourceRank = (sourceIndexOnReceiver - firstSource) / peerGroupCount;
  }
  /* Whole slots only, so each source's slots start chunk-aligned. */
  const int numSlots = (sourceCount > 0 && chunkSize > 0) ? (int)(regionSize / (size_t)sourceCount / chunkSize) : 0;
  const size_t dataSize = (size_t)numSlots * chunkSize;
  return {regionStart + (size_t)sourceRank * dataSize, dataSize, numSlots, sourceCount, sourceRank};
}

static ncclResult_t stagingSetReceiverSourceLayout(StagingFlowCtrl* fc, size_t regionStart, size_t regionSize,
                                                    size_t chunkSize, int receiverSourceCount,
                                                    int sourceIndexOnReceiver, int targetIndexOnSource,
                                                    int peerGroupCount, int rank, const char* path) {
  const StagingReceiverSourceLayout layout = stagingResolveReceiverSourceLayout(
    regionStart, regionSize, chunkSize, receiverSourceCount, sourceIndexOnReceiver, targetIndexOnSource,
    peerGroupCount);
  NCCL_M2N_CHECK_ARG(layout.sourceCount > 0 && layout.sourceRank >= 0 && layout.sourceRank < layout.sourceCount &&
                       layout.numSlots >= 1,
                     rank,
                     "[STAGING] invalid receiver %s layout (sources=%d source=%d target=%d group=%d "
                     "group_sources=%d source_rank=%d data=%zu chunk=%zu)",
                     path, receiverSourceCount, sourceIndexOnReceiver, targetIndexOnSource,
                     edgePeerGroup(sourceIndexOnReceiver, targetIndexOnSource, peerGroupCount), layout.sourceCount,
                     layout.sourceRank, layout.dataSize, chunkSize);
  setFcDataRegion(fc, layout.dataOffset, layout.numSlots, chunkSize);
  return ncclSuccess;
}

static size_t getMaxPeerGroupSize(const StagingTransferDescriptor* desc) {
  size_t maxPeers = 1;
  if (desc->peerGroupSizeBound > 0) {
    maxPeers = std::max(maxPeers, (size_t)desc->peerGroupSizeBound);
  }
  if (desc->isDest && desc->numSources > 0) {
    maxPeers = std::max(maxPeers, (size_t)desc->numSources);
  }
  if (desc->isSource && desc->numTargets > 0) {
    maxPeers = std::max(maxPeers, (size_t)desc->numTargets);
  }
  for (int j = 0; j < desc->numTargets; j++) {
    if (desc->destNumSources[j] > 0) {
      maxPeers = std::max(maxPeers, (size_t)desc->destNumSources[j]);
    }
  }
  return maxPeers;
}

int stagingResolveNumChannelsForTransfer(const StagingTransferDescriptor* desc) {
  const ReshardStagingRuntimeConfig& config = reshardGetStagingRuntimeConfig();
  if (config.numChannelsFixed) {
    return clampNumChannels(config.numChannels);
  }
  if (config.hostRmaDefault) {
    if (desc == nullptr) {
      return clampNumChannels(config.numChannels);
    }
    const size_t maxPeerGroupSize = getMaxPeerGroupSize(desc);
    const int peerGroups = ceilDivInt((int)maxPeerGroupSize, std::max(1, config.peersPerChannel));
    return clampNumChannels(std::min(std::max(1, peerGroups), config.numChannels));
  }
  if (desc != nullptr && config.peersPerChannel > 0) {
    size_t maxPeerGroupSize = getMaxPeerGroupSize(desc);
    int peerGroupCount = ceilDivInt((int)maxPeerGroupSize, config.peersPerChannel);
    int requested = peerGroupCount;
    if (desc->ctaHeuristicPeerCount > 0) {
      int targetCtas = config.targetCtas;
      if (!config.targetCtasExplicit) {
        targetCtas = stagingDefaultTargetCtasForPeerGroups(peerGroupCount);
      }
      int requestedCtas = std::max(peerGroupCount, targetCtas);
      if (desc->hasLocalFanout && desc->maxEdgeBytes > 0 && config.chunkSize > 0 && config.channelDataSize > 0) {
        size_t dataRegionSize = stagingPipeUsesSharedDataRegion() ? config.channelDataSize : config.channelDataSize / 2;
        int maxChannelPeerCount = ceilDivInt((int)maxPeerGroupSize, peerGroupCount);
        size_t peerRegionSize = maxChannelPeerCount > 0 ? dataRegionSize / (size_t)maxChannelPeerCount : 0;
        size_t slotsPerPeer = peerRegionSize / config.chunkSize;
        if (slotsPerPeer > 0) {
          size_t maxEdgeChunks = 1 + (desc->maxEdgeBytes - 1) / config.chunkSize;
          int maxCtasPerPeerGroup = std::max(1, STAGING_MAX_CHANNELS / std::max(1, peerGroupCount));
          int creditSafeCtas = ceilDivSizeClamped(maxEdgeChunks, slotsPerPeer, maxCtasPerPeerGroup);
          requestedCtas = std::max(requestedCtas, peerGroupCount * creditSafeCtas);
        }
      }
      requested = requestedCtas;
    }
    return requested > STAGING_MAX_CHANNELS ? STAGING_MAX_CHANNELS : clampNumChannels(requested);
  }
  return clampNumChannels(STAGING_DEFAULT_NUM_CHANNELS);
}

/* ======================================================================
 * stagingPrepareTransfer
 * ====================================================================*/

ncclResult_t stagingPrepareTransfer(const StagingBufferState* state, const StagingTransferDescriptor* desc,
                                    ncclWindow_t rdmaWindow, ncclWindow_t lsaWindow, StagingKernelParams* params) {
  NCCL_M2N_CHECK_ARG(state != nullptr && state->initialized, -1,
                     "[STAGING] stagingPrepareTransfer called with uninitialized state");
  NCCL_M2N_CHECK_ARG(desc != nullptr, -1, "[STAGING] stagingPrepareTransfer called with null descriptor");
  NCCL_M2N_CHECK_ARG(params != nullptr, -1, "[STAGING] stagingPrepareTransfer called with null params");

  memset(params, 0, sizeof(*params));

  const int R = desc->myWorldRank;
  const int numChannels = state->numChannels;
  const size_t channelSize = state->channelSize;
  const size_t C = state->controlRegionSize;
  const int controlSlot = desc->controlSlot;
  NCCL_M2N_CHECK_ARG(state->chunkSize > 0 && channelSize > C, R,
                     "[STAGING] invalid staging sizes: channel=%zu ctrl=%zu chunk=%zu", channelSize, C,
                     state->chunkSize);
  NCCL_M2N_CHECK_ARG(state->controlSlotCount > 0 && controlSlot >= 0 && controlSlot < state->controlSlotCount, R,
                     "[STAGING] invalid controlSlot=%d for controlSlotCount=%d", controlSlot, state->controlSlotCount);
  const size_t channelDataSize = state->channelDataSize;
  const bool pipe = stagingPipeUsesSharedDataRegion();
  const size_t minChannelDataSize = pipe ? state->chunkSize : 2 * state->chunkSize;
  NCCL_M2N_CHECK_ARG(channelDataSize >= minChannelDataSize, R,
                     "[STAGING] channel data size %zu is too small for chunk size %zu", channelDataSize,
                     state->chunkSize);
  const size_t dataRegionSize = pipe ? channelDataSize : channelDataSize / 2;

  NCCL_M2N_CHECK_ARG(desc->numTargets >= 0 && desc->numTargets <= MAX_TARGETS && desc->numSources >= 0 &&
                       desc->numSources <= MAX_SOURCES,
                     R, "[STAGING] invalid peer counts: targets=%d/%d sources=%d/%d", desc->numTargets, MAX_TARGETS,
                     desc->numSources, MAX_SOURCES);
  NCCL_M2N_CHECK_ARG(stagingLsaFollowersFitKernelCapacity(desc->numLsaFollowers), R,
                     "[STAGING] LSA fan-out followers exceed kernel capacity: followers=%d max=%d",
                     desc->numLsaFollowers, STAGING_LSA_FANOUT_MAX_FOLLOWERS);
  int rdmaSourceCount = 0;
  int lsaTargetCount = 0;
  for (int j = 0; j < desc->numTargets; j++) {
    if (!desc->targets[j].isRdma) {
      lsaTargetCount++;
    }
  }
  if (desc->isDest) {
    for (int j = 0; j < desc->numSources; j++) {
      if (desc->sources[j].isRdma) {
        rdmaSourceCount++;
      }
    }
  }
  NCCL_M2N_CHECK_ARG(stagingLsaFanoutFitsTargetCapacity(rdmaSourceCount, desc->numLsaFollowers), R,
                     "[STAGING] LSA fan-out target count exceeds staging target capacity: rdmaSources=%d "
                     "followers=%d max=%d",
                     rdmaSourceCount, desc->numLsaFollowers, MAX_TARGETS);
  NCCL_M2N_CHECK_ARG(stagingLsaFanoutHasTargetDescriptors(rdmaSourceCount, desc->numLsaFollowers, lsaTargetCount), R,
                     "[STAGING] LSA fan-out target list incomplete: rdmaSources=%d followers=%d lsaTargets=%d",
                     rdmaSourceCount, desc->numLsaFollowers, lsaTargetCount);

  const size_t maxPeerGroupSize = getMaxPeerGroupSize(desc);
  // Only PIPE consumes the compact peer-to-channel map. DIRECT needs every
  // channel populated for its dense channel-major staging map.
  const bool groupPeersByChannel = reshardGetCopyAlgorithm() == RESHARD_COPY_ALGO_PIPE && state->peersPerChannel > 0;
  int peerGroupCount = 1;
  size_t maxChannelPeerCount = maxPeerGroupSize;
  if (groupPeersByChannel) {
    const int peerGroups = ceilDivInt((int)maxPeerGroupSize, state->peersPerChannel);
    NCCL_M2N_CHECK_ARG(peerGroups > 0, R, "[STAGING] invalid peerGroupCount=0 for max peers %zu", maxPeerGroupSize);
    const int ginChannelsPerPeer = std::max(1, desc->pipeGinChannelsPerPeer);
    const int requestedPeerGroupCount = std::max(peerGroups, ceilDivInt(numChannels, ginChannelsPerPeer));
    if (requestedPeerGroupCount > numChannels) {
      STAGING_LOG(R,
                  "peersPerChannel=%d requests %d peer groups for %zu peers, capping to numChannels=%d",
                  state->peersPerChannel, requestedPeerGroupCount, maxPeerGroupSize, numChannels);
    }
    peerGroupCount = std::min(numChannels, requestedPeerGroupCount);
    maxChannelPeerCount =
      std::max((size_t)ceilDivInt((int)maxPeerGroupSize, peerGroupCount), maxPeersInChannelGroup(desc, peerGroupCount));
  }

  const size_t maxChunkSize = dataRegionSize / maxChannelPeerCount;
  NCCL_M2N_CHECK_ARG(maxChunkSize > 0, R, "[STAGING] channel data region too small (%zu B / %zu peers)", dataRegionSize,
                     maxChannelPeerCount);
  /* PIPE chunk boundaries drive peer-visible signals and must not depend on
   * rank-local fanout. Receiver FIFO validation below rejects insufficient capacity. */
  const size_t chunkSize = pipe ? state->chunkSize : std::min(state->chunkSize, maxChunkSize);
  /* LSA followers bulk-copy whole 16-byte granules from 16-byte-aligned slots.
   * Check on every PIPE rank, not only on followers: a rejection on some ranks
   * would leave their peers waiting in the kernel. */
  NCCL_M2N_CHECK_ARG(!pipe || (chunkSize % 16 == 0 && state->channelSize % 16 == 0 &&
                               state->channelDataSize % 16 == 0 && state->controlRegionSize % 16 == 0),
                     R, "[STAGING] PIPE staging sizes must be multiples of 16 bytes (chunk=%zu channel=%zu)", chunkSize,
                     state->channelSize);
  if (!pipe && chunkSize != state->chunkSize) {
    STAGING_LOG(R,
                "  chunkSize adjusted: requested=%zu effective=%zu max_peer_group=%zu "
                "max_channel_peer_group=%zu data_region=%zu peersPerChannel=%d peerGroups=%d",
                state->chunkSize, chunkSize, maxPeerGroupSize, maxChannelPeerCount, dataRegionSize,
                state->peersPerChannel, peerGroupCount);
  }

  STAGING_LOG(R,
              "stagingPrepareTransfer() ENTRY is_src=%d is_dst=%d "
              "numTargets=%d numSources=%d numChannels=%d controlSlot=%d/%d peersPerChannel=%d peerGroups=%d",
              desc->isSource, desc->isDest, desc->numTargets, desc->numSources, numChannels, controlSlot,
              state->controlSlotCount, state->peersPerChannel, peerGroupCount);

  /* ----------------------------------------------------------------
   * 1. Classify peers into RDMA / LSA buckets.
   * ---------------------------------------------------------------- */
  int numRdmaTargets = 0;
  int numLsaTargets = 0;
  int numRdmaSources = 0;
  int numLsaSources = 0;

  int targetRdmaIdx[MAX_TARGETS];
  int targetLsaIdx[MAX_TARGETS];
  int sourceRdmaIdx[MAX_SOURCES];
  int sourceLsaIdx[MAX_SOURCES];

  memset(targetRdmaIdx, -1, sizeof(targetRdmaIdx));
  memset(targetLsaIdx, -1, sizeof(targetLsaIdx));
  memset(sourceRdmaIdx, -1, sizeof(sourceRdmaIdx));
  memset(sourceLsaIdx, -1, sizeof(sourceLsaIdx));

  for (int j = 0; j < desc->numTargets; j++) {
    if (desc->targets[j].isRdma) {
      targetRdmaIdx[j] = numRdmaTargets++;
    } else {
      targetLsaIdx[j] = numLsaTargets++;
    }
  }

  if (desc->isDest) {
    for (int j = 0; j < desc->numSources; j++) {
      if (desc->sources[j].isRdma) {
        sourceRdmaIdx[j] = numRdmaSources++;
      } else {
        sourceLsaIdx[j] = numLsaSources++;
      }
    }
  }

  STAGING_LOG(R, "  classify: rdma_t=%d lsa_t=%d rdma_s=%d lsa_s=%d", numRdmaTargets, numLsaTargets, numRdmaSources,
              numLsaSources);

  /* ----------------------------------------------------------------
   * 2. Signal/counter budgets.  GIN is RDMA-only: LSA fanout uses the
   * staging-buffer control region, so it must not inflate this stride.
   * ---------------------------------------------------------------- */
  int numRdmaPeers = std::max(numRdmaTargets, numRdmaSources);
  StagingRdmaSignalLayout rdmaSignalLayout{};
  if (desc->pipeGinPeerCapacity != 0 || desc->pipeGinChannelsPerPeer != 0) {
    NCCL_M2N_CHECK_ARG(desc->pipeGinPeerCapacity > 0 && desc->pipeGinChannelsPerPeer > 0, R,
                       "[STAGING] PIPE GIN map requires positive peer and channel capacities (peers=%d channels=%d)",
                       desc->pipeGinPeerCapacity, desc->pipeGinChannelsPerPeer);
    NCCL_M2N_CHECK_ARG(numRdmaPeers <= desc->pipeGinPeerCapacity, R,
                       "[STAGING] PIPE RDMA peer count %d exceeds GIN peer capacity %d", numRdmaPeers,
                       desc->pipeGinPeerCapacity);
    rdmaSignalLayout = {true, desc->pipeGinChannelsPerPeer};
  }
  const int ginPeerCount = rdmaSignalLayout.dense ? desc->pipeGinPeerCapacity : numRdmaPeers;
  const int ginChannelsPerPeer = rdmaSignalLayout.dense ? rdmaSignalLayout.channelsPerPeer : numChannels;
  const size_t ginCounterCount = (size_t)ginChannelsPerPeer * (size_t)ginPeerCount;
  NCCL_M2N_CHECK_ARG(ginCounterCount <= (size_t)std::numeric_limits<int>::max() / 2U, R,
                     "[STAGING] GIN map exceeds NCCL int capacity (peers=%d channels=%d)", ginPeerCount,
                     ginChannelsPerPeer);
  const size_t ginSignalCount = ginCounterCount * 2U;

  /* ----------------------------------------------------------------
   * 3. Top-level params.
   * ---------------------------------------------------------------- */
  params->numChannels = numChannels;
  params->myRank = desc->myWorldRank;
  params->myLocalRank = desc->myLocalRank;
  params->isSource = desc->isSource;
  params->isDest = desc->isDest;
  params->srcBuffer = desc->srcBuffer;
  params->dstBuffer = desc->dstBuffer;
  params->stagingBuffer = state->buffer;
  params->rdmaWindow = rdmaWindow;
  params->rdmaWindowB = rdmaWindow;
  params->lsaWindow = lsaWindow;
  params->chunkSize = chunkSize;
  params->ginSignalCount = (int)ginSignalCount;
  params->ginCounterCount = (int)ginCounterCount;
  params->pipeReadySlotOffset = (size_t)controlSlot * STAGING_CTRL_REGION_SIZE;
  params->pipeReadyChannelStride = state->adaptiveChannelLayout ? state->controlRegionSize : state->channelSize;
  params->numRdmaTargets = numRdmaTargets;
  params->numLsaTargets = numLsaTargets;
  params->numRdmaSources = numRdmaSources;
  params->numLsaSources = numLsaSources;
  params->numLsaFollowers = desc->numLsaFollowers;
  params->numRingTargets = desc->numRingTargets;
  params->ndims = desc->ndims;

  for (int d = 0; d < desc->ndims; d++) {
    params->srcDims[d] = desc->srcDims[d];
    params->dstDims[d] = desc->dstDims[d];
    params->srcStrides[d] = desc->srcStrides[d];
    params->dstStrides[d] = desc->dstStrides[d];
  }

  /* ----------------------------------------------------------------
   * 4. Per-source / per-target sub-region sizing.
   * ---------------------------------------------------------------- */
  const size_t groupedPeerSize = groupPeersByChannel ? dataRegionSize / maxChannelPeerCount : 0;
  int sourceNumSlots = (int)(dataRegionSize / chunkSize);

  /* ----------------------------------------------------------------
   * 5. Per-channel descriptors.
   * ---------------------------------------------------------------- */
  for (int ch = 0; ch < numChannels; ch++) {
    size_t channelBase = stagingChannelControlBase(state, ch);
    size_t ctrlSlotBase = channelBase + (size_t)controlSlot * STAGING_CTRL_REGION_SIZE;
    size_t rdmaRegionStart = stagingChannelDataBase(state, ch);
    size_t lsaRegionStart = pipe ? rdmaRegionStart : rdmaRegionStart + dataRegionSize;
    int peerGroup = groupPeersByChannel ? channelPeerGroup(ch, peerGroupCount) : 0;
    int channelPeerRank = groupPeersByChannel ? channelRankInPeerGroup(ch, peerGroupCount) : ch;
    int channelPeerCount =
      groupPeersByChannel ? channelsInPeerGroup(numChannels, peerGroupCount, peerGroup) : numChannels;
    if (rdmaSignalLayout.dense && numRdmaPeers > 0) {
      NCCL_M2N_CHECK_ARG(channelPeerCount <= rdmaSignalLayout.channelsPerPeer, R,
                         "[STAGING] PIPE channels per peer group %d exceeds GIN capacity %d", channelPeerCount,
                         rdmaSignalLayout.channelsPerPeer);
    }

    int activeTargetCount =
      groupPeersByChannel ? countTargetsForGroup(desc, peerGroup, peerGroupCount) : desc->numTargets;
    size_t perTargetSize = 0;
    int perTargetSlots = 0;
    if (desc->isSource && desc->numTargets > 0 && (!pipe || numLsaTargets > 0)) {
      perTargetSize = groupPeersByChannel ? groupedPeerSize : dataRegionSize / desc->numTargets;
      perTargetSlots = (int)(perTargetSize / chunkSize);
      NCCL_M2N_CHECK_ARG(perTargetSlots >= 1, R,
                         "[STAGING] per-target sub-region too small (%zu B / %d active targets, chunk=%zu)",
                         perTargetSize, activeTargetCount, chunkSize);
    }

    int rdmaTargetRanks[MAX_TARGETS];
    int activeRdmaTargetCount = 0;
    size_t rdmaTargetSize = 0;
    int rdmaTargetSlots = 0;
    if (pipe) {
      memset(rdmaTargetRanks, -1, sizeof(rdmaTargetRanks));
      if (desc->isSource && numRdmaTargets > 0) {
        for (int j = 0; j < desc->numTargets; j++) {
          if (!desc->targets[j].isRdma) {
            continue;
          }
          int sourceKey = 0;
          int targetKey = 0;
          channelEdgeKeyForTarget(desc, j, &sourceKey, &targetKey);
          if (groupPeersByChannel && !channelHasEdge(ch, sourceKey, targetKey, peerGroupCount)) {
            continue;
          }
          rdmaTargetRanks[j] = activeRdmaTargetCount++;
        }
      }
      rdmaTargetSize = activeRdmaTargetCount > 0 ? dataRegionSize / (size_t)activeRdmaTargetCount : 0;
      rdmaTargetSlots = activeRdmaTargetCount > 0 ? (int)(rdmaTargetSize / chunkSize) : 0;
      NCCL_M2N_CHECK_ARG(activeRdmaTargetCount == 0 || rdmaTargetSlots >= 1, R,
                         "[STAGING] local RDMA sub-region too small (%zu B / %d active RDMA targets, chunk=%zu)",
                         rdmaTargetSize, activeRdmaTargetCount, chunkSize);
    }

    StagingRegion* rdmaReg = &params->rdmaRegions[ch];
    rdmaReg->dataOffset = rdmaRegionStart;
    rdmaReg->regionSize = dataRegionSize;
    rdmaReg->chunkSize = chunkSize;
    rdmaReg->numSlots = sourceNumSlots;
    rdmaReg->window = rdmaWindow;

    StagingRegion* lsaReg = &params->lsaRegions[ch];
    lsaReg->dataOffset = lsaRegionStart;
    lsaReg->regionSize = dataRegionSize;
    lsaReg->chunkSize = chunkSize;
    lsaReg->numSlots = sourceNumSlots;
    lsaReg->window = pipe ? rdmaWindow : lsaWindow;

    /* ---------- 5a. Dest-side per-source flow control ---------- */
    if (desc->isDest) {
      for (int j = 0; j < desc->numSources; j++) {
        if (!desc->sources[j].isRdma) {
          continue;
        }
        int sourceKey = 0;
        int targetKey = 0;
        channelEdgeKeyForSource(desc, j, &sourceKey, &targetKey);
        if (groupPeersByChannel && !channelHasEdge(ch, sourceKey, targetKey, peerGroupCount)) {
          continue;
        }
        int rdmaJ = sourceRdmaIdx[j];

        StagingPeerInfo* pi = &params->rdmaSources[ch][rdmaJ];
        pi->peerWorldRank = desc->sources[j].peerWorldRank;
        pi->peerShardIdx = desc->sources[j].peerShardIdx;
        pi->peerLocalRank = -1;
        pi->rdmaTransport = STAGING_RDMA_TRANSPORT_PARENT;
        pi->plan = desc->sources[j].plan;
        pi->active = true;
        pi->channelRank = channelPeerRank;
        pi->channelCount = channelPeerCount;
        setPeerChunkRange(pi, chunkSize);

        StagingFlowCtrl* fc = &pi->fc;
        initFlowCtrl(fc);
        fc->remoteRank = desc->sources[j].peerWorldRank;

        NCCL_M2N_CHECK(stagingSetReceiverSourceLayout(fc, rdmaRegionStart, dataRegionSize, chunkSize,
                                                       desc->numSources, sourceKey, targetKey, peerGroupCount, R,
                                                       "RDMA"));

        int myTargetIdxOnSource = desc->rdmaTargetIndexOnSource[j];
        int srcNumTargets = desc->sourceNumRdmaTargets[j];
        NCCL_M2N_CHECK_ARG(myTargetIdxOnSource >= 0 && myTargetIdxOnSource < srcNumTargets, R,
                           "[STAGING] invalid remote RDMA target ordinal %d/%d for source %d", myTargetIdxOnSource,
                           srcNumTargets, j);

        setFcRdmaSignals(fc, rdmaSignalLayout, ch, channelPeerRank, numRdmaSources, rdmaJ, myTargetIdxOnSource,
                         srcNumTargets);
        fc->cursorHeadOffset = ctrlSlotBase + (size_t)j * STAGING_CTRL_ENTRY_SIZE + CTRL_FIELD_CURSOR_HEAD;
      }

      for (int j = 0; j < desc->numSources; j++) {
        if (desc->sources[j].isRdma) {
          continue;
        }
        int sourceKey = 0;
        int targetKey = 0;
        channelEdgeKeyForSource(desc, j, &sourceKey, &targetKey);
        if (groupPeersByChannel && !channelHasEdge(ch, sourceKey, targetKey, peerGroupCount)) {
          continue;
        }
        int lsaJ = sourceLsaIdx[j];

        StagingPeerInfo* pi = &params->lsaSources[ch][lsaJ];
        pi->peerWorldRank = desc->sources[j].peerWorldRank;
        pi->peerShardIdx = desc->sources[j].peerShardIdx;
        pi->peerLocalRank = desc->sources[j].peerLocalRank;
        pi->plan = desc->sources[j].plan;
        pi->active = true;
        pi->channelRank = channelPeerRank;
        pi->channelCount = channelPeerCount;
        setPeerChunkRange(pi, chunkSize);

        StagingFlowCtrl* fc = &pi->fc;
        initFlowCtrl(fc);
        fc->remoteRank = desc->sources[j].peerLocalRank;

        NCCL_M2N_CHECK(stagingSetReceiverSourceLayout(fc, lsaRegionStart, dataRegionSize, chunkSize,
                                                       desc->numSources, sourceKey, targetKey, peerGroupCount, R,
                                                       "LSA"));

        int sourceLsaHeadIdxForMe = desc->sourceLsaHeadIndexOnProvider[j];
        setFcLsaConsumer(fc, ctrlSlotBase, j, sourceLsaHeadIdxForMe);
        fc->cursorHeadOffset = ctrlSlotBase + (size_t)j * STAGING_CTRL_ENTRY_SIZE + CTRL_FIELD_CURSOR_HEAD;
      }
    }

    /* ---------- 5b. Per-target flow control (src + dst fan-out) ---------- */
    if (desc->numTargets > 0) {
      for (int j = 0; j < desc->numTargets; j++) {
        if (!desc->targets[j].isRdma) {
          continue;
        }
        int sourceKey = 0;
        int targetKey = 0;
        channelEdgeKeyForTarget(desc, j, &sourceKey, &targetKey);
        int mySrcIdxOnDest = desc->sourceIndexOnDest[j];
        if (groupPeersByChannel && !channelHasEdge(ch, sourceKey, targetKey, peerGroupCount)) {
          continue;
        }
        int rdmaJ = targetRdmaIdx[j];

        StagingPeerInfo* pi = &params->rdmaTargets[ch][rdmaJ];
        pi->peerWorldRank = desc->targets[j].peerWorldRank;
        pi->peerShardIdx = desc->targets[j].peerShardIdx;
        pi->peerLocalRank = -1;
        pi->rdmaTransport = STAGING_RDMA_TRANSPORT_PARENT;
        pi->plan = desc->targets[j].plan;
        pi->active = true;
        pi->channelRank = channelPeerRank;
        pi->channelCount = channelPeerCount;
        setPeerChunkRange(pi, chunkSize);

        StagingFlowCtrl* fc = &pi->fc;
        initFlowCtrl(fc);
        fc->remoteRank = desc->targets[j].peerWorldRank;

        NCCL_M2N_CHECK(stagingSetReceiverSourceLayout(fc, rdmaRegionStart, dataRegionSize, chunkSize,
                                                       desc->destNumSources[j], mySrcIdxOnDest, targetKey,
                                                       peerGroupCount, R, "RDMA"));

        int rdmaSrcIdxOnDest = desc->rdmaSourceIndexOnDest[j];
        int destNumRdmaSrc = desc->destNumRdmaSources[j];
        NCCL_M2N_CHECK_ARG(rdmaSrcIdxOnDest >= 0 && rdmaSrcIdxOnDest < destNumRdmaSrc, R,
                           "[STAGING] invalid remote RDMA source ordinal %d/%d for target %d", rdmaSrcIdxOnDest,
                           destNumRdmaSrc, j);
        setFcRdmaSignals(fc, rdmaSignalLayout, ch, channelPeerRank, numRdmaTargets, rdmaJ, rdmaSrcIdxOnDest,
                         destNumRdmaSrc);
        fc->localPutCounter = stagingRdmaCounterId(rdmaSignalLayout, ch, channelPeerRank, rdmaJ, numRdmaTargets);
        fc->cursorTailOffset =
          ctrlSlotBase + (size_t)(STAGING_LOCAL_FC_BASE + j) * STAGING_CTRL_ENTRY_SIZE + CTRL_FIELD_CURSOR_TAIL;
      }

      for (int j = 0; j < desc->numTargets; j++) {
        if (desc->targets[j].isRdma) {
          continue;
        }
        int sourceKey = 0;
        int targetKey = 0;
        channelEdgeKeyForTarget(desc, j, &sourceKey, &targetKey);
        int mySrcIdxOnDest = desc->sourceIndexOnDest[j];
        if (groupPeersByChannel && !channelHasEdge(ch, sourceKey, targetKey, peerGroupCount)) {
          continue;
        }
        int lsaJ = targetLsaIdx[j];

        StagingPeerInfo* pi = &params->lsaTargets[ch][lsaJ];
        pi->peerWorldRank = desc->targets[j].peerWorldRank;
        pi->peerShardIdx = desc->targets[j].peerShardIdx;
        pi->peerLocalRank = desc->targets[j].peerLocalRank;
        pi->plan = desc->targets[j].plan;
        pi->active = true;
        pi->channelRank = channelPeerRank;
        pi->channelCount = channelPeerCount;
        setPeerChunkRange(pi, chunkSize);

        StagingFlowCtrl* fc = &pi->fc;
        initFlowCtrl(fc);
        fc->remoteRank = desc->targets[j].peerLocalRank;

        NCCL_M2N_CHECK(stagingSetReceiverSourceLayout(fc, lsaRegionStart, dataRegionSize, chunkSize,
                                                       desc->destNumSources[j], mySrcIdxOnDest, targetKey,
                                                       peerGroupCount, R, "LSA"));

        setFcLsaProducer(fc, ctrlSlotBase, lsaJ, mySrcIdxOnDest);
        fc->cursorTailOffset =
          ctrlSlotBase + (size_t)(STAGING_LOCAL_FC_BASE + j) * STAGING_CTRL_ENTRY_SIZE + CTRL_FIELD_CURSOR_TAIL;
      }
    }

    /* ---------- 5c. Per-target local pipeline FC (source side only) ---------- */
    if (desc->isSource) {
      for (int j = 0; j < desc->numTargets; j++) {
        int sourceKey = 0;
        int targetKey = 0;
        channelEdgeKeyForTarget(desc, j, &sourceKey, &targetKey);
        if (groupPeersByChannel && !channelHasEdge(ch, sourceKey, targetKey, peerGroupCount)) {
          continue;
        }
        int ctrlIdx = STAGING_LOCAL_FC_BASE + j;

        if (desc->targets[j].isRdma) {
          int rdmaJ = targetRdmaIdx[j];
          int targetRank = 0;
          size_t targetSize = 0;
          int targetSlots = 0;
          if (pipe) {
            targetRank = rdmaTargetRanks[j];
            NCCL_M2N_CHECK_ARG(targetRank >= 0 && targetRank < activeRdmaTargetCount, R,
                               "[STAGING] invalid local RDMA target layout (rank=%d count=%d)", targetRank,
                               activeRdmaTargetCount);
            targetSize = rdmaTargetSize;
            targetSlots = rdmaTargetSlots;
          } else {
            targetRank = groupPeersByChannel ? targetRankInGroup(desc, j, peerGroup, peerGroupCount) : j;
            NCCL_M2N_CHECK_ARG((size_t)targetRank < maxChannelPeerCount, R,
                               "[STAGING] local target rank %d exceeds grouped peer slots %zu", targetRank,
                               maxChannelPeerCount);
            targetSize = perTargetSize;
            targetSlots = perTargetSlots;
          }
          StagingFlowCtrl* lrf = &params->localRdmaFc[ch][rdmaJ];
          initFlowCtrl(lrf);

          size_t targetDataOff = rdmaRegionStart + (size_t)targetRank * targetSize;
          setFcLocalPipeline(lrf, desc->myWorldRank, ctrlSlotBase, ctrlIdx, targetDataOff, targetSlots, chunkSize,
                             CTRL_FIELD_RDMA_TAIL, CTRL_FIELD_RDMA_HEAD);
          lrf->localPutCounter = stagingRdmaCounterId(rdmaSignalLayout, ch, channelPeerRank, rdmaJ, numRdmaTargets);
        } else {
          int targetRank = groupPeersByChannel ? targetRankInGroup(desc, j, peerGroup, peerGroupCount) : j;
          NCCL_M2N_CHECK_ARG((size_t)targetRank < maxChannelPeerCount, R,
                             "[STAGING] local target rank %d exceeds grouped peer slots %zu", targetRank,
                             maxChannelPeerCount);
          int lsaJ = targetLsaIdx[j];
          StagingFlowCtrl* llf = &params->localLsaFc[ch][lsaJ];
          initFlowCtrl(llf);

          size_t targetDataOff = lsaRegionStart + (size_t)targetRank * perTargetSize;
          setFcLocalPipeline(llf, desc->myWorldRank, ctrlSlotBase, ctrlIdx, targetDataOff, perTargetSlots, chunkSize,
                             CTRL_FIELD_LSA_TAIL, CTRL_FIELD_LSA_HEAD);
        }
      }
    }

    if (pipe) {
      int activeRdmaTargets = 0;
      int activeRdmaSources = 0;
      for (int t = 0; t < numRdmaTargets; t++) {
        activeRdmaTargets += params->rdmaTargets[ch][t].active ? 1 : 0;
      }
      for (int s = 0; s < numRdmaSources; s++) {
        activeRdmaSources += params->rdmaSources[ch][s].active ? 1 : 0;
      }
      NCCL_M2N_CHECK_ARG(activeRdmaTargets <= 1 && activeRdmaSources <= 1, R,
                         "[STAGING] PIPE channel %d has multiple RDMA peers (targets=%d sources=%d)", ch,
                         activeRdmaTargets, activeRdmaSources);
    }
  }

  STAGING_LOG(R,
              "stagingPrepareTransfer() EXIT (success): "
              "rdma_t=%d lsa_t=%d rdma_s=%d lsa_s=%d gin_sig=%d gin_cnt=%d ring_t=%d",
              numRdmaTargets, numLsaTargets, numRdmaSources, numLsaSources, ginSignalCount, ginCounterCount,
              desc->numRingTargets);
  return ncclSuccess;
}

/* ======================================================================
 * stagingBufferFinalize
 * ====================================================================*/

ncclResult_t stagingBufferFinalize(StagingBufferState* state) {
  if (!state || !state->initialized) {
    return ncclSuccess;
  }

  STAGING_LOG(-1, "stagingBufferFinalize() ENTRY buffer=%p total=%zu", state->buffer, state->totalSize);

  if (state->buffer) {
    STAGING_NCCLCHECK(ncclMemFree(state->buffer));
    state->buffer = nullptr;
  }
  if (state->devParams) {
    STAGING_CUDACHECK(cudaFree(state->devParams));
    state->devParams = nullptr;
  }
  stagingPipeHostRmaPipelineDestroy(state->hostRmaPipeline);
  state->hostRmaPipeline = nullptr;
  for (int i = 0; i < STAGING_PIPE_CONTROL_SLOTS; i++) {
    StagingPipePlanCacheEntry& entry = state->pipePlanCache[i];
    delete entry.hostParams;
    entry.hostParams = nullptr;
    delete entry.hostPlan;
    entry.hostPlan = nullptr;
    if (entry.devParams) {
      STAGING_CUDACHECK(cudaFree(entry.devParams));
      entry.devParams = nullptr;
    }
    if (entry.devPlan) {
      STAGING_CUDACHECK(cudaFree(entry.devPlan));
      entry.devPlan = nullptr;
    }
    entry = {};
  }

  state->totalSize = 0;
  state->numChannels = 0;
  state->capacityChannels = 0;
  state->channelSize = 0;
  state->channelDataSize = 0;
  state->dataCapacity = 0;
  state->controlSlotCount = 0;
  state->controlRegionSize = 0;
  state->chunkSize = 0;
  state->peersPerChannel = 0;
  state->adaptiveChannelLayout = false;
  state->pipePlanCacheNextVictim = 0;
  state->initialized = false;
  return ncclSuccess;
}
