/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

/*************************************************************************
 * staging_buffer.h — host-side API for the staging-buffer infrastructure
 * that backs ncclReshard.
 *
 *   stagingBufferInit       — allocate a symmetric staging pool with
 *                             ncclMemAlloc + zero its control region (NOT
 *                             collective; window registration is the
 *                             caller's responsibility, see reshard_cache).
 *   stagingPrepareTransfer  — translate a host-side
 *                             StagingTransferDescriptor into a packed
 *                             StagingKernelParams ready to upload.
 *   stagingBufferFinalize   — free the pool + matching device-side params
 *                             buffer.
 *
 * No MPI dependency.  Logging picks up `desc->myWorldRank` so we can
 * tag traces without going through MPI_Comm_rank.
 ************************************************************************/

#ifndef NCCL_STAGING_BUFFER_H_
#define NCCL_STAGING_BUFFER_H_

#include "nccl.h"
#include "reshard_split.h"
#include "staging_types.h"

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

struct StagingPipePlanCacheEntry {
  ReshardStagingPipeSignature signature;

  /* Host/device copies of the immutable PIPE launch metadata.  Runtime
   * values that change every call stay in StagingPipeCallParams and are
   * passed as kernel arguments instead of being copied into these blocks. */
  StagingKernelParams* hostParams;
  StagingPipeDevicePlan* hostPlan;
  StagingKernelParams* devParams;
  StagingPipeDevicePlan* devPlan;

  bool hostValid;
  bool deviceValid;
};

/* ======================================================================
 * Host-side handle returned by stagingBufferInit().
 *
 * Holds the staging buffer allocation and a lazily allocated device-side
 * StagingKernelParams scratch slot for DIRECT. PIPE uses the fixed
 * associative plan cache below so alternating mesh/tensor shapes do not rebuild or
 * re-upload static metadata after warmup.
 * Window registrations live in reshard_cache so that each (comm,
 * staging_buffer) pair gets exactly one collective registration.
 * ====================================================================*/
struct StagingBufferState {
  void* buffer;
  size_t totalSize;
  int numChannels;
  int capacityChannels;
  size_t channelSize;
  size_t channelDataSize;
  size_t dataCapacity;
  int controlSlotCount;
  size_t controlRegionSize;
  size_t chunkSize;
  int peersPerChannel;
  bool adaptiveChannelLayout;
  bool initialized;

  StagingKernelParams* devParams; /* cudaMalloc'd lazily; DIRECT uses it as scratch. */
  void* hostRmaPipeline; /* Host-RMA CUDA streams/events, retained across plans. */

  /* PIPE plan entries are associative. Persistent-control slots isolate
   * cursor/counter state; plans use the same rank-uniform transfer identity
   * so every endpoint selects the same GIN control-slot offset. */
  StagingPipePlanCacheEntry pipePlanCache[STAGING_PIPE_CONTROL_SLOTS];
  int pipePlanCacheNextVictim;
};

/* Toggle [STAGING] verbose logging at runtime (parallels
 * NCCL_RESHARD_LOG_LEVEL=DEBUG). */
void stagingSetVerbose(bool verbose);

/* Resolve the active staging channel count for a transfer. A fixed channel
 * count (an explicit override) is returned directly. Host-RMA defaults cap
 * active peer lanes while retaining one physical pool allocation.
 * Otherwise, when peersPerChannel is set, the result is derived from the
 * transfer peer groups. Descriptors may also set ctaHeuristicPeerCount to
 * enable a peer-count-aware CTA heuristic:
 * small peer-group counts get multiple CTAs per peer group, then taper to
 * one CTA per peer group as peer-group count grows. Explicit
 * NCCL_RESHARD_STAGING_TARGET_CTAS still overrides the default heuristic. */
int stagingResolveNumChannelsForTransfer(const StagingTransferDescriptor* desc);

/* Allocate the staging pool. DIRECT and device PIPE lazily allocate their
 * device-only plan/scratch slots.
 * NOT collective. Sizes default from reshard_limits.h, with internal
 * staging-pool overrides read during initialization. */
ncclResult_t stagingBufferInit(StagingBufferState* state);
ncclResult_t stagingBufferInitWithNumChannels(StagingBufferState* state, int numChannels);
ncclResult_t stagingBufferInitWithNumChannelsAndControlSlots(StagingBufferState* state, int numChannels,
                                                             int controlSlotCount);

/* Destroys Host-RMA CUDA scheduling resources attached to a staging pool. */
void stagingPipeHostRmaPipelineDestroy(void* pipeline);

/* Select the active channel layout within an initialized pool. Default
 * Host-RMA pools retain a fixed total data capacity and repartition it over
 * active peer lanes; all other paths retain fixed per-channel capacity. */
ncclResult_t stagingBufferConfigureActiveChannels(StagingBufferState* state, int numChannels);

/* Translate a host descriptor into device-ready kernel params.  The
 * caller is expected to have already registered both windows on the
 * staging buffer (see ncclReshard host entry). */
ncclResult_t stagingPrepareTransfer(const StagingBufferState* state, const StagingTransferDescriptor* desc,
                                    ncclWindow_t rdmaWindow, ncclWindow_t lsaWindow, StagingKernelParams* params);

/* Free the staging buffer + devParams slot.  Window deregistration is
 * driven by the global reshard_cache teardown. */
ncclResult_t stagingBufferFinalize(StagingBufferState* state);

#ifdef __cplusplus
}
#endif

#endif /* NCCL_STAGING_BUFFER_H_ */
