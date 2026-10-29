/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

/*************************************************************************
 * staging_types.h — device + host-shared types for the copy/staging
 * resharding path (the engine behind ncclReshard).
 *
 * Every type here is consumed by both staging_buffer.cc / staging_prepare.cc
 * (host) and staging_kernel.cu (device).  Anything that is purely host-side
 * lives in staging_buffer.h; anything device-only lives in
 * staging_primitives.cuh.
 *
 * The protocol is a per-channel, credit-based ring of fixed-size chunks.
 * Producers (source ranks) push payload + signal a tail; consumers (dest
 * ranks) drain the ring + signal a head to release credits.  Two
 * independent channels exist per CTA: one over RDMA windows for inter-node
 * peers, one over LSA windows for intra-node peers.  See the design doc
 * for the full state machine.
 ************************************************************************/

#ifndef NCCL_STAGING_TYPES_H_
#define NCCL_STAGING_TYPES_H_

#include "nccl.h"

#include "nccl_m2n.h"
#include "reshard_limits.h"

#include <cstddef>
#include <cstdint>

/* We deliberately do NOT pull in nccl_device.h here.  This header is
 * shared between host TUs (staging_buffer.cc, staging_prepare.cc) and
 * device TUs (staging_kernel.cu, reshard_staging.cu).  nccl_device.h
 * defines __device__ always-inline helpers that g++ refuses to inline
 * when compiling pure host TUs, which broke the library link.
 *
 * The only nccl_device symbol this header needs is the trivial
 * ncclGinSignal_t typedef (uint32_t).  Mirroring it locally is safe:
 * the real typedef in nccl_device/core.h is identical (also
 * `typedef uint32_t ncclGinSignal_t;`), and C++11+ permits multiple
 * identical typedefs in the same translation unit. */
typedef uint32_t ncclGinSignal_t;

constexpr inline bool stagingLsaFollowersFitKernelCapacity(int numLsaFollowers) {
  return numLsaFollowers >= 0 && numLsaFollowers <= STAGING_LSA_FANOUT_MAX_FOLLOWERS;
}

constexpr inline bool stagingLsaFanoutFitsTargetCapacity(int numRdmaSources, int numLsaFollowers) {
  return numRdmaSources <= 0 || numLsaFollowers <= MAX_TARGETS / numRdmaSources;
}

constexpr inline bool stagingLsaFanoutHasTargetDescriptors(int numRdmaSources, int numLsaFollowers, int numLsaTargets) {
  return numRdmaSources <= 0 || numLsaFollowers <= 0 || numLsaTargets >= numRdmaSources * numLsaFollowers;
}

#ifdef __cplusplus
extern "C" {
#endif

enum {
  STAGING_RDMA_TRANSPORT_PARENT = 0,
  STAGING_RDMA_TRANSPORT_SPLIT_A = 1,
  STAGING_RDMA_TRANSPORT_SPLIT_B = 2,
};

/* ======================================================================
 * Flow control state for one producer-consumer pair.
 *
 * Used in three flavours (selected via the boolean fields):
 *   1) Remote RDMA peer            (isLocal=false, useGinSignal=true)
 *   2) Remote LSA peer             (isLocal=false, useGinSignal=false)
 *   3) Local inter-warp pipeline   (isLocal=true,  useGinSignal=false)
 *      — used to sequence Type 1 -> Type 4 (RDMA) and
 *        Type 2 -> Type 3 (LSA) intra-CTA hand-offs.
 * ====================================================================*/
struct StagingFlowCtrl {
  int remoteRank;
  bool isLocal;
  bool useGinSignal;

  /* Path A — staging-buffer-memory based (LSA + local pipeline). */
  size_t localTailOffset;
  size_t localHeadOffset;
  size_t remoteTailOffset;
  size_t remoteHeadOffset;

  /* Path B — gin signals (RDMA peers only). */
  ncclGinSignal_t localTailSignal;
  ncclGinSignal_t localHeadSignal;
  ncclGinSignal_t remoteTailSignal;
  ncclGinSignal_t remoteHeadSignal;
  uint64_t tailSignalBase;
  uint64_t headSignalBase;

  /* Path A bases (LSA staging-buffer-memory path).  The active PIPE
   * persistent-counter path keeps peer-visible counters monotonically
   * increasing and uses per-edge cursor state instead of a prologue snapshot,
   * so these bases normally remain zero.  They are retained for the generic
   * LSA primitive helpers and mirror tailSignalBase / headSignalBase for
   * the GIN-signal path. */
  uint64_t lsaTailBase;
  uint64_t lsaHeadBase;

  /* Path C — local DMA put-completion counter (consumer-side). */
  uint32_t localPutCounter;

  /* Persistent local cursors used by the PIPE NCCL-style FIFO path.
   * Producers keep their absolute tail locally; consumers keep their absolute
   * consumed head locally.  Peer-visible head/tail still live in the transport
   * signal/counter fields above. */
  size_t cursorTailOffset;
  size_t cursorHeadOffset;

  /* Per-peer data sub-region inside the channel's data area. */
  size_t peerDataOffset;
  int peerNumSlots;
  size_t peerChunkSize;

  /* Kernel-runtime shadow state — host leaves zero-initialized. */
  uint64_t shadowTail;
  uint64_t lastTailVal;
  uint64_t localHeadVal;
};

/* ======================================================================
 * One channel's worth of staging memory (RDMA OR LSA — two of these
 * exist per channel, one for each transport).
 * ====================================================================*/
struct StagingRegion {
  size_t dataOffset;
  size_t regionSize;
  size_t chunkSize;
  int numSlots;
  ncclWindow_t window;
};

/* ======================================================================
 * Per-(source,dest) shard transfer plan.  Mirrors the layout that
 * reshard_mesh.cc / reshard_loadbalance.cc already produce for the
 * window path; staging_prepare.cc just memcpys those plans into here.
 * ====================================================================*/
struct StagingTransferPlan {
  int numOuterLoops;
  size_t outerCounts[NCCL_RESHARD_MAX_TENSOR_DIMS];
  size_t outerSrcStrides[NCCL_RESHARD_MAX_TENSOR_DIMS];
  size_t outerDstStrides[NCCL_RESHARD_MAX_TENSOR_DIMS];
  size_t srcBaseOffset;
  size_t dstBaseOffset;
  size_t innerSize;
  size_t totalInnerTransfers;
  bool isContiguous;
  size_t totalBytes;
};

/* ======================================================================
 * One peer's per-channel descriptor: flow control + transfer plan.
 * ====================================================================*/
struct StagingPeerInfo {
  int peerWorldRank;
  int peerShardIdx;
  int peerLocalRank; /* physical LSA rank for LSA peers, -1 for RDMA */
  int rdmaTransport;  /* STAGING_RDMA_TRANSPORT_*; only meaningful for RDMA peers. */

  StagingFlowCtrl fc;
  StagingTransferPlan plan;

  /* Channel assignment.  Legacy staging leaves every peer active on every
   * channel with channelRank=channel_id and channelCount=numChannels.  When
   * peer grouping is enabled, only the channels assigned to this peer are
   * active, and chunk ranges are split across that smaller channel group. */
  bool active;
  int channelRank;
  int channelCount;

  /* Compact per-channel work range.  This lets control/signal paths avoid
   * touching the full tensor transfer plan unless they actually copy data. */
  size_t totalBytes;
  /* Producer/consumer chunk contract for this edge. */
  size_t logicalChunkSize;
  size_t chunkStart;
  size_t chunkEnd;
};

/* ======================================================================
 * Top-level kernel parameters.  Launched with gridDim.x = numChannels;
 * each CTA picks its slice via blockIdx.x.
 * ====================================================================*/
struct StagingKernelParams {
  /* Global config. */
  int numChannels;
  int myRank;
  int myLocalRank;
  bool isSource;
  bool isDest;

  /* Caller buffers (NOT registered as windows — used for local copies). */
  void* srcBuffer;
  void* dstBuffer;

  /* Staging buffer base pointer (device). */
  void* stagingBuffer;

  /* Per-channel region descriptors (indexed by blockIdx.x). */
  StagingRegion rdmaRegions[STAGING_MAX_CHANNELS];
  StagingRegion lsaRegions[STAGING_MAX_CHANNELS];

  ncclWindow_t rdmaWindow;
  ncclWindow_t rdmaWindowB; /* Split PIPE commB RDMA window; same as rdmaWindow on the single-comm path. */
  ncclWindow_t lsaWindow;

  /* Per-channel, per-target local pipeline flow control. */
  StagingFlowCtrl localRdmaFc[STAGING_MAX_CHANNELS][MAX_TARGETS];
  StagingFlowCtrl localLsaFc[STAGING_MAX_CHANNELS][MAX_TARGETS];

  /* Source side: per-target. */
  StagingPeerInfo rdmaTargets[STAGING_MAX_CHANNELS][MAX_TARGETS];
  int numRdmaTargets;

  /* Dual purpose:
   *   src side -> LSA targets (dest leaders on same node)
   *   dst side -> LSA fan-out followers
   * (a rank is either source or dest, never both, so no conflict). */
  StagingPeerInfo lsaTargets[STAGING_MAX_CHANNELS][MAX_TARGETS];
  int numLsaTargets;

  /* Dest side: per-source. */
  StagingPeerInfo rdmaSources[STAGING_MAX_CHANNELS][MAX_SOURCES];
  int numRdmaSources;

  StagingPeerInfo lsaSources[STAGING_MAX_CHANNELS][MAX_SOURCES];
  int numLsaSources;

  /* Number of LSA fan-out followers per RDMA source (= num_local_reps - 1).
   * Indexes into lsaTargets[] when iterating fan-out — the kernel reads
   * lsaTargets[ch][s * numLsaFollowers + f]. */
  int numLsaFollowers;

  /* Inter-node ring forwarding: the LAST `numRingTargets` entries
   * of rdmaTargets[] are dest-side ring forwarders. */
  int numRingTargets;

  /* Chunking + signal-bank counts (filled by stagingPrepareTransfer). */
  size_t chunkSize;
  int ginSignalCount;
  int ginCounterCount;
  /* Start-of-call readiness in this graph's control slot. */
  size_t pipeReadySlotOffset;
  size_t pipeReadyChannelStride;
  uint64_t epoch;

  /* Tensor metadata for pack/unpack strides. */
  size_t srcDims[NCCL_RESHARD_MAX_TENSOR_DIMS];
  size_t dstDims[NCCL_RESHARD_MAX_TENSOR_DIMS];
  size_t srcStrides[NCCL_RESHARD_MAX_TENSOR_DIMS];
  size_t dstStrides[NCCL_RESHARD_MAX_TENSOR_DIMS];
  int ndims;
};

/* Per-call fields for the persistent-counter PIPE path.  The large
 * StagingKernelParams block is treated as an immutable, cached device plan;
 * these are the values that can change between ncclReshard calls. */
struct StagingPipeCallParams {
  void* srcBuffer;
  void* dstBuffer;
  uint64_t epoch;
  size_t maxChunkRounds;

  bool splitComm;
  bool hasLocalFanout;
  int splitCommBContextBase;
  int splitCommBContextCount;
};

struct StagingPipeCopyLayout {
  /* Compact copy descriptor used by the PIPE kernels.  It keeps only the
   * local tensor traversal needed at pack/unpack time, separate from the
   * transport/control metadata in StagingPipePeerEdge. */
  int numOuterLoops;
  size_t outerCounts[NCCL_RESHARD_MAX_TENSOR_DIMS];
  size_t outerStrides[NCCL_RESHARD_MAX_TENSOR_DIMS];
  size_t baseOffset;
  size_t innerSize;
  bool isContiguous;
};

struct StagingPipePeerEdge {
  /* Transport/control descriptor for one active edge in one staging channel.
   * Chunk ranges are precomputed on the host so control warps can drive the
   * FIFO without touching the full tensor transfer plan. */
  int peerWorldRank;
  int peerLocalRank;
  int rdmaTransport;
  int copyLayoutIndex;
  bool active;

  size_t totalBytes;
  /* Copied from StagingPeerInfo; used by both PIPE execution paths. */
  size_t logicalChunkSize;
  int channelRank;
  int channelCount;
  size_t chunkStart;
  size_t chunkEnd;

  StagingFlowCtrl fc;
};

struct StagingPipeTrainerEdge {
  /* Trainer edges need both the remote generator FIFO and the local
   * pack-to-RDMA FIFO consumed by the trainer control warp. */
  StagingPipePeerEdge peer;
  StagingFlowCtrl localFc;
};

struct StagingPipeDevicePlan {
  /* Immutable per-shape plan uploaded once per cache slot.  Active-channel
   * launch maps compact gridDim.x while preserving channel ids used for
   * staging-buffer, GIN context, and counter indexing. */
  StagingPipeTrainerEdge rdmaTargets[STAGING_MAX_CHANNELS][MAX_TARGETS];
  StagingPipePeerEdge lsaTargets[STAGING_MAX_CHANNELS][MAX_TARGETS];
  StagingPipePeerEdge rdmaSources[STAGING_MAX_CHANNELS][MAX_SOURCES];
  StagingPipePeerEdge lsaSources[STAGING_MAX_CHANNELS][MAX_SOURCES];

  StagingPipeCopyLayout rdmaTargetLayouts[STAGING_MAX_CHANNELS][MAX_TARGETS];
  StagingPipeCopyLayout rdmaSourceLayouts[STAGING_MAX_CHANNELS][MAX_SOURCES];
  StagingPipeCopyLayout lsaSourceLayouts[STAGING_MAX_CHANNELS][MAX_SOURCES];

  int trainerRdmaLaunchChannels[STAGING_MAX_CHANNELS];
  int generatorRdmaLaunchChannels[STAGING_MAX_CHANNELS];
  int generatorLsaLaunchChannels[STAGING_MAX_CHANNELS];
  int numTrainerRdmaLaunchChannels;
  int numGeneratorRdmaLaunchChannels;
  int numGeneratorLsaLaunchChannels;
};

/* ======================================================================
 * Host-side peer descriptor — how staging_prepare.cc tells
 * stagingPrepareTransfer() about its peers.  Independent of the
 * device-side StagingPeerInfo so the host can keep extra metadata
 * (sorted positions etc.) without bloating the kernel param block.
 * ====================================================================*/
struct StagingPeerDescriptor {
  int peerWorldRank;
  int peerShardIdx;
  bool isRdma;
  int peerLocalRank;

  StagingTransferPlan plan;
};

/* ======================================================================
 * Top-level transfer descriptor passed by the host entry to
 * stagingPrepareTransfer().  Aggregates everything the kernel needs to
 * know that does NOT depend on the staging-buffer allocation itself.
 * ====================================================================*/
struct StagingTransferDescriptor {
  int myWorldRank;
  int myLocalRank;
  bool isSource;
  bool isDest;

  void* srcBuffer;
  void* dstBuffer;

  StagingPeerDescriptor targets[MAX_TARGETS];
  int numTargets;
  int numRingTargets; /* tail of targets[] reserved for ring fwd */
  int controlSlot;    /* graph-specific control slot within each staging channel */

  StagingPeerDescriptor sources[MAX_SOURCES];
  int numSources;

  /* num_local_reps - 1: per-source LSA fan-out fan-out factor on the
   * dest side.  Forwarded into StagingKernelParams::numLsaFollowers
   * for the kernel's fan-out indexing. */
  int numLsaFollowers;

  /* Cross-rank sorting metadata.  Data-region indices retain the complete
   * source/target ordering so local LSA fanout has stable buffer offsets.
   * GIN IDs use the RDMA-only ordinals below: local LSA edges never consume
   * a GIN signal or counter slot. */
  int destNumSources[MAX_TARGETS];
  int sourceIndexOnDest[MAX_TARGETS];
  int destNumRdmaSources[MAX_TARGETS];
  int rdmaSourceIndexOnDest[MAX_TARGETS];
  /* targetIndexOnSource is the remote producer's full target slot for data
   * placement; rdmaTargetIndexOnSource is its compact GIN ordinal.
   * channelTargetIndexOnSource is the producer-side edge used for channel
   * grouping; ring forwarding needs these to differ. */
  int targetIndexOnSource[MAX_SOURCES];
  int rdmaTargetIndexOnSource[MAX_SOURCES];
  int channelTargetIndexOnSource[MAX_SOURCES];
  int sourceNumTargets[MAX_SOURCES];
  int sourceNumRdmaTargets[MAX_SOURCES];
  int sourceLsaHeadIndexOnProvider[MAX_SOURCES];
  /* PIPE's dense GIN map uses (RDMA peer ordinal, channel rank within that
   * peer group). Zero keeps the legacy channel-major map for PACK. */
  int pipeGinPeerCapacity;
  int pipeGinChannelsPerPeer;
  int peerGroupSizeBound;
  int ctaHeuristicPeerCount;
  size_t maxEdgeBytes;
  bool hasLocalFanout;

  /* Tensor metadata. */
  size_t srcDims[NCCL_RESHARD_MAX_TENSOR_DIMS];
  size_t dstDims[NCCL_RESHARD_MAX_TENSOR_DIMS];
  size_t srcStrides[NCCL_RESHARD_MAX_TENSOR_DIMS];
  size_t dstStrides[NCCL_RESHARD_MAX_TENSOR_DIMS];
  int ndims;
};

#ifdef __cplusplus
}
#endif

#endif /* NCCL_STAGING_TYPES_H_ */
