/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

/*************************************************************************
 * Tensor Reshard — Internal Type Definitions
 *
 * Central registry of all internal structs consumed by the host-side
 * prep code, caches, and device kernels.  Not part of the public C API.
 *
 * All translation units in the library include this header for struct
 * definitions; function declarations for cross-TU calls are in
 * reshard_internal.h.
 ************************************************************************/

#ifndef NCCL_RESHARD_TYPES_H_
#define NCCL_RESHARD_TYPES_H_

#include <cstddef>
#include <cstdint>

#include "cuda_runtime.h"
#include "nccl.h"

#include "reshard_limits.h"

/* Algorithm + load-balance mode are library-internal: callers configure
 * them via env vars only (NCCL_RESHARD_ALGORITHM / NCCL_RESHARD_LB_MODE). */
typedef enum {
  RESHARD_ALGO_AUTO = -1,
  RESHARD_ALGO_RING = 0,
  RESHARD_ALGO_DIRECT = 1
} ReshardAlgorithm;

typedef enum {
  RESHARD_LB_UNIFORM = 0,
  RESHARD_LB_NODE_AWARE = 1
} ReshardLoadBalanceMode;

typedef enum {
  RESHARD_COPY_ALGO_PIPE = 0,
  RESHARD_COPY_ALGO_DIRECT = 1,
  /* Pack each destination's bytes contiguously into the staging buffer
   * (CE cudaMemcpy3D), transfer with the hierarchical user-window kernel
   * (reshardKernelUserWindow) over contiguous plans, then unpack. */
  RESHARD_COPY_ALGO_PACK = 2
} ReshardCopyAlgorithm;

typedef enum {
  RESHARD_PIPE_NET_DEVICE = 0,
  RESHARD_PIPE_NET_HOST_RMA = 1
} ReshardPipeNetMode;

/* ncclDevComm is defined in nccl_device.h; only TUs that need the
   DevCommCacheEntry (reshard_cache.cc) include that header directly. */

/* ======================================================================
 * Derived Group Information (computed from mesh)
 * ====================================================================*/

typedef struct {
  /* Replicated axes have size 1 and tensor dimension -1. Shard indices
   * flatten the sharded mesh axes in placement order (last axis fastest). */
  int axisSizes[NCCL_RESHARD_MESH_NDIMS];
  int axisTensorDims[NCCL_RESHARD_MESH_NDIMS];
  int shardCounts[NCCL_RESHARD_MAX_TENSOR_DIMS];
} ncclReshardShardLayout_t;

typedef struct {
  ncclReshardShardLayout_t shardLayout;
  int repMeshDim;
  int shardCount;
  int repCount;
  int meshPos[NCCL_RESHARD_MESH_NDIMS];
  int shardIdx;
  int repIdx;
} ncclReshardMeshGroupInfo;

/* ======================================================================
 * Transfer Plan
 * ====================================================================*/

typedef struct {
  int numOuterLoops;
  size_t outerCounts[NCCL_RESHARD_MAX_TENSOR_DIMS];
  size_t outerSrcStrides[NCCL_RESHARD_MAX_TENSOR_DIMS];
  size_t outerDstStrides[NCCL_RESHARD_MAX_TENSOR_DIMS];

  size_t srcBaseOffset;
  size_t dstBaseOffset;

  size_t innerSize;
  size_t totalInnerTransfers;

  size_t overlapStart[NCCL_RESHARD_MAX_TENSOR_DIMS];
  size_t overlapEnd[NCCL_RESHARD_MAX_TENSOR_DIMS];
} ncclReshardTransferPlan;

/* ======================================================================
 * RING (hierarchical) kernel param structs
 * ====================================================================*/

typedef struct {
  unsigned int signalBase;

  ncclReshardTransferPlan plan;

  bool isContiguous;
  size_t totalBytes;

  /* Source shard index that produced this entry.  Used by the
   * PACK copy mode to compute a rank-independent contiguous
   * staging offset (cumulative bytes of smaller src shards feeding
   * this dst shard).  -1 when unused. */
  int srcShardIdx;
} ncclReshardSourceInfo;

typedef struct {
  int dstShardIdx;
  int dstWorldRank;
  size_t overlapStart[NCCL_RESHARD_MAX_TENSOR_DIMS];
  size_t overlapEnd[NCCL_RESHARD_MAX_TENSOR_DIMS];

  ncclReshardTransferPlan plan;

  bool isContiguous;
  size_t totalBytes;
  size_t windowOffset;
} ncclReshardTargetInfo;

typedef struct {
  ncclWindow_t window;

  size_t srcDims[NCCL_RESHARD_MAX_TENSOR_DIMS];
  size_t dstDims[NCCL_RESHARD_MAX_TENSOR_DIMS];
  size_t srcStrides[NCCL_RESHARD_MAX_TENSOR_DIMS];
  size_t dstStrides[NCCL_RESHARD_MAX_TENSOR_DIMS];
  int ndims;

  ncclReshardShardLayout_t srcShardLayout;
  ncclReshardShardLayout_t dstShardLayout;
  int srcShardCount;
  int dstShardCount;

  bool isSource;
  bool isDest;
  int mySrcShardIdx;
  int myDstShardIdx;
  int mySrcRepIdx;
  int myDstRepIdx;
  int myWorldRank;

  size_t elementsPerChunk;
  size_t chunkSizeBytes;
  int totalCtas;
  unsigned int mySignalBase;

  ncclReshardSourceInfo sources[MAX_SOURCES];
  int numSources;

  ncclReshardTargetInfo targets[MAX_TARGETS];
  int numTargets;

  int localFollowerWorldRanks[MAX_LOCAL_FOLLOWERS];
  int numLocalFollowers;
  int ringNextWorldRank;
  bool isRingLast;

  int localRepIdx;
  int numLocalReps;
  bool isLeaderForSources;

  /* Split-comm classification: this dest rank is the head of its ring chain
   * (a source injects it directly), so it receives over commA.  Downstream
   * ring-recipient leaders have this false and must wait on the commB ring. */
  bool destInjectedDirectly;

  size_t myWindowOffset;
  size_t ringNextWindowOffset;
  size_t localFollowerWindowOffsets[MAX_LOCAL_FOLLOWERS];
} ncclReshardParams;

/* ======================================================================
 * DIRECT algorithm param structs
 * ====================================================================*/

typedef struct {
  int dstShardIdx;
  int dstRepIdx;
  int dstWorldRank;
  size_t overlapStart[NCCL_RESHARD_MAX_TENSOR_DIMS];
  size_t overlapEnd[NCCL_RESHARD_MAX_TENSOR_DIMS];
  size_t totalBytes;
  bool isContiguous;
  size_t windowOffset;

  ncclReshardTransferPlan plan;
} ncclReshardDirectTargetInfo;

typedef struct {
  unsigned int signalBase;
  size_t totalBytes;
  bool isContiguous;

  ncclReshardTransferPlan plan;
} ncclReshardDirectSourceInfo;

typedef struct {
  ncclWindow_t window;

  size_t srcDims[NCCL_RESHARD_MAX_TENSOR_DIMS];
  size_t dstDims[NCCL_RESHARD_MAX_TENSOR_DIMS];
  size_t srcStrides[NCCL_RESHARD_MAX_TENSOR_DIMS];
  size_t dstStrides[NCCL_RESHARD_MAX_TENSOR_DIMS];
  int ndims;

  ncclReshardShardLayout_t srcShardLayout;
  ncclReshardShardLayout_t dstShardLayout;
  int srcShardCount;
  int dstShardCount;

  bool isSource;
  bool isDest;
  int myWorldRank;
  int mySrcShardIdx;
  int mySrcRepIdx;
  int myDstShardIdx;
  int myDstRepIdx;

  size_t elementsPerChunk;
  int totalCtas;
  unsigned int mySignalBase;

  size_t myWindowOffset;

  ncclReshardDirectTargetInfo targets[MAX_DIRECT_TARGETS];
  int numTargets;

  ncclReshardDirectSourceInfo sources[MAX_DIRECT_SOURCES];
  int numSources;
} ncclReshardDirectParams;

/* ======================================================================
 * Load Balancer
 * ====================================================================*/

typedef struct {
  int srcRepCount;
  int dstRepCount;
  int dstGpusPerDomain;
  int dstRepStartRank;
  int dstRepStride;
  ReshardLoadBalanceMode mode;
  /* Origin subtracted before dividing a parent-comm rank by dstGpusPerDomain
   * to obtain its NVL-domain index.  0 reproduces the absolute mapping used by
   * the non-split path (unchanged).  The split path sets this to the dst mesh
   * start rank so a gen domain whose origin is not a multiple of the domain
   * size (asymmetric-PP: dstStart=16, domain=32) is not split into phantom
   * nodes. */
  int dstNodeAnchor;
  /* Split-comm RING (QP-scalability) only.  When `strided` is true the
   * NODE_AWARE mapping funnels each source rep's injection into the first
   * `numInjectionDomains` (= K = numInjectionReps * domainsPerRep) NVL
   * domains: source rep i injects the first domain of injection-rep i and
   * rings to rep i+numInjectionReps, i+2*numInjectionReps, ...  Default
   * false => the contiguous baseline mapping is used unchanged (non-split
   * path is unaffected). */
  int numInjectionDomains;
  /* Split-comm strided injection only.  Number of NVL domains a single
   * destination replica spans (= numGenDomains / dstRepCount, clamped to
   * >= 1).  1 in the common reps/domain >= 1 regime; > 1 when a replica
   * straddles multiple domains.  Used to recover numInjectionReps
   * (= numInjectionDomains / domainsPerRep) and to map source reps to the
   * first domain of their owned replica.  0 is treated as 1. */
  int domainsPerRep;
  bool strided;
} ncclReshardRepLoadBalancer;

/* ======================================================================
 * Cache entry types (used by reshard_cache.cc)
 * ====================================================================*/

enum ReshardInternalWindowKind {
  RESHARD_INTERNAL_WINDOW_PACK,
  RESHARD_INTERNAL_WINDOW_STAGING,
  RESHARD_INTERNAL_WINDOW_SPLIT,
};

struct WindowCacheEntry {
  ncclComm_t comm;
  void* windowBuffer;
  size_t windowSize;
  ReshardInternalWindowKind kind;
  ncclWindow_t window;
  bool valid;
};

struct WindowCache {
  WindowCacheEntry entries[MAX_WINDOW_CACHE_ENTRIES];
  int count;
  int nextIdx;
};

/* DevCommCacheEntry is defined in reshard_cache.cc (needs nccl_device.h
   for the ncclDevComm by-value member). */

/* ======================================================================
 * Per-comm staging buffer pool (used by reshard_cache.cc)
 * ====================================================================*/

struct StagingBufferState;

struct StagingBufferPoolEntry {
  ncclComm_t comm;
  uint64_t poolKey;
  StagingBufferState* state;
  cudaStream_t stream;
  cudaEvent_t event;
  int capacityChannels;
  bool reserved;
  bool poisoned;
  bool allocated;
};

#endif /* NCCL_RESHARD_TYPES_H_ */
