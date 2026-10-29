/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

/*************************************************************************
 * Tensor Reshard — Split-Comm RING Path (QP-scalability optimization)
 *
 * For the RING algorithm under NODE_AWARE load balancing, the single
 * full all-to-all ncclDevComm spanning every trainer + generator rank
 * costs O(N^2) GIN queue pairs.  This module forms two narrower comms
 * instead:
 *
 *   commA (FULL all-to-all): trainer ranks + the entire FIRST generator
 *         NVL domain.  Carries the cross-group GIN puts (source ->
 *         first-domain leaders).
 *
 *   commB (RAIL only): ALL generator ranks.  Carries the intra-generator
 *         cross-NVL ring forwarding and the LSA fan-out.
 *
 * This is additive and non-destructive. PACK uses its legacy
 * node-aware admission; PIPE uses split comms for node-aware destination
 * replication. Both use the same split communicator construction and
 * forwarding path after admission.
 *
 * NOTE: the generator NVL-domain size (lsaSize) is topology dependent
 * and is only known after commB is formed and its DevComm is probed.
 * commA's membership (which generator ranks join) is therefore derived
 * AFTER commB exists; the probed lsaSize is broadcast across the parent
 * comm so trainer-only ranks (which are not in commB) agree on the
 * eligibility decision and commA membership.
 ************************************************************************/

#ifndef NCCLM2N_RESHARD_SPLIT_H_
#define NCCLM2N_RESHARD_SPLIT_H_

#include "cuda_runtime.h"
#include "nccl.h"
#include "nccl_m2n.h"
#include "reshard_types.h"
#include "reshard_pipe_resources.h"

#include <cstddef>
#include <cstdint>

struct ReshardDevCommUse;
struct ncclDevComm;
enum ReshardDevCommBarrierKind : int;

/* Rail GIN connection type for the generator-only sub-comm (commB).
 * The rest of the library only ever requests NCCL_GIN_CONNECTION_FULL.
 * The rail-only enum name must be confirmed against the on-cluster NCCL
 * device headers; override at build time with
 *   -DNCCLM2N_GIN_RAIL_CONNECTION=<enum>
 * if the symbol differs in the installed NCCL. */
#ifndef NCCLM2N_GIN_RAIL_CONNECTION
#define NCCLM2N_GIN_RAIL_CONNECTION NCCL_GIN_CONNECTION_RAIL
#endif

/* ======================================================================
 * Sub-comm container
 *
 * Result of (collectively) attempting to form the split comms for a
 * given parent comm + src/dst mesh layout.  Cached per parent comm; all
 * ranks of the parent comm observe identical `active`, `lsaSize`, and
 * `numGenDomains` so the dispatch decision is consistent.
 * ====================================================================*/
struct ReshardSplitComms {
  bool active; /* true => split path is in use for this parent comm */

  ncclComm_t commA; /* trainer + first K gen NVL domains (FULL); NULL if not member */
  ncclComm_t commB; /* all generator ranks (RAIL); NULL if not member */

  ncclComm_t parentComm; /* the parent comm this split was derived from */

  int lsaSize; /* generator NVL-domain size (from commB probe DevComm) */
  int srcLsaSize; /* trainer NVL-domain size (from commA probe DevComm, src ranks); 0 if unknown */
  int numGenDomains; /* number of NVL domains the generator spans */

  /* K = min(srcRepCount, numGenDomains): the number of generator NVL
   * domains that act as injection legs (one per active source rep).
   * commA's generator block is exactly the first K*lsaSize generator
   * ranks (parent ranks [dstStartRank, dstStartRank + K*lsaSize)).  When
   * `strided` is true the load balancer must place every source rep's
   * injection target within those first K domains (rep i -> domain i,
   * ringing to i+K, i+2K, ...).  When false the contiguous baseline
   * mapping already lands all injections in the first K domains (this
   * only holds when K == numGenDomains, i.e. srcRepCount >= numGenDomains). */
  int numInjectionDomains;

  /* Number of NVL domains a single destination replica spans
   * (= numGenDomains / dstRepCount, clamped to >= 1).  Threaded into the
   * load balancer so the strided helpers can recover numInjectionReps
   * (= numInjectionDomains / domainsPerRep) and map each source rep to the
   * first domain of the replica it injects. */
  int domainsPerRep;

  bool strided; /* true => split path needs the strided node assignment */

  /* Parent-comm geometry captured at formation (world == parent comm). */
  int parentRank; /* this rank's id within the parent comm */
  int parentSize; /* parent comm size */
  int srcStartRank; /* srcMesh->startRank (parent-relative) */
  int srcMeshSize;
  int dstStartRank; /* dstMesh->startRank (parent-relative) */
  int dstMeshSize;

  /* Membership / translated rank of THIS rank. -1 when not a member. */
  bool inA;
  bool inB;
  int rankInA;
  int rankInB;

  /* Slot within commB's DevComm. */
  int slotIdx;
  int slotCount;
  int commBPoolOwnerIndex;
  /* Stable PIPE FIFO-history namespace; unlike slotIdx, this never wraps
   * with execution concurrency. PACK does not use these fields. */
  int pipeNamespaceId;
  int pipeNamespaceCapacity;

  bool valid; /* internal: entry has been populated */
};

/* Rank-uniform identity for PIPE persistent control and GIN-resource
 * slots. The cache is intentionally a small linear table, so use direct
 * equality rather than a hash: any field that changes rank mapping or the
 * split topology must select a distinct slot. */
struct ReshardStagingMeshSignature {
  int srcStartRank;
  int srcMeshDims[NCCL_RESHARD_MESH_NDIMS];
  int srcPlacements[NCCL_RESHARD_MESH_NDIMS];
  int dstStartRank;
  int dstMeshDims[NCCL_RESHARD_MESH_NDIMS];
  int dstPlacements[NCCL_RESHARD_MESH_NDIMS];
  int srcGpusPerDomain;
  int dstGpusPerDomain;
  int loadBalanceMode;
  bool splitStrided;
  int splitNumInjectionDomains;
  int splitDomainsPerRep;

  bool operator==(const ReshardStagingMeshSignature& other) const {
    if (srcStartRank != other.srcStartRank || dstStartRank != other.dstStartRank ||
        srcGpusPerDomain != other.srcGpusPerDomain || dstGpusPerDomain != other.dstGpusPerDomain ||
        loadBalanceMode != other.loadBalanceMode || splitStrided != other.splitStrided ||
        splitNumInjectionDomains != other.splitNumInjectionDomains || splitDomainsPerRep != other.splitDomainsPerRep) {
      return false;
    }
    for (int d = 0; d < NCCL_RESHARD_MESH_NDIMS; d++) {
      if (srcMeshDims[d] != other.srcMeshDims[d] || srcPlacements[d] != other.srcPlacements[d] ||
          dstMeshDims[d] != other.dstMeshDims[d] || dstPlacements[d] != other.dstPlacements[d]) {
        return false;
      }
    }
    return true;
  }
};

/* Tensor-shape identity for immutable PIPE plans. It intentionally omits
 * mesh and split topology, which belong exclusively to ReshardStagingMeshSignature. */
struct ReshardStagingTensorSignature {
  int srcNdims;
  ncclDataType_t srcDtype;
  size_t srcLocalShape[NCCL_RESHARD_MAX_TENSOR_DIMS];
  int dstNdims;
  ncclDataType_t dstDtype;
  size_t dstLocalShape[NCCL_RESHARD_MAX_TENSOR_DIMS];

  bool operator==(const ReshardStagingTensorSignature& other) const {
    if (srcNdims != other.srcNdims || srcDtype != other.srcDtype || dstNdims != other.dstNdims ||
        dstDtype != other.dstDtype) {
      return false;
    }
    for (int d = 0; d < NCCL_RESHARD_MAX_TENSOR_DIMS; d++) {
      if (srcLocalShape[d] != other.srcLocalShape[d] || dstLocalShape[d] != other.dstLocalShape[d]) {
        return false;
      }
    }
    return true;
  }
};

/* Canonical identity for a PIPE transfer. Every rank receives the same mesh
 * and tensor descriptors, while its cached plan retains rank-local channels
 * and peer maps. Keeping this identity rank-uniform makes persistent GIN
 * control-slot selection agree across every edge. */
struct ReshardStagingPipeSignature {
  ReshardStagingMeshSignature meshSignature;
  ReshardStagingTensorSignature tensorSignature;

  bool operator==(const ReshardStagingPipeSignature& other) const {
    return meshSignature == other.meshSignature && tensorSignature == other.tensorSignature;
  }
};

/* ======================================================================
 * Cross-comm RING kernel parameters
 *
 * Mirrors ncclReshardParams but every peer rank is expressed in the
 * sub-comm it is reached through, and signal bases are per-sub-comm:
 *   - Source puts (source -> first-domain leader) ride commA, so target
 *     dstWorldRank and source signalBase are commA-relative.
 *   - Ring forwarding + LSA fan-out ride commB, so ringNext / followers
 *     are commB-relative and use commB-relative signal bases.
 *
 * Populated by reshard_split_prepare.cc; consumed by the dual-DevComm
 * kernel in reshard_split_user_window.cu.
 * ====================================================================*/
typedef struct {
  ncclWindow_t windowA; /* window registered on commA (source-side puts) */
  ncclWindow_t windowB; /* window registered on commB (ring + LSA) */

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

  /* This rank's id in each sub-comm; -1 when not a member. */
  int myRankInA;
  int myRankInB;

  /* True when this DEST rank receives its sources directly from the
   * trainer over commA (i.e. it is on the first gen NVL domain).  False
   * when it receives via the commB cross-NVL ring.  Selects which GIN
   * the dest waits on. */
  bool destRecvViaCommA;

  size_t elementsPerChunk;
  size_t chunkSizeBytes;
  int totalCtas;

  /* Source-side targets: ncclReshardTargetInfo::dstWorldRank holds the
   * leader's commA-relative rank.  sources[].signalBase is the WAIT
   * index (commA-relative for first-domain dests, commB-relative for
   * downstream dests); sourceSignalBaseB[] is the commB ring-FORWARD
   * index (always srcShardIdx * totalCtas), distinct because commA and
   * commB have independent signal spaces. */
  ncclReshardSourceInfo sources[MAX_SOURCES];
  unsigned int sourceSignalBaseB[MAX_SOURCES];
  int numSources;

  ncclReshardTargetInfo targets[MAX_TARGETS];
  int numTargets;

  /* Replication peers, commB-relative. */
  int localFollowerRanksB[MAX_LOCAL_FOLLOWERS];
  int numLocalFollowers;
  int ringNextRankB;
  bool isRingLast;

  int localRepIdx;
  int numLocalReps;
  bool isLeaderForSources;

  size_t myWindowOffset;
  size_t ringNextWindowOffset;
  size_t localFollowerWindowOffsets[MAX_LOCAL_FOLLOWERS];

  /* Registered staging-window size in bytes (symmetric across ranks).
   * Used ONLY by the kernelTrace-gated bounds checks to flag a put whose
   * src-read or dst-write offset overflows the staging window. */
  size_t windowCapacityBytes;

  /* commB partition fields. Each staging bucket gets disjoint GIN context
   * and signal ranges; commA remains unslotted. */
  int slotIdx;
  int ctxPerSlotB;
  int signalsPerSlotB;

  /* Debug: when nonzero, the split kernel emits per-rank device printf at
   * the commB (Phase B) barrier/waitSignal boundaries to localize hangs.
   * Set from NCCL_RESHARD_SPLIT_KERNEL_TRACE in reshardLaunchPackSplit. */
  int kernelTrace;
} ncclReshardParamsSplit;

/* ======================================================================
 * reshard_split_comm.cc — sub-comm formation / teardown
 * ====================================================================*/

/* PIPE uses split comms only for node-aware destination replication. */
bool reshardShouldAttemptPipeSplitComms(const ncclDistTensor_t* srcTensor, const ncclDistTensor_t* dstTensor);

ncclResult_t reshardGetOrCreateSplitComms(ncclComm_t comm, const ncclMesh_t* srcMesh, const ncclMesh_t* dstMesh,
                                          int srcRepCount, int dstRepCount, bool dstRepStrided, int numCtas,
                                          cudaStream_t stream, ReshardSplitComms* out);

/* Split and parent PIPE launches share one reusable staging allocation per
 * parent communicator.  A (mesh, channel) signature owns a unique control
 * slice in that allocation.  The mesh signature records split topology, so
 * an active split path cannot alias the parent path. */
int reshardGetPersistentControlSlotCount();
ncclResult_t reshardGetOrCreatePersistentControlSlot(ncclComm_t parentComm,
                                                     const ReshardStagingPipeSignature& signature, int rank,
                                                     int* outSlot);

/* Ensure (and cache) the per-staging-buffer windows + optional sized DevComms
 * on the split sub-comms: windowA/devCommA (FULL) on commA and
 * windowB/devCommB (RAIL) on commB.  Collective over the respective sub-comms;
 * safe to call every reshard. Pass null DevComm outputs when only split windows
 * are needed.
 *
 * commA and commB use the shared event-ordered DevComm cache. PACK sizes commB
 * resources by physical lanes; PIPE sizes persistent signals/counters by
 * reserved parent namespaces and contexts by execution lanes. A pooled PIPE
 * owner's resource plan cannot change while persistent FIFO state is live. */
ncclResult_t reshardSplitEnsureResources(const ReshardSplitComms* sc, void* stagingBuffer, size_t stagingCapacity,
                                         int barrierCount, ReshardDevCommBarrierKind barrierKind, int ginSignalCountA,
                                         int ginCounterCountA, const ReshardCommBResourcePlan& resourcesB,
                                         cudaStream_t stream,
                                         ncclWindow_t* outWindowA, ncclWindow_t* outWindowB, ncclDevComm* outDevCommA,
                                         ReshardDevCommUse* outDevCommAUse, ncclDevComm* outDevCommB,
                                         ReshardDevCommUse* outDevCommBUse);

/* Order reuse of one commB resource lane across parent communicators. PACK's
 * staging-slot event already provides the same fence, so these are active for
 * PIPE only. */
ncclResult_t reshardSplitPrepareCommBResourceLane(const ReshardSplitComms* sc, cudaStream_t stream);
ncclResult_t reshardSplitRecordCommBResourceLane(const ReshardSplitComms* sc, cudaStream_t stream);

/* ======================================================================
 * reshard_split_prepare.cc — cross-comm RING param builder
 * ====================================================================*/

/* Translate a parent-comm-relative ncclReshardParams (as built by
 * prepareReshardParams, including any PACK contiguous-plan rewrite)
 * into a ncclReshardParamsSplit whose peers are commA/commB-relative
 * and whose signal bases are per-sub-comm.  windowA/windowB are the
 * staging windows registered on commA/commB. */
void buildSplitReshardParams(const ncclReshardParams* base, const ReshardSplitComms* sc, int numCtas,
                             ncclWindow_t windowA, ncclWindow_t windowB, ncclReshardParamsSplit* out);

/* ======================================================================
 * reshard_split_user_window.cu — dual-DevComm RING kernel + PACK
 * split dispatch.
 * ====================================================================*/

/* PACK split launch.  Called from reshardCopyPackNormalized after the
 * (split-domain-sized) base params + pack + contiguous rewrite are built,
 * once the caller has already formed the split comms (sc->active == true)
 * and chosen dstGpusPerDomain=commB.lsaSize / srcGpusPerDomain=commA
 * probe.  Registers the staging windows on commA/commB, creates the two
 * DevComms (FULL / RAIL), builds the split params, and launches the
 * dual-DevComm kernel.  Caller is responsible for the RING + NODE_AWARE +
 * eligibility gating (this function assumes the split is active). */
ncclResult_t reshardLaunchPackSplit(const ReshardSplitComms* sc, void* stagingBuffer, size_t stagingCapacity,
                                    const ncclReshardParams* baseParams, int numCtas, cudaStream_t stream);

#endif /* NCCLM2N_RESHARD_SPLIT_H_ */
