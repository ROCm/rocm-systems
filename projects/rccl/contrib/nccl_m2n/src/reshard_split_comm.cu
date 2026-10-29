/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

/*************************************************************************
 * Tensor Reshard — Split-Comm Formation (RING / NODE_AWARE)
 *
 * Forms (and caches) the two narrower sub-comms used by the QP-scalability
 * RING path:
 *
 *   commB (RAIL, formed FIRST): all generator ranks.  We probe its
 *         DevComm to learn the generator NVL-domain size (lsaSize),
 *         which is topology dependent and not known until the gen-only
 *         comm exists.
 *
 *   commA (FULL all-to-all, formed SECOND): trainer ranks + the entire
 *         FIRST generator NVL domain.  Membership depends on lsaSize, so
 *         it can only be derived after commB is probed.
 *
 * lsaSize is broadcast across the parent comm (ncclAllReduce MAX) so
 * trainer-only ranks — which are not members of commB — agree on the
 * eligibility decision and on commA membership.
 *
 * Algorithm, load-balance, and copy-mode admission remain at the dispatch
 * sites. Once admitted, PACK and PIPE share this communicator
 * construction and forwarding topology.
 ************************************************************************/

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include "cuda_runtime.h"
#include "nccl.h"
#include "nccl_device.h"

#include "nccl_m2n.h"
#include "reshard_call_setup.h"
#include "reshard_types.h"
#include "m2n_log.h"
#include "m2n_checks.h"
#include "reshard_internal.h"
#include "reshard_split.h"


static std::mutex gCrossNicOverrideMutex;
static int gCrossNicOverrideUsers = 0;
static bool gCrossNicHadValue = false;
static std::string gCrossNicSavedValue;

ScopedCrossNicRailOverride::ScopedCrossNicRailOverride(bool enabled) : active_(enabled) {
  if (!active_) return;
  std::lock_guard<std::mutex> guard(gCrossNicOverrideMutex);
  if (gCrossNicOverrideUsers++ != 0) return;

  const char* previous = getenv("NCCL_CROSS_NIC");
  gCrossNicHadValue = previous != nullptr;
  if (gCrossNicHadValue) gCrossNicSavedValue = previous;
  (void)setenv("NCCL_CROSS_NIC", "0", 1);
}

ScopedCrossNicRailOverride::~ScopedCrossNicRailOverride() {
  if (!active_) return;
  std::lock_guard<std::mutex> guard(gCrossNicOverrideMutex);
  if (--gCrossNicOverrideUsers != 0) return;

  if (gCrossNicHadValue) {
    (void)setenv("NCCL_CROSS_NIC", gCrossNicSavedValue.c_str(), 1);
  } else {
    (void)unsetenv("NCCL_CROSS_NIC");
  }
}

bool reshardShouldAttemptPipeSplitComms(const ncclDistTensor_t* srcTensor, const ncclDistTensor_t* dstTensor) {
  if (srcTensor == nullptr || dstTensor == nullptr || dstTensor->mesh == nullptr) {
    return false;
  }
  if (!reshardGetSplitCommEnabled() || reshardEffectiveLbMode(srcTensor, dstTensor) != RESHARD_LB_NODE_AWARE) {
    return false;
  }

  ncclReshardMeshGroupInfo dstInfo{};
  computeMeshGroupInfo(dstTensor, dstTensor->mesh->startRank, &dstInfo);
  return dstInfo.repCount > 1;
}

namespace {

struct CommBGroupFingerprint {
  uint64_t lo;
  uint64_t hi;

  bool operator==(const CommBGroupFingerprint& other) const {
    return lo == other.lo && hi == other.hi;
  }
};

/* commB (RAIL, all generator ranks) state. Every parent keeps an alias entry,
 * while the first generator-side entry for an ordered generator group owns
 * the reusable communicator, probe DevComm, and resource-lane fences. */
struct CommBSharedEntry {
  ncclComm_t parent;
  int srcStart;
  int dstStart;
  int dstSize;
  int srcSize;
  int resourceLane;
  int resourceLaneCount;
  int poolOwnerIndex;
  CommBGroupFingerprint groupFingerprint;
  int pipeNamespaceId;
  int pipeNamespaceCapacity;
  ReshardPipeNamespacePool pipeNamespaces; /* populated only in the owner */
  ReshardCommBResourcePlan pipeResourcePlan;
  bool pipeResourcePlanValid;

  ncclComm_t commB;         /* NULL when this rank is not a generator rank */
  ncclDevComm probeDevComm; /* RAIL probe DevComm on commB used to read gen lsaSize (rail-local QPs only) */
  bool probeValid;
  int lsaSize;       /* generator NVL-domain size */
  int numGenDomains; /* number of NVL domains the generator spans */
  int rankInB;       /* this rank's id within commB; -1 if not a gen rank */
  bool isDst;        /* whether THIS rank is a generator rank */
  bool pooled;
  bool ownsCommB;

  cudaEvent_t laneCompletion[MAX_SPLIT_CONCURRENCY];
  bool laneCompletionRecorded[MAX_SPLIT_CONCURRENCY];
  bool lanePoisoned[MAX_SPLIT_CONCURRENCY];

  bool used;
};

/* Per-parent split geometry cached after commB formation. */
struct CommBParentEntry {
  ncclComm_t parent;
  int dstStart; /* gen rank set this parent targets (key into gCommBShared) */
  int dstSize;
  int lsaSize;
  int numGenDomains;
  /* Parent-comm geometry captured at formation. */
  int parentRank;
  int parentSize;
  int srcStart;
  int srcSize;
  bool isDst;
  int rankInB;
  int resourceLane;
  int resourceLaneCount;
  int poolOwnerIndex;
  int pipeNamespaceId;
  int pipeNamespaceCapacity;
  bool used;
};

/* commA = all source ranks + the first K generator NVL domains.  Its
 * membership depends on K = the injection-domain count, which varies per
 * parameter (different srcRepCount), so commA is cached keyed by (parent
 * comm, K).  Multiple commA's may coexist for one parent comm. */
struct CommACacheEntry {
  ncclComm_t parent;
  int srcStart;
  int srcSize;
  int dstStart;
  int dstSize;
  int K;
  ncclComm_t commA; /* NULL when this rank is not a member */
  ncclDevComm probeDevCommA; /* DevComm on commA used to read src NVL size */
  bool probeAValid;
  int rankInA;
  int srcLsaSize;
  bool inA;
  bool used;
};

/* PIPE has one reusable staging allocation per parent communicator.  Every
 * persistent cursor slice in that allocation is allocated here for both
 * parent and split launches; split topology is in meshSignature. */
struct PersistentControlSlotEntry {
  ncclComm_t parent;
  ReshardStagingPipeSignature signature;
  int slot;
  bool used;
};

constexpr int kMaxSplitCommEntries = MAX_DEVCOMM_CACHE_ENTRIES;
constexpr int kMaxPersistentControlPools = MAX_DEVCOMM_CACHE_ENTRIES;
/* Split and non-split PIPE use the same fixed persistent-control-slot budget. */
constexpr int kPersistentControlSlots = STAGING_PIPE_CONTROL_SLOTS;
constexpr int kMaxPersistentControlSlotEntries = kMaxPersistentControlPools * kPersistentControlSlots;
/* Protected by the documented single-host-thread reshard contract. */
CommBSharedEntry gCommBShared[kMaxSplitCommEntries];
int gCommBSharedCount = 0;
int gCommBCreateCount = 0;
int gCommBReuseCount = 0;
CommBParentEntry gCommBParent[kMaxSplitCommEntries];
int gCommBParentCount = 0;
CommACacheEntry gCommACache[kMaxSplitCommEntries];
int gCommACount = 0;
PersistentControlSlotEntry gPersistentControlSlots[kMaxPersistentControlSlotEntries];
int gPersistentControlSlotCount = 0;

constexpr int kSplitBroadcastScratchMaxDevices = 16;
struct SplitBroadcastScratch {
  int cudaDev;
  int* dev;
};
SplitBroadcastScratch gSplitBroadcastScratch[kSplitBroadcastScratchMaxDevices];
int gSplitBroadcastScratchCount = 0;
/* Protect scratch-cache mutation only. A collective must not hold this mutex:
 * other local ranks need to enter the same collective concurrently. */
std::mutex gSplitBroadcastScratchMutex;

struct CommBParticipantMetadata {
  uint64_t deviceUuidLo;
  uint64_t deviceUuidHi;
  int resourceLane;
  int isDst;
};

/* Exchanged only when installing a new parent alias. Trainers contribute no
 * namespace proposal; every rank observes generator agreement or failure. */
struct PipeNamespaceMetadata {
  int id;
  int capacity;
  int lanes;
  int peersPerControl;
  int channelsPerPeer;
  int contextsPerLane;
};

struct SplitMetadataScratch {
  int cudaDev;
  unsigned char* dev;
  size_t capacityBytes;
};

SplitMetadataScratch gSplitMetadataScratch[kSplitBroadcastScratchMaxDevices];
int gSplitMetadataScratchCount = 0;

ncclResult_t getSplitBroadcastScratch(int** out) {
  NCCL_M2N_CHECK_ARG(out != nullptr, -1, "split broadcast scratch requires output pointer");
  *out = nullptr;
  int cudaDev = -1;
  NCCL_M2N_CUDACHECK(cudaGetDevice(&cudaDev));
  for (int i = 0; i < gSplitBroadcastScratchCount; i++) {
    if (gSplitBroadcastScratch[i].cudaDev == cudaDev) {
      *out = gSplitBroadcastScratch[i].dev;
      return ncclSuccess;
    }
  }
  NCCL_M2N_CHECK_ARG(gSplitBroadcastScratchCount < kSplitBroadcastScratchMaxDevices, -1,
                     "split broadcast scratch cache exhausted (%d CUDA devices)", kSplitBroadcastScratchMaxDevices);
  int* dev = nullptr;
  NCCL_M2N_CUDACHECK(cudaMalloc(&dev, 2 * sizeof(int)));
  gSplitBroadcastScratch[gSplitBroadcastScratchCount++] = {cudaDev, dev};
  *out = dev;
  return ncclSuccess;
}

ncclResult_t getSplitMetadataScratch(size_t requiredBytes, unsigned char** out) {
  NCCL_M2N_CHECK_ARG(out != nullptr && requiredBytes > 0, -1,
                     "split metadata scratch requires a positive size and output pointer");
  *out = nullptr;
  int cudaDev = -1;
  NCCL_M2N_CUDACHECK(cudaGetDevice(&cudaDev));
  for (int i = 0; i < gSplitMetadataScratchCount; i++) {
    SplitMetadataScratch& s = gSplitMetadataScratch[i];
    if (s.cudaDev != cudaDev) continue;
    if (s.capacityBytes < requiredBytes) {
      if (s.dev != nullptr) NCCL_M2N_CUDACHECK(cudaFree(s.dev));
      s.dev = nullptr;
      NCCL_M2N_CUDACHECK(cudaMalloc(reinterpret_cast<void**>(&s.dev), requiredBytes));
      s.capacityBytes = requiredBytes;
    }
    *out = s.dev;
    return ncclSuccess;
  }
  NCCL_M2N_CHECK_ARG(gSplitMetadataScratchCount < kSplitBroadcastScratchMaxDevices, -1,
                     "split metadata scratch cache exhausted (%d CUDA devices)", kSplitBroadcastScratchMaxDevices);
  SplitMetadataScratch& s = gSplitMetadataScratch[gSplitMetadataScratchCount++];
  s = {};
  s.cudaDev = cudaDev;
  NCCL_M2N_CUDACHECK(cudaMalloc(reinterpret_cast<void**>(&s.dev), requiredBytes));
  s.capacityBytes = requiredBytes;
  *out = s.dev;
  return ncclSuccess;
}

void freeSplitBroadcastScratch() {
  std::lock_guard<std::mutex> guard(gSplitBroadcastScratchMutex);
  int savedCudaDev = -1;
  const cudaError_t getDeviceResult = cudaGetDevice(&savedCudaDev);
  for (int i = 0; i < gSplitBroadcastScratchCount; i++) {
    SplitBroadcastScratch& s = gSplitBroadcastScratch[i];
    if (s.dev != nullptr) {
      if (getDeviceResult == cudaSuccess && s.cudaDev >= 0) {
        NCCL_M2N_CUDACHECK_WARN(cudaSetDevice(s.cudaDev));
      }
      NCCL_M2N_CUDACHECK_WARN(cudaFree(s.dev));
    }
    s = {};
  }
  for (int i = 0; i < gSplitMetadataScratchCount; i++) {
    SplitMetadataScratch& s = gSplitMetadataScratch[i];
    if (s.dev != nullptr) {
      if (getDeviceResult == cudaSuccess && s.cudaDev >= 0) {
        NCCL_M2N_CUDACHECK_WARN(cudaSetDevice(s.cudaDev));
      }
      NCCL_M2N_CUDACHECK_WARN(cudaFree(s.dev));
    }
    s = {};
  }
  if (getDeviceResult == cudaSuccess && savedCudaDev >= 0) {
    NCCL_M2N_CUDACHECK_WARN(cudaSetDevice(savedCudaDev));
  }
  gSplitBroadcastScratchCount = 0;
  gSplitMetadataScratchCount = 0;
}

static void commBHashBytes(CommBGroupFingerprint* hash, const void* data, size_t bytes) {
  const unsigned char* p = static_cast<const unsigned char*>(data);
  for (size_t i = 0; i < bytes; i++) {
    hash->lo ^= static_cast<uint64_t>(p[i]);
    hash->lo *= 1099511628211ULL;
    hash->hi ^= static_cast<uint64_t>(p[i] + 0x9dU);
    hash->hi *= 14029467366897019727ULL;
  }
}

template <typename Metadata>
static ncclResult_t gatherSplitMetadata(ncclComm_t comm, int parentSize, int parentRank, const Metadata& local,
                                        cudaStream_t stream, std::vector<Metadata>* metadata) {
  NCCL_M2N_CHECK_ARG(metadata != nullptr && parentSize > 0, parentRank,
                     "split commB metadata gather requires a positive parent size and output");
  metadata->assign(parentSize, {});
  const size_t sendBytes = sizeof(local);
  const size_t recvBytes = static_cast<size_t>(parentSize) * sendBytes;
  unsigned char* scratch = nullptr;
  {
    std::lock_guard<std::mutex> guard(gSplitBroadcastScratchMutex);
    NCCL_M2N_CHECK(getSplitMetadataScratch(sendBytes + recvBytes, &scratch));
  }
  unsigned char* send = scratch;
  unsigned char* recv = scratch + sendBytes;

  M2nApiUnlock apiUnlock;
  cudaError_t cerr = cudaMemcpyAsync(send, &local, sendBytes, cudaMemcpyHostToDevice, stream);
  ncclResult_t rc = ncclSuccess;
  if (cerr == cudaSuccess) {
    rc = ncclAllGather(send, recv, sendBytes, ncclUint8, comm, stream);
    if (rc == ncclSuccess || rc == ncclInProgress) rc = m2nWaitCommReady(comm);
  }
  if (cerr == cudaSuccess && rc == ncclSuccess) {
    cerr = cudaMemcpyAsync(metadata->data(), recv, recvBytes, cudaMemcpyDeviceToHost, stream);
  }
  if (cerr == cudaSuccess && rc == ncclSuccess) cerr = cudaStreamSynchronize(stream);
  if (rc != ncclSuccess) {
    NCCL_M2N_FAIL(rc, parentRank, "NCCL allgather for split commB metadata failed: %s", ncclGetErrorString(rc));
  }
  if (cerr != cudaSuccess) {
    NCCL_M2N_FAIL(ncclSystemError, parentRank, "CUDA error in split commB metadata gather: %s",
                  cudaGetErrorString(cerr));
  }
  return ncclSuccess;
}

static ncclResult_t gatherCommBParticipantMetadata(ncclComm_t comm, int parentSize, int parentRank, bool isDst,
                                                    int resourceLane, cudaStream_t stream,
                                                    std::vector<CommBParticipantMetadata>* metadata) {
  int cudaDev = -1;
  NCCL_M2N_CUDACHECK(cudaGetDevice(&cudaDev));
  cudaDeviceProp deviceProp{};
  NCCL_M2N_CUDACHECK(cudaGetDeviceProperties(&deviceProp, cudaDev));
  CommBParticipantMetadata local{};
  memcpy(&local.deviceUuidLo, deviceProp.uuid.bytes, sizeof(local.deviceUuidLo));
  memcpy(&local.deviceUuidHi, deviceProp.uuid.bytes + sizeof(local.deviceUuidLo), sizeof(local.deviceUuidHi));
  local.resourceLane = resourceLane;
  local.isDst = isDst ? 1 : 0;
  return gatherSplitMetadata(comm, parentSize, parentRank, local, stream, metadata);
}

static bool deriveCommBPoolKey(const std::vector<CommBParticipantMetadata>& metadata, int dstStart, int dstSize,
                               bool physicalLane, int* resourceLane, CommBGroupFingerprint* fingerprint) {
  if (resourceLane == nullptr || fingerprint == nullptr || dstStart < 0 || dstSize <= 0 ||
      dstStart + dstSize > static_cast<int>(metadata.size())) {
    return false;
  }
  *resourceLane = metadata[dstStart].resourceLane;
  if (physicalLane && *resourceLane < 0) return false;
  *fingerprint = {1469598103934665603ULL, 1609587929392839161ULL};
  for (int rank = 0; rank < static_cast<int>(metadata.size()); rank++) {
    const bool expectedDst = rank >= dstStart && rank < dstStart + dstSize;
    if ((metadata[rank].isDst != 0) != expectedDst) return false;
    if (!expectedDst) continue;
    if (physicalLane && metadata[rank].resourceLane != *resourceLane) return false;
    commBHashBytes(fingerprint, &metadata[rank].deviceUuidLo, sizeof(metadata[rank].deviceUuidLo));
    commBHashBytes(fingerprint, &metadata[rank].deviceUuidHi, sizeof(metadata[rank].deviceUuidHi));
    const int rankInB = rank - dstStart;
    commBHashBytes(fingerprint, &rankInB, sizeof(rankInB));
  }
  return true;
}

CommBSharedEntry* findCommBShared(ncclComm_t parent, int srcStart, int srcSize, int dstStart, int dstSize) {
  for (int i = 0; i < gCommBSharedCount; i++) {
    if (gCommBShared[i].used && gCommBShared[i].parent == parent && gCommBShared[i].srcStart == srcStart &&
        gCommBShared[i].srcSize == srcSize && gCommBShared[i].dstStart == dstStart &&
        gCommBShared[i].dstSize == dstSize) {
      return &gCommBShared[i];
    }
  }
  return nullptr;
}

CommBSharedEntry* findCommBPoolOwner(int dstSize, const CommBGroupFingerprint& groupFingerprint) {
  for (int i = 0; i < gCommBSharedCount; i++) {
    CommBSharedEntry& e = gCommBShared[i];
    if (e.used && e.pooled && e.ownsCommB && e.dstSize == dstSize && e.groupFingerprint == groupFingerprint) {
      return &e;
    }
  }
  return nullptr;
}

CommBParentEntry* findCommBParent(ncclComm_t comm, int srcStart, int srcSize, int dstStart, int dstSize) {
  for (int i = 0; i < gCommBParentCount; i++) {
    if (gCommBParent[i].used && gCommBParent[i].parent == comm && gCommBParent[i].srcStart == srcStart &&
        gCommBParent[i].srcSize == srcSize && gCommBParent[i].dstStart == dstStart &&
        gCommBParent[i].dstSize == dstSize) {
      return &gCommBParent[i];
    }
  }
  return nullptr;
}

CommACacheEntry* findCommA(ncclComm_t comm, const CommBParentEntry* b, int K) {
  CommACacheEntry* best = nullptr;
  for (int i = 0; i < gCommACount; i++) {
    if (gCommACache[i].used && gCommACache[i].parent == comm && gCommACache[i].srcStart == b->srcStart &&
        gCommACache[i].srcSize == b->srcSize && gCommACache[i].dstStart == b->dstStart &&
        gCommACache[i].dstSize == b->dstSize && gCommACache[i].K >= K) {
      /* Keep the smallest sufficient communicator to avoid unused injection domains. */
      if (best == nullptr || gCommACache[i].K < best->K) {
        best = &gCommACache[i];
      }
    }
  }
  return best;
}

bool samePersistentControlPool(const PersistentControlSlotEntry& e, ncclComm_t parentComm) {
  return e.used && e.parent == parentComm;
}

/* Create the minimal DevComm on `comm` whose only purpose is to report
 * lsaSize.  lsaSize is the LSA (NVLink/same-node) team size, populated from
 * comm topology at devcomm creation.  We request a cheap GIN connection
 * (RAIL via NCCLM2N_GIN_RAIL_CONNECTION at the call sites) with ZERO GIN
 * resources (no signals/contexts/barriers): RAIL allocates only rail-local
 * QPs, unlike a FULL probe's all-to-all base-RMA set.
 *
 * NOTE: a GIN-LESS probe (ginConnectionType = NONE) is NOT usable on this
 * NCCL build -- ncclDevCommCreate rejects a NONE DevComm on a multi-domain
 * comm with ncclInvalidUsage (observed at 64 nodes).  RAIL is the proven,
 * low-cost connection (commB's real kernel DevComm is itself RAIL), and
 * NCCL accepts RAIL on any GIN-capable comm (the only RAIL guard rejects
 * comms whose globalGinSupport == NONE).
 *
 * Set ginConnectionType directly; the NCCL M2N compatibility floor is
 * NCCL 2.30.5, where this field is available. */
ncclResult_t createProbeDevComm(ncclComm_t comm, int ginConnectionType, ncclDevComm* out) {
  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  /* Request the chosen GIN connection but no GIN resources; lsaSize is read
   * from comm topology regardless of resource counts. */
  reqs.ginConnectionType = (decltype(reqs.ginConnectionType))ginConnectionType;
  memset(out, 0, sizeof(*out));
  {
    M2nApiUnlock apiUnlock;
    NCCL_M2N_CHECK(ncclDevCommCreate(comm, &reqs, out));
    NCCL_M2N_CHECK(m2nWaitCommReady(comm));
  }
  return ncclSuccess;
}

/* AllReduce(MAX) a single int over the parent comm so every rank learns
 * the value contributed by the generator ranks (trainer ranks pass 0). */
static ncclResult_t broadcastMaxInt(ncclComm_t comm, cudaStream_t stream, int localValue, int* out) {
  M2nApiUnlock apiUnlock;
  int* dev = nullptr;
  {
    std::lock_guard<std::mutex> guard(gSplitBroadcastScratchMutex);
    NCCL_M2N_CHECK(getSplitBroadcastScratch(&dev));
  }
  int host = localValue;
  ncclResult_t rc = ncclSuccess;
  /* FIXME: Simplify these staged checks after the device allocation has automatic cleanup. */
  cudaError_t cerr = cudaMemcpyAsync(dev, &host, sizeof(int), cudaMemcpyHostToDevice, stream);
  if (cerr == cudaSuccess) {
    /* Drive the AllReduce to completion (a blocking comm returns ncclSuccess
     * immediately; a non-blocking comm is spun until the collective is enqueued
     * on `stream`) BEFORE enqueuing the device->host copy of the reduced value,
     * so the D2H never reads an un-enqueued result. Scratch is per CUDA device,
     * so local NCCL ranks use distinct allocations. */
    rc = ncclAllReduce(dev, dev + 1, 1, ncclInt, ncclMax, comm, stream);
    if (rc == ncclSuccess || rc == ncclInProgress) {
      rc = m2nWaitCommReady(comm);
    }
  }
  if (cerr == cudaSuccess && rc == ncclSuccess) {
    cerr = cudaMemcpyAsync(&host, dev + 1, sizeof(int), cudaMemcpyDeviceToHost, stream);
  }
  if (cerr == cudaSuccess && rc == ncclSuccess) {
    cerr = cudaStreamSynchronize(stream);
  }
  if (rc != ncclSuccess) {
    NCCL_M2N_FAIL(rc, -1, "NCCL allreduce in split lsaSize broadcast failed: %s", ncclGetErrorString(rc));
  }
  if (cerr != cudaSuccess) {
    NCCL_M2N_FAIL(ncclSystemError, -1, "CUDA error in split lsaSize broadcast: %s", cudaGetErrorString(cerr));
  }
  *out = host;
  return ncclSuccess;
}

/* Form or fetch commB (RAIL, all generator ranks) and probe its NVL-domain
 * size. Parents with the same ordered generator group share one commB. PACK
 * maps the current physical staging slot to a resource lane; PIPE derives
 * its lane from the agreed persistent parent namespace.
 *
 * MUST be called collectively by every parent-comm rank. It only forms and
 * probes commB; dispatch sites decide whether split comms are needed. */
ncclResult_t ensureCommBShared(ncclComm_t comm, const ncclMesh_t* srcMesh, const ncclMesh_t* dstMesh,
                               cudaStream_t stream, CommBSharedEntry** out) {
  int parentRank = -1, parentSize = 0;
  NCCL_M2N_CHECK(ncclCommUserRank(comm, &parentRank));
  NCCL_M2N_CHECK(ncclCommCount(comm, &parentSize));

  const int dstStart = dstMesh->startRank;
  const int dstSize = dstMesh->dims[0] * dstMesh->dims[1];
  const int srcStart = srcMesh->startRank;
  const int srcSize = srcMesh->dims[0] * srcMesh->dims[1];
  const bool isDst = (parentRank >= dstStart && parentRank < dstStart + dstSize);

  CommBSharedEntry* shared = findCommBShared(comm, srcStart, srcSize, dstStart, dstSize);

  const int resourceLaneCount = reshardGetSplitSlotCount();
  NCCL_M2N_CHECK_ARG(resourceLaneCount > 0 && resourceLaneCount <= MAX_SPLIT_CONCURRENCY, parentRank,
                     "split-comm: invalid commB resource-lane count %d", resourceLaneCount);
  int proposedResourceLane = -1;
  const bool physicalLane = reshardGetCopyAlgorithm() == RESHARD_COPY_ALGO_PACK;
  if (physicalLane) {
    proposedResourceLane = getStagingPhysicalSlotIndex(comm);
  }

  int resourceLane = -1;
  CommBGroupFingerprint groupFingerprint{};
  std::vector<CommBParticipantMetadata> metadata;
  NCCL_M2N_CHECK(gatherCommBParticipantMetadata(comm, parentSize, parentRank, isDst, proposedResourceLane, stream,
                                                &metadata));
  bool poolingEligible = deriveCommBPoolKey(metadata, dstStart, dstSize, physicalLane, &resourceLane,
                                            &groupFingerprint) &&
                         (!physicalLane || (resourceLane >= 0 && resourceLane < resourceLaneCount));
  if (!poolingEligible && parentRank == 0) {
    RESHARD_WARN(parentRank,
                 "split-comm: generator resource lanes or participant metadata disagree; "
                 "falling back to the parent DevComm");
  }
  if (!poolingEligible) {
    *out = nullptr;
    return ncclSuccess;
  }

  CommBSharedEntry* reusable = shared;
  if (reusable == nullptr && poolingEligible && isDst) {
    reusable = findCommBPoolOwner(dstSize, groupFingerprint);
    if (reusable != nullptr && reusable->rankInB != parentRank - dstStart) reusable = nullptr;
  }

  /* Distinguish all-hit, all-miss, and partial-hit states among generator
   * ranks. A partial hit must not mix independent NCCL communicator handles;
   * it falls back to the unsplit parent DevComm. */
  const int localHasCommB = (isDst && reusable != nullptr && reusable->commB != nullptr) ? 1 : 0;
  const int localMissingCommB = isDst && !localHasCommB ? 1 : 0;
  int anyHasCommB = 0, anyMissingCommB = 0;
  NCCL_M2N_CHECK(broadcastMaxInt(comm, stream, localHasCommB, &anyHasCommB));
  NCCL_M2N_CHECK(broadcastMaxInt(comm, stream, localMissingCommB, &anyMissingCommB));
  const bool reuseCommB = anyHasCommB != 0 && anyMissingCommB == 0;
  const bool partialPoolHit = anyHasCommB != 0 && anyMissingCommB != 0;
  if (partialPoolHit) {
    if (parentRank == 0) {
      RESHARD_WARN(parentRank,
                   "split-comm: partial generator commB pool hit; falling back to the parent DevComm");
    }
    *out = nullptr;
    return ncclSuccess;
  }

  const int localCacheFull = (shared == nullptr && gCommBSharedCount >= kMaxSplitCommEntries) ? 1 : 0;
  int cacheFull = 0;
  NCCL_M2N_CHECK(broadcastMaxInt(comm, stream, localCacheFull, &cacheFull));
  if (cacheFull) {
    NCCL_M2N_FAIL(ncclInvalidArgument, parentRank,
                  "split-comm: commB shared cache full (%d); refusing gen set [%d,%d). Increase "
                  "kMaxSplitCommEntries",
                  kMaxSplitCommEntries, dstStart, dstStart + dstSize);
  }

  ncclComm_t commB = nullptr;
  ncclDevComm probeDevComm{};
  bool probeValid = false;
  int localLsa = 0;
  int rankInB = -1;
  bool ownsCommB = false;
  int poolOwnerIndex = -1;

  if (reuseCommB) {
    /* Only generator ranks own handles; trainer ranks learn the topology from
     * the lsaSize broadcast below and store a null-handle parent alias. */
    if (isDst && reusable != nullptr) {
      commB = reusable->commB;
      probeDevComm = reusable->probeDevComm;
      probeValid = reusable->probeValid;
      localLsa = reusable->lsaSize;
      rankInB = reusable->rankInB;
      poolOwnerIndex = reusable->poolOwnerIndex;
      gCommBReuseCount++;
    }
  } else {
    /* Form commB (all generator ranks) — collective over the parent.
     *
     * To make commB natively RAIL (globalGinSupport == NCCL_GIN_CONNECTION_RAIL)
     * we temporarily force NCCL_CROSS_NIC=0 across the split.  ncclCommSplit
     * recomputes globalGinSupport from ncclParamCrossNic() during
     * initTransportsRank and freezes it on the new comm, so only this split
     * call needs the override.  ncclGinConnectOnce then builds commB's GIN
     * connection over the rail team only (nRanks/lsaSize peers) and hands that
     * reduced peer list to the backend — the GDAKI backend honors this at the
     * connection level even though it ignores the FULL-comm rankStride path.
     * All parent ranks run this collectively (NCCL ANDs crossNicSupport across
     * every rank, so the override must be symmetric).  Needs
     * NCCL_NO_CACHE=NCCL_CROSS_NIC so the param is re-read here instead of
     * served from its first-read cache.  commA is split with NCCL_CROSS_NIC
     * restored and stays FULL (all-to-all). */
    const int colorB = isDst ? 0 : NCCL_SPLIT_NOCOLOR;
    const bool forceRail = reshardGetCommBForceRail();
    {
      ScopedCrossNicRailOverride crossNicOverride(forceRail);
      if (forceRail && parentRank == 0) {
        RESHARD_INFO(parentRank, "split-comm: commB forced RAIL via NCCL_CROSS_NIC=0; "
                                 "requires NCCL_NO_CACHE=NCCL_CROSS_NIC to take effect");
      }
      {
        M2nApiUnlock apiUnlock;
        NCCL_M2N_CHECK(ncclCommSplit(comm, colorB, parentRank, &commB, nullptr));
        NCCL_M2N_CHECK(m2nWaitCommReady(comm));
      }
      if (isDst) {
        NCCL_M2N_CHECK_ARG(commB != nullptr, parentRank, "split-comm: ncclCommSplit did not publish commB");
        M2nApiUnlock apiUnlock;
        NCCL_M2N_CHECK(m2nWaitCommReady(commB));
      }
    }

    /* Probe commB for lsaSize (generator ranks only). */
    if (isDst && commB != nullptr) {
      NCCL_M2N_CHECK(createProbeDevComm(commB, NCCLM2N_GIN_RAIL_CONNECTION, &probeDevComm));
      probeValid = true;
      localLsa = (probeDevComm.lsaSize > 0) ? probeDevComm.lsaSize : 0;
      NCCL_M2N_CHECK(ncclCommUserRank(commB, &rankInB));
      ownsCommB = true;
      poolOwnerIndex = gCommBSharedCount;
      gCommBCreateCount++;
    }
  }

  /* Broadcast lsaSize to every parent-comm rank (trainers contribute 0).
   * Keeps the collective symmetric and lets trainer ranks learn the gen
   * NVL-domain size for their per-parent eligibility / K computation. */
  int lsaSize = 0;
  NCCL_M2N_CHECK(broadcastMaxInt(comm, stream, localLsa, &lsaSize));

  int numGenDomains = 0;
  if (lsaSize > 0 && dstSize > 0) {
    numGenDomains = (dstSize + lsaSize - 1) / lsaSize;
  }

  if (reuseCommB && shared != nullptr) {
    /* Already cached for this rank; nothing to (re)store. */
    *out = shared;
    return ncclSuccess;
  }

  int pipeNamespaceId = -1;
  int pipeNamespaceCapacity = 0;
  if (reshardGetCopyAlgorithm() == RESHARD_COPY_ALGO_PIPE) {
    const int capacity = gReshardPipeCommBParentCapacity;
    PipeNamespaceMetadata local{-1, capacity, resourceLaneCount, reshardGetPipeGinPeersPerSlot(),
                                reshardGetPipeGinChannelsPerPeer(), reshardGetGinContextCount()};
    ReshardPipeNamespacePool emptyPool{};
    if (isDst) {
      const ReshardPipeNamespacePool& pool = reuseCommB ?
        gCommBShared[poolOwnerIndex].pipeNamespaces : emptyPool;
      local.id = pool.propose(reinterpret_cast<uintptr_t>(comm), capacity);
    }
    std::vector<PipeNamespaceMetadata> namespaceMetadata;
    NCCL_M2N_CHECK(gatherSplitMetadata(comm, parentSize, parentRank, local, stream, &namespaceMetadata));
    const PipeNamespaceMetadata& agreed = namespaceMetadata[dstStart];
    NCCL_M2N_CHECK_ARG(agreed.id >= 0 && agreed.id < agreed.capacity &&
                         agreed.capacity <= RESHARD_PIPE_MAX_PARENT_NAMESPACES, parentRank,
                       "split PIPE commB parent namespace capacity exhausted or invalid (%d); "
                       "set NCCL_RESHARD_GEN_COMM_PARENT_CAPACITY before runtime initialization", agreed.capacity);
    for (int rank = 0; rank < parentSize; rank++) {
      const PipeNamespaceMetadata& peer = namespaceMetadata[rank];
      const bool generator = rank >= dstStart && rank < dstStart + dstSize;
      NCCL_M2N_CHECK_ARG((!generator || peer.id == agreed.id) && peer.capacity == agreed.capacity &&
                           peer.lanes == agreed.lanes &&
                           peer.peersPerControl == agreed.peersPerControl &&
                           peer.channelsPerPeer == agreed.channelsPerPeer && peer.contextsPerLane == agreed.contextsPerLane,
                         parentRank, "split PIPE commB namespace/resource-plan disagreement at parent rank %d", rank);
    }
    pipeNamespaceId = agreed.id;
    pipeNamespaceCapacity = agreed.capacity;
    resourceLane = reshardPipeResourceLane(pipeNamespaceId, agreed.lanes);
  }

  /* Store a new shared entry.  Even trainer-only ranks store one (commB
   * == NULL) so their per-rank cache state is consistent with future
   * lookups; they never use the commB handle. */
  CommBSharedEntry e;
  memset(&e, 0, sizeof(e));
  e.parent = comm;
  e.srcStart = srcStart;
  e.dstStart = dstStart;
  e.dstSize = dstSize;
  e.srcSize = srcSize;
  e.resourceLane = resourceLane;
  e.resourceLaneCount = resourceLaneCount;
  e.poolOwnerIndex = poolOwnerIndex;
  e.groupFingerprint = groupFingerprint;
  e.pipeNamespaceId = pipeNamespaceId;
  e.pipeNamespaceCapacity = pipeNamespaceCapacity;
  e.commB = commB;
  e.probeDevComm = probeDevComm;
  e.probeValid = probeValid;
  e.lsaSize = lsaSize;
  e.numGenDomains = numGenDomains;
  e.rankInB = rankInB;
  e.isDst = isDst;
  e.pooled = poolingEligible;
  e.ownsCommB = ownsCommB;
  e.used = true;

  gCommBShared[gCommBSharedCount] = e;
  if (isDst && pipeNamespaceId >= 0) {
    CommBSharedEntry& owner = gCommBShared[poolOwnerIndex];
    NCCL_M2N_CHECK_ARG(owner.pipeNamespaces.commit(reinterpret_cast<uintptr_t>(comm), pipeNamespaceCapacity,
                                                   pipeNamespaceId),
                       parentRank, "split PIPE commB namespace commit disagrees with verified proposal");
  }
  *out = &gCommBShared[gCommBSharedCount];
  gCommBSharedCount++;
  if (parentRank == dstStart) {
    RESHARD_INFO(parentRank,
                 "split-comm: commB-pool action=%s pooled=%d resourceLane=%d/%d group=%016llx:%016llx "
                 "dst=[%d,%d) ownersCreated=%d aliasesReused=%d pipeNamespace=%d/%d",
                 reuseCommB ? "reuse" : "create", (int)poolingEligible, resourceLane, resourceLaneCount,
                 (unsigned long long)groupFingerprint.hi, (unsigned long long)groupFingerprint.lo, dstStart,
                 dstStart + dstSize, gCommBCreateCount, gCommBReuseCount, pipeNamespaceId, pipeNamespaceCapacity);
  }
  return ncclSuccess;
}

/* Form (or fetch the cached) commA = all source ranks + the first K gen
 * NVL domains, and probe the source (trainer) NVL-domain size.  commA
 * membership depends on K, so it is cached keyed by (parent comm, K).
 * MUST be called collectively by every parent-comm rank (all ranks agree
 * on K for a given parameter). */
ncclResult_t ensureCommA(ncclComm_t comm, const CommBParentEntry* b, int K, cudaStream_t stream,
                         CommACacheEntry** out) {
  CommACacheEntry* hit = findCommA(comm, b, K);
  if (hit != nullptr) {
    *out = hit;
    return ncclSuccess;
  }

  const int parentRank = b->parentRank;
  const int localCacheFull = (gCommACount >= kMaxSplitCommEntries) ? 1 : 0;
  int cacheFull = 0;
  NCCL_M2N_CHECK(broadcastMaxInt(comm, stream, localCacheFull, &cacheFull));
  if (cacheFull) {
    NCCL_M2N_FAIL(ncclInvalidArgument, parentRank,
                  "split-comm: commA cache full (%d); refusing comm %p K=%d. Increase kMaxSplitCommEntries",
                  kMaxSplitCommEntries, (void*)comm, K);
  }
  const bool isSrc = (parentRank >= b->srcStart && parentRank < b->srcStart + b->srcSize);
  /* commB's LSA teams are lsaSize-sized contiguous blocks of gen ranks in
   * parent order, so the first K injection domains are parent ranks
   * [dstStart, dstStart + K*lsaSize). */
  const int commAGenRanks = K * b->lsaSize;
  const bool inInjectionDomains = b->isDst && (parentRank < b->dstStart + commAGenRanks);
  const bool inA = isSrc || inInjectionDomains;

  ncclComm_t commA = nullptr;
  {
    const int colorA = inA ? 0 : NCCL_SPLIT_NOCOLOR;
    const int keyA = isSrc ? (parentRank - b->srcStart) : (b->srcSize + parentRank - b->dstStart);
    M2nApiUnlock apiUnlock;
    NCCL_M2N_CHECK(ncclCommSplit(comm, colorA, keyA, &commA, nullptr));
    NCCL_M2N_CHECK(m2nWaitCommReady(comm));
  }

  int rankInA = -1;
  if (inA) {
    NCCL_M2N_CHECK_ARG(commA != nullptr, parentRank, "split-comm: ncclCommSplit did not publish commA");
    {
      M2nApiUnlock apiUnlock;
      NCCL_M2N_CHECK(m2nWaitCommReady(commA));
    }
    NCCL_M2N_CHECK(ncclCommUserRank(commA, &rankInA));
  }

  /* Probe commA for the SOURCE (trainer) NVL-domain size: only source
   * ranks contribute (they report the trainer domain), MAX-broadcast over
   * the parent comm.  The probe DevComm is created by ALL commA members. */
  ncclDevComm probeDevCommA;
  bool probeAValid = false;
  int srcLocalLsa = 0;
  if (inA && commA != nullptr) {
    NCCL_M2N_CHECK(createProbeDevComm(commA, NCCLM2N_GIN_RAIL_CONNECTION, &probeDevCommA));
    probeAValid = true;
    if (isSrc) {
      srcLocalLsa = (probeDevCommA.lsaSize > 0) ? probeDevCommA.lsaSize : 0;
    }
  }
  int srcLsaSize = 0;
  NCCL_M2N_CHECK(broadcastMaxInt(comm, stream, srcLocalLsa, &srcLsaSize));

  CommACacheEntry e;
  memset(&e, 0, sizeof(e));
  e.parent = comm;
  e.srcStart = b->srcStart;
  e.srcSize = b->srcSize;
  e.dstStart = b->dstStart;
  e.dstSize = b->dstSize;
  e.K = K;
  e.commA = commA;
  e.probeDevCommA = probeDevCommA;
  e.probeAValid = probeAValid;
  e.rankInA = rankInA;
  e.srcLsaSize = srcLsaSize;
  e.inA = inA;
  e.used = true;

  gCommACache[gCommACount] = e;
  *out = &gCommACache[gCommACount];
  gCommACount++;
  return ncclSuccess;
}

} // namespace

#ifdef NCCL_M2N_TESTING
ncclResult_t reshardTestBroadcastMaxInt(ncclComm_t comm, cudaStream_t stream, int localValue, int* out) {
  return broadcastMaxInt(comm, stream, localValue, out);
}
#endif

int reshardGetPersistentControlSlotCount() {
  return kPersistentControlSlots;
}

void stagingPipeControlSlotCacheReset() {
  for (int i = 0; i < gPersistentControlSlotCount; i++) {
    gPersistentControlSlots[i] = {};
  }
  gPersistentControlSlotCount = 0;
}

ncclResult_t reshardGetOrCreatePersistentControlSlot(ncclComm_t parentComm,
                                                     const ReshardStagingPipeSignature& signature, int rank,
                                                     int* outSlot) {
  NCCL_M2N_CHECK_ARG(parentComm != nullptr && outSlot != nullptr, rank,
                     "PIPE persistent control slot requires parent comm and output");

  bool usedSlots[kPersistentControlSlots] = {};
  for (int i = 0; i < gPersistentControlSlotCount; i++) {
    PersistentControlSlotEntry& e = gPersistentControlSlots[i];
    if (!samePersistentControlPool(e, parentComm)) {
      continue;
    }
    if (e.signature == signature) {
      *outSlot = e.slot;
      return ncclSuccess;
    }
    if (e.slot >= 0 && e.slot < kPersistentControlSlots) {
      usedSlots[e.slot] = true;
    }
  }

  int slot = -1;
  for (int i = 0; i < kPersistentControlSlots; i++) {
    if (!usedSlots[i]) {
      slot = i;
      break;
    }
  }
  NCCL_M2N_CHECK_ARG(slot >= 0, rank, "PIPE persistent control slots exhausted (%d) for parent comm %p",
                     kPersistentControlSlots, (void*)parentComm);
  NCCL_M2N_CHECK_ARG(gPersistentControlSlotCount < kMaxPersistentControlSlotEntries, rank,
                     "PIPE persistent control-slot cache full (%d)", kMaxPersistentControlSlotEntries);

  PersistentControlSlotEntry e;
  memset(&e, 0, sizeof(e));
  e.parent = parentComm;
  e.signature = signature;
  e.slot = slot;
  e.used = true;
  gPersistentControlSlots[gPersistentControlSlotCount++] = e;
  *outSlot = slot;
  return ncclSuccess;
}

/* Phase gate: the strided node assignment in the load balancer (needed
 * when srcRepCount < numGenDomains so injections funnel into the first K
 * domains) is implemented in reshard_loadbalance.cc / reshard_prepare.cc
 * (strided getTargetRepRange/getSourceRepForDest + K-stepped ring walk).
 * When false we fall back to the single-DevComm path for the strided
 * regime; when true the split path serves it too. */
static constexpr bool kSplitStridedSupported = true;
static constexpr int kSplitRepRepKBound = 4;

ncclResult_t reshardGetOrCreateSplitComms(ncclComm_t comm, const ncclMesh_t* srcMesh, const ncclMesh_t* dstMesh,
                                          int srcRepCount, int dstRepCount, bool dstRepStrided, int numCtas,
                                          cudaStream_t stream, ReshardSplitComms* out) {
  NCCL_M2N_CHECK_ARG(out != nullptr && comm != nullptr && srcMesh != nullptr && dstMesh != nullptr, -1,
                     "reshardGetOrCreateSplitComms: comm, srcMesh, dstMesh, and output must be non-null");
  memset(out, 0, sizeof(*out));
  out->commA = nullptr;
  out->commB = nullptr;
  out->rankInA = -1;
  out->rankInB = -1;
  out->slotIdx = -1;
  out->commBPoolOwnerIndex = -1;

  // Reject the entire multi-bucket profile before any split setup collective.
  if (reshardGetCopyAlgorithm() == RESHARD_COPY_ALGO_PACK && gReshardStagingBucketCount > 1) {
    out->parentComm = comm;
    return ncclSuccess;
  }

  {
    M2nApiUnlock apiUnlock;
    NCCL_M2N_CHECK(m2nWaitCommReady(comm));
  }
  int parentRank = -1, parentSize = 0;
  NCCL_M2N_CHECK(ncclCommUserRank(comm, &parentRank));
  NCCL_M2N_CHECK(ncclCommCount(comm, &parentSize));

  const int srcStart = srcMesh->startRank;
  const int srcSize = srcMesh->dims[0] * srcMesh->dims[1];
  const int dstStart = dstMesh->startRank;
  const int dstSize = dstMesh->dims[0] * dstMesh->dims[1];

  /* commA orders source ranks before destination-only ranks. The split
   * parameter translation currently relies on that disjoint layout. */
  const bool meshesOverlap = srcStart < dstStart + dstSize && dstStart < srcStart + srcSize;
  if (meshesOverlap) {
    RESHARD_INFO(parentRank, "split-comm: overlapping source/destination meshes -> fallback to parent DevComm");
    out->parentComm = comm;
    return ncclSuccess;
  }

  /* On the first reshard for this parent comm, lazily form or load commB. */
  CommBParentEntry* pe = findCommBParent(comm, srcStart, srcSize, dstStart, dstSize);
  CommBSharedEntry* se = nullptr;

  if (pe == nullptr) {
    const int localParentCacheFull = (gCommBParentCount >= kMaxSplitCommEntries) ? 1 : 0;
    int parentCacheFull = 0;
    NCCL_M2N_CHECK(broadcastMaxInt(comm, stream, localParentCacheFull, &parentCacheFull));
    if (parentCacheFull) {
      NCCL_M2N_FAIL(ncclInvalidArgument, parentRank,
                    "split-comm: commB parent cache full (%d); refusing comm %p. Increase kMaxSplitCommEntries",
                    kMaxSplitCommEntries, (void*)comm);
    }

    /* Stage 1: commB, cached for this parent and mesh geometry. */
    NCCL_M2N_CHECK(ensureCommBShared(comm, srcMesh, dstMesh, stream, &se));
    if (se == nullptr) {
      out->active = false;
      out->parentComm = comm;
      return ncclSuccess;
    }

    CommBParentEntry e;
    memset(&e, 0, sizeof(e));
    e.parent = comm;
    e.dstStart = dstStart;
    e.dstSize = dstSize;
    e.lsaSize = se->lsaSize;
    e.numGenDomains = se->numGenDomains;
    e.parentRank = parentRank;
    e.parentSize = parentSize;
    e.srcStart = srcStart;
    e.srcSize = srcSize;
    e.isDst = se->isDst;
    e.rankInB = se->rankInB;
    e.resourceLane = se->resourceLane;
    e.resourceLaneCount = se->resourceLaneCount;
    e.poolOwnerIndex = se->poolOwnerIndex;
    e.pipeNamespaceId = se->pipeNamespaceId;
    e.pipeNamespaceCapacity = se->pipeNamespaceCapacity;
    e.used = true;

    gCommBParent[gCommBParentCount] = e;
    pe = &gCommBParent[gCommBParentCount];
    gCommBParentCount++;
  } else {
    se = findCommBShared(comm, pe->srcStart, pe->srcSize, pe->dstStart, pe->dstSize);
  }

  /* Base geometry (returned regardless of activation). */
  ReshardSplitComms r;
  memset(&r, 0, sizeof(r));
  r.parentComm = comm;
  r.lsaSize = pe->lsaSize;
  r.numGenDomains = pe->numGenDomains;
  r.parentRank = parentRank;
  r.parentSize = parentSize;
  r.srcStartRank = srcStart;
  r.srcMeshSize = srcSize;
  r.dstStartRank = dstStart;
  r.dstMeshSize = dstSize;
  r.commB = (se != nullptr) ? se->commB : nullptr;
  r.inB = pe->isDst;
  r.rankInB = pe->rankInB;
  r.slotIdx = pe->resourceLane;
  r.slotCount = pe->resourceLaneCount;
  r.commBPoolOwnerIndex = pe->poolOwnerIndex;
  r.pipeNamespaceId = pe->pipeNamespaceId;
  r.pipeNamespaceCapacity = pe->pipeNamespaceCapacity;
  r.valid = true;

  const int maxConcurrency = pe->resourceLaneCount;
  if (se == nullptr) {
    RESHARD_INFO(parentRank, "split-comm: missing cached commB shared state; falling back to parent DevComm");
    r.active = false;
    *out = r;
    return ncclSuccess;
  }

  /* K = number of injection NVL domains.  commA's generator block is the
   * first K*lsaSize generator ranks and must hold one COMPLETE destination
   * replica so the commB ring can replicate it to the remaining copies.
   *
   * A destination replica spans `domainsPerRep` NVL domains:
   *   domainsPerRep = numGenDomains / dstRepCount  (>= 1).
   *   - reps/domain >= 1 (case 1): a replica fits within one domain, so
   *     domainsPerRep clamps to 1 and K reduces to min(srcRepCount,
   *     numGenDomains) -- identical to the original behavior.
   *   - reps/domain <  1 (case 2): a replica spans >1 domains, so each
   *     injected replica needs domainsPerRep domains in commA.
   *
   * We directly inject `numInjectionReps = min(srcRepCount, dstRepCount)`
   * replicas (preserving the "parallel multi-inject with min" policy), each
   * occupying domainsPerRep domains, capped at the whole generator.  The
   * commB ring then strides by K domains to the remaining replicas.
   * `needStrided` is true whenever commA does not already span every gen
   * domain (i.e. some domains receive their copy via the ring). */
  if (!dstRepStrided && dstRepCount > 0 && pe->numGenDomains >= dstRepCount && pe->numGenDomains % dstRepCount != 0) {
    RESHARD_INFO(parentRank, "split-comm: generator domains (%d) do not divide destination replicas (%d) -> fallback",
                 pe->numGenDomains, dstRepCount);
    r.active = false;
    *out = r;
    return ncclSuccess;
  }
  int domainsPerRep = dstRepStrided ? pe->numGenDomains : 1;
  if (!dstRepStrided && dstRepCount > 0 && pe->numGenDomains >= dstRepCount) {
    domainsPerRep = pe->numGenDomains / dstRepCount;
    if (domainsPerRep < 1) domainsPerRep = 1;
  }
  int numInjectionReps = pe->numGenDomains;
  if (srcRepCount > 0) {
    numInjectionReps = (srcRepCount < dstRepCount) ? srcRepCount : dstRepCount;
    if (numInjectionReps < 1) numInjectionReps = 1;
  }
  bool singleRepInject = false;
  if (reshardGetSplitSingleRepInject()) {
    if (domainsPerRep > 1) {
      numInjectionReps = 1;
      singleRepInject = true;
    } else if (kSplitRepRepKBound > 0 && numInjectionReps > kSplitRepRepKBound) {
      numInjectionReps = kSplitRepRepKBound;
      singleRepInject = true;
    }
  }
  int numInjectionDomains = numInjectionReps * domainsPerRep;
  if (numInjectionDomains > pe->numGenDomains || numInjectionDomains <= 0) {
    numInjectionDomains = pe->numGenDomains;
  }
  bool needStrided = (numInjectionDomains < pe->numGenDomains);

  if (needStrided && !kSplitStridedSupported) {
    /* Strided LB not active: fall back to the single-DevComm path for
     * THIS parameter only (commB stays cached, other K's unaffected). */
    RESHARD_INFO(parentRank, "split-comm: K=%d strided unsupported -> fallback (numGenDomains=%d srcRepCount=%d)",
                 numInjectionDomains, pe->numGenDomains, srcRepCount);
    r.active = false;
    *out = r;
    return ncclSuccess;
  }

  /* Stage 2: commA for this K (cached per (parent comm, K)). */
  CommACacheEntry* a = nullptr;
  NCCL_M2N_CHECK(ensureCommA(comm, pe, numInjectionDomains, stream, &a));

  const int slotIdx = pe->resourceLane;
  if (slotIdx < 0 || slotIdx >= maxConcurrency) {
    NCCL_M2N_FAIL(ncclInvalidArgument, parentRank, "split-comm: invalid commB resource lane %d for %d partitions",
                  slotIdx, maxConcurrency);
  }

  r.active = true;
  r.numInjectionDomains = numInjectionDomains;
  r.domainsPerRep = domainsPerRep;
  r.strided = needStrided;
  r.commA = a->commA;
  r.inA = a->inA;
  r.rankInA = a->rankInA;
  r.srcLsaSize = a->srcLsaSize;
  r.slotIdx = slotIdx;

  RESHARD_INFO(parentRank,
               "split-comm: parentSize=%d src=[%d,%d) dst=[%d,%d) lsaSize=%d numGenDomains=%d K=%d strided=%d "
               "srcRepCount=%d dstRepCount=%d domainsPerRep=%d singleRepInject=%d slot=%d/%d active=1",
               parentSize, srcStart, srcStart + srcSize, dstStart, dstStart + dstSize, pe->lsaSize, pe->numGenDomains,
               numInjectionDomains, (int)needStrided, srcRepCount, dstRepCount, domainsPerRep, (int)singleRepInject,
               slotIdx, maxConcurrency);

  *out = r;
  return ncclSuccess;
}

ncclResult_t reshardSplitEnsureResources(const ReshardSplitComms* sc, void* stagingBuffer, size_t stagingCapacity,
                                         int barrierCount, ReshardDevCommBarrierKind barrierKind, int ginSignalCountA,
                                         int ginCounterCountA, const ReshardCommBResourcePlan& resourcesB,
                                         cudaStream_t stream,
                                         ncclWindow_t* outWindowA, ncclWindow_t* outWindowB, ncclDevComm* outDevCommA,
                                         ReshardDevCommUse* outDevCommAUse, ncclDevComm* outDevCommB,
                                         ReshardDevCommUse* outDevCommBUse) {
  NCCL_M2N_CHECK_ARG(sc != nullptr && stagingBuffer != nullptr, -1,
                     "reshardSplitEnsureResources: split state and staging buffer must be non-null");
  if (outWindowA != nullptr) *outWindowA = nullptr;
  if (outWindowB != nullptr) *outWindowB = nullptr;
  if (outDevCommA != nullptr) memset(outDevCommA, 0, sizeof(*outDevCommA));
  if (outDevCommB != nullptr) memset(outDevCommB, 0, sizeof(*outDevCommB));
  if (outDevCommBUse != nullptr) *outDevCommBUse = {};

  /* commA (FULL): trainer + first gen NVL domain.  commA is PER-PARENT
   * (not shared), so its window + DevComm stay cached under the commA key;
   * cacheFinalize() tears them down before
   * reshardSplitCommFinalize() destroys commA. */
  if (sc->inA && sc->commA != nullptr) {
    ncclWindow_t* cw =
      findCachedInternalWindowByPtr(sc->commA, stagingBuffer, stagingCapacity, RESHARD_INTERNAL_WINDOW_SPLIT);
    ncclWindow_t winA = nullptr;
    if (cw != nullptr) {
      winA = *cw;
    } else {
      {
        M2nApiUnlock apiUnlock;
        NCCL_M2N_CHECK(ncclCommWindowRegister(sc->commA, stagingBuffer, stagingCapacity, &winA,
                                               NCCL_WIN_COLL_SYMMETRIC | NCCL_WIN_STRICT_ORDERING));
        NCCL_M2N_CHECK(m2nWaitCommReady(sc->commA));
      }
      NCCL_M2N_CHECK(cacheInternalWindow(sc->commA, stagingBuffer, stagingCapacity, RESHARD_INTERNAL_WINDOW_SPLIT,
                                       winA));
    }
    if (outWindowA != nullptr) *outWindowA = winA;

    const bool needDevCommA = outDevCommA != nullptr || outDevCommAUse != nullptr;
    NCCL_M2N_CHECK_ARG(!needDevCommA || (outDevCommA != nullptr && outDevCommAUse != nullptr), sc->parentRank,
                       "reshardSplitEnsureResources: commA DevComm and use outputs must be supplied together");
    if (needDevCommA) {
      NCCL_M2N_CHECK(reshardGetOrCreateDevCommWithRequirements(
        sc->commA, barrierCount, ginSignalCountA, ginCounterCountA, barrierKind, reshardGetGinContextCount(),
        NCCL_GIN_CONNECTION_FULL, stream, outDevCommA, outDevCommAUse,
        reshardGetCopyAlgorithm() == RESHARD_COPY_ALGO_PIPE));
    }
  }

  /* commB (RAIL): all generator ranks. PIPE fixes one kernel resource plan
   * per owner: namespaces isolate persistent FIFO histories, while lanes
   * bound execution contexts. PACK retains its requirement-variant behavior. */
  if (sc->inB && sc->commB != nullptr) {
    ncclWindow_t* cw =
      findCachedInternalWindowByPtr(sc->commB, stagingBuffer, stagingCapacity, RESHARD_INTERNAL_WINDOW_SPLIT);
    ncclWindow_t winB = nullptr;
    if (cw != nullptr) {
      winB = *cw;
    } else {
      {
        M2nApiUnlock apiUnlock;
        NCCL_M2N_CHECK(ncclCommWindowRegister(sc->commB, stagingBuffer, stagingCapacity, &winB,
                                               NCCL_WIN_COLL_SYMMETRIC | NCCL_WIN_STRICT_ORDERING));
        NCCL_M2N_CHECK(m2nWaitCommReady(sc->commB));
      }
      NCCL_M2N_CHECK(cacheInternalWindow(sc->commB, stagingBuffer, stagingCapacity, RESHARD_INTERNAL_WINDOW_SPLIT,
                                       winB));
    }
    if (outWindowB != nullptr) *outWindowB = winB;

    const bool needDevCommB = outDevCommB != nullptr || outDevCommBUse != nullptr;
    NCCL_M2N_CHECK_ARG(!needDevCommB || (outDevCommB != nullptr && outDevCommBUse != nullptr), sc->parentRank,
                       "reshardSplitEnsureResources: commB DevComm and use outputs must be supplied together");
    if (needDevCommB) {
      if (reshardGetCopyAlgorithm() == RESHARD_COPY_ALGO_PIPE) {
        NCCL_M2N_CHECK_ARG(sc->commBPoolOwnerIndex >= 0 && sc->commBPoolOwnerIndex < gCommBSharedCount,
                           sc->parentRank, "split PIPE resource plan requires a valid commB owner");
        CommBSharedEntry& owner = gCommBShared[sc->commBPoolOwnerIndex];
        NCCL_M2N_CHECK_ARG(!owner.pipeResourcePlanValid || owner.pipeResourcePlan == resourcesB,
                           sc->parentRank, "split PIPE commB resource plan cannot change while FIFO histories are live");
        owner.pipeResourcePlan = resourcesB;
        owner.pipeResourcePlanValid = true;
      }
      NCCL_M2N_CHECK(reshardGetOrCreateDevCommWithRequirements(
        sc->commB, resourcesB.barriers, resourcesB.signals, resourcesB.counters, barrierKind, resourcesB.contexts,
        NCCLM2N_GIN_RAIL_CONNECTION,
        stream, outDevCommB, outDevCommBUse, reshardGetCopyAlgorithm() == RESHARD_COPY_ALGO_PIPE));
    }
  }

  return ncclSuccess;
}

ncclResult_t reshardSplitPrepareCommBResourceLane(const ReshardSplitComms* sc, cudaStream_t stream) {
  NCCL_M2N_CHECK_ARG(sc != nullptr, -1, "commB resource-lane prepare requires split state");
  if (reshardGetCopyAlgorithm() != RESHARD_COPY_ALGO_PIPE || !sc->active || !sc->inB) return ncclSuccess;
  NCCL_M2N_CHECK_ARG(sc->slotIdx >= 0 && sc->slotIdx < sc->slotCount, sc->parentRank,
                     "commB resource-lane prepare received invalid lane %d/%d", sc->slotIdx, sc->slotCount);
  NCCL_M2N_CHECK_ARG(sc->commBPoolOwnerIndex >= 0 && sc->commBPoolOwnerIndex < gCommBSharedCount, sc->parentRank,
                     "commB resource-lane prepare has invalid pool owner %d", sc->commBPoolOwnerIndex);
  CommBSharedEntry& owner = gCommBShared[sc->commBPoolOwnerIndex];
  NCCL_M2N_CHECK_ARG(owner.used && owner.ownsCommB && owner.commB == sc->commB, sc->parentRank,
                     "commB resource-lane prepare found an invalid pool owner");
  NCCL_M2N_CHECK_ARG(!owner.lanePoisoned[sc->slotIdx], sc->parentRank,
                     "commB resource lane %d is unavailable after its completion stream could not be synchronized",
                     sc->slotIdx);
  cudaEvent_t& event = owner.laneCompletion[sc->slotIdx];
  if (event == nullptr) {
    NCCL_M2N_CUDACHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
  }
  if (owner.laneCompletionRecorded[sc->slotIdx]) {
    NCCL_M2N_CUDACHECK(cudaStreamWaitEvent(stream, event, 0));
  }
  return ncclSuccess;
}

ncclResult_t reshardSplitRecordCommBResourceLane(const ReshardSplitComms* sc, cudaStream_t stream) {
  NCCL_M2N_CHECK_ARG(sc != nullptr, -1, "commB resource-lane record requires split state");
  if (reshardGetCopyAlgorithm() != RESHARD_COPY_ALGO_PIPE || !sc->active || !sc->inB) return ncclSuccess;
  NCCL_M2N_CHECK_ARG(sc->slotIdx >= 0 && sc->slotIdx < sc->slotCount, sc->parentRank,
                     "commB resource-lane record received invalid lane %d/%d", sc->slotIdx, sc->slotCount);
  NCCL_M2N_CHECK_ARG(sc->commBPoolOwnerIndex >= 0 && sc->commBPoolOwnerIndex < gCommBSharedCount, sc->parentRank,
                     "commB resource-lane record has invalid pool owner %d", sc->commBPoolOwnerIndex);
  CommBSharedEntry& owner = gCommBShared[sc->commBPoolOwnerIndex];
  cudaEvent_t event = owner.laneCompletion[sc->slotIdx];
  NCCL_M2N_CHECK_ARG(event != nullptr, sc->parentRank, "commB resource lane %d has no completion event",
                     sc->slotIdx);
  const ncclResult_t result =
    reshardRecordCompletionEvent(event, stream, "commB resource lane", &owner.lanePoisoned[sc->slotIdx]);
  owner.laneCompletionRecorded[sc->slotIdx] = result == ncclSuccess;
  return result;
}

#ifdef NCCL_M2N_TESTING
void reshardSplitSetLaneCompletionForTest(int lane, cudaEvent_t event) {
  CommBSharedEntry& e = gCommBShared[0];
  e.used = e.ownsCommB = true;
  if (e.resourceLaneCount <= lane) e.resourceLaneCount = lane + 1;
  e.laneCompletion[lane] = event;
  e.laneCompletionRecorded[lane] = true;
  gCommBSharedCount = 1;
}
#endif

void reshardSplitSynchronizeCommBLanes() {
  if (reshardResourcesNeedQuarantine()) return;
  for (int i = 0; i < gCommBSharedCount; i++) {
    CommBSharedEntry& e = gCommBShared[i];
    if (!e.used || !e.ownsCommB) continue;
    for (int lane = 0; lane < e.resourceLaneCount; lane++) {
      if (e.lanePoisoned[lane]) {
        reshardRequireResourceQuarantine();
        return;
      }
      if (!e.laneCompletionRecorded[lane]) continue;
      const cudaError_t result = cudaEventSynchronize(e.laneCompletion[lane]);
      if (result != cudaSuccess) {
        RESHARD_WARN(-1, "Retaining split resources after commB lane %d could not be synchronized: %s", lane,
                       cudaGetErrorString(result));
        reshardRequireResourceQuarantine();
        return;
      }
    }
  }
}

void reshardSplitCommFinalize() {
  freeSplitBroadcastScratch();

  if (reshardResourcesNeedQuarantine()) {
    RESHARD_WARN(-1, "Retaining split communicators because GPU work could not be fenced safely");
    for (int i = 0; i < gCommACount; i++) {
      gCommACache[i] = {};
    }
    gCommACount = 0;
    for (int i = 0; i < gCommBParentCount; i++) {
      gCommBParent[i] = {};
    }
    gCommBParentCount = 0;
    for (int i = 0; i < gCommBSharedCount; i++) {
      gCommBShared[i] = {};
    }
    gCommBSharedCount = 0;
    gCommBCreateCount = 0;
    gCommBReuseCount = 0;
    return;
  }

  /* commA entries first: destroy each probe DevComm before its commA. */
  for (int i = 0; i < gCommACount; i++) {
    CommACacheEntry& e = gCommACache[i];
    if (!e.used) {
      continue;
    }
    if (e.probeAValid && e.commA != nullptr) {
      NCCL_M2N_CHECK_WARN(ncclDevCommDestroy(e.commA, &e.probeDevCommA));
    }
    if (e.commA != nullptr) {
      NCCL_M2N_CHECK_WARN(ncclCommDestroy(e.commA));
    }
    e = {};
  }
  gCommACount = 0;

  /* Per-parent eligibility entries hold no CUDA handles. */
  for (int i = 0; i < gCommBParentCount; i++) {
    gCommBParent[i] = {};
  }
  gCommBParentCount = 0;

  /* commB probe DevComms are private to split formation. Cached kernel
   * DevComms were released by cacheFinalize() before this communicator. */
  for (int i = 0; i < gCommBSharedCount; i++) {
    CommBSharedEntry& e = gCommBShared[i];
    if (!e.used) {
      continue;
    }
    if (e.ownsCommB && e.probeValid && e.commB != nullptr) {
      NCCL_M2N_CHECK_WARN(ncclDevCommDestroy(e.commB, &e.probeDevComm));
    }
    if (e.ownsCommB) {
      for (int lane = 0; lane < e.resourceLaneCount; lane++) {
        if (e.laneCompletion[lane] != nullptr) {
          // Already drained before cacheFinalize() released kernel DevComms.
          NCCL_M2N_CUDACHECK_WARN(cudaEventDestroy(e.laneCompletion[lane]));
        }
      }
    }
    if (e.ownsCommB && e.commB != nullptr) {
      NCCL_M2N_CHECK_WARN(ncclCommDestroy(e.commB));
    }
    e = {};
  }
  gCommBSharedCount = 0;
  gCommBCreateCount = 0;
  gCommBReuseCount = 0;
}
