/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

#include "cuda_runtime.h"
#include <mutex>
#include "nccl.h"
#include "reshard_types.h"
#include "m2n_checks.h"
#include "m2n_log.h"
#include "reshard_internal.h"

/* ======================================================================
 * Bounded PACK staging pool.
 *
 * Buffers are fixed-size best-fit buckets. A communicator keeps one stable
 * slot per bucket so window registration remains rank-uniform across calls.
 * Distinct communicators reuse the physical slots in round-robin waves.
 * ====================================================================*/

/* The slot memo is intentionally stable for the lifetime of a communicator.
 * Every rank must register its symmetric staging window in lockstep. If a
 * communicator changed slots between reshard sequences, ranks taking
 * different sequences could enter ncclCommWindowRegister for different
 * windows and desynchronize the collective. The slot index itself need not
 * be the same on every rank. */
struct StagingSlot {
  void* buffer;
  size_t size;
  cudaEvent_t doneEvent;
  cudaStream_t lastStream;
  int bucketIdx;
  int physicalSlotIdx;
  bool eventRecorded;
  bool reserved;
  bool poisoned;
};

struct StagingBucketRT {
  size_t size;
  int numSlots;
  int nextSlotCursor;
  StagingSlot slots[MAX_SPLIT_CONCURRENCY];
  bool allocated;
};

struct CommBucketSlot {
  ncclComm_t comm;
  int bucketIdx;
  int slotIdx;
};

struct StagingDevicePool {
  int cudaDev;
  StagingBucketRT buckets[kMaxStagingBuckets];
  int bucketCount;
  bool built;
  CommBucketSlot commSlots[MAX_PACK_STAGING_ENTRIES * kMaxStagingBuckets];
  int commSlotCount;
};

static StagingDevicePool gStagingDevicePools[MAX_PACK_STAGING_ENTRIES];
static int gStagingDevicePoolCount = 0;
static int gPackStagingAllocationCount = 0;

struct PackRmaWarmupState {
  ncclComm_t comm;
  bool warmed;
};
static PackRmaWarmupState
  gPackRmaWarmupStates[MAX_PACK_STAGING_ENTRIES * kMaxStagingBuckets];
static int gPackRmaWarmupStateCount = 0;

static std::mutex gStagingPoolMutex;
static thread_local ncclComm_t gCurrentStagingComm = nullptr;
static thread_local StagingSlot* gCurrentStagingSlot = nullptr;

static StagingDevicePool* findStagingDevicePool(int cudaDev) {
  for (int i = 0; i < gStagingDevicePoolCount; i++) {
    if (gStagingDevicePools[i].cudaDev == cudaDev) return &gStagingDevicePools[i];
  }
  return nullptr;
}

static ncclResult_t acquireStagingDevicePool(StagingDevicePool** outPool) {
  int cudaDev = -1;
  NCCL_M2N_CUDACHECK(cudaGetDevice(&cudaDev));
  *outPool = findStagingDevicePool(cudaDev);
  if (*outPool != nullptr) return ncclSuccess;
  NCCL_M2N_CHECK_ARG(gStagingDevicePoolCount < MAX_PACK_STAGING_ENTRIES, -1,
                     "PACK staging device-pool table full (%d entries); increase "
                     "MAX_PACK_STAGING_ENTRIES",
                     MAX_PACK_STAGING_ENTRIES);
  *outPool = &gStagingDevicePools[gStagingDevicePoolCount++];
  (*outPool)->cudaDev = cudaDev;
  return ncclSuccess;
}

static int lookupCommBucketSlot(const StagingDevicePool& pool, ncclComm_t comm, int bucketIdx) {
  for (int i = 0; i < pool.commSlotCount; i++) {
    if (pool.commSlots[i].comm == comm && pool.commSlots[i].bucketIdx == bucketIdx) {
      return pool.commSlots[i].slotIdx;
    }
  }
  return -1;
}

static bool recordCommBucketSlot(StagingDevicePool* pool, ncclComm_t comm, int bucketIdx, int slotIdx) {
  if (pool->commSlotCount >= (int)(sizeof(pool->commSlots) / sizeof(pool->commSlots[0]))) return false;
  pool->commSlots[pool->commSlotCount++] = {comm, bucketIdx, slotIdx};
  return true;
}

static void buildStagingPoolMeta(StagingDevicePool* pool) {
  if (pool->built) return;
  pool->bucketCount = gReshardStagingBucketCount;
  int physicalSlotBase = 0;
  for (int i = 0; i < pool->bucketCount; i++) {
    StagingBucketRT& bucket = pool->buckets[i];
    bucket.size = gReshardStagingBuckets[i].size;
    bucket.numSlots = gReshardStagingBuckets[i].numSlots;
    bucket.nextSlotCursor = 0;
    bucket.allocated = false;
    for (int slot = 0; slot < bucket.numSlots; slot++) {
      bucket.slots[slot] = {};
      bucket.slots[slot].size = bucket.size;
      bucket.slots[slot].bucketIdx = i;
      bucket.slots[slot].physicalSlotIdx = physicalSlotBase + slot;
    }
    physicalSlotBase += bucket.numSlots;
  }
  pool->built = true;
}

static ncclResult_t ensureStagingSlotAllocated(StagingBucketRT* bucket, int slotIdx) {
  StagingSlot* slot = &bucket->slots[slotIdx];
  if (slot->buffer != nullptr) return ncclSuccess;
  cudaEvent_t event = nullptr;
  NCCL_M2N_CUDACHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
  void* buffer = nullptr;
  const ncclResult_t result = ncclMemAlloc(&buffer, bucket->size);
  if (result != ncclSuccess) {
    NCCL_M2N_CUDACHECK_WARN(cudaEventDestroy(event));
    return result;
  }
  slot->doneEvent = event;
  slot->buffer = buffer;
  bucket->allocated = true;
  gPackStagingAllocationCount++;
  RESHARD_DEBUG(-1, "PACK staging slot allocated: bucket=%d slot=%d size=%zu B", slot->bucketIdx,
                slotIdx, bucket->size);
  return ncclSuccess;
}

/* A reservation is abandoned when it outlived the call that took it.
 *
 * A slot is reserved by ensurePackStagingBuffer and released by
 * packStagingRecordEvent. Any error between the two returns early and would
 * otherwise leave the slot reserved for good -- and a reserved slot is never
 * reacquired, so the communicator fails every later call with a spurious
 * concurrency error and stops entering collectives its peers still enter.
 *
 * Ownership is deliberately NOT thread-local: the public API lock serializes
 * reshard calls but does not pin them to a thread, so a retry after a failure
 * can arrive on a different thread than the one that abandoned the slot. What
 * does hold is that a reservation with no M2N call in flight besides this one
 * cannot belong to a live call -- so treat it as abandoned.
 *
 * Poison it rather than reusing it: the completion event was never recorded, so
 * neither the buffer contents nor any in-flight device work are known. */
static bool stagingReservationIsAbandoned() {
  return !m2nApiHasConcurrentCalls();
}

static void reclaimAbandonedSlot(StagingSlot* slot) {
  slot->reserved = false;
  slot->eventRecorded = false;
  slot->poisoned = true;
}

static ncclResult_t acquireStagingSlot(ncclComm_t comm, size_t requiredBytes, cudaStream_t stream,
                                       StagingSlot** outSlot) {
  std::lock_guard<std::mutex> poolLock(gStagingPoolMutex);
  StagingDevicePool* pool = nullptr;
  NCCL_M2N_CHECK(acquireStagingDevicePool(&pool));
  buildStagingPoolMeta(pool);
  StagingBucketRT* bucket = nullptr;
  int bucketIdx = -1;
  for (int i = 0; i < pool->bucketCount; i++) {
    if (pool->buckets[i].size >= requiredBytes) {
      bucket = &pool->buckets[i];
      bucketIdx = i;
      break;
    }
  }
  if (bucket == nullptr) {
    const size_t largest = pool->bucketCount > 0 ? pool->buckets[pool->bucketCount - 1].size : 0;
    if (reshardStagingBucketsUseImplicitDefault()) {
      NCCL_M2N_FAIL(ncclInvalidArgument, -1,
                    "PACK staging request %zu B exceeds the %zu-B bucket in the implicit default "
                    "NCCL_RESHARD_PACK_BUFFSIZES=%zu:%d; configure a bucket of at least %zu B (for example "
                    "NCCL_RESHARD_PACK_BUFFSIZES=%zu:%d)",
                    requiredBytes, largest, largest, kDefaultStagingBucketSlots, requiredBytes, requiredBytes,
                    kDefaultStagingBucketSlots);
    }
    NCCL_M2N_FAIL(ncclInvalidArgument, -1,
                  "PACK staging request %zu B exceeds the largest configured buffer %zu B; add or enlarge "
                  "an NCCL_RESHARD_PACK_BUFFSIZES bucket to at least %zu B",
                  requiredBytes, largest, requiredBytes);
  }

  int slotIdx = lookupCommBucketSlot(*pool, comm, bucketIdx);
  const bool newMapping = slotIdx < 0;
  if (newMapping) {
    for (int i = 0; i < bucket->numSlots; i++) {
      const int candidate = (bucket->nextSlotCursor + i) % bucket->numSlots;
      if (!bucket->slots[candidate].poisoned) {
        slotIdx = candidate;
        break;
      }
    }
    NCCL_M2N_CHECK_ARG(slotIdx >= 0, -1, "staging bucket %d has no reusable slots (%d configured)", bucketIdx,
                       bucket->numSlots);
  }
  StagingSlot* slot = &bucket->slots[slotIdx];
  /* Reclaim before the poison check, not after: reclaiming marks the slot
   * poisoned, and a check that already ran would let this very call reuse a
   * slot it just declared unusable. */
  if (slot->reserved && stagingReservationIsAbandoned()) {
    reclaimAbandonedSlot(slot);
  }
  if (slot->poisoned) {
    NCCL_M2N_FAIL(ncclUnhandledCudaError, -1, "Staging bucket %d slot %d is unavailable", bucketIdx, slotIdx);
  }
  if (slot->reserved) {
    NCCL_M2N_FAIL(ncclInvalidUsage, -1,
                  "Concurrent reshard operations sharing staging slot %d:%d are unsupported; serialize calls at "
                  "the caller (communicator %p)",
                  bucketIdx, slotIdx, (void*)comm);
  }
  NCCL_M2N_CHECK(ensureStagingSlotAllocated(bucket, slotIdx));
  if (newMapping) {
    NCCL_M2N_CHECK_ARG(recordCommBucketSlot(pool, comm, bucketIdx, slotIdx), -1,
                       "PACK staging communicator memo is full (%zu entries); increase "
                       "MAX_PACK_STAGING_ENTRIES",
                       sizeof(pool->commSlots) / sizeof(pool->commSlots[0]));
    bucket->nextSlotCursor = (slotIdx + 1) % bucket->numSlots;
  }
  if (slot->eventRecorded && slot->lastStream != stream) {
    NCCL_M2N_CUDACHECK(cudaStreamWaitEvent(stream, slot->doneEvent, 0));
  }
  slot->reserved = true;
  gCurrentStagingComm = comm;
  gCurrentStagingSlot = slot;
  *outSlot = slot;
  return ncclSuccess;
}

ncclResult_t ensurePackStagingBuffer(ncclComm_t comm, size_t requiredBytes, cudaStream_t stream) {
  StagingSlot* slot = nullptr;
  return acquireStagingSlot(comm, requiredBytes, stream, &slot);
}

void* getPackStagingBuffer(ncclComm_t comm) {
  return (comm == gCurrentStagingComm && gCurrentStagingSlot != nullptr) ? gCurrentStagingSlot->buffer : nullptr;
}

size_t getPackStagingCapacity(ncclComm_t comm) {
  return (comm == gCurrentStagingComm && gCurrentStagingSlot != nullptr) ? gCurrentStagingSlot->size : 0;
}

static PackRmaWarmupState* findPackRmaWarmupState(ncclComm_t comm) {
  for (int i = 0; i < gPackRmaWarmupStateCount; i++) {
    if (gPackRmaWarmupStates[i].comm == comm) {
      return &gPackRmaWarmupStates[i];
    }
  }
  return nullptr;
}

static ncclResult_t getOrCreatePackRmaWarmupState(ncclComm_t comm, PackRmaWarmupState** outState) {
  PackRmaWarmupState* state = findPackRmaWarmupState(comm);
  if (state == nullptr) {
    const int capacity = (int)(sizeof(gPackRmaWarmupStates) / sizeof(gPackRmaWarmupStates[0]));
    NCCL_M2N_CHECK_ARG(gPackRmaWarmupStateCount < capacity, -1,
                       "PACK host-RMA warmup-state table is full (%d entries)", capacity);
    state = &gPackRmaWarmupStates[gPackRmaWarmupStateCount++];
    *state = {};
    state->comm = comm;
  }
  *outState = state;
  return ncclSuccess;
}

ncclResult_t getPackRmaWarmed(ncclComm_t comm, bool* warmed) {
  NCCL_M2N_CHECK_ARG(warmed != nullptr, -1, "PACK host-RMA warmed-state output must be non-null");
  std::lock_guard<std::mutex> poolLock(gStagingPoolMutex);
  PackRmaWarmupState* state = nullptr;
  NCCL_M2N_CHECK(getOrCreatePackRmaWarmupState(comm, &state));
  *warmed = state->warmed;
  return ncclSuccess;
}

ncclResult_t setPackRmaWarmed(ncclComm_t comm, bool warmed) {
  std::lock_guard<std::mutex> poolLock(gStagingPoolMutex);
  PackRmaWarmupState* state = nullptr;
  NCCL_M2N_CHECK(getOrCreatePackRmaWarmupState(comm, &state));
  state->warmed = warmed;
  return ncclSuccess;
}

int getStagingBucketIndex(ncclComm_t comm) {
  std::lock_guard<std::mutex> poolLock(gStagingPoolMutex);
  StagingSlot* slot = (comm == gCurrentStagingComm) ? gCurrentStagingSlot : nullptr;
  return (slot != nullptr) ? slot->bucketIdx : -1;
}

int getStagingPhysicalSlotIndex(ncclComm_t comm) {
  std::lock_guard<std::mutex> poolLock(gStagingPoolMutex);
  StagingSlot* slot = (comm == gCurrentStagingComm) ? gCurrentStagingSlot : nullptr;
  return (slot != nullptr) ? slot->physicalSlotIdx : -1;
}

ncclResult_t packStagingRecordEvent(ncclComm_t comm, cudaStream_t stream) {
  std::lock_guard<std::mutex> poolLock(gStagingPoolMutex);
  if (comm == gCurrentStagingComm && gCurrentStagingSlot != nullptr) {
    StagingSlot* slot = gCurrentStagingSlot;
    /* Release the reservation even when recording fails; the poisoned slot is
     * skipped while healthy slots remain available. */
    const ncclResult_t result =
      reshardRecordCompletionEvent(slot->doneEvent, stream, "PACK staging slot", &slot->poisoned);
    slot->lastStream = stream;
    slot->eventRecorded = (result == ncclSuccess);
    slot->reserved = false;
    gCurrentStagingComm = nullptr;
    gCurrentStagingSlot = nullptr;
    NCCL_M2N_CHECK(result);
  }
  return ncclSuccess;
}

void packStagingSynchronize() {
  std::lock_guard<std::mutex> poolLock(gStagingPoolMutex);
  if (reshardResourcesNeedQuarantine()) return;
  int currentDevice = -1;
  NCCL_M2N_CUDACHECK_WARN(cudaGetDevice(&currentDevice));
  for (int device = 0; device < gStagingDevicePoolCount; device++) {
    StagingDevicePool& pool = gStagingDevicePools[device];
    NCCL_M2N_CUDACHECK_WARN(cudaSetDevice(pool.cudaDev));
    for (int bucketIdx = 0; bucketIdx < pool.bucketCount; bucketIdx++) {
      StagingBucketRT& bucket = pool.buckets[bucketIdx];
      if (!bucket.allocated) continue;
      for (int slotIdx = 0; slotIdx < bucket.numSlots; slotIdx++) {
        StagingSlot& slot = bucket.slots[slotIdx];
        if (slot.eventRecorded) {
          const cudaError_t result = cudaEventSynchronize(slot.doneEvent);
          if (result != cudaSuccess) {
            RESHARD_WARN(-1, "Retaining PACK resources after staging slot could not be synchronized: %s",
                           cudaGetErrorString(result));
            reshardRequireResourceQuarantine();
          }
        }
      }
    }
  }
  if (currentDevice >= 0) NCCL_M2N_CUDACHECK_WARN(cudaSetDevice(currentDevice));
}

void packStagingFinalize() {
  std::lock_guard<std::mutex> poolLock(gStagingPoolMutex);
  gPackStagingAllocationCount = 0;
  if (reshardResourcesNeedQuarantine()) {
    RESHARD_WARN(-1, "Retaining PACK staging buffers because GPU work could not be fenced safely");
    for (int device = 0; device < gStagingDevicePoolCount; device++) {
      gStagingDevicePools[device] = {};
    }
    gStagingDevicePoolCount = 0;
    for (int i = 0; i < gPackRmaWarmupStateCount; i++) {
      gPackRmaWarmupStates[i] = {};
    }
    gPackRmaWarmupStateCount = 0;
    gCurrentStagingComm = nullptr;
    gCurrentStagingSlot = nullptr;
    return;
  }
  int currentDevice = -1;
  NCCL_M2N_CUDACHECK_WARN(cudaGetDevice(&currentDevice));
  for (int device = 0; device < gStagingDevicePoolCount; device++) {
    StagingDevicePool& pool = gStagingDevicePools[device];
    bool hasAllocation = false;
    for (int bucketIdx = 0; bucketIdx < pool.bucketCount && !hasAllocation; bucketIdx++) {
      hasAllocation = pool.buckets[bucketIdx].allocated;
    }
    if (!pool.built && !hasAllocation) continue;
    if (hasAllocation) NCCL_M2N_CUDACHECK_WARN(cudaSetDevice(pool.cudaDev));
    for (int bucketIdx = 0; bucketIdx < pool.bucketCount; bucketIdx++) {
      StagingBucketRT& bucket = pool.buckets[bucketIdx];
      for (int slotIdx = 0; slotIdx < bucket.numSlots; slotIdx++) {
        if (bucket.slots[slotIdx].doneEvent != nullptr) cudaEventDestroy(bucket.slots[slotIdx].doneEvent);
        if (bucket.slots[slotIdx].buffer != nullptr) {
          ncclMemFree(bucket.slots[slotIdx].buffer);
        }
      }
      bucket = {};
    }
    pool = {};
  }
  if (currentDevice >= 0) NCCL_M2N_CUDACHECK_WARN(cudaSetDevice(currentDevice));
  gStagingDevicePoolCount = 0;
  for (int i = 0; i < gPackRmaWarmupStateCount; i++) {
    gPackRmaWarmupStates[i] = {};
  }
  gPackRmaWarmupStateCount = 0;
  gCurrentStagingComm = nullptr;
  gCurrentStagingSlot = nullptr;
}

#ifdef NCCL_M2N_TESTING
int packStagingAllocationCountForTest() {
  std::lock_guard<std::mutex> poolLock(gStagingPoolMutex);
  return gPackStagingAllocationCount;
}
#endif
