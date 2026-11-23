/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

#include <cstdio>
#include <cstring>
#include <memory>
#include <new>
#include <vector>
#include "nccl.h"

// Host-only TU: pull host-visible types/decls from nccl_device.h. Scope
// the macro to this include so it doesn't leak to subsequent headers.
#define NCCL_HOSTLIB_ONLY
#include "nccl_device.h"
#undef NCCL_HOSTLIB_ONLY

#include "reshard_types.h"
#include "m2n_log.h"
#include "m2n_checks.h"
#include "reshard_internal.h"
#include "staging_buffer.h"

struct DevCommCacheEntry {
  ReshardDevCommCacheKey key;
  bool valid;
  bool reserved;
  ncclDevComm devComm;
  /* Per-entry ordering/failure fence. Shared uses are additionally drained by
   * existing PACK-slot / PIPE-lane events before cache teardown. No DevComm is
   * evicted during the runtime epoch. */
  cudaEvent_t completionEvent;
  std::shared_ptr<ReshardDevCommUseState> useState;
};

/* One pool entry per (comm, dev) — a non-blocking stream paired with
 * caller-to-work readiness and work-to-caller completion events.  The events
 * are reused across calls so we avoid cudaEvent{Create,Destroy} on every
 * reshard. */
struct StreamPoolEntry {
  ncclComm_t comm;
  int dev;
  cudaStream_t stream = nullptr;
  cudaEvent_t readyEvent = nullptr;
  cudaEvent_t doneEvent = nullptr;
};

static WindowCache gInternalWindowCache = {};
static std::unique_ptr<DevCommCacheEntry[]> gDevcommCache;
static int gDevcommCacheCount = 0;
static int gDevcommCacheCapacity = 0;

/* CUDA handle lifetimes are tied to the last finalize call in the
 * process-lifetime init/finalize epoch (so we can destroy them before the CUDA
 * context tears down) — see cacheFinalize.
 * The vector itself just owns memory; handles inside it are released
 * explicitly there before the vector is cleared. */
static std::vector<StreamPoolEntry> gStreamPool;

#ifdef NCCL_M2N_TESTING
static bool gFailNextCompletionEventRecordForTest = false;
static bool gFailNextCompletionStreamSynchronizeForTest = false;
static bool gFailNextCacheEventSynchronizeForTest = false;
#endif

/* Without a valid fence, quarantine the whole runtime epoch because cached GPU
 * resources may still be in use. */
static bool gReshardResourceQuarantineRequired = false;

static cudaError_t recordEvent(cudaEvent_t event, cudaStream_t stream) {
#ifdef NCCL_M2N_TESTING
  if (gFailNextCompletionEventRecordForTest) {
    gFailNextCompletionEventRecordForTest = false;
    return cudaErrorUnknown;
  }
#endif
  return cudaEventRecord(event, stream);
}

static cudaError_t synchronizeStream(cudaStream_t stream) {
#ifdef NCCL_M2N_TESTING
  if (gFailNextCompletionStreamSynchronizeForTest) {
    gFailNextCompletionStreamSynchronizeForTest = false;
    return cudaErrorUnknown;
  }
#endif
  return cudaStreamSynchronize(stream);
}

static cudaError_t synchronizeEvent(cudaEvent_t event) {
#ifdef NCCL_M2N_TESTING
  if (gFailNextCacheEventSynchronizeForTest) {
    gFailNextCacheEventSynchronizeForTest = false;
    return cudaErrorUnknown;
  }
#endif
  return cudaEventSynchronize(event);
}

/* An armed injection that its test never reached would otherwise fire inside
 * whatever ran next. Teardown disarms them all. */
static void resetCompletionFailureInjection() {
#ifdef NCCL_M2N_TESTING
  gFailNextCompletionEventRecordForTest = false;
  gFailNextCompletionStreamSynchronizeForTest = false;
  gFailNextCacheEventSynchronizeForTest = false;
#endif
}

ncclResult_t reshardRecordCompletionEvent(cudaEvent_t event, cudaStream_t stream, const char* resource,
                                          bool* poisoned) {
  const cudaError_t recordResult = recordEvent(event, stream);
  if (recordResult == cudaSuccess) {
    return ncclSuccess;
  }

  // Work may already be enqueued. Fence it before allowing resource reuse.
  const cudaError_t syncResult = synchronizeStream(stream);
  if (syncResult != cudaSuccess) {
    gReshardResourceQuarantineRequired = true;
    if (poisoned != nullptr) {
      *poisoned = true;
    }
    NCCL_M2N_FAIL(ncclSystemError, -1, "Failed to record %s completion event (%s) and synchronize its stream (%s)",
                  resource, cudaGetErrorString(recordResult), cudaGetErrorString(syncResult));
  }
  NCCL_M2N_FAIL(ncclSystemError, -1,
                "Failed to record %s completion event: %s; synchronized its stream before reuse", resource,
                cudaGetErrorString(recordResult));
}

#ifdef NCCL_M2N_TESTING
void reshardFailNextCompletionEventRecordForTest(bool bFailStreamSynchronize) {
  gFailNextCompletionEventRecordForTest = true;
  gFailNextCompletionStreamSynchronizeForTest = bFailStreamSynchronize;
}

void reshardFailNextCacheEventSynchronizeForTest() {
  gFailNextCacheEventSynchronizeForTest = true;
}
#endif

bool reshardResourcesNeedQuarantine() {
  return gReshardResourceQuarantineRequired;
}

void reshardRequireResourceQuarantine() {
  gReshardResourceQuarantineRequired = true;
}

void reshardClearResourceQuarantine() {
  gReshardResourceQuarantineRequired = false;
}

static ncclWindow_t* findCachedWindowByPtr(WindowCache* cache, ncclComm_t comm, void* buffer, size_t size,
                                           ReshardInternalWindowKind kind) {
  for (int i = 0; i < cache->count; i++) {
    WindowCacheEntry& e = cache->entries[i];
    if (e.valid && e.comm == comm && e.windowBuffer == buffer && e.windowSize == size && e.kind == kind) {
      return &e.window;
    }
  }
  return nullptr;
}

/* Entries evicted while another public call was in flight. Their teardown is
 * collective, and eviction is data-dependent per rank, so it cannot simply run
 * with the lock released - a rank that is not evicting would never join. Defer
 * instead, and drain once this process has no other public call in flight. */
static std::vector<WindowCacheEntry> gRetiredWindowCacheEntries;

static ncclResult_t reclaimRetiredWindowEntriesIfIdle() {
  if (m2nApiHasConcurrentCalls() || gRetiredWindowCacheEntries.empty()) {
    return ncclSuccess;
  }
  std::vector<WindowCacheEntry> retired;
  retired.swap(gRetiredWindowCacheEntries);
  {
    M2nApiUnlock apiUnlock;
    for (WindowCacheEntry& entry : retired) {
      if (entry.valid) {
        NCCL_M2N_CHECK_WARN(ncclCommWindowDeregister(entry.comm, entry.window));
      }
    }
  }
  return ncclSuccess;
}

/* The API lock proves no other host call is running; it does not prove the GPU
 * is finished. Sync the entry's completion event before destroying it. */
static ncclResult_t synchronizeDevCommCacheEntry(DevCommCacheEntry* entry) {
  if (entry->useState != nullptr && entry->useState->bPoisoned) {
    RESHARD_WARN(-1, "Retaining poisoned DevComm because its last use could not be fenced safely");
    gReshardResourceQuarantineRequired = true;
    return ncclSystemError;
  }
  if (entry->completionEvent != nullptr) {
    const cudaError_t syncResult = synchronizeEvent(entry->completionEvent);
    if (syncResult != cudaSuccess) {
      gReshardResourceQuarantineRequired = true;
      if (entry->useState != nullptr) {
        entry->useState->bPoisoned = true;
      }
      RESHARD_WARN(-1, "Retaining DevComm after its completion event could not be synchronized: %s",
                   cudaGetErrorString(syncResult));
      return ncclSystemError;
    }
  }
  return ncclSuccess;
}

static ncclResult_t destroyDevCommCacheEntry(DevCommCacheEntry* entry) {
  ncclResult_t result = ncclSuccess;
  if (entry->completionEvent != nullptr) {
    const cudaError_t eventResult = cudaEventDestroy(entry->completionEvent);
    if (eventResult != cudaSuccess) {
      RESHARD_WARN(-1, "Failed to destroy DevComm completion event: %s", cudaGetErrorString(eventResult));
      result = ncclSystemError;
    }
    entry->completionEvent = nullptr;
  }
  entry->useState.reset();
  const ncclResult_t devCommResult = ncclDevCommDestroy(entry->key.comm, &entry->devComm);
  if (devCommResult != ncclSuccess) {
    RESHARD_WARN(-1, "Failed to destroy cached DevComm: %s", ncclGetErrorString(devCommResult));
    if (result == ncclSuccess) {
      result = devCommResult;
    }
  }
  entry->valid = false;
  return result;
}

/* On a failed fence the entry is dropped from the cache but its DevComm is
 * deliberately leaked: destroying it could free memory a running kernel is
 * still reading. The epoch is quarantined instead. */
static ncclResult_t releaseDevCommCacheEntry(DevCommCacheEntry* entry) {
  const ncclResult_t syncResult = synchronizeDevCommCacheEntry(entry);
  if (syncResult != ncclSuccess) {
    entry->valid = false;
    return syncResult;
  }
  return destroyDevCommCacheEntry(entry);
}

static ncclResult_t cacheWindow(WindowCache* cache, ncclComm_t comm, void* windowBuffer, size_t windowSize,
                                ReshardInternalWindowKind kind, ncclWindow_t window) {
  int idx;
  if (cache->count >= MAX_WINDOW_CACHE_ENTRIES) {
    idx = cache->nextIdx;
    WindowCacheEntry& old = cache->entries[idx];
    RESHARD_WARN(-1, "Window cache full (%d entries), replacing entry at index %d", MAX_WINDOW_CACHE_ENTRIES, idx);
    if (old.valid) {
      if (m2nApiHasConcurrentCalls()) {
        try {
          gRetiredWindowCacheEntries.push_back(old);
        } catch (const std::bad_alloc&) {
          NCCL_M2N_FAIL(ncclSystemError, -1, "failed to retain an in-use window cache entry");
        }
      } else {
        M2nApiUnlock apiUnlock;
        NCCL_M2N_CHECK_WARN(ncclCommWindowDeregister(old.comm, old.window));
      }
    }
    cache->nextIdx = (cache->nextIdx + 1) % MAX_WINDOW_CACHE_ENTRIES;
  } else {
    idx = cache->count++;
  }
  WindowCacheEntry& e = cache->entries[idx];
  e.comm = comm;
  e.windowBuffer = windowBuffer;
  e.windowSize = windowSize;
  e.kind = kind;
  e.window = window;
  e.valid = true;
  return ncclSuccess;
}

ncclWindow_t* findCachedInternalWindowByPtr(ncclComm_t comm, void* buffer, size_t size,
                                            ReshardInternalWindowKind kind) {
  return findCachedWindowByPtr(&gInternalWindowCache, comm, buffer, size, kind);
}

ncclResult_t cacheInternalWindow(ncclComm_t comm, void* buffer, size_t size, ReshardInternalWindowKind kind,
                                 ncclWindow_t window) {
  return cacheWindow(&gInternalWindowCache, comm, buffer, size, kind, window);
}

ncclDevComm* findCachedDevComm(const ReshardDevCommCacheKey& key, cudaEvent_t* outCompletionEvent,
                               std::shared_ptr<ReshardDevCommUseState>* outUseState) {
  if (outCompletionEvent != nullptr) {
    *outCompletionEvent = nullptr;
  }
  if (outUseState != nullptr) {
    outUseState->reset();
  }
  for (int i = 0; i < gDevcommCacheCount; i++) {
    DevCommCacheEntry& e = gDevcommCache[i];
    if (e.valid && e.key == key) {
      if (outCompletionEvent != nullptr) {
        *outCompletionEvent = e.completionEvent;
      }
      if (outUseState != nullptr) {
        *outUseState = e.useState;
      }
      return &e.devComm;
    }
  }
  return nullptr;
}

ReshardDevCommCacheReservation::~ReshardDevCommCacheReservation() {
  if (index >= 0) gDevcommCache[index].reserved = false;
}

ncclResult_t reserveDevCommCacheEntry(ReshardDevCommCacheReservation* reservation) {
  NCCL_M2N_CHECK_ARG(reservation != nullptr && reservation->index < 0, -1,
                     "DevComm cache reservation must be empty");
  if (gDevcommCache == nullptr) {
    NCCL_M2N_CHECK_ARG(gReshardDevCommCacheCapacity > 0, -1, "Invalid DevComm cache capacity");
    try {
      gDevcommCache.reset(new DevCommCacheEntry[gReshardDevCommCacheCapacity]());
    } catch (const std::bad_alloc&) {
      NCCL_M2N_FAIL(ncclSystemError, -1, "Failed to allocate DevComm cache with capacity %d",
                    gReshardDevCommCacheCapacity);
    }
    gDevcommCacheCapacity = gReshardDevCommCacheCapacity;
  }
  int idx = 0;
  while (idx < gDevcommCacheCount && (gDevcommCache[idx].valid || gDevcommCache[idx].reserved)) ++idx;
  if (idx == gDevcommCacheCapacity) {
    NCCL_M2N_FAIL(ncclInvalidUsage, -1,
                  "DevComm cache capacity exhausted (%d entries); runtime eviction is disabled. "
                  "Increase NCCL_RESHARD_DEVCOMM_CACHE_CAPACITY before initialization and restart. "
                  "Capacity errors are local; abort the distributed job on error",
                  gDevcommCacheCapacity);
  }
  if (idx == gDevcommCacheCount) ++gDevcommCacheCount;
  gDevcommCache[idx].reserved = true;
  reservation->index = idx;
  return ncclSuccess;
}

ncclResult_t cacheDevComm(const ReshardDevCommCacheKey& key, const ncclDevComm* devComm,
                           bool retainPersistentState, ReshardDevCommCacheReservation* reservation) {
  ReshardDevCommCacheReservation localReservation;
  if (reservation == nullptr) {
    NCCL_M2N_CHECK(reserveDevCommCacheEntry(&localReservation));
    reservation = &localReservation;
  }
  NCCL_M2N_CHECK_ARG(reservation->index >= 0 && reservation->index < gDevcommCacheCount &&
                         gDevcommCache[reservation->index].reserved, -1, "Invalid DevComm cache reservation");

  /* A failure leaves existing entries untouched and releases the reservation. */
  cudaEvent_t completionEvent = nullptr;
  std::shared_ptr<ReshardDevCommUseState> useState;
  const cudaError_t eventResult = cudaEventCreateWithFlags(&completionEvent, cudaEventDisableTiming);
  if (eventResult != cudaSuccess) {
    NCCL_M2N_FAIL(ncclSystemError, -1, "Failed to create DevComm completion event: %s",
                  cudaGetErrorString(eventResult));
  }
  try {
    useState = std::make_shared<ReshardDevCommUseState>();
  } catch (const std::bad_alloc&) {
    NCCL_M2N_CUDACHECK_WARN(cudaEventDestroy(completionEvent));
    NCCL_M2N_FAIL(ncclSystemError, -1, "Failed to create DevComm use state");
  }
  useState->bSerializeUses = !reshardUseInternalStreams();
  useState->bRetainPersistentState = retainPersistentState;

  DevCommCacheEntry& e = gDevcommCache[reservation->index];
  e.key = key;
  e.devComm = *devComm;
  e.completionEvent = completionEvent;
  e.useState = std::move(useState);
  e.valid = true;
  e.reserved = false;
  reservation->index = -1;
  return ncclSuccess;
}

/* ======================================================================
 * Staging buffer pool -- per-comm entries, mirroring TransposeBufferEntry.
 * ====================================================================*/

static StagingBufferPoolEntry gStagingPool[MAX_STAGING_BUFFER_ENTRIES];
static int gStagingPoolCount = 0;

#ifdef NCCL_M2N_TESTING
void stagingBufferPoolSetCompletionEventForTest(cudaEvent_t event) {
  gStagingPool[0] = {};
  gStagingPool[0].event = event;
  gStagingPool[0].allocated = true;
  gStagingPoolCount = 1;
}
#endif

static StagingBufferPoolEntry* findStagingPoolEntry(ncclComm_t comm, uint64_t poolKey) {
  for (int i = 0; i < gStagingPoolCount; i++) {
    if (gStagingPool[i].comm == comm && gStagingPool[i].poolKey == poolKey && gStagingPool[i].allocated) {
      return &gStagingPool[i];
    }
  }
  return nullptr;
}

ncclResult_t ensureStagingBufferPool(ncclComm_t comm, uint64_t poolKey, cudaStream_t stream, int activeChannels,
                                     int capacityChannels, int controlSlotCount,
                                     StagingBufferState** outState) {
  NCCL_M2N_CHECK_ARG(outState != nullptr, -1, "ensureStagingBufferPool called with null outState");
  *outState = nullptr;

  StagingBufferPoolEntry* entry = findStagingPoolEntry(comm, poolKey);
  if (entry != nullptr) {
    NCCL_M2N_CHECK_ARG(activeChannels <= entry->capacityChannels, -1,
                       "Staging buffer pool capacity %d is smaller than active channel count %d",
                       entry->capacityChannels, activeChannels);
    if (entry->reserved) {
      NCCL_M2N_FAIL(ncclInvalidUsage, -1,
                    "Concurrent reshard operations on communicator %p are unsupported; serialize calls at the caller",
                    (void*)comm);
    }
    NCCL_M2N_CHECK_ARG(!entry->poisoned, -1,
                       "Staging buffer pool entry is unusable after its completion stream could not be synchronized");
    NCCL_M2N_CHECK_ARG(entry->event != nullptr, -1, "Staging buffer pool entry has no ordering event");
    NCCL_M2N_CUDACHECK(cudaStreamWaitEvent(stream, entry->event, 0));
    entry->reserved = true;
    const ncclResult_t configureResult = stagingBufferConfigureActiveChannels(entry->state, activeChannels);
    if (configureResult != ncclSuccess) {
      entry->reserved = false;
      return configureResult;
    }
    *outState = entry->state;
    return ncclSuccess;
  }
  if (gStagingPoolCount >= MAX_STAGING_BUFFER_ENTRIES) {
    NCCL_M2N_FAIL(ncclSystemError, -1,
                  "Staging buffer pool full (%d entries); increase MAX_STAGING_BUFFER_ENTRIES",
                  MAX_STAGING_BUFFER_ENTRIES);
  }

  cudaEvent_t event = nullptr;
  NCCL_M2N_CUDACHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));

  std::unique_ptr<StagingBufferState> state(new (std::nothrow) StagingBufferState());
  if (state == nullptr) {
    NCCL_M2N_CUDACHECK_WARN(cudaEventDestroy(event));
    NCCL_M2N_FAIL(ncclSystemError, -1, "Failed to allocate host memory for staging buffer state");
  }

  ncclResult_t r = stagingBufferInitWithNumChannelsAndControlSlots(state.get(), capacityChannels, controlSlotCount);
  if (r != ncclSuccess) {
    stagingBufferFinalize(state.get()); // reclaim anything init allocated before failing
    NCCL_M2N_CUDACHECK_WARN(cudaEventDestroy(event));
    return r;
  }
  r = stagingBufferConfigureActiveChannels(state.get(), activeChannels);
  if (r != ncclSuccess) {
    stagingBufferFinalize(state.get());
    NCCL_M2N_CUDACHECK_WARN(cudaEventDestroy(event));
    return r;
  }

  StagingBufferPoolEntry& e = gStagingPool[gStagingPoolCount++];
  e.comm = comm;
  e.poolKey = poolKey;
  e.capacityChannels = capacityChannels;
  e.event = event;
  e.state = state.release();
  e.reserved = true;
  e.poisoned = false;
  e.allocated = true;
  RESHARD_DEBUG(-1,
                "Staging buffer pool: allocated for comm %p pool=0x%016llx activeChannels=%d capacityChannels=%d "
                "controlSlots=%d (%zu bytes) [slot %d]",
                (void*)comm, (unsigned long long)poolKey, activeChannels, e.capacityChannels, controlSlotCount,
                e.state->totalSize, gStagingPoolCount - 1);
  *outState = e.state;
  return ncclSuccess;
}

ncclResult_t stagingBufferPoolRecordEvent(ncclComm_t comm, uint64_t poolKey, cudaStream_t stream) {
  StagingBufferPoolEntry* e = findStagingPoolEntry(comm, poolKey);
  if (e != nullptr) {
    const ncclResult_t result = reshardRecordCompletionEvent(e->event, stream, "staging buffer", &e->poisoned);
    e->reserved = false;
    return result;
  }
  return ncclSuccess;
}

void stagingBufferPoolFinalize() {
  for (int i = 0; i < gStagingPoolCount; i++) {
    if (gStagingPool[i].event != nullptr) {
      // Sync before destroy/free: a finalize racing an in-flight staging kernel
      // would otherwise free the buffer under a running kernel (use-after-free).
      // Warn-only: teardown must not abort.
      NCCL_M2N_CUDACHECK_WARN(cudaEventSynchronize(gStagingPool[i].event));
      NCCL_M2N_CUDACHECK_WARN(cudaEventDestroy(gStagingPool[i].event));
    }
    if (gStagingPool[i].state != nullptr) {
      stagingBufferFinalize(gStagingPool[i].state);
      delete gStagingPool[i].state;
    }
    gStagingPool[i] = {};
  }
  gStagingPoolCount = 0;
}

void cacheFinalize() {
  if (reshardResourcesNeedQuarantine()) {
    /* Post-entry host-RMA failures can leave work referencing any of these
     * objects. Drop local ownership without destroying the underlying
     * resources; coordinated process shutdown is the only safe recovery. */
    RESHARD_WARN(-1, "Retaining cached windows, DevComms, streams, and staging buffers because GPU work could not "
                     "be fenced safely");
    gInternalWindowCache = {};
    gRetiredWindowCacheEntries.clear();
    for (int i = 0; i < gDevcommCacheCount; i++) {
      gDevcommCache[i] = {};
    }
    gDevcommCacheCount = 0;
    gDevcommCache.reset();
    gDevcommCacheCapacity = 0;
    gStreamPool.clear();
    for (int i = 0; i < gStagingPoolCount; i++) {
      gStagingPool[i] = {};
    }
    gStagingPoolCount = 0;
    stagingPipeControlSlotCacheReset();
    resetCompletionFailureInjection();
    return;
  }

  /* Drain anything deferred by an eviction that raced a concurrent call. */
  (void)reclaimRetiredWindowEntriesIfIdle();

  /* Only the window/DevComm teardown below is collective. The stream pool and
   * staging teardown that follow touch process-local state and must stay under
   * the lock, or a concurrent public call could observe them mid-destruction. */
  {
    M2nApiUnlock apiUnlock;
    for (int i = 0; i < gInternalWindowCache.count; i++) {
      WindowCacheEntry& e = gInternalWindowCache.entries[i];
      if (e.valid) {
        NCCL_M2N_CHECK_WARN(ncclCommWindowDeregister(e.comm, e.window));
        e.valid = false;
      }
    }
    gInternalWindowCache.count = 0;
    gInternalWindowCache.nextIdx = 0;

    for (int i = 0; i < gDevcommCacheCount; i++) {
      DevCommCacheEntry& e = gDevcommCache[i];
      if (e.valid) {
        NCCL_M2N_CHECK_WARN(releaseDevCommCacheEntry(&e));
      }
    }
    gDevcommCacheCount = 0;
    gDevcommCache.reset();
    gDevcommCacheCapacity = 0;
  }

  for (StreamPoolEntry& e : gStreamPool) {
    if (e.doneEvent != nullptr) {
      NCCL_M2N_CUDACHECK_WARN(cudaEventDestroy(e.doneEvent));
    }
    if (e.readyEvent != nullptr) {
      NCCL_M2N_CUDACHECK_WARN(cudaEventDestroy(e.readyEvent));
    }
    if (e.stream != nullptr) {
      NCCL_M2N_CUDACHECK_WARN(cudaStreamDestroy(e.stream));
    }
  }
  gStreamPool.clear();

  stagingBufferPoolFinalize();
  stagingPipeControlSlotCacheReset();
  resetCompletionFailureInjection();
}

ncclResult_t streamPoolAcquire(ncclComm_t comm, int dev, cudaStream_t* outStream, cudaEvent_t* outReadyEvent,
                               cudaEvent_t* outDoneEvent) {
  NCCL_M2N_CHECK_ARG(outStream != nullptr && outReadyEvent != nullptr && outDoneEvent != nullptr, -1,
                     "streamPoolAcquire: output stream and events must be non-null");
  NCCL_M2N_CHECK_ARG(reshardUseInternalStreams(), -1,
                     "streamPoolAcquire called while NCCL_RESHARD_USE_INTERNAL_STREAMS=0; caller must bypass the "
                     "pool");

  /* Find existing entry for (comm, dev). */
  for (StreamPoolEntry& e : gStreamPool) {
    if (e.comm == comm && e.dev == dev) {
      *outStream = e.stream;
      *outReadyEvent = e.readyEvent;
      *outDoneEvent = e.doneEvent;
      return ncclSuccess;
    }
  }
  /* Lazy-create stream + events for the new (comm, dev). */
  StreamPoolEntry fresh;
  fresh.comm = comm;
  fresh.dev = dev;
  if (cudaStreamCreateWithFlags(&fresh.stream, cudaStreamNonBlocking) != cudaSuccess ||
      cudaEventCreateWithFlags(&fresh.readyEvent, cudaEventDisableTiming) != cudaSuccess ||
      cudaEventCreateWithFlags(&fresh.doneEvent, cudaEventDisableTiming) != cudaSuccess) {
    if (fresh.doneEvent != nullptr) {
      NCCL_M2N_CUDACHECK_WARN(cudaEventDestroy(fresh.doneEvent));
    }
    if (fresh.readyEvent != nullptr) {
      NCCL_M2N_CUDACHECK_WARN(cudaEventDestroy(fresh.readyEvent));
    }
    if (fresh.stream != nullptr) {
      NCCL_M2N_CUDACHECK_WARN(cudaStreamDestroy(fresh.stream));
    }
    NCCL_M2N_FAIL(ncclSystemError, -1, "Failed to create CUDA stream-pool resources for device %d", dev);
  }
  try {
    gStreamPool.push_back(fresh);
  } catch (...) {
    NCCL_M2N_CUDACHECK_WARN(cudaEventDestroy(fresh.doneEvent));
    NCCL_M2N_CUDACHECK_WARN(cudaEventDestroy(fresh.readyEvent));
    NCCL_M2N_CUDACHECK_WARN(cudaStreamDestroy(fresh.stream));
    NCCL_M2N_FAIL(ncclSystemError, -1, "Failed to allocate stream-pool entry for device %d", dev);
  }
  *outStream = fresh.stream;
  *outReadyEvent = fresh.readyEvent;
  *outDoneEvent = fresh.doneEvent;
  return ncclSuccess;
}
