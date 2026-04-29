/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "argcheck.h"
#include "bootstrap.h"
#include "comm.h"
#include "debug.h"
#include "mem_manager.h"
#include "nccl.h"
#include "param.h"

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <new>

NCCL_PARAM(MemManagerDisable, "DISABLE_MEM_MANAGER", 0);

namespace {

hipMemAllocationProp defaultProp(int dev) {
  hipMemAllocationProp prop{};
  prop.type                = hipMemAllocationTypePinned;
  prop.location.type       = hipMemLocationTypeDevice;
  prop.location.id         = dev;
  prop.requestedHandleType = hipMemHandleTypePosixFileDescriptor;
  return prop;
}

hipError_t setLocalAccess(void* va, size_t size, int dev) {
  hipMemAccessDesc access{};
  access.location.type = hipMemLocationTypeDevice;
  access.location.id   = dev;
  access.flags         = hipMemAccessFlagsProtReadWrite;
  return hipMemSetAccess(va, size, &access, 1);
}

ncclResult_t quiesceBarrier(struct ncclComm* comm, int tag) {
  if (comm->nRanks <= 1 || comm->bootstrap == nullptr) return ncclSuccess;
  return bootstrapBarrier(comm->bootstrap, comm->rank, comm->nRanks, tag);
}

ncclResult_t ensureCanary(struct ncclComm* comm) {
  ncclMemManager* mgr = comm->memManager;
  struct ncclSuspendCanary* c = &mgr->canary;
  if (c->valid) return ncclSuccess;

  int vmmSupported = 0;
  hipError_t e = hipDeviceGetAttribute(
      &vmmSupported, hipDeviceAttributeVirtualMemoryManagementSupported,
      comm->cudaDev);
  if (e != hipSuccess || !vmmSupported) {
    WARN("MemManager: HIP virtual memory management not supported on "
         "device %d (hipDeviceGetAttribute=%d, vmm=%d)",
         comm->cudaDev, (int)e, vmmSupported);
    return ncclInvalidUsage;
  }

  hipMemAllocationProp prop = defaultProp(comm->cudaDev);
  size_t granularity = 0;
  e = hipMemGetAllocationGranularity(&granularity, &prop,
                                     hipMemAllocationGranularityMinimum);
  if (e != hipSuccess || granularity == 0) {
    WARN("MemManager: hipMemGetAllocationGranularity failed (%d)", (int)e);
    return ncclSystemError;
  }
  size_t size = granularity;

  hipMemGenericAllocationHandle_t handle = nullptr;
  void* va = nullptr;

  e = hipMemCreate(&handle, size, &prop, 0);
  if (e != hipSuccess) {
    WARN("MemManager: canary hipMemCreate(size=%zu) failed (%d)", size, (int)e);
    return ncclSystemError;
  }
  e = hipMemAddressReserve(&va, size, 0, nullptr, 0);
  if (e != hipSuccess) {
    (void)hipMemRelease(handle);
    WARN("MemManager: canary hipMemAddressReserve failed (%d)", (int)e);
    return ncclSystemError;
  }
  e = hipMemMap(va, size, 0, handle, 0);
  if (e != hipSuccess) {
    (void)hipMemAddressFree(va, size);
    (void)hipMemRelease(handle);
    WARN("MemManager: canary hipMemMap failed (%d)", (int)e);
    return ncclSystemError;
  }
  e = setLocalAccess(va, size, comm->cudaDev);
  if (e != hipSuccess) {
    (void)hipMemUnmap(va, size);
    (void)hipMemAddressFree(va, size);
    (void)hipMemRelease(handle);
    WARN("MemManager: canary hipMemSetAccess failed (%d)", (int)e);
    return ncclSystemError;
  }

  c->va      = va;
  c->size    = size;
  c->handle  = handle;
  c->cudaDev = comm->cudaDev;
  c->valid   = 1;
  INFO(NCCL_INIT,
       "MemManager: canary VA=%p size=%zu dev=%d created (rank=%d)",
       va, size, comm->cudaDev, comm->rank);
  return ncclSuccess;
}

ncclResult_t suspendCanary(struct ncclComm* comm) {
  struct ncclSuspendCanary* c = &comm->memManager->canary;
  if (!c->valid || c->handle == nullptr) {
    WARN("MemManager: canary not initialized");
    return ncclInternalError;
  }
  hipError_t e = hipMemUnmap(c->va, c->size);
  if (e != hipSuccess) {
    WARN("MemManager: canary hipMemUnmap failed (%d)", (int)e);
    return ncclUnhandledCudaError;
  }
  e = hipMemRelease(c->handle);
  if (e != hipSuccess) {
    WARN("MemManager: canary hipMemRelease failed (%d)", (int)e);
    return ncclUnhandledCudaError;
  }
  c->handle = nullptr;
  return ncclSuccess;
}

ncclResult_t resumeCanary(struct ncclComm* comm) {
  struct ncclSuspendCanary* c = &comm->memManager->canary;
  if (!c->valid) return ncclSuccess;          // never used; nothing to do
  if (c->handle != nullptr) return ncclSuccess; // already mapped

  hipMemAllocationProp prop = defaultProp(c->cudaDev);
  hipMemGenericAllocationHandle_t newHandle = nullptr;
  hipError_t e = hipMemCreate(&newHandle, c->size, &prop, 0);
  if (e != hipSuccess) {
    WARN("MemManager: canary hipMemCreate failed (%d)", (int)e);
    return ncclSystemError;
  }
  e = hipMemMap(c->va, c->size, 0, newHandle, 0);
  if (e != hipSuccess) {
    (void)hipMemRelease(newHandle);
    WARN("MemManager: canary hipMemMap failed (%d)", (int)e);
    return ncclSystemError;
  }
  e = setLocalAccess(c->va, c->size, c->cudaDev);
  if (e != hipSuccess) {
    (void)hipMemUnmap(c->va, c->size);
    (void)hipMemRelease(newHandle);
    WARN("MemManager: canary hipMemSetAccess failed (%d)", (int)e);
    return ncclSystemError;
  }
  c->handle = newHandle;
  return ncclSuccess;
}

bool snapshotEntryForSuspend(ncclDynMemEntry* entry) {
  hipPointerAttribute_t pa{};
  hipError_t pe = hipPointerGetAttributes(&pa, entry->ptr);
  if (pe != hipSuccess) {
    (void)hipGetLastError();
    return false;
  }

  hipMemGenericAllocationHandle_t handle = nullptr;
  hipError_t e = hipMemRetainAllocationHandle(&handle, entry->ptr);
  if (e != hipSuccess || handle == nullptr) {
    (void)hipGetLastError();
    return false;
  }

  hipMemAllocationProp prop{};
  hipError_t propErr = hipMemGetAllocationPropertiesFromHandle(&prop, handle);
  hipDeviceptr_t base = 0;
  size_t live = 0;
  hipError_t rangeErr = hipMemGetAddressRange(&base, &live, (hipDeviceptr_t)entry->ptr);

  if (propErr != hipSuccess || rangeErr != hipSuccess || live == 0) {
    (void)hipMemRelease(handle);
    return false;
  }

  entry->prop   = prop;
  entry->handle = handle;     // we're about to release it via unmap+release
  if (entry->size == 0) entry->size = live;
  return true;
}

}  // namespace

ncclResult_t ncclMemManagerInit(struct ncclComm* comm) {
  if (ncclParamMemManagerDisable()) return ncclSuccess;
  if (comm == nullptr) return ncclInvalidArgument;

  ncclMemManager* mgr = (ncclMemManager*)calloc(1, sizeof(*mgr));
  if (mgr == nullptr) return ncclSystemError;
  new (&mgr->lock) std::mutex();

  mgr->entries     = nullptr;
  mgr->numEntries  = 0;
  mgr->released    = 0;
  mgr->refCount    = 1;
  mgr->commCudaDev = comm->cudaDev;

  comm->memManager = mgr;

  __atomic_store_n(&mgr->initialized, 1, __ATOMIC_RELEASE);

  INFO(NCCL_INIT, "MemManager: initialized for device %d (rank=%d)",
       comm->cudaDev, comm->rank);
  return ncclSuccess;
}

ncclResult_t ncclMemManagerDestroy(struct ncclComm* comm) {
  if (ncclParamMemManagerDisable()) return ncclSuccess;
  if (comm == nullptr) return ncclInvalidArgument;
  if (comm->memManager == nullptr) return ncclSuccess;

  ncclMemManager* mgr = comm->memManager;

  if (!__atomic_load_n(&mgr->initialized, __ATOMIC_ACQUIRE)) {
    comm->memManager = nullptr;
    return ncclSuccess;
  }

  int refCount = __atomic_sub_fetch(&mgr->refCount, 1, __ATOMIC_ACQ_REL);
  if (refCount > 0) {
    INFO(NCCL_INIT, "MemManager: decremented refCount to %d", refCount);
    comm->memManager = nullptr;
    return ncclSuccess;
  }

  __atomic_store_n(&mgr->initialized, 0, __ATOMIC_RELEASE);

  ncclDynMemEntry* e = mgr->entries;
  while (e != nullptr) {
    ncclDynMemEntry* next = e->next;
    if (e->cpuBackup != nullptr) free(e->cpuBackup);
    if (!e->isImportedFromPeer && e->desc.local.exportedPeerRanks != nullptr) {
      free(e->desc.local.exportedPeerRanks);
    }
    free(e);
    e = next;
  }
  mgr->entries    = nullptr;
  mgr->numEntries = 0;

  mgr->lock.~mutex();
  free(mgr);
  comm->memManager = nullptr;

  INFO(NCCL_INIT, "MemManager: destroyed");
  return ncclSuccess;
}

namespace {

ncclResult_t memTrackInternal(
    struct ncclMemManager*           manager,
    void*                            ptr,
    size_t                           size,
    hipMemGenericAllocationHandle_t  handle,
    hipMemAllocationHandleType       handleType,
    ncclMemType_t                    memType,
    bool                             isImportedFromPeer,
    int                              ownerRank,
    int                              ownerDev,
    void*                            ownerPtr) {
  if (ncclParamMemManagerDisable()) return ncclSuccess;
  if (manager == nullptr || ptr == nullptr) return ncclInternalError;
  if (!__atomic_load_n(&manager->initialized, __ATOMIC_ACQUIRE)) {
    WARN("MemManager: cannot track ptr=%p, manager not initialized", ptr);
    return ncclInternalError;
  }

  if (memType == ncclMemPersist) {
    if (isImportedFromPeer) {
      __atomic_add_fetch(&manager->totalPersistImported, size, __ATOMIC_RELAXED);
    } else {
      __atomic_add_fetch(&manager->totalPersist, size, __ATOMIC_RELAXED);
    }
    return ncclSuccess;
  }

  ncclDynMemEntry* entry = (ncclDynMemEntry*)calloc(1, sizeof(*entry));
  if (entry == nullptr) {
    WARN("MemManager: failed to allocate entry");
    return ncclSystemError;
  }
  entry->ptr        = ptr;
  entry->size       = size;
  entry->handle     = handle;
  entry->handleType = handleType;
  entry->memType    = memType;
  entry->state      = ncclDynMemStateActive;
  entry->cudaDev    = manager->commCudaDev;
  entry->cpuBackup  = nullptr;
  entry->isImportedFromPeer = isImportedFromPeer;

  if (isImportedFromPeer) {
    entry->desc.imported.ownerRank = ownerRank;
    entry->desc.imported.ownerDev  = ownerDev;
    entry->desc.imported.ownerPtr  = ownerPtr;
  } else {
    entry->desc.local.shareableHandleValid  = false;
    entry->desc.local.numExportedPeers      = 0;
    entry->desc.local.exportedPeersCapacity = 0;
    entry->desc.local.exportedPeerRanks     = nullptr;
  }

  {
    std::lock_guard<std::mutex> lock(manager->lock);
    entry->next        = manager->entries;
    manager->entries   = entry;
    manager->numEntries++;
  }

  if (isImportedFromPeer) {
    if (memType == ncclMemScratch) {
      __atomic_add_fetch(&manager->totalScratchImported, size, __ATOMIC_RELAXED);
    } else if (memType == ncclMemOffload) {
      __atomic_add_fetch(&manager->totalOffloadImported, size, __ATOMIC_RELAXED);
    }
  } else {
    if (memType == ncclMemScratch) {
      __atomic_add_fetch(&manager->totalScratch, size, __ATOMIC_RELAXED);
    } else if (memType == ncclMemOffload) {
      __atomic_add_fetch(&manager->totalOffload, size, __ATOMIC_RELAXED);
    }
  }
  return ncclSuccess;
}

}  // namespace

ncclResult_t ncclMemTrack(
    struct ncclMemManager*           manager,
    void*                            ptr,
    size_t                           size,
    hipMemGenericAllocationHandle_t  handle,
    hipMemAllocationHandleType       handleType,
    ncclMemType_t                    memType) {
  return memTrackInternal(manager, ptr, size, handle, handleType, memType,
                          /*isImportedFromPeer=*/false,
                          /*ownerRank=*/-1,
                          /*ownerDev=*/-1,
                          /*ownerPtr=*/nullptr);
}

ncclResult_t ncclMemTrackImportFromPeer(
    struct ncclMemManager*           manager,
    void*                            ptr,
    size_t                           size,
    hipMemGenericAllocationHandle_t  handle,
    hipMemAllocationHandleType       handleType,
    ncclMemType_t                    memType,
    int                              ownerRank,
    int                              ownerDev,
    void*                            ownerPtr) {
  return memTrackInternal(manager, ptr, size, handle, handleType, memType,
                          /*isImportedFromPeer=*/true,
                          ownerRank, ownerDev, ownerPtr);
}

ncclResult_t ncclMemUntrack(struct ncclMemManager* manager, void* ptr,
                            size_t size) {
  if (ncclParamMemManagerDisable()) return ncclSuccess;
  if (manager == nullptr || ptr == nullptr) return ncclInternalError;
  if (!__atomic_load_n(&manager->initialized, __ATOMIC_ACQUIRE)) {
    return ncclInternalError;
  }

  size_t           entrySize          = 0;
  bool             isImportedFromPeer = false;
  ncclMemType_t    memType            = ncclMemScratch;

  {
    std::lock_guard<std::mutex> lock(manager->lock);

    ncclDynMemEntry* prev  = nullptr;
    ncclDynMemEntry* entry = manager->entries;

    while (entry != nullptr) {
      if (entry->ptr == ptr) {
        if (prev == nullptr) {
          manager->entries = entry->next;
        } else {
          prev->next = entry->next;
        }
        manager->numEntries--;

        if (entry->cpuBackup != nullptr) {
          manager->cpuBackupUsage -= entry->size;
          free(entry->cpuBackup);
        }

        if (!entry->isImportedFromPeer &&
            entry->desc.local.exportedPeerRanks != nullptr) {
          free(entry->desc.local.exportedPeerRanks);
        }

        entrySize          = entry->size;
        isImportedFromPeer = entry->isImportedFromPeer;
        memType            = entry->memType;
        free(entry);
        break;
      }
      prev  = entry;
      entry = entry->next;
    }
  }

  if (entrySize > 0) {
    if (isImportedFromPeer) {
      if (memType == ncclMemScratch)
        __atomic_sub_fetch(&manager->totalScratchImported, entrySize, __ATOMIC_RELAXED);
      else if (memType == ncclMemOffload)
        __atomic_sub_fetch(&manager->totalOffloadImported, entrySize, __ATOMIC_RELAXED);
    } else {
      if (memType == ncclMemScratch)
        __atomic_sub_fetch(&manager->totalScratch, entrySize, __ATOMIC_RELAXED);
      else if (memType == ncclMemOffload)
        __atomic_sub_fetch(&manager->totalOffload, entrySize, __ATOMIC_RELAXED);
    }
  } else {
    __atomic_sub_fetch(&manager->totalPersist, size, __ATOMIC_RELAXED);
  }
  return ncclSuccess;
}

ncclResult_t ncclDynMemMarkExportToPeer(struct ncclMemManager* manager,
                                        void* ptr, int peerRank) {
  if (ncclParamMemManagerDisable()) return ncclSuccess;
  if (manager == nullptr || ptr == nullptr) return ncclInternalError;
  if (!__atomic_load_n(&manager->initialized, __ATOMIC_ACQUIRE)) {
    return ncclInternalError;
  }

  std::lock_guard<std::mutex> lock(manager->lock);

  ncclDynMemEntry* entry = manager->entries;
  while (entry != nullptr && entry->ptr != ptr) entry = entry->next;
  if (entry == nullptr) {
    WARN("MemManager: cannot mark export for ptr=%p (not tracked)", ptr);
    return ncclInternalError;
  }
  if (entry->isImportedFromPeer) {
    WARN("MemManager: cannot export an imported entry (ptr=%p)", ptr);
    return ncclInternalError;
  }

  for (int i = 0; i < entry->desc.local.numExportedPeers; ++i) {
    if (entry->desc.local.exportedPeerRanks[i] == peerRank) return ncclSuccess;
  }

  if (entry->desc.local.numExportedPeers >=
      entry->desc.local.exportedPeersCapacity) {
    int newCap = entry->desc.local.exportedPeersCapacity == 0
                     ? NCCL_MEM_EXPORT_PEERS_INIT
                     : entry->desc.local.exportedPeersCapacity * 2;
    int* grown = (int*)realloc(entry->desc.local.exportedPeerRanks,
                               (size_t)newCap * sizeof(int));
    if (grown == nullptr) return ncclSystemError;
    entry->desc.local.exportedPeerRanks     = grown;
    entry->desc.local.exportedPeersCapacity = newCap;
  }

  entry->desc.local.exportedPeerRanks[entry->desc.local.numExportedPeers++] =
      peerRank;
  return ncclSuccess;
}

ncclResult_t ncclCommMemSuspend(struct ncclComm* comm) {
  if (ncclParamMemManagerDisable()) {
    WARN("MemManager: suspend failed, memory manager is disabled");
    return ncclInvalidUsage;
  }
  if (comm == nullptr) return ncclInvalidArgument;
  if (comm->memManager == nullptr) return ncclInvalidUsage;
  ncclMemManager* mgr = comm->memManager;

  if (mgr->released) {
    WARN("MemManager: already suspended");
    return ncclInvalidUsage;
  }

  NCCLCHECK(ensureCanary(comm));

  if (hipDeviceSynchronize() != hipSuccess) {
    WARN("MemManager: hipDeviceSynchronize failed before suspend");
    return ncclUnhandledCudaError;
  }

  NCCLCHECK(quiesceBarrier(comm, /*tag=*/0xBEEF));
  NCCLCHECK(suspendCanary(comm));
  size_t releasedScratch = 0;
  size_t releasedOffload = 0;
  int    releasedCount   = 0;
  int    skippedCount    = 0;

  ncclDynMemEntry* entry = mgr->entries;
  while (entry != nullptr) {
    if (entry->isImportedFromPeer) {
      entry = entry->next;
      continue;
    }
    if (entry->state == ncclDynMemStateReleased) {
      entry = entry->next;
      continue;
    }

    if (!snapshotEntryForSuspend(entry)) {
      ++skippedCount;
      entry = entry->next;
      continue;
    }

    if (entry->memType == ncclMemOffload) {
      entry->cpuBackup = malloc(entry->size);
      if (entry->cpuBackup == nullptr) {
        WARN("MemManager: malloc(%zu) for host backup failed", entry->size);
        (void)hipMemRelease(entry->handle);
        entry->handle = 0;
        return ncclSystemError;
      }
      hipError_t e = hipMemcpy(entry->cpuBackup, entry->ptr, entry->size,
                               hipMemcpyDeviceToHost);
      if (e != hipSuccess) {
        WARN("MemManager: hipMemcpy DtoH failed for %p size %zu (%d)",
             entry->ptr, entry->size, (int)e);
        free(entry->cpuBackup);
        entry->cpuBackup = nullptr;
        (void)hipMemRelease(entry->handle);
        entry->handle = 0;
        return ncclUnhandledCudaError;
      }
      mgr->cpuBackupUsage += entry->size;
      releasedOffload     += entry->size;
    } else {
      releasedScratch += entry->size;
    }

    // unmap -> release; skip hipMemAddressFree so VA reservation survives
    if (hipMemUnmap(entry->ptr, entry->size) != hipSuccess) {
      WARN("MemManager: hipMemUnmap failed for %p size %zu",
           entry->ptr, entry->size);
      if (entry->cpuBackup != nullptr) {
        free(entry->cpuBackup);
        entry->cpuBackup = nullptr;
        mgr->cpuBackupUsage -= entry->size;
      }
      (void)hipMemRelease(entry->handle);
      entry->handle = 0;
      return ncclUnhandledCudaError;
    }
    if (hipMemRelease(entry->handle) != hipSuccess) {
      WARN("MemManager: hipMemRelease failed for %p", entry->ptr);
      entry->handle = 0;
      return ncclUnhandledCudaError;
    }
    entry->handle = 0;
    entry->state  = ncclDynMemStateReleased;
    ++releasedCount;

    entry = entry->next;
  }

  mgr->released = 1;

  INFO(NCCL_INIT,
       "MemManager: rank %d suspended canary + %d entries (scratch=%zu, "
       "offload=%zu, cpuBackup=%zu, skipped=%d non-VMM)",
       comm->rank, releasedCount, releasedScratch, releasedOffload,
       mgr->cpuBackupUsage, skippedCount);
  return ncclSuccess;
}

ncclResult_t ncclCommMemResume(struct ncclComm* comm) {
  if (ncclParamMemManagerDisable()) {
    WARN("MemManager: resume failed, memory manager is disabled");
    return ncclInvalidUsage;
  }
  if (comm == nullptr) return ncclInvalidArgument;
  if (comm->memManager == nullptr) return ncclInvalidUsage;
  ncclMemManager* mgr = comm->memManager;

  if (!mgr->released) {
    WARN("MemManager: not in suspended state");
    return ncclInvalidUsage;
  }

  size_t restoredBytes = 0;
  int    restoredCount = 0;

  ncclDynMemEntry* entry = mgr->entries;
  while (entry != nullptr) {
    if (entry->isImportedFromPeer) {
      entry = entry->next;
      continue;
    }
    if (entry->state != ncclDynMemStateReleased) {
      entry = entry->next;
      continue;
    }

    hipMemGenericAllocationHandle_t newHandle = nullptr;
    hipError_t e = hipMemCreate(&newHandle, entry->size, &entry->prop, 0);
    if (e != hipSuccess) {
      WARN("MemManager: resume hipMemCreate failed for VA=%p size=%zu (%d)",
           entry->ptr, entry->size, (int)e);
      return ncclSystemError;
    }
    e = hipMemMap(entry->ptr, entry->size, 0, newHandle, 0);
    if (e != hipSuccess) {
      (void)hipMemRelease(newHandle);
      WARN("MemManager: resume hipMemMap failed for VA=%p (%d)", entry->ptr, (int)e);
      return ncclSystemError;
    }
    e = setLocalAccess(entry->ptr, entry->size, entry->cudaDev);
    if (e != hipSuccess) {
      (void)hipMemUnmap(entry->ptr, entry->size);
      (void)hipMemRelease(newHandle);
      WARN("MemManager: resume hipMemSetAccess failed for VA=%p (%d)",
           entry->ptr, (int)e);
      return ncclSystemError;
    }
    entry->handle = newHandle;

    if (entry->memType == ncclMemOffload && entry->cpuBackup != nullptr) {
      e = hipMemcpy(entry->ptr, entry->cpuBackup, entry->size,
                    hipMemcpyHostToDevice);
      if (e != hipSuccess) {
        WARN("MemManager: resume hipMemcpy HtoD failed for VA=%p (%d)",
             entry->ptr, (int)e);
        return ncclUnhandledCudaError;
      }
      free(entry->cpuBackup);
      entry->cpuBackup = nullptr;
      mgr->cpuBackupUsage -= entry->size;
    }

    entry->state = ncclDynMemStateActive;
    ++restoredCount;
    restoredBytes += entry->size;

    entry = entry->next;
  }

  NCCLCHECK(resumeCanary(comm));
  NCCLCHECK(quiesceBarrier(comm, /*tag=*/0xBEEF));

  // P2P re-import not yet implemented (requires ncclProxyClientGetFdBlocking)
  if (comm->nRanks > 1 && comm->bootstrap != nullptr) {
    static int warnedOnce = 0;
    if (__atomic_exchange_n(&warnedOnce, 1, __ATOMIC_RELAXED) == 0) {
      INFO(NCCL_INIT,
           "MemManager: peer P2P re-import not yet implemented; relying on "
           "the caller to drain inter-rank traffic before suspend");
    }
  }

  NCCLCHECK(quiesceBarrier(comm, /*tag=*/0xCAFE));

  mgr->released = 0;

  INFO(NCCL_INIT,
       "MemManager: rank %d resumed canary + %d entries (%zu bytes); "
       "comm is usable again",
       comm->rank, restoredCount, restoredBytes);
  return ncclSuccess;
}

ncclResult_t ncclCommSuspendCanaryFree(struct ncclComm* comm) {
  if (comm == nullptr) return ncclSuccess;
  if (comm->memManager == nullptr) return ncclSuccess;
  struct ncclSuspendCanary* c = &comm->memManager->canary;
  if (!c->valid) return ncclSuccess;
  if (c->handle != nullptr) {
    (void)hipMemUnmap(c->va, c->size);
    (void)hipMemRelease(c->handle);
    c->handle = nullptr;
  }
  if (c->va != nullptr) {
    (void)hipMemAddressFree(c->va, c->size);
    c->va = nullptr;
  }
  c->valid = 0;
  return ncclSuccess;
}

ncclResult_t ncclCommSuspendForceResumeForDestroy(struct ncclComm* comm) {
  if (comm == nullptr || comm->memManager == nullptr) return ncclSuccess;
  ncclMemManager* mgr = comm->memManager;
  if (!mgr->released) return ncclSuccess;

  int    count = 0;
  size_t bytes = 0;

  ncclDynMemEntry* entry = mgr->entries;
  while (entry != nullptr) {
    if (entry->isImportedFromPeer || entry->state != ncclDynMemStateReleased) {
      entry = entry->next;
      continue;
    }

    hipMemGenericAllocationHandle_t newHandle = nullptr;
    if (hipMemCreate(&newHandle, entry->size, &entry->prop, 0) != hipSuccess) {
      entry = entry->next;
      continue;
    }
    if (hipMemMap(entry->ptr, entry->size, 0, newHandle, 0) != hipSuccess) {
      (void)hipMemRelease(newHandle);
      entry = entry->next;
      continue;
    }
    (void)setLocalAccess(entry->ptr, entry->size, entry->cudaDev);

    entry->handle = newHandle;
    entry->state  = ncclDynMemStateActive;
    if (entry->cpuBackup != nullptr) {
      free(entry->cpuBackup);
      entry->cpuBackup = nullptr;
      mgr->cpuBackupUsage -= entry->size;
    }
    ++count;
    bytes += entry->size;
    entry = entry->next;
  }

  mgr->released = 0;

  INFO(NCCL_INIT,
       "MemManager: ncclCommDestroy force-resumed %d suspended entries "
       "(%zu bytes) so destructor walk can free them",
       count, bytes);
  return ncclSuccess;
}

extern "C" {

ncclResult_t ncclCommSuspend_impl(ncclComm_t comm, int flags) {
  NCCLCHECK(CommCheck(comm, "ncclCommSuspend", "comm"));
  NCCLCHECK(ncclCommEnsureReady(comm));

  if (flags == 0 || (flags & ~NCCL_SUSPEND_MEM) != 0) {
    WARN("ncclCommSuspend: invalid flags 0x%x (only NCCL_SUSPEND_MEM=0x%x supported)",
         flags, NCCL_SUSPEND_MEM);
    return ncclInvalidArgument;
  }

  if (ncclParamMemManagerDisable()) {
    WARN("MemManager: suspend not supported, memory manager is disabled");
    return ncclInvalidUsage;
  }

  if (flags & NCCL_SUSPEND_MEM) {
    if (comm->memManager != nullptr && comm->memManager->refCount > 1) {
      WARN("ncclCommSuspend: not supported on split_share comms (refCount=%d)",
           comm->memManager->refCount);
      return ncclInvalidUsage;
    }

    INFO(NCCL_INIT, "ncclCommSuspend: rank %d suspending memory", comm->rank);
    struct ncclMemManagerTask* task =
        (struct ncclMemManagerTask*)calloc(1, sizeof(*task));
    if (task == nullptr) return ncclSystemError;
    task->comm = comm;
    ncclIntruQueueEnqueue(&comm->suspendTaskQueue, task);

    ncclResult_t ret = ncclSuccess;
    while (!ncclIntruQueueEmpty(&comm->suspendTaskQueue)) {
      struct ncclMemManagerTask* t =
          ncclIntruQueueDequeue(&comm->suspendTaskQueue);
      ret = ncclCommMemSuspend(t->comm);
      free(t);
      if (ret != ncclSuccess) {
        while (!ncclIntruQueueEmpty(&comm->suspendTaskQueue)) {
          free(ncclIntruQueueDequeue(&comm->suspendTaskQueue));
        }
        if (!comm->config.blocking) (void)ncclCommSetAsyncError(comm, ret);
        return ret;
      }
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclCommResume_impl(ncclComm_t comm) {
  NCCLCHECK(CommCheck(comm, "ncclCommResume", "comm"));
  NCCLCHECK(ncclCommEnsureReady(comm));

  if (ncclParamMemManagerDisable()) {
    WARN("MemManager: resume not supported, memory manager is disabled");
    return ncclInvalidUsage;
  }
  if (comm->memManager != nullptr && comm->memManager->refCount > 1) {
    WARN("ncclCommResume: not supported on split_share comms (refCount=%d)",
         comm->memManager->refCount);
    return ncclInvalidUsage;
  }

  INFO(NCCL_INIT, "ncclCommResume: rank %d resuming all resources", comm->rank);
  struct ncclMemManagerTask* task =
      (struct ncclMemManagerTask*)calloc(1, sizeof(*task));
  if (task == nullptr) return ncclSystemError;
  task->comm = comm;
  ncclIntruQueueEnqueue(&comm->resumeTaskQueue, task);

  ncclResult_t ret = ncclSuccess;
  while (!ncclIntruQueueEmpty(&comm->resumeTaskQueue)) {
    struct ncclMemManagerTask* t =
        ncclIntruQueueDequeue(&comm->resumeTaskQueue);
    ret = ncclCommMemResume(t->comm);
    free(t);
    if (ret != ncclSuccess) {
      while (!ncclIntruQueueEmpty(&comm->resumeTaskQueue)) {
        free(ncclIntruQueueDequeue(&comm->resumeTaskQueue));
      }
      if (!comm->config.blocking) (void)ncclCommSetAsyncError(comm, ret);
      return ret;
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclCommMemStats_impl(ncclComm_t comm, ncclCommMemStat_t stat,
                                   uint64_t* value) {
  NCCLCHECK(CommCheck(comm, "ncclCommMemStats", "comm"));
  if (value == nullptr) return ncclInvalidArgument;
  NCCLCHECK(ncclCommEnsureReady(comm));

  if (ncclParamMemManagerDisable()) {
    WARN("MemManager: MemStats not supported, memory manager is disabled");
    return ncclInvalidUsage;
  }

  if (comm->memManager == nullptr) {
    *value = 0;
    return ncclSuccess;
  }

  ncclMemManager* mgr = comm->memManager;
  size_t totalPersist =
      __atomic_load_n(&mgr->totalPersist, __ATOMIC_RELAXED);
  size_t totalScratch =
      __atomic_load_n(&mgr->totalScratch, __ATOMIC_RELAXED);
  size_t totalOffload =
      __atomic_load_n(&mgr->totalOffload, __ATOMIC_RELAXED);

  switch (stat) {
    case ncclStatGpuMemTotal:
      *value = totalPersist + totalScratch + totalOffload;
      return ncclSuccess;
    case ncclStatGpuMemPersist:
      *value = totalPersist;
      return ncclSuccess;
    case ncclStatGpuMemSuspend:
      *value = totalScratch + totalOffload;
      return ncclSuccess;
    case ncclStatGpuMemSuspended:
      *value = mgr->released ? 1 : 0;
      return ncclSuccess;
    default:
      WARN("ncclCommMemStats: unknown stat %d", (int)stat);
      return ncclInvalidArgument;
  }
}

}  // extern "C"
