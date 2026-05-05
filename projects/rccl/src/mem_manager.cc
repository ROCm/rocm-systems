/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Modifications Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "comm.h"
#include "alloc.h"
#include "checks.h"
#include "argcheck.h"
#include "rocmwrap.h"
#include "debug.h"
#include "bootstrap.h"
#include "proxy.h"
#include "transport.h"
#include "nvtx.h"
#include "param.h"
#include "group.h"
#include <string.h>
#include <stdlib.h>
#include <mutex>

// Internal parameter to disable memory manager for testing
NCCL_PARAM(MemManagerDisable, "DISABLE_MEM_MANAGER", 0);

// Initialize memory manager
ncclResult_t ncclMemManagerInit(struct ncclComm* comm) {
  if (ncclParamMemManagerDisable()) return ncclSuccess;
  if (comm == nullptr) return ncclInvalidArgument;

  ncclMemManager* mgr;
  NCCLCHECK(ncclCalloc(&mgr, 1));
  // Explicitly construct std::mutex using placement new
  new (&mgr->lock) std::mutex();

  mgr->entries = nullptr;
  mgr->numEntries = 0;
  mgr->released = 0;
  mgr->refCount = 1;
  mgr->totalPersist = 0;
  mgr->totalPersistImported = 0;
  mgr->totalScratch = 0;
  mgr->totalScratchImported = 0;
  mgr->totalOffload = 0;
  mgr->totalOffloadImported = 0;
  mgr->cpuBackupUsage = 0;
  mgr->commCudaDev = comm->cudaDev;

  __atomic_store_n(&mgr->initialized, 1, __ATOMIC_RELEASE);

  comm->memManager = mgr;

  INFO(NCCL_ALLOC, "MemManager: Initialized for device %d", comm->cudaDev);
  return ncclSuccess;
}

// Destroy memory manager and free all resources
ncclResult_t ncclMemManagerDestroy(struct ncclComm* comm) {
  if (ncclParamMemManagerDisable()) return ncclSuccess;
  if (comm == nullptr) return ncclInvalidArgument;
  if (comm->memManager == nullptr) return ncclSuccess;

  ncclMemManager* mgr = comm->memManager;

  if (!__atomic_load_n(&mgr->initialized, __ATOMIC_ACQUIRE)) {
    comm->memManager = nullptr;
    return ncclSuccess;
  }

  // Decrement reference count
  int refCount = ncclAtomicRefCountDecrement(&mgr->refCount);

  if (refCount > 0) {
    // Other comms still using this manager
    INFO(NCCL_ALLOC, "MemManager: Decremented refCount to %d", refCount);
    comm->memManager = nullptr;  // Clear this comm's pointer
    return ncclSuccess;
  }

  // refCount == 0, at this point proxy threads should be joined
  INFO(NCCL_ALLOC, "MemManager: Destroying (refCount=0)");
  __atomic_store_n(&mgr->initialized, 0, __ATOMIC_RELEASE);

  ncclDynMemEntry* entry = mgr->entries;
  while (entry != nullptr) {
    ncclDynMemEntry* next = entry->next;

    // Free CPU backup if exists
    if (entry->cpuBackup != nullptr) {
      ncclCudaHostFree(entry->cpuBackup);
    }

    // Close shareable FD if valid (defensive cleanup for POSIX FD handle type)
    if (!entry->isImportedFromPeer &&
        entry->desc.local.shareableHandleValid &&
        entry->handleType == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR &&
        entry->desc.local.shareableHandle.fd >= 0) {
      close(entry->desc.local.shareableHandle.fd);
      entry->desc.local.shareableHandle.fd = -1;
      entry->desc.local.shareableHandleValid = false;
    }

    // Only local entries have exportedPeerRanks (imported entries use desc.imported union member)
    if (!entry->isImportedFromPeer && entry->desc.local.exportedPeerRanks != nullptr) {
      free(entry->desc.local.exportedPeerRanks);
    }

    // Free the entry itself
    free(entry);
    entry = next;
  }

  mgr->entries = nullptr;
  mgr->numEntries = 0;

  // Explicitly call destructor for std::mutex
  mgr->lock.~mutex();
  // Free the manager struct
  free(mgr);
  comm->memManager = nullptr;

  INFO(NCCL_ALLOC, "MemManager: Destroyed");
  return ncclSuccess;
}

// Internal helper to create and track memory entry
static ncclResult_t ncclMemTrackInternal(
  struct ncclMemManager* manager,
  void* ptr,
  size_t size,
  CUmemGenericAllocationHandle handle,
  CUmemAllocationHandleType handleType,
  ncclMemType_t memType,
  bool isImportedFromPeer,
  int ownerRank,
  int ownerDev,
  void* ownerPtr
) {
  if (ncclParamMemManagerDisable()) return ncclSuccess;
  if (manager == nullptr || ptr == nullptr) return ncclInternalError;
  if (!__atomic_load_n(&manager->initialized, __ATOMIC_ACQUIRE)) {
    WARN("MemManager: Cannot track allocation ptr=%p, manager not initialized", ptr);
    return ncclInternalError;
  }

  // Persistent memory: atomic update only
  if (memType == ncclMemPersist) {
    if (isImportedFromPeer) {
      (void)__atomic_add_fetch(&manager->totalPersistImported, size, __ATOMIC_RELAXED);
      TRACE(NCCL_ALLOC, "MemManager: Track Persistent Import ptr=%p size=%zu from rank=%d",
            ptr, size, ownerRank);
    } else {
      (void)__atomic_add_fetch(&manager->totalPersist, size, __ATOMIC_RELAXED);
      TRACE(NCCL_ALLOC, "MemManager: Track Persistent ptr=%p size=%zu dev=%d",
            ptr, size, manager->commCudaDev);
    }
    return ncclSuccess;
  }

  // Scratch/Offload: create linked list entry
  ncclDynMemEntry* entry = (ncclDynMemEntry*)malloc(sizeof(ncclDynMemEntry));
  if (entry == nullptr) {
    WARN("MemManager: Failed to allocate memory entry");
    return ncclSystemError;
  }

  // Initialize common fields
  memset(entry, 0, sizeof(ncclDynMemEntry));
  entry->ptr = ptr;
  entry->size = size;
  entry->handle = handle;
  entry->handleType = handleType;
  entry->memType = memType;
  entry->state = ncclDynMemStateActive;
  entry->cudaDev = manager->commCudaDev;
  entry->cpuBackup = nullptr;
  entry->isImportedFromPeer = isImportedFromPeer;

  // Initialize ownership-specific fields
  if (isImportedFromPeer) {
    entry->desc.imported.ownerRank = ownerRank;
    entry->desc.imported.ownerDev = ownerDev;
    entry->desc.imported.ownerPtr = ownerPtr;
  } else {
    if (handleType == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
      entry->desc.local.shareableHandle.fd = -1;  // avoid using 0 which is stdin
    }
    entry->desc.local.shareableHandleValid = false;
    entry->desc.local.numExportedPeers = 0;
    entry->desc.local.exportedPeersCapacity = 0;
    entry->desc.local.exportedPeerRanks = nullptr;
  }

  { // lock the mutex to add the entry to the linked list
    std::lock_guard<std::mutex> lock(manager->lock);
    // Add to linked list (prepend)
    entry->next = manager->entries;
    manager->entries = entry;
    manager->numEntries++;
  } // lock_guard automatically releases mutex

  // Update statistics
  if (isImportedFromPeer) {
    if (memType == ncclMemScratch) {
      (void)__atomic_add_fetch(&manager->totalScratchImported, size, __ATOMIC_RELAXED);
    } else if (memType == ncclMemOffload) {
      (void)__atomic_add_fetch(&manager->totalOffloadImported, size, __ATOMIC_RELAXED);
    }
    TRACE(NCCL_ALLOC, "MemManager: Track imported ptr=%p size=%zu type=%d from rank=%d entries=%d",
          ptr, size, memType, ownerRank, manager->numEntries);
  } else {
    if (memType == ncclMemScratch) {
      (void)__atomic_add_fetch(&manager->totalScratch, size, __ATOMIC_RELAXED);
    } else if (memType == ncclMemOffload) {
      (void)__atomic_add_fetch(&manager->totalOffload, size, __ATOMIC_RELAXED);
    }
    TRACE(NCCL_ALLOC, "MemManager: Track ptr=%p size=%zu type=%d dev=%d entries=%d",
          ptr, size, memType, manager->commCudaDev, manager->numEntries);
  }

  return ncclSuccess;
}

// Track a new allocation
ncclResult_t ncclMemTrack(
  struct ncclMemManager* manager,
  void* ptr,
  size_t size,
  CUmemGenericAllocationHandle handle,
  CUmemAllocationHandleType handleType,
  ncclMemType_t memType
) {
  return ncclMemTrackInternal(manager, ptr, size, handle, handleType, memType,
                              false, -1, -1, nullptr);
}

// Track imported allocation from peer
ncclResult_t ncclMemTrackImportFromPeer(
  struct ncclMemManager* manager,
  void* ptr,
  size_t size,
  CUmemGenericAllocationHandle handle,
  CUmemAllocationHandleType handleType,
  ncclMemType_t memType,
  int ownerRank,
  int ownerDev,
  void* ownerPtr
) {
  return ncclMemTrackInternal(manager, ptr, size, handle, handleType, memType,
                              true, ownerRank, ownerDev, ownerPtr);
}

// Untrack allocation
ncclResult_t ncclMemUntrack(struct ncclMemManager* manager, void* ptr, size_t size) {
  if (ncclParamMemManagerDisable()) return ncclSuccess;
  if (manager == nullptr || ptr == nullptr) return ncclInternalError;

  // Atomic check to avoid locking destroyed mutex
  if (!__atomic_load_n(&manager->initialized, __ATOMIC_ACQUIRE)) {
    WARN("MemManager: Cannot untrack allocation ptr=%p, manager not initialized", ptr);
    return ncclInternalError;
  }

  // Variables to save values before releasing lock
  size_t entrySize = 0;
  [[maybe_unused]] int numEntries = 0;  // May be unused if TRACE compiled out
  bool isImportedFromPeer = false;
  ncclMemType_t memType = ncclMemScratch;

  {
    std::lock_guard<std::mutex> lock(manager->lock);

    ncclDynMemEntry* prev = nullptr;
    ncclDynMemEntry* entry = manager->entries;

    while (entry != nullptr) {
      if (entry->ptr == ptr) {
        // Remove from linked list
        if (prev == nullptr) {
          manager->entries = entry->next;
        } else {
          prev->next = entry->next;
        }
        manager->numEntries--;

        // Free CPU backup if exists
        if (entry->cpuBackup != nullptr) {
          manager->cpuBackupUsage -= entry->size;
          ncclCudaHostFree(entry->cpuBackup);
        }

        // Close shareable FD if valid (defensive cleanup for POSIX FD handle type)
        if (!entry->isImportedFromPeer &&
            entry->desc.local.shareableHandleValid &&
            entry->handleType == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR &&
            entry->desc.local.shareableHandle.fd >= 0) {
          close(entry->desc.local.shareableHandle.fd);
          entry->desc.local.shareableHandle.fd = -1;
          entry->desc.local.shareableHandleValid = false;
        }

        // Only local entries have exportedPeerRanks (imported entries use desc.imported union member)
        if (!entry->isImportedFromPeer && entry->desc.local.exportedPeerRanks != nullptr) {
          free(entry->desc.local.exportedPeerRanks);
        }

        // Save values before unlock for logging (may be unused if TRACE is compiled out)
        entrySize = entry->size;
        numEntries = manager->numEntries;
        isImportedFromPeer = entry->isImportedFromPeer;
        memType = entry->memType;

        // Safety check: log if tracked size doesn't match passed size
        if (entrySize != size) {
          INFO(NCCL_ALLOC, "MemManager: Untrack size mismatch ptr=%p tracked=%zu passed=%zu", ptr, entrySize, size);
        }

        free(entry);
        break;
      }
      prev = entry;
      entry = entry->next;
    }
  } // lock_guard automatically releases mutex

  // Update statistics
  if (entrySize > 0) {
    // Entry found in linked list
    if (isImportedFromPeer) {
      if (memType == ncclMemScratch) {
        (void)__atomic_sub_fetch(&manager->totalScratchImported, entrySize, __ATOMIC_RELAXED);
      } else if (memType == ncclMemOffload) {
        (void)__atomic_sub_fetch(&manager->totalOffloadImported, entrySize, __ATOMIC_RELAXED);
      }
    } else {
      if (memType == ncclMemScratch) {
        (void)__atomic_sub_fetch(&manager->totalScratch, entrySize, __ATOMIC_RELAXED);
      } else if (memType == ncclMemOffload) {
        (void)__atomic_sub_fetch(&manager->totalOffload, entrySize, __ATOMIC_RELAXED);
      }
    }

    TRACE(NCCL_ALLOC, "MemManager: Untrack ptr=%p size=%zu entries=%d",
          ptr, entrySize, numEntries);
  } else {
    // Entry not found in linked list - must be persistent memory
    (void)__atomic_sub_fetch(&manager->totalPersist, size, __ATOMIC_RELAXED);
    TRACE(NCCL_ALLOC, "MemManager: Untrack Persistent ptr=%p size=%zu", ptr, size);
  }

  return ncclSuccess;
}

// Track that a buffer is being shared with a peer (for suspend/resume coordination)
// NOTE: Only for dynamic memory (scratch/offload) that's in the linked list.
// Persistent memory doesn't need export tracking since it's never suspended.
// Call this after allocating dynamic memory and the peer imports it.
ncclResult_t ncclDynMemMarkExportToPeer(struct ncclMemManager* manager, void* ptr, int peerRank) {
  if (ncclParamMemManagerDisable()) return ncclSuccess;
  if (manager == nullptr || ptr == nullptr) return ncclInternalError;
  if (!__atomic_load_n(&manager->initialized, __ATOMIC_ACQUIRE)) {
    WARN("MemManager: Cannot mark export for ptr=%p, manager not initialized", ptr);
    return ncclInternalError;
  }
  std::lock_guard<std::mutex> lock(manager->lock);

  // Find entry in linked list (only contains scratch/offload, not persistent)
  ncclDynMemEntry* entry = manager->entries;
  while (entry != nullptr && entry->ptr != ptr) {
    entry = entry->next;
  }

  if (entry == nullptr) {
    WARN("MemManager: Cannot mark export for ptr=%p - not found in tracked entries. "
         "Only dynamic memory (scratch/offload) needs export tracking for suspend/resume.", ptr);
    return ncclInternalError;
  }

  // Verify this is a local entry, not an imported one
  if (entry->isImportedFromPeer) {
    WARN("MemManager: Cannot mark export for ptr=%p - this is an imported buffer, not a local one", ptr);
    return ncclInternalError;
  }

  // Check if peer already exists
  for (int i = 0; i < entry->desc.local.numExportedPeers; i++) {
    if (entry->desc.local.exportedPeerRanks[i] == peerRank) {
      WARN("MemManager: Buffer ptr=%p already exported to peer rank %d", ptr, peerRank);
      return ncclInternalError;
    }
  }

  if (entry->desc.local.numExportedPeers >= entry->desc.local.exportedPeersCapacity) {
    int newCapacity = entry->desc.local.exportedPeersCapacity == 0 ? NCCL_MEM_EXPORT_PEERS_INIT :
                      entry->desc.local.exportedPeersCapacity * 2;
    ncclResult_t ret = ncclRealloc(&entry->desc.local.exportedPeerRanks,
                                   entry->desc.local.exportedPeersCapacity,
                                   newCapacity);
    if (ret != ncclSuccess) {
      WARN("MemManager: Failed to grow exportedPeerRanks array for ptr=%p", ptr);
      return ret;
    }
    entry->desc.local.exportedPeersCapacity = newCapacity;
  }

  // Add peer to export list
  entry->desc.local.exportedPeerRanks[entry->desc.local.numExportedPeers++] = peerRank;

  TRACE(NCCL_ALLOC, "MemManager: ExportToPeer ptr=%p peerRank=%d numExportedPeers=%d",
        ptr, peerRank, entry->desc.local.numExportedPeers);
  return ncclSuccess;
}
