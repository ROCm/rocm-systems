/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Modifications Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_MEM_MANAGER_H_
#define NCCL_MEM_MANAGER_H_

#include "nccl.h"
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <stdbool.h>
#include <mutex>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef HIP_FABRIC_API
// FABRIC handle support lifted for HIP runtimes that do not expose it yet
#ifndef HIP_IPC_HANDLE_SIZE
#define HIP_IPC_HANDLE_SIZE 64
#endif
typedef struct hipMemFabricHandle_st {
    unsigned char data[HIP_IPC_HANDLE_SIZE];
} hipMemFabricHandle_compat_t;
#else
typedef hipMemFabricHandle_t hipMemFabricHandle_compat_t;
#ifdef __cplusplus
// Guard against silent ABI drift if HIP ever changes the FABRIC handle layout:
// our compat type is hard-coded to 64 bytes (HIP_IPC_HANDLE_SIZE), so any
// future divergence in the real hipMemFabricHandle_t size must trip this
// assert and be addressed explicitly.
static_assert(sizeof(hipMemFabricHandle_compat_t) == 64,
              "hipMemFabricHandle_t size diverged from the 64-byte compat layout; "
              "update HIP_IPC_HANDLE_SIZE / serialization code in mem_manager.{h,cc}");
#endif
#endif

struct ncclComm;

// Initial capacity for exported peers array
#define NCCL_MEM_EXPORT_PEERS_INIT 8

// Memory Type for NCCL allocations
typedef enum {
  ncclMemPersist  = 0,  // Persistent memory - track stats only, never release/offload
  ncclMemScratch  = 1,  // Free without saving
  ncclMemOffload  = 2   // Copy to CPU before free, restore on resume
} ncclMemType_t;

// Memory entry state
typedef enum {
  ncclDynMemStateActive   = 0,  // Memory is allocated and usable
  ncclDynMemStateReleased = 1   // Memory has been released
} ncclDynMemState_t;

// Local owned memory descriptor
typedef struct ncclDynMemLocalDesc {
  // Shareable handle for P2P exports
  // TODO: Remove the 'fd' field - POSIX FD handles are converted on-demand via proxy
  // (ncclProxyClientGetFdBlocking), so we no longer export them upfront. Only FABRIC
  // handles need upfront export since they can be shared directly via messaging.
  union {
    int                          fd;            // For POSIX_FILE_DESCRIPTOR (unused)
    hipMemFabricHandle_compat_t  fabricHandle;  // For FABRIC
  } shareableHandle;
  bool                           shareableHandleValid;
  // Peer tracking for P2P exports
  int                            numExportedPeers;
  int                            exportedPeersCapacity;
  int*                           exportedPeerRanks;
} ncclDynMemLocalDesc;

// Imported from peer memory descriptor
typedef struct ncclDynMemImportDesc {
  int                            ownerRank;     // Rank that owns the original buffer
  int                            ownerDev;      // CUDA device of the owner
  void*                          ownerPtr;      // Owner's virtual address
} ncclDynMemImportDesc;

// Individual tracked memory entry (only track scratch and offload allocations)
typedef struct ncclDynMemEntry {
  void*                            ptr;           // GPU virtual address
  size_t                           size;          // Allocation size
  hipMemGenericAllocationHandle_t  handle;        // Physical memory handle
  hipMemAllocationHandleType       handleType;
  ncclMemType_t                    memType;
  ncclDynMemState_t                state;
  int                              cudaDev;

  // CPU backup for OFFLOAD type memory
  void*                            cpuBackup;     // Host memory for offloaded data

  // Ownership type and type-specific data
  bool                             isImportedFromPeer;  // true if this is a peer-imported buffer
  union {
    ncclDynMemLocalDesc            local;
    ncclDynMemImportDesc           imported;
  } desc;

  // Linked list pointer
  struct ncclDynMemEntry*          next;
} ncclDynMemEntry;

// P2P Handle Exchange Structure
typedef struct ncclDynMemP2pHandleInfo {
  void*    ptr;
  int      ownerRank;
  int      ownerDev;
  size_t   size;
  int      handleType;
  union {
    uint64_t                     handleData;
    hipMemFabricHandle_compat_t  fabricHandle;
  };
} ncclDynMemP2pHandleInfo;

// Memory manager attached to ncclComm
typedef struct ncclMemManager {
  ncclDynMemEntry*  entries;  // Linked list of tracked allocations, only track scratch and offload allocations
  int               numEntries;
  std::mutex        lock;
  int               released;
  int               initialized;
  int               refCount;

  size_t            totalPersist;
  size_t            totalPersistImported;
  size_t            totalScratch;
  size_t            totalScratchImported;
  size_t            totalOffload;
  size_t            totalOffloadImported;
  size_t            cpuBackupUsage;

  int               commCudaDev;
} ncclMemManager;

struct ncclMemManagerTask {
  struct ncclMemManagerTask* next;
  struct ncclComm* comm;
};

// Initialize memory manager
ncclResult_t ncclMemManagerInit(struct ncclComm* comm);

// Destroy memory manager and free all resources
ncclResult_t ncclMemManagerDestroy(struct ncclComm* comm);

// Track a new allocation
ncclResult_t ncclMemTrack(
  struct ncclMemManager* manager,
  void* ptr,
  size_t size,
  hipMemGenericAllocationHandle_t handle,
  hipMemAllocationHandleType handleType,
  ncclMemType_t memType
);

// Track imported allocation from peer
ncclResult_t ncclMemTrackImportFromPeer(
  struct ncclMemManager* manager,
  void* ptr,
  size_t size,
  hipMemGenericAllocationHandle_t handle,
  hipMemAllocationHandleType handleType,
  ncclMemType_t memType,
  int ownerRank,
  int ownerDev,
  void* ownerPtr
);

// Untrack allocation
ncclResult_t ncclMemUntrack(struct ncclMemManager* manager, void* ptr, size_t size);

// Add peer info for buffers in the linked list entries (only for dynamic memory: scratch/offload)
ncclResult_t ncclDynMemMarkExportToPeer(struct ncclMemManager* manager, void* ptr, int peerRank);

ncclResult_t ncclCommMemSuspend(struct ncclComm* comm);
ncclResult_t ncclCommMemResume(struct ncclComm* comm);

// RCCL: Public API _impl entry points (dispatched from src/misc/api_trace.cc)
ncclResult_t ncclCommSuspend_impl(ncclComm_t comm, int flags);
ncclResult_t ncclCommResume_impl(ncclComm_t comm);
ncclResult_t ncclCommMemStats_impl(ncclComm_t comm, ncclCommMemStat_t stat, uint64_t* value);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
// RCCL: true if a tracked entry for `ptr` was already torn down by Suspend
// (state==Released). ncclCuMemFree / ncclCudaFree use it to skip the real
// teardown for Suspended entries while still freeing persistent / untracked
// pointers on Destroy-while-Suspended.
static inline bool ncclMemEntryAlreadyReleased(struct ncclMemManager* manager,
                                               void* ptr) {
  if (manager == nullptr) return false;
  if (!__atomic_load_n(&manager->released, __ATOMIC_ACQUIRE)) return false;
  std::lock_guard<std::mutex> lock(manager->lock);
  for (ncclDynMemEntry* e = manager->entries; e != nullptr; e = e->next) {
    if (e->ptr == ptr) return e->state == ncclDynMemStateReleased;
  }
  return false;
}
#endif

#endif /* NCCL_MEM_MANAGER_H_ */
