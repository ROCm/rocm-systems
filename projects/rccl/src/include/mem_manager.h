/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_MEM_MANAGER_H_
#define NCCL_MEM_MANAGER_H_

#include "nccl.h"
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <mutex>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ncclComm;

#define NCCL_MEM_EXPORT_PEERS_INIT 8

typedef enum {
  ncclMemPersist = 0,
  ncclMemScratch = 1,
  ncclMemOffload = 2
} ncclMemType_t;

typedef enum {
  ncclDynMemStateActive   = 0,
  ncclDynMemStateReleased = 1
} ncclDynMemState_t;

typedef struct ncclDynMemLocalDesc {
  bool shareableHandleValid;
  int  numExportedPeers;
  int  exportedPeersCapacity;
  int* exportedPeerRanks;
} ncclDynMemLocalDesc;

typedef struct ncclDynMemImportDesc {
  int   ownerRank;
  int   ownerDev;
  void* ownerPtr;
} ncclDynMemImportDesc;

typedef struct ncclDynMemEntry {
  void*                            ptr;        // Device VA
  size_t                           size;
  hipMemGenericAllocationHandle_t  handle;     // VMM physical handle
  hipMemAllocationHandleType       handleType;
  ncclMemType_t                    memType;
  ncclDynMemState_t                state;
  int                              cudaDev;

  void* cpuBackup;
  hipMemAllocationProp prop;

  bool isImportedFromPeer;
  union {
    ncclDynMemLocalDesc  local;
    ncclDynMemImportDesc imported;
  } desc;

  struct ncclDynMemEntry* next;
} ncclDynMemEntry;

struct ncclSuspendCanary {
  void*                            va;       // reserved virtual address
  size_t                           size;     // bytes (>= granularity)
  hipMemGenericAllocationHandle_t  handle;   // physical handle when active
  int                              cudaDev;
  int                              valid;    // 1 iff `va` reservation alive
};

typedef struct ncclMemManager {
  ncclDynMemEntry* entries;       // Linked list of scratch/offload entries.
  int              numEntries;
  std::mutex       lock;
  int              released;      // 1 iff suspended (physical mappings dropped)
  int              initialized;
  int              refCount;      // Future-proof: split_share comm sharing

  size_t totalPersist;
  size_t totalPersistImported;
  size_t totalScratch;
  size_t totalScratchImported;
  size_t totalOffload;
  size_t totalOffloadImported;
  size_t cpuBackupUsage;          // Bytes currently held in host backups.

  int commCudaDev;

  struct ncclSuspendCanary canary;
} ncclMemManager;

struct ncclMemManagerTask {
  struct ncclMemManagerTask* next;
  struct ncclComm*           comm;
};

ncclResult_t ncclMemManagerInit(struct ncclComm* comm);
ncclResult_t ncclMemManagerDestroy(struct ncclComm* comm);

ncclResult_t ncclMemTrack(
    struct ncclMemManager*           manager,
    void*                            ptr,
    size_t                           size,
    hipMemGenericAllocationHandle_t  handle,
    hipMemAllocationHandleType       handleType,
    ncclMemType_t                    memType);

ncclResult_t ncclMemTrackImportFromPeer(
    struct ncclMemManager*           manager,
    void*                            ptr,
    size_t                           size,
    hipMemGenericAllocationHandle_t  handle,
    hipMemAllocationHandleType       handleType,
    ncclMemType_t                    memType,
    int                              ownerRank,
    int                              ownerDev,
    void*                            ownerPtr);

ncclResult_t ncclMemUntrack(
    struct ncclMemManager* manager,
    void*                  ptr,
    size_t                 size);

ncclResult_t ncclDynMemMarkExportToPeer(
    struct ncclMemManager* manager,
    void*                  ptr,
    int                    peerRank);

ncclResult_t ncclCommMemSuspend(struct ncclComm* comm);
ncclResult_t ncclCommMemResume(struct ncclComm* comm);
ncclResult_t ncclCommSuspendCanaryFree(struct ncclComm* comm);
ncclResult_t ncclCommSuspendForceResumeForDestroy(struct ncclComm* comm);

ncclResult_t ncclCommSuspend_impl(ncclComm_t comm, int flags);
ncclResult_t ncclCommResume_impl(ncclComm_t comm);
ncclResult_t ncclCommMemStats_impl(
    ncclComm_t comm, ncclCommMemStat_t stat, uint64_t* value);

#ifdef __cplusplus
}
#endif

#endif // NCCL_MEM_MANAGER_H_
