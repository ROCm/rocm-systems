/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _NCCL_GIN_HOST_H_
#define _NCCL_GIN_HOST_H_

#include "allocator.h"
#include "nccl.h"
#include "nccl_gin.h"
#include "nccl_device/core_tmp.h"
#include "os.h"
#include "nccl_device/gin/gin_device_host_common.h"
#include <atomic>
#include <shared_mutex>
#include <thread>

struct ncclGinStateDevComm {
  int contextCount;
  int backendIndex;  // index into ncclGinState::backends[] for the backend that owns these contexts
  void* ginCtx[NCCL_GIN_MAX_CONNECTIONS];
  ncclNetDeviceHandle_t* devHandles[NCCL_GIN_MAX_CONNECTIONS];
  struct ncclGinStateDevComm* next;
};

struct ncclGinBackendState {
  ncclGinType_t ginType;      // GIN backend type.
  ncclGin_t* ncclGin;
  void* ginInstance;          // Plugin's per-comm opaque context.
  int pluginIndex;            // Index into pluginLibs[].
  int ginCommCount;
  void* ginComms[NCCL_GIN_MAX_CONNECTIONS];
  ncclNetProperties_t ginProps[NCCL_GIN_MAX_CONNECTIONS];
  bool supportsStrongSignals;
  bool supportsVASignals;
  // False exactly while ginComms[] hold live handles. Set before closeColl by
  // ncclGinHostFinalize and by ncclGinConnectOnce's failure path: register /
  // deregister / DevCommSetup must not walk ginComms[] once they are NULLed.
  // ncclGinConnectOnce clears it when a retry repopulates them.
  bool closed;
  // Bumped on each successful connect, so a window records which generation of
  // ginComms[] it was registered against. ginComms[] is shared through sharedRes
  // while windows live per comm, so after a sibling's teardown and a survivor's
  // reconnect the two disagree; ncclGinDeregister compares them instead of pairing
  // a reconnected collComm with a mhandle from the closed connection. 0 means
  // never connected, which no registered window can carry.
  uint32_t connectGeneration;
};

struct ncclGinState {
  ncclAffinity cpuAffinity;
  bool connected;
  bool supported;              // True if any backend is loaded on this comm.
  int proxyNthreads;           // Number of GIN progress threads.
  bool proxyThreadsCreated;     // Set once the GIN progress thread is spawned.
  std::atomic<bool> proxyThreadStopSignal;  // Signals the GIN progress thread to exit.
  // Use std::shared_timed_mutex (C++14) rather than std::shared_mutex (C++17)
  // because NCCL targets C++14 for CUDA < 13.
  std::shared_timed_mutex devCommRwMutex;  // Readers: proxy threads; Writer: main thread (setup/free).
  // When true, readers skip locking devCommRwMutex. Prevents writer starvation.
  std::atomic<bool> writePending;
  std::thread thread[NCCL_GIN_MAX_CONNECTIONS];
  ncclResult_t asyncResult;

  struct ncclGinStateDevComm* devComms;
  ncclGinConnectionType_t ginConnectionType;

  // Set by ncclGinHostFinalize and never cleared. backends[] then survives only
  // for ncclGinFinalize on the last sharedRes reference: each ginInstance still
  // points at the comm that ran ncclGinInit, which may already be freed, so no
  // comm on this sharedRes may advertise, select or reconnect these backends.
  bool finalized;
  int numActiveBackends;
  struct ncclGinBackendState backends[NCCL_GIN_MAX_ACTIVE_BACKENDS];
};

extern int64_t ncclParamGinType();

// Get the GIN type from comm. ginType is set to the GIN type that can be used
// by the comm to communicate with other nodes.
ncclResult_t ncclGetGinType(struct ncclComm* comm, ncclGinType_t* ginType);
ncclResult_t ncclGetRailedGinType(struct ncclComm* comm, ncclGinType_t* ginType);

// FIXME change to ncclGinState instead of ncclComm, no need to pass comm
ncclResult_t ncclGinConnectOnce(struct ncclComm* comm);
ncclResult_t ncclGinHostFinalize(struct ncclComm* comm);
ncclResult_t ncclGinDevCommSetup(struct ncclComm* comm, struct ncclDevCommRequirements const* reqs,
                                 struct ncclDevComm* devComm, uint32_t deviceCodeVersion);
ncclResult_t ncclGinDevCommFree(struct ncclComm* comm, struct ncclDevComm const* devComm);
ncclResult_t ncclGinRegister(struct ncclComm* comm, void* address, size_t size,
                             void* ginHostWins[NCCL_GIN_MAX_CONNECTIONS * NCCL_GIN_MAX_ACTIVE_BACKENDS],
                             ncclGinWindow_t ginDevWins[NCCL_GIN_MAX_CONNECTIONS * NCCL_GIN_MAX_ACTIVE_BACKENDS],
                             uint32_t ginWinGenerations[NCCL_GIN_MAX_ACTIVE_BACKENDS],
                             int winFlags, bool multiSegment = false, int memType = NCCL_PTR_CUDA);
ncclResult_t ncclGinDeregister(struct ncclComm* comm,
                               void* ginHostWins[NCCL_GIN_MAX_CONNECTIONS * NCCL_GIN_MAX_ACTIVE_BACKENDS],
                               uint32_t const ginWinGenerations[NCCL_GIN_MAX_ACTIVE_BACKENDS]);

ncclResult_t ncclGinQueryLastError(struct ncclGinState* ginState, bool* hasError);

ncclResult_t ncclGinSetDefaultBackend(struct ncclComm* comm, uint64_t globalBitmask);

#endif
