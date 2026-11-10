/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_REGISTER_H_
#define NCCL_REGISTER_H_

#include "device.h"

#include <cuda.h>
#include <stdint.h>

struct ncclMcArenaReg;

int64_t ncclParamLocalRegister();
int64_t ncclParamGraphRegister();

enum {
  NET_REG_COMPLETE = 0x01,
  NVLS_REG_COMPLETE = 0x02,
  NVLS_REG_POSSIBLE = 0x04,
  NVLS_REG_NO_SUPPORT = 0x08,
  COLLNET_REG_COMPLETE = 0x10,
  IPC_REG_COMPLETE = 0x20
};

struct ncclPeerRegIpcAddr {
  uintptr_t* devPeerRmtAddrs;
  uintptr_t* hostPeerRmtAddrs;
};

struct ncclRegNetHandles {
  void* handle;
  struct ncclProxyConnector* proxyConn;
  struct ncclRegNetHandles* next;
};

struct ncclReg {
  // common attributes
  uintptr_t begAddr, endAddr; // page aligned
  int localRefs;
  int graphRefs;
  uint32_t state;
  // net reg
  struct ncclRegNetHandles* netHandleHead;
  // nvls reg
  // Committed arena registration, owned here: NVLS_REG_COMPLETE implies non-NULL.
  struct ncclMcArenaReg* nvlsUbReg;
  // collnet reg
  void* collnetHandle;
  // gin reg
  void** ginMhandles;
  void** ginHandles;
  struct ncclProxyConnector* collnetProxyconn;
  // general ipc reg
  struct ncclPeerRegIpcAddr regIpcAddrs;
  struct ncclIpcRegInfo** ipcInfos;  // Dynamically allocated, sized to ipcInfosSize
  int ipcInfosSize;                  // Size of ipcInfos array (localRanks or nRanks for cross-clique)
  // RCCL: NET segment cache, written only by rcclNetRegSegmentCount / rcclNetRegCommit (rccl_wrap.cc).
  struct { int nSegments; bool allPeers; } rcclNet;
  // RCCL: set by rcclRegAutoLocal for a record RCCL created on the user's behalf. The record spans the whole
  // allocation and bufferId is its HIP buffer ID; a different ID at the same address means the allocation was freed
  // and reused. retiredNext links the record into ncclRegCache::rcclRetired once it is out of the cache.
  struct { bool owned; unsigned long long bufferId; struct ncclReg* retiredNext; } rcclAuto;
};

struct ncclRegCache {
  struct ncclReg** slots;
  int capacity, population;
  uintptr_t pageSize;
  // RCCL: stale RCCL-owned records. Their peer imports stay mapped until ncclRegCleanup, because unmapping one
  // while collectives are in flight deadlocks (hipMemUnmap in the peer's proxy waits for a kernel that waits
  // on this rank). rcclAutoBytes counts the live and retired RCCL-owned bytes against
  // RCCL_RING_AUTO_REGISTER_MAX_BYTES.
  struct ncclReg* rcclRetired;
  size_t rcclAutoBytes;
};

ncclResult_t ncclRegCleanup(struct ncclComm* comm);
// Returns in *reg the cache record covering [buff, buff + size), creating one owned by RCCL for the buffer's whole
// allocation when the user has not registered it. *reg is NULL when the buffer cannot be registered; that is not an
// error. Does not IPC-map.
ncclResult_t rcclRegAutoLocal(struct ncclComm* comm, void* buff, size_t size, struct ncclReg** reg);
// Called by ncclRegFind and ncclRegister for an RCCL-owned record covering buff. If buff now belongs to a different
// allocation, sets *stale, drops RCCL's reference and, when nobody else holds the record, moves it out of the cache
// onto the retired list. The caller must not use a stale record.
ncclResult_t rcclRegCheckOwned(struct ncclComm* comm, const void* buff, struct ncclReg* reg, bool* stale);
// Releases every RCCL-owned record, live or retired, including its peer imports. Call it while the peers'
// proxies are still running and no collective on comm is in flight.
ncclResult_t rcclRegAutoRelease(struct ncclComm* comm);
ncclResult_t ncclCommGraphRegister(const ncclComm_t comm, void* buff, size_t size, void** handle);
ncclResult_t ncclCommGraphDeregister(const ncclComm_t comm, struct ncclReg* handle);
ncclResult_t ncclRegLocalIsValid(struct ncclReg* reg, bool* isValid);

#endif
