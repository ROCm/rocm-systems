/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifdef ENABLE_ROCSHMEM_GIN

#include "gin/gin_host_anvil.h"
#include "bootstrap.h"
#include "comm.h"
#include "dev_runtime.h"
#include "gin/gin_host_rocshmem.h"
#include "nccl_device/gin/anvil/gin_anvil_device_host_common.h"

// Anvil host API (src/sdma); include path from ROCSHMEM_SOURCE_DIR or mono-repo sibling.
#include "sdma/anvil.hpp"
#include <rocshmem/rocshmem.hpp>

#include <hip/hip_runtime.h>
#include <map>

struct ginAnvilCtx {
  struct ncclComm* comm;
  void* collComm;
  ncclNetDeviceHandle_v11_t* devHandle;

  int nContexts;
  ncclGinAnvilGPUContext* gpuCtxDev;

  // Device arrays
  void** queuesDev;         // [nRanks]
  uint64_t** signalsBaseDev;// [nRanks]

  // Host mirrors
  void** queuesHost;
  uint64_t** signalsBaseHost;

  uint64_t* signalsAlloc;   // rocshmem_malloc base (all contexts); remote SDMA atomics require symmetric heap
  uint64_t* countersAlloc;  // hipMalloc base (local-only)

  bool hasError;
};

struct ginAnvilMemHandle {
  ncclGinAnvilMemHandle* devHandle;
  void* addr;
  size_t size;
};

static ncclResult_t allocDeviceArray(void*** devPtr, void** hostData, size_t count) {
  if (hipMalloc(devPtr, count * sizeof(void*)) != hipSuccess) return ncclSystemError;
  if (hipMemcpy(*devPtr, hostData, count * sizeof(void*), hipMemcpyHostToDevice) != hipSuccess) return ncclSystemError;
  return ncclSuccess;
}

ncclResult_t ncclGinAnvilCreateContext(struct ncclComm* comm, void* collComm, int devId,
                                       int nSignals, int nCounters, int nContexts,
                                       void** outGinCtx, ncclNetDeviceHandle_v11_t** outDevHandle) {
  ncclResult_t ret = ncclSuccess;
  ginAnvilCtx* ctx = nullptr;
  uint64_t* signalsLocal = nullptr;
  uint64_t* countersLocal = nullptr;
  ncclGinAnvilGPUContext* gpuCtxHostArr = nullptr;
  struct ncclDevrState* devr = nullptr;
  if (nContexts < 1) nContexts = 1;
  NCCLCHECK(ncclCalloc(&ctx, 1));
  ctx->comm = comm;
  ctx->collComm = collComm;
  ctx->nContexts = nContexts;
  ctx->hasError = false;

  // GIN_ANVIL uses MI300 xGMI SDMA and is valid only for single-node jobs.
  if (comm->nNodes > 1) {
    WARN("GIN anvil (NCCL_GIN_TYPE=5) requires a single-node communicator (nNodes=%d)", comm->nNodes);
    ret = ncclInvalidUsage;
    goto fail;
  }

  devr = &comm->devrState;
  // rocshmem_malloc and Anvil endpoint setup require a live rocSHMEM runtime.
  NCCLCHECK(ncclDevrInitOnce(comm));
  NCCLCHECK(ncclGinRocshmemEnsureInit(comm));

  // Indexed signals live on the rocSHMEM symmetric heap so peer SDMA signal atomics
  // (GIN barrier + put completion) can write destination signal cells. hipMalloc pages
  // fault as read-only under remote SDMA stores.
  if (nSignals > 0) {
    size_t signalsBytes = (size_t)nSignals * (size_t)nContexts * sizeof(uint64_t);
    signalsLocal = (uint64_t*)rocshmem::rocshmem_malloc(signalsBytes);
    if (signalsLocal == nullptr) {
      WARN("GIN anvil: rocshmem_malloc failed for %d signals x %d contexts", nSignals, nContexts);
      ret = ncclSystemError;
      goto fail;
    }
    ctx->signalsAlloc = signalsLocal;
    hipMemset(signalsLocal, 0, signalsBytes);
  }
  if (nCounters > 0) {
    size_t countersBytes = (size_t)nCounters * (size_t)nContexts * sizeof(uint64_t);
    if (hipMalloc((void**)&countersLocal, countersBytes) != hipSuccess) { ret = ncclSystemError; goto fail; }
    ctx->countersAlloc = countersLocal;
    hipMemset(countersLocal, 0, countersBytes);
  }

  // Enable xGMI P2P for in-node SDMA queues (rocshmem IPC covers symmetric heap).
  for (int r = 0; r < comm->nRanks; r++) {
    if (r == comm->rank) continue;
    if (comm->rankToNode[r] != comm->rankToNode[comm->rank]) continue;
    int peerDev = comm->peerInfo[r].cudaDev;
    if (comm->cudaDev == peerDev) continue;
    rocshmem::anvil::EnablePeerAccess(comm->cudaDev, peerDev);
  }

  // Build per-peer signal bases for SDMA on this GPU. Anvil SDMA packets carry an
  // address in the *source* GPU VA space (like rocshmem AMOs), not the peer's raw
  // local pointer. rocshmem_ptr(localBase, pe) returns the IPC-mapped address.
  NCCLCHECKGOTO(ncclCalloc(&ctx->signalsBaseHost, comm->nRanks), ret, fail_signals);
  for (int r = 0; r < comm->nRanks; r++) {
    if (signalsLocal == nullptr) {
      ctx->signalsBaseHost[r] = nullptr;
      continue;
    }
    if (r == comm->rank) {
      ctx->signalsBaseHost[r] = signalsLocal;
    } else if (comm->rankToNode[r] != comm->rankToNode[comm->rank]) {
      ctx->signalsBaseHost[r] = nullptr;
    } else {
      void* remote = rocshmem::rocshmem_ptr(signalsLocal, r);
      if (remote == nullptr) {
        WARN("GIN anvil: rocshmem_ptr(signals=%p, pe=%d) failed", (void*)signalsLocal, r);
        ret = ncclSystemError;
        goto fail_signals;
      }
      ctx->signalsBaseHost[r] = (uint64_t*)remote;
    }
  }

  NCCLCHECKGOTO(ncclCalloc(&ctx->queuesHost, comm->nRanks), ret, fail_signals);

  // Create SDMA queues for each peer in-node.
  if (!rocshmem::anvil::initEndpoint()) {
    WARN("GIN anvil: anvil initEndpoint failed");
    ret = ncclSystemError;
    goto fail_signals;
  }
  for (int r = 0; r < comm->nRanks; r++) {
    if (r == comm->rank) { ctx->queuesHost[r] = nullptr; continue; }
    if (comm->rankToNode[r] != comm->rankToNode[comm->rank]) { ctx->queuesHost[r] = nullptr; continue; }
    int srcDev = comm->cudaDev;
    int dstDev = comm->peerInfo[r].cudaDev;
    rocshmem::anvil::anvil.connect(srcDev, dstDev, /*numChannels=*/1);
    auto* q = rocshmem::anvil::anvil.getSdmaQueue(srcDev, dstDev, 0);
    if (q == nullptr) {
      WARN("GIN anvil: getSdmaQueue failed for dev %d -> %d", srcDev, dstDev);
      ret = ncclSystemError;
      goto fail_signals;
    }
    ctx->queuesHost[r] = (void*)q->deviceHandle();
  }

  // Allocate device arrays for queues and peer signal bases.
  NCCLCHECKGOTO(allocDeviceArray(&ctx->queuesDev, ctx->queuesHost, comm->nRanks), ret, fail_signals);
  if (hipMalloc(&ctx->signalsBaseDev, comm->nRanks * sizeof(uint64_t*)) != hipSuccess) { ret = ncclSystemError; goto fail_signals; }
  if (hipMemcpy(ctx->signalsBaseDev, ctx->signalsBaseHost, comm->nRanks * sizeof(uint64_t*), hipMemcpyHostToDevice) != hipSuccess) { ret = ncclSystemError; goto fail_signals; }

  // Build per-context GPU contexts (proxy uses an array indexed by contextId).
  NCCLCHECKGOTO(ncclCalloc(&gpuCtxHostArr, nContexts), ret, fail_signals);
  for (int contextId = 0; contextId < nContexts; contextId++) {
    ncclGinAnvilGPUContext* h = &gpuCtxHostArr[contextId];
    memset(h, 0, sizeof(*h));
    h->queues = ctx->queuesDev;
    h->signalsBase = ctx->signalsBaseDev;
    h->signals = signalsLocal ? signalsLocal + (size_t)contextId * (size_t)nSignals : nullptr;
    h->counters = countersLocal ? countersLocal + (size_t)contextId * (size_t)nCounters : nullptr;
    h->nSignals = nSignals;
    h->nCounters = nCounters;
    h->nRanks = comm->nRanks;
    h->rank = comm->rank;
    h->myNode = comm->rankToNode[comm->rank];
    h->lsaStrideBytes = devr->bigSize;
    h->lsaRank0Base = (uintptr_t)devr->lsaFlatBase;
  }

  if (hipMalloc(&ctx->gpuCtxDev, (size_t)nContexts * sizeof(ncclGinAnvilGPUContext)) != hipSuccess) {
    ret = ncclSystemError;
    goto fail_signals;
  }
  if (hipMemcpy(ctx->gpuCtxDev, gpuCtxHostArr, (size_t)nContexts * sizeof(ncclGinAnvilGPUContext),
                hipMemcpyHostToDevice) != hipSuccess) {
    ret = ncclSystemError;
    goto fail_signals;
  }

  // Create net device handle
  NCCLCHECK(ncclCalloc(&ctx->devHandle, 1));
  ctx->devHandle->netDeviceType = NCCL_NET_DEVICE_GIN_ANVIL;
  ctx->devHandle->netDeviceVersion = NCCL_GIN_ANVIL_VERSION;
  ctx->devHandle->needsProxyProgress = 0;
  ctx->devHandle->handle = ctx->gpuCtxDev;
  ctx->devHandle->size = 0;

  *outGinCtx = ctx;
  *outDevHandle = ctx->devHandle;
  INFO(NCCL_INIT, "GIN anvil: context created (%d signals, %d counters, %d contexts, signalsDev=%p gpuCtxDev=%p)",
       nSignals, nCounters, nContexts, (void*)signalsLocal, (void*)ctx->gpuCtxDev);
  free(gpuCtxHostArr);
  gpuCtxHostArr = nullptr;
  return ncclSuccess;

fail_signals:
  free(gpuCtxHostArr);
fail:
  if (ctx) {
    if (ctx->queuesDev) hipFree(ctx->queuesDev);
    if (ctx->signalsBaseDev) hipFree(ctx->signalsBaseDev);
    if (ctx->gpuCtxDev) hipFree(ctx->gpuCtxDev);
    if (ctx->devHandle) free(ctx->devHandle);
    free(ctx->queuesHost);
    free(ctx->signalsBaseHost);
    if (ctx->signalsAlloc) rocshmem::rocshmem_free(ctx->signalsAlloc);
    if (ctx->countersAlloc) hipFree(ctx->countersAlloc);
    free(ctx);
  } else {
    if (signalsLocal) rocshmem::rocshmem_free(signalsLocal);
    if (countersLocal) hipFree(countersLocal);
  }
  return ret;
}

ncclResult_t ncclGinAnvilDestroyContext(ncclGin_t*, void* ginCtx) {
  ginAnvilCtx* ctx = (ginAnvilCtx*)ginCtx;
  if (ctx == nullptr) return ncclSuccess;
  if (ctx->signalsAlloc) rocshmem::rocshmem_free(ctx->signalsAlloc);
  if (ctx->countersAlloc) hipFree(ctx->countersAlloc);
  if (ctx->queuesDev) hipFree(ctx->queuesDev);
  if (ctx->signalsBaseDev) hipFree(ctx->signalsBaseDev);
  if (ctx->gpuCtxDev) hipFree(ctx->gpuCtxDev);
  free(ctx->queuesHost);
  free(ctx->signalsBaseHost);
  free(ctx->devHandle);
  free(ctx);
  return ncclSuccess;
}

ncclResult_t ncclGinAnvilRegister(ncclGin_t*, void* ginCtx, void* addr, size_t size,
                                  int, int, void** mhandle, void** ginHandle) {
  ginAnvilCtx* ctx = (ginAnvilCtx*)ginCtx;
  ginAnvilMemHandle* mh = nullptr;
  NCCLCHECK(ncclCalloc(&mh, 1));
  mh->addr = addr;
  mh->size = size;

  // Resolve this address into LSA flat VA space; required for symmetric addressing.
  void* lsaSelfAddr = nullptr;
  NCCLCHECK(ncclDevrGetLsaSelfAddr(ctx->comm, addr, &lsaSelfAddr));
  if (lsaSelfAddr == nullptr) {
    WARN("GIN anvil: cannot resolve %p into LSA flat VA; unsupported", addr);
    free(mh);
    return ncclInvalidUsage;
  }
  struct ncclDevrState* devr = &ctx->comm->devrState;
  uintptr_t rank0Base = (uintptr_t)lsaSelfAddr - (uintptr_t)devr->lsaSelf * (uintptr_t)devr->bigSize;

  // Device handle contains symmetric base info.
  ncclGinAnvilMemHandle hostMh;
  hostMh.lsaRank0Base = rank0Base;
  hostMh.lsaStrideBytes = devr->bigSize;

  if (hipMalloc(&mh->devHandle, sizeof(ncclGinAnvilMemHandle)) != hipSuccess) {
    free(mh);
    return ncclSystemError;
  }
  if (hipMemcpy(mh->devHandle, &hostMh, sizeof(hostMh), hipMemcpyHostToDevice) != hipSuccess) {
    hipFree(mh->devHandle);
    free(mh);
    return ncclSystemError;
  }

  *mhandle = mh;
  *ginHandle = mh->devHandle;
  return ncclSuccess;
}

ncclResult_t ncclGinAnvilDeregister(ncclGin_t*, void*, void* mhandle) {
  ginAnvilMemHandle* mh = (ginAnvilMemHandle*)mhandle;
  if (mh == nullptr) return ncclSuccess;
  if (mh->devHandle) hipFree(mh->devHandle);
  free(mh);
  return ncclSuccess;
}

ncclResult_t ncclGinAnvilProgress(ncclGin_t*, void*) { return ncclSuccess; }

ncclResult_t ncclGinAnvilQueryLastError(ncclGin_t*, void* ginCtx, bool* hasError) {
  ginAnvilCtx* ctx = (ginAnvilCtx*)ginCtx;
  *hasError = ctx ? ctx->hasError : false;
  return ncclSuccess;
}

#endif // ENABLE_ROCSHMEM_GIN

