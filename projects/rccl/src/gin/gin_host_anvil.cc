/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifdef ENABLE_ROCSHMEM_GIN

#include "gin/gin_host_anvil.h"
#include "alloc.h"
#include "bootstrap.h"
#include "comm.h"
#include "dev_runtime.h"
#include "p2p.h"
#include "nccl_device/gin/anvil/gin_anvil_device_host_common.h"

// Anvil host API (src/sdma); include path from ROCSHMEM_SOURCE_DIR or mono-repo sibling.
#include "sdma/anvil.hpp"

#include <hip/hip_runtime.h>
#include <map>

// Match ncclCuMemAlloc / ncclP2pImportShareableBuffer granularity alignment.
static size_t ginAnvilAlignCuMemBytes(size_t bytes, int cudaDev) {
  CUmemAllocationProp prop = {};
  size_t granularity = 0;
  CUdevice dev;
  CUCHECK(cuDeviceGet(&dev, cudaDev));
#if defined(HIP_VMM_UNCACHED_MEMORY)
  prop.type = hipMemAllocationTypeUncached;
#else
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
#endif
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.requestedHandleTypes = ncclCuMemHandleType;
  prop.location.id = dev;
  CUCHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  ALIGN_SIZE(bytes, granularity);
  return bytes;
}

// SDMA signal atomics target the peer's native GPU VA. Grant each same-node peer
// READWRITE access on this rank's signal allocation (owner mapping only had local access).
static ncclResult_t ginAnvilGrantSignalPeerAccess(struct ncclComm* comm, void* signalsLocal,
                                                  size_t signalsBytes) {
  if (signalsLocal == nullptr || signalsBytes == 0) return ncclSuccess;
  for (int r = 0; r < comm->nRanks; r++) {
    if (r == comm->rank) continue;
    if (comm->rankToNode[r] != comm->rankToNode[comm->rank]) continue;
    int peerDev = comm->peerInfo[r].cudaDev;
    if (peerDev == comm->cudaDev) continue;
    CUmemAccessDesc accessDesc = {};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = peerDev;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CUCHECK(cuMemSetAccess((CUdeviceptr)signalsLocal, signalsBytes, &accessDesc, 1));
    INFO(NCCL_INIT | NCCL_NET,
         "GIN anvil: granted peer rank %d dev %d RW access to signals %p bytes %zu", r, peerDev,
         signalsLocal, signalsBytes);
  }
  return ncclSuccess;
}

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

  uint64_t* signalsAlloc;   // shareable cuMem base (all contexts)
  uint64_t* countersAlloc;   // hipMalloc base (local-only)
  ncclIpcDesc signalsIpcDesc;
  size_t signalsBytes;
  bool signalsShareable;

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

static void freeSignalBases(ginAnvilCtx* ctx) {
  if (ctx == nullptr || ctx->comm == nullptr) return;
  free(ctx->signalsBaseHost);
  ctx->signalsBaseHost = nullptr;
}

// Exchange cuMem handles and map peer signal allocs into this GPU (imported RW view).
// Remote signal updates use GPU atomics on the import; SDMA is used for data puts only.
struct ginAnvilSignalExport {
  ncclIpcDesc ipcDesc;
  void* directPtr;
};

static ncclResult_t setupSignalBases(ginAnvilCtx* ctx, struct ncclComm* comm, uint64_t* signalsLocal) {
  ncclResult_t ret = ncclSuccess;
  struct ginAnvilSignalExport* allExports = nullptr;

  NCCLCHECKGOTO(ncclCalloc(&ctx->signalsBaseHost, comm->nRanks), ret, fail);

  if (signalsLocal == nullptr) goto fail;

  ctx->signalsBaseHost[comm->rank] = signalsLocal;

  NCCLCHECKGOTO(ncclCalloc(&allExports, comm->nRanks), ret, fail);
  allExports[comm->rank].ipcDesc = ctx->signalsIpcDesc;
  allExports[comm->rank].directPtr = signalsLocal;
  NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, allExports, sizeof(struct ginAnvilSignalExport)), ret, fail);

  for (int r = 0; r < comm->nRanks; r++) {
    if (r == comm->rank) continue;
    if (comm->rankToNode[r] != comm->rankToNode[comm->rank]) {
      ctx->signalsBaseHost[r] = nullptr;
      continue;
    }
    void* mapped = nullptr;
    NCCLCHECKGOTO(ncclP2pImportShareableBuffer(comm, r, ctx->signalsBytes, &allExports[r].ipcDesc,
                                               &mapped, allExports[r].directPtr, ncclMemPersist),
                  ret, fail);
    ctx->signalsBaseHost[r] = (uint64_t*)mapped;
  }

  for (int r = 0; r < comm->nRanks; r++) {
    if (r == comm->rank || ctx->signalsBaseHost[r] == nullptr) continue;
    INFO(NCCL_INIT | NCCL_NET,
         "GIN anvil: signalsBase[rank %d] peer %d -> %p (import owner %p)", comm->rank, r,
         (void*)ctx->signalsBaseHost[r], allExports[r].directPtr);
  }

fail:
  free(allExports);
  if (ret != ncclSuccess) freeSignalBases(ctx);
  return ret;
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
  ctx->signalsShareable = false;
  ctx->signalsBytes = 0;
  memset(&ctx->signalsIpcDesc, 0, sizeof(ctx->signalsIpcDesc));

  // GIN_ANVIL uses MI300 xGMI SDMA and is valid only for single-node jobs.
  if (comm->nNodes > 1) {
    WARN("GIN anvil (NCCL_GIN_TYPE=5) requires a single-node communicator (nNodes=%d)", comm->nNodes);
    ret = ncclInvalidUsage;
    goto fail;
  }

  devr = &comm->devrState;
  NCCLCHECK(ncclDevrInitOnce(comm));

  if (nSignals > 0) {
    ctx->signalsBytes = (size_t)nSignals * (size_t)nContexts * sizeof(uint64_t);
    ctx->signalsBytes = ginAnvilAlignCuMemBytes(ctx->signalsBytes, comm->cudaDev);
    NCCLCHECKGOTO(ncclP2pAllocateShareableBuffer(ctx->signalsBytes, /*refcount=*/0, &ctx->signalsIpcDesc,
                                                 (void**)&signalsLocal, /*peerRank=*/-1, comm->memManager,
                                                 ncclMemPersist),
                  ret, fail);
    ctx->signalsAlloc = signalsLocal;
    ctx->signalsShareable = true;
    CUDACHECKGOTO(cudaMemset(signalsLocal, 0, ctx->signalsBytes), ret, fail);
    NCCLCHECKGOTO(ginAnvilGrantSignalPeerAccess(comm, signalsLocal, ctx->signalsBytes), ret, fail);
  }
  if (nCounters > 0) {
    size_t countersBytes = (size_t)nCounters * (size_t)nContexts * sizeof(uint64_t);
    if (hipMalloc((void**)&countersLocal, countersBytes) != hipSuccess) { ret = ncclSystemError; goto fail; }
    ctx->countersAlloc = countersLocal;
    hipMemset(countersLocal, 0, countersBytes);
  }

  for (int r = 0; r < comm->nRanks; r++) {
    if (r == comm->rank) continue;
    if (comm->rankToNode[r] != comm->rankToNode[comm->rank]) continue;
    int peerDev = comm->peerInfo[r].cudaDev;
    if (comm->cudaDev == peerDev) continue;
    rocshmem::anvil::EnablePeerAccess(comm->cudaDev, peerDev);
  }

  NCCLCHECKGOTO(setupSignalBases(ctx, comm, signalsLocal), ret, fail_signals);
  NCCLCHECKGOTO(ncclCalloc(&ctx->queuesHost, comm->nRanks), ret, fail_signals);

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

  NCCLCHECKGOTO(allocDeviceArray(&ctx->queuesDev, ctx->queuesHost, comm->nRanks), ret, fail_signals);
  if (hipMalloc(&ctx->signalsBaseDev, comm->nRanks * sizeof(uint64_t*)) != hipSuccess) {
    ret = ncclSystemError;
    goto fail_signals;
  }
  if (hipMemcpy(ctx->signalsBaseDev, ctx->signalsBaseHost, comm->nRanks * sizeof(uint64_t*),
                hipMemcpyHostToDevice) != hipSuccess) {
    ret = ncclSystemError;
    goto fail_signals;
  }

  NCCLCHECKGOTO(ncclCalloc(&gpuCtxHostArr, nContexts), ret, fail_signals);
  for (int contextId = 0; contextId < nContexts; contextId++) {
    ncclGinAnvilGPUContext* h = &gpuCtxHostArr[contextId];
    memset(h, 0, sizeof(*h));
    h->queues = ctx->queuesDev;
    h->signalsBase = ctx->signalsBaseDev;
    h->signals = signalsLocal ? signalsLocal + (size_t)contextId * (size_t)nSignals : nullptr;
    h->signalsContextOffset = (uint32_t)((size_t)contextId * (size_t)nSignals);
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
    freeSignalBases(ctx);
    if (ctx->signalsAlloc) {
      if (ctx->signalsShareable) NCCLCHECK(ncclCuMemFree(ctx->signalsAlloc, comm->memManager));
      else hipFree(ctx->signalsAlloc);
    }
    if (ctx->countersAlloc) hipFree(ctx->countersAlloc);
    free(ctx);
  } else {
    if (countersLocal) hipFree(countersLocal);
  }
  return ret;
}

ncclResult_t ncclGinAnvilDestroyContext(ncclGin_t*, void* ginCtx) {
  ginAnvilCtx* ctx = (ginAnvilCtx*)ginCtx;
  if (ctx == nullptr) return ncclSuccess;
  freeSignalBases(ctx);
  if (ctx->signalsAlloc) {
    if (ctx->signalsShareable) NCCLCHECK(ncclCuMemFree(ctx->signalsAlloc, ctx->comm->memManager));
    else hipFree(ctx->signalsAlloc);
  }
  if (ctx->countersAlloc) hipFree(ctx->countersAlloc);
  if (ctx->queuesDev) hipFree(ctx->queuesDev);
  if (ctx->signalsBaseDev) hipFree(ctx->signalsBaseDev);
  if (ctx->gpuCtxDev) hipFree(ctx->gpuCtxDev);
  free(ctx->queuesHost);
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

  void* lsaSelfAddr = nullptr;
  NCCLCHECK(ncclDevrGetLsaSelfAddr(ctx->comm, addr, &lsaSelfAddr));
  if (lsaSelfAddr == nullptr) {
    WARN("GIN anvil: cannot resolve %p into LSA flat VA; unsupported", addr);
    free(mh);
    return ncclInvalidUsage;
  }
  struct ncclDevrState* devr = &ctx->comm->devrState;
  uintptr_t rank0Base = (uintptr_t)lsaSelfAddr - (uintptr_t)devr->lsaSelf * (uintptr_t)devr->bigSize;

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
