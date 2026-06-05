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

  ncclGinAnvilGPUContext gpuCtxHost;
  ncclGinAnvilGPUContext* gpuCtxDev;

  // Device arrays
  void** queuesDev;         // [nRanks]
  uint64_t** signalsBaseDev;// [nRanks]

  // Host mirrors
  void** queuesHost;
  uint64_t** signalsBaseHost;

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
                                       int nSignals, int nCounters, void** outGinCtx,
                                       ncclNetDeviceHandle_v11_t** outDevHandle) {
  ncclResult_t ret = ncclSuccess;
  ginAnvilCtx* ctx = nullptr;
  uint64_t* signalsLocal = nullptr;
  uint64_t* countersLocal = nullptr;
  struct ncclDevrState* devr = nullptr;
  NCCLCHECK(ncclCalloc(&ctx, 1));
  ctx->comm = comm;
  ctx->collComm = collComm;
  ctx->hasError = false;

  // GIN_ANVIL uses MI300 xGMI SDMA and is valid only for single-node jobs.
  if (comm->nNodes > 1) {
    WARN("GIN anvil (NCCL_GIN_TYPE=5) requires a single-node communicator (nNodes=%d)", comm->nNodes);
    ret = ncclInvalidUsage;
    goto fail;
  }

  // Signals must live on the rocSHMEM symmetric heap so peer GPUs can SDMA-atomic
  // them (plain hipMalloc is not reliably peer-writable from SDMA).
  if (nSignals > 0) {
    signalsLocal = (uint64_t*)rocshmem::rocshmem_malloc(sizeof(uint64_t) * nSignals);
    if (!signalsLocal) {
      WARN("GIN anvil: rocshmem_malloc failed for %d signals", nSignals);
      ret = ncclSystemError;
      goto fail;
    }
    hipMemset(signalsLocal, 0, sizeof(uint64_t) * nSignals);
  }
  if (nCounters > 0) {
    if (hipMalloc(&countersLocal, sizeof(uint64_t) * nCounters) != hipSuccess) { ret = ncclSystemError; goto fail; }
    hipMemset(countersLocal, 0, sizeof(uint64_t) * nCounters);
  }

  // Exchange the per-rank signal base pointers inside the LSA team (single node assumption).
  // We rely on symmetric ranks being in the same node subset (devr->lsaRankList).
  devr = &comm->devrState;
  // Ensure symmetric runtime is initialized (for lsaRankList/bigSize fields).
  NCCLCHECK(ncclDevrInitOnce(comm));

  // Anvil SDMA uses rocSHMEM host runtime (initEndpoint); ensure it is ready even
  // when librccl is built GIN-only and the test binary owns rocshmem symbols.
  NCCLCHECK(ncclGinRocshmemEnsureInit(comm));

  // For LSA flat VA accesses, expose signals in that space as well by mapping them
  // into symmetric memory is non-trivial. For now, we require signals to live in
  // LSA flat space by allocating them from the symmetric resource window.
  // If not available, we fall back to exchanging raw pointers (requires peer access).
  // Enable peer access on all GPUs in the node.
  for (int r = 0; r < comm->nRanks; r++) {
    if (r == comm->rank) continue;
    if (comm->rankToNode[r] != comm->rankToNode[comm->rank]) continue;
    int peerDev = comm->peerInfo[r].cudaDev;
    if (comm->cudaDev == peerDev) continue;
    rocshmem::anvil::EnablePeerAccess(comm->cudaDev, peerDev);
  }

  // Gather pointers across all ranks (world). For off-node ranks, entries are nullptr.
  NCCLCHECKGOTO(ncclCalloc(&ctx->signalsBaseHost, comm->nRanks), ret, fail_signals);
  ctx->signalsBaseHost[comm->rank] = signalsLocal;
  NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, ctx->signalsBaseHost, sizeof(uint64_t*)), ret, fail_signals);

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
    ctx->queuesHost[r] = (void*)q->deviceHandle();
  }

  // Allocate device arrays for queues and peer signal bases.
  NCCLCHECKGOTO(allocDeviceArray(&ctx->queuesDev, ctx->queuesHost, comm->nRanks), ret, fail_signals);
  if (hipMalloc(&ctx->signalsBaseDev, comm->nRanks * sizeof(uint64_t*)) != hipSuccess) { ret = ncclSystemError; goto fail_signals; }
  if (hipMemcpy(ctx->signalsBaseDev, ctx->signalsBaseHost, comm->nRanks * sizeof(uint64_t*), hipMemcpyHostToDevice) != hipSuccess) { ret = ncclSystemError; goto fail_signals; }

  // Build GPU context and copy to device.
  memset(&ctx->gpuCtxHost, 0, sizeof(ctx->gpuCtxHost));
  ctx->gpuCtxHost.queues = ctx->queuesDev;
  ctx->gpuCtxHost.signalsBase = ctx->signalsBaseDev;
  ctx->gpuCtxHost.signals = signalsLocal;
  ctx->gpuCtxHost.counters = countersLocal;
  ctx->gpuCtxHost.nSignals = nSignals;
  ctx->gpuCtxHost.nCounters = nCounters;
  ctx->gpuCtxHost.nRanks = comm->nRanks;
  ctx->gpuCtxHost.rank = comm->rank;
  ctx->gpuCtxHost.myNode = comm->rankToNode[comm->rank];
  ctx->gpuCtxHost.lsaStrideBytes = devr->bigSize;
  // lsaRank0Base is per-window; kept here only for debugging.
  ctx->gpuCtxHost.lsaRank0Base = (uintptr_t)devr->lsaFlatBase;

  if (hipMalloc(&ctx->gpuCtxDev, sizeof(ncclGinAnvilGPUContext)) != hipSuccess) { ret = ncclSystemError; goto fail_signals; }
  if (hipMemcpy(ctx->gpuCtxDev, &ctx->gpuCtxHost, sizeof(ctx->gpuCtxHost), hipMemcpyHostToDevice) != hipSuccess) { ret = ncclSystemError; goto fail_signals; }

  // Create net device handle
  NCCLCHECK(ncclCalloc(&ctx->devHandle, 1));
  ctx->devHandle->netDeviceType = NCCL_NET_DEVICE_GIN_ANVIL;
  ctx->devHandle->netDeviceVersion = NCCL_GIN_ANVIL_VERSION;
  ctx->devHandle->needsProxyProgress = 0;
  ctx->devHandle->handle = ctx->gpuCtxDev;
  ctx->devHandle->size = sizeof(ncclGinAnvilGPUContext);

  *outGinCtx = ctx;
  *outDevHandle = ctx->devHandle;
  INFO(NCCL_INIT, "GIN anvil: context created (%d signals, %d counters)", nSignals, nCounters);
  return ncclSuccess;

fail_signals:
fail:
  if (ctx) {
    if (ctx->queuesDev) hipFree(ctx->queuesDev);
    if (ctx->signalsBaseDev) hipFree(ctx->signalsBaseDev);
    if (ctx->gpuCtxDev) hipFree(ctx->gpuCtxDev);
    if (ctx->devHandle) free(ctx->devHandle);
    free(ctx->queuesHost);
    free(ctx->signalsBaseHost);
    free(ctx);
  }
  if (signalsLocal) rocshmem::rocshmem_free(signalsLocal);
  if (countersLocal) hipFree(countersLocal);
  return ret;
}

ncclResult_t ncclGinAnvilDestroyContext(ncclGin_t*, void* ginCtx) {
  ginAnvilCtx* ctx = (ginAnvilCtx*)ginCtx;
  if (ctx == nullptr) return ncclSuccess;
  if (ctx->gpuCtxDev) {
    ncclGinAnvilGPUContext hostCtx;
    if (hipMemcpy(&hostCtx, ctx->gpuCtxDev, sizeof(hostCtx), hipMemcpyDeviceToHost) == hipSuccess) {
      if (hostCtx.signals) rocshmem::rocshmem_free(hostCtx.signals);
      if (hostCtx.counters) hipFree(hostCtx.counters);
    }
  }
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

