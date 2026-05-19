/*************************************************************************
 * Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifdef ENABLE_ROCSHMEM

#include "gin/gin_host_rocshmem.h"
#include "comm.h"
#include "nccl_device/gin/rocshmem/gin_rocshmem_device_host_common.h"

#include <hip/hip_runtime.h>

// Host-side context that wraps the GPU context and IB resources
struct ginRocshmemCtx {
  struct ncclComm *comm;
  void *collComm;
  ncclNetDeviceHandle_v11_t *devHandle;

  // GPU-side context (allocated via hipMalloc)
  ncclGinRocshmemGPUContext *gpuCtxDev;
  // Host-side copy for setup
  ncclGinRocshmemGPUContext gpuCtxHost;

  int nRanks;
  int rank;
  int nSignals;
  int nCounters;

  bool hasError;

  // TODO: IB resources (PD, QPs, MRs) via rocshmem gin_qp_factory
};

// Host-side memory handle that wraps the GPU handle
struct ginRocshmemMemHandle {
  ncclGinRocshmemMemHandle *devHandle;  // GPU-side handle
  // TODO: ibv_mr, host-side state
};

ncclResult_t ncclGinRocshmemCreateContext(struct ncclComm *comm, void *collComm, int devId,
                                          int nSignals, int nCounters, void **outGinCtx,
                                          ncclNetDeviceHandle_v11_t **outDevHandle) {
  ncclResult_t ret = ncclSuccess;
  struct ginRocshmemCtx *ctx = NULL;

  NCCLCHECK(ncclCalloc(&ctx, 1));
  ctx->comm = comm;
  ctx->collComm = collComm;
  ctx->nRanks = comm->nRanks;
  ctx->rank = comm->rank;
  ctx->nSignals = nSignals;
  ctx->nCounters = nCounters;
  ctx->hasError = false;

  // Allocate device handle
  NCCLCHECK(ncclCalloc(&ctx->devHandle, 1));
  ctx->devHandle->netDeviceType = NCCL_NET_DEVICE_GIN_ROCSHMEM;
  ctx->devHandle->netDeviceVersion = NCCL_GIN_ROCSHMEM_VERSION;
  ctx->devHandle->needsProxyProgress = 0;  // GPU-direct, no CPU proxy needed

  // Allocate GPU context
  if (hipMalloc(&ctx->gpuCtxDev, sizeof(ncclGinRocshmemGPUContext)) != hipSuccess) {
    WARN("GIN rocshmem: failed to allocate GPU context");
    ret = ncclSystemError;
    goto fail;
  }

  // Initialize host copy of GPU context
  memset(&ctx->gpuCtxHost, 0, sizeof(ncclGinRocshmemGPUContext));
  ctx->gpuCtxHost.nRanks = ctx->nRanks;
  ctx->gpuCtxHost.rank = ctx->rank;
  ctx->gpuCtxHost.nSignals = nSignals;
  ctx->gpuCtxHost.nCounters = nCounters;

  // Allocate signal and counter arrays on GPU
  if (nSignals > 0) {
    if (hipMalloc(&ctx->gpuCtxHost.signals, sizeof(uint64_t) * nSignals) != hipSuccess) {
      ret = ncclSystemError;
      goto fail;
    }
    hipMemset(ctx->gpuCtxHost.signals, 0, sizeof(uint64_t) * nSignals);
  }
  if (nCounters > 0) {
    if (hipMalloc(&ctx->gpuCtxHost.counters, sizeof(uint64_t) * nCounters) != hipSuccess) {
      ret = ncclSystemError;
      goto fail;
    }
    hipMemset(ctx->gpuCtxHost.counters, 0, sizeof(uint64_t) * nCounters);
  }

  // Allocate per-peer signal remote address/key arrays
  if (hipMalloc(&ctx->gpuCtxHost.signal_rkeys, sizeof(uint32_t) * ctx->nRanks) != hipSuccess ||
      hipMalloc(&ctx->gpuCtxHost.signal_raddrs, sizeof(uintptr_t) * ctx->nRanks) != hipSuccess) {
    ret = ncclSystemError;
    goto fail;
  }

  // Allocate putValue staging buffer (8 bytes)
  if (hipMalloc(&ctx->gpuCtxHost.putValueStagingBuf, 8) != hipSuccess) {
    ret = ncclSystemError;
    goto fail;
  }

  // TODO: Create QPs via rocshmem gin_qp_factory
  // TODO: Connect QPs (exchange dest_info via bootstrap)
  // TODO: Allocate and register signal MRs with IBV_ACCESS_REMOTE_ATOMIC
  // TODO: Exchange signal rkeys and base addresses via bootstrap allgather
  // TODO: Register putValue staging buffer MR, set putValueStagingLkey

  // Copy GPU context to device
  if (hipMemcpy(ctx->gpuCtxDev, &ctx->gpuCtxHost, sizeof(ncclGinRocshmemGPUContext),
                hipMemcpyHostToDevice) != hipSuccess) {
    ret = ncclSystemError;
    goto fail;
  }

  ctx->devHandle->handle = ctx->gpuCtxDev;
  ctx->devHandle->size = sizeof(ncclGinRocshmemGPUContext);

  *outGinCtx = ctx;
  *outDevHandle = ctx->devHandle;
  return ncclSuccess;

fail:
  if (ctx) {
    if (ctx->gpuCtxHost.signals) hipFree(ctx->gpuCtxHost.signals);
    if (ctx->gpuCtxHost.counters) hipFree(ctx->gpuCtxHost.counters);
    if (ctx->gpuCtxHost.signal_rkeys) hipFree(ctx->gpuCtxHost.signal_rkeys);
    if (ctx->gpuCtxHost.signal_raddrs) hipFree(ctx->gpuCtxHost.signal_raddrs);
    if (ctx->gpuCtxHost.putValueStagingBuf) hipFree(ctx->gpuCtxHost.putValueStagingBuf);
    if (ctx->gpuCtxDev) hipFree(ctx->gpuCtxDev);
    free(ctx->devHandle);
    free(ctx);
  }
  return ret;
}

ncclResult_t ncclGinRocshmemDestroyContext(ncclGin_t *ginComm, void *ginCtx) {
  struct ginRocshmemCtx *ctx = (struct ginRocshmemCtx *)ginCtx;
  if (ctx == NULL) return ncclSuccess;

  // TODO: Destroy QPs via rocshmem gin_qp_factory
  // TODO: Deregister signal/counter/staging MRs

  if (ctx->gpuCtxHost.signals) hipFree(ctx->gpuCtxHost.signals);
  if (ctx->gpuCtxHost.counters) hipFree(ctx->gpuCtxHost.counters);
  if (ctx->gpuCtxHost.signal_rkeys) hipFree(ctx->gpuCtxHost.signal_rkeys);
  if (ctx->gpuCtxHost.signal_raddrs) hipFree(ctx->gpuCtxHost.signal_raddrs);
  if (ctx->gpuCtxHost.putValueStagingBuf) hipFree(ctx->gpuCtxHost.putValueStagingBuf);
  if (ctx->gpuCtxDev) hipFree(ctx->gpuCtxDev);
  free(ctx->devHandle);
  free(ctx);
  return ncclSuccess;
}

ncclResult_t ncclGinRocshmemRegister(ncclGin_t *ginComm, void *ginCtx, void *addr, size_t size,
                                     int type, int mr_flags, void **mhandle, void **ginHandle) {
  struct ginRocshmemCtx *ctx = (struct ginRocshmemCtx *)ginCtx;
  struct ginRocshmemMemHandle *mh = NULL;

  NCCLCHECK(ncclCalloc(&mh, 1));

  // Allocate GPU-side mem handle
  if (hipMalloc(&mh->devHandle, sizeof(ncclGinRocshmemMemHandle)) != hipSuccess) {
    free(mh);
    return ncclSystemError;
  }

  // Populate host copy of mem handle
  ncclGinRocshmemMemHandle hostMh;
  hostMh.baseAddr = (uintptr_t)addr;

  // TODO: Register buffer with ibv_reg_mr
  // TODO: Exchange rkeys via bootstrap allgather
  // TODO: Set hostMh.lkey and hostMh.rkey from MR
  hostMh.lkey = 0;  // placeholder
  hostMh.rkey = 0;  // placeholder

  // Copy to device
  if (hipMemcpy(mh->devHandle, &hostMh, sizeof(ncclGinRocshmemMemHandle),
                hipMemcpyHostToDevice) != hipSuccess) {
    hipFree(mh->devHandle);
    free(mh);
    return ncclSystemError;
  }

  *mhandle = mh;
  *ginHandle = mh->devHandle;
  return ncclSuccess;
}

ncclResult_t ncclGinRocshmemDeregister(ncclGin_t *ginComm, void *ginCtx, void *mhandle) {
  struct ginRocshmemMemHandle *mh = (struct ginRocshmemMemHandle *)mhandle;
  if (mh == NULL) return ncclSuccess;

  // TODO: Deregister ibv_mr

  if (mh->devHandle) hipFree(mh->devHandle);
  free(mh);
  return ncclSuccess;
}

ncclResult_t ncclGinRocshmemProgress(ncclGin_t *ginComm, void *ginCtx) {
  // No-op: GPU-direct backend, no CPU progress needed
  return ncclSuccess;
}

ncclResult_t ncclGinRocshmemQueryLastError(ncclGin_t *ginComm, void *ginCtx, bool *hasError) {
  struct ginRocshmemCtx *ctx = (struct ginRocshmemCtx *)ginCtx;
  *hasError = ctx ? ctx->hasError : false;
  return ncclSuccess;
}

#endif // ENABLE_ROCSHMEM
