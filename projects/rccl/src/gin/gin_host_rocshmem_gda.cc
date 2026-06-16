/*************************************************************************
 * Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifdef ENABLE_ROCSHMEM_GIN

#include "gin/gin_host_rocshmem_gda.h"
#include "comm.h"
#include "bootstrap.h"
#include "nccl_device/gin/rocshmem_gda/gin_rocshmem_device_host_common_gda.h"

#include "gin/gin_qp_factory.h"
#include <hip/hip_runtime.h>

// Host-side context that wraps the GPU context and IB resources
struct ginRocshmemCtx {
  struct ncclComm *comm;
  void *collComm;
  ncclNetDeviceHandle_v11_t *devHandle;

  // GPU-side context (allocated via hipMalloc)
  ncclGinRocshmemGdaGPUContext *gpuCtxDev;
  // Host-side copy for setup
  ncclGinRocshmemGdaGPUContext gpuCtxHost;

  int nRanks;
  int rank;
  int nSignals;
  int nCounters;

  bool hasError;

  // QP set from gin_qp_factory (owns IB resources)
  rocshmem_gin_qp_set_t qpSet;

  // MRs for signal/counter buffers
  void *signalMr;
  void *counterMr;
};

// Host-side memory handle that wraps the GPU handle
struct ginRocshmemMemHandle {
  ncclGinRocshmemGdaMemHandle *devHandle;  // GPU-side handle
  void *mr;                              // ibv_mr from gin_qp_factory
  uint32_t *rkeys_dev;                   // GPU array of per-peer rkeys
  uintptr_t *remote_vas_dev;                    // GPU array of per-peer base VAs
};

// Bootstrap allgather wrapper for gin_qp_factory callback
struct ginBootstrapCtx {
  struct ncclComm *comm;
};

static int ginBootstrapAllgather(void *ctx, void *buf, size_t perRankSize) {
  struct ginBootstrapCtx *bctx = (struct ginBootstrapCtx *)ctx;
  ncclResult_t ret = bootstrapAllGather(bctx->comm->bootstrap, buf, perRankSize);
  return (ret == ncclSuccess) ? 0 : -1;
}

ncclResult_t ncclGinRocshmemGdaCreateContext(struct ncclComm *comm, void *collComm, int devId,
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
  ctx->qpSet = nullptr;
  ctx->signalMr = nullptr;
  ctx->counterMr = nullptr;
  // Allocate device handle
  NCCLCHECK(ncclCalloc(&ctx->devHandle, 1));
  ctx->devHandle->netDeviceType = NCCL_NET_DEVICE_GIN_ROCSHMEM_GDA;
  ctx->devHandle->netDeviceVersion = NCCL_GIN_ROCSHMEM_VERSION;
  ctx->devHandle->needsProxyProgress = 0;

  // Allocate GPU context
  if (hipMalloc(&ctx->gpuCtxDev, sizeof(ncclGinRocshmemGdaGPUContext)) != hipSuccess) {
    WARN("GIN rocshmem: failed to allocate GPU context");
    ret = ncclSystemError;
    goto fail;
  }

  // Initialize host copy of GPU context
  memset(&ctx->gpuCtxHost, 0, sizeof(ncclGinRocshmemGdaGPUContext));
  ctx->gpuCtxHost.nRanks = ctx->nRanks;
  ctx->gpuCtxHost.rank = ctx->rank;
  ctx->gpuCtxHost.nSignals = nSignals;
  ctx->gpuCtxHost.nCounters = nCounters;

  // Create QPs via gin_qp_factory
  {
    struct ginBootstrapCtx bctx = { comm };
    void **gpu_qp_ptrs = nullptr;
    int rc = rocshmem_gin_create_qps(ctx->nRanks, ctx->rank,
                                      ginBootstrapAllgather, &bctx,
                                      &ctx->qpSet, &gpu_qp_ptrs);
    if (rc != 0) {
      WARN("GIN rocshmem: failed to create QPs via gin_qp_factory");
      ret = ncclSystemError;
      goto fail;
    }
    // gpu_qp_ptrs is a GPU array of QueuePair* — store in GPU context
    ctx->gpuCtxHost.qps = (rocshmem::QueuePair**)gpu_qp_ptrs;

    // Initialize rocshmem __constant__ device memory with the detected provider.
    // Initialize rocshmem __constant__ device memory (constmem + logd).
    // Implementation lives in librocshmem.a; resolved via -rdynamic.
    rocshmem_gin_init_constmem(rocshmem_gin_get_provider(ctx->qpSet), ctx->rank);
  }

  // Allocate signal and counter arrays on GPU
  if (nSignals > 0) {
    if (hipExtMallocWithFlags((void**)&ctx->gpuCtxHost.signals, sizeof(uint64_t) * nSignals, hipDeviceMallocFinegrained) != hipSuccess) {
      ret = ncclSystemError;
      goto fail;
    }
    if (hipMemset(ctx->gpuCtxHost.signals, 0, sizeof(uint64_t) * nSignals) != hipSuccess) {
      ret = ncclSystemError;
      goto fail;
    }

    // Register signal buffer for RDMA atomic access
    uint32_t sigLkey, sigRkey;
    if (rocshmem_gin_reg_mr(ctx->qpSet, ctx->gpuCtxHost.signals,
                             sizeof(uint64_t) * nSignals, /*atomic=*/1,
                             &ctx->signalMr, &sigLkey, &sigRkey) != 0) {
      WARN("GIN rocshmem: failed to register signal buffer MR");
      ret = ncclSystemError;
      goto fail;
    }

    // Exchange signal rkeys and base addresses via bootstrap
    if (hipMalloc(&ctx->gpuCtxHost.signal_rkeys, sizeof(uint32_t) * ctx->nRanks) != hipSuccess ||
        hipMalloc(&ctx->gpuCtxHost.signal_raddrs, sizeof(uintptr_t) * ctx->nRanks) != hipSuccess) {
      ret = ncclSystemError;
      goto fail;
    }

    {
      // Allgather signal rkeys
      uint32_t *rkeys_buf = (uint32_t *)malloc(sizeof(uint32_t) * ctx->nRanks);
      uintptr_t *raddrs_buf = (uintptr_t *)malloc(sizeof(uintptr_t) * ctx->nRanks);
      rkeys_buf[ctx->rank] = sigRkey;
      raddrs_buf[ctx->rank] = (uintptr_t)ctx->gpuCtxHost.signals;

      bootstrapAllGather(comm->bootstrap, rkeys_buf, sizeof(uint32_t));
      bootstrapAllGather(comm->bootstrap, raddrs_buf, sizeof(uintptr_t));

      if (hipMemcpy(ctx->gpuCtxHost.signal_rkeys, rkeys_buf,
                    sizeof(uint32_t) * ctx->nRanks, hipMemcpyHostToDevice) != hipSuccess ||
          hipMemcpy(ctx->gpuCtxHost.signal_raddrs, raddrs_buf,
                    sizeof(uintptr_t) * ctx->nRanks, hipMemcpyHostToDevice) != hipSuccess) {
        free(rkeys_buf);
        free(raddrs_buf);
        ret = ncclSystemError;
        goto fail;
      }

      free(rkeys_buf);
      free(raddrs_buf);
    }
  }

  if (nCounters > 0) {
    if (hipExtMallocWithFlags((void**)&ctx->gpuCtxHost.counters, sizeof(uint64_t) * nCounters, hipDeviceMallocFinegrained) != hipSuccess) {
      ret = ncclSystemError;
      goto fail;
    }
    if (hipMemset(ctx->gpuCtxHost.counters, 0, sizeof(uint64_t) * nCounters) != hipSuccess) {
      ret = ncclSystemError;
      goto fail;
    }
  }

  // Copy GPU context to device
  if (hipMemcpy(ctx->gpuCtxDev, &ctx->gpuCtxHost, sizeof(ncclGinRocshmemGdaGPUContext),
                hipMemcpyHostToDevice) != hipSuccess) {
    ret = ncclSystemError;
    goto fail;
  }

  ctx->devHandle->handle = ctx->gpuCtxDev;
  ctx->devHandle->size = sizeof(ncclGinRocshmemGdaGPUContext);

  *outGinCtx = ctx;
  *outDevHandle = ctx->devHandle;
  INFO(NCCL_INIT, "GIN rocshmem: context created with %d QPs, %d signals, %d counters",
       ctx->nRanks, nSignals, nCounters);
  return ncclSuccess;

fail:
  if (ctx) {
    if (ctx->signalMr) rocshmem_gin_dereg_mr(ctx->signalMr);
    if (ctx->qpSet) rocshmem_gin_destroy_qps(ctx->qpSet);
    if (ctx->gpuCtxHost.signals) (void)hipFree(ctx->gpuCtxHost.signals);
    if (ctx->gpuCtxHost.counters) (void)hipFree(ctx->gpuCtxHost.counters);
    if (ctx->gpuCtxHost.signal_rkeys) (void)hipFree(ctx->gpuCtxHost.signal_rkeys);
    if (ctx->gpuCtxHost.signal_raddrs) (void)hipFree(ctx->gpuCtxHost.signal_raddrs);
    if (ctx->gpuCtxDev) (void)hipFree(ctx->gpuCtxDev);
    free(ctx->devHandle);
    free(ctx);
  }
  return ret;
}

ncclResult_t ncclGinRocshmemGdaDestroyContext(ncclGin_t *ginComm, void *ginCtx) {
  struct ginRocshmemCtx *ctx = (struct ginRocshmemCtx *)ginCtx;
  if (ctx == NULL) return ncclSuccess;

  if (ctx->signalMr) rocshmem_gin_dereg_mr(ctx->signalMr);
  if (ctx->counterMr) rocshmem_gin_dereg_mr(ctx->counterMr);
  if (ctx->qpSet) rocshmem_gin_destroy_qps(ctx->qpSet);

  if (ctx->gpuCtxHost.signals) (void)hipFree(ctx->gpuCtxHost.signals);
  if (ctx->gpuCtxHost.counters) (void)hipFree(ctx->gpuCtxHost.counters);
  if (ctx->gpuCtxHost.signal_rkeys) (void)hipFree(ctx->gpuCtxHost.signal_rkeys);
  if (ctx->gpuCtxHost.signal_raddrs) (void)hipFree(ctx->gpuCtxHost.signal_raddrs);
  if (ctx->gpuCtxDev) (void)hipFree(ctx->gpuCtxDev);
  free(ctx->devHandle);
  free(ctx);
  return ncclSuccess;
}

ncclResult_t ncclGinRocshmemGdaRegister(ncclGin_t *ginComm, void *ginCtx, void *addr, size_t size,
                                     int type, int mr_flags, void **mhandle, void **ginHandle) {
  struct ginRocshmemCtx *ctx = (struct ginRocshmemCtx *)ginCtx;
  struct ginRocshmemMemHandle *mh = NULL;

  NCCLCHECK(ncclCalloc(&mh, 1));

  // Allocate GPU-side mem handle
  if (hipMalloc(&mh->devHandle, sizeof(ncclGinRocshmemGdaMemHandle)) != hipSuccess) {
    free(mh);
    return ncclSystemError;
  }

  // Register buffer with iova=0 — remote addr in WQE is offset, not VA
  uint32_t lkey, rkey;
  if (rocshmem_gin_reg_mr_vmm(ctx->qpSet, addr, size, /*atomic=*/0,
                                 &mh->mr, &lkey, &rkey) != 0) {
    WARN("GIN rocshmem GDA: MR registration failed for buffer %p size %zu", addr, size);
    (void)hipFree(mh->devHandle);
    free(mh);
    return ncclSystemError;
  }

  INFO(NCCL_INIT, "GIN rocshmem GDA Register: rank=%d addr=%p size=%zu lkey=0x%x rkey=0x%x",
       ctx->rank, addr, size, lkey, rkey);

  // Allgather rkeys and base VAs across all peers
  uint32_t *rkeys_buf = (uint32_t *)malloc(sizeof(uint32_t) * ctx->nRanks);
  uintptr_t *vas_buf = (uintptr_t *)malloc(sizeof(uintptr_t) * ctx->nRanks);
  rkeys_buf[ctx->rank] = rkey;
  vas_buf[ctx->rank] = (uintptr_t)addr;
  bootstrapAllGather(ctx->comm->bootstrap, rkeys_buf, sizeof(uint32_t));
  bootstrapAllGather(ctx->comm->bootstrap, vas_buf, sizeof(uintptr_t));

  for (int i = 0; i < ctx->nRanks; i++) {
    INFO(NCCL_INIT, "GIN rocshmem GDA Register: rank=%d peer=%d rkey=0x%x va=%p",
         ctx->rank, i, rkeys_buf[i], (void*)vas_buf[i]);
  }

  uint32_t *rkeys_dev = nullptr;
  uintptr_t *remote_vas_dev = nullptr;
  if (hipMalloc(&rkeys_dev, sizeof(uint32_t) * ctx->nRanks) != hipSuccess ||
      hipMalloc(&remote_vas_dev, sizeof(uintptr_t) * ctx->nRanks) != hipSuccess) {
    free(rkeys_buf);
    free(vas_buf);
    (void)hipFree(rkeys_dev);
    rocshmem_gin_dereg_mr(mh->mr);
    (void)hipFree(mh->devHandle);
    free(mh);
    return ncclSystemError;
  }
  if (hipMemcpy(rkeys_dev, rkeys_buf, sizeof(uint32_t) * ctx->nRanks, hipMemcpyHostToDevice) != hipSuccess ||
      hipMemcpy(remote_vas_dev, vas_buf, sizeof(uintptr_t) * ctx->nRanks, hipMemcpyHostToDevice) != hipSuccess) {
    free(rkeys_buf);
    free(vas_buf);
    (void)hipFree(rkeys_dev);
    (void)hipFree(remote_vas_dev);
    rocshmem_gin_dereg_mr(mh->mr);
    (void)hipFree(mh->devHandle);
    free(mh);
    return ncclSystemError;
  }
  free(rkeys_buf);
  free(vas_buf);

  // Populate host copy of mem handle
  ncclGinRocshmemGdaMemHandle hostMh;
  hostMh.local_va = (uintptr_t)addr;
  hostMh.remote_vas = remote_vas_dev;
  hostMh.lkey = lkey;
  hostMh.rkeys = rkeys_dev;

  // Copy to device
  if (hipMemcpy(mh->devHandle, &hostMh, sizeof(ncclGinRocshmemGdaMemHandle),
                hipMemcpyHostToDevice) != hipSuccess) {
    (void)hipFree(rkeys_dev);
    rocshmem_gin_dereg_mr(mh->mr);
    (void)hipFree(mh->devHandle);
    free(mh);
    return ncclSystemError;
  }

  mh->rkeys_dev = rkeys_dev;
  mh->remote_vas_dev = remote_vas_dev;
  *mhandle = mh;
  *ginHandle = mh->devHandle;
  return ncclSuccess;
}

ncclResult_t ncclGinRocshmemGdaDeregister(ncclGin_t *ginComm, void *ginCtx, void *mhandle) {
  struct ginRocshmemMemHandle *mh = (struct ginRocshmemMemHandle *)mhandle;
  if (mh == NULL) return ncclSuccess;

  if (mh->rkeys_dev) (void)hipFree(mh->rkeys_dev);
  if (mh->remote_vas_dev) (void)hipFree(mh->remote_vas_dev);
  if (mh->mr) rocshmem_gin_dereg_mr(mh->mr);
  if (mh->devHandle) (void)hipFree(mh->devHandle);
  free(mh);
  return ncclSuccess;
}

ncclResult_t ncclGinRocshmemGdaProgress(ncclGin_t *ginComm, void *ginCtx) {
  // No-op: GPU-direct backend, no CPU progress needed
  return ncclSuccess;
}

ncclResult_t ncclGinRocshmemGdaQueryLastError(ncclGin_t *ginComm, void *ginCtx, bool *hasError) {
  struct ginRocshmemCtx *ctx = (struct ginRocshmemCtx *)ginCtx;
  *hasError = ctx ? ctx->hasError : false;
  return ncclSuccess;
}

///////////////////////////////////////////////////////////////////////////////
// Plugin-facing variants (no ncclComm available)
///////////////////////////////////////////////////////////////////////////////

ncclResult_t ncclGinRocshmemGdaCreateContextFromPlugin(int nSignals, int nCounters,
                                                       void **outGinCtx,
                                                       ncclNetDeviceHandle_v11_t **outDevHandle) {
  return ncclGinRocshmemGdaCreateContext(NULL, NULL, 0, nSignals, nCounters, outGinCtx, outDevHandle);
}

ncclResult_t ncclGinRocshmemGdaRegisterFromPlugin(void *addr, size_t size, int type,
                                                  uint64_t mr_flags, void **mhandle, void **ginHandle) {
  return ncclGinRocshmemGdaRegister(NULL, NULL, addr, size, type, (int)mr_flags, mhandle, ginHandle);
}

ncclResult_t ncclGinRocshmemGdaDeregisterFromPlugin(void *mhandle) {
  return ncclGinRocshmemGdaDeregister(NULL, NULL, mhandle);
}

ncclResult_t ncclGinRocshmemGdaDestroyContextFromPlugin(void *ginCtx) {
  return ncclGinRocshmemGdaDestroyContext(NULL, ginCtx);
}

#endif // ENABLE_ROCSHMEM_GIN
