/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifdef ENABLE_ROCSHMEM_GIN

/**
 * Built-in GIN plugin for the rocshmem GDA (QueuePair) backend.
 *
 * Follows upstream vtable pattern like GDAKI:
 * - connect() creates QPs and stores qpSet in collComm
 * - regMrSym(collComm) uses qpSet for IB MR registration
 * - createContext(collComm) sets up signals, counters, GPU context
 */

#include "gin/gin_host_rocshmem_gda.h"  // ginRocshmemInitCtx, SetInitContext
#include "comm.h"
#include "bootstrap.h"
#include "nccl_device/gin/rocshmem_gda/gin_rocshmem_device_host_common_gda.h"
#include "gin/gin_rocshmem_gda_factory.h"
#include <hip/hip_runtime.h>

// collComm: per-connection state, returned by connect()
struct ginRocshmemGdaCollCtx {
  int nranks;
  int rank;
  struct ncclComm *comm;
  rocshmem_gin_qp_set_t qpSet;
  void **gpu_qp_ptrs;
};

// ginCtx: per-context state, returned by createContext()
struct ginRocshmemGdaGinCtx {
  ncclNetDeviceHandle_v11_t *devHandle;
  ncclGinRocshmemGdaGPUContext *gpuCtxDev;
  ncclGinRocshmemGdaGPUContext gpuCtxHost;
  int nRanks;
  int rank;
  int nSignals;
  int nCounters;
  bool hasError;
  void *signalMr;
  void *counterMr;
  rocshmem_gin_qp_set_t qpSet;  // borrowed from collComm, not owned
};

// Host-side memory handle
struct ginRocshmemGdaMemHandle {
  ncclGinRocshmemGdaMemHandle *devHandle;
  void *mr;
  uint32_t *rkeys_dev;
  uintptr_t *remote_vas_dev;
};

struct ginRocshmemGdaListenCtx {
  int dev;
};

// Bootstrap allgather wrapper
static int ginGdaBootstrapAllgather(void *ctx, void *buf, size_t perRankSize) {
  return (bootstrapAllGather(ctx, buf, perRankSize) == ncclSuccess) ? 0 : -1;
}

///////////////////////////////////////////////////////////////////////////////
// Vtable implementations
///////////////////////////////////////////////////////////////////////////////

static ncclResult_t ginRocshmemGdaInit(void** ctx, uint64_t commId, ncclDebugLogger_t logFunction) {
  const char *gin_type = getenv("NCCL_GIN_TYPE");
  if (!gin_type || atoi(gin_type) != NCCL_NET_DEVICE_GIN_ROCSHMEM_GDA)
    return ncclInternalError;
  if (rocshmem_gin_probe_devices() <= 0)
    return ncclInternalError;
  // Allocate init context; bootstrap filled by ncclGinRocshmemSetInitContext
  struct ginRocshmemInitCtx *ictx = new ginRocshmemInitCtx{};
  *ctx = ictx;
  return ncclSuccess;
}

static ncclResult_t ginRocshmemGdaDevices(int* ndev) {
  *ndev = 1;
  return ncclSuccess;
}

static ncclResult_t ginRocshmemGdaGetProperties(int dev, ncclNetProperties_v12_t* props) {
  memset(props, 0, sizeof(*props));
  props->name = const_cast<char*>("rocshmem-gda");
  props->pciPath = nullptr;
  props->guid = 0;
  props->ptrSupport = NCCL_PTR_CUDA;
  props->netDeviceType = NCCL_NET_DEVICE_GIN_ROCSHMEM_GDA;
  props->netDeviceVersion = NCCL_GIN_ROCSHMEM_VERSION;
  props->maxP2pBytes = 1ULL << 30;
  props->maxCollBytes = 1ULL << 30;
  return ncclSuccess;
}

static ncclResult_t ginRocshmemGdaListen(void* ctx, int dev, void* handle, void** listenComm) {
  auto* lctx = new ginRocshmemGdaListenCtx;
  lctx->dev = dev;
  *listenComm = lctx;
  memset(handle, 0, NCCL_NET_HANDLE_MAXSIZE);
  return ncclSuccess;
}

static ncclResult_t ginRocshmemGdaConnect(void* ctx, void* handles[], int nranks, int rank,
                                          void* listenComm, void** collComm) {
  struct ginRocshmemInitCtx *ictx = (struct ginRocshmemInitCtx *)ctx;
  auto *cctx = new ginRocshmemGdaCollCtx{};
  cctx->nranks = nranks;
  cctx->rank = rank;
  cctx->comm = ictx->comm;

  // Create QPs during connect (like GDAKI creates IB connections in connect)
  int rc = rocshmem_gin_create_qps(nranks, rank,
                                    ginGdaBootstrapAllgather, cctx->comm->bootstrap,
                                    &cctx->qpSet, &cctx->gpu_qp_ptrs);
  if (rc != 0) {
    WARN("GIN rocshmem-gda: failed to create QPs");
    delete cctx;
    return ncclSystemError;
  }

  // Initialize rocshmem __constant__ device memory
  rocshmem_gin_init_constmem(rocshmem_gin_get_provider(cctx->qpSet), rank);

  INFO(NCCL_INIT, "GIN rocshmem-gda: QPs created (%d ranks)", nranks);
  *collComm = cctx;
  return ncclSuccess;
}

static ncclResult_t ginRocshmemGdaCloseListen(void* listenComm) {
  delete (ginRocshmemGdaListenCtx*)listenComm;
  return ncclSuccess;
}

static ncclResult_t ginRocshmemGdaCloseColl(void* collComm) {
  struct ginRocshmemGdaCollCtx *cctx = (struct ginRocshmemGdaCollCtx *)collComm;
  if (cctx) {
    if (cctx->qpSet) rocshmem_gin_destroy_qps(cctx->qpSet);
    delete cctx;
  }
  return ncclSuccess;
}

static ncclResult_t ginRocshmemGdaFinalize(void* ctx) {
  delete (ginRocshmemInitCtx*)ctx;
  return ncclSuccess;
}

///////////////////////////////////////////////////////////////////////////////
// regMrSym: memory registration using QP set from collComm
///////////////////////////////////////////////////////////////////////////////

static ncclResult_t ginRocshmemGdaRegMrSym(void* collComm, void* data, size_t size,
                                           int type, uint64_t mrFlags,
                                           void** mhandle, void** ginHandle) {
  struct ginRocshmemGdaCollCtx *cctx = (struct ginRocshmemGdaCollCtx *)collComm;
  struct ginRocshmemGdaMemHandle *mh = NULL;

  NCCLCHECK(ncclCalloc(&mh, 1));

  if (hipMalloc(&mh->devHandle, sizeof(ncclGinRocshmemGdaMemHandle)) != hipSuccess) {
    free(mh);
    return ncclSystemError;
  }

  // Register buffer with QP set's PD
  uint32_t lkey, rkey;
  if (rocshmem_gin_reg_mr_vmm(cctx->qpSet, data, size, /*atomic=*/0,
                               &mh->mr, &lkey, &rkey) != 0) {
    WARN("GIN rocshmem-gda: MR registration failed for buffer %p size %zu", data, size);
    (void)hipFree(mh->devHandle);
    free(mh);
    return ncclSystemError;
  }

  INFO(NCCL_INIT, "GIN rocshmem-gda: registered addr=%p size=%zu lkey=0x%x rkey=0x%x",
       data, size, lkey, rkey);

  // Allgather rkeys and base VAs across all peers
  uint32_t *rkeys_buf = (uint32_t *)malloc(sizeof(uint32_t) * cctx->nranks);
  uintptr_t *vas_buf = (uintptr_t *)malloc(sizeof(uintptr_t) * cctx->nranks);
  rkeys_buf[cctx->rank] = rkey;
  vas_buf[cctx->rank] = (uintptr_t)data;

  bootstrapAllGather(cctx->comm->bootstrap, rkeys_buf, sizeof(uint32_t));
  bootstrapAllGather(cctx->comm->bootstrap, vas_buf, sizeof(uintptr_t));

  // Copy rkeys and VAs to GPU arrays
  if (hipMalloc(&mh->rkeys_dev, sizeof(uint32_t) * cctx->nranks) != hipSuccess ||
      hipMalloc(&mh->remote_vas_dev, sizeof(uintptr_t) * cctx->nranks) != hipSuccess) {
    free(rkeys_buf); free(vas_buf);
    rocshmem_gin_dereg_mr(mh->mr);
    (void)hipFree(mh->devHandle);
    free(mh);
    return ncclSystemError;
  }

  (void)hipMemcpy(mh->rkeys_dev, rkeys_buf, sizeof(uint32_t) * cctx->nranks, hipMemcpyHostToDevice);
  (void)hipMemcpy(mh->remote_vas_dev, vas_buf, sizeof(uintptr_t) * cctx->nranks, hipMemcpyHostToDevice);

  // Set GPU-side mem handle
  ncclGinRocshmemGdaMemHandle hostMh;
  hostMh.local_va = (uintptr_t)data;
  hostMh.lkey = lkey;
  hostMh.rkeys = mh->rkeys_dev;
  hostMh.remote_vas = mh->remote_vas_dev;

  (void)hipMemcpy(mh->devHandle, &hostMh, sizeof(ncclGinRocshmemGdaMemHandle), hipMemcpyHostToDevice);

  free(rkeys_buf);
  free(vas_buf);

  *mhandle = mh;
  *ginHandle = mh->devHandle;
  return ncclSuccess;
}

static ncclResult_t ginRocshmemGdaRegMrSymDmaBuf(void* collComm, void* data, size_t size,
                                                  int type, uint64_t offset, int fd,
                                                  uint64_t mrFlags, void** mhandle, void** ginHandle) {
  return ginRocshmemGdaRegMrSym(collComm, data, size, type, mrFlags, mhandle, ginHandle);
}

static ncclResult_t ginRocshmemGdaDeregMrSym(void* collComm, void* mhandle) {
  struct ginRocshmemGdaMemHandle *mh = (struct ginRocshmemGdaMemHandle *)mhandle;
  if (!mh) return ncclSuccess;
  if (mh->mr) rocshmem_gin_dereg_mr(mh->mr);
  if (mh->rkeys_dev) (void)hipFree(mh->rkeys_dev);
  if (mh->remote_vas_dev) (void)hipFree(mh->remote_vas_dev);
  if (mh->devHandle) (void)hipFree(mh->devHandle);
  free(mh);
  return ncclSuccess;
}

///////////////////////////////////////////////////////////////////////////////
// createContext: lightweight setup (signals, counters, GPU context)
///////////////////////////////////////////////////////////////////////////////

static ncclResult_t ginRocshmemGdaCreateContext(void* collComm, ncclGinConfig_v13_t* config,
                                                void** outGinCtx, ncclNetDeviceHandle_v11_t** outDevHandle) {
  struct ginRocshmemGdaCollCtx *cctx = (struct ginRocshmemGdaCollCtx *)collComm;
  ncclResult_t ret = ncclSuccess;

  auto *ctx = new ginRocshmemGdaGinCtx{};
  ctx->nRanks = cctx->nranks;
  ctx->rank = cctx->rank;
  ctx->nSignals = config->nSignals;
  ctx->nCounters = config->nCounters;
  ctx->hasError = false;
  ctx->qpSet = cctx->qpSet;  // borrow, not own
  ctx->signalMr = nullptr;
  ctx->counterMr = nullptr;

  NCCLCHECK(ncclCalloc(&ctx->devHandle, 1));
  ctx->devHandle->netDeviceType = NCCL_NET_DEVICE_GIN_ROCSHMEM_GDA;
  ctx->devHandle->netDeviceVersion = NCCL_GIN_ROCSHMEM_VERSION;
  ctx->devHandle->needsProxyProgress = 0;

  if (hipMalloc(&ctx->gpuCtxDev, sizeof(ncclGinRocshmemGdaGPUContext)) != hipSuccess) {
    ret = ncclSystemError; goto fail;
  }

  memset(&ctx->gpuCtxHost, 0, sizeof(ncclGinRocshmemGdaGPUContext));
  ctx->gpuCtxHost.nRanks = ctx->nRanks;
  ctx->gpuCtxHost.rank = ctx->rank;
  ctx->gpuCtxHost.nSignals = config->nSignals;
  ctx->gpuCtxHost.nCounters = config->nCounters;

  // QP pointers were created in connect and saved in collComm
  ctx->gpuCtxHost.qps = (rocshmem::QueuePair**)cctx->gpu_qp_ptrs;

  // Allocate signals
  if (config->nSignals > 0) {
    if (hipExtMallocWithFlags((void**)&ctx->gpuCtxHost.signals, sizeof(uint64_t) * config->nSignals,
                              hipDeviceMallocFinegrained) != hipSuccess) {
      ret = ncclSystemError; goto fail;
    }
    (void)hipMemset(ctx->gpuCtxHost.signals, 0, sizeof(uint64_t) * config->nSignals);

    uint32_t sigLkey, sigRkey;
    if (rocshmem_gin_reg_mr(cctx->qpSet, ctx->gpuCtxHost.signals,
                             sizeof(uint64_t) * config->nSignals, /*atomic=*/1,
                             &ctx->signalMr, &sigLkey, &sigRkey) != 0) {
      ret = ncclSystemError; goto fail;
    }

    // Allgather signal rkeys and addresses
    if (hipMalloc(&ctx->gpuCtxHost.signal_rkeys, sizeof(uint32_t) * ctx->nRanks) != hipSuccess ||
        hipMalloc(&ctx->gpuCtxHost.signal_raddrs, sizeof(uintptr_t) * ctx->nRanks) != hipSuccess) {
      ret = ncclSystemError; goto fail;
    }

    uint32_t *rkeys_buf = (uint32_t *)malloc(sizeof(uint32_t) * ctx->nRanks);
    uintptr_t *raddrs_buf = (uintptr_t *)malloc(sizeof(uintptr_t) * ctx->nRanks);
    rkeys_buf[ctx->rank] = sigRkey;
    raddrs_buf[ctx->rank] = (uintptr_t)ctx->gpuCtxHost.signals;

    bootstrapAllGather(cctx->comm->bootstrap, rkeys_buf, sizeof(uint32_t));
    bootstrapAllGather(cctx->comm->bootstrap, raddrs_buf, sizeof(uintptr_t));

    (void)hipMemcpy(ctx->gpuCtxHost.signal_rkeys, rkeys_buf,
                    sizeof(uint32_t) * ctx->nRanks, hipMemcpyHostToDevice);
    (void)hipMemcpy(ctx->gpuCtxHost.signal_raddrs, raddrs_buf,
                    sizeof(uintptr_t) * ctx->nRanks, hipMemcpyHostToDevice);
    free(rkeys_buf);
    free(raddrs_buf);
  }

  if (config->nCounters > 0) {
    if (hipExtMallocWithFlags((void**)&ctx->gpuCtxHost.counters, sizeof(uint64_t) * config->nCounters,
                              hipDeviceMallocFinegrained) != hipSuccess) {
      ret = ncclSystemError; goto fail;
    }
    (void)hipMemset(ctx->gpuCtxHost.counters, 0, sizeof(uint64_t) * config->nCounters);
  }

  // Copy GPU context
  (void)hipMemcpy(ctx->gpuCtxDev, &ctx->gpuCtxHost, sizeof(ncclGinRocshmemGdaGPUContext),
                  hipMemcpyHostToDevice);

  ctx->devHandle->handle = ctx->gpuCtxDev;
  ctx->devHandle->size = sizeof(ncclGinRocshmemGdaGPUContext);

  *outGinCtx = ctx;
  *outDevHandle = ctx->devHandle;
  INFO(NCCL_INIT, "GIN rocshmem-gda: context created (%d signals, %d counters)", config->nSignals, config->nCounters);
  return ncclSuccess;

fail:
  if (ctx) {
    if (ctx->signalMr) rocshmem_gin_dereg_mr(ctx->signalMr);
    if (ctx->gpuCtxHost.signals) (void)hipFree(ctx->gpuCtxHost.signals);
    if (ctx->gpuCtxHost.counters) (void)hipFree(ctx->gpuCtxHost.counters);
    if (ctx->gpuCtxHost.signal_rkeys) (void)hipFree(ctx->gpuCtxHost.signal_rkeys);
    if (ctx->gpuCtxHost.signal_raddrs) (void)hipFree(ctx->gpuCtxHost.signal_raddrs);
    if (ctx->gpuCtxDev) (void)hipFree(ctx->gpuCtxDev);
    free(ctx->devHandle);
    delete ctx;
  }
  return ret;
}

static ncclResult_t ginRocshmemGdaDestroyContext(void* ginCtx) {
  struct ginRocshmemGdaGinCtx *ctx = (struct ginRocshmemGdaGinCtx *)ginCtx;
  if (!ctx) return ncclSuccess;
  if (ctx->signalMr) rocshmem_gin_dereg_mr(ctx->signalMr);
  if (ctx->counterMr) rocshmem_gin_dereg_mr(ctx->counterMr);
  // qpSet is owned by collComm, not destroyed here
  if (ctx->gpuCtxHost.signals) (void)hipFree(ctx->gpuCtxHost.signals);
  if (ctx->gpuCtxHost.counters) (void)hipFree(ctx->gpuCtxHost.counters);
  if (ctx->gpuCtxHost.signal_rkeys) (void)hipFree(ctx->gpuCtxHost.signal_rkeys);
  if (ctx->gpuCtxHost.signal_raddrs) (void)hipFree(ctx->gpuCtxHost.signal_raddrs);
  if (ctx->gpuCtxDev) (void)hipFree(ctx->gpuCtxDev);
  free(ctx->devHandle);
  delete ctx;
  return ncclSuccess;
}

///////////////////////////////////////////////////////////////////////////////
// Progress / error query
///////////////////////////////////////////////////////////////////////////////

static ncclResult_t ginRocshmemGdaGinProgress(void* ginCtx) {
  return ncclSuccess;
}

static ncclResult_t ginRocshmemGdaQueryLastError(void* ginCtx, bool* hasError) {
  *hasError = false;
  return ncclSuccess;
}

///////////////////////////////////////////////////////////////////////////////
// Plugin vtable
///////////////////////////////////////////////////////////////////////////////

__attribute__((visibility("default")))
ncclGin_t ncclGinRocshmemGdaPlugin = {
  .name            = "rocshmem-gda",
  .init            = ginRocshmemGdaInit,
  .devices         = ginRocshmemGdaDevices,
  .getProperties   = ginRocshmemGdaGetProperties,
  .listen          = ginRocshmemGdaListen,
  .connect         = ginRocshmemGdaConnect,
  .createContext   = ginRocshmemGdaCreateContext,
  .regMrSym        = ginRocshmemGdaRegMrSym,
  .regMrSymDmaBuf  = ginRocshmemGdaRegMrSymDmaBuf,
  .deregMrSym      = ginRocshmemGdaDeregMrSym,
  .destroyContext  = ginRocshmemGdaDestroyContext,
  .closeColl       = ginRocshmemGdaCloseColl,
  .closeListen     = ginRocshmemGdaCloseListen,
  .iput            = NULL,
  .iputSignal      = NULL,
  .iget            = NULL,
  .iflush          = NULL,
  .test            = NULL,
  .ginProgress     = ginRocshmemGdaGinProgress,
  .queryLastError  = ginRocshmemGdaQueryLastError,
  .finalize        = ginRocshmemGdaFinalize,
};

#endif // ENABLE_ROCSHMEM_GIN
