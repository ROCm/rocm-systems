/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifdef ENABLE_ROCSHMEM_GIN

/**
 * Built-in GIN plugin for rocshmem API path (GIN_TYPE=4).
 *
 * Follows the upstream vtable pattern: init() returns a ginRocshmemInitCtx,
 * connect() creates a ginRocshmemCollCtx (= collComm), and all subsequent
 * vtable calls receive collComm with the state they need.
 */

#include "gin/gin_host_rocshmem_api.h"
#include "comm.h"
#include "dev_runtime.h"
#include "bootstrap.h"
#include "nccl_device/gin/rocshmem_api/gin_rocshmem_api_device_host_common.h"
#include "plugin/nccl_net.h"

#include <rocshmem/rocshmem.hpp>
#include <hip/hip_runtime.h>
#include <map>

// collComm: per-connection state, returned by connect(), passed to all vtable calls
struct ginRocshmemCollCtx {
  int nranks;
  int rank;
  struct ncclComm *comm;
};

// ginCtx: per-context state, returned by createContext(), passed to operations
struct ginRocshmemGinCtx {
  ncclNetDeviceHandle_v11_t *devHandle;
  ncclGinRocshmemGPUContext *gpuCtxDev;
  ncclGinRocshmemGPUContext gpuCtxHost;
  int nRanks;
  int rank;
  int nSignals;
  int nCounters;
  bool hasError;
};

struct ginRocshmemMemHandle {
  ncclGinRocshmemMemHandle *devHandle;
  void *addr;
  size_t size;
};

// Refcount for buffer registration: ncclGinRegister calls regMrSym once per
// connection for the same buffer.
static std::map<void*, int> bufferRegRefcount;

struct ginRocshmemListenCtx {
  int dev;
};

///////////////////////////////////////////////////////////////////////////////
// Vtable implementations
///////////////////////////////////////////////////////////////////////////////

static ncclResult_t ginRocshmemInit(void** ctx, uint64_t commId, ncclDebugLogger_t logFunction) {
  const char *gin_type = getenv("NCCL_GIN_TYPE");
  if (!gin_type || atoi(gin_type) != NCCL_NET_DEVICE_GIN_ROCSHMEM_API)
    return ncclInternalError;
  // Allocate init context; devrState/bootstrap filled by ncclGinRocshmemSetInitContext
  struct ginRocshmemInitCtx *ictx = new ginRocshmemInitCtx{};
  *ctx = ictx;
  return ncclSuccess;
}

static ncclResult_t ginRocshmemDevices(int* ndev) {
  *ndev = 1;
  return ncclSuccess;
}

static ncclResult_t ginRocshmemGetProperties(int dev, ncclNetProperties_v12_t* props) {
  memset(props, 0, sizeof(*props));
  props->name = "rocshmem";
  props->pciPath = nullptr;
  props->guid = 0;
  props->ptrSupport = NCCL_PTR_CUDA;
  props->netDeviceType = NCCL_NET_DEVICE_GIN_ROCSHMEM_API;
  props->netDeviceVersion = NCCL_GIN_ROCSHMEM_VERSION;
  props->maxP2pBytes = 1ULL << 30;
  props->maxCollBytes = 1ULL << 30;
  return ncclSuccess;
}

static ncclResult_t ginRocshmemListen(void* ctx, int dev, void* handle, void** listenComm) {
  auto* lctx = new ginRocshmemListenCtx;
  lctx->dev = dev;
  *listenComm = lctx;
  memset(handle, 0, NCCL_NET_HANDLE_MAXSIZE);
  return ncclSuccess;
}

static ncclResult_t ginRocshmemConnect(void* ctx, void* handles[], int nranks, int rank,
                                       void* listenComm, void** collComm) {
  struct ginRocshmemInitCtx *ictx = (struct ginRocshmemInitCtx *)ctx;
  auto* cctx = new ginRocshmemCollCtx;
  cctx->nranks = nranks;
  cctx->rank = rank;
  cctx->comm = ictx->comm;
  *collComm = cctx;
  return ncclSuccess;
}

static ncclResult_t ginRocshmemCloseListen(void* listenComm) {
  delete (ginRocshmemListenCtx*)listenComm;
  return ncclSuccess;
}

static ncclResult_t ginRocshmemCloseColl(void* collComm) {
  delete (ginRocshmemCollCtx*)collComm;
  return ncclSuccess;
}

static ncclResult_t ginRocshmemFinalize(void* ctx) {
  delete (ginRocshmemInitCtx*)ctx;
  return ncclSuccess;
}

///////////////////////////////////////////////////////////////////////////////
// createContext: called per-devComm, receives collComm
///////////////////////////////////////////////////////////////////////////////

static ncclResult_t ginRocshmemCreateContext(void* collComm, ncclGinConfig_v13_t* config,
                                              void** outGinCtx, ncclNetDeviceHandle_v11_t** outDevHandle) {
  struct ginRocshmemCollCtx *cctx = (struct ginRocshmemCollCtx *)collComm;
  ncclResult_t ret = ncclSuccess;

  auto *ctx = new ginRocshmemGinCtx{};
  ctx->nRanks = cctx->nranks;
  ctx->rank = cctx->rank;
  ctx->nSignals = config->nSignals;
  ctx->nCounters = config->nCounters;
  ctx->hasError = false;

  NCCLCHECK(ncclCalloc(&ctx->devHandle, 1));
  ctx->devHandle->netDeviceType = NCCL_NET_DEVICE_GIN_ROCSHMEM_API;
  ctx->devHandle->netDeviceVersion = NCCL_GIN_ROCSHMEM_VERSION;
  ctx->devHandle->needsProxyProgress = 0;

  if (hipMalloc(&ctx->gpuCtxDev, sizeof(ncclGinRocshmemGPUContext)) != hipSuccess) {
    ret = ncclSystemError;
    goto fail;
  }

  memset(&ctx->gpuCtxHost, 0, sizeof(ncclGinRocshmemGPUContext));
  ctx->gpuCtxHost.nRanks = ctx->nRanks;
  ctx->gpuCtxHost.rank = ctx->rank;
  ctx->gpuCtxHost.nSignals = config->nSignals;
  ctx->gpuCtxHost.nCounters = config->nCounters;

  if (config->nSignals > 0) {
    ctx->gpuCtxHost.signals = (uint64_t*)rocshmem::rocshmem_malloc(sizeof(uint64_t) * config->nSignals);
    if (!ctx->gpuCtxHost.signals) { ret = ncclSystemError; goto fail; }
    hipMemset(ctx->gpuCtxHost.signals, 0, sizeof(uint64_t) * config->nSignals);
  }

  if (config->nCounters > 0) {
    if (hipMalloc(&ctx->gpuCtxHost.counters, sizeof(uint64_t) * config->nCounters) != hipSuccess) {
      ret = ncclSystemError; goto fail;
    }
    hipMemset(ctx->gpuCtxHost.counters, 0, sizeof(uint64_t) * config->nCounters);
  }

  if (hipMemcpy(ctx->gpuCtxDev, &ctx->gpuCtxHost, sizeof(ncclGinRocshmemGPUContext),
                hipMemcpyHostToDevice) != hipSuccess) {
    ret = ncclSystemError; goto fail;
  }

  ctx->devHandle->handle = ctx->gpuCtxDev;
  ctx->devHandle->size = sizeof(ncclGinRocshmemGPUContext);

  *outGinCtx = ctx;
  *outDevHandle = ctx->devHandle;
  INFO(NCCL_INIT, "GIN rocshmem-api: context created (%d signals, %d counters)",
       config->nSignals, config->nCounters);
  return ncclSuccess;

fail:
  if (ctx) {
    if (ctx->gpuCtxHost.signals) rocshmem::rocshmem_free(ctx->gpuCtxHost.signals);
    if (ctx->gpuCtxHost.counters) hipFree(ctx->gpuCtxHost.counters);
    if (ctx->gpuCtxDev) hipFree(ctx->gpuCtxDev);
    free(ctx->devHandle);
    delete ctx;
  }
  return ret;
}

static ncclResult_t ginRocshmemDestroyContext(void* ginCtx) {
  struct ginRocshmemGinCtx *ctx = (struct ginRocshmemGinCtx *)ginCtx;
  if (!ctx) return ncclSuccess;
  if (ctx->gpuCtxHost.signals) rocshmem::rocshmem_free(ctx->gpuCtxHost.signals);
  if (ctx->gpuCtxHost.counters) hipFree(ctx->gpuCtxHost.counters);
  if (ctx->gpuCtxDev) hipFree(ctx->gpuCtxDev);
  free(ctx->devHandle);
  delete ctx;
  return ncclSuccess;
}

///////////////////////////////////////////////////////////////////////////////
// regMrSym / deregMrSym: memory registration, receives collComm
///////////////////////////////////////////////////////////////////////////////

static ncclResult_t ginRocshmemRegMrSym(void* collComm, void* data, size_t size,
                                         int type, uint64_t mrFlags,
                                         void** mhandle, void** ginHandle) {
  struct ginRocshmemCollCtx *cctx = (struct ginRocshmemCollCtx *)collComm;
  struct ncclDevrState *devr = &cctx->comm->devrState;
  struct ginRocshmemMemHandle *mh = NULL;

  NCCLCHECK(ncclCalloc(&mh, 1));

  auto &refcount = bufferRegRefcount[data];

  void *lsaSelfAddr = nullptr;
  NCCLCHECK(ncclDevrGetLsaSelfAddr(devr, data, &lsaSelfAddr));
  if (lsaSelfAddr == nullptr) {
    WARN("GIN rocshmem-api: could not resolve LSA flat addr for %p", data);
    free(mh);
    return ncclSystemError;
  }

  if (refcount == 0) {
    int rc = rocshmem::rocshmem_buffer_register_vmm(lsaSelfAddr, size, devr->lsaSelf,
                 devr->lsaSize, (ptrdiff_t)devr->bigSize);
    if (rc != 0) {
      WARN("GIN rocshmem-api: buffer register failed for %p (lsaSelf=%p) size %zu",
           data, lsaSelfAddr, size);
      bufferRegRefcount.erase(data);
      free(mh);
      return ncclSystemError;
    }
    INFO(NCCL_INIT, "GIN rocshmem-api: registered addr=%p lsaSelf=%p +%zu", data, lsaSelfAddr, size);
  }
  refcount++;

  mh->addr = data;
  mh->size = size;

  if (hipMalloc(&mh->devHandle, sizeof(ncclGinRocshmemMemHandle)) != hipSuccess) {
    free(mh);
    return ncclSystemError;
  }

  ncclGinRocshmemMemHandle hostMh;
  hostMh.baseAddr = (uintptr_t)lsaSelfAddr;

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

static ncclResult_t ginRocshmemRegMrSymDmaBuf(void* collComm, void* data, size_t size,
                                               int type, uint64_t offset, int fd,
                                               uint64_t mrFlags, void** mhandle, void** ginHandle) {
  return ginRocshmemRegMrSym(collComm, data, size, type, mrFlags, mhandle, ginHandle);
}

static ncclResult_t ginRocshmemDeregMrSym(void* collComm, void* mhandle) {
  struct ginRocshmemMemHandle *mh = (struct ginRocshmemMemHandle *)mhandle;
  if (!mh) return ncclSuccess;

  if (mh->addr) {
    auto &refcount = bufferRegRefcount[mh->addr];
    refcount--;
    if (refcount <= 0) {
      bufferRegRefcount.erase(mh->addr);
    }
  }
  if (mh->devHandle) hipFree(mh->devHandle);
  free(mh);
  return ncclSuccess;
}

///////////////////////////////////////////////////////////////////////////////
// Progress / error query
///////////////////////////////////////////////////////////////////////////////

static ncclResult_t ginRocshmemGinProgress(void* ginCtx) {
  return ncclSuccess;
}

static ncclResult_t ginRocshmemQueryLastError(void* ginCtx, bool* hasError) {
  *hasError = false;
  return ncclSuccess;
}

///////////////////////////////////////////////////////////////////////////////
// Plugin vtable
///////////////////////////////////////////////////////////////////////////////

__attribute__((visibility("default")))
ncclGin_t ncclGinRocshmemApiPlugin = {
  .name            = "rocshmem-api",
  .init            = ginRocshmemInit,
  .devices         = ginRocshmemDevices,
  .getProperties   = ginRocshmemGetProperties,
  .listen          = ginRocshmemListen,
  .connect         = ginRocshmemConnect,
  .createContext   = ginRocshmemCreateContext,
  .regMrSym        = ginRocshmemRegMrSym,
  .regMrSymDmaBuf  = ginRocshmemRegMrSymDmaBuf,
  .deregMrSym      = ginRocshmemDeregMrSym,
  .destroyContext  = ginRocshmemDestroyContext,
  .closeColl       = ginRocshmemCloseColl,
  .closeListen     = ginRocshmemCloseListen,
  .iput            = NULL,
  .iputSignal      = NULL,
  .iget            = NULL,
  .iflush          = NULL,
  .test            = NULL,
  .ginProgress     = ginRocshmemGinProgress,
  .queryLastError  = ginRocshmemQueryLastError,
  .finalize        = ginRocshmemFinalize,
};

#endif // ENABLE_ROCSHMEM_GIN
