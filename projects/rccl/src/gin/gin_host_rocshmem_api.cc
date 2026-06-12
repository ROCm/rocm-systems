/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifdef ENABLE_ROCSHMEM_GIN

#include "gin/gin_host_rocshmem_api.h"
#include "comm.h"
#include "dev_runtime.h"
#include "nccl_device/gin/rocshmem/gin_rocshmem_api_device_host_common.h"

#include <rocshmem/rocshmem.hpp>
#include <hip/hip_runtime.h>
#include <map>

// Refcount for buffer registration: ncclGinRegister calls us once per
// GIN context for the same buffer.
static std::map<void*, int> bufferRegRefcount;


struct ginRocshmemCtx {
  struct ncclComm *comm;
  void *collComm;
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

ncclResult_t ncclGinRocshmemCreateContext(struct ncclComm *comm, void *collComm, int devId,
                                          int nSignals, int nCounters, void **outGinCtx,
                                          ncclNetDeviceHandle_v11_t **outDevHandle) {
  ncclResult_t ret = ncclSuccess;
  struct ginRocshmemCtx *ctx = NULL;

  NCCLCHECK(ncclCalloc(&ctx, 1));
  ctx->comm = comm;
  ctx->collComm = collComm;
  if (comm) {
    ctx->nRanks = comm->nRanks;
    ctx->rank = comm->rank;
  } else {
    // Plugin path: get rank/npes from rocshmem
    ctx->nRanks = rocshmem::rocshmem_n_pes();
    ctx->rank = rocshmem::rocshmem_my_pe();
  }
  ctx->nSignals = nSignals;
  ctx->nCounters = nCounters;
  ctx->hasError = false;

  NCCLCHECK(ncclCalloc(&ctx->devHandle, 1));
  ctx->devHandle->netDeviceType = NCCL_NET_DEVICE_GIN_ROCSHMEM_API;
  ctx->devHandle->netDeviceVersion = NCCL_GIN_ROCSHMEM_VERSION;
  ctx->devHandle->needsProxyProgress = 0;

  if (hipMalloc(&ctx->gpuCtxDev, sizeof(ncclGinRocshmemGPUContext)) != hipSuccess) {
    WARN("GIN rocshmem: failed to allocate GPU context");
    ret = ncclSystemError;
    goto fail;
  }

  memset(&ctx->gpuCtxHost, 0, sizeof(ncclGinRocshmemGPUContext));
  ctx->gpuCtxHost.nRanks = ctx->nRanks;
  ctx->gpuCtxHost.rank = ctx->rank;
  ctx->gpuCtxHost.nSignals = nSignals;
  ctx->gpuCtxHost.nCounters = nCounters;

  // Signals on symmetric heap (needed for remote atomic access)
  if (nSignals > 0) {
    ctx->gpuCtxHost.signals = (uint64_t*)rocshmem::rocshmem_malloc(sizeof(uint64_t) * nSignals);
    if (!ctx->gpuCtxHost.signals) {
      WARN("GIN rocshmem: rocshmem_malloc failed for signals");
      ret = ncclSystemError;
      goto fail;
    }
    hipMemset(ctx->gpuCtxHost.signals, 0, sizeof(uint64_t) * nSignals);
  }

  // Counters local only
  if (nCounters > 0) {
    if (hipMalloc(&ctx->gpuCtxHost.counters, sizeof(uint64_t) * nCounters) != hipSuccess) {
      ret = ncclSystemError;
      goto fail;
    }
    hipMemset(ctx->gpuCtxHost.counters, 0, sizeof(uint64_t) * nCounters);
  }

  if (hipMemcpy(ctx->gpuCtxDev, &ctx->gpuCtxHost, sizeof(ncclGinRocshmemGPUContext),
                hipMemcpyHostToDevice) != hipSuccess) {
    ret = ncclSystemError;
    goto fail;
  }

  ctx->devHandle->handle = ctx->gpuCtxDev;
  ctx->devHandle->size = sizeof(ncclGinRocshmemGPUContext);

  *outGinCtx = ctx;
  *outDevHandle = ctx->devHandle;
  INFO(NCCL_INIT, "GIN rocshmem: context created (%d signals, %d counters)", nSignals, nCounters);
  return ncclSuccess;

fail:
  if (ctx) {
    if (ctx->gpuCtxHost.signals) rocshmem::rocshmem_free(ctx->gpuCtxHost.signals);
    if (ctx->gpuCtxHost.counters) hipFree(ctx->gpuCtxHost.counters);
    if (ctx->gpuCtxDev) hipFree(ctx->gpuCtxDev);
    free(ctx->devHandle);
    free(ctx);
  }
  return ret;
}

ncclResult_t ncclGinRocshmemDestroyContext(ncclGin_t *ginComm, void *ginCtx) {
  struct ginRocshmemCtx *ctx = (struct ginRocshmemCtx *)ginCtx;
  if (ctx == NULL) return ncclSuccess;

  if (ctx->gpuCtxHost.signals) rocshmem::rocshmem_free(ctx->gpuCtxHost.signals);
  if (ctx->gpuCtxHost.counters) hipFree(ctx->gpuCtxHost.counters);
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

  auto &refcount = bufferRegRefcount[addr];
  // lsaSelfAddr: the LSA flat VA for self rank for this memory region.
  // All cross-PE memory access goes through the LSA flat space (cuMemSetAccess
  // is only called for the LSA flat mappings, not for primaryAddr).
  // We must register and address-reference the buffer at its LSA flat VA so
  // that ipc_resolve_remote can translate to other ranks' LSA flat VAs.
  void* lsaSelfAddr = nullptr;
  NCCLCHECK(ncclDevrGetLsaSelfAddr(ctx->comm, addr, &lsaSelfAddr));
  if (lsaSelfAddr == nullptr) {
    WARN("GIN rocshmem: could not resolve LSA flat addr for %p", addr);
    free(mh);
    return ncclSystemError;
  }

  if (refcount == 0) {
    struct ncclDevrState* devr = &ctx->comm->devrState;

    // Always use stride-based VMM registration: each PE's LSA flat VA differs
    // by exactly bigSize, so remote_bases[pe] = lsaSelfAddr + (pe-lsaSelf)*bigSize.
    int rc = rocshmem::rocshmem_buffer_register_vmm(lsaSelfAddr, size, devr->lsaSelf,
                 devr->lsaSize, (ptrdiff_t)devr->bigSize);
    if (rc != 0) {
      WARN("GIN rocshmem: buffer register failed for %p (lsaSelf=%p) size %zu",
           addr, lsaSelfAddr, size);
      bufferRegRefcount.erase(addr);
      free(mh);
      return ncclSystemError;
    }

    INFO(NCCL_INIT, "GIN rocshmem: registered addr=%p lsaSelf=%p +%zu", addr, lsaSelfAddr, size);
  }
  refcount++;
  mh->addr = addr;
  mh->size = size;

  if (hipMalloc(&mh->devHandle, sizeof(ncclGinRocshmemMemHandle)) != hipSuccess) {
    free(mh);
    return ncclSystemError;
  }

  // Store the LSA flat VA for self rank. The device-side code will use this as
  // the local_base to compute offsets, and ipc_resolve_remote will translate to
  // the remote rank's LSA flat VA (which is P2P accessible).
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

ncclResult_t ncclGinRocshmemDeregister(ncclGin_t *ginComm, void *ginCtx, void *mhandle) {
  struct ginRocshmemMemHandle *mh = (struct ginRocshmemMemHandle *)mhandle;
  if (mh == NULL) return ncclSuccess;

  if (mh->addr) {
    auto &refcount = bufferRegRefcount[mh->addr];
    refcount--;
    if (refcount <= 0) {
      // Registered via rocshmem_buffer_register_vmm (constant-memory table),
      // no corresponding rocshmem_buffer_unregister needed.
      bufferRegRefcount.erase(mh->addr);
    }
  }
  if (mh->devHandle) hipFree(mh->devHandle);
  free(mh);
  return ncclSuccess;
}

ncclResult_t ncclGinRocshmemProgress(ncclGin_t *ginComm, void *ginCtx) {
  return ncclSuccess;
}

ncclResult_t ncclGinRocshmemQueryLastError(ncclGin_t *ginComm, void *ginCtx, bool *hasError) {
  struct ginRocshmemCtx *ctx = (struct ginRocshmemCtx *)ginCtx;
  *hasError = ctx ? ctx->hasError : false;
  return ncclSuccess;
}

////////////////////////////////////////////////////////////////////////////////
// ncclGin_t vtable for rocshmem: provides the plugin interface so that
// gin_host.cc's ncclGinConnectOnce can drive init/listen/connect.
// Rocshmem transport is already established via rocshmem_init(), so
// listen/connect/closeListen are stubs.
////////////////////////////////////////////////////////////////////////////////

static ncclResult_t ncclGinRocshmemInit(void** ctx, uint64_t commId, ncclDebugLogger_t logFunction) {
  *ctx = nullptr;
  return ncclSuccess;
}

static ncclResult_t ncclGinRocshmemDevices(int* ndev) {
  *ndev = 1;
  return ncclSuccess;
}

static ncclResult_t ncclGinRocshmemGetProperties(int dev, ncclNetProperties_t* props) {
  memset(props, 0, sizeof(*props));
  props->name = (char*)"rocshmem";
  props->pciPath = NULL;
  props->guid = 0;
  props->ptrSupport = NCCL_PTR_CUDA;
  props->netDeviceType = NCCL_NET_DEVICE_GIN_ROCSHMEM_API;
  props->netDeviceVersion = NCCL_GIN_ROCSHMEM_VERSION;
  return ncclSuccess;
}

static ncclResult_t ncclGinRocshmemListen(void* ctx, int dev, void* handle, void** listenComm) {
  *listenComm = (void*)0x1;
  return ncclSuccess;
}

static ncclResult_t ncclGinRocshmemConnect(void* ctx, void* handles[], int nranks, int rank,
                                           void* listenComm, void** collComm) {
  *collComm = (void*)0x1;
  return ncclSuccess;
}

static ncclResult_t ncclGinRocshmemCloseListen(void* listenComm) {
  return ncclSuccess;
}

static ncclResult_t ncclGinRocshmemCloseColl(void* collComm) {
  return ncclSuccess;
}

static ncclResult_t ncclGinRocshmemFinalize(void* ctx) {
  return ncclSuccess;
}

///////////////////////////////////////////////////////////////////////////////
// Plugin-facing variants (no ncclComm available)
///////////////////////////////////////////////////////////////////////////////

ncclResult_t ncclGinRocshmemCreateContextFromPlugin(int nSignals, int nCounters,
                                                     void **outGinCtx,
                                                     ncclNetDeviceHandle_v11_t **outDevHandle) {
  // Delegate to the full version with NULL comm — only needs rocshmem symmetric heap
  return ncclGinRocshmemCreateContext(NULL, NULL, 0, nSignals, nCounters, outGinCtx, outDevHandle);
}

ncclResult_t ncclGinRocshmemRegisterFromPlugin(void *addr, size_t size, int type,
                                                uint64_t mr_flags, void **mhandle, void **ginHandle) {
  return ncclGinRocshmemRegister(NULL, NULL, addr, size, type, mr_flags, mhandle, ginHandle);
}

ncclResult_t ncclGinRocshmemDeregisterFromPlugin(void *mhandle) {
  return ncclGinRocshmemDeregister(NULL, NULL, mhandle);
}

ncclResult_t ncclGinRocshmemDestroyContextFromPlugin(void *ginCtx) {
  return ncclGinRocshmemDestroyContext(NULL, ginCtx);
}

#endif // ENABLE_ROCSHMEM_GIN
