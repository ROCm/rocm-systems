/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifdef ENABLE_ROCSHMEM_GIN

#include "gin/gin_host_rocshmem.h"
#include "comm.h"
#include "dev_runtime.h"
#include "nccl_device/gin/rocshmem/gin_rocshmem_device_host_common.h"

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
  void *addr;            // primaryAddr (key for refcount map)
  void *registeredAddr;  // address passed to rocshmem register
  size_t size;
  bool isSym;            // true if registered via sym_buffer_register
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
  ctx->devHandle->netDeviceType = NCCL_NET_DEVICE_GIN_ROCSHMEM;
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

  // lsaSelfAddr: the LSA flat VA for this rank, valid only when comm is available.
  // LSA flat space is the only P2P-accessible address range for VMM/ncclMemAlloc buffers.
  // For non-VMM (hipMalloc'd) buffers, lsaSelfAddr stays nullptr — use addr directly.
  void* lsaSelfAddr = nullptr;
  if (ctx != NULL && ctx->comm != NULL) {
    ncclDevrGetLsaSelfAddr(ctx->comm, addr, &lsaSelfAddr);
  }
  void *regAddr = lsaSelfAddr ? lsaSelfAddr : addr;

  if (refcount == 0) {
    if (ctx != NULL && ctx->comm != NULL) {
      // With comm: sym_buffer_register for rkey + remote VA allgather (needed for GDA).
      struct ncclDevrState* devr = &ctx->comm->devrState;
      int rc = rocshmem::rocshmem_sym_buffer_register(regAddr, size,
                   devr->lsaSelf, devr->lsaSize, (ptrdiff_t)devr->bigSize);
      if (rc != 0) {
        WARN("GIN rocshmem: sym_buffer_register failed for %p (reg=%p) size %zu",
             addr, regAddr, size);
        bufferRegRefcount.erase(addr);
        free(mh);
        return ncclSystemError;
      }
      INFO(NCCL_INIT, "GIN rocshmem: registered addr=%p reg=%p +%zu", addr, regAddr, size);
    } else {
      // No comm (plugin path): local-only registration (IPC handles, no rkey allgather).
      int rc = rocshmem::rocshmem_buffer_register(addr, size);
      if (rc != 0) {
        WARN("GIN rocshmem: rocshmem_buffer_register failed for %p size %zu", addr, size);
        bufferRegRefcount.erase(addr);
        free(mh);
        return ncclSystemError;
      }
      INFO(NCCL_INIT, "GIN rocshmem: registered local addr=%p +%zu", addr, size);
    }
  }
  refcount++;
  mh->addr = addr;
  mh->registeredAddr = regAddr;
  mh->isSym = (ctx != NULL && ctx->comm != NULL);
  mh->size = size;

  if (hipMalloc(&mh->devHandle, sizeof(ncclGinRocshmemMemHandle)) != hipSuccess) {
    free(mh);
    return ncclSystemError;
  }

  // Store the address the device will use as base for offset computations.
  // LSA flat VA for VMM buffers (P2P accessible); plain addr for plugin path.
  ncclGinRocshmemMemHandle hostMh;
  hostMh.baseAddr = (uintptr_t)regAddr;

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
      if (mh->registeredAddr) {
        if (mh->isSym)
          rocshmem::rocshmem_sym_buffer_unregister(mh->registeredAddr);
        else
          rocshmem::rocshmem_buffer_unregister(mh->registeredAddr);
      }
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
