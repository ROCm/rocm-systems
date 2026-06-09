/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifdef ENABLE_ROCSHMEM_GIN

/**
 * Built-in GIN plugin for rocshmem.
 *
 * Currently a thin shim: rocshmem_init() (called from init.cc) handles
 * all transport setup. The plugin reports NCCL_NET_DEVICE_GIN_ROCSHMEM_API
 * and provides stubs for listen/connect (no-ops since rocshmem manages
 * its own QP connectivity).
 *
 * In a later stage, this plugin will perform explicit QP creation and
 * connection, replacing the rocshmem_init-based transport with
 * GIN-managed queue pairs.
 */

#include "gin/gin_host_rocshmem_api.h"
#include "comm.h"
#include "nccl_device/gin/rocshmem/gin_rocshmem_api_device_host_common.h"
#include "plugin/nccl_net.h"

#include <rocshmem/rocshmem.hpp>

// Stub listen/connect context — no actual transport needed yet
struct ginRocshmemListenCtx {
  int dev;
};

struct ginRocshmemCollCtx {
  int nranks;
  int rank;
};

static ncclResult_t ginRocshmemInit(void** ctx, uint64_t commId, ncclDebugLogger_t logFunction) {
  // rocshmem_init already called from init.cc
  *ctx = nullptr;
  return ncclSuccess;
}

static ncclResult_t ginRocshmemDevices(int* ndev) {
  *ndev = 1;
  return ncclSuccess;
}

static ncclResult_t ginRocshmemGetProperties(int dev, ncclNetProperties_v11_t* props) {
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
  // No-op: rocshmem manages its own connectivity
  auto* lctx = new ginRocshmemListenCtx;
  lctx->dev = dev;
  *listenComm = lctx;
  // Write something into handle so allgather has data to exchange
  memset(handle, 0, NCCL_NET_HANDLE_MAXSIZE);
  return ncclSuccess;
}

static ncclResult_t ginRocshmemConnect(void* ctx, void* handles[], int nranks, int rank,
                                       int nConnections, int queueDepth,
                                       void* listenComm, void** collComm) {
  // No-op: rocshmem manages its own connectivity
  auto* cctx = new ginRocshmemCollCtx;
  cctx->nranks = nranks;
  cctx->rank = rank;
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
  return ncclSuccess;
}

// Delegate to gin_host_rocshmem_api.cc implementations
// Note: createContext/regMrSym don't use collComm for rocshmem (transport is internal)

static ncclResult_t ginRocshmemCreateContext(void* collComm, int nSignals, int nCounters,
                                              int nContexts,
                                              void** ginCtx, ncclNetDeviceHandle_v11_t** devHandle) {
  // We need the ncclComm, but the plugin interface only gives us collComm.
  // For now, create the context without ncclComm — the rocshmem context
  // only needs nSignals/nCounters and rocshmem's symmetric heap.
  // TODO: plumb ncclComm through when needed for QP-level integration.
  return ncclGinRocshmemCreateContextFromPlugin(nSignals, nCounters, ginCtx, devHandle);
}

static ncclResult_t ginRocshmemRegMrSym(void* collComm, void* data, size_t size,
                                         int type, uint64_t mrFlags,
                                         void** mhandle, void** ginHandle) {
  return ncclGinRocshmemRegisterFromPlugin(data, size, type, mrFlags, mhandle, ginHandle);
}

static ncclResult_t ginRocshmemRegMrSymDmaBuf(void* collComm, void* data, size_t size,
                                               int type, uint64_t offset, int fd,
                                               uint64_t mrFlags, void** mhandle, void** ginHandle) {
  // DMA-buf variant — delegate to regular registration for now
  return ginRocshmemRegMrSym(collComm, data, size, type, mrFlags, mhandle, ginHandle);
}

static ncclResult_t ginRocshmemDeregMrSym(void* collComm, void* mhandle) {
  return ncclGinRocshmemDeregisterFromPlugin(mhandle);
}

static ncclResult_t ginRocshmemDestroyContext(void* ginCtx) {
  return ncclGinRocshmemDestroyContextFromPlugin(ginCtx);
}

static ncclResult_t ginRocshmemGinProgress(void* collComm) {
  return ncclSuccess;
}

static ncclResult_t ginRocshmemQueryLastError(void* ginCtx, bool* hasError) {
  *hasError = false;
  return ncclSuccess;
}

// Not used for rocshmem (device-initiated only)
static ncclResult_t ginRocshmemIput(void*, uint64_t, void*, size_t, uint64_t, void*, uint32_t, int, void**) {
  return ncclInternalError;
}
static ncclResult_t ginRocshmemIputSignal(void*, uint64_t, void*, size_t, uint64_t, void*, uint32_t, uint64_t, void*, uint64_t, uint32_t, int, void**) {
  return ncclInternalError;
}
static ncclResult_t ginRocshmemTest(void*, void*, int*) {
  return ncclInternalError;
}

__attribute__((visibility("default")))
ncclGin_t ncclGinRocshmem = {
  .name            = "rocshmem",
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
  .iput            = ginRocshmemIput,
  .iputSignal      = ginRocshmemIputSignal,
  .test            = ginRocshmemTest,
  .ginProgress     = ginRocshmemGinProgress,
  .queryLastError  = ginRocshmemQueryLastError,
  .finalize        = ginRocshmemFinalize,
};

#endif // ENABLE_ROCSHMEM_GIN
