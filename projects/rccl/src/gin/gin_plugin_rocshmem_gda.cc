/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifdef ENABLE_ROCSHMEM_GIN

/**
 * Built-in GIN plugin for the rocshmem QP backend.
 *
 * Uses gin_qp_factory directly — no rocshmem_init(), no symmetric heap.
 * QP creation and connection are handled in ncclGinRocshmemGdaCreateContext().
 */

#include "gin/gin_host_rocshmem_gda.h"
#include "comm.h"
#include "nccl_device/gin/rocshmem/gin_rocshmem_device_host_common_gda.h"
#include "plugin/nccl_net.h"

struct ginRocshmemGdaListenCtx {
  int dev;
};

struct ginRocshmemGdaCollCtx {
  int nranks;
  int rank;
};

static ncclResult_t ginRocshmemGdaInit(void** ctx, uint64_t commId, ncclDebugLogger_t logFunction) {
  *ctx = nullptr;
  return ncclSuccess;
}

static ncclResult_t ginRocshmemGdaDevices(int* ndev) {
  *ndev = 1;
  return ncclSuccess;
}

static ncclResult_t ginRocshmemGdaGetProperties(int dev, ncclNetProperties_v11_t* props) {
  memset(props, 0, sizeof(*props));
  props->name = const_cast<char*>("rocshmem_gda");
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
  auto* cctx = new ginRocshmemGdaCollCtx;
  cctx->nranks = nranks;
  cctx->rank = rank;
  *collComm = cctx;
  return ncclSuccess;
}

static ncclResult_t ginRocshmemGdaCloseListen(void* listenComm) {
  delete (ginRocshmemGdaListenCtx*)listenComm;
  return ncclSuccess;
}

static ncclResult_t ginRocshmemGdaCloseColl(void* collComm) {
  delete (ginRocshmemGdaCollCtx*)collComm;
  return ncclSuccess;
}

static ncclResult_t ginRocshmemGdaFinalize(void* ctx) {
  return ncclSuccess;
}

static ncclResult_t ginRocshmemGdaCreateContext(void* collComm, int nSignals, int nCounters,
                                                void** ginCtx, ncclNetDeviceHandle_v11_t** devHandle) {
  return ncclGinRocshmemGdaCreateContextFromPlugin(nSignals, nCounters, ginCtx, devHandle);
}

static ncclResult_t ginRocshmemGdaRegMrSym(void* collComm, void* data, size_t size,
                                           int type, uint64_t mrFlags,
                                           void** mhandle, void** ginHandle) {
  return ncclGinRocshmemGdaRegisterFromPlugin(data, size, type, mrFlags, mhandle, ginHandle);
}

static ncclResult_t ginRocshmemGdaRegMrSymDmaBuf(void* collComm, void* data, size_t size,
                                                  int type, uint64_t offset, int fd,
                                                  uint64_t mrFlags, void** mhandle, void** ginHandle) {
  return ginRocshmemGdaRegMrSym(collComm, data, size, type, mrFlags, mhandle, ginHandle);
}

static ncclResult_t ginRocshmemGdaDeregMrSym(void* collComm, void* mhandle) {
  return ncclGinRocshmemGdaDeregisterFromPlugin(mhandle);
}

static ncclResult_t ginRocshmemGdaDestroyContext(void* ginCtx) {
  return ncclGinRocshmemGdaDestroyContextFromPlugin(ginCtx);
}

static ncclResult_t ginRocshmemGdaGinProgress(void* collComm) {
  return ncclSuccess;
}

static ncclResult_t ginRocshmemGdaQueryLastError(void* ginCtx, bool* hasError) {
  *hasError = false;
  return ncclSuccess;
}

static ncclResult_t ginRocshmemGdaIput(void*, uint64_t, void*, size_t, uint64_t, void*, uint32_t, void**) {
  return ncclInternalError;
}
static ncclResult_t ginRocshmemGdaIputSignal(void*, uint64_t, void*, size_t, uint64_t, void*, uint32_t, uint64_t, void*, uint64_t, uint32_t, void**) {
  return ncclInternalError;
}
static ncclResult_t ginRocshmemGdaTest(void*, void*, int*) {
  return ncclInternalError;
}

__attribute__((visibility("default")))
ncclGin_v11_t ncclGinRocshmemGdaPlugin = {
  .name            = "rocshmem_gda",
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
  .iput            = ginRocshmemGdaIput,
  .iputSignal      = ginRocshmemGdaIputSignal,
  .test            = ginRocshmemGdaTest,
  .ginProgress     = ginRocshmemGdaGinProgress,
  .queryLastError  = ginRocshmemGdaQueryLastError,
  .finalize        = ginRocshmemGdaFinalize,
};

#endif // ENABLE_ROCSHMEM_GIN
