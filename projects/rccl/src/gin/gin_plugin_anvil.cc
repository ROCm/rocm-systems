/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifdef ENABLE_ROCSHMEM_GIN

#include "gin/gin_host_anvil.h"
#include "comm.h"

// Plugin interface for Anvil: transport is already "there" (intra-node SDMA),
// so listen/connect/closeListen are stubs like the rocshmem plugin.
#include "nccl_device/gin/anvil/gin_anvil_device_host_common.h"

static ncclResult_t ncclGinAnvilInit(void** ctx, uint64_t, ncclDebugLogger_t) {
  *ctx = nullptr;
  return ncclSuccess;
}

static ncclResult_t ncclGinAnvilDevices(int* ndev) {
  *ndev = 1;
  return ncclSuccess;
}

static ncclResult_t ncclGinAnvilGetProperties(int, ncclNetProperties_t* props) {
  memset(props, 0, sizeof(*props));
  props->name = (char*)"anvil";
  props->ptrSupport = NCCL_PTR_CUDA;
  props->netDeviceType = NCCL_NET_DEVICE_GIN_ANVIL;
  props->netDeviceVersion = NCCL_GIN_ANVIL_VERSION;
  return ncclSuccess;
}

static ncclResult_t ncclGinAnvilListen(void*, int, void*, void** listenComm) {
  *listenComm = (void*)0x1;
  return ncclSuccess;
}

static ncclResult_t ncclGinAnvilConnect(void*, void*[], int, int, int, int, void*, void** collComm) {
  *collComm = (void*)0x1;
  return ncclSuccess;
}

static ncclResult_t ncclGinAnvilCloseListen(void*) { return ncclSuccess; }
static ncclResult_t ncclGinAnvilCloseColl(void*) { return ncclSuccess; }
static ncclResult_t ncclGinAnvilFinalize(void*) { return ncclSuccess; }

ncclGin_t ncclGinAnvilPlugin = {
  "GIN_ANVIL",
  ncclGinAnvilInit,
  ncclGinAnvilDevices,
  ncclGinAnvilGetProperties,
  ncclGinAnvilListen,
  ncclGinAnvilConnect,
  NULL, // createContext handled by ncclGinAnvilCreateContext in gin_host.cc
  NULL, // regMrSym handled by ncclGinAnvilRegister in gin_host.cc
  NULL,
  NULL, // deregMrSym handled by ncclGinAnvilDeregister in gin_host.cc
  NULL, // destroyContext handled by ncclGinAnvilDestroyContext in gin_host.cc
  ncclGinAnvilCloseColl,
  ncclGinAnvilCloseListen,
  NULL,
  NULL,
  NULL,
  NULL,
  NULL,
  ncclGinAnvilFinalize
};

#endif // ENABLE_ROCSHMEM_GIN

