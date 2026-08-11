/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Minimal external RCCL net plugin used only by NetPluginReloadTests
// (AICOMRCCL-1534). It reports one device so RCCL selects it as the active net
// plugin, and records every real init() call by appending a line to the file
// named in RCCL_NET_RELOAD_COUNTER_FILE. The test counts those lines to detect
// whether the plugin is reloaded after a communicator is destroyed.
//
// Compiled as C++ (the RCCL project only enables CXX/HIP), so the plugin symbol
// is exported with C linkage and default visibility for RCCL's dlsym() lookup.
//
// RCCL_NET_TEST_PLUGIN_MODE=assign_fail selects an incompatible UNPACK device
// version (assignment rejected after init) and records init/finalize via
// RCCL_NET_ASSIGN_FAIL_{INIT,FINALIZE}_FILE (NCCL 2.28.7 net.cc fix).

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "net.h" // from plugins/net/example/nccl

#define __hidden __attribute__((visibility("hidden")))
#define __exported __attribute__((visibility("default")))
#define NCCL_PLUGIN_MAX_RECVS 1

static char kPluginName[] = "ReloadTest";

static void recordLine(const char* envVar) {
  const char* path = getenv(envVar);
  if (path == nullptr) return;

  FILE* f = fopen(path, "a");
  if (f == nullptr) return;

  fputs("1\n", f);
  fclose(f);
}

static int assignFailMode() {
  const char* mode = getenv("RCCL_NET_TEST_PLUGIN_MODE");
  return mode && strcmp(mode, "assign_fail") == 0;
}

__hidden ncclResult_t pluginInit(void** ctx, uint64_t commId, ncclNetCommConfig_v12_t* config,
                                 ncclDebugLogger_t logFunction, ncclProfilerCallback_t profFunction) {
  (void)ctx; (void)commId; (void)config; (void)logFunction; (void)profFunction;
  recordLine("RCCL_NET_RELOAD_COUNTER_FILE");
  recordLine("RCCL_NET_ASSIGN_FAIL_INIT_FILE");
  return ncclSuccess;
}

__hidden ncclResult_t pluginFinalize(void* ctx) {
  (void)ctx;
  recordLine("RCCL_NET_ASSIGN_FAIL_FINALIZE_FILE");
  return ncclSuccess;
}

__hidden ncclResult_t pluginDevices(int* ndev) { *ndev = 1; return ncclSuccess; }

__hidden ncclResult_t pluginGetProperties(int dev, ncclNetProperties_v12_t* props) {
  props->name = kPluginName;
  props->pciPath = nullptr;
  props->guid = 0;
  props->ptrSupport = NCCL_PTR_HOST;
  props->regIsGlobal = 0;
  props->forceFlush = 0;
  props->speed = 100000;
  props->port = 0;
  props->latency = 0;
  props->maxComms = 1024 * 1024;
  props->maxRecvs = NCCL_PLUGIN_MAX_RECVS;
  if (assignFailMode()) {
    props->netDeviceType = NCCL_NET_DEVICE_UNPACK;
    props->netDeviceVersion = NCCL_NET_DEVICE_UNPACK_VERSION - 1;
  } else {
    props->netDeviceType = NCCL_NET_DEVICE_HOST;
    props->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
  }
  props->vProps.ndevs = 1;
  props->vProps.devs[0] = dev;
  props->maxP2pBytes = NCCL_MAX_NET_SIZE_BYTES;
  props->maxCollBytes = NCCL_MAX_NET_SIZE_BYTES;
  props->maxMultiRequestSize = 1;
  props->railId = NCCL_NET_ID_UNDEF;
  props->planeId = NCCL_NET_ID_UNDEF;
  return ncclSuccess;
}

__hidden ncclResult_t pluginListen(void* ctx, int dev, void* handle, void** listenComm) { return ncclInternalError; }
__hidden ncclResult_t pluginConnect(void* ctx, int dev, void* handle, void** sendComm, ncclNetDeviceHandle_v12_t** sendDevComm) { return ncclInternalError; }
__hidden ncclResult_t pluginAccept(void* listenComm, void** recvComm, ncclNetDeviceHandle_v12_t** recvDevComm) { return ncclInternalError; }
__hidden ncclResult_t pluginRegMr(void* comm, void* data, size_t size, int type, void** mhandle) { return ncclInternalError; }
__hidden ncclResult_t pluginRegMrDmaBuf(void* comm, void* data, size_t size, int type, uint64_t offset, int fd, void** mhandle) { return ncclInternalError; }
__hidden ncclResult_t pluginDeregMr(void* comm, void* mhandle) { return ncclInternalError; }
__hidden ncclResult_t pluginIsend(void* sendComm, void* data, size_t size, int tag, void* mhandle, void* phandle, void** request) { return ncclInternalError; }
__hidden ncclResult_t pluginIrecv(void* recvComm, int n, void** data, size_t* sizes, int* tags, void** mhandles, void** phandles, void** request) { return ncclInternalError; }
__hidden ncclResult_t pluginIflush(void* recvComm, int n, void** data, int* sizes, void** mhandles, void** request) { return ncclInternalError; }
__hidden ncclResult_t pluginTest(void* request, int* done, int* size) { return ncclInternalError; }
__hidden ncclResult_t pluginCloseSend(void* sendComm) { return ncclInternalError; }
__hidden ncclResult_t pluginCloseRecv(void* recvComm) { return ncclInternalError; }
__hidden ncclResult_t pluginCloseListen(void* listenComm) { return ncclInternalError; }
__hidden ncclResult_t pluginIrecvConsumed(void* recvComm, int n, void* request) { return ncclInternalError; }
__hidden ncclResult_t pluginGetDeviceMr(void* comm, void* mhandle, void** dptr_mhandle) { return ncclInternalError; }
__hidden ncclResult_t pluginMakeVDevice(int* d, ncclNetVDeviceProps_v12_t* props) { return ncclInternalError; }
__hidden ncclResult_t pluginSetNetAttr(void* ctx, ncclNetAttr_v12_t* netAttr) { return ncclSuccess; }

extern "C" __exported const ncclNet_v12_t ncclNetPlugin_v12 = {
  kPluginName,
  pluginInit,
  pluginDevices,
  pluginGetProperties,
  pluginListen,
  pluginConnect,
  pluginAccept,
  pluginRegMr,
  pluginRegMrDmaBuf,
  pluginDeregMr,
  pluginIsend,
  pluginIrecv,
  pluginIflush,
  pluginTest,
  pluginCloseSend,
  pluginCloseRecv,
  pluginCloseListen,
  pluginGetDeviceMr,
  pluginIrecvConsumed,
  pluginMakeVDevice,
  pluginFinalize,
  pluginSetNetAttr,
};
