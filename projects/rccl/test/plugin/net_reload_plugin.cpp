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
// The vtable below is built from the vendored example net headers, which are a
// separate copy of the v10 plugin ABI from the one getNcclNet_v10() casts the
// dlsym'd symbol to. Keeping the plugin on v10 is deliberate, and a mismatch
// between the two copies cannot silently weaken the test: a changed struct
// layout makes the loader call through wrong offsets and the test fails, and if
// v10 support is ever dropped the symbol is never looked up, the counter file
// stays empty and the test asserts with an init count of 0.

#include <stdio.h>
#include <stdlib.h>
#include "net.h" // from plugins/net/example/nccl

#define __hidden __attribute__((visibility("hidden")))
#define __exported __attribute__((visibility("default")))
#define NCCL_PLUGIN_MAX_RECVS 1

static char kPluginName[] = "ReloadTest";

static void recordLoad() {
  const char* path = getenv("RCCL_NET_RELOAD_COUNTER_FILE");
  if (path == nullptr) return;

  FILE* f = fopen(path, "a");
  if (f == nullptr) return;

  fputs("1\n", f);
  fclose(f);
}

__hidden ncclResult_t pluginInit_v10(ncclDebugLogger_t logFunction, ncclProfilerCallback_t profFunction) {
  (void)logFunction; (void)profFunction;
  recordLoad();
  return ncclSuccess;
}

__hidden ncclResult_t pluginDevices(int* ndev) { *ndev = 1; return ncclSuccess; }

__hidden ncclResult_t pluginGetProperties_v10(int dev, ncclNetProperties_v10_t* props) {
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
  props->netDeviceType = NCCL_NET_DEVICE_HOST;
  props->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
  props->vProps.ndevs = 1;
  props->vProps.devs[0] = dev;
  props->maxP2pBytes = NCCL_MAX_NET_SIZE_BYTES;
  props->maxCollBytes = NCCL_MAX_NET_SIZE_BYTES;
  return ncclSuccess;
}

__hidden ncclResult_t pluginListen_v10(int d, void* handle, void** listenComm) { return ncclInternalError; }
__hidden ncclResult_t pluginConnect_v10(int dev, ncclNetCommConfig_v10_t* config, void* handle, void** sendComm, ncclNetDeviceHandle_v10_t** sendDevComm) { return ncclInternalError; }
__hidden ncclResult_t pluginAccept(void* listenComm, void** recvComm, ncclNetDeviceHandle_v10_t** recvDevComm) { return ncclInternalError; }
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
__hidden ncclResult_t pluginMakeVDevice_v10(int* d, ncclNetVDeviceProps_v10_t* props) { return ncclInternalError; }

extern "C" __exported const ncclNet_v10_t ncclNetPlugin_v10 = {
  kPluginName,
  pluginInit_v10,
  pluginDevices,
  pluginGetProperties_v10,
  pluginListen_v10,
  pluginConnect_v10,
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
  pluginMakeVDevice_v10,
};
