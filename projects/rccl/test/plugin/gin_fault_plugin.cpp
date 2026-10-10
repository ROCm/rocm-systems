/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// GIN plugin stub for GinConnectLeakMPITest: fails the Nth connect() and records connect/close counts.

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "nccl_net.h"
#include "gin/gin_v14.h"

#define __hidden __attribute__((visibility("hidden")))

static int ginConnectCalls = 0;
static int ginCloseCollCalls = 0;
static int ginCloseListenCalls = 0;

static int ginEnvInt(const char* name, int fallback) {
  const char* v = getenv(name);
  return (v != nullptr && *v != '\0') ? atoi(v) : fallback;
}

static void ginWriteCounters() {
  const char* path = getenv("RCCL_TEST_GIN_FAULT_COUNTER_FILE");
  if (path == nullptr || *path == '\0') return;
  int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_NOFOLLOW, 0600);
  if (fd < 0) return;
  FILE* f = fdopen(fd, "w");
  if (f == nullptr) {
    close(fd);
    return;
  }
  fprintf(f, "connect %d\ncloseColl %d\ncloseListen %d\n", ginConnectCalls, ginCloseCollCalls,
          ginCloseListenCalls);
  fclose(f);
}

__hidden ncclResult_t ginInit(void** ctx, uint64_t commId, ncclDebugLogger_t logFunction) {
  // init() runs once per communicator, so each one counts its connects from 1.
  ginConnectCalls = ginCloseCollCalls = ginCloseListenCalls = 0;
  *ctx = calloc(1, 1);
  return *ctx ? ncclSuccess : ncclSystemError;
}

__hidden ncclResult_t ginDevices(int* ndev) {
  *ndev = 1;
  return ncclSuccess;
}

__hidden ncclResult_t ginGetGinProperties(ncclGinProperties_v14_t* ginProps) {
  ginProps->supportsStrongSignals = true;
  ginProps->supportsVASignals = true;
  return ncclSuccess;
}

__hidden ncclResult_t ginGetProperties(int dev, ncclNetProperties_v12_t* props) {
  memset(props, 0, sizeof(*props));
  props->name = (char*)"GIN Fault";
  props->ptrSupport = NCCL_PTR_CUDA;
  props->speed = 100000;
  props->maxComms = 1024 * 1024;
  props->maxRecvs = 1;
  // GPI, not PROXY: external PROXY plugins lose to the built-in proxy, and no in-tree backend claims GPI.
  props->netDeviceType = NCCL_NET_DEVICE_GIN_GPI;
  props->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
  props->vProps.ndevs = 1;
  props->vProps.devs[0] = dev;
  props->maxP2pBytes = 1L << 30;
  props->maxCollBytes = 1L << 30;
  props->maxMultiRequestSize = 1;
  return ncclSuccess;
}

__hidden ncclResult_t ginListen(void* ctx, int dev, void* handle, void** listenComm) {
  memset(handle, 0, NCCL_NET_HANDLE_MAXSIZE);
  *listenComm = calloc(1, 1);
  return *listenComm ? ncclSuccess : ncclSystemError;
}

__hidden ncclResult_t ginConnect(void* ctx, void* handles[], int nranks, int rank, void* listenComm,
                                 void** collComm) {
  ginConnectCalls++;
  if (ginConnectCalls == ginEnvInt("RCCL_TEST_GIN_FAULT_FAIL_CONNECT_AT", 0)) {
    ginWriteCounters();
    return ncclSystemError;
  }
  *collComm = calloc(1, 1);
  return *collComm ? ncclSuccess : ncclSystemError;
}

__hidden ncclResult_t ginCloseColl(void* collComm) {
  ginCloseCollCalls++;
  ginWriteCounters();
  free(collComm);
  return ncclSuccess;
}

__hidden ncclResult_t ginCloseListen(void* listenComm) {
  ginCloseListenCalls++;
  ginWriteCounters();
  free(listenComm);
  return ncclSuccess;
}

__hidden ncclResult_t ginFinalize(void* ctx) {
  free(ctx);
  return ncclSuccess;
}

// Setup fails before any context or registration exists, so nothing past connect() is reachable.
__hidden ncclResult_t ginCreateContext(void*, ncclGinConfig_v14_t*, void**, ncclNetDeviceHandle_v11_t**) {
  return ncclInternalError;
}
__hidden ncclResult_t ginRegMrSym(void*, void*, size_t, int, uint64_t, void**, void**) { return ncclInternalError; }
__hidden ncclResult_t ginRegMrSymDmaBuf(void*, void*, size_t, int, uint64_t, int, uint64_t, void**, void**) {
  return ncclInternalError;
}
__hidden ncclResult_t ginDeregMrSym(void*, void*) { return ncclInternalError; }
__hidden ncclResult_t ginDestroyContext(void*) { return ncclInternalError; }
__hidden ncclResult_t ginProgress(void*) { return ncclInternalError; }
__hidden ncclResult_t ginQueryLastError(void*, bool* hasError) {
  *hasError = false;
  return ncclSuccess;
}

extern "C" __attribute__((visibility("default"))) const ncclGin_v14_t ncclGinPlugin_v14 = {
  .name = "GinFault",
  .init = ginInit,
  .devices = ginDevices,
  .getGinProperties = ginGetGinProperties,
  .getProperties = ginGetProperties,
  .listen = ginListen,
  .connect = ginConnect,
  .createContext = ginCreateContext,
  .regMrSym = ginRegMrSym,
  .regMrSymDmaBuf = ginRegMrSymDmaBuf,
  .deregMrSym = ginDeregMrSym,
  .destroyContext = ginDestroyContext,
  .closeColl = ginCloseColl,
  .closeListen = ginCloseListen,
  .ginProgress = ginProgress,
  .queryLastError = ginQueryLastError,
  .finalize = ginFinalize,
};
