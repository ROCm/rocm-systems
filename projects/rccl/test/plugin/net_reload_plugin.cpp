/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Minimal external RCCL test plugins. This single translation unit hosts two
// independent, self-contained plugins that share the same reload-counter
// mechanism (recordLine appends "1\n" to the file named in an env var so a test
// can count how many times a callback fired):
//
//   1. Net plugin (ncclNetPlugin_v12) used by NetPluginReloadTests
//      (AICOMRCCL-1534). It reports one device so RCCL selects it as the active
//      net plugin, and records every real init() call via
//      RCCL_NET_RELOAD_COUNTER_FILE. The test counts those lines to detect
//      whether the plugin is reloaded after a communicator is destroyed.
//      RCCL_NET_TEST_PLUGIN_MODE=assign_fail selects an incompatible UNPACK
//      device version (assignment rejected after init) and records
//      init/finalize via RCCL_NET_ASSIGN_FAIL_{INIT,FINALIZE}_FILE
//      (NCCL 2.28.7 net.cc fix).
//
//   2. One-sided RMA plugin (ncclGinPlugin_v13 / ncclRmaPlugin_v13, an
//      ncclGin_v13_t vtable) used by RmaExternalPluginLoad.* to validate only the
//      loader/assign path in src/plugin/gin.cc: src/plugin/gin/gin_v13.cc resolves
//      the symbols via dlsym, gin.cc promotes the RMA plugin to InitReady, calls
//      init()/devices(), and assigns the vtable to comm->rmaState.rmaProxyState.
//      init() records that it ran via RCCL_RMA_RELOAD_INIT_FILE. The RMA DATA path
//      (iput/iputSignal/iget) is NOT implemented here (a loopback memcpy/atomic
//      cannot service a real cross-rank put/signal): those ops move no data and
//      raise no signal, and the data path is validated separately with a real
//      plugin on real IB. They DO record their invocation via
//      RCCL_RMA_RELOAD_COUNTER_FILE (iputSignal also RCCL_RMA_SIGNAL_COUNTER_FILE)
//      so RmaExternalPluginPutSignal.* can prove the host put/signal APIs reach
//      the plugin primitives (a fire-and-forget check: no wait, no validation).
//
// Compiled as C++ (the RCCL project only enables CXX/HIP), so each plugin symbol
// is exported with C linkage and default visibility for RCCL's dlsym() lookup.
//
// RCCL_NET_TEST_PLUGIN_MODE selects a failure to inject:
//   assign_fail   - incompatible UNPACK device version, so assignment is rejected
//                   after a successful init (NCCL 2.28.7 net.cc fix). Records
//                   init/finalize via RCCL_NET_ASSIGN_FAIL_{INIT,FINALIZE}_FILE.
//   init_fail     - init() reports an error (AICOMRCCL-1891 / NCCL 2.29.7 net.cc
//                   fix: finalize() must not run for an init() that failed).
//   devices_fail  - init() succeeds but devices() reports an error, which is the
//                   companion case where finalize() must still run.
// Both new modes record via RCCL_NET_TEST_{INIT,FINALIZE}_FILE.
//
// Both plugins share the real RCCL plugin headers (nccl_net.h transitively
// provides the v12 net ABI, net_device.h and the GIN proxy constants); this
// keeps the two vtables in one consistent header world so they can live in a
// single translation unit without the example net headers redefining the v12
// net types that gin_v13.h also pulls in.

#include <cstdint>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nccl_net.h"     // ncclNet_v12_t + v12 net ABI, NCCL_NET_DEVICE_*
#include "gin/gin_v13.h"  // ncclGin_v13_t, ncclGinConfig_v13_t

#define __hidden __attribute__((visibility("hidden")))
#define __exported __attribute__((visibility("default")))
#define NCCL_PLUGIN_MAX_RECVS 1

static char kPluginName[] = "ReloadTest";

enum PluginTestMode {
  kModeDefault,
  kModeAssignFail,
  kModeInitFail,
  kModeDevicesFail,
};

static void recordLine(const char* envVar) {
  const char* path = getenv(envVar);
  if (path == nullptr) return;

  FILE* f = fopen(path, "a");
  if (f == nullptr) return;

  fputs("1\n", f);
  fclose(f);
}

static PluginTestMode testMode() {
  const char* mode = getenv("RCCL_NET_TEST_PLUGIN_MODE");
  if (mode == nullptr) return kModeDefault;
  if (strcmp(mode, "assign_fail") == 0) return kModeAssignFail;
  if (strcmp(mode, "init_fail") == 0) return kModeInitFail;
  if (strcmp(mode, "devices_fail") == 0) return kModeDevicesFail;
  return kModeDefault;
}

__hidden ncclResult_t pluginInit(void** ctx, uint64_t commId, ncclNetCommConfig_v12_t* config,
                                 ncclDebugLogger_t logFunction, ncclProfilerCallback_t profFunction) {
  (void)ctx; (void)commId; (void)config; (void)logFunction; (void)profFunction;
  recordLine("RCCL_NET_RELOAD_COUNTER_FILE");
  recordLine("RCCL_NET_ASSIGN_FAIL_INIT_FILE");
  recordLine("RCCL_NET_TEST_INIT_FILE");
  // Recorded before failing, so a test can tell "init was never attempted" apart
  // from "init was attempted and rejected".
  if (testMode() == kModeInitFail) return ncclSystemError;
  return ncclSuccess;
}

__hidden ncclResult_t pluginFinalize(void* ctx) {
  (void)ctx;
  recordLine("RCCL_NET_ASSIGN_FAIL_FINALIZE_FILE");
  recordLine("RCCL_NET_TEST_FINALIZE_FILE");
  return ncclSuccess;
}

__hidden ncclResult_t pluginDevices(int* ndev) {
  if (testMode() == kModeDevicesFail) return ncclSystemError;
  *ndev = 1;
  return ncclSuccess;
}

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
  if (testMode() == kModeAssignFail) {
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

// ---------------------------------------------------------------------------
// One-sided RMA plugin (ncclGinPlugin_v13 / ncclRmaPlugin_v13)
// ---------------------------------------------------------------------------
//
static char kRmaPluginName[] = "RmaReloadStub";

struct RmaStubMr {
  void* base;
  size_t size;
};

struct RmaStubComm {
  int rank;
  int nranks;
};

struct RmaStubCtx {
  RmaStubComm* comm;
};

static int gRmaRequestSentinel = 0;

__hidden ncclResult_t stubInit(void** ctx, uint64_t /*commId*/, ncclDebugLogger_t /*logFunction*/) {
  recordLine("RCCL_RMA_RELOAD_INIT_FILE");
  if (ctx) *ctx = new RmaStubComm{0, 1};
  return ncclSuccess;
}

__hidden ncclResult_t stubDevices(int* ndev) {
  if (ndev) *ndev = 1;
  return ncclSuccess;
}

__hidden ncclResult_t stubGetProperties(int dev, ncclNetProperties_v12_t* props) {
  memset(props, 0, sizeof(*props));
  props->name = kRmaPluginName;
  props->pciPath = nullptr;
  props->guid = 0;
  props->ptrSupport = NCCL_PTR_HOST | NCCL_PTR_CUDA;
  props->regIsGlobal = 0;
  props->forceFlush = 0;
  props->speed = 100000;
  props->port = 0;
  props->latency = 0;
  props->maxComms = 1024 * 1024;
  props->maxRecvs = 1;
  // Selecting the proxy backend is gated on this device type.
  props->netDeviceType = NCCL_NET_DEVICE_GIN_PROXY;
  props->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
  props->maxP2pBytes = NCCL_MAX_NET_SIZE_BYTES;
  props->maxCollBytes = NCCL_MAX_NET_SIZE_BYTES;
  props->maxMultiRequestSize = 1;
  props->vProps.ndevs = 1;
  props->vProps.devs[0] = dev;
  props->railId = NCCL_NET_ID_UNDEF;
  props->planeId = NCCL_NET_ID_UNDEF;
  return ncclSuccess;
}

__hidden ncclResult_t stubListen(void* /*ctx*/, int /*dev*/, void* handle, void** listenComm) {
  if (handle) memset(handle, 0, NCCL_NET_HANDLE_MAXSIZE);
  if (listenComm) *listenComm = new RmaStubComm{0, 1};
  return ncclSuccess;
}

__hidden ncclResult_t stubConnect(void* /*ctx*/, void* /*handles*/[], int nranks, int rank, void* /*listenComm*/,
                                  void** collComm) {
  if (collComm) *collComm = new RmaStubComm{rank, nranks};
  return ncclSuccess;
}

__hidden ncclResult_t stubCreateContext(void* collComm, ncclGinConfig_v13_t* /*config*/, void** ginCtx,
                                        ncclNetDeviceHandle_v11_t** devHandle) {
  if (ginCtx) *ginCtx = new RmaStubCtx{static_cast<RmaStubComm*>(collComm)};
  if (devHandle) *devHandle = nullptr;
  return ncclSuccess;
}

__hidden ncclResult_t stubRegMrSym(void* /*collComm*/, void* data, size_t size, int /*type*/, uint64_t /*mrFlags*/,
                                   void** mhandle, void** ginHandle) {
  RmaStubMr* mr = new RmaStubMr{data, size};
  if (mhandle) *mhandle = mr;
  if (ginHandle) *ginHandle = mr;
  return ncclSuccess;
}

__hidden ncclResult_t stubRegMrSymDmaBuf(void* collComm, void* data, size_t size, int type, uint64_t /*offset*/,
                                         int /*fd*/, uint64_t mrFlags, void** mhandle, void** ginHandle) {
  return stubRegMrSym(collComm, data, size, type, mrFlags, mhandle, ginHandle);
}

__hidden ncclResult_t stubDeregMrSym(void* /*collComm*/, void* mhandle) {
  delete static_cast<RmaStubMr*>(mhandle);
  return ncclSuccess;
}

__hidden ncclResult_t stubDestroyContext(void* ginCtx) {
  delete static_cast<RmaStubCtx*>(ginCtx);
  return ncclSuccess;
}

__hidden ncclResult_t stubCloseColl(void* collComm) {
  delete static_cast<RmaStubComm*>(collComm);
  return ncclSuccess;
}

__hidden ncclResult_t stubCloseListen(void* listenComm) {
  delete static_cast<RmaStubComm*>(listenComm);
  return ncclSuccess;
}

// Data ops move NO data and raise NO signal
__hidden ncclResult_t stubIput(void* /*ginCtx*/, int /*context*/, uint64_t /*srcOff*/, void* /*srcMhandle*/,
                               size_t /*size*/, uint64_t /*dstOff*/, void* /*dstMhandle*/, uint32_t /*rank*/,
                               void** request) {
  recordLine("RCCL_RMA_RELOAD_COUNTER_FILE");
  if (request) *request = &gRmaRequestSentinel;
  return ncclSuccess;
}

__hidden ncclResult_t stubIputSignal(void* /*ginCtx*/, int /*context*/, uint64_t /*srcOff*/, void* /*srcMhandle*/,
                                     size_t /*size*/, uint64_t /*dstOff*/, void* /*dstMhandle*/, uint32_t /*rank*/,
                                     uint64_t /*signalOff*/, void* /*signalMhandle*/, uint64_t /*signalValue*/,
                                     uint32_t /*signalOp*/, void** request) {
  recordLine("RCCL_RMA_RELOAD_COUNTER_FILE");
  recordLine("RCCL_RMA_SIGNAL_COUNTER_FILE");
  if (request) *request = &gRmaRequestSentinel;
  return ncclSuccess;
}

__hidden ncclResult_t stubIget(void* /*ginCtx*/, int /*context*/, uint64_t /*remoteOff*/, void* /*remoteMhandle*/,
                               size_t /*size*/, uint64_t /*localOff*/, void* /*localMhandle*/, uint32_t /*rank*/,
                               void** request) {
  recordLine("RCCL_RMA_RELOAD_COUNTER_FILE");
  if (request) *request = &gRmaRequestSentinel;
  return ncclSuccess;
}

__hidden ncclResult_t stubIflush(void* /*ginCtx*/, int /*context*/, void* /*mhandle*/, uint32_t /*rank*/,
                                 void** request) {
  if (request) *request = &gRmaRequestSentinel;
  return ncclSuccess;
}

// Every op is synchronous, so requests are always complete.
__hidden ncclResult_t stubTest(void* /*collComm*/, void* /*request*/, int* done) {
  if (done) *done = 1;
  return ncclSuccess;
}

__hidden ncclResult_t stubGinProgress(void* /*ginCtx*/) { return ncclSuccess; }

__hidden ncclResult_t stubQueryLastError(void* /*ginCtx*/, bool* hasError) {
  if (hasError) *hasError = false;
  return ncclSuccess;
}

__hidden ncclResult_t stubFinalize(void* ctx) {
  delete static_cast<RmaStubComm*>(ctx);
  return ncclSuccess;
}

#define RMA_RELOAD_STUB_VTABLE_FIELDS \
  kRmaPluginName, \
  stubInit, \
  stubDevices, \
  stubGetProperties, \
  stubListen, \
  stubConnect, \
  stubCreateContext, \
  stubRegMrSym, \
  stubRegMrSymDmaBuf, \
  stubDeregMrSym, \
  stubDestroyContext, \
  stubCloseColl, \
  stubCloseListen, \
  stubIput, \
  stubIputSignal, \
  stubIget, \
  stubIflush, \
  stubTest, \
  stubGinProgress, \
  stubQueryLastError, \
  stubFinalize

extern "C" __exported const ncclGin_v13_t ncclGinPlugin_v13 = { RMA_RELOAD_STUB_VTABLE_FIELDS };
extern "C" __exported const ncclGin_v13_t ncclRmaPlugin_v13 = { RMA_RELOAD_STUB_VTABLE_FIELDS };
