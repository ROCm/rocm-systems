// RCCL_COMPAT_STUB: aggregated stubs for NCCL upstream symbols not yet
// implemented in RCCL. Each function/object below is tagged with
// RCCL_COMPAT_STUB on its closing brace for traceability.

#include "scheduler.h"
#include "socket.h"
#include "param.h"
#include "env.h"
#include "plugin/plugin.h"
#include "dev_runtime.h"
#include "gin.h"
#include "gin/gin_host.h"
#include "rma/rma.h"
#include "rma/rma_proxy.h"
#include "rma/rma_ce.h"
#include "enqueue.h"
#include "os.h"

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <dlfcn.h>

ncclResult_t ncclScheduleBcastTasksToPlan(
    struct ncclComm*, struct ncclKernelPlan*, struct ncclKernelPlanBudget*
  ) {
  return ncclInternalError;
} // RCCL_COMPAT_STUB

ncclResult_t ncclGetAlgoInfo(
    struct ncclComm*, struct ncclTaskColl*,
    int, int, int, ncclSimInfo_t*
  ) {
  return ncclInternalError;
} // RCCL_COMPAT_STUB

const char* ncclVersionToString(int version, char* buf, size_t bufLen) {
  int major = version / 10000;
  int minor = (version / 100) % 100;
  int patch = version % 100;
  snprintf(buf, bufLen, "%d.%d.%d", major, minor, patch);
  return buf;
} // RCCL_COMPAT_STUB

// GIN host stubs - NVIDIA-only feature, init must succeed as no-op
ncclResult_t ncclGinInit(struct ncclComm*) { return ncclSuccess; } // RCCL_COMPAT_STUB
ncclResult_t ncclGinInitFromParent(struct ncclComm*, struct ncclComm*) { return ncclSuccess; } // RCCL_COMPAT_STUB
ncclResult_t ncclGinGetDevCount(int, int* nPhysDev, int* nVirtDev) {
  if (nPhysDev) *nPhysDev = 0;
  if (nVirtDev) *nVirtDev = 0;
  return ncclSuccess;
} // RCCL_COMPAT_STUB
ncclResult_t ncclGinHostFinalize(struct ncclComm*) { return ncclSuccess; } // RCCL_COMPAT_STUB
ncclResult_t getGlobalGinType(struct ncclComm*, ncclGinType_t* ginType) {
  if (ginType) *ginType = NCCL_GIN_TYPE_NONE;
  return ncclSuccess;
} // RCCL_COMPAT_STUB
ncclResult_t getGlobalRailedGinType(struct ncclComm*, ncclGinType_t* ginType) {
  if (ginType) *ginType = NCCL_GIN_TYPE_NONE;
  return ncclSuccess;
} // RCCL_COMPAT_STUB
ncclResult_t ncclGinConnectOnce(struct ncclComm*, ncclGinConnectionType_t, int, int) { return ncclInternalError; } // RCCL_COMPAT_STUB
ncclResult_t ncclGinRegister(struct ncclComm*, void*, size_t,
                             void* [NCCL_GIN_MAX_CONNECTIONS],
                             ncclGinWindow_t [NCCL_GIN_MAX_CONNECTIONS], int) { return ncclInternalError; } // RCCL_COMPAT_STUB
ncclResult_t ncclGinDeregister(struct ncclComm*, void* [NCCL_GIN_MAX_CONNECTIONS]) { return ncclSuccess; } // RCCL_COMPAT_STUB
ncclResult_t ncclGinAllocSignalsCounters(struct ncclComm*, int, uint32_t* outSignal0,
                                         int, uint32_t* outCounter0) {
  if (outSignal0) *outSignal0 = 0;
  if (outCounter0) *outCounter0 = 0;
  return ncclInternalError;
} // RCCL_COMPAT_STUB
ncclResult_t ncclGinFreeSignalsCounters(struct ncclComm*, uint32_t, int, uint32_t, int) { return ncclSuccess; } // RCCL_COMPAT_STUB
ncclResult_t ncclGinQueryLastError(struct ncclGinState*, bool* hasError) {
  if (hasError) *hasError = false;
  return ncclSuccess;
} // RCCL_COMPAT_STUB

// GIN GDAKI stubs - referenced by net_ib.cc functions even though we NULLed the
// vtable entries; provide no-op stubs so the symbols resolve.
ncclResult_t ncclGinGdakiRegMrSym(void*, void*, size_t, int, uint64_t, void**, void**) { return ncclInternalError; } // RCCL_COMPAT_STUB
ncclResult_t ncclGinGdakiCreateContext(void*, int, int, int, int, int, void**, ncclNetDeviceHandle_v11_t**) { return ncclInternalError; } // RCCL_COMPAT_STUB
ncclResult_t ncclGinGdakiDeregMrSym(void*, void*) { return ncclSuccess; } // RCCL_COMPAT_STUB
ncclResult_t ncclGinGdakiDestroyContext(void*) { return ncclSuccess; } // RCCL_COMPAT_STUB
ncclResult_t ncclGinGdakiProgress(void*) { return ncclSuccess; } // RCCL_COMPAT_STUB
ncclResult_t ncclGinGdakiQueryLastError(void*, bool* hasError) {
  if (hasError) *hasError = false;
  return ncclSuccess;
} // RCCL_COMPAT_STUB

// RMA stubs - new in NCCL 2.30, NVIDIA RDMA path
ncclResult_t ncclLaunchRma(struct ncclComm*, struct ncclKernelPlan*) { return ncclInternalError; } // RCCL_COMPAT_STUB
ncclResult_t scheduleRmaTasksToPlan(struct ncclComm*, struct ncclKernelPlan*) { return ncclInternalError; } // RCCL_COMPAT_STUB
ncclResult_t ncclRmaCeInit(struct ncclComm*) { return ncclSuccess; } // RCCL_COMPAT_STUB
ncclResult_t ncclRmaProxyConnectOnce(struct ncclComm*) { return ncclSuccess; } // RCCL_COMPAT_STUB
ncclResult_t ncclRmaProxyFinalize(struct ncclComm*) { return ncclSuccess; } // RCCL_COMPAT_STUB
ncclResult_t ncclRmaProxyRegister(struct ncclComm*, void*, size_t,
                                  void* [NCCL_GIN_MAX_CONNECTIONS],
                                  ncclGinWindow_t [NCCL_GIN_MAX_CONNECTIONS]) { return ncclInternalError; } // RCCL_COMPAT_STUB
ncclResult_t ncclRmaProxyDeregister(struct ncclComm*, void* [NCCL_GIN_MAX_CONNECTIONS]) { return ncclSuccess; } // RCCL_COMPAT_STUB
