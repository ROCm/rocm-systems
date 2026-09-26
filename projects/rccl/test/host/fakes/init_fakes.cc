/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// What is left after every seam moved to the fakes file named for its owning production TU: the
// src/init.cc-only seams, the NCCL_PARAMs whose owning TU has no fakes file on this link line, and
// the Reset/Install chains. See init_fakes.h.

#include "init_fakes.h"

#include <cstdint>
#include <cstring>

#include "recorder.h"

// micro_getenv / SetMicroEnv / ClearMicroEnv / the getenv interposer / ncclGetEnv
// moved to env_fakes.cc so every microtest binary shares ONE env implementation:
// a second, map-only copy cannot intercept production raw getenv() call sites.

int64_t ncclParamEnqueueRearchEnable() { return g_loadParam("ENQUEUE_REARCH_ENABLE", 0); }

int64_t ncclParamRasDiagnostics() { return g_loadParam("RUN_RAS_DIAGNOSTICS", 0); }
int64_t ncclParamDiagnostics() { return g_loadParam("RUN_DIAGNOSTICS", 0); }
// src/ras/ras.cc (NCCL 2.32): init.cc gates the progress-counter monitor on RAS being enabled.
int64_t ncclParamRasEnable() { return g_loadParam("RAS_ENABLE", 1); }

// src/cft_dev_runtime.cc: CFT needs CUDA >= 13.3 logical endpoints, so on HIP every capability is off.
ncclResult_t ncclGpuCftSupport(struct ncclComm*, int* gpuCftSupport, bool* gpuCftMulticastSupport,
                               bool* gpuCftCountedSupport) {
  *gpuCftSupport = 0;
  *gpuCftMulticastSupport = false;
  *gpuCftCountedSupport = false;
  return ncclSuccess;
}

// src/ras/progress_monitor.cc (NCCL 2.32). NCCL_PROGRESS_COUNTERS defaults to 0, so the suites here never
// start a monitor; Destroy runs on every comm teardown and must succeed.
ncclResult_t ncclProgressCounterMonitorInit(struct ncclComm*) { return ncclSuccess; }
ncclResult_t ncclProgressCounterMonitorDestroy(struct ncclComm*) { return ncclSuccess; }

// src/device/common.cu's __global__ ncclProgressCounterCaptureGpuTime. init.cc only takes its address to hand
// to the (faked) kernel launch, so a host function with the same mangled name satisfies the link.
void ncclProgressCounterCaptureGpuTime(uint64_t*) {}
int64_t rcclParamIntraGraphGen() { return g_loadParam("INTRA_GRAPH_GEN", 0); }

// Dead seam: no src/*.cc defines ncclTopoGetStrFromSys and no unit under test calls it. Kept as-is
// rather than deleted, since removing it is a behaviour question this move is not answering.
ncclResult_t ncclTopoGetStrFromSys(const char* /*path*/, const char* fileName, char* strValue) {
  if (!strValue) return ncclSuccess;
  if (fileName && std::strcmp(fileName, "version") == 0)
    std::strcpy(strValue, "Linux version 6.8.0-microtest");
  else if (fileName && std::strcmp(fileName, "numa_balancing") == 0)
    std::strcpy(strValue, "0");
  else
    std::strcpy(strValue, "microtest");
  return ncclSuccess;
}

// ncclNetInit/ncclNetInitFromParent live in init-test.cc: they need the full ncclComm/ncclNet_t layout.
ncclResult_t g_ncclNetInitResult = ncclSuccess;

void InstallCommAllocSuccess() {
  g_ncclNetInitResult = ncclSuccess;
  g_ncclGinInitResult = ncclSuccess;
  g_ncclStrongStreamResult = ncclSuccess;
  g_ncclMemManagerInitResult = ncclSuccess;
  g_amdSmiInitResult = ncclSuccess;
  g_hipDeviceGetAttributeResult = hipSuccess;
  g_hipDeviceGetPCIBusIdResult  = hipSuccess;
  g_hipEventCreateResult        = hipSuccess;
  g_hipMemPoolResult            = hipSuccess;
  g_hipStreamCreateResult       = hipSuccess;
}

void InstallDevCommSetupSuccess() {
  InstallCommAllocSuccess();
  g_hipAsyncOpsResult = hipSuccess;
}

void ResetInitFakes() {
  ResetAmdSmiFakes();
  ResetBootstrapStubs();
  ResetEnqueueFakes();
  ResetEnvFakes();
  ResetEnvPluginFakes();
  ResetGinFakes();
  ResetGroupFakes();
  ResetHipFakes();
  ResetLibcInterposers();
  ResetMemManagerFakes();
  ResetNcclFakes();
  ResetNcclStubs();
  ResetOsFakes();
  ResetRcclWrapFakes();
  ResetRecorderFakes();
  ResetRocmWrapFakes();
  ResetStrongStreamStubs();
  ResetTopoStubs();
  ResetTransportStubs();
  ResetTuningFakes();
  g_ncclNetInitResult = ncclSuccess;
}
