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

// ncclParam* referenced by init.cc but not declared inside it, so the redirected NCCL_PARAM does not
// cover them. Each stays here rather than moving to its owner's fakes file because that file is on a
// link line whose unit under test already defines the symbol (enqueue.cc:1985 for LaunchOrderImplicit),
// or has no fakes file at all. The trailing comment names the definition each copies.
int64_t ncclParamLaunchOrderImplicit() { return g_loadParam("LAUNCH_ORDER_IMPLICIT", 0); }  // enqueue.cc:1985

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
