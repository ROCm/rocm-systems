/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Definitions behind DdaCollSeams.h, plus the HIP kernel registration a TU defining __global__ functions emits.

#include "algorithms/dda/DdaCollSeams.h"

#include "fakes/param_redirect.h"
#include "rccl_common.h"

std::vector<DdaLaunch> g_ddaLaunches;

RCCL_PARAM(DdaLL, "DDA_LL", 1);
RCCL_PARAM(DdaLLOneShotThreshold, "DDA_LL_ONESHOT_THRESHOLD", (size_t)(1) * 1024 * 1024);
RCCL_PARAM(DdaLL128OneShotThreshold, "DDA_LL128_ONESHOT_THRESHOLD", (size_t)(4) * 1024 * 1024);

static size_t DefaultDdaLLThreshold(const ncclComm*, ncclFunc_t) { return kDdaLLBaseDefault; }
static size_t DefaultDdaLL128Threshold(const ncclComm*, ncclFunc_t) { return 0; }  // DDA_LL128 unset, no table: off
static size_t DefaultDdaLLTwoShotThreshold(const ncclComm*) { return (size_t)(2) * 1024 * 1024; }
static size_t DefaultDdaLL128TwoShotThreshold(const ncclComm*) { return (size_t)(64) * 1024 * 1024; }
static int DefaultDdaFabricVmmBlocks(const ncclComm* comm) { return comm != nullptr ? comm->ddaFabricMaxBlocks : 0; }
static int DefaultDdaFabricAgLL128Threads(const ncclComm*, size_t) { return 0; }  // no table: 512 threads
static int DefaultDdaFabricAgLL128Blocks(const ncclComm* comm, size_t) {
  return comm != nullptr ? comm->ddaFabricMaxBlocks : 0;
}
static bool DefaultDdaDatatypeSupported(const ncclComm* comm, ncclDataType_t datatype) {
  switch (datatype) {
  case ncclFloat32:
  case ncclFloat16:
  case ncclBfloat16:
    return true;
  case ncclFloat8e4m3:
  case ncclFloat8e5m2:
    return comm != nullptr && comm->archName != nullptr && std::strncmp(comm->archName, "gfx1250", 7) == 0;
  default:
    return false;
  }
}

std::function<size_t(const ncclComm*, ncclFunc_t)> g_rcclDdaLLThreshold = DefaultDdaLLThreshold;
std::function<size_t(const ncclComm*, ncclFunc_t)> g_rcclDdaLL128Threshold = DefaultDdaLL128Threshold;
std::function<size_t(const ncclComm*)> g_rcclDdaLLTwoShotThreshold = DefaultDdaLLTwoShotThreshold;
std::function<size_t(const ncclComm*)> g_rcclDdaLL128TwoShotThreshold = DefaultDdaLL128TwoShotThreshold;
std::function<int(const ncclComm*)> g_rcclDdaFabricVmmBlocks = DefaultDdaFabricVmmBlocks;
std::function<int(const ncclComm*, size_t)> g_rcclDdaFabricAgLL128Threads = DefaultDdaFabricAgLL128Threads;
std::function<int(const ncclComm*, size_t)> g_rcclDdaFabricAgLL128Blocks = DefaultDdaFabricAgLL128Blocks;
std::function<bool(const ncclComm*, ncclDataType_t)> g_rcclDdaDatatypeSupported = DefaultDdaDatatypeSupported;

size_t rcclDdaLLThreshold(const ncclComm* comm, ncclFunc_t func) { return g_rcclDdaLLThreshold(comm, func); }
size_t rcclDdaLL128Threshold(const ncclComm* comm, ncclFunc_t func) { return g_rcclDdaLL128Threshold(comm, func); }
size_t rcclDdaLLTwoShotThreshold(const ncclComm* comm) { return g_rcclDdaLLTwoShotThreshold(comm); }
size_t rcclDdaLL128TwoShotThreshold(const ncclComm* comm) { return g_rcclDdaLL128TwoShotThreshold(comm); }
int rcclDdaFabricVmmBlocks(const ncclComm* comm) { return g_rcclDdaFabricVmmBlocks(comm); }
int rcclDdaFabricAgLL128Threads(const ncclComm* comm, size_t totalBytes) {
  return g_rcclDdaFabricAgLL128Threads(comm, totalBytes);
}
int rcclDdaFabricAgLL128Blocks(const ncclComm* comm, size_t totalBytes) {
  return g_rcclDdaFabricAgLL128Blocks(comm, totalBytes);
}
bool rcclDdaDatatypeSupported(const ncclComm* comm, ncclDataType_t datatype) {
  return g_rcclDdaDatatypeSupported(comm, datatype);
}

void ResetDdaCollSeams() {
  g_ddaLaunches.clear();
  g_rcclDdaLLThreshold = DefaultDdaLLThreshold;
  g_rcclDdaLL128Threshold = DefaultDdaLL128Threshold;
  g_rcclDdaLLTwoShotThreshold = DefaultDdaLLTwoShotThreshold;
  g_rcclDdaLL128TwoShotThreshold = DefaultDdaLL128TwoShotThreshold;
  g_rcclDdaFabricVmmBlocks = DefaultDdaFabricVmmBlocks;
  g_rcclDdaFabricAgLL128Threads = DefaultDdaFabricAgLL128Threads;
  g_rcclDdaFabricAgLL128Blocks = DefaultDdaFabricAgLL128Blocks;
  g_rcclDdaDatatypeSupported = DefaultDdaDatatypeSupported;
}

// Weak so a shared definition in fakes/hip_fakes.cc takes over without a clash. The test TUs embed an empty GPU
// binary (-fcuda-include-gpubinary) so clang does not reference an undefined __hip_fatbin_<cuid>.
extern "C" {
__attribute__((weak)) void** __hipRegisterFatBinary(const void*) {
  static void* handle = nullptr;
  return &handle;
}
__attribute__((weak)) void __hipRegisterFunction(void**, const void*, char*, const char*, unsigned int, void*, void*,
                                                 void*, void*, int*) {}
__attribute__((weak)) void __hipRegisterVar(void**, char*, char*, const char*, int, size_t, int, int) {}
__attribute__((weak)) void __hipUnregisterFatBinary(void**) {}
__attribute__((weak)) hipError_t __hipPopCallConfiguration(dim3*, dim3*, size_t*, hipStream_t*) {
  return hipErrorNotSupported;
}
// clang >= 23 registers per-kernel profile shadows under -fprofile-instr-generate; these keep the ROCm profile runtime out.
__attribute__((weak)) void __llvm_profile_offload_register_shadow_variable(void*) {}
__attribute__((weak)) void __llvm_profile_offload_register_section_shadow_variable(void*) {}
}  // extern "C"
