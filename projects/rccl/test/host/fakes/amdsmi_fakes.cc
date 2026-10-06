/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Implementation of the src/misc/amdsmi_wrap.cc fakes. See amdsmi_fakes.h.

#include "amdsmi_fakes.h"

#include "amdsmi_wrap.h"
#include "signature-drift.h"

ASSERT_HOOK_MATCHES_PROD(g_amdSmiDiagGpuCount, amd_smi_diagGpuCount);
ASSERT_HOOK_MATCHES_PROD(g_amdSmiDiagGpuModel, amd_smi_diagGpuModel);
ASSERT_HOOK_MATCHES_PROD(g_amdSmiDiagEccCounts, amd_smi_diagEccCounts);
ASSERT_HOOK_MATCHES_PROD(g_amdSmiDiagXgmiLinks, amd_smi_diagXgmiLinks);

ncclResult_t DefaultAmdSmiGetDeviceIndexByPciBusId(const char*, uint32_t* deviceIndex) {
  if (deviceIndex) *deviceIndex = static_cast<uint32_t>(-1);  // -1 -> skip fabric block
  return ncclSuccess;
}
std::function<ncclResult_t(const char*, uint32_t*)> g_amdSmiGetDeviceIndexByPciBusId =
    DefaultAmdSmiGetDeviceIndexByPciBusId;
ncclResult_t amd_smi_getDeviceIndexByPciBusId(const char* busId, uint32_t* deviceIndex) {
  return g_amdSmiGetDeviceIndexByPciBusId(busId, deviceIndex);
}

ncclResult_t DefaultAmdSmiGetFabricDeviceInfo(uint32_t, struct amdsmiFabricDeviceInfo*) {
  return ncclSuccess;
}
std::function<ncclResult_t(uint32_t, struct amdsmiFabricDeviceInfo*)> g_amdSmiGetFabricDeviceInfo =
    DefaultAmdSmiGetFabricDeviceInfo;
ncclResult_t amd_smi_getFabricDeviceInfo(uint32_t deviceIndex, struct amdsmiFabricDeviceInfo* info) {
  return g_amdSmiGetFabricDeviceInfo(deviceIndex, info);
}

ncclResult_t g_amdSmiInitResult = ncclSuccess;
ncclResult_t amd_smi_init() { return g_amdSmiInitResult; }

namespace {
template <typename... Args>
ncclResult_t AmdSmiDiagUnavailable(Args...) {
  return ncclSystemError;
}
}  // namespace

int g_amdSmiDiagInitCalls = 0;
ncclResult_t amd_smi_diagInit() {
  ++g_amdSmiDiagInitCalls;
  return ncclSystemError;
}

std::function<ncclResult_t(uint32_t*)> g_amdSmiDiagGpuCount = AmdSmiDiagUnavailable<uint32_t*>;
ncclResult_t amd_smi_diagGpuCount(uint32_t* count) { return g_amdSmiDiagGpuCount(count); }

std::function<ncclResult_t(int64_t, char*, size_t)> g_amdSmiDiagGpuModel =
    AmdSmiDiagUnavailable<int64_t, char*, size_t>;
ncclResult_t amd_smi_diagGpuModel(int64_t busId, char* model, size_t len) {
  return g_amdSmiDiagGpuModel(busId, model, len);
}

std::function<ncclResult_t(int64_t, struct amdsmiDiagEccCounts*)> g_amdSmiDiagEccCounts =
    AmdSmiDiagUnavailable<int64_t, struct amdsmiDiagEccCounts*>;
ncclResult_t amd_smi_diagEccCounts(int64_t busId, struct amdsmiDiagEccCounts* counts) {
  return g_amdSmiDiagEccCounts(busId, counts);
}

std::function<ncclResult_t(int64_t, struct amdsmiDiagXgmiLinks*)> g_amdSmiDiagXgmiLinks =
    AmdSmiDiagUnavailable<int64_t, struct amdsmiDiagXgmiLinks*>;
ncclResult_t amd_smi_diagXgmiLinks(int64_t busId, struct amdsmiDiagXgmiLinks* links) {
  return g_amdSmiDiagXgmiLinks(busId, links);
}

void ResetAmdSmiFakes() {
  g_amdSmiGetDeviceIndexByPciBusId = DefaultAmdSmiGetDeviceIndexByPciBusId;
  g_amdSmiGetFabricDeviceInfo = DefaultAmdSmiGetFabricDeviceInfo;
  g_amdSmiInitResult = ncclSuccess;
  g_amdSmiDiagInitCalls = 0;
  g_amdSmiDiagGpuCount = AmdSmiDiagUnavailable<uint32_t*>;
  g_amdSmiDiagGpuModel = AmdSmiDiagUnavailable<int64_t, char*, size_t>;
  g_amdSmiDiagEccCounts = AmdSmiDiagUnavailable<int64_t, struct amdsmiDiagEccCounts*>;
  g_amdSmiDiagXgmiLinks = AmdSmiDiagUnavailable<int64_t, struct amdsmiDiagXgmiLinks*>;
}
