/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Implementation of the src/misc/amdsmi_wrap.cc fakes. See amdsmi_fakes.h.

#include "amdsmi_fakes.h"
#include "signature-drift.h"

ASSERT_HOOK_MATCHES_PROD(g_amdSmiGetDeviceIndexByPciBusId, amd_smi_getDeviceIndexByPciBusId);
ASSERT_HOOK_MATCHES_PROD(g_amdSmiGetFabricDeviceInfo,      amd_smi_getFabricDeviceInfo);
ASSERT_HOOK_MATCHES_PROD(g_amdSmiGetNumDevice,             amd_smi_getNumDevice);
ASSERT_HOOK_MATCHES_PROD(g_amdSmiGetDevicePciBusIdString,  amd_smi_getDevicePciBusIdString);
ASSERT_HOOK_MATCHES_PROD(g_amdSmiGetLinkInfo,              amd_smi_getLinkInfo);
ASSERT_HOOK_MATCHES_PROD(g_amdSmiFabricTelemetryAcquire,   amd_smi_fabricTelemetryAcquire);
ASSERT_HOOK_MATCHES_PROD(g_amdSmiFabricTelemetryRelease,   amd_smi_fabricTelemetryRelease);

#include "amdsmi_wrap.h"
#include "signature-drift.h"

ASSERT_HOOK_MATCHES_PROD(g_amdSmiDiagGpuCount, amd_smi_diagGpuCount);
ASSERT_HOOK_MATCHES_PROD(g_amdSmiDiagGpuModel, amd_smi_diagGpuModel);
ASSERT_HOOK_MATCHES_PROD(g_amdSmiDiagDriverVersion, amd_smi_diagDriverVersion);
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
int g_amdSmiInitCalls = 0;
ncclResult_t amd_smi_init() {
  g_amdSmiInitCalls++;
  return g_amdSmiInitResult;
}

static ncclResult_t DefaultAmdSmiGetNumDevice(uint32_t* numDevs) {
  *numDevs = 0;
  return ncclSuccess;
}
std::function<ncclResult_t(uint32_t*)> g_amdSmiGetNumDevice = DefaultAmdSmiGetNumDevice;
ncclResult_t amd_smi_getNumDevice(uint32_t* numDevs) { return g_amdSmiGetNumDevice(numDevs); }

static ncclResult_t DefaultAmdSmiGetDevicePciBusIdString(uint32_t, char*, size_t) { return ncclInternalError; }
std::function<ncclResult_t(uint32_t, char*, size_t)> g_amdSmiGetDevicePciBusIdString =
    DefaultAmdSmiGetDevicePciBusIdString;
ncclResult_t amd_smi_getDevicePciBusIdString(uint32_t deviceIndex, char* pciBusId, size_t len) {
  return g_amdSmiGetDevicePciBusIdString(deviceIndex, pciBusId, len);
}

static ncclResult_t DefaultAmdSmiGetLinkInfo(int, int, amdsmi_link_type_t*, int*, int*) { return ncclInternalError; }
std::function<ncclResult_t(int, int, amdsmi_link_type_t*, int*, int*)> g_amdSmiGetLinkInfo = DefaultAmdSmiGetLinkInfo;
ncclResult_t amd_smi_getLinkInfo(int srcDev, int dstDev, amdsmi_link_type_t* type, int* hops, int* count) {
  return g_amdSmiGetLinkInfo(srcDev, dstDev, type, hops, count);
}

ncclResult_t DefaultAmdSmiFabricTelemetryAcquire(uint32_t, uint64_t, int, bool* acquired) {
  if (acquired) *acquired = false;  // telemetry is opt-in and off by default
  return ncclSuccess;
}
std::function<ncclResult_t(uint32_t, uint64_t, int, bool*)> g_amdSmiFabricTelemetryAcquire =
    DefaultAmdSmiFabricTelemetryAcquire;
ncclResult_t amd_smi_fabricTelemetryAcquire(uint32_t deviceIndex, uint64_t commHash, int rank, bool* acquired) {
  return g_amdSmiFabricTelemetryAcquire(deviceIndex, commHash, rank, acquired);
}

ncclResult_t DefaultAmdSmiFabricTelemetryRelease(uint32_t) { return ncclSuccess; }
std::function<ncclResult_t(uint32_t)> g_amdSmiFabricTelemetryRelease = DefaultAmdSmiFabricTelemetryRelease;
ncclResult_t amd_smi_fabricTelemetryRelease(uint32_t deviceIndex) {
  return g_amdSmiFabricTelemetryRelease(deviceIndex);
}

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

std::function<ncclResult_t(int64_t, char*, size_t)> g_amdSmiDiagDriverVersion =
    AmdSmiDiagUnavailable<int64_t, char*, size_t>;
ncclResult_t amd_smi_diagDriverVersion(int64_t busId, char* version, size_t len) {
  return g_amdSmiDiagDriverVersion(busId, version, len);
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
  g_amdSmiInitCalls = 0;
  g_amdSmiGetNumDevice = DefaultAmdSmiGetNumDevice;
  g_amdSmiGetDevicePciBusIdString = DefaultAmdSmiGetDevicePciBusIdString;
  g_amdSmiGetLinkInfo = DefaultAmdSmiGetLinkInfo;
  g_amdSmiFabricTelemetryAcquire = DefaultAmdSmiFabricTelemetryAcquire;
  g_amdSmiFabricTelemetryRelease = DefaultAmdSmiFabricTelemetryRelease;
  g_amdSmiDiagInitCalls = 0;
  g_amdSmiDiagGpuCount = AmdSmiDiagUnavailable<uint32_t*>;
  g_amdSmiDiagGpuModel = AmdSmiDiagUnavailable<int64_t, char*, size_t>;
  g_amdSmiDiagDriverVersion = AmdSmiDiagUnavailable<int64_t, char*, size_t>;
  g_amdSmiDiagEccCounts = AmdSmiDiagUnavailable<int64_t, struct amdsmiDiagEccCounts*>;
  g_amdSmiDiagXgmiLinks = AmdSmiDiagUnavailable<int64_t, struct amdsmiDiagXgmiLinks*>;
}
