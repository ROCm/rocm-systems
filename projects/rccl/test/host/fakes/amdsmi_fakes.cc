/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Implementation of the src/misc/amdsmi_wrap.cc fakes. See amdsmi_fakes.h.

#include "amdsmi_fakes.h"

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

void ResetAmdSmiFakes() {
  g_amdSmiGetDeviceIndexByPciBusId = DefaultAmdSmiGetDeviceIndexByPciBusId;
  g_amdSmiGetFabricDeviceInfo = DefaultAmdSmiGetFabricDeviceInfo;
  g_amdSmiInitResult = ncclSuccess;
  g_amdSmiGetNumDevice = DefaultAmdSmiGetNumDevice;
  g_amdSmiGetDevicePciBusIdString = DefaultAmdSmiGetDevicePciBusIdString;
  g_amdSmiGetLinkInfo = DefaultAmdSmiGetLinkInfo;
}
