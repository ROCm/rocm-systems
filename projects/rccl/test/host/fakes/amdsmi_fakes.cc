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

void ResetAmdSmiFakes() {
  g_amdSmiGetDeviceIndexByPciBusId = DefaultAmdSmiGetDeviceIndexByPciBusId;
  g_amdSmiGetFabricDeviceInfo = DefaultAmdSmiGetFabricDeviceInfo;
  g_amdSmiInitResult = ncclSuccess;
}
