/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Fakes for the symbols defined by src/misc/amdsmi_wrap.cc, named after the
// PRODUCTION TU that owns them rather than the test that needed them first.

#ifndef RCCL_TEST_HOST_AMDSMI_FAKES_H_
#define RCCL_TEST_HOST_AMDSMI_FAKES_H_

#include <cstdint>
#include <functional>

#include "nccl.h"

struct amdsmiFabricDeviceInfo;

// fillInfo's UALoE/MNNVL probe. The default answers -1, i.e. what a host with no fabric device reports.
ncclResult_t DefaultAmdSmiGetDeviceIndexByPciBusId(const char* busId, uint32_t* deviceIndex);
extern std::function<ncclResult_t(const char*, uint32_t*)> g_amdSmiGetDeviceIndexByPciBusId;

// The default leaves the caller's struct untouched, so fillInfo's own fabricSupported=false stands.
ncclResult_t DefaultAmdSmiGetFabricDeviceInfo(uint32_t deviceIndex, struct amdsmiFabricDeviceInfo* info);
extern std::function<ncclResult_t(uint32_t, struct amdsmiFabricDeviceInfo*)> g_amdSmiGetFabricDeviceInfo;

extern ncclResult_t g_amdSmiInitResult;

void ResetAmdSmiFakes();

#endif  // RCCL_TEST_HOST_AMDSMI_FAKES_H_
