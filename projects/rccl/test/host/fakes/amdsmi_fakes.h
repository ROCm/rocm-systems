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

// commAlloc's fabric telemetry opt-in. The default declines, i.e. what every host
// reports unless RCCL_FABRIC_TELEMETRY_ENABLE is set, which leaves commFree with
// nothing to release.
ncclResult_t DefaultAmdSmiFabricTelemetryAcquire(uint32_t deviceIndex, uint64_t commHash, int rank, bool* acquired);
extern std::function<ncclResult_t(uint32_t, uint64_t, int, bool*)> g_amdSmiFabricTelemetryAcquire;

// commFree's matching release. Nothing to undo in the default, which never acquired.
ncclResult_t DefaultAmdSmiFabricTelemetryRelease(uint32_t deviceIndex);
extern std::function<ncclResult_t(uint32_t)> g_amdSmiFabricTelemetryRelease;

void ResetAmdSmiFakes();

#endif  // RCCL_TEST_HOST_AMDSMI_FAKES_H_
