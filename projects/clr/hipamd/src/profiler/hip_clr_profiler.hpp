/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <thread>

#include "hip/hip_runtime_api.h"
#include "hip/amd_detail/hip_profiler_ext.h"
#include "platform/prof_protocol.h"


// ============================================================
// Internal profiler API
// ============================================================
void HipProfilerInitExt();
void HipProfilerEnableExt();
void HipProfilerDisableExt();
void HipProfilerResetExt();
void HipProfilerWriteJsonExt(const char* filepath);

// Called from each *Layer wrapper.
HipApiRecordExt* HipGetActiveRecordExt(uint32_t api_id);

// Declared in hip_clr_dispatch_wrappers.cpp; called by Enable/Disable/Init
struct HipDispatchTable;
void HipProfilerInstallWrappersExt(HipDispatchTable* tbl);
void HipProfilerRemoveWrappersExt(HipDispatchTable* tbl);

// API name table — indexed by api_id, same order as UpdateDispatchTable
extern const char* const kHipApiNamesExt[];
extern const size_t      kHipApiNamesCountExt;
