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
uint64_t HipProfilerEnableExt();
uint64_t HipProfilerDisableExt();

// Called from each *Layer wrapper.
HipApiRecordExt* HipGetActiveRecordExt(uint32_t api_id);

// Declared in hip_clr_dispatch_wrappers.cpp; called by Enable/Disable/Init
struct HipDispatchTable;
struct HipCompilerDispatchTable;
// Called once at init: captures g_next and pre-builds g_wrapper_tbl.
void HipProfilerBuildWrapperTableExt(HipDispatchTable* tbl);
void HipProfilerInstallWrappersExt(HipDispatchTable* tbl);
void HipProfilerRemoveWrappersExt(HipDispatchTable* tbl);
// Compiler dispatch table (___hipPushCallConfiguration / hipLaunchByPtr path).
void HipProfilerInstallCompilerWrappersExt(HipCompilerDispatchTable* tbl);
void HipProfilerRemoveCompilerWrappersExt(HipCompilerDispatchTable* tbl);

// API name table — indexed by api_id, same order as UpdateDispatchTable
extern const char* const kHipApiNamesExt[];
extern const size_t      kHipApiNamesCountExt;

// Capture kernel arguments from a void** args array.
// Walks the kernel signature (user params only), packs each arg as
// {uint32_t size; uint8_t data[size];} into a heap-allocated blob,
// and stores kernel_args / kernel_args_size on the record.
// func may be a hipFunction_t (module path) or a host function pointer
// (hipLaunchKernel path — pass as_hip_func=false); in the latter case
// hipGetFuncBySymbol is called via the dispatch table to resolve it.
void HipCaptureKernelArgsExt(HipApiRecordExt* rec, hipFunction_t func, void** args);
