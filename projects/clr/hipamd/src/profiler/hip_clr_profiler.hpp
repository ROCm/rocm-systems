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
#include <vector>

#include "hip/hip_runtime_api.h"
#include "hip/amd_detail/hip_profiler_ext.h"
#include "platform/prof_protocol.h"


// Internal HIP function — probe-safe variant of hipGetFuncBySymbol that never
// sets the sticky last_error_.  Used by the profiler to resolve kernel symbols
// without polluting the user-visible error state.
namespace hip {
hipError_t ihipGetFuncBySymbol(hipFunction_t* functionPtr, const void* symbolPtr);
}

// ============================================================
// Internal profiler API
// ============================================================
void HipProfilerInitExt();
uint64_t HipProfilerEnableExt();
uint64_t HipProfilerDisableExt();

// Called from each *Layer wrapper.
hipApiRecordExt* HipGetActiveRecordExt(uint32_t api_id);

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
// and stores kernel_args / kernel_args_size directly on the GPU activity struct.
// func must be a resolved hipFunction_t.
void HipCaptureKernelArgsExt(hipGpuActivityExt* gact, hipFunction_t func, void** args);

// Capture kernel arguments from a pre-packed kernargs buffer (the `extra` path).
// kernargs points to the contiguous ABI buffer; kernargs_size is its byte length.
// Uses desc.offset_ from the kernel signature to locate each argument.
void HipCaptureKernelArgsPackedExt(hipGpuActivityExt* gact, hipFunction_t func,
                                   const void* kernargs, size_t kernargs_size);

// ============================================================
// Graph node info — captured at hipGraphInstantiate time.
// Stored per graphExec so the GPU activity callback can fill in
// per-node dims and kernel args for graph launch spill nodes.
//
// All per-op fields (op, copy_kind, grid/block, kernel_args, src/dst, bytes)
// live directly in gpu (hipGpuActivityExt).  gpu.kernel_name is a stable
// const char* into the g_kernel_names interning table (non-owning).
// gpu.kernel_args is an owned blob freed by the destructor.
// ============================================================
struct HipGraphNodeInfoExt {
  hipGpuActivityExt gpu;  // all op-specific fields; gpu.kernel_args is owned

  HipGraphNodeInfoExt() : gpu{} {}
  ~HipGraphNodeInfoExt() { delete[] gpu.kernel_args; }

  // Non-copyable; move transfers kernel_args ownership.
  HipGraphNodeInfoExt(const HipGraphNodeInfoExt&) = delete;
  HipGraphNodeInfoExt& operator=(const HipGraphNodeInfoExt&) = delete;
  HipGraphNodeInfoExt(HipGraphNodeInfoExt&& o) noexcept : gpu(o.gpu) {
    o.gpu.kernel_args      = nullptr;
    o.gpu.kernel_args_size = 0;
  }
};

// Store/erase/lookup node info for a given graphExec.
void HipStoreGraphExecNodesExt(hipGraphExec_t exec, std::vector<HipGraphNodeInfoExt> nodes);
void HipEraseGraphExecNodesExt(hipGraphExec_t exec);
const std::vector<HipGraphNodeInfoExt>* HipGetGraphExecNodesExt(hipGraphExec_t exec);

// Capture kernel name and args for one graph kernel node.
// Interns the mangled name into g_kernel_names and stores a stable const char*
// in info->gpu.kernel_name.  Writes the arg blob into info->gpu.kernel_args/size.
// func must be a resolved hipFunction_t (obtained via hipGetFuncBySymbol).
void HipCaptureGraphNodeArgsExt(HipGraphNodeInfoExt* info, hipFunction_t func, void** args);
