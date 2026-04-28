/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

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
// and stores kernel_args / kernel_args_size directly on the GPU activity struct.
// func must be a resolved hipFunction_t.
void HipCaptureKernelArgsExt(HipGpuActivityExt* gact, hipFunction_t func, void** args);

// Capture kernel arguments from a pre-packed kernargs buffer (the `extra` path).
// kernargs points to the contiguous ABI buffer; kernargs_size is its byte length.
// Uses desc.offset_ from the kernel signature to locate each argument.
void HipCaptureKernelArgsPackedExt(HipGpuActivityExt* gact, hipFunction_t func,
                                   const void* kernargs, size_t kernargs_size);

// ============================================================
// Graph node info — captured at hipGraphInstantiate time.
// Stored per graphExec so the GPU activity callback can fill in
// per-node dims and kernel args for graph launch spill nodes.
// ============================================================
struct HipGraphNodeInfoExt {
  // Common
  uint8_t        op;              // HipGpuOpExt: HIP_OP_DISPATCH_EXT or HIP_OP_COPY_EXT
  // Dispatch fields (op==HIP_OP_DISPATCH_EXT)
  std::string    kernel_name;    // mangled kernel name from hipKernelNodeParams
  const uint8_t* kernel_args;    // owned blob; NULL for copy nodes
  uint32_t       kernel_args_size;
  uint32_t       grid_x, grid_y, grid_z;
  uint32_t       block_x, block_y, block_z;
  // Copy fields (op==HIP_OP_COPY_EXT)
  const void*    src;            // source address captured at instantiate time
  const void*    dst;            // destination address
  uint64_t       bytes;          // byte count (for matching in callback)
  HipCopyKindExt copy_kind;      // for matching in callback

  HipGraphNodeInfoExt()
    : op(0), kernel_args(nullptr), kernel_args_size(0)
    , grid_x(0), grid_y(0), grid_z(0)
    , block_x(0), block_y(0), block_z(0)
    , src(nullptr), dst(nullptr), bytes(0)
    , copy_kind(HIP_COPY_KIND_UNKNOWN_EXT) {}
  ~HipGraphNodeInfoExt() { delete[] kernel_args; }

  // Non-copyable; only move is needed.
  HipGraphNodeInfoExt(const HipGraphNodeInfoExt&) = delete;
  HipGraphNodeInfoExt& operator=(const HipGraphNodeInfoExt&) = delete;
  HipGraphNodeInfoExt(HipGraphNodeInfoExt&& o) noexcept
    : op(o.op)
    , kernel_name(std::move(o.kernel_name))
    , kernel_args(o.kernel_args), kernel_args_size(o.kernel_args_size)
    , grid_x(o.grid_x), grid_y(o.grid_y), grid_z(o.grid_z)
    , block_x(o.block_x), block_y(o.block_y), block_z(o.block_z)
    , src(o.src), dst(o.dst), bytes(o.bytes), copy_kind(o.copy_kind) {
    o.kernel_args = nullptr; o.kernel_args_size = 0;
  }
};

// Store/erase/lookup node info for a given graphExec.
void HipStoreGraphExecNodesExt(hipGraphExec_t exec, std::vector<HipGraphNodeInfoExt> nodes);
void HipEraseGraphExecNodesExt(hipGraphExec_t exec);
const std::vector<HipGraphNodeInfoExt>* HipGetGraphExecNodesExt(hipGraphExec_t exec);

// Capture dims+args for one graph kernel node into HipGraphNodeInfoExt.
// func must be a resolved hipFunction_t (obtained via hipGetFuncBySymbol).
void HipCaptureGraphNodeArgsExt(HipGraphNodeInfoExt* info, hipFunction_t func, void** args);
