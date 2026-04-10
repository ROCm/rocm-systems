/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <new>
#include <thread>
#include <vector>

#include "hip/hip_runtime_api.h"
#include "hip/amd_detail/hip_profiler_ext.h"
#include "platform/prof_protocol.h"

// ============================================================
// Internal GPU activity record (mirrors activity_record_t)
// Accumulated per-slot in the _pad1 area of HipApiRecordExt.
// ============================================================
struct HipGpuRecord {
  uint32_t   op;           // OP_ID_DISPATCH=0, OP_ID_COPY=1, OP_ID_BARRIER=2
  uint64_t   begin_ns;
  uint64_t   end_ns;
  int        device_id;
  uint64_t   queue_id;
  union {
    size_t      bytes;
    const char* kernel_name;
  };
};

// ============================================================
// Access the gpu_ops accumulation vector stored in _pad1.
// Valid only while the record lives in a chunk (before GetRecords).
// _pad1 is 80 bytes; std::vector is ≤24 bytes on any supported ABI.
// ============================================================
static_assert(sizeof(std::vector<HipGpuRecord>) <= sizeof(HipApiRecordExt{})._pad1,
              "gpu_ops vector does not fit in _pad1");

inline std::vector<HipGpuRecord>& GpuOps(HipApiRecordExt& r) {
  return *std::launder(reinterpret_cast<std::vector<HipGpuRecord>*>(r._pad1));
}
inline const std::vector<HipGpuRecord>& GpuOps(const HipApiRecordExt& r) {
  return *std::launder(reinterpret_cast<const std::vector<HipGpuRecord>*>(r._pad1));
}

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
