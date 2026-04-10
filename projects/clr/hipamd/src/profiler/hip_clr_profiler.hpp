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
// Access the gpu_ops overflow vector stored in HipApiRecordExt::_pad1.
// Used only when a single API call produces more than one GPU activity record
// (e.g. hipGraphLaunch with multiple kernel/copy nodes).  Op-1 lives directly
// in HipApiRecordExt::gpu; ops 2..N spill into this vector.
//
// _pad1 is 80 bytes; std::vector<HipGpuActivityExt> is 24 bytes on any ABI.
// Spill entries are zero-initialised (gpu_op_count=0, gpu_ops=nullptr, _pad1={0})
// and are ready to push into g_export_gpu_buf without further fixup.
// ============================================================
static_assert(sizeof(std::vector<HipGpuActivityExt>) <= sizeof(HipApiRecordExt{})._pad1,
              "gpu_ops spill vector does not fit in _pad1");

inline std::vector<HipGpuActivityExt>& GpuOps(HipApiRecordExt& r) {
  return *std::launder(reinterpret_cast<std::vector<HipGpuActivityExt>*>(r._pad1));
}
inline const std::vector<HipGpuActivityExt>& GpuOps(const HipApiRecordExt& r) {
  return *std::launder(reinterpret_cast<const std::vector<HipGpuActivityExt>*>(r._pad1));
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
