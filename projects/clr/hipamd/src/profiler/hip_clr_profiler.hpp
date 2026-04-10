/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

#include "hip/hip_runtime_api.h"
#include "platform/prof_protocol.h"

// ============================================================
// Internal GPU activity record (mirrors activity_record_t)
// ============================================================
struct HipClrGpuRecord {
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
// Combined CPU + GPU profiling record — mirrors reference ProfRecord
// ============================================================
struct HipClrProfRecord {
  uint32_t          api_id;
  uint64_t          thread_id;
  std::chrono::high_resolution_clock::time_point start_;
  std::chrono::high_resolution_clock::time_point end_;
  hipError_t        result;
  bool              has_gpu;
  HipClrGpuRecord   gpu;
};

// ============================================================
// Internal profiler API
// ============================================================
void HipClrProfilerInit();
void HipClrProfilerEnable();
void HipClrProfilerDisable();
void HipClrProfilerReset();
void HipClrProfilerWriteJson(const char* filepath);
// out_records: pointer to first chunk (caller walks chunks of out_chunk_size)
// out_count:      total number of valid records across all chunks
// out_chunk_size: capacity of each chunk (last chunk may be partially filled)
void HipClrProfilerGetRecords(const HipClrProfRecord** out_records,
                               size_t* out_count,
                               size_t* out_chunk_size);

// Called from each *Layer wrapper — mirrors reference GetActiveRecord().
// Returns nullptr when profiling is disabled (wrapper skips end_ stamp).
HipClrProfRecord* HipClrGetActiveRecord(uint32_t api_id);

// Declared in hip_clr_dispatch_wrappers.cpp; called by Enable/Disable/Init
struct HipDispatchTable;
void HipClrProfilerInstallWrappers(HipDispatchTable* tbl);
void HipClrProfilerRemoveWrappers(HipDispatchTable* tbl);

// API name table — indexed by api_id, same order as UpdateDispatchTable
extern const char* const kHipClrApiNames[];
extern const size_t      kHipClrApiNamesCount;
