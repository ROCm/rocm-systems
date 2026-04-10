/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stddef.h>
#include <stdint.h>
#include "hip/hip_runtime_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief GPU activity record returned by hipClrProfilerGetRecords().
 */
typedef struct {
  uint32_t    op;           /**< 0=dispatch, 1=copy, 2=barrier */
  uint64_t    begin_ns;     /**< GPU begin timestamp (ns) */
  uint64_t    end_ns;       /**< GPU end timestamp (ns) */
  int         device_id;
  uint64_t    queue_id;
  uint64_t    bytes;        /**< Bytes transferred (copy ops) */
  const char* kernel_name;  /**< Kernel name (dispatch ops, may be NULL) */
} HipClrGpuActivity;

/**
 * @brief Per-HIP-API profiling record returned by hipClrProfilerGetRecords().
 *
 * Timestamps are nanoseconds from the Unix epoch (clock_gettime CLOCK_REALTIME
 * on Linux, or QueryPerformanceCounter-based on Windows).
 */
typedef struct {
  uint32_t         api_id;
  uint64_t         thread_id;         /**< Hash of std::thread::id */
  uint64_t         start_ns;
  uint64_t         end_ns;
  int              has_gpu_activity;  /**< Non-zero when gpu field is valid */
  HipClrGpuActivity gpu;
} HipClrApiRecord;

/**
 * @brief Enable built-in CLR profiling at runtime.
 *
 * Equivalent to setting GPU_CLR_PROFILE=1 before process start, but can be
 * called at any time from application code.
 */
hipError_t hipClrProfilerEnable(void);

/**
 * @brief Disable built-in CLR profiling.  Already-collected records are kept.
 */
hipError_t hipClrProfilerDisable(void);

/**
 * @brief Return a pointer to all records collected since init or last Reset.
 *
 * The returned pointer is owned by the profiler.  It remains valid until the
 * next call to hipClrProfilerReset() or library unload.  Calling
 * hipClrProfilerGetRecords() again after new records have been collected will
 * rebuild the flat export buffer, so callers should process records before
 * issuing further HIP calls when profiling is active.
 *
 * @param[out] records  Set to the base of the flat record array.
 * @param[out] count    Set to the number of valid records.
 */
hipError_t hipClrProfilerGetRecords(const HipClrApiRecord** records, size_t* count);

/**
 * @brief Clear all accumulated records and free internal storage.
 */
hipError_t hipClrProfilerReset(void);

/**
 * @brief Flush all accumulated records to a Chrome Trace Event JSON file.
 *
 * @param filepath  Destination file path, or NULL to use "hip_clr_trace.json".
 */
hipError_t hipClrProfilerWriteJson(const char* filepath);

#ifdef __cplusplus
}
#endif
