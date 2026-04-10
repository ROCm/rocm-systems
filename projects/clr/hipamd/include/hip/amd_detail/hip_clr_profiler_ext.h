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
 *
 * Fixed size: 128 bytes.  Fields beyond the active payload are reserved for
 * future use and must be treated as zero by callers.
 */
typedef struct {
  union {
    uint64_t _flags_u64;        /**< Raw 64-bit access to the packed flags word */
    struct {
      uint64_t op        : 3;   /**< 0=dispatch, 1=copy, 2=barrier */
      uint64_t is_graph  : 1;   /**< Set when the op was launched from a HIP graph */
      uint64_t           : 12;  /**< Unnamed reserved bits — must be zero */
      uint64_t device_id : 16;  /**< Device index (up to 65535 devices) */
      uint64_t queue_id  : 16;  /**< Queue/stream index (up to 65535 queues) */
      uint64_t           : 16;  /**< Unnamed reserved bits — must be zero */
    };
  };
  uint64_t    begin_ns;         /**< GPU begin timestamp (ns) */
  uint64_t    end_ns;           /**< GPU end timestamp (ns) */
  union {
    uint64_t    bytes;          /**< Bytes transferred (op==copy) */
    const char* kernel_name;    /**< Kernel name        (op==dispatch, may be NULL) */
  };
  uint8_t     _pad1[128 - (8+8+8+8)]; /**< Padding to 128 bytes */
} HipClrGpuActivity;
#ifdef __cplusplus
static_assert(sizeof(HipClrGpuActivity) == 128, "HipClrGpuActivity must be 128 bytes");
#endif

/**
 * @brief Per-HIP-API profiling record returned by hipClrProfilerGetRecords().
 *
 * Timestamps are nanoseconds from the Unix epoch (clock_gettime CLOCK_REALTIME
 * on Linux, or QueryPerformanceCounter-based on Windows).
 *
 * The gpu field is valid only when has_gpu_activity != 0.
 *
 * Fixed size: 256 bytes (40-byte CPU header + 128-byte HipClrGpuActivity + 88-byte pad).
 */
typedef struct {
  /* CPU call info — first 128-byte half */
  union {
    uint64_t _header_u64;             /**< Raw 64-bit access to the flags word */
    struct {
      uint64_t api_id           : 16; /**< HIP API identifier (up to 65535 APIs) */
      uint64_t has_gpu_activity : 1;  /**< Non-zero when gpu field is valid */
      uint64_t                  : 47; /**< Unnamed reserved bits — must be zero */
    };
  };
  uint64_t          thread_id;        /**< Hash of std::thread::id */
  uint64_t          start_ns;         /**< CPU call begin (ns) */
  uint64_t          end_ns;           /**< CPU call end (ns) */
  /* Padding to 128-byte CPU half: 128 - (8+8+8+8) = 96 */
  uint8_t           _pad1[96];
  /* GPU activity — second 128-byte half (valid when has_gpu_activity != 0) */
  HipClrGpuActivity gpu;
} HipClrApiRecord;
#ifdef __cplusplus
static_assert(sizeof(HipClrApiRecord) == 256, "HipClrApiRecord must be 256 bytes");
#endif

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
