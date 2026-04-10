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
 * @brief GPU operation type stored in HipClrGpuActivity::op.
 */
typedef enum {
  HIP_CLR_OP_DISPATCH = 0, /**< Kernel dispatch */
  HIP_CLR_OP_COPY     = 1, /**< Memory copy (DMA / blit) */
  HIP_CLR_OP_BARRIER  = 2, /**< Barrier / fence */
} HipClrGpuOp;

/**
 * @brief GPU activity record returned by hipClrProfilerGetRecords().
 *
 * Fixed size: 128 bytes.  Fields beyond the active payload are reserved for
 * future use and must be treated as zero by callers.
 *
 * When embedded as HipClrApiRecord::gpu this struct describes the first (or
 * only) GPU operation for the API call.  gpu_op_count and gpu_ops expose all
 * operations (>1 for hipGraphLaunch with multiple nodes).
 */
typedef struct HipClrGpuActivity {
  union {
    uint64_t _flags_u64;        /**< Raw 64-bit access to the packed flags word */
    struct {
      uint64_t op        : 3;   /**< HipClrGpuOp value */
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
  /* Originally _pad1[96].  First 16 bytes repurposed for multi-op support;
   * remaining 80 bytes stay reserved and must be treated as zero. */
  uint32_t          gpu_op_count;              /**< Total GPU ops (0=none, 1=single, >1=graph) */
  uint32_t          _reserved_u32;             /**< Reserved — must be zero */
  const struct HipClrGpuActivity* gpu_ops;    /**< gpu_ops[0..gpu_op_count-1]; for gpu_op_count==1
                                                   points to the enclosing gpu field itself */
  uint8_t     _pad1[80];                       /**< Remaining reserved padding — must be zero */
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
 * The gpu field is valid only when has_gpu_activity != 0.  It always holds the
 * first GPU operation.  gpu.gpu_op_count gives the total count; gpu.gpu_ops
 * points to all operations (profiler-owned flat buffer).
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
  hipStream_t       stream;           /**< Stream argument, or NULL for default/no-stream APIs */
  /* Padding to 128-byte CPU half: 128 - (8+8+8+8+8) = 88 */
  uint8_t           _pad1[88];
  /* GPU activity — second 128-byte half (valid when has_gpu_activity != 0) */
  HipClrGpuActivity gpu;
} HipClrApiRecord;
#ifdef __cplusplus
static_assert(sizeof(HipClrApiRecord) == 256, "HipClrApiRecord must be 256 bytes");
#endif

/**
 * @brief Return the HIP API name for a given api_id.
 *
 * @param api_id  The api_id from HipClrApiRecord::api_id.
 * @return        A constant string (e.g. "hipLaunchKernel"), or "unknown" if
 *                api_id is out of range.  The pointer is valid for the lifetime
 *                of the process.
 */
const char* hipClrApiName(uint32_t api_id);

/**
 * @brief Return the number of tracked HIP API entries.
 *
 * Valid api_id values are in [0, hipClrApiCount()).
 */
uint32_t hipClrApiCount(void);

/**
 * @brief Return the name of a GPU op type.
 *
 * @param op  A HipClrGpuOp value.
 * @return    "Dispatch", "Copy", "Barrier", or "Unknown".
 */
const char* hipClrGpuOpName(uint32_t op);

/**
 * @brief Enable built-in CLR profiling at runtime.
 *
 * Equivalent to setting GPU_CLR_PROFILE_OUTPUT=<path> before process start,
 * but can be called at any time from application code.
 */
hipError_t hipClrProfilerEnable(void);

/**
 * @brief Disable built-in CLR profiling.  Already-collected records are kept.
 */
hipError_t hipClrProfilerDisable(void);

/**
 * @brief Return a pointer to all records collected since init or last Reset.
 *
 * The returned pointer and the gpu.gpu_ops buffer it references are owned by
 * the profiler and remain valid until the next call to hipClrProfilerGetRecords()
 * or hipClrProfilerReset().
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
