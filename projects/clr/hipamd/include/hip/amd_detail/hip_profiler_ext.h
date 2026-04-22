/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef HIP_PROFILER_EXT_H
#define HIP_PROFILER_EXT_H

/**
 * @file hip_profiler_ext.h
 *
 * HIP built-in profiling extension — BETA API.
 *
 * This interface is under active development. Structures, function signatures,
 * and enum values may change in future releases without notice.
 * Do not use in production code.
 *
 * Version: 0.1.0
 */

#include <stddef.h>
#include <stdint.h>
#include "hip/hip_runtime_api.h"

#define HIP_PROFILER_EXT_VERSION_MAJOR 0
#define HIP_PROFILER_EXT_VERSION_MINOR 1
#define HIP_PROFILER_EXT_VERSION_PATCH 0

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief GPU operation type stored in HipGpuActivityExt::op.
 */
typedef enum {
  HIP_OP_DISPATCH_EXT = 0, /**< Kernel dispatch */
  HIP_OP_COPY_EXT     = 1, /**< Memory copy (DMA / blit) */
  HIP_OP_BARRIER_EXT  = 2, /**< Barrier / fence */
} HipGpuOpExt;

/**
 * @brief Memory copy direction stored in HipGpuActivityExt::copy_kind.
 * Valid only when op == HIP_OP_COPY_EXT.
 *
 * Each value corresponds to one OpenCL CL_COMMAND_* copy command so callers
 * can distinguish rectangular copies, image copies, and format-conversion copies
 * without consulting any internal OpenCL headers.
 *
 * The 4-bit copy_kind bitfield supports values 0–15; this enum uses 0–12.
 */
typedef enum {
  HIP_COPY_KIND_UNKNOWN_EXT         =  0, /**< Direction not determined */
  /* Buffer ↔ host (SDMA / PCIe) */
  HIP_COPY_KIND_H2D_EXT             =  1, /**< Host buffer → device buffer */
  HIP_COPY_KIND_H2D_RECT_EXT        =  2, /**< Host buffer → device buffer, rectangular region */
  HIP_COPY_KIND_H2D_IMAGE_EXT       =  3, /**< Host buffer → device image */
  HIP_COPY_KIND_D2H_EXT             =  4, /**< Device buffer → host buffer */
  HIP_COPY_KIND_D2H_RECT_EXT        =  5, /**< Device buffer → host buffer, rectangular region */
  HIP_COPY_KIND_D2H_IMAGE_EXT       =  6, /**< Device image → host buffer */
  /* Device ↔ device (GPU blit / compute engine) */
  HIP_COPY_KIND_D2D_EXT             =  7, /**< Device buffer → device buffer */
  HIP_COPY_KIND_D2D_RECT_EXT        =  8, /**< Device buffer → device buffer, rectangular region */
  HIP_COPY_KIND_D2D_IMAGE_EXT       =  9, /**< Device image → device image */
  HIP_COPY_KIND_BUFFER_TO_IMAGE_EXT = 10, /**< Device buffer → device image (format conversion) */
  HIP_COPY_KIND_IMAGE_TO_BUFFER_EXT = 11, /**< Device image → device buffer (format conversion) */
  HIP_COPY_KIND_FILL_EXT            = 12, /**< Device buffer fill (pattern written by compute engine) */
} HipCopyKindExt;

/**
 * @brief Returns non-zero if the copy kind crosses PCIe and uses the SDMA engine.
 *
 * Convenience predicate for callers that want to separate SDMA transfers from
 * device-side blit/compute copies without enumerating every kind individually.
 */
static inline int hipCopyKindIsSDMAExt(HipCopyKindExt kind) {
  return kind == HIP_COPY_KIND_H2D_EXT       ||
         kind == HIP_COPY_KIND_H2D_RECT_EXT  ||
         kind == HIP_COPY_KIND_H2D_IMAGE_EXT ||
         kind == HIP_COPY_KIND_D2H_EXT       ||
         kind == HIP_COPY_KIND_D2H_RECT_EXT  ||
         kind == HIP_COPY_KIND_D2H_IMAGE_EXT;
}

/**
 * @brief GPU activity record returned by hipProfilerGetRecordsExt().
 *
 * Fixed size: 128 bytes.  Fields beyond the active payload are reserved for
 * future use and must be treated as zero by callers.
 *
 * When embedded as HipApiRecordExt::gpu this struct describes the first (or
 * only) GPU operation for the API call.  gpu_op_count and gpu_ops expose all
 * operations (>1 for hipGraphLaunch with multiple nodes).
 */
typedef struct HipGpuActivityExt {
  union {
    uint64_t _flags_u64;        /**< Raw 64-bit access to the packed flags word */
    struct {
      uint64_t op        : 3;   /**< HipGpuOpExt value */
      uint64_t is_graph  : 1;   /**< Set when the op was launched from a HIP graph */
      uint64_t copy_kind : 4;   /**< HipCopyKindExt; valid when op==HIP_OP_COPY_EXT */
      uint64_t           : 8;   /**< Unnamed reserved bits — must be zero */
      uint64_t device_id : 16;  /**< Device index (up to 65535 devices) */
      uint64_t queue_id  : 16;  /**< Queue/stream index (up to 65535 queues) */
      uint64_t           : 16;  /**< Unnamed reserved bits — must be zero */
    };
  };
  uint64_t    begin_ns;         /**< GPU begin timestamp (ns) */
  uint64_t    end_ns;           /**< GPU end timestamp (ns) */
  union {
    uint64_t    bytes;          /**< Bytes transferred (op==HIP_OP_COPY_EXT) */
    const char* kernel_name;    /**< Kernel name        (op==HIP_OP_DISPATCH_EXT, may be NULL) */
  };
  /* Originally _pad1[96].  First 16 bytes repurposed for multi-op support;
   * remaining 80 bytes stay reserved and must be treated as zero. */
  uint32_t          gpu_op_count;            /**< Total GPU ops (0=none, 1=single, >1=graph) */
  uint32_t          _reserved_u32;           /**< Reserved — must be zero */
  const struct HipGpuActivityExt* gpu_ops;  /**< gpu_ops[0..gpu_op_count-1]; for gpu_op_count==1
                                                  points to the enclosing gpu field itself */
  uint8_t     _pad1[80];                     /**< Remaining reserved padding — must be zero */
} HipGpuActivityExt;
#ifdef __cplusplus
static_assert(sizeof(HipGpuActivityExt) == 128, "HipGpuActivityExt must be 128 bytes");
#endif

/**
 * @brief Per-HIP-API profiling record returned by hipProfilerGetRecordsExt().
 *
 * Timestamps are nanoseconds from the Unix epoch (clock_gettime CLOCK_REALTIME
 * on Linux, or QueryPerformanceCounter-based on Windows).
 *
 * The gpu field is valid only when has_gpu_activity != 0.  It always holds the
 * first GPU operation.  gpu.gpu_op_count gives the total count; gpu.gpu_ops
 * points to all operations (profiler-owned flat buffer).
 *
 * Fixed size: 256 bytes (48-byte CPU header + 128-byte HipGpuActivityExt + 80-byte pad).
 */
typedef struct {
  /* CPU call info — first 128-byte half */
  const char*  api_name;              /**< Points into the DLL's API name table; never NULL */
  union {
    uint64_t _flags_u64;              /**< Raw 64-bit access to the flags word */
    struct {
      uint64_t has_gpu_activity : 1;  /**< Non-zero when gpu field is valid */
      uint64_t                  : 63; /**< Unnamed reserved bits — must be zero */
    };
  };
  uint64_t          thread_id;        /**< Hash of std::thread::id */
  uint64_t          start_ns;         /**< CPU call begin (ns) */
  uint64_t          end_ns;           /**< CPU call end (ns) */
  hipStream_t       stream;           /**< Stream argument, or NULL for default/no-stream APIs */
  /* Padding to 128-byte CPU half: 128 - (8+8+8+8+8+8) = 80 */
  uint8_t           _pad1[80];
  /* GPU activity — second 128-byte half (valid when has_gpu_activity != 0) */
  HipGpuActivityExt gpu;
} HipApiRecordExt;
#ifdef __cplusplus
static_assert(sizeof(HipApiRecordExt) == 256, "HipApiRecordExt must be 256 bytes");
#endif

/**
 * @brief Enable built-in profiling at runtime.
 *
 * Equivalent to setting GPU_CLR_PROFILE_OUTPUT=<path> before process start,
 * but can be called at any time from application code.
 */
hipError_t hipProfilerEnableExt(void);

/**
 * @brief Disable built-in profiling.  Already-collected records are kept.
 *
 * Drains all pending GPU work before returning, ensuring that every in-flight
 * GPU activity callback has fired and all records are fully populated.
 *
 * Must be called before hipProfilerGetRecordsExt() or hipProfilerResetExt().
 * Calling either of those without a prior disable may return incomplete records
 * or free memory that is still being written by the GPU completion thread.
 */
hipError_t hipProfilerDisableExt(void);

/**
 * @brief Return the raw profiler chunk array without copying.
 *
 * Records are stored internally as an array of fixed-size chunks.  This call
 * exposes those chunks directly — no allocation, no copy.
 *
 * Iteration pattern:
 * @code
 *   const HipApiRecordExt* const* chunks;
 *   size_t chunk_count, chunk_size, total;
 *   hipProfilerGetRecordsExt(&chunks, &chunk_count, &chunk_size, &total);
 *   for (size_t c = 0; c < chunk_count; ++c) {
 *     size_t n = (total - c * chunk_size < chunk_size)
 *                ? total - c * chunk_size : chunk_size;
 *     for (size_t i = 0; i < n; ++i) {
 *       const HipApiRecordExt* r = &chunks[c][i];
 *       // use r->api_name, r->start_ns, r->end_ns, r->gpu, ...
 *     }
 *   }
 * @endcode
 *
 * Lifetime: the returned pointers are owned by the profiler and remain valid
 * until the next call to hipProfilerResetExt().  Do NOT call
 * hipProfilerResetExt() while iterating.
 *
 * Note: HipApiRecordExt::_pad1 is used internally to store a spill vector for
 * multi-op graph launches.  Treat it as opaque; do not read or write it.
 * Use gpu.gpu_ops[0..gpu_op_count-1] to access all GPU operations.
 *
 * @param[out] chunks       Set to the profiler's internal chunk pointer array.
 * @param[out] chunk_count  Number of chunks (length of the chunks array).
 * @param[out] chunk_size   Capacity of each chunk in records.
 * @param[out] total_count  Total number of valid records across all chunks.
 */
hipError_t hipProfilerGetRecordsExt(const HipApiRecordExt* const** chunks,
                                     size_t* chunk_count,
                                     size_t* chunk_size,
                                     size_t* total_count);

/**
 * @brief Clear all accumulated records and free internal storage.
 */
hipError_t hipProfilerResetExt(void);

#ifdef __cplusplus
}
#endif

#endif /* HIP_PROFILER_EXT_H */
