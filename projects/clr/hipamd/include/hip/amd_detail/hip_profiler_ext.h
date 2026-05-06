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
 * @brief GPU operation type stored in hipGpuActivityExt::op.
 */
typedef enum {
  HIP_OP_DISPATCH_EXT = 0, /**< Kernel dispatch */
  HIP_OP_COPY_EXT     = 1, /**< Memory copy (DMA / blit) */
  HIP_OP_BARRIER_EXT  = 2, /**< Barrier / fence */
} HipGpuOpExt;

/**
 * @brief Memory copy direction stored in hipGpuActivityExt::copy_kind.
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
 * When embedded as hipApiRecordExt::gpu this struct describes the first (or
 * only) GPU operation for the API call.  gpu_op_count and gpu_ops expose all
 * operations (>1 for hipGraphLaunch with multiple nodes).
 */
typedef struct hipGpuActivityExt {
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
  /* Originally _pad1[96].  First 16 bytes repurposed for multi-op linked list;
   * next 40 bytes repurposed for op-specific payload (dispatch dims/args or copy src/dst);
   * remaining 40 bytes stay reserved and must be treated as zero by external callers. */
  uint32_t                   gpu_op_count; /**< Total GPU ops (0=none, 1=in gpu field, >1=graph) */
  uint32_t                   _reserved_u32;/**< Reserved — must be zero */
  const struct hipGpuActivityExt* next;    /**< Next spill node (ops 2..N); NULL at tail or when
                                                 gpu_op_count <= 1.  rec->gpu.next is the head. */
  /* Op-specific payload — 40 bytes, two arms sharing the same storage. */
  union {
    struct { /* op==HIP_OP_DISPATCH_EXT: grid/block dims and kernel arg blob.
              * Dims are valid when launched from a single hipLaunchKernel or a graph node
              * captured at hipGraphInstantiate time; zero for barriers/copies. */
      uint32_t    grid_x;          /**< Grid X (number of blocks) */
      uint32_t    grid_y;          /**< Grid Y */
      uint32_t    grid_z;          /**< Grid Z */
      uint32_t    block_x;         /**< Block X (threads per block) */
      uint32_t    block_y;         /**< Block Y */
      uint32_t    block_z;         /**< Block Z */
      const uint8_t* kernel_args;  /**< Packed arg blob, or NULL.  Owned by the profiler. */
      uint32_t    kernel_args_size;/**< Byte length of kernel_args blob */
      uint32_t    _reserved_dispatch; /**< Reserved — must be zero */
    };
    struct { /* op==HIP_OP_COPY_EXT: source and destination addresses.
              * Populated for both direct copies and graph copy nodes captured at
              * hipGraphInstantiate time.  NULL when address is unavailable. */
      const void* src;             /**< Source address (host or device) */
      const void* dst;             /**< Destination address (host or device) */
      uint8_t     _reserved_copy[24]; /**< Reserved — must be zero */
    };
  };
  uint8_t     _pad1[40];           /**< Remaining reserved padding — must be zero */
} hipGpuActivityExt;
#ifdef __cplusplus
static_assert(sizeof(hipGpuActivityExt) == 128, "hipGpuActivityExt must be 128 bytes");
#endif

/**
 * @brief Per-HIP-API profiling record returned by hipProfilerGetRecordsExt().
 *
 * Timestamps are nanoseconds from the Unix epoch (clock_gettime CLOCK_REALTIME
 * on Linux, or QueryPerformanceCounter-based on Windows).
 *
 * The gpu field is valid only when has_gpu_activity != 0.  It always holds the
 * first GPU operation.  gpu.gpu_op_count gives the total count; for graph launches
 * with more than one op, gpu.next is the head of the spill linked list (ops 2..N).
 *
 * Fixed size: 256 bytes (48-byte CPU header + 8-byte _spill_tail + 24-byte memory ptrs +
 *             4-byte chunk_id + 44-byte pad + 128-byte hipGpuActivityExt).
 *
 * Dispatch grid/block dims and kernel argument blobs are stored in the hipGpuActivityExt
 * dispatch union arm (grid_x/y/z, block_x/y/z, kernel_args, kernel_args_size).
 * Copy source/destination addresses are stored in the copy union arm (src, dst).
 * Both are available per-GPU-op for direct launches and graph nodes.
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
  /* Internal tail pointer for the gpu spill linked list — used by the profiler
   * runtime to achieve O(1) append without scanning to the end of the list.
   * Callers must treat this field as opaque and must not read or write it. */
  const struct hipGpuActivityExt* _spill_tail; /**< Internal — do not use */
  /* Memory operation pointers and size.
   * memory1: destination/allocated pointer (hipMalloc, hipMemcpy dst, hipGraphLaunch exec handle).
   * memory2: source pointer for copies.
   * size:    byte count for allocations, copies, and memsets.
   * Zero for kernel launches and other non-memory APIs. */
  void*    memory1;  /**< Dst/allocated ptr for memory ops, or graphExec for hipGraphLaunch */
  void*    memory2;  /**< Src ptr for copies */
  uint64_t size;     /**< Bytes for allocs/copies/sets */
  /* chunk_id: monotonically increasing sequence number assigned when the record is
   * allocated.  Used by hipProfilerRegisterChunkCallbackExt to order delivery.
   * The 4 bytes come from shrinking _pad1 by 4; struct size stays 256. */
  uint32_t          chunk_id;         /**< Delivery sequence number — assigned at allocation */
  /* Remaining padding to complete the 128-byte CPU half.
   * Reserved — must be treated as zero by callers. */
  uint8_t           _pad1[44];
  /* GPU activity — second 128-byte half (valid when has_gpu_activity != 0) */
  hipGpuActivityExt gpu;
} hipApiRecordExt;
#ifdef __cplusplus
static_assert(sizeof(hipApiRecordExt) == 256, "hipApiRecordExt must be 256 bytes");
#endif

/**
 * @brief Enable built-in profiling at runtime.
 *
 * Increments an internal reference counter.  Profiling becomes active on the
 * first call (ref count 0 → 1).  Subsequent calls increase the counter further,
 * allowing multiple independent profiling sessions to coexist — profiling stays
 * active until the matching number of hipProfilerDisableExt() calls has been
 * made.
 *
 * @param[out] start_record_id  Set to the record counter value at the moment
 *                              profiling was enabled (i.e. the index of the
 *                              first record that will be attributed to this
 *                              session).  May be NULL.
 * @param[in]  state            Feature flags for this profiling session.
 *                              Pass 0 for default behaviour (all features off).
 *                              Reserved for future use — all bits must be zero
 *                              in this version; non-zero values are accepted but
 *                              ignored.
 * @return hipSuccess
 */
hipError_t hipProfilerEnableExt(uint64_t* start_record_id, uint64_t state);

/**
 * @brief Disable built-in profiling.  Already-collected records are kept.
 *
 * Decrements the internal reference counter.  When the counter reaches zero
 * all pending GPU work is drained before profiling is deactivated, ensuring
 * that every in-flight GPU activity callback has fired and all records for
 * this session are fully populated.
 *
 * Must be paired with a prior call to hipProfilerEnableExt().  Calling this
 * function more times than hipProfilerEnableExt() has no effect once the
 * counter is already zero.
 *
 * @param[out] end_record_id  Set to the record counter value at the moment
 *                            profiling was disabled (i.e. one past the index
 *                            of the last record for this session).  May be NULL.
 * @return hipSuccess
 */
hipError_t hipProfilerDisableExt(uint64_t* end_record_id);

/**
 * @brief Return the raw profiler chunk array without copying.
 *
 * Records are stored internally as an array of fixed-size chunks.  This call
 * exposes those chunks directly — no allocation, no copy.
 *
 * Iteration pattern:
 * @code
 *   const hipApiRecordExt* const* chunks;
 *   size_t chunk_count, chunk_size, total;
 *   hipProfilerGetRecordsExt(&chunks, &chunk_count, &chunk_size, &total);
 *   for (size_t c = 0; c < chunk_count; ++c) {
 *     size_t n = (total - c * chunk_size < chunk_size)
 *                ? total - c * chunk_size : chunk_size;
 *     for (size_t i = 0; i < n; ++i) {
 *       const hipApiRecordExt* r = &chunks[c][i];
 *       // use r->api_name, r->start_ns, r->end_ns, r->gpu, ...
 *     }
 *   }
 * @endcode
 *
 * Lifetime: the returned pointers are owned by the profiler and remain valid
 * for the lifetime of the process.
 *
 * Note: hipApiRecordExt::_spill_tail is used internally by the profiler runtime
 * to maintain the gpu spill linked list.  Treat it as opaque; do not read or write it.
 * Use gpu.gpu_ops[0..gpu_op_count-1] to access all GPU operations.
 *
 * @param[out] chunks       Set to the profiler's internal chunk pointer array.
 * @param[out] chunk_count  Number of chunks (length of the chunks array).
 * @param[out] chunk_size   Capacity of each chunk in records.
 * @param[out] total_count  Total number of valid records across all chunks.
 */
hipError_t hipProfilerGetRecordsExt(const hipApiRecordExt* const** chunks,
                                    size_t* chunk_count,
                                    size_t* chunk_size,
                                    size_t* total_count);

/**
 * @brief Callback type for streaming delivery of completed activity records.
 *
 * @param records     Pointer to the slab — an array of count records aligned to a
 *                    kChunkSize boundary.  Valid only for the duration of the callback;
 *                    do not store this pointer.  The profiler frees the records immediately
 *                    after the callback returns.
 * @param count       Number of valid records in the slab (always kChunkSize except for the
 *                    last partial slab delivered at exit).
 * @param chunk_id    chunk_id of records[0].  Records within the slab are contiguous, so
 *                    the last record has chunk_id == chunk_id + count - 1.
 * @param user_data   Opaque pointer supplied to hipProfilerRegisterChunkCallbackExt.
 */
typedef void (*hipProfilerChunkCallback)(
    const hipApiRecordExt* records,
    uint32_t               count,
    uint32_t               chunk_id,
    void*                  user_data
);

/**
 * @brief Register a chunk callback for streaming delivery of completed activity records.
 *
 * When registered, records are delivered in chunks as GPU activity completes,
 * and freed immediately after the callback returns.  Records pointer is valid
 * only for the duration of the callback — do not store it.
 *
 * If not registered, all records are buffered and written at exit (existing
 * behaviour).  Must be called before hipProfilerEnableExt or setting
 * GPU_CLR_PROFILE_OUTPUT to take effect.
 *
 * @param cb         Chunk delivery callback.  Pass NULL to unregister.
 * @param user_data  Opaque pointer forwarded to every callback invocation.
 * @return hipSuccess
 */
hipError_t hipProfilerRegisterChunkCallbackExt(hipProfilerChunkCallback cb, void* user_data);

#ifdef __cplusplus
}
#endif

#endif /* HIP_PROFILER_EXT_H */
