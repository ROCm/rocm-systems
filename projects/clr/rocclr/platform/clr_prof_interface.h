/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @file clr_prof_interface.h
 * @brief CLR Native Profiling Interface — stable C ABI
 *
 * Provides a direct, roctracer-free profiling interface that any client
 * (rpd_tracer, rocprofiler, custom tools) can use by subscribing to typed
 * event callbacks emitted by CLR.
 *
 * Versioning: every struct carries a struct_size field that must be set to
 * sizeof(<struct>) by the caller before passing it to CLR.  CLR uses this
 * to detect ABI mismatches and to safely skip fields added in future versions.
 *
 * Exported from libamdhip64.so.
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Version ─────────────────────────────────────────────────────────────── */

#define CLR_PROF_VERSION_MAJOR 1
#define CLR_PROF_VERSION_MINOR 0

/* ── Opaque subscriber handle ─────────────────────────────────────────────── */

typedef struct clr_prof_subscriber_s* clr_prof_subscriber_t;

/* ── GPU Activity (async, delivered from CLR worker thread) ───────────────── */

typedef enum {
  CLR_PROF_OP_KERNEL_DISPATCH = 0,
  CLR_PROF_OP_MEMCPY          = 1,
  CLR_PROF_OP_BARRIER         = 2,
} clr_prof_gpu_op_t;

typedef struct {
  uint32_t          struct_size;    /**< Must be set to sizeof(clr_prof_gpu_record_t). */
  clr_prof_gpu_op_t op;
  uint64_t          correlation_id; /**< Links this GPU record to the API record that
                                         submitted the work. */
  uint64_t          begin_ns;       /**< GPU hardware start timestamp, nanoseconds. */
  uint64_t          end_ns;         /**< GPU hardware end timestamp, nanoseconds. */
  int32_t           device_id;      /**< Driver node (logical device) id. */
  uint64_t          queue_id;       /**< Queue/stream index on the device. */
  union {
    const char* kernel_name;        /**< Valid when op == CLR_PROF_OP_KERNEL_DISPATCH.
                                         Pointer is valid only during the callback. */
    size_t bytes;                   /**< Valid when op == CLR_PROF_OP_MEMCPY. */
  };
} clr_prof_gpu_record_t;

/**
 * GPU activity callback.  Invoked from a CLR worker thread after the GPU
 * completes the operation.  Must not block for extended periods.
 */
typedef void (*clr_prof_gpu_activity_cb_t)(const clr_prof_gpu_record_t* record, void* user_data);

/* ── HIP API Tracing (synchronous, in-thread at every HIP call) ───────────── */

typedef enum {
  CLR_PROF_PHASE_ENTER = 0,
  CLR_PROF_PHASE_EXIT  = 1,
} clr_prof_phase_t;

typedef struct {
  uint32_t          struct_size;    /**< Must be set to sizeof(clr_prof_api_record_t). */
  clr_prof_phase_t  phase;
  uint32_t          api_id;         /**< hip_api_id_t value identifying the HIP function. */
  uint64_t          correlation_id; /**< CLR-managed monotonic ID; same value on enter
                                         and exit for the same call. */
  uint64_t          timestamp_ns;   /**< Host clock (CLOCK_MONOTONIC) at enter or exit. */
  uint32_t          pid;
  uint32_t          tid;
  /**
   * Pointer to the hip_api_data_t argument union for this api_id.
   * Valid only during the ENTER phase (arguments are populated before the
   * actual HIP implementation runs).  NULL during EXIT phase.
   */
  const void*       api_args;
} clr_prof_api_record_t;

/**
 * HIP API callback.  Called synchronously, in the application thread that
 * issued the HIP call.  Do not call back into HIP from within this callback.
 */
typedef void (*clr_prof_api_cb_t)(const clr_prof_api_record_t* record, void* user_data);

/* ── Code Object / Kernel Symbol Events ──────────────────────────────────── */

typedef enum {
  CLR_PROF_CODE_OBJECT_LOAD   = 0,
  CLR_PROF_CODE_OBJECT_UNLOAD = 1,
} clr_prof_code_object_op_t;

typedef struct {
  uint32_t                   struct_size;
  clr_prof_code_object_op_t  op;
  const char*                kernel_name;      /**< Demangled kernel name. */
  const char*                kernel_name_raw;  /**< Mangled (raw) kernel name. */
  uint64_t                   base_address;     /**< HSA code object load address. */
  int32_t                    device_id;
} clr_prof_code_object_record_t;

typedef void (*clr_prof_code_object_cb_t)(const clr_prof_code_object_record_t* record,
                                          void* user_data);

/* ── Queue Lifecycle Events ───────────────────────────────────────────────── */

typedef enum {
  CLR_PROF_QUEUE_CREATE  = 0,
  CLR_PROF_QUEUE_DESTROY = 1,
} clr_prof_queue_op_t;

typedef struct {
  uint32_t            struct_size;
  clr_prof_queue_op_t op;
  int32_t             device_id;
  uint64_t            queue_id;
} clr_prof_queue_record_t;

typedef void (*clr_prof_queue_cb_t)(const clr_prof_queue_record_t* record, void* user_data);

/* ── Subscriber Callbacks Descriptor ─────────────────────────────────────── */

/**
 * Fill in only the callbacks you need; set unused pointers to NULL.
 * CLR skips callbacks that are NULL, so you pay only for what you use.
 */
typedef struct {
  uint32_t                    struct_size; /**< Must be sizeof(clr_prof_callbacks_t). */
  clr_prof_gpu_activity_cb_t  gpu_activity;
  clr_prof_api_cb_t           hip_api;
  clr_prof_code_object_cb_t   code_object;
  clr_prof_queue_cb_t         queue;
  void*                       user_data;   /**< Passed as-is to every callback. */
} clr_prof_callbacks_t;

/* ── API Filter ───────────────────────────────────────────────────────────── */

/**
 * Restrict which HIP API IDs trigger hip_api callbacks.
 * Pass NULL to clr_prof_subscribe() to trace all APIs.
 */
typedef struct {
  uint32_t        struct_size;
  const uint32_t* api_ids; /**< Array of hip_api_id_t values. */
  uint32_t        count;   /**< Number of entries in api_ids. */
} clr_prof_api_filter_t;

/* ── Registration ─────────────────────────────────────────────────────────── */

/**
 * Register a new subscriber.
 *
 * @param callbacks  Pointer to a caller-filled clr_prof_callbacks_t.  CLR
 *                   copies the contents; the caller can free it after return.
 * @param filter     Optional API filter.  NULL means trace all HIP APIs.
 * @return           Opaque subscriber handle, or NULL on error.
 *
 * Thread-safe.  May be called before or after hipInit().
 */
clr_prof_subscriber_t clr_prof_subscribe(const clr_prof_callbacks_t*  callbacks,
                                          const clr_prof_api_filter_t* filter);

/**
 * Unregister a subscriber.
 *
 * Blocks until any in-flight callbacks for this subscriber have completed,
 * then removes it.  The handle is invalid after this call.
 *
 * Thread-safe.
 */
void clr_prof_unsubscribe(clr_prof_subscriber_t subscriber);

/* ── Utilities ────────────────────────────────────────────────────────────── */

/**
 * Return the CLR-managed correlation ID associated with the current thread.
 * This is the same value reported in clr_prof_api_record_t::correlation_id
 * and clr_prof_gpu_record_t::correlation_id.  Useful to correlate external
 * instrumentation (e.g., roctx markers) with CLR records.
 *
 * Returns 0 when no HIP call is in progress on this thread.
 */
uint64_t clr_prof_get_correlation_id(void);

/** Human-readable name for a hip_api_id_t value (e.g., "hipLaunchKernel"). */
const char* clr_prof_api_name(uint32_t api_id);

/** Human-readable name for a clr_prof_gpu_op_t value. */
const char* clr_prof_gpu_op_name(clr_prof_gpu_op_t op);

/** Populate *major and *minor with the CLR profiling interface version. */
void clr_prof_version(uint32_t* major, uint32_t* minor);

#ifdef __cplusplus
}
#endif
