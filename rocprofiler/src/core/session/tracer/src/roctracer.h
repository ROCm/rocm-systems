/* Copyright (c) 2018-2022 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#ifndef SRC_TOOLS_TRACER_SRC_ROCTRACER_H_
#define SRC_TOOLS_TRACER_SRC_ROCTRACER_H_

#include <hip/hip_runtime.h>
#include <hip/hip_deprecated.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <stddef.h>
#include <stdint.h>

#include <optional>
#include <string>

#include "hip_ostream_ops.h"
#include "hsa_ostream_ops.h"
#include "hsa_prof_str.h"
#include "rocprofiler.h"
#include "src/core/memory/generic_buffer.h"

typedef struct {
  rocprofiler_session_id_t session_id;
  rocprofiler_buffer_id_t buffer_id;
} session_buffer_id_t;

typedef session_buffer_id_t roctracer_pool_t;

/* Correlation id */
typedef uint64_t activity_correlation_id_t;

typedef uint32_t activity_kind_t;
typedef uint32_t activity_op_t;

typedef uint64_t roctracer_timestamp_t;

typedef rocprofiler_tracer_activity_domain_t roctracer_domain_t;
typedef rocprofiler_tracer_activity_domain_t activity_domain_t;


// Prof_Protocol
/* Activity record type */
typedef struct activity_record_s {
  uint32_t domain;      /* activity domain id */
  activity_kind_t kind; /* activity kind */
  activity_op_t op;     /* activity op */
  union {
    struct {
      activity_correlation_id_t correlation_id; /* activity ID */
      roctracer_timestamp_t begin_ns;           /* host begin timestamp */
      roctracer_timestamp_t end_ns;             /* host end timestamp */
    };
    struct {
      uint32_t se;    /* sampled SE */
      uint64_t cycle; /* sample cycle */
      uint64_t pc;    /* sample PC */
    } pc_sample;
  };
  union {
    struct {
      int device_id;     /* device id */
      uint64_t queue_id; /* queue id */
    };
    struct {
      uint32_t process_id; /* device id */
      uint32_t thread_id;  /* thread id */
    };
    struct {
      activity_correlation_id_t external_id; /* external correlation id */
    };
  };
  union {
    size_t bytes;            /* data size bytes */
    const char* kernel_name; /* kernel name */
    const char* mark_message;
  };
} activity_record_t;

typedef activity_record_t roctracer_record_t;

/* Activity sync callback type */
typedef void (*activity_sync_callback_t)(activity_domain_t cid, activity_record_t* record,
                                         const void* data, void* arg);
/* Activity async callback type */
typedef void (*activity_async_callback_t)(activity_domain_t op, void* record, void* arg);


/* API callback type */
typedef void (*activity_rtapi_callback_t)(activity_domain_t domain, uint32_t cid, const void* data,
                                          void* arg);
typedef activity_rtapi_callback_t roctracer_rtapi_callback_t;

typedef roctracer_timestamp_t (*roctracer_get_timestamp_t)();
typedef rocprofiler_timestamp_t (*rocprofiler_get_timestamp_t)();

typedef uint32_t activity_kind_t;
typedef uint32_t activity_op_t;

/* API callback phase */
typedef enum { ACTIVITY_API_PHASE_ENTER = 0, ACTIVITY_API_PHASE_EXIT = 1 } activity_api_phase_t;

const char* roctracer_op_string(uint32_t domain, uint32_t op);

/* Trace record types */

/**
 * Memory pool allocator callback.
 *
 * If \p *ptr is NULL, then allocate memory of \p size bytes and save address
 * in \p *ptr.
 *
 * If \p *ptr is non-NULL and size is non-0, then reallocate the memory at \p
 * *ptr with size \p size and save the address in \p *ptr. The memory will have
 * been allocated by the same callback.
 *
 * If \p *ptr is non-NULL and size is 0, then deallocate the memory at \p *ptr.
 * The memory will have been allocated by the same callback.
 *
 * \p size is the size of the memory allocation or reallocation, or 0 if
 * deallocating.
 *
 * \p arg Argument provided
 */
typedef void (*roctracer_allocator_t)(char** ptr, size_t size, void* arg);

/**
 * Memory pool buffer callback.
 *
 * The callback that will be invoked when a memory pool buffer becomes full or
 * is flushed.
 *
 * \p begin pointer to first entry entry in the buffer.
 *
 * \p end pointer to one past the end entry in the buffer.
 *
 * \p arg the argument specified when the callback was defined.
 */
typedef void (*roctracer_buffer_callback_t)(const char* begin, const char* end, void* arg);

/**
 * Memory pool properties.
 *
 * Defines the properties when a tracer memory pool is created.
 */
typedef struct {
  /**
   * ROC Tracer mode.
   */
  uint32_t mode;

  /**
   * Size of buffer in bytes.
   */
  size_t buffer_size;

  /**
   * The allocator function to use to allocate and deallocate the buffer. If
   * NULL then \p malloc, \p realloc, and \p free are used.
   */
  roctracer_allocator_t alloc_fun;

  /**
   * The argument to pass when invoking the \p alloc_fun allocator.
   */
  void* alloc_arg;

  /**
   * The function to call when a buffer becomes full or is flushed.
   */
  roctracer_buffer_callback_t buffer_callback_fun;

  /**
   * The argument to pass when invoking the \p buffer_callback_fun callback.
   */
  void* buffer_callback_arg;
} roctracer_properties_t;

/**
 * ROC Tracer API status codes.
 */
typedef enum {
  /**
   * The function has executed successfully.
   */
  ROCTRACER_STATUS_SUCCESS = 0,
  /**
   * A generic error has occurred.
   */
  ROCTRACER_STATUS_ERROR = -1,
  /**
   * The domain ID is invalid.
   */
  ROCTRACER_STATUS_ERROR_INVALID_DOMAIN_ID = -2,
  /**
   * An invalid argument was given to the function.
   */
  ROCTRACER_STATUS_ERROR_INVALID_ARGUMENT = -3,
  /**
   * No default pool is defined.
   */
  ROCTRACER_STATUS_ERROR_DEFAULT_POOL_UNDEFINED = -4,
  /**
   * The default pool is already defined.
   */
  ROCTRACER_STATUS_ERROR_DEFAULT_POOL_ALREADY_DEFINED = -5,
  /**
   * Memory allocation error.
   */
  ROCTRACER_STATUS_ERROR_MEMORY_ALLOCATION = -6,
  /**
   * External correlation ID pop mismatch.
   */
  ROCTRACER_STATUS_ERROR_MISMATCHED_EXTERNAL_CORRELATION_ID = -7,
  /**
   * The operation is not currently implemented.  This error may be reported by
   * any function.  Check the \ref known_limitations section to determine the
   * status of the library implementation of the interface.
   */
  ROCTRACER_STATUS_ERROR_NOT_IMPLEMENTED = -8,
  /**
   * Deprecated error code.
   */
  ROCTRACER_STATUS_UNINIT = 2,
  /**
   * Deprecated error code.
   */
  ROCTRACER_STATUS_BREAK = 3,
  /**
   * Deprecated error code.
   */
  ROCTRACER_STATUS_BAD_DOMAIN = ROCTRACER_STATUS_ERROR_INVALID_DOMAIN_ID,
  /**
   * Deprecated error code.
   */
  ROCTRACER_STATUS_BAD_PARAMETER = ROCTRACER_STATUS_ERROR_INVALID_ARGUMENT,
  /**
   * Deprecated error code.
   */
  ROCTRACER_STATUS_HIP_API_ERR = 6,
  /**
   * Deprecated error code.
   */
  ROCTRACER_STATUS_HIP_OPS_ERR = 7,
  /**
   * Deprecated error code.
   */
  ROCTRACER_STATUS_HCC_OPS_ERR = ROCTRACER_STATUS_HIP_OPS_ERR,
  /**
   * Deprecated error code.
   */
  ROCTRACER_STATUS_HSA_ERR = 7,
  /**
   * Deprecated error code.
   */
  ROCTRACER_STATUS_ROCTX_ERR = 8,
} roctracer_status_t;

/**
 * Query textual name of an operation of a domain.
 * @param[in] domain Domain being queried.
 * @param[in] op Operation within \p domain.
 * @param[in] kind \todo Define kind.
 * @return Returns the NUL terminated string for the operation name, or NULL if
 * the domain or operation are invalid.  The string is owned by the ROC Tracer
 * library.
 */
const char* roctracer_op_string(uint32_t domain, uint32_t op, uint32_t kind);

/**
 * Query the operation code given a domain and the name of an operation.
 * @param[in] domain The domain being queried.
 * @param[in] str The NUL terminated name of the operation name being queried.
 * @param[out] op The operation code.
 * @param[out] kind If not NULL then the operation kind code.
 */
void roctracer_op_code(uint32_t domain, const char* str, uint32_t* op, uint32_t* kind);

/**
 * Set the properties of a domain.
 * @param[in] domain The domain.
 * @param[in] properties The properties. Each domain defines its own type for
 * the properties. Some domains require the properties to be set before they
 * can be enabled.
 */
void roctracer_set_properties(roctracer_domain_t domain, void* properties);

/**
 * Enable runtime API callback for a specific operation of a domain.
 * @param domain The domain.
 * @param op The operation ID in \p domain.
 * @param callback The callback to invoke each time the operation is performed
 * on entry and exit.
 * @param pool Value to pass as last argument of \p callback.
 */
void roctracer_enable_op_callback(roctracer_domain_t domain, uint32_t op,
                                  roctracer_rtapi_callback_t callback);

/**
 * Enable runtime API callback for all operations of a domain.
 * @param domain The domain
 * @param callback The callback to invoke each time the operation is performed
 * on entry and exit.
 * @param arg Value to pass as last argument of \p callback.
 */
void roctracer_enable_domain_callback(roctracer_domain_t domain,
                                      roctracer_rtapi_callback_t callback,
                                      void* user_data = nullptr);

/**
 * Disable runtime API callback for a specific operation of a domain.
 * @param domain The domain
 * @param op The operation in \p domain.
 */
void roctracer_disable_op_callback(roctracer_domain_t domain, uint32_t op);

/**
 * Disable runtime API callback for all operations of a domain.
 * @param domain The domain
 */
void roctracer_disable_domain_callback(roctracer_domain_t domain);

/**
 * Enable activity record logging for a specified operation of a domain using
 * the default memory pool.
 * @param[in] domain The domain.
 * @param[in] op The activity operation ID in \p domain.
 */
void roctracer_enable_op_activity(roctracer_domain_t domain, uint32_t op, roctracer_pool_t pool);

/**
 * Enable activity record logging for all operations of a domain using the
 * default memory pool.
 * @param[in] domain The domain.
 */
void roctracer_enable_domain_activity(roctracer_domain_t domain, roctracer_pool_t pool);

/**
 * Disable activity record logging for a specified operation of a domain.
 * @param[in] domain The domain.
 * @param[in] op The activity operation ID in \p domain.
 */
void roctracer_disable_op_activity(roctracer_domain_t domain, uint32_t op);

/**
 * Disable activity record logging for all operations of a domain.
 * @param[in] domain The domain.
 */
void roctracer_disable_domain_activity(roctracer_domain_t domain);

std::optional<std::string> GetHipKernelName(uint32_t cid, hip_api_data_t* data);

// HIP Support
typedef enum {
  HIP_OP_ID_DISPATCH = 0,
  HIP_OP_ID_COPY = 1,
  HIP_OP_ID_BARRIER = 2,
  HIP_OP_ID_NUMBER = 3
} hip_op_id_t;

// HSA Support
// HSA OP ID enumeration
enum hsa_op_id_t {
  HSA_OP_ID_DISPATCH = 0,
  HSA_OP_ID_COPY = 1,
  HSA_OP_ID_BARRIER = 2,
  HSA_OP_ID_RESERVED1 = 3,
  HSA_OP_ID_NUMBER
};

// HSA EVT ID enumeration
enum hsa_evt_id_t {
  HSA_EVT_ID_ALLOCATE = 0,  // Memory allocate callback
  HSA_EVT_ID_DEVICE = 1,    // Device assign callback
  HSA_EVT_ID_MEMCOPY = 2,   // Memcopy callback
  HSA_EVT_ID_SUBMIT = 3,    // Packet submission callback
  HSA_EVT_ID_KSYMBOL = 4,   // Loading/unloading of kernel symbol
  HSA_EVT_ID_CODEOBJ = 5,   // Loading/unloading of device code object
  HSA_EVT_ID_NUMBER
};

struct hsa_ops_properties_t {
  void* reserved1[4];
};

// ROCTx Support
typedef uint64_t roctx_range_id_t;

/**
 *  ROCTX API ID enumeration
 */
enum roctx_api_id_t {
  ROCTX_API_ID_roctxMarkA = 0,
  ROCTX_API_ID_roctxRangePushA = 1,
  ROCTX_API_ID_roctxRangePop = 2,
  ROCTX_API_ID_roctxRangeStartA = 3,
  ROCTX_API_ID_roctxRangeStop = 4,
  ROCTX_API_ID_NUMBER,
};

/**
 *  ROCTX callbacks data type
 */
typedef struct roctx_api_data_s {
  union {
    struct {
      const char* message;
      roctx_range_id_t id;
    };
    struct {
      const char* message;
    } roctxMarkA;
    struct {
      const char* message;
    } roctxRangePushA;
    struct {
      const char* message;
    } roctxRangePop;
    struct {
      const char* message;
      roctx_range_id_t id;
    } roctxRangeStartA;
    struct {
      const char* message;
      roctx_range_id_t id;
    } roctxRangeStop;
  } args;
} roctx_api_data_t;

// External Support
/* Extension opcodes */
typedef enum { ACTIVITY_EXT_OP_MARK = 0, ACTIVITY_EXT_OP_EXTERN_ID = 1 } activity_ext_op_t;

typedef void (*roctracer_start_cb_t)();
typedef void (*roctracer_stop_cb_t)();
typedef struct {
  roctracer_start_cb_t start_cb;
  roctracer_stop_cb_t stop_cb;
} roctracer_ext_properties_t;

// Tracing start
void roctracer_start();

// Tracing stop
void roctracer_stop();

// Notifies that the calling thread is entering an external region.
// Push an external correlation id for the calling thread.
void roctracer_activity_push_external_correlation_id(activity_correlation_id_t id);

// Notifies that the calling thread is leaving an external region.
// Pop an external correlation id for the calling thread.
// 'lastId' returns the last external correlation if not NULL
void roctracer_activity_pop_external_correlation_id(activity_correlation_id_t* last_id);

#endif /* SRC_TOOLS_TRACER_SRC_ROCTRACER_H_ */
