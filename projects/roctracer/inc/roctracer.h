////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2015, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 AMD Research and AMD HSA Software Development
//
//                 Advanced Micro Devices, Inc.
//
//                 www.amd.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//
// ROC Profiler API
//
// The goal of the implementation is to provide a HW specific low-level
// performance analysis interface for profiling of GPU compute applications.
// The profiling includes HW performance counters (PMC) with complex
// performance metrics and thread traces (SQTT). The profiling is supported
// by the SQTT, PMC and Callback APIs.
//
// The library can be used by a tool library loaded by HSA runtime or by
// higher level HW independent performance analysis API like PAPI.
//
// The library is written on C and will be based on AQLprofile AMD specific
// HSA extension. The library implementation requires HSA API intercepting and
// a profiling queue supporting a submit callback interface.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef INC_ROCTRACER_H_
#define INC_ROCTRACER_H_

#include <stdint.h>

#include <hip/hip_runtime.h>
#include <hip/hip_cbapi.h>

#define ROCTRACER_VERSION_MAJOR 1
#define ROCTRACER_VERSION_MINOR 0

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

////////////////////////////////////////////////////////////////////////////////
// Returning library version
uint32_t roctracer_version_major();
uint32_t roctracer_version_minor();

////////////////////////////////////////////////////////////////////////////////
// Library errors enumaration
typedef enum {
  ROCTRACER_STATUS_SUCCESS = 0,
  ROCTRACER_STATUS_ERROR = 1,
  ROCTRACER_STATUS_UNINIT = 2,
  ROCTRACER_STATUS_BREAK = 3,
  ROCTRACER_STATUS_BAD_DOMAIN = 4,
  ROCTRACER_STATUS_HIP_API_ERR = 5,
} roctracer_status_t;

////////////////////////////////////////////////////////////////////////////////
// Returning the last error
const char* roctracer_error_string();

////////////////////////////////////////////////////////////////////////////////
// Traced runtime API domains

// Traced API domains
typedef enum {
  ROCTRACER_DOMAIN_ANY = 0,                        // Any domain
  ROCTRACER_DOMAIN_HIP_API = 1,                    // HIP domain
  ROCTRACER_DOMAIN_HCC_OPS = 2,                    // HCC domain
  ROCTRACER_DOMAIN_NUMBER
} roctracer_domain_t;

// Traced calls ID enumeration
typedef hip_cb_id_t roctracer_hip_api_cid_t;

// Correlation ID type
typedef uint64_t roctracer_correletion_id_t;

// Validates tracing domains revisions consistency
roctracer_status_t roctracer_validate_domains();

// Return ID string by given domain and activity/API ID
// NULL returned on the error and the library errno is set
const char* roctracer_id_string(
  const uint32_t& domain,                                               // API domain
  const uint32_t& cid);                                                 // API call ID

////////////////////////////////////////////////////////////////////////////////
// Callback API
//
// ROC profiler frontend provides support for runtime API callbacks and activity
// records logging. The API callbacks provide the API calls arguments and are
// called on different phases, on enter, on exit, on kernel completion.
// Methods return non-zero on error and library errno is set.

// API callback phase
typedef enum {
  ROCTRACER_API_PHASE_ENTER = 0,
  ROCTRACER_API_PHASE_EXIT = 1,
  ROCTRACER_API_PHASE_COMPLETE = 2,
} roctracer_feature_kind_t;

// API calback data
typedef hip_cb_fun_t roctracer_api_callback_t;

// Enable runtime API callbacks
roctracer_status_t roctracer_enable_api_callback(
    roctracer_domain_t domain,                    // runtime API domain
    uint32_t cid,                            // API call ID
    roctracer_api_callback_t callback,                         // callback function pointer
    void* arg);                                     // [in/out] callback arg

// Disable runtime API callbacks
roctracer_status_t roctracer_disable_api_callback(
    roctracer_domain_t domain,                    // runtime API domain
    uint32_t cid);                            // API call ID

////////////////////////////////////////////////////////////////////////////////
// Activity API
//
// The activity records are asynchronously logged to the pool and can be associated
// with the respective API callbacks using the correlation ID. Activity API can
// be used to enable collecting of the records with timestamping data for API
// calls and the kernel submits.
// Methods return non zero on error and library errno is set.

// Roctracer pool type
typedef void roctracer_pool_t;

// Activity record
typedef hip_act_record_t roctracer_record_t;
typedef hip_ops_record_t roctracer_async_record_t;

// Return next record
static inline int roctracer_next_record(
    const roctracer_record_t* record,                                   // [in] record ptr
    const roctracer_record_t** next)                                    // [out] next record ptr
{
  *next = (record->op_id != 0) ?
    reinterpret_cast<const roctracer_async_record_t*>(record) + 1 :
    record + 1;
  return ROCTRACER_STATUS_SUCCESS;
}

// Tracer allocator type
typedef void (*roctracer_allocator_t)(
    char** ptr,                                                         // memory pointer
    size_t size,                                                        // memory size
    void* arg);                                                         // allocator arg

// Pool callback type
typedef void (*roctracer_buffer_callback_t)(
    const char* begin,                                                  // [in] available buffered trace records
    const char* end,                                                    // [in] end of buffered trace records
    void* arg);                                                         // [in/out] callback arg

// Tracer properties
typedef struct {
    uint32_t mode;                                                      // roctracer mode
    size_t buffer_size;                                                 // buffer size
    roctracer_allocator_t alloc_fun;                                    // memory alocator function pointer
    void* alloc_arg;                                                    // memory alocator function pointer
    roctracer_buffer_callback_t buffer_callback_fun;                    // tracer record callback function
    void* buffer_callback_arg;                                          // tracer record callback arg
} roctracer_properties_t;

// Create tracer memory pool
// The first invocation sets the default pool
roctracer_status_t roctracer_open_pool(
    const roctracer_properties_t* properties,                           // tracer pool properties
    roctracer_pool_t** pool = NULL);                                    // [out] returns tracer pool if not NULL,
                                                                        // otherwise sets the default one if it is not set yet
                                                                        // otherwise the error is generated

// Close tracer memory pool
roctracer_status_t roctracer_close_pool(
    roctracer_pool_t* pool = NULL);                                     // [in] memory pool, NULL is a default one

// Return current default pool
// Set new default pool if the argument is not NULL
roctracer_pool_t* roctracer_default_pool(
    roctracer_pool_t* pool = NULL);                                     // [in] new default pool if not NULL

// Enable activity records logging
roctracer_status_t roctracer_enable_api_activity(
    roctracer_domain_t domain,                                      // runtime API domain
    uint32_t activity_kind,                                             // activity kind
    roctracer_pool_t* pool = NULL);                                     // memory pool, NULL is a default one

// Disable activity records logging
roctracer_status_t roctracer_disable_api_activity(
    roctracer_domain_t domain,                                      // runtime API domain
    uint32_t activity_kind);                                            // activity kind

// Flush available activity records
roctracer_status_t roctracer_flush_api_activity(
    roctracer_pool_t* pool = NULL);                                     // memory pool, NULL is a default one

#ifdef __cplusplus
}  // extern "C" block
#endif  // __cplusplus

#endif  // INC_ROCTRACER_H_
