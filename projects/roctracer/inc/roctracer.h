/*
Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.

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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

////////////////////////////////////////////////////////////////////////////////
//
// ROC Tracer API
//
// ROC-tracer library, Runtimes Generic Callback/Activity APIs.
// The goal of the implementation is to provide a generic independent from
// specific runtime profiler to trace API and asyncronous activity.
//
// The API provides functionality for registering the runtimes API callbacks and
// asyncronous activity records pool support.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef INC_ROCTRACER_H_
#define INC_ROCTRACER_H_

#include <stdint.h>
#include <stddef.h>

#include "inc/roctracer/prof_protocol.h"

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
  ROCTRACER_STATUS_BAD_PARAMETER = 5,
  ROCTRACER_STATUS_HIP_API_ERR = 6,
  ROCTRACER_STATUS_HCC_OPS_ERR = 7,
} roctracer_status_t;

////////////////////////////////////////////////////////////////////////////////
// Returning the last error
const char* roctracer_error_string();

////////////////////////////////////////////////////////////////////////////////
// Traced runtime API domains

// Activity domain type
typedef activity_domain_t roctracer_domain_t;

// Return ID string by given domain and activity/API ID
// NULL returned on the error and the library errno is set
const char* roctracer_id_string(
  const uint32_t& domain,                                 // tracing domain
  const uint32_t& id);                                    // activity ID

////////////////////////////////////////////////////////////////////////////////
// Callback API
//
// ROC tracer provides support for runtime API callbacks and activity
// records logging. The API callbacks provide the API calls arguments and are
// called on different phases, on enter, on exit, on kernel completion.
// Methods return non-zero on error and library errno is set.

typedef activity_rtapi_callback_t roctracer_rtapi_callback_t;

// Enable runtime API callbacks
roctracer_status_t roctracer_enable_api_callback(
    activity_domain_t domain,                             // runtime API domain
    uint32_t kind,                                        // API kind
    uint32_t id,                                          // API call ID
    activity_rtapi_callback_t callback,                   // callback function pointer
    void* arg);                                           // [in/out] callback arg

// Disable runtime API callbacks
roctracer_status_t roctracer_disable_api_callback(
    activity_domain_t domain,                             // runtime API domain
    uint32_t kind,                                        // API kind
    uint32_t id);                                         // API call ID

////////////////////////////////////////////////////////////////////////////////
// Activity API
//
// The activity records are asynchronously logged to the pool and can be associated
// with the respective API callbacks using the correlation ID. Activity API can
// be used to enable collecting of the records with timestamping data for API
// calls and the kernel submits.
// Methods return non zero on error and library errno is set.

// Activity record type
typedef activity_record_t roctracer_record_t;

// Return next record
static inline int roctracer_next_record(
    const activity_record_t* record,                      // [in] record ptr
    const activity_record_t** next)                       // [out] next record ptr
{
  *next = record + 1;
  return ROCTRACER_STATUS_SUCCESS;
}

// Tracer allocator type
typedef void (*roctracer_allocator_t)(
    char** ptr,                                           // memory pointer
    size_t size,                                          // memory size
    void* arg);                                           // allocator arg

// Pool callback type
typedef void (*roctracer_buffer_callback_t)(
    const char* begin,                                    // [in] available buffered trace records
    const char* end,                                      // [in] end of buffered trace records
    void* arg);                                           // [in/out] callback arg

// Tracer properties
typedef struct {
    uint32_t mode;                                        // roctracer mode
    size_t buffer_size;                                   // buffer size
    roctracer_allocator_t alloc_fun;                      // memory alocator function pointer
    void* alloc_arg;                                      // memory alocator function pointer
    roctracer_buffer_callback_t buffer_callback_fun;      // tracer record callback function
    void* buffer_callback_arg;                            // tracer record callback arg
} roctracer_properties_t;

// Tracer memory pool type
typedef void roctracer_pool_t;

// Create tracer memory pool
// The first invocation sets the default pool
roctracer_status_t roctracer_open_pool(
    const roctracer_properties_t* properties,             // tracer pool properties
    roctracer_pool_t** pool = NULL);                      // [out] returns tracer pool if not NULL,
                                                          // otherwise sets the default one if it is not set yet
                                                          // otherwise the error is generated

// Close tracer memory pool
roctracer_status_t roctracer_close_pool(
    roctracer_pool_t* pool = NULL);                       // [in] memory pool, NULL is a default one

// Return current default pool
// Set new default pool if the argument is not NULL
roctracer_pool_t* roctracer_default_pool(
    roctracer_pool_t* pool = NULL);                       // [in] new default pool if not NULL

// Enable activity records logging
roctracer_status_t roctracer_enable_api_activity(
    activity_domain_t domain,                             // runtime API domain
    uint32_t kind,                                        // activity kind
    uint32_t id,                                          // activity ID
    roctracer_pool_t* pool = NULL);                       // memory pool, NULL is a default one

// Disable activity records logging
roctracer_status_t roctracer_disable_api_activity(
    activity_domain_t domain,                             // runtime API domain
    uint32_t kind,                                        // activity kind
    uint32_t id);                                         // activity ID

// Flush available activity records
roctracer_status_t roctracer_flush_api_activity(
    roctracer_pool_t* pool = NULL);                       // memory pool, NULL is a default one

#ifdef __cplusplus
}  // extern "C" block
#endif  // __cplusplus

#endif  // INC_ROCTRACER_H_
