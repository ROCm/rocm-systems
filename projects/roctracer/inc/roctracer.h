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
#ifndef __cplusplus
#include <stdbool.h>
#endif

#include <ext/prof_protocol.h>

#define ROCTRACER_VERSION_MAJOR 4
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
  ROCTRACER_STATUS_HSA_ERR = 7,
  ROCTRACER_STATUS_ROCTX_ERR = 8,
} roctracer_status_t;

////////////////////////////////////////////////////////////////////////////////
// Returning the last error
const char* roctracer_error_string();

////////////////////////////////////////////////////////////////////////////////
// Traced runtime domains

// Activity domain type
typedef activity_domain_t roctracer_domain_t;

// Return Op string by given domain and Op code
// NULL returned on the error and the library errno is set
const char* roctracer_op_string(
  uint32_t domain,                                        // tracing domain
  uint32_t op,                                            // activity op ID
  uint32_t kind);                                         // activity kind

// Return Op code and kind by given string
roctracer_status_t roctracer_op_code(
  uint32_t domain,                                        // tracing domain
  const char* str,                                        // [in] op string
  uint32_t* op,                                           // [out] op code
  uint32_t* kind);                                        // [out] op kind code if not NULL

////////////////////////////////////////////////////////////////////////////////
// Callback API
//
// ROC tracer provides support for runtime API callbacks and activity
// records logging. The API callbacks provide the API calls arguments and are
// called on different phases, on enter, on exit, on kernel completion.
// Methods return non-zero on error and library errno is set.

// Runtime API callback type
typedef activity_rtapi_callback_t roctracer_rtapi_callback_t;

// Enable runtime API callbacks
roctracer_status_t roctracer_enable_op_callback(
    activity_domain_t domain,                             // tracing domain
    uint32_t op,                                          // API call ID
    activity_rtapi_callback_t callback,                   // callback function pointer
    void* arg);                                           // [in/out] callback arg
roctracer_status_t roctracer_enable_domain_callback(
    activity_domain_t domain,                             // tracing domain
    activity_rtapi_callback_t callback,                   // callback function pointer
    void* arg);                                           // [in/out] callback arg
roctracer_status_t roctracer_enable_callback(
    activity_rtapi_callback_t callback,                   // callback function pointer
    void* arg);                                           // [in/out] callback arg

// Disable runtime API callbacks
roctracer_status_t roctracer_disable_op_callback(
    activity_domain_t domain,                             // tracing domain
    uint32_t op);                                         // API call ID
roctracer_status_t roctracer_disable_domain_callback(
    activity_domain_t domain);                            // tracing domain
roctracer_status_t roctracer_disable_callback();

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
static inline roctracer_status_t roctracer_next_record(
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
roctracer_status_t roctracer_open_pool_expl(
    const roctracer_properties_t* properties,             // tracer pool properties
    roctracer_pool_t** pool);                             // [out] returns tracer pool if not NULL,
                                                          // otherwise sets the default one if it is not set yet
static inline roctracer_status_t roctracer_open_pool(
    const roctracer_properties_t* properties)             // tracer pool properties
{
    return roctracer_open_pool_expl(properties, NULL);
}
                                                          // otherwise the error is generated

// Close tracer memory pool
roctracer_status_t roctracer_close_pool_expl(
    roctracer_pool_t* pool);                              // [in] memory pool, NULL is a default one
static inline roctracer_status_t roctracer_close_pool()
{
    return roctracer_close_pool_expl(NULL);
}

// Return current default pool
// Set new default pool if the argument is not NULL
roctracer_pool_t* roctracer_default_pool_expl(
    roctracer_pool_t* pool);                              // [in] new default pool if not NULL
static inline roctracer_pool_t* roctracer_default_pool()
{
    return roctracer_default_pool_expl(NULL);
}

// Enable activity records logging
roctracer_status_t roctracer_enable_op_activity_expl(
    activity_domain_t domain,                             // tracing domain
    uint32_t op,                                          // activity op ID
    roctracer_pool_t* pool);                              // memory pool, NULL is a default one
static inline roctracer_status_t roctracer_enable_op_activity(
    activity_domain_t domain,                             // tracing domain
    uint32_t op)                                          // activity op ID
{
    return roctracer_enable_op_activity_expl(domain, op, NULL);
}
roctracer_status_t roctracer_enable_domain_activity_expl(
    activity_domain_t domain,                             // tracing domain
    roctracer_pool_t* pool);                              // memory pool, NULL is a default one
static inline roctracer_status_t roctracer_enable_domain_activity(
    activity_domain_t domain)                             // tracing domain
{
    return roctracer_enable_domain_activity_expl(domain, NULL);
}
roctracer_status_t roctracer_enable_activity_expl(
    roctracer_pool_t* pool);                       // memory pool, NULL is a default one
static inline roctracer_status_t roctracer_enable_activity()
{
    return roctracer_enable_activity_expl(NULL);
}

// Disable activity records logging
roctracer_status_t roctracer_disable_op_activity(
    activity_domain_t domain,                             // tracing domain
    uint32_t op);                                         // activity op ID
roctracer_status_t roctracer_disable_domain_activity(
    activity_domain_t domain);                            // tracing domain
roctracer_status_t roctracer_disable_activity();

// Flush available activity records
roctracer_status_t roctracer_flush_activity_expl(
    roctracer_pool_t* pool);                              // memory pool, NULL is a default one
static inline roctracer_status_t roctracer_flush_activity()
{
    return roctracer_flush_activity_expl(NULL);
}

// Get system timestamp
roctracer_status_t roctracer_get_timestamp(
    uint64_t* timestamp);                                 // [out] return timestamp

// Load/Unload methods
bool roctracer_load();
void roctracer_unload();
void roctracer_flush_buf();

// Set properties
roctracer_status_t roctracer_set_properties(
    roctracer_domain_t domain,                            // tracing domain
    void* propertes);                                     // tracing properties

#ifdef __cplusplus
}  // extern "C" block
#endif  // __cplusplus

#endif  // INC_ROCTRACER_H_
