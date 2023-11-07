// MIT License
//
// Copyright (c) 2023 ROCm Developer Tools
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <rocprofiler/agent.h>
#include <rocprofiler/defines.h>
#include <rocprofiler/fwd.h>

ROCPROFILER_EXTERN_C_INIT

/**
 * @defgroup BUFFER_TRACING_SERVICE Asynchronous Tracing Service
 * @brief Receive callbacks for batches of records from an internal (background) thread
 *
 * @{
 */

/**
 * @brief ROCProfiler Buffer HSA API Tracer Record.
 */
typedef struct
{
    uint64_t                                  size;
    rocprofiler_service_buffer_tracing_kind_t kind;
    rocprofiler_correlation_id_t              correlation_id;
    rocprofiler_tracing_operation_t           operation;  // rocprofiler/hsa.h
    rocprofiler_timestamp_t                   start_timestamp;
    rocprofiler_timestamp_t                   end_timestamp;
    rocprofiler_thread_id_t                   thread_id;
} rocprofiler_buffer_tracing_hsa_api_record_t;

/**
 * @brief ROCProfiler Buffer HIP API Tracer Record.
 */
typedef struct
{
    uint64_t                                  size;
    rocprofiler_service_buffer_tracing_kind_t kind;
    rocprofiler_correlation_id_t              correlation_id;
    rocprofiler_tracing_operation_t           operation;  // rocprofiler/hip.h
    rocprofiler_timestamp_t                   start_timestamp;
    rocprofiler_timestamp_t                   end_timestamp;
    rocprofiler_thread_id_t                   thread_id;
} rocprofiler_buffer_tracing_hip_api_record_t;

/**
 * @brief ROCProfiler Buffer Marker Tracer Record.
 */
typedef struct
{
    uint64_t                                  size;
    rocprofiler_service_buffer_tracing_kind_t kind;
    rocprofiler_correlation_id_t              correlation_id;
    rocprofiler_tracing_operation_t           operation;  // rocprofiler/marker.h
    rocprofiler_timestamp_t                   timestamp;
    rocprofiler_thread_id_t                   thread_id;
    uint64_t                                  marker_id;  // rocprofiler_marker_id_t
    // const char* message; // (Need Review?)
} rocprofiler_buffer_tracing_marker_record_t;

/**
 * @brief ROCProfiler Buffer Memory Copy Tracer Record.
 */
typedef struct
{
    uint64_t                                  size;
    rocprofiler_service_buffer_tracing_kind_t kind;
    rocprofiler_correlation_id_t              correlation_id;
    /**
     * Memory copy operation that can be derived from
     * ::rocprofiler_tracing_operation_t
     */
    uint32_t                operation;
    rocprofiler_timestamp_t start_timestamp;
    rocprofiler_timestamp_t end_timestamp;
    rocprofiler_queue_id_t  queue_id;
} rocprofiler_buffer_tracing_memory_copy_record_t;

/**
 * @brief ROCProfiler Buffer Kernel Dispatch Tracer Record.
 */
typedef struct
{
    uint64_t                                  size;
    rocprofiler_service_buffer_tracing_kind_t kind;
    rocprofiler_correlation_id_t              correlation_id;
    rocprofiler_timestamp_t                   start_timestamp;
    rocprofiler_timestamp_t                   end_timestamp;
    rocprofiler_agent_id_t                    agent_id;
    rocprofiler_queue_id_t                    queue_id;
    rocprofiler_kernel_id_t                   kernel_id;
    uint32_t                                  private_segment_size;
    uint32_t                                  group_segment_size;
    rocprofiler_dim3_t                        workgroup_size;
    rocprofiler_dim3_t                        grid_size;
} rocprofiler_buffer_tracing_kernel_dispatch_record_t;

/**
 * @brief ROCProfiler Buffer Page Migration Tracer Record.
 */
typedef struct
{
    uint64_t                                  size;
    rocprofiler_service_buffer_tracing_kind_t kind;
    rocprofiler_correlation_id_t              correlation_id;
    rocprofiler_timestamp_t                   start_timestamp;
    rocprofiler_timestamp_t                   end_timestamp;
    rocprofiler_queue_id_t                    queue_id;
    // Not Sure What is the info needed here?
} rocprofiler_buffer_tracing_page_migration_record_t;

/**
 * @brief ROCProfiler Buffer Scratch Memory Tracer Record.
 */
typedef struct
{
    uint64_t                                  size;
    rocprofiler_service_buffer_tracing_kind_t kind;
    rocprofiler_correlation_id_t              correlation_id;
    rocprofiler_timestamp_t                   start_timestamp;
    rocprofiler_timestamp_t                   end_timestamp;
    rocprofiler_queue_id_t                    queue_id;
    // Not Sure What is the info needed here?
} rocprofiler_buffer_tracing_scratch_memory_record_t;

/**
 * @brief ROCProfiler Buffer Queue Scheduling Tracer Record.
 */
typedef struct
{
    uint64_t                                  size;
    rocprofiler_service_buffer_tracing_kind_t kind;
    rocprofiler_correlation_id_t              correlation_id;
    rocprofiler_timestamp_t                   start_timestamp;
    rocprofiler_timestamp_t                   end_timestamp;
    rocprofiler_queue_id_t                    queue_id;
    // Not Sure What is the info needed here?
} rocprofiler_buffer_tracing_queue_scheduling_record_t;

/**
 * @brief ROCProfiler Code Object Tracer Buffer Record.
 *
 * We need to guarantee that these records are in the buffer before the
 * corresponding Exit Phase API calls are called.
 */
// typedef struct {
//   rocprofiler_buffer_tracing_record_header_t header;
//   rocprofiler_tracing_code_object_kind_id_t kind;
// } rocprofiler_buffer_tracing_code_object_header_t;

/**
 * @brief ROCProfiler Code Object Load Tracer Buffer Record.
 *
 */
// typedef struct {
//   rocprofiler_buffer_tracing_code_object_header_t header;
//   uint64_t load_base; // code object load base
//   uint64_t load_size; // code object load size
//   const char *uri;    // URI string (NULL terminated)
//   rocprofiler_timestamp_t timestamp;
//   // uint32_t storage_type; // code object storage type (Need Review?)
//   // int storage_file;      // origin file descriptor (Need Review?)
//   // uint64_t memory_base;  // origin memory base (Need Review?)
//   // uint64_t memory_size;  // origin memory size (Need Review?)
//   // uint64_t load_delta;   // code object load delta (Need Review?)
// } rocprofiler_buffer_tracing_code_object_load_record_t;

/**
 * @brief ROCProfiler Code Object UnLoad Tracer Buffer Record.
 *
 */
// typedef struct {
//   rocprofiler_buffer_tracing_code_object_header_t header;
//   uint64_t load_base; // code object load base
//   rocprofiler_timestamp_t timestamp;
// } rocprofiler_buffer_tracing_code_object_unload_record_t;

/**
 * @brief ROCProfiler Code Object Kernel Symbol Tracer Buffer Record.
 *
 */
// typedef struct {
//   rocprofiler_buffer_tracing_code_object_header_t header;
//   const char *kernel_name;    // kernel name string (NULL terminated)
//   uint64_t kernel_descriptor; // kernel descriptor (Need to be changed from
//                               // uint64_t to ::rocprofiler_address_t)
//   // rocprofiler_timestamp_t timestamp; // (Need Review?)
// } rocprofiler_buffer_tracing_code_object_kernel_symbol_record_t;

/**
 * @brief ROCProfiler Buffer External Correlation Tracer Record.
 */
typedef struct
{
    uint64_t                                  size;
    rocprofiler_service_buffer_tracing_kind_t kind;
    rocprofiler_correlation_id_t              correlation_id;
} rocprofiler_buffer_tracing_correlation_record_t;

/**
 * @brief Callback function for mapping @ref rocprofiler_service_buffer_tracing_kind_t ids to
 * string names. @see rocprofiler_iterate_buffer_trace_kind_names.
 */
typedef int (*rocprofiler_buffer_tracing_kind_cb_t)(rocprofiler_service_buffer_tracing_kind_t kind,
                                                    void*                                     data);

/**
 * @brief Callback function for mapping the operations of a given @ref
 * rocprofiler_service_buffer_tracing_kind_t to string names. @see
 * rocprofiler_iterate_buffer_trace_kind_operation_names.
 */
typedef int (*rocprofiler_buffer_tracing_kind_operation_cb_t)(
    rocprofiler_service_buffer_tracing_kind_t kind,
    uint32_t                                  operation,
    void*                                     data);

/**
 * @brief Configure Buffer Tracing Service.
 *
 * @param [in] context_id
 * @param [in] kind
 * @param [in] operations
 * @param [in] operations_count
 * @param [in] buffer_id
 * @return ::rocprofiler_status_t
 *
 */
rocprofiler_status_t ROCPROFILER_API
rocprofiler_configure_buffer_tracing_service(rocprofiler_context_id_t                  context_id,
                                             rocprofiler_service_buffer_tracing_kind_t kind,
                                             rocprofiler_tracing_operation_t*          operations,
                                             size_t                  operations_count,
                                             rocprofiler_buffer_id_t buffer_id);

/**
 * @brief Query the name of the buffer tracing kind. The name retrieved from this function is a
 * string literal that is encoded in the read-only section of the binary (i.e. it is always
 * "allocated" and never "deallocated").
 *
 * @param [in] kind Buffer tracing domain
 * @param [out] name If non-null and the name is a constant string that does not require dynamic
 * allocation, this paramter will be set to the address of the string literal, otherwise it will
 * be set to nullptr
 * @param [out] name_len If non-null, this will be assigned the length of the name (regardless of
 * the name is a constant string or requires dynamic allocation)
 * @return ::rocprofiler_status_t
 * @retval ::ROCPROFILER_STATUS_ERROR_KIND_NOT_FOUND Returned if the domain id is not valid
 * @retval ::ROCPROFILER_STATUS_SUCCESS Returned if a valid domain, regardless if there is a
 * constant string or not.
 */
rocprofiler_status_t
rocprofiler_query_buffer_tracing_kind_name(rocprofiler_service_buffer_tracing_kind_t kind,
                                           const char**                              name,
                                           uint64_t* name_len) ROCPROFILER_API;

/**
 * @brief Query the name of the buffer tracing kind. The name retrieved from this function is a
 * string literal that is encoded in the read-only section of the binary (i.e. it is always
 * "allocated" and never "deallocated").
 *
 * @param [in] kind Buffer tracing domain
 * @param [in] operation Enumeration id value which maps to a specific API function or event type
 * @param [out] name If non-null and the name is a constant string that does not require dynamic
 * allocation, this paramter will be set to the address of the string literal, otherwise it will
 * be set to nullptr
 * @param [out] name_len If non-null, this will be assigned the length of the name (regardless of
 * the name is a constant string or requires dynamic allocation)
 * @return ::rocprofiler_status_t
 * @retval ::ROCPROFILER_STATUS_ERROR_KIND_NOT_FOUND An invalid domain id
 * @retval ::ROCPROFILER_STATUS_ERROR_OPERATION_NOT_FOUND The operation number is not recognized for
 * the given domain
 * @retval ::ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED Rocprofiler does not support providing the
 * operation name within this domain
 * @retval ::ROCPROFILER_STATUS_SUCCESS Valid domain and operation, regardless of whether there is a
 * constant string or not.
 */
rocprofiler_status_t
rocprofiler_query_buffer_tracing_kind_operation_name(rocprofiler_service_buffer_tracing_kind_t kind,
                                                     uint32_t     operation,
                                                     const char** name,
                                                     uint64_t*    name_len) ROCPROFILER_API;

/**
 * @brief Iterate over all the mappings of the buffer tracing kinds and get a buffer with the id
 * mapped to a constant string. The strings provided in the arg will be valid pointers for the
 * entire duration of the program. It is recommended to call this function once and cache this data
 * in the client instead of making multiple on-demand calls.
 *
 * @param [in] callback Callback function invoked for each enumeration value in @ref
 * rocprofiler_service_buffer_tracing_kind_t with the exception of the `NONE` and `LAST` values.
 * @param [in] data User data passed back into the callback
 */
rocprofiler_status_t ROCPROFILER_API
rocprofiler_iterate_buffer_tracing_kinds(rocprofiler_buffer_tracing_kind_cb_t callback, void* data)
    ROCPROFILER_NONNULL(1);

/**
 * @brief Iterates over all the operations for a given @ref
 * rocprofiler_service_buffer_tracing_kind_t and invokes the callback with the kind and operation
 * id. This is useful to build a map of the operation names during tool initialization instead of
 * querying rocprofiler everytime in the callback hotpath.
 *
 * @param [in] kind which buffer tracing kind operations to iterate over
 * @param [in] callback Callback function invoked for each operation associated with @ref
 * rocprofiler_service_buffer_tracing_kind_t with the exception of the `NONE` and `LAST` values.
 * @param [in] data User data passed back into the callback
 */
rocprofiler_status_t ROCPROFILER_API
rocprofiler_iterate_buffer_tracing_kind_operations(
    rocprofiler_service_buffer_tracing_kind_t      kind,
    rocprofiler_buffer_tracing_kind_operation_cb_t callback,
    void*                                          data) ROCPROFILER_NONNULL(2);

/** @} */

ROCPROFILER_EXTERN_C_FINI
