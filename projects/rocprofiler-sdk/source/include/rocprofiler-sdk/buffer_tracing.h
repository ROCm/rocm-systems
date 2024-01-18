// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/defines.h>
#include <rocprofiler-sdk/fwd.h>

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
    uint64_t                          size;             ///< size of this struct
    rocprofiler_buffer_tracing_kind_t kind;             ///< ::ROCPROFILER_CALLBACK_TRACING_HSA_API
    rocprofiler_correlation_id_t      correlation_id;   ///< correlation ids for record
    rocprofiler_tracing_operation_t   operation;        ///< ::rocprofiler_hsa_api_id_t
    rocprofiler_timestamp_t           start_timestamp;  ///< start time in nanoseconds
    rocprofiler_timestamp_t           end_timestamp;    ///< end time in nanoseconds
    rocprofiler_thread_id_t           thread_id;        ///< id for thread generating this record
} rocprofiler_buffer_tracing_hsa_api_record_t;

/**
 * @brief ROCProfiler Buffer HIP API Tracer Record.
 */
typedef struct
{
    uint64_t                          size;             ///< size of this struct
    rocprofiler_buffer_tracing_kind_t kind;             ///< ::ROCPROFILER_CALLBACK_TRACING_HIP_API
    rocprofiler_correlation_id_t      correlation_id;   ///< correlation ids for record
    rocprofiler_tracing_operation_t   operation;        ///< ::rocprofiler_hip_api_id_t
    rocprofiler_timestamp_t           start_timestamp;  ///< start time in nanoseconds
    rocprofiler_timestamp_t           end_timestamp;    ///< end time in nanoseconds
    rocprofiler_thread_id_t           thread_id;        ///< id for thread generating this record
} rocprofiler_buffer_tracing_hip_api_record_t;

/**
 * @brief ROCProfiler Buffer Marker Tracer Record.
 */
typedef struct
{
    uint64_t                          size;  ///< size of this struct
    rocprofiler_buffer_tracing_kind_t kind;  ///< ::ROCPROFILER_CALLBACK_TRACING_MARKER_API
    rocprofiler_correlation_id_t      correlation_id;   ///< correlation ids for record
    rocprofiler_tracing_operation_t   operation;        ///< ::rocprofiler_marker_api_id_t
    rocprofiler_timestamp_t           start_timestamp;  ///< start time in nanoseconds
    rocprofiler_timestamp_t           end_timestamp;    ///< end time in nanoseconds
    rocprofiler_thread_id_t           thread_id;        ///< id for thread generating this record
    uint64_t                          marker_id;        ///< rocprofiler_marker_id_t
    // const char* message; // (Need Review?)
} rocprofiler_buffer_tracing_marker_api_record_t;

/**
 * @brief ROCProfiler Buffer Memory Copy Tracer Record.
 */
typedef struct
{
    uint64_t                            size;  ///< size of this struct
    rocprofiler_buffer_tracing_kind_t   kind;  ///< ::ROCPROFILER_BUFFER_TRACING_MEMORY_COPY
    rocprofiler_correlation_id_t        correlation_id;   ///< correlation ids for record
    rocprofiler_memory_copy_operation_t operation;        ///< memory copy direction
    rocprofiler_timestamp_t             start_timestamp;  ///< start time in nanoseconds
    rocprofiler_timestamp_t             end_timestamp;    ///< end time in nanoseconds
    rocprofiler_agent_id_t              dst_agent_id;     ///< destination agent of copy
    rocprofiler_agent_id_t              src_agent_id;     ///< source agent of copy
} rocprofiler_buffer_tracing_memory_copy_record_t;

/**
 * @brief ROCProfiler Buffer Kernel Dispatch Tracer Record.
 */
typedef struct
{
    uint64_t                          size;  ///< size of this struct
    rocprofiler_buffer_tracing_kind_t kind;  ///< ::ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH
    rocprofiler_correlation_id_t      correlation_id;        ///< correlation ids for record
    rocprofiler_timestamp_t           start_timestamp;       ///< start time in nanoseconds
    rocprofiler_timestamp_t           end_timestamp;         ///< end time in nanoseconds
    rocprofiler_agent_id_t            agent_id;              ///< agent kernel was dispatched on
    rocprofiler_queue_id_t            queue_id;              ///< queue kernel was dispatched on
    rocprofiler_kernel_id_t           kernel_id;             ///< identifier for kernel
    uint32_t                          private_segment_size;  /// runtime private memory segment size
    uint32_t                          group_segment_size;    /// runtime group memory segment size
    rocprofiler_dim3_t                workgroup_size;  /// runtime workgroup size (grid * threads)
    rocprofiler_dim3_t                grid_size;       /// runtime grid size
} rocprofiler_buffer_tracing_kernel_dispatch_record_t;

/**
 * @brief ROCProfiler Buffer Page Migration Tracer Record. Not implemented.
 */
typedef struct
{
    uint64_t                          size;  ///< size of this struct
    rocprofiler_buffer_tracing_kind_t kind;  ///< ROCPROFILER_BUFFER_TRACING_PAGE_MIGRATION
    rocprofiler_correlation_id_t      correlation_id;   ///< correlation ids for record
    rocprofiler_timestamp_t           start_timestamp;  ///< start time in nanoseconds
    rocprofiler_timestamp_t           end_timestamp;    ///< end time in nanoseconds
    // Not Sure What is the info needed here?
} rocprofiler_buffer_tracing_page_migration_record_t;

/**
 * @brief ROCProfiler Buffer Scratch Memory Tracer Record. Not implemented.
 */
typedef struct
{
    uint64_t                          size;  ///< size of this struct
    rocprofiler_buffer_tracing_kind_t kind;  ///< ::ROCPROFILER_BUFFER_TRACING_SCRATCH_MEMORY
    rocprofiler_correlation_id_t      correlation_id;   ///< correlation ids for record
    rocprofiler_timestamp_t           start_timestamp;  ///< start time in nanoseconds
    rocprofiler_timestamp_t           end_timestamp;    ///< end time in nanoseconds
    // Not Sure What is the info needed here?
} rocprofiler_buffer_tracing_scratch_memory_record_t;

/**
 * @brief ROCProfiler Buffer Queue Scheduling Tracer Record. Not implemented.
 */
typedef struct
{
    uint64_t                          size;  ///< size of this struct
    rocprofiler_buffer_tracing_kind_t kind;  ///< ::ROCPROFILER_BUFFER_TRACING_SCRATCH_MEMORY
    rocprofiler_correlation_id_t      correlation_id;   ///< correlation ids for record
    rocprofiler_timestamp_t           start_timestamp;  ///< start time in nanoseconds
    rocprofiler_timestamp_t           end_timestamp;    ///< end time in nanoseconds
    // Not Sure What is the info needed here?
} rocprofiler_buffer_tracing_queue_scheduling_record_t;

/**
 * @brief ROCProfiler Buffer External Correlation Tracer Record. Not implemented.
 */
typedef struct
{
    uint64_t                          size;
    rocprofiler_buffer_tracing_kind_t kind;
    rocprofiler_correlation_id_t      correlation_id;
} rocprofiler_buffer_tracing_correlation_record_t;

/**
 * @brief Callback function for mapping @ref rocprofiler_buffer_tracing_kind_t ids to
 * string names. @see rocprofiler_iterate_buffer_trace_kind_names.
 */
typedef int (*rocprofiler_buffer_tracing_kind_cb_t)(rocprofiler_buffer_tracing_kind_t kind,
                                                    void*                             data);

/**
 * @brief Callback function for mapping the operations of a given @ref
 * rocprofiler_buffer_tracing_kind_t to string names. @see
 * rocprofiler_iterate_buffer_trace_kind_operation_names.
 */
typedef int (*rocprofiler_buffer_tracing_kind_operation_cb_t)(
    rocprofiler_buffer_tracing_kind_t kind,
    uint32_t                          operation,
    void*                             data);

/**
 * @brief Configure Buffer Tracing Service.
 *
 * @param [in] context_id Associated context to control activation of service
 * @param [in] kind Buffer tracing category
 * @param [in] operations Array of specific operations (if desired)
 * @param [in] operations_count Number of specific operations (if non-null set of operations)
 * @param [in] buffer_id Buffer to store the records in
 * @return ::rocprofiler_status_t
 * @retval ::ROCPROFILER_STATUS_ERROR_CONFIGURATION_LOCKED ::rocprofiler_configure initialization
 * phase has passed
 * @retval ::ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND context is not valid
 * @retval ::ROCPROFILER_STATUS_ERROR_SERVICE_ALREADY_CONFIGURED Context has already been configured
 * for the ::rocprofiler_buffer_tracing_kind_t kind
 * @retval ::ROCPROFILER_STATUS_ERROR_KIND_NOT_FOUND Invalid ::rocprofiler_buffer_tracing_kind_t
 * @retval ::ROCPROFILER_STATUS_ERROR_OPERATION_NOT_FOUND Invalid operation id for
 * ::rocprofiler_buffer_tracing_kind_t kind was found
 *
 */
rocprofiler_status_t
rocprofiler_configure_buffer_tracing_service(rocprofiler_context_id_t          context_id,
                                             rocprofiler_buffer_tracing_kind_t kind,
                                             rocprofiler_tracing_operation_t*  operations,
                                             size_t                            operations_count,
                                             rocprofiler_buffer_id_t buffer_id) ROCPROFILER_API;

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
rocprofiler_query_buffer_tracing_kind_name(rocprofiler_buffer_tracing_kind_t kind,
                                           const char**                      name,
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
rocprofiler_query_buffer_tracing_kind_operation_name(rocprofiler_buffer_tracing_kind_t kind,
                                                     uint32_t                          operation,
                                                     const char**                      name,
                                                     uint64_t* name_len) ROCPROFILER_API;

/**
 * @brief Iterate over all the buffer tracing kinds and invokes the callback for each buffer tracing
 * kind.
 *
 * This is typically used to invoke ::rocprofiler_iterate_buffer_tracing_kind_operations for each
 * buffer tracing kind.
 *
 * @param [in] callback Callback function invoked for each enumeration value in @ref
 * rocprofiler_buffer_tracing_kind_t with the exception of the `NONE` and `LAST` values.
 * @param [in] data User data passed back into the callback
 */
rocprofiler_status_t
rocprofiler_iterate_buffer_tracing_kinds(rocprofiler_buffer_tracing_kind_cb_t callback,
                                         void* data) ROCPROFILER_API ROCPROFILER_NONNULL(1);

/**
 * @brief Iterates over all the operations for a given @ref
 * rocprofiler_buffer_tracing_kind_t and invokes the callback with the kind and operation
 * id. This is useful to build a map of the operation names during tool initialization instead of
 * querying rocprofiler everytime in the callback hotpath.
 *
 * @param [in] kind which buffer tracing kind operations to iterate over
 * @param [in] callback Callback function invoked for each operation associated with @ref
 * rocprofiler_buffer_tracing_kind_t with the exception of the `NONE` and `LAST` values.
 * @param [in] data User data passed back into the callback
 */
rocprofiler_status_t
rocprofiler_iterate_buffer_tracing_kind_operations(
    rocprofiler_buffer_tracing_kind_t              kind,
    rocprofiler_buffer_tracing_kind_operation_cb_t callback,
    void*                                          data) ROCPROFILER_API ROCPROFILER_NONNULL(2);

/** @} */

ROCPROFILER_EXTERN_C_FINI
