// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include <rocprofiler-sdk/hip/api_args.h>
#include <rocprofiler-sdk/kfd/page_migration_args.h>
#include <rocprofiler-sdk/rocdecode/api_args.h>
#include <rocprofiler-sdk/rocdecode/api_id.h>

#include <stdint.h>

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
typedef struct rocprofiler_buffer_tracing_hsa_api_record_t
{
    uint64_t                          size;  ///< size of this struct
    rocprofiler_buffer_tracing_kind_t kind;
    rocprofiler_tracing_operation_t   operation;
    rocprofiler_correlation_id_t      correlation_id;   ///< correlation ids for record
    rocprofiler_timestamp_t           start_timestamp;  ///< start time in nanoseconds
    rocprofiler_timestamp_t           end_timestamp;    ///< end time in nanoseconds
    rocprofiler_thread_id_t           thread_id;        ///< id for thread generating this record

    /// @var kind
    /// @brief ::ROCPROFILER_CALLBACK_TRACING_HSA_CORE_API,
    /// ::ROCPROFILER_CALLBACK_TRACING_HSA_AMD_EXT_API,
    /// ::ROCPROFILER_CALLBACK_TRACING_HSA_IMAGE_EXT_API, or
    /// ::ROCPROFILER_CALLBACK_TRACING_HSA_FINALIZE_EXT_API
    /// @var operation
    /// @brief Specification of the API function, e.g., ::rocprofiler_hsa_core_api_id_t,
    /// ::rocprofiler_hsa_amd_ext_api_id_t, ::rocprofiler_hsa_image_ext_api_id_t, or
    /// ::rocprofiler_hsa_finalize_ext_api_id_t
} rocprofiler_buffer_tracing_hsa_api_record_t;

/**
 * @brief ROCProfiler Buffer HIP API Tracer Record.
 */
typedef struct rocprofiler_buffer_tracing_hip_api_record_t
{
    uint64_t                          size;  ///< size of this struct
    rocprofiler_buffer_tracing_kind_t kind;
    rocprofiler_tracing_operation_t   operation;
    rocprofiler_correlation_id_t      correlation_id;   ///< correlation ids for record
    rocprofiler_timestamp_t           start_timestamp;  ///< start time in nanoseconds
    rocprofiler_timestamp_t           end_timestamp;    ///< end time in nanoseconds
    rocprofiler_thread_id_t           thread_id;        ///< id for thread generating this record

    /// @var kind
    /// @brief ::ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API or
    /// ::ROCPROFILER_CALLBACK_TRACING_HIP_COMPILER_API
    /// @var operation
    /// @brief Specification of the API function, e.g., ::rocprofiler_hip_runtime_api_id_t or
    /// ::rocprofiler_hip_compiler_api_id_t
} rocprofiler_buffer_tracing_hip_api_record_t;

/**
 * @brief ROCProfiler Buffer HIP API Tracer Record.
 */
typedef struct rocprofiler_buffer_tracing_hip_api_ext_record_t
{
    uint64_t                          size;  ///< size of this struct
    rocprofiler_buffer_tracing_kind_t kind;
    rocprofiler_tracing_operation_t   operation;
    rocprofiler_correlation_id_t      correlation_id;   ///< correlation ids for record
    rocprofiler_timestamp_t           start_timestamp;  ///< start time in nanoseconds
    rocprofiler_timestamp_t           end_timestamp;    ///< end time in nanoseconds
    rocprofiler_thread_id_t           thread_id;        ///< id for thread generating this record
    rocprofiler_hip_api_args_t        args;             ///< arguments of function call
    rocprofiler_hip_api_retval_t      retval;           ///< return value of function call

    /// @var kind
    /// @brief ::ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API or
    /// ::ROCPROFILER_CALLBACK_TRACING_HIP_COMPILER_API
    /// @var operation
    /// @brief Specification of the API function, e.g., ::rocprofiler_hip_runtime_api_id_t or
    /// ::rocprofiler_hip_compiler_api_id_t
} rocprofiler_buffer_tracing_hip_api_ext_record_t;

/**
 * @brief Additional trace data for OpenMP target routines
 */

typedef struct rocprofiler_buffer_tracing_ompt_target_t
{
    int32_t     kind;        // ompt_target_t target region kind
    int32_t     device_num;  // ompt device number for the region
    uint64_t    task_id;     // Task ID from the task_data argument to the OMPT callback
    uint64_t    target_id;   // Target identifier from the target_data argument to the callback
    const void* codeptr_ra;  // pointer to the callsite of the target region
} rocprofiler_buffer_tracing_ompt_target_t;

typedef struct rocprofiler_buffer_tracing_ompt_target_data_op_t
{
    uint64_t    host_op_id;      // from the host_op_id argument to the OMPT callback
    int32_t     optype;          // ompt_target_data_op_t kind of operation
    int32_t     src_device_num;  // ompt device number for data source
    int32_t     dst_device_num;  // ompt device number for data destination
    int32_t     reserved;        // for padding
    uint64_t    bytes;           // size in bytes of the operation
    const void* codeptr_ra;      // pointer to the callsite of the target_data_op
} rocprofiler_buffer_tracing_ompt_target_data_op_t;

typedef struct rocprofiler_buffer_tracing_ompt_target_kernel_t
{
    uint64_t host_op_id;           // from the host_op_id argument to the OMPT callback
    int32_t  device_num;           // strangely missing from the OpenMP spec,
    uint32_t requested_num_teams;  // from the compiler
} rocprofiler_buffer_tracing_ompt_target_kernel_t;

/**
 * @brief ROCProfiler Buffer OMPT Tracer Record.
 */
typedef struct rocprofiler_buffer_tracing_ompt_record_t
{
    uint64_t                          size;  ///< size of this struct
    rocprofiler_buffer_tracing_kind_t kind;
    rocprofiler_tracing_operation_t   operation;
    rocprofiler_correlation_id_t      correlation_id;   ///< correlation ids for record
    rocprofiler_timestamp_t           start_timestamp;  ///< start time in nanoseconds
    rocprofiler_timestamp_t           end_timestamp;    ///< end time in nanoseconds
    rocprofiler_thread_id_t           thread_id;        ///< id for thread generating this record
    union
    {
        rocprofiler_buffer_tracing_ompt_target_t         target;
        rocprofiler_buffer_tracing_ompt_target_data_op_t target_data_op;
        rocprofiler_buffer_tracing_ompt_target_kernel_t  target_kernel;
        uint64_t                                         reserved[5];
    };

    /// @var kind
    /// @brief ::ROCPROFILER_BUFFER_TRACING_OMPT
    /// @var operation
    /// @brief Specification of the ::rocprofiler_ompt_operation_t
} rocprofiler_buffer_tracing_ompt_record_t;

/**
 * @brief ROCProfiler Buffer Marker Tracer Record.
 */
typedef struct rocprofiler_buffer_tracing_marker_api_record_t
{
    uint64_t                          size;  ///< size of this struct
    rocprofiler_buffer_tracing_kind_t kind;
    rocprofiler_tracing_operation_t   operation;
    rocprofiler_correlation_id_t      correlation_id;   ///< correlation ids for record
    rocprofiler_timestamp_t           start_timestamp;  ///< start time in nanoseconds
    rocprofiler_timestamp_t           end_timestamp;    ///< end time in nanoseconds
    rocprofiler_thread_id_t           thread_id;        ///< id for thread generating this record

    /// @var kind
    /// @brief ::ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_API,
    /// ::ROCPROFILER_CALLBACK_TRACING_MARKER_CONTROL_API, or
    /// ::ROCPROFILER_CALLBACK_TRACING_MARKER_NAME_API
    /// @var operation
    /// @brief Specification of the API function, e.g., ::rocprofiler_marker_core_api_id_t,
    /// ::rocprofiler_marker_control_api_id_t, or
    /// ::rocprofiler_marker_name_api_id_t
} rocprofiler_buffer_tracing_marker_api_record_t;

/**
 * @brief ROCProfiler Buffer RCCL API Record.
 */
typedef struct rocprofiler_buffer_tracing_rccl_api_record_t
{
    uint64_t                          size;  ///< size of this struct
    rocprofiler_buffer_tracing_kind_t kind;
    rocprofiler_tracing_operation_t   operation;
    rocprofiler_correlation_id_t      correlation_id;   ///< correlation ids for record
    rocprofiler_timestamp_t           start_timestamp;  ///< start time in nanoseconds
    rocprofiler_timestamp_t           end_timestamp;    ///< end time in nanoseconds
    rocprofiler_thread_id_t           thread_id;        ///< id for thread generating this record

    /// @var kind
    /// @brief ::ROCPROFILER_CALLBACK_TRACING_RCCL_API
    /// @var operation
    /// @brief Specification of the API function, e.g., ::rocprofiler_rccl_api_id_t
} rocprofiler_buffer_tracing_rccl_api_record_t;

/**
 * @brief ROCProfiler Buffer rocDecode API Record.
 */
typedef struct rocprofiler_buffer_tracing_rocdecode_api_record_t
{
    uint64_t                          size;  ///< size of this struct
    rocprofiler_buffer_tracing_kind_t kind;
    rocprofiler_rocdecode_api_id_t    operation;
    rocprofiler_correlation_id_t      correlation_id;   ///< correlation ids for record
    rocprofiler_timestamp_t           start_timestamp;  ///< start time in nanoseconds
    rocprofiler_timestamp_t           end_timestamp;    ///< end time in nanoseconds
    rocprofiler_thread_id_t           thread_id;        ///< id for thread generating this record

    /// @var kind
    /// @brief ::ROCPROFILER_CALLBACK_TRACING_ROCDECODE_API
    /// @var operation
    /// @brief Specification of the API function, e.g., ::rocprofiler_rocdecode_api_id_t
} rocprofiler_buffer_tracing_rocdecode_api_record_t;

/**
 * @brief An extended ROCProfiler rocDecode API Tracer Record which includes function arguments.
 * Pointers are not dereferenced.
 */
typedef struct rocprofiler_buffer_tracing_rocdecode_api_ext_record_t
{
    uint64_t                           size;  ///< size of this struct
    rocprofiler_buffer_tracing_kind_t  kind;
    rocprofiler_rocdecode_api_id_t     operation;
    rocprofiler_correlation_id_t       correlation_id;   ///< correlation ids for record
    rocprofiler_timestamp_t            start_timestamp;  ///< start time in nanoseconds
    rocprofiler_timestamp_t            end_timestamp;    ///< end time in nanoseconds
    rocprofiler_thread_id_t            thread_id;        ///< id for thread generating this record
    rocprofiler_rocdecode_api_args_t   args;             ///< arguments of function call
    rocprofiler_rocdecode_api_retval_t retval;           ///< return value of function call

    /// @var kind
    /// @brief ::ROCPROFILER_BUFFER_TRACING_ROCDECODE_API_EXT
    /// @var operation
    /// @brief Specification of the API function (@see
    /// ::rocprofiler_rocdecode_api_id_t)
} rocprofiler_buffer_tracing_rocdecode_api_ext_record_t;

/**
 * @brief ROCProfiler Buffer rocJPEG API Record.
 */
typedef struct rocprofiler_buffer_tracing_rocjpeg_api_record_t
{
    uint64_t                          size;  ///< size of this struct
    rocprofiler_buffer_tracing_kind_t kind;
    rocprofiler_tracing_operation_t   operation;
    rocprofiler_correlation_id_t      correlation_id;   ///< correlation ids for record
    rocprofiler_timestamp_t           start_timestamp;  ///< start time in nanoseconds
    rocprofiler_timestamp_t           end_timestamp;    ///< end time in nanoseconds
    rocprofiler_thread_id_t           thread_id;        ///< id for thread generating this record

    /// @var kind
    /// @brief ::ROCPROFILER_CALLBACK_TRACING_ROCJPEG_API
    /// @var operation
    /// @brief Specification of the API function (@see
    /// ::rocprofiler_rocjpeg_api_id_t)
} rocprofiler_buffer_tracing_rocjpeg_api_record_t;

/**
 * @brief ROCProfiler Buffer Memory Copy Tracer Record.
 */
typedef struct rocprofiler_buffer_tracing_memory_copy_record_t
{
    uint64_t                            size;  ///< size of this struct
    rocprofiler_buffer_tracing_kind_t   kind;
    rocprofiler_memory_copy_operation_t operation;
    rocprofiler_async_correlation_id_t  correlation_id;   ///< correlation ids for record
    rocprofiler_thread_id_t             thread_id;        ///< id for thread that triggered copy
    rocprofiler_timestamp_t             start_timestamp;  ///< start time in nanoseconds
    rocprofiler_timestamp_t             end_timestamp;    ///< end time in nanoseconds
    rocprofiler_agent_id_t              dst_agent_id;     ///< destination agent of copy
    rocprofiler_agent_id_t              src_agent_id;     ///< source agent of copy
    uint64_t                            bytes;            ///< bytes copied
    rocprofiler_address_t               dst_address;      ///< destination address
    rocprofiler_address_t               src_address;      ///< source address

    /// @var kind
    /// @brief ::ROCPROFILER_BUFFER_TRACING_MEMORY_COPY
    /// @var operation
    /// @brief Specification of the memory copy direction (@see
    /// ::rocprofiler_memory_copy_operation_t)
} rocprofiler_buffer_tracing_memory_copy_record_t;

/**
 * @brief ROCProfiler Buffer Memory Allocation Tracer Record.
 */
typedef struct rocprofiler_buffer_tracing_memory_allocation_record_t
{
    uint64_t                                  size;  ///< size of this struct
    rocprofiler_buffer_tracing_kind_t         kind;
    rocprofiler_memory_allocation_operation_t operation;
    rocprofiler_correlation_id_t              correlation_id;  ///< correlation ids for record
    rocprofiler_thread_id_t                   thread_id;  ///< id for thread that triggered copy
    rocprofiler_timestamp_t                   start_timestamp;  ///< start time in nanoseconds
    rocprofiler_timestamp_t                   end_timestamp;    ///< end time in nanoseconds
    rocprofiler_agent_id_t agent_id;         ///< agent information for memory allocation
    rocprofiler_address_t  address;          ///< starting address for memory allocation
    uint64_t               allocation_size;  ///< size for memory allocation

    /// @var kind
    /// @brief ::ROCPROFILER_BUFFER_TRACING_MEMORY_ALLOCATION
    /// @var operation
    /// @brief Specification of the memory allocation function (@see
    /// ::rocprofiler_memory_allocation_operation_t
} rocprofiler_buffer_tracing_memory_allocation_record_t;

/**
 * @brief ROCProfiler Buffer Kernel Dispatch Tracer Record.
 */
typedef struct rocprofiler_buffer_tracing_kernel_dispatch_record_t
{
    uint64_t                                size;  ///< size of this struct
    rocprofiler_buffer_tracing_kind_t       kind;  ///< ::ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH
    rocprofiler_kernel_dispatch_operation_t operation;
    rocprofiler_async_correlation_id_t      correlation_id;  ///< correlation ids for record
    rocprofiler_thread_id_t                 thread_id;       ///< id for thread that launched kernel
    rocprofiler_timestamp_t                 start_timestamp;  ///< start time in nanoseconds
    rocprofiler_timestamp_t                 end_timestamp;    ///< end time in nanoseconds
    rocprofiler_kernel_dispatch_info_t      dispatch_info;    ///< Dispatch info

    /// @var operation
    /// @brief Kernel dispatch buffer records only report the ::ROCPROFILER_KERNEL_DISPATCH_COMPLETE
    /// operation because there are no "real" wrapper around the enqueuing of an individual kernel
    /// dispatch
} rocprofiler_buffer_tracing_kernel_dispatch_record_t;

/**
 * @brief ROCProfiler Buffer Page Migration Tracer Record
 */
typedef struct rocprofiler_buffer_tracing_page_migration_record_t
{
    uint64_t                               size;  ///< size of this struct
    rocprofiler_buffer_tracing_kind_t      kind;  ///< ROCPROFILER_BUFFER_TRACING_PAGE_MIGRATION
    rocprofiler_page_migration_operation_t operation;
    rocprofiler_timestamp_t                timestamp;  ///< start time in nanoseconds
    uint32_t                               pid;
    rocprofiler_page_migration_args_t      args;
} rocprofiler_buffer_tracing_page_migration_record_t;

/**
 * @brief ROCProfiler Buffer Scratch Memory Tracer Record
 */
typedef struct rocprofiler_buffer_tracing_scratch_memory_record_t
{
    uint64_t                               size;  ///< size of this struct
    rocprofiler_buffer_tracing_kind_t      kind;  ///< ::ROCPROFILER_BUFFER_TRACING_SCRATCH_MEMORY
    rocprofiler_scratch_memory_operation_t operation;       ///< specification of the kind
    rocprofiler_correlation_id_t           correlation_id;  ///< correlation ids for record
    rocprofiler_agent_id_t                 agent_id;        ///< agent kernel was dispatched on
    rocprofiler_queue_id_t                 queue_id;        ///< queue kernel was dispatched on
    rocprofiler_thread_id_t                thread_id;  ///< id for thread generating this record
    rocprofiler_timestamp_t                start_timestamp;  ///< start time in nanoseconds
    rocprofiler_timestamp_t                end_timestamp;    ///< end time in nanoseconds
    rocprofiler_scratch_alloc_flag_t       flags;
} rocprofiler_buffer_tracing_scratch_memory_record_t;

/**
 * @brief ROCProfiler Buffer Correlation ID Retirement Tracer Record.
 */
typedef struct rocprofiler_buffer_tracing_correlation_id_retirement_record_t
{
    uint64_t                          size;  ///< size of this struct
    rocprofiler_buffer_tracing_kind_t kind;
    rocprofiler_timestamp_t           timestamp;
    uint64_t                          internal_correlation_id;

    /// @var kind
    /// @brief ::ROCPROFILER_BUFFER_TRACING_CORRELATION_ID_RETIREMENT
    /// @var timestamp
    /// @brief Timestamp (in nanosec) of when rocprofiler detected the correlation ID could be
    /// retired. Due to clock skew between the CPU and GPU, this may at times, *appear* to be before
    /// the kernel or memory copy completed but the reality is that if this ever occurred, the API
    /// would report a FATAL error
    /// @var internal_correlation_id
    /// @brief Only internal correlation ID is provided
} rocprofiler_buffer_tracing_correlation_id_retirement_record_t;

/**
 * @brief ROCProfiler Buffer Runtime Initialization Tracer Record.
 */
typedef struct rocprofiler_buffer_tracing_runtime_initialization_record_t
{
    uint64_t                                       size;  ///< size of this struct
    rocprofiler_buffer_tracing_kind_t              kind;
    rocprofiler_runtime_initialization_operation_t operation;
    rocprofiler_correlation_id_t                   correlation_id;
    rocprofiler_thread_id_t                        thread_id;
    rocprofiler_timestamp_t                        timestamp;
    uint64_t                                       version;
    uint64_t                                       instance;

    /// @var kind
    /// @brief ::ROCPROFILER_BUFFER_TRACING_RUNTIME_INITIALIZATION
    /// @var operation
    /// @brief Indicates which runtime was initialized/loaded
    /// @var correlation_id
    /// @brief Correlation ID for these records are always zero
    /// @var thread_id
    /// @brief ID for thread which loaded this runtime
    /// @var timestamp
    /// @brief Timestamp (in nanosec) of when runtime was initialized/loaded
    /// @var version
    /// @brief The version number of the runtime
    ///
    /// Version number is encoded as: (10000 * MAJOR) + (100 * MINOR) + PATCH
    /// @var instance
    /// @brief Number of times this runtime had been loaded previously
} rocprofiler_buffer_tracing_runtime_initialization_record_t;

/**
 * @brief Callback function for mapping ::rocprofiler_buffer_tracing_kind_t ids to
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
    rocprofiler_tracing_operation_t   operation,
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
rocprofiler_configure_buffer_tracing_service(rocprofiler_context_id_t               context_id,
                                             rocprofiler_buffer_tracing_kind_t      kind,
                                             const rocprofiler_tracing_operation_t* operations,
                                             size_t                  operations_count,
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
                                                     rocprofiler_tracing_operation_t   operation,
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
 * @return ::rocprofiler_status_t
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
 * @return ::rocprofiler_status_t
 */
rocprofiler_status_t
rocprofiler_iterate_buffer_tracing_kind_operations(
    rocprofiler_buffer_tracing_kind_t              kind,
    rocprofiler_buffer_tracing_kind_operation_cb_t callback,
    void*                                          data) ROCPROFILER_API ROCPROFILER_NONNULL(2);

/**
 * @brief Callback function for iterating over the function arguments to a traced function.
 * This function will be invoked for each argument.
 * @see rocprofiler_iterate_buffer_tracing_record_args
 *
 * @param [in] kind domain
 * @param [in] operation associated domain operation
 * @param [in] arg_number the argument number, starting at zero
 * @param [in] arg_value_addr the address of the argument stored by rocprofiler.
 * @param [in] arg_indirection_count the total number of indirection levels for the argument, e.g.
 * int == 0, int* == 1, int** == 2
 * @param [in] arg_type the typeid name of the argument (not demangled)
 * @param [in] arg_name the name of the argument in the prototype (or rocprofiler union)
 * @param [in] arg_value_str conversion of the argument to a string, e.g. operator<< overload
 * @param [in] data user data
 */
typedef int (*rocprofiler_buffer_tracing_operation_args_cb_t)(
    rocprofiler_buffer_tracing_kind_t kind,
    rocprofiler_tracing_operation_t   operation,
    uint32_t                          arg_number,
    const void* const                 arg_value_addr,
    int32_t                           arg_indirection_count,
    const char*                       arg_type,
    const char*                       arg_name,
    const char*                       arg_value_str,
    void*                             data);

/**
 * @brief Iterates over all the arguments for the traced function (when available). This is
 * particularly useful when tools want to annotate traces with the function arguments. See
 * @example samples/api_buffered_tracing/client.cpp for a usage example.
 *
 * In contrast to ::rocprofiler_iterate_callback_tracing_kind_operation_args, this function
 * cannot dereference pointer arguments since there is a high probability that the pointer
 * address references the stack and the buffer tracing record is delivered after the
 * stack variables of the corresponding function have been destroyed.
 *
 * @param [in] record Buffer record
 * @param [in] callback The callback function which will be invoked for each argument
 * @param [in] user_data Data to be passed to each invocation of the callback
 * @return ::rocprofiler_status_t
 */
rocprofiler_status_t
rocprofiler_iterate_buffer_tracing_record_args(
    rocprofiler_record_header_t                    record,
    rocprofiler_buffer_tracing_operation_args_cb_t callback,
    void* user_data) ROCPROFILER_API ROCPROFILER_NONNULL(2);

/** @} */

ROCPROFILER_EXTERN_C_FINI
