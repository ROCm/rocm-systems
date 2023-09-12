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

#include <stddef.h>
#include <stdint.h>

/** @defgroup SYMBOL_VERSIONING_GROUP Symbol Versions
 *
 * The names used for the shared library versioned symbols.
 *
 * Every function is annotated with one of the version macros defined in this
 * section.  Each macro specifies a corresponding symbol version string.  After
 * dynamically loading the shared library with @p dlopen, the address of each
 * function can be obtained using @p dlsym with the name of the function and
 * its corresponding symbol version string.  An error will be reported by @p
 * dlvsym if the installed library does not support the version for the
 * function specified in this version of the interface.
 *
 * @{
 */

/**
 * The function was introduced in version 10.0 of the interface and has the
 * symbol version string of ``"ROCPROFILER_10.0"``.
 */
#define ROCPROFILER_VERSION_10_0

/** @} */

/** @defgroup VERSIONING_GROUP Library Versioning
 *
 * Version information about the interface and the associated installed
 * library.
 *
 * The semantic version of the interface following semver.org rules. A context
 * that uses this interface is only compatible with the installed library if
 * the major version numbers match and the interface minor version number is
 * less than or equal to the installed library minor version number.
 */

#include "rocprofiler/defines.h"
#include "rocprofiler/hip.h"
#include "rocprofiler/hsa.h"
#include "rocprofiler/marker.h"
#include "rocprofiler/version.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/**
 * @fn void rocprofiler_get_version(uint32_t* major, uint32_t* minor, uint32_t* patch)
 * @param [out] major The major version number is stored if non-NULL.
 * @param [out] minor The minor version number is stored if non-NULL.
 * @param [out] patch The patch version number is stored if non-NULL.
 * @addtogroup VERSIONING_GROUP
 *
 * @brief Query the version of the installed library.
 *
 * Return the version of the installed library.  This can be used to check if
 * it is compatible with this interface version.  This function can be used
 * even when the library is not initialized.
 */
void ROCPROFILER_API
rocprofiler_get_version(uint32_t* major, uint32_t* minor, uint32_t* patch)
    ROCPROFILER_NONNULL(1, 2, 3);

/**
 * @defgroup STATUS_CODES Status codes
 * @{
 */

/**
 * @brief Status codes.
 *
 */
typedef enum
{
    ROCPROFILER_STATUS_SUCCESS = 0,
    ROCPROFILER_STATUS_ERROR,
    ROCPROFILER_STATUS_ERROR_BUFFER_NOT_FOUND,
    ROCPROFILER_STATUS_ERROR_FILTER_NOT_FOUND,
    ROCPROFILER_STATUS_ERROR_INCORRECT_DOMAIN,
    ROCPROFILER_STATUS_ERROR_INVALID_DOMAIN_ID,
    ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND,
    ROCPROFILER_STATUS_ERROR_HAS_ACTIVE_CONTEXT,
    ROCPROFILER_STATUS_ERROR_INVALID_OPERATION_ID,
    ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_ACTIVE,
    ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED,
    ROCPROFILER_STATUS_LAST,
} rocprofiler_status_t;

/** @} */

/**
 * @defgroup CONTEXT_OPERATIONS Context
 * @{
 */

/**
 * @brief Context ID.
 *
 */
typedef struct
{
    uint64_t handle;
} rocprofiler_context_id_t;

/**
 * The NULL Context handle.
 */
#define ROCPROFILER_CONTEXT_NONE ROCPROFILER_HANDLE_LITERAL(rocprofiler_context_id_t, 0)

/**
 * @brief Create context.
 *
 * @param context_id [out] Context identifier
 * @return ::rocprofiler_status_t
 */
rocprofiler_status_t ROCPROFILER_API
rocprofiler_create_context(rocprofiler_context_id_t* context_id) ROCPROFILER_NONNULL(1);

/**
 * @brief Start context.
 *
 * @param [in] context_id
 * @return ::rocprofiler_status_t
 */
rocprofiler_status_t ROCPROFILER_API
rocprofiler_start_context(rocprofiler_context_id_t context_id);

/**
 * @brief Stop context.
 *
 * @param [in] context_id
 * @return ::rocprofiler_status_t
 */
rocprofiler_status_t ROCPROFILER_API
rocprofiler_stop_context(rocprofiler_context_id_t context_id);

/** @} */

/**
 * @defgroup RECORDS ROCProfiler Records
 * @{
 */

/** @} */

/**
 * @brief Buffer ID.
 * @addtogroup BUFFER_HANDLING
 */
typedef struct
{
    uint64_t handle;
} rocprofiler_buffer_id_t;

/** @defgroup SERVICE_OPERATIONS Services
 * @{
 */

/**
 * @brief Agent type.
 */
typedef enum
{
    ROCPROFILER_AGENT_TYPE_NONE = 0,  ///< agent is unknown type
    ROCPROFILER_AGENT_TYPE_CPU,       ///< agent is CPU
    ROCPROFILER_AGENT_TYPE_GPU,       ///< agent is GPU
    ROCPROFILER_AGENT_TYPE_LAST,
} rocprofiler_agent_type_t;

/**
 * @brief Agent Identifier
 */
typedef struct
{
    uint64_t handle;
} rocprofiler_agent_id_t;

typedef struct rocprofiler_pc_sampling_configuration_s rocprofiler_pc_sampling_configuration_t;

typedef struct rocprofiler_pc_sampling_config_array_s
{
    rocprofiler_pc_sampling_configuration_t* data;
    size_t                                   size;
} rocprofiler_pc_sampling_config_array_t;

/**
 * @brief Agent.
 */
typedef struct
{
    rocprofiler_agent_id_t                 id;
    rocprofiler_agent_type_t               type;
    const char*                            name;
    rocprofiler_pc_sampling_config_array_t pc_sampling_configs;
} rocprofiler_agent_t;

/**
 * @brief Callback function type for querying the available agents
 *
 * @param [in] agents Array of pointers to agents
 * @param [in] num_agents Number of agents in array
 * @param [in] user_data Data pointer passback
 * @return ::rocprofiler_status_t
 */
typedef rocprofiler_status_t (*rocprofiler_available_agents_cb_t)(rocprofiler_agent_t** agents,
                                                                  size_t                num_agents,
                                                                  void*                 user_data);

/**
 * @brief Receive synchronous callback with an array of available agents at moment of invocation
 *
 * @param [in] callback Callback function accepting list of agents
 * @param [in] agent_size Should be set to sizeof(rocprofiler_agent_t)
 * @param [in] user_data Data pointer provided to callback
 * @return ::rocprofiler_status_t
 */
rocprofiler_status_t ROCPROFILER_API
rocprofiler_query_available_agents(rocprofiler_available_agents_cb_t callback,
                                   size_t                            agent_size,
                                   void* user_data) ROCPROFILER_NONNULL(1);

/**
 * @brief Queue ID.
 */
typedef struct
{
    uint64_t handle;
} rocprofiler_queue_id_t;

/**
 * @brief Thread ID
 */
typedef uint64_t rocprofiler_thread_id_t;

/**
 * @brief ROCProfiler Record Correlation ID.
 * To be reviewed?
 */
typedef struct
{
    uint64_t handle;
} rocprofiler_correlation_id_t;

/**
 * @brief ROCProfiler Timestamp.
 *
 */
typedef uint64_t rocprofiler_timestamp_t;

/**
 * @brief ROCProfiler Address.
 */
typedef uint64_t rocprofiler_address_t;

/** @defgroup TRACING_SERVICES Tracing Services
 * @{
 */

/**
 * @brief Tracing Domain ID.
 *
 * Domains for tracing
 *
 * if the value is equal to zero that means all operations will be considered
 * for tracing.
 *
 */
typedef enum
{
    ROCPROFILER_TRACER_ACTIVITY_DOMAIN_NONE = 0,
    ROCPROFILER_TRACER_ACTIVITY_DOMAIN_HSA_API,
    ROCPROFILER_TRACER_ACTIVITY_DOMAIN_HIP_API,
    ROCPROFILER_TRACER_ACTIVITY_DOMAIN_MARKER_API,
    ROCPROFILER_TRACER_ACTIVITY_DOMAIN_ROCTX = ROCPROFILER_TRACER_ACTIVITY_DOMAIN_MARKER_API,
    ROCPROFILER_TRACER_ACTIVITY_DOMAIN_KFD_API,
    ROCPROFILER_TRACER_ACTIVITY_DOMAIN_EXT_API,
    ROCPROFILER_TRACER_ACTIVITY_DOMAIN_HSA_OPS,
    ROCPROFILER_TRACER_ACTIVITY_DOMAIN_HIP_OPS,
    ROCPROFILER_TRACER_ACTIVITY_DOMAIN_HSA_EVT,
    ROCPROFILER_TRACER_ACTIVITY_DOMAIN_LAST
} rocprofiler_tracer_activity_domain_t;

/**
 * @brief Tracing Operation ID.
 *
 * Depending on the kind, operations can be determined
 *
 * if the value is equal to zero that means all operations will be considered
 * for tracing.
 *
 */
typedef uint32_t rocprofiler_trace_operation_t;

/** @defgroup CALLBACK_TRACING_SERVICE Callback Tracing Service
 * @{
 */

/**
 * @brief Service Callback Tracing Kind.
 */
typedef enum
{
    ROCPROFILER_SERVICE_CALLBACK_TRACING_NONE            = 0,
    ROCPROFILER_SERVICE_CALLBACK_TRACING_HSA_API         = 1,
    ROCPROFILER_SERVICE_CALLBACK_TRACING_HIP_API         = 2,
    ROCPROFILER_SERVICE_CALLBACK_TRACING_MARKER          = 3,
    ROCPROFILER_SERVICE_CALLBACK_TRACING_CODE_OBJECT     = 4,
    ROCPROFILER_SERVICE_CALLBACK_TRACING_KERNEL_DISPATCH = 5,
    ROCPROFILER_SERVICE_CALLBACK_TRACING_HELPER_THREAD   = 6,
    // TODO: Is tracing runtime threads possible?
    // ROCPROFILER_SERVICE_CALLBACK_TRACING_RUNTIME_THREAD = 7,
    ROCPROFILER_SERVICE_CALLBACK_TRACING_LAST,
} rocprofiler_service_callback_tracing_kind_t;

/**
 * @defgroup HSA_API_CALLBACK_TRACING_RECORDS HSA API Callback Tracing Records
 * @{
 */

/**
 * @brief ROCProfiler HSA API Callback Data.
 *
 * Depending on the operation kind, the data can be casted to the corresponding
 * structure.
 *
 */
typedef void* rocprofiler_hsa_api_callback_api_data_t;

/**
 * @brief ROCProfiler HSA API Callback Data.
 */
typedef struct
{
    rocprofiler_correlation_id_t            correlation_id;
    rocprofiler_hsa_api_callback_api_data_t data;  // Arguments or api_data?
} rocprofiler_hsa_api_callback_tracer_data_t;

/**
 * @brief ROCProfiler HIP API Callback Data.
 *
 * Depending on the operation kind, the data can be casted to the corresponding
 * structure.
 *
 */
typedef void* rocprofiler_hip_api_callback_api_data_t;

/**
 * @brief ROCProfiler HIP API Tracer Callback Data.
 */
typedef struct
{
    rocprofiler_correlation_id_t            correlation_id;
    rocprofiler_address_t                   host_kernel_address;
    rocprofiler_hip_api_callback_api_data_t data;  // Arguments or api_data?
} rocprofiler_hip_api_callback_tracer_data_t;

/**
 * @brief ROCProfiler Marker Callback Data.
 *
 * Depending on the operation kind, the data can be casted to the corresponding
 * structure.
 *
 */
typedef void* rocprofiler_marker_callback_api_data_t;

/**
 * @brief ROCProfiler Marker Tracer Callback Data.
 */
typedef struct
{
    rocprofiler_correlation_id_t           correlation_id;
    rocprofiler_marker_callback_api_data_t data;  // Arguments or api_data?
} rocprofiler_marker_callback_tracer_data_t;

/**
 * @brief ROCProfiler Tracing Helper Thread.
 *
 */
typedef enum
{

    ROCPROFILER_TRACING_HELPER_THREAD_START    = 0,
    ROCPROFILER_TRACING_HELPER_THREAD_COMPLETE = 1,
    ROCPROFILER_TRACING_HELPER_THREAD_LAST,
} rocprofiler_tracing_helper_thread_operation_t;

/**
 * @brief ROCProfiler Helper Thread Callback Data.
 *
 */
typedef struct
{
    rocprofiler_tracing_helper_thread_operation_t id;
} rocprofiler_helper_thread_callback_tracer_data_t;

/**
 * @brief ROCProfiler Code Object Tracer Operation.
 */
typedef enum
{
    ROCPROFILER_TRACING_CODE_OBJECT_NONE                            = 0,
    ROCPROFILER_TRACING_CODE_OBJECT_LOAD                            = 1,
    ROCPROFILER_TRACING_CODE_OBJECT_UNLOAD                          = 2,
    ROCPROFILER_TRACING_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER   = 3,
    ROCPROFILER_TRACING_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_UNREGISTER = 4,
    // Should we remove these as they will be part of hipRegisterFunction API
    // tracing? ROCPROFILER_TRACING_CODE_OBJECT_REGISTER_HOST_KERNEL_SYMBOL = 5,
    // (?) ROCPROFILER_TRACING_CODE_OBJECT_UNREGISTER_HOST_KERNEL_SYMBOL = 6, (?)
    ROCPROFILER_TRACING_CODE_OBJECT_LAST,
} rocprofiler_tracing_code_object_operation_t;

/**
 * @brief ROCProfiler Code Object Load Tracer Callback Record.
 */
typedef struct
{
    uint64_t    load_base;  // code object load base
    uint64_t    load_size;  // code object load size
    const char* uri;        // URI string (NULL terminated)
                            // uint32_t storage_type; // code object storage type (Need Review?)
                            // int storage_file;      // origin file descriptor (Need Review?)
                            // uint64_t memory_base;  // origin memory base (Need Review?)
                            // uint64_t memory_size;  // origin memory size (Need Review?)
                            // uint64_t load_delta;   // code object load delta (Need Review?)
} rocprofiler_callback_tracer_code_object_load_data_t;

/**
 * @brief ROCProfiler Code Object UnLoad Tracer Callback Record.
 *
 */
typedef struct
{
    uint64_t load_base;  // code object load base
} rocprofiler_callback_tracer_code_object_unload_data_t;

/**
 * @brief ROCProfiler Code Object Device Kernel Symbol Tracer Callback Record.
 *
 */
typedef struct
{
    const char*           kernel_name;        // kernel name string (NULL terminated)
    rocprofiler_address_t kernel_descriptor;  // kernel descriptor
} rocprofiler_callback_tracer_code_object_device_kernel_symbol_data_t;

/**
 * @brief ROCProfiler Code Object Register Host Kernel Symbol Tracer Callback
 * Record.
 *
 */
typedef struct
{
    rocprofiler_address_t host_address;  // host address
    // Should this be nullptr if it is unregister?
    const char*           kernel_name;        // kernel name string (NULL terminated)
    rocprofiler_address_t kernel_descriptor;  // kernel descriptor
} rocprofiler_callback_tracer_code_object_register_host_kernel_symbol_data_t;

/** @} */

/**
 * @brief API Tracing callback data.
 *
 * This can be casted to:
 *  rocprofiler_hsa_callback_data_t if the record kind is
 * @ref ROCPROFILER_SERVICE_CALLBACK_TRACING_HSA_API
 *  rocprofiler_hip_callback_data_t if the record kind is
 * @ref ROCPROFILER_SERVICE_CALLBACK_TRACING_HIP_API
 *  rocprofiler_marker_callback_data_t if the record kind is
 * @ref ROCPROFILER_SERVICE_CALLBACK_TRACING_MARKER
 *
 */
typedef void* rocprofiler_tracer_callback_data_t;

/**
 * @brief API Tracing callback operation kind.
 *
 * Depending on the ::rocprofiler_service_callback_tracing_kind_t
 * the operation kind can be determined from the following:
 *  rocprofiler_marker_trace_record_operation_t for Markers
 *  rocprofiler_hsa_trace_record_operation_t for HSA API
 *  rocprofiler_hip_trace_record_operation_t for HIP API
 *  rocprofiler_code_object_record_operation_t for Code object tracing
 *
 */
typedef uint32_t rocprofiler_tracer_callback_operation_t;

/**
 * @brief API Tracing callback function.
 */
typedef void (*rocprofiler_tracer_callback_t)(rocprofiler_service_callback_tracing_kind_t kind,
                                              rocprofiler_tracer_callback_operation_t     operation,
                                              rocprofiler_tracer_callback_data_t          data,
                                              void* callback_args);

/**
 * @brief Configure Callback Tracing Service.
 *
 * @param [in] context_id
 * @param [in] kind
 * @param [in] operations
 * @param [in] operations_count
 * @param [in] callback
 * @param [in] callback_args
 * @return ::rocprofiler_status_t
 *
 */
rocprofiler_status_t ROCPROFILER_API
rocprofiler_configure_callback_tracing_service(rocprofiler_context_id_t context_id,
                                               rocprofiler_service_callback_tracing_kind_t kind,
                                               rocprofiler_trace_operation_t* operations,
                                               size_t                         operations_count,
                                               rocprofiler_tracer_callback_t  callback,
                                               void*                          callback_args);

/**
 * @brief Query Callback Trace Kind Name.
 *
 * @param [in] kind
 * @param [out] name if nullptr, size will be returned
 * @param [out] size
 * @return ::rocprofiler_status_t
 */
rocprofiler_status_t ROCPROFILER_API
rocprofiler_query_callback_trace_kind_name(rocprofiler_service_callback_tracing_kind_t kind,
                                           const char*                                 name,
                                           size_t* size) ROCPROFILER_NONNULL(3);

/**
 * @brief General Operation kind
 *
 * That can be used to represent one of the following:
 * - ::rocprofiler_trace_record_hsa_operation_kind_t
 * - ::rocprofiler_trace_record_hip_operation_kind_t
 * - ::rocprofiler_trace_record_marker_operation_kind_t
 *
 */
typedef uint32_t rocprofiler_trace_record_operation_kind_t;

/**
 * @brief Query callback kind operation name.
 *
 * @param [in] kind
 * @param [in] api_trace_operation
 * @param [out] name if nullptr, size will be returned
 * @param [out] size
 * @return ::rocprofiler_status_t
 */
rocprofiler_status_t ROCPROFILER_API
rocprofiler_query_callback_kind_operation_name(
    rocprofiler_service_callback_tracing_kind_t kind,
    rocprofiler_trace_record_operation_kind_t   api_trace_operation,
    const char*                                 name,
    size_t*                                     size) ROCPROFILER_NONNULL(4);

/** @} */

/** @defgroup BUFFER_TRACING_SERVICE Buffer Tracing Service
 * @{
 */

/**
 * @brief Service Buffer Tracing Kind.
 */
typedef enum
{
    ROCPROFILER_SERVICE_BUFFER_TRACING_NONE                 = 0,
    ROCPROFILER_SERVICE_BUFFER_TRACING_HSA_API              = 1,
    ROCPROFILER_SERVICE_BUFFER_TRACING_HIP_API              = 2,
    ROCPROFILER_SERVICE_BUFFER_TRACING_MARKER               = 3,
    ROCPROFILER_SERVICE_BUFFER_TRACING_MEMORY_COPY          = 4,
    ROCPROFILER_SERVICE_BUFFER_TRACING_KERNEL_DISPATCH      = 5,
    ROCPROFILER_SERVICE_BUFFER_TRACING_PAGE_MIGRATION       = 6,
    ROCPROFILER_SERVICE_BUFFER_TRACING_SCRATCH_MEMORY       = 7,
    ROCPROFILER_SERVICE_BUFFER_TRACING_EXTERNAL_CORRELATION = 8,
    // To determine if this is possible to implement?
    // ROCPROFILER_SERVICE_BUFFER_TRACING_QUEUE_SCHEDULING = 9,
    // Do we need to keep it in buffer tracing?
    // ROCPROFILER_SERVICE_BUFFER_TRACING_CODE_OBJECT = 10,
    ROCPROFILER_SERVICE_BUFFER_TRACING_LAST,
} rocprofiler_service_buffer_tracing_kind_t;

/**
 * @brief ROCProfiler Buffer Tracing Record Header.
 */
typedef struct
{
    rocprofiler_service_buffer_tracing_kind_t kind;
    rocprofiler_correlation_id_t              correlation_id;
} rocprofiler_buffer_tracing_record_header_t;

/**
 * @defgroup HSA_API_CALLBACK_TRACING_RECORDS HSA API Callback Tracing Records
 * @{
 */

/**
 * @brief ROCProfiler Buffer HSA API Tracer Record.
 */
typedef struct
{
    rocprofiler_buffer_tracing_record_header_t    header;
    rocprofiler_trace_record_hsa_operation_kind_t operation;  // rocprofiler/hsa.h
    rocprofiler_timestamp_t                       start_timestamp;
    rocprofiler_timestamp_t                       end_timestamp;
    rocprofiler_thread_id_t                       thread_id;
} rocprofiler_buffer_tracing_hsa_api_record_t;

/**
 * @brief ROCProfiler Buffer HIP API Tracer Record.
 */
typedef struct
{
    rocprofiler_buffer_tracing_record_header_t    header;
    rocprofiler_trace_record_hip_operation_kind_t operation;  // rocprofiler/hip.h
    rocprofiler_timestamp_t                       start_timestamp;
    rocprofiler_timestamp_t                       end_timestamp;
    rocprofiler_thread_id_t                       thread_id;
} rocprofiler_buffer_tracing_hip_api_record_t;

/**
 * @brief ROCProfiler Buffer Marker Tracer Record.
 */
typedef struct
{
    rocprofiler_buffer_tracing_record_header_t       header;
    rocprofiler_trace_record_marker_operation_kind_t operation;  // rocprofiler/marker.h
    rocprofiler_timestamp_t                          timestamp;
    rocprofiler_thread_id_t                          thread_id;
    uint64_t                                         marker_id;  // rocprofiler_marker_id_t
    // const char* message; // (Need Review?)
} rocprofiler_buffer_tracing_marker_record_t;

/**
 * @brief Memory Copy Operation.
 */
typedef enum
{
    ROCPROFILER_TRACER_MEMORY_NONE                = 0,
    ROCPROFILER_TRACER_MEMORY_COPY_DEVICE_TO_HOST = 1,
    ROCPROFILER_TRACER_MEMORY_HOST_TO_DEVICE      = 2,
    ROCPROFILER_TRACER_MEMORY_DEVICE_TO_DEVICE    = 3,
    ROCPROFILER_TRACER_MEMORY_LAST,
} rocprofiler_trace_memory_copy_operation_t;

/**
 * @brief ROCProfiler Buffer Memory Copy Tracer Record.
 */
typedef struct
{
    rocprofiler_buffer_tracing_record_header_t header;
    /**
     * Memory copy operation that can be derived from
     * ::rocprofiler_trace_record_operation_kind_t
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
    rocprofiler_buffer_tracing_record_header_t header;
    rocprofiler_timestamp_t                    start_timestamp;
    rocprofiler_timestamp_t                    end_timestamp;
    rocprofiler_queue_id_t                     queue_id;
    const char*                                kernel_name;
} rocprofiler_buffer_tracing_kernel_dispatch_record_t;

/**
 * @brief ROCProfiler Buffer Page Migration Tracer Record.
 */
typedef struct
{
    rocprofiler_buffer_tracing_record_header_t header;
    rocprofiler_timestamp_t                    start_timestamp;
    rocprofiler_timestamp_t                    end_timestamp;
    rocprofiler_queue_id_t                     queue_id;
    // Not Sure What is the info needed here?
} rocprofiler_buffer_tracing_page_migration_record_t;

/**
 * @brief ROCProfiler Buffer Scratch Memory Tracer Record.
 */
typedef struct
{
    rocprofiler_buffer_tracing_record_header_t header;
    rocprofiler_timestamp_t                    start_timestamp;
    rocprofiler_timestamp_t                    end_timestamp;
    rocprofiler_queue_id_t                     queue_id;
    // Not Sure What is the info needed here?
} rocprofiler_buffer_tracing_scratch_memory_record_t;

/**
 * @brief ROCProfiler Buffer Queue Scheduling Tracer Record.
 */
typedef struct
{
    rocprofiler_buffer_tracing_record_header_t header;
    rocprofiler_timestamp_t                    start_timestamp;
    rocprofiler_timestamp_t                    end_timestamp;
    rocprofiler_queue_id_t                     queue_id;
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
 * @brief ROCProfiler External Correlation ID.
 *
 */
typedef struct
{
    uint64_t id;
} rocprofiler_external_correlation_id_t;

/**
 * @brief ROCProfiler Buffer External Correlation Tracer Record.
 */
typedef struct
{
    rocprofiler_buffer_tracing_record_header_t header;
    rocprofiler_external_correlation_id_t      external_correlation_id;
} rocprofiler_buffer_tracing_external_correlation_record_t;

/** @} */

/**
 * @brief ROCProfiler Push External Correlation ID.
 *
 * @param external_correlation_id
 * @return rocprofiler_status_t
 */
rocprofiler_status_t ROCPROFILER_API
rocprofiler_push_external_correlation_id(
    rocprofiler_external_correlation_id_t external_correlation_id);

/**
 * @brief ROCProfiler Push External Correlation ID.
 *
 * @param external_correlation_id
 * @return rocprofiler_status_t
 */
rocprofiler_status_t ROCPROFILER_API
rocprofiler_pop_external_correlation_id(
    rocprofiler_external_correlation_id_t* external_correlation_id);

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
                                             rocprofiler_trace_operation_t*            operations,
                                             size_t                  operations_count,
                                             rocprofiler_buffer_id_t buffer_id);

/**
 * @brief Query Buffer Trace Kind Name.
 *
 * @param [in] kind
 * @param [out] name if nullptr, size will be returned
 * @param [out] size
 * @return ::rocprofiler_status_t
 */
rocprofiler_status_t ROCPROFILER_API
rocprofiler_query_buffer_trace_kind_name(rocprofiler_service_buffer_tracing_kind_t kind,
                                         const char*                               name,
                                         size_t* size) ROCPROFILER_NONNULL(3);

/**
 * @brief Query buffer kind operation name.
 *
 * @param [in] kind
 * @param [in] api_trace_operation_id
 * @param [out] name if nullptr, size will be returned
 * @param [out] size
 * @return ::rocprofiler_status_t
 */
rocprofiler_status_t ROCPROFILER_API
rocprofiler_query_buffer_kind_operation_name(
    rocprofiler_service_buffer_tracing_kind_t kind,
    rocprofiler_trace_record_operation_kind_t api_trace_operation_id,
    const char*                               name,
    size_t*                                   size) ROCPROFILER_NONNULL(4);

/** @} */

/** @} */

/** @defgroup PROFILE_CONFIG Profile Configurations
 * @{
 */

/**
 * @brief Counter ID.
 *
 */
typedef struct
{
    uint64_t handle;
} rocprofiler_counter_id_t;

/**
 * @brief Profile Configurations
 *
 */
typedef struct
{
    uint64_t handle;
} rocprofiler_profile_config_id_t;

/**
 * @brief Create Profile Configuration.
 *
 * @param [in] agent Agent identifier
 * @param [in] counters_list List of GPU counters
 * @param [in] counters_count Size of counters list
 * @param [out] config_id Identifier for GPU counters group
 * @return ::rocprofiler_status_t
 */
rocprofiler_status_t ROCPROFILER_API
rocprofiler_create_profile_config(rocprofiler_agent_t              agent,
                                  rocprofiler_counter_id_t*        counters_list,
                                  size_t                           counters_count,
                                  rocprofiler_profile_config_id_t* config_id)
    ROCPROFILER_NONNULL(4);

/**
 * @brief Destroy Profile Configuration.
 *
 * @param [in] config_id
 * @return ::rocprofiler_status_t
 */
rocprofiler_status_t ROCPROFILER_API
rocprofiler_destroy_profile_config(rocprofiler_profile_config_id_t config_id);

/** @} */

/** @defgroup PROFILE_COUNTING Profile Counting
 * @{
 */

/**
 * @brief Needs non-typedef specification?
 */
typedef uint32_t rocprofiler_counter_instance_id_t;

/**
 * @brief ROCProfiler Profile Counting Counter per instance.
 */
typedef struct
{
    rocprofiler_counter_id_t          counter_id;
    rocprofiler_counter_instance_id_t instance_id;
    double                            counter_value;
} rocprofiler_record_counter_t;

/** @defgroup DISPATCH_PROFILE_COUNTING_SERVICE Dispatch Profile Counting
 * Service
 * @{
 */

/**
 * @brief ROCProfiler Profile Counting Data.
 *
 */
typedef struct
{
    rocprofiler_timestamp_t start_timestamp;
    rocprofiler_timestamp_t end_timestamp;
    /**
     * Counters, including identifiers to get counter information and Counters
     * values
     *
     * Should it be a record per counter?
     */
    rocprofiler_record_counter_t* counters;
    uint64_t                      counters_count;
    rocprofiler_correlation_id_t  correlation_id;
} rocprofiler_dispatch_profile_counting_record_t;

/**
 * @brief Kernel Dispatch Callback
 *
 * @param [out] queue_id
 * @param [out] agent_id
 * @param [out] correlation_id
 * @param [out] dispatch_packet It can be used to get the kernel descriptor and then using
 * code_object tracing, we can get the kernel name. `dispatch_packet->reserved2` is the
 * correlation_id used to correlate the dispatch packet with the corresponding API call.
 * @param [out] callback_data_args
 * @param [in] config
 */
typedef void (*rocprofiler_profile_counting_dispatch_callback_t)(
    rocprofiler_queue_id_t              queue_id,
    rocprofiler_agent_t                 agent_id,
    rocprofiler_correlation_id_t        correlation_id,
    const hsa_kernel_dispatch_packet_t* dispatch_packet,
    void*                               callback_data_args,
    rocprofiler_profile_config_id_t*    config);

/**
 * @brief Configure Dispatch Profile Counting Service.
 *
 * @param [in] context_id
 * @param [in] agent_id
 * @param [in] buffer_id
 * @param [in] callback
 * @param [in] callback_data_args
 * @return ::rocprofiler_status_t
 */
rocprofiler_status_t ROCPROFILER_API
rocprofiler_configure_dispatch_profile_counting_service(
    rocprofiler_context_id_t                         context_id,
    rocprofiler_agent_t                              agent_id,
    rocprofiler_buffer_id_t                          buffer_id,
    rocprofiler_profile_counting_dispatch_callback_t callback,
    void*                                            callback_data_args);

/** @} */

/** @defgroup AGENT_PROFILE_COUNTING_SERVICE Agent Profile Counting Service
 * @{
 */

/**
 * @brief ROCProfiler Agent Profile Counting Data.
 *
 */
typedef struct
{
    /**
     * Counters, including identifiers to get counter information and Counters
     * values
     */
    rocprofiler_record_counter_t* counters;
    uint64_t                      counters_count;
} rocprofiler_agent_profile_counting_data_t;

/**
 * @brief Configure Profile Counting Service for agent.
 *
 * @param [in] buffer_id
 * @param [in] profile_config_id
 * @return ::rocprofiler_status_t
 */
rocprofiler_status_t ROCPROFILER_API
rocprofiler_configure_agent_profile_counting_service(
    rocprofiler_buffer_id_t         buffer_id,
    rocprofiler_profile_config_id_t profile_config_id);

/**
 * @brief Sample Profile Counting Service for agent.
 *
 * @param [out] data // It is always a size of one
 * @return ::rocprofiler_status_t
 */
rocprofiler_status_t ROCPROFILER_API
rocprofiler_sample_agent_profile_counting_service(rocprofiler_agent_profile_counting_data_t* data);

/** @} */

/**
 * @brief Query Counter name.
 *
 * @param [in] counter_id
 * @param [out] name if nullptr, size will be returned
 * @param [out] size
 * @return ::rocprofiler_status_t
 */
rocprofiler_status_t ROCPROFILER_API
rocprofiler_query_counter_name(rocprofiler_counter_id_t counter_id, const char* name, size_t* size)
    ROCPROFILER_NONNULL(3);

/**
 * @brief Query Counter Instances Count.
 *
 * @param [in] counter_id
 * @param [out] instance_count
 * @return rocprofiler_status_t
 */
rocprofiler_status_t ROCPROFILER_API
rocprofiler_query_counter_instance_count(rocprofiler_counter_id_t counter_id,
                                         size_t* instance_count) ROCPROFILER_NONNULL(2);

/**
 * @brief Query Agent Counters Availability.
 *
 * @param [in] agent
 * @param [out] counters_list
 * @param [out] counters_count
 * @return ::rocprofiler_status_t
 */
rocprofiler_status_t ROCPROFILER_API
rocprofiler_query_agent_supported_counters(rocprofiler_agent_t       agent,
                                           rocprofiler_counter_id_t* counters_list,
                                           size_t* counters_count) ROCPROFILER_NONNULL(2, 3);

/** @} */

/** @defgroup PC_SAMPLING_SERVICE PC Sampling Service
 * @{
 */

/**
 * @brief ROCProfiler PC Sampling Record.
 *
 */
typedef struct
{
    uint64_t pc;
    uint64_t dispatch_id;
    uint64_t timestamp;
    uint64_t hardware_id;
    union
    {
        uint8_t arb_value;
    };
    union
    {
        void* data;
    };
} rocprofiler_pc_sampling_record_t;

/**
 * @brief PC Sampling Method.
 *
 */
typedef enum
{
    ROCPROFILER_PC_SAMPLING_METHOD_NONE       = 0,
    ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC = 1,
    ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP  = 2,
    ROCPROFILER_PC_SAMPLING_METHOD_LAST,
} rocprofiler_pc_sampling_method_t;

/**
 * @brief PC Sampling Unit.
 *
 */
typedef enum
{
    ROCPROFILER_PC_SAMPLING_UNIT_NONE         = 0,
    ROCPROFILER_PC_SAMPLING_UNIT_INSTRUCTIONS = 1,
    ROCPROFILER_PC_SAMPLING_UNIT_CYCLES       = 2,
    ROCPROFILER_PC_SAMPLING_UNIT_TIME         = 3,
    ROCPROFILER_PC_SAMPLING_UNIT_LAST,
} rocprofiler_pc_sampling_unit_t;

/**
 * @brief Create PC Sampling Service.
 *
 * @param [in] context_id
 * @param [in] agent
 * @param [in] method
 * @param [in] unit
 * @param [in] interval
 * @param [in] buffer_id
 * @return ::rocprofiler_status_t
 *
 */
rocprofiler_status_t ROCPROFILER_API
rocprofiler_configure_pc_sampling_service(rocprofiler_context_id_t         context_id,
                                          rocprofiler_agent_t              agent,
                                          rocprofiler_pc_sampling_method_t method,
                                          rocprofiler_pc_sampling_unit_t   unit,
                                          uint64_t                         interval,
                                          rocprofiler_buffer_id_t          buffer_id);

struct rocprofiler_pc_sampling_configuration_s
{
    rocprofiler_pc_sampling_method_t method;
    rocprofiler_pc_sampling_unit_t   unit;
    size_t                           min_interval;
    size_t                           max_interval;
    uint64_t                         flags;
};

/**
 * @brief Query PC Sampling Configuration.
 *
 * @param [in] agent
 * @param [out] config
 * @param [out] config_count
 * @return ::rocprofiler_status_t
 */
rocprofiler_status_t ROCPROFILER_API
rocprofiler_query_pc_sampling_agent_configurations(rocprofiler_agent_t                      agent,
                                                   rocprofiler_pc_sampling_configuration_t* config,
                                                   size_t* config_count) ROCPROFILER_NONNULL(2, 3);

/** @} */

/** @defgroup SPM_SERVICE SPM Service
 * @{
 */

/**
 * @brief ROCProfiler SPM Record.
 *
 */
typedef struct
{
    /**
     * Counters, including identifiers to get counter information and Counters
     * values
     */
    rocprofiler_record_counter_t* counters;
    uint64_t                      counters_count;
} rocprofiler_spm_record_t;

/**
 * @brief Configure SPM Service.
 *
 * @param [in] context_id
 * @param [in] buffer_id
 * @param [in] profile_config
 * @param [in] interval
 * @return ::rocprofiler_status_t
 */
rocprofiler_status_t ROCPROFILER_API
rocprofiler_configure_spm_service(rocprofiler_context_id_t        context_id,
                                  rocprofiler_buffer_id_t         buffer_id,
                                  rocprofiler_profile_config_id_t profile_config,
                                  uint64_t                        interval);

/** @} */

/** @} */

/** @defgroup BUFFER_HANDLING Buffer
 * @{
 *
 * Every Buffer is associated with a specific service kind.
 * OR
 * Every Buffer is associated with a specific service ID.
 *
 */

// TODO: We need to add rocprofiler_record_header_t
/**
 * @brief Generic record with a type and a pointer to data
 */
typedef struct
{
    uint64_t kind;
    void*    payload;
} rocprofiler_record_header_t;

typedef rocprofiler_record_header_t rocprofiler_record_tracer_t;

/**
 * @brief  Async callback function.
 *
 * @code{.cpp}
 *  for(size_t i = 0; i < num_headers; ++i)
 *  {
 *      rocprofiler_record_header_t* hdr = headers[i];
 *      if(hdr->kind == ROCPROFILER_RECORD_KIND_PC_SAMPLE)
 *      {
 *          auto* data = static_cast<rocprofiler_pc_sample_t*>(&hdr->payload);
 *          ...
 *      }
 *  }
 * @endcode
 */
typedef void (*rocprofiler_buffer_callback_t)(rocprofiler_context_id_t      context,
                                              rocprofiler_buffer_id_t       buffer_id,
                                              rocprofiler_record_header_t** headers,
                                              size_t                        num_headers,
                                              void*                         data,
                                              uint64_t                      drop_count);

/**
 * @brief Actions when Buffer is full.
 *
 */
typedef enum
{
    ROCPROFILER_BUFFER_POLICY_NONE = 0,
    /**
     * Drop records when buffer is full.
     */
    ROCPROFILER_BUFFER_POLICY_DISCARD = 1,
    /**
     * Block when buffer is full.
     */
    ROCPROFILER_BUFFER_POLICY_LOSSLESS = 2,
    ROCPROFILER_BUFFER_POLICY_LAST,
} rocprofiler_buffer_policy_t;

/**
 * @brief Create buffer.
 *
 * @param [in] context Context identifier associated with buffer
 * @param [in] size Size of the buffer in bytes
 * @param [in] watermark - watermark size, where the callback is called, if set
 * to 0 then the callback will be called on every record
 * @param [in] policy Behavior policy when buffer is full
 * @param [in] callback Callback to invoke when buffer is flushed/full
 * @param [in] callback_data Data to provide in callback function
 * @param [out] buffer_id Identification handle for buffer
 * @return ::rocprofiler_status_t
 */
rocprofiler_status_t ROCPROFILER_API
rocprofiler_create_buffer(rocprofiler_context_id_t      context,
                          size_t                        size,
                          size_t                        watermark,
                          rocprofiler_buffer_policy_t   policy,
                          rocprofiler_buffer_callback_t callback,
                          void*                         callback_data,
                          rocprofiler_buffer_id_t*      buffer_id) ROCPROFILER_NONNULL(5, 7);

/**
 * @brief Destroy buffer.
 *
 * @param [in] buffer_id
 * @return ::rocprofiler_status_t
 *
 * Note: This will destroy the buffer even if it is not empty. The user can
 * call @ref ::rocprofiler_flush_buffer before it to make sure the buffer is empty.
 */
rocprofiler_status_t ROCPROFILER_API
rocprofiler_destroy_buffer(rocprofiler_buffer_id_t buffer_id);

/**
 * @brief Flush buffer.
 *
 * @param [in] buffer_id
 * @return ::rocprofiler_status_t
 */
rocprofiler_status_t ROCPROFILER_API
rocprofiler_flush_buffer(rocprofiler_buffer_id_t buffer_id);

/** @} */

#ifdef __cplusplus
}  // extern "C" block
#endif  // __cplusplus
