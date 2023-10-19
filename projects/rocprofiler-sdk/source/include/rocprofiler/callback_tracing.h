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

#include <rocprofiler/defines.h>
#include <rocprofiler/fwd.h>
#include <rocprofiler/hsa.h>

ROCPROFILER_EXTERN_C_INIT

/**
 * @defgroup CALLBACK_TRACING_SERVICE Synchronous Tracing Services
 * @brief Receive immediate callbacks on the calling thread
 *
 * @{
 */

/**
 * @brief ROCProfiler HSA API Callback Data.
 */
typedef struct
{
    size_t                       size;  ///< provides the size of this struct
    rocprofiler_hsa_api_args_t   args;
    rocprofiler_hsa_api_retval_t retval;
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
    size_t                                  size;
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
    size_t                                 size;
    rocprofiler_correlation_id_t           correlation_id;
    rocprofiler_marker_callback_api_data_t data;  // Arguments or api_data?
} rocprofiler_marker_callback_tracer_data_t;

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

/**
 * @brief API Tracing callback function. This function is invoked twice per API function: once
 * before the function is invoked and once after the function is invoked.  The external correlation
 * id value within the record is assigned the value at the top of the external correlation id stack.
 * It is permissible to invoke @ref rocprofiler_push_external_correlation_id within the enter phase;
 * when a new external correlation id is pushed during the enter phase, rocprofiler will use that
 * external correlation id for any async events and provide the new external correlation id during
 * the exit callback... In other words, pushing a new external correlation id within the enter
 * callback will result in that external correlation id value in the exit callback (which may or may
 * not be different from the external correlation id value in the enter callback). If a tool pushes
 * new external correlation ids in the enter phase, it is recommended to pop the external
 * correlation id in the exit callback.
 *
 * @param record [in] Callback record data
 * @param user_data [in,out] This paramter can be used to retain information in between the enter
 * and exit phases.
 * @param callback_data [in] User data provided when configuring the callback tracing service
 */
typedef void (*rocprofiler_callback_tracing_cb_t)(rocprofiler_callback_tracing_record_t record,
                                                  rocprofiler_user_data_t*              user_data,
                                                  void* callback_data) ROCPROFILER_NONNULL(2);

/**
 * @brief Callback function for mapping @ref rocprofiler_service_callback_tracing_kind_t ids to
 * string names. @see rocprofiler_iterate_callback_tracing_kind_names.
 */
typedef int (*rocprofiler_callback_tracing_kind_cb_t)(
    rocprofiler_service_callback_tracing_kind_t kind,
    void*                                       data);

/**
 * @brief Callback function for mapping the operations of a given @ref
 * rocprofiler_service_callback_tracing_kind_t to string names. @see
 * rocprofiler_iterate_callback_tracing_kind_operation_names.
 */
typedef int (*rocprofiler_callback_tracing_kind_operation_cb_t)(
    rocprofiler_service_callback_tracing_kind_t kind,
    uint32_t                                    operation,
    void*                                       data);

/**
 * @brief Callback function for iterating over the function arguments to a traced function.
 * This function will be invoked for each argument.
 * @see rocprofiler_iterate_callback_tracing_operation_args
 *
 * @param kind [in] domain
 * @param operation [in] associated domain operation
 * @param arg_number [in] the argument number, starting at zero
 * @param arg_name [in] the name of the argument in the prototype (or rocprofiler union)
 * @param arg_value_str [in] conversion of the argument to a string, e.g. operator<< overload
 * @param arg_value_addr [in] the address of the argument stored by rocprofiler.
 * @param data [in] user data
 */
typedef int (*rocprofiler_callback_tracing_operation_args_cb_t)(
    rocprofiler_service_callback_tracing_kind_t kind,
    uint32_t                                    operation,
    uint32_t                                    arg_number,
    const char*                                 arg_name,
    const char*                                 arg_value_str,
    const void* const                           arg_value_addr,
    void*                                       data);

/**
 * @brief Configure Callback Tracing Service. The callback tracing service provides two synchronous
 * callbacks around an API function on the same thread as the application which is invoking the API
 * function. This function can only be invoked once per @ref
 * rocprofiler_service_callback_tracing_kind_t value, i.e. it can be invoked once for the HSA API,
 * once for the HIP API, and so on but it will fail if it is invoked for the HSA API twice. Please
 * note, the callback API does have the potentially non-trivial overhead of copying the function
 * arguments into the record. If you are willing to let rocprofiler record the timestamps, do not
 * require synchronous notifications of the API calls, and want to lowest possible overhead, use the
 * @see BUFFER_TRACING_SERVICE.
 *
 * @param [in] context_id Context to associate the service with
 * @param [in] kind The domain of the callback tracing service
 * @param [in] operations Array of operations in the domain (i.e. enum values which identify
 * specific API functions). If this is null, all API functions in the domain will be traced
 * @param [in] operations_count If the operations array is non-null, set this to the size of the
 * array.
 * @param [in] callback The function to invoke before and after an API function
 * @param [in] callback_args Data provided to every invocation of the callback function
 * @return ::rocprofiler_status_t Will return @ref ROCPROFILER_STATUS_ERROR_CONFIGURATION_LOCKED if
 * invoked outside of the initialization function in @ref rocprofiler_tool_configure_result_t
 * provided to rocprofiler via @ref rocprofiler_configure function. Will return @ref
 * ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND if the provided context is not valid/registered. Will
 * return @ref ROCPROFILER_STATUS_ERROR_SERVICE_ALREADY_CONFIGURED if the same @ref
 * rocprofiler_service_callback_tracing_kind_t value is provided more than once (per context) -- in
 * other words, we do not support overriding or combining the operations in separate function calls.
 *
 */
rocprofiler_status_t ROCPROFILER_API
rocprofiler_configure_callback_tracing_service(rocprofiler_context_id_t context_id,
                                               rocprofiler_service_callback_tracing_kind_t kind,
                                               rocprofiler_tracing_operation_t*  operations,
                                               size_t                            operations_count,
                                               rocprofiler_callback_tracing_cb_t callback,
                                               void*                             callback_args);

/**
 * @brief Query the name of the callback tracing kind. The name retrieved from this function is a
 * string literal that is encoded in the read-only section of the binary (i.e. it is always
 * "allocated" and never "deallocated").
 *
 * @param kind [in] Callback tracing domain
 * @param name [out] If non-null and the name is a constant string that does not require dynamic
 * allocation, this paramter will be set to the address of the string literal, otherwise it will
 * be set to nullptr
 * @param name_len [out] If non-null, this will be assigned the length of the name (regardless of
 * the name is a constant string or requires dynamic allocation)
 * @return rocprofiler_status_t
 */
rocprofiler_status_t
rocprofiler_query_callback_tracing_kind_name(rocprofiler_service_callback_tracing_kind_t kind,
                                             const char**                                name,
                                             uint64_t* name_len) ROCPROFILER_API;

/**
 * @brief Query the name of the callback tracing kind. The name retrieved from this function is a
 * string literal that is encoded in the read-only section of the binary (i.e. it is always
 * "allocated" and never "deallocated").
 *
 * @param kind [in] Callback tracing domain
 * @param operation [in] Enumeration id value which maps to a specific API function or event type
 * @param name [out] If non-null and the name is a constant string that does not require dynamic
 * allocation, this paramter will be set to the address of the string literal, otherwise it will
 * be set to nullptr
 * @param name_len [out] If non-null, this will be assigned the length of the name (regardless of
 * the name is a constant string or requires dynamic allocation)
 * @return rocprofiler_status_t Returns @ref ROCPROFILER_STATUS_ERROR_KIND_NOT_FOUND if the
 * domain id is not valid. Returns @ref ROCPROFILER_STATUS_SUCCESS for a valid domain regardless if
 * there is a constant string or not.
 */
rocprofiler_status_t
rocprofiler_query_callback_tracing_kind_operation_name(
    rocprofiler_service_callback_tracing_kind_t kind,
    uint32_t                                    operation,
    const char**                                name,
    uint64_t*                                   name_len) ROCPROFILER_API;

/**
 * @brief Iterate over all the mappings of the callback tracing kinds and get a callback with the id
 * mapped to a constant string. The strings provided in the arg will be valid pointers for the
 * entire duration of the program. It is recommended to call this function once and cache this data
 * in the client instead of making multiple on-demand calls.
 *
 * @param [in] callback Callback function invoked for each enumeration value in @ref
 * rocprofiler_service_callback_tracing_kind_t with the exception of the `NONE` and `LAST` values.
 * @param [in] data User data passed back into the callback
 * @return rocprofiler_status_t Returns @ref ROCPROFILER_STATUS_ERROR_KIND_NOT_FOUND on an invalid
 * domain id. Returns @ref ROCPROFILER_STATUS_ERROR_OPERATION_NOT_FOUND if the operation number is
 * not recognized for the given domain. Returns @ref ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED if
 * rocprofiler does not support providing the operation name within this domain. Returns @ref
 * ROCPROFILER_STATUS_SUCCESS for valid domain and operation regardless of whether there is a
 * constant string or not.
 */
rocprofiler_status_t ROCPROFILER_API
rocprofiler_iterate_callback_tracing_kinds(rocprofiler_callback_tracing_kind_cb_t callback,
                                           void* data) ROCPROFILER_NONNULL(1);

/**
 * @brief Iterates over all the mappings of the operations for a given @ref
 * rocprofiler_service_callback_tracing_kind_t and invokes the callback with the kind, operation id,
 * and the string mapping to the operation id. The strings provided in the callback arg will be
 * valid pointers for the entire duration of the program. It is recommended to call this function
 * once per kind, and cache this data in the client instead of making multiple on-demand calls.
 *
 * @param [in] kind which tracing callback kind operations to iterate over
 * @param [in] callback Callback function invoked for each operation associated with @ref
 * rocprofiler_service_callback_tracing_kind_t with the exception of the `NONE` and `LAST` values.
 * @param [in] data User data passed back into the callback
 */
rocprofiler_status_t ROCPROFILER_API
rocprofiler_iterate_callback_tracing_kind_operations(
    rocprofiler_service_callback_tracing_kind_t      kind,
    rocprofiler_callback_tracing_kind_operation_cb_t callback,
    void*                                            data) ROCPROFILER_NONNULL(2);

/**
 * @brief Iterates over all the arguments for the traced function (when available). This is
 * particularly useful when tools want to annotate traces with the function arguments. See
 * @example samples/api_callback_tracing/client.cpp for a usage example.
 *
 * @param[in] record Record provided by service callback
 * @param[in] callback The callback function which will be invoked for each argument
 * @param[in] user_data Data to be passed to each invocation of the callback
 */
rocprofiler_status_t ROCPROFILER_API
rocprofiler_iterate_callback_tracing_kind_operation_args(
    rocprofiler_callback_tracing_record_t            record,
    rocprofiler_callback_tracing_operation_args_cb_t callback,
    void*                                            user_data) ROCPROFILER_NONNULL(2);

/** @} */

ROCPROFILER_EXTERN_C_FINI
