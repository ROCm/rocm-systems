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

/** @defgroup CALLBACK_TRACING_SERVICE Synchronous Tracing Services
 *
 *  Receive immediate callbacks on the calling thread
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
 * @brief API Tracing callback function.
 */
typedef void (*rocprofiler_callback_tracing_cb_t)(rocprofiler_callback_tracing_record_t record,
                                                  void*                                 user_data);

/**
 * @brief Callback function for mapping @ref rocprofiler_service_callback_tracing_kind_t ids to
 * string names. @see rocprofiler_iterate_callback_tracing_kind_names.
 */
typedef int (*rocprofiler_callback_tracing_kind_name_cb_t)(
    rocprofiler_service_callback_tracing_kind_t kind,
    const char*                                 kind_name,
    void*                                       data);

/**
 * @brief Callback function for mapping the operations of a given @ref
 * rocprofiler_service_callback_tracing_kind_t to string names. @see
 * rocprofiler_iterate_callback_tracing_kind_operation_names.
 */
typedef int (*rocprofiler_callback_tracing_operation_name_cb_t)(
    rocprofiler_service_callback_tracing_kind_t kind,
    uint32_t                                    operation,
    const char*                                 operation_name,
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
                                               rocprofiler_tracing_operation_t*  operations,
                                               size_t                            operations_count,
                                               rocprofiler_callback_tracing_cb_t callback,
                                               void*                             callback_args);

/**
 * @brief Iterate over all the mappings of the callback tracing kinds and get a callback with the id
 * mapped to a constant string. The strings provided in the arg will be valid pointers for the
 * entire duration of the program. It is recommended to call this function once and cache this data
 * in the client instead of making multiple on-demand calls.
 *
 * @param [in] callback Callback function invoked for each enumeration value in @ref
 * rocprofiler_service_callback_tracing_kind_t with the exception of the `NONE` and `LAST` values.
 * @param [in] data User data passed back into the callback
 */
rocprofiler_status_t ROCPROFILER_API
rocprofiler_iterate_callback_tracing_kind_names(
    rocprofiler_callback_tracing_kind_name_cb_t callback,
    void*                                       data) ROCPROFILER_NONNULL(1);

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
rocprofiler_iterate_callback_tracing_kind_operation_names(
    rocprofiler_service_callback_tracing_kind_t      kind,
    rocprofiler_callback_tracing_operation_name_cb_t callback,
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
rocprofiler_iterate_callback_tracing_operation_args(
    rocprofiler_callback_tracing_record_t            record,
    rocprofiler_callback_tracing_operation_args_cb_t callback,
    void*                                            user_data) ROCPROFILER_NONNULL(2);

/** @} */

ROCPROFILER_EXTERN_C_FINI
