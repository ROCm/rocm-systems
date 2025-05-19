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

#include <rocprofiler-sdk/defines.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/hip.h>
#include <rocprofiler-sdk/hsa.h>
#include <rocprofiler-sdk/marker.h>
#include <rocprofiler-sdk/ompt.h>
#include <rocprofiler-sdk/rccl.h>
#include <rocprofiler-sdk/rocdecode.h>
#include <rocprofiler-sdk/rocjpeg.h>

#include <hsa/hsa.h>
#include <hsa/hsa_amd_tool.h>
#include <hsa/hsa_ext_amd.h>
#include <hsa/hsa_ven_amd_loader.h>

#include <stdint.h>

ROCPROFILER_EXTERN_C_INIT

/**
 * @defgroup CALLBACK_TRACING_SERVICE Synchronous Tracing Services
 * @brief Receive immediate callbacks on the calling thread
 *
 * @{
 */

/**
 * @brief ROCProfiler Enumeration for code object storage types (identical values to
 * `hsa_ven_amd_loader_code_object_storage_type_t` enumeration)
 */
typedef enum rocprofiler_code_object_storage_type_t
{
    ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_NONE = HSA_VEN_AMD_LOADER_CODE_OBJECT_STORAGE_TYPE_NONE,
    ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_FILE = HSA_VEN_AMD_LOADER_CODE_OBJECT_STORAGE_TYPE_FILE,
    ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_MEMORY =
        HSA_VEN_AMD_LOADER_CODE_OBJECT_STORAGE_TYPE_MEMORY,
    ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_LAST,
} rocprofiler_code_object_storage_type_t;

/**
 * @brief ROCProfiler HSA API Callback Data.
 */
typedef struct rocprofiler_callback_tracing_hsa_api_data_t
{
    uint64_t                     size;  ///< size of this struct
    rocprofiler_hsa_api_args_t   args;
    rocprofiler_hsa_api_retval_t retval;
} rocprofiler_callback_tracing_hsa_api_data_t;

/**
 * @brief ROCProfiler HIP runtime and compiler API Tracer Callback Data.
 */
typedef struct rocprofiler_callback_tracing_hip_api_data_t
{
    uint64_t                     size;  ///< size of this struct
    rocprofiler_hip_api_args_t   args;
    rocprofiler_hip_api_retval_t retval;  ///< return value of function call

    /// @var args
    /// @brief Arguments of the function call. @see
    /// ::rocprofiler_iterate_callback_tracing_kind_operation_args for generic
    /// access/stringification of the arguments.
} rocprofiler_callback_tracing_hip_api_data_t;

/**
 * @brief ROCProfiler OMPT Callback Data
 */
typedef struct rocprofiler_callback_tracing_ompt_data_t
{
    uint64_t                size;  ///< size of this struct
    rocprofiler_ompt_args_t args;
} rocprofiler_callback_tracing_ompt_data_t;

/**
 * @brief ROCProfiler Marker Tracer Callback Data.
 */
typedef struct rocprofiler_callback_tracing_marker_api_data_t
{
    uint64_t                        size;  ///< size of this struct
    rocprofiler_marker_api_args_t   args;
    rocprofiler_marker_api_retval_t retval;
} rocprofiler_callback_tracing_marker_api_data_t;

/**
 * @brief ROCProfiler RCCL API Callback Data.
 */
typedef struct rocprofiler_callback_tracing_rccl_api_data_t
{
    uint64_t                      size;  ///< size of this struct
    rocprofiler_rccl_api_args_t   args;
    rocprofiler_rccl_api_retval_t retval;
} rocprofiler_callback_tracing_rccl_api_data_t;

/**
 * @brief ROCProfiler rocDecode API Callback Data.
 */
typedef struct rocprofiler_callback_tracing_rocdecode_api_data_t
{
    uint64_t                           size;  ///< size of this struct
    rocprofiler_rocdecode_api_args_t   args;
    rocprofiler_rocdecode_api_retval_t retval;
} rocprofiler_callback_tracing_rocdecode_api_data_t;

/**
 * @brief ROCProfiler rocJPEG API Callback Data.
 */
typedef struct rocprofiler_callback_tracing_rocjpeg_api_data_t
{
    uint64_t                         size;  ///< size of this struct
    rocprofiler_rocjpeg_api_args_t   args;
    rocprofiler_rocjpeg_api_retval_t retval;
} rocprofiler_callback_tracing_rocjpeg_api_data_t;

/**
 * @brief ROCProfiler Code Object Load Tracer Callback Record.
 */
typedef struct rocprofiler_callback_tracing_code_object_load_data_t
{
    uint64_t size;            ///< size of this struct
    uint64_t code_object_id;  ///< unique code object identifier
    union
    {
        rocprofiler_agent_id_t rocp_agent;  ///< Deprecated. Renamed to agent_id
        rocprofiler_agent_id_t agent_id;  ///< The agent on which this loaded code object is loaded
    };
    hsa_agent_t hsa_agent;  ///< Deprecated. The agent on which this loaded code object is loaded
    const char* uri;        ///< The URI name from which the code object was loaded
    uint64_t    load_base;
    uint64_t    load_size;
    int64_t     load_delta;
    rocprofiler_code_object_storage_type_t storage_type;

    /// @var load_base
    /// @brief The base memory address at which the code object is loaded. This is the base address
    /// of the allocation for the lowest addressed segment of the code object that is loaded. Note
    /// that any non-loaded segments before the first loaded segment are ignored.
    ///
    /// @var load_size
    /// @brief The byte size of the loaded code objects contiguous memory allocation.
    ///
    /// @var load_delta
    /// @brief The signed byte address difference of the memory address at which the code object is
    /// loaded minus the virtual address specified in the code object that is loaded.
    ///
    /// @var storage_type
    /// @brief storage type of the code object reader used to load the loaded code object
    ///

    union
    {
        struct
        {
            int storage_file;  ///< file descriptor of the code object that was loaded. Access this
                               ///< field if ::rocprofiler_code_object_storage_type_t is
                               ///< ::ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_FILE
        };
        struct
        {
            uint64_t memory_base;  ///< The memory address of the first byte of the code object that
                                   ///< was loaded. Access this
                                   ///< field if ::rocprofiler_code_object_storage_type_t is
                                   ///< ::ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_MEMORY
            uint64_t
                memory_size;  ///< The memory size in bytes of the code object that was loaded.
                              ///< Access this field if ::rocprofiler_code_object_storage_type_t is
                              ///< ::ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_MEMORY
        };
    };
} rocprofiler_callback_tracing_code_object_load_data_t;

/**
 * @brief The NULL value of a code object id. Used when code object is unknown.
 */
#define ROCPROFILER_CODE_OBJECT_ID_NONE ROCPROFILER_UINT64_C(0)

/**
 * @brief ROCProfiler Code Object Kernel Symbol Tracer Callback Record.
 *
 */
typedef struct rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t
{
    uint64_t              size;            ///< size of this struct
    uint64_t              kernel_id;       ///< unique symbol identifier value
    uint64_t              code_object_id;  ///< parent unique code object identifier
    const char*           kernel_name;     ///< name of the kernel
    uint64_t              kernel_object;
    uint32_t              kernarg_segment_size;
    uint32_t              kernarg_segment_alignment;
    uint32_t              group_segment_size;
    uint32_t              private_segment_size;
    uint32_t              sgpr_count;        ///< Scalar general purpose register count
    uint32_t              arch_vgpr_count;   ///< Architecture vector general purpose register count
    uint32_t              accum_vgpr_count;  ///< Accum vector general purpose register count
    int64_t               kernel_code_entry_byte_offset;
    rocprofiler_address_t kernel_address;

    /// @var kernel_object
    /// @brief kernel object handle, used in the kernel dispatch packet
    ///
    /// @var kernarg_segment_size
    /// @brief size of memory (in bytes) allocated for kernel arguments. Will be multiple of 16
    ///
    /// @var kernarg_segment_alignment
    /// @brief Alignment (in bytes) of the buffer used to pass arguments to the kernel
    ///
    /// @var group_segment_size
    /// @brief Size of static group segment memory required by the kernel (per work-group), in
    /// bytes. AKA: LDS size
    ///
    /// @var private_segment_size
    /// @brief Size of static private, spill, and arg segment memory required by this kernel (per
    /// work-item), in bytes. AKA: scratch size
    ///
    /// @var kernel_code_entry_byte_offset
    /// @brief Relative offset from kernel_object address to calculate the first address of a
    /// kernel.
    ///
    /// @var kernel_address
    /// @brief The first address of a kernel. Useful for PC sampling.
    ///
} rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t;
// rename struct

typedef struct rocprofiler_callback_tracing_code_object_host_kernel_symbol_register_data_t
{
    uint64_t              size;              ///< size of this struct
    uint64_t              host_function_id;  ///< unique host function identifier value
    uint64_t              kernel_id;         ///< unique symbol identifier value
    uint64_t              code_object_id;    ///< parent unique code object identifier
    rocprofiler_address_t host_function;     ///< kernel host function pointer
    rocprofiler_address_t modules;           ///< reference address where modules will be loaded
    const char*           device_function;
    uint32_t              thread_limit;    ///< thread limit
    rocprofiler_dim3_t    thread_ids;      ///< thread ids address
    rocprofiler_dim3_t    block_ids;       ///< block ids address
    rocprofiler_dim3_t    block_dims;      ///< block dimensions address
    rocprofiler_dim3_t    grid_dims;       ///< grid dimensions address
    uint64_t              workgroup_size;  ///< workgroup size address

    /// @var device_function
    /// @brief device function name used to map the metadata during kernel launch
} rocprofiler_callback_tracing_code_object_host_kernel_symbol_register_data_t;

/**
 * @brief ROCProfiler Kernel Dispatch Callback Tracer Record.
 *
 */
typedef struct rocprofiler_callback_tracing_kernel_dispatch_data_t
{
    uint64_t                           size;             ///< size of this struct
    rocprofiler_timestamp_t            start_timestamp;  ///< start time in nanoseconds
    rocprofiler_timestamp_t            end_timestamp;    ///< end time in nanoseconds
    rocprofiler_kernel_dispatch_info_t dispatch_info;    ///< Dispatch info
} rocprofiler_callback_tracing_kernel_dispatch_data_t;

/**
 * @brief ROCProfiler Memory Copy Callback Tracer Record.
 *
 * The timestamps in this record will only be non-zero in the ::ROCPROFILER_CALLBACK_PHASE_EXIT
 * callback
 */
typedef struct rocprofiler_callback_tracing_memory_copy_data_t
{
    uint64_t                size;             ///< size of this struct
    rocprofiler_timestamp_t start_timestamp;  ///< start time in nanoseconds
    rocprofiler_timestamp_t end_timestamp;    ///< end time in nanoseconds
    rocprofiler_agent_id_t  dst_agent_id;     ///< destination agent of copy
    rocprofiler_agent_id_t  src_agent_id;     ///< source agent of copy
    uint64_t                bytes;            ///< bytes copied
    rocprofiler_address_t   dst_address;      ///< destination address
    rocprofiler_address_t   src_address;      ///< source address
} rocprofiler_callback_tracing_memory_copy_data_t;

/**
 * @brief ROCProfiler Memory Allocation Tracer Record.
 */
typedef struct rocprofiler_callback_tracing_memory_allocation_data_t
{
    uint64_t                size;             ///< size of this struct
    rocprofiler_timestamp_t start_timestamp;  ///< start time in nanoseconds
    rocprofiler_timestamp_t end_timestamp;    ///< end time in nanoseconds
    rocprofiler_agent_id_t  agent_id;         ///< agent id for memory allocation
    rocprofiler_address_t   address;          ///< starting address for memory allocation
    uint64_t                allocation_size;  ///< size of memory allocation
} rocprofiler_callback_tracing_memory_allocation_data_t;

/**
 * @brief ROCProfiler Scratch Memory Callback Data.
 */
typedef struct rocprofiler_callback_tracing_scratch_memory_data_t
{
    uint64_t                          size;  ///< size of this struct
    rocprofiler_agent_id_t            agent_id;
    rocprofiler_queue_id_t            queue_id;
    rocprofiler_scratch_alloc_flag_t  flags;
    hsa_amd_tool_event_kind_t         args_kind;
    rocprofiler_scratch_memory_args_t args;
} rocprofiler_callback_tracing_scratch_memory_data_t;

/**
 * @brief ROCProfiler Runtime Initialization Data.
 */
typedef struct rocprofiler_callback_tracing_runtime_initialization_data_t
{
    uint64_t size;  ///< size of this struct
    uint64_t version;
    uint64_t instance;  ///< Number of times this runtime had been loaded previously

    /// @var version
    /// @brief The version number of the library
    ///
    /// Version number is encoded as: (10000 * MAJOR) + (100 * MINOR) + PATCH
} rocprofiler_callback_tracing_runtime_initialization_data_t;

/**
 * @brief ROCProfiler Stream Handle Callback Data.
 */
typedef struct rocprofiler_callback_tracing_hip_stream_data_t
{
    uint64_t                size;          ///< size of this struct
    rocprofiler_stream_id_t stream_id;     ///< HIP stream ID
    rocprofiler_address_t   stream_value;  ///< HIP stream value
} rocprofiler_callback_tracing_hip_stream_data_t;

/**
 * @brief API Tracing callback function. This function is invoked twice per API function: once
 * before the function is invoked and once after the function is invoked.  The external correlation
 * id value within the record is assigned the value at the top of the external correlation id stack.
 * It is permissible to invoke ::rocprofiler_push_external_correlation_id within the enter phase;
 * when a new external correlation id is pushed during the enter phase, rocprofiler will use that
 * external correlation id for any async events and provide the new external correlation id during
 * the exit callback... In other words, pushing a new external correlation id within the enter
 * callback will result in that external correlation id value in the exit callback (which may or may
 * not be different from the external correlation id value in the enter callback). If a tool pushes
 * new external correlation ids in the enter phase, it is recommended to pop the external
 * correlation id in the exit callback.
 *
 * @param [in] record Callback record data
 * @param [in,out] user_data This paramter can be used to retain information in between the enter
 * and exit phases.
 * @param [in] callback_data User data provided when configuring the callback tracing service
 */
typedef void (*rocprofiler_callback_tracing_cb_t)(rocprofiler_callback_tracing_record_t record,
                                                  rocprofiler_user_data_t*              user_data,
                                                  void* callback_data) ROCPROFILER_NONNULL(2);

/**
 * @brief Callback function for mapping ::rocprofiler_callback_tracing_kind_t ids to
 * string names. @see rocprofiler_iterate_callback_tracing_kind_names.
 */
typedef int (*rocprofiler_callback_tracing_kind_cb_t)(rocprofiler_callback_tracing_kind_t kind,
                                                      void*                               data);

/**
 * @brief Callback function for mapping the operations of a given
 * ::rocprofiler_callback_tracing_kind_t to string names. @see
 * rocprofiler_iterate_callback_tracing_kind_operation_names.
 */
typedef int (*rocprofiler_callback_tracing_kind_operation_cb_t)(
    rocprofiler_callback_tracing_kind_t kind,
    rocprofiler_tracing_operation_t     operation,
    void*                               data);

/**
 * @brief Callback function for iterating over the function arguments to a traced function.
 * This function will be invoked for each argument.
 * @see rocprofiler_iterate_callback_tracing_operation_args
 *
 * @param [in] kind domain
 * @param [in] operation associated domain operation
 * @param [in] arg_number the argument number, starting at zero
 * @param [in] arg_value_addr the address of the argument stored by rocprofiler.
 * @param [in] arg_indirection_count the total number of indirection levels for the argument, e.g.
 * int == 0, int* == 1, int** == 2
 * @param [in] arg_type the typeid name of the argument
 * @param [in] arg_name the name of the argument in the prototype (or rocprofiler union)
 * @param [in] arg_value_str conversion of the argument to a string, e.g. operator<< overload
 * @param [in] arg_dereference_count the number of times the argument was dereferenced when it was
 * converted to a string
 * @param [in] data user data
 */
typedef int (*rocprofiler_callback_tracing_operation_args_cb_t)(
    rocprofiler_callback_tracing_kind_t kind,
    rocprofiler_tracing_operation_t     operation,
    uint32_t                            arg_number,
    const void* const                   arg_value_addr,
    int32_t                             arg_indirection_count,
    const char*                         arg_type,
    const char*                         arg_name,
    const char*                         arg_value_str,
    int32_t                             arg_dereference_count,
    void*                               data);

/**
 * @brief Configure Callback Tracing Service. The callback tracing service provides two synchronous
 * callbacks around an API function on the same thread as the application which is invoking the API
 * function. This function can only be invoked once per ::rocprofiler_callback_tracing_kind_t value,
 * i.e. it can be invoked once for the HSA API, once for the HIP API, and so on but it will fail if
 * it is invoked for the HSA API twice. Please note, the callback API does have the potentially
 * non-trivial overhead of copying the function arguments into the record. If you are willing to let
 * rocprofiler record the timestamps, do not require synchronous notifications of the API calls, and
 * want to lowest possible overhead, use the
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
 * @return ::rocprofiler_status_t
 * @retval ::ROCPROFILER_STATUS_ERROR_CONFIGURATION_LOCKED Invoked outside of the initialization
 * function in ::rocprofiler_tool_configure_result_t provided to rocprofiler via
 * ::rocprofiler_configure function
 * @retval ::ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND The provided context is not valid/registered
 * @retval ::ROCPROFILER_STATUS_ERROR_SERVICE_ALREADY_CONFIGURED if the same
 * ::rocprofiler_callback_tracing_kind_t value is provided more than once (per context) -- in other
 * words, we do not support overriding or combining the operations in separate function calls.
 *
 */
rocprofiler_status_t
rocprofiler_configure_callback_tracing_service(rocprofiler_context_id_t               context_id,
                                               rocprofiler_callback_tracing_kind_t    kind,
                                               const rocprofiler_tracing_operation_t* operations,
                                               size_t                            operations_count,
                                               rocprofiler_callback_tracing_cb_t callback,
                                               void* callback_args) ROCPROFILER_API;

/**
 * @brief Query the name of the callback tracing kind. The name retrieved from this function is a
 * string literal that is encoded in the read-only section of the binary (i.e. it is always
 * "allocated" and never "deallocated").
 *
 * @param [in] kind Callback tracing domain
 * @param [out] name If non-null and the name is a constant string that does not require dynamic
 * allocation, this paramter will be set to the address of the string literal, otherwise it will
 * be set to nullptr
 * @param [out] name_len If non-null, this will be assigned the length of the name (regardless of
 * the name is a constant string or requires dynamic allocation)
 * @return ::rocprofiler_status_t
 */
rocprofiler_status_t
rocprofiler_query_callback_tracing_kind_name(rocprofiler_callback_tracing_kind_t kind,
                                             const char**                        name,
                                             uint64_t* name_len) ROCPROFILER_API;

/**
 * @brief Query the name of the callback tracing kind. The name retrieved from this function is a
 * string literal that is encoded in the read-only section of the binary (i.e. it is always
 * "allocated" and never "deallocated").
 *
 * @param [in] kind Callback tracing domain
 * @param [in] operation Enumeration id value which maps to a specific API function or event type
 * @param [out] name If non-null and the name is a constant string that does not require dynamic
 * allocation, this paramter will be set to the address of the string literal, otherwise it will
 * be set to nullptr
 * @param [out] name_len If non-null, this will be assigned the length of the name (regardless of
 * the name is a constant string or requires dynamic allocation)
 * @return ::rocprofiler_status_t
 * @retval ::ROCPROFILER_STATUS_ERROR_KIND_NOT_FOUND Domain id is not valid
 * @retval ::ROCPROFILER_STATUS_SUCCESS Valid domain provided, regardless if there is a constant
 * string or not.
 */
rocprofiler_status_t
rocprofiler_query_callback_tracing_kind_operation_name(rocprofiler_callback_tracing_kind_t kind,
                                                       rocprofiler_tracing_operation_t operation,
                                                       const char**                    name,
                                                       uint64_t* name_len) ROCPROFILER_API;

/**
 * @brief Iterate over all the mappings of the callback tracing kinds and get a callback for each
 * kind.
 *
 * @param [in] callback Callback function invoked for each enumeration value in
 * ::rocprofiler_callback_tracing_kind_t with the exception of the `NONE` and `LAST` values.
 * @param [in] data User data passed back into the callback
 * @return ::rocprofiler_status_t
 */
rocprofiler_status_t
rocprofiler_iterate_callback_tracing_kinds(rocprofiler_callback_tracing_kind_cb_t callback,
                                           void* data) ROCPROFILER_API ROCPROFILER_NONNULL(1);

/**
 * @brief Iterates over all the mappings of the operations for a given
 * ::rocprofiler_callback_tracing_kind_t and invokes the callback with the kind id, operation id,
 * and user-provided data.
 *
 * @param [in] kind which tracing callback kind operations to iterate over
 * @param [in] callback Callback function invoked for each operation associated with
 * ::rocprofiler_callback_tracing_kind_t with the exception of the `NONE` and `LAST` values.
 * @param [in] data User data passed back into the callback
 * @return ::rocprofiler_status_t
 * @retval ::ROCPROFILER_STATUS_ERROR_KIND_NOT_FOUND Invalid domain id
 * @retval ::ROCPROFILER_STATUS_SUCCESS Valid domain
 */
rocprofiler_status_t
rocprofiler_iterate_callback_tracing_kind_operations(
    rocprofiler_callback_tracing_kind_t              kind,
    rocprofiler_callback_tracing_kind_operation_cb_t callback,
    void*                                            data) ROCPROFILER_API ROCPROFILER_NONNULL(2);

/**
 * @brief Iterates over all the arguments for the traced function (when available). This is
 * particularly useful when tools want to annotate traces with the function arguments. See
 * @example samples/api_callback_tracing/client.cpp for a usage example.
 *
 * It is recommended to use this function when the record phase is ::ROCPROFILER_CALLBACK_PHASE_EXIT
 * or ::ROCPROFILER_CALLBACK_PHASE_NONE. When the phase is ::ROCPROFILER_CALLBACK_PHASE_ENTER, the
 * function may have output parameters which have not set. In the case of an output parameter with
 * one level of indirection, e.g. `int* output_len`, this is considered safe since the output
 * parameter is either null or, in the worst case scenario, pointing to an uninitialized value which
 * will result in garbage values to be stringified. However, if the output parameter has more than
 * one level of indirection, e.g. `const char** output_name`, this can result in a segmentation
 * fault because the dereferenced output parameter may be uninitialized and point to an invalid
 address. E.g.:
 *
 * @code{.cpp}
 * struct dim3
 * {
 *     int x;
 *     int y;
 *     int z;
 * };
 *
 * static dim3 default_dims = {.x = 1, .y = 1, .z = 1};
 *
 * void set_dim_x(int val, dim3* output_dims) { output_dims->x = val; }
 *
 * void get_default_dims(dim3** output_dims) { *output_dims = default_dims; }
 *
 * int main()
 * {
 *     dim3  my_dims;       // uninitialized value. x, y, and z may be set to random values
 *     dim3* current_dims;  // uninitialized pointer. May be set to invalid address
 *
 *     set_dim_x(3, &my_dims);  // if rocprofiler-sdk wrapped this function and tried to stringify
 *                              // in the enter phase, dereferencing my_dims is not problematic
 *                              // since there is an actual dim3 allocation
 *
 *     get_default_dims(&current_dims);  // if rocprofiler-sdk wrapped this function,
 *                                       // and tried to stringify in the enter phase,
 *                                       // current_dims may point to an address outside
 *                                       // of the address space of this process and
 *                                       // cause a segfault
 * }
 * @endcode
 *
 *
 * @param[in] record Record provided by service callback
 * @param[in] callback The callback function which will be invoked for each argument
 * @param[in] max_dereference_count In the callback enter phase, certain arguments may be output
 * parameters which have not been set. When the output parameter has multiple levels of indirection,
 * it may be invalid to dereference the output parameter more than once and doing so may result in a
 * segmentation fault. Thus, it is recommended to set this parameter to a maximum value of 1 when
 * the phase is ::ROCPROFILER_CALLBACK_PHASE_ENTER to ensure that output parameters which point to
 * uninitialized pointers do not cause segmentation faults.
 * @param[in] user_data Data to be passed to each invocation of the callback
 */
rocprofiler_status_t
rocprofiler_iterate_callback_tracing_kind_operation_args(
    rocprofiler_callback_tracing_record_t            record,
    rocprofiler_callback_tracing_operation_args_cb_t callback,
    int32_t                                          max_dereference_count,
    void* user_data) ROCPROFILER_API ROCPROFILER_NONNULL(2);

/** @} */

ROCPROFILER_EXTERN_C_FINI
