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

ROCPROFILER_EXTERN_C_INIT

/**
 * @defgroup INTERCEPT_TABLE Intercept table for runtime libraries
 * @brief Enable tools to wrap the runtime API function calls of HIP, HSA, and ROCTx before and
 * after the "real" implementation is called.
 *
 * When an application invokes the public function from the HIP, HSA, and ROCTx libraries, these
 * functions invoke a function pointer which, when properly chained, allow tools to wrap these
 * function calls to collect information. When this capability is used alongside the rocprofiler API
 * tracing, tools will wrap the rocprofiler wrappers of the API function, e.g. if the tool installs
 * a wrapper around the `hsa_init` function called `tool_hsa_init`, and rocprofiler installs a
 * wrapper around the `hsa_init` function called `rocp_hsa_init`, and within the HSA runtime
 * library, the "real" implementation of the `hsa_init` invokes a function called `real_hsa_init`,
 * the invocation chain (starting from within the user application) will be: `<application>` ->
 * `hsa_init` -> `tool_hsa_init` -> `rocp_hsa_init` -> `real_hsa_init`. The return sequence will be
 * the inverse of invocation chain: `real_hsa_init` -> `rocp_hsa_init` -> `tool_hsa_init` ->
 * `<application>`. Thus, it is important for tools that use this feature to (A) call the next
 * function in the chain and (B) properly handle the return value.
 *
 * @{
 */

/**
 * @brief (experimental) Callback type when a new runtime library is loaded. @see
 * rocprofiler_at_intercept_table_registration
 * @param [in] type Type of API table
 * @param [in] lib_version Major, minor, and patch version of library encoded into single number
 * similar to ::ROCPROFILER_VERSION
 * @param [in] lib_instance The number of times this runtime library has been registered previously
 * @param [in] tables An array of pointers to the API tables
 * @param [in] num_tables The size of the array of pointers to the API tables
 * @param [in] user_data The pointer to the data provided to
 * ::rocprofiler_at_intercept_table_registration
 */
ROCPROFILER_SDK_EXPERIMENTAL
typedef void (*rocprofiler_intercept_library_cb_t)(rocprofiler_intercept_table_t type,
                                                   uint64_t                      lib_version,
                                                   uint64_t                      lib_instance,
                                                   void**                        tables,
                                                   uint64_t                      num_tables,
                                                   void*                         user_data);

/**
 * @brief (experimental) Invoke this function to receive callbacks when a ROCm library registers its
 * API intercept table with rocprofiler. Use the ::rocprofiler_intercept_table_t enumeration for
 * specifying which raw API tables the tool would like to have access to. E.g. including
 * ::ROCPROFILER_HSA_TABLE in the ::rocprofiler_at_intercept_table_registration function call
 * communicates to rocprofiler that, when rocprofiler receives a `HsaApiTable` instance, the tool
 * would like rocprofiler to provide it access too.
 *
 * When the HIP, HSA, and ROCTx libraries are initialized (either explicitly or on the first
 * invocation of one of their public API functions), these runtimes will provide a table of function
 * pointers to the rocprofiler library via the rocprofiler-register library if the
 * `rocprofiler_configure` symbol is visible in the application's symbol table. The vast majority of
 * tools will want to use the @ref CALLBACK_TRACING_SERVICE to trace these runtime APIs, however,
 * some tools may want or require installing their own intercept functions in lieu of receiving
 * these callbacks and those tools should use the ::rocprofiler_at_intercept_table_registration
 * to install their intercept functions. There are no restrictions to where or how early this
 * function can be invoked but it will return ::ROCPROFILER_STATUS_ERROR_CONFIGURATION_LOCKED if it
 * is invoked after rocprofiler has requested all the tool configurations. Thus, it is highly
 * recommended to invoke this function within the ::rocprofiler_configure function or the
 * callback passed to the ::rocprofiler_force_configure function -- the reason for this
 * recommendation is that if ::rocprofiler_at_intercept_table_registration is invoked in one of
 * these locations, rocprofiler can guarantee that the tool will be passed the API table because, at
 * the first instance of a runtime registering it's API table, rocprofiler will ensure that, in the
 * case of the former, rocprofiler will invoke all of the ::rocprofiler_configure symbols that
 * are visible before checking the list of tools which want to receive the API tables and, in the
 * case of the latter, ::rocprofiler_force_configure will fail with error code
 * ::ROCPROFILER_STATUS_ERROR_CONFIGURATION_LOCKED if a runtime has already been registered (and,
 * therefore, already scanned and invoked the visible ::rocprofiler_configure symbols and
 * completed the tool initialization). If ::rocprofiler_at_intercept_table_registration is
 * invoked outside of these recommended places, even if it is done before the `main` function starts
 * (e.g. in a library init/constructor function), it is possible that another library, such as
 * ROCm-aware MPI, caused the HIP and HSA runtime libraries to be initialized when that library was
 * loaded. In this aforementioned scenario, if the ROCm-aware MPI library library init/constructor
 * function runs before your library init/constructor function, rocprofiler will have already
 * processed the API table and will not provide the API table to the tool due to the fact that the
 * API may already be in use and, thus, any modifications to the table might result in thread-safety
 * violations or more disastrous consequences.
 *
 * @param [in] callback Callback to tool invoked when a runtime registers their API table with
 * rocprofiler
 * @param [in] libs Bitwise-or of libraries, e.g. `ROCPROFILER_HSA_TABLE |
 * ROCPROFILER_HIP_RUNTIME_TABLE | ROCPROFILER_MARKER_CORE_TABLE` means the callbacks will be
 * invoked whenever the HSA, HIP runtime, and ROCTx core API tables register their intercept
 * table(s).
 * @param [in] data Data to provide to callback(s)
 * @return ::rocprofiler_status_t
 * @retval ::ROCPROFILER_STATUS_SUCCESS Callback was registered for specified runtime(s)
 * @retval ::ROCPROFILER_STATUS_ERROR_CONFIGURATION_LOCKED rocprofiler has already initialized
 * @retval ::ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT this error code is returned if
 * `ROCPROFILER_TABLE` is included in bitwise-or of the libs
 * @retval ::ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED this error code is returned if one of the
 * specified libraries does not have support for API intercept tables (which should not be the case
 * by the time this code is publicly released)
 *
 * @code{.cpp}
 * namespace
 * {
 * // this function generates a wrapper around the original function that
 * // prints out the function name and then invokes the original function
 * template <size_t Idx, typename RetT, typename... Args>
 * auto
 * generate_wrapper(const char* name, RetT (*func)(Args...))
 * {
 *     using functor_type = RetT (*)(Args...);
 *
 *     // save function name, "real function"
 *     static const auto*  func_name       = name;
 *     static functor_type underlying_func = func;
 *     static functor_type wrapped_func    = [](Args... args) -> RetT {
 *         std::cout << "Wrapping " << func_name << "..." << std::endl;
 *         if(underlying_func) return underlying_func(args...);
 *         if constexpr(!std::is_void<RetT>::value) return RetT{};
 *     };
 *
 *     return wrapped_func;
 * }
 *
 *
 * // this macro installs the wrapper in place of the original function
 * #define GENERATE_WRAPPER(TABLE, FUNC)                                                         \
 *      TABLE->FUNC##_fn = generate_wrapper<__COUNTER__>(#FUNC, TABLE->FUNC##_fn)
 *
 *
 * // this is the function that gets called when the HSA runtime
 * // intercept table is registered with rocprofiler
 * void
 * api_registration_callback(rocprofiler_intercept_table_t type,
 *                           uint64_t                        lib_version,
 *                           uint64_t                        lib_instance,
 *                           void**                          tables,
 *                           uint64_t                        num_tables,
 *                           void*                           user_data)
 * {
 *     if(type != ROCPROFILER_HSA_TABLE)
 *         throw std::runtime_error{"unexpected library type: " +
 *                                  std::to_string(static_cast<int>(type))};
 *     if(lib_instance != 0) throw std::runtime_error{"multiple instances of HSA runtime library"};
 *     if(num_tables != 1) throw std::runtime_error{"expected only one table of type HsaApiTable"};
 *
 *     auto* hsa_api_table = static_cast<HsaApiTable*>(tables[0]);
 *     GENERATE_WRAPPER(hsa_api_table->core_, hsa_agent_get_info);
 *     GENERATE_WRAPPER(hsa_api_table->core_, hsa_agent_iterate_isas);
 *     GENERATE_WRAPPER(hsa_api_table->core_, hsa_code_object_reader_create_from_memory);
 *     GENERATE_WRAPPER(hsa_api_table->core_, hsa_executable_create_alt);
 *     GENERATE_WRAPPER(hsa_api_table->core_, hsa_executable_freeze);
 *     GENERATE_WRAPPER(hsa_api_table->core_, hsa_executable_get_symbol_by_name);
 *     GENERATE_WRAPPER(hsa_api_table->core_, hsa_executable_iterate_symbols);
 *     GENERATE_WRAPPER(hsa_api_table->core_, hsa_executable_load_agent_code_object);
 *     GENERATE_WRAPPER(hsa_api_table->core_, hsa_executable_symbol_get_info);
 *     GENERATE_WRAPPER(hsa_api_table->core_, hsa_isa_get_info_alt);
 *     GENERATE_WRAPPER(hsa_api_table->core_, hsa_iterate_agents);
 *     GENERATE_WRAPPER(hsa_api_table->core_, hsa_queue_add_write_index_screlease);
 *     GENERATE_WRAPPER(hsa_api_table->core_, hsa_queue_create);
 *     GENERATE_WRAPPER(hsa_api_table->core_, hsa_queue_load_read_index_relaxed);
 *     GENERATE_WRAPPER(hsa_api_table->core_, hsa_queue_load_read_index_scacquire);
 *     GENERATE_WRAPPER(hsa_api_table->core_, hsa_signal_create);
 *     GENERATE_WRAPPER(hsa_api_table->core_, hsa_signal_destroy);
 *     GENERATE_WRAPPER(hsa_api_table->core_, hsa_signal_load_relaxed);
 *     GENERATE_WRAPPER(hsa_api_table->core_, hsa_signal_silent_store_relaxed);
 *     GENERATE_WRAPPER(hsa_api_table->core_, hsa_signal_store_screlease);
 *     GENERATE_WRAPPER(hsa_api_table->core_, hsa_signal_wait_scacquire);
 *     GENERATE_WRAPPER(hsa_api_table->core_, hsa_system_get_info);
 *     GENERATE_WRAPPER(hsa_api_table->core_, hsa_system_get_major_extension_table);
 * }
 * }  // namespace
 *
 *
 * extern "C" rocprofiler_tool_configure_result_t*
 * rocprofiler_configure(uint32_t                 version,
 *                       const char*              runtime_version,
 *                       uint32_t                 priority,
 *                       rocprofiler_client_id_t* id)
 * {
 *     // set the client name
 *     id->name = "ExampleTool";
 *
 *     // specify that we only want to intercept the HSA library
 *     rocprofiler_at_intercept_table_registration(api_registration_callback,
 *                                             ROCPROFILER_HSA_TABLE, nullptr);
 *
 *     return nullptr;
 * }
 * @endcode
 *
 * @example intercept_table/client.cpp
 * Example demonstrating ::rocprofiler_at_intercept_table_registration usage
 */
ROCPROFILER_SDK_EXPERIMENTAL
rocprofiler_status_t
rocprofiler_at_intercept_table_registration(rocprofiler_intercept_library_cb_t callback,
                                            int                                libs,
                                            void* data) ROCPROFILER_API;

/** @} */

ROCPROFILER_EXTERN_C_FINI
