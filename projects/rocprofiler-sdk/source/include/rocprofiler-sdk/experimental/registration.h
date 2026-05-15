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
#include <rocprofiler-sdk/registration.h>

ROCPROFILER_EXTERN_C_INIT

/**
 * @defgroup EXPERIMENTAL_REGISTRATION_GROUP Experimental tool registration
 *
 * @brief Data types and functions for tool registration with rocprofiler
 * @{
 */

/**
 * @brief (experimental)
 *
 */
ROCPROFILER_SDK_EXPERIMENTAL
typedef void (*rocprofiler_client_detach_t)(rocprofiler_client_id_t);

/**
 * @brief Prototype for the start of the attach function that will be called after the
 * configuration.
 * @param [in] tool_data `tool_data` field returned from ::rocprofiler_configure_attach in
 * ::rocprofiler_tool_configure_result_t.
 */
ROCPROFILER_SDK_EXPERIMENTAL
typedef int (*rocprofiler_tool_attach_t)(rocprofiler_client_detach_t detach_func,
                                         rocprofiler_context_id_t*   context_ids,
                                         uint64_t                    context_ids_length,
                                         void*                       tool_data);

/**
 * @brief Prototype for the detach function where a tool can temporarily suspend operations.
 * @param [in] tool_data `tool_data` field returned from ::rocprofiler_configure in
 * ::rocprofiler_tool_configure_attach_result_t.
 */
ROCPROFILER_SDK_EXPERIMENTAL
typedef void (*rocprofiler_tool_detach_t)(void* tool_data);

/**
 * @brief (EXPERIMENTAL) Extended data structure containing initialization, finalization,
 * attach/detach, and data.
 *
 * This is an experimental extension of ::rocprofiler_tool_configure_result_t that adds support for
 * runtime attachment and detachment of tools. The `tool_attach` and `tool_detach` function
 * pointers allow tools to handle dynamic attachment scenarios where they may need to suspend and
 * resume profiling operations.
 *
 * The `size` field is used for ABI reasons and should be set to
 * `sizeof(rocprofiler_tool_configure_result_t)`
 */
typedef struct ROCPROFILER_SDK_EXPERIMENTAL rocprofiler_tool_configure_attach_result_t
{
    size_t                    size;         ///< size of this struct (in case of future extensions)
    rocprofiler_tool_attach_t tool_attach;  ///< after configuration
    rocprofiler_tool_detach_t tool_detach;  ///< end of attach session
    void*                     tool_data;    ///< data to provide to init and fini callbacks
} rocprofiler_tool_configure_attach_result_t;

/**
 * @brief (experimental) This is the special function that tools define to enable rocprofiler
 * attachment support.
 *
 * @param version
 * @param runtime_version
 * @param priority
 * @param client_id
 * @return rocprofiler_tool_configure_attach_result_t*
 */
ROCPROFILER_SDK_EXPERIMENTAL
rocprofiler_tool_configure_attach_result_t*
rocprofiler_configure_attach(uint32_t                 version,
                             const char*              runtime_version,
                             uint32_t                 priority,
                             rocprofiler_client_id_t* client_id) ROCPROFILER_PUBLIC_API;

/**
 * @brief Function pointer typedef for ::rocprofiler_configure_attach function
 * @param [in] version The version of rocprofiler: `(10000 * major) + (100 * minor) + patch`
 * @param [in] runtime_version String descriptor of the rocprofiler version and other relevant info.
 * @param [in] priority How many client tools were initialized before this client tool
 * @param [in, out] client_id tool identifier value.
 */
ROCPROFILER_SDK_EXPERIMENTAL
typedef rocprofiler_tool_configure_attach_result_t* (*rocprofiler_configure_attach_func_t)(
    uint32_t                 version,
    const char*              runtime_version,
    uint32_t                 priority,
    rocprofiler_client_id_t* client_id);

/** @} */

ROCPROFILER_EXTERN_C_FINI
