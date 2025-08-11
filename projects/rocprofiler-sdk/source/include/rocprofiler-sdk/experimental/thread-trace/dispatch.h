// MIT License
//
// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include <rocprofiler-sdk/experimental/thread-trace/core.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/hsa.h>

ROCPROFILER_EXTERN_C_INIT

/**
 * @defgroup THREAD_TRACE Thread Trace Service
 * @brief Provides API calls to enable and handle thread trace data
 *
 * @{
 */

typedef enum rocprofiler_thread_trace_control_flags_t
{
    ROCPROFILER_THREAD_TRACE_CONTROL_NONE           = 0,
    ROCPROFILER_THREAD_TRACE_CONTROL_START_AND_STOP = 3
} rocprofiler_thread_trace_control_flags_t;

/**
 * @brief Callback to be triggered every kernel dispatch, indicating to start and/or stop ATT
 * @param [in] agent_id agent_id.
 * @param [in] queue_id queue_id.
 * @param [in] correlation_id internal correlation id.
 * @param [in] kernel_id kernel_id.
 * @param [in] dispatch_id dispatch_id.
 * @param [in] userdata_config Userdata passed back from
 * rocprofiler_configure_dispatch_thread_trace_service.
 * @param [out] userdata_shader Userdata to be passed in shader_callback
 */
typedef rocprofiler_thread_trace_control_flags_t (*rocprofiler_thread_trace_dispatch_callback_t)(
    rocprofiler_agent_id_t             agent_id,
    rocprofiler_queue_id_t             queue_id,
    rocprofiler_async_correlation_id_t correlation_id,
    rocprofiler_kernel_id_t            kernel_id,
    rocprofiler_dispatch_id_t          dispatch_id,
    void*                              userdata_config,
    rocprofiler_user_data_t*           userdata_shader);

/**
 * @brief Enables the thread trace service for dispatch-based tracing.
 * The tool has an option to enable/disable thread trace on every dispatch callback.
 * This service serializes all traced kernels, and optionally all non-traced kernels.
 * @param [in] context_id id of the context used for start/stop thread_trace.
 * @param [in] agent_id rocprofiler_agent_id_t to configure thread trace.
 * @param [in] parameters List of ATT-specific parameters.
 * @param [in] num_parameters Number of parameters. Zero is allowed.
 * @param [in] dispatch_callback Control fn which decides when TT starts/stop collecting.
 * @param [in] shader_callback Callback fn where the collected data will be sent to.
 * @param [in] callback_userdata Passed back to user in dispatch_callback.
 * @return ::rocprofiler_status_t
 * @retval ROCPROFILER_STATUS_SUCCESS on success
 * @retval ROCPROFILER_STATUS_ERROR_CONFIGURATION_LOCKED for configuration locked
 * @retval ROCPROFILER_STATUS_ERROR_CONTEXT_INVALID for conflicting configurations in the same ctx
 * @retval ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND for invalid context id
 * @retval ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT for invalid
 * rocprofiler_thread_trace_parameter_t
 * @retval ROCPROFILER_STATUS_ERROR_SERVICE_ALREADY_CONFIGURED if already configured
 */
rocprofiler_status_t
rocprofiler_configure_dispatch_thread_trace_service(
    rocprofiler_context_id_t                        context_id,
    rocprofiler_agent_id_t                          agent_id,
    rocprofiler_thread_trace_parameter_t*           parameters,
    size_t                                          num_parameters,
    rocprofiler_thread_trace_dispatch_callback_t    dispatch_callback,
    rocprofiler_thread_trace_shader_data_callback_t shader_callback,
    void*                                           callback_userdata) ROCPROFILER_API;

/** @} */

ROCPROFILER_EXTERN_C_FINI
