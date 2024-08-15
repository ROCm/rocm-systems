// MIT License
//
// Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include <rocprofiler-sdk/amd_detail/thread_trace_core.h>
#include <rocprofiler-sdk/defines.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/hsa.h>

ROCPROFILER_EXTERN_C_INIT

/**
 * @defgroup THREAD_TRACE Thread Trace Service
 * @brief Provides API calls to enable and handle thread trace data
 *
 * @{
 */

typedef enum
{
    ROCPROFILER_ATT_CONTROL_NONE           = 0,
    ROCPROFILER_ATT_CONTROL_START_AND_STOP = 3
} rocprofiler_att_control_flags_t;

/**
 * @brief Callback to be triggered every kernel dispatch, indicating to start and/or stop ATT
 */
typedef rocprofiler_att_control_flags_t (*rocprofiler_att_dispatch_callback_t)(
    rocprofiler_queue_id_t       queue_id,
    const rocprofiler_agent_t*   agent,
    rocprofiler_correlation_id_t correlation_id,
    rocprofiler_kernel_id_t      kernel_id,
    rocprofiler_dispatch_id_t    dispatch_id,
    rocprofiler_user_data_t*     userdata_shader,
    void*                        userdata_config);

/**
 * @brief Enables the advanced thread trace service for dispatch-based tracing.
 * The tool has an option to enable/disable thread trace on every dispatch callback.
 * This service enables kernel serialization.
 * @param [in] context_id context_id.
 * @param [in] parameters List of ATT-specific parameters.
 * @param [in] num_parameters Number of parameters. Zero is allowed.
 * @param [in] dispatch_callback Control fn which decides when ATT starts/stop collecting.
 * @param [in] shader_callback Callback fn where the collected data will be sent to.
 * @param [in] callback_userdata Passed back to user.
 */
rocprofiler_status_t
rocprofiler_configure_dispatch_thread_trace_service(
    rocprofiler_context_id_t               context_id,
    rocprofiler_att_parameter_t*           parameters,
    size_t                                 num_parameters,
    rocprofiler_att_dispatch_callback_t    dispatch_callback,
    rocprofiler_att_shader_data_callback_t shader_callback,
    void*                                  callback_userdata) ROCPROFILER_API;

/** @} */

ROCPROFILER_EXTERN_C_FINI
