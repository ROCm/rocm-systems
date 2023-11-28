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

#include <rocprofiler/agent.h>
#include <rocprofiler/defines.h>
#include <rocprofiler/fwd.h>
#include <rocprofiler/hsa.h>
#include <rocprofiler/profile_config.h>

ROCPROFILER_EXTERN_C_INIT

/**
 * @defgroup DISPATCH_PROFILE_COUNTING_SERVICE Dispatch Profile Counting Service
 * @brief Per-dispatch hardware counter collection service
 *
 * @{
 */

/**
 * @brief Kernel Dispatch Callback. This is a callback that is invoked before the kernel
 *        is enqueued into the HSA queue. What counters to collect for a kernel are set
 *        via passing back a profile config (config) in this callback. These counters
 *        will be collected and emplaced in the buffer with @ref rocprofiler_buffer_id_t used when
 *        setting up this callback.
 *
 * @param [in] queue_id        Queue the kernel dispatch packet is being enqueued onto
 * @param [in] agent           Agent of this queue
 * @param [in] correlation_id  Correlation ID for this dispatch
 * @param [in] dispatch_packet Kernel dispatch packet about to be enqueued into HSA
 * @param [in] callback_data_args Callback supplied via buffered_dispatch_profile_counting_service
 * @param [out] config         Profile config detailing the counters to collect for this kernel
 */
typedef void (*rocprofiler_profile_counting_dispatch_callback_t)(
    rocprofiler_queue_id_t              queue_id,
    const rocprofiler_agent_t*          agent,
    rocprofiler_correlation_id_t        correlation_id,
    const hsa_kernel_dispatch_packet_t* dispatch_packet,
    void*                               callback_data_args,
    rocprofiler_profile_config_id_t*    config);

/**
 * @brief Configure buffered dispatch profile Counting Service.
 *        Collects the counters in dispatch packets and stores them
 *        in a buffer with @p buffer_id. The buffer may contain packets from more than
 *        one dispatch (denoted by correlation id). Will trigger the
 *        callback based on the parameters setup in @p buffer_id.
 *
 * // TODO(aelwazir): Should this be per agent?
 *
 * @param [in] context_id context id
 * @param [in] buffer_id id of the buffer to use for the counting service
 * @param [in] callback callback to perform when dispatch is enqueued
 * @param [in] callback_data_args callback data
 * @return ::rocprofiler_status_t
 */
rocprofiler_status_t ROCPROFILER_API
rocprofiler_configure_buffered_dispatch_profile_counting_service(
    rocprofiler_context_id_t                         context_id,
    rocprofiler_buffer_id_t                          buffer_id,
    rocprofiler_profile_counting_dispatch_callback_t callback,
    void*                                            callback_data_args);
/** @} */

ROCPROFILER_EXTERN_C_FINI
