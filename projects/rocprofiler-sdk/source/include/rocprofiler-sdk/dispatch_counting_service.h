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
#include <rocprofiler-sdk/counter_config.h>
#include <rocprofiler-sdk/defines.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/hsa.h>

ROCPROFILER_EXTERN_C_INIT

/**
 * @defgroup dispatch_counting_service Dispatch Profile Counting Service
 * @brief Per-dispatch hardware counter collection service
 *
 * @{
 */

/**
 * @brief (experimental) Kernel dispatch data for profile counting callbacks.
 *
 */
typedef struct ROCPROFILER_SDK_EXPERIMENTAL rocprofiler_dispatch_counting_service_data_t
{
    uint64_t                           size;             ///< Size of this struct
    rocprofiler_async_correlation_id_t correlation_id;   ///< Correlation ID for this dispatch
    rocprofiler_timestamp_t            start_timestamp;  ///< start time in nanoseconds
    rocprofiler_timestamp_t            end_timestamp;    ///< end time in nanoseconds
    rocprofiler_kernel_dispatch_info_t dispatch_info;    ///< Dispatch info
} rocprofiler_dispatch_counting_service_data_t;

/**
 * @brief (experimental) ROCProfiler Profile Counting Counter Record Header Information
 *
 * This is buffer equivalent of ::rocprofiler_dispatch_counting_service_data_t
 */
typedef struct ROCPROFILER_SDK_EXPERIMENTAL rocprofiler_dispatch_counting_service_record_t
{
    uint64_t size;         ///< Size of this struct
    uint64_t num_records;  ///< number of ::rocprofiler_counter_record_t records
    rocprofiler_async_correlation_id_t correlation_id;   ///< Correlation ID for this dispatch
    rocprofiler_timestamp_t            start_timestamp;  ///< start time in nanoseconds
    rocprofiler_timestamp_t            end_timestamp;    ///< end time in nanoseconds
    rocprofiler_kernel_dispatch_info_t dispatch_info;    ///< Contains the `dispatch_id`
} rocprofiler_dispatch_counting_service_record_t;

/**
 * @brief (experimental) Kernel Dispatch Callback. This is a callback that is invoked before the
 * kernel is enqueued into the HSA queue. What counters to collect for a kernel are set via passing
 * back a profile config (config) in this callback. These counters will be collected and emplaced in
 * the buffer with ::rocprofiler_buffer_id_t used when setting up this callback.
 *
 * @param [in] dispatch_data      @see ::rocprofiler_dispatch_counting_service_data_t
 * @param [out] config            Profile config detailing the counters to collect for this kernel
 * @param [out] user_data         User data unique to this dispatch. Returned in record callback
 * @param [in] callback_data_args Callback supplied via buffered_dispatch_counting_service
 */
ROCPROFILER_SDK_EXPERIMENTAL
typedef void (*rocprofiler_dispatch_counting_service_cb_t)(
    rocprofiler_dispatch_counting_service_data_t dispatch_data,
    rocprofiler_counter_config_id_t*             config,
    rocprofiler_user_data_t*                     user_data,
    void*                                        callback_data_args);

/**
 * @brief (experimental) Counting record callback. This is a callback is invoked when the kernel
 *        execution is complete and contains the counter profile data requested in
 *        ::rocprofiler_dispatch_counting_service_cb_t. Only used with
 *        ::rocprofiler_configure_callback_dispatch_counting_service.
 *
 * @param [in] dispatch_data      @see ::rocprofiler_dispatch_counting_service_data_t
 * @param [in] record_data        Counter record data.
 * @param [in] record_count       Number of counter records.
 * @param [in] user_data          User data instance from dispatch callback
 * @param [in] callback_data_args Callback supplied via buffered_dispatch_counting_service
 */
ROCPROFILER_SDK_EXPERIMENTAL
typedef void (*rocprofiler_dispatch_counting_record_cb_t)(
    rocprofiler_dispatch_counting_service_data_t dispatch_data,
    rocprofiler_counter_record_t*                record_data,
    size_t                                       record_count,
    rocprofiler_user_data_t                      user_data,
    void*                                        callback_data_args);

/**
 * @brief (experimental) Configure buffered dispatch profile Counting Service.
 *        Collects the counters in dispatch packets and stores them
 *        in a buffer with @p buffer_id. The buffer may contain packets from more than
 *        one dispatch (denoted by correlation id). Will trigger the
 *        callback based on the parameters setup in buffer_id_t.
 *
 *        NOTE: Interface is up for comment as to whether restrictions
 *        on agent should be made here (limiting the CB based on agent)
 *        or if the restriction should be performed by the tool in
 *        ::rocprofiler_dispatch_counting_service_cb_t (i.e.
 *        tool code checking the agent param to see if they want to profile
 *        it).
 *
 *        Interface is up for comment as to whether restrictions
 *        on agent should be made here (limiting the CB based on agent)
 *        or if the restriction should be performed by the tool in
 *        ::rocprofiler_dispatch_counting_service_cb_t (i.e.
 *        tool code checking the agent param to see if they want to profile
 *        it).
 *
 * @param [in] context_id context id
 * @param [in] buffer_id id of the buffer to use for the counting service
 * @param [in] callback callback to perform when dispatch is enqueued
 * @param [in] callback_data_args callback data
 * @return ::rocprofiler_status_t
 */
ROCPROFILER_SDK_EXPERIMENTAL
rocprofiler_status_t
rocprofiler_configure_buffer_dispatch_counting_service(
    rocprofiler_context_id_t                   context_id,
    rocprofiler_buffer_id_t                    buffer_id,
    rocprofiler_dispatch_counting_service_cb_t callback,
    void*                                      callback_data_args) ROCPROFILER_API;

/**
 * @brief (experimental) Configure buffered dispatch profile Counting Service.
 *        Collects the counters in dispatch packets and calls a callback
 *        with the counters collected during that dispatch.
 *
 * @param [in] context_id context id
 * @param [in] dispatch_callback callback to perform when dispatch is enqueued
 * @param [in] dispatch_callback_args callback data for dispatch callback
 * @param [in] record_callback  Record callback for completed profile data
 * @param [in] record_callback_args Callback args for record callback
 * @return ::rocprofiler_status_t
 */
ROCPROFILER_SDK_EXPERIMENTAL
rocprofiler_status_t
rocprofiler_configure_callback_dispatch_counting_service(
    rocprofiler_context_id_t                   context_id,
    rocprofiler_dispatch_counting_service_cb_t dispatch_callback,
    void*                                      dispatch_callback_args,
    rocprofiler_dispatch_counting_record_cb_t  record_callback,
    void*                                      record_callback_args) ROCPROFILER_API;

/** @} */

ROCPROFILER_EXTERN_C_FINI
