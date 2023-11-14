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
    rocprofiler_record_counter_t*       records,
    size_t                              record_count,
    rocprofiler_profile_config_id_t     config);

/**
 * @brief Configure Dispatch Profile Counting Service.
 *
 * @param [in] context_id context id
 * @param [in] profile profile config to use for dispatch
 * @param [in] callback callback
 * @param [in] callback_data_args callback data
 * @return ::rocprofiler_status_t
 */
rocprofiler_status_t ROCPROFILER_API
rocprofiler_configure_dispatch_profile_counting_service(
    rocprofiler_context_id_t                         context_id,
    rocprofiler_profile_config_id_t                  profile,
    rocprofiler_profile_counting_dispatch_callback_t callback,
    void*                                            callback_data_args);

/** @} */

ROCPROFILER_EXTERN_C_FINI
