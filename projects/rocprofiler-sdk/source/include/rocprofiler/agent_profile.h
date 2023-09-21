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

ROCPROFILER_EXTERN_C_INIT

/**
 * @defgroup AGENT_PROFILE_COUNTING_SERVICE Agent Profile Counting Service
 * @brief needs brief description
 *
 * @{
 */

/**
 * @brief ROCProfiler Agent Profile Counting Data.
 *
 * Counters, including identifiers to get counter information and Counters values
 */
typedef struct
{
    /**
     */
    rocprofiler_record_counter_t* counters;
    uint64_t                      counters_count;
} rocprofiler_agent_profile_counting_data_t;

/**
 * @brief Configure Profile Counting Service for agent.
 *
 * @param [in] buffer_id
 * @param [in] profile_config_id
 * @return ::rocprofiler_status_t
 */
rocprofiler_status_t ROCPROFILER_API
rocprofiler_configure_agent_profile_counting_service(
    rocprofiler_buffer_id_t         buffer_id,
    rocprofiler_profile_config_id_t profile_config_id);

/**
 * @brief Sample Profile Counting Service for agent.
 *
 * @param [out] data // It is always a size of one
 * @return ::rocprofiler_status_t
 */
rocprofiler_status_t ROCPROFILER_API
rocprofiler_sample_agent_profile_counting_service(rocprofiler_agent_profile_counting_data_t* data);

/** @} */

ROCPROFILER_EXTERN_C_FINI
