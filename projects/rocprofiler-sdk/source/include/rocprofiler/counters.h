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

#include <rocprofiler/agent.h>
#include <rocprofiler/defines.h>
#include <rocprofiler/fwd.h>

ROCPROFILER_EXTERN_C_INIT

/** @defgroup COUNTERS Hardware counters
 * @{
 */

/**
 * @brief Query Counter name.
 *
 * @param [in] counter_id
 * @param [out] name if nullptr, size will be returned
 * @param [out] size
 * @return ::rocprofiler_status_t
 */
rocprofiler_status_t ROCPROFILER_API
rocprofiler_query_counter_name(rocprofiler_counter_id_t counter_id, const char* name, size_t* size)
    ROCPROFILER_NONNULL(3);

/**
 * @brief Query Counter Instances Count.
 *
 * @param [in] counter_id
 * @param [out] instance_count
 * @return rocprofiler_status_t
 */
rocprofiler_status_t ROCPROFILER_API
rocprofiler_query_counter_instance_count(rocprofiler_counter_id_t counter_id,
                                         size_t* instance_count) ROCPROFILER_NONNULL(2);

/**
 * @brief Query Agent Counters Availability.
 *
 * @param [in] agent
 * @param [out] counters_list
 * @param [out] counters_count
 * @return ::rocprofiler_status_t
 */
rocprofiler_status_t ROCPROFILER_API
rocprofiler_query_agent_supported_counters(rocprofiler_agent_t       agent,
                                           rocprofiler_counter_id_t* counters_list,
                                           size_t* counters_count) ROCPROFILER_NONNULL(2, 3);

/** @} */

ROCPROFILER_EXTERN_C_FINI
