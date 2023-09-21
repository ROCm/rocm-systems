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

/**
 * @defgroup PROFILE_CONFIG Profile Configurations
 * @brief Group one or more hardware counters into a unique handle
 *
 * @{
 */

/**
 * @brief Create Profile Configuration.
 *
 * @param [in] agent Agent identifier
 * @param [in] counters_list List of GPU counters
 * @param [in] counters_count Size of counters list
 * @param [out] config_id Identifier for GPU counters group
 * @return ::rocprofiler_status_t
 */
rocprofiler_status_t ROCPROFILER_API
rocprofiler_create_profile_config(rocprofiler_agent_t              agent,
                                  rocprofiler_counter_id_t*        counters_list,
                                  size_t                           counters_count,
                                  rocprofiler_profile_config_id_t* config_id)
    ROCPROFILER_NONNULL(4);

/**
 * @brief Destroy Profile Configuration.
 *
 * @param [in] config_id
 * @return ::rocprofiler_status_t
 */
rocprofiler_status_t ROCPROFILER_API
rocprofiler_destroy_profile_config(rocprofiler_profile_config_id_t config_id);

/** @} */

ROCPROFILER_EXTERN_C_FINI
