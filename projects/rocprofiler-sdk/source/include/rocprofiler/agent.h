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
 * @defgroup AGENTS Agent Information
 * @brief needs brief description
 *
 * @{
 */

/**
 * @brief Agent.
 */
typedef struct
{
    rocprofiler_agent_id_t                 id;
    rocprofiler_agent_type_t               type;
    const char*                            name;
    rocprofiler_pc_sampling_config_array_t pc_sampling_configs;
} rocprofiler_agent_t;

/**
 * @brief Callback function type for querying the available agents
 *
 * @param [in] agents Array of pointers to agents
 * @param [in] num_agents Number of agents in array
 * @param [in] user_data Data pointer passback
 * @return ::rocprofiler_status_t
 */
typedef rocprofiler_status_t (*rocprofiler_available_agents_cb_t)(rocprofiler_agent_t** agents,
                                                                  size_t                num_agents,
                                                                  void*                 user_data);

/**
 * @brief Receive synchronous callback with an array of available agents at moment of invocation
 *
 * @param [in] callback Callback function accepting list of agents
 * @param [in] agent_size Should be set to sizeof(rocprofiler_agent_t)
 * @param [in] user_data Data pointer provided to callback
 * @return ::rocprofiler_status_t
 */
rocprofiler_status_t ROCPROFILER_API
rocprofiler_query_available_agents(rocprofiler_available_agents_cb_t callback,
                                   size_t                            agent_size,
                                   void* user_data) ROCPROFILER_NONNULL(1);

/** @} */

ROCPROFILER_EXTERN_C_FINI
