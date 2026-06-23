// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <rocprofiler-sdk/experimental/spm.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <set>
#include <string>
#include <vector>

#define CLIENT_API __attribute__((visibility("default")))

#define ROCPROFILER_CALL(result, msg)                                                              \
    {                                                                                              \
        rocprofiler_status_t CHECKSTATUS = result;                                                 \
        if(CHECKSTATUS != ROCPROFILER_STATUS_SUCCESS)                                              \
        {                                                                                          \
            std::string status_msg = rocprofiler_get_status_string(CHECKSTATUS);                   \
            std::cerr << "[" #result "][" << __FILE__ << ":" << __LINE__ << "] " << msg            \
                      << " failed with error code " << CHECKSTATUS << ": " << status_msg           \
                      << std::endl;                                                                \
            std::stringstream errmsg{};                                                            \
            errmsg << "[" #result "][" << __FILE__ << ":" << __LINE__ << "] " << msg " failure ("  \
                   << status_msg << ")";                                                           \
            throw std::runtime_error(errmsg.str());                                                \
        }                                                                                          \
    }

int
start() CLIENT_API;

rocprofiler_context_id_t&
get_client_ctx();

/**
 * Returns all GPU agents visible to rocprofiler on the system
 */
std::vector<rocprofiler_agent_v0_t>
get_gpu_device_agents();

/**
 * Iterate all SPM-supported counters on an agent and return those matching
 * the requested counter names.
 */
std::vector<rocprofiler_counter_id_t>
get_matched_spm_counters(rocprofiler_agent_id_t       agent_id,
                         const std::set<std::string>& counters_to_collect);

/**
 * Create an SPM counter config from a set of already-matched counter IDs and
 * sampling parameters.
 */
rocprofiler_counter_config_id_t
create_spm_counter_config(rocprofiler_agent_id_t                    agent_id,
                          std::vector<rocprofiler_counter_id_t>     counters,
                          std::vector<rocprofiler_spm_parameters_t> params);
