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

#include "common.hpp"

#include <iostream>
#include <sstream>
#include <stdexcept>

int
start()
{
    return 1;
}

rocprofiler_context_id_t&
get_client_ctx()
{
    static rocprofiler_context_id_t ctx{0};
    return ctx;
}

std::vector<rocprofiler_agent_v0_t>
get_gpu_device_agents()
{
    std::vector<rocprofiler_agent_v0_t> agents;

    rocprofiler_query_available_agents_cb_t iterate_cb = [](rocprofiler_agent_version_t agents_ver,
                                                            const void**                agents_arr,
                                                            size_t                      num_agents,
                                                            void*                       udata) {
        if(agents_ver != ROCPROFILER_AGENT_INFO_VERSION_0)
            throw std::runtime_error{"unexpected rocprofiler agent version"};
        auto* agents_v = static_cast<std::vector<rocprofiler_agent_v0_t>*>(udata);
        for(size_t i = 0; i < num_agents; ++i)
        {
            const auto* agent = static_cast<const rocprofiler_agent_v0_t*>(agents_arr[i]);
            if(agent->type == ROCPROFILER_AGENT_TYPE_GPU) agents_v->emplace_back(*agent);
        }
        return ROCPROFILER_STATUS_SUCCESS;
    };

    ROCPROFILER_CALL(
        rocprofiler_query_available_agents(ROCPROFILER_AGENT_INFO_VERSION_0,
                                           iterate_cb,
                                           sizeof(rocprofiler_agent_t),
                                           const_cast<void*>(static_cast<const void*>(&agents))),
        "query available agents");
    return agents;
}

std::vector<rocprofiler_counter_id_t>
get_matched_spm_counters(rocprofiler_agent_id_t       agent_id,
                         const std::set<std::string>& counters_to_collect)
{
    std::vector<rocprofiler_counter_id_t> gpu_counters;

    ROCPROFILER_CALL(rocprofiler_spm_iterate_agent_supported_counters(
                         agent_id,
                         [](rocprofiler_agent_id_t,
                            rocprofiler_counter_id_t* counters,
                            size_t                    num_counters,
                            void*                     user_data) {
                             auto* vec =
                                 static_cast<std::vector<rocprofiler_counter_id_t>*>(user_data);
                             for(size_t i = 0; i < num_counters; i++)
                             {
                                 vec->push_back(counters[i]);
                             }
                             return ROCPROFILER_STATUS_SUCCESS;
                         },
                         static_cast<void*>(&gpu_counters)),
                     "Could not fetch supported counters");

    std::vector<rocprofiler_counter_id_t> collect_counters;
    for(auto& counter : gpu_counters)
    {
        rocprofiler_counter_info_v0_t info;
        ROCPROFILER_CALL(
            rocprofiler_query_counter_info(
                counter, ROCPROFILER_COUNTER_INFO_VERSION_0, static_cast<void*>(&info)),
            "Could not query info for counter");
        if(counters_to_collect.count(std::string(info.name)) > 0)
        {
            std::clog << "Counter: " << counter.handle << " " << info.name << "\n";
            collect_counters.push_back(counter);
        }
    }

    return collect_counters;
}

rocprofiler_counter_config_id_t
create_spm_counter_config(rocprofiler_agent_id_t                    agent_id,
                          std::vector<rocprofiler_counter_id_t>     counters,
                          std::vector<rocprofiler_spm_parameters_t> params)
{
    rocprofiler_counter_config_id_t            profile = {.handle = 0};
    std::vector<rocprofiler_spm_parameters_t*> input_params{};
    for(auto& param : params)
    {
        input_params.push_back(&param);
    }

    ROCPROFILER_CALL(rocprofiler_spm_create_counter_config(agent_id,
                                                           counters.data(),
                                                           counters.size(),
                                                           input_params.data(),
                                                           input_params.size(),
                                                           &profile),
                     "Could not construct profile cfg");
    return profile;
}
