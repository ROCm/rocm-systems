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

#include <rocprofiler/agent.h>
#include <rocprofiler/fwd.h>
#include <rocprofiler/rocprofiler.h>

#include "lib/rocprofiler/hsa/agent.hpp"

#include <vector>

extern "C" {
rocprofiler_status_t
rocprofiler_query_available_agents(rocprofiler_available_agents_cb_t callback,
                                   size_t                            agent_size,
                                   void*                             user_data)
{
    using pc_sampling_config_vec_t = std::vector<rocprofiler_pc_sampling_configuration_t>;

    auto pc_sampling_configs = std::vector<pc_sampling_config_vec_t>{};
    auto get_agents          = [&pc_sampling_configs]() {
        static const auto _default_pc_config =
            rocprofiler_pc_sampling_configuration_t{ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP,
                                                    ROCPROFILER_PC_SAMPLING_UNIT_TIME,
                                                    1UL,
                                                    1000000000UL,
                                                    0};
        auto        temporaries_ = std::vector<rocprofiler_agent_t>{};
        const auto& agent_info   = rocprofiler::hsa::all_agents();
        for(const auto& agent : agent_info)
        {
            auto& _data = pc_sampling_configs.emplace_back();
            if(agent.isGpu()) _data = {_default_pc_config};
            temporaries_.emplace_back(rocprofiler_agent_t{
                .id   = rocprofiler_agent_id_t{.handle = temporaries_.size()},
                .type = (agent.isCpu() ? ROCPROFILER_AGENT_TYPE_CPU
                                                : (agent.isGpu() ? ROCPROFILER_AGENT_TYPE_GPU
                                                                 : ROCPROFILER_AGENT_TYPE_NONE)),
                .name = agent.getNameChar(),
                .pc_sampling_configs =
                    rocprofiler_pc_sampling_config_array_t{_data.data(), _data.size()}});
        }
        return temporaries_;
    };

    auto agents   = get_agents();
    auto pointers = std::vector<rocprofiler_agent_t*>{};
    pointers.reserve(agents.size());
    for(auto& agent : agents)
    {
        pointers.emplace_back(&agent);
    }

    assert(agent_size <= sizeof(rocprofiler_agent_t) &&
           "rocprofiler_agent_t used by caller is ABI-incompatible with rocprofiler_agent_t in "
           "rocprofiler");
    return callback(pointers.data(), pointers.size(), user_data);
}
}
