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


#include "lib/rocprofiler-sdk/spm/core.hpp"
#include "lib/rocprofiler-sdk/spm/asynchandler.hpp"

namespace rocprofiler
{
namespace spm
{

    /**
 * @brief The functions checks if the `ROCPROFILER_SPM_BETA_ENABLED` is set.
 * If so, it will enable SPM service. Otherwise, the API is reported
 * as not implemented.
 *
 * The SPM is in experimental phase .
   By enabling the `ROCPROFILER_SPM_BETA_ENABLED`,
 * user accepts all consequences of using early implementation of SPM API.
 */
bool
is_spm_explicitly_enabled()
{
    auto spm_sampling_enabled = rocprofiler::common::get_env("ROCPROFILER_SPM_BETA_ENABLED", false);

    if(!spm_sampling_enabled)
        ROCP_INFO << " SPM unavailable. The feature is implicitly disabled. "
                  << "To use it on a supported architecture, "
                  << "set ROCPROFILER_SPM_BETA_ENABLED=ON in the environment";

    return spm_sampling_enabled;
}

bool asynchandler(rocprofiler_agent_id_t agent_id)
{
    spm_callback_data  *callback_data = nullptr;

    spm_get_controller()._callback_data.wlock([&](auto& map)
    {
        auto it = map.find(agent_id.handle);
        callback_data = it->second.front().get();
    });

    if(!callback_data->is_profiling) return false; 

    if(callback_data->config_switch == true)
    {
        spm_get_controller()._agent_state_map.wlock([&](auto& map) {
            auto it = map.find(agent_id.handle);
            ROCP_FATAL_IF(it == map.end())
                << "agent state map does not have an entnry for agent in async handler";
            auto& state_queue = it->second;
            if(state_queue.size() == 1)
                state_queue.front()->spm_packet->kfd_start();
            else
            {
                auto rel_pkt = std::move(state_queue.front()->spm_packet);
                rel_pkt->kfd_stop();
                state_queue.erase(state_queue.begin());
                ROCP_FATAL_IF(state_queue.empty())
                    << "agent state map has no entry for context switch async handler";
                state_queue.front()->spm_packet->kfd_start();
            }
        });
    }
    
    return false;
}

}
}