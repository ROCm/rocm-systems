// MIT License
//
// Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "lib/rocprofiler-sdk/agent_mapping.hpp"

namespace rocprofiler
{
namespace agent
{
namespace
{
bool
keys_match(const mapping_agent_view& rocp_agent, uint32_t driver_node_id, mapping_policy policy)
{
    if(policy == mapping_policy::wsl) return rocp_agent.node_id == driver_node_id;
    return rocp_agent.logical_node_id == static_cast<int64_t>(driver_node_id);
}
}  // namespace

agent_mapping_result
compute_agent_mapping(const std::vector<mapping_agent_view>& rocp_agents,
                      const std::vector<mapping_hsa_view>&   hsa_agents,
                      mapping_policy                         policy)
{
    auto result  = agent_mapping_result{};
    auto claimed = std::vector<bool>(rocp_agents.size(), false);

    for(size_t hidx = 0; hidx < hsa_agents.size(); ++hidx)
    {
        const auto& hsa_agent = hsa_agents.at(hidx);

        if(!hsa_agent.has_node_id)
        {
            ++result.unqueryable_count;
            continue;
        }

        auto matched = false;
        for(size_t ridx = 0; ridx < rocp_agents.size(); ++ridx)
        {
            if(claimed.at(ridx)) continue;
            if(!keys_match(rocp_agents.at(ridx), hsa_agent.driver_node_id, policy)) continue;

            claimed.at(ridx) = true;
            result.pairs.emplace_back(
                agent_mapping_result::pair{ridx, hidx, hsa_agent.driver_node_id});
            matched = true;
            break;
        }

        if(!matched)
        {
            if(hsa_agent.is_gpu)
                result.unmatched_gpu_node_ids.emplace_back(hsa_agent.driver_node_id);
            else
                result.unmatched_other_node_ids.emplace_back(hsa_agent.driver_node_id);
        }
    }

    return result;
}
}  // namespace agent
}  // namespace rocprofiler
