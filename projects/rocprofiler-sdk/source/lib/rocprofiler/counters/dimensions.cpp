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

#include "dimensions.hpp"

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include <fmt/core.h>

#include "lib/common/synchronized.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler/aql/helpers.hpp"
#include "lib/rocprofiler/hsa/queue_controller.hpp"

namespace rocprofiler
{
namespace counters
{
// namespace
// {
// void
// create_block_dimensions(std::string&                  agent,
//                         std::string&                  block_name,
//                         std::vector<MetricDimension>& dimension_list)
// {
//     static std::atomic<uint64_t> id = 0;
//     // query hsa/aqlprofile/kfd etc here to get dimension sizes
//     // create MetricDimension objects and push_back() in dimension_list
// }

// }  // namespace

// BlockDimensionMap&
// getBlockDimensionsMap(std::string& agent)
// {
//     static std::unique_ptr<BlockDimensionMap> map = [&]() {
//         auto data = std::make_unique<BlockDimensionMap>();
//         // TODO: populate this vector with list of all blocks
//         std::vector<std::string> block_names;

//         for(auto& block : block_names)
//         {
//             auto& dimensions = data->emplace(block,
//             std::vector<MetricDimension>()).first->second; create_block_dimensions(agent, block,
//             dimensions);
//         }
//         return data;
//     }();
//     return *map;
// }

// const AgentBlockDimensionsMap&
// getAgentBlockDimensionsMap()
// {
//     static std::unique_ptr<AgentBlockDimensionsMap> map = [&]() {
//         auto data = std::make_unique<AgentBlockDimensionsMap>();
//         // TODO: fill this up with agent iteration or through xml
//         std::vector<std::string> agent_names;

//         // insert the BlockDimensionMap for each agent
//         for(auto& agent : agent_names)
//         {
//             auto& val = getBlockDimensionsMap(agent);
//             data->emplace(agent, val);
//         }
//         return data;
//     }();
//     return *map;
// }

std::vector<MetricDimension>
getBlockDimensions(const std::string& agent, const Metric& metric)
{
    if(!metric.special().empty())
    {
        // Special non-hardware counters without dimension data
        return std::vector<MetricDimension>{
            {dimension_map().at(ROCPROFILER_DIMENSION_NONE), 1, ROCPROFILER_DIMENSION_NONE}};
    }

    for(const auto& [_, maybe_agent] : hsa::get_queue_controller().get_supported_agents())
    {
        if(maybe_agent.name() == agent)
        {
            return std::vector<MetricDimension>{
                {dimension_map().at(ROCPROFILER_DIMENSION_SHADER_ENGINE),
                 maybe_agent.get_rocp_agent()->num_shader_banks,
                 ROCPROFILER_DIMENSION_SHADER_ENGINE},
                {dimension_map().at(ROCPROFILER_DIMENSION_XCC),
                 maybe_agent.get_rocp_agent()->num_xcc,
                 ROCPROFILER_DIMENSION_XCC},
                {dimension_map().at(ROCPROFILER_DIMENSION_CU),
                 maybe_agent.get_rocp_agent()->cu_count,
                 ROCPROFILER_DIMENSION_CU},
                {dimension_map().at(ROCPROFILER_DIMENSION_AGENT),
                 maybe_agent.get_rocp_agent()->id.handle,
                 ROCPROFILER_DIMENSION_AGENT}};

            // auto query_info = aql::get_query_info(maybe_agent.get_agent(), metric);
            // return std::vector<MetricDimension>{
            //     {metric.block(), query_info.instance_count, ROCPROFILER_DIMENSION_NONE}};
        }
    }

    return {};
}

}  // namespace counters
}  // namespace rocprofiler
