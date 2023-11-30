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
#include "lib/rocprofiler-sdk/aql/helpers.hpp"
#include "lib/rocprofiler-sdk/aql/packet_construct.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"

namespace rocprofiler
{
namespace counters
{
std::vector<MetricDimension>
getBlockDimensions(std::string_view agent, const Metric& metric)
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
            // To be returned when instance counting is functional with AQL profiler
            // return std::vector<MetricDimension>{
            //     {dimension_map().at(ROCPROFILER_DIMENSION_SHADER_ENGINE),
            //      maybe_agent.get_rocp_agent()->num_shader_banks,
            //      ROCPROFILER_DIMENSION_SHADER_ENGINE},
            //     {dimension_map().at(ROCPROFILER_DIMENSION_XCC),
            //      maybe_agent.get_rocp_agent()->num_xcc,
            //      ROCPROFILER_DIMENSION_XCC},
            //     {dimension_map().at(ROCPROFILER_DIMENSION_CU),
            //      maybe_agent.get_rocp_agent()->cu_count,
            //      ROCPROFILER_DIMENSION_CU},
            //     {dimension_map().at(ROCPROFILER_DIMENSION_AGENT),
            //      maybe_agent.get_rocp_agent()->id.handle,
            //      ROCPROFILER_DIMENSION_AGENT}};
            aql::AQLPacketConstruct pkt_gen(maybe_agent, {metric});
            return std::vector<MetricDimension>{
                {metric.block(), pkt_gen.get_all_events().size(), ROCPROFILER_DIMENSION_NONE}};
        }
    }

    return {};
}

}  // namespace counters
}  // namespace rocprofiler
