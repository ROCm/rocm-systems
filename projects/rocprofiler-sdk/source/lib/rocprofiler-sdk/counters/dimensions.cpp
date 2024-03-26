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

#include "lib/common/static_object.hpp"
#include "lib/common/synchronized.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/aql/helpers.hpp"
#include "lib/rocprofiler-sdk/aql/packet_construct.hpp"
#include "lib/rocprofiler-sdk/counters/evaluate_ast.hpp"
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
        return std::vector<MetricDimension>{{dimension_map().at(ROCPROFILER_DIMENSION_INSTANCE),
                                             1,
                                             ROCPROFILER_DIMENSION_INSTANCE}};
    }

    std::unordered_map<rocprofiler_profile_counter_instance_types, uint64_t> count;

    std::vector<MetricDimension> ret;

    for(const auto& [_, maybe_agent] :
        CHECK_NOTNULL(hsa::get_queue_controller())->get_supported_agents())
    {
        if(maybe_agent.name() == agent)
        {
            aql::AQLPacketConstruct pkt_gen(maybe_agent, {metric});
            const auto&             events = pkt_gen.get_counter_events(metric);

            for(const auto& event : events)
            {
                std::map<int, uint64_t> dims;
                auto status = aql::get_dim_info(maybe_agent.get_hsa_agent(), event, 0, dims);
                CHECK_EQ(status, ROCPROFILER_STATUS_SUCCESS)
                    << rocprofiler_get_status_string(status);

                for(const auto& [id, extent] : dims)
                {
                    if(const auto* inst_type =
                           rocprofiler::common::get_val(aqlprofile_id_to_rocprof_instance(), id))
                    {
                        count.emplace(*inst_type, 0).first->second = extent;
                    }
                    else
                    {
                        LOG(ERROR) << "Unknown AQL Profiler Dimension " << id << " " << extent;
                    }
                }
            }
        }
    }

    ret.reserve(count.size());
    for(const auto& [dim, size] : count)
    {
        ret.emplace_back(dimension_map().at(dim), size, dim);
    }

    return ret;
}

const std::unordered_map<uint64_t, std::vector<MetricDimension>>&
get_dimension_cache()
{
    static auto*& cache =
        common::static_object<std::unordered_map<uint64_t, std::vector<MetricDimension>>>::
            construct([]() -> std::unordered_map<uint64_t, std::vector<MetricDimension>> {
                std::unordered_map<uint64_t, std::vector<MetricDimension>> dims;
                /**
                 * Fails if HSA is not loaded by retruning nothing. This should not remain after
                 * AQL is transistioned away from HSA.
                 */
                if(CHECK_NOTNULL(rocprofiler::hsa::get_queue_controller())
                       ->get_supported_agents()
                       .empty())
                {
                    return {};
                }

                const auto& asts = counters::get_ast_map();
                for(const auto& [gfx, metrics] : asts)
                {
                    for(const auto& [metric, ast] : metrics)
                    {
                        auto ast_copy = ast;
                        try
                        {
                            dims.emplace(ast.out_id().handle, ast_copy.set_dimensions());
                        } catch(std::runtime_error& e)
                        {
                            LOG(ERROR) << metric << " has improper dimensions"
                                       << " " << e.what();
                            throw;
                        }
                    }
                }
                return dims;
            }());
    return *cache;
}

}  // namespace counters
}  // namespace rocprofiler
