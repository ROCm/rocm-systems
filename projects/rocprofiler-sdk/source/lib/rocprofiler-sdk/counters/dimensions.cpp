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

#include "dimensions.hpp"

#include "lib/common/static_object.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/aql/helpers.hpp"
#include "lib/rocprofiler-sdk/aql/packet_construct.hpp"
#include "lib/rocprofiler-sdk/counters/evaluate_ast.hpp"
#include "lib/rocprofiler-sdk/counters/metrics.hpp"

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/rocprofiler.h>
#include <rocprofiler-sdk/cxx/hash.hpp>
#include <rocprofiler-sdk/cxx/operators.hpp>

#include <fmt/core.h>

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace rocprofiler
{
namespace counters
{
std::vector<MetricDimension>
getBlockDimensions(rocprofiler_agent_id_t agent_id, const Metric& metric)
{
    if(!metric.constant().empty())
    {
        // Special non-hardware counters without dimension data
        return std::vector<MetricDimension>{{dimension_map().at(ROCPROFILER_DIMENSION_INSTANCE),
                                             1,
                                             ROCPROFILER_DIMENSION_INSTANCE}};
    }

    std::unordered_map<rocprofiler_profile_counter_instance_types, uint64_t> count;

    std::vector<MetricDimension> ret;

    aql::CounterPacketConstruct pkt_gen(agent_id, {metric});
    const auto&                 events = pkt_gen.get_counter_events(metric);

    for(const auto& event : events)
    {
        auto dims   = std::map<int, uint64_t>{};
        auto status = aql::get_dim_info(agent_id, event, 0, dims);
        CHECK_EQ(status, ROCPROFILER_STATUS_SUCCESS) << rocprofiler_get_status_string(status);

        for(const auto& [id, extent] : dims)
        {
            if(const auto* inst_type =
                   rocprofiler::common::get_val(aqlprofile_id_to_rocprof_instance(), id))
            {
                count.emplace(*inst_type, 0).first->second = extent;
            }
            else
            {
                ROCP_WARNING << "Unknown AQL Profiler Dimension " << id << " " << extent;
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

namespace
{
metric_dims
generate_dimensions(rocprofiler_agent_id_t agent_id, bool spm = false)
{
    std::unordered_map<uint64_t, std::vector<MetricDimension>> dims;

    // Get the agent to determine which architecture's metrics to load
    const auto* agent = rocprofiler::agent::get_agent(agent_id);
    if(!agent) return {.id_to_dim = dims};

    const auto  asts = counters::get_ast_map();
    const auto* arch_asts =
        rocprofiler::common::get_val(asts->arch_to_counter_asts, std::string(agent->name));
    if(!arch_asts) return {.id_to_dim = dims};

    auto        metrics_map = spm ? counters::loadMetrics() : nullptr;
    const auto* id_map      = metrics_map ? &metrics_map->id_to_metric : nullptr;

    for(const auto& [metric, ast] : *arch_asts)
    {
        auto ast_copy = ast;
        try
        {
            // Generate dimensions for this specific agent
            auto metric_dims_v = ast_copy.set_dimensions(agent_id);

            if(spm && id_map)
            {
                auto base_metric_id    = counters::get_base_metric_from_counter_id(ast.out_id());
                const auto* metric_ptr = rocprofiler::common::get_val(*id_map, base_metric_id);
                if(metric_ptr && isSupportSpm(*metric_ptr, agent_id))
                {
                    if(!metric_ptr->expression().empty())
                    {
                        metric_dims_v.clear();
                        metric_dims_v.emplace_back(dimension_map().at(ROCPROFILER_DIMENSION_XCC),
                                                   agent->num_xcc,
                                                   ROCPROFILER_DIMENSION_XCC);
                    }
                    dims.emplace(ast.out_id().handle, std::move(metric_dims_v));
                }
            }
            else
            {
                dims.emplace(ast.out_id().handle, std::move(metric_dims_v));
            }
        } catch(std::runtime_error& e)
        {
            ROCP_FATAL << metric << " has improper dimensions"
                       << " " << e.what();
        }
    }
    return {.id_to_dim = dims};
}
}  // namespace

std::shared_ptr<const metric_dims>
get_dimension_cache(rocprofiler_agent_id_t agent_id, bool reload, bool spm)
{
    using DimSync = common::Synchronized<
        std::unordered_map<rocprofiler_agent_id_t, std::shared_ptr<const metric_dims>>>;
    struct pmc_dim_tag
    {};
    struct spm_dim_tag
    {};
    static DimSync*& pmc_dim_data = common::static_object<DimSync, pmc_dim_tag>::construct();
    static DimSync*& spm_dim_data = common::static_object<DimSync, spm_dim_tag>::construct();

    auto*& dim_data = spm ? spm_dim_data : pmc_dim_data;

    if(!dim_data) return nullptr;

    // Check if we need to generate (first time or reload)
    auto needs_generation = dim_data->rlock([agent_id, reload](const auto& data) {
        return reload || !rocprofiler::common::get_val(data, agent_id);
    });

    if(needs_generation)
    {
        return dim_data->wlock([agent_id, spm](auto& data) -> std::shared_ptr<const metric_dims> {
            auto new_dims = std::make_shared<const metric_dims>(generate_dimensions(agent_id, spm));
            data[agent_id] = new_dims;
            return new_dims;
        });
    }

    return dim_data->rlock([agent_id](const auto& data) -> std::shared_ptr<const metric_dims> {
        if(const auto* ptr = rocprofiler::common::get_val(data, agent_id))
        {
            return *ptr;
        }
        return nullptr;
    });
}

}  // namespace counters
}  // namespace rocprofiler
