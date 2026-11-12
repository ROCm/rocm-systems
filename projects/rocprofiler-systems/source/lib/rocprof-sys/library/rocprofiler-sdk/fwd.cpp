// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/rocprofiler-sdk/fwd.hpp"
#include "core/agent.hpp"
#include "core/agent_manager.hpp"
#include "core/state.hpp"

#include "logger/debug.hpp"

#include <algorithm>
#include <exception>
#include <memory>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <fmt/ranges.h>

#include <utility>
#include <vector>

namespace rocprofsys::rocprofiler_sdk
{
namespace
{
using agent_vec_t = std::vector<std::shared_ptr<agent>>;

rocprofiler_status_t
dimensions_info_callback(rocprofiler_counter_id_t /*id*/,
                         const rocprofiler_record_dimension_info_t* dim_info,
                         long unsigned int num_dims, void* user_data)
{
    auto* dimensions_info =
        static_cast<std::vector<rocprofiler_record_dimension_info_t>*>(user_data);
    dimensions_info->reserve(num_dims);
    for(size_t j = 0; j < num_dims; j++)
    {
        dimensions_info->emplace_back(dim_info[j]);
    }

    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
// NOLINTNEXTLINE(misc-const-correctness) - signature must match
// rocprofiler_available_counters_cb_t exactly
counters_supported_callback(rocprofiler_agent_id_t    agent_id,
                            rocprofiler_counter_id_t* counters, size_t num_counters,
                            void* user_data)
{
    using value_type = agent_counter_info_map_t::mapped_type;

    auto* data_v = static_cast<agent_counter_info_map_t*>(user_data);
    data_v->emplace(agent_id, value_type{});
    for(size_t i = 0; i < num_counters; ++i)
    {
        auto _info     = rocprofiler_counter_info_v0_t{};
        auto _dim_info = std::vector<rocprofiler_record_dimension_info_t>{};

        ROCPROFILER_CALL(rocprofiler_query_counter_info(
            counters[i], ROCPROFILER_COUNTER_INFO_VERSION_0, &_info));

        // populate local vector
        ROCPROFILER_CALL(rocprofiler_iterate_counter_dimensions(
            counters[i], dimensions_info_callback, &_dim_info));

        if(!_info.is_constant)
        {
            data_v->at(agent_id).emplace_back(agent_id, _info, std::move(_dim_info));
        }
    }
    return ROCPROFILER_STATUS_SUCCESS;
}

void
sort_counter_info(std::vector<rocprofiler_tool_counter_info_t>& counters)
{
    std::ranges::sort(counters, [](const auto& lhs, const auto& rhs) {
        return lhs.id.handle < rhs.id.handle;
    });

    for(auto& counter : counters)
    {
        std::ranges::sort(counter.dimension_info, [](const auto& lhs, const auto& rhs) {
            return lhs.id < rhs.id;
        });
    }
}

agent_counter_info_map_t
get_agent_counter_info(const agent_vec_t& agents)
{
    auto data = agent_counter_info_map_t{};

    for(const auto& itr : agents)
    {
        const auto agent_id = rocprofiler_agent_id_t{ itr->handle };

        auto status = rocprofiler_iterate_agent_supported_counters(
            agent_id, counters_supported_callback, &data);

        if(status != ROCPROFILER_STATUS_SUCCESS)
        {
            LOG_WARNING(
                "rocprofiler_iterate_agent_supported_counters failed for agent {} "
                "with status {} (Agent HW architecture may not be supported)",
                agent_id.handle, static_cast<int>(status));
            // Skip processing for this agent if it's not supported
            continue;
        }

        // Only process if the agent was successfully added to the map
        auto const agent_it = data.find(agent_id);
        if(agent_it == data.end())
        {
            continue;
        }

        sort_counter_info(agent_it->second);
    }

    return data;
}
}  // namespace

rocprofiler_tool_counter_info_t::rocprofiler_tool_counter_info_t(
    rocprofiler_agent_id_t _agent_id, parent_type _info, dimension_info_vec_t&& _dim_info)
: parent_type{ _info }
, agent_id{ _agent_id }
, dimension_info{ std::move(_dim_info) }
{}

void
client_data::initialize_counter_info()
{
    auto const gpu_agents =
        get_agent_manager_instance().get_agents_by_type(agent_type::gpu);

    if(agent_counter_info.size() != gpu_agents.size())
    {
        agent_counter_info = get_agent_counter_info(gpu_agents);
    }
}
}  // namespace rocprofsys::rocprofiler_sdk
