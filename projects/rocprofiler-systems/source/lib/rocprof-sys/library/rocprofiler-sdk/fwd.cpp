// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/rocprofiler-sdk/fwd.hpp"
#include "core/state.hpp"

#include "logger/debug.hpp"

#include <algorithm>
#include <exception>
#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/cxx/name_info.hpp>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <fmt/ranges.h>

#include <utility>

namespace rocprofsys::rocprofiler_sdk
{
namespace
{
using tool_agent_vec_t = std::vector<tool_agent>;

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

agent_counter_info_map_t
get_agent_counter_info(const tool_agent_vec_t& _agents)
{
    auto _data = agent_counter_info_map_t{};

    for(auto const itr : _agents)
    {
        const auto& _agent_id = rocprofiler_agent_id_t{ itr.agent->handle };

        auto status = rocprofiler_iterate_agent_supported_counters(
            _agent_id, counters_supported_callback, &_data);

        if(status != ROCPROFILER_STATUS_SUCCESS)
        {
            LOG_WARNING(
                "rocprofiler_iterate_agent_supported_counters failed for agent {} "
                "with status {} (Agent HW architecture may not be supported)",
                _agent_id.handle, static_cast<int>(status));
            // Skip processing for this agent if it's not supported
            continue;
        }

        // Only process if the agent was successfully added to the map
        auto const agent_it = _data.find(_agent_id);
        if(agent_it != _data.end())
        {
            std::sort(agent_it->second.begin(), agent_it->second.end(),
                      [](const auto& lhs, const auto& rhs) {
                          return (lhs.id.handle < rhs.id.handle);
                      });

            for(auto& citr : agent_it->second)
            {
                std::ranges::sort(
                    citr.dimension_info,
                    [](const auto& lhs, const auto& rhs) { return (lhs.id < rhs.id); });
            }
        }
    }

    return _data;
}
}  // namespace

rocprofiler_tool_counter_info_t::rocprofiler_tool_counter_info_t(
    rocprofiler_agent_id_t _agent_id, parent_type _info, dimension_info_vec_t&& _dim_info)
: parent_type{ _info }
, agent_id{ _agent_id }
, dimension_info{ std::move(_dim_info) }
{}

void
client_data::initialize()
{
    set_agents();
}

void
client_data::initialize_counter_info()
{
    auto const& agent_mngr = get_agent_manager_instance();

    if(agent_mngr.get_agents().empty())
    {
        initialize();
    }
    else if(gpu_agents.empty() && cpu_agents.empty())
    {
        set_agents();
    }

    if(agent_counter_info.size() != gpu_agents.size())
    {
        agent_counter_info = get_agent_counter_info(gpu_agents);
    }
}

void
client_data::set_agents()
{
    auto& agent_mngr = get_agent_manager_instance();

    auto const fill_agents = [&](agent_type type, std::vector<tool_agent>& out) {
        const auto& _agents = agent_mngr.get_agents_by_type(type);
        for(const auto& agent : _agents)
        {
            out.emplace_back(tool_agent{ .device_id = agent->device_type_index,
                                         .agent     = agent.get() });
        }
    };

    fill_agents(agent_type::gpu, gpu_agents);
    fill_agents(agent_type::cpu, cpu_agents);
}
}  // namespace rocprofsys::rocprofiler_sdk
