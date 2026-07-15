// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "agent_manager.hpp"
#include <cstdint>

#include "logger/debug.hpp"

#include <algorithm>
#include <iterator>

namespace rocprofsys
{

agent_manager&
get_agent_manager_instance()
{
    static agent_manager _instance;
    return _instance;
}

agent_manager::agent_manager(std::vector<std::shared_ptr<agent>> agents)
: _agents(std::move(agents))
{}

void
agent_manager::insert_agent(agent& _agent)
{
    _agent.device_type_index = _agent_counts[_agent.type]++;
    _agents.emplace_back(std::make_shared<agent>(_agent));

    LOG_TRACE("Inserting agent with device handle: {}, and agent id: {}, device type: {}",
              _agent.device_id, _agent.device_type_index, to_string(_agent.type));
}

const agent&
agent_manager::get_agent_by_type_index(size_t type_index, agent_type type) const
{
    auto _agent =
        std::find_if(_agents.begin(), _agents.end(), [&](const auto& agent_ptr) {
            return agent_ptr->type == type && agent_ptr->device_type_index == type_index;
        });
    if(_agent == _agents.end())
    {
        throw std::out_of_range(fmt::format(
            "Agent not found for type index: {}, type: {}", type_index, to_string(type)));
    }
    return **_agent;
}

const agent&
agent_manager::get_agent_by_id(size_t device_id, agent_type type) const
{
    LOG_TRACE("Getting agent for device id: {}, type {}", device_id, to_string(type));
    auto _agent =
        std::find_if(_agents.begin(), _agents.end(), [&](const auto& agent_ptr) {
            return agent_ptr->type == type && agent_ptr->device_id == device_id;
        });
    if(_agent == _agents.end())
    {
        throw std::out_of_range(fmt::format("Agent not found for device id: {}, type: {}",
                                            device_id, to_string(type)));
    }
    return **_agent;
}

const agent&
agent_manager::get_agent_by_handle(std::uint64_t device_handle, agent_type type) const
{
    LOG_TRACE("Getting agent for device handle: {}, type {}", device_handle,
              to_string(type));
    auto _agent =
        std::find_if(_agents.begin(), _agents.end(), [&](const auto& agent_ptr) {
            return agent_ptr->type == type && agent_ptr->handle == device_handle;
        });
    if(_agent == _agents.end())
    {
        throw std::out_of_range(
            fmt::format("Agent not found for device handle: {}, type: {}", device_handle,
                        to_string(type)));
    }
    return **_agent;
}

const agent&
agent_manager::get_agent_by_handle(size_t device_handle) const
{
    LOG_TRACE("Getting agent for device handle: {}", device_handle);
    auto _agent =
        std::find_if(_agents.begin(), _agents.end(), [&](const auto& agent_ptr) {
            return agent_ptr->handle == device_handle;
        });
    if(_agent == _agents.end())
    {
        throw std::out_of_range(
            fmt::format("Agent not found for device handle: {}", device_handle));
    }
    return **_agent;
}

std::vector<std::shared_ptr<agent>>
agent_manager::get_agents_by_type(agent_type type) const
{
    LOG_TRACE("Getting agent for device type: {}", to_string(type));

    std::vector<std::shared_ptr<agent>> agents;
    std::copy_if(std::begin(_agents), std::end(_agents), std::back_inserter(agents),
                 [&type](const auto& agent_ptr) { return agent_ptr->type == type; });
    return agents;
}

std::vector<std::shared_ptr<agent>>
agent_manager::get_agents() const
{
    return _agents;
}

size_t
agent_manager::get_gpu_agents_count() const
{
    return get_agent_count(agent_type::GPU);
}

size_t
agent_manager::get_cpu_agents_count() const
{
    return get_agent_count(agent_type::CPU);
}

size_t
agent_manager::get_agent_count(agent_type type) const
{
    auto it = _agent_counts.find(type);
    return it != _agent_counts.end() ? it->second : 0;
}

}  // namespace rocprofsys
