// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "agent_manager.hpp"

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
    LOG_TRACE("Inserting agent with device handle: {}, and agent id: {}, device type: {}",
              _agent.device_id,
              (_agent.type == agent_type::GPU ? _gpu_agents_cnt : _cpu_agents_cnt),
              (_agent.type == agent_type::GPU ? "GPU" : "CPU"));

    _agent.device_type_index =
        (_agent.type == agent_type::GPU ? _gpu_agents_cnt++ : _cpu_agents_cnt++);
    _agents.emplace_back(std::make_shared<agent>(_agent));
}

const agent&
agent_manager::get_agent_by_type_index(size_t type_index, agent_type type) const
{
    LOG_TRACE("Getting agent for type: {}, with type index: {}",
              (type == agent_type::GPU ? "GPU" : "CPU"), type_index);
    auto _agent =
        std::find_if(_agents.begin(), _agents.end(), [&](const auto& agent_ptr) {
            return agent_ptr->type == type && agent_ptr->device_type_index == type_index;
        });
    if(_agent == _agents.end())
    {
        throw std::out_of_range(
            fmt::format("Agent not found for type index: {}, type: {}", type_index,
                        (type == agent_type::GPU ? "GPU" : "CPU")));
    }
    return **_agent;
}

const agent&
agent_manager::get_agent_by_id(size_t device_id, agent_type type) const
{
    LOG_TRACE("Getting agent for device id: {}, type {}", device_id,
              (type == agent_type::GPU ? "GPU" : "CPU"));
    auto _agent =
        std::find_if(_agents.begin(), _agents.end(), [&](const auto& agent_ptr) {
            return agent_ptr->type == type && agent_ptr->device_id == device_id;
        });
    if(_agent == _agents.end())
    {
        throw std::out_of_range(fmt::format("Agent not found for device id: {}, type: {}",
                                            device_id,
                                            (type == agent_type::GPU ? "GPU" : "CPU")));
    }
    return **_agent;
}

const agent&
agent_manager::get_agent_by_handle(uint64_t device_handle, agent_type type) const
{
    LOG_TRACE("Getting agent for device handle: {}, type {}", device_handle,
              (type == agent_type::GPU ? "GPU" : "CPU"));
    auto _agent =
        std::find_if(_agents.begin(), _agents.end(), [&](const auto& agent_ptr) {
            return agent_ptr->type == type && agent_ptr->handle == device_handle;
        });
    if(_agent == _agents.end())
    {
        throw std::out_of_range(
            fmt::format("Agent not found for device handle: {}, type: {}", device_handle,
                        (type == agent_type::GPU ? "GPU" : "CPU")));
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
    LOG_TRACE("Getting agent for device type: {}",
              (type == agent_type::GPU ? "GPU" : "CPU"));

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
    return _gpu_agents_cnt;
}

size_t
agent_manager::get_cpu_agents_count() const
{
    return _cpu_agents_cnt;
}

}  // namespace rocprofsys
