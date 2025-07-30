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
#include "debug.hpp"
#include <algorithm>
#include <iterator>

namespace rocprofsys
{

agent_manager&
agent_manager::get_instance()
{
    static agent_manager instance;
    return instance;
}

void
agent_manager::insert_agent(agent& _agent)
{
    ROCPROFSYS_VERBOSE(
        3, "Inserting agent with device handle: %lu, and agent id: %ld, device type: %s",
        _agent.device_id,
        (_agent.type == agent_type::GPU ? _gpu_agents_cnt : _cpu_agents_cnt),
        (_agent.type == agent_type::GPU ? "GPU" : "CPU"));

    _agent.device_id =
        (_agent.type == agent_type::GPU ? _gpu_agents_cnt++ : _cpu_agents_cnt++);
    _agents.emplace_back(std::make_shared<agent>(_agent));
}

const agent&
agent_manager::get_agent_by_id(size_t device_id, agent_type type) const
{
    ROCPROFSYS_VERBOSE(3, "Getting agent for device id: %ld, type %s\n", device_id,
                       (type == agent_type::GPU) ? "GPU" : "CPU");
    auto _agent =
        std::find_if(_agents.begin(), _agents.end(), [&](const auto& agent_ptr) {
            return agent_ptr->type == type && agent_ptr->device_id == device_id;
        });
    if(_agent == _agents.end())
    {
        std::ostringstream oss;
        oss << "Agent not found for device id: " << device_id
            << ", type: " << (type == agent_type::GPU ? "GPU" : "CPU");
        throw std::out_of_range(oss.str());
    }
    return **_agent;
}

const agent&
agent_manager::get_agent_by_handle(uint64_t device_handle, agent_type type) const
{
    ROCPROFSYS_VERBOSE(3, "Getting agent for device handle: %ld, type %s\n",
                       device_handle, (type == agent_type::GPU ? "GPU" : "CPU"));
    auto _agent =
        std::find_if(_agents.begin(), _agents.end(), [&](const auto& agent_ptr) {
            return agent_ptr->type == type && agent_ptr->id == device_handle;
        });
    if(_agent == _agents.end())
    {
        std::ostringstream oss;
        oss << "Agent not found for device handle: " << device_handle
            << ", type: " << (type == agent_type::GPU ? "GPU" : "CPU");
        throw std::out_of_range(oss.str());
    }
    return **_agent;
}

const agent&
agent_manager::get_agent_by_handle(size_t device_handle) const
{
    ROCPROFSYS_VERBOSE(3, "Getting agent for device handle: %ld\n", device_handle);
    auto _agent =
        std::find_if(_agents.begin(), _agents.end(), [&](const auto& agent_ptr) {
            return agent_ptr->id == device_handle;
        });
    if(_agent == _agents.end())
    {
        std::ostringstream oss;
        oss << "Agent not found for device handle: " << device_handle;
        throw std::out_of_range(oss.str());
    }
    return **_agent;
}

std::vector<std::shared_ptr<agent>>
agent_manager::get_agents_by_type(agent_type type) const
{
    ROCPROFSYS_VERBOSE(3, "Getting agent for device type: %s\n",
                       type == agent_type::GPU ? "GPU" : "CPU");

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
