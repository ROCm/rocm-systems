// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once
#include <cstddef>
#include <memory>
#include <unordered_map>
#include <vector>

#include "agent.hpp"

namespace rocprofsys
{

struct agent_manager
{
    agent_manager() = default;
    agent_manager(std::vector<std::shared_ptr<agent>> agents);
    agent_manager(const agent_manager&)            = delete;
    agent_manager& operator=(const agent_manager&) = delete;
    agent_manager(agent_manager&&)                 = delete;
    agent_manager& operator=(agent_manager&&)      = delete;
    ~agent_manager()                               = default;

    void         insert_agent(agent& agent);
    const agent& get_agent_by_type_index(size_t type_index, agent_type type) const;
    const agent& get_agent_by_id(size_t device_id, agent_type type) const;
    const agent& get_agent_by_handle(size_t device_handle, agent_type type) const;
    const agent& get_agent_by_handle(size_t device_handle) const;

    std::vector<std::shared_ptr<agent>> get_agents_by_type(agent_type type) const;

    std::vector<std::shared_ptr<agent>> get_agents() const;

    size_t get_gpu_agents_count() const;
    size_t get_cpu_agents_count() const;

private:
    size_t                                 get_agent_count(agent_type type) const;
    std::vector<std::shared_ptr<agent>>    _agents;
    std::unordered_map<agent_type, size_t> _agent_counts;
};

agent_manager&
get_agent_manager_instance();

}  // namespace rocprofsys
