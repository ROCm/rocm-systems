// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <concepts>
#include <cstddef>
#include <memory>
#include <ranges>
#include <vector>

#include "agent_policy.hpp"

namespace rocprofsys::policies
{

template <typename Manager, typename Agent, typename AgentType>
concept agent_manager_policy =
    agent_policy<Agent, AgentType> &&
    requires(Manager& manager, const Manager& const_manager, Agent& agent_ref,
             std::vector<std::shared_ptr<Agent>> agents, std::size_t index,
             AgentType type) {
        { Manager(agents) };
        { manager.insert_agent(agent_ref) };
        {
            const_manager.get_agent_by_type_index(index, type)
        } -> std::convertible_to<const Agent&>;
        {
            const_manager.get_agent_by_id(index, type)
        } -> std::convertible_to<const Agent&>;
        {
            const_manager.get_agent_by_handle(index, type)
        } -> std::convertible_to<const Agent&>;
        { const_manager.get_agent_by_handle(index) } -> std::convertible_to<const Agent&>;
        { const_manager.get_agents_by_type(type) } -> std::ranges::range;
        { const_manager.get_agents() } -> std::ranges::range;
        { const_manager.get_gpu_agents_count() } -> std::convertible_to<std::size_t>;
        { const_manager.get_cpu_agents_count() } -> std::convertible_to<std::size_t>;
    };
}  // namespace rocprofsys::policies
