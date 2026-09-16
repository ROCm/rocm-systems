// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace rocprofsys::policies
{
template <typename Agent, typename AgentType>
concept agent_policy = requires(const Agent& agent) {
    { agent.type } -> std::convertible_to<AgentType>;
    { agent.handle } -> std::convertible_to<std::uint64_t>;
    { agent.device_id } -> std::convertible_to<std::uint64_t>;
    { agent.node_id } -> std::convertible_to<std::uint32_t>;
    { agent.logical_node_id } -> std::convertible_to<std::int32_t>;
    { agent.logical_node_type_id } -> std::convertible_to<std::int32_t>;
    { agent.name } -> std::convertible_to<std::string_view>;
    { agent.model_name } -> std::convertible_to<std::string_view>;
    { agent.vendor_name } -> std::convertible_to<std::string_view>;
    { agent.product_name } -> std::convertible_to<std::string_view>;
    { agent.device_type_index } -> std::convertible_to<std::size_t>;
    { agent.agent_info } -> std::convertible_to<std::string_view>;
    { agent.location_id } -> std::convertible_to<std::uint32_t>;
    { agent.domain } -> std::convertible_to<std::uint32_t>;
    { agent.hip_visible } -> std::convertible_to<bool>;
};
}  // namespace rocprofsys::policies
