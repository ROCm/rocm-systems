// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "policies/agent_policy.hpp"

#include <concepts>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace rocprofsys::policies::trace_cache
{

template <typename Registry, typename Process, typename Pmc, typename Thread,
          typename Track, typename Agent, typename AgentType>
concept metadata_registry_policy =
    agent_policy<Agent, AgentType> &&
    requires(Registry& registry, const Registry& const_registry, const Process& process,
             const Pmc& pmc_info, const Thread& thread_info, const Track& track_info,
             const std::uint64_t& handle, const std::string_view& name,
             const std::uint32_t& thread_id, const std::string& filepath,
             std::vector<std::shared_ptr<Agent>>& agents) {
        { Registry() };

        { registry.set_process(process) };
        { registry.add_pmc_info(pmc_info) };
        { registry.add_thread_info(thread_info) };
        { registry.add_track(track_info) };
        { registry.add_queue(handle) };
        { registry.add_stream(handle) };
        { registry.add_string(name) };

        { const_registry.get_process_info() } -> std::convertible_to<Process>;
        { const_registry.get_pmc_info(name) } -> std::convertible_to<std::optional<Pmc>>;
        {
            const_registry.get_thread_info(thread_id)
        } -> std::convertible_to<std::optional<Thread>>;
        {
            const_registry.get_track_info(name)
        } -> std::convertible_to<std::optional<Track>>;
        { const_registry.get_pmc_info_list() } -> std::convertible_to<std::vector<Pmc>>;
        {
            const_registry.get_thread_info_list()
        } -> std::convertible_to<std::vector<Thread>>;
        {
            const_registry.get_track_info_list()
        } -> std::convertible_to<std::vector<Track>>;
        {
            const_registry.get_queue_list()
        } -> std::convertible_to<std::vector<std::uint64_t>>;
        {
            const_registry.get_stream_list()
        } -> std::convertible_to<std::vector<std::uint64_t>>;
        {
            const_registry.get_string_list()
        } -> std::convertible_to<std::vector<std::string_view>>;
        { const_registry.save_to_file(filepath, agents) } -> std::convertible_to<bool>;
    };

}  // namespace rocprofsys::policies::trace_cache
