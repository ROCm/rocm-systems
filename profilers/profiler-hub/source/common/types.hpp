// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <functional>
#include <optional>
#include <string>

#include "profiler-hub/writer_types.hpp"

namespace profiler_hub::internal_types
{

struct agent_unique_id_t
{
    std::optional<std::string> agent_type;
    size_t                     type_index;

    agent_unique_id_t(const writer_types::agent_unique_id_t& agent_unique_id)
    : agent_type(agent_unique_id.agent_type.has_value()
                     ? std::make_optional<std::string>(agent_unique_id.agent_type.value())
                     : std::nullopt)
    , type_index(agent_unique_id.type_index)
    {}

    bool operator==(const agent_unique_id_t& other) const noexcept
    {
        return agent_type == other.agent_type && type_index == other.type_index;
    }
};

struct pmc_info_unique_id_t
{
    std::string                      name;
    std::optional<agent_unique_id_t> agent_id;

    pmc_info_unique_id_t(const writer_types::pmc_info_unique_id_t& pmc_info_unique_id)
    : name(pmc_info_unique_id.name)
    , agent_id(
          pmc_info_unique_id.agent_id.has_value()
              ? std::make_optional<agent_unique_id_t>(pmc_info_unique_id.agent_id.value())
              : std::nullopt)
    {}

    bool operator==(const pmc_info_unique_id_t& other) const noexcept
    {
        return name == other.name && agent_id == other.agent_id;
    }
};

struct track_info_t
{
    std::optional<std::string> name;

    writer_types::node_id_t                   node_id{};
    std::optional<writer_types::process_id_t> process_id;
    std::optional<writer_types::thread_id_t>  thread_id;

    track_info_t(const writer_types::track_info_t& track_info)
    : name(track_info.name.has_value()
               ? std::make_optional<std::string>(track_info.name.value())
               : std::nullopt)
    , node_id(track_info.node_id)
    , process_id(track_info.process_id)
    , thread_id(track_info.thread_id)
    {}

    bool operator==(const track_info_t& other) const noexcept
    {
        return name == other.name && node_id == other.node_id &&
               process_id == other.process_id && thread_id == other.thread_id;
    }
};

namespace hashing
{
struct owned_agent_unique_id_hash
{
    std::size_t operator()(const agent_unique_id_t& agent) const noexcept
    {
        return std::hash<std::string>{}(agent.agent_type.value_or("NULL")) ^
               std::hash<size_t>{}(agent.type_index);
    }
};

struct owned_pmc_unique_id_hash
{
    std::size_t operator()(const pmc_info_unique_id_t& pmc) const noexcept
    {
        if(pmc.agent_id.has_value())
        {
            return owned_agent_unique_id_hash{}(*pmc.agent_id) ^
                   std::hash<std::string>{}(pmc.name);
        }
        return std::hash<std::string>{}(pmc.name);
    }
};

struct track_info_hash
{
    std::size_t operator()(const track_info_t& track_info) const noexcept
    {
        std::string const track_name_value =
            track_info.name.has_value() ? track_info.name.value().data() : "";
        size_t const process_id_value =
            track_info.process_id.has_value() ? track_info.process_id.value() : 0;
        size_t const thread_id_value =
            track_info.thread_id.has_value() ? track_info.thread_id.value() : 0;

        return std::hash<size_t>{}(track_info.node_id) ^
               std::hash<std::string>{}(track_name_value) ^
               std::hash<size_t>{}(process_id_value) ^
               std::hash<size_t>{}(thread_id_value);
    }
};
}  // namespace hashing
}  // namespace profiler_hub::internal_types
