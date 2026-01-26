// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include <rocstorage/writer_types.hpp>

#include <functional>
#include <string>

namespace rocstorage
{
namespace hashing
{

struct agent_unique_id_hash
{
    std::size_t operator()(const writer_api::agent_unique_id_t& agent) const noexcept
    {
        return std::hash<std::string>{}(agent.agent_type) ^
               std::hash<size_t>{}(agent.type_index);
    }
};

struct pmc_unique_id_hash
{
    std::size_t operator()(const writer_api::pmc_info_unique_id_t& pmc) const noexcept
    {
        if(pmc.agent_id.has_value())
        {
            return agent_unique_id_hash{}(*pmc.agent_id) ^
                   std::hash<std::string>{}(pmc.name);
        }
        return std::hash<std::string>{}(pmc.name);
    }
};

struct track_info_hash
{
    std::size_t operator()(const writer_api::track_info_t& track_info) const noexcept
    {
        std::string track_name_value =
            track_info.name.has_value() ? track_info.name.value() : "";
        size_t process_id_value =
            track_info.process_id.has_value() ? track_info.process_id.value() : 0;
        size_t thread_id_value =
            track_info.thread_id.has_value() ? track_info.thread_id.value() : 0;

        return std::hash<size_t>{}(track_info.node_id) ^
               std::hash<std::string>{}(track_name_value) ^
               std::hash<size_t>{}(process_id_value) ^
               std::hash<size_t>{}(thread_id_value);
    }
};

}  // namespace hashing
}  // namespace rocstorage
