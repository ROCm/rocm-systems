// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include <rocstorage/writer.hpp>

#include "autoincrementer.hpp"
#include "entity_utility.hpp"

#include <string>
#include <unordered_map>
#include <unordered_set>

namespace rocstorage
{

using primary_key = size_t;

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
        return agent_unique_id_hash{}(*pmc.agent_id) ^ std::hash<std::string>{}(pmc.name);
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

struct data_identifiers
{
    ~data_identifiers() = default;

    [[nodiscard]] auto& node_info() { return node_info_utility; }
    [[nodiscard]] auto& process_info() { return process_info_utility; }
    [[nodiscard]] auto& agent_info() { return agent_info_utility; }
    [[nodiscard]] auto& pmc_info() { return pmc_info_utility; }
    [[nodiscard]] auto& thread_info() { return thread_info_utility; }
    [[nodiscard]] auto& stream_info() { return stream_info_utility; }
    [[nodiscard]] auto& queue_info() { return queue_info_utility; }
    [[nodiscard]] auto& code_object_info() { return code_object_info_utility; }
    [[nodiscard]] auto& kernel_symbol_info() { return kernel_symbol_info_utility; }
    [[nodiscard]] auto& track_info() { return track_info_utility; }
    [[nodiscard]] auto& string_info() { return string_info_utility; }
    [[nodiscard]] auto& process_info_primary_key_provider()
    {
        return m_process_info_primary_key_provider;
    }
    [[nodiscard]] auto& agent_info_primary_key_provider()
    {
        return m_agent_info_primary_key_provider;
    }
    [[nodiscard]] auto& pmc_info_primary_key_provider()
    {
        return m_pmc_info_primary_key_provider;
    }
    [[nodiscard]] auto& thread_info_primary_key_provider()
    {
        return m_thread_info_primary_key_provider;
    }
    [[nodiscard]] auto& stream_info_primary_key_provider()
    {
        return m_stream_info_primary_key_provider;
    }
    [[nodiscard]] auto& queue_info_primary_key_provider()
    {
        return m_queue_info_primary_key_provider;
    }
    [[nodiscard]] auto& track_info_primary_key_provider()
    {
        return m_track_info_primary_key_provider;
    }
    [[nodiscard]] auto& string_info_primary_key_provider()
    {
        return m_string_info_primary_key_provider;
    }
    [[nodiscard]] auto& event_data_primary_key_provider()
    {
        return m_event_data_primary_key_provider;
    }
    [[nodiscard]] auto& sample_data_primary_key_provider()
    {
        return m_sample_data_primary_key_provider;
    }
    [[nodiscard]] auto& region_data_primary_key_provider()
    {
        return m_region_data_primary_key_provider;
    }
    [[nodiscard]] auto& arg_primary_key_provider() { return m_arg_primary_key_provider; }
    [[nodiscard]] auto& pmc_event_data_primary_key_provider()
    {
        return m_pmc_event_data_primary_key_provider;
    }
    [[nodiscard]] auto& kernel_dispatch_data_primary_key_provider()
    {
        return m_kernel_dispatch_data_primary_key_provider;
    }
    [[nodiscard]] auto& memory_copy_data_primary_key_provider()
    {
        return m_memory_copy_data_primary_key_provider;
    }
    [[nodiscard]] auto& memory_alloc_data_primary_key_provider()
    {
        return m_memory_alloc_data_primary_key_provider;
    }

private:
    entity_utility<std::unordered_set<writer_api::node_id_t>> node_info_utility{};
    entity_utility<std::unordered_map<writer_api::process_id_t, primary_key>>
        process_info_utility{};
    entity_utility<std::unordered_map<writer_api::agent_unique_id_t,
                                      primary_key,
                                      hashing::agent_unique_id_hash>>
        agent_info_utility{};
    entity_utility<std::unordered_map<writer_api::pmc_info_unique_id_t,
                                      primary_key,
                                      hashing::pmc_unique_id_hash>>
        pmc_info_utility{};
    entity_utility<std::unordered_map<writer_api::thread_id_t, primary_key>>
        thread_info_utility{};
    entity_utility<std::unordered_map<writer_api::stream_id_t, primary_key>>
        stream_info_utility{};
    entity_utility<std::unordered_map<writer_api::queue_id_t, primary_key>>
        queue_info_utility{};
    entity_utility<std::unordered_set<writer_api::code_object_id_t>>
        code_object_info_utility{};
    entity_utility<std::unordered_set<writer_api::kernel_symbol_id_t>>
        kernel_symbol_info_utility{};
    entity_utility<std::unordered_map<writer_api::track_info_t,
                                      primary_key,
                                      hashing::track_info_hash>>
                                                                 track_info_utility{};
    entity_utility<std::unordered_map<std::string, primary_key>> string_info_utility{};

    autoincrementer<primary_key> m_process_info_primary_key_provider{ "process_info" };

    autoincrementer<primary_key> m_agent_info_primary_key_provider{ "agent_info" };
    autoincrementer<primary_key> m_pmc_info_primary_key_provider{ "pmc_info" };

    autoincrementer<primary_key> m_thread_info_primary_key_provider{ "thread_info" };
    autoincrementer<primary_key> m_stream_info_primary_key_provider{ "stream_info" };
    autoincrementer<primary_key> m_queue_info_primary_key_provider{ "queue_info" };
    autoincrementer<primary_key> m_track_info_primary_key_provider{ "track_info" };
    autoincrementer<primary_key> m_string_info_primary_key_provider{ "string_info" };

    autoincrementer<primary_key> m_event_data_primary_key_provider{ "event_data" };
    autoincrementer<primary_key> m_sample_data_primary_key_provider{ "sample_data" };
    autoincrementer<primary_key> m_region_data_primary_key_provider{ "region_data" };
    autoincrementer<primary_key> m_arg_primary_key_provider{ "arg" };
    autoincrementer<primary_key> m_pmc_event_data_primary_key_provider{
        "pmc_event_data"
    };
    autoincrementer<primary_key> m_kernel_dispatch_data_primary_key_provider{
        "kernel_dispatch_data"
    };
    autoincrementer<primary_key> m_memory_copy_data_primary_key_provider{
        "memory_copy_data"
    };
    autoincrementer<primary_key> m_memory_alloc_data_primary_key_provider{
        "memory_alloc_data"
    };
};
}  // namespace rocstorage
