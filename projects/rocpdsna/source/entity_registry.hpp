// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "rocpdsna/writer_types.hpp"

#include "common/types.hpp"
#include "entity_utility.hpp"

#include <cstddef>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace rocpdsna
{

using primary_key_t = size_t;

struct entity_registry
{
    // V3 & V4 registries
    [[nodiscard]] auto& node_info() { return m_node_info; }
    [[nodiscard]] auto& process_info() { return m_process_info; }
    [[nodiscard]] auto& agent_info() { return m_agent_info; }
    [[nodiscard]] auto& pmc_info() { return m_pmc_info; }
    [[nodiscard]] auto& thread_info() { return m_thread_info; }
    [[nodiscard]] auto& stream_info() { return m_stream_info; }
    [[nodiscard]] auto& queue_info() { return m_queue_info; }
    [[nodiscard]] auto& code_object_info() { return m_code_object_info; }
    [[nodiscard]] auto& kernel_symbol_info() { return m_kernel_symbol_info; }
    [[nodiscard]] auto& track_info() { return m_track_info; }
    [[nodiscard]] auto& string_info() { return m_string_info; }

    // V4 only registries
    [[nodiscard]] auto& category_info() { return m_category_info; }
    [[nodiscard]] auto& address_range_info() { return m_address_range_info; }
    [[nodiscard]] auto& source_code_info() { return m_source_code_info; }
    [[nodiscard]] auto& pc_info() { return m_pc_info; }

private:
    // =========================================================================
    // V3 & V4 registries
    // =========================================================================

    entity_utility<std::unordered_set<writer_types::node_id_t>> m_node_info{};

    entity_utility<std::unordered_map<writer_types::process_id_t, primary_key_t>>
        m_process_info{};

    entity_utility<
        std::unordered_map<internal_types::agent_unique_id_t,
                           primary_key_t,
                           internal_types::hashing::owned_agent_unique_id_hash>>
        m_agent_info{};

    entity_utility<std::unordered_map<internal_types::pmc_info_unique_id_t,
                                      primary_key_t,
                                      internal_types::hashing::owned_pmc_unique_id_hash>>
        m_pmc_info{};

    entity_utility<std::unordered_map<writer_types::thread_id_t, primary_key_t>>
        m_thread_info{};

    entity_utility<std::unordered_map<writer_types::stream_id_t, primary_key_t>>
        m_stream_info{};

    entity_utility<std::unordered_map<writer_types::queue_id_t, primary_key_t>>
        m_queue_info{};

    entity_utility<std::unordered_set<writer_types::code_object_id_t>>
        m_code_object_info{};

    entity_utility<std::unordered_set<writer_types::kernel_symbol_id_t>>
        m_kernel_symbol_info{};

    entity_utility<std::unordered_map<internal_types::track_info_t,
                                      primary_key_t,
                                      internal_types::hashing::track_info_hash>>
        m_track_info{};

    entity_utility<std::unordered_map<std::string, primary_key_t>> m_string_info{};

    // =========================================================================
    // V4 only registries
    // =========================================================================

    // Category info registry for v4 schema (categories stored in rocpd_info_category
    // table)
    entity_utility<std::unordered_map<std::string, primary_key_t>> m_category_info{};

    // Address range info registry for v4 schema (rocpd_info_address_range table)
    // Maps address_range_id_t → DB primary key
    entity_utility<std::unordered_map<writer_types::address_range_id_t, primary_key_t>>
        m_address_range_info{};

    // Source code info registry for v4 schema (rocpd_info_source_code table)
    // Maps source_code_id_t → DB primary key
    entity_utility<std::unordered_map<writer_types::source_code_id_t, primary_key_t>>
        m_source_code_info{};

    // Program counter info registry for v4 schema (rocpd_info_pc table)
    // Maps pc_id_t → DB primary key
    entity_utility<std::unordered_map<writer_types::pc_id_t, primary_key_t>> m_pc_info{};
};

}  // namespace rocpdsna
