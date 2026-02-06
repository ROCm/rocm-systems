// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "rocstorage/reader.hpp"
#include "rocstorage/reader_types.hpp"
#include "rocstorage/storage.hpp"

#include "data_storage/database.hpp"
#include "data_storage/read_statements.hpp"
#include "entity_utility.hpp"

#include <memory>

namespace rocstorage
{

struct topology_key_t
{
    size_t nid{};
    size_t pid{};
    size_t tid{};

    bool operator==(const topology_key_t& other) const
    {
        return nid == other.nid && pid == other.pid && tid == other.tid;
    }
};

struct topology_key_hash_t
{
    size_t operator()(const topology_key_t& k) const
    {
        size_t h = std::hash<size_t>{}(k.nid);
        h ^= std::hash<size_t>{}(k.pid) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<size_t>{}(k.tid) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct reader_t::impl
{
    explicit impl(std::unique_ptr<rocstorage::storage_t> storage);

    // Info table accessors (cached)
    [[nodiscard]] reader_types::node_info_list_t          get_all_nodes();
    [[nodiscard]] reader_types::process_info_list_t       get_all_processes();
    [[nodiscard]] reader_types::thread_info_list_t        get_all_threads();
    [[nodiscard]] reader_types::agent_info_list_t         get_all_agents();
    [[nodiscard]] reader_types::track_info_list_t         get_all_tracks();
    [[nodiscard]] reader_types::kernel_symbol_info_list_t get_all_kernel_symbols();
    [[nodiscard]] reader_types::code_object_info_list_t   get_all_code_objects();
    [[nodiscard]] reader_types::stream_info_list_t        get_all_streams();
    [[nodiscard]] reader_types::queue_info_list_t         get_all_queues();
    [[nodiscard]] reader_types::pmc_info_list_t           get_all_pmc_infos();

    // Timeline event queries
    [[nodiscard]] reader_types::timeline_event_list_t get_events(
        const reader_types::event_filter_t& filter);

    [[nodiscard]] reader_types::timeline_event_list_t get_events_for_track(
        reader_types::track_info_ptr_t      track,
        const reader_types::event_filter_t& filter);

    [[nodiscard]] size_t get_event_count(const reader_types::event_filter_t& filter);

private:
    void initialize_string_list();
    void initialize_all_info_lists();

    // Converts raw SQL results to timeline_event_t, resolving FKs
    [[nodiscard]] reader_types::timeline_event_list_t build_timeline_events(
        const std::vector<data_storage::schema_v3::timeline_event_result>& results,
        reader_types::event_type_t                                         type);

    // Applies limit/offset to merged event list
    void apply_pagination(reader_types::timeline_event_list_t& events,
                          const reader_types::pagination_t&    pagination);

    std::unique_ptr<rocstorage::storage_t>                    m_storage;
    std::shared_ptr<data_storage::database>                   m_database;
    std::shared_ptr<data_storage::schema_v3::read_statements> m_read_statements;

    reader_types::node_info_list_t          m_node_info_list;
    reader_types::process_info_list_t       m_process_info_list;
    reader_types::thread_info_list_t        m_thread_info_list;
    reader_types::agent_info_list_t         m_agent_info_list;
    reader_types::track_info_list_t         m_track_info_list;
    reader_types::kernel_symbol_info_list_t m_kernel_symbol_info_list;
    reader_types::code_object_info_list_t   m_code_object_info_list;
    reader_types::stream_info_list_t        m_stream_info_list;
    reader_types::queue_info_list_t         m_queue_info_list;
    reader_types::pmc_info_list_t           m_pmc_info_list;

    std::unordered_map<size_t, std::string> m_string_info_utility;

    std::unordered_map<size_t, reader_types::node_info_ptr_t>    m_node_info_utility;
    std::unordered_map<size_t, reader_types::process_info_ptr_t> m_process_info_utility;
    std::unordered_map<size_t, reader_types::thread_info_ptr_t>  m_thread_info_utility;
    std::unordered_map<size_t, reader_types::agent_info_ptr_t>   m_agent_info_utility;
    std::unordered_map<size_t, reader_types::track_info_ptr_t>   m_track_info_utility;
    std::unordered_map<size_t, reader_types::kernel_symbol_info_ptr_t>
        m_kernel_symbol_info_utility;
    std::unordered_map<size_t, reader_types::code_object_info_ptr_t>
        m_code_object_info_utility;
    std::unordered_map<size_t, reader_types::stream_info_ptr_t> m_stream_info_utility;
    std::unordered_map<size_t, reader_types::queue_info_ptr_t>  m_queue_info_utility;
    std::unordered_map<size_t, reader_types::pmc_info_ptr_t>    m_pmc_info_utility;

    // Track lookup maps (populated during get_all_tracks)
    std::
        unordered_map<topology_key_t, reader_types::track_info_ptr_t, topology_key_hash_t>
            m_topology_to_track_ptr;

    std::unordered_map<reader_types::track_info_ptr_t, topology_key_t>
        m_track_ptr_to_topology;

    std::unordered_map<reader_types::track_info_ptr_t, size_t> m_track_ptr_to_db_id;
};

}  // namespace rocstorage
