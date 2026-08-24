// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "profiler-hub/reader.hpp"
#include "profiler-hub/reader_types.hpp"
#include "profiler-hub/storage.hpp"

#include "data_storage/backends/sqlite_backend.hpp"
#include "data_storage/read_statements.hpp"
#include "data_storage/read_statements_v4.hpp"
#include "entity_utility.hpp"

#include <memory>
#include <optional>

namespace profiler_hub
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

// Per-track routing info used to scope get_interval_track / get_scalar_track to a
// track's identity. Keyed by track_info_t::id (real rocpd_track id or synthetic id).
struct track_query_info_t
{
    reader_types::track_type_t type{};
    size_t                     nid{};
    size_t                     pid{};
    std::optional<size_t>      tid;        // cpu_thread
    std::optional<size_t>      agent_id;   // gpu_queue, memory; dma (dst_agent_id)
    std::optional<size_t>      queue_id;   // gpu_queue, memory, dma
    std::optional<size_t>      stream_id;  // stream
    std::optional<size_t>      pmc_id;     // kernel_dispatch_pmc
    size_t real_track_id{};                // counter: rocpd_track id for sample.track_id
    bool   region_is_sample{};             // cpu_thread: true => sample-kind region track
};

struct reader_t::impl
{
    explicit impl(std::unique_ptr<profiler_hub::storage_t> storage);

    // Info table accessors (cached)
    [[nodiscard]] reader_types::node_info_list_t          get_all_nodes();
    [[nodiscard]] reader_types::process_info_list_t       get_all_processes();
    [[nodiscard]] reader_types::thread_info_list_t        get_all_threads();
    [[nodiscard]] reader_types::agent_info_list_t         get_all_agents();
    [[nodiscard]] reader_types::track_info_list_t         get_tracks();
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

    // Track-scoped event queries
    [[nodiscard]] reader_types::interval_entry_list_t get_interval_track(
        size_t                              track_id,
        const reader_types::event_filter_t& filter);

    [[nodiscard]] reader_types::scalar_sample_list_t get_scalar_track(
        size_t                              track_id,
        const reader_types::event_filter_t& filter);

    [[nodiscard]] reader_types::track_stats_t get_track_stats(size_t track_id);

    [[nodiscard]] reader_types::flow_list_t get_flows(
        const reader_types::event_filter_t& filter);

    [[nodiscard]] reader_types::flow_list_t get_flows_for_event(
        const reader_types::event_id_t& id);

    [[nodiscard]] reader_types::flow_list_t get_flows_for_chain(
        const reader_types::flow_id_t& flow_id);

    [[nodiscard]] reader_types::flow_list_t get_flows_in_window(
        const std::vector<reader_types::track_id_t>& tracks,
        const reader_types::time_window_t&           window,
        uint32_t                                     max_edges);

    // Scalar / pmc detail queries (by event handle)
    [[nodiscard]] std::optional<reader_types::pmc_event_data_t> get_pmc_event_details(
        const reader_types::event_id_t& id);

    // Event detail queries
    [[nodiscard]] std::optional<reader_types::region_data_t> get_region_details(
        const reader_types::timeline_event_t& event);

    [[nodiscard]] std::optional<reader_types::kernel_dispatch_data_t>
    get_kernel_dispatch_details(const reader_types::timeline_event_t& event);

    [[nodiscard]] std::optional<reader_types::memory_copy_data_t> get_memory_copy_details(
        const reader_types::timeline_event_t& event);

    [[nodiscard]] std::optional<reader_types::memory_alloc_data_t>
    get_memory_alloc_details(const reader_types::timeline_event_t& event);

    // Event detail queries (by event handle)
    [[nodiscard]] std::optional<reader_types::region_data_t> get_region_details(
        const reader_types::event_id_t& id);

    [[nodiscard]] std::optional<reader_types::kernel_dispatch_data_t>
    get_kernel_dispatch_details(const reader_types::event_id_t& id);

    [[nodiscard]] std::optional<reader_types::memory_copy_data_t> get_memory_copy_details(
        const reader_types::event_id_t& id);

    [[nodiscard]] std::optional<reader_types::memory_alloc_data_t>
    get_memory_alloc_details(const reader_types::event_id_t& id);

    [[nodiscard]] std::optional<reader_types::event_info_t> get_event_info(
        const reader_types::event_id_t& id);

    // Event property queries
    [[nodiscard]] reader_types::call_stack_t get_call_stack(
        const reader_types::timeline_event_t& event);

    [[nodiscard]] reader_types::source_context_list_t get_source_context(
        const reader_types::timeline_event_t& event);

    [[nodiscard]] reader_types::call_stack_t get_call_stack(
        const reader_types::event_id_t& id);

    [[nodiscard]] reader_types::source_context_list_t get_source_context(
        const reader_types::event_id_t& id);

    [[nodiscard]] reader_types::arg_data_list_t get_arguments(
        const reader_types::timeline_event_t& event);

    [[nodiscard]] reader_types::arg_data_list_t get_arguments(
        const reader_types::event_id_t& id);

    [[nodiscard]] reader_types::timeline_event_list_t get_correlated_events(
        const reader_types::timeline_event_t& event);

    // Database metadata
    [[nodiscard]] reader_types::time_window_t  get_time_range();
    [[nodiscard]] reader_types::event_counts_t get_event_counts(
        const reader_types::time_window_t& window);

    // Aggregate (GROUP BY name) summaries
    [[nodiscard]] reader_types::event_summary_list_t get_kernel_summary(
        const reader_types::time_window_t& window);
    [[nodiscard]] reader_types::event_summary_list_t get_region_summary(
        const reader_types::time_window_t& window);

private:
    void initialize_string_list();
    void initialize_all_info_lists();

    // Synthesize gpu_queue + dma tracks (no rocpd_track rows exist for them in v3).
    void synthesize_derived_tracks();

    // Build tracks from real rocpd_track rows (v4.0: every swimlane is a track row).
    void build_v4_tracks();

    // Resolve event metadata from event-specific table by db_id and type.
    // Returns event_id_result containing event_id + stack_id + call_stack JSON etc.
    [[nodiscard]] std::optional<data_storage::event_id_result> resolve_event_metadata(
        const reader_types::timeline_event_t& event);

    // Build event_data_t from event_id (queries rocpd_event, parses JSON)
    [[nodiscard]] reader_types::event_data_ptr_t build_event_data(
        const data_storage::event_id_result& event_meta);

    // Converts raw SQL results to timeline_event_t, resolving FKs
    [[nodiscard]] reader_types::timeline_event_list_t build_timeline_events(
        const std::vector<data_storage::timeline_event_result>& results,
        reader_types::event_type_t                              type);

    // Applies limit/offset to merged event list
    void apply_pagination(reader_types::timeline_event_list_t& events,
                          const reader_types::pagination_t&    pagination);

    [[nodiscard]] reader_types::pmc_event_data_t build_pmc_event_data(
        const data_storage::scalar_detail_result& row);

    std::unique_ptr<profiler_hub::storage_t>            m_storage;
    std::shared_ptr<data_storage::sqlite_backend>       m_backend;
    std::shared_ptr<data_storage::read_statements_base> m_read_statements;

    // True when the opened database is a v4.0 rocpd schema (rocpd_timestamp spine
    // present). Selected once at construction; drives the v4 read paths and guards
    // the legacy v3-only reader surface (which returns empty/nullopt on v4).
    bool m_is_v4{ false };

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

    // Track lookup maps (populated during get_tracks)
    std::
        unordered_map<topology_key_t, reader_types::track_info_ptr_t, topology_key_hash_t>
            m_topology_to_track_ptr;

    std::unordered_map<reader_types::track_info_ptr_t, topology_key_t>
        m_track_ptr_to_topology;

    std::unordered_map<reader_types::track_info_ptr_t, size_t> m_track_ptr_to_db_id;

    std::unordered_map<size_t, track_query_info_t> m_track_query_info;
};

}  // namespace profiler_hub
