// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "profiler-hub/cpp/reader.hpp"
#include "profiler-hub/cpp/reader_types.hpp"
#include "profiler-hub/cpp/storage.hpp"

#include "data_storage/backends/sqlite_backend.hpp"
#include "data_storage/read_statements.hpp"
#include "entity_utility.hpp"
#include "reader_catalog.hpp"

#include <memory>
#include <optional>
#include <unordered_map>

namespace profiler_hub
{

struct reader_t::impl
{
    explicit impl(std::unique_ptr<profiler_hub::storage_t> storage);

    // Pooled/shared-catalog construction: does not build anything itself;
    // `catalog` (non-null) is shared with sibling connections and may be
    // populated by any of them via build_catalog_category().
    impl(std::unique_ptr<profiler_hub::storage_t> storage,
         std::shared_ptr<reader_catalog_t>        catalog);

    void build_catalog_category(reader_t::catalog_category_t category,
                                reader_catalog_t&            catalog);

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

    [[nodiscard]] reader_types::counter_timeline_event_list_t
    get_counter_events_for_track(reader_types::track_info_ptr_t      track,
                                 const reader_types::event_filter_t& filter);

    [[nodiscard]] size_t get_event_count(const reader_types::event_filter_t& filter);

    // Event detail queries
    [[nodiscard]] std::optional<reader_types::region_data_t> get_region_details(
        const reader_types::timeline_event_t& event);

    [[nodiscard]] std::optional<reader_types::kernel_dispatch_data_t>
    get_kernel_dispatch_details(const reader_types::timeline_event_t& event);

    [[nodiscard]] std::optional<reader_types::memory_copy_data_t> get_memory_copy_details(
        const reader_types::timeline_event_t& event);

    [[nodiscard]] std::optional<reader_types::memory_alloc_data_t>
    get_memory_alloc_details(const reader_types::timeline_event_t& event);

    // Event property queries
    [[nodiscard]] reader_types::call_stack_t get_call_stack(
        const reader_types::timeline_event_t& event);

    [[nodiscard]] reader_types::source_context_list_t get_source_context(
        const reader_types::timeline_event_t& event);

    [[nodiscard]] reader_types::arg_data_list_t get_arguments(
        const reader_types::timeline_event_t& event);

    [[nodiscard]] reader_types::timeline_event_list_t get_correlated_events(
        const reader_types::timeline_event_t& event);

    // Database metadata
    [[nodiscard]] reader_types::time_window_t  get_data_time_range();
    [[nodiscard]] reader_types::event_counts_t get_event_counts(
        const reader_types::time_window_t& window);

private:
    // Resolve event metadata from event-specific table by db_id and type.
    // Returns event_id_result containing event_id + stack_id + call_stack JSON etc.
    [[nodiscard]] std::optional<data_storage::schema_v3::event_id_result>
    resolve_event_metadata(const reader_types::timeline_event_t& event);

    // Build event_data_t from event_id (queries rocpd_event, parses JSON)
    [[nodiscard]] reader_types::event_data_ptr_t build_event_data(
        const data_storage::schema_v3::event_id_result& event_meta);

    // Converts raw SQL results to timeline_event_t, resolving FKs
    [[nodiscard]] reader_types::timeline_event_list_t build_timeline_events(
        const std::vector<data_storage::schema_v3::timeline_event_result>& results,
        reader_types::event_type_t                                         type);

    // Applies limit/offset to merged event list
    void apply_pagination(reader_types::timeline_event_list_t& events,
                          const reader_types::pagination_t&    pagination);

    // Fetches events for the 4 optiq-parity category tracks (agent+queue
    // and stream), which have no rocpd_track row and so aren't in
    // m_track_ptr_to_topology/m_track_ptr_to_db_id.
    [[nodiscard]] reader_types::timeline_event_list_t get_category_track_events(
        const reader_types::track_info_ptr_t& track,
        const reader_types::event_filter_t&   filter);

    std::unique_ptr<profiler_hub::storage_t>                  m_storage;
    std::shared_ptr<data_storage::sqlite_backend>             m_backend;
    std::shared_ptr<data_storage::schema_v3::read_statements> m_read_statements;

    std::shared_ptr<reader_catalog_t> m_catalog;
};

}  // namespace profiler_hub
