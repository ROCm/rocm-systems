// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "profiler-hub/cpp/reader.hpp"
#include "profiler-hub/cpp/reader_types.hpp"
#include "profiler-hub/cpp/storage.hpp"

#include "data_storage/backends/sqlite_backend.hpp"
#include "event_query.hpp"
#include "histogram_grid.hpp"
#include "reader_catalog.hpp"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

struct ph_table_event;

namespace profiler_hub
{

/**
 * @brief One trace format's read path behind reader_t.
 *
 * Every format fills the same reader_catalog_t and answers the same queries.
 * Only the storage access behind them differs, so callers above reader_t never
 * see which format they read. A format registers its factory in
 * formats/trace_formats.cpp.
 *
 * The catalog is shared by every connection of one trace source. Pooled
 * readers receive it unbuilt and fill it through build_catalog_category().
 */
struct reader_t::impl
{
    virtual ~impl() = default;

    impl(const impl&)            = delete;
    impl& operator=(const impl&) = delete;
    impl(impl&&)                 = delete;
    impl& operator=(impl&&)      = delete;

    /** @brief Builds every catalog category in dependency order. */
    void build_catalog();

    virtual void build_catalog_category(reader_t::catalog_category_t category,
                                        reader_catalog_t&            catalog) = 0;
    virtual void ensure_track_topology_indexes()                   = 0;

    // Info table accessors (cached)
    [[nodiscard]] reader_types::node_info_list_t            get_all_nodes();
    [[nodiscard]] reader_types::process_info_list_t         get_all_processes();
    [[nodiscard]] reader_types::thread_info_list_t          get_all_threads();
    [[nodiscard]] reader_types::agent_info_list_t           get_all_agents();
    [[nodiscard]] reader_types::track_info_list_t           get_all_tracks();
    [[nodiscard]] reader_types::stream_hardware_link_list_t get_stream_hardware_links();
    [[nodiscard]] reader_types::kernel_symbol_info_list_t   get_all_kernel_symbols();
    [[nodiscard]] reader_types::code_object_info_list_t     get_all_code_objects();
    [[nodiscard]] reader_types::stream_info_list_t          get_all_streams();
    [[nodiscard]] reader_types::queue_info_list_t           get_all_queues();
    [[nodiscard]] reader_types::pmc_info_list_t             get_all_pmc_infos();

    // Timeline event queries
    [[nodiscard]] virtual reader_types::timeline_event_list_t get_events(
        const reader_types::event_filter_t& filter) = 0;

    [[nodiscard]] virtual reader_types::timeline_event_list_t get_events_for_track(
        reader_types::track_info_ptr_t      track,
        const reader_types::event_filter_t& filter) = 0;

    [[nodiscard]] virtual std::optional<std::pair<size_t, size_t>> get_event_id_span(
        reader_types::event_type_t type) = 0;

    virtual void visit_track_events_in_id_range(
        const reader_types::track_info_ptr_t& track,
        reader_types::event_type_t            type,
        size_t                                id_begin,
        size_t                                id_end,
        reader_t::event_visitor_t             visitor,
        void*                                 context) = 0;

    [[nodiscard]] virtual reader_types::counter_timeline_event_list_t
    get_counter_events_for_track(reader_types::track_info_ptr_t      track,
                                 const reader_types::event_filter_t& filter) = 0;

    [[nodiscard]] virtual size_t get_event_count(
        const reader_types::event_filter_t& filter) = 0;

    [[nodiscard]] virtual reader_types::histogram_t get_histogram(size_t max_buckets) = 0;

    /** @brief Histogram for this file's tracks on a grid shared by every source. */
    [[nodiscard]] virtual reader_types::histogram_t get_histogram(
        const histogram_grid_t& grid) = 0;

    // Event detail queries. A format without the data keeps the empty default.
    [[nodiscard]] virtual std::optional<reader_types::region_data_t> get_region_details(
        const reader_types::timeline_event_t& event);

    [[nodiscard]] virtual std::optional<reader_types::kernel_dispatch_data_t>
    get_kernel_dispatch_details(const reader_types::timeline_event_t& event);

    [[nodiscard]] virtual std::optional<reader_types::memory_copy_data_t>
    get_memory_copy_details(const reader_types::timeline_event_t& event);

    [[nodiscard]] virtual std::optional<reader_types::memory_alloc_data_t>
    get_memory_alloc_details(const reader_types::timeline_event_t& event);

    // Track the same placement ph_get_track_events uses for this row.
    [[nodiscard]] virtual reader_types::track_info_ptr_t track_of(
        const reader_types::timeline_event_t& event) = 0;

    // Event property queries. A format without the data keeps the empty default.
    [[nodiscard]] virtual reader_types::call_stack_t get_call_stack(
        const reader_types::timeline_event_t& event);

    [[nodiscard]] virtual reader_types::source_context_list_t get_source_context(
        const reader_types::timeline_event_t& event);

    [[nodiscard]] virtual reader_types::arg_data_list_t get_arguments(
        const reader_types::timeline_event_t& event);

    [[nodiscard]] virtual reader_types::timeline_event_list_t get_correlated_events(
        const reader_types::timeline_event_t& event);

    // Database metadata
    [[nodiscard]] virtual reader_types::time_window_t  get_data_time_range() = 0;
    [[nodiscard]] virtual reader_types::event_counts_t get_event_counts(
        const reader_types::time_window_t& window) = 0;

    [[nodiscard]] virtual uint32_t read_table_events(const resolved_event_query& query,
                                                     uint32_t        event_type,
                                                     uint64_t        after_id,
                                                     uint64_t        last_id,
                                                     ph_table_event* rows,
                                                     uint32_t        capacity) = 0;

    /** @brief Groups matching rows by display name inside SQLite.
     *         Returns false when this format has no summary query. An empty
     *         @p rows with true means the filter matched no events. */
    [[nodiscard]] virtual bool summarize_name_durations(
        const resolved_event_query&           query,
        uint32_t                              event_type,
        std::vector<name_duration_summary_t>& rows);
    [[nodiscard]] virtual bool table_event_id_span(uint32_t  event_type,
                                                   uint64_t& first_id,
                                                   uint64_t& last_id) = 0;

protected:
    // reader_t keeps the category enum private. Formats name it through here.
    using catalog_category_t = reader_t::catalog_category_t;

    impl(std::unique_ptr<profiler_hub::storage_t> storage,
         std::shared_ptr<reader_catalog_t>        catalog);

    /** @brief Opens this reader's own read connection to a SQLite trace. */
    [[nodiscard]] std::shared_ptr<data_storage::sqlite_backend> open_sqlite() const;

    [[nodiscard]] std::string database_path() const;

    // Applies limit/offset to merged event list
    static void apply_pagination(reader_types::timeline_event_list_t& events,
                                 const reader_types::pagination_t&    pagination);

    std::unique_ptr<profiler_hub::storage_t> m_storage;
    std::shared_ptr<reader_catalog_t>        m_catalog;
};

}  // namespace profiler_hub
