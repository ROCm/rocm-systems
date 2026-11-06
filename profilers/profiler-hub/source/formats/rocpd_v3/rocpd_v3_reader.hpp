// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "reader_impl.hpp"

#include "data_storage/backends/sqlite_backend.hpp"
#include "data_storage/read_statements.hpp"
#include "reader_catalog.hpp"

#include <memory>
#include <optional>

namespace profiler_hub
{
class trim_schema_t;

/** @brief Reader for rocpd schema 3: GUID-suffixed per-node tables. */
class rocpd_v3_reader final : public reader_t::impl
{
public:
    rocpd_v3_reader(std::unique_ptr<profiler_hub::storage_t> storage,
                    std::shared_ptr<reader_catalog_t>        catalog);

    void build_catalog_category(catalog_category_t category,
                                reader_catalog_t&  catalog) override;
    void ensure_track_topology_indexes() override;

    // Timeline event queries
    [[nodiscard]] reader_types::timeline_event_list_t get_events(
        const reader_types::event_filter_t& filter) override;

    [[nodiscard]] reader_types::timeline_event_list_t get_events_for_track(
        reader_types::track_info_ptr_t      track,
        const reader_types::event_filter_t& filter) override;

    [[nodiscard]] std::optional<std::pair<size_t, size_t>> get_event_id_span(
        reader_types::event_type_t type) override;

    void visit_track_events_in_id_range(const reader_types::track_info_ptr_t& track,
                                        reader_types::event_type_t            type,
                                        size_t                                id_begin,
                                        size_t                                id_end,
                                        reader_t::event_visitor_t             visitor,
                                        void* context) override;

    [[nodiscard]] reader_types::counter_timeline_event_list_t
    get_counter_events_for_track(reader_types::track_info_ptr_t      track,
                                 const reader_types::event_filter_t& filter) override;

    [[nodiscard]] size_t get_event_count(
        const reader_types::event_filter_t& filter) override;

    [[nodiscard]] reader_types::histogram_t get_histogram(size_t max_buckets) override;
    [[nodiscard]] reader_types::histogram_t get_histogram(
        const histogram_grid_t& grid) override;

    // Event detail queries
    [[nodiscard]] std::optional<reader_types::region_data_t> get_region_details(
        const reader_types::timeline_event_t& event) override;

    [[nodiscard]] std::optional<reader_types::kernel_dispatch_data_t>
    get_kernel_dispatch_details(const reader_types::timeline_event_t& event) override;

    [[nodiscard]] std::optional<reader_types::memory_copy_data_t> get_memory_copy_details(
        const reader_types::timeline_event_t& event) override;

    [[nodiscard]] std::optional<reader_types::memory_alloc_data_t>
    get_memory_alloc_details(const reader_types::timeline_event_t& event) override;

    [[nodiscard]] reader_types::track_info_ptr_t track_of(
        const reader_types::timeline_event_t& event) override;

    // Event property queries
    [[nodiscard]] reader_types::call_stack_t get_call_stack(
        const reader_types::timeline_event_t& event) override;

    [[nodiscard]] reader_types::source_context_list_t get_source_context(
        const reader_types::timeline_event_t& event) override;

    [[nodiscard]] reader_types::arg_data_list_t get_arguments(
        const reader_types::timeline_event_t& event) override;

    [[nodiscard]] reader_types::timeline_event_list_t get_correlated_events(
        const reader_types::timeline_event_t& event) override;

    // Database metadata
    [[nodiscard]] reader_types::time_window_t  get_data_time_range() override;
    [[nodiscard]] reader_types::event_counts_t get_event_counts(
        const reader_types::time_window_t& window) override;

    [[nodiscard]] uint32_t read_table_events(const resolved_event_query& query,
                                             uint32_t                    event_type,
                                             uint64_t                    after_id,
                                             uint64_t                    last_id,
                                             ph_table_event*             rows,
                                             uint32_t                    capacity) override;
    [[nodiscard]] bool     table_event_id_span(uint32_t  event_type,
                                               uint64_t& first_id,
                                               uint64_t& last_id) override;
    [[nodiscard]] bool     summarize_name_durations(
        const resolved_event_query&           query,
        uint32_t                              event_type,
        std::vector<name_duration_summary_t>& rows) override;

private:
    // Resolve event metadata from event-specific table by db_id and type.
    // Returns event_id_result containing event_id + stack_id + source context etc.
    [[nodiscard]] std::optional<data_storage::schema_v3::event_id_result>
    resolve_event_metadata(const reader_types::timeline_event_t& event);

    // Build event_data_t from event_id (queries rocpd_event, parses JSON)
    [[nodiscard]] reader_types::event_data_ptr_t build_event_data(
        const data_storage::schema_v3::event_id_result& event_meta);

    // Fetches events for the 4 optiq-parity category tracks (agent+queue
    // and stream), which have no rocpd_track row and so aren't in
    // m_track_ptr_to_topology/m_track_ptr_to_db_id.
    [[nodiscard]] reader_types::timeline_event_list_t get_category_track_events(
        const reader_types::track_info_ptr_t& track,
        const reader_types::event_filter_t&   filter);

    std::shared_ptr<data_storage::sqlite_backend>             m_backend;
    std::shared_ptr<data_storage::schema_v3::read_statements> m_read_statements;
};

[[nodiscard]] std::unique_ptr<reader_t::impl> make_rocpd_v3_reader(
    std::unique_ptr<profiler_hub::storage_t> storage,
    std::shared_ptr<reader_catalog_t>        catalog);

[[nodiscard]] std::unique_ptr<trim_schema_t> make_rocpd_v3_trim_schema();

}  // namespace profiler_hub
