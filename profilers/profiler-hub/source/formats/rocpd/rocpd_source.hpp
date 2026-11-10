// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/connection_pool.hpp"
#include "format_source.hpp"
#include "pooled_connection_source.hpp"

#include <memory>

namespace profiler_hub
{

/** @brief A rocpd trace file read through a pool of SQLite connections. */
class rocpd_source final : public format_source
{
public:
    explicit rocpd_source(const source_request_t& request);

    [[nodiscard]] const std::shared_ptr<reader_catalog_t>& catalog() const override;
    void                             load_catalog(common::thread_pool&   workers,
                                                  const catalog_cache_t& cache) override;
    [[nodiscard]] connection_source& track_reads() override;

    [[nodiscard]] reader_types::track_info_ptr_t track_of(
        const reader_types::timeline_event_t& event) override;
    [[nodiscard]] reader_types::timeline_event_list_t correlated_events(
        const reader_types::timeline_event_t& event) override;
    [[nodiscard]] reader_types::call_stack_t call_stack(
        const reader_types::timeline_event_t& event) override;
    [[nodiscard]] std::optional<reader_types::region_data_t> region_details(
        const reader_types::timeline_event_t& event) override;
    [[nodiscard]] std::optional<reader_types::kernel_dispatch_data_t>
    kernel_dispatch_details(const reader_types::timeline_event_t& event) override;
    [[nodiscard]] std::optional<reader_types::memory_copy_data_t> memory_copy_details(
        const reader_types::timeline_event_t& event) override;
    [[nodiscard]] std::optional<reader_types::memory_alloc_data_t> memory_alloc_details(
        const reader_types::timeline_event_t& event) override;

    [[nodiscard]] bool summarize_name_durations(
        const resolved_event_query&           query,
        uint32_t                              event_type,
        std::vector<name_duration_summary_t>& rows) override;
    [[nodiscard]] bool     supports_table_events() const override;
    [[nodiscard]] uint32_t read_table_events(const resolved_event_query& query,
                                             uint32_t                    event_type,
                                             uint64_t                    after_id,
                                             uint64_t                    last_id,
                                             ph_table_event*             rows,
                                             uint32_t capacity) override;
    [[nodiscard]] bool     table_event_id_span(uint32_t  event_type,
                                               uint64_t& first_id,
                                               uint64_t& last_id) override;

    [[nodiscard]] reader_types::histogram_t histogram(
        const histogram_grid_t& grid) override;

private:
    reader_types::trace_format_t      m_trace_format;
    std::shared_ptr<reader_catalog_t> m_catalog;
    common::connection_pool           m_pool;
    pooled_connection_source          m_connections;
};

}  // namespace profiler_hub
