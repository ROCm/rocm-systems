// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "formats/rocpd/rocpd_source.hpp"

#include "catalog_aggregate_store.hpp"
#include "populate_reader_catalog.hpp"

#include <cstddef>
#include <utility>

namespace profiler_hub
{
namespace
{
constexpr size_t k_read_connections = 8;
}  // namespace

rocpd_source::rocpd_source(const source_request_t& request)
: format_source{ "rocpd" }
, m_trace_format{ request.format }
, m_catalog{ std::make_shared<reader_catalog_t>() }
, m_pool{ request.path, k_read_connections, m_catalog, request.uuid, request.format }
, m_connections{ m_pool }
{}

const std::shared_ptr<reader_catalog_t>&
rocpd_source::catalog() const
{
    return m_catalog;
}

void
rocpd_source::load_catalog(common::thread_pool& workers, const catalog_cache_t& cache)
{
    const bool legacy_rocpd = m_trace_format.version.major == 2;
    const bool modern_rocpd = m_trace_format.version.major >= 3;
    if(legacy_rocpd || modern_rocpd)
    {
        m_catalog->aggregate_cache_root   = cache.root;
        m_catalog->aggregate_cache_paths  = cache.paths;
        m_catalog->aggregate_cache_source = cache.index;
    }
    populate_reader_catalog(workers, m_pool, *m_catalog);
    if(legacy_rocpd && !m_catalog->derived_aggregates_ready)
    {
        save_catalog_aggregates(*m_catalog);
    }
    if(modern_rocpd && !m_catalog->derived_aggregates_ready)
    {
        save_derived_tracks(*m_catalog);
    }
    m_catalog->publish_memory_activity_pmc_infos();
}

connection_source&
rocpd_source::track_reads()
{
    return m_connections;
}

reader_types::track_info_ptr_t
rocpd_source::track_of(const reader_types::timeline_event_t& event)
{
    return m_pool.with_connection([&event](common::connection& connection) {
        return connection.reader().track_of(event);
    });
}

reader_types::timeline_event_list_t
rocpd_source::correlated_events(const reader_types::timeline_event_t& event)
{
    return m_pool.with_connection([&event](common::connection& connection) {
        return connection.reader().get_correlated_events(event);
    });
}

reader_types::call_stack_t
rocpd_source::call_stack(const reader_types::timeline_event_t& event)
{
    return m_pool.with_connection([&event](common::connection& connection) {
        return connection.reader().get_call_stack(event);
    });
}

std::optional<reader_types::region_data_t>
rocpd_source::region_details(const reader_types::timeline_event_t& event)
{
    return m_pool.with_connection([&event](common::connection& connection) {
        return connection.reader().get_region_details(event);
    });
}

std::optional<reader_types::kernel_dispatch_data_t>
rocpd_source::kernel_dispatch_details(const reader_types::timeline_event_t& event)
{
    return m_pool.with_connection([&event](common::connection& connection) {
        return connection.reader().get_kernel_dispatch_details(event);
    });
}

std::optional<reader_types::memory_copy_data_t>
rocpd_source::memory_copy_details(const reader_types::timeline_event_t& event)
{
    return m_pool.with_connection([&event](common::connection& connection) {
        return connection.reader().get_memory_copy_details(event);
    });
}

std::optional<reader_types::memory_alloc_data_t>
rocpd_source::memory_alloc_details(const reader_types::timeline_event_t& event)
{
    return m_pool.with_connection([&event](common::connection& connection) {
        return connection.reader().get_memory_alloc_details(event);
    });
}

bool
rocpd_source::summarize_name_durations(const resolved_event_query&           query,
                                       uint32_t                              event_type,
                                       std::vector<name_duration_summary_t>& rows)
{
    return m_pool.with_connection([&](common::connection& connection) {
        return connection.reader().summarize_name_durations(query, event_type, rows);
    });
}

bool
rocpd_source::supports_table_events() const
{
    return true;
}

uint32_t
rocpd_source::read_table_events(const resolved_event_query& query,
                                uint32_t                    event_type,
                                uint64_t                    after_id,
                                uint64_t                    last_id,
                                ph_table_event*             rows,
                                uint32_t                    capacity)
{
    return m_pool.with_connection([&](common::connection& connection) {
        return connection.reader().read_table_events(
            query, event_type, after_id, last_id, rows, capacity);
    });
}

bool
rocpd_source::table_event_id_span(uint32_t  event_type,
                                  uint64_t& first_id,
                                  uint64_t& last_id)
{
    return m_pool.with_connection([&](common::connection& connection) {
        return connection.reader().table_event_id_span(event_type, first_id, last_id);
    });
}

reader_types::histogram_t
rocpd_source::histogram(const histogram_grid_t& grid)
{
    return m_pool.with_connection([&grid](common::connection& connection) {
        return connection.reader().get_histogram(
            grid.origin, grid.bucket_size, grid.bucket_count);
    });
}

}  // namespace profiler_hub
