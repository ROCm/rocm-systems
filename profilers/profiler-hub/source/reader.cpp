// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "profiler-hub/cpp/reader.hpp"
#include "profiler-hub/cpp/storage.hpp"
#include "formats/trace_formats.hpp"
#include "reader_impl.hpp"
#include "storage_impl.hpp"

#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <utility>

namespace profiler_hub
{

namespace
{
reader_types::trace_format_t
format_of(const std::string& path)
{
    const reader_types::file_probe_result_t probe = probe_file(path);
    if(probe.status == PH_RESULT_SUCCESS)
    {
        return { probe.trace_schema, probe.schema_version };
    }
#if PROFILER_HUB_ENABLE_ROCPD_V3
    // A path the probe does not recognize, such as a database the writer has
    // not flushed yet, keeps the rocpd schema 3 reader.
    return { reader_types::trace_schema_t::rocpd, version_t{ 3, 0, 0 } };
#else
    throw std::invalid_argument{ "trace format is not supported" };
#endif
}
}  // namespace

reader_t::reader_t(std::unique_ptr<profiler_hub::storage_t> storage)
{
    if(!storage)
    {
        throw std::invalid_argument("Provided pointer to a non-existing storage!");
    }
    const reader_types::trace_format_t format =
        format_of(storage->m_impl->get_database_path());
    m_impl = make_reader_impl(
        format, std::move(storage), std::make_shared<reader_catalog_t>());
    m_impl->build_catalog();
}

reader_t::reader_t(std::unique_ptr<profiler_hub::storage_t> storage,
                   std::shared_ptr<reader_catalog_t>        catalog,
                   const reader_types::trace_format_t&      format)
: m_impl(make_reader_impl(format, std::move(storage), std::move(catalog)))
{}

reader_t::~reader_t() = default;

// ============================================================================
// Format-neutral part of every read backend
// ============================================================================

reader_t::impl::impl(std::unique_ptr<profiler_hub::storage_t> storage,
                     std::shared_ptr<reader_catalog_t>        catalog)
: m_storage(storage ? std::move(storage)
                    : throw std::invalid_argument(
                          "Provided pointer to a non-existing storage!"))
, m_catalog(std::move(catalog))
{}

std::shared_ptr<data_storage::sqlite_backend>
reader_t::impl::open_sqlite() const
{
    return m_storage->m_impl->create_database(storage_t::impl::storage_type_t::read);
}

std::string
reader_t::impl::database_path() const
{
    return m_storage->m_impl->get_database_path();
}

void
reader_t::impl::build_catalog()
{
    ensure_track_topology_indexes();
    for(const catalog_category_t category : { catalog_category_t::string_list,
                                              catalog_category_t::nodes,
                                              catalog_category_t::processes,
                                              catalog_category_t::threads,
                                              catalog_category_t::agents,
                                              catalog_category_t::tracks,
                                              catalog_category_t::stream_hardware_links,
                                              catalog_category_t::code_objects,
                                              catalog_category_t::kernel_symbols,
                                              catalog_category_t::streams,
                                              catalog_category_t::queues,
                                              catalog_category_t::pmc_infos })
    {
        build_catalog_category(category, *m_catalog);
    }
}

reader_types::node_info_list_t
reader_t::impl::get_all_nodes()
{
    return m_catalog->nodes;
}

reader_types::process_info_list_t
reader_t::impl::get_all_processes()
{
    return m_catalog->processes;
}

reader_types::thread_info_list_t
reader_t::impl::get_all_threads()
{
    return m_catalog->threads;
}

reader_types::agent_info_list_t
reader_t::impl::get_all_agents()
{
    return m_catalog->agents;
}

reader_types::track_info_list_t
reader_t::impl::get_all_tracks()
{
    return m_catalog->tracks;
}

reader_types::stream_hardware_link_list_t
reader_t::impl::get_stream_hardware_links()
{
    return m_catalog->stream_hardware_links;
}

reader_types::kernel_symbol_info_list_t
reader_t::impl::get_all_kernel_symbols()
{
    return m_catalog->kernel_symbols;
}

reader_types::code_object_info_list_t
reader_t::impl::get_all_code_objects()
{
    return m_catalog->code_objects;
}

reader_types::stream_info_list_t
reader_t::impl::get_all_streams()
{
    return m_catalog->streams;
}

reader_types::queue_info_list_t
reader_t::impl::get_all_queues()
{
    return m_catalog->queues;
}

reader_types::pmc_info_list_t
reader_t::impl::get_all_pmc_infos()
{
    return m_catalog->pmc_infos;
}

std::optional<reader_types::region_data_t>
reader_t::impl::get_region_details(const reader_types::timeline_event_t&)
{
    return std::nullopt;
}

bool
reader_t::impl::summarize_name_durations(const resolved_event_query&,
                                         uint32_t,
                                         std::vector<name_duration_summary_t>& rows)
{
    rows.clear();
    return false;
}

std::optional<reader_types::kernel_dispatch_data_t>
reader_t::impl::get_kernel_dispatch_details(const reader_types::timeline_event_t&)
{
    return std::nullopt;
}

std::optional<reader_types::memory_copy_data_t>
reader_t::impl::get_memory_copy_details(const reader_types::timeline_event_t&)
{
    return std::nullopt;
}

std::optional<reader_types::memory_alloc_data_t>
reader_t::impl::get_memory_alloc_details(const reader_types::timeline_event_t&)
{
    return std::nullopt;
}

reader_types::call_stack_t
reader_t::impl::get_call_stack(const reader_types::timeline_event_t&)
{
    return {};
}

reader_types::source_context_list_t
reader_t::impl::get_source_context(const reader_types::timeline_event_t&)
{
    return {};
}

reader_types::arg_data_list_t
reader_t::impl::get_arguments(const reader_types::timeline_event_t&)
{
    return {};
}

reader_types::timeline_event_list_t
reader_t::impl::get_correlated_events(const reader_types::timeline_event_t&)
{
    return {};
}

void
reader_t::impl::apply_pagination(reader_types::timeline_event_list_t& events,
                                 const reader_types::pagination_t&    pagination)
{
    if(pagination.offset.has_value())
    {
        auto off = pagination.offset.value();
        if(off >= events.size())
        {
            events.clear();
            return;
        }
        events.erase(events.begin(), events.begin() + static_cast<ptrdiff_t>(off));
    }

    if(pagination.limit.has_value())
    {
        auto lim = pagination.limit.value();
        if(lim < events.size())
        {
            events.resize(lim);
        }
    }
}

// ============================================================================
// reader_t forwarding
// ============================================================================

void
reader_t::ensure_track_topology_indexes()
{
    m_impl->ensure_track_topology_indexes();
}

void
reader_t::build_catalog_category(catalog_category_t category, reader_catalog_t& catalog)
{
    m_impl->build_catalog_category(category, catalog);
}

reader_types::node_info_list_t
reader_t::get_all_nodes() const
{
    return m_impl->get_all_nodes();
}

reader_types::process_info_list_t
reader_t::get_all_processes() const
{
    return m_impl->get_all_processes();
}

reader_types::thread_info_list_t
reader_t::get_all_threads() const
{
    return m_impl->get_all_threads();
}

reader_types::agent_info_list_t
reader_t::get_all_agents() const
{
    return m_impl->get_all_agents();
}

reader_types::queue_info_list_t
reader_t::get_all_queues() const
{
    return m_impl->get_all_queues();
}

reader_types::stream_info_list_t
reader_t::get_all_streams() const
{
    return m_impl->get_all_streams();
}

reader_types::pmc_info_list_t
reader_t::get_all_pmc_info() const
{
    return m_impl->get_all_pmc_infos();
}

reader_types::code_object_info_list_t
reader_t::get_all_code_objects() const
{
    return m_impl->get_all_code_objects();
}

reader_types::kernel_symbol_info_list_t
reader_t::get_all_kernel_symbols() const
{
    return m_impl->get_all_kernel_symbols();
}

reader_types::track_info_list_t
reader_t::get_all_tracks() const
{
    return m_impl->get_all_tracks();
}

reader_types::stream_hardware_link_list_t
reader_t::get_stream_hardware_links() const
{
    return m_impl->get_stream_hardware_links();
}

reader_types::timeline_event_list_t
reader_t::get_events_for_track(reader_types::track_info_ptr_t      track,
                               const reader_types::event_filter_t& filter) const
{
    return m_impl->get_events_for_track(std::move(track), filter);
}

std::optional<std::pair<size_t, size_t>>
reader_t::get_event_id_span(reader_types::event_type_t type) const
{
    return m_impl->get_event_id_span(type);
}

void
reader_t::visit_track_events_in_id_range(const reader_types::track_info_ptr_t& track,
                                         reader_types::event_type_t            type,
                                         size_t                                id_begin,
                                         size_t                                id_end,
                                         event_visitor_t                       visitor,
                                         void* context) const
{
    m_impl->visit_track_events_in_id_range(
        track, type, id_begin, id_end, visitor, context);
}

reader_types::counter_timeline_event_list_t
reader_t::get_counter_events_for_track(reader_types::track_info_ptr_t      track,
                                       const reader_types::event_filter_t& filter) const
{
    return m_impl->get_counter_events_for_track(std::move(track), filter);
}

reader_types::timeline_event_list_t
reader_t::get_events(const reader_types::event_filter_t& filter) const
{
    return m_impl->get_events(filter);
}

size_t
reader_t::get_event_count(const reader_types::event_filter_t& filter) const
{
    return m_impl->get_event_count(filter);
}

std::optional<reader_types::region_data_t>
reader_t::get_region_details(const reader_types::timeline_event_t& event) const
{
    return m_impl->get_region_details(event);
}

std::optional<reader_types::kernel_dispatch_data_t>
reader_t::get_kernel_dispatch_details(const reader_types::timeline_event_t& event) const
{
    return m_impl->get_kernel_dispatch_details(event);
}

std::optional<reader_types::memory_copy_data_t>
reader_t::get_memory_copy_details(const reader_types::timeline_event_t& event) const
{
    return m_impl->get_memory_copy_details(event);
}

std::optional<reader_types::memory_alloc_data_t>
reader_t::get_memory_alloc_details(const reader_types::timeline_event_t& event) const
{
    return m_impl->get_memory_alloc_details(event);
}

reader_types::track_info_ptr_t
reader_t::track_of(const reader_types::timeline_event_t& event) const
{
    return m_impl->track_of(event);
}

std::optional<reader_types::sample_data_t>
reader_t::get_sample_details(const reader_types::timeline_event_t& event) const
{
    (void) event;
    return std::nullopt;
}

std::optional<reader_types::pmc_event_data_t>
reader_t::get_pmc_event_details(const reader_types::timeline_event_t& event) const
{
    (void) event;
    return std::nullopt;
}

reader_types::call_stack_t
reader_t::get_call_stack(const reader_types::timeline_event_t& event) const
{
    return m_impl->get_call_stack(event);
}

reader_types::source_context_list_t
reader_t::get_source_context(const reader_types::timeline_event_t& event) const
{
    return m_impl->get_source_context(event);
}

reader_types::arg_data_list_t
reader_t::get_arguments(const reader_types::timeline_event_t& event) const
{
    return m_impl->get_arguments(event);
}

reader_types::timeline_event_list_t
reader_t::get_correlated_events(const reader_types::timeline_event_t& event) const
{
    return m_impl->get_correlated_events(event);
}

reader_types::event_summary_list_t
reader_t::get_kernel_summary(const reader_types::time_window_t& window) const
{
    (void) window;
    return {};
}

reader_types::event_summary_list_t
reader_t::get_region_summary(const reader_types::time_window_t& window) const
{
    (void) window;
    return {};
}

reader_types::histogram_t
reader_t::get_histogram(size_t max_buckets) const
{
    return m_impl->get_histogram(max_buckets);
}

reader_types::histogram_t
reader_t::get_histogram(reader_types::timestamp_ns_t origin,
                        reader_types::timestamp_ns_t bucket_size,
                        size_t                       bucket_count) const
{
    return m_impl->get_histogram(histogram_grid_t{ origin, bucket_size, bucket_count });
}

reader_types::time_window_t
reader_t::get_data_time_range() const
{
    return m_impl->get_data_time_range();
}

reader_types::event_counts_t
reader_t::get_event_counts(const reader_types::time_window_t& window) const
{
    return m_impl->get_event_counts(window);
}

uint32_t
reader_t::read_table_events(const resolved_event_query& query,
                            uint32_t                    event_type,
                            uint64_t                    after_id,
                            uint64_t                    last_id,
                            ph_table_event*             rows,
                            uint32_t                    capacity)
{
    return m_impl->read_table_events(query, event_type, after_id, last_id, rows, capacity);
}

bool
reader_t::summarize_name_durations(const resolved_event_query&           query,
                                   uint32_t                              event_type,
                                   std::vector<name_duration_summary_t>& rows)
{
    return m_impl->summarize_name_durations(query, event_type, rows);
}

bool
reader_t::table_event_id_span(uint32_t event_type, uint64_t& first_id, uint64_t& last_id)
{
    return m_impl->table_event_id_span(event_type, first_id, last_id);
}

}  // namespace profiler_hub
