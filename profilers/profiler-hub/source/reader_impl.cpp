// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "reader_impl.hpp"
#include "json_serializers.hpp"
#include "profiler-hub/cpp/reader.hpp"
#include "profiler-hub/cpp/storage.hpp"
#include "storage_impl.hpp"

#include "queries/select/table_select_query.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

namespace profiler_hub
{

namespace
{

class timeline_event_builder
{
public:
    explicit timeline_event_builder(const reader_catalog_t& catalog)
    : m_catalog{ catalog }
    {}

    [[nodiscard]] reader_types::timeline_event_t operator()(
        const data_storage::schema_v3::timeline_event_result& row,
        reader_types::event_type_t                            type)
    {
        reader_types::timeline_event_t event;
        event.unique_identifier = { .id = row.id, .type = type };
        event.start_timestamp   = row.start_timestamp;
        event.end_timestamp     = row.end_timestamp;
        event.display_name      = resolve_string(row.display_name_id, m_display_name);
        event.category          = resolve_string(row.category_id, m_category);
        event.track             = resolve_track(row, type);
        return event;
    }

private:
    struct cached_string
    {
        std::optional<size_t> id;
        std::string_view      value;
    };

    std::string_view resolve_string(const std::optional<size_t>& id,
                                    cached_string&               cache) const
    {
        if(!id.has_value()) return {};
        if(cache.id == id) return cache.value;

        const auto it = m_catalog.strings_by_id.find(id.value());
        cache         = { id,
                  it != m_catalog.strings_by_id.end() ? std::string_view{ it->second }
                                                              : std::string_view{} };
        return cache.value;
    }

    reader_types::track_info_ptr_t resolve_track(
        const data_storage::schema_v3::timeline_event_result& row,
        reader_types::event_type_t                            type)
    {
        const auto kind = agent_queue_kind(type);
        if(kind.has_value())
        {
            const auto it = m_catalog.agent_queue_track_by_key.find(
                agent_queue_track_key_t{ kind.value(),
                                         row.nid,
                                         row.agent_id.value_or(0),
                                         row.queue_id.value_or(0) });
            if(it != m_catalog.agent_queue_track_by_key.end() && it->second)
            {
                return it->second;
            }
        }

        if(type != reader_types::event_type_t::region && row.stream_id.has_value())
        {
            const auto it = m_catalog.stream_track_by_key.find(stream_track_key_t{
                row.nid, row.pid.value_or(0), row.stream_id.value() });
            if(it != m_catalog.stream_track_by_key.end() && it->second)
            {
                return it->second;
            }
        }

        if(row.track_id.has_value())
        {
            const auto it = m_catalog.sample_track_by_db_id.find(row.track_id.value());
            if(it != m_catalog.sample_track_by_db_id.end() && it->second)
            {
                return it->second;
            }
        }

        const topology_key_t topology{ row.nid,
                                       row.pid.value_or(0),
                                       row.tid.value_or(0) };
        if(m_last_topology != topology)
        {
            const auto it = m_catalog.track_by_topology.find(topology);
            m_last_topology_track =
                it != m_catalog.track_by_topology.end() ? it->second : nullptr;
            m_last_topology = topology;
        }
        return m_last_topology_track;
    }

    static std::optional<reader_types::track_kind_t> agent_queue_kind(
        reader_types::event_type_t type)
    {
        using reader_types::event_type_t;
        using reader_types::track_kind_t;
        switch(type)
        {
            case event_type_t::kernel_dispatch:
                return track_kind_t::kernel_dispatch_agent_queue;
            case event_type_t::memory_copy: return track_kind_t::memory_copy_agent_queue;
            case event_type_t::memory_allocate:
                return track_kind_t::memory_allocate_agent_queue;
            case event_type_t::region:
            case event_type_t::sample:
            case event_type_t::pmc_event: break;
        }
        return std::nullopt;
    }

    const reader_catalog_t&        m_catalog;
    cached_string                  m_display_name;
    cached_string                  m_category;
    std::optional<topology_key_t>  m_last_topology;
    reader_types::track_info_ptr_t m_last_topology_track;
};

using reader_types::event_type_t;

constexpr std::array event_types{ event_type_t::region,
                                  event_type_t::kernel_dispatch,
                                  event_type_t::memory_allocate,
                                  event_type_t::memory_copy };

struct time_range
{
    size_t start;
    size_t end;
};

std::optional<time_range>
range_of(const reader_types::time_window_t& window)
{
    if(!window.start.has_value() || !window.end.has_value()) return std::nullopt;
    return time_range{ window.start.value(), window.end.value() };
}

template <typename Fn>
void
for_each_selected_type(const reader_types::event_filter_t& filter, Fn&& fn)
{
    for(const auto type : event_types)
    {
        if(filter.types.empty() ||
           std::find(filter.types.begin(), filter.types.end(), type) !=
               filter.types.end())
        {
            fn(type);
        }
    }
}

const data_storage::schema_v3::read_statements::timeline_event_statement_set&
statements_for(const data_storage::schema_v3::read_statements& stmts,
               reader_types::event_type_t                      type)
{
    if(type == reader_types::event_type_t::kernel_dispatch)
        return stmts.kernel_dispatch_statements();
    if(type == reader_types::event_type_t::memory_allocate)
        return stmts.memory_allocate_statements();
    if(type == reader_types::event_type_t::memory_copy)
        return stmts.memory_copy_statements();
    return stmts.region_statements();
}

bool
serves_events_of(reader_types::track_kind_t kind, event_type_t type)
{
    switch(kind)
    {
        case reader_types::track_kind_t::kernel_dispatch_agent_queue:
            return type == event_type_t::kernel_dispatch;
        case reader_types::track_kind_t::memory_allocate_agent_queue:
            return type == event_type_t::memory_allocate;
        case reader_types::track_kind_t::memory_copy_agent_queue:
            return type == event_type_t::memory_copy;
        case reader_types::track_kind_t::stream: return type != event_type_t::region;
        case reader_types::track_kind_t::thread:
        case reader_types::track_kind_t::thread_sample:
        case reader_types::track_kind_t::pmc_agent:
        case reader_types::track_kind_t::pmc_kernel:
        case reader_types::track_kind_t::pmc_memory: return false;
    }
    return false;
}

template <typename Base, typename Timed>
size_t
run_count(const Base& base, const Timed& timed, const std::optional<time_range>& range)
{
    auto results =
        range ? timed(range->end, range->start).to_vector() : base().to_vector();
    return results.empty() ? 0 : results.front().count;
}

size_t
count_events(const data_storage::schema_v3::read_statements& stmts,
             event_type_t                                    type,
             const std::optional<time_range>&                range)
{
    switch(type)
    {
        case event_type_t::region:
            return run_count(
                stmts.region_count(), stmts.region_count_time_filtered(), range);
        case event_type_t::kernel_dispatch:
            return run_count(stmts.kernel_dispatch_count(),
                             stmts.kernel_dispatch_count_time_filtered(),
                             range);
        case event_type_t::memory_copy:
            return run_count(stmts.memory_copy_count(),
                             stmts.memory_copy_count_time_filtered(),
                             range);
        case event_type_t::memory_allocate:
            return run_count(stmts.memory_alloc_count(),
                             stmts.memory_alloc_count_time_filtered(),
                             range);
        default: return 0;
    }
}

}  // namespace

reader_t::impl::impl(std::unique_ptr<profiler_hub::storage_t> storage)
: m_storage(storage ? std::move(storage)
                    : throw std::invalid_argument(
                          "Provided pointer to a non-existing storage!"))
, m_backend(m_storage->m_impl->create_database(storage_t::impl::storage_type_t::read))
, m_read_statements(
      std::make_shared<data_storage::schema_v3::read_statements>(m_backend,
                                                                 m_backend->get_uuid()))
, m_catalog(std::make_shared<reader_catalog_t>())
{
    ensure_track_topology_indexes();
    m_catalog->build_all(*m_read_statements);
}

reader_t::impl::impl(std::unique_ptr<profiler_hub::storage_t> storage,
                     std::shared_ptr<reader_catalog_t>        catalog)
: m_storage(storage ? std::move(storage)
                    : throw std::invalid_argument(
                          "Provided pointer to a non-existing storage!"))
, m_backend(m_storage->m_impl->create_database(storage_t::impl::storage_type_t::read))
, m_read_statements(
      std::make_shared<data_storage::schema_v3::read_statements>(m_backend,
                                                                 m_backend->get_uuid()))
, m_catalog(std::move(catalog))
{}

void
reader_t::impl::ensure_track_topology_indexes()
{
    m_read_statements->create_track_topology_indexes();
}

void
reader_t::impl::build_catalog_category(reader_t::catalog_category_t category,
                                       reader_catalog_t&            catalog)
{
    switch(category)
    {
        case reader_t::catalog_category_t::string_list:
            catalog.build_string_list(*m_read_statements);
            break;
        case reader_t::catalog_category_t::nodes:
            catalog.build_nodes(*m_read_statements);
            break;
        case reader_t::catalog_category_t::processes:
            catalog.build_processes(*m_read_statements);
            break;
        case reader_t::catalog_category_t::threads:
            catalog.build_threads(*m_read_statements);
            break;
        case reader_t::catalog_category_t::agents:
            catalog.build_agents(*m_read_statements);
            break;
        case reader_t::catalog_category_t::tracks:
            catalog.build_tracks(*m_read_statements);
            break;
        case reader_t::catalog_category_t::stream_hardware_links:
            catalog.build_stream_hardware_links(*m_read_statements);
            break;
        case reader_t::catalog_category_t::code_objects:
            catalog.build_code_objects(*m_read_statements);
            break;
        case reader_t::catalog_category_t::kernel_symbols:
            catalog.build_kernel_symbols(*m_read_statements);
            break;
        case reader_t::catalog_category_t::streams:
            catalog.build_streams(*m_read_statements);
            break;
        case reader_t::catalog_category_t::queues:
            catalog.build_queues(*m_read_statements);
            break;
        case reader_t::catalog_category_t::pmc_infos:
            catalog.build_pmc_infos(*m_read_statements);
            break;
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

reader_types::timeline_event_list_t
reader_t::impl::get_events(const reader_types::event_filter_t& filter)
{
    reader_types::timeline_event_list_t all_events;

    const auto             range = range_of(filter.time_window);
    timeline_event_builder build_event{ *m_catalog };

    for_each_selected_type(filter, [&](event_type_t type) {
        const auto append =
            [&](const data_storage::schema_v3::timeline_event_result& row) {
                all_events.push_back(build_event(row, type));
            };

        const auto& stmts = statements_for(*m_read_statements, type);
        if(range)
        {
            stmts.time_filtered(range->end, range->start).for_each(append);
        }
        else
        {
            stmts.base().for_each(append);
        }
    });

    apply_pagination(all_events, filter.pagination);
    return all_events;
}

reader_types::timeline_event_list_t
reader_t::impl::get_events_for_track(reader_types::track_info_ptr_t      track,
                                     const reader_types::event_filter_t& filter)
{
    if(!track) return {};

    switch(track->category)
    {
        case reader_types::track_kind_t::kernel_dispatch_agent_queue:
        case reader_types::track_kind_t::memory_allocate_agent_queue:
        case reader_types::track_kind_t::memory_copy_agent_queue:
        case reader_types::track_kind_t::stream:
            return get_category_track_events(track, filter);
        case reader_types::track_kind_t::thread:
        case reader_types::track_kind_t::thread_sample:
        case reader_types::track_kind_t::pmc_agent:
        case reader_types::track_kind_t::pmc_kernel:
        case reader_types::track_kind_t::pmc_memory: break;
    }

    auto topo_it = m_catalog->topology_by_track.find(track);
    if(topo_it == m_catalog->topology_by_track.end()) return {};

    auto db_id_it = m_catalog->db_id_by_track.find(track);
    if(db_id_it == m_catalog->db_id_by_track.end()) return {};

    const auto& topo  = topo_it->second;
    auto        db_id = db_id_it->second;

    reader_types::timeline_event_list_t all_events;

    const auto range = range_of(filter.time_window);
    if(!range)
    {
        all_events.reserve(track->event_count);
    }

    timeline_event_builder build_event{ *m_catalog };

    for_each_selected_type(filter, [&](event_type_t type) {
        const auto append =
            [&](const data_storage::schema_v3::timeline_event_result& row) {
                all_events.push_back(build_event(row, type));
            };

        const auto& stmts = statements_for(*m_read_statements, type);
        if(range)
        {
            stmts
                .track_and_time_filtered(topo.nid,
                                         topo.pid,
                                         topo.tid,
                                         range->end,
                                         range->start,
                                         db_id,
                                         range->end,
                                         range->start)
                .for_each(append);
        }
        else
        {
            stmts.track_filtered(topo.nid, topo.pid, topo.tid, db_id).for_each(append);
        }
    });

    apply_pagination(all_events, filter.pagination);
    return all_events;
}

std::optional<std::pair<size_t, size_t>>
reader_t::impl::get_event_id_span(reader_types::event_type_t type)
{
    const auto rows = statements_for(*m_read_statements, type).id_span().to_vector();
    if(rows.empty() || !rows.front().min_id.has_value() ||
       !rows.front().max_id.has_value())
        return std::nullopt;
    return std::pair{ rows.front().min_id.value(), rows.front().max_id.value() };
}

void
reader_t::impl::visit_track_events_in_id_range(
    const reader_types::track_info_ptr_t& track,
    reader_types::event_type_t            type,
    size_t                                id_begin,
    size_t                                id_end,
    reader_t::event_visitor_t             visitor,
    void*                                 context)
{
    const auto topo_it = m_catalog->topology_by_track.find(track);
    if(topo_it == m_catalog->topology_by_track.end()) return;
    const auto& topo = topo_it->second;

    std::optional<size_t>  last_id;
    std::string_view       last_name;
    timeline_event_builder build_event{ *m_catalog };

    statements_for(*m_read_statements, type)
        .track_range_filtered(topo.nid, topo.pid, topo.tid, id_begin, id_end)
        .for_each([&](const data_storage::schema_v3::event_range_result& row) {
            data_storage::schema_v3::timeline_event_result located;
            located.id              = row.id;
            located.start_timestamp = row.start_timestamp;
            located.end_timestamp   = row.end_timestamp;
            located.display_name_id = row.display_name_id;
            located.nid             = topo.nid;
            located.pid             = topo.pid;
            located.tid             = topo.tid;
            located.agent_id        = row.agent_id;
            located.queue_id        = row.queue_id;
            located.stream_id       = row.stream_id;
            const auto resolved     = build_event(located, type);
            if(resolved.track && resolved.track->id != track->id) return;

            std::string_view name;
            if(row.display_name_id.has_value())
            {
                if(last_id != row.display_name_id)
                {
                    const auto it = m_catalog->strings_by_id.find(*row.display_name_id);
                    last_name     = it != m_catalog->strings_by_id.end()
                                        ? std::string_view{ it->second }
                                        : std::string_view{};
                    last_id       = row.display_name_id;
                }
                name = last_name;
            }
            visitor(
                context, { row.id, type }, row.start_timestamp, row.end_timestamp, name);
        });
}

reader_types::timeline_event_list_t
reader_t::impl::get_category_track_events(const reader_types::track_info_ptr_t& track,
                                          const reader_types::event_filter_t&   filter)
{
    if(!track->node_info) return {};
    const auto nid = track->node_info->node_id;

    reader_types::timeline_event_list_t all_events;

    const auto range     = range_of(filter.time_window);
    const bool is_stream = track->category == reader_types::track_kind_t::stream;
    timeline_event_builder build_event{ *m_catalog };

    for_each_selected_type(filter, [&](event_type_t type) {
        if(!serves_events_of(track->category, type)) return;

        const auto append =
            [&](const data_storage::schema_v3::timeline_event_result& row) {
                all_events.push_back(build_event(row, type));
            };

        const auto& stmts = statements_for(*m_read_statements, type);
        if(is_stream)
        {
            if(range && stmts.stream_time_filtered)
            {
                stmts
                    .stream_time_filtered(
                        nid, track->db_pid, track->stream_id, range->end, range->start)
                    .for_each(append);
            }
            else if(!range && stmts.stream_filtered)
            {
                stmts.stream_filtered(nid, track->db_pid, track->stream_id)
                    .for_each(append);
            }
        }
        else if(range)
        {
            const bool unrecorded = track->agent_id == 0 || track->queue_id == 0;
            if(unrecorded && stmts.agent_queue_unrecorded_time_filtered)
            {
                stmts
                    .agent_queue_unrecorded_time_filtered(
                        nid, track->agent_id, track->queue_id, range->end, range->start)
                    .for_each(append);
            }
            else if(stmts.agent_queue_time_filtered)
            {
                stmts
                    .agent_queue_time_filtered(
                        nid, track->agent_id, track->queue_id, range->end, range->start)
                    .for_each(append);
            }
        }
        else if(stmts.agent_queue_filtered)
        {
            const bool unrecorded = track->agent_id == 0 || track->queue_id == 0;
            if(unrecorded && stmts.agent_queue_unrecorded_filtered)
            {
                stmts
                    .agent_queue_unrecorded_filtered(
                        nid, track->agent_id, track->queue_id)
                    .for_each(append);
            }
            else
            {
                stmts.agent_queue_filtered(nid, track->agent_id, track->queue_id)
                    .for_each(append);
            }
        }
    });

    apply_pagination(all_events, filter.pagination);
    return all_events;
}

reader_types::counter_timeline_event_list_t
reader_t::impl::get_counter_events_for_track(reader_types::track_info_ptr_t      track,
                                             const reader_types::event_filter_t& filter)
{
    if(!track || !track->node_info) return {};
    if(track->category != reader_types::track_kind_t::pmc_agent &&
       track->category != reader_types::track_kind_t::pmc_memory)
    {
        return {};
    }

    const auto nid = track->node_info->node_id;

    const bool has_time =
        filter.time_window.start.has_value() && filter.time_window.end.has_value();

    reader_types::counter_timeline_event_list_t events;
    if(track->category == reader_types::track_kind_t::pmc_memory)
    {
        if(m_catalog == nullptr) return {};
        const std::vector<memory_activity_point_t>* points =
            m_catalog->memory_activity_points(track->id);
        if(points == nullptr) return {};
        events.reserve(points->size());
        for(const memory_activity_point_t& point : *points)
        {
            if(has_time &&
               (point.start < filter.time_window.start.value() ||
                point.start > filter.time_window.end.value()))
            {
                continue;
            }
            events.push_back(
                reader_types::counter_timeline_event_t{ .unique_identifier = {},
                                                        .timestamp         = point.start,
                                                        .value             = point.value,
                                                        .track             = track });
        }
        return events;
    }

    std::vector<data_storage::schema_v3::pmc_sample_result> results;
    if(has_time)
    {
        results = m_read_statements
                      ->pmc_sample_time_filtered_statement()(nid,
                                                             track->agent_id,
                                                             track->pmc_id,
                                                             filter.time_window.start.value(),
                                                             filter.time_window.end.value())
                      .to_vector();
    }
    else
    {
        results = m_read_statements
                      ->pmc_sample_statement()(nid, track->agent_id, track->pmc_id)
                      .to_vector();
    }

    events.reserve(results.size());
    for(const auto& result : results)
    {
        events.push_back(
            reader_types::counter_timeline_event_t{ .unique_identifier = {},
                                                    .timestamp         = result.timestamp,
                                                    .value             = result.value,
                                                    .track             = track });
    }

    return events;
}

size_t
reader_t::impl::get_event_count(const reader_types::event_filter_t& filter)
{
    const auto range = range_of(filter.time_window);

    size_t total = 0;
    for_each_selected_type(filter, [&](event_type_t type) {
        total += count_events(*m_read_statements, type, range);
    });
    return total;
}

// ============================================================================
// Event metadata resolution helpers
// ============================================================================

std::optional<data_storage::schema_v3::event_id_result>
reader_t::impl::resolve_event_metadata(const reader_types::timeline_event_t& event)
{
    auto db_id = event.unique_identifier.id;

    std::vector<data_storage::schema_v3::event_id_result> results;
    switch(event.unique_identifier.type)
    {
        case reader_types::event_type_t::region:
            results = m_read_statements->region_event_id()(db_id).to_vector();
            break;
        case reader_types::event_type_t::kernel_dispatch:
            results = m_read_statements->kernel_dispatch_event_id()(db_id).to_vector();
            break;
        case reader_types::event_type_t::memory_copy:
            results = m_read_statements->memory_copy_event_id()(db_id).to_vector();
            break;
        case reader_types::event_type_t::memory_allocate:
            results = m_read_statements->memory_alloc_event_id()(db_id).to_vector();
            break;
        default: return std::nullopt;
    }

    if(results.empty()) return std::nullopt;
    return results.front();
}

reader_types::event_data_ptr_t
reader_t::impl::build_event_data(
    const data_storage::schema_v3::event_id_result& event_meta)
{
    auto event_data             = std::make_shared<reader_types::event_data_t>();
    event_data->stack_id        = event_meta.stack_id.value_or(0);
    event_data->parent_stack_id = event_meta.parent_stack_id.value_or(0);
    event_data->correlation_id  = event_meta.correlation_id.value_or(0);
    event_data->extdata         = event_meta.event_extdata;

    if(event_meta.category_id.has_value())
    {
        auto it = m_catalog->strings_by_id.find(event_meta.category_id.value());
        if(it != m_catalog->strings_by_id.end())
        {
            event_data->event_category = it->second;
        }
    }

    event_data->call_stack =
        json_serializers::deserialize_call_stack(event_meta.call_stack);
    event_data->line_info_list =
        json_serializers::deserialize_source_context(event_meta.line_info);

    return event_data;
}

// ============================================================================
// Event detail methods
// ============================================================================

reader_types::track_info_ptr_t
reader_t::impl::track_of(const reader_types::timeline_event_t& event)
{
    const auto& statements =
        statements_for(*m_read_statements, event.unique_identifier.type);
    const auto rows = statements.by_id(event.unique_identifier.id).to_vector();
    if(rows.empty()) return {};

    timeline_event_builder build_event{ *m_catalog };
    return build_event(rows.front(), event.unique_identifier.type).track;
}

std::optional<reader_types::region_data_t>
reader_t::impl::get_region_details(const reader_types::timeline_event_t& event)
{
    if(event.unique_identifier.type != reader_types::event_type_t::region)
    {
        return std::nullopt;
    }

    auto results =
        m_read_statements->region_detail()(event.unique_identifier.id).to_vector();
    if(results.empty())
    {
        return std::nullopt;
    }

    const auto& r = results.front();

    reader_types::region_data_t data;
    data.start_timestamp = r.start;
    data.end_timestamp   = r.end;
    data.extdata         = r.extdata;

    if(r.name_id.has_value())
    {
        auto it = m_catalog->strings_by_id.find(r.name_id.value());
        if(it != m_catalog->strings_by_id.end())
        {
            data.name = it->second;
        }
    }

    if(r.event_id.has_value())
    {
        auto event_meta = resolve_event_metadata(event);
        if(event_meta.has_value())
        {
            data.event = build_event_data(event_meta.value());
        }
    }

    return data;
}

std::optional<reader_types::kernel_dispatch_data_t>
reader_t::impl::get_kernel_dispatch_details(const reader_types::timeline_event_t& event)
{
    if(event.unique_identifier.type != reader_types::event_type_t::kernel_dispatch)
        return std::nullopt;

    auto results = m_read_statements->kernel_dispatch_detail()(event.unique_identifier.id)
                       .to_vector();
    if(results.empty()) return std::nullopt;

    const auto& r = results.front();

    reader_types::kernel_dispatch_data_t data;
    data.dispatch_id          = r.dispatch_id;
    data.start_timestamp      = r.start;
    data.end_timestamp        = r.end;
    data.private_segment_size = r.private_segment_size.value_or(0);
    data.group_segment_size   = r.group_segment_size.value_or(0);
    data.workgroup_size_x     = r.workgroup_size_x;
    data.workgroup_size_y     = r.workgroup_size_y;
    data.workgroup_size_z     = r.workgroup_size_z;
    data.grid_size_x          = r.grid_size_x;
    data.grid_size_y          = r.grid_size_y;
    data.grid_size_z          = r.grid_size_z;
    data.extdata              = r.extdata;

    if(r.region_name_id.has_value())
    {
        auto it = m_catalog->strings_by_id.find(r.region_name_id.value());
        if(it != m_catalog->strings_by_id.end()) data.name = it->second;
    }

    if(r.kernel_id.has_value())
    {
        auto it = m_catalog->kernel_symbols_by_id.find(r.kernel_id.value());
        if(it != m_catalog->kernel_symbols_by_id.end())
        {
            data.kernel_symbol_info = it->second;
            if(it->second && it->second->code_object_info)
                data.code_object_info = it->second->code_object_info;
        }
    }

    auto node_it = m_catalog->nodes_by_id.find(r.nid);
    if(node_it != m_catalog->nodes_by_id.end()) data.node_info = node_it->second;

    if(r.pid.has_value())
    {
        auto it = m_catalog->processes_by_id.find(r.pid.value());
        if(it != m_catalog->processes_by_id.end()) data.process_info = it->second;
    }

    if(r.tid.has_value())
    {
        auto it = m_catalog->threads_by_id.find(r.tid.value());
        if(it != m_catalog->threads_by_id.end()) data.thread_info = it->second;
    }

    if(r.event_id.has_value())
    {
        auto event_meta = resolve_event_metadata(event);
        if(event_meta.has_value()) data.event = build_event_data(event_meta.value());
    }

    return data;
}

std::optional<reader_types::memory_copy_data_t>
reader_t::impl::get_memory_copy_details(const reader_types::timeline_event_t& event)
{
    if(event.unique_identifier.type != reader_types::event_type_t::memory_copy)
        return std::nullopt;

    auto results =
        m_read_statements->memory_copy_detail()(event.unique_identifier.id).to_vector();
    if(results.empty()) return std::nullopt;

    const auto& r = results.front();

    reader_types::memory_copy_data_t data;
    data.start_timestamp = r.start;
    data.end_timestamp   = r.end;
    data.dst_address     = r.dst_address;
    data.src_address     = r.src_address;
    data.size            = r.size;
    data.extdata         = r.extdata;

    if(r.name_id.has_value())
    {
        auto it = m_catalog->strings_by_id.find(r.name_id.value());
        if(it != m_catalog->strings_by_id.end()) data.name = it->second;
    }

    if(r.region_name_id.has_value())
    {
        auto it = m_catalog->strings_by_id.find(r.region_name_id.value());
        if(it != m_catalog->strings_by_id.end()) data.region_name = it->second;
    }

    if(r.dst_agent_id.has_value())
    {
        auto it = m_catalog->agents_by_id.find(r.dst_agent_id.value());
        if(it != m_catalog->agents_by_id.end()) data.dst_agent_id = it->second;
    }

    if(r.src_agent_id.has_value())
    {
        auto it = m_catalog->agents_by_id.find(r.src_agent_id.value());
        if(it != m_catalog->agents_by_id.end()) data.src_agent_id = it->second;
    }

    auto node_it = m_catalog->nodes_by_id.find(r.nid);
    if(node_it != m_catalog->nodes_by_id.end()) data.node_info = node_it->second;

    if(r.pid.has_value())
    {
        auto it = m_catalog->processes_by_id.find(r.pid.value());
        if(it != m_catalog->processes_by_id.end()) data.process_info = it->second;
    }

    if(r.tid.has_value())
    {
        auto it = m_catalog->threads_by_id.find(r.tid.value());
        if(it != m_catalog->threads_by_id.end()) data.thread_info = it->second;
    }

    if(r.event_id.has_value())
    {
        auto event_meta = resolve_event_metadata(event);
        if(event_meta.has_value()) data.event = build_event_data(event_meta.value());
    }

    return data;
}

std::optional<reader_types::memory_alloc_data_t>
reader_t::impl::get_memory_alloc_details(const reader_types::timeline_event_t& event)
{
    if(event.unique_identifier.type != reader_types::event_type_t::memory_allocate)
        return std::nullopt;

    auto results =
        m_read_statements->memory_alloc_detail()(event.unique_identifier.id).to_vector();
    if(results.empty()) return std::nullopt;

    const auto& r = results.front();

    reader_types::memory_alloc_data_t data;
    data.type            = r.type.value_or("");
    data.level           = r.level.value_or("");
    data.start_timestamp = r.start;
    data.end_timestamp   = r.end;
    data.address         = r.address;
    data.size            = r.size;
    data.extdata         = r.extdata;

    auto node_it = m_catalog->nodes_by_id.find(r.nid);
    if(node_it != m_catalog->nodes_by_id.end()) data.node_info = node_it->second;

    if(r.pid.has_value())
    {
        auto it = m_catalog->processes_by_id.find(r.pid.value());
        if(it != m_catalog->processes_by_id.end()) data.process_info = it->second;
    }

    if(r.tid.has_value())
    {
        auto it = m_catalog->threads_by_id.find(r.tid.value());
        if(it != m_catalog->threads_by_id.end()) data.thread_info = it->second;
    }

    if(r.event_id.has_value())
    {
        auto event_meta = resolve_event_metadata(event);
        if(event_meta.has_value()) data.event = build_event_data(event_meta.value());
    }

    return data;
}

// ============================================================================
// Event property methods
// ============================================================================

reader_types::call_stack_t
reader_t::impl::get_call_stack(const reader_types::timeline_event_t& event)
{
    using reader_types::event_type_t;
    if(event.unique_identifier.type != event_type_t::region) return {};

    const auto source_rows =
        m_read_statements->region_statements().by_id(event.unique_identifier.id).to_vector();
    if(source_rows.empty()) return {};

    timeline_event_builder build_event{ *m_catalog };
    const auto             source = build_event(source_rows.front(), event_type_t::region);
    if(!source.track) return {};

    reader_types::event_filter_t filter;
    filter.types.push_back(event_type_t::region);
    auto regions = get_events_for_track(source.track, filter);
    std::sort(
        regions.begin(),
        regions.end(),
        [](const reader_types::timeline_event_t& lhs,
           const reader_types::timeline_event_t& rhs) {
            return lhs.start_timestamp < rhs.start_timestamp;
        });

    struct active_region_t
    {
        const reader_types::timeline_event_t* event;
        size_t                                level;
    };

    std::vector<active_region_t> active;
    reader_types::call_stack_t   stack;
    for(const auto& region : regions)
    {
        auto it = active.begin();
        while(it != active.end())
        {
            if(region.start_timestamp >= it->event->end_timestamp)
            {
                it = active.erase(it);
            }
            else
            {
                ++it;
            }
        }

        size_t level = 0;
        for(const auto& ancestor : active)
        {
            if(region.end_timestamp < ancestor.event->start_timestamp) continue;
            if(level < ancestor.level) break;
            level = ancestor.level + 1;
        }

        if(region.unique_identifier.id == event.unique_identifier.id)
        {
            auto add_frame = [&](const reader_types::timeline_event_t& frame_event) {
                reader_types::stack_frame_t      frame;
                reader_types::program_counter_info_t program_counter;
                program_counter.event_id = frame_event.unique_identifier.id;
                program_counter.function = std::string(frame_event.display_name);

                const auto frame_meta = resolve_event_metadata(frame_event);
                if(frame_meta.has_value())
                {
                    program_counter.extdata = frame_meta->line_info;
                    const auto source_context =
                        json_serializers::deserialize_source_context(frame_meta->line_info);
                    if(!source_context.empty())
                    {
                        const auto& source_info = source_context.front();
                        if(source_info.program_counter.has_value())
                        {
                            program_counter.filename =
                                source_info.program_counter->filename;
                            program_counter.line_number =
                                source_info.program_counter->line_number;
                        }
                        else if(source_info.source_code.has_value())
                        {
                            program_counter.filename =
                                source_info.source_code->filename.value_or("");
                            program_counter.line_number =
                                source_info.source_code->starting_line_number;
                        }
                    }
                }

                frame.program_counter = std::move(program_counter);
                stack.push_back(std::move(frame));
            };

            add_frame(region);
            for(auto ancestor = active.rbegin(); ancestor != active.rend(); ++ancestor)
            {
                if(ancestor->level >= level) continue;
                add_frame(*ancestor->event);
            }
            return stack;
        }

        active.push_back({ &region, level });
        std::sort(
            active.begin(),
            active.end(),
            [](const active_region_t& lhs, const active_region_t& rhs) {
                return lhs.level < rhs.level;
            });
    }
    return {};
}

reader_types::source_context_list_t
reader_t::impl::get_source_context(const reader_types::timeline_event_t& event)
{
    auto event_meta = resolve_event_metadata(event);
    if(!event_meta.has_value()) return {};

    return json_serializers::deserialize_source_context(event_meta->line_info);
}

reader_types::arg_data_list_t
reader_t::impl::get_arguments(const reader_types::timeline_event_t& event)
{
    auto event_meta = resolve_event_metadata(event);
    if(!event_meta.has_value() || !event_meta->event_id.has_value()) return {};

    auto results =
        m_read_statements->arg_detail()(event_meta->event_id.value()).to_vector();

    reader_types::arg_data_list_t args;
    args.reserve(results.size());
    for(const auto& r : results)
    {
        auto arg      = std::make_shared<reader_types::arg_data_t>();
        arg->position = r.position;
        arg->type     = r.type;
        arg->name     = r.name;
        arg->value    = r.value;
        arg->extdata  = r.extdata;
        args.push_back(std::move(arg));
    }
    return args;
}

reader_types::timeline_event_list_t
reader_t::impl::get_correlated_events(const reader_types::timeline_event_t& event)
{
    static_assert(static_cast<int>(reader_types::event_type_t::region) == 0);
    static_assert(static_cast<int>(reader_types::event_type_t::kernel_dispatch) == 1);
    static_assert(static_cast<int>(reader_types::event_type_t::memory_copy) == 2);
    static_assert(static_cast<int>(reader_types::event_type_t::memory_allocate) == 3);

    const auto& stmts = m_read_statements->correlated_event_statements();
    const decltype(stmts.region)* statement = nullptr;
    switch(event.unique_identifier.type)
    {
        case reader_types::event_type_t::region:
            statement = &stmts.region;
            break;
        case reader_types::event_type_t::kernel_dispatch:
            statement = &stmts.kernel_dispatch;
            break;
        case reader_types::event_type_t::memory_copy:
            statement = &stmts.memory_copy;
            break;
        case reader_types::event_type_t::memory_allocate:
            statement = &stmts.memory_allocate;
            break;
        default: return {};
    }

    reader_types::timeline_event_list_t all_events;
    timeline_event_builder              build_event{ *m_catalog };

    (*statement)(event.unique_identifier.id)
        .for_each([&](const data_storage::schema_v3::correlated_event_row& row) {
            using reader_types::event_type_t;
            if(row.kind > static_cast<size_t>(event_type_t::memory_allocate)) return;

            data_storage::schema_v3::timeline_event_result mapped;
            mapped.id              = row.id;
            mapped.start_timestamp = row.start_timestamp;
            mapped.end_timestamp   = row.end_timestamp;
            mapped.display_name_id = row.display_name_id;
            mapped.category_id     = row.category_id;
            mapped.nid             = row.nid;
            mapped.pid             = row.pid;
            mapped.tid             = row.tid;
            mapped.track_id        = row.track_id;
            mapped.agent_id        = row.agent_id;
            mapped.queue_id        = row.queue_id;
            mapped.stream_id       = row.stream_id;
            all_events.push_back(
                build_event(mapped, static_cast<event_type_t>(row.kind)));
        });
    return all_events;
}

// ============================================================================
// Database metadata methods
// ============================================================================

reader_types::time_window_t
reader_t::impl::get_data_time_range()
{
    size_t global_min = std::numeric_limits<size_t>::max();
    size_t global_max = 0;

    auto process_range = [&](const auto& stmt) {
        auto results = stmt().to_vector();
        if(!results.empty())
        {
            if(results.front().min_start.has_value())
            {
                global_min = std::min(global_min, results.front().min_start.value());
            }
            if(results.front().max_end.has_value())
            {
                global_max = std::max(global_max, results.front().max_end.value());
            }
        }
    };

    process_range(m_read_statements->region_time_range());
    process_range(m_read_statements->kernel_dispatch_time_range());
    process_range(m_read_statements->memory_copy_time_range());
    process_range(m_read_statements->memory_alloc_time_range());

    reader_types::time_window_t window;
    if(global_min != std::numeric_limits<size_t>::max())
    {
        window.start = global_min;
        window.end   = global_max;
    }
    return window;
}

reader_types::event_counts_t
reader_t::impl::get_event_counts(const reader_types::time_window_t& window)
{
    const auto range = range_of(window);

    reader_types::event_counts_t counts;
    for(const auto type : event_types)
    {
        counts[type] = count_events(*m_read_statements, type, range);
    }
    return counts;
}

}  // namespace profiler_hub
