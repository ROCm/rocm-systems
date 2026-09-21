// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "reader_impl.hpp"
#include "json_serializers.hpp"
#include "profiler-hub/cpp/reader.hpp"
#include "profiler-hub/cpp/storage.hpp"
#include "storage_impl.hpp"

#include "queries/select/table_select_query.hpp"

#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace profiler_hub
{

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

reader_types::timeline_event_list_t
reader_t::impl::build_timeline_events(
    const std::vector<data_storage::schema_v3::timeline_event_result>& results,
    reader_types::event_type_t                                         type)
{
    reader_types::timeline_event_list_t events;
    events.reserve(results.size());

    for(const auto& result : results)
    {
        reader_types::timeline_event_t event;
        event.unique_identifier = { result.id, type };
        event.start_timestamp   = result.start_timestamp;
        event.end_timestamp     = result.end_timestamp;

        if(result.display_name_id.has_value())
        {
            auto it = m_catalog->string_utility.find(result.display_name_id.value());
            if(it != m_catalog->string_utility.end())
            {
                event.display_name = it->second;
            }
        }

        if(result.category_id.has_value())
        {
            auto it = m_catalog->string_utility.find(result.category_id.value());
            if(it != m_catalog->string_utility.end())
            {
                event.category = it->second;
            }
        }

        // Track resolution: try sample-based track_id first, fall back to topology
        if(result.track_id.has_value())
        {
            auto it = m_catalog->track_utility.find(result.track_id.value());
            if(it != m_catalog->track_utility.end())
            {
                event.track = it->second;
            }
        }

        if(!event.track)
        {
            topology_key_t topo{ result.nid,
                                 result.pid.value_or(0),
                                 result.tid.value_or(0) };
            auto           it = m_catalog->topology_to_track.find(topo);
            if(it != m_catalog->topology_to_track.end())
            {
                event.track = it->second;
            }
        }

        events.push_back(std::move(event));
    }

    return events;
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

    bool query_all    = filter.types.empty();
    auto should_query = [&](reader_types::event_type_t t) {
        return query_all || std::find(filter.types.begin(), filter.types.end(), t) !=
                                filter.types.end();
    };

    bool has_time =
        filter.time_window.start.has_value() && filter.time_window.end.has_value();

    auto query_event_type =
        [&](const data_storage::schema_v3::read_statements::timeline_event_statement_set&
                                       stmts,
            reader_types::event_type_t type) {
            std::vector<data_storage::schema_v3::timeline_event_result> results;
            if(has_time)
            {
                results = stmts
                              .time_filtered(filter.time_window.end.value(),
                                             filter.time_window.start.value())
                              .to_vector();
            }
            else
            {
                results = stmts.base().to_vector();
            }

            auto events = build_timeline_events(results, type);
            all_events.insert(all_events.end(),
                              std::make_move_iterator(events.begin()),
                              std::make_move_iterator(events.end()));
        };

    if(should_query(reader_types::event_type_t::region))
    {
        query_event_type(m_read_statements->region_statements(),
                         reader_types::event_type_t::region);
    }

    if(should_query(reader_types::event_type_t::kernel_dispatch))
    {
        query_event_type(m_read_statements->kernel_dispatch_statements(),
                         reader_types::event_type_t::kernel_dispatch);
    }

    if(should_query(reader_types::event_type_t::memory_allocate))
    {
        query_event_type(m_read_statements->memory_allocate_statements(),
                         reader_types::event_type_t::memory_allocate);
    }

    if(should_query(reader_types::event_type_t::memory_copy))
    {
        query_event_type(m_read_statements->memory_copy_statements(),
                         reader_types::event_type_t::memory_copy);
    }

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
        case reader_types::track_kind_t::pmc_agent: break;
    }

    auto topo_it = m_catalog->track_to_topology.find(track);
    if(topo_it == m_catalog->track_to_topology.end()) return {};

    auto db_id_it = m_catalog->track_to_db_id.find(track);
    if(db_id_it == m_catalog->track_to_db_id.end()) return {};

    const auto& topo  = topo_it->second;
    auto        db_id = db_id_it->second;

    reader_types::timeline_event_list_t all_events;

    bool query_all    = filter.types.empty();
    auto should_query = [&](reader_types::event_type_t t) {
        return query_all || std::find(filter.types.begin(), filter.types.end(), t) !=
                                filter.types.end();
    };

    bool has_time =
        filter.time_window.start.has_value() && filter.time_window.end.has_value();

    auto query_event_type =
        [&](const data_storage::schema_v3::read_statements::timeline_event_statement_set&
                                       stmts,
            reader_types::event_type_t type) {
            std::vector<data_storage::schema_v3::timeline_event_result> results;
            if(has_time)
            {
                const auto window_end   = filter.time_window.end.value();
                const auto window_start = filter.time_window.start.value();
                results                 = stmts
                              .track_and_time_filtered(topo.nid,
                                                       topo.pid,
                                                       topo.tid,
                                                       window_end,
                                                       window_start,
                                                       db_id,
                                                       window_end,
                                                       window_start)
                              .to_vector();
            }
            else
            {
                results =
                    stmts.track_filtered(topo.nid, topo.pid, topo.tid, db_id).to_vector();
            }

            auto events = build_timeline_events(results, type);
            all_events.insert(all_events.end(),
                              std::make_move_iterator(events.begin()),
                              std::make_move_iterator(events.end()));
        };

    if(should_query(reader_types::event_type_t::region))
    {
        query_event_type(m_read_statements->region_statements(),
                         reader_types::event_type_t::region);
    }

    if(should_query(reader_types::event_type_t::kernel_dispatch))
    {
        query_event_type(m_read_statements->kernel_dispatch_statements(),
                         reader_types::event_type_t::kernel_dispatch);
    }

    if(should_query(reader_types::event_type_t::memory_allocate))
    {
        query_event_type(m_read_statements->memory_allocate_statements(),
                         reader_types::event_type_t::memory_allocate);
    }

    if(should_query(reader_types::event_type_t::memory_copy))
    {
        query_event_type(m_read_statements->memory_copy_statements(),
                         reader_types::event_type_t::memory_copy);
    }

    apply_pagination(all_events, filter.pagination);
    return all_events;
}

reader_types::timeline_event_list_t
reader_t::impl::get_category_track_events(const reader_types::track_info_ptr_t& track,
                                          const reader_types::event_filter_t&   filter)
{
    if(!track->node_info) return {};
    const auto nid = track->node_info->node_id;

    reader_types::timeline_event_list_t all_events;

    bool query_all    = filter.types.empty();
    auto should_query = [&](reader_types::event_type_t t) {
        return query_all || std::find(filter.types.begin(), filter.types.end(), t) !=
                                filter.types.end();
    };

    const bool has_time =
        filter.time_window.start.has_value() && filter.time_window.end.has_value();
    const auto window_end   = has_time ? filter.time_window.end.value() : 0;
    const auto window_start = has_time ? filter.time_window.start.value() : 0;

    auto append_events = [&](const auto& results, reader_types::event_type_t type) {
        auto events = build_timeline_events(results, type);
        all_events.insert(all_events.end(),
                          std::make_move_iterator(events.begin()),
                          std::make_move_iterator(events.end()));
    };

    auto query_agent_queue =
        [&](const data_storage::schema_v3::read_statements::timeline_event_statement_set&
                                       stmts,
            reader_types::event_type_t type) {
            if(has_time)
            {
                if(!stmts.agent_queue_time_filtered) return;
                append_events(stmts
                                  .agent_queue_time_filtered(nid,
                                                             track->agent_id,
                                                             track->queue_id,
                                                             window_end,
                                                             window_start)
                                  .to_vector(),
                              type);
            }
            else
            {
                if(!stmts.agent_queue_filtered) return;
                append_events(
                    stmts.agent_queue_filtered(nid, track->agent_id, track->queue_id)
                        .to_vector(),
                    type);
            }
        };

    auto query_stream =
        [&](const data_storage::schema_v3::read_statements::timeline_event_statement_set&
                                       stmts,
            reader_types::event_type_t type) {
            if(has_time)
            {
                if(!stmts.stream_time_filtered) return;
                append_events(stmts
                                  .stream_time_filtered(nid,
                                                        track->db_pid,
                                                        track->stream_id,
                                                        window_end,
                                                        window_start)
                                  .to_vector(),
                              type);
            }
            else
            {
                if(!stmts.stream_filtered) return;
                append_events(stmts.stream_filtered(nid, track->db_pid, track->stream_id)
                                  .to_vector(),
                              type);
            }
        };

    switch(track->category)
    {
        case reader_types::track_kind_t::kernel_dispatch_agent_queue:
            if(should_query(reader_types::event_type_t::kernel_dispatch))
            {
                query_agent_queue(m_read_statements->kernel_dispatch_statements(),
                                  reader_types::event_type_t::kernel_dispatch);
            }
            break;
        case reader_types::track_kind_t::memory_allocate_agent_queue:
            if(should_query(reader_types::event_type_t::memory_allocate))
            {
                query_agent_queue(m_read_statements->memory_allocate_statements(),
                                  reader_types::event_type_t::memory_allocate);
            }
            break;
        case reader_types::track_kind_t::memory_copy_agent_queue:
            if(should_query(reader_types::event_type_t::memory_copy))
            {
                query_agent_queue(m_read_statements->memory_copy_statements(),
                                  reader_types::event_type_t::memory_copy);
            }
            break;
        case reader_types::track_kind_t::stream:
            if(should_query(reader_types::event_type_t::kernel_dispatch))
            {
                query_stream(m_read_statements->kernel_dispatch_statements(),
                             reader_types::event_type_t::kernel_dispatch);
            }
            if(should_query(reader_types::event_type_t::memory_allocate))
            {
                query_stream(m_read_statements->memory_allocate_statements(),
                             reader_types::event_type_t::memory_allocate);
            }
            if(should_query(reader_types::event_type_t::memory_copy))
            {
                query_stream(m_read_statements->memory_copy_statements(),
                             reader_types::event_type_t::memory_copy);
            }
            break;
        case reader_types::track_kind_t::thread:
        case reader_types::track_kind_t::pmc_agent: break;
    }

    apply_pagination(all_events, filter.pagination);
    return all_events;
}

reader_types::counter_timeline_event_list_t
reader_t::impl::get_counter_events_for_track(reader_types::track_info_ptr_t      track,
                                             const reader_types::event_filter_t& filter)
{
    if(!track || !track->node_info) return {};

    const auto nid = track->node_info->node_id;

    const bool has_time =
        filter.time_window.start.has_value() && filter.time_window.end.has_value();

    std::vector<data_storage::schema_v3::pmc_sample_result> results;
    if(has_time)
    {
        results =
            m_read_statements
                ->pmc_sample_time_filtered_statement()(nid,
                                                       track->agent_id,
                                                       track->pmc_id,
                                                       filter.time_window.start.value(),
                                                       filter.time_window.end.value())
                .to_vector();
    }
    else
    {
        results =
            m_read_statements->pmc_sample_statement()(nid, track->agent_id, track->pmc_id)
                .to_vector();
    }

    reader_types::counter_timeline_event_list_t events;
    events.reserve(results.size());
    for(const auto& result : results)
    {
        events.push_back(reader_types::counter_timeline_event_t{
            .timestamp = result.timestamp,
            .value     = result.value,
            .track     = track,
        });
    }

    return events;
}

size_t
reader_t::impl::get_event_count(const reader_types::event_filter_t& filter)
{
    const bool query_all    = filter.types.empty();
    auto       should_count = [&](reader_types::event_type_t t) {
        return query_all || std::find(filter.types.begin(), filter.types.end(), t) !=
                                filter.types.end();
    };

    const bool has_time =
        filter.time_window.start.has_value() && filter.time_window.end.has_value();

    auto run_count = [&](const auto& base_stmt, const auto& time_stmt) -> size_t {
        auto results = has_time ? time_stmt(filter.time_window.end.value(),
                                            filter.time_window.start.value())
                                      .to_vector()
                                : base_stmt().to_vector();
        return results.empty() ? 0 : results.front().count;
    };

    size_t total = 0;
    if(should_count(reader_types::event_type_t::region))
    {
        total += run_count(m_read_statements->region_count(),
                           m_read_statements->region_count_time_filtered());
    }
    if(should_count(reader_types::event_type_t::kernel_dispatch))
    {
        total += run_count(m_read_statements->kernel_dispatch_count(),
                           m_read_statements->kernel_dispatch_count_time_filtered());
    }
    if(should_count(reader_types::event_type_t::memory_copy))
    {
        total += run_count(m_read_statements->memory_copy_count(),
                           m_read_statements->memory_copy_count_time_filtered());
    }
    if(should_count(reader_types::event_type_t::memory_allocate))
    {
        total += run_count(m_read_statements->memory_alloc_count(),
                           m_read_statements->memory_alloc_count_time_filtered());
    }
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
        auto it = m_catalog->string_utility.find(event_meta.category_id.value());
        if(it != m_catalog->string_utility.end())
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
        auto it = m_catalog->string_utility.find(r.name_id.value());
        if(it != m_catalog->string_utility.end())
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
        auto it = m_catalog->string_utility.find(r.region_name_id.value());
        if(it != m_catalog->string_utility.end()) data.name = it->second;
    }

    if(r.kernel_id.has_value())
    {
        auto it = m_catalog->kernel_symbol_utility.find(r.kernel_id.value());
        if(it != m_catalog->kernel_symbol_utility.end())
        {
            data.kernel_symbol_info = it->second;
            if(it->second && it->second->code_object_info)
                data.code_object_info = it->second->code_object_info;
        }
    }

    auto node_it = m_catalog->node_utility.find(r.nid);
    if(node_it != m_catalog->node_utility.end()) data.node_info = node_it->second;

    if(r.pid.has_value())
    {
        auto it = m_catalog->process_utility.find(r.pid.value());
        if(it != m_catalog->process_utility.end()) data.process_info = it->second;
    }

    if(r.tid.has_value())
    {
        auto it = m_catalog->thread_utility.find(r.tid.value());
        if(it != m_catalog->thread_utility.end()) data.thread_info = it->second;
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
        auto it = m_catalog->string_utility.find(r.name_id.value());
        if(it != m_catalog->string_utility.end()) data.name = it->second;
    }

    if(r.region_name_id.has_value())
    {
        auto it = m_catalog->string_utility.find(r.region_name_id.value());
        if(it != m_catalog->string_utility.end()) data.region_name = it->second;
    }

    if(r.dst_agent_id.has_value())
    {
        auto it = m_catalog->agent_utility.find(r.dst_agent_id.value());
        if(it != m_catalog->agent_utility.end()) data.dst_agent_id = it->second;
    }

    if(r.src_agent_id.has_value())
    {
        auto it = m_catalog->agent_utility.find(r.src_agent_id.value());
        if(it != m_catalog->agent_utility.end()) data.src_agent_id = it->second;
    }

    auto node_it = m_catalog->node_utility.find(r.nid);
    if(node_it != m_catalog->node_utility.end()) data.node_info = node_it->second;

    if(r.pid.has_value())
    {
        auto it = m_catalog->process_utility.find(r.pid.value());
        if(it != m_catalog->process_utility.end()) data.process_info = it->second;
    }

    if(r.tid.has_value())
    {
        auto it = m_catalog->thread_utility.find(r.tid.value());
        if(it != m_catalog->thread_utility.end()) data.thread_info = it->second;
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

    auto node_it = m_catalog->node_utility.find(r.nid);
    if(node_it != m_catalog->node_utility.end()) data.node_info = node_it->second;

    if(r.pid.has_value())
    {
        auto it = m_catalog->process_utility.find(r.pid.value());
        if(it != m_catalog->process_utility.end()) data.process_info = it->second;
    }

    if(r.tid.has_value())
    {
        auto it = m_catalog->thread_utility.find(r.tid.value());
        if(it != m_catalog->thread_utility.end()) data.thread_info = it->second;
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
    auto event_meta = resolve_event_metadata(event);
    if(!event_meta.has_value()) return {};

    return json_serializers::deserialize_call_stack(event_meta->call_stack);
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
    auto event_meta = resolve_event_metadata(event);
    if(!event_meta.has_value() || !event_meta->stack_id.has_value()) return {};

    auto stack_id          = event_meta->stack_id.value();
    auto excluded_event_id = event_meta->event_id.value_or(0);

    reader_types::timeline_event_list_t all_events;

    const auto& stmts = m_read_statements->correlated_event_statements();

    auto query_type = [&](const auto& stmt, reader_types::event_type_t type) {
        auto results = stmt(stack_id, excluded_event_id).to_vector();
        auto events  = build_timeline_events(results, type);
        all_events.insert(all_events.end(),
                          std::make_move_iterator(events.begin()),
                          std::make_move_iterator(events.end()));
    };

    query_type(stmts.region, reader_types::event_type_t::region);
    query_type(stmts.kernel_dispatch, reader_types::event_type_t::kernel_dispatch);
    query_type(stmts.memory_copy, reader_types::event_type_t::memory_copy);
    query_type(stmts.memory_allocate, reader_types::event_type_t::memory_allocate);

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
    const bool has_time = window.start.has_value() && window.end.has_value();

    auto get_count = [&](const auto& base_stmt, const auto& time_stmt) -> size_t {
        auto results =
            has_time ? time_stmt(window.end.value(), window.start.value()).to_vector()
                     : base_stmt().to_vector();
        return results.empty() ? 0 : results.front().count;
    };

    reader_types::event_counts_t counts;
    counts[reader_types::event_type_t::region] =
        get_count(m_read_statements->region_count(),
                  m_read_statements->region_count_time_filtered());
    counts[reader_types::event_type_t::kernel_dispatch] =
        get_count(m_read_statements->kernel_dispatch_count(),
                  m_read_statements->kernel_dispatch_count_time_filtered());
    counts[reader_types::event_type_t::memory_copy] =
        get_count(m_read_statements->memory_copy_count(),
                  m_read_statements->memory_copy_count_time_filtered());
    counts[reader_types::event_type_t::memory_allocate] =
        get_count(m_read_statements->memory_alloc_count(),
                  m_read_statements->memory_alloc_count_time_filtered());
    return counts;
}

}  // namespace profiler_hub
