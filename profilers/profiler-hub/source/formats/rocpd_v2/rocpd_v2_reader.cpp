// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocpd_v2_reader.hpp"

#include "common/debug.hpp"
#include "histogram_grid.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <exception>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace profiler_hub
{

using reader_types::event_type_t;
using reader_types::track_kind_t;
using rocpd_v2::api_row_t;
using rocpd_v2::k_api_columns;
using rocpd_v2::k_node_id;
using rocpd_v2::op_row_t;

namespace rocpd_v2
{
std::string
op_columns()
{
    return fmt::format("O.id, O.start, O.end, O.description_id, O.opType_id, O.gpuId, "
                       "O.queueId, O.opType_id IN {}",
                       k_copy_op_types);
}

std::string
op_kind_where(std::string_view alias, bool is_copy)
{
    return fmt::format(
        "{}.opType_id {}IN {}", alias, is_copy ? "" : "NOT ", k_copy_op_types);
}

std::string
quote_text(std::string_view text)
{
    std::string quoted = "'";
    for(const char ch : text)
    {
        if(ch == '\'') quoted += "''";
        else quoted += ch;
    }
    quoted += '\'';
    return quoted;
}

std::optional<size_t>
parse_address(const std::optional<std::string>& text)
{
    if(!text.has_value() || text->empty()) return std::nullopt;
    try
    {
        return static_cast<size_t>(std::stoull(*text, nullptr, 16));
    } catch(const std::exception&)
    {
        return std::nullopt;
    }
}
}  // namespace rocpd_v2

namespace
{
struct id_span_row_t
{
    std::optional<size_t> first_id;
    std::optional<size_t> last_id;
};

struct count_row_t
{
    size_t count{};
};

struct interval_row_t
{
    size_t start_ts{};
    size_t end_ts{};
};

struct monitor_row_t
{
    size_t                timestamp{};
    std::optional<double> value;
};

struct sample_point_t
{
    size_t timestamp{};
    double value{};
};

struct time_range_row_t
{
    std::optional<size_t> min_start;
    std::optional<size_t> max_end;
};

struct region_detail_row_t
{
    size_t                start{};
    size_t                end{};
    std::optional<size_t> name_id;
    std::optional<size_t> args_id;
};

struct kernel_detail_row_t
{
    size_t                start{};
    size_t                end{};
    size_t                gpu_id{};
    std::optional<size_t> description_id;
    std::optional<size_t> op_type_id;
    std::optional<size_t> grid_x;
    std::optional<size_t> grid_y;
    std::optional<size_t> grid_z;
    std::optional<size_t> workgroup_x;
    std::optional<size_t> workgroup_y;
    std::optional<size_t> workgroup_z;
    std::optional<size_t> group_segment_size;
    std::optional<size_t> private_segment_size;
    std::optional<size_t> kernel_name_id;
    std::optional<size_t> pid;
    std::optional<size_t> tid;
};

struct copy_detail_row_t
{
    size_t                     start{};
    size_t                     end{};
    size_t                     gpu_id{};
    std::optional<size_t>      description_id;
    std::optional<size_t>      op_type_id;
    std::optional<size_t>      size;
    std::optional<std::string> dst;
    std::optional<std::string> src;
    std::optional<size_t>      api_name_id;
    std::optional<size_t>      pid;
    std::optional<size_t>      tid;
};

struct location_row_t
{
    size_t first{};
    size_t second{};
    size_t is_copy{};
};

constexpr std::array k_event_types{ event_type_t::region,
                                    event_type_t::kernel_dispatch,
                                    event_type_t::memory_copy };

auto
api_rows(data_storage::sqlite_backend& backend, const std::string& sql)
{
    return backend
        .create_read_statement_executor<api_row_t>(sql,
                                                   &api_row_t::id,
                                                   &api_row_t::start,
                                                   &api_row_t::end,
                                                   &api_row_t::name_id,
                                                   &api_row_t::pid,
                                                   &api_row_t::tid)()
        .to_vector();
}

auto
op_rows(data_storage::sqlite_backend& backend, const std::string& sql)
{
    return backend
        .create_read_statement_executor<op_row_t>(sql,
                                                  &op_row_t::id,
                                                  &op_row_t::start,
                                                  &op_row_t::end,
                                                  &op_row_t::description_id,
                                                  &op_row_t::op_type_id,
                                                  &op_row_t::gpu_id,
                                                  &op_row_t::queue_id,
                                                  &op_row_t::is_copy)()
        .to_vector();
}

size_t
count_of(data_storage::sqlite_backend& backend, const std::string& sql)
{
    const auto rows =
        backend.create_read_statement_executor<count_row_t>(sql, &count_row_t::count)()
            .to_vector();
    return rows.empty() ? 0 : rows.front().count;
}

std::vector<interval_row_t>
interval_rows(data_storage::sqlite_backend& backend, const std::string& sql)
{
    return backend
        .create_read_statement_executor<interval_row_t>(
            sql, &interval_row_t::start_ts, &interval_row_t::end_ts)()
        .to_vector();
}

std::vector<sample_point_t>
monitor_samples(data_storage::sqlite_backend&         backend,
                const reader_types::track_info_t&     track,
                const reader_types::time_window_t&    window)
{
    std::string sql = fmt::format(
        "SELECT start, CAST(value AS REAL) FROM rocpd_monitor WHERE deviceId = {} AND "
        "monitorType = {}",
        track.agent_id,
        rocpd_v2::quote_text(track.name));
    if(window.start.has_value() && window.end.has_value())
    {
        sql += fmt::format(
            " AND start >= {} AND start <= {}", *window.start, *window.end);
    }
    sql += " ORDER BY start";

    std::vector<sample_point_t> samples;
    backend
        .create_read_statement_executor<monitor_row_t>(
            sql, &monitor_row_t::timestamp, &monitor_row_t::value)()
        .for_each([&](const monitor_row_t& row) {
            samples.push_back(sample_point_t{ row.timestamp, row.value.value_or(0.0) });
        });
    return samples;
}

std::string
time_where(std::string_view alias, const reader_types::time_window_t& window)
{
    if(!window.start.has_value() || !window.end.has_value()) return {};
    return fmt::format(
        " AND {0}.start <= {1} AND {0}.end >= {2}", alias, *window.end, *window.start);
}

bool
selects(const reader_types::event_filter_t& filter, event_type_t type)
{
    return filter.types.empty() ||
           std::find(filter.types.begin(), filter.types.end(), type) != filter.types.end();
}

bool
is_op_type(event_type_t type)
{
    return type == event_type_t::kernel_dispatch || type == event_type_t::memory_copy;
}

std::string
event_select(event_type_t type, const std::string& where)
{
    if(type == event_type_t::region)
    {
        return fmt::format("SELECT {} FROM rocpd_api A WHERE 1{}", k_api_columns, where);
    }
    return fmt::format("SELECT {} FROM rocpd_op O WHERE {}{}",
                       rocpd_v2::op_columns(),
                       rocpd_v2::op_kind_where("O", type == event_type_t::memory_copy),
                       where);
}

std::string
count_select(event_type_t type, const reader_types::time_window_t& window)
{
    if(type == event_type_t::region)
    {
        return "SELECT COUNT(*) FROM rocpd_api A WHERE 1" + time_where("A", window);
    }
    return "SELECT COUNT(*) FROM rocpd_op O WHERE " +
           rocpd_v2::op_kind_where("O", type == event_type_t::memory_copy) +
           time_where("O", window);
}
}  // namespace

std::unique_ptr<reader_t::impl>
make_rocpd_v2_reader(std::unique_ptr<profiler_hub::storage_t> storage,
                     std::shared_ptr<reader_catalog_t>        catalog)
{
    return std::make_unique<rocpd_v2_reader>(std::move(storage), std::move(catalog));
}

rocpd_v2_reader::rocpd_v2_reader(std::unique_ptr<profiler_hub::storage_t> storage,
                                 std::shared_ptr<reader_catalog_t>        catalog)
: reader_t::impl(std::move(storage), std::move(catalog))
, m_backend(open_sqlite())
{
    m_has_kernel_api = has_table("rocpd_kernelapi");
    m_has_copy_api   = has_table("rocpd_copyapi");
}

bool
rocpd_v2_reader::has_table(std::string_view name)
{
    return count_of(*m_backend,
                    "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = " +
                        rocpd_v2::quote_text(name)) != 0;
}

void
rocpd_v2_reader::ensure_track_topology_indexes()
{
    constexpr std::array<std::string_view, 5> k_indexes{
        "CREATE INDEX IF NOT EXISTS ph_rocpd_v2_api_track ON rocpd_api(pid, tid, start)",
        "CREATE INDEX IF NOT EXISTS ph_rocpd_v2_op_track ON rocpd_op(gpuId, queueId, start)",
        "CREATE INDEX IF NOT EXISTS ph_rocpd_v2_monitor_track ON "
        "rocpd_monitor(deviceId, monitorType, start)",
        "CREATE INDEX IF NOT EXISTS ph_rocpd_v2_api_ops_api ON rocpd_api_ops(api_id)",
        "CREATE INDEX IF NOT EXISTS ph_rocpd_v2_api_ops_op ON rocpd_api_ops(op_id)",
    };
    for(const std::string_view sql : k_indexes)
    {
        try
        {
            m_backend->execute(std::string{ sql });
        } catch(const std::exception& error)
        {
            // A read-only file still opens. Its track queries scan instead.
            LOG_WARNING("rocpd schema 2 index was not created: {}", error.what());
        }
    }
}

// ============================================================================
// Lookups
// ============================================================================

std::string_view
rocpd_v2_reader::text_of(std::optional<size_t> string_id) const
{
    if(!string_id.has_value()) return {};
    const auto found = m_catalog->strings_by_id.find(string_id.value());
    return found != m_catalog->strings_by_id.end() ? std::string_view{ found->second }
                                                   : std::string_view{};
}

std::optional<size_t>
rocpd_v2_reader::op_name_id(std::optional<size_t> description_id,
                            std::optional<size_t> op_type_id) const
{
    if(!text_of(description_id).empty()) return description_id;
    return op_type_id;
}

size_t
rocpd_v2_reader::canonical_string_id(size_t string_id) const
{
    const auto found = m_catalog->canonical_string_ids.find(string_id);
    return found != m_catalog->canonical_string_ids.end() ? found->second : string_id;
}

reader_types::track_info_ptr_t
rocpd_v2_reader::thread_track(size_t pid, size_t tid) const
{
    const auto found = m_catalog->track_by_topology.find(topology_key_t{ k_node_id, pid, tid });
    return found != m_catalog->track_by_topology.end() ? found->second : nullptr;
}

reader_types::track_info_ptr_t
rocpd_v2_reader::op_track(bool is_copy, size_t gpu_id, size_t queue_id) const
{
    const auto kind = is_copy ? track_kind_t::memory_copy_agent_queue
                              : track_kind_t::kernel_dispatch_agent_queue;
    const auto found = m_catalog->agent_queue_track_by_key.find(
        agent_queue_track_key_t{ kind, k_node_id, gpu_id, queue_id });
    return found != m_catalog->agent_queue_track_by_key.end() ? found->second : nullptr;
}

reader_types::thread_info_ptr_t
rocpd_v2_reader::thread_info(size_t pid, size_t tid) const
{
    const auto track = thread_track(pid, tid);
    return track ? track->thread_info : nullptr;
}

reader_types::process_info_ptr_t
rocpd_v2_reader::process_info(size_t pid) const
{
    const auto found = m_catalog->processes_by_id.find(pid);
    return found != m_catalog->processes_by_id.end() ? found->second : nullptr;
}

reader_types::agent_info_ptr_t
rocpd_v2_reader::agent_info(size_t gpu_id) const
{
    const auto found = m_catalog->agents_by_id.find(gpu_id);
    return found != m_catalog->agents_by_id.end() ? found->second : nullptr;
}

reader_types::timeline_event_t
rocpd_v2_reader::region_event(const api_row_t& row) const
{
    reader_types::timeline_event_t event;
    event.unique_identifier = { .id = row.id, .type = event_type_t::region };
    event.start_timestamp   = row.start;
    event.end_timestamp     = row.end;
    event.display_name      = text_of(row.name_id);
    event.track             = thread_track(row.pid, row.tid);
    return event;
}

reader_types::timeline_event_t
rocpd_v2_reader::op_event(const op_row_t& row) const
{
    const bool                     is_copy = row.is_copy != 0;
    reader_types::timeline_event_t event;
    event.unique_identifier = {
        .id   = row.id,
        .type = is_copy ? event_type_t::memory_copy : event_type_t::kernel_dispatch
    };
    event.start_timestamp = row.start;
    event.end_timestamp   = row.end;
    event.display_name    = text_of(op_name_id(row.description_id, row.op_type_id));
    event.category        = text_of(row.op_type_id);
    event.track           = op_track(is_copy, row.gpu_id, row.queue_id);
    return event;
}

std::string
rocpd_v2_reader::kernel_api_columns() const
{
    if(!m_has_kernel_api) return "NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL";
    return "K.gridX, K.gridY, K.gridZ, K.workgroupX, K.workgroupY, K.workgroupZ, "
           "K.groupSegmentSize, K.privateSegmentSize, K.kernelName_id";
}

std::string
rocpd_v2_reader::copy_api_columns() const
{
    if(!m_has_copy_api) return "NULL, NULL, NULL";
    return "C.size, C.dst, C.src";
}

std::string
rocpd_v2_reader::op_api_joins() const
{
    // An operation may be linked more than once. The first link keeps one
    // output row per operation.
    std::string joins = "LEFT JOIN rocpd_api_ops AO ON AO.rowid = (SELECT MIN(rowid) FROM "
                        "rocpd_api_ops WHERE op_id = O.id) "
                        "LEFT JOIN rocpd_api A ON A.id = AO.api_id";
    if(m_has_kernel_api) joins += " LEFT JOIN rocpd_kernelapi K ON K.api_ptr_id = AO.api_id";
    if(m_has_copy_api) joins += " LEFT JOIN rocpd_copyapi C ON C.api_ptr_id = AO.api_id";
    return joins;
}

std::optional<api_row_t>
rocpd_v2_reader::launching_api(size_t op_id)
{
    const auto found = api_rows(
        *m_backend,
        fmt::format("SELECT {} FROM rocpd_api_ops AO JOIN rocpd_api A ON A.id = AO.api_id "
                    "WHERE AO.op_id = {} ORDER BY AO.rowid LIMIT 1",
                    k_api_columns,
                    op_id));
    if(found.empty()) return std::nullopt;
    return found.front();
}

// ============================================================================
// Timeline events
// ============================================================================

reader_types::timeline_event_list_t
rocpd_v2_reader::get_events(const reader_types::event_filter_t& filter)
{
    reader_types::timeline_event_list_t events;
    for(const event_type_t type : k_event_types)
    {
        if(!selects(filter, type)) continue;
        const std::string sql =
            event_select(type, time_where(is_op_type(type) ? "O" : "A", filter.time_window));
        if(type == event_type_t::region)
        {
            for(const api_row_t& row : api_rows(*m_backend, sql))
                events.push_back(region_event(row));
        }
        else
        {
            for(const op_row_t& row : op_rows(*m_backend, sql))
                events.push_back(op_event(row));
        }
    }
    apply_pagination(events, filter.pagination);
    return events;
}

reader_types::timeline_event_list_t
rocpd_v2_reader::get_events_for_track(reader_types::track_info_ptr_t      track,
                                      const reader_types::event_filter_t& filter)
{
    if(!track) return {};

    reader_types::timeline_event_list_t events;
    events.reserve(track->event_count);
    if(track->category == track_kind_t::thread)
    {
        if(!selects(filter, event_type_t::region)) return {};
        const auto topology = m_catalog->topology_by_track.find(track);
        if(topology == m_catalog->topology_by_track.end()) return {};
        const std::string where = fmt::format(" AND A.pid = {} AND A.tid = {}{}",
                                              topology->second.pid,
                                              topology->second.tid,
                                              time_where("A", filter.time_window));
        for(const api_row_t& row : api_rows(*m_backend, event_select(event_type_t::region, where)))
        {
            auto event  = region_event(row);
            event.track = track;
            events.push_back(event);
        }
    }
    else if(track->category == track_kind_t::kernel_dispatch_agent_queue ||
            track->category == track_kind_t::memory_copy_agent_queue)
    {
        const auto type = track->category == track_kind_t::memory_copy_agent_queue
                              ? event_type_t::memory_copy
                              : event_type_t::kernel_dispatch;
        if(!selects(filter, type)) return {};
        const std::string where = fmt::format(" AND O.gpuId = {} AND O.queueId = {}{}",
                                              track->agent_id,
                                              track->queue_id,
                                              time_where("O", filter.time_window));
        for(const op_row_t& row : op_rows(*m_backend, event_select(type, where)))
        {
            auto event  = op_event(row);
            event.track = track;
            events.push_back(event);
        }
    }
    apply_pagination(events, filter.pagination);
    return events;
}

std::optional<std::pair<size_t, size_t>>
rocpd_v2_reader::get_event_id_span(reader_types::event_type_t type)
{
    std::string sql;
    if(type == event_type_t::region)
    {
        sql = "SELECT MIN(id), MAX(id) FROM rocpd_api";
    }
    else if(is_op_type(type))
    {
        sql = "SELECT MIN(O.id), MAX(O.id) FROM rocpd_op O WHERE " +
              rocpd_v2::op_kind_where("O", type == event_type_t::memory_copy);
    }
    else
    {
        return std::nullopt;
    }
    const auto rows = m_backend
                          ->create_read_statement_executor<id_span_row_t>(
                              sql, &id_span_row_t::first_id, &id_span_row_t::last_id)()
                          .to_vector();
    if(rows.empty() || !rows.front().first_id.has_value() ||
       !rows.front().last_id.has_value())
    {
        return std::nullopt;
    }
    return std::pair{ rows.front().first_id.value(), rows.front().last_id.value() };
}

void
rocpd_v2_reader::visit_track_events_in_id_range(const reader_types::track_info_ptr_t& track,
                                                reader_types::event_type_t            type,
                                                size_t                    id_begin,
                                                size_t                    id_end,
                                                reader_t::event_visitor_t visitor,
                                                void*                     context)
{
    // Only host threads are read in id ranges, and their rows are API calls.
    if(!track || type != event_type_t::region) return;
    const auto topology = m_catalog->topology_by_track.find(track);
    if(topology == m_catalog->topology_by_track.end()) return;

    const std::string sql = event_select(
        event_type_t::region,
        fmt::format(" AND A.pid = {} AND A.tid = {} AND A.id >= {} AND A.id < {}",
                    topology->second.pid,
                    topology->second.tid,
                    id_begin,
                    id_end));
    for(const api_row_t& row : api_rows(*m_backend, sql))
    {
        visitor(context, { row.id, event_type_t::region }, row.start, row.end,
                text_of(row.name_id));
    }
}

reader_types::counter_timeline_event_list_t
rocpd_v2_reader::get_counter_events_for_track(reader_types::track_info_ptr_t      track,
                                              const reader_types::event_filter_t& filter)
{
    if(!track || track->category != track_kind_t::pmc_agent) return {};

    reader_types::counter_timeline_event_list_t events;
    for(const sample_point_t& sample : monitor_samples(*m_backend, *track, filter.time_window))
    {
        events.push_back(reader_types::counter_timeline_event_t{ .unique_identifier = {},
                                                                 .timestamp = sample.timestamp,
                                                                 .value     = sample.value,
                                                                 .track     = track });
    }
    return events;
}

size_t
rocpd_v2_reader::get_event_count(const reader_types::event_filter_t& filter)
{
    size_t total = 0;
    for(const event_type_t type : k_event_types)
    {
        if(selects(filter, type)) total += count_of(*m_backend, count_select(type, filter.time_window));
    }
    return total;
}

reader_types::histogram_t
rocpd_v2_reader::get_histogram(size_t max_buckets)
{
    return get_histogram(make_histogram_grid(*m_catalog, max_buckets));
}

reader_types::histogram_t
rocpd_v2_reader::get_histogram(const histogram_grid_t& grid)
{
    return build_histogram(
        *m_catalog,
        grid,
        [this](const reader_types::track_info_ptr_t& track,
               const histogram_grid_t& grid) -> reader_types::histogram_bucket_list_t {
            switch(track->category)
            {
                case track_kind_t::thread:
                {
                    const auto topology = m_catalog->topology_by_track.find(track);
                    if(topology == m_catalog->topology_by_track.end()) return {};
                    return accumulate_overlaps(
                        interval_rows(*m_backend,
                                      fmt::format("SELECT start, end FROM rocpd_api WHERE "
                                                  "pid = {} AND tid = {}",
                                                  topology->second.pid,
                                                  topology->second.tid)),
                        grid);
                }
                case track_kind_t::kernel_dispatch_agent_queue:
                case track_kind_t::memory_copy_agent_queue:
                    return accumulate_overlaps(
                        interval_rows(
                            *m_backend,
                            fmt::format(
                                "SELECT O.start, O.end FROM rocpd_op O WHERE O.gpuId = {} "
                                "AND O.queueId = {} AND {}",
                                track->agent_id,
                                track->queue_id,
                                rocpd_v2::op_kind_where(
                                    "O",
                                    track->category ==
                                        track_kind_t::memory_copy_agent_queue))),
                        grid);
                case track_kind_t::pmc_agent:
                    return accumulate_samples(monitor_samples(*m_backend, *track, {}), grid);
                default: return {};
            }
        });
}

// ============================================================================
// Event details
// ============================================================================

reader_types::track_info_ptr_t
rocpd_v2_reader::track_of(const reader_types::timeline_event_t& event)
{
    const auto type = event.unique_identifier.type;
    const auto id   = event.unique_identifier.id;
    std::string sql;
    if(type == event_type_t::region)
    {
        sql = fmt::format("SELECT pid, tid, 0 FROM rocpd_api WHERE id = {}", id);
    }
    else if(is_op_type(type))
    {
        sql = fmt::format("SELECT gpuId, queueId, opType_id IN {} FROM rocpd_op WHERE id = {}",
                          rocpd_v2::k_copy_op_types,
                          id);
    }
    else
    {
        return {};
    }
    const auto rows = m_backend
                          ->create_read_statement_executor<location_row_t>(
                              sql,
                              &location_row_t::first,
                              &location_row_t::second,
                              &location_row_t::is_copy)()
                          .to_vector();
    if(rows.empty()) return {};
    const location_row_t& row = rows.front();
    if(type == event_type_t::region) return thread_track(row.first, row.second);
    return op_track(row.is_copy != 0, row.first, row.second);
}

std::optional<reader_types::region_data_t>
rocpd_v2_reader::get_region_details(const reader_types::timeline_event_t& event)
{
    if(event.unique_identifier.type != event_type_t::region) return std::nullopt;

    const auto rows = m_backend
                          ->create_read_statement_executor<region_detail_row_t>(
                              fmt::format("SELECT start, end, apiName_id, args_id FROM "
                                          "rocpd_api WHERE id = {}",
                                          event.unique_identifier.id),
                              &region_detail_row_t::start,
                              &region_detail_row_t::end,
                              &region_detail_row_t::name_id,
                              &region_detail_row_t::args_id)()
                          .to_vector();
    if(rows.empty()) return std::nullopt;

    const region_detail_row_t& row = rows.front();
    reader_types::region_data_t data;
    data.start_timestamp = row.start;
    data.end_timestamp   = row.end;
    data.name            = std::string{ text_of(row.name_id) };
    const std::string_view args = text_of(row.args_id);
    if(!args.empty())
    {
        data.args.push_back(reader_types::arg_data_t{
            .position = 0, .type = {}, .name = "args", .value = std::string{ args } });
    }
    return data;
}

reader_types::arg_data_list_t
rocpd_v2_reader::get_arguments(const reader_types::timeline_event_t& event)
{
    const auto details = get_region_details(event);
    if(!details.has_value()) return {};

    reader_types::arg_data_list_t args;
    for(const reader_types::arg_data_t& arg : details->args)
    {
        args.push_back(std::make_shared<reader_types::arg_data_t>(arg));
    }
    return args;
}

std::optional<reader_types::kernel_dispatch_data_t>
rocpd_v2_reader::get_kernel_dispatch_details(const reader_types::timeline_event_t& event)
{
    if(event.unique_identifier.type != event_type_t::kernel_dispatch) return std::nullopt;

    const std::string sql = fmt::format(
        "SELECT O.start, O.end, O.gpuId, O.description_id, O.opType_id, {}, A.pid, A.tid "
        "FROM rocpd_op O {} WHERE O.id = {}",
        kernel_api_columns(),
        op_api_joins(),
        event.unique_identifier.id);
    const auto rows = m_backend
                          ->create_read_statement_executor<kernel_detail_row_t>(
                              sql,
                              &kernel_detail_row_t::start,
                              &kernel_detail_row_t::end,
                              &kernel_detail_row_t::gpu_id,
                              &kernel_detail_row_t::description_id,
                              &kernel_detail_row_t::op_type_id,
                              &kernel_detail_row_t::grid_x,
                              &kernel_detail_row_t::grid_y,
                              &kernel_detail_row_t::grid_z,
                              &kernel_detail_row_t::workgroup_x,
                              &kernel_detail_row_t::workgroup_y,
                              &kernel_detail_row_t::workgroup_z,
                              &kernel_detail_row_t::group_segment_size,
                              &kernel_detail_row_t::private_segment_size,
                              &kernel_detail_row_t::kernel_name_id,
                              &kernel_detail_row_t::pid,
                              &kernel_detail_row_t::tid)()
                          .to_vector();
    if(rows.empty()) return std::nullopt;

    const kernel_detail_row_t&           row = rows.front();
    reader_types::kernel_dispatch_data_t data;
    data.dispatch_id          = event.unique_identifier.id;
    data.start_timestamp      = row.start;
    data.end_timestamp        = row.end;
    data.grid_size_x          = row.grid_x.value_or(0);
    data.grid_size_y          = row.grid_y.value_or(0);
    data.grid_size_z          = row.grid_z.value_or(0);
    data.workgroup_size_x     = row.workgroup_x.value_or(0);
    data.workgroup_size_y     = row.workgroup_y.value_or(0);
    data.workgroup_size_z     = row.workgroup_z.value_or(0);
    data.group_segment_size   = row.group_segment_size;
    data.private_segment_size = row.private_segment_size;

    std::string_view name = text_of(row.description_id);
    if(name.empty()) name = text_of(row.kernel_name_id);
    if(name.empty()) name = text_of(row.op_type_id);
    data.name = std::string{ name };

    if(const auto node = m_catalog->nodes_by_id.find(k_node_id);
       node != m_catalog->nodes_by_id.end())
    {
        data.node_info = node->second;
    }
    data.agent_info = agent_info(row.gpu_id);
    if(row.pid.has_value())
    {
        data.process_info = process_info(row.pid.value());
        if(row.tid.has_value()) data.thread_info = thread_info(row.pid.value(), row.tid.value());
    }
    return data;
}

std::optional<reader_types::memory_copy_data_t>
rocpd_v2_reader::get_memory_copy_details(const reader_types::timeline_event_t& event)
{
    if(event.unique_identifier.type != event_type_t::memory_copy) return std::nullopt;

    const std::string sql = fmt::format(
        "SELECT O.start, O.end, O.gpuId, O.description_id, O.opType_id, {}, A.apiName_id, "
        "A.pid, A.tid FROM rocpd_op O {} WHERE O.id = {}",
        copy_api_columns(),
        op_api_joins(),
        event.unique_identifier.id);
    const auto rows = m_backend
                          ->create_read_statement_executor<copy_detail_row_t>(
                              sql,
                              &copy_detail_row_t::start,
                              &copy_detail_row_t::end,
                              &copy_detail_row_t::gpu_id,
                              &copy_detail_row_t::description_id,
                              &copy_detail_row_t::op_type_id,
                              &copy_detail_row_t::size,
                              &copy_detail_row_t::dst,
                              &copy_detail_row_t::src,
                              &copy_detail_row_t::api_name_id,
                              &copy_detail_row_t::pid,
                              &copy_detail_row_t::tid)()
                          .to_vector();
    if(rows.empty()) return std::nullopt;

    const copy_detail_row_t&         row = rows.front();
    reader_types::memory_copy_data_t data;
    data.start_timestamp = row.start;
    data.end_timestamp   = row.end;
    data.size            = row.size.value_or(0);
    data.dst_address     = rocpd_v2::parse_address(row.dst);
    data.src_address     = rocpd_v2::parse_address(row.src);
    data.name        = std::string{ text_of(op_name_id(row.description_id, row.op_type_id)) };
    data.region_name = std::string{ text_of(row.api_name_id) };
    data.dst_agent_id = agent_info(row.gpu_id);

    if(const auto node = m_catalog->nodes_by_id.find(k_node_id);
       node != m_catalog->nodes_by_id.end())
    {
        data.node_info = node->second;
    }
    if(row.pid.has_value())
    {
        data.process_info = process_info(row.pid.value());
        if(row.tid.has_value()) data.thread_info = thread_info(row.pid.value(), row.tid.value());
    }
    return data;
}

// ============================================================================
// Event properties
// ============================================================================

reader_types::call_stack_t
rocpd_v2_reader::get_call_stack(const reader_types::timeline_event_t& event)
{
    if(event.unique_identifier.type != event_type_t::region) return {};

    const auto source = api_rows(*m_backend,
                                 event_select(event_type_t::region,
                                              fmt::format(" AND A.id = {}",
                                                          event.unique_identifier.id)));
    if(source.empty()) return {};
    const api_row_t& self = source.front();

    // Every call on the thread that encloses this one is an active caller.
    // The latest start is the innermost frame.
    const auto callers = api_rows(
        *m_backend,
        event_select(event_type_t::region,
                     fmt::format(" AND A.pid = {} AND A.tid = {} AND A.start <= {} AND "
                                 "A.end >= {} AND A.id != {} ORDER BY A.start DESC, A.end",
                                 self.pid,
                                 self.tid,
                                 self.start,
                                 self.end,
                                 self.id)));

    reader_types::call_stack_t stack;
    const auto add_frame = [&](const api_row_t& row) {
        reader_types::program_counter_info_t program_counter;
        program_counter.event_id = row.id;
        program_counter.function = std::string{ text_of(row.name_id) };
        reader_types::stack_frame_t frame;
        frame.program_counter = std::move(program_counter);
        stack.push_back(std::move(frame));
    };
    add_frame(self);
    for(const api_row_t& caller : callers)
    {
        add_frame(caller);
    }
    return stack;
}

reader_types::timeline_event_list_t
rocpd_v2_reader::get_correlated_events(const reader_types::timeline_event_t& event)
{
    const auto type = event.unique_identifier.type;
    const auto id   = event.unique_identifier.id;

    reader_types::timeline_event_list_t events;
    if(type == event_type_t::region)
    {
        const std::string sql = fmt::format(
            "SELECT {} FROM rocpd_api_ops AO JOIN rocpd_op O ON O.id = AO.op_id WHERE "
            "AO.api_id = {} ORDER BY O.start",
            rocpd_v2::op_columns(),
            id);
        for(const op_row_t& row : op_rows(*m_backend, sql))
            events.push_back(op_event(row));
    }
    else if(is_op_type(type))
    {
        const std::string sql = fmt::format(
            "SELECT {} FROM rocpd_api_ops AO JOIN rocpd_api A ON A.id = AO.api_id WHERE "
            "AO.op_id = {} ORDER BY A.start",
            k_api_columns,
            id);
        for(const api_row_t& row : api_rows(*m_backend, sql))
            events.push_back(region_event(row));
    }
    return events;
}

// ============================================================================
// Database metadata
// ============================================================================

reader_types::time_window_t
rocpd_v2_reader::get_data_time_range()
{
    size_t global_min = std::numeric_limits<size_t>::max();
    size_t global_max = 0;
    for(const char* source :
        { "rocpd_api", "rocpd_op", "rocpd_monitor WHERE deviceId > 0" })
    {
        m_backend
            ->create_read_statement_executor<time_range_row_t>(
                fmt::format("SELECT MIN(start), MAX(end) FROM {}", source),
                &time_range_row_t::min_start,
                &time_range_row_t::max_end)()
            .for_each([&](const time_range_row_t& row) {
                if(row.min_start.has_value())
                    global_min = std::min(global_min, row.min_start.value());
                if(row.max_end.has_value())
                    global_max = std::max(global_max, row.max_end.value());
            });
    }

    reader_types::time_window_t window;
    if(global_min != std::numeric_limits<size_t>::max())
    {
        window.start = global_min;
        window.end   = global_max;
    }
    return window;
}

reader_types::event_counts_t
rocpd_v2_reader::get_event_counts(const reader_types::time_window_t& window)
{
    reader_types::event_counts_t counts;
    for(const event_type_t type : k_event_types)
    {
        counts[type] = count_of(*m_backend, count_select(type, window));
    }
    return counts;
}

}  // namespace profiler_hub
