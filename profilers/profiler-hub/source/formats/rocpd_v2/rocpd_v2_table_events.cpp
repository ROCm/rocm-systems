// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "event_query.hpp"
#include "rocpd_v2_reader.hpp"

#include "profiler-hub/c/profiler_hub_types.h"

#include <fmt/format.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace profiler_hub
{
namespace
{

struct table_event_row_t
{
    size_t                     id{};
    size_t                     start{};
    size_t                     end{};
    std::optional<size_t>      name_id;
    std::optional<size_t>      category_id;
    std::optional<size_t>      pid;
    std::optional<size_t>      tid;
    std::optional<size_t>      agent_id;
    std::optional<size_t>      queue_id;
    std::optional<size_t>      grid_size_x;
    std::optional<size_t>      grid_size_y;
    std::optional<size_t>      grid_size_z;
    std::optional<size_t>      workgroup_size_x;
    std::optional<size_t>      workgroup_size_y;
    std::optional<size_t>      workgroup_size_z;
    std::optional<size_t>      group_segment_size;
    std::optional<size_t>      private_segment_size;
    std::optional<size_t>      byte_size;
    std::optional<std::string> dst_address;
    std::optional<std::string> src_address;
};

struct sample_row_t
{
    size_t                id{};
    size_t                start{};
    size_t                end{};
    std::optional<double> value;
};

struct id_span_row_t
{
    std::optional<size_t> first_id;
    std::optional<size_t> last_id;
};

using data_storage::bind_types;
using reader_types::event_type_t;
using reader_types::track_kind_t;

std::optional<event_type_t>
reader_event_type(uint32_t event_type)
{
    switch(event_type)
    {
        case PH_EVENT_TYPE_REGION: return event_type_t::region;
        case PH_EVENT_TYPE_KERNEL_DISPATCH: return event_type_t::kernel_dispatch;
        case PH_EVENT_TYPE_MEMORY_COPY: return event_type_t::memory_copy;
        default: return std::nullopt;
    }
}

std::string
join_ids(const std::vector<size_t>& ids)
{
    std::string text;
    for(const size_t id : ids)
    {
        if(!text.empty()) text += ',';
        text += std::to_string(id);
    }
    return text;
}

void
add_and(std::string& where, const std::string& clause)
{
    if(clause.empty()) return;
    where += " AND " + clause;
}

std::string
time_clause(std::string_view alias, const resolved_event_query& query)
{
    if(!query.time.has_value()) return {};
    return fmt::format(
        "{0}.start <= {1} AND {0}.end >= {2}", alias, query.time->end, query.time->start);
}

const reader_types::track_info_ptr_t*
find_track(const reader_catalog_t& catalog, uint32_t track_id)
{
    const auto found =
        std::find_if(catalog.tracks.begin(),
                     catalog.tracks.end(),
                     [track_id](const reader_types::track_info_ptr_t& track) {
                         return track && track->id == track_id;
                     });
    return found != catalog.tracks.end() ? &*found : nullptr;
}

// Returns false when the query asks for tracks and none of them hold this type.
bool
track_clause(const reader_catalog_t&     catalog,
             event_type_t                type,
             const resolved_event_query& query,
             std::string&                clause)
{
    if(query.tracks.empty()) return true;

    const auto  kind = type == event_type_t::region ? track_kind_t::thread
                       : type == event_type_t::kernel_dispatch
                           ? track_kind_t::kernel_dispatch_agent_queue
                           : track_kind_t::memory_copy_agent_queue;
    std::string any;
    for(const uint32_t track_id : query.tracks)
    {
        const auto* track = find_track(catalog, track_id);
        if(track == nullptr || (*track)->category != kind) continue;
        std::string one;
        if(kind == track_kind_t::thread)
        {
            const auto topology = catalog.topology_by_track.find(*track);
            if(topology == catalog.topology_by_track.end()) continue;
            one = fmt::format(
                "A.pid = {} AND A.tid = {}", topology->second.pid, topology->second.tid);
        }
        else
        {
            one = fmt::format("O.gpuId = {} AND O.queueId = {}",
                              (*track)->agent_id,
                              (*track)->queue_id);
        }
        if(!any.empty()) any += " OR ";
        any += "(" + one + ")";
    }
    if(any.empty()) return false;
    clause = "(" + any + ")";
    return true;
}

std::string
string_clause(event_type_t type, const resolved_event_query& query)
{
    if(!query.has_string_filter || query.string_ids.empty()) return {};
    const std::string ids = join_ids(query.string_ids);
    if(type == event_type_t::region) return "A.apiName_id IN (" + ids + ")";
    // An operation without a description is named by its type, and its
    // category is its type.
    return "(O.description_id IN (" + ids + ") OR O.opType_id IN (" + ids + "))";
}

uint64_t
or_zero(const std::optional<size_t>& value)
{
    return value.has_value() ? static_cast<uint64_t>(value.value()) : 0;
}

// Runs "... id > ? AND id <= ? ORDER BY id LIMIT ?" for one page of rows.
template <typename Row, typename Statement, typename Publish>
uint32_t
read_page(Statement& statement,
          uint64_t   after_id,
          uint64_t   last_id,
          uint32_t   capacity,
          Publish&&  publish)
{
    // SQLite binds signed 64-bit integers, so an unbounded range stops at INT64_MAX.
    const uint64_t upper_id =
        std::min<uint64_t>(last_id, static_cast<uint64_t>(INT64_MAX));
    if(after_id >= upper_id) return 0;

    uint32_t filled = 0;
    statement(static_cast<size_t>(after_id),
              static_cast<size_t>(upper_id),
              static_cast<size_t>(capacity))
        .for_each([&](const Row& row) { publish(row, filled++); });
    return filled;
}

}  // namespace

uint32_t
rocpd_v2_reader::read_table_events(const resolved_event_query& query,
                                   uint32_t                    event_type,
                                   uint64_t                    after_id,
                                   uint64_t                    last_id,
                                   ph_table_event*             rows,
                                   uint32_t                    capacity)
{
    if(rows == nullptr || capacity == 0) return 0;
    if(!query.event_types.empty() &&
       std::find(query.event_types.begin(), query.event_types.end(), event_type) ==
           query.event_types.end())
    {
        return 0;
    }
    // Every row belongs to the one synthesized node.
    if(query.node_id.has_value() && *query.node_id != rocpd_v2::k_node_id) return 0;

    if(event_type == PH_EVENT_TYPE_PMC_SAMPLE)
    {
        std::string where = "M.deviceId > 0";
        add_and(where, time_clause("M", query));
        if(query.agent_id.has_value())
        {
            add_and(where, fmt::format("M.deviceId = {}", *query.agent_id));
        }
        if(!query.tracks.empty())
        {
            std::string any;
            for(const uint32_t track_id : query.tracks)
            {
                const auto* track = find_track(*m_catalog, track_id);
                if(track == nullptr || (*track)->category != track_kind_t::pmc_agent)
                    continue;
                if(!any.empty()) any += " OR ";
                any += fmt::format("(M.deviceId = {} AND M.monitorType = {})",
                                   (*track)->agent_id,
                                   rocpd_v2::quote_text((*track)->name));
            }
            if(any.empty()) return 0;
            add_and(where, "(" + any + ")");
        }
        auto statement =
            m_backend->create_read_statement_executor<sample_row_t,
                                                      bind_types<size_t, size_t, size_t>>(
                "SELECT M.id, M.start, M.end, CAST(M.value AS REAL) FROM rocpd_monitor M "
                "WHERE " +
                    where + " AND M.id > ? AND M.id <= ? ORDER BY M.id LIMIT ?",
                &sample_row_t::id,
                &sample_row_t::start,
                &sample_row_t::end,
                &sample_row_t::value);
        return read_page<sample_row_t>(statement,
                                       after_id,
                                       last_id,
                                       capacity,
                                       [&](const sample_row_t& row, uint32_t index) {
                                           ph_table_event& out = rows[index];
                                           out                 = ph_table_event{};
                                           out.id              = row.id;
                                           out.start           = row.start;
                                           out.end             = row.end;
                                           out.value           = row.value.value_or(0.0);
                                       });
    }

    const auto type = reader_event_type(event_type);
    if(!type.has_value()) return 0;
    const bool        is_region = type.value() == event_type_t::region;
    const std::string alias     = is_region ? "A" : "O";

    std::string where =
        is_region
            ? std::string{ "1" }
            : rocpd_v2::op_kind_where("O", type.value() == event_type_t::memory_copy);
    add_and(where, time_clause(alias, query));
    if(query.process_id.has_value())
    {
        add_and(where, fmt::format("A.pid = {}", *query.process_id));
    }
    if(query.thread_id.has_value())
    {
        const auto thread = m_catalog->threads_by_id.find(*query.thread_id);
        if(thread == m_catalog->threads_by_id.end() || !thread->second) return 0;
        add_and(where,
                fmt::format("A.pid = {} AND A.tid = {}",
                            thread->second->parent_process_id.value_or(0),
                            thread->second->thread_id));
    }
    if(query.agent_id.has_value())
    {
        // Host calls run on no agent.
        if(is_region) return 0;
        add_and(where, fmt::format("O.gpuId = {}", *query.agent_id));
    }
    std::string tracks;
    if(!track_clause(*m_catalog, type.value(), query, tracks)) return 0;
    add_and(where, tracks);
    if(query.has_string_filter)
    {
        const std::string strings = string_clause(type.value(), query);
        if(strings.empty()) return 0;
        add_and(where, strings);
    }

    constexpr std::string_view k_no_kernel =
        "NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL";
    constexpr std::string_view k_no_copy = "NULL, NULL, NULL";
    std::string                sql;
    if(is_region)
    {
        sql =
            fmt::format("SELECT A.id, A.start, A.end, A.apiName_id, NULL, A.pid, A.tid, "
                        "NULL, NULL, {}, {} FROM rocpd_api A",
                        k_no_kernel,
                        k_no_copy);
    }
    else
    {
        const bool        is_copy = type.value() == event_type_t::memory_copy;
        const std::string kernel =
            !is_copy && m_has_kernel_api
                ? "K.gridX, K.gridY, K.gridZ, K.workgroupX, K.workgroupY, K.workgroupZ, "
                  "K.groupSegmentSize, K.privateSegmentSize"
                : std::string{ k_no_kernel };
        const std::string copy = is_copy ? copy_api_columns() : std::string{ k_no_copy };
        sql                    = fmt::format(
            "SELECT O.id, O.start, O.end, O.description_id, O.opType_id, A.pid, "
                               "A.tid, O.gpuId, O.queueId, {}, {} FROM rocpd_op O {}",
            kernel,
            copy,
            op_api_joins());
    }
    sql += fmt::format(" WHERE {} AND {}.id > ? AND {}.id <= ? ORDER BY {}.id LIMIT ?",
                       where,
                       alias,
                       alias,
                       alias);

    auto statement =
        m_backend->create_read_statement_executor<table_event_row_t,
                                                  bind_types<size_t, size_t, size_t>>(
            sql,
            &table_event_row_t::id,
            &table_event_row_t::start,
            &table_event_row_t::end,
            &table_event_row_t::name_id,
            &table_event_row_t::category_id,
            &table_event_row_t::pid,
            &table_event_row_t::tid,
            &table_event_row_t::agent_id,
            &table_event_row_t::queue_id,
            &table_event_row_t::grid_size_x,
            &table_event_row_t::grid_size_y,
            &table_event_row_t::grid_size_z,
            &table_event_row_t::workgroup_size_x,
            &table_event_row_t::workgroup_size_y,
            &table_event_row_t::workgroup_size_z,
            &table_event_row_t::group_segment_size,
            &table_event_row_t::private_segment_size,
            &table_event_row_t::byte_size,
            &table_event_row_t::dst_address,
            &table_event_row_t::src_address);

    const auto published_string = [this](std::optional<size_t> id) -> uint64_t {
        return id.has_value() ? canonical_string_id(id.value()) : 0;
    };
    return read_page<table_event_row_t>(
        statement,
        after_id,
        last_id,
        capacity,
        [&](const table_event_row_t& row, uint32_t index) {
            ph_table_event& out = rows[index];
            out                 = ph_table_event{};
            out.id              = row.id;
            out.start           = row.start;
            out.end             = row.end;
            out.name_id         = published_string(
                is_region ? row.name_id : op_name_id(row.name_id, row.category_id));
            out.category_id = published_string(row.category_id);
            out.nid         = rocpd_v2::k_node_id;
            out.process_id  = or_zero(row.pid);
            out.pid         = or_zero(row.pid);
            out.tid         = or_zero(row.tid);
            if(row.pid.has_value() && row.tid.has_value())
            {
                if(const auto thread = thread_info(row.pid.value(), row.tid.value()))
                {
                    out.thread_id = thread->id;
                }
            }
            out.agent_id             = or_zero(row.agent_id);
            out.queue_id             = or_zero(row.queue_id);
            out.grid_size_x          = or_zero(row.grid_size_x);
            out.grid_size_y          = or_zero(row.grid_size_y);
            out.grid_size_z          = or_zero(row.grid_size_z);
            out.workgroup_size_x     = or_zero(row.workgroup_size_x);
            out.workgroup_size_y     = or_zero(row.workgroup_size_y);
            out.workgroup_size_z     = or_zero(row.workgroup_size_z);
            out.group_segment_size   = or_zero(row.group_segment_size);
            out.private_segment_size = or_zero(row.private_segment_size);
            out.size                 = or_zero(row.byte_size);
            out.address              = or_zero(rocpd_v2::parse_address(row.dst_address));
            out.src_address          = or_zero(rocpd_v2::parse_address(row.src_address));
        });
}

bool
rocpd_v2_reader::table_event_id_span(uint32_t  event_type,
                                     uint64_t& first_id,
                                     uint64_t& last_id)
{
    first_id = 0;
    last_id  = 0;

    std::string sql;
    switch(event_type)
    {
        case PH_EVENT_TYPE_REGION: sql = "SELECT MIN(id), MAX(id) FROM rocpd_api"; break;
        case PH_EVENT_TYPE_KERNEL_DISPATCH:
        case PH_EVENT_TYPE_MEMORY_COPY:
            sql = "SELECT MIN(O.id), MAX(O.id) FROM rocpd_op O WHERE " +
                  rocpd_v2::op_kind_where("O", event_type == PH_EVENT_TYPE_MEMORY_COPY);
            break;
        case PH_EVENT_TYPE_PMC_SAMPLE:
            sql = "SELECT MIN(id), MAX(id) FROM rocpd_monitor WHERE deviceId > 0";
            break;
        default: return false;
    }

    bool found = false;
    m_backend
        ->create_read_statement_executor<id_span_row_t>(
            sql, &id_span_row_t::first_id, &id_span_row_t::last_id)()
        .for_each([&](const id_span_row_t& row) {
            if(!row.first_id.has_value() || !row.last_id.has_value()) return;
            first_id = static_cast<uint64_t>(row.first_id.value());
            last_id  = static_cast<uint64_t>(row.last_id.value());
            found    = true;
        });
    return found;
}

namespace
{

struct duration_summary_row
{
    std::optional<std::string> name;
    int64_t                    count{};
    std::optional<int64_t>     min_duration;
    std::optional<int64_t>     max_duration;
    std::optional<int64_t>     total_duration;
};

uint64_t
non_negative(int64_t value)
{
    return value < 0 ? 0 : static_cast<uint64_t>(value);
}

void
add_summary(std::unordered_map<std::string, name_duration_summary_t>& groups,
            std::string                                               name,
            uint64_t                                                  count,
            uint64_t                                                  min_duration,
            uint64_t                                                  max_duration,
            uint64_t                                                  total_duration)
{
    if(count == 0) return;
    name_duration_summary_t& group = groups[std::move(name)];
    if(group.count == 0)
    {
        group.min_duration = min_duration;
        group.max_duration = max_duration;
    }
    else
    {
        group.min_duration = std::min(group.min_duration, min_duration);
        group.max_duration = std::max(group.max_duration, max_duration);
    }
    group.count += count;
    group.total_duration += total_duration;
}

}  // namespace

bool
rocpd_v2_reader::summarize_name_durations(const resolved_event_query& query,
                                          uint32_t                    event_type,
                                          std::vector<name_duration_summary_t>& rows)
{
    rows.clear();
    if(m_backend == nullptr || m_catalog == nullptr) return false;
    if(!query.event_types.empty() &&
       std::find(query.event_types.begin(), query.event_types.end(), event_type) ==
           query.event_types.end())
    {
        return true;
    }
    if(query.node_id.has_value() && *query.node_id != rocpd_v2::k_node_id) return true;

    const auto type = reader_event_type(event_type);
    if(!type.has_value()) return true;
    const bool        is_region = type.value() == event_type_t::region;
    const std::string alias     = is_region ? "A" : "O";

    std::string where =
        is_region
            ? std::string{ "1" }
            : rocpd_v2::op_kind_where("O", type.value() == event_type_t::memory_copy);
    add_and(where, time_clause(alias, query));
    if(query.process_id.has_value())
    {
        add_and(where, fmt::format("A.pid = {}", *query.process_id));
    }
    if(query.thread_id.has_value())
    {
        const auto thread = m_catalog->threads_by_id.find(*query.thread_id);
        if(thread == m_catalog->threads_by_id.end() || !thread->second) return true;
        add_and(where,
                fmt::format("A.pid = {} AND A.tid = {}",
                            thread->second->parent_process_id.value_or(0),
                            thread->second->thread_id));
    }
    if(query.agent_id.has_value())
    {
        if(is_region) return true;
        add_and(where, fmt::format("O.gpuId = {}", *query.agent_id));
    }
    std::string tracks;
    if(!track_clause(*m_catalog, type.value(), query, tracks)) return true;
    add_and(where, tracks);
    if(query.has_string_filter)
    {
        const std::string strings = string_clause(type.value(), query);
        if(strings.empty()) return true;
        add_and(where, strings);
    }

    std::string name_expr;
    std::string from;
    if(is_region)
    {
        name_expr = "IFNULL(N.string, '')";
        from      = "rocpd_api A LEFT JOIN rocpd_string N ON N.id = A.apiName_id";
    }
    else
    {
        name_expr = m_has_kernel_api
                        ? "CASE WHEN IFNULL(KN.string, '') != '' THEN KN.string "
                          "WHEN IFNULL(SD.string, '') != '' THEN SD.string "
                          "ELSE IFNULL(ST.string, '') END"
                        : "CASE WHEN IFNULL(SD.string, '') != '' THEN SD.string "
                          "ELSE IFNULL(ST.string, '') END";
        from      = "rocpd_op O " + op_api_joins() +
               " LEFT JOIN rocpd_string SD ON SD.id = O.description_id"
               " LEFT JOIN rocpd_string ST ON ST.id = O.opType_id";
        if(m_has_kernel_api)
        {
            from += " LEFT JOIN rocpd_string KN ON KN.id = K.kernelName_id";
        }
    }

    const std::string duration = alias + ".end - " + alias + ".start";
    const std::string sql      = "SELECT " + name_expr + ", COUNT(*), MIN(" + duration +
                            "), MAX(" + duration + "), SUM(" + duration + ") FROM " +
                            from + " WHERE " + where + " GROUP BY 1";

    std::unordered_map<std::string, name_duration_summary_t> groups;
    m_backend
        ->create_read_statement_executor<duration_summary_row>(
            sql,
            &duration_summary_row::name,
            &duration_summary_row::count,
            &duration_summary_row::min_duration,
            &duration_summary_row::max_duration,
            &duration_summary_row::total_duration)()
        .for_each([&](const duration_summary_row& row) {
            add_summary(groups,
                        row.name.value_or(""),
                        row.count > 0 ? static_cast<uint64_t>(row.count) : 0,
                        non_negative(row.min_duration.value_or(0)),
                        non_negative(row.max_duration.value_or(0)),
                        non_negative(row.total_duration.value_or(0)));
        });
    rows.reserve(groups.size());
    for(auto& entry : groups)
    {
        entry.second.name = entry.first;
        rows.push_back(std::move(entry.second));
    }
    return true;
}

}  // namespace profiler_hub
