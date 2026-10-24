// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "event_query.hpp"
#include "reader_catalog.hpp"
#include "reader_impl.hpp"

#include "data_storage/backends/database_backend.hpp"
#include "profiler-hub/c/profiler_hub_types.h"

#include <algorithm>
#include <cstring>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace profiler_hub
{
namespace
{

struct table_event_row
{
    size_t                     id{};
    size_t                     start_timestamp{};
    size_t                     end_timestamp{};
    std::optional<size_t>      name_id;
    std::optional<size_t>      category_id;
    std::optional<size_t>      kernel_id;
    size_t                     nid{};
    std::optional<size_t>      pid;
    std::optional<size_t>      tid;
    std::optional<size_t>      agent_id;
    std::optional<size_t>      queue_id;
    std::optional<size_t>      stream_id;
    std::optional<size_t>      src_agent_id;
    std::optional<size_t>      grid_size_x;
    std::optional<size_t>      grid_size_y;
    std::optional<size_t>      grid_size_z;
    std::optional<size_t>      workgroup_size_x;
    std::optional<size_t>      workgroup_size_y;
    std::optional<size_t>      workgroup_size_z;
    std::optional<size_t>      group_segment_size;
    std::optional<size_t>      private_segment_size;
    std::optional<size_t>      byte_size;
    std::optional<size_t>      address;
    std::optional<size_t>      src_address;
    std::optional<std::string> mem_type;
    std::optional<std::string> level;
    std::optional<size_t>      sample_track_id;
};

struct table_source
{
    const char* table{};
    const char* alias{};
    const char* name_id{};
    const char* kernel_id{};
    const char* agent_id{};
    const char* src_agent_id{};
    const char* location{};
    const char* sizes{};
    const char* agent_column{};
    bool        allows_stream{ false };
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
        case PH_EVENT_TYPE_REGION_SAMPLE: return event_type_t::region;
        case PH_EVENT_TYPE_KERNEL_DISPATCH: return event_type_t::kernel_dispatch;
        case PH_EVENT_TYPE_MEMORY_COPY: return event_type_t::memory_copy;
        case PH_EVENT_TYPE_MEMORY_ALLOCATE: return event_type_t::memory_allocate;
        default: return std::nullopt;
    }
}

std::optional<track_kind_t>
agent_queue_kind(event_type_t type)
{
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

bool
serves(track_kind_t kind, event_type_t type)
{
    switch(kind)
    {
        case track_kind_t::kernel_dispatch_agent_queue:
            return type == event_type_t::kernel_dispatch;
        case track_kind_t::memory_allocate_agent_queue:
            return type == event_type_t::memory_allocate;
        case track_kind_t::memory_copy_agent_queue:
            return type == event_type_t::memory_copy;
        case track_kind_t::stream: return type != event_type_t::region;
        case track_kind_t::thread:
        case track_kind_t::thread_sample: return true;
        case track_kind_t::pmc_agent: return false;
    }
    return false;
}

const table_source*
source_for(event_type_t type)
{
    static const table_source region{ "rocpd_region",
                                      "R",
                                      "R.name_id",
                                      "NULL",
                                      "NULL",
                                      "NULL",
                                      "NULL, NULL, NULL",
                                      "NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, "
                                      "NULL, NULL, NULL",
                                      nullptr,
                                      false };
    static const table_source kernel{ "rocpd_kernel_dispatch",
                                      "K",
                                      "K.region_name_id",
                                      "K.kernel_id",
                                      "K.agent_id",
                                      "NULL",
                                      "K.agent_id, K.queue_id, K.stream_id",
                                      "K.grid_size_x, K.grid_size_y, K.grid_size_z, "
                                      "K.workgroup_size_x, K.workgroup_size_y, "
                                      "K.workgroup_size_z, K.group_segment_size, "
                                      "K.private_segment_size, NULL, NULL, NULL",
                                      "agent_id",
                                      true };
    static const table_source copy{ "rocpd_memory_copy",
                                    "MC",
                                    "MC.name_id",
                                    "NULL",
                                    "MC.dst_agent_id",
                                    "MC.src_agent_id",
                                    "MC.dst_agent_id, MC.queue_id, MC.stream_id",
                                    "NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, "
                                    "MC.size, MC.dst_address, MC.src_address",
                                    "dst_agent_id",
                                    true };
    static const table_source alloc{ "rocpd_memory_allocate",
                                     "MA",
                                     "NULL",
                                     "NULL",
                                     "MA.agent_id",
                                     "NULL",
                                     "MA.agent_id, MA.queue_id, MA.stream_id",
                                     "NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, "
                                     "MA.size, MA.address, NULL",
                                     "agent_id",
                                     true };
    switch(type)
    {
        case event_type_t::region: return &region;
        case event_type_t::kernel_dispatch: return &kernel;
        case event_type_t::memory_copy: return &copy;
        case event_type_t::memory_allocate: return &alloc;
        case event_type_t::sample:
        case event_type_t::pmc_event: break;
    }
    return nullptr;
}

std::string
join_ids(const std::vector<size_t>& ids)
{
    std::string text;
    for(size_t index = 0; index < ids.size(); ++index)
    {
        if(index != 0)
        {
            text += ',';
        }
        text += std::to_string(ids[index]);
    }
    return text;
}

std::string
id_or_unrecorded(const std::string& column, size_t value)
{
    if(value == 0)
    {
        return "(" + column + " = 0 OR " + column + " IS NULL)";
    }
    return column + " = " + std::to_string(value);
}

void
add_and(std::string& where, const std::string& clause)
{
    if(clause.empty()) return;
    if(!where.empty())
    {
        where += " AND ";
    }
    where += clause;
}

std::string
string_clause(const table_source& source, event_type_t type, const resolved_event_query& query)
{
    if(!query.has_string_filter) return {};

    std::string clause;
    const auto  append_in = [&](const char* column, const std::vector<size_t>& ids) {
        if(column == nullptr || std::strcmp(column, "NULL") == 0 || ids.empty()) return;
        if(!clause.empty())
        {
            clause += " OR ";
        }
        clause += std::string(column) + " IN (" + join_ids(ids) + ")";
    };

    append_in(source.name_id, query.string_ids);
    if(type == event_type_t::kernel_dispatch)
    {
        append_in("K.kernel_id", query.kernel_symbol_ids);
    }
    if(type == event_type_t::memory_copy)
    {
        append_in("MC.region_name_id", query.string_ids);
    }
    if(query.include_category)
    {
        append_in("E.category_id", query.string_ids);
    }
    if(clause.empty()) return {};
    return "(" + clause + ")";
}

std::string
track_clause(const table_source&        source,
             event_type_t               type,
             const reader_catalog_t&    catalog,
             const resolved_event_query& query)
{
    if(query.tracks.empty()) return {};

    std::string clause;
    for(const uint32_t track_id : query.tracks)
    {
        const auto found = std::find_if(
            catalog.tracks.begin(),
            catalog.tracks.end(),
            [track_id](const reader_types::track_info_ptr_t& track) {
                return track && track->id == track_id;
            });
        if(found == catalog.tracks.end() || !serves((*found)->category, type))
        {
            continue;
        }
        const auto& track = **found;
        std::string one;
        if(track.category == track_kind_t::thread)
        {
            const auto topo = catalog.topology_by_track.find(*found);
            if(topo == catalog.topology_by_track.end()) continue;
            one = std::string(source.alias) + ".nid = " + std::to_string(topo->second.nid) +
                  " AND " + source.alias + ".pid = " + std::to_string(topo->second.pid) +
                  " AND " + source.alias + ".tid = " + std::to_string(topo->second.tid) +
                  " AND S.track_id IS NULL";
        }
        else if(track.category == track_kind_t::thread_sample)
        {
            const auto db_id = catalog.db_id_by_track.find(*found);
            if(db_id == catalog.db_id_by_track.end()) continue;
            one = "S.track_id = " + std::to_string(db_id->second);
        }
        else if(track.category == track_kind_t::stream)
        {
            if(!source.allows_stream || track.node_info == nullptr) continue;
            one = std::string(source.alias) +
                  ".nid = " + std::to_string(track.node_info->node_id) + " AND " +
                  source.alias + ".pid = " + std::to_string(track.db_pid) + " AND " +
                  source.alias + ".stream_id = " + std::to_string(track.stream_id);
        }
        else if(source.agent_column != nullptr && track.node_info != nullptr)
        {
            const std::string agent =
                std::string(source.alias) + "." + source.agent_column;
            const std::string queue = std::string(source.alias) + ".queue_id";
            one = std::string(source.alias) +
                  ".nid = " + std::to_string(track.node_info->node_id) + " AND " +
                  id_or_unrecorded(agent, track.agent_id) + " AND " +
                  id_or_unrecorded(queue, track.queue_id);
        }
        if(one.empty()) continue;
        if(!clause.empty())
        {
            clause += " OR ";
        }
        clause += "(" + one + ")";
    }
    if(clause.empty()) return {};
    return "(" + clause + ")";
}

uint64_t
or_zero(const std::optional<size_t>& value)
{
    return value.has_value() ? static_cast<uint64_t>(value.value()) : 0;
}

void
copy_text(char* dest, size_t size, const std::optional<std::string>& value)
{
    if(size == 0) return;
    dest[0] = '\0';
    if(!value.has_value()) return;
    const size_t count = std::min(size - 1, value->size());
    std::memcpy(dest, value->data(), count);
    dest[count] = '\0';
}

std::optional<size_t>
resolved_track_id(const reader_catalog_t& catalog,
                  event_type_t            type,
                  const table_event_row&  row)
{
    const size_t pid = row.pid.value_or(0);
    const size_t tid = row.tid.value_or(0);
    if(const auto kind = agent_queue_kind(type); kind.has_value())
    {
        const auto it = catalog.agent_queue_track_by_key.find(agent_queue_track_key_t{
            kind.value(), row.nid, row.agent_id.value_or(0), row.queue_id.value_or(0) });
        if(it != catalog.agent_queue_track_by_key.end() && it->second)
        {
            return it->second->id;
        }
    }
    if(type != event_type_t::region && row.stream_id.has_value())
    {
        const auto it = catalog.stream_track_by_key.find(
            stream_track_key_t{ row.nid, pid, row.stream_id.value() });
        if(it != catalog.stream_track_by_key.end() && it->second)
        {
            return it->second->id;
        }
    }
    if(row.sample_track_id.has_value())
    {
        const auto it = catalog.sample_track_by_db_id.find(row.sample_track_id.value());
        if(it != catalog.sample_track_by_db_id.end() && it->second)
        {
            return it->second->id;
        }
    }
    const auto it =
        catalog.track_by_topology.find(topology_key_t{ row.nid, pid, tid });
    if(it != catalog.track_by_topology.end() && it->second)
    {
        return it->second->id;
    }
    return std::nullopt;
}

void
publish(const reader_catalog_t& catalog, const table_event_row& row, ph_table_event& out)
{
    out                = ph_table_event{};
    out.id             = row.id;
    out.start          = row.start_timestamp;
    out.end            = row.end_timestamp;
    out.name_id        = or_zero(row.name_id);
    out.category_id    = or_zero(row.category_id);
    out.kernel_id      = or_zero(row.kernel_id);
    out.nid            = row.nid;
    out.process_id     = or_zero(row.pid);
    out.thread_id      = or_zero(row.tid);
    out.pid            = out.process_id;
    out.tid            = out.thread_id;
    out.agent_id       = or_zero(row.agent_id);
    out.src_agent_id   = or_zero(row.src_agent_id);
    out.queue_id       = or_zero(row.queue_id);
    out.stream_id      = or_zero(row.stream_id);
    out.grid_size_x    = or_zero(row.grid_size_x);
    out.grid_size_y    = or_zero(row.grid_size_y);
    out.grid_size_z    = or_zero(row.grid_size_z);
    out.workgroup_size_x = or_zero(row.workgroup_size_x);
    out.workgroup_size_y = or_zero(row.workgroup_size_y);
    out.workgroup_size_z = or_zero(row.workgroup_size_z);
    out.group_segment_size   = or_zero(row.group_segment_size);
    out.private_segment_size = or_zero(row.private_segment_size);
    out.size                 = or_zero(row.byte_size);
    out.address              = or_zero(row.address);
    out.src_address          = or_zero(row.src_address);
    copy_text(out.mem_type, sizeof(out.mem_type), row.mem_type);
    copy_text(out.level, sizeof(out.level), row.level);

    if(row.pid.has_value())
    {
        const auto it = catalog.processes_by_id.find(row.pid.value());
        if(it != catalog.processes_by_id.end() && it->second)
        {
            out.pid = it->second->pid;
        }
    }
    if(row.tid.has_value())
    {
        const auto it = catalog.threads_by_id.find(row.tid.value());
        if(it != catalog.threads_by_id.end() && it->second)
        {
            out.tid = it->second->thread_id;
        }
    }
    if(row.kernel_id.has_value())
    {
        const auto it = catalog.kernel_symbols_by_id.find(row.kernel_id.value());
        if(it != catalog.kernel_symbols_by_id.end() && it->second)
        {
            out.static_group_segment_size = or_zero(it->second->group_segment_size);
            out.static_private_segment_size =
                or_zero(it->second->private_segment_size);
        }
    }
}

std::string
select_list(const table_source& source)
{
    const char* type_expr  = std::string(source.table) == "rocpd_memory_allocate"
                                 ? "MA.type"
                                 : "NULL";
    const char* level_expr = std::string(source.table) == "rocpd_memory_allocate"
                                 ? "MA.level"
                                 : "NULL";
    return std::string(source.alias) + ".id, " + source.alias + ".start, " + source.alias +
           ".end, " + source.name_id + ", E.category_id, " + source.kernel_id + ", " +
           source.alias + ".nid, " + source.alias + ".pid, " + source.alias + ".tid, " +
           source.location + ", " + source.src_agent_id + ", " + source.sizes + ", " +
           type_expr + ", " + level_expr + ", S.track_id";
}

}  // namespace

uint32_t
reader_t::impl::read_table_events(const resolved_event_query& query,
                                  uint32_t                    event_type,
                                  uint64_t                    after_id,
                                  ph_table_event*             rows,
                                  uint32_t                    capacity)
{
    if(rows == nullptr || capacity == 0 || m_catalog == nullptr) return 0;
    const auto type = reader_event_type(event_type);
    if(!type.has_value()) return 0;
    if(!query.event_types.empty() &&
       std::find(query.event_types.begin(), query.event_types.end(), event_type) ==
           query.event_types.end())
    {
        return 0;
    }

    const table_source* source = source_for(type.value());
    if(source == nullptr) return 0;

    const std::string text_filter = string_clause(*source, type.value(), query);
    if(query.has_string_filter && text_filter.empty()) return 0;

    const std::string tracks = track_clause(*source, type.value(), *m_catalog, query);
    if(!query.tracks.empty() && tracks.empty()) return 0;

    std::string where;
    if(query.time.has_value())
    {
        add_and(where,
                std::string(source->alias) + ".start <= " +
                    std::to_string(query.time->end) + " AND " + source->alias +
                    ".end >= " + std::to_string(query.time->start));
    }
    if(query.node_id.has_value())
    {
        add_and(where,
                std::string(source->alias) + ".nid = " + std::to_string(*query.node_id));
    }
    if(query.process_id.has_value())
    {
        add_and(where,
                std::string(source->alias) +
                    ".pid = " + std::to_string(*query.process_id));
    }
    if(query.thread_id.has_value())
    {
        add_and(where,
                std::string(source->alias) +
                    ".tid = " + std::to_string(*query.thread_id));
    }
    if(type.value() == event_type_t::region)
    {
        if(event_type == PH_EVENT_TYPE_REGION_SAMPLE)
        {
            add_and(where, "S.track_id IS NOT NULL");
        }
        else
        {
            add_and(where, "S.track_id IS NULL");
        }
    }
    add_and(where, text_filter);
    add_and(where, tracks);

    const std::string sql =
        "SELECT " + select_list(*source) + " FROM " + source->table + "_" +
        m_backend->get_uuid() + " " + source->alias +
        " INNER JOIN rocpd_event E ON E.id = " + source->alias +
        ".event_id LEFT JOIN rocpd_sample S ON S.event_id = " + source->alias +
        ".event_id WHERE " + (where.empty() ? std::string() : where + " AND ") +
        source->alias + ".id > ? ORDER BY " + source->alias + ".id LIMIT ?";

    auto statement =
        m_backend->create_read_statement_executor<table_event_row, bind_types<size_t, size_t>>(
            sql,
            &table_event_row::id,
            &table_event_row::start_timestamp,
            &table_event_row::end_timestamp,
            &table_event_row::name_id,
            &table_event_row::category_id,
            &table_event_row::kernel_id,
            &table_event_row::nid,
            &table_event_row::pid,
            &table_event_row::tid,
            &table_event_row::agent_id,
            &table_event_row::queue_id,
            &table_event_row::stream_id,
            &table_event_row::src_agent_id,
            &table_event_row::grid_size_x,
            &table_event_row::grid_size_y,
            &table_event_row::grid_size_z,
            &table_event_row::workgroup_size_x,
            &table_event_row::workgroup_size_y,
            &table_event_row::workgroup_size_z,
            &table_event_row::group_segment_size,
            &table_event_row::private_segment_size,
            &table_event_row::byte_size,
            &table_event_row::address,
            &table_event_row::src_address,
            &table_event_row::mem_type,
            &table_event_row::level,
            &table_event_row::sample_track_id);

    std::unordered_set<size_t> allowed;
    allowed.reserve(query.tracks.size());
    for(const uint32_t track_id : query.tracks)
    {
        allowed.insert(track_id);
    }

    uint64_t cursor = after_id;
    uint32_t filled = 0;
    while(filled < capacity)
    {
        size_t   got     = 0;
        uint64_t scanned = cursor;
        bool     stop    = false;
        statement(static_cast<size_t>(cursor), static_cast<size_t>(capacity))
            .for_each([&](const table_event_row& row) {
                if(stop) return;
                ++got;
                scanned = row.id;
                if(!allowed.empty())
                {
                    const auto track = resolved_track_id(*m_catalog, type.value(), row);
                    if(!track.has_value() || allowed.find(track.value()) == allowed.end())
                    {
                        return;
                    }
                }
                publish(*m_catalog, row, rows[filled]);
                ++filled;
                if(filled == capacity)
                {
                    stop = true;
                }
            });
        if(got == 0 || got < capacity) break;
        cursor = scanned;
    }
    return filled;
}

}  // namespace profiler_hub
