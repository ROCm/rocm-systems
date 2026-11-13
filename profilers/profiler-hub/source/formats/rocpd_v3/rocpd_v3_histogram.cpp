// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "histogram_grid.hpp"
#include "rocpd_v3_reader.hpp"

#include <fmt/format.h>

#include <string>
#include <utility>
#include <vector>

namespace profiler_hub
{
namespace
{
struct duration_source_t
{
    const char* table;
    const char* alias;
    const char* agent_column;
    int         type_code;
};

constexpr duration_source_t k_region{ "rocpd_region", "R", nullptr, 1 };
constexpr duration_source_t k_kernel{ "rocpd_kernel_dispatch", "K", "agent_id", 2 };
constexpr duration_source_t k_allocate{ "rocpd_memory_allocate", "MA", "agent_id", 3 };
constexpr duration_source_t k_copy{ "rocpd_memory_copy", "MC", "dst_agent_id", 4 };
constexpr duration_source_t k_hip{ "rocpd_hip_event", "HE", "agent_id", 5 };

std::string
interval_select(const duration_source_t& source,
                const std::string&       uuid,
                const std::string&       sample_table,
                const std::string&       where)
{
    return fmt::format(
        "SELECT ({0}.id + ({1} << 60)) AS event_id, {0}.start AS start_ts, "
        "{0}.end AS end_ts FROM {2}_{3} {0} "
        "LEFT JOIN {4} S ON S.event_id = {0}.event_id WHERE {5}",
        source.alias,
        source.type_code,
        source.table,
        uuid,
        sample_table,
        where);
}

std::string
thread_where(const duration_source_t& source, const topology_key_t& topology)
{
    return fmt::format(
        "{0}.nid = {1} AND {0}.pid = {2} AND {0}.tid = {3} AND S.track_id IS NULL",
        source.alias,
        topology.nid,
        topology.pid,
        topology.tid);
}

std::string
sample_where(size_t sample_track_id, const std::string& sample_table)
{
    return fmt::format(
        "S.track_id = {0} AND EXISTS (SELECT 1 FROM {1} WHERE track_id = {0})",
        sample_track_id,
        sample_table);
}

// A recorded id compares equal, so the column can be indexed. 0 is the
// unrecorded value and also has to match NULL.
std::string
match_id(const std::string& column, size_t id)
{
    if(id == 0) return fmt::format("({0} = 0 OR {0} IS NULL)", column);
    return fmt::format("{0} = {1}", column, id);
}

std::string
agent_where(const duration_source_t& source, size_t nid, size_t agent_id, size_t queue_id)
{
    return fmt::format(
        "{0}.nid = {1} AND {2} AND {3}",
        source.alias,
        nid,
        match_id(fmt::format("{}.{}", source.alias, source.agent_column), agent_id),
        match_id(fmt::format("{}.queue_id", source.alias), queue_id));
}

std::string
stream_where(const duration_source_t& source, size_t nid, size_t db_pid, size_t stream_id)
{
    return fmt::format("{0}.nid = {1} AND {0}.pid = {2} AND {0}.stream_id = {3}",
                       source.alias,
                       nid,
                       db_pid,
                       stream_id);
}

void
append_union(std::string& sql, const std::string& select)
{
    if(!sql.empty()) sql += " UNION ALL ";
    sql += select;
}

std::string
duration_source_sql(const reader_catalog_t&                         catalog,
                    const reader_types::track_info_ptr_t&           track,
                    const data_storage::schema_v3::read_statements& statements)
{
    const auto& uuid         = statements.trace_uuid();
    const auto  sample_table = statements.joined_sample_table();
    std::string sql;

    switch(track->category)
    {
        case reader_types::track_kind_t::thread:
        {
            const auto topology = catalog.topology_by_track.find(track);
            if(topology == catalog.topology_by_track.end()) return {};
            for(const duration_source_t& source :
                { k_region, k_kernel, k_allocate, k_copy })
            {
                append_union(sql,
                             interval_select(source,
                                             uuid,
                                             sample_table,
                                             thread_where(source, topology->second)));
            }
            break;
        }
        case reader_types::track_kind_t::thread_sample:
        {
            const auto sample_id = catalog.db_id_by_track.find(track);
            if(sample_id == catalog.db_id_by_track.end()) return {};
            const auto where = sample_where(sample_id->second, sample_table);
            for(const duration_source_t& source :
                { k_region, k_kernel, k_allocate, k_copy })
            {
                append_union(sql, interval_select(source, uuid, sample_table, where));
            }
            break;
        }
        case reader_types::track_kind_t::kernel_dispatch_agent_queue:
        case reader_types::track_kind_t::memory_allocate_agent_queue:
        case reader_types::track_kind_t::memory_copy_agent_queue:
        case reader_types::track_kind_t::hip_event_agent_queue:
        {
            if(!track->node_info) return {};
            const duration_source_t& source =
                track->category == reader_types::track_kind_t::kernel_dispatch_agent_queue
                    ? k_kernel
                : track->category ==
                        reader_types::track_kind_t::memory_allocate_agent_queue
                    ? k_allocate
                : track->category == reader_types::track_kind_t::hip_event_agent_queue
                    ? k_hip
                    : k_copy;
            sql = interval_select(
                source,
                uuid,
                sample_table,
                agent_where(
                    source, track->node_info->node_id, track->agent_id, track->queue_id));
            break;
        }
        case reader_types::track_kind_t::stream:
        {
            if(!track->node_info) return {};
            for(const duration_source_t& source : { k_kernel, k_allocate, k_copy, k_hip })
            {
                if(source.table == k_hip.table && !statements.hip_event_table_present())
                {
                    continue;
                }
                append_union(sql,
                             interval_select(source,
                                             uuid,
                                             sample_table,
                                             stream_where(source,
                                                          track->node_info->node_id,
                                                          track->db_pid,
                                                          track->stream_id)));
            }
            break;
        }
        case reader_types::track_kind_t::pmc_agent:
        case reader_types::track_kind_t::pmc_kernel:
        case reader_types::track_kind_t::pmc_memory: break;
    }
    return sql;
}
}  // namespace

reader_types::histogram_t
rocpd_v3_reader::get_histogram(size_t max_buckets)
{
    return get_histogram(make_histogram_grid(*m_catalog, max_buckets));
}

reader_types::histogram_t
rocpd_v3_reader::get_histogram(const histogram_grid_t& grid)
{
    return build_histogram(
        *m_catalog,
        grid,
        [this](const reader_types::track_info_ptr_t& track,
               const histogram_grid_t& grid) -> reader_types::histogram_bucket_list_t {
            if(track->category == reader_types::track_kind_t::pmc_agent)
            {
                if(!track->node_info) return {};
                return accumulate_samples(
                    m_read_statements
                        ->pmc_sample_statement()(
                            track->node_info->node_id, track->agent_id, track->pmc_id)
                        .to_vector(),
                    grid);
            }
            const std::string sql =
                duration_source_sql(*m_catalog, track, *m_read_statements);
            if(sql.empty()) return {};
            return accumulate_overlaps(m_read_statements->query_intervals(sql), grid);
        });
}

}  // namespace profiler_hub
