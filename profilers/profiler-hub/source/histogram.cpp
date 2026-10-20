// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "reader_impl.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
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

std::string
interval_select(const duration_source_t& source,
                const std::string&       uuid,
                const std::string&       where)
{
    return fmt::format(
        "SELECT ({0}.id + ({1} << 60)) AS event_id, {0}.start AS start_ts, "
        "{0}.end AS end_ts FROM {2}_{3} {0} "
        "LEFT JOIN rocpd_sample S ON S.event_id = {0}.event_id WHERE {4}",
        source.alias,
        source.type_code,
        source.table,
        uuid,
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
sample_where(size_t sample_track_id)
{
    return fmt::format(
        "S.track_id = {0} AND EXISTS (SELECT 1 FROM rocpd_sample WHERE track_id = {0})",
        sample_track_id);
}

std::string
agent_where(const duration_source_t& source, size_t nid, size_t agent_id, size_t queue_id)
{
    return fmt::format("{0}.nid = {1} AND {0}.{2} = {3} AND {0}.queue_id = {4}",
                       source.alias,
                       nid,
                       source.agent_column,
                       agent_id,
                       queue_id);
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
    const auto& uuid = statements.trace_uuid();
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
                             interval_select(
                                 source, uuid, thread_where(source, topology->second)));
            }
            break;
        }
        case reader_types::track_kind_t::thread_sample:
        {
            const auto sample_id = catalog.db_id_by_track.find(track);
            if(sample_id == catalog.db_id_by_track.end()) return {};
            const auto where = sample_where(sample_id->second);
            for(const duration_source_t& source :
                { k_region, k_kernel, k_allocate, k_copy })
            {
                append_union(sql, interval_select(source, uuid, where));
            }
            break;
        }
        case reader_types::track_kind_t::kernel_dispatch_agent_queue:
        case reader_types::track_kind_t::memory_allocate_agent_queue:
        case reader_types::track_kind_t::memory_copy_agent_queue:
        {
            if(!track->node_info) return {};
            const duration_source_t& source =
                track->category == reader_types::track_kind_t::kernel_dispatch_agent_queue
                    ? k_kernel
                : track->category ==
                        reader_types::track_kind_t::memory_allocate_agent_queue
                    ? k_allocate
                    : k_copy;
            sql = interval_select(
                source,
                uuid,
                agent_where(
                    source, track->node_info->node_id, track->agent_id, track->queue_id));
            break;
        }
        case reader_types::track_kind_t::stream:
        {
            if(!track->node_info) return {};
            for(const duration_source_t& source : { k_kernel, k_allocate, k_copy })
            {
                append_union(sql,
                             interval_select(source,
                                             uuid,
                                             stream_where(source,
                                                          track->node_info->node_id,
                                                          track->db_pid,
                                                          track->stream_id)));
            }
            break;
        }
        case reader_types::track_kind_t::pmc_agent: break;
    }
    return sql;
}

struct bucket_accumulator_t
{
    size_t event_count{};
    double value{};
};

reader_types::histogram_bucket_list_t
accumulate_overlaps(const std::vector<data_storage::schema_v3::interval_result>& rows,
                    reader_types::timestamp_ns_t                                origin,
                    reader_types::timestamp_ns_t bucket_size,
                    size_t                       bucket_count)
{
    std::vector<bucket_accumulator_t> buckets(bucket_count);
    for(const auto& row : rows)
    {
        if(row.end_ts <= row.start_ts || bucket_size == 0) continue;
        const size_t end_mark = row.end_ts - 1;
        if(end_mark < origin) continue;

        size_t start_bucket =
            row.start_ts <= origin ? 0 : (row.start_ts - origin) / bucket_size;
        size_t end_bucket = (end_mark - origin) / bucket_size;
        if(start_bucket >= bucket_count) continue;
        if(end_bucket >= bucket_count) end_bucket = bucket_count - 1;
        if(end_bucket < start_bucket) continue;

        for(size_t bucket = start_bucket; bucket <= end_bucket; ++bucket)
        {
            const size_t bucket_start  = origin + bucket * bucket_size;
            const size_t bucket_end    = bucket_start + bucket_size;
            const size_t overlap_start = std::max(row.start_ts, bucket_start);
            const size_t overlap_end   = std::min(row.end_ts, bucket_end);
            if(overlap_end <= overlap_start) continue;
            buckets[bucket].event_count += 1;
            buckets[bucket].value += static_cast<double>(overlap_end - overlap_start);
        }
    }

    reader_types::histogram_bucket_list_t result;
    for(size_t index = 0; index < buckets.size(); ++index)
    {
        if(buckets[index].event_count == 0) continue;
        result.push_back(reader_types::histogram_bucket_t{
            index, buckets[index].event_count, buckets[index].value });
    }
    return result;
}

reader_types::histogram_bucket_list_t
accumulate_samples(const std::vector<data_storage::schema_v3::pmc_sample_result>& rows,
                   reader_types::timestamp_ns_t                                   origin,
                   reader_types::timestamp_ns_t bucket_size,
                   size_t                       bucket_count)
{
    struct sample_accumulator_t
    {
        size_t count{};
        double sum{};
    };
    std::vector<sample_accumulator_t> buckets(bucket_count);
    for(const auto& row : rows)
    {
        if(row.timestamp < origin || bucket_size == 0) continue;
        const size_t index = (row.timestamp - origin) / bucket_size;
        if(index >= bucket_count) continue;
        buckets[index].count += 1;
        buckets[index].sum += row.value;
    }

    reader_types::histogram_bucket_list_t populated;
    for(size_t index = 0; index < buckets.size(); ++index)
    {
        if(buckets[index].count == 0) continue;
        populated.push_back(reader_types::histogram_bucket_t{
            index,
            buckets[index].count,
            buckets[index].sum / static_cast<double>(buckets[index].count) });
    }

    if(populated.size() < 2) return populated;

    reader_types::histogram_bucket_list_t filled;
    for(size_t index = 0; index < populated.size(); ++index)
    {
        filled.push_back(populated[index]);
        if(index + 1 == populated.size()) break;
        for(size_t gap = populated[index].bucket_index + 1;
            gap < populated[index + 1].bucket_index;
            ++gap)
        {
            filled.push_back(
                reader_types::histogram_bucket_t{ gap, 0, populated[index].value });
        }
    }
    return filled;
}

struct histogram_grid_t
{
    reader_types::timestamp_ns_t origin{};
    reader_types::timestamp_ns_t bucket_size{};
    size_t                       bucket_count{};
};

// max_buckets is an upper bound. The grid never has more buckets than
// requested, and a short trace has fewer.
histogram_grid_t
make_grid(const reader_catalog_t& catalog, size_t max_buckets)
{
    if(max_buckets == 0)
    {
        throw std::invalid_argument{ "max_buckets must be non-zero" };
    }

    bool                         any    = false;
    reader_types::timestamp_ns_t origin = 0;
    reader_types::timestamp_ns_t end    = 0;
    for(const auto& track : catalog.tracks)
    {
        if(track->event_count == 0) continue;
        if(!any || track->start_ts < origin) origin = track->start_ts;
        if(!any || track->end_ts > end) end = track->end_ts;
        any = true;
    }
    if(!any || end < origin)
    {
        return {};
    }

    const reader_types::timestamp_ns_t duration = end - origin;
    reader_types::timestamp_ns_t bucket_size = (duration + max_buckets) / max_buckets;
    if(bucket_size == 0) bucket_size = 1;
    const size_t bucket_count =
        duration == 0 ? 1 : (duration + bucket_size) / bucket_size;
    return { origin, bucket_size, bucket_count };
}
}  // namespace

reader_types::histogram_t
reader_t::impl::get_histogram(size_t max_buckets)
{
    const histogram_grid_t    grid = make_grid(*m_catalog, max_buckets);
    reader_types::histogram_t histogram;
    histogram.origin_ts    = grid.origin;
    histogram.bucket_size  = grid.bucket_size;
    histogram.bucket_count = grid.bucket_count;
    histogram.tracks.reserve(m_catalog->tracks.size());

    for(const reader_types::track_info_ptr_t& track : m_catalog->tracks)
    {
        reader_types::track_histogram_t track_histogram;
        track_histogram.track = track;
        if(grid.bucket_count == 0)
        {
            histogram.tracks.push_back(std::move(track_histogram));
            continue;
        }

        if(track->category == reader_types::track_kind_t::pmc_agent)
        {
            if(track->node_info)
            {
                const auto rows = m_read_statements
                                      ->pmc_sample_statement()(track->node_info->node_id,
                                                               track->agent_id,
                                                               track->pmc_id)
                                      .to_vector();
                track_histogram.buckets = accumulate_samples(
                    rows, grid.origin, grid.bucket_size, grid.bucket_count);
            }
        }
        else
        {
            const std::string sql =
                duration_source_sql(*m_catalog, track, *m_read_statements);
            if(!sql.empty())
            {
                track_histogram.buckets =
                    accumulate_overlaps(m_read_statements->query_intervals(sql),
                                        grid.origin,
                                        grid.bucket_size,
                                        grid.bucket_count);
            }
        }
        histogram.tracks.push_back(std::move(track_histogram));
    }
    return histogram;
}

}  // namespace profiler_hub
