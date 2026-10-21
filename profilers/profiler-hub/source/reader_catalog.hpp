// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "profiler-hub/cpp/reader_types.hpp"

#include "data_storage/read_statements.hpp"

#include <cstddef>
#include <functional>
#include <limits>
#include <optional>
#include <unordered_map>
#include <vector>

namespace profiler_hub
{

struct topology_key_t
{
    size_t nid{};
    size_t pid{};
    size_t tid{};

    bool operator==(const topology_key_t&) const = default;
};

struct topology_key_hash_t
{
    size_t operator()(const topology_key_t& k) const
    {
        size_t h = std::hash<size_t>{}(k.nid);
        h ^= std::hash<size_t>{}(k.pid) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<size_t>{}(k.tid) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct stream_track_key_t
{
    size_t nid{};
    size_t pid{};
    size_t stream_id{};

    bool operator==(const stream_track_key_t&) const = default;
};

struct stream_track_key_hash_t
{
    size_t operator()(const stream_track_key_t& key) const
    {
        size_t hash = std::hash<size_t>{}(key.nid);
        hash ^= std::hash<size_t>{}(key.pid) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^=
            std::hash<size_t>{}(key.stream_id) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        return hash;
    }
};

struct agent_queue_track_key_t
{
    reader_types::track_kind_t category{};
    size_t                     nid{};
    size_t                     agent_id{};
    size_t                     queue_id{};

    bool operator==(const agent_queue_track_key_t&) const = default;
};

struct agent_queue_track_key_hash_t
{
    size_t operator()(const agent_queue_track_key_t& key) const
    {
        size_t hash = std::hash<int>{}(static_cast<int>(key.category));
        hash ^= std::hash<size_t>{}(key.nid) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^=
            std::hash<size_t>{}(key.agent_id) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^=
            std::hash<size_t>{}(key.queue_id) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        return hash;
    }
};

/** @brief Accumulated event count plus [min start, max end] time span for
 *         a track being derived from one or more grouped event queries. */
struct track_range_stats_t
{
    size_t count{};
    size_t min_start{ std::numeric_limits<size_t>::max() };
    size_t max_end{};
    bool   has_range{};

    void merge(std::optional<size_t> row_min_start, std::optional<size_t> row_max_end)
    {
        if(row_min_start.has_value())
        {
            min_start = std::min(min_start, row_min_start.value());
            has_range = true;
        }
        if(row_max_end.has_value())
        {
            max_end   = std::max(max_end, row_max_end.value());
            has_range = true;
        }
    }
};

/** @brief track_range_stats_t plus the owning node, process, and thread.
 *         tid is the OS thread id (rocpd_info_thread.tid) and is what the
 *         track name prints. thread_row_id is rocpd_info_thread.id, the key
 *         threads_by_id is indexed by. The read path identifies a sample
 *         track by db track id, not by topology. */
struct thread_sample_stats_t : track_range_stats_t
{
    size_t nid{};
    size_t pid{};
    size_t tid{};
    size_t thread_row_id{};
};

/**
 * @brief Cached, read-only trace metadata: nodes/processes/threads/agents/
 *        tracks/stream-to-hardware links/code objects/kernel symbols/streams/
 *        queues/pmc info, plus the lookup indices built alongside them.
 *
 * Identical for a given trace file regardless of which SQLite connection
 * queried it -- immutable once built, safe to share (via shared_ptr) across
 * multiple reader_t instances/connections instead of every one rebuilding
 * its own copy.
 *
 * Build methods are grouped by dependency (see each group's comment); a
 * caller building this incrementally (e.g. populate_reader_catalog's parallel
 * bootstrap) MUST respect that order. build_all() does so sequentially and
 * is what a standalone (non-pooled) reader_t uses.
 */
struct reader_catalog_t
{
    // ---- Sequential prefix: each depends on the previous. ----
    void build_string_list(data_storage::schema_v3::read_statements& stmts);
    void build_nodes(data_storage::schema_v3::read_statements& stmts);
    void build_processes(data_storage::schema_v3::read_statements& stmts);
    void build_threads(data_storage::schema_v3::read_statements& stmts);
    void build_agents(data_storage::schema_v3::read_statements& stmts);

    // ---- Independent of the lists below, once the prefix above is done. ----
    void build_tracks(data_storage::schema_v3::read_statements& stmts);

    // ---- Depends on tracks (build_tracks() above). ----
    void build_stream_hardware_links(data_storage::schema_v3::read_statements& stmts);

    // ---- Independent of each other and of tracks, once the prefix is done. ----
    void build_code_objects(data_storage::schema_v3::read_statements& stmts);
    void build_streams(data_storage::schema_v3::read_statements& stmts);
    void build_queues(data_storage::schema_v3::read_statements& stmts);
    void build_pmc_infos(data_storage::schema_v3::read_statements& stmts);

    // ---- Depends on code_objects (build_code_objects() above). ----
    void build_kernel_symbols(data_storage::schema_v3::read_statements& stmts);

    /** @brief Builds every category sequentially, in dependency order. */
    void build_all(data_storage::schema_v3::read_statements& stmts);

    reader_types::node_info_list_t            nodes;
    reader_types::process_info_list_t         processes;
    reader_types::thread_info_list_t          threads;
    reader_types::agent_info_list_t           agents;
    reader_types::track_info_list_t           tracks;
    reader_types::stream_hardware_link_list_t stream_hardware_links;
    reader_types::kernel_symbol_info_list_t   kernel_symbols;
    reader_types::code_object_info_list_t     code_objects;
    reader_types::stream_info_list_t          streams;
    reader_types::queue_info_list_t           queues;
    reader_types::pmc_info_list_t             pmc_infos;

    std::unordered_map<size_t, std::string> strings_by_id;

    std::unordered_map<size_t, reader_types::node_info_ptr_t>    nodes_by_id;
    std::unordered_map<size_t, reader_types::process_info_ptr_t> processes_by_id;
    std::unordered_map<size_t, reader_types::thread_info_ptr_t>  threads_by_id;
    std::unordered_map<size_t, reader_types::agent_info_ptr_t>   agents_by_id;
    std::unordered_map<size_t, reader_types::kernel_symbol_info_ptr_t>
                                                                     kernel_symbols_by_id;
    std::unordered_map<size_t, reader_types::code_object_info_ptr_t> code_objects_by_id;

    // Track lookup maps (populated by build_tracks()).
    std::
        unordered_map<topology_key_t, reader_types::track_info_ptr_t, topology_key_hash_t>
                                                                       track_by_topology;
    std::unordered_map<reader_types::track_info_ptr_t, topology_key_t> topology_by_track;
    std::unordered_map<reader_types::track_info_ptr_t, size_t>         db_id_by_track;
    std::unordered_map<size_t, reader_types::track_info_ptr_t> sample_track_by_db_id;
    std::unordered_map<stream_track_key_t,
                       reader_types::track_info_ptr_t,
                       stream_track_key_hash_t>
        stream_track_by_key;
    std::unordered_map<agent_queue_track_key_t,
                       reader_types::track_info_ptr_t,
                       agent_queue_track_key_hash_t>
        agent_queue_track_by_key;

private:
    void link_owner(reader_types::track_info_t& track,
                    size_t                      nid,
                    std::optional<size_t>       pid = std::nullopt,
                    std::optional<size_t>       tid = std::nullopt) const;

    // Discovers "thread" tracks directly from the duration-event tables
    // (region/kernel_dispatch/memory_allocate/memory_copy, grouped by
    // (nid,pid,tid)) -- a track only exists if this returns a non-empty
    // group for it, matching optiq's own "no rocpd_track dependency"
    // discovery philosophy (see build_tracks()'s doc comment). Returns
    // per-key summed event count plus [min start, max end] time span;
    // called from build_tracks().
    [[nodiscard]] std::
        unordered_map<topology_key_t, track_range_stats_t, topology_key_hash_t>
        discover_thread_tracks(data_storage::schema_v3::read_statements& stmts);

    // Discovers "thread_sample" tracks: duration events tagged with a real
    // rocpd_sample.track_id (via writer_t::register_track_info() +
    // trace_environment_t::track_name), grouped by that db track id -- a
    // separate track per named sample, distinct from the untagged "thread"
    // track discover_thread_tracks() returns for the same (nid,pid,tid).
    // Called from build_tracks().
    [[nodiscard]] std::unordered_map<size_t, thread_sample_stats_t>
    discover_thread_sample_tracks(data_storage::schema_v3::read_statements& stmts);

    // Appends optiq-parity category tracks (kernel-dispatch/memory-allocate/
    // memory-copy, per agent+queue and per host-stream) to `tracks`,
    // continuing synthetic ids from `next_synthetic_id`. Called from
    // build_tracks().
    void add_category_tracks(data_storage::schema_v3::read_statements& stmts,
                             size_t&                                   next_synthetic_id);
};

}  // namespace profiler_hub
