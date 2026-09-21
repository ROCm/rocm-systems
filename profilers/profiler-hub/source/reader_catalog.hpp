// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "profiler-hub/cpp/reader_types.hpp"

#include "data_storage/read_statements.hpp"

#include <cstddef>
#include <functional>
#include <unordered_map>
#include <vector>

namespace profiler_hub
{

struct topology_key_t
{
    size_t nid{};
    size_t pid{};
    size_t tid{};

    bool operator==(const topology_key_t& other) const
    {
        return nid == other.nid && pid == other.pid && tid == other.tid;
    }
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

/**
 * @brief Cached, read-only trace metadata: nodes/processes/threads/agents/
 *        tracks/code objects/kernel symbols/streams/queues/pmc info, plus
 *        the lookup indices built alongside them.
 *
 * Identical for a given trace file regardless of which SQLite connection
 * queried it -- immutable once built, safe to share (via shared_ptr) across
 * multiple reader_t instances/connections instead of every one rebuilding
 * its own copy.
 *
 * Build methods are grouped by dependency (see each group's comment); a
 * caller building this incrementally (e.g. connection_pool's parallel
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

    // ---- Independent of each other, once the prefix above is done. ----
    void build_agents(data_storage::schema_v3::read_statements& stmts);
    void build_tracks(data_storage::schema_v3::read_statements& stmts);
    void build_code_objects(data_storage::schema_v3::read_statements& stmts);
    void build_streams(data_storage::schema_v3::read_statements& stmts);
    void build_queues(data_storage::schema_v3::read_statements& stmts);
    void build_pmc_infos(data_storage::schema_v3::read_statements& stmts);

    // ---- Depends on code_objects (build_code_objects() above). ----
    void build_kernel_symbols(data_storage::schema_v3::read_statements& stmts);

    /** @brief Builds every category sequentially, in dependency order. */
    void build_all(data_storage::schema_v3::read_statements& stmts);

    reader_types::node_info_list_t          nodes;
    reader_types::process_info_list_t       processes;
    reader_types::thread_info_list_t        threads;
    reader_types::agent_info_list_t         agents;
    reader_types::track_info_list_t         tracks;
    reader_types::kernel_symbol_info_list_t kernel_symbols;
    reader_types::code_object_info_list_t   code_objects;
    reader_types::stream_info_list_t        streams;
    reader_types::queue_info_list_t         queues;
    reader_types::pmc_info_list_t           pmc_infos;

    std::unordered_map<size_t, std::string> string_utility;

    std::unordered_map<size_t, reader_types::node_info_ptr_t>    node_utility;
    std::unordered_map<size_t, reader_types::process_info_ptr_t> process_utility;
    std::unordered_map<size_t, reader_types::thread_info_ptr_t>  thread_utility;
    std::unordered_map<size_t, reader_types::agent_info_ptr_t>   agent_utility;
    std::unordered_map<size_t, reader_types::track_info_ptr_t>   track_utility;
    std::unordered_map<size_t, reader_types::kernel_symbol_info_ptr_t>
        kernel_symbol_utility;
    std::unordered_map<size_t, reader_types::code_object_info_ptr_t> code_object_utility;
    std::unordered_map<size_t, reader_types::stream_info_ptr_t>      stream_utility;
    std::unordered_map<size_t, reader_types::queue_info_ptr_t>       queue_utility;
    std::unordered_map<size_t, reader_types::pmc_info_ptr_t>         pmc_utility;

    // Track lookup maps (populated by build_tracks()).
    std::
        unordered_map<topology_key_t, reader_types::track_info_ptr_t, topology_key_hash_t>
                                                                       topology_to_track;
    std::unordered_map<reader_types::track_info_ptr_t, topology_key_t> track_to_topology;
    std::unordered_map<reader_types::track_info_ptr_t, size_t>         track_to_db_id;

private:
    // Discovers "thread" tracks directly from the duration-event tables
    // (region/kernel_dispatch/memory_allocate/memory_copy, grouped by
    // (nid,pid,tid)) -- a track only exists if this returns a non-empty
    // group for it, matching optiq's own "no rocpd_track dependency"
    // discovery philosophy (see build_tracks()'s doc comment). Returns
    // per-key summed event counts; called from build_tracks().
    [[nodiscard]] std::unordered_map<topology_key_t, size_t, topology_key_hash_t>
    discover_thread_tracks(data_storage::schema_v3::read_statements& stmts);

    // Appends optiq-parity category tracks (kernel-dispatch/memory-allocate/
    // memory-copy, per agent+queue and per host-stream) to `tracks`,
    // continuing synthetic ids from `next_synthetic_id`. Called from
    // build_tracks().
    void add_category_tracks(data_storage::schema_v3::read_statements& stmts,
                             size_t&                                   next_synthetic_id);
};

}  // namespace profiler_hub
