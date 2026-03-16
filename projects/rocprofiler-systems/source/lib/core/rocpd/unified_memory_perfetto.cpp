// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "unified_memory_perfetto.hpp"
#include "logger/debug.hpp"

#include <spdlog/fmt/fmt.h>
#include <sqlite3.h>
#include <map>

#if defined(TIMEMORY_USE_PERFETTO)
#    include <timemory/components/perfetto/backends.hpp>
#else
#    include <perfetto.h>
#endif

namespace rocprofsys
{
namespace rocpd
{

// Define counter track types for unified memory metrics
using um_bandwidth_track = perfetto_counter_track<category::kfd_page_migrate>;
using um_fault_rate_track = perfetto_counter_track<category::kfd_page_fault>;

// ===================================================================
// Public Methods
// ===================================================================

void
unified_memory_perfetto::setup_tracks(sqlite3* db, const std::string& upid)
{
    if(!db) return;

    // Query unique agents that have migration data
    std::string query =
        "SELECT DISTINCT dst_agent_id, dst_agent_name "
        "FROM " +
        get_table_name("unified_memory_migration_summary", upid) + ";";

    std::map<uint64_t, std::string> agents;

    execute_query(db, query, [&](sqlite3_stmt* stmt) {
        uint64_t    agent_id = static_cast<uint64_t>(sqlite3_column_int64(stmt, 0));
        const char* name     = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        if(name) agents[agent_id] = std::string(name);
    });

    // Create bandwidth tracks for each agent
    for(const auto& [agent_id, agent_name] : agents)
    {
        create_bandwidth_track(agent_id, agent_name);
    }

    // Query unique agents that have page fault data
    query = "SELECT DISTINCT agent_id, agent_name "
            "FROM " +
            get_table_name("unified_memory_page_faults", upid) + ";";

    agents.clear();

    execute_query(db, query, [&](sqlite3_stmt* stmt) {
        uint64_t    agent_id = static_cast<uint64_t>(sqlite3_column_int64(stmt, 0));
        const char* name     = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        if(name) agents[agent_id] = std::string(name);
    });

    // Create fault rate tracks for each agent
    for(const auto& [agent_id, agent_name] : agents)
    {
        create_fault_rate_track(agent_id, agent_name);
    }
}

void
unified_memory_perfetto::emit_migration_bandwidth_counters(sqlite3*           db,
                                                            const std::string& upid)
{
    if(!db) return;

    // Query migration timeline for bandwidth data
    std::string query =
        "SELECT "
        "  timestamp_ns, "
        "  bandwidth_gbps, "
        "  dst_agent_id, "
        "  dst_agent_name "
        "FROM " +
        get_table_name("unified_memory_migration_timeline", upid) +
        " ORDER BY timestamp_ns;";

    execute_query(db, query, [&](sqlite3_stmt* stmt) {
        uint64_t timestamp_ns = static_cast<uint64_t>(sqlite3_column_int64(stmt, 0));
        double   bandwidth    = sqlite3_column_double(stmt, 1);
        uint64_t agent_id     = static_cast<uint64_t>(sqlite3_column_int64(stmt, 2));

        // Emit counter value to Perfetto
        if(um_bandwidth_track::exists(agent_id, 0))
        {
            TRACE_COUNTER("unified_memory_bandwidth", um_bandwidth_track::at(agent_id, 0),
                          timestamp_ns, bandwidth);
        }
    });
}

void
unified_memory_perfetto::emit_page_fault_rate_counters(sqlite3*           db,
                                                        const std::string& upid)
{
    if(!db) return;

    // Query page fault timeline
    std::string query =
        "SELECT "
        "  timestamp_ns, "
        "  agent_id, "
        "  agent_name "
        "FROM " +
        get_table_name("unified_memory_page_fault_timeline", upid) +
        " ORDER BY timestamp_ns;";

    // Calculate fault rate using a sliding window
    std::map<uint64_t, std::vector<uint64_t>> agent_fault_times;
    const uint64_t window_ns = 1000000000;  // 1 second window

    execute_query(db, query, [&](sqlite3_stmt* stmt) {
        uint64_t timestamp_ns = static_cast<uint64_t>(sqlite3_column_int64(stmt, 0));
        uint64_t agent_id     = static_cast<uint64_t>(sqlite3_column_int64(stmt, 1));

        // Store fault timestamp
        agent_fault_times[agent_id].push_back(timestamp_ns);

        // Calculate rate: count faults in last 1 second window
        auto&    fault_times = agent_fault_times[agent_id];
        uint64_t window_start = (timestamp_ns > window_ns) ? (timestamp_ns - window_ns) : 0;

        // Remove old faults outside the window
        fault_times.erase(
            std::remove_if(fault_times.begin(), fault_times.end(),
                           [window_start](uint64_t t) { return t < window_start; }),
            fault_times.end());

        // Calculate rate (faults per second)
        double fault_rate = static_cast<double>(fault_times.size());

        // Emit counter value
        if(um_fault_rate_track::exists(agent_id, 0))
        {
            TRACE_COUNTER("unified_memory_fault_rate", um_fault_rate_track::at(agent_id, 0),
                          timestamp_ns, fault_rate);
        }
    });
}

void
unified_memory_perfetto::emit_all_counters(sqlite3* db, const std::string& upid)
{
    if(!db) return;

    // Setup tracks first
    setup_tracks(db, upid);

    // Emit all counter values
    emit_migration_bandwidth_counters(db, upid);
    emit_page_fault_rate_counters(db, upid);

    LOG_INFO("Unified memory Perfetto counters emitted");
}

// ===================================================================
// Private Helper Methods
// ===================================================================

size_t
unified_memory_perfetto::create_bandwidth_track(uint64_t           agent_id,
                                                 const std::string& agent_name)
{
    // Check if track already exists
    if(um_bandwidth_track::exists(agent_id, 0))
    {
        return 0;
    }

    // Create track with descriptive name
    std::string track_name =
        fmt::format("Unified Memory Bandwidth [{}]", agent_name);

    return um_bandwidth_track::emplace(
        agent_id,       // Index (agent ID)
        track_name,     // Track name
        "GB/s",         // Units
        nullptr,        // Category (use default)
        1,              // Unit multiplier
        false           // Not incremental
    );
}

size_t
unified_memory_perfetto::create_fault_rate_track(uint64_t           agent_id,
                                                  const std::string& agent_name)
{
    // Check if track already exists
    if(um_fault_rate_track::exists(agent_id, 0))
    {
        return 0;
    }

    // Create track with descriptive name
    std::string track_name =
        fmt::format("Unified Memory Page Fault Rate [{}]", agent_name);

    return um_fault_rate_track::emplace(
        agent_id,       // Index (agent ID)
        track_name,     // Track name
        "faults/sec",   // Units
        nullptr,        // Category (use default)
        1,              // Unit multiplier
        false           // Not incremental
    );
}

void
unified_memory_perfetto::execute_query(sqlite3*                           db,
                                        const std::string&                 query,
                                        std::function<void(sqlite3_stmt*)> callback)
{
    sqlite3_stmt* stmt = nullptr;
    int           rc   = sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, nullptr);

    if(rc != SQLITE_OK)
    {
        LOG_ERROR("Failed to prepare SQL query: {}", query);
        LOG_ERROR("SQLite error: {}", sqlite3_errmsg(db));
        return;
    }

    while((rc = sqlite3_step(stmt)) == SQLITE_ROW)
    {
        callback(stmt);
    }

    if(rc != SQLITE_DONE)
    {
        LOG_ERROR("SQL query execution failed: {}", query);
        LOG_ERROR("SQLite error: {}", sqlite3_errmsg(db));
    }

    sqlite3_finalize(stmt);
}

std::string
unified_memory_perfetto::get_table_name(const std::string& base_name,
                                         const std::string& upid)
{
    if(upid.empty())
    {
        return base_name;
    }
    else if(upid[0] == '_')
    {
        return base_name + upid;
    }
    else
    {
        return base_name + "_" + upid;
    }
}

}  // namespace rocpd
}  // namespace rocprofsys
