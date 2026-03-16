// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "unified_memory_aggregator.hpp"
#include "logger/debug.hpp"

#include <sqlite3.h>
#include <stdexcept>

namespace rocprofsys
{
namespace rocpd
{

// ===================================================================
// Constructor / Destructor
// ===================================================================

unified_memory_aggregator::unified_memory_aggregator(const std::string& db_path)
: m_db(nullptr)
, m_owns_db(true)
, m_upid("")
{
    int rc = sqlite3_open(db_path.c_str(), &m_db);
    if(rc != SQLITE_OK)
    {
        LOG_ERROR("Failed to open database: {}", db_path);
        throw std::runtime_error("Failed to open database for unified memory aggregation");
    }

    const char* upid_query = "SELECT value FROM rocpd_metadata WHERE tag = 'uuid' LIMIT 1;";
    sqlite3_stmt* stmt;
    rc = sqlite3_prepare_v2(m_db, upid_query, -1, &stmt, nullptr);
    if(rc == SQLITE_OK)
    {
        if(sqlite3_step(stmt) == SQLITE_ROW)
        {
            const char* upid = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
            if(upid) m_upid = std::string("_") + upid;
        }
        sqlite3_finalize(stmt);
    }
}

unified_memory_aggregator::unified_memory_aggregator(sqlite3* db, const std::string& upid)
: m_db(db)
, m_owns_db(false)
, m_upid(upid.empty() ? "" : ("_" + upid))
{
    if(!m_db)
    {
        throw std::runtime_error("Database handle cannot be null");
    }
}

unified_memory_aggregator::~unified_memory_aggregator()
{
    if(m_owns_db && m_db)
    {
        sqlite3_close(m_db);
    }
}

// ===================================================================
// Public Methods
// ===================================================================

unified_memory_data
unified_memory_aggregator::aggregate()
{
    unified_memory_data data;

    data.device_summaries      = get_migration_stats_by_device();
    data.fault_stats           = get_page_fault_stats();
    data.overall_summary       = get_overall_summary();
    data.fault_summary_by_type = get_page_fault_summary_by_type();

    return data;
}

std::vector<device_migration_summary>
unified_memory_aggregator::get_migration_stats_by_device()
{
    std::vector<device_migration_summary> results;
    std::map<uint64_t, device_migration_summary> device_map;

    std::string query = "SELECT "
                        "  direction, "
                        "  dst_agent_id, "
                        "  dst_agent_name, "
                        "  dst_agent_type, "
                        "  count, "
                        "  avg_size_bytes, "
                        "  min_size_bytes, "
                        "  max_size_bytes, "
                        "  total_size_bytes, "
                        "  total_time_ns, "
                        "  avg_time_ns, "
                        "  min_time_ns, "
                        "  max_time_ns, "
                        "  bandwidth_gbps "
                        "FROM " +
                        get_table_name("unified_memory_migration_summary") +
                        " ORDER BY dst_agent_id, direction;";

    execute_query(query, [&](sqlite3_stmt* stmt) {
        migration_stats stats;
        stats.direction = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        uint64_t agent_id =
            static_cast<uint64_t>(sqlite3_column_int64(stmt, 1));
        const char* agent_name =
            reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        const char* agent_type =
            reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));

        stats.count            = static_cast<uint64_t>(sqlite3_column_int64(stmt, 4));
        stats.avg_size_bytes   = sqlite3_column_double(stmt, 5);
        stats.min_size_bytes   = sqlite3_column_double(stmt, 6);
        stats.max_size_bytes   = sqlite3_column_double(stmt, 7);
        stats.total_size_bytes = sqlite3_column_double(stmt, 8);
        stats.total_time_ns    = static_cast<uint64_t>(sqlite3_column_int64(stmt, 9));
        stats.avg_time_ns      = sqlite3_column_double(stmt, 10);
        stats.min_time_ns      = sqlite3_column_double(stmt, 11);
        stats.max_time_ns      = sqlite3_column_double(stmt, 12);
        stats.bandwidth_gbps   = sqlite3_column_double(stmt, 13);

        if(device_map.find(agent_id) == device_map.end())
        {
            device_migration_summary summary;
            summary.agent_id   = agent_id;
            summary.agent_name = agent_name ? agent_name : "";
            summary.agent_type = agent_type ? agent_type : "";
            device_map[agent_id] = summary;
        }

        device_map[agent_id].migrations.push_back(stats);
    });

    for(auto& [agent_id, summary] : device_map)
    {
        results.push_back(std::move(summary));
    }

    return results;
}

std::vector<page_fault_stats>
unified_memory_aggregator::get_page_fault_stats()
{
    std::vector<page_fault_stats> results;

    std::string query = "SELECT "
                        "  agent_id, "
                        "  agent_name, "
                        "  agent_type, "
                        "  total_faults, "
                        "  read_faults, "
                        "  write_faults, "
                        "  faults_migrated, "
                        "  faults_updated, "
                        "  avg_fault_time_ns, "
                        "  min_fault_time_ns, "
                        "  max_fault_time_ns "
                        "FROM " +
                        get_table_name("unified_memory_page_faults") +
                        " ORDER BY agent_type, agent_name;";

    execute_query(query, [&](sqlite3_stmt* stmt) {
        page_fault_stats stats;
        stats.agent_id   = static_cast<uint64_t>(sqlite3_column_int64(stmt, 0));
        const char* name = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        const char* type = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        stats.agent_name = name ? name : "";
        stats.agent_type = type ? type : "";

        stats.total_faults      = static_cast<uint64_t>(sqlite3_column_int64(stmt, 3));
        stats.read_faults       = static_cast<uint64_t>(sqlite3_column_int64(stmt, 4));
        stats.write_faults      = static_cast<uint64_t>(sqlite3_column_int64(stmt, 5));
        stats.faults_migrated   = static_cast<uint64_t>(sqlite3_column_int64(stmt, 6));
        stats.faults_updated    = static_cast<uint64_t>(sqlite3_column_int64(stmt, 7));
        stats.avg_fault_time_ns = sqlite3_column_double(stmt, 8);
        stats.min_fault_time_ns = sqlite3_column_double(stmt, 9);
        stats.max_fault_time_ns = sqlite3_column_double(stmt, 10);

        results.push_back(stats);
    });

    return results;
}

std::optional<unified_memory_summary>
unified_memory_aggregator::get_overall_summary()
{
    std::optional<unified_memory_summary> result;

    std::string query = "SELECT "
                        "  host_to_device_count, "
                        "  device_to_host_count, "
                        "  device_to_device_count, "
                        "  host_to_device_bytes, "
                        "  device_to_host_bytes, "
                        "  device_to_device_bytes, "
                        "  total_migrations, "
                        "  total_bytes_migrated, "
                        "  total_migration_time_ns, "
                        "  overall_bandwidth_gbps "
                        "FROM " +
                        get_table_name("unified_memory_summary") + ";";

    execute_query(query, [&](sqlite3_stmt* stmt) {
        unified_memory_summary summary;
        summary.host_to_device_count =
            static_cast<uint64_t>(sqlite3_column_int64(stmt, 0));
        summary.device_to_host_count =
            static_cast<uint64_t>(sqlite3_column_int64(stmt, 1));
        summary.device_to_device_count =
            static_cast<uint64_t>(sqlite3_column_int64(stmt, 2));
        summary.host_to_device_bytes   = sqlite3_column_double(stmt, 3);
        summary.device_to_host_bytes   = sqlite3_column_double(stmt, 4);
        summary.device_to_device_bytes = sqlite3_column_double(stmt, 5);
        summary.total_migrations =
            static_cast<uint64_t>(sqlite3_column_int64(stmt, 6));
        summary.total_bytes_migrated    = sqlite3_column_double(stmt, 7);
        summary.total_migration_time_ns = static_cast<uint64_t>(sqlite3_column_int64(stmt, 8));
        summary.overall_bandwidth_gbps  = sqlite3_column_double(stmt, 9);

        result = summary;
    });

    return result;
}

std::vector<page_fault_summary_by_type>
unified_memory_aggregator::get_page_fault_summary_by_type()
{
    std::vector<page_fault_summary_by_type> results;

    std::string query = "SELECT "
                        "  agent_type, "
                        "  total_faults, "
                        "  total_read_faults, "
                        "  total_write_faults, "
                        "  total_faults_migrated, "
                        "  total_faults_updated, "
                        "  avg_fault_time_ns "
                        "FROM " +
                        get_table_name("unified_memory_page_faults_by_type") +
                        " ORDER BY agent_type;";

    execute_query(query, [&](sqlite3_stmt* stmt) {
        page_fault_summary_by_type summary;
        const char* type = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        summary.agent_type = type ? type : "";

        summary.total_faults =
            static_cast<uint64_t>(sqlite3_column_int64(stmt, 1));
        summary.total_read_faults =
            static_cast<uint64_t>(sqlite3_column_int64(stmt, 2));
        summary.total_write_faults =
            static_cast<uint64_t>(sqlite3_column_int64(stmt, 3));
        summary.total_faults_migrated =
            static_cast<uint64_t>(sqlite3_column_int64(stmt, 4));
        summary.total_faults_updated =
            static_cast<uint64_t>(sqlite3_column_int64(stmt, 5));
        summary.avg_fault_time_ns = sqlite3_column_double(stmt, 6);

        results.push_back(summary);
    });

    return results;
}

bool
unified_memory_aggregator::has_unified_memory_data() const
{
    bool has_data = false;

    std::string query =
        "SELECT COUNT(*) FROM " + get_table_name("rocpd_event") + " e "
        "JOIN " + get_table_name("rocpd_string") + " s ON s.id = e.category_id "
        "WHERE s.string IN ('kfd_page_migrate', 'kfd_page_fault');";

    const_cast<unified_memory_aggregator*>(this)->execute_query(
        query, [&](sqlite3_stmt* stmt) {
            int64_t count = sqlite3_column_int64(stmt, 0);
            has_data      = (count > 0);
        });

    return has_data;
}

// ===================================================================
// Private Methods
// ===================================================================

void
unified_memory_aggregator::execute_query(const std::string&                 query,
                                         std::function<void(sqlite3_stmt*)> callback)
{
    sqlite3_stmt* stmt = nullptr;
    int           rc   = sqlite3_prepare_v2(m_db, query.c_str(), -1, &stmt, nullptr);

    if(rc != SQLITE_OK)
    {
        LOG_ERROR("Failed to prepare SQL query: {}", query);
        LOG_ERROR("SQLite error: {}", sqlite3_errmsg(m_db));
        return;
    }

    while((rc = sqlite3_step(stmt)) == SQLITE_ROW)
    {
        callback(stmt);
    }

    if(rc != SQLITE_DONE)
    {
        LOG_ERROR("SQL query execution failed: {}", query);
        LOG_ERROR("SQLite error: {}", sqlite3_errmsg(m_db));
    }

    sqlite3_finalize(stmt);
}

std::string
unified_memory_aggregator::get_table_name(const std::string& base_name) const
{
    return base_name + m_upid;
}

}  // namespace rocpd
}  // namespace rocprofsys
