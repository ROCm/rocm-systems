// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

// Forward declarations
struct sqlite3;
struct sqlite3_stmt;

namespace rocprofsys
{
namespace rocpd
{
namespace data_storage
{
class database;
}

// ===================================================================
// Data Structures for Unified Memory Statistics
// ===================================================================

/**
 * @brief Migration statistics for a specific direction (Host→Device, Device→Host, etc.)
 */
struct migration_stats
{
    std::string direction;           // "Host To Device", "Device To Host", "Device To Device"
    uint64_t    count;               // Number of migration events
    double      avg_size_bytes;      // Average migration size in bytes
    double      min_size_bytes;      // Minimum migration size in bytes
    double      max_size_bytes;      // Maximum migration size in bytes
    double      total_size_bytes;    // Total bytes migrated
    uint64_t    total_time_ns;       // Total time spent in nanoseconds
    double      avg_time_ns;         // Average time per migration
    double      min_time_ns;         // Minimum migration time
    double      max_time_ns;         // Maximum migration time
    double      bandwidth_gbps;      // Bandwidth in GB/s
};

/**
 * @brief Per-device migration summary
 */
struct device_migration_summary
{
    uint64_t    agent_id;            // Agent ID
    std::string agent_name;          // Agent name (e.g., "gfx90a")
    std::string agent_type;          // "CPU" or "GPU"

    std::vector<migration_stats> migrations;  // Stats per direction
};

/**
 * @brief Page fault statistics for a specific agent
 */
struct page_fault_stats
{
    uint64_t    agent_id;            // Agent ID
    std::string agent_name;          // Agent name
    std::string agent_type;          // "CPU" or "GPU"

    uint64_t total_faults;           // Total page faults
    uint64_t read_faults;            // Read page faults
    uint64_t write_faults;           // Write page faults
    uint64_t faults_migrated;        // Faults resolved via migration
    uint64_t faults_updated;         // Faults resolved via update

    double   avg_fault_time_ns;      // Average fault handling time
    double   min_fault_time_ns;      // Minimum fault time
    double   max_fault_time_ns;      // Maximum fault time
};

/**
 * @brief Overall unified memory summary statistics
 */
struct unified_memory_summary
{
    // Migration counts by direction
    uint64_t host_to_device_count;
    uint64_t device_to_host_count;
    uint64_t device_to_device_count;

    // Migration bytes by direction
    double   host_to_device_bytes;
    double   device_to_host_bytes;
    double   device_to_device_bytes;

    // Overall totals
    uint64_t total_migrations;
    double   total_bytes_migrated;
    uint64_t total_migration_time_ns;
    double   overall_bandwidth_gbps;
};

/**
 * @brief Page fault summary by agent type (CPU vs GPU)
 */
struct page_fault_summary_by_type
{
    std::string agent_type;          // "CPU" or "GPU"
    uint64_t    total_faults;
    uint64_t    total_read_faults;
    uint64_t    total_write_faults;
    uint64_t    total_faults_migrated;
    uint64_t    total_faults_updated;
    double      avg_fault_time_ns;
};

/**
 * @brief Complete unified memory profiling data
 */
struct unified_memory_data
{
    std::vector<device_migration_summary>    device_summaries;
    std::vector<page_fault_stats>            fault_stats;
    std::optional<unified_memory_summary>    overall_summary;
    std::vector<page_fault_summary_by_type>  fault_summary_by_type;

    bool has_data() const
    {
        return !device_summaries.empty() || !fault_stats.empty();
    }
};

// ===================================================================
// Unified Memory Aggregator Class
// ===================================================================

/**
 * @brief Aggregates unified memory profiling data from SQL views
 *
 * This class queries the unified_memory_* SQL views created in Phase 2
 * and structures the data into C++ objects for consumption by output formatters.
 */
class unified_memory_aggregator
{
public:
    /**
     * @brief Construct aggregator with database connection
     * @param db_path Path to the SQLite database file
     */
    explicit unified_memory_aggregator(const std::string& db_path);

    /**
     * @brief Construct aggregator with existing database handle
     * @param db SQLite database handle
     * @param upid Unique process ID for table name suffixes
     */
    unified_memory_aggregator(sqlite3* db, const std::string& upid);

    ~unified_memory_aggregator();

    // Non-copyable, non-movable
    unified_memory_aggregator(const unified_memory_aggregator&)            = delete;
    unified_memory_aggregator& operator=(const unified_memory_aggregator&) = delete;
    unified_memory_aggregator(unified_memory_aggregator&&)                 = delete;
    unified_memory_aggregator& operator=(unified_memory_aggregator&&)      = delete;

    /**
     * @brief Aggregate all unified memory data from database
     * @return Complete unified memory profiling data
     */
    unified_memory_data aggregate();

    /**
     * @brief Get migration statistics by device
     * @return Vector of per-device migration summaries
     */
    std::vector<device_migration_summary> get_migration_stats_by_device();

    /**
     * @brief Get page fault statistics by agent
     * @return Vector of per-agent page fault statistics
     */
    std::vector<page_fault_stats> get_page_fault_stats();

    /**
     * @brief Get overall unified memory summary
     * @return Overall statistics across all devices
     */
    std::optional<unified_memory_summary> get_overall_summary();

    /**
     * @brief Get page fault summary by agent type (CPU vs GPU)
     * @return Vector of fault summaries grouped by type
     */
    std::vector<page_fault_summary_by_type> get_page_fault_summary_by_type();

    /**
     * @brief Check if database contains any unified memory data
     * @return true if KFD events (page migrations or faults) exist
     */
    bool has_unified_memory_data() const;

private:
    /**
     * @brief Execute SQL query and return results
     * @param query SQL query string
     * @param callback Function to process each row
     */
    void execute_query(const std::string& query,
                       std::function<void(sqlite3_stmt*)> callback);

    /**
     * @brief Get table name with UPID suffix
     * @param base_name Base table/view name
     * @return Fully qualified table name
     */
    std::string get_table_name(const std::string& base_name) const;

private:
    sqlite3*    m_db;           // SQLite database handle
    bool        m_owns_db;      // Whether we opened the DB ourselves
    std::string m_upid;         // Unique process ID for table suffixes
};

}  // namespace rocpd
}  // namespace rocprofsys
