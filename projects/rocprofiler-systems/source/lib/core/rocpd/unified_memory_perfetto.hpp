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

#include "unified_memory_aggregator.hpp"
#include "core/perfetto.hpp"
#include "core/categories.hpp"

#include <cstdint>
#include <string>

namespace rocprofsys
{
namespace rocpd
{

/**
 * @brief Perfetto counter track integration for unified memory profiling
 *
 * This class creates and populates Perfetto counter tracks to visualize
 * unified memory statistics over time (bandwidth, page fault rates, etc.)
 */
class unified_memory_perfetto
{
public:
    /**
     * @brief Setup Perfetto counter tracks for unified memory
     * @param db Database handle
     * @param upid Unique process ID for table name suffixes
     */
    static void setup_tracks(sqlite3* db, const std::string& upid);

    /**
     * @brief Emit counter values for migration bandwidth timeline
     * @param db Database handle
     * @param upid Unique process ID
     */
    static void emit_migration_bandwidth_counters(sqlite3*           db,
                                                   const std::string& upid);

    /**
     * @brief Emit counter values for page fault rate timeline
     * @param db Database handle
     * @param upid Unique process ID
     */
    static void emit_page_fault_rate_counters(sqlite3* db, const std::string& upid);

    /**
     * @brief Emit all unified memory counter tracks
     * @param db Database handle
     * @param upid Unique process ID
     *
     * This is the main entry point that sets up tracks and emits all counters
     */
    static void emit_all_counters(sqlite3* db, const std::string& upid);

private:
    /**
     * @brief Create counter track for migration bandwidth (per agent)
     * @param agent_id Agent ID
     * @param agent_name Agent name for track label
     * @return Track index
     */
    static size_t create_bandwidth_track(uint64_t agent_id, const std::string& agent_name);

    /**
     * @brief Create counter track for page fault rate (per agent)
     * @param agent_id Agent ID
     * @param agent_name Agent name for track label
     * @return Track index
     */
    static size_t create_fault_rate_track(uint64_t agent_id, const std::string& agent_name);

    /**
     * @brief Execute SQL query and process results with callback
     * @param db Database handle
     * @param query SQL query string
     * @param callback Function to process each row
     */
    static void execute_query(sqlite3*                           db,
                              const std::string&                 query,
                              std::function<void(sqlite3_stmt*)> callback);

    /**
     * @brief Get table name with UPID suffix
     * @param base_name Base table/view name
     * @param upid Unique process ID
     * @return Fully qualified table name
     */
    static std::string get_table_name(const std::string& base_name,
                                      const std::string& upid);
};

}  // namespace rocpd
}  // namespace rocprofsys
