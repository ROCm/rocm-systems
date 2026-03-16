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

#include <nlohmann/json.hpp>
#include <string>

namespace rocprofsys
{
namespace rocpd
{

/**
 * @brief JSON serialization for unified memory profiling data
 *
 * Provides conversion functions between unified memory data structures
 * and JSON format for easy consumption by analysis tools.
 */
class unified_memory_json
{
public:
    /**
     * @brief Convert unified memory data to JSON
     * @param data Unified memory profiling data
     * @return JSON object
     */
    static nlohmann::json to_json(const unified_memory_data& data);

    /**
     * @brief Convert migration statistics to JSON
     * @param stats Migration statistics
     * @return JSON object
     */
    static nlohmann::json to_json(const migration_stats& stats);

    /**
     * @brief Convert device migration summary to JSON
     * @param device Device migration summary
     * @return JSON object
     */
    static nlohmann::json to_json(const device_migration_summary& device);

    /**
     * @brief Convert page fault stats to JSON
     * @param stats Page fault statistics
     * @return JSON object
     */
    static nlohmann::json to_json(const page_fault_stats& stats);

    /**
     * @brief Convert unified memory summary to JSON
     * @param summary Overall summary statistics
     * @return JSON object
     */
    static nlohmann::json to_json(const unified_memory_summary& summary);

    /**
     * @brief Convert page fault summary by type to JSON
     * @param summary Fault summary by type
     * @return JSON object
     */
    static nlohmann::json to_json(const page_fault_summary_by_type& summary);

    /**
     * @brief Write JSON to file
     * @param data Unified memory data
     * @param filepath Output file path
     * @param pretty_print Whether to format JSON with indentation
     * @return true if successful, false otherwise
     */
    static bool write_to_file(const unified_memory_data& data,
                              const std::string&         filepath,
                              bool                       pretty_print = true);

    /**
     * @brief Generate JSON directly from database
     * @param db_path Path to database file
     * @param filepath Output JSON file path
     * @param pretty_print Whether to format JSON with indentation
     * @return true if successful, false otherwise
     */
    static bool write_from_db(const std::string& db_path,
                              const std::string& filepath,
                              bool               pretty_print = true);

    /**
     * @brief Get JSON string representation
     * @param data Unified memory data
     * @param pretty_print Whether to format JSON with indentation
     * @return JSON string
     */
    static std::string to_string(const unified_memory_data& data,
                                 bool                       pretty_print = true);
};

}  // namespace rocpd
}  // namespace rocprofsys

// ===================================================================
// nlohmann::json ADL (Argument-Dependent Lookup) Serializers
// ===================================================================
// These allow direct use of nlohmann::json conversion functions

namespace rocprofsys
{
namespace rocpd
{

inline void
to_json(nlohmann::json& j, const migration_stats& stats)
{
    j = nlohmann::json{
        {"direction", stats.direction},
        {"count", stats.count},
        {"avg_size_bytes", stats.avg_size_bytes},
        {"min_size_bytes", stats.min_size_bytes},
        {"max_size_bytes", stats.max_size_bytes},
        {"total_size_bytes", stats.total_size_bytes},
        {"total_time_ns", stats.total_time_ns},
        {"avg_time_ns", stats.avg_time_ns},
        {"min_time_ns", stats.min_time_ns},
        {"max_time_ns", stats.max_time_ns},
        {"bandwidth_gbps", stats.bandwidth_gbps}
    };
}

inline void
to_json(nlohmann::json& j, const device_migration_summary& device)
{
    j = nlohmann::json{
        {"agent_id", device.agent_id},
        {"agent_name", device.agent_name},
        {"agent_type", device.agent_type},
        {"migrations", device.migrations}
    };
}

inline void
to_json(nlohmann::json& j, const page_fault_stats& stats)
{
    j = nlohmann::json{
        {"agent_id", stats.agent_id},
        {"agent_name", stats.agent_name},
        {"agent_type", stats.agent_type},
        {"total_faults", stats.total_faults},
        {"read_faults", stats.read_faults},
        {"write_faults", stats.write_faults},
        {"faults_migrated", stats.faults_migrated},
        {"faults_updated", stats.faults_updated},
        {"avg_fault_time_ns", stats.avg_fault_time_ns},
        {"min_fault_time_ns", stats.min_fault_time_ns},
        {"max_fault_time_ns", stats.max_fault_time_ns}
    };
}

inline void
to_json(nlohmann::json& j, const unified_memory_summary& summary)
{
    j = nlohmann::json{
        {"host_to_device_count", summary.host_to_device_count},
        {"device_to_host_count", summary.device_to_host_count},
        {"device_to_device_count", summary.device_to_device_count},
        {"host_to_device_bytes", summary.host_to_device_bytes},
        {"device_to_host_bytes", summary.device_to_host_bytes},
        {"device_to_device_bytes", summary.device_to_device_bytes},
        {"total_migrations", summary.total_migrations},
        {"total_bytes_migrated", summary.total_bytes_migrated},
        {"total_migration_time_ns", summary.total_migration_time_ns},
        {"overall_bandwidth_gbps", summary.overall_bandwidth_gbps}
    };
}

inline void
to_json(nlohmann::json& j, const page_fault_summary_by_type& summary)
{
    j = nlohmann::json{
        {"agent_type", summary.agent_type},
        {"total_faults", summary.total_faults},
        {"total_read_faults", summary.total_read_faults},
        {"total_write_faults", summary.total_write_faults},
        {"total_faults_migrated", summary.total_faults_migrated},
        {"total_faults_updated", summary.total_faults_updated},
        {"avg_fault_time_ns", summary.avg_fault_time_ns}
    };
}

inline void
to_json(nlohmann::json& j, const unified_memory_data& data)
{
    j = nlohmann::json{
        {"device_summaries", data.device_summaries},
        {"page_fault_stats", data.fault_stats},
        {"fault_summary_by_type", data.fault_summary_by_type}
    };

    if(data.overall_summary)
    {
        j["overall_summary"] = *data.overall_summary;
    }
}

}  // namespace rocpd
}  // namespace rocprofsys
