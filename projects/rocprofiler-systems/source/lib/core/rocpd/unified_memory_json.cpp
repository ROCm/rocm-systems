// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "unified_memory_json.hpp"
#include "unified_memory_aggregator.hpp"
#include "logger/debug.hpp"

#include <fstream>

namespace rocprofsys
{
namespace rocpd
{

// ===================================================================
// Public Static Methods - Conversion
// ===================================================================

nlohmann::json
unified_memory_json::to_json(const unified_memory_data& data)
{
    nlohmann::json j;

    // Use ADL (Argument-Dependent Lookup) serializers defined in header
    rocprofsys::rocpd::to_json(j, data);

    return j;
}

nlohmann::json
unified_memory_json::to_json(const migration_stats& stats)
{
    nlohmann::json j;
    rocprofsys::rocpd::to_json(j, stats);
    return j;
}

nlohmann::json
unified_memory_json::to_json(const device_migration_summary& device)
{
    nlohmann::json j;
    rocprofsys::rocpd::to_json(j, device);
    return j;
}

nlohmann::json
unified_memory_json::to_json(const page_fault_stats& stats)
{
    nlohmann::json j;
    rocprofsys::rocpd::to_json(j, stats);
    return j;
}

nlohmann::json
unified_memory_json::to_json(const unified_memory_summary& summary)
{
    nlohmann::json j;
    rocprofsys::rocpd::to_json(j, summary);
    return j;
}

nlohmann::json
unified_memory_json::to_json(const page_fault_summary_by_type& summary)
{
    nlohmann::json j;
    rocprofsys::rocpd::to_json(j, summary);
    return j;
}

// ===================================================================
// Public Static Methods - File I/O
// ===================================================================

bool
unified_memory_json::write_to_file(const unified_memory_data& data,
                                    const std::string&         filepath,
                                    bool                       pretty_print)
{
    try
    {
        std::ofstream file(filepath);
        if(!file.is_open())
        {
            LOG_ERROR("Failed to open file for writing: {}", filepath);
            return false;
        }

        nlohmann::json j = to_json(data);

        if(pretty_print)
        {
            file << j.dump(2);  // 2-space indentation
        }
        else
        {
            file << j.dump();
        }

        file.close();

        LOG_INFO("Unified memory JSON written to: {}", filepath);
        return true;
    }
    catch(const std::exception& e)
    {
        LOG_ERROR("Failed to write unified memory JSON to file: {}", e.what());
        return false;
    }
}

bool
unified_memory_json::write_from_db(const std::string& db_path,
                                    const std::string& filepath,
                                    bool               pretty_print)
{
    try
    {
        unified_memory_aggregator aggregator(db_path);

        if(!aggregator.has_unified_memory_data())
        {
            LOG_WARNING("No unified memory data found in database: {}", db_path);
            return false;
        }

        auto data = aggregator.aggregate();
        return write_to_file(data, filepath, pretty_print);
    }
    catch(const std::exception& e)
    {
        LOG_ERROR("Failed to generate JSON from database: {}", e.what());
        return false;
    }
}

std::string
unified_memory_json::to_string(const unified_memory_data& data, bool pretty_print)
{
    nlohmann::json j = to_json(data);

    if(pretty_print)
    {
        return j.dump(2);  // 2-space indentation
    }
    else
    {
        return j.dump();
    }
}

}  // namespace rocpd
}  // namespace rocprofsys
