// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "unified_memory_integration.hpp"
#include "unified_memory_aggregator.hpp"
#include "unified_memory_formatter.hpp"
#include "unified_memory_json.hpp"
#include "unified_memory_perfetto.hpp"
#include "logger/debug.hpp"
#include "core/config.hpp"

#include <sqlite3.h>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace rocprofsys
{
namespace rocpd
{

namespace fs = std::filesystem;

// ===================================================================
// Public Methods
// ===================================================================

void
unified_memory_integration::generate_outputs(const std::string& db_path,
                                               const std::string& output_dir, int pid)
{
    if(!is_enabled())
    {
        return;
    }

    LOG_INFO("Generating unified memory profiling outputs...");

    check_xnack_environment();

    try
    {
        if(!fs::exists(db_path))
        {
            LOG_WARNING("Database file not found: {}", db_path);
            return;
        }

        unified_memory_aggregator aggregator(db_path);
        if(!aggregator.has_unified_memory_data())
        {
            LOG_INFO(
                "No unified memory data found in database. Skipping unified memory report "
                "generation.");
            return;
        }

        std::string upid = get_upid_from_database(db_path);

        if(!fs::exists(output_dir))
        {
            fs::create_directories(output_dir);
        }

        std::string text_output_path = fs::path(output_dir) / "unified_memory.txt";
        generate_text_output(db_path, text_output_path, pid);

        std::string json_output_path = fs::path(output_dir) / "unified_memory.json";
        generate_json_output(db_path, json_output_path);

        sqlite3* db = nullptr;
        int      rc = sqlite3_open(db_path.c_str(), &db);
        if(rc == SQLITE_OK && db)
        {
            generate_perfetto_counters(db, upid);
            sqlite3_close(db);
        }
        else
        {
            LOG_ERROR("Failed to open database for Perfetto counter generation: {}",
                      db_path);
        }

        LOG_INFO("Unified memory profiling outputs generated successfully");
        LOG_INFO("  Text report: {}", text_output_path);
        LOG_INFO("  JSON report: {}", json_output_path);
    }
    catch(const std::exception& e)
    {
        LOG_ERROR("Failed to generate unified memory outputs: {}", e.what());
    }
}

void
unified_memory_integration::check_xnack_environment()
{
    const char* xnack = std::getenv("HSA_XNACK");

    if(!xnack || std::string(xnack) != "1")
    {
        LOG_WARNING(
            "HSA_XNACK is not set to 1. Unified memory profiling may show limited data.");
        LOG_WARNING(
            "For accurate page fault and migration tracking, set: export HSA_XNACK=1");
        LOG_WARNING(
            "XNACK support requires CDNA2+ GPUs (MI200 series or newer) and compatible ROCm "
            "version.");
    }
    else
    {
        LOG_INFO("HSA_XNACK=1 detected. Unified memory profiling is properly configured.");
    }
}

bool
unified_memory_integration::is_enabled()
{
    auto enabled = config::get_setting_value<bool>("ROCPROFSYS_USE_UNIFIED_MEMORY_PROFILING");
    return enabled.value_or(false);
}

// ===================================================================
// Private Helper Methods
// ===================================================================

void
unified_memory_integration::generate_text_output(const std::string& db_path,
                                                   const std::string& output_path, int pid)
{
    try
    {
        std::ofstream output_file(output_path);
        if(!output_file.is_open())
        {
            LOG_ERROR("Failed to open text output file: {}", output_path);
            return;
        }

        unified_memory_formatter::format_text_output_from_db(db_path, pid, output_file);
        output_file.close();

        LOG_DEBUG("Text output written to: {}", output_path);
    }
    catch(const std::exception& e)
    {
        LOG_ERROR("Failed to generate text output: {}", e.what());
    }
}

void
unified_memory_integration::generate_json_output(const std::string& db_path,
                                                   const std::string& output_path)
{
    try
    {
        bool success = unified_memory_json::write_from_db(db_path, output_path, true);
        if(!success)
        {
            LOG_ERROR("Failed to write JSON output to: {}", output_path);
        }
        else
        {
            LOG_DEBUG("JSON output written to: {}", output_path);
        }
    }
    catch(const std::exception& e)
    {
        LOG_ERROR("Failed to generate JSON output: {}", e.what());
    }
}

void
unified_memory_integration::generate_perfetto_counters(sqlite3*           db,
                                                         const std::string& upid)
{
    try
    {
        unified_memory_perfetto::emit_all_counters(db, upid);
        LOG_DEBUG("Perfetto counters emitted");
    }
    catch(const std::exception& e)
    {
        LOG_ERROR("Failed to generate Perfetto counters: {}", e.what());
    }
}

std::string
unified_memory_integration::get_upid_from_database(const std::string& db_path)
{
    std::string upid;

    sqlite3*      db   = nullptr;
    sqlite3_stmt* stmt = nullptr;

    int rc = sqlite3_open(db_path.c_str(), &db);
    if(rc != SQLITE_OK)
    {
        LOG_WARNING("Failed to open database to get UPID: {}", db_path);
        return "";
    }

    const char* query = "SELECT value FROM rocpd_metadata WHERE tag = 'uuid' LIMIT 1;";
    rc                = sqlite3_prepare_v2(db, query, -1, &stmt, nullptr);

    if(rc == SQLITE_OK)
    {
        if(sqlite3_step(stmt) == SQLITE_ROW)
        {
            const char* uuid = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
            if(uuid) upid = std::string(uuid);
        }
        sqlite3_finalize(stmt);
    }

    sqlite3_close(db);

    return upid;
}

}  // namespace rocpd
}  // namespace rocprofsys
