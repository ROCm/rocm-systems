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

#include "database.hpp"
#include "common/md5sum.hpp"
#include "debug.hpp"
#include "node_info.hpp"

#include <config.hpp>
#include <fstream>
#include <regex>
#include <string>
#include <timemory/environment/types.hpp>
#include <timemory/utility/filepath.hpp>
#include <unistd.h>

namespace
{
void
create_directory_for_database_file(const std::string& db_file)
{
    auto _db_dirname = tim::filepath::dirname(db_file);
    if(!tim::filepath::direxists(_db_dirname))
    {
        tim::filepath::makedir(_db_dirname);
    }
}
}  // namespace
namespace rocprofsys
{
namespace rocpd
{
namespace data_storage
{

static int
enhanced_trace_callback_v2(unsigned int trace_type, void* user_data, void* p, void* x)
{
    auto* tracker = static_cast<database_performance_tracker*>(user_data);
    if(!tracker) return 0;

    switch(trace_type)
    {
        case SQLITE_TRACE_STMT:
        {
            sqlite3_stmt* stmt = static_cast<sqlite3_stmt*>(p);
            char*         sql  = (char*) sqlite3_sql(stmt);
            if(sql && (memcmp(sql, "INSERT INTO ", sizeof("INSERT INTO ") - 1) == 0))
            {
                tracker->query_count++;
                std::string name_tag(sql + sizeof("INSERT INTO ") - 1, 20);
                tracker->queries[name_tag].count++;
            }
            break;
        }
        case SQLITE_TRACE_PROFILE:
        {
            sqlite3_stmt*   stmt    = static_cast<sqlite3_stmt*>(p);
            sqlite3_uint64* time_ns = static_cast<sqlite3_uint64*>(x);

            tracker->total_execution_time += std::chrono::nanoseconds(*time_ns);

            char* sql = (char*) sqlite3_sql(stmt);
            if(sql && (memcmp(sql, "INSERT INTO ", sizeof("INSERT INTO ") - 1) == 0))
            {
                std::string name_tag(sql + sizeof("INSERT INTO ") - 1, 20);
                tracker->queries[name_tag].time += std::chrono::nanoseconds(*time_ns);
            }
            break;
        }
        case SQLITE_TRACE_ROW:
        {
            tracker->row_count++;
            break;
        }
        case SQLITE_TRACE_CLOSE:
        {
            ROCPROFSYS_VERBOSE(1, "SQLite Database connection closing\n");
            tracker->print_summary();
            break;
        }
        default: break;
    }
    return 0;
}

database::database(int pid, int ppid)
{
    auto _tag        = std::to_string(pid);
    auto db_name     = std::string{ "rocpd" };
    auto abs_db_path = rocprofsys::get_database_absolute_path(db_name, _tag);
    create_directory_for_database_file(abs_db_path);
    ROCPROFSYS_VERBOSE(0, "Database: %s\r\n", abs_db_path.c_str());

    auto db_memory_name = std::string("file:").append(_tag).append("?mode=memory");
    std::cout << "Memory db name " << db_memory_name << "\n";
    validate_sqlite3_result(sqlite3_open(db_memory_name.c_str(), &_sqlite3_db_temp), "",
                            "database open failed!");
    validate_sqlite3_result(sqlite3_open(abs_db_path.c_str(), &_sqlite3_db), "",
                            "database open failed!");

    // Initialize performance tracker
    m_perf_tracker = std::make_unique<database_performance_tracker>();
    m_perf_tracker->reset();

    // Setup comprehensive tracing with sqlite3_trace_v2
    unsigned int trace_mask =
        SQLITE_TRACE_STMT | SQLITE_TRACE_PROFILE | SQLITE_TRACE_ROW | SQLITE_TRACE_CLOSE;

    sqlite3_trace_v2(_sqlite3_db_temp, trace_mask, enhanced_trace_callback_v2,
                     m_perf_tracker.get());

    m_upid = generate_upid(pid, ppid);
}

database::~database()
{
    sqlite3_close(_sqlite3_db_temp);
    sqlite3_close(_sqlite3_db);

    std::cout << "Bind duration: " << (float) bind_duration / 1000 / 1000
              << "ms, Step duration: " << (float) step_duration / 1000 / 1000
              << "ms, Reset duration: " << (float) reset_duration / 1000 / 1000 << "ms\n";
}

void
database::initialize_schema()
{
    auto get_file_path = [](const std::string_view filename) {
        auto _rocprofsys_root = tim::get_env<std::string>(
            "rocprofiler_systems_ROOT", tim::get_env<std::string>("ROCPROFSYS_ROOT", ""));
        if(!_rocprofsys_root.empty() &&
           tim::filepath::direxists(std::string(_rocprofsys_root)))
        {
            auto new_file_path = std::string(_rocprofsys_root)
                                     .append("/share/rocprofiler-systems/")
                                     .append(filename);
            if(tim::filepath::exists(new_file_path))
            {
                return new_file_path;
            }
        }
        // TODO:  Update to look for the system's rocpd schema
        return std::string("source/lib/core/rocpd/data_storage/schema/").append(filename);
    };

    std::vector<std::string_view> schema_files = { "rocpd_tables.sql", "rocpd_views.sql",
                                                   "data_views.sql", "marker_views.sql",
                                                   "summary_views.sql" };

    // Process each schema file
    for(const auto& schema_file : schema_files)
    {
        auto          file_path = get_file_path(schema_file);
        std::ifstream file(file_path);
        if(!file.is_open())
        {
            throw std::runtime_error(
                std::string("Failed to open schema file ").append(file_path));
        }

        std::stringstream ss_query;
        ss_query << file.rdbuf();
        std::string query = ss_query.str();

        std::regex upid_pattern("\\{\\{uuid\\}\\}");
        std::regex guid_pattern("\\{\\{guid\\}\\}");
        std::regex view_upid_pattern("\\{\\{view_upid\\}\\}");

        auto upid = get_upid();

        query = std::regex_replace(query, upid_pattern, "_" + upid);
        query = std::regex_replace(query, guid_pattern, upid);
        query = std::regex_replace(query, view_upid_pattern, "");

        validate_sqlite3_result(
            sqlite3_exec(_sqlite3_db_temp, query.c_str(), 0, 0, 0), query.c_str(),
            std::string("Invalid schema file, init database failed!").append(file_path));
        file.close();
    }
}

void
database::execute_query(const std::string& query)
{
    validate_sqlite3_result(sqlite3_exec(_sqlite3_db_temp, query.c_str(), 0, 0, 0),
                            "Failed to execute query - ", query);
}

std::string
database::get_upid()
{
    return m_upid;
}

std::string
database::generate_upid(const int pid, const int ppid)
{
    auto n_info = node_info::get_instance();
    auto guid   = common::md5sum{ n_info.id, pid, ppid };
    return guid.hexdigest();
}

size_t
database::get_last_insert_id() const
{
    return sqlite3_last_insert_rowid(_sqlite3_db_temp);
}

void
database::flush()
{
    auto* backup = sqlite3_backup_init(_sqlite3_db, "main", _sqlite3_db_temp, "main");
    if(backup)
    {
        sqlite3_backup_step(backup, -1);  // Copy all pages
        sqlite3_backup_finish(backup);
    }
    // m_perf_tracker->print_summary();
}

}  // namespace data_storage
}  // namespace rocpd
}  // namespace rocprofsys
