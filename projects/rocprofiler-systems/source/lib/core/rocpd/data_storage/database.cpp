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
database&
database::get_instance()
{
    static database _instance;
    return _instance;
}

database::database()
{
    auto db_name     = std::string_view{ "rocpd.db" };
    auto abs_db_path = rocprofsys::get_database_absolute_path(db_name);
    create_directory_for_database_file(abs_db_path);
    ROCPROFSYS_VERBOSE(0, "Database: %s\r\n", abs_db_path.c_str());

    validate_sqlite3_result(sqlite3_open(":memory:", &_sqlite3_db_temp), "",
                            "database open failed!");
    validate_sqlite3_result(sqlite3_open(abs_db_path.c_str(), &_sqlite3_db), "",
                            "database open failed!");
}

database::~database()
{
    sqlite3_close(_sqlite3_db_temp);
    sqlite3_close(_sqlite3_db);
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
        return std::string(
                   "rocprofiler-systems/source/lib/core/rocpd/data_storage/schema/")
            .append(filename);
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
        std::regex view_upid_pattern("\\{\\{view_upid\\}\\}");

        query = std::regex_replace(query, upid_pattern, "_" + get_upid());
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
    static std::string _upid = []() {
        auto n_info = node_info::get_instance();
        auto guid   = common::md5sum{ n_info.id, getpid(), getppid() };
        return guid.hexdigest();
    }();
    return _upid;
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
}

}  // namespace data_storage
}  // namespace rocpd
}  // namespace rocprofsys
