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
#include "common/directory.hpp"

#include <iostream>
#include <regex>
#include <string>

#if defined(USE_SCHEMA_FROM_ROCPROFILER_SDK_ROCPD) &&                                    \
    USE_SCHEMA_FROM_ROCPROFILER_SDK_ROCPD > 0
#    include <rocprofiler-sdk-rocpd/rocpd.h>
#    include <rocprofiler-sdk-rocpd/types.h>
#else
#    include "schema/data_views.hpp"
#    include "schema/marker_views.hpp"
#    include "schema/rocpd_tables.hpp"
#    include "schema/rocpd_views.hpp"
#    include "schema/summary_views.hpp"

namespace
{
enum rocpd_sql_schema_kind_t
{
    ROCPD_SQL_SCHEMA_NONE = 0,
    ROCPD_SQL_SCHEMA_ROCPD_TABLES,
    ROCPD_SQL_SCHEMA_ROCPD_INDEXES,
    ROCPD_SQL_SCHEMA_ROCPD_VIEWS,
    ROCPD_SQL_SCHEMA_ROCPD_DATA_VIEWS,
    ROCPD_SQL_SCHEMA_ROCPD_SUMMARY_VIEWS,
    ROCPD_SQL_SCHEMA_ROCPD_MARKER_VIEWS,
    ROCPD_SQL_SCHEMA_LAST,
};
}  // namespace

#endif

namespace
{
void
create_directory_for_database_file(const std::string& db_file)
{
    auto _db_dirname = rocstorage::common::dirname(db_file);
    if(!rocstorage::common::direxists(_db_dirname))
    {
        rocstorage::common::makedir(_db_dirname);
    }
}

std::string
process_schema_template(std::string_view schema_content, const std::string& upid)
{
    std::string query = std::string(schema_content);

    std::regex upid_pattern("\\{\\{uuid\\}\\}");
    std::regex guid_pattern("\\{\\{guid\\}\\}");
    std::regex view_upid_pattern("\\{\\{view_upid\\}\\}");

    query = std::regex_replace(query, upid_pattern, "_" + upid);
    query = std::regex_replace(query, guid_pattern, upid);
    query = std::regex_replace(query, view_upid_pattern, "");

    return query;
}

#if defined(USE_SCHEMA_FROM_ROCPROFILER_SDK_ROCPD) &&                                    \
    USE_SCHEMA_FROM_ROCPROFILER_SDK_ROCPD > 0
void
load_schema_cb(rocpd_sql_engine_t, rocpd_sql_schema_kind_t, rocpd_sql_options_t,
               const rocpd_sql_schema_jinja_variables_t*, const char*,
               const char* schema_content, void* user_data)
{
    if(user_data == nullptr || schema_content == nullptr)
    {
        std::cerr << "Invalid user data or schema content pointer" << std::endl;
        return;
    }
    auto* query = static_cast<std::string*>(user_data);
    if(query == nullptr)
    {
        std::cerr << "Invalid query pointer" << std::endl;
        return;
    }
    *query = std::string(schema_content);
}
#endif

std::string
get_schema_query(rocpd_sql_schema_kind_t schema_kind, const std::string& uuid)
{
#if defined(USE_SCHEMA_FROM_ROCPROFILER_SDK_ROCPD) &&                                    \
    USE_SCHEMA_FROM_ROCPROFILER_SDK_ROCPD > 0
    const auto                         jinja_size = 2 * uuid.size();
    rocpd_sql_schema_jinja_variables_t info{ jinja_size, uuid.c_str(), uuid.c_str() };

    std::string query;
    auto        status = rocpd_sql_load_schema(ROCPD_SQL_ENGINE_SQLITE3, schema_kind,
                                               ROCPD_SQL_OPTIONS_NONE, &info, load_schema_cb,
                                               nullptr, 0, &query);
    if(status != ROCPD_STATUS_SUCCESS)
    {
        std::cerr << "Unable to load rocpd schema. Error code: " << status << std::endl;
    }
    return query;
#else
    std::string_view schema_content;

    switch(schema_kind)
    {
        case ROCPD_SQL_SCHEMA_ROCPD_TABLES:
            schema_content = rocpd::data_storage::schema::ROCPD_TABLES_SQL;
            break;
        case ROCPD_SQL_SCHEMA_ROCPD_VIEWS:
            schema_content = rocpd::data_storage::schema::ROCPD_VIEWS_SQL;
            break;
        case ROCPD_SQL_SCHEMA_ROCPD_DATA_VIEWS:
            schema_content = rocpd::data_storage::schema::DATA_VIEWS_SQL;
            break;
        case ROCPD_SQL_SCHEMA_ROCPD_MARKER_VIEWS:
            schema_content = rocpd::data_storage::schema::MARKER_VIEWS_SQL;
            break;
        case ROCPD_SQL_SCHEMA_ROCPD_SUMMARY_VIEWS:
            schema_content = rocpd::data_storage::schema::SUMMARY_VIEWS_SQL;
            break;
        default:
            throw std::runtime_error("Unknown schema kind: " +
                                     std::to_string(schema_kind));
    }

    return process_schema_template(schema_content, uuid);
#endif
}

}  // namespace

namespace rocstorage
{
namespace data_storage
{
database::database(const std::string& abs_db_path, const std::string& uuid)
{
    create_directory_for_database_file(abs_db_path);
    std::cout << "Database: " << abs_db_path << std::endl;

    validate_sqlite3_result(sqlite3_open(":memory:", &_sqlite3_db_temp), "",
                            "database open failed!");
    validate_sqlite3_result(sqlite3_open(abs_db_path.c_str(), &_sqlite3_db), "",
                            "database open failed!");
    m_uuid = uuid;
}

database::~database()
{
    sqlite3_close(_sqlite3_db_temp);
    sqlite3_close(_sqlite3_db);
}

void
database::initialize_schema()
{
    const auto uuid = get_uuid();

    const std::vector<rocpd_sql_schema_kind_t> schema_kinds = {
        ROCPD_SQL_SCHEMA_ROCPD_TABLES, ROCPD_SQL_SCHEMA_ROCPD_VIEWS,
        ROCPD_SQL_SCHEMA_ROCPD_DATA_VIEWS, ROCPD_SQL_SCHEMA_ROCPD_MARKER_VIEWS,
        ROCPD_SQL_SCHEMA_ROCPD_SUMMARY_VIEWS
    };

    for(const auto& schema_kind : schema_kinds)
    {
        const std::string query = get_schema_query(schema_kind, uuid);

        if(query.empty())
        {
            std::cerr << "Failed to get schema query for schema kind: " << schema_kind
                      << std::endl;
            continue;
        }

        validate_sqlite3_result(sqlite3_exec(_sqlite3_db_temp, query.c_str(), 0, 0, 0),
                                query.c_str(),
                                std::string("Invalid schema, init database failed!"));
    }
}

void
database::execute_query(const std::string& query)
{
    validate_sqlite3_result(sqlite3_exec(_sqlite3_db_temp, query.c_str(), 0, 0, 0),
                            "Failed to execute query - ", query);
}

std::string
database::get_uuid()
{
    return m_uuid;
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
}  // namespace rocstorage
