// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "database.hpp"

#include "debug.hpp"
#include "directory.hpp"

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

#if !defined(USE_SCHEMA_FROM_ROCPROFILER_SDK_ROCPD) ||                                   \
    USE_SCHEMA_FROM_ROCPROFILER_SDK_ROCPD == 0
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
#endif

#if defined(USE_SCHEMA_FROM_ROCPROFILER_SDK_ROCPD) &&                                    \
    USE_SCHEMA_FROM_ROCPROFILER_SDK_ROCPD > 0
void
load_schema_cb(rocpd_sql_engine_t,
               rocpd_sql_schema_kind_t,
               rocpd_sql_options_t,
               const rocpd_sql_schema_jinja_variables_t*,
               const char*,
               const char* schema_content,
               void*       user_data)
{
    if(user_data == nullptr || schema_content == nullptr)
    {
        LOG_ERROR("Invalid user data or schema content pointer");
        return;
    }
    auto* query = static_cast<std::string*>(user_data);
    if(query == nullptr)
    {
        LOG_ERROR("Invalid query pointer");
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
    auto        status = rocpd_sql_load_schema(ROCPD_SQL_ENGINE_SQLITE3,
                                        schema_kind,
                                        ROCPD_SQL_OPTIONS_NONE,
                                        &info,
                                        load_schema_cb,
                                        nullptr,
                                        0,
                                        &query);
    if(status != ROCPD_STATUS_SUCCESS)
    {
        LOG_ERROR("Unable to load rocpd schema (error code: {})",
                  static_cast<int>(status));
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
database::database(std::string                 db_path,
                   std::string                 uuid,
                   rocstorage::database_type_t database_type)
: m_db_path{ std::move(db_path) }
, m_uuid{ std::move(uuid) }
, m_database_type{ database_type }
{
    create_directory_for_database_file(m_db_path);
    LOG_INFO("rocstorage database initialized (uuid: {}, path: {})", m_uuid, m_db_path);

    if(m_database_type == rocstorage::database_type_t::in_memory)
    {
        validate_sqlite3_result(
            sqlite3_open(":memory:", &m_sqlite3), "", "database open failed!");
    }
    else if(m_database_type == rocstorage::database_type_t::on_disk)
    {
        validate_sqlite3_result(
            sqlite3_open(m_db_path.c_str(), &m_sqlite3), "", "database open failed!");
    }
}

database::~database() { sqlite3_close(m_sqlite3); }

std::string
database::get_uuid() const
{
    return m_uuid;
}

void
database::initialize_schema()
{
    if(m_initialized)
    {
        throw std::runtime_error("Database already initialized!");
    }

    const std::vector<rocpd_sql_schema_kind_t> schema_kinds = {
        ROCPD_SQL_SCHEMA_ROCPD_TABLES,
        ROCPD_SQL_SCHEMA_ROCPD_VIEWS,
        ROCPD_SQL_SCHEMA_ROCPD_DATA_VIEWS,
        ROCPD_SQL_SCHEMA_ROCPD_MARKER_VIEWS,
        ROCPD_SQL_SCHEMA_ROCPD_SUMMARY_VIEWS
    };

    for(const auto& schema_kind : schema_kinds)
    {
        const std::string query = get_schema_query(schema_kind, m_uuid);

        if(query.empty())
        {
            LOG_ERROR("Failed to get schema query for schema kind: {}",
                      static_cast<int>(schema_kind));
            continue;
        }

        validate_sqlite3_result(sqlite3_exec(m_sqlite3, query.c_str(), 0, 0, 0),
                                query.c_str(),
                                std::string("Invalid schema, init database failed!"));
    }

    m_initialized = true;
}

void
database::execute_query(const std::string& query)
{
    validate_sqlite3_result(sqlite3_exec(m_sqlite3, query.c_str(), 0, 0, 0),
                            "Failed to execute query:",
                            query);
}

size_t
database::get_last_insert_id() const
{
    return sqlite3_last_insert_rowid(m_sqlite3);
}

void
database::flush()
{
    if(m_database_type != rocstorage::database_type_t::in_memory)
    {
        LOG_WARNING("Flushing database is not supported for database type: {}",
                    static_cast<int>(m_database_type));
        return;
    }

    if(m_flushed)
    {
        throw std::runtime_error("Database already flushed!");
    }

    sqlite3* out_db;
    validate_sqlite3_result(
        sqlite3_open(m_db_path.c_str(), &out_db), "", "database open failed!");
    auto* backup = sqlite3_backup_init(out_db, "main", m_sqlite3, "main");
    if(backup)
    {
        sqlite3_backup_step(backup, -1);
        sqlite3_backup_finish(backup);
    }
    sqlite3_close(out_db);
    m_flushed = true;
}

}  // namespace data_storage
}  // namespace rocstorage
