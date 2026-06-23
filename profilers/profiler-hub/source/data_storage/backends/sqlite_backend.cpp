// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "sqlite_backend.hpp"

#include "debug.hpp"
#include "directory.hpp"

#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(USE_SCHEMA_FROM_ROCPROFILER_SDK_ROCPD) &&                                    \
    USE_SCHEMA_FROM_ROCPROFILER_SDK_ROCPD > 0
#    include <rocprofiler-sdk-rocpd/sql.h>
#    include <rocprofiler-sdk-rocpd/types.h>
#else
#    include <regex>

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
    auto db_dirname = profiler_hub::common::dirname(db_file);
    if(!profiler_hub::common::direxists(db_dirname))
    {
        profiler_hub::common::makedir(db_dirname);
    }
}

#if defined(USE_SCHEMA_FROM_ROCPROFILER_SDK_ROCPD) &&                                    \
    USE_SCHEMA_FROM_ROCPROFILER_SDK_ROCPD > 0
void
load_schema_cb(rocpd_sql_engine_t /*unused*/,
               rocpd_sql_schema_kind_t /*unused*/,
               rocpd_sql_options_t /*unused*/,
               const rocpd_sql_schema_jinja_variables_t* /*unused*/,
               const char* /*unused*/,
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
    rocpd_sql_schema_jinja_variables_t const info{
        sizeof(rocpd_sql_schema_jinja_variables_t), uuid.c_str(), uuid.c_str()
    };

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

    std::string query_str = std::string(schema_content);

    std::regex upid_pattern("\\{\\{uuid\\}\\}");
    std::regex guid_pattern("\\{\\{guid\\}\\}");
    std::regex view_upid_pattern("\\{\\{view_upid\\}\\}");

    query_str = std::regex_replace(query_str, upid_pattern, "_" + uuid);
    query_str = std::regex_replace(query_str, guid_pattern, uuid);
    query_str = std::regex_replace(query_str, view_upid_pattern, "");

    return query_str;
#endif
}

}  // namespace

namespace profiler_hub::data_storage
{

std::shared_ptr<sqlite_backend>
sqlite_backend::create(std::string db_path, std::string uuid, storage_mode_t mode)
{
    auto backend = std::shared_ptr<sqlite_backend>(
        new sqlite_backend(std::move(db_path), std::move(uuid), mode));

    // discover_uuids() uses create_read_statement_executor which calls
    // shared_from_this(). This must happen after the shared_ptr is fully
    // constructed -- calling shared_from_this() inside the constructor
    // throws bad_weak_ptr.
    if(backend->m_initialized)
    {
        auto uuids = backend->discover_uuids();
        if(uuids.size() == 1)
        {
            backend->m_uuid = uuids[0];
        }
    }

    return backend;
}

sqlite_backend::sqlite_backend(std::string db_path, std::string uuid, storage_mode_t mode)
: m_db_path{ std::move(db_path) }
, m_uuid{ std::move(uuid) }
, m_mode{ mode }
{
    if(std::filesystem::exists(m_db_path))
    {
        m_mode        = storage_mode_t::on_disk;
        m_initialized = true;
    }
    else
    {
        create_directory_for_database_file(m_db_path);
    }

    if(m_mode == storage_mode_t::in_memory)
    {
        validate_sqlite3_result(
            sqlite3_open(":memory:", &m_sqlite3), "", "database open failed!");
    }
    else if(m_mode == storage_mode_t::on_disk)
    {
        validate_sqlite3_result(
            sqlite3_open(m_db_path.c_str(), &m_sqlite3), "", "database open failed!");
    }

    validate_sqlite3_result(
        sqlite3_prepare_v2(m_sqlite3, "BEGIN TRANSACTION", -1, &m_begin_stmt, nullptr),
        "BEGIN TRANSACTION",
        "prepare failed");
    validate_sqlite3_result(
        sqlite3_prepare_v2(m_sqlite3, "COMMIT", -1, &m_commit_stmt, nullptr),
        "COMMIT",
        "prepare failed");
    validate_sqlite3_result(
        sqlite3_prepare_v2(m_sqlite3, "ROLLBACK", -1, &m_rollback_stmt, nullptr),
        "ROLLBACK",
        "prepare failed");

    LOG_INFO("profiler_hub database initialized (uuid: {}, path: {})", m_uuid, m_db_path);
}

sqlite_backend::~sqlite_backend()
{
    sqlite3_finalize(m_begin_stmt);
    sqlite3_finalize(m_commit_stmt);
    sqlite3_finalize(m_rollback_stmt);
    sqlite3_close(m_sqlite3);
}

std::vector<std::string>
sqlite_backend::discover_uuids()
{
    struct uuid_result
    {
        std::string uuid;
    };
    auto uuid_query_executor = create_read_statement_executor<uuid_result>(
        "SELECT DISTINCT replace(name, rtrim(name, replace(name, '_', '')), '') "
        "AS guid "
        "FROM sqlite_master WHERE type='table' AND name LIKE 'rocpd_%';",
        &uuid_result::uuid);

    auto result = uuid_query_executor().to_vector();

    std::vector<std::string> uuids;
    uuids.reserve(result.size());
    for(const auto& row : result)
    {
        uuids.push_back(row.uuid);
    }
    return uuids;
}

std::string
sqlite_backend::get_uuid() const
{
    return m_uuid;
}

void
sqlite_backend::initialize_schema()
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

        validate_sqlite3_result(
            sqlite3_exec(m_sqlite3, query.c_str(), nullptr, nullptr, nullptr),
            query.c_str(),
            std::string("Invalid schema, init database failed!"));
    }

    m_initialized = true;
}

void
sqlite_backend::execute(const std::string& query)
{
    validate_sqlite3_result(
        sqlite3_exec(m_sqlite3, query.c_str(), nullptr, nullptr, nullptr),
        "Failed to execute query:",
        query);
}

void
sqlite_backend::flush()
{
    if(m_mode != storage_mode_t::in_memory)
    {
        LOG_WARNING("Flushing database is not supported for database type: {}",
                    static_cast<int>(m_mode));
        return;
    }

    if(m_flushed)
    {
        throw std::runtime_error("Database already flushed!");
    }

    sqlite3* out_db = nullptr;
    validate_sqlite3_result(
        sqlite3_open(m_db_path.c_str(), &out_db), "", "database open failed!");
    auto* backup = sqlite3_backup_init(out_db, "main", m_sqlite3, "main");
    if(backup == nullptr)
    {
        const std::string err = sqlite3_errmsg(out_db);
        sqlite3_close(out_db);
        throw std::runtime_error("sqlite3_backup_init failed: " + err);
    }

    const auto step_rc   = sqlite3_backup_step(backup, -1);
    const auto finish_rc = sqlite3_backup_finish(backup);
    sqlite3_close(out_db);

    if(step_rc != SQLITE_DONE)
    {
        throw std::runtime_error("sqlite3_backup_step failed: rc=" +
                                 std::to_string(step_rc));
    }
    if(finish_rc != SQLITE_OK)
    {
        throw std::runtime_error("sqlite3_backup_finish failed: rc=" +
                                 std::to_string(finish_rc));
    }
    m_flushed = true;
}

}  // namespace profiler_hub::data_storage
