// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "database_backend.hpp"

#include "debug.hpp"
#include "directory.hpp"

#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <regex>

#include "schema/data_views.hpp"
#include "schema/rocpd_metadata.hpp"
#include "schema/rocpd_tables.hpp"
#include "schema/rocpd_views.hpp"
#include "schema/summary_views.hpp"

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
    ROCPD_SQL_SCHEMA_ROCPD_METADATA,
    ROCPD_SQL_SCHEMA_LAST,
};
}  // namespace

namespace
{
[[maybe_unused]] void
create_directory_for_database_file(const std::string& db_file)
{
    auto db_dirname = profiler_hub::common::dirname(db_file);
    if(!profiler_hub::common::direxists(db_dirname))
    {
        profiler_hub::common::makedir(db_dirname);
    }
}

[[maybe_unused]] std::string
get_schema_query(rocpd_sql_schema_kind_t schema_kind, const std::string& uuid)
{
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
        case ROCPD_SQL_SCHEMA_ROCPD_SUMMARY_VIEWS:
            schema_content = rocpd::data_storage::schema::SUMMARY_VIEWS_SQL;
            break;
        case ROCPD_SQL_SCHEMA_ROCPD_METADATA:
            schema_content = rocpd::data_storage::schema::ROCPD_METADATA_SQL;
            break;
        default:
            throw std::runtime_error("Unknown schema kind: " +
                                     std::to_string(schema_kind));
    }

    std::string query_str = std::string(schema_content);

    std::regex upid_pattern("\\{\\{uuid\\}\\}");
    std::regex guid_pattern("\\{\\{guid\\}\\}");
    std::regex view_upid_pattern("\\{\\{view_upid\\}\\}");

    // rocpd_metadata.sql references these; hardcoded to the baseline schema version
    // (3.0.0) for now until schema files are selected by version rather than always
    // taking the latest cloned/bundled copy.
    std::regex schema_version_pattern("\\{\\{schema_version\\}\\}");
    std::regex schema_version_major_pattern("\\{\\{schema_version_major\\}\\}");
    std::regex schema_version_minor_pattern("\\{\\{schema_version_minor\\}\\}");
    std::regex schema_version_patch_pattern("\\{\\{schema_version_patch\\}\\}");

    query_str = std::regex_replace(query_str, upid_pattern, "_" + uuid);
    query_str = std::regex_replace(query_str, guid_pattern, uuid);
    query_str = std::regex_replace(query_str, view_upid_pattern, "");
    query_str = std::regex_replace(query_str, schema_version_pattern, "3.0.1");
    query_str = std::regex_replace(query_str, schema_version_major_pattern, "3");
    query_str = std::regex_replace(query_str, schema_version_minor_pattern, "0");
    query_str = std::regex_replace(query_str, schema_version_patch_pattern, "1");

    return query_str;
}

}  // namespace

namespace profiler_hub::data_storage
{

template <typename SqlitePolicy>
std::shared_ptr<database_backend<SqlitePolicy>>
database_backend<SqlitePolicy>::create(std::string    db_path,
                                       std::string    uuid,
                                       storage_mode_t mode)
{
    auto backend = std::shared_ptr<database_backend>(
        new database_backend(std::move(db_path), std::move(uuid), mode));

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

template <typename SqlitePolicy>
database_backend<SqlitePolicy>::database_backend(std::string    db_path,
                                                 std::string    uuid,
                                                 storage_mode_t mode)
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
            SqlitePolicy::open(":memory:", &m_sqlite3), "", "database open failed!");
    }
    else if(m_mode == storage_mode_t::on_disk)
    {
        validate_sqlite3_result(SqlitePolicy::open(m_db_path.c_str(), &m_sqlite3),
                                "",
                                "database open failed!");
    }

    validate_sqlite3_result(
        SqlitePolicy::prepare(m_sqlite3, "BEGIN TRANSACTION", &m_begin_stmt),
        "BEGIN TRANSACTION",
        "prepare failed");
    validate_sqlite3_result(SqlitePolicy::prepare(m_sqlite3, "COMMIT", &m_commit_stmt),
                            "COMMIT",
                            "prepare failed");
    validate_sqlite3_result(
        SqlitePolicy::prepare(m_sqlite3, "ROLLBACK", &m_rollback_stmt),
        "ROLLBACK",
        "prepare failed");

    LOG_INFO("profiler_hub database initialized (uuid: {}, path: {})", m_uuid, m_db_path);
}

template <typename SqlitePolicy>
database_backend<SqlitePolicy>::~database_backend()
{
    SqlitePolicy::finalize(m_begin_stmt);
    SqlitePolicy::finalize(m_commit_stmt);
    SqlitePolicy::finalize(m_rollback_stmt);
    SqlitePolicy::close(m_sqlite3);
}

template <typename SqlitePolicy>
std::vector<std::string>
database_backend<SqlitePolicy>::discover_uuids()
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

template <typename SqlitePolicy>
void
database_backend<SqlitePolicy>::initialize_schema()
{
    if(m_initialized)
    {
        throw std::runtime_error("Database already initialized!");
    }

    const std::vector<rocpd_sql_schema_kind_t> schema_kinds = {
        ROCPD_SQL_SCHEMA_ROCPD_TABLES,
        ROCPD_SQL_SCHEMA_ROCPD_VIEWS,
        ROCPD_SQL_SCHEMA_ROCPD_DATA_VIEWS,
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

        validate_sqlite3_result(SqlitePolicy::exec(m_sqlite3, query.c_str()),
                                query.c_str(),
                                std::string("Invalid schema, init database failed!"));
    }

    m_initialized = true;
}

template <typename SqlitePolicy>
void
database_backend<SqlitePolicy>::execute(const std::string& query)
{
    validate_sqlite3_result(
        SqlitePolicy::exec(m_sqlite3, query.c_str()), "Failed to execute query:", query);
}

template <typename SqlitePolicy>
void
database_backend<SqlitePolicy>::flush()
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

    std::string backup_errmsg;
    const int   rc =
        SqlitePolicy::backup_to_file(m_sqlite3, m_db_path.c_str(), backup_errmsg);
    if(rc != SqlitePolicy::result_ok)
    {
        throw std::runtime_error("Database flush (backup) failed: rc=" +
                                 std::to_string(rc) + ": " + backup_errmsg);
    }
    m_flushed = true;
}

}  // namespace profiler_hub::data_storage
