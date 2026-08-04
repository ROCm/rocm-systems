// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "database_backend.hpp"
#include "debug.hpp"
#include "directory.hpp"
#include "profiler-hub/version.hpp"
#include "schema_catalog.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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

/**
 * Substitutes the {{uuid}}, {{schema_version}}, etc. placeholders in `schema_content`
 * (a compiled-in SQL constant from schema_catalog.hpp) with their actual runtime values.
 * @return The finished query string, or empty if `schema_content` is null.
 */
[[maybe_unused]] std::string
get_schema_query(const char*                    schema_content,
                 const std::string&             uuid,
                 const profiler_hub::version_t& schema_version)
{
    if(schema_content == nullptr)
    {
        LOG_ERROR("No compiled-in schema content provided");
        return {};
    }
    std::string query_str = schema_content;

    std::regex upid_pattern("\\{\\{uuid\\}\\}");
    std::regex guid_pattern("\\{\\{guid\\}\\}");
    std::regex view_upid_pattern("\\{\\{view_upid\\}\\}");
    std::regex schema_version_pattern("\\{\\{schema_version\\}\\}");
    std::regex schema_version_major_pattern("\\{\\{schema_version_major\\}\\}");
    std::regex schema_version_minor_pattern("\\{\\{schema_version_minor\\}\\}");
    std::regex schema_version_patch_pattern("\\{\\{schema_version_patch\\}\\}");

    query_str = std::regex_replace(query_str, upid_pattern, "_" + uuid);
    query_str = std::regex_replace(query_str, guid_pattern, uuid);
    query_str = std::regex_replace(query_str, view_upid_pattern, "");
    query_str =
        std::regex_replace(query_str, schema_version_pattern, schema_version.to_string());
    query_str = std::regex_replace(
        query_str, schema_version_major_pattern, std::to_string(schema_version.major));
    query_str = std::regex_replace(
        query_str, schema_version_minor_pattern, std::to_string(schema_version.minor));
    query_str = std::regex_replace(
        query_str, schema_version_patch_pattern, std::to_string(schema_version.patch));

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

/**
 * Creates the rocpd schema for `schema_version` (or the latest available version,
 * if version_t{}) and marks the database as initialized.
 */
template <typename SqlitePolicy>
void
database_backend<SqlitePolicy>::initialize_schema(profiler_hub::version_t schema_version)
{
    if(m_initialized)
    {
        LOG_WARNING("Database already initialized!");
        return;
    }

    // known_schema_versions() caches its table in a function-local static, so every
    // subsequent initialize_schema() call (e.g. across versions/instances) reuses it
    // instead of rebuilding it.
    const auto& versions = known_schema_versions();

    // version_t{} (all-zero) means "use whatever is the latest compiled-in version".
    profiler_hub::version_t resolved_version = schema_version;
    if(schema_version.is_latest())
    {
        for(const auto& entry : versions)
        {
            if(resolved_version.is_latest() || resolved_version < entry.version)
            {
                resolved_version = entry.version;
            }
        }
    }

    const auto version_it = std::find_if(
        versions.begin(), versions.end(), [&resolved_version](const auto& entry) {
            return entry.version == resolved_version;
        });
    if(version_it == versions.end())
    {
        throw std::runtime_error("Unsupported rocpd schema version requested: " +
                                 schema_version.to_string());
    }

    for(const char* schema_content : version_it->sql)
    {
        const std::string query =
            get_schema_query(schema_content, m_uuid, resolved_version);
        if(query.empty())
        {
            LOG_ERROR("Failed to get schema query for version {}",
                      resolved_version.to_string());
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
