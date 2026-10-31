// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "database_backend.hpp"
#include "debug.hpp"
#include "directory.hpp"
#include "embedded_schema.hpp"
#include "profiler-hub/version.hpp"

#include <filesystem>
#include <regex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{

using rocpd::data_storage::schema::embedded_schema_t;
using rocpd::data_storage::schema::find_embedded_schema;
using rocpd::data_storage::schema::k_embedded_schemas;

/**
 * Creates the parent directory of `db_file` when it does not already exist.
 * Called for a new database path, before the file itself is opened. An existing
 * directory is left unchanged.
 */
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
 * (a compiled-in SQL constant from the versions.yml headers) with their actual runtime
 * values.
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

/**
 * Selects the embedded RocPD schema for `schema_version`.
 * A concrete version is looked up by "MAJOR.MINOR.PATCH". version_t{} (all zero)
 * selects the highest major.minor.patch in k_embedded_schemas.
 * @return The matching catalog row, or nullptr when the version is not embedded.
 */
[[maybe_unused]] const embedded_schema_t*
resolve_embedded_schema(const profiler_hub::version_t& schema_version)
{
    if(!schema_version.is_latest())
    {
        return find_embedded_schema(schema_version.to_string());
    }
    // version_t{} selects the highest major.minor.patch in k_embedded_schemas.
    const embedded_schema_t* newest_schema = nullptr;
    for(const auto& schema : k_embedded_schemas)
    {
        const profiler_hub::version_t embedded_version{ schema.major,
                                                        schema.minor,
                                                        schema.patch };
        if(newest_schema == nullptr ||
           profiler_hub::version_t{ newest_schema->major,
                                    newest_schema->minor,
                                    newest_schema->patch } < embedded_version)
        {
            newest_schema = &schema;
        }
    }
    return newest_schema;
}

}  // namespace

namespace profiler_hub::data_storage
{

/**
 * Factory for a backend owned by `shared_ptr`.
 * The constructor opens `db_path` in `mode`. When that file already exists, the
 * backend is treated as initialized and, if it contains exactly one RocPD uuid,
 * that uuid replaces `uuid`. Uuid discovery uses `shared_from_this()`, so it runs
 * only after this `shared_ptr` exists.
 * @return The open backend.
 */
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

/**
 * Opens the connection for `db_path`.
 * An existing file is opened on disk and marked initialized, because its schema is
 * already present. A new path gets its parent directory created, then the connection
 * is opened in memory or on disk according to `mode`. BEGIN, COMMIT, and ROLLBACK
 * are prepared for later transactions.
 */
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

/**
 * Finalizes the prepared transaction statements and closes the connection.
 */
template <typename SqlitePolicy>
database_backend<SqlitePolicy>::~database_backend()
{
    SqlitePolicy::finalize(m_begin_stmt);
    SqlitePolicy::finalize(m_commit_stmt);
    SqlitePolicy::finalize(m_rollback_stmt);
    SqlitePolicy::close(m_sqlite3);
}

/**
 * Reads the distinct RocPD uuids from table names in sqlite_master.
 * Names look like `rocpd_<table>_<uuid>`; the query returns the text after the last
 * underscore.
 * @return One entry per distinct uuid. Empty when the database has no rocpd tables.
 */
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
 * Creates the RocPD schema for `schema_version` and marks the database initialized.
 * version_t{} (all zero) uses the newest embedded schema. An already-initialized
 * database logs a warning and returns. Tables, indexes, views, data views, summary
 * views, and metadata are executed in that order, with uuid and version placeholders
 * filled in. An unknown version throws.
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

    // version_t{} (all-zero) means the newest entry generated from versions.yml.
    const auto* schema = resolve_embedded_schema(schema_version);
    if(schema == nullptr)
    {
        throw std::runtime_error("Unsupported rocpd schema version requested: " +
                                 schema_version.to_string());
    }

    const profiler_hub::version_t resolved_version{ schema->major,
                                                    schema->minor,
                                                    schema->patch };
    const char* schema_parts[] = { schema->tables,        schema->indexes,
                                   schema->views,         schema->data_views,
                                   schema->summary_views, schema->metadata };
    for(const char* schema_content : schema_parts)
    {
        // A missing schema here means a broken build, so stop instead of a partial DB.
        if(schema_content == nullptr)
        {
            throw std::runtime_error("Missing compiled-in rocpd schema for version " +
                                     resolved_version.to_string());
        }

        const std::string query =
            get_schema_query(schema_content, m_uuid, resolved_version);
        if(query.empty())
        {
            continue;
        }

        validate_sqlite3_result(SqlitePolicy::exec(m_sqlite3, query.c_str()),
                                query.c_str(),
                                std::string("Invalid schema, init database failed!"));
    }
    m_initialized = true;
}

/**
 * Runs `query` on the open connection.
 * Throws when SQLite reports an error.
 */
template <typename SqlitePolicy>
void
database_backend<SqlitePolicy>::execute(const std::string& query)
{
    validate_sqlite3_result(
        SqlitePolicy::exec(m_sqlite3, query.c_str()), "Failed to execute query:", query);
}

/**
 * Copies an in-memory database onto `m_db_path`.
 * On-disk backends log a warning and return. A second flush of the same backend throws.
 */
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
