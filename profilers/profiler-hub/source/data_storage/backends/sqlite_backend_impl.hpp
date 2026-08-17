// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "database_backend.hpp"

#include "debug.hpp"
#include "directory.hpp"
#include "schema_manifest.hpp"

#include <dlfcn.h>

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

[[nodiscard]] static const std::filesystem::path&
schema_directory()
{
    static const std::filesystem::path installed_schema_dir = [] {
        Dl_info library_info{};
        if(dladdr(reinterpret_cast<const void*>(&schema_directory), &library_info) != 0 &&
           library_info.dli_fname != nullptr)
        {
            const auto schema_dir = std::filesystem::path{ library_info.dli_fname }
                                        .parent_path()
                                        .parent_path() /
                                    "share/profiler-hub/schema";
            if(std::filesystem::exists(schema_dir / "versions.yml"))
            {
                return schema_dir;
            }
        }

        throw std::runtime_error("Unable to locate profiler-hub schema directory "
                                 "relative to the loaded library");
    }();

    return installed_schema_dir;
}

[[nodiscard]] static std::filesystem::path
schema_file_path(const std::string& schema_file_name)
{
    return schema_directory() / schema_file_name;
}

// Reads raw SQL copied by cmake/rocprofiler-sdk-rocpd.cmake into the schema directory.
[[maybe_unused]] std::string
read_schema_file(const std::string& schema_file_name)
{
    const std::filesystem::path schema_path = schema_file_path(schema_file_name);

    std::ifstream schema_file(schema_path, std::ios::in | std::ios::binary);
    if(!schema_file)
    {
        throw std::runtime_error("Failed to open schema file: " + schema_path.string());
    }

    std::ostringstream schema_stream;
    schema_stream << schema_file.rdbuf();
    return schema_stream.str();
}

[[maybe_unused]] std::string
get_schema_query(rocpd_sql_schema_kind_t                                schema_kind,
                 const profiler_hub::data_storage::kind_filename_map_t& kind_paths,
                 const profiler_hub::version_t&                         version,
                 const std::string&                                     uuid)
{
    std::string query_str;

    switch(schema_kind)
    {
        case ROCPD_SQL_SCHEMA_ROCPD_TABLES:
            query_str = read_schema_file(kind_paths.at("rocpd_tables"));
            break;
        case ROCPD_SQL_SCHEMA_ROCPD_INDEXES:
            query_str = read_schema_file(kind_paths.at("rocpd_indexes"));
            break;
        case ROCPD_SQL_SCHEMA_ROCPD_VIEWS:
            query_str = read_schema_file(kind_paths.at("rocpd_views"));
            break;
        case ROCPD_SQL_SCHEMA_ROCPD_DATA_VIEWS:
            query_str = read_schema_file(kind_paths.at("rocpd_data_views"));
            break;
        case ROCPD_SQL_SCHEMA_ROCPD_SUMMARY_VIEWS:
            query_str = read_schema_file(kind_paths.at("rocpd_summary_views"));
            break;
        case ROCPD_SQL_SCHEMA_ROCPD_METADATA:
            query_str = read_schema_file(kind_paths.at("rocpd_metadata"));
            break;
        default:
            throw std::runtime_error("Unknown schema kind: " +
                                     std::to_string(schema_kind));
    }

    static const std::regex upid_pattern("\\{\\{uuid\\}\\}");
    static const std::regex guid_pattern("\\{\\{guid\\}\\}");
    static const std::regex view_upid_pattern("\\{\\{view_upid\\}\\}");
    static const std::regex schema_version_pattern("\\{\\{schema_version\\}\\}");
    static const std::regex schema_version_major_pattern(
        "\\{\\{schema_version_major\\}\\}");
    static const std::regex schema_version_minor_pattern(
        "\\{\\{schema_version_minor\\}\\}");
    static const std::regex schema_version_patch_pattern(
        "\\{\\{schema_version_patch\\}\\}");

    query_str = std::regex_replace(query_str, upid_pattern, "_" + uuid);
    query_str = std::regex_replace(query_str, guid_pattern, uuid);
    query_str = std::regex_replace(query_str, view_upid_pattern, "");
    query_str =
        std::regex_replace(query_str, schema_version_pattern, version.to_string());
    query_str = std::regex_replace(
        query_str, schema_version_major_pattern, std::to_string(version.major));
    query_str = std::regex_replace(
        query_str, schema_version_minor_pattern, std::to_string(version.minor));
    query_str = std::regex_replace(
        query_str, schema_version_patch_pattern, std::to_string(version.patch));

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
database_backend<SqlitePolicy>::initialize_schema(profiler_hub::version_t schema_version)
{
    if(m_initialized)
    {
        LOG_WARNING("Database already initialized!");
        return;
    }

    version_file_map_t      version_file_map;
    profiler_hub::version_t latest_version;
    // Load the manifest before resolving which schema version to initialize.
    load_schema_manifest(schema_directory(), version_file_map, latest_version);

    // Resolve "latest" or an explicit request to the manifest entry we will apply.
    const profiler_hub::version_t resolved_version =
        resolve_schema_version(version_file_map, latest_version, schema_version);
    const auto& kind_paths = version_file_map.at(resolved_version.to_string());

    LOG_INFO("Initializing rocpd schema version {} (uuid: {})",
             resolved_version.to_string(),
             m_uuid);

    // Order matters: indexes are created on tables that must already exist.
    const std::vector<rocpd_sql_schema_kind_t> schema_kinds = {
        ROCPD_SQL_SCHEMA_ROCPD_TABLES,        ROCPD_SQL_SCHEMA_ROCPD_INDEXES,
        ROCPD_SQL_SCHEMA_ROCPD_VIEWS,         ROCPD_SQL_SCHEMA_ROCPD_DATA_VIEWS,
        ROCPD_SQL_SCHEMA_ROCPD_SUMMARY_VIEWS, ROCPD_SQL_SCHEMA_ROCPD_METADATA,
    };
    for(const auto& schema_kind : schema_kinds)
    {
        const std::string query =
            get_schema_query(schema_kind, kind_paths, resolved_version, m_uuid);

        if(query.empty())
        {
            throw std::runtime_error("Empty schema SQL for kind " +
                                     std::to_string(static_cast<int>(schema_kind)) +
                                     " (schema version " + resolved_version.to_string() +
                                     ")");
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
