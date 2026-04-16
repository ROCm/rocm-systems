// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "sqlite_backend.hpp"

#include "debug.hpp"
#include "directory.hpp"

#include <filesystem>
#include <regex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if !defined(ROCPD_SQL_HAS_SCHEMA_VERSION)
#    define ROCPD_SQL_HAS_SCHEMA_VERSION 0
#endif

#if defined(USE_SCHEMA_FROM_ROCPROFILER_SDK_ROCPD) &&                                    \
    USE_SCHEMA_FROM_ROCPROFILER_SDK_ROCPD > 0
#    include <rocprofiler-sdk-rocpd/sql.h>
#    include <rocprofiler-sdk-rocpd/types.h>
#else
#    include <regex>

// V3 schema headers (namespace: rocpd::data_storage::schema_v3)
#    include "schema/3.0.0/data_views.hpp"
#    include "schema/3.0.0/rocpd_tables.hpp"
#    include "schema/3.0.0/rocpd_views.hpp"
#    include "schema/3.0.0/summary_views.hpp"

// V4 schema headers (namespace: rocpd::data_storage::schema_v4)
#    include "schema/4.0.0/data_views.hpp"
#    include "schema/4.0.0/rocpd_metadata.hpp"
#    include "schema/4.0.0/rocpd_tables.hpp"
#    include "schema/4.0.0/rocpd_views.hpp"
#    include "schema/4.0.0/summary_views.hpp"

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

// Helper to check if version is v4 or later
inline bool
is_v4_or_later(const rocpdsna::version_t& version)
{
    return version.major == 4 && version.minor == 0 && version.patch == 0;
}

}  // namespace

#endif

namespace
{
void
create_directory_for_database_file(const std::string& db_file)
{
    auto db_dirname = rocpdsna::common::dirname(db_file);
    if(!rocpdsna::common::direxists(db_dirname))
    {
        rocpdsna::common::makedir(db_dirname);
    }
}

#if defined(USE_SCHEMA_FROM_ROCPROFILER_SDK_ROCPD) &&                                    \
    USE_SCHEMA_FROM_ROCPROFILER_SDK_ROCPD > 0
void
load_schema_cb(rocpd_sql_engine_t      engine,
               rocpd_sql_schema_kind_t schema_kind,
               rocpd_sql_options_t     options,
#    if ROCPD_SQL_HAS_SCHEMA_VERSION
               rocpd_version_triplet_t schema_version,
#    endif
               const rocpd_sql_schema_jinja_variables_t* jinja_vars,
               const char*                               filename,
               const char*                               schema_content,
               void*                                     user_data)
{
    (void) engine;
    (void) schema_kind;
    (void) options;
#    if ROCPD_SQL_HAS_SCHEMA_VERSION
    (void) schema_version;
#    endif
    (void) jinja_vars;
    (void) filename;

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
get_schema_query(rocpd_sql_schema_kind_t schema_kind,
                 const std::string&      uuid,
                 rocpdsna::version_t     version)
{
    std::string query;
#if defined(USE_SCHEMA_FROM_ROCPROFILER_SDK_ROCPD) &&                                    \
    USE_SCHEMA_FROM_ROCPROFILER_SDK_ROCPD > 0
    const auto                               jinja_size = 2 * uuid.size();
    rocpd_sql_schema_jinja_variables_t const info{ jinja_size,
                                                   uuid.c_str(),
                                                   uuid.c_str() };

#    if ROCPD_SQL_HAS_SCHEMA_VERSION
    rocpd_version_triplet_t sdk_schema_version = { version.major,
                                                   version.minor,
                                                   version.patch };

    auto status = rocpd_sql_load_schema(ROCPD_SQL_ENGINE_SQLITE3,
                                        schema_kind,
                                        ROCPD_SQL_OPTIONS_NONE,
                                        sdk_schema_version,
                                        &info,
                                        load_schema_cb,
                                        nullptr,
                                        0,
                                        &query);
#    else
    auto status = rocpd_sql_load_schema(ROCPD_SQL_ENGINE_SQLITE3,
                                        schema_kind,
                                        ROCPD_SQL_OPTIONS_NONE,
                                        &info,
                                        load_schema_cb,
                                        nullptr,
                                        0,
                                        &query);
#    endif

    if(status != ROCPD_STATUS_SUCCESS)
    {
        LOG_ERROR("Unable to load rocpd schema (error code: {})",
                  static_cast<int>(status));
    }
#else
    // Runtime version selection for local schema files
    std::string_view schema_content;
    const bool       use_v4 = is_v4_or_later(version);

    switch(schema_kind)
    {
        case ROCPD_SQL_SCHEMA_ROCPD_TABLES:
            schema_content = use_v4 ? rocpd::data_storage::schema_v4::ROCPD_TABLES_SQL
                                    : rocpd::data_storage::schema_v3::ROCPD_TABLES_SQL;
            break;
        case ROCPD_SQL_SCHEMA_ROCPD_VIEWS:
            schema_content = use_v4 ? rocpd::data_storage::schema_v4::ROCPD_VIEWS_SQL
                                    : rocpd::data_storage::schema_v3::ROCPD_VIEWS_SQL;
            break;
        case ROCPD_SQL_SCHEMA_ROCPD_DATA_VIEWS:
            schema_content = use_v4 ? rocpd::data_storage::schema_v4::DATA_VIEWS_SQL
                                    : rocpd::data_storage::schema_v3::DATA_VIEWS_SQL;
            break;
        case ROCPD_SQL_SCHEMA_ROCPD_SUMMARY_VIEWS:
            schema_content = use_v4 ? rocpd::data_storage::schema_v4::SUMMARY_VIEWS_SQL
                                    : rocpd::data_storage::schema_v3::SUMMARY_VIEWS_SQL;
            break;
        case ROCPD_SQL_SCHEMA_ROCPD_METADATA:
            // v4 has rocpd_metadata.sql, v3 does not (metadata is in rocpd_tables.sql for
            // v3)
            schema_content =
                use_v4 ? rocpd::data_storage::schema_v4::ROCPD_METADATA_SQL : "";
            break;
        default:
            throw std::runtime_error("Unknown schema kind: " +
                                     std::to_string(schema_kind));
    }

    query = std::string(schema_content);

    std::regex upid_pattern("\\{\\{uuid\\}\\}");
    std::regex guid_pattern("\\{\\{guid\\}\\}");
    std::regex view_upid_pattern("\\{\\{view_upid\\}\\}");

    query = std::regex_replace(query, upid_pattern, "_" + uuid);
    query = std::regex_replace(query, guid_pattern, uuid);
    query = std::regex_replace(query, view_upid_pattern, "");
#endif
    // Substitute schema version variables (needed for rocpd_metadata.sql)
    std::regex schema_version_pattern(R"(\{\{schema_version\}\})");
    std::regex schema_version_major_pattern(R"(\{\{schema_version_major\}\})");
    std::regex schema_version_minor_pattern(R"(\{\{schema_version_minor\}\})");
    std::regex schema_version_patch_pattern(R"(\{\{schema_version_patch\}\})");

    std::string version_string = std::to_string(version.major) + "." +
                                 std::to_string(version.minor) + "." +
                                 std::to_string(version.patch);

    query = std::regex_replace(query, schema_version_pattern, version_string);
    query = std::regex_replace(
        query, schema_version_major_pattern, std::to_string(version.major));
    query = std::regex_replace(
        query, schema_version_minor_pattern, std::to_string(version.minor));
    query = std::regex_replace(
        query, schema_version_patch_pattern, std::to_string(version.patch));

    return query;
}

}  // namespace

namespace rocpdsna::data_storage
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

    LOG_INFO("rocpdsna database initialized (uuid: {}, path: {})", m_uuid, m_db_path);
}

sqlite_backend::~sqlite_backend() { sqlite3_close(m_sqlite3); }

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
sqlite_backend::initialize_schema(rocpdsna::version_t version)
{
    if(m_initialized)
    {
        throw std::runtime_error("Database already initialized!");
    }

    LOG_INFO("Initializing schema version {}.{}.{}",
             version.major,
             version.minor,
             version.patch);

    std::vector<rocpd_sql_schema_kind_t> schema_kinds = {
        ROCPD_SQL_SCHEMA_ROCPD_TABLES,
        ROCPD_SQL_SCHEMA_ROCPD_VIEWS,
        ROCPD_SQL_SCHEMA_ROCPD_DATA_VIEWS,
        ROCPD_SQL_SCHEMA_ROCPD_SUMMARY_VIEWS,
    };

#if ROCPD_SQL_HAS_SCHEMA_VERSION
    schema_kinds.push_back(ROCPD_SQL_SCHEMA_ROCPD_METADATA);
#else
    // Old SDK has MARKER_VIEWS instead of METADATA
    schema_kinds.push_back(ROCPD_SQL_SCHEMA_ROCPD_MARKER_VIEWS);
#endif

    for(const auto& schema_kind : schema_kinds)
    {
        const std::string query = get_schema_query(schema_kind, m_uuid, version);

        if(query.empty())
        {
            LOG_INFO("Skipping empty schema query for schema kind: {}",
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
    if(backup != nullptr)
    {
        sqlite3_backup_step(backup, -1);
        sqlite3_backup_finish(backup);
    }
    sqlite3_close(out_db);
    m_flushed = true;
}

}  // namespace rocpdsna::data_storage
