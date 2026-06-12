// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "database.hpp"
#include "common/md5sum.hpp"
#include "node_info.hpp"
#include <cstdint>

#include "logger/debug.hpp"

#include <config.hpp>
#include <regex>
#include <string>
#include <timemory/environment/types.hpp>
#include <timemory/utility/filepath.hpp>
#include <unistd.h>

#if defined(ROCPROFSYS_USE_ROCPD_LIBRARY) && ROCPROFSYS_USE_ROCPD_LIBRARY > 0
#    include <dlfcn.h>
#    include <rocprofiler-sdk-rocpd/rocpd.h>
#    include <rocprofiler-sdk-rocpd/types.h>
#    include <rocprofiler-sdk-rocpd/version.h>

#    ifndef ROCPROSYS_CALCULATE_VERSION
#        define ROCPROSYS_CALCULATE_VERSION(MAJOR, MINOR, PATCH)                         \
            ((100 * 100 * MAJOR) + (100 * MINOR) + (PATCH))
#    endif

#    define ROCPROFSYS_USE_ROCPD_SCHEMA_VERSION 3, 0, 1

#    define ROCPROFSYS_ROCPD_COMPILE_VERSION ROCPD_VERSION
#    define ROCPROFSYS_ROCPD_NEW_API_VERSION ROCPROSYS_CALCULATE_VERSION(1, 3, 2)
#else
#    include "core/rocpd/data_storage/schema/data_views.hpp"
#    include "core/rocpd/data_storage/schema/marker_views.hpp"
#    include "core/rocpd/data_storage/schema/rocpd_tables.hpp"
#    include "core/rocpd/data_storage/schema/rocpd_views.hpp"
#    include "core/rocpd/data_storage/schema/summary_views.hpp"

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
    auto _db_dirname = tim::filepath::dirname(db_file);
    if(!tim::filepath::direxists(_db_dirname))
    {
        tim::filepath::makedir(_db_dirname);
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

#if defined(ROCPROFSYS_USE_ROCPD_LIBRARY) && ROCPROFSYS_USE_ROCPD_LIBRARY > 0
#    if ROCPROFSYS_ROCPD_COMPILE_VERSION >= ROCPROFSYS_ROCPD_NEW_API_VERSION
// New API (>= 1.3.2): callback includes schema_version parameter
void
load_schema_cb(rocpd_sql_engine_t, rocpd_sql_schema_kind_t, rocpd_sql_options_t,
               rocpd_version_triplet_t, const rocpd_sql_schema_jinja_variables_t*,
               const char*, const char* schema_content, void* user_data)
{
    if(user_data == nullptr || schema_content == nullptr)
    {
        LOG_WARNING("Invalid user data or schema content pointer");
        return;
    }
    auto* query = static_cast<std::string*>(user_data);
    if(query == nullptr)
    {
        LOG_WARNING("Invalid query pointer");
        return;
    }
    *query = std::string(schema_content);
}

// Old API (< 1.3.2) callback typedef - no schema_version parameter.
// Used at runtime when the loaded library is older than the headers we compiled against.
using rocpd_sql_load_schema_cb_v1_t = void (*)(rocpd_sql_engine_t,
                                               rocpd_sql_schema_kind_t,
                                               rocpd_sql_options_t,
                                               const rocpd_sql_schema_jinja_variables_t*,
                                               const char*, const char*, void*);

// Old API (< 1.3.2) function pointer typedef - 8 params, no schema_version.
using rocpd_sql_load_schema_fn_v1_t =
    rocpd_status_t (*)(rocpd_sql_engine_t, rocpd_sql_schema_kind_t, rocpd_sql_options_t,
                       const rocpd_sql_schema_jinja_variables_t*,
                       rocpd_sql_load_schema_cb_v1_t, const char**, std::uint64_t, void*);
#    endif

// Legacy API (< 1.3.2): callback does not have schema_version parameter
void
load_schema_cb_legacy(rocpd_sql_engine_t, rocpd_sql_schema_kind_t, rocpd_sql_options_t,
                      const rocpd_sql_schema_jinja_variables_t*, const char*,
                      const char* schema_content, void* user_data)
{
    if(user_data == nullptr || schema_content == nullptr)
    {
        LOG_WARNING("Invalid user data or schema content pointer");
        return;
    }
    auto* query = static_cast<std::string*>(user_data);
    if(query == nullptr)
    {
        LOG_WARNING("Invalid query pointer");
        return;
    }
    *query = std::string(schema_content);
}
#endif

std::string
get_schema_query(rocpd_sql_schema_kind_t schema_kind, const std::string& upid)
{
#if defined(ROCPROFSYS_USE_ROCPD_LIBRARY) && ROCPROFSYS_USE_ROCPD_LIBRARY > 0
    std::uint32_t rt_major = 0, rt_minor = 0, rt_patch = 0;
    rocpd_get_version(&rt_major, &rt_minor, &rt_patch);
    const std::uint32_t runtime_version =
        ROCPROSYS_CALCULATE_VERSION(rt_major, rt_minor, rt_patch);

    if(runtime_version != ROCPROFSYS_ROCPD_COMPILE_VERSION)
    {
        LOG_WARNING("rocpd compile-time version {} differs from runtime version {}",
                    ROCPROFSYS_ROCPD_COMPILE_VERSION, runtime_version);
    }

    const rocpd_sql_schema_jinja_variables_t info{ 2 * upid.size(), upid.c_str(),
                                                   upid.c_str() };
    rocpd_status_t                           status = ROCPD_STATUS_ERROR;
    std::string                              query;

#    if ROCPROFSYS_ROCPD_COMPILE_VERSION >= ROCPROFSYS_ROCPD_NEW_API_VERSION
    if(runtime_version >= ROCPROFSYS_ROCPD_NEW_API_VERSION)
    {
        // fixed to the schema version rocprof-sys supports
        rocpd_version_triplet_t schema_version{ ROCPROFSYS_USE_ROCPD_SCHEMA_VERSION };
        status = rocpd_sql_load_schema(ROCPD_SQL_ENGINE_SQLITE3, schema_kind,
                                       ROCPD_SQL_OPTIONS_NONE, schema_version, &info,
                                       load_schema_cb, nullptr, 0, &query);
    }
    else
    {
        // Runtime library is older than the headers we compiled against.
        // Bypass the link-time new-ABI symbol and call the old 8-param signature via
        // dlsym so there is no ABI mismatch.
        void* sym            = dlsym(RTLD_DEFAULT, "rocpd_sql_load_schema");
        auto  load_schema_v1 = reinterpret_cast<rocpd_sql_load_schema_fn_v1_t>(sym);
        if(load_schema_v1)
        {
            status = load_schema_v1(ROCPD_SQL_ENGINE_SQLITE3, schema_kind,
                                    ROCPD_SQL_OPTIONS_NONE, &info, load_schema_cb_legacy,
                                    nullptr, 0, &query);
        }
        else
        {
            LOG_WARNING("rocpd runtime {} < {}; schema load unavailable", runtime_version,
                        ROCPROFSYS_ROCPD_NEW_API_VERSION);
        }
    }
#    else
    status = rocpd_sql_load_schema(ROCPD_SQL_ENGINE_SQLITE3, schema_kind,
                                   ROCPD_SQL_OPTIONS_NONE, &info, load_schema_cb_legacy,
                                   nullptr, 0, &query);
#    endif

    if(status != ROCPD_STATUS_SUCCESS)
    {
        LOG_WARNING("Unable to load rocpd schema. Error code: {0:X}",
                    static_cast<int>(status));
    }
    return query;
#else
    std::string_view schema_content;

    switch(schema_kind)
    {
        case ROCPD_SQL_SCHEMA_ROCPD_TABLES:
            schema_content = rocprofsys::rocpd::data_storage::schema::ROCPD_TABLES_SQL;
            break;
        case ROCPD_SQL_SCHEMA_ROCPD_VIEWS:
            schema_content = rocprofsys::rocpd::data_storage::schema::ROCPD_VIEWS_SQL;
            break;
        case ROCPD_SQL_SCHEMA_ROCPD_DATA_VIEWS:
            schema_content = rocprofsys::rocpd::data_storage::schema::DATA_VIEWS_SQL;
            break;
        case ROCPD_SQL_SCHEMA_ROCPD_MARKER_VIEWS:
            schema_content = rocprofsys::rocpd::data_storage::schema::MARKER_VIEWS_SQL;
            break;
        case ROCPD_SQL_SCHEMA_ROCPD_SUMMARY_VIEWS:
            schema_content = rocprofsys::rocpd::data_storage::schema::SUMMARY_VIEWS_SQL;
            break;
        default:
            LOG_WARNING("Unknown schema kind: {}", static_cast<int>(schema_kind));
            return "";
    }

    return process_schema_template(schema_content, upid);
#endif
}

}  // namespace

namespace rocprofsys
{
namespace rocpd
{
namespace data_storage
{
database::database(int pid, int ppid, std::string output_path)
: m_path(std::move(output_path))
{
    create_directory_for_database_file(m_path);
    LOG_INFO("Database: {}", m_path);

    validate_sqlite3_result(sqlite3_open(":memory:", &_sqlite3_db_temp), "",
                            "database open failed!");
    validate_sqlite3_result(sqlite3_open(m_path.c_str(), &_sqlite3_db), "",
                            "database open failed!");
    m_upid = generate_upid(pid, ppid);
}

database::~database()
{
    sqlite3_close(_sqlite3_db_temp);
    sqlite3_close(_sqlite3_db);
}

void
database::initialize_schema()
{
    const auto upid = get_upid();

// This #IF can be removed now that MARKER_VIEWS is aliased to METADATA.
// Kept it for clarity
#if defined(ROCPROFSYS_USE_ROCPD_LIBRARY) && ROCPROFSYS_USE_ROCPD_LIBRARY > 0 &&         \
    ROCPROFSYS_ROCPD_COMPILE_VERSION >= ROCPROFSYS_ROCPD_NEW_API_VERSION
    const std::vector<rocpd_sql_schema_kind_t> schema_kinds = {
        ROCPD_SQL_SCHEMA_ROCPD_TABLES, ROCPD_SQL_SCHEMA_ROCPD_VIEWS,
        ROCPD_SQL_SCHEMA_ROCPD_DATA_VIEWS, ROCPD_SQL_SCHEMA_ROCPD_METADATA,
        ROCPD_SQL_SCHEMA_ROCPD_SUMMARY_VIEWS
    };
#else
    const std::vector<rocpd_sql_schema_kind_t> schema_kinds = {
        ROCPD_SQL_SCHEMA_ROCPD_TABLES, ROCPD_SQL_SCHEMA_ROCPD_VIEWS,
        ROCPD_SQL_SCHEMA_ROCPD_DATA_VIEWS, ROCPD_SQL_SCHEMA_ROCPD_MARKER_VIEWS,
        ROCPD_SQL_SCHEMA_ROCPD_SUMMARY_VIEWS
    };
#endif

    for(const auto& schema_kind : schema_kinds)
    {
        const std::string query = get_schema_query(schema_kind, upid);

        if(query.empty())
        {
            LOG_WARNING("Failed to get schema query for schema kind: {0:X}",
                        static_cast<int>(schema_kind));
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
}

}  // namespace data_storage
}  // namespace rocpd
}  // namespace rocprofsys
