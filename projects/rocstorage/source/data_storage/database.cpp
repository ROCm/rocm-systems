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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "database.hpp"

#include "directory.hpp"
#include "uuid_validation.h"

#include <regex>
#include <string>

#if defined(USE_SCHEMA_FROM_ROCPROFILER_SDK_ROCPD) &&                          \
    USE_SCHEMA_FROM_ROCPROFILER_SDK_ROCPD > 0
#include <rocprofiler-sdk-rocpd/rocpd.h>
#include <rocprofiler-sdk-rocpd/types.h>
#else
#include "schema/data_views.hpp"
#include "schema/marker_views.hpp"
#include "schema/rocpd_tables.hpp"
#include "schema/rocpd_views.hpp"
#include "schema/summary_views.hpp"

namespace {
enum rocpd_sql_schema_kind_t {
  ROCPD_SQL_SCHEMA_NONE = 0,
  ROCPD_SQL_SCHEMA_ROCPD_TABLES,
  ROCPD_SQL_SCHEMA_ROCPD_INDEXES,
  ROCPD_SQL_SCHEMA_ROCPD_VIEWS,
  ROCPD_SQL_SCHEMA_ROCPD_DATA_VIEWS,
  ROCPD_SQL_SCHEMA_ROCPD_SUMMARY_VIEWS,
  ROCPD_SQL_SCHEMA_ROCPD_MARKER_VIEWS,
  ROCPD_SQL_SCHEMA_LAST,
};
} // namespace

#endif

namespace {
void create_directory_for_database_file(const std::string &db_file) {
  auto _db_dirname = rocstorage::common::dirname(db_file);
  if (!rocstorage::common::direxists(_db_dirname)) {
    rocstorage::common::makedir(_db_dirname);
  }
}
#if !defined(USE_SCHEMA_FROM_ROCPROFILER_SDK_ROCPD)
std::string process_schema_template(std::string_view schema_content,
                                    const std::string &upid) {
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

#if defined(USE_SCHEMA_FROM_ROCPROFILER_SDK_ROCPD) &&                          \
    USE_SCHEMA_FROM_ROCPROFILER_SDK_ROCPD > 0
void load_schema_cb(rocpd_sql_engine_t, rocpd_sql_schema_kind_t,
                    rocpd_sql_options_t,
                    const rocpd_sql_schema_jinja_variables_t *, const char *,
                    const char *schema_content, void *user_data) {
  if (user_data == nullptr || schema_content == nullptr) {
    spdlog::error("Invalid user data or schema content pointer");
    return;
  }
  auto *query = static_cast<std::string *>(user_data);
  if (query == nullptr) {
    spdlog::error("Invalid query pointer");
    return;
  }
  *query = std::string(schema_content);
}
#endif

std::string get_schema_query(rocpd_sql_schema_kind_t schema_kind,
                             const std::string &uuid) {
#if defined(USE_SCHEMA_FROM_ROCPROFILER_SDK_ROCPD) &&                          \
    USE_SCHEMA_FROM_ROCPROFILER_SDK_ROCPD > 0
  const auto jinja_size = 2 * uuid.size();
  rocpd_sql_schema_jinja_variables_t info{jinja_size, uuid.c_str(),
                                          uuid.c_str()};

  std::string query;
  auto status = rocpd_sql_load_schema(ROCPD_SQL_ENGINE_SQLITE3, schema_kind,
                                      ROCPD_SQL_OPTIONS_NONE, &info,
                                      load_schema_cb, nullptr, 0, &query);
  if (status != ROCPD_STATUS_SUCCESS) {
    spdlog::error("Unable to load rocpd schema (error code: {})",
                  static_cast<int>(status));
  }
  return query;
#else
  std::string_view schema_content;

  switch (schema_kind) {
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

} // namespace

namespace rocstorage {
namespace data_storage {
database::database(std::string db_path, std::string uuid, database_mode mode)
    : m_db_path{std::move(db_path)}, m_uuid{std::move(uuid)}, m_mode{mode} {

  if (m_mode == database_mode::read_only) {
    // Read-only mode: open existing database for reading
    spdlog::info("rocstorage database opened in read-only mode (path: {})", m_db_path);
    validate_sqlite3_result(
        sqlite3_open_v2(m_db_path.c_str(), &m_sqlite3_db, SQLITE_OPEN_READONLY, nullptr),
        "", "database open failed!");
    // Mark as initialized to skip initialize_schema() - existing database has schema
    m_initialized = true;
  } else {
    // Validate UUID to ensure compatibility with rocstorage-reader's GUID extraction
    if (!validation::is_valid_uuid(m_uuid)) {
      throw std::invalid_argument(
          "Invalid UUID: " + validation::get_uuid_validation_error(m_uuid) +
          "\nUUIDs must contain only alphanumeric characters to ensure "
          "compatibility with rocstorage-reader.");
    }

    create_directory_for_database_file(m_db_path);

    if (m_mode == database_mode::wal) {
      // WAL mode: open file directly for concurrent read/write access
      spdlog::info("rocstorage database initialized in WAL mode (uuid: {}, path: {})",
                   m_uuid, m_db_path);
      validate_sqlite3_result(sqlite3_open(m_db_path.c_str(), &m_sqlite3_db), "",
                              "database open failed!");
      // Enable WAL mode for concurrent access
      validate_sqlite3_result(
          sqlite3_exec(m_sqlite3_db, "PRAGMA journal_mode=WAL;", nullptr, nullptr, nullptr),
          "PRAGMA journal_mode=WAL", "Failed to enable WAL mode!");
      // Use NORMAL synchronous mode for better performance with reasonable safety
      validate_sqlite3_result(
          sqlite3_exec(m_sqlite3_db, "PRAGMA synchronous=NORMAL;", nullptr, nullptr, nullptr),
          "PRAGMA synchronous=NORMAL", "Failed to set synchronous mode!");
    } else {
      // In-memory mode: write to memory, flush to disk at end (current behavior)
      spdlog::info("rocstorage database initialized (uuid: {}, path: {})", m_uuid,
                   m_db_path);
      validate_sqlite3_result(sqlite3_open(":memory:", &m_sqlite3_db), "",
                              "database open failed!");
    }
  }
}

database::~database() { sqlite3_close(m_sqlite3_db); }

std::string database::get_uuid() const { return m_uuid; }

std::string database::get_path() const { return m_db_path; }

database_mode database::get_mode() const { return m_mode; }

result<std::unique_ptr<database>> database::open_readonly(const std::string &path) {
  // Check if file exists and is a valid SQLite database
  sqlite3 *test_db;
  int rc = sqlite3_open_v2(path.c_str(), &test_db,
                           SQLITE_OPEN_READONLY, nullptr);
  if (rc != SQLITE_OK) {
    std::string err_msg = sqlite3_errmsg(test_db);
    sqlite3_close(test_db);
    return error(error_code::file_not_found,
                 "Failed to open database: " + err_msg, path, rc);
  }
  sqlite3_close(test_db);

  // Create database instance with read_only mode
  // Use empty UUID since we're opening an existing database
  try {
    auto db = std::unique_ptr<database>(new database(path, "", database_mode::read_only));
    return db;
  } catch (const std::exception &e) {
    return error(error_code::db_access_failed,
                 std::string("Failed to open database: ") + e.what(), path);
  }
}

status database::execute_query(const std::string &query, row_callback callback) {
  sqlite3_stmt *stmt = nullptr;
  int rc = sqlite3_prepare_v2(m_sqlite3_db, query.c_str(), -1, &stmt, nullptr);

  if (rc != SQLITE_OK) {
    return error(error_code::query_error,
                 std::string("Failed to prepare query: ") + sqlite3_errmsg(m_sqlite3_db),
                 query, rc);
  }

  auto cleanup = [](sqlite3_stmt *s) { sqlite3_finalize(s); };
  std::unique_ptr<sqlite3_stmt, decltype(cleanup)> stmt_guard(stmt, cleanup);

  while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
    query_row row(stmt);
    if (!callback(row)) {
      break; // Callback requested stop
    }
  }

  if (rc != SQLITE_DONE && rc != SQLITE_ROW) {
    return error(error_code::query_error,
                 std::string("Query execution failed: ") + sqlite3_errmsg(m_sqlite3_db),
                 query, rc);
  }

  return ok();
}

void database::initialize_schema() {
  if (m_initialized) {
    throw std::runtime_error("Database already initialized!");
  }

  const std::vector<rocpd_sql_schema_kind_t> schema_kinds = {
      ROCPD_SQL_SCHEMA_ROCPD_TABLES, ROCPD_SQL_SCHEMA_ROCPD_VIEWS,
      ROCPD_SQL_SCHEMA_ROCPD_DATA_VIEWS, ROCPD_SQL_SCHEMA_ROCPD_MARKER_VIEWS,
      ROCPD_SQL_SCHEMA_ROCPD_SUMMARY_VIEWS};

  for (const auto &schema_kind : schema_kinds) {
    const std::string query = get_schema_query(schema_kind, m_uuid);

    if (query.empty()) {
      spdlog::error("Failed to get schema query for schema kind: {}",
                    static_cast<int>(schema_kind));
      continue;
    }

    validate_sqlite3_result(
        sqlite3_exec(m_sqlite3_db, query.c_str(), 0, 0, 0), query.c_str(),
        std::string("Invalid schema, init database failed!"));
  }

  m_initialized = true;
}

void database::execute_query(const std::string &query) {
  validate_sqlite3_result(
      sqlite3_exec(m_sqlite3_db, query.c_str(), 0, 0, 0),
      "Failed to execute query - ", query);
}

size_t database::get_last_insert_id() const {
  return sqlite3_last_insert_rowid(m_sqlite3_db);
}

void database::flush() {
  if (m_flushed) {
    throw std::runtime_error("Database already flushed!");
  }

  if (m_mode == database_mode::wal) {
    // WAL mode: data is already on disk, just checkpoint to ensure durability
    sqlite3_wal_checkpoint_v2(m_sqlite3_db, nullptr, SQLITE_CHECKPOINT_FULL,
                              nullptr, nullptr);
  } else {
    // In-memory mode: backup to file
    sqlite3 *out_db;
    validate_sqlite3_result(sqlite3_open(m_db_path.c_str(), &out_db), "",
                            "database open failed!");
    auto *backup = sqlite3_backup_init(out_db, "main", m_sqlite3_db, "main");
    if (backup) {
      sqlite3_backup_step(backup, -1);
      sqlite3_backup_finish(backup);
    }
    sqlite3_close(out_db);
  }
  m_flushed = true;
}

} // namespace data_storage
} // namespace rocstorage
