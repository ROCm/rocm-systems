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

#include <rocstorage/database.hpp>
#include <rocstorage/reader.hpp>
#include <rocstorage/trace.hpp>

#include "error.hpp"
#include "reader/database/rocprofvis_db.h"
#include "reader/database/rocprofvis_db_future.h"
#include "reader/database/rocprofvis_db_rocpd.h"
#include "reader/database/rocprofvis_db_rocprof.h"
#include "reader/datamodel/rocprofvis_dm_trace.h"

#include <fstream>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace rocstorage {

using namespace RocProfVis::DataModel;

// Helper to get typed internal pointer
static inline Database* db_internal(rocprofvis_dm_database_t handle) {
  return static_cast<Database*>(handle);
}

namespace {
// SQLite file header
constexpr std::string_view kSqliteFileHeader = "SQLite format 3";

// Check if file exists and has valid SQLite header
bool is_valid_sqlite_file(const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    return false;
  }
  char header[16] = {0};
  file.read(header, 16);
  return file.gcount() == 16 &&
         std::string_view(header, kSqliteFileHeader.size()) == kSqliteFileHeader;
}
} // namespace

// Impl holds state for async operations (Future, callback)
struct database::Impl {
  std::unique_ptr<Future> future_;
  progress_callback callback_;

  std::unique_ptr<Future> allocate_future(progress_callback callback) {
    callback_ = std::move(callback);

    if (callback_) {
      return std::make_unique<Future>(
          [](rocprofvis_db_filename_t filename,
             rocprofvis_db_progress_percent_t percent,
             rocprofvis_db_status_t status,
             rocprofvis_db_status_message_t message, void *user_data) {
            auto *self = static_cast<Impl *>(user_data);
            if (self->callback_) {
              self->callback_(filename, percent, status == kRPVDbSuccess,
                              message);
            }
          },
          this);
    }

    return std::make_unique<Future>(nullptr, nullptr);
  }
};

// Private default constructor
database::database() : impl_(std::make_unique<Impl>()), internal_(nullptr), owning_(false) {}

// Non-owning view constructor
database::database(rocprofvis_dm_database_t internal)
    : impl_(std::make_unique<Impl>()), internal_(internal), owning_(false) {}

database::~database() {
  if (owning_ && internal_) {
    delete db_internal(internal_);
  }
}

database::database(database &&other) noexcept
    : impl_(std::move(other.impl_)),
      internal_(other.internal_),
      owning_(other.owning_) {
  other.internal_ = nullptr;
  other.owning_ = false;
}

database &database::operator=(database &&other) noexcept {
  if (this != &other) {
    if (owning_ && internal_) {
      delete db_internal(internal_);
    }
    impl_ = std::move(other.impl_);
    internal_ = other.internal_;
    owning_ = other.owning_;
    other.internal_ = nullptr;
    other.owning_ = false;
  }
  return *this;
}

result<std::unique_ptr<database>> database::open(const std::string &path,
                                                  database_type type) {
  if (path.empty()) {
    return error(error_code::invalid_parameter, "Empty database path");
  }

  if (!is_valid_sqlite_file(path)) {
    return error(error_code::file_not_found,
                 "File does not exist or is not a valid SQLite database: " +
                     path);
  }

  // Determine database type
  auto db_type = static_cast<rocprofvis_db_type_t>(type);
  if (db_type == kAutodetect) {
    db_type = Database::Autodetect(path.c_str());
    if (db_type == kAutodetect) {
      db_type = kRocprofSqlite;
    }
  }

  // Create internal database
  Database* internal_db = nullptr;
  try {
    if (db_type == kRocpdSqlite) {
      internal_db = new RocpdDatabase(path.c_str());
    } else if (db_type == kRocprofSqlite) {
      internal_db = new RocprofDatabase(path.c_str());
    } else {
      return error(error_code::invalid_parameter, "Unsupported database type");
    }

    if (internal_db->Open() != kRocProfVisDmResultSuccess) {
      delete internal_db;
      return error(error_code::db_access_failed, "Failed to open database: " + path);
    }
  } catch (const std::exception &e) {
    delete internal_db;
    return error(error_code::db_access_failed,
                 std::string("Failed to open database: ") + e.what());
  }

  // Create owning wrapper
  auto db = std::unique_ptr<database>(new database());
  db->internal_ = internal_db;
  db->owning_ = true;
  return std::move(db);
}

status database::bind(trace &t) {
  if (!internal_) {
    return error(error_code::invalid_parameter, "Database not initialized");
  }

  // Get internal trace handle
  auto *internal_trace = static_cast<Trace *>(t.c_handle());
  if (!internal_trace) {
    return error(error_code::invalid_parameter, "Invalid trace");
  }

  // Bind trace to database
  rocprofvis_dm_db_bind_struct *bind_data = nullptr;
  auto result = internal_trace->BindDatabase(internal_, bind_data);
  if (result != kRocProfVisDmResultSuccess) {
    return from_c_result(result, "Failed to bind trace to database");
  }

  // Tell database about the trace binding
  result = db_internal(internal_)->BindTrace(bind_data);
  if (result != kRocProfVisDmResultSuccess) {
    return from_c_result(result, "Failed to bind database to trace");
  }

  return ok();
}

bool database::read_metadata_async(progress_callback callback) {
  if (!internal_) return false;

  impl_->future_ = impl_->allocate_future(std::move(callback));
  auto result = db_internal(internal_)->ReadTraceMetadataAsync(impl_->future_.get());
  return result == kRocProfVisDmResultSuccess;
}

status database::try_read_metadata_async(progress_callback callback) {
  if (!internal_) {
    return error(error_code::invalid_parameter, "Database not initialized");
  }

  impl_->future_ = impl_->allocate_future(std::move(callback));
  auto c_result = db_internal(internal_)->ReadTraceMetadataAsync(impl_->future_.get());
  if (c_result != kRocProfVisDmResultSuccess) {
    return from_c_result(c_result, "Failed to start metadata read");
  }

  return ok();
}

bool database::read_slice_async(uint64_t start, uint64_t end,
                                const std::vector<uint32_t> &track_ids,
                                progress_callback callback) {
  if (!internal_ || track_ids.empty()) return false;
  if (track_ids.size() > std::numeric_limits<uint16_t>::max()) return false;

  impl_->future_ = impl_->allocate_future(std::move(callback));
  auto result = db_internal(internal_)->ReadTraceSliceAsync(
      start, end, static_cast<uint16_t>(track_ids.size()),
      const_cast<uint32_t *>(track_ids.data()), impl_->future_.get());

  return result == kRocProfVisDmResultSuccess;
}

status database::try_read_slice_async(uint64_t start, uint64_t end,
                                      const std::vector<uint32_t> &track_ids,
                                      progress_callback callback) {
  if (!internal_) {
    return error(error_code::invalid_parameter, "Database not initialized");
  }
  if (track_ids.empty()) {
    return error(error_code::invalid_parameter, "No track IDs specified");
  }
  if (track_ids.size() > std::numeric_limits<uint16_t>::max()) {
    return error(error_code::invalid_parameter,
                 "Too many track IDs specified (max " +
                     std::to_string(std::numeric_limits<uint16_t>::max()) + ")");
  }

  impl_->future_ = impl_->allocate_future(std::move(callback));
  auto c_result = db_internal(internal_)->ReadTraceSliceAsync(
      start, end, static_cast<uint16_t>(track_ids.size()),
      const_cast<uint32_t *>(track_ids.data()), impl_->future_.get());

  if (c_result != kRocProfVisDmResultSuccess) {
    return from_c_result(c_result, "Failed to start slice read");
  }

  return ok();
}

bool database::read_event_property_async(event_property_type type, uint64_t id,
                                         progress_callback callback) {
  if (!internal_) return false;

  impl_->future_ = impl_->allocate_future(std::move(callback));

  rocprofvis_dm_event_id_t c_id;
  c_id.value = id;

  auto result = db_internal(internal_)->ReadEventPropertyAsync(
      static_cast<rocprofvis_dm_event_property_type_t>(type), c_id,
      impl_->future_.get());

  return result == kRocProfVisDmResultSuccess;
}

uint64_t database::execute_query_async(const std::string &query,
                                       const std::string &description,
                                       progress_callback callback) {
  if (!internal_ || query.empty()) return 0;

  impl_->future_ = impl_->allocate_future(std::move(callback));

  uint64_t table_id = 0;
  auto result = db_internal(internal_)->ExecuteQueryAsync(
      query.c_str(), description.empty() ? "" : description.c_str(),
      impl_->future_.get(), &table_id);

  if (result != kRocProfVisDmResultSuccess) {
    return 0;
  }

  return table_id;
}

bool database::wait(uint64_t timeout_sec) {
  if (!impl_ || !impl_->future_) return false;

  uint64_t timeout_ms = (timeout_sec == 0) ? UINT64_MAX : timeout_sec * 1000;
  auto result = impl_->future_->WaitForCompletion(timeout_ms);
  return result == kRocProfVisDmResultSuccess;
}

status database::try_wait(uint64_t timeout_sec) {
  if (!impl_) {
    return error(error_code::invalid_parameter, "Database not initialized");
  }
  if (!impl_->future_) {
    return error(error_code::invalid_parameter, "No pending operation");
  }

  uint64_t timeout_ms = (timeout_sec == 0) ? UINT64_MAX : timeout_sec * 1000;
  auto c_result = impl_->future_->WaitForCompletion(timeout_ms);
  if (c_result != kRocProfVisDmResultSuccess) {
    return from_c_result(c_result, "Operation failed or timed out");
  }

  return ok();
}

void database::cancel() {
  if (impl_ && impl_->future_) {
    impl_->future_->SetInterrupted();
  }
}

std::string database::build_table_query(const table_query_options &options) const {
  if (!internal_) return "";

  std::vector<const char *> filter_ptrs;
  for (const auto &s : options.string_table_filters) {
    filter_ptrs.push_back(s.c_str());
  }

  std::string query;
  auto result = db_internal(internal_)->BuildTableQuery(
      options.start_time, options.end_time,
      static_cast<uint16_t>(options.track_ids.size()),
      const_cast<uint32_t *>(options.track_ids.data()),
      options.where_clause.empty() ? nullptr : options.where_clause.c_str(),
      options.filter.empty() ? nullptr : options.filter.c_str(),
      options.group_by.empty() ? nullptr : options.group_by.c_str(),
      options.group_columns.empty() ? nullptr : options.group_columns.c_str(),
      options.sort_column.empty() ? nullptr : options.sort_column.c_str(),
      static_cast<rocprofvis_dm_sort_order_t>(options.order),
      static_cast<uint16_t>(filter_ptrs.size()),
      filter_ptrs.empty() ? nullptr : filter_ptrs.data(), options.max_count,
      options.offset, options.count_only, options.summary, query);

  if (result != kRocProfVisDmResultSuccess) {
    return "";
  }

  return query;
}

bool database::export_to_csv_async(const std::string &query,
                                   const std::string &file_path,
                                   progress_callback callback) {
  if (!internal_ || query.empty() || file_path.empty()) return false;

  impl_->future_ = impl_->allocate_future(std::move(callback));

  auto result = db_internal(internal_)->ExportTableCSVAsync(
      query.c_str(), file_path.c_str(), impl_->future_.get());

  return result == kRocProfVisDmResultSuccess;
}

bool database::save_trimmed_async(uint64_t start, uint64_t end,
                                  const std::string &new_path,
                                  progress_callback callback) {
  if (!internal_ || new_path.empty()) return false;

  impl_->future_ = impl_->allocate_future(std::move(callback));

  auto result = db_internal(internal_)->SaveTrimmedDataAsync(
      start, end, new_path.c_str(), impl_->future_.get());

  return result == kRocProfVisDmResultSuccess;
}

uint64_t database::memory_footprint() const {
  if (!internal_) return 0;
  return db_internal(internal_)->GetMemoryFootprint();
}

rocprofvis_dm_database_t database::c_handle() const {
  return internal_;
}

} // namespace rocstorage
