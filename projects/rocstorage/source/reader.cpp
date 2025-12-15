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

#include <rocstorage/reader.hpp>

#include "reader/datamodel/internal_types.h"
#include "reader/datamodel/rocprofvis_dm_trace.h"
#include "reader/database/rocprofvis_db.h"
#include "reader/database/rocprofvis_db_future.h"
#include "reader/database/rocprofvis_db_rocpd.h"
#include "reader/database/rocprofvis_db_rocprof.h"

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string_view>

namespace rocstorage {

using namespace RocProfVis::DataModel;

namespace {
// Helper to safely convert nullable C strings to std::string
inline std::string safe_string(const char* str) {
  return str ? std::string(str) : std::string();
}

// Default timeout for synchronous operations (30 seconds)
constexpr uint64_t kDefaultTimeoutMs = 30000;

// SQLite file header
constexpr std::string_view kSqliteFileHeader = "SQLite format 3";

// Check if file exists and has valid SQLite header
bool is_valid_sqlite_file(const std::string& path) {
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

// ======================== track implementation ========================

struct track::Impl {
  explicit Impl(Track *handle) : handle_(handle) {
    if (!handle_) {
      throw std::invalid_argument("Invalid track handle");
    }
  }

  Track *handle_;
};

track::track(void *handle)
    : impl_(std::make_unique<Impl>(static_cast<Track *>(handle))) {}

track::~track() = default;

track::track(track &&) noexcept = default;
track &track::operator=(track &&) noexcept = default;

uint32_t track::id() const { return impl_->handle_->TrackId(); }

track_category track::category() const {
  return static_cast<track_category>(impl_->handle_->Category());
}

std::string track::category_string() const {
  return safe_string(impl_->handle_->CategoryString());
}

uint64_t track::node_id() const { return impl_->handle_->NodeId(); }

std::string track::process_name() const {
  return safe_string(impl_->handle_->Process());
}

std::string track::subprocess_name() const {
  return safe_string(impl_->handle_->SubProcess());
}

uint64_t track::num_records() const { return impl_->handle_->NumRecords(); }

uint64_t track::min_timestamp() const {
  return impl_->handle_->MinTimestamp();
}

uint64_t track::max_timestamp() const {
  return impl_->handle_->MaxTimestamp();
}

uint64_t track::num_slices() const {
  return impl_->handle_->NumberOfSlices();
}

// ======================== reader implementation ========================

struct reader::Impl {
  explicit Impl(const std::string &path, database_type type)
      : trace_(std::make_unique<Trace>()), database_(nullptr),
        future_(nullptr) {
    if (path.empty()) {
      throw std::runtime_error("Empty database path");
    }

    // SQLite creates the file if it doesn't exist, which we don't want
    if (!is_valid_sqlite_file(path)) {
      throw std::runtime_error("File does not exist or is not a valid SQLite database: " + path);
    }

    // Determine database type
    auto db_type = static_cast<rocprofvis_db_type_t>(type);
    if (db_type == kAutodetect) {
      db_type = Database::Autodetect(path.c_str());
      // If autodetect failed, try rocprof format as fallback
      if (db_type == kAutodetect) {
        db_type = kRocprofSqlite;
      }
    }

    // Open database
    if (db_type == kRocpdSqlite) {
      database_ = std::make_unique<RocpdDatabase>(path.c_str());
    } else if (db_type == kRocprofSqlite) {
      database_ = std::make_unique<RocprofDatabase>(path.c_str());
    } else {
      throw std::runtime_error("Unsupported database type");
    }

    if (database_->Open() != kRocProfVisDmResultSuccess) {
      throw std::runtime_error("Failed to open database: " + path);
    }

    // Bind trace to database
    rocprofvis_dm_db_bind_struct *bind_data = nullptr;
    auto result = trace_->BindDatabase(database_.get(), bind_data);
    if (result != kRocProfVisDmResultSuccess) {
      throw std::runtime_error("Failed to bind trace to database");
    }

    // Tell database about the trace binding (so BindObject() works)
    result = database_->BindTrace(bind_data);
    if (result != kRocProfVisDmResultSuccess) {
      throw std::runtime_error("Failed to bind database to trace");
    }
  }

  bool is_ready() const { return trace_ && database_; }

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

  std::unique_ptr<Trace> trace_;
  std::unique_ptr<Database> database_;
  std::unique_ptr<Future> future_;
  progress_callback callback_;
};

std::unique_ptr<reader> reader::open(const std::string &path,
                                     database_type type) {
  try {
    auto r = std::unique_ptr<reader>(new reader());
    r->impl_ = std::make_unique<Impl>(path, type);
    return r;
  } catch (const std::exception &) {
    return nullptr;
  }
}

reader::~reader() = default;

reader::reader(reader &&) noexcept = default;
reader &reader::operator=(reader &&) noexcept = default;

bool reader::read_metadata() {
  if (!impl_ || !impl_->is_ready()) {
    return false;
  }

  impl_->future_ = impl_->allocate_future(nullptr);
  auto result =
      impl_->database_->ReadTraceMetadataAsync(impl_->future_.get());
  if (result != kRocProfVisDmResultSuccess) {
    return false;
  }

  result = impl_->future_->WaitForCompletion(kDefaultTimeoutMs);
  return result == kRocProfVisDmResultSuccess;
}

bool reader::read_metadata_async(progress_callback callback) {
  if (!impl_ || !impl_->is_ready()) {
    return false;
  }

  impl_->future_ = impl_->allocate_future(std::move(callback));
  auto result =
      impl_->database_->ReadTraceMetadataAsync(impl_->future_.get());
  return result == kRocProfVisDmResultSuccess;
}

bool reader::wait(uint64_t timeout_sec) {
  if (!impl_ || !impl_->future_) {
    return false;
  }

  // Convert seconds to milliseconds (WaitForCompletion uses ms)
  // 0 = wait indefinitely
  uint64_t timeout_ms = (timeout_sec == 0) ? UINT64_MAX : timeout_sec * 1000;
  auto result = impl_->future_->WaitForCompletion(timeout_ms);
  return result == kRocProfVisDmResultSuccess;
}

void reader::cancel() {
  if (impl_ && impl_->future_) {
    impl_->future_->SetInterrupted();
  }
}

uint64_t reader::start_time() const {
  if (!impl_ || !impl_->trace_) {
    return 0;
  }
  return impl_->trace_->StartTime();
}

uint64_t reader::end_time() const {
  if (!impl_ || !impl_->trace_) {
    return 0;
  }
  return impl_->trace_->EndTime();
}

uint64_t reader::num_tracks() const {
  if (!impl_ || !impl_->trace_) {
    return 0;
  }
  return impl_->trace_->NumberOfTracks();
}

std::unique_ptr<track> reader::get_track(uint64_t index) const {
  if (!impl_ || !impl_->trace_ || index >= num_tracks()) {
    return nullptr;
  }

  rocprofvis_dm_track_t handle = nullptr;
  auto result = impl_->trace_->GetPropertyAsHandle(kRPVDMTrackHandleIndexed,
                                                     index, &handle);
  if (result != kRocProfVisDmResultSuccess || !handle) {
    return nullptr;
  }

  return std::unique_ptr<track>(new track(handle));
}

std::vector<std::unique_ptr<track>> reader::get_tracks() const {
  std::vector<std::unique_ptr<track>> tracks;
  auto count = num_tracks();
  tracks.reserve(count);

  for (uint64_t i = 0; i < count; ++i) {
    if (auto t = get_track(i)) {
      tracks.push_back(std::move(t));
    }
  }

  return tracks;
}

bool reader::read_slice(uint64_t start, uint64_t end,
                        const std::vector<uint32_t> &track_ids) {
  if (!impl_ || !impl_->is_ready() || track_ids.empty()) {
    return false;
  }

  impl_->future_ = impl_->allocate_future(nullptr);
  auto result = impl_->database_->ReadTraceSliceAsync(
      start, end, static_cast<uint16_t>(track_ids.size()),
      const_cast<uint32_t *>(track_ids.data()), impl_->future_.get());

  if (result != kRocProfVisDmResultSuccess) {
    return false;
  }

  result = impl_->future_->WaitForCompletion(kDefaultTimeoutMs);
  return result == kRocProfVisDmResultSuccess;
}

bool reader::read_slice_async(uint64_t start, uint64_t end,
                              const std::vector<uint32_t> &track_ids,
                              progress_callback callback) {
  if (!impl_ || !impl_->is_ready() || track_ids.empty()) {
    return false;
  }

  impl_->future_ = impl_->allocate_future(std::move(callback));
  auto result = impl_->database_->ReadTraceSliceAsync(
      start, end, static_cast<uint16_t>(track_ids.size()),
      const_cast<uint32_t *>(track_ids.data()), impl_->future_.get());

  return result == kRocProfVisDmResultSuccess;
}

uint64_t reader::memory_footprint() const {
  if (!impl_ || !impl_->trace_) {
    return 0;
  }
  uint64_t footprint = 0;
  impl_->trace_->GetPropertyAsUint64(kRPVDMTraceMemoryFootprintUInt64, 0,
                                       &footprint);
  return footprint;
}

rocprofvis_dm_trace_t reader::c_trace_handle() const {
  return impl_ ? impl_->trace_.get() : nullptr;
}

rocprofvis_dm_database_t reader::c_database_handle() const {
  return impl_ ? impl_->database_.get() : nullptr;
}

// ======================== Memory Management ========================

bool reader::delete_slice(uint64_t start, uint64_t end) {
  if (!impl_ || !impl_->trace_) {
    return false;
  }
  return impl_->trace_->DeleteSliceAtTimeRange(start, end) ==
         kRocProfVisDmResultSuccess;
}

bool reader::delete_slice(uint32_t track_id, void *slice_handle) {
  if (!impl_ || !impl_->trace_) {
    return false;
  }
  return impl_->trace_->DeleteSliceByHandle(track_id, slice_handle) ==
         kRocProfVisDmResultSuccess;
}

bool reader::delete_all_slices() {
  if (!impl_ || !impl_->trace_) {
    return false;
  }
  return impl_->trace_->DeleteAllSlices() == kRocProfVisDmResultSuccess;
}

bool reader::delete_all_tables() {
  if (!impl_ || !impl_->trace_) {
    return false;
  }
  return impl_->trace_->DeleteAllTables() == kRocProfVisDmResultSuccess;
}

bool reader::delete_table(uint64_t table_id) {
  if (!impl_ || !impl_->trace_) {
    return false;
  }
  return impl_->trace_->DeleteTableAt(table_id) == kRocProfVisDmResultSuccess;
}

// ======================== Event Properties ========================

bool reader::read_event_property(event_property_type type, event_id id) {
  if (!impl_ || !impl_->is_ready()) {
    return false;
  }

  impl_->future_ = impl_->allocate_future(nullptr);

  rocprofvis_dm_event_id_t c_id;
  c_id.value = id.raw();

  auto result = impl_->database_->ReadEventPropertyAsync(
      static_cast<rocprofvis_dm_event_property_type_t>(type), c_id,
      impl_->future_.get());

  if (result != kRocProfVisDmResultSuccess) {
    return false;
  }

  result = impl_->future_->WaitForCompletion(kDefaultTimeoutMs);
  return result == kRocProfVisDmResultSuccess;
}

bool reader::read_event_property_async(event_property_type type, event_id id,
                                       progress_callback callback) {
  if (!impl_ || !impl_->is_ready()) {
    return false;
  }

  impl_->future_ = impl_->allocate_future(std::move(callback));

  rocprofvis_dm_event_id_t c_id;
  c_id.value = id.raw();

  auto result = impl_->database_->ReadEventPropertyAsync(
      static_cast<rocprofvis_dm_event_property_type_t>(type), c_id,
      impl_->future_.get());

  return result == kRocProfVisDmResultSuccess;
}

std::unique_ptr<flow_trace> reader::get_flow_trace(event_id id) const {
  if (!impl_ || !impl_->trace_) {
    return nullptr;
  }

  rocprofvis_dm_handle_t handle = nullptr;
  auto result = impl_->trace_->GetPropertyAsHandle(
      kRPVDMFlowTraceHandleByEventID, id.raw(), &handle);

  if (result != kRocProfVisDmResultSuccess || !handle) {
    return nullptr;
  }

  return std::unique_ptr<flow_trace>(new flow_trace(handle));
}

std::unique_ptr<stack_trace> reader::get_stack_trace(event_id id) const {
  if (!impl_ || !impl_->trace_) {
    return nullptr;
  }

  rocprofvis_dm_handle_t handle = nullptr;
  auto result = impl_->trace_->GetPropertyAsHandle(
      kRPVDMStackTraceHandleByEventID, id.raw(), &handle);

  if (result != kRocProfVisDmResultSuccess || !handle) {
    return nullptr;
  }

  return std::unique_ptr<stack_trace>(new stack_trace(handle));
}

std::unique_ptr<ext_data> reader::get_ext_data(event_id id) const {
  if (!impl_ || !impl_->trace_) {
    return nullptr;
  }

  rocprofvis_dm_handle_t handle = nullptr;
  auto result = impl_->trace_->GetPropertyAsHandle(kRPVDMExtInfoHandleByEventID,
                                                     id.raw(), &handle);

  if (result != kRocProfVisDmResultSuccess || !handle) {
    return nullptr;
  }

  return std::unique_ptr<ext_data>(new ext_data(handle));
}

bool reader::delete_event_property(event_property_type type, event_id id) {
  if (!impl_ || !impl_->trace_) {
    return false;
  }

  rocprofvis_dm_event_id_t c_id;
  c_id.value = id.raw();

  return impl_->trace_->DeleteEventPropertyFor(
             static_cast<rocprofvis_dm_event_property_type_t>(type), c_id) ==
         kRocProfVisDmResultSuccess;
}

bool reader::delete_all_event_properties(event_property_type type) {
  if (!impl_ || !impl_->trace_) {
    return false;
  }

  return impl_->trace_->DeleteAllEventPropertiesFor(
             static_cast<rocprofvis_dm_event_property_type_t>(type)) ==
         kRocProfVisDmResultSuccess;
}

// ======================== Query Execution ========================

std::string reader::build_table_query(const table_query_options &options) const {
  if (!impl_ || !impl_->database_) {
    return "";
  }

  // Build C-style string array for string_table_filters
  std::vector<const char *> filter_ptrs;
  for (const auto &s : options.string_table_filters) {
    filter_ptrs.push_back(s.c_str());
  }

  std::string query;
  auto result = impl_->database_->BuildTableQuery(
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

uint64_t reader::execute_query(const std::string &query,
                               const std::string &description) {
  if (!impl_ || !impl_->is_ready() || query.empty()) {
    return 0;
  }

  impl_->future_ = impl_->allocate_future(nullptr);

  uint64_t table_id = 0;
  auto result = impl_->database_->ExecuteQueryAsync(
      query.c_str(), description.empty() ? "" : description.c_str(),
      impl_->future_.get(), &table_id);

  if (result != kRocProfVisDmResultSuccess) {
    return 0;
  }

  result = impl_->future_->WaitForCompletion(kDefaultTimeoutMs);
  if (result != kRocProfVisDmResultSuccess) {
    return 0;
  }

  return table_id;
}

uint64_t reader::execute_query_async(const std::string &query,
                                     const std::string &description,
                                     progress_callback callback) {
  if (!impl_ || !impl_->is_ready() || query.empty()) {
    return 0;
  }

  impl_->future_ = impl_->allocate_future(std::move(callback));

  uint64_t table_id = 0;
  auto result = impl_->database_->ExecuteQueryAsync(
      query.c_str(), description.empty() ? "" : description.c_str(),
      impl_->future_.get(), &table_id);

  if (result != kRocProfVisDmResultSuccess) {
    return 0;
  }

  return table_id;
}

uint64_t reader::num_tables() const {
  if (!impl_ || !impl_->trace_) {
    return 0;
  }

  uint64_t count = 0;
  impl_->trace_->GetPropertyAsUint64(kRPVDMNumberOfTablesUInt64, 0, &count);
  return count;
}

std::unique_ptr<query_result> reader::get_table(uint64_t table_id) const {
  if (!impl_ || !impl_->trace_) {
    return nullptr;
  }

  rocprofvis_dm_handle_t handle = nullptr;
  auto result = impl_->trace_->GetPropertyAsHandle(kRPVDMTableHandleByID,
                                                     table_id, &handle);

  if (result != kRocProfVisDmResultSuccess || !handle) {
    return nullptr;
  }

  return std::unique_ptr<query_result>(new query_result(handle));
}

// ======================== Export / Save ========================

bool reader::export_to_csv(const std::string &query,
                           const std::string &file_path) {
  if (!impl_ || !impl_->is_ready() || query.empty() || file_path.empty()) {
    return false;
  }

  impl_->future_ = impl_->allocate_future(nullptr);

  auto result = impl_->database_->ExportTableCSVAsync(
      query.c_str(), file_path.c_str(), impl_->future_.get());

  if (result != kRocProfVisDmResultSuccess) {
    return false;
  }

  result = impl_->future_->WaitForCompletion(kDefaultTimeoutMs);
  return result == kRocProfVisDmResultSuccess;
}

bool reader::export_to_csv_async(const std::string &query,
                                 const std::string &file_path,
                                 progress_callback callback) {
  if (!impl_ || !impl_->is_ready() || query.empty() || file_path.empty()) {
    return false;
  }

  impl_->future_ = impl_->allocate_future(std::move(callback));

  auto result = impl_->database_->ExportTableCSVAsync(
      query.c_str(), file_path.c_str(), impl_->future_.get());

  return result == kRocProfVisDmResultSuccess;
}

bool reader::save_trimmed(uint64_t start, uint64_t end,
                          const std::string &new_path) {
  if (!impl_ || !impl_->is_ready() || new_path.empty()) {
    return false;
  }

  impl_->future_ = impl_->allocate_future(nullptr);

  auto result = impl_->database_->SaveTrimmedDataAsync(
      start, end, new_path.c_str(), impl_->future_.get());

  if (result != kRocProfVisDmResultSuccess) {
    return false;
  }

  result = impl_->future_->WaitForCompletion(kDefaultTimeoutMs);
  return result == kRocProfVisDmResultSuccess;
}

bool reader::save_trimmed_async(uint64_t start, uint64_t end,
                                const std::string &new_path,
                                progress_callback callback) {
  if (!impl_ || !impl_->is_ready() || new_path.empty()) {
    return false;
  }

  impl_->future_ = impl_->allocate_future(std::move(callback));

  auto result = impl_->database_->SaveTrimmedDataAsync(
      start, end, new_path.c_str(), impl_->future_.get());

  return result == kRocProfVisDmResultSuccess;
}

// ======================== flow_trace implementation ========================

struct flow_trace::Impl {
  explicit Impl(void *handle) : handle_(handle) {
    if (!handle_) {
      throw std::invalid_argument("Invalid flow trace handle");
    }
  }

  void *handle_;
};

flow_trace::flow_trace(void *handle)
    : impl_(std::make_unique<Impl>(handle)) {}

flow_trace::~flow_trace() = default;

flow_trace::flow_trace(flow_trace &&) noexcept = default;
flow_trace &flow_trace::operator=(flow_trace &&) noexcept = default;

uint64_t flow_trace::num_endpoints() const {
  auto *base = static_cast<DmBase *>(impl_->handle_);
  uint64_t count = 0;
  base->GetPropertyAsUint64(kRPVDMNumberOfEndpointsUInt64, 0, &count);
  return count;
}

flow_endpoint flow_trace::get_endpoint(uint64_t index) const {
  auto *base = static_cast<DmBase *>(impl_->handle_);
  flow_endpoint ep;

  uint64_t u64_val = 0;
  char *str_val = nullptr;

  base->GetPropertyAsUint64(kRPVDMEndpointTrackIDUInt64Indexed, index, &u64_val);
  ep.track_id = static_cast<uint32_t>(u64_val);

  base->GetPropertyAsUint64(kRPVDMEndpointIDUInt64Indexed, index, &u64_val);
  ep.event_id = u64_val;

  base->GetPropertyAsUint64(kRPVDMEndpointTimestampUInt64Indexed, index, &u64_val);
  ep.start_timestamp = u64_val;

  base->GetPropertyAsUint64(kRPVDMEndpointEndTimestampUInt64Indexed, index, &u64_val);
  ep.end_timestamp = u64_val;

  base->GetPropertyAsCharPtr(kRPVDMEndpointCategoryCharPtrIndexed, index, &str_val);
  ep.category = safe_string(str_val);

  base->GetPropertyAsCharPtr(kRPVDMEndpointSymbolCharPtrIndexed, index, &str_val);
  ep.symbol = safe_string(str_val);

  base->GetPropertyAsUint64(kRPVDMEndpointLevelUInt64Indexed, index, &u64_val);
  ep.level = u64_val;

  return ep;
}

std::vector<flow_endpoint> flow_trace::get_endpoints() const {
  std::vector<flow_endpoint> endpoints;
  auto count = num_endpoints();
  endpoints.reserve(count);
  for (uint64_t i = 0; i < count; ++i) {
    endpoints.push_back(get_endpoint(i));
  }
  return endpoints;
}

// ======================== stack_trace implementation ========================

struct stack_trace::Impl {
  explicit Impl(void *handle) : handle_(handle) {
    if (!handle_) {
      throw std::invalid_argument("Invalid stack trace handle");
    }
  }

  void *handle_;
};

stack_trace::stack_trace(void *handle)
    : impl_(std::make_unique<Impl>(handle)) {}

stack_trace::~stack_trace() = default;

stack_trace::stack_trace(stack_trace &&) noexcept = default;
stack_trace &stack_trace::operator=(stack_trace &&) noexcept = default;

uint64_t stack_trace::num_frames() const {
  auto *base = static_cast<DmBase *>(impl_->handle_);
  uint64_t count = 0;
  base->GetPropertyAsUint64(kRPVDMNumberOfFramesUInt64, 0, &count);
  return count;
}

stack_frame stack_trace::get_frame(uint64_t index) const {
  auto *base = static_cast<DmBase *>(impl_->handle_);
  stack_frame frame;

  uint64_t u64_val = 0;
  char *str_val = nullptr;

  base->GetPropertyAsUint64(kRPVDMFrameDepthUInt64Indexed, index, &u64_val);
  frame.depth = u64_val;

  base->GetPropertyAsCharPtr(kRPVDMFrameSymbolCharPtrIndexed, index, &str_val);
  frame.symbol = safe_string(str_val);

  base->GetPropertyAsCharPtr(kRPVDMFrameArgsCharPtrIndexed, index, &str_val);
  frame.arguments = safe_string(str_val);

  base->GetPropertyAsCharPtr(kRPVDMFrameCodeLineCharPtrIndexed, index, &str_val);
  frame.code_line = safe_string(str_val);

  return frame;
}

std::vector<stack_frame> stack_trace::get_frames() const {
  std::vector<stack_frame> frames;
  auto count = num_frames();
  frames.reserve(count);
  for (uint64_t i = 0; i < count; ++i) {
    frames.push_back(get_frame(i));
  }
  return frames;
}

// ======================== ext_data implementation ========================

struct ext_data::Impl {
  explicit Impl(void *handle) : handle_(handle) {
    if (!handle_) {
      throw std::invalid_argument("Invalid ext_data handle");
    }
  }

  void *handle_;
};

ext_data::ext_data(void *handle) : impl_(std::make_unique<Impl>(handle)) {}

ext_data::~ext_data() = default;

ext_data::ext_data(ext_data &&) noexcept = default;
ext_data &ext_data::operator=(ext_data &&) noexcept = default;

uint64_t ext_data::num_records() const {
  auto *base = static_cast<DmBase *>(impl_->handle_);
  uint64_t count = 0;
  base->GetPropertyAsUint64(kRPVDMNumberOfExtDataRecordsUInt64, 0, &count);
  return count;
}

ext_data_record ext_data::get_record(uint64_t index) const {
  auto *base = static_cast<DmBase *>(impl_->handle_);
  ext_data_record record;

  char *str_val = nullptr;
  uint64_t u64_val = 0;

  base->GetPropertyAsCharPtr(kRPVDMExtDataCategoryCharPtrIndexed, index, &str_val);
  record.category = safe_string(str_val);

  base->GetPropertyAsCharPtr(kRPVDMExtDataNameCharPtrIndexed, index, &str_val);
  record.name = safe_string(str_val);

  base->GetPropertyAsCharPtr(kRPVDMExtDataValueCharPtrIndexed, index, &str_val);
  record.value = safe_string(str_val);

  base->GetPropertyAsUint64(kRPVDMExtDataTypeUint64Indexed, index, &u64_val);
  record.type = u64_val;

  return record;
}

std::vector<ext_data_record> ext_data::get_records() const {
  std::vector<ext_data_record> records;
  auto count = num_records();
  records.reserve(count);
  for (uint64_t i = 0; i < count; ++i) {
    records.push_back(get_record(i));
  }
  return records;
}

// ======================== table_row implementation ========================

struct table_row::Impl {
  explicit Impl(void *handle) : handle_(handle) {
    if (!handle_) {
      throw std::invalid_argument("Invalid table_row handle");
    }
  }

  void *handle_;
};

table_row::table_row(void *handle) : impl_(std::make_unique<Impl>(handle)) {}

table_row::~table_row() = default;

table_row::table_row(table_row &&) noexcept = default;
table_row &table_row::operator=(table_row &&) noexcept = default;

uint64_t table_row::num_cells() const {
  auto *base = static_cast<DmBase *>(impl_->handle_);
  uint64_t count = 0;
  base->GetPropertyAsUint64(kRPVDMNumberOfTableRowCellsUInt64, 0, &count);
  return count;
}

std::string table_row::get_cell(uint64_t index) const {
  auto *base = static_cast<DmBase *>(impl_->handle_);
  char *str_val = nullptr;
  base->GetPropertyAsCharPtr(kRPVDMExtTableRowCellValueCharPtrIndexed, index,
                             &str_val);
  return safe_string(str_val);
}

std::vector<std::string> table_row::get_cells() const {
  std::vector<std::string> cells;
  auto count = num_cells();
  cells.reserve(count);
  for (uint64_t i = 0; i < count; ++i) {
    cells.push_back(get_cell(i));
  }
  return cells;
}

// ======================== query_result implementation ========================

struct query_result::Impl {
  explicit Impl(void *handle) : handle_(handle) {
    if (!handle_) {
      throw std::invalid_argument("Invalid query_result handle");
    }
  }

  void *handle_;
};

query_result::query_result(void *handle)
    : impl_(std::make_unique<Impl>(handle)) {}

query_result::~query_result() = default;

query_result::query_result(query_result &&) noexcept = default;
query_result &query_result::operator=(query_result &&) noexcept = default;

uint64_t query_result::id() const {
  auto *base = static_cast<DmBase *>(impl_->handle_);
  uint64_t val = 0;
  base->GetPropertyAsUint64(kRPVDMNumberOfTableIdUInt64, 0, &val);
  return val;
}

std::string query_result::description() const {
  auto *base = static_cast<DmBase *>(impl_->handle_);
  char *str_val = nullptr;
  base->GetPropertyAsCharPtr(kRPVDMExtTableDescriptionCharPtr, 0, &str_val);
  return safe_string(str_val);
}

std::string query_result::query() const {
  auto *base = static_cast<DmBase *>(impl_->handle_);
  char *str_val = nullptr;
  base->GetPropertyAsCharPtr(kRPVDMExtTableQueryCharPtr, 0, &str_val);
  return safe_string(str_val);
}

uint64_t query_result::num_columns() const {
  auto *base = static_cast<DmBase *>(impl_->handle_);
  uint64_t count = 0;
  base->GetPropertyAsUint64(kRPVDMNumberOfTableColumnsUInt64, 0, &count);
  return count;
}

std::string query_result::column_name(uint64_t index) const {
  auto *base = static_cast<DmBase *>(impl_->handle_);
  char *str_val = nullptr;
  base->GetPropertyAsCharPtr(kRPVDMExtTableColumnNameCharPtrIndexed, index,
                             &str_val);
  return safe_string(str_val);
}

std::vector<std::string> query_result::column_names() const {
  std::vector<std::string> names;
  auto count = num_columns();
  names.reserve(count);
  for (uint64_t i = 0; i < count; ++i) {
    names.push_back(column_name(i));
  }
  return names;
}

uint64_t query_result::num_rows() const {
  auto *base = static_cast<DmBase *>(impl_->handle_);
  uint64_t count = 0;
  base->GetPropertyAsUint64(kRPVDMNumberOfTableRowsUInt64, 0, &count);
  return count;
}

std::unique_ptr<table_row> query_result::get_row(uint64_t index) const {
  auto *base = static_cast<DmBase *>(impl_->handle_);
  rocprofvis_dm_handle_t handle = nullptr;
  auto result =
      base->GetPropertyAsHandle(kRPVDMExtTableRowHandleIndexed, index, &handle);

  if (result != kRocProfVisDmResultSuccess || !handle) {
    return nullptr;
  }

  return std::unique_ptr<table_row>(new table_row(handle));
}

} // namespace rocstorage