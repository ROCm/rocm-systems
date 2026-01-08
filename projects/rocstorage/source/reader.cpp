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
#include <rocstorage/database.hpp>
#include <rocstorage/trace.hpp>

#include "error.hpp"
#include "reader/datamodel/internal_types.h"
#include "reader/datamodel/rocprofvis_dm_trace.h"
#include "reader/database/rocprofvis_db.h"
#include "reader/database/rocprofvis_db_future.h"
#include "reader/database/rocprofvis_db_rocpd.h"
#include "reader/database/rocprofvis_db_rocprof.h"

#include <cstdint>
#include <fstream>
#include <limits>
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
  explicit Impl(const std::string &path, database_type type) {
    // Create trace (owns internal Trace)
    trace_ = std::make_unique<rocstorage::trace>();

    // Open database (owns internal Database)
    auto db_result = rocstorage::database::open(path, type);
    if (!db_result) {
      throw std::runtime_error(db_result.get_error().message());
    }
    database_ = std::move(db_result.value());

    // Bind trace to database
    auto bind_status = database_->bind(*trace_);
    if (!bind_status) {
      throw std::runtime_error(bind_status.get_error().message());
    }
  }

  bool is_ready() const { return trace_ && database_; }

  std::unique_ptr<rocstorage::trace> trace_;
  std::unique_ptr<rocstorage::database> database_;
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

result<std::unique_ptr<reader>> reader::try_open(const std::string &path,
                                                  database_type type) {
  if (path.empty()) {
    return error(error_code::invalid_parameter, "Empty database path");
  }

  // SQLite creates the file if it doesn't exist, which we don't want
  if (!is_valid_sqlite_file(path)) {
    return error(error_code::file_not_found,
                 "File does not exist or is not a valid SQLite database: " + path);
  }

  try {
    auto r = std::unique_ptr<reader>(new reader());
    r->impl_ = std::make_unique<Impl>(path, type);
    return std::move(r);
  } catch (const std::exception &e) {
    return error(error_code::db_access_failed,
                 std::string("Failed to open database: ") + e.what());
  }
}

reader::~reader() = default;

reader::reader(reader &&) noexcept = default;
reader &reader::operator=(reader &&) noexcept = default;

bool reader::read_metadata() {
  if (!impl_ || !impl_->is_ready()) {
    return false;
  }
  return impl_->database_->read_metadata_async() && impl_->database_->wait();
}

status reader::try_read_metadata() {
  if (!impl_) {
    return error(error_code::invalid_parameter, "Reader not initialized");
  }
  if (!impl_->is_ready()) {
    return error(error_code::not_loaded, "Reader not ready");
  }

  auto start_status = impl_->database_->try_read_metadata_async();
  if (!start_status) {
    return start_status;
  }

  return impl_->database_->try_wait();
}

bool reader::read_metadata_async(progress_callback callback) {
  if (!impl_ || !impl_->is_ready()) {
    return false;
  }
  return impl_->database_->read_metadata_async(std::move(callback));
}

bool reader::wait(uint64_t timeout_sec) {
  if (!impl_ || !impl_->database_) {
    return false;
  }
  return impl_->database_->wait(timeout_sec);
}

void reader::cancel() {
  if (impl_ && impl_->database_) {
    impl_->database_->cancel();
  }
}

uint64_t reader::start_time() const {
  if (!impl_ || !impl_->trace_) {
    return 0;
  }
  return impl_->trace_->start_time();
}

uint64_t reader::end_time() const {
  if (!impl_ || !impl_->trace_) {
    return 0;
  }
  return impl_->trace_->end_time();
}

uint64_t reader::num_tracks() const {
  if (!impl_ || !impl_->trace_) {
    return 0;
  }
  return impl_->trace_->num_tracks();
}

std::unique_ptr<track> reader::get_track(uint64_t index) const {
  if (!impl_ || !impl_->trace_) {
    return nullptr;
  }
  return impl_->trace_->get_track(index);
}

result<std::unique_ptr<track>> reader::try_get_track(uint64_t index) const {
  if (!impl_) {
    return error(error_code::invalid_parameter, "Reader not initialized");
  }
  if (!impl_->trace_) {
    return error(error_code::not_loaded, "Trace not loaded");
  }
  return impl_->trace_->try_get_track(index);
}

std::vector<std::unique_ptr<track>> reader::get_tracks() const {
  if (!impl_ || !impl_->trace_) {
    return {};
  }
  return impl_->trace_->get_tracks();
}

bool reader::read_slice(uint64_t start, uint64_t end,
                        const std::vector<uint32_t> &track_ids) {
  if (!impl_ || !impl_->is_ready()) {
    return false;
  }
  return impl_->database_->read_slice_async(start, end, track_ids) &&
         impl_->database_->wait();
}

status reader::try_read_slice(uint64_t start, uint64_t end,
                               const std::vector<uint32_t> &track_ids) {
  if (!impl_) {
    return error(error_code::invalid_parameter, "Reader not initialized");
  }
  if (!impl_->is_ready()) {
    return error(error_code::not_loaded, "Reader not ready");
  }

  auto start_status = impl_->database_->try_read_slice_async(start, end, track_ids);
  if (!start_status) {
    return start_status;
  }

  return impl_->database_->try_wait();
}

bool reader::read_slice_async(uint64_t start, uint64_t end,
                              const std::vector<uint32_t> &track_ids,
                              progress_callback callback) {
  if (!impl_ || !impl_->is_ready()) {
    return false;
  }
  return impl_->database_->read_slice_async(start, end, track_ids, std::move(callback));
}

uint64_t reader::memory_footprint() const {
  if (!impl_ || !impl_->trace_) {
    return 0;
  }
  return impl_->trace_->memory_footprint();
}

rocprofvis_dm_trace_t reader::c_trace_handle() const {
  return impl_ && impl_->trace_ ? impl_->trace_->c_handle() : nullptr;
}

rocprofvis_dm_database_t reader::c_database_handle() const {
  return impl_ && impl_->database_ ? impl_->database_->c_handle() : nullptr;
}

// ======================== Sub-object Access ========================

trace *reader::get_trace() {
  return impl_ ? impl_->trace_.get() : nullptr;
}

const trace *reader::get_trace() const {
  return impl_ ? impl_->trace_.get() : nullptr;
}

database *reader::get_database() {
  return impl_ ? impl_->database_.get() : nullptr;
}

const database *reader::get_database() const {
  return impl_ ? impl_->database_.get() : nullptr;
}

// ======================== Memory Management ========================

bool reader::delete_slice(uint64_t start, uint64_t end) {
  if (!impl_ || !impl_->trace_) {
    return false;
  }
  return impl_->trace_->delete_slice(start, end);
}

bool reader::delete_slice(uint32_t track_id, void *slice_handle) {
  if (!impl_ || !impl_->trace_) {
    return false;
  }
  return impl_->trace_->delete_slice(track_id, slice_handle);
}

bool reader::delete_all_slices() {
  if (!impl_ || !impl_->trace_) {
    return false;
  }
  return impl_->trace_->delete_all_slices();
}

bool reader::delete_all_tables() {
  if (!impl_ || !impl_->trace_) {
    return false;
  }
  return impl_->trace_->delete_all_tables();
}

bool reader::delete_table(uint64_t table_id) {
  if (!impl_ || !impl_->trace_) {
    return false;
  }
  return impl_->trace_->delete_table(table_id);
}

// ======================== Event Properties ========================

bool reader::read_event_property(event_property_type type, event_id id) {
  if (!impl_ || !impl_->is_ready()) {
    return false;
  }
  return impl_->database_->read_event_property_async(type, id.raw()) &&
         impl_->database_->wait();
}

bool reader::read_event_property_async(event_property_type type, event_id id,
                                       progress_callback callback) {
  if (!impl_ || !impl_->is_ready()) {
    return false;
  }
  return impl_->database_->read_event_property_async(type, id.raw(),
                                                     std::move(callback));
}

std::unique_ptr<flow_trace> reader::get_flow_trace(event_id id) const {
  if (!impl_ || !impl_->trace_) {
    return nullptr;
  }
  return impl_->trace_->get_flow_trace(id);
}

std::unique_ptr<stack_trace> reader::get_stack_trace(event_id id) const {
  if (!impl_ || !impl_->trace_) {
    return nullptr;
  }
  return impl_->trace_->get_stack_trace(id);
}

std::unique_ptr<ext_data> reader::get_ext_data(event_id id) const {
  if (!impl_ || !impl_->trace_) {
    return nullptr;
  }
  return impl_->trace_->get_ext_data(id);
}

bool reader::delete_event_property(event_property_type type, event_id id) {
  if (!impl_ || !impl_->trace_) {
    return false;
  }
  return impl_->trace_->delete_event_property(type, id);
}

bool reader::delete_all_event_properties(event_property_type type) {
  if (!impl_ || !impl_->trace_) {
    return false;
  }
  return impl_->trace_->delete_all_event_properties(type);
}

// ======================== Query Execution ========================

std::string reader::build_table_query(const table_query_options &options) const {
  if (!impl_ || !impl_->database_) {
    return "";
  }
  return impl_->database_->build_table_query(options);
}

uint64_t reader::execute_query(const std::string &query,
                               const std::string &description) {
  if (!impl_ || !impl_->is_ready() || query.empty()) {
    return 0;
  }

  uint64_t table_id = impl_->database_->execute_query_async(query, description);
  if (table_id == 0) {
    return 0;
  }

  if (!impl_->database_->wait()) {
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
  return impl_->database_->execute_query_async(query, description,
                                               std::move(callback));
}

uint64_t reader::num_tables() const {
  if (!impl_ || !impl_->trace_) {
    return 0;
  }
  return impl_->trace_->num_tables();
}

std::unique_ptr<query_result> reader::get_table(uint64_t table_id) const {
  if (!impl_ || !impl_->trace_) {
    return nullptr;
  }
  return impl_->trace_->get_table(table_id);
}

// ======================== Export / Save ========================

bool reader::export_to_csv(const std::string &query,
                           const std::string &file_path) {
  if (!impl_ || !impl_->is_ready() || query.empty() || file_path.empty()) {
    return false;
  }
  return impl_->database_->export_to_csv_async(query, file_path) &&
         impl_->database_->wait();
}

bool reader::export_to_csv_async(const std::string &query,
                                 const std::string &file_path,
                                 progress_callback callback) {
  if (!impl_ || !impl_->is_ready() || query.empty() || file_path.empty()) {
    return false;
  }
  return impl_->database_->export_to_csv_async(query, file_path,
                                               std::move(callback));
}

bool reader::save_trimmed(uint64_t start, uint64_t end,
                          const std::string &new_path) {
  if (!impl_ || !impl_->is_ready() || new_path.empty()) {
    return false;
  }
  return impl_->database_->save_trimmed_async(start, end, new_path) &&
         impl_->database_->wait();
}

bool reader::save_trimmed_async(uint64_t start, uint64_t end,
                                const std::string &new_path,
                                progress_callback callback) {
  if (!impl_ || !impl_->is_ready() || new_path.empty()) {
    return false;
  }
  return impl_->database_->save_trimmed_async(start, end, new_path,
                                              std::move(callback));
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