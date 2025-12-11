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

struct track::impl {
  explicit impl(Track *handle) : m_handle(handle) {
    if (!m_handle) {
      throw std::invalid_argument("Invalid track handle");
    }
  }

  Track *m_handle;
};

track::track(void *handle)
    : m_impl(std::make_unique<impl>(static_cast<Track *>(handle))) {}

track::~track() = default;

track::track(track &&) noexcept = default;
track &track::operator=(track &&) noexcept = default;

uint32_t track::id() const { return m_impl->m_handle->TrackId(); }

track_category track::category() const {
  return static_cast<track_category>(m_impl->m_handle->Category());
}

std::string track::category_string() const {
  return safe_string(m_impl->m_handle->CategoryString());
}

uint64_t track::node_id() const { return m_impl->m_handle->NodeId(); }

std::string track::process_name() const {
  return safe_string(m_impl->m_handle->Process());
}

std::string track::subprocess_name() const {
  return safe_string(m_impl->m_handle->SubProcess());
}

uint64_t track::num_records() const { return m_impl->m_handle->NumRecords(); }

uint64_t track::min_timestamp() const {
  return m_impl->m_handle->MinTimestamp();
}

uint64_t track::max_timestamp() const {
  return m_impl->m_handle->MaxTimestamp();
}

uint64_t track::num_slices() const {
  return m_impl->m_handle->NumberOfSlices();
}

// ======================== reader implementation ========================

struct reader::impl {
  explicit impl(const std::string &path, database_type type)
      : m_trace(std::make_unique<Trace>()), m_database(nullptr),
        m_future(nullptr) {
    if (path.empty()) {
      throw std::runtime_error("Empty database path");
    }

    // Check if file exists and is a valid SQLite database
    // (SQLite creates the file if it doesn't exist, which we don't want)
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
      m_database = std::make_unique<RocpdDatabase>(path.c_str());
    } else if (db_type == kRocprofSqlite) {
      m_database = std::make_unique<RocprofDatabase>(path.c_str());
    } else {
      throw std::runtime_error("Unsupported database type");
    }

    if (m_database->Open() != kRocProfVisDmResultSuccess) {
      throw std::runtime_error("Failed to open database: " + path);
    }

    // Bind trace to database
    rocprofvis_dm_db_bind_struct *bind_data = nullptr;
    auto result = m_trace->BindDatabase(m_database.get(), bind_data);
    if (result != kRocProfVisDmResultSuccess) {
      throw std::runtime_error("Failed to bind trace to database");
    }

    // Tell database about the trace binding (so BindObject() works)
    result = m_database->BindTrace(bind_data);
    if (result != kRocProfVisDmResultSuccess) {
      throw std::runtime_error("Failed to bind database to trace");
    }
  }

  bool is_ready() const { return m_trace && m_database; }

  std::unique_ptr<Future> allocate_future(progress_callback callback) {
    m_callback = std::move(callback);

    if (m_callback) {
      return std::make_unique<Future>(
          [](rocprofvis_db_filename_t filename,
             rocprofvis_db_progress_percent_t percent,
             rocprofvis_db_status_t status,
             rocprofvis_db_status_message_t message, void *user_data) {
            auto *self = static_cast<impl *>(user_data);
            if (self->m_callback) {
              self->m_callback(filename, percent, status == kRPVDbSuccess,
                               message);
            }
          },
          this);
    }

    return std::make_unique<Future>(nullptr, nullptr);
  }

  std::unique_ptr<Trace> m_trace;
  std::unique_ptr<Database> m_database;
  std::unique_ptr<Future> m_future;
  progress_callback m_callback;
};

std::unique_ptr<reader> reader::open(const std::string &path,
                                     database_type type) {
  try {
    auto r = std::unique_ptr<reader>(new reader());
    r->m_impl = std::make_unique<impl>(path, type);
    return r;
  } catch (const std::exception &) {
    return nullptr;
  }
}

reader::~reader() = default;

reader::reader(reader &&) noexcept = default;
reader &reader::operator=(reader &&) noexcept = default;

bool reader::read_metadata() {
  if (!m_impl || !m_impl->is_ready()) {
    return false;
  }

  m_impl->m_future = m_impl->allocate_future(nullptr);
  auto result =
      m_impl->m_database->ReadTraceMetadataAsync(m_impl->m_future.get());
  if (result != kRocProfVisDmResultSuccess) {
    return false;
  }

  result = m_impl->m_future->WaitForCompletion(kDefaultTimeoutMs);
  return result == kRocProfVisDmResultSuccess;
}

bool reader::read_metadata_async(progress_callback callback) {
  if (!m_impl || !m_impl->is_ready()) {
    return false;
  }

  m_impl->m_future = m_impl->allocate_future(std::move(callback));
  auto result =
      m_impl->m_database->ReadTraceMetadataAsync(m_impl->m_future.get());
  return result == kRocProfVisDmResultSuccess;
}

bool reader::wait(uint64_t timeout_sec) {
  if (!m_impl || !m_impl->m_future) {
    return false;
  }

  // Convert seconds to milliseconds (WaitForCompletion uses ms)
  // 0 = wait indefinitely
  uint64_t timeout_ms = (timeout_sec == 0) ? UINT64_MAX : timeout_sec * 1000;
  auto result = m_impl->m_future->WaitForCompletion(timeout_ms);
  return result == kRocProfVisDmResultSuccess;
}

void reader::cancel() {
  if (m_impl && m_impl->m_future) {
    m_impl->m_future->SetInterrupted();
  }
}

uint64_t reader::start_time() const {
  if (!m_impl || !m_impl->m_trace) {
    return 0;
  }
  return m_impl->m_trace->StartTime();
}

uint64_t reader::end_time() const {
  if (!m_impl || !m_impl->m_trace) {
    return 0;
  }
  return m_impl->m_trace->EndTime();
}

uint64_t reader::num_tracks() const {
  if (!m_impl || !m_impl->m_trace) {
    return 0;
  }
  return m_impl->m_trace->NumberOfTracks();
}

std::unique_ptr<track> reader::get_track(uint64_t index) const {
  if (!m_impl || !m_impl->m_trace || index >= num_tracks()) {
    return nullptr;
  }

  rocprofvis_dm_track_t handle = nullptr;
  auto result = m_impl->m_trace->GetPropertyAsHandle(kRPVDMTrackHandleIndexed,
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
  if (!m_impl || !m_impl->is_ready() || track_ids.empty()) {
    return false;
  }

  m_impl->m_future = m_impl->allocate_future(nullptr);
  auto result = m_impl->m_database->ReadTraceSliceAsync(
      start, end, static_cast<uint16_t>(track_ids.size()),
      const_cast<uint32_t *>(track_ids.data()), m_impl->m_future.get());

  if (result != kRocProfVisDmResultSuccess) {
    return false;
  }

  result = m_impl->m_future->WaitForCompletion(kDefaultTimeoutMs);
  return result == kRocProfVisDmResultSuccess;
}

bool reader::read_slice_async(uint64_t start, uint64_t end,
                              const std::vector<uint32_t> &track_ids,
                              progress_callback callback) {
  if (!m_impl || !m_impl->is_ready() || track_ids.empty()) {
    return false;
  }

  m_impl->m_future = m_impl->allocate_future(std::move(callback));
  auto result = m_impl->m_database->ReadTraceSliceAsync(
      start, end, static_cast<uint16_t>(track_ids.size()),
      const_cast<uint32_t *>(track_ids.data()), m_impl->m_future.get());

  return result == kRocProfVisDmResultSuccess;
}

uint64_t reader::memory_footprint() const {
  if (!m_impl || !m_impl->m_trace) {
    return 0;
  }
  uint64_t footprint = 0;
  m_impl->m_trace->GetPropertyAsUint64(kRPVDMTraceMemoryFootprintUInt64, 0,
                                       &footprint);
  return footprint;
}

rocprofvis_dm_trace_t reader::c_trace_handle() const {
  return m_impl ? m_impl->m_trace.get() : nullptr;
}

rocprofvis_dm_database_t reader::c_database_handle() const {
  return m_impl ? m_impl->m_database.get() : nullptr;
}

} // namespace rocstorage