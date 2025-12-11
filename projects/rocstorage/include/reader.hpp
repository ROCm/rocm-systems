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

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

// Forward declarations for C API interop
typedef void *rocprofvis_dm_trace_t;
typedef void *rocprofvis_dm_database_t;

namespace rocstorage {

/// Track category enumeration matching C API
enum class track_category {
  not_a_track = 0,
  pmc = 1,
  region = 2,
  kernel_dispatch = 3,
  sqtt = 4,
  nic = 5,
  memory_allocation = 6,
  memory_copy = 7,
  stream = 8,
  region_main = 9,
  region_sample = 10,
};

/// Database type enumeration
enum class database_type {
  autodetect = 0,
  rocpd_sqlite = 1,
  rocprof_sqlite = 2,
};

/// Progress callback for async operations
using progress_callback =
    std::function<void(const char *filename, uint16_t percent, bool success,
                       const char *message)>;

/// Represents a track in the trace (e.g., PMC track, kernel dispatch track)
class track {
public:
  ~track();

  track(const track &) = delete;
  track &operator=(const track &) = delete;
  track(track &&) noexcept;
  track &operator=(track &&) noexcept;

  /// Get track ID
  uint32_t id() const;

  /// Get track category
  track_category category() const;

  /// Get category as string
  std::string category_string() const;

  /// Get node ID
  uint64_t node_id() const;

  /// Get process name (PID, GPU ID, etc.)
  std::string process_name() const;

  /// Get subprocess name (TID, Queue ID, PMC name, etc.)
  std::string subprocess_name() const;

  /// Get total number of records in this track
  uint64_t num_records() const;

  /// Get minimum timestamp
  uint64_t min_timestamp() const;

  /// Get maximum timestamp
  uint64_t max_timestamp() const;

  /// Get number of loaded slices
  uint64_t num_slices() const;

private:
  friend class reader;
  explicit track(void *handle);

  struct impl;
  std::unique_ptr<impl> m_impl;
};

/// Main reader class for accessing profiling data
class reader {
public:
  /// Open a database file for reading
  /// @param path Path to the database file
  /// @param type Database type (autodetect by default)
  /// @return Reader instance, or nullptr on failure
  static std::unique_ptr<reader>
  open(const std::string &path,
       database_type type = database_type::autodetect);

  ~reader();

  reader(const reader &) = delete;
  reader &operator=(const reader &) = delete;
  reader(reader &&) noexcept;
  reader &operator=(reader &&) noexcept;

  /// Read metadata (tracks, time bounds) synchronously
  /// @return true on success
  bool read_metadata();

  /// Read metadata asynchronously
  /// @param callback Optional progress callback
  /// @return true if operation started successfully
  bool read_metadata_async(progress_callback callback = nullptr);

  /// Wait for any pending async operation
  /// @param timeout_sec Timeout in seconds (0 = infinite)
  /// @return true if operation completed successfully
  bool wait(uint64_t timeout_sec = 0);

  /// Cancel any pending async operation
  void cancel();

  /// Get trace start time
  uint64_t start_time() const;

  /// Get trace end time
  uint64_t end_time() const;

  /// Get number of tracks
  uint64_t num_tracks() const;

  /// Get track by index
  /// @param index Track index (0 to num_tracks() - 1)
  /// @return Track object, or nullptr if index is invalid
  std::unique_ptr<track> get_track(uint64_t index) const;

  /// Get all tracks
  std::vector<std::unique_ptr<track>> get_tracks() const;

  /// Read a time slice for specified tracks
  /// @param start Start timestamp
  /// @param end End timestamp
  /// @param track_ids Vector of track IDs to read
  /// @return true on success
  bool read_slice(uint64_t start, uint64_t end,
                  const std::vector<uint32_t> &track_ids);

  /// Read a time slice asynchronously
  /// @param start Start timestamp
  /// @param end End timestamp
  /// @param track_ids Vector of track IDs to read
  /// @param callback Optional progress callback
  /// @return true if operation started successfully
  bool read_slice_async(uint64_t start, uint64_t end,
                        const std::vector<uint32_t> &track_ids,
                        progress_callback callback = nullptr);

  /// Get memory footprint of loaded data
  uint64_t memory_footprint() const;

  /// Get the underlying C API trace handle (for interop)
  rocprofvis_dm_trace_t c_trace_handle() const;

  /// Get the underlying C API database handle (for interop)
  rocprofvis_dm_database_t c_database_handle() const;

private:
  reader() = default;

  struct impl;
  std::unique_ptr<impl> m_impl;
};

} // namespace rocstorage