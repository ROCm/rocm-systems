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

#include <rocstorage/enum_definitions.h>
#include <rocstorage/reader.hpp>
#include <rocstorage/writer.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace rocm {

/// Configuration for storage mode and behavior
struct storage_config {
  /// Storage access modes
  enum class mode {
    unknown,    ///< Unknown/unset mode (will use detect_defaults)
    write,      ///< Write-only: in-memory SQLite, flush to disk at end (max write perf)
    read_write, ///< Read-write: file-based SQLite with WAL mode (concurrent access)
    read        ///< Read-only: open existing database for analysis
  };

  mode storage_mode = mode::unknown;

  // Read-write mode options
  std::string wal_directory;  ///< Directory for WAL files (empty = platform default)
  bool use_ram_disk = true;   ///< Prefer RAM disk (/dev/shm) if available

  // Read-only mode options
  bool cache_in_memory = false;    ///< Load entire DB into memory for repeated queries
  size_t connection_pool_size = 0; ///< Connection pool size (0 = auto/hardware_concurrency)

  // Shared options
  bool sync_on_close = true; ///< Ensure data persisted before destructor returns

  /// Create write-only configuration (default, max performance)
  static storage_config write_only();

  /// Create read-write configuration (concurrent access via WAL)
  static storage_config read_write();

  /// Create read-only configuration (for analysis)
  static storage_config read_only();

  /// Detect platform-appropriate defaults
  static storage_config detect_defaults();

  /// Get platform-specific default WAL directory
  static std::string default_wal_directory();
};

/// Track data (value type, internally managed by storage)
class track_view {
public:
  uint32_t id() const { return id_; }
  rocstorage::track_category category() const { return category_; }
  std::string_view category_string() const { return category_string_; }
  uint64_t node_id() const { return node_id_; }
  std::string_view process_name() const { return process_name_; }
  std::string_view subprocess_name() const { return subprocess_name_; }
  uint64_t num_records() const { return num_records_; }
  uint64_t min_timestamp() const { return min_timestamp_; }
  uint64_t max_timestamp() const { return max_timestamp_; }

private:
  friend class storage;

  uint32_t id_ = 0;
  rocstorage::track_category category_ = kRocProfVisDmNotATrack;
  std::string category_string_;
  uint64_t node_id_ = 0;
  std::string process_name_;
  std::string subprocess_name_;
  uint64_t num_records_ = 0;
  uint64_t min_timestamp_ = 0;
  uint64_t max_timestamp_ = 0;
};

/// Unified storage facade for reading and writing profiling data
class storage {
public:
  /// Create storage for writing (existing constructor, uses write-only mode)
  explicit storage(std::string database_path, std::string uuid);

  /// Create storage with explicit configuration
  static std::unique_ptr<storage> create(const std::string &path,
                                         const std::string &uuid,
                                         const storage_config &config);

  /// Open existing database for reading
  static std::unique_ptr<storage> open(const std::string &path);

  virtual ~storage();

  storage(const storage &) = delete;
  storage(storage &&) = delete;
  storage &operator=(const storage &) = delete;
  storage &operator=(storage &&) = delete;

  // ==================== OO API ====================

  /// Load metadata from database
  bool load();

  /// Check if metadata has been loaded
  bool is_loaded() const;

  /// Get trace start time (nanoseconds)
  uint64_t start_time() const;

  /// Get trace end time (nanoseconds)
  uint64_t end_time() const;

  /// Get number of tracks
  size_t num_tracks() const;

  /// Get all tracks (range-based iteration)
  const std::vector<track_view> &tracks() const;

  /// Get track by index
  const track_view &track(size_t index) const;

  /// Get database path
  std::string_view path() const;

  // ==================== Low-level Access ====================

  std::shared_ptr<rocstorage::writer> get_writer() const;
  std::shared_ptr<rocstorage::reader> get_reader() const;

private:
  storage();

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace rocm
