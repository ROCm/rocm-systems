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
#include <rocstorage/result.hpp>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

// Forward declaration for C API interop
typedef void *rocprofvis_dm_database_t;

namespace rocstorage {

class trace;
struct table_query_options;

/// Progress callback for async operations
using progress_callback =
    std::function<void(const char *filename, uint16_t percent, bool success,
                       const char *message)>;

/// Database wrapper for reading profiling data from SQLite files.
///
/// This class wraps the internal RocProfVis::DataModel::Database and provides
/// a public API for database operations. It can be used as either:
/// - An owning wrapper (open() creates and owns internal Database)
/// - A non-owning view (pointer constructor wraps existing internal Database)
///
/// Use this class directly when you need fine-grained control over database
/// operations, or use the `reader` class for a higher-level convenience API.
class database {
public:
  /// Open a database file (creates owning wrapper)
  /// @param path Path to the database file
  /// @param type Database type (autodetect by default)
  /// @return Result containing database instance or error with details
  [[nodiscard]] static result<std::unique_ptr<database>>
  open(const std::string &path, database_type type = kAutodetect);

  /// Create a non-owning view of an existing internal Database
  /// @param internal Pointer to internal Database (caller retains ownership)
  explicit database(rocprofvis_dm_database_t internal);

  ~database();

  database(const database &) = delete;
  database &operator=(const database &) = delete;
  database(database &&) noexcept;
  database &operator=(database &&) noexcept;

  /// Bind a trace to this database for data loading
  /// @param t Trace to bind (must outlive database operations)
  /// @return Status indicating success or error
  [[nodiscard]] status bind(trace &t);

  // ==================== Async Operations ====================

  /// Read trace metadata asynchronously
  /// @param callback Optional progress callback
  /// @return true if operation started successfully
  bool read_metadata_async(progress_callback callback = nullptr);

  /// Read trace metadata with error information
  /// @param callback Optional progress callback
  /// @return Status indicating success or error
  [[nodiscard]] status try_read_metadata_async(progress_callback callback = nullptr);

  /// Read a time slice asynchronously
  /// @param start Start timestamp
  /// @param end End timestamp
  /// @param track_ids Vector of track IDs to read
  /// @param callback Optional progress callback
  /// @return true if operation started successfully
  bool read_slice_async(uint64_t start, uint64_t end,
                        const std::vector<uint32_t> &track_ids,
                        progress_callback callback = nullptr);

  /// Read a time slice with error information
  /// @param start Start timestamp
  /// @param end End timestamp
  /// @param track_ids Vector of track IDs to read
  /// @param callback Optional progress callback
  /// @return Status indicating success or error
  [[nodiscard]] status try_read_slice_async(uint64_t start, uint64_t end,
                                            const std::vector<uint32_t> &track_ids,
                                            progress_callback callback = nullptr);

  /// Read event property asynchronously
  /// @param type Type of property to read
  /// @param id Event ID (as raw 64-bit value)
  /// @param callback Optional progress callback
  /// @return true if operation started successfully
  bool read_event_property_async(event_property_type type, uint64_t id,
                                 progress_callback callback = nullptr);

  /// Execute a SQL query asynchronously
  /// @param query SQL query string
  /// @param description Description for the result table
  /// @param callback Optional progress callback
  /// @return Table ID that will be assigned, or 0 on failure
  uint64_t execute_query_async(const std::string &query,
                               const std::string &description = "",
                               progress_callback callback = nullptr);

  /// Wait for any pending async operation
  /// @param timeout_sec Timeout in seconds (0 = infinite)
  /// @return true if operation completed successfully
  bool wait(uint64_t timeout_sec = 0);

  /// Wait for any pending async operation with error information
  /// @param timeout_sec Timeout in seconds (0 = infinite)
  /// @return Status indicating success or error
  [[nodiscard]] status try_wait(uint64_t timeout_sec = 0);

  /// Cancel any pending async operation
  void cancel();

  // ==================== Query Building ====================

  /// Build a SQL query string from options
  /// @param options Query options
  /// @return SQL query string, or empty string on failure
  std::string build_table_query(const table_query_options &options) const;

  // ==================== Export / Save ====================

  /// Export query results to CSV file asynchronously
  /// @param query SQL query string
  /// @param file_path Output file path
  /// @param callback Optional progress callback
  /// @return true if operation started successfully
  bool export_to_csv_async(const std::string &query,
                           const std::string &file_path,
                           progress_callback callback = nullptr);

  /// Save a trimmed copy of the database asynchronously
  /// @param start Start timestamp
  /// @param end End timestamp
  /// @param new_path Path for the new database file
  /// @param callback Optional progress callback
  /// @return true if operation started successfully
  bool save_trimmed_async(uint64_t start, uint64_t end,
                          const std::string &new_path,
                          progress_callback callback = nullptr);

  // ==================== Memory ====================

  /// Get memory footprint of database structures
  /// @return Memory usage in bytes
  uint64_t memory_footprint() const;

  // ==================== C API Interop ====================

  /// Get the underlying C API database handle (for interop)
  rocprofvis_dm_database_t c_handle() const;

  /// Check if this database is valid (has internal handle)
  explicit operator bool() const { return internal_ != nullptr; }

private:
  friend class reader;

  database();

  struct Impl;
  std::unique_ptr<Impl> impl_;  // For Future and callback management
  rocprofvis_dm_database_t internal_;
  bool owning_;
};

} // namespace rocstorage
