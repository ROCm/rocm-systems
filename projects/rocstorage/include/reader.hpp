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
#include <rocstorage/enum_definitions.h>
#include <rocstorage/result.hpp>

// Forward declarations for C API interop
typedef void *rocprofvis_dm_trace_t;
typedef void *rocprofvis_dm_database_t;

namespace rocstorage {


/// Event ID combining 60-bit event ID and 4-bit operation type
struct event_id {
  uint64_t id : 60;
  uint64_t operation : 4;

  event_id() : id(0), operation(0) {}
  event_id(uint64_t event_id, event_operation op)
      : id(event_id), operation(static_cast<uint64_t>(op)) {}

  /// Get as raw 64-bit value (for C API interop)
  uint64_t raw() const { return (static_cast<uint64_t>(operation) << 60) | id; }

  /// Create from raw 64-bit value
  static event_id from_raw(uint64_t value) {
    event_id e;
    e.id = value & 0x0FFFFFFFFFFFFFFFULL;
    e.operation = value >> 60;
    return e;
  }
};

/// Progress callback for async operations
using progress_callback =
    std::function<void(const char *filename, uint16_t percent, bool success,
                       const char *message)>;

// Forward declarations
class track;
class flow_trace;
class stack_trace;
class ext_data;
class query_result;
class table_row;

/// Options for building a table query
struct table_query_options {
  uint64_t start_time = 0;
  uint64_t end_time = UINT64_MAX;
  std::vector<uint32_t> track_ids;
  std::string where_clause;
  std::string filter;
  std::string group_by;
  std::string group_columns;
  std::string sort_column;
  sort_order order = kRPVDMSortOrderAsc;
  std::vector<std::string> string_table_filters;
  uint64_t max_count = 0;
  uint64_t offset = 0;
  bool count_only = false;
  bool summary = false;
};

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

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/// Represents a flow endpoint in a flow trace
struct flow_endpoint {
  uint32_t track_id;
  uint64_t event_id;
  uint64_t start_timestamp;
  uint64_t end_timestamp;
  std::string category;
  std::string symbol;
  uint64_t level;
};

/// Flow trace showing connections between events
class flow_trace {
public:
  ~flow_trace();

  flow_trace(const flow_trace &) = delete;
  flow_trace &operator=(const flow_trace &) = delete;
  flow_trace(flow_trace &&) noexcept;
  flow_trace &operator=(flow_trace &&) noexcept;

  /// Get number of endpoints in this flow
  uint64_t num_endpoints() const;

  /// Get endpoint by index
  flow_endpoint get_endpoint(uint64_t index) const;

  /// Get all endpoints
  std::vector<flow_endpoint> get_endpoints() const;

private:
  friend class reader;
  explicit flow_trace(void *handle);

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/// Represents a frame in a stack trace
struct stack_frame {
  uint64_t depth;
  std::string symbol;
  std::string arguments;
  std::string code_line;
};

/// Stack trace for an event
class stack_trace {
public:
  ~stack_trace();

  stack_trace(const stack_trace &) = delete;
  stack_trace &operator=(const stack_trace &) = delete;
  stack_trace(stack_trace &&) noexcept;
  stack_trace &operator=(stack_trace &&) noexcept;

  /// Get number of frames in this stack trace
  uint64_t num_frames() const;

  /// Get frame by index
  stack_frame get_frame(uint64_t index) const;

  /// Get all frames
  std::vector<stack_frame> get_frames() const;

private:
  friend class reader;
  explicit stack_trace(void *handle);

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/// Represents a single extended data record
struct ext_data_record {
  std::string category;
  std::string name;
  std::string value;
  uint64_t type;
};

/// Extended data for an event
class ext_data {
public:
  ~ext_data();

  ext_data(const ext_data &) = delete;
  ext_data &operator=(const ext_data &) = delete;
  ext_data(ext_data &&) noexcept;
  ext_data &operator=(ext_data &&) noexcept;

  /// Get number of records
  uint64_t num_records() const;

  /// Get record by index
  ext_data_record get_record(uint64_t index) const;

  /// Get all records
  std::vector<ext_data_record> get_records() const;

private:
  friend class reader;
  explicit ext_data(void *handle);

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/// Represents a row in a query result table
class table_row {
public:
  ~table_row();

  table_row(const table_row &) = delete;
  table_row &operator=(const table_row &) = delete;
  table_row(table_row &&) noexcept;
  table_row &operator=(table_row &&) noexcept;

  /// Get number of cells in this row
  uint64_t num_cells() const;

  /// Get cell value by index as string
  std::string get_cell(uint64_t index) const;

  /// Get all cell values
  std::vector<std::string> get_cells() const;

private:
  friend class query_result;
  explicit table_row(void *handle);

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/// Result of a SQL query execution
class query_result {
public:
  ~query_result();

  query_result(const query_result &) = delete;
  query_result &operator=(const query_result &) = delete;
  query_result(query_result &&) noexcept;
  query_result &operator=(query_result &&) noexcept;

  /// Get table ID
  uint64_t id() const;

  /// Get description
  std::string description() const;

  /// Get the SQL query that produced this result
  std::string query() const;

  /// Get number of columns
  uint64_t num_columns() const;

  /// Get column name by index
  std::string column_name(uint64_t index) const;

  /// Get all column names
  std::vector<std::string> column_names() const;

  /// Get number of rows
  uint64_t num_rows() const;

  /// Get row by index
  std::unique_ptr<table_row> get_row(uint64_t index) const;

private:
  friend class reader;
  explicit query_result(void *handle);

  struct Impl;
  std::unique_ptr<Impl> impl_;
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
       database_type type = kAutodetect);

  /// Open a database file for reading with detailed error information
  /// @param path Path to the database file
  /// @param type Database type (autodetect by default)
  /// @return Result containing reader instance or error with details
  [[nodiscard]] static result<std::unique_ptr<reader>>
  try_open(const std::string &path,
           database_type type = kAutodetect);

  ~reader();

  reader(const reader &) = delete;
  reader &operator=(const reader &) = delete;
  reader(reader &&) noexcept;
  reader &operator=(reader &&) noexcept;

  /// Read metadata (tracks, time bounds) synchronously
  /// @return true on success
  bool read_metadata();

  /// Read metadata with detailed error information
  /// @return status indicating success or error with details
  [[nodiscard]] status try_read_metadata();

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

  /// Get track by index with error information
  /// @param index Track index (0 to num_tracks() - 1)
  /// @return Result with track object or error
  [[nodiscard]] result<std::unique_ptr<track>> try_get_track(uint64_t index) const;

  /// Get all tracks
  std::vector<std::unique_ptr<track>> get_tracks() const;

  /// Read a time slice for specified tracks
  /// @param start Start timestamp
  /// @param end End timestamp
  /// @param track_ids Vector of track IDs to read
  /// @return true on success
  bool read_slice(uint64_t start, uint64_t end,
                  const std::vector<uint32_t> &track_ids);

  /// Read a time slice with error information
  /// @param start Start timestamp
  /// @param end End timestamp
  /// @param track_ids Vector of track IDs to read
  /// @return status indicating success or error with details
  [[nodiscard]] status try_read_slice(uint64_t start, uint64_t end,
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

  // ==================== Memory Management ====================

  /// Delete a time slice by time range
  /// @param start Start timestamp of slice to delete
  /// @param end End timestamp of slice to delete
  /// @return true on success
  bool delete_slice(uint64_t start, uint64_t end);

  /// Delete a time slice by track ID and slice handle
  /// @param track_id Track ID
  /// @param slice_handle Opaque slice handle
  /// @return true on success
  bool delete_slice(uint32_t track_id, void *slice_handle);

  /// Delete all loaded time slices
  /// @return true on success
  bool delete_all_slices();

  /// Delete all loaded tables (from query results)
  /// @return true on success
  bool delete_all_tables();

  /// Delete a table by ID
  /// @param table_id Table ID to delete
  /// @return true on success
  bool delete_table(uint64_t table_id);

  // ==================== Event Properties ====================

  /// Read event property (flow trace, stack trace, or extended data)
  /// @param type Type of property to read
  /// @param id Event ID
  /// @return true on success
  bool read_event_property(event_property_type type, event_id id);

  /// Read event property asynchronously
  /// @param type Type of property to read
  /// @param id Event ID
  /// @param callback Optional progress callback
  /// @return true if operation started successfully
  bool read_event_property_async(event_property_type type, event_id id,
                                 progress_callback callback = nullptr);

  /// Get flow trace for an event (after reading)
  /// @param id Event ID
  /// @return Flow trace object, or nullptr if not available
  std::unique_ptr<flow_trace> get_flow_trace(event_id id) const;

  /// Get stack trace for an event (after reading)
  /// @param id Event ID
  /// @return Stack trace object, or nullptr if not available
  std::unique_ptr<stack_trace> get_stack_trace(event_id id) const;

  /// Get extended data for an event (after reading)
  /// @param id Event ID
  /// @return Extended data object, or nullptr if not available
  std::unique_ptr<ext_data> get_ext_data(event_id id) const;

  /// Delete event property
  /// @param type Type of property to delete
  /// @param id Event ID
  /// @return true on success
  bool delete_event_property(event_property_type type, event_id id);

  /// Delete all event properties of a given type
  /// @param type Type of properties to delete
  /// @return true on success
  bool delete_all_event_properties(event_property_type type);

  // ==================== Query Execution ====================

  /// Build a SQL query string from options
  /// @param options Query options
  /// @return SQL query string, or empty string on failure
  std::string build_table_query(const table_query_options &options) const;

  /// Execute a SQL query and store results
  /// @param query SQL query string
  /// @param description Description for the result table
  /// @return Table ID, or 0 on failure
  uint64_t execute_query(const std::string &query,
                         const std::string &description = "");

  /// Execute a SQL query asynchronously
  /// @param query SQL query string
  /// @param description Description for the result table
  /// @param callback Optional progress callback
  /// @return Table ID that will be assigned, or 0 on failure
  uint64_t execute_query_async(const std::string &query,
                               const std::string &description = "",
                               progress_callback callback = nullptr);

  /// Get number of stored query result tables
  uint64_t num_tables() const;

  /// Get a query result table by ID
  /// @param table_id Table ID
  /// @return Query result, or nullptr if not found
  std::unique_ptr<query_result> get_table(uint64_t table_id) const;

  // ==================== Export / Save ====================

  /// Export query results to CSV file
  /// @param query SQL query string
  /// @param file_path Output file path
  /// @return true on success
  bool export_to_csv(const std::string &query, const std::string &file_path);

  /// Export query results to CSV file asynchronously
  /// @param query SQL query string
  /// @param file_path Output file path
  /// @param callback Optional progress callback
  /// @return true if operation started successfully
  bool export_to_csv_async(const std::string &query,
                           const std::string &file_path,
                           progress_callback callback = nullptr);

  /// Save a trimmed copy of the database
  /// @param start Start timestamp
  /// @param end End timestamp
  /// @param new_path Path for the new database file
  /// @return true on success
  bool save_trimmed(uint64_t start, uint64_t end, const std::string &new_path);

  /// Save a trimmed copy of the database asynchronously
  /// @param start Start timestamp
  /// @param end End timestamp
  /// @param new_path Path for the new database file
  /// @param callback Optional progress callback
  /// @return true if operation started successfully
  bool save_trimmed_async(uint64_t start, uint64_t end,
                          const std::string &new_path,
                          progress_callback callback = nullptr);

  // ==================== C API Interop ====================

  /// Get the underlying C API trace handle (for interop)
  rocprofvis_dm_trace_t c_trace_handle() const;

  /// Get the underlying C API database handle (for interop)
  rocprofvis_dm_database_t c_database_handle() const;

private:
  reader() = default;

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace rocstorage