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
#include <memory>
#include <string>
#include <vector>

// Forward declaration for C API interop
typedef void *rocprofvis_dm_trace_t;

namespace rocstorage {

// Forward declarations
class track;
class flow_trace;
class stack_trace;
class ext_data;
class query_result;
struct event_id;

/// Trace wrapper for accessing profiling trace data.
///
/// This class wraps the internal RocProfVis::DataModel::Trace and provides
/// a public API for trace operations. It can be used as either:
/// - An owning wrapper (default constructor creates internal Trace)
/// - A non-owning view (pointer constructor wraps existing internal Trace)
///
/// Use this class directly when you need fine-grained control over trace
/// data, or use the `reader` class for a higher-level convenience API.
class trace {
public:
  /// Create an owning trace (creates internal Trace object)
  trace();

  /// Create a non-owning view of an existing internal Trace
  /// @param internal Pointer to internal Trace (caller retains ownership)
  explicit trace(rocprofvis_dm_trace_t internal);

  ~trace();

  trace(const trace &) = delete;
  trace &operator=(const trace &) = delete;
  trace(trace &&) noexcept;
  trace &operator=(trace &&) noexcept;

  // ==================== Time Bounds ====================

  /// Get trace start time
  /// @return Start timestamp in nanoseconds
  uint64_t start_time() const;

  /// Get trace end time
  /// @return End timestamp in nanoseconds
  uint64_t end_time() const;

  // ==================== Track Access ====================

  /// Get number of tracks
  /// @return Number of tracks in the trace
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
  /// @return Vector of all track objects
  std::vector<std::unique_ptr<track>> get_tracks() const;

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

  /// Delete a table by ID
  /// @param table_id Table ID to delete
  /// @return true on success
  bool delete_table(uint64_t table_id);

  /// Delete all loaded tables (from query results)
  /// @return true on success
  bool delete_all_tables();

  /// Delete event property
  /// @param type Type of property to delete
  /// @param id Event ID
  /// @return true on success
  bool delete_event_property(event_property_type type, event_id id);

  /// Delete all event properties of a given type
  /// @param type Type of properties to delete
  /// @return true on success
  bool delete_all_event_properties(event_property_type type);

  // ==================== Event Properties ====================

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

  // ==================== Query Results ====================

  /// Get number of stored query result tables
  /// @return Number of tables
  uint64_t num_tables() const;

  /// Get a query result table by ID
  /// @param table_id Table ID
  /// @return Query result, or nullptr if not found
  std::unique_ptr<query_result> get_table(uint64_t table_id) const;

  // ==================== Memory ====================

  /// Get memory footprint of trace data
  /// @return Memory usage in bytes
  uint64_t memory_footprint() const;

  // ==================== C API Interop ====================

  /// Get the underlying C API trace handle (for interop)
  rocprofvis_dm_trace_t c_handle() const;

  /// Check if this trace is valid (has internal handle)
  explicit operator bool() const { return internal_ != nullptr; }

private:
  friend class database;
  friend class reader;

  rocprofvis_dm_trace_t internal_;
  bool owning_;
};

} // namespace rocstorage
