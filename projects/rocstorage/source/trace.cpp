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

#include <rocstorage/trace.hpp>
#include <rocstorage/reader.hpp>

#include "error.hpp"
#include "reader/datamodel/internal_types.h"
#include "reader/datamodel/rocprofvis_dm_trace.h"

namespace rocstorage {

using namespace RocProfVis::DataModel;

// Helper to get typed internal pointer
static inline Trace* internal(rocprofvis_dm_trace_t handle) {
  return static_cast<Trace*>(handle);
}

// Owning constructor - creates new internal Trace
trace::trace() : internal_(new Trace()), owning_(true) {}

// Non-owning view constructor
trace::trace(rocprofvis_dm_trace_t handle) : internal_(handle), owning_(false) {}

trace::~trace() {
  if (owning_ && internal_) {
    delete internal(internal_);
  }
}

trace::trace(trace &&other) noexcept
    : internal_(other.internal_), owning_(other.owning_) {
  other.internal_ = nullptr;
  other.owning_ = false;
}

trace &trace::operator=(trace &&other) noexcept {
  if (this != &other) {
    if (owning_ && internal_) {
      delete internal(internal_);
    }
    internal_ = other.internal_;
    owning_ = other.owning_;
    other.internal_ = nullptr;
    other.owning_ = false;
  }
  return *this;
}

uint64_t trace::start_time() const {
  if (!internal_) return 0;
  return internal(internal_)->StartTime();
}

uint64_t trace::end_time() const {
  if (!internal_) return 0;
  return internal(internal_)->EndTime();
}

uint64_t trace::num_tracks() const {
  if (!internal_) return 0;
  return internal(internal_)->NumberOfTracks();
}

std::unique_ptr<track> trace::get_track(uint64_t index) const {
  if (!internal_ || index >= num_tracks()) {
    return nullptr;
  }

  rocprofvis_dm_track_t handle = nullptr;
  auto result = internal(internal_)->GetPropertyAsHandle(
      kRPVDMTrackHandleIndexed, index, &handle);
  if (result != kRocProfVisDmResultSuccess || !handle) {
    return nullptr;
  }

  return std::unique_ptr<track>(new track(handle));
}

result<std::unique_ptr<track>> trace::try_get_track(uint64_t index) const {
  if (!internal_) {
    return error(error_code::invalid_parameter, "Trace not initialized");
  }
  if (index >= num_tracks()) {
    return error(error_code::index_out_of_bounds,
                 "Track index " + std::to_string(index) + " out of range (0-" +
                     std::to_string(num_tracks() - 1) + ")");
  }

  rocprofvis_dm_track_t handle = nullptr;
  auto c_result = internal(internal_)->GetPropertyAsHandle(
      kRPVDMTrackHandleIndexed, index, &handle);
  if (c_result != kRocProfVisDmResultSuccess) {
    return from_c_result(c_result,
                         "Failed to get track at index " + std::to_string(index));
  }
  if (!handle) {
    return error(error_code::invalid_property,
                 "Track handle is null for index " + std::to_string(index));
  }

  return std::unique_ptr<track>(new track(handle));
}

std::vector<std::unique_ptr<track>> trace::get_tracks() const {
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

bool trace::delete_slice(uint64_t start, uint64_t end) {
  if (!internal_) return false;
  return internal(internal_)->DeleteSliceAtTimeRange(start, end) ==
         kRocProfVisDmResultSuccess;
}

bool trace::delete_slice(uint32_t track_id, void *slice_handle) {
  if (!internal_) return false;
  return internal(internal_)->DeleteSliceByHandle(track_id, slice_handle) ==
         kRocProfVisDmResultSuccess;
}

bool trace::delete_all_slices() {
  if (!internal_) return false;
  return internal(internal_)->DeleteAllSlices() == kRocProfVisDmResultSuccess;
}

bool trace::delete_table(uint64_t table_id) {
  if (!internal_) return false;
  return internal(internal_)->DeleteTableAt(table_id) == kRocProfVisDmResultSuccess;
}

bool trace::delete_all_tables() {
  if (!internal_) return false;
  return internal(internal_)->DeleteAllTables() == kRocProfVisDmResultSuccess;
}

bool trace::delete_event_property(event_property_type type, event_id id) {
  if (!internal_) return false;

  rocprofvis_dm_event_id_t c_id;
  c_id.value = id.raw();

  return internal(internal_)->DeleteEventPropertyFor(
             static_cast<rocprofvis_dm_event_property_type_t>(type), c_id) ==
         kRocProfVisDmResultSuccess;
}

bool trace::delete_all_event_properties(event_property_type type) {
  if (!internal_) return false;

  return internal(internal_)->DeleteAllEventPropertiesFor(
             static_cast<rocprofvis_dm_event_property_type_t>(type)) ==
         kRocProfVisDmResultSuccess;
}

std::unique_ptr<flow_trace> trace::get_flow_trace(event_id id) const {
  if (!internal_) return nullptr;

  rocprofvis_dm_handle_t handle = nullptr;
  auto result = internal(internal_)->GetPropertyAsHandle(
      kRPVDMFlowTraceHandleByEventID, id.raw(), &handle);

  if (result != kRocProfVisDmResultSuccess || !handle) {
    return nullptr;
  }

  return std::unique_ptr<flow_trace>(new flow_trace(handle));
}

std::unique_ptr<stack_trace> trace::get_stack_trace(event_id id) const {
  if (!internal_) return nullptr;

  rocprofvis_dm_handle_t handle = nullptr;
  auto result = internal(internal_)->GetPropertyAsHandle(
      kRPVDMStackTraceHandleByEventID, id.raw(), &handle);

  if (result != kRocProfVisDmResultSuccess || !handle) {
    return nullptr;
  }

  return std::unique_ptr<stack_trace>(new stack_trace(handle));
}

std::unique_ptr<ext_data> trace::get_ext_data(event_id id) const {
  if (!internal_) return nullptr;

  rocprofvis_dm_handle_t handle = nullptr;
  auto result = internal(internal_)->GetPropertyAsHandle(
      kRPVDMExtInfoHandleByEventID, id.raw(), &handle);

  if (result != kRocProfVisDmResultSuccess || !handle) {
    return nullptr;
  }

  return std::unique_ptr<ext_data>(new ext_data(handle));
}

uint64_t trace::num_tables() const {
  if (!internal_) return 0;

  uint64_t count = 0;
  internal(internal_)->GetPropertyAsUint64(kRPVDMNumberOfTablesUInt64, 0, &count);
  return count;
}

std::unique_ptr<query_result> trace::get_table(uint64_t table_id) const {
  if (!internal_) return nullptr;

  rocprofvis_dm_handle_t handle = nullptr;
  auto result = internal(internal_)->GetPropertyAsHandle(
      kRPVDMTableHandleByID, table_id, &handle);

  if (result != kRocProfVisDmResultSuccess || !handle) {
    return nullptr;
  }

  return std::unique_ptr<query_result>(new query_result(handle));
}

uint64_t trace::memory_footprint() const {
  if (!internal_) return 0;
  uint64_t footprint = 0;
  internal(internal_)->GetPropertyAsUint64(kRPVDMTraceMemoryFootprintUInt64, 0,
                                           &footprint);
  return footprint;
}

rocprofvis_dm_trace_t trace::c_handle() const {
  return internal_;
}

} // namespace rocstorage
