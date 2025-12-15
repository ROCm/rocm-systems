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

/// @file reader_test_adapters.hpp
/// @brief Adapter infrastructure for testing C and C++ reader APIs uniformly
///
/// This file provides an abstract interface (IReaderAdapter) and concrete
/// implementations for both the C and C++ APIs. Tests can use parameterized
/// testing to verify both APIs produce equivalent results.

#include <rocstorage/reader.hpp>
#include <rocstorage/rocprofvis_c_interface.h>

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

namespace rocstorage::testing {

/// API mode for parameterized tests
enum class ApiMode { C, Cpp };

inline std::string ApiModeToString(ApiMode mode) {
  switch (mode) {
  case ApiMode::C:
    return "C_API";
  case ApiMode::Cpp:
    return "Cpp_API";
  }
  return "Unknown";
}

/// Abstract interface for reader operations.
/// Both C and C++ APIs are wrapped to implement this interface.
class IReaderAdapter {
public:
  virtual ~IReaderAdapter() = default;

  virtual bool is_valid() const = 0;
  virtual bool read_metadata() = 0;
  virtual bool wait(uint64_t timeout_sec) = 0;
  virtual void cancel() = 0;

  virtual uint64_t start_time() const = 0;
  virtual uint64_t end_time() const = 0;
  virtual uint64_t num_tracks() const = 0;
  virtual uint64_t memory_footprint() const = 0;
  virtual uint64_t num_tables() const = 0;

  // Track info
  virtual uint32_t track_id(uint64_t index) const = 0;
  virtual int track_category(uint64_t index) const = 0;
  virtual std::string track_category_string(uint64_t index) const = 0;
  virtual uint64_t track_node_id(uint64_t index) const = 0;
  virtual std::string track_process_name(uint64_t index) const = 0;
  virtual std::string track_subprocess_name(uint64_t index) const = 0;
  virtual uint64_t track_num_records(uint64_t index) const = 0;
  virtual uint64_t track_min_timestamp(uint64_t index) const = 0;
  virtual uint64_t track_max_timestamp(uint64_t index) const = 0;

  // Memory management
  virtual bool delete_all_slices() = 0;
  virtual bool delete_all_tables() = 0;
  virtual bool delete_all_event_properties(int type) = 0;

  // Slice reading
  virtual bool read_slice(uint64_t start, uint64_t end,
                          const std::vector<uint32_t> &track_ids) = 0;
};

/// C++ API adapter
class CppReaderAdapter : public IReaderAdapter {
public:
  explicit CppReaderAdapter(const std::string &path) {
    m_reader = rocstorage::reader::open(path);
  }

  bool is_valid() const override { return m_reader != nullptr; }

  bool read_metadata() override {
    return m_reader ? m_reader->read_metadata() : false;
  }

  bool wait(uint64_t timeout_sec) override {
    return m_reader ? m_reader->wait(timeout_sec) : false;
  }

  void cancel() override {
    if (m_reader)
      m_reader->cancel();
  }

  uint64_t start_time() const override {
    return m_reader ? m_reader->start_time() : 0;
  }

  uint64_t end_time() const override {
    return m_reader ? m_reader->end_time() : 0;
  }

  uint64_t num_tracks() const override {
    return m_reader ? m_reader->num_tracks() : 0;
  }

  uint64_t memory_footprint() const override {
    return m_reader ? m_reader->memory_footprint() : 0;
  }

  uint64_t num_tables() const override {
    return m_reader ? m_reader->num_tables() : 0;
  }

  uint32_t track_id(uint64_t index) const override {
    if (!m_reader)
      return 0;
    auto track = m_reader->get_track(index);
    return track ? track->id() : 0;
  }

  int track_category(uint64_t index) const override {
    if (!m_reader)
      return 0;
    auto track = m_reader->get_track(index);
    return track ? static_cast<int>(track->category()) : 0;
  }

  std::string track_category_string(uint64_t index) const override {
    if (!m_reader)
      return "";
    auto track = m_reader->get_track(index);
    return track ? track->category_string() : "";
  }

  uint64_t track_node_id(uint64_t index) const override {
    if (!m_reader)
      return 0;
    auto track = m_reader->get_track(index);
    return track ? track->node_id() : 0;
  }

  std::string track_process_name(uint64_t index) const override {
    if (!m_reader)
      return "";
    auto track = m_reader->get_track(index);
    return track ? track->process_name() : "";
  }

  std::string track_subprocess_name(uint64_t index) const override {
    if (!m_reader)
      return "";
    auto track = m_reader->get_track(index);
    return track ? track->subprocess_name() : "";
  }

  uint64_t track_num_records(uint64_t index) const override {
    if (!m_reader)
      return 0;
    auto track = m_reader->get_track(index);
    return track ? track->num_records() : 0;
  }

  uint64_t track_min_timestamp(uint64_t index) const override {
    if (!m_reader)
      return 0;
    auto track = m_reader->get_track(index);
    return track ? track->min_timestamp() : 0;
  }

  uint64_t track_max_timestamp(uint64_t index) const override {
    if (!m_reader)
      return 0;
    auto track = m_reader->get_track(index);
    return track ? track->max_timestamp() : 0;
  }

  bool delete_all_slices() override {
    return m_reader ? m_reader->delete_all_slices() : false;
  }

  bool delete_all_tables() override {
    return m_reader ? m_reader->delete_all_tables() : false;
  }

  bool delete_all_event_properties(int type) override {
    return m_reader ? m_reader->delete_all_event_properties(
                          static_cast<rocstorage::event_property_type>(type))
                    : false;
  }

  bool read_slice(uint64_t start, uint64_t end,
                  const std::vector<uint32_t> &track_ids) override {
    return m_reader ? m_reader->read_slice(start, end, track_ids) : false;
  }

  // Access to underlying C++ reader for API-specific tests
  rocstorage::reader *get_reader() { return m_reader.get(); }

private:
  std::unique_ptr<rocstorage::reader> m_reader;
};

/// C API adapter
class CReaderAdapter : public IReaderAdapter {
public:
  explicit CReaderAdapter(const std::string &path) {
    m_db = rocprofvis_db_open_database(path.c_str(), kAutodetect);
    if (m_db) {
      m_trace = rocprofvis_dm_create_trace();
      if (m_trace) {
        if (rocprofvis_dm_bind_trace_to_database(m_trace, m_db) !=
            kRocProfVisDmResultSuccess) {
          rocprofvis_dm_delete_trace(m_trace);
          m_trace = nullptr;
        }
      }
    }
    m_future = rocprofvis_db_future_alloc(nullptr, nullptr);
  }

  ~CReaderAdapter() override {
    if (m_future) {
      rocprofvis_db_future_free(m_future);
    }
    if (m_trace) {
      rocprofvis_dm_delete_trace(m_trace);
    }
  }

  bool is_valid() const override {
    return m_trace != nullptr && m_db != nullptr;
  }

  bool read_metadata() override {
    if (!is_valid() || !m_future)
      return false;
    auto result = rocprofvis_db_read_metadata_async(m_db, m_future);
    if (result != kRocProfVisDmResultSuccess)
      return false;
    result = rocprofvis_db_future_wait(m_future, 30);
    return result == kRocProfVisDmResultSuccess;
  }

  bool wait(uint64_t timeout_sec) override {
    if (!m_future)
      return false;
    auto result = rocprofvis_db_future_wait(m_future, timeout_sec);
    return result == kRocProfVisDmResultSuccess;
  }

  void cancel() override {
    if (m_future) {
      rocprofvis_db_future_cancel(m_future);
    }
  }

  uint64_t start_time() const override {
    return rocprofvis_dm_get_property_as_uint64(m_trace, kRPVDMStartTimeUInt64,
                                                0);
  }

  uint64_t end_time() const override {
    return rocprofvis_dm_get_property_as_uint64(m_trace, kRPVDMEndTimeUInt64, 0);
  }

  uint64_t num_tracks() const override {
    return rocprofvis_dm_get_property_as_uint64(m_trace,
                                                kRPVDMNumberOfTracksUInt64, 0);
  }

  uint64_t memory_footprint() const override {
    return rocprofvis_dm_get_property_as_uint64(
        m_trace, kRPVDMTraceMemoryFootprintUInt64, 0);
  }

  uint64_t num_tables() const override {
    return rocprofvis_dm_get_property_as_uint64(m_trace,
                                                kRPVDMNumberOfTablesUInt64, 0);
  }

  uint32_t track_id(uint64_t index) const override {
    auto track = get_track_handle(index);
    if (!track)
      return 0;
    return static_cast<uint32_t>(
        rocprofvis_dm_get_property_as_uint64(track, kRPVDMTrackIdUInt64, 0));
  }

  int track_category(uint64_t index) const override {
    auto track = get_track_handle(index);
    if (!track)
      return 0;
    return static_cast<int>(rocprofvis_dm_get_property_as_uint64(
        track, kRPVDMTrackCategoryEnumUInt64, 0));
  }

  std::string track_category_string(uint64_t index) const override {
    auto track = get_track_handle(index);
    if (!track)
      return "";
    const char *str = rocprofvis_dm_get_property_as_charptr(
        track, kRPVDMTrackCategoryEnumCharPtr, 0);
    return str ? str : "";
  }

  uint64_t track_node_id(uint64_t index) const override {
    auto track = get_track_handle(index);
    if (!track)
      return 0;
    return rocprofvis_dm_get_property_as_uint64(track, kRPVDMTrackNodeIdUInt64,
                                                0);
  }

  std::string track_process_name(uint64_t index) const override {
    auto track = get_track_handle(index);
    if (!track)
      return "";
    const char *str = rocprofvis_dm_get_property_as_charptr(
        track, kRPVDMTrackMainProcessNameCharPtr, 0);
    return str ? str : "";
  }

  std::string track_subprocess_name(uint64_t index) const override {
    auto track = get_track_handle(index);
    if (!track)
      return "";
    const char *str = rocprofvis_dm_get_property_as_charptr(
        track, kRPVDMTrackSubProcessNameCharPtr, 0);
    return str ? str : "";
  }

  uint64_t track_num_records(uint64_t index) const override {
    auto track = get_track_handle(index);
    if (!track)
      return 0;
    return rocprofvis_dm_get_property_as_uint64(track,
                                                kRPVDMTrackNumRecordsUInt64, 0);
  }

  uint64_t track_min_timestamp(uint64_t index) const override {
    auto track = get_track_handle(index);
    if (!track)
      return 0;
    return rocprofvis_dm_get_property_as_uint64(
        track, kRPVDMTrackMinimumTimestampUInt64, 0);
  }

  uint64_t track_max_timestamp(uint64_t index) const override {
    auto track = get_track_handle(index);
    if (!track)
      return 0;
    return rocprofvis_dm_get_property_as_uint64(
        track, kRPVDMTrackMaximumTimestampUInt64, 0);
  }

  bool delete_all_slices() override {
    return rocprofvis_dm_delete_all_time_slices(m_trace) ==
           kRocProfVisDmResultSuccess;
  }

  bool delete_all_tables() override {
    return rocprofvis_dm_delete_all_tables(m_trace) ==
           kRocProfVisDmResultSuccess;
  }

  bool delete_all_event_properties(int type) override {
    return rocprofvis_dm_delete_all_event_properties_for(
               m_trace,
               static_cast<rocprofvis_dm_event_property_type_t>(type)) ==
           kRocProfVisDmResultSuccess;
  }

  bool read_slice(uint64_t start, uint64_t end,
                  const std::vector<uint32_t> &track_ids) override {
    if (!is_valid() || !m_future || track_ids.empty())
      return false;

    auto result = rocprofvis_db_read_trace_slice_async(
        m_db, start, end, static_cast<uint16_t>(track_ids.size()),
        const_cast<uint32_t *>(track_ids.data()), m_future);

    if (result != kRocProfVisDmResultSuccess)
      return false;

    result = rocprofvis_db_future_wait(m_future, 30);
    return result == kRocProfVisDmResultSuccess;
  }

  // Access to underlying C handles for API-specific tests
  rocprofvis_dm_trace_t get_trace() { return m_trace; }
  rocprofvis_dm_database_t get_database() { return m_db; }

private:
  rocprofvis_dm_handle_t get_track_handle(uint64_t index) const {
    return rocprofvis_dm_get_property_as_handle(m_trace, kRPVDMTrackHandleIndexed,
                                                index);
  }

  rocprofvis_dm_database_t m_db = nullptr;
  rocprofvis_dm_trace_t m_trace = nullptr;
  rocprofvis_db_future_t m_future = nullptr;
};

/// Factory to create the appropriate adapter
inline std::unique_ptr<IReaderAdapter> CreateAdapter(ApiMode mode,
                                                     const std::string &path) {
  switch (mode) {
  case ApiMode::C:
    return std::make_unique<CReaderAdapter>(path);
  case ApiMode::Cpp:
    return std::make_unique<CppReaderAdapter>(path);
  }
  return nullptr;
}

/// Custom name generator for parameterized tests
inline std::string ParityTestNameGenerator(
    const ::testing::TestParamInfo<ApiMode> &info) {
  return ApiModeToString(info.param);
}

} // namespace rocstorage::testing