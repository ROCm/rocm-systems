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

/// @file reader_test.cpp
/// @brief Unified tests for C and C++ reader APIs
///
/// This file tests both APIs using parameterized tests to ensure parity.
/// API-specific tests (like C++ move semantics) are included as regular tests.

#include "reader_test_adapters.hpp"

#include <rocstorage/reader.hpp>
#include <rocstorage/rocprofvis_c_interface.h>

#include <gtest/gtest.h>

#include <fstream>
#include <memory>
#include <string>
#include <type_traits>

#ifndef ROCSTORAGE_TEST_FIXTURE_PATH
#define ROCSTORAGE_TEST_FIXTURE_PATH ""
#endif

namespace {

using namespace rocstorage::testing;

// ============================================================================
// Test fixture path helper
// ============================================================================

std::string GetTestFixturePath() {
  return std::string(ROCSTORAGE_TEST_FIXTURE_PATH) + "/reader_test_trace.db";
}

bool TestFixtureExists() {
  std::ifstream f(GetTestFixturePath());
  return f.good();
}

// ============================================================================
// Parameterized tests - run against both C and C++ APIs
// ============================================================================

class ReaderParityTest : public ::testing::TestWithParam<ApiMode> {
protected:
  void SetUp() override {
    if (!TestFixtureExists()) {
      GTEST_SKIP() << "Test fixture not found: " << GetTestFixturePath()
                   << " (run 'make generate_test_fixtures' to create it)";
    }
    m_adapter = CreateAdapter(GetParam(), GetTestFixturePath());
  }

  std::unique_ptr<IReaderAdapter> m_adapter;
};

INSTANTIATE_TEST_SUITE_P(ApiParity, ReaderParityTest,
                         ::testing::Values(ApiMode::C, ApiMode::Cpp),
                         ParityTestNameGenerator);

TEST_P(ReaderParityTest, AdapterIsValid) { EXPECT_TRUE(m_adapter->is_valid()); }

TEST_P(ReaderParityTest, ReadMetadataSucceeds) {
  EXPECT_TRUE(m_adapter->read_metadata());
}

TEST_P(ReaderParityTest, StartTimeIsNonZeroAfterMetadata) {
  ASSERT_TRUE(m_adapter->read_metadata());
  EXPECT_GT(m_adapter->start_time(), 0u);
}

TEST_P(ReaderParityTest, EndTimeGreaterThanStartTime) {
  ASSERT_TRUE(m_adapter->read_metadata());
  EXPECT_GE(m_adapter->end_time(), m_adapter->start_time());
}

TEST_P(ReaderParityTest, NumTracksIsNonZero) {
  ASSERT_TRUE(m_adapter->read_metadata());
  EXPECT_GT(m_adapter->num_tracks(), 0u);
}

TEST_P(ReaderParityTest, TrackPropertiesAreValid) {
  ASSERT_TRUE(m_adapter->read_metadata());
  uint64_t num_tracks = m_adapter->num_tracks();
  ASSERT_GT(num_tracks, 0u);

  for (uint64_t i = 0; i < num_tracks; ++i) {
    EXPECT_GE(m_adapter->track_id(i), 0u);
    EXPECT_GE(m_adapter->track_category(i), 0);
    EXPECT_FALSE(m_adapter->track_category_string(i).empty());
  }
}

TEST_P(ReaderParityTest, DeleteAllSlicesSucceeds) {
  ASSERT_TRUE(m_adapter->read_metadata());
  EXPECT_TRUE(m_adapter->delete_all_slices());
}

TEST_P(ReaderParityTest, DeleteAllTablesSucceeds) {
  ASSERT_TRUE(m_adapter->read_metadata());
  EXPECT_TRUE(m_adapter->delete_all_tables());
}

TEST_P(ReaderParityTest, DeleteAllEventPropertiesSucceeds) {
  ASSERT_TRUE(m_adapter->read_metadata());
  EXPECT_TRUE(m_adapter->delete_all_event_properties(kRPVDMEventFlowTrace));
  EXPECT_TRUE(m_adapter->delete_all_event_properties(kRPVDMEventStackTrace));
  EXPECT_TRUE(m_adapter->delete_all_event_properties(kRPVDMEventExtData));
}

TEST_P(ReaderParityTest, ReadSliceSucceeds) {
  ASSERT_TRUE(m_adapter->read_metadata());

  uint64_t num_tracks = m_adapter->num_tracks();
  ASSERT_GT(num_tracks, 0u);

  std::vector<uint32_t> track_ids;
  for (uint64_t i = 0; i < num_tracks; ++i) {
    track_ids.push_back(m_adapter->track_id(i));
  }

  uint64_t start = m_adapter->start_time();
  uint64_t end = m_adapter->end_time();
  EXPECT_TRUE(m_adapter->read_slice(start, end, track_ids));
}

TEST_P(ReaderParityTest, MemoryFootprintIsNonNegative) {
  ASSERT_TRUE(m_adapter->read_metadata());
  EXPECT_GE(m_adapter->memory_footprint(), 0u);
}

TEST_P(ReaderParityTest, CancelDoesNotCrash) {
  // Cancel should not crash even without pending operation
  m_adapter->cancel();
}

// ============================================================================
// Direct parity tests - compare C and C++ API results
// ============================================================================

class DirectParityTest : public ::testing::Test {
protected:
  void SetUp() override {
    if (!TestFixtureExists()) {
      GTEST_SKIP() << "Test fixture not found: " << GetTestFixturePath()
                   << " (run 'make generate_test_fixtures' to create it)";
    }
    m_cpp = CreateAdapter(ApiMode::Cpp, GetTestFixturePath());
    m_c = CreateAdapter(ApiMode::C, GetTestFixturePath());
    ASSERT_TRUE(m_cpp->is_valid());
    ASSERT_TRUE(m_c->is_valid());
    ASSERT_TRUE(m_cpp->read_metadata());
    ASSERT_TRUE(m_c->read_metadata());
  }

  std::unique_ptr<IReaderAdapter> m_cpp;
  std::unique_ptr<IReaderAdapter> m_c;
};

TEST_F(DirectParityTest, StartTimeMatches) {
  EXPECT_EQ(m_cpp->start_time(), m_c->start_time());
}

TEST_F(DirectParityTest, EndTimeMatches) {
  EXPECT_EQ(m_cpp->end_time(), m_c->end_time());
}

TEST_F(DirectParityTest, NumTracksMatches) {
  EXPECT_EQ(m_cpp->num_tracks(), m_c->num_tracks());
}

TEST_F(DirectParityTest, NumTablesMatches) {
  EXPECT_EQ(m_cpp->num_tables(), m_c->num_tables());
}

TEST_F(DirectParityTest, MemoryFootprintMatches) {
  EXPECT_EQ(m_cpp->memory_footprint(), m_c->memory_footprint());
}

TEST_F(DirectParityTest, AllTrackIdsMatch) {
  uint64_t num_tracks = m_cpp->num_tracks();
  ASSERT_EQ(num_tracks, m_c->num_tracks());

  for (uint64_t i = 0; i < num_tracks; ++i) {
    EXPECT_EQ(m_cpp->track_id(i), m_c->track_id(i)) << "Track index: " << i;
  }
}

TEST_F(DirectParityTest, AllTrackCategoriesMatch) {
  uint64_t num_tracks = m_cpp->num_tracks();
  for (uint64_t i = 0; i < num_tracks; ++i) {
    EXPECT_EQ(m_cpp->track_category(i), m_c->track_category(i))
        << "Track index: " << i;
  }
}

TEST_F(DirectParityTest, AllTrackCategoryStringsMatch) {
  uint64_t num_tracks = m_cpp->num_tracks();
  for (uint64_t i = 0; i < num_tracks; ++i) {
    EXPECT_EQ(m_cpp->track_category_string(i), m_c->track_category_string(i))
        << "Track index: " << i;
  }
}

TEST_F(DirectParityTest, AllTrackNodeIdsMatch) {
  uint64_t num_tracks = m_cpp->num_tracks();
  for (uint64_t i = 0; i < num_tracks; ++i) {
    EXPECT_EQ(m_cpp->track_node_id(i), m_c->track_node_id(i))
        << "Track index: " << i;
  }
}

TEST_F(DirectParityTest, AllTrackProcessNamesMatch) {
  uint64_t num_tracks = m_cpp->num_tracks();
  for (uint64_t i = 0; i < num_tracks; ++i) {
    EXPECT_EQ(m_cpp->track_process_name(i), m_c->track_process_name(i))
        << "Track index: " << i;
  }
}

TEST_F(DirectParityTest, AllTrackSubprocessNamesMatch) {
  uint64_t num_tracks = m_cpp->num_tracks();
  for (uint64_t i = 0; i < num_tracks; ++i) {
    EXPECT_EQ(m_cpp->track_subprocess_name(i), m_c->track_subprocess_name(i))
        << "Track index: " << i;
  }
}

TEST_F(DirectParityTest, AllTrackNumRecordsMatch) {
  uint64_t num_tracks = m_cpp->num_tracks();
  for (uint64_t i = 0; i < num_tracks; ++i) {
    EXPECT_EQ(m_cpp->track_num_records(i), m_c->track_num_records(i))
        << "Track index: " << i;
  }
}

TEST_F(DirectParityTest, AllTrackTimestampRangesMatch) {
  uint64_t num_tracks = m_cpp->num_tracks();
  for (uint64_t i = 0; i < num_tracks; ++i) {
    EXPECT_EQ(m_cpp->track_min_timestamp(i), m_c->track_min_timestamp(i))
        << "Track index: " << i;
    EXPECT_EQ(m_cpp->track_max_timestamp(i), m_c->track_max_timestamp(i))
        << "Track index: " << i;
  }
}

// ============================================================================
// C++ API-specific tests
// ============================================================================

class CppApiTest : public ::testing::Test {};

TEST_F(CppApiTest, OpenReturnsNullptrForNonexistentFile) {
  auto reader = rocstorage::reader::open("nonexistent_file.db");
  EXPECT_EQ(reader, nullptr);
}

TEST_F(CppApiTest, OpenReturnsNullptrForInvalidPath) {
  auto reader = rocstorage::reader::open("");
  EXPECT_EQ(reader, nullptr);
}

TEST_F(CppApiTest, ReaderIsMoveConstructible) {
  EXPECT_TRUE(std::is_move_constructible<rocstorage::reader>::value);
  EXPECT_TRUE(std::is_move_assignable<rocstorage::reader>::value);
  EXPECT_FALSE(std::is_copy_constructible<rocstorage::reader>::value);
  EXPECT_FALSE(std::is_copy_assignable<rocstorage::reader>::value);
}

TEST_F(CppApiTest, TrackIsMoveConstructible) {
  EXPECT_TRUE(std::is_move_constructible<rocstorage::track>::value);
  EXPECT_TRUE(std::is_move_assignable<rocstorage::track>::value);
  EXPECT_FALSE(std::is_copy_constructible<rocstorage::track>::value);
  EXPECT_FALSE(std::is_copy_assignable<rocstorage::track>::value);
}

class CppIntegrationTest : public ::testing::Test {
protected:
  void SetUp() override {
    if (!TestFixtureExists()) {
      GTEST_SKIP() << "Test fixture not found: " << GetTestFixturePath()
                   << " (run 'make generate_test_fixtures' to create it)";
    }
  }
};

TEST_F(CppIntegrationTest, OpenSucceedsForValidDatabase) {
  auto reader = rocstorage::reader::open(GetTestFixturePath());
  ASSERT_NE(reader, nullptr);
}

TEST_F(CppIntegrationTest, CHandlesAreAccessible) {
  auto reader = rocstorage::reader::open(GetTestFixturePath());
  ASSERT_NE(reader, nullptr);

  EXPECT_NE(reader->c_trace_handle(), nullptr);
  EXPECT_NE(reader->c_database_handle(), nullptr);
}

TEST_F(CppIntegrationTest, GetTracksReturnsVector) {
  auto reader = rocstorage::reader::open(GetTestFixturePath());
  ASSERT_NE(reader, nullptr);
  ASSERT_TRUE(reader->read_metadata());

  auto tracks = reader->get_tracks();
  EXPECT_EQ(tracks.size(), reader->num_tracks());
}

TEST_F(CppIntegrationTest, GetTrackReturnsNullptrForInvalidIndex) {
  auto reader = rocstorage::reader::open(GetTestFixturePath());
  ASSERT_NE(reader, nullptr);
  ASSERT_TRUE(reader->read_metadata());

  auto track = reader->get_track(99999);
  EXPECT_EQ(track, nullptr);
}

TEST_F(CppIntegrationTest, WaitReturnsFalseWithoutPendingOperation) {
  auto reader = rocstorage::reader::open(GetTestFixturePath());
  ASSERT_NE(reader, nullptr);

  bool result = reader->wait(1);
  EXPECT_FALSE(result);
}

TEST_F(CppIntegrationTest, ReadSliceWithEmptyTrackIdsReturnsFalse) {
  auto reader = rocstorage::reader::open(GetTestFixturePath());
  ASSERT_NE(reader, nullptr);
  ASSERT_TRUE(reader->read_metadata());

  std::vector<uint32_t> empty_ids;
  bool result = reader->read_slice(0, 1000000, empty_ids);
  EXPECT_FALSE(result);
}

// ============================================================================
// C API-specific tests
// ============================================================================

class CApiTest : public ::testing::Test {};

TEST_F(CApiTest, CreateTraceReturnsValidHandle) {
  rocprofvis_dm_trace_t trace = rocprofvis_dm_create_trace();
  ASSERT_NE(trace, nullptr);
  rocprofvis_dm_delete_trace(trace);
}

TEST_F(CApiTest, DeleteTraceSucceeds) {
  rocprofvis_dm_trace_t trace = rocprofvis_dm_create_trace();
  ASSERT_NE(trace, nullptr);
  auto result = rocprofvis_dm_delete_trace(trace);
  EXPECT_EQ(result, kRocProfVisDmResultSuccess);
}

TEST_F(CApiTest, BindToNullDatabaseFails) {
  // The function asserts on null database parameter, which terminates the process.
  // Use EXPECT_DEATH to verify this defensive behavior.
  rocprofvis_dm_trace_t trace = rocprofvis_dm_create_trace();
  ASSERT_NE(trace, nullptr);

  EXPECT_DEATH(rocprofvis_dm_bind_trace_to_database(trace, nullptr),
               "DATABASE_CANNOT_BE_NULL");

  rocprofvis_dm_delete_trace(trace);
}

TEST_F(CApiTest, FutureAllocAndFreeDoNotCrash) {
  auto future = rocprofvis_db_future_alloc(nullptr, nullptr);
  ASSERT_NE(future, nullptr);
  rocprofvis_db_future_free(future);
}

TEST_F(CApiTest, FutureCancelDoesNotCrash) {
  auto future = rocprofvis_db_future_alloc(nullptr, nullptr);
  ASSERT_NE(future, nullptr);
  rocprofvis_db_future_cancel(future);
  rocprofvis_db_future_free(future);
}

} // namespace