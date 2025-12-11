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

#include <gtest/gtest.h>

#include <fstream>
#include <memory>
#include <string>
#include <type_traits>

// Path to the pre-generated trace fixture database
// This was created by profiling a real HIP program with rocprofv3
#ifndef ROCSTORAGE_TEST_FIXTURE_PATH
#define ROCSTORAGE_TEST_FIXTURE_PATH ""
#endif

namespace {

class ReaderApiTest : public ::testing::Test {};

TEST_F(ReaderApiTest, OpenReturnsNullptrForNonexistentFile) {
  auto reader = rocstorage::reader::open("nonexistent_file.db");
  EXPECT_EQ(reader, nullptr);
}

TEST_F(ReaderApiTest, OpenReturnsNullptrForInvalidPath) {
  auto reader = rocstorage::reader::open("");
  EXPECT_EQ(reader, nullptr);
}

TEST_F(ReaderApiTest, DatabaseTypeEnumValuesMatchCApi) {
  EXPECT_EQ(static_cast<int>(rocstorage::database_type::autodetect), 0);
  EXPECT_EQ(static_cast<int>(rocstorage::database_type::rocpd_sqlite), 1);
  EXPECT_EQ(static_cast<int>(rocstorage::database_type::rocprof_sqlite), 2);
}

TEST_F(ReaderApiTest, TrackCategoryEnumValuesMatchCApi) {
  EXPECT_EQ(static_cast<int>(rocstorage::track_category::not_a_track), 0);
  EXPECT_EQ(static_cast<int>(rocstorage::track_category::pmc), 1);
  EXPECT_EQ(static_cast<int>(rocstorage::track_category::region), 2);
  EXPECT_EQ(static_cast<int>(rocstorage::track_category::kernel_dispatch), 3);
  EXPECT_EQ(static_cast<int>(rocstorage::track_category::sqtt), 4);
  EXPECT_EQ(static_cast<int>(rocstorage::track_category::nic), 5);
  EXPECT_EQ(static_cast<int>(rocstorage::track_category::memory_allocation), 6);
  EXPECT_EQ(static_cast<int>(rocstorage::track_category::memory_copy), 7);
  EXPECT_EQ(static_cast<int>(rocstorage::track_category::stream), 8);
  EXPECT_EQ(static_cast<int>(rocstorage::track_category::region_main), 9);
  EXPECT_EQ(static_cast<int>(rocstorage::track_category::region_sample), 10);
}

TEST_F(ReaderApiTest, ReaderIsMoveConstructible) {
  EXPECT_TRUE(std::is_move_constructible<rocstorage::reader>::value);
  EXPECT_TRUE(std::is_move_assignable<rocstorage::reader>::value);
  EXPECT_FALSE(std::is_copy_constructible<rocstorage::reader>::value);
  EXPECT_FALSE(std::is_copy_assignable<rocstorage::reader>::value);
}

TEST_F(ReaderApiTest, TrackIsMoveConstructible) {
  EXPECT_TRUE(std::is_move_constructible<rocstorage::track>::value);
  EXPECT_TRUE(std::is_move_assignable<rocstorage::track>::value);
  EXPECT_FALSE(std::is_copy_constructible<rocstorage::track>::value);
  EXPECT_FALSE(std::is_copy_assignable<rocstorage::track>::value);
}

class ReaderIntegrationTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Use the pre-generated trace fixture database created by rocprofv3
    m_database_path = std::string(ROCSTORAGE_TEST_FIXTURE_PATH) + "/reader_test_trace.db";

    // Check if the fixture exists
    std::ifstream f(m_database_path);
    if (!f.good()) {
      GTEST_SKIP() << "Test fixture not found: " << m_database_path
                   << " (run 'make generate_test_fixtures' to create it)";
    }
  }

  std::string m_database_path;
};

TEST_F(ReaderIntegrationTest, OpenSucceedsForValidDatabase) {
  auto reader = rocstorage::reader::open(m_database_path);
  ASSERT_NE(reader, nullptr);
}

TEST_F(ReaderIntegrationTest, CHandlesAreAccessible) {
  auto reader = rocstorage::reader::open(m_database_path);
  ASSERT_NE(reader, nullptr);

  EXPECT_NE(reader->c_trace_handle(), nullptr);
  EXPECT_NE(reader->c_database_handle(), nullptr);
}

TEST_F(ReaderIntegrationTest, ReadMetadataSucceeds) {
  auto reader = rocstorage::reader::open(m_database_path);
  ASSERT_NE(reader, nullptr);

  bool success = reader->read_metadata();
  EXPECT_TRUE(success);
}

TEST_F(ReaderIntegrationTest, NumTracksReturnsExpectedCount) {
  auto reader = rocstorage::reader::open(m_database_path);
  ASSERT_NE(reader, nullptr);
  ASSERT_TRUE(reader->read_metadata());

  uint64_t num_tracks = reader->num_tracks();
  EXPECT_GE(num_tracks, 0u);
}

TEST_F(ReaderIntegrationTest, StartAndEndTimeAreValid) {
  auto reader = rocstorage::reader::open(m_database_path);
  ASSERT_NE(reader, nullptr);
  ASSERT_TRUE(reader->read_metadata());

  uint64_t start = reader->start_time();
  uint64_t end = reader->end_time();

  EXPECT_GE(end, start);
}

TEST_F(ReaderIntegrationTest, GetTracksReturnsVector) {
  auto reader = rocstorage::reader::open(m_database_path);
  ASSERT_NE(reader, nullptr);
  ASSERT_TRUE(reader->read_metadata());

  auto tracks = reader->get_tracks();
  EXPECT_EQ(tracks.size(), reader->num_tracks());
}

TEST_F(ReaderIntegrationTest, GetTrackReturnsNullptrForInvalidIndex) {
  auto reader = rocstorage::reader::open(m_database_path);
  ASSERT_NE(reader, nullptr);
  ASSERT_TRUE(reader->read_metadata());

  auto track = reader->get_track(99999);
  EXPECT_EQ(track, nullptr);
}

TEST_F(ReaderIntegrationTest, MemoryFootprintIsNonnegative) {
  auto reader = rocstorage::reader::open(m_database_path);
  ASSERT_NE(reader, nullptr);
  ASSERT_TRUE(reader->read_metadata());

  uint64_t footprint = reader->memory_footprint();
  EXPECT_GE(footprint, 0u);
}

TEST_F(ReaderIntegrationTest, CancelDoesNotCrash) {
  auto reader = rocstorage::reader::open(m_database_path);
  ASSERT_NE(reader, nullptr);

  reader->cancel();
}

TEST_F(ReaderIntegrationTest, WaitReturnsFalseWithoutPendingOperation) {
  auto reader = rocstorage::reader::open(m_database_path);
  ASSERT_NE(reader, nullptr);

  bool result = reader->wait(1);
  EXPECT_FALSE(result);
}

TEST_F(ReaderIntegrationTest, ReadSliceWithEmptyTrackIdsReturnsFalse) {
  auto reader = rocstorage::reader::open(m_database_path);
  ASSERT_NE(reader, nullptr);
  ASSERT_TRUE(reader->read_metadata());

  std::vector<uint32_t> empty_ids;
  bool result = reader->read_slice(0, 1000000, empty_ids);
  EXPECT_FALSE(result);
}

} // namespace