// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "simdojo/components/backing_line_versions.h"

#include <gtest/gtest.h>

#include <memory>
#include <optional>
#include <thread>

namespace simdojo {
namespace {

TEST(BackingLineVersionsTest, CrossLineStampAuthorizesOnlyCapturedBytes) {
  BackingLineVersions lines(128, 4);
  const auto read = lines.try_capture_read(7, 120, 20);
  ASSERT_TRUE(read);

  EXPECT_TRUE(read->covers(7, 120, 20));
  EXPECT_FALSE(read->covers(7, 119, 1));
  EXPECT_FALSE(read->covers(8, 120, 20));
  EXPECT_TRUE(read->try_acquire_fill(7, 120, 8));
  EXPECT_TRUE(read->try_acquire_fill(7, 128, 12));
  EXPECT_FALSE(read->try_acquire_fill(7, 119, 1));
  EXPECT_FALSE(read->try_acquire_fill(7, 128, 13));
}

TEST(BackingLineVersionsTest, WriterWaitsForFillAndChangesOnlyTouchedLine) {
  BackingLineVersions lines(128, 4);
  const auto read = lines.try_capture_read(7, 120, 20);
  ASSERT_TRUE(read);

  {
    auto fill = read->try_acquire_fill(7, 128, 12);
    ASSERT_TRUE(fill);
    bool writer_acquired = true;
    bool maintenance_cleared = true;
    std::thread contender([&] {
      auto writer = lines.try_acquire_writer();
      writer_acquired = static_cast<bool>(writer);
      maintenance_cleared = lines.try_invalidate_all();
    });
    contender.join();
    EXPECT_FALSE(writer_acquired);
    EXPECT_FALSE(maintenance_cleared);
  }
  {
    auto writer = lines.try_acquire_writer();
    ASSERT_TRUE(writer);
    writer.advance(7, 128, 1);
  }
  EXPECT_TRUE(read->try_acquire_fill(7, 120, 8));
  EXPECT_FALSE(read->try_acquire_fill(7, 128, 12));
}

TEST(BackingLineVersionsTest, AuthorityMoveAssignmentUnlocksBeforeDroppingLastState) {
  BackingLineVersions::FillAuthority fill;
  {
    auto lines = std::make_unique<BackingLineVersions>(128, 4);
    auto read = lines->try_capture_read(7, 128, 1);
    ASSERT_TRUE(read);
    fill = read->try_acquire_fill(7, 128, 1);
    ASSERT_TRUE(fill);
  }
  fill = {};
  EXPECT_FALSE(fill);

  BackingLineVersions::WriterAuthority writer;
  {
    auto lines = std::make_unique<BackingLineVersions>(128, 4);
    writer = lines->try_acquire_writer();
    ASSERT_TRUE(writer);
  }
  writer = {};
  EXPECT_FALSE(writer);
}

TEST(BackingLineVersionsTest, UsesTheSelectedLineSizeAndCapacity) {
  BackingLineVersions lines(64, 2);
  const auto stamp = lines.try_capture_read(3, 63, 2);
  ASSERT_TRUE(stamp);
  {
    auto writer = lines.try_acquire_writer();
    ASSERT_TRUE(writer);
    writer.advance(3, 64, 1);
  }
  EXPECT_TRUE(stamp->try_acquire_fill(3, 63, 1));
  EXPECT_FALSE(stamp->try_acquire_fill(3, 64, 1));
  EXPECT_FALSE(lines.try_capture_read(3, 0, 129));
  EXPECT_FALSE(BackingLineVersions(0, 1).try_capture_read(3, 0, 1));
  EXPECT_FALSE(BackingLineVersions(96, 1).try_capture_read(3, 0, 1));
  EXPECT_FALSE(BackingLineVersions(64, 0).try_capture_read(3, 0, 1));
}

TEST(BackingLineVersionsTest, MaintenanceClearCannotReviveOlderReadStamp) {
  BackingLineVersions lines(128, 1);
  const auto old_read = lines.try_capture_read(7, 128, 1);
  ASSERT_TRUE(old_read);

  lines.invalidate_all();
  std::optional<BackingLineVersions::ReadStamp> new_read;
  {
    auto effect = lines.try_acquire_writer();
    ASSERT_TRUE(effect);
    effect.invalidate_all();
    new_read = lines.capture_read_under_writer(effect, 7, 128, 1);
  }
  ASSERT_TRUE(new_read);
  EXPECT_FALSE(old_read->try_acquire_fill(7, 128, 1));
  EXPECT_TRUE(new_read->try_acquire_fill(7, 128, 1));
}

TEST(BackingLineVersionsTest, EvictionCannotReviveOlderReadStamp) {
  BackingLineVersions lines(128, 1);
  const auto old_read = lines.try_capture_read(7, 0, 1);
  ASSERT_TRUE(old_read);
  ASSERT_TRUE(lines.try_capture_read(8, 0, 1));
  const auto new_read = lines.try_capture_read(7, 0, 1);
  ASSERT_TRUE(new_read);

  EXPECT_FALSE(old_read->try_acquire_fill(7, 0, 1));
  EXPECT_TRUE(new_read->try_acquire_fill(7, 0, 1));
  EXPECT_FALSE(lines.try_capture_read(7, 0, 257));
}

} // namespace
} // namespace simdojo
