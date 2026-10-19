// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/canonical_line_freshness.h"

#include <gtest/gtest.h>

#include <memory>
#include <optional>
#include <thread>

namespace rocjitsu {
namespace {

TEST(CanonicalLineFreshnessTest, CrossLineStampAuthorizesOnlyCapturedBytes) {
  CanonicalLineFreshness lines;
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

TEST(CanonicalLineFreshnessTest, WriterWaitsForFillAndChangesOnlyTouchedLine) {
  CanonicalLineFreshness lines;
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

TEST(CanonicalLineFreshnessTest, AuthorityMoveAssignmentUnlocksBeforeDroppingLastState) {
  CanonicalLineFreshness::FillAuthority fill;
  {
    auto lines = std::make_unique<CanonicalLineFreshness>();
    auto read = lines->try_capture_read(7, 128, 1);
    ASSERT_TRUE(read);
    fill = read->try_acquire_fill(7, 128, 1);
    ASSERT_TRUE(fill);
  }
  fill = {};
  EXPECT_FALSE(fill);

  CanonicalLineFreshness::WriterAuthority writer;
  {
    auto lines = std::make_unique<CanonicalLineFreshness>();
    writer = lines->try_acquire_writer();
    ASSERT_TRUE(writer);
  }
  writer = {};
  EXPECT_FALSE(writer);
}

TEST(CanonicalLineFreshnessTest, MaintenanceClearCannotReviveOlderReadStamp) {
  CanonicalLineFreshness lines(1);
  const auto old_read = lines.try_capture_read(7, 128, 1);
  ASSERT_TRUE(old_read);

  lines.invalidate_all();
  std::optional<CanonicalLineFreshness::ReadStamp> new_read;
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

TEST(CanonicalLineFreshnessTest, EvictionCannotReviveOlderReadStamp) {
  CanonicalLineFreshness lines(1);
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
} // namespace rocjitsu
