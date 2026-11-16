// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "util/bit.h"

#include <gtest/gtest.h>

TEST(UtilBit, RangeBoundsDoNotWrapAndPermitEmptyEndRanges) {
  EXPECT_TRUE(util::is_range_in_bounds(0, 0, 0));
  EXPECT_TRUE(util::is_range_in_bounds(8, 0, 8));
  EXPECT_TRUE(util::is_range_in_bounds(3, 5, 8));
  EXPECT_FALSE(util::is_range_in_bounds(8, 1, 8));
  EXPECT_FALSE(util::is_range_in_bounds(9, 0, 8));
  EXPECT_FALSE(util::is_range_in_bounds(3, 6, 8));
  EXPECT_TRUE(util::is_range_in_bounds(UINT64_MAX, 0, UINT64_MAX));
  EXPECT_TRUE(util::is_range_in_bounds(1, UINT64_MAX - 1, UINT64_MAX));
  EXPECT_FALSE(util::is_range_in_bounds(UINT64_MAX, 1, UINT64_MAX));
  EXPECT_FALSE(util::is_range_in_bounds(UINT64_MAX - 1, 4, UINT64_MAX));
  EXPECT_FALSE(util::is_range_in_bounds(1, UINT64_MAX, UINT64_MAX));
}
