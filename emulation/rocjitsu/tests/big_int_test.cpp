// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "util/big_int.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>

namespace {

template <typename Wide> void expect_signed_128_arithmetic() {
  const auto zero = Wide{};
  const auto max64 = static_cast<Wide>(std::numeric_limits<uint64_t>::max());
  const auto twice_max64 = max64 + max64;
  const auto negative_max64 = -max64;

  EXPECT_TRUE(twice_max64 > max64);
  EXPECT_TRUE(negative_max64 < zero);
  EXPECT_TRUE(negative_max64 - max64 < negative_max64);
  EXPECT_FALSE(negative_max64 + max64 < zero);
  EXPECT_FALSE(negative_max64 + max64 > zero);
  EXPECT_EQ(static_cast<uint64_t>(negative_max64), 1u);
  EXPECT_EQ(static_cast<uint64_t>(twice_max64), std::numeric_limits<uint64_t>::max() - 1);
}

TEST(BigIntTest, Int128SupportsSignedWidenedArithmetic) {
  expect_signed_128_arithmetic<util::int128_t>();
}

TEST(BigIntTest, FallbackInt128SupportsSignedWidenedArithmetic) {
  expect_signed_128_arithmetic<util::detail::fallback_int128_t>();
}

} // namespace
