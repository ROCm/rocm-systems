// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "util/big_int.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <random>

namespace {

template <typename Wide> void expect_documented_signed_128_arithmetic() {
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
  expect_documented_signed_128_arithmetic<util::int128_t>();
}

TEST(BigIntTest, FallbackInt128SupportsSignedWidenedArithmetic) {
  expect_documented_signed_128_arithmetic<util::detail::fallback_int128_t>();
}

template <typename Wide> void expect_documented_unsigned_128_arithmetic() {
  const Wide one{1};
  const Wide max64{std::numeric_limits<uint64_t>::max()};
  const Wide limb = one << 64;
  const Wide top = one << 127;
  const Wide max128 = Wide{0} - one;
  EXPECT_TRUE(top > max64);
  EXPECT_TRUE(max128 >= top);
  EXPECT_TRUE(one < top);
  EXPECT_TRUE(limb <= limb);
  EXPECT_NE(top, limb);
  EXPECT_TRUE(static_cast<bool>(limb));
  EXPECT_FALSE(static_cast<bool>(Wide{0}));
  EXPECT_EQ(static_cast<uint32_t>(Wide{0x12345678abcdef01ULL}), 0xabcdef01u);
  EXPECT_EQ(max64 + one, limb);
  EXPECT_EQ(limb - one, max64);
  EXPECT_EQ(max128 + one, Wide{0});
  EXPECT_EQ(static_cast<uint64_t>(max64 * max64), 1u);
  EXPECT_EQ(static_cast<uint64_t>((max64 * max64) >> 64), std::numeric_limits<uint64_t>::max() - 1);
  EXPECT_EQ(max128 * max128, one);
  EXPECT_EQ((limb + one) * (limb + 2), (limb * 3) + 2);
  EXPECT_EQ((top | max64) & limb, Wide{0});
  EXPECT_EQ((top | max64) & max64, max64);
  EXPECT_EQ(limb << 63, top);
  EXPECT_EQ(top >> 127, one);
  EXPECT_EQ(top >> 64, one << 63);
  EXPECT_EQ(one << 0, one);
  EXPECT_EQ(one >> 0, one);
  Wide carry = max64;
  EXPECT_EQ(++carry, limb);
  carry >>= 64;
  EXPECT_EQ(carry, one);
}

TEST(BigIntTest, Uint128SupportsUnsignedArithmetic) {
  expect_documented_unsigned_128_arithmetic<util::uint128_t>();
}

TEST(BigIntTest, FallbackUint128SupportsUnsignedArithmetic) {
  expect_documented_unsigned_128_arithmetic<util::detail::fallback_uint128_t>();
}

#if defined(__SIZEOF_INT128__)
TEST(BigIntTest, FallbackUint128MatchesNativeAcrossBothWords) {
  using Fallback = util::detail::fallback_uint128_t;
  using Native = util::uint128_t;
  const auto expect_equal = [](Fallback actual, Native expected) {
    EXPECT_EQ(static_cast<uint64_t>(actual), static_cast<uint64_t>(expected));
    EXPECT_EQ(static_cast<uint64_t>(actual >> 64), static_cast<uint64_t>(expected >> 64));
  };
  std::mt19937_64 random(1250950);
  for (int i = 0; i < 10000; ++i) {
    const uint64_t a0 = random(), a1 = random(), b0 = random(), b1 = random();
    const Fallback a = Fallback{a0} | (Fallback{a1} << 64);
    const Fallback b = Fallback{b0} | (Fallback{b1} << 64);
    const Native na = Native{a0} | (Native{a1} << 64);
    const Native nb = Native{b0} | (Native{b1} << 64);
    const int shift = static_cast<int>(random() % 128);
    expect_equal(a + b, na + nb);
    expect_equal(a - b, na - nb);
    expect_equal(a * b, na * nb);
    expect_equal(a & b, na & nb);
    expect_equal(a | b, na | nb);
    expect_equal(a << shift, na << shift);
    expect_equal(a >> shift, na >> shift);
    EXPECT_EQ(a < b, na < nb);
    EXPECT_EQ(a > b, na > nb);
  }
}
#endif

} // namespace
