// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/amdgpu/shared/conversion.h"
#include "util/simd.h"

#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>

namespace {

namespace conversion = rocjitsu::amdgpu::conversion;
namespace denorm = rocjitsu::amdgpu::input_denormal;

constexpr denorm::Policy kKeep{false};
constexpr denorm::Policy kFlush{true};

// Saturating floor and floor(x + 0.5) of a finite value, computed exactly in double.
int32_t reference(float value, bool nearest) {
  const double rounded = std::floor(static_cast<double>(value) + (nearest ? 0.5 : 0.0));
  if (rounded >= 2147483647.0)
    return std::numeric_limits<int32_t>::max();
  if (rounded <= -2147483648.0)
    return std::numeric_limits<int32_t>::min();
  return static_cast<int32_t>(rounded);
}

TEST(ConversionTest, FloorAndNearestMatchExactRoundingOfFiniteValues) {
  for (uint64_t i = 0; i < (uint64_t{1} << 32); i += 997) {
    const auto bits = static_cast<uint32_t>(i);
    const float value = std::bit_cast<float>(bits);
    if (std::isnan(value))
      continue;
    ASSERT_EQ(conversion::FloorI32{kKeep}(bits), static_cast<uint32_t>(reference(value, false)))
        << std::hex << bits;
    ASSERT_EQ(conversion::NearestI32{kKeep}(bits), static_cast<uint32_t>(reference(value, true)))
        << std::hex << bits;
  }
}

TEST(ConversionTest, FloorAndNearestSaturateNanBySign) {
  for (const uint32_t nan : {0x7fc00000u, 0x7f800001u, 0x7fffffffu}) {
    EXPECT_EQ(conversion::FloorI32{kKeep}(nan), 0x7fffffffu);
    EXPECT_EQ(conversion::FloorI32{kKeep}(nan | 0x80000000u), 0x80000000u);
    EXPECT_EQ(conversion::NearestI32{kKeep}(nan), 0x7fffffffu);
    EXPECT_EQ(conversion::NearestI32{kKeep}(nan | 0x80000000u), 0x80000000u);
  }
}

TEST(ConversionTest, InputFlushPrecedesRounding) {
  EXPECT_EQ(conversion::FloorI32{kKeep}(0x80000001u), 0xffffffffu);
  EXPECT_EQ(conversion::FloorI32{kFlush}(0x80000001u), 0u);
  EXPECT_EQ(conversion::NearestI32{kFlush}(0x807fffffu), 0u);
  // Ties round toward +infinity.
  EXPECT_EQ(conversion::NearestI32{kKeep}(0xbfc00000u), 0xffffffffu); // -1.5 -> -1
  EXPECT_EQ(conversion::NearestI32{kKeep}(0x3fc00000u), 2u);          // 1.5 -> 2
}

TEST(ConversionTest, SimdLanesMatchScalarLanes) {
  using V = util::native<uint32_t>;
  std::mt19937 random(5);
  for (int i = 0; i < 20000; ++i) {
    const V input([&](auto) { return static_cast<uint32_t>(random()); });
    for (const denorm::Policy policy : {kKeep, kFlush}) {
      const V floor = conversion::FloorI32{policy}(input);
      const V nearest = conversion::NearestI32{policy}(input);
      for (std::size_t lane = 0; lane < V::size(); ++lane) {
        ASSERT_EQ(floor[lane], conversion::FloorI32{policy}(uint32_t(input[lane])));
        ASSERT_EQ(nearest[lane], conversion::NearestI32{policy}(uint32_t(input[lane])));
      }
    }
  }
}

} // namespace
