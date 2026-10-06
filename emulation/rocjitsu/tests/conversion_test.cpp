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

TEST(ConversionTest, PackU8RoundsInModeAndKeepsOtherBytes) {
  const conversion::PackU8 nearest{0, 0, kKeep, conversion::integer_rounding(0)};
  const conversion::PackU8 down{0, 0, kKeep, conversion::integer_rounding(2)};
  const conversion::PackU8 up{0, 0, kKeep, conversion::integer_rounding(1)};
  EXPECT_EQ(nearest(0x3fc00000u, 0u, 0u), 2u); // 1.5
  EXPECT_EQ(nearest(0x40200000u, 0u, 0u), 2u); // 2.5 ties to even
  EXPECT_EQ(down(0x3fc00000u, 0u, 0u), 1u);
  EXPECT_EQ(up(0x3f000000u, 0u, 0u), 1u);                        // 0.5
  EXPECT_EQ(up(0xbf000000u, 0u, 0u), 0u);                        // -0.5 rounds to -0
  EXPECT_EQ(nearest(0x437f8000u, 2u, 0x11223344u), 0x11ff3344u); // 255.5 saturates
  EXPECT_EQ(nearest(0x7fc00000u, 3u, 0x11223344u), 0x00223344u); // NaN gives zero
  const conversion::PackU8 negated{0, 1, kKeep, conversion::integer_rounding(0)};
  EXPECT_EQ(negated(0xc0400000u, 1u, 0u), 0x300u); // -(-3)
}

template <typename From, typename To>
conversion::ToFloat<From, To> to_float(uint32_t mode, uint32_t omod = 0, bool clamp = false) {
  conversion::ToFloat<From, To> stages;
  stages.rounding = {mode, false, false};
  stages.output = {omod, clamp, true, mode, false};
  return stages;
}

TEST(ConversionTest, IntegerSourcesUseTheirLowBits) {
  const auto i16 = to_float<conversion::I16, rocjitsu::amdgpu::fp_format::F16>(0);
  const auto u16 = to_float<conversion::U16, rocjitsu::amdgpu::fp_format::F16>(0);
  EXPECT_EQ(i16(0x12348000u), 0xf800u); // -32768
  EXPECT_EQ(i16(0xffffffffu), 0xbc00u); // -1
  EXPECT_EQ(u16(0xffff0001u), 0x3c00u); // 1
  EXPECT_EQ(u16(0x0000fff0u), 0x7c00u); // rounds up to infinity
  const auto i32 = to_float<conversion::I32, rocjitsu::amdgpu::fp_format::F32>(3);
  EXPECT_EQ(i32(0x80000000u), 0xcf000000u);
  EXPECT_EQ(i32(0x80000001u), 0xceffffffu);
}

TEST(ConversionTest, OutputModifiersFollowRounding) {
  // Round toward zero first, then scale: 0xffffffff -> 0x4f7fffff -> 0x4fffffff.
  const auto u32 = to_float<conversion::U32, rocjitsu::amdgpu::fp_format::F32>(3, 1);
  EXPECT_EQ(u32(0xffffffffu), 0x4fffffffu);
  const auto clamped = to_float<conversion::I16, rocjitsu::amdgpu::fp_format::F16>(0, 0, true);
  EXPECT_EQ(clamped(0x7fffu), 0x3c00u);
  EXPECT_EQ(clamped(0xffffu), 0u);
}

TEST(ConversionTest, FloatSourcesTakeModifiersBeforeFlushing) {
  conversion::ToFloat<rocjitsu::amdgpu::fp_format::F32, rocjitsu::amdgpu::fp_format::F64> wide;
  wide.neg = 1;
  EXPECT_EQ(wide(0x00000001u), uint64_t{0xb6a0000000000000});
  wide.input = kFlush;
  EXPECT_EQ(wide(0x00000001u), uint64_t{0x8000000000000000});
  // Scalar lanes of a narrowing conversion return the destination lane type.
  conversion::ToFloat<rocjitsu::amdgpu::fp_format::F64, rocjitsu::amdgpu::fp_format::F32> narrow;
  narrow.abs = 1;
  const uint32_t one = narrow(uint64_t{0xbff0000000000000});
  EXPECT_EQ(one, 0x3f800000u);
}

TEST(ConversionTest, ToFloatSimdLanesMatchScalarLanes) {
  using V = util::native<uint32_t>;
  std::mt19937 random(6);
  for (uint32_t mode = 0; mode < 4; ++mode) {
    auto i16 = to_float<conversion::I16, rocjitsu::amdgpu::fp_format::F16>(mode, mode);
    auto f16 = to_float<rocjitsu::amdgpu::fp_format::F32, rocjitsu::amdgpu::fp_format::F16>(mode);
    f16.rounding.flush_tiny = mode % 2 == 0;
    f16.neg = mode & 1u;
    for (int i = 0; i < 2000; ++i) {
      const V input([&](auto) { return static_cast<uint32_t>(random()); });
      const V half = i16(input);
      const V narrowed = f16(input);
      for (std::size_t lane = 0; lane < V::size(); ++lane) {
        ASSERT_EQ(half[lane], i16(uint32_t(input[lane])));
        ASSERT_EQ(narrowed[lane], f16(uint32_t(input[lane])));
      }
    }
  }
}

} // namespace
