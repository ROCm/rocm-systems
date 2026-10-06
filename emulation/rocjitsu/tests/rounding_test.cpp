// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/amdgpu/shared/fp_format.h"
#include "rocjitsu/isa/arch/amdgpu/shared/rounding.h"
#include "util/simd.h"

#include <gtest/gtest.h>

#include <bit>
#include <cfenv>
#include <cstdint>
#include <random>

#if defined(__F16C__)
#include <immintrin.h>
#endif

namespace {

namespace rounding = rocjitsu::amdgpu::rounding;
namespace fmt = rocjitsu::amdgpu::fp_format;

constexpr uint32_t kModes[] = {rounding::NEAREST_EVEN, rounding::TOWARD_POSITIVE,
                               rounding::TOWARD_NEGATIVE, rounding::TOWARD_ZERO};

// Named wrappers keep template argument commas out of the assertion macros.
template <typename V> V f32_to_f16(V bits, const rounding::Policy &policy) {
  return rounding::narrow<fmt::F32, fmt::F16>(bits, policy);
}
template <typename V> V f64_to_f32(V bits, const rounding::Policy &policy) {
  return rounding::narrow<fmt::F64, fmt::F32>(bits, policy);
}
template <typename V> V f32_to_f64(V bits) { return rounding::widen<fmt::F32, fmt::F64>(bits); }
template <typename V, typename Mask>
V integer_to_f16(V magnitude, const Mask &negative, const rounding::Policy &policy) {
  return rounding::from_integer<fmt::F16>(magnitude, negative, policy);
}
template <typename V, typename Mask>
V integer_to_f32(V magnitude, const Mask &negative, const rounding::Policy &policy) {
  return rounding::from_integer<fmt::F32>(magnitude, negative, policy);
}

int host_round_mode(uint32_t mode) {
  switch (mode) {
  case rounding::TOWARD_POSITIVE:
    return FE_UPWARD;
  case rounding::TOWARD_NEGATIVE:
    return FE_DOWNWARD;
  case rounding::TOWARD_ZERO:
    return FE_TOWARDZERO;
  default:
    return FE_TONEAREST;
  }
}

/// Run host IEEE conversions in one rounding mode, restoring nearest-even afterwards.
class HostRounding {
public:
  explicit HostRounding(uint32_t mode) { std::fesetround(host_round_mode(mode)); }
  ~HostRounding() { std::fesetround(FE_TONEAREST); }
  HostRounding(const HostRounding &) = delete;
  HostRounding &operator=(const HostRounding &) = delete;
};

uint32_t host_f64_to_f32(uint64_t bits) {
  volatile double value = std::bit_cast<double>(bits);
  return std::bit_cast<uint32_t>(static_cast<float>(value));
}

// Without flushing or saturation the rules are those of IEEE 754 conversion.
TEST(RoundingTest, NarrowF64ToF32MatchesIeeeInEveryMode) {
  std::mt19937_64 random(1);
  for (uint32_t mode : kModes) {
    const rounding::Policy policy{mode, false, false};
    HostRounding host(mode);
    for (int i = 0; i < 200000; ++i) {
      uint64_t bits = random();
      // Most samples land near the F32 range, including its subnormals.
      if (i % 4 != 0)
        bits = (bits & 0x800fffffffffffffull) | (uint64_t{random() % 320 + 1023 - 170} << 52);
      ASSERT_EQ(f64_to_f32(bits, policy), uint64_t{host_f64_to_f32(bits)})
          << std::hex << bits << " mode " << mode;
    }
  }
}

#if defined(__F16C__)
uint32_t host_f32_to_f16(uint32_t bits, uint32_t mode) {
  const __m128 value = _mm_castsi128_ps(_mm_cvtsi32_si128(static_cast<int>(bits)));
  switch (mode) {
  case rounding::TOWARD_POSITIVE:
    return static_cast<uint16_t>(_mm_extract_epi16(_mm_cvtps_ph(value, _MM_FROUND_TO_POS_INF), 0));
  case rounding::TOWARD_NEGATIVE:
    return static_cast<uint16_t>(_mm_extract_epi16(_mm_cvtps_ph(value, _MM_FROUND_TO_NEG_INF), 0));
  case rounding::TOWARD_ZERO:
    return static_cast<uint16_t>(_mm_extract_epi16(_mm_cvtps_ph(value, _MM_FROUND_TO_ZERO), 0));
  default:
    return static_cast<uint16_t>(
        _mm_extract_epi16(_mm_cvtps_ph(value, _MM_FROUND_TO_NEAREST_INT), 0));
  }
}

TEST(RoundingTest, NarrowF32ToF16MatchesIeeeInEveryMode) {
  for (uint32_t mode : kModes) {
    const rounding::Policy policy{mode, false, false};
    for (uint64_t i = 0; i < (uint64_t{1} << 32); i += 4099) {
      const auto bits = static_cast<uint32_t>(i);
      ASSERT_EQ(f32_to_f16(bits, policy), host_f32_to_f16(bits, mode))
          << std::hex << bits << " mode " << mode;
    }
  }
}

TEST(RoundingTest, IntegerToF16MatchesIeeeInEveryMode) {
  for (uint32_t mode : kModes) {
    const rounding::Policy policy{mode, false, false};
    for (uint32_t value = 0; value < 0x10000u; ++value) {
      // Each 16-bit integer is exact in F32, so the host rounds once.
      const auto as_signed = static_cast<int16_t>(value);
      const uint32_t magnitude =
          as_signed < 0 ? 0u - static_cast<uint32_t>(as_signed) : static_cast<uint32_t>(as_signed);
      ASSERT_EQ(integer_to_f16(magnitude, as_signed < 0, policy),
                host_f32_to_f16(std::bit_cast<uint32_t>(static_cast<float>(as_signed)), mode))
          << as_signed;
      ASSERT_EQ(integer_to_f16(value, false, policy),
                host_f32_to_f16(std::bit_cast<uint32_t>(static_cast<float>(value)), mode))
          << value;
    }
  }
}
#endif

TEST(RoundingTest, IntegerToF32MatchesIeeeInEveryMode) {
  std::mt19937_64 random(2);
  for (uint32_t mode : kModes) {
    const rounding::Policy policy{mode, false, false};
    HostRounding host(mode);
    for (int i = 0; i < 200000; ++i) {
      const auto value = static_cast<uint32_t>(random() >> (random() % 40));
      volatile uint32_t unsigned_value = value;
      volatile int32_t signed_value = static_cast<int32_t>(value);
      const uint32_t magnitude = signed_value < 0 ? 0u - value : value;
      ASSERT_EQ(integer_to_f32(magnitude, signed_value < 0, policy),
                std::bit_cast<uint32_t>(static_cast<float>(signed_value)))
          << static_cast<int32_t>(value);
      ASSERT_EQ(integer_to_f32(value, false, policy),
                std::bit_cast<uint32_t>(static_cast<float>(unsigned_value)))
          << value;
    }
  }
}

TEST(RoundingTest, WidenF32ToF64IsExact) {
  std::mt19937_64 random(3);
  for (int i = 0; i < 200000; ++i) {
    auto bits = static_cast<uint32_t>(random());
    if (i % 2 == 0)
      bits &= 0x807fffffu; // zero or subnormal
    volatile float value = std::bit_cast<float>(bits);
    ASSERT_EQ(f32_to_f64(uint64_t{bits}), std::bit_cast<uint64_t>(static_cast<double>(value)))
        << std::hex << bits;
  }
  // A signaling NaN is quieted and keeps its payload.
  EXPECT_EQ(f32_to_f64(uint64_t{0xff800001u}), uint64_t{0xfff8000020000000});
}

// gfx1201 judges tininess after rounding: 0x387ff000 rounds up to the smallest
// normal half, while the exact 11-bit value 0x387fe000 below it is tiny.
TEST(RoundingTest, TininessIsJudgedAfterRounding) {
  const rounding::Policy flush{rounding::NEAREST_EVEN, true, false};
  EXPECT_EQ(f32_to_f16(0x387ff000u, flush), 0x0400u);
  EXPECT_EQ(f32_to_f16(0xb87ff000u, flush), 0x8400u);
  EXPECT_EQ(f32_to_f16(0x387fe000u, flush), 0x0000u);
  EXPECT_EQ(f32_to_f16(0xb87fe000u, flush), 0x8000u);
  // Without flushing, that tiny value's subnormal encoding rounds up to the smallest normal.
  EXPECT_EQ(f32_to_f16(0x387fe000u, rounding::Policy{}), 0x0400u);
  // Directed rounding decides whether the boundary value carries.
  const rounding::Policy flush_zero{rounding::TOWARD_ZERO, true, false};
  const rounding::Policy flush_up{rounding::TOWARD_POSITIVE, true, false};
  EXPECT_EQ(f32_to_f16(0x387ff000u, flush_zero), 0x0000u);
  EXPECT_EQ(f32_to_f16(0x387fe001u, flush_up), 0x0400u);
  // F64 to F32 uses the same rule.
  EXPECT_EQ(f64_to_f32(uint64_t{0x380fffffe0000000}, flush), 0u);
  EXPECT_EQ(f64_to_f32(uint64_t{0x380ffffff0000000}, flush), uint64_t{0x00800000});
  EXPECT_EQ(f64_to_f32(uint64_t{0xb6a0000000000000}, flush), uint64_t{0x80000000});
}

TEST(RoundingTest, OverflowFollowsModeUnlessSaturated) {
  const uint32_t too_large = 0x47800000u; // 65536
  EXPECT_EQ(f32_to_f16(too_large, {rounding::NEAREST_EVEN}), 0x7c00u);
  EXPECT_EQ(f32_to_f16(too_large, {rounding::TOWARD_ZERO}), 0x7bffu);
  EXPECT_EQ(f32_to_f16(too_large | 0x80000000u, {rounding::TOWARD_POSITIVE}), 0xfbffu);
  EXPECT_EQ(f32_to_f16(too_large | 0x80000000u, {rounding::TOWARD_NEGATIVE}), 0xfc00u);
  const rounding::Policy saturate{rounding::NEAREST_EVEN, false, true};
  EXPECT_EQ(f32_to_f16(too_large, saturate), 0x7bffu);
  EXPECT_EQ(integer_to_f16(65535u, true, saturate), 0xfbffu);
  // Saturation applies to finite values only.
  EXPECT_EQ(f32_to_f16(0xff800000u, saturate), 0xfc00u);
}

TEST(RoundingTest, NarrowQuietsNanAndKeepsLeadingPayload) {
  EXPECT_EQ(f32_to_f16(0x7f800001u, {}), 0x7e00u);
  EXPECT_EQ(f32_to_f16(0xffa12345u, {}), 0xff09u);
  EXPECT_EQ(f64_to_f32(uint64_t{0x7ff0000000000001}, {}), uint64_t{0x7fc00000});
  // FP16_OVFL does not change infinities or NaNs.
  EXPECT_EQ(f32_to_f16(0x7f800000u, {0, false, true}), 0x7c00u);
}

TEST(RoundingTest, SimdLanesMatchScalarLanes) {
  using V32 = util::native<uint32_t>;
  using V64 = util::native<uint64_t>;
  std::mt19937_64 random(4);
  for (uint32_t mode : kModes) {
    for (bool flush : {false, true}) {
      const rounding::Policy policy{mode, flush, !flush};
      for (int i = 0; i < 2000; ++i) {
        const V32 in32([&](auto) { return static_cast<uint32_t>(random()); });
        const V64 in64([&](auto) { return random() >> (random() % 2 ? 20 : 0); });
        const V32 narrowed = f32_to_f16(in32, policy);
        const V32 converted = integer_to_f32(in32, in32 > V32(1u << 31), policy);
        const V64 narrowed64 = f64_to_f32(in64, policy);
        const V64 widened = f32_to_f64(in64 & V64(0xffffffffu));
        for (std::size_t lane = 0; lane < V32::size(); ++lane) {
          const uint32_t value = in32[lane];
          ASSERT_EQ(narrowed[lane], f32_to_f16(value, policy));
          ASSERT_EQ(converted[lane], integer_to_f32(value, value > (1u << 31), policy));
        }
        for (std::size_t lane = 0; lane < V64::size(); ++lane) {
          const uint64_t value = in64[lane];
          ASSERT_EQ(narrowed64[lane], f64_to_f32(value, policy));
          ASSERT_EQ(widened[lane], f32_to_f64(value & 0xffffffffu));
        }
      }
    }
  }
}

TEST(FpFormatTest, BiasAndFractionMask) {
  EXPECT_EQ(fmt::F16::kBias, 15u);
  EXPECT_EQ(fmt::F16::kFraction, 0x3ffu);
  EXPECT_EQ(fmt::F32::kBias, 127u);
  EXPECT_EQ(fmt::F32::kFraction, 0x7fffffu);
  EXPECT_EQ(fmt::F64::kBias, uint64_t{1023});
  EXPECT_EQ(fmt::F64::kFraction, uint64_t{0xfffffffffffff});
}

// SIMD lanes count the same bits as std::bit_width, at 32 and 64 bits.
TEST(FpFormatTest, BitWidthMatchesScalarOnSimdLanes) {
  constexpr uint64_t kValues[] = {0,           1,          2,           3,
                                  0x3ff,       0x400,      0x7fffff,    0x800000,
                                  0x80000000,  0xffffffff, 0x123456789, uint64_t{1} << 52,
                                  ~uint64_t{0}};
  for (const uint64_t value : kValues) {
    const uint32_t low = static_cast<uint32_t>(value);
    EXPECT_EQ(fmt::bit_width(low), static_cast<uint32_t>(std::bit_width(low))) << value;
    EXPECT_EQ(fmt::bit_width(value), static_cast<uint64_t>(std::bit_width(value))) << value;
    const util::native<uint32_t> lanes32(low);
    const util::native<uint64_t> lanes64(value);
    EXPECT_EQ(uint32_t(fmt::bit_width(lanes32)[0]), static_cast<uint32_t>(std::bit_width(low)))
        << value;
    EXPECT_EQ(uint64_t(fmt::bit_width(lanes64)[0]), static_cast<uint64_t>(std::bit_width(value)))
        << value;
  }
}

struct ShiftCase {
  uint32_t value;
  uint32_t count;
  bool negative;
  // Rounded magnitude for nearest even, toward +inf, toward -inf, toward zero.
  uint32_t expected[4];
};

// 0b1011 / 4 = 2.75; 0b1010 / 4 = 2.5 (tie, even below); 0b1110 / 4 = 3.5 (tie,
// even above); 0b1000 / 4 is exact. Dropping more bits than the value has
// leaves a nonzero remainder below one half.
constexpr ShiftCase kShiftCases[] = {
    {0b1011, 2, false, {3, 3, 2, 2}}, {0b1011, 2, true, {3, 2, 3, 2}},
    {0b1010, 2, false, {2, 3, 2, 2}}, {0b1110, 2, true, {4, 3, 4, 3}},
    {0b1000, 2, false, {2, 2, 2, 2}}, {0b1000, 2, true, {2, 2, 2, 2}},
    {0x7ff, 12, false, {0, 1, 0, 0}}, {0x7ff, 12, true, {0, 0, 1, 0}},
};

TEST(RoundingTest, ShiftRightFollowsModeAndSign) {
  for (const ShiftCase &test : kShiftCases) {
    for (uint32_t mode : kModes) {
      SCOPED_TRACE(testing::Message() << test.value << " >> " << test.count << " negative "
                                      << test.negative << " mode " << mode);
      EXPECT_EQ(rounding::shift_right(test.value, test.count, test.negative, mode),
                test.expected[mode]);
      const util::native<uint32_t> value(test.value);
      const util::native<uint32_t> count(test.count);
      const auto negative = util::native<uint32_t>(test.negative ? 1u : 0u) != 0u;
      EXPECT_EQ(uint32_t(rounding::shift_right(value, count, negative, mode)[0]),
                test.expected[mode]);
    }
  }
}

// F16 significands with the leading one at bit 12. 0x7ff * 2^-25 = 0x3bff *
// 2^-14 is tiny after rounding to 11 bits, so a flushing policy gives zero
// even though its subnormal encoding rounds up to the smallest normal; gfx1201
// V_LDEXP_F16 does the same.
TEST(RoundingTest, RoundSignificandJudgesTininessAfterRounding) {
  constexpr unsigned kTop = fmt::F16::kMantissaBits + 2;
  const auto round = [](uint32_t significand, int exponent, bool negative,
                        const rounding::Policy &policy) {
    return rounding::round_significand<fmt::F16, kTop>(
        significand << 2, uint32_t(int(rounding::kExponentOrigin) + exponent), negative, policy);
  };
  // Biased exponent 0 is one step below the smallest normal.
  EXPECT_EQ(round(0x7ffu, 0, false, {}), 0x0400u);
  EXPECT_EQ(round(0x7ffu, 0, false, {rounding::NEAREST_EVEN, true, false}), 0u);
  EXPECT_EQ(round(0x7ffu, 0, false, {rounding::TOWARD_ZERO, false, false}), 0x03ffu);
  EXPECT_EQ(round(0x400u, 1, false, {rounding::NEAREST_EVEN, true, false}), 0x0400u);
  // Overflow follows the rounding direction, or saturates.
  EXPECT_EQ(round(0x400u, 31, false, {}), 0x7c00u);
  EXPECT_EQ(round(0x400u, 31, true, {rounding::TOWARD_POSITIVE, false, false}), 0x7bffu);
  EXPECT_EQ(round(0x400u, 31, false, {rounding::NEAREST_EVEN, false, true}), 0x7bffu);
}

} // namespace
