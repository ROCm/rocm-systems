// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/amdgpu/shared/conversion.h"
#include "util/simd.h"

#include <gtest/gtest.h>

#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>

#if defined(__F16C__)
#include <immintrin.h>
#endif

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
    ASSERT_EQ(conversion::FloorI32{.input = kKeep}(bits),
              static_cast<uint32_t>(reference(value, false)))
        << std::hex << bits;
    ASSERT_EQ(conversion::NearestI32{.input = kKeep}(bits),
              static_cast<uint32_t>(reference(value, true)))
        << std::hex << bits;
  }
}

TEST(ConversionTest, FloorAndNearestSaturateNanBySign) {
  for (const uint32_t nan : {0x7fc00000u, 0x7f800001u, 0x7fffffffu}) {
    EXPECT_EQ(conversion::FloorI32{.input = kKeep}(nan), 0x7fffffffu);
    EXPECT_EQ(conversion::FloorI32{.input = kKeep}(nan | 0x80000000u), 0x80000000u);
    EXPECT_EQ(conversion::NearestI32{.input = kKeep}(nan), 0x7fffffffu);
    EXPECT_EQ(conversion::NearestI32{.input = kKeep}(nan | 0x80000000u), 0x80000000u);
  }
}

TEST(ConversionTest, InputFlushPrecedesRounding) {
  EXPECT_EQ(conversion::FloorI32{.input = kKeep}(0x80000001u), 0xffffffffu);
  EXPECT_EQ(conversion::FloorI32{.input = kFlush}(0x80000001u), 0u);
  EXPECT_EQ(conversion::NearestI32{.input = kFlush}(0x807fffffu), 0u);
  // Ties round toward +infinity.
  EXPECT_EQ(conversion::NearestI32{.input = kKeep}(0xbfc00000u), 0xffffffffu); // -1.5 -> -1
  EXPECT_EQ(conversion::NearestI32{.input = kKeep}(0x3fc00000u), 2u);          // 1.5 -> 2
}

TEST(ConversionTest, FloorAndNearestApplySourceModifiersBeforeFlush) {
  EXPECT_EQ((conversion::FloorI32{.abs = 1, .input = kKeep}(0xbfc00000u)), 1u);          // |-1.5|
  EXPECT_EQ((conversion::FloorI32{.neg = 1, .input = kKeep}(0x3fc00000u)), 0xfffffffeu); // -1.5
  EXPECT_EQ((conversion::NearestI32{.abs = 1, .neg = 1, .input = kKeep}(0x3fc00000u)),
            0xffffffffu); // -|1.5| -> -1
  // NEG makes the positive subnormal negative, and the flush keeps that sign.
  EXPECT_EQ((conversion::FloorI32{.neg = 1, .input = kKeep}(0x00000001u)), 0xffffffffu);
  EXPECT_EQ((conversion::FloorI32{.neg = 1, .input = kFlush}(0x00000001u)), 0u);
}

TEST(ConversionTest, SimdLanesMatchScalarLanes) {
  using V = util::native<uint32_t>;
  std::mt19937 random(5);
  for (int i = 0; i < 20000; ++i) {
    const V input([&](auto) { return static_cast<uint32_t>(random()); });
    for (const denorm::Policy policy : {kKeep, kFlush}) {
      const V floor = conversion::FloorI32{.input = policy}(input);
      const V nearest = conversion::NearestI32{.input = policy}(input);
      for (std::size_t lane = 0; lane < V::size(); ++lane) {
        ASSERT_EQ(floor[lane], conversion::FloorI32{.input = policy}(uint32_t(input[lane])));
        ASSERT_EQ(nearest[lane], conversion::NearestI32{.input = policy}(uint32_t(input[lane])));
      }
    }
  }
}

TEST(ConversionTest, PackU8RoundsInModeAndKeepsOtherBytes) {
  const conversion::PackU8 nearest{0, 0, kKeep, conversion::mode_rounding(0)};
  const conversion::PackU8 down{0, 0, kKeep, conversion::mode_rounding(2)};
  const conversion::PackU8 up{0, 0, kKeep, conversion::mode_rounding(1)};
  EXPECT_EQ(nearest(0x3fc00000u, 0u, 0u), 2u); // 1.5
  EXPECT_EQ(nearest(0x40200000u, 0u, 0u), 2u); // 2.5 ties to even
  EXPECT_EQ(down(0x3fc00000u, 0u, 0u), 1u);
  EXPECT_EQ(up(0x3f000000u, 0u, 0u), 1u);                        // 0.5
  EXPECT_EQ(up(0xbf000000u, 0u, 0u), 0u);                        // -0.5 rounds to -0
  EXPECT_EQ(nearest(0x437f8000u, 2u, 0x11223344u), 0x11ff3344u); // 255.5 saturates
  EXPECT_EQ(nearest(0x7fc00000u, 3u, 0x11223344u), 0x00223344u); // NaN gives zero
  const conversion::PackU8 negated{0, 1, kKeep, conversion::mode_rounding(0)};
  EXPECT_EQ(negated(0xc0400000u, 1u, 0u), 0x300u); // -(-3)
}

TEST(ConversionTest, PackB32F16ModifiesFlushesAndQuietsEachHalf) {
  const conversion::PackB32F16 keep{0, 0, kKeep, true};
  EXPECT_EQ(keep(0x12347c01u, 0x5678fc00u), 0xfc007e01u); // upper source bits are ignored
  EXPECT_EQ(keep(0x03ffu, 0x8001u), 0x800103ffu);
  const conversion::PackB32F16 flush{0, 0, kFlush, true};
  EXPECT_EQ(flush(0x03ffu, 0x8001u), 0x80000000u);
  const conversion::PackB32F16 modified{1, 2, kKeep, true};
  EXPECT_EQ(modified(0xbc00u, 0x3c00u), 0xbc003c00u);
  // Targets that keep signaling NaNs copy their payload unchanged.
  const conversion::PackB32F16 raw{0, 0, kKeep, false};
  EXPECT_EQ(raw(0x7c01u, 0u), 0x7c01u);
}

TEST(ConversionTest, Fp8DecodeIsExactAndCanonicalizesNan) {
  // Every non-NaN byte decodes to a value that encodes back to the same byte.
  for (uint32_t byte = 0; byte < 0x100u; ++byte) {
    const uint32_t fp8 = conversion::decode_fp8<conversion::Fp8>(byte);
    const uint32_t bf8 = conversion::decode_fp8<conversion::Bf8>(byte);
    if ((byte & 0x7fu) == 0x7fu)
      EXPECT_EQ(fp8, 0xffc00000u) << byte;
    else
      EXPECT_EQ(conversion::encode_fp8<conversion::Fp8>(fp8, false), byte) << byte;
    if ((byte & 0x7fu) > 0x7cu)
      EXPECT_EQ(bf8, 0xffc00000u) << byte;
    else
      EXPECT_EQ(conversion::encode_fp8<conversion::Bf8>(bf8, false), byte) << byte;
  }
  EXPECT_EQ(conversion::decode_fp8<conversion::Fp8>(0x7eu), 0x43e00000u); // 448
  EXPECT_EQ(conversion::decode_fp8<conversion::Fp8>(0x01u), 0x3b000000u); // 2^-9
  EXPECT_EQ(conversion::decode_fp8<conversion::Bf8>(0xfcu), 0xff800000u);
}

TEST(ConversionTest, Fp8EncodeOverflowAndNan) {
  // 464 ties to the even 448; 472 rounds up past it to the FP8 NaN pattern, or
  // to 448 when saturating.
  EXPECT_EQ(conversion::encode_fp8<conversion::Fp8>(0x43e80000u, false), 0x7eu);
  EXPECT_EQ(conversion::encode_fp8<conversion::Fp8>(0x43ec0000u, false), 0x7fu);
  EXPECT_EQ(conversion::encode_fp8<conversion::Fp8>(0xc3ec0000u, true), 0xfeu);
  EXPECT_EQ(conversion::encode_fp8<conversion::Bf8>(0x7f7fffffu, false), 0x7cu);
  EXPECT_EQ(conversion::encode_fp8<conversion::Bf8>(0x7f7fffffu, true), 0x7bu);
  // Infinity keeps the overflow pattern even when saturating.
  EXPECT_EQ(conversion::encode_fp8<conversion::Fp8>(0x7f800000u, true), 0x7fu);
  EXPECT_EQ(conversion::encode_fp8<conversion::Bf8>(0xff800000u, true), 0xfcu);
  // Every NaN encodes to one pattern.
  EXPECT_EQ(conversion::encode_fp8<conversion::Fp8>(0x7fc00000u, false), 0xffu);
  EXPECT_EQ(conversion::encode_fp8<conversion::Bf8>(0xff800001u, false), 0xfeu);
}

TEST(ConversionTest, Fp8StochasticRoundingAddsRandomBits) {
  // 1 + 2^-4 lies between FP8 1.0 and 1.125: truncated, or rounded up by large random bits.
  EXPECT_EQ(conversion::encode_fp8_stochastic<conversion::Fp8>(0x3f880000u, 0u, false), 0x38u);
  EXPECT_EQ(conversion::encode_fp8_stochastic<conversion::Fp8>(0x3f880000u, 0x80000000u, false),
            0x39u);
  // A value headed for the subnormal range is truncated at normal precision first.
  EXPECT_EQ(conversion::encode_fp8_stochastic<conversion::Fp8>(0x384b85fdu, 0xffffffffu, false),
            0x01u);
}

TEST(ConversionTest, ToFp8AppliesModifiersAndFlushesInputs) {
  const conversion::ToFp8<conversion::Fp8> plain{0, 0, kKeep, false};
  EXPECT_EQ(plain(0x3f800000u, 0x80000000u), 0x8038u);
  const conversion::ToFp8<conversion::Fp8> negated{0, 1, kKeep, false};
  EXPECT_EQ(negated(0x3f800000u, 0x80000000u), 0x80b8u);
  const conversion::ToFp8<conversion::Bf8> flush{0, 0, kFlush, false};
  EXPECT_EQ(flush.stochastic(0x807fffffu, 0xffffffffu), 0x80u);
}

template <typename From, typename To>
conversion::ToFloat<From, To> to_float(uint32_t mode, uint32_t omod = 0, bool clamp = false) {
  conversion::ToFloat<From, To> stages;
  stages.rounding = {conversion::mode_rounding(mode), false, false};
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

namespace fmt = rocjitsu::amdgpu::fp_format;
using conversion::FloatRounding;
using conversion::Rounding;

constexpr Rounding kDirections[] = {Rounding::NEAREST_EVEN, Rounding::CEIL, Rounding::FLOOR,
                                    Rounding::TOWARD_ZERO};

// Named wrappers keep template argument commas out of the assertion macros.
template <typename V> V f32_to_f16(V bits, const FloatRounding &policy) {
  return conversion::narrow<fmt::F32, fmt::F16>(bits, policy);
}
template <typename V> V f64_to_f32(V bits, const FloatRounding &policy) {
  return conversion::narrow<fmt::F64, fmt::F32>(bits, policy);
}
template <typename V> V f32_to_f64(V bits) { return conversion::widen<fmt::F32, fmt::F64>(bits); }
template <typename V, typename Mask>
V integer_to_f16(V magnitude, const Mask &negative, const FloatRounding &policy) {
  return conversion::from_integer<fmt::F16>(magnitude, negative, policy);
}
template <typename V, typename Mask>
V integer_to_f32(V magnitude, const Mask &negative, const FloatRounding &policy) {
  return conversion::from_integer<fmt::F32>(magnitude, negative, policy);
}

int host_round_mode(Rounding direction) {
  switch (direction) {
  case Rounding::CEIL:
    return FE_UPWARD;
  case Rounding::FLOOR:
    return FE_DOWNWARD;
  case Rounding::TOWARD_ZERO:
    return FE_TOWARDZERO;
  default:
    return FE_TONEAREST;
  }
}

/// Run host IEEE conversions in one rounding mode, restoring nearest-even afterwards.
class HostRounding {
public:
  explicit HostRounding(Rounding direction) { std::fesetround(host_round_mode(direction)); }
  ~HostRounding() { std::fesetround(FE_TONEAREST); }
  HostRounding(const HostRounding &) = delete;
  HostRounding &operator=(const HostRounding &) = delete;
};

uint32_t host_f64_to_f32(uint64_t bits) {
  volatile double value = std::bit_cast<double>(bits);
  return std::bit_cast<uint32_t>(static_cast<float>(value));
}

// Without flushing or saturation the rules are those of IEEE 754 conversion.
TEST(FloatRoundingTest, NarrowF64ToF32MatchesIeeeInEveryMode) {
  std::mt19937_64 random(1);
  for (const Rounding direction : kDirections) {
    const FloatRounding policy{direction, false, false};
    HostRounding host(direction);
    for (int i = 0; i < 200000; ++i) {
      uint64_t bits = random();
      // Most samples land near the F32 range, including its subnormals.
      if (i % 4 != 0)
        bits = (bits & 0x800fffffffffffffull) | (uint64_t{random() % 320 + 1023 - 170} << 52);
      ASSERT_EQ(f64_to_f32(bits, policy), uint64_t{host_f64_to_f32(bits)})
          << std::hex << bits << " direction " << static_cast<int>(direction);
    }
  }
}

#if defined(__F16C__)
uint32_t host_f32_to_f16(uint32_t bits, Rounding direction) {
  const __m128 value = _mm_castsi128_ps(_mm_cvtsi32_si128(static_cast<int>(bits)));
  switch (direction) {
  case Rounding::CEIL:
    return static_cast<uint16_t>(_mm_extract_epi16(_mm_cvtps_ph(value, _MM_FROUND_TO_POS_INF), 0));
  case Rounding::FLOOR:
    return static_cast<uint16_t>(_mm_extract_epi16(_mm_cvtps_ph(value, _MM_FROUND_TO_NEG_INF), 0));
  case Rounding::TOWARD_ZERO:
    return static_cast<uint16_t>(_mm_extract_epi16(_mm_cvtps_ph(value, _MM_FROUND_TO_ZERO), 0));
  default:
    return static_cast<uint16_t>(
        _mm_extract_epi16(_mm_cvtps_ph(value, _MM_FROUND_TO_NEAREST_INT), 0));
  }
}

TEST(FloatRoundingTest, NarrowF32ToF16MatchesIeeeInEveryMode) {
  for (const Rounding direction : kDirections) {
    const FloatRounding policy{direction, false, false};
    for (uint64_t i = 0; i < (uint64_t{1} << 32); i += 4099) {
      const auto bits = static_cast<uint32_t>(i);
      ASSERT_EQ(f32_to_f16(bits, policy), host_f32_to_f16(bits, direction))
          << std::hex << bits << " direction " << static_cast<int>(direction);
    }
  }
}

TEST(FloatRoundingTest, IntegerToF16MatchesIeeeInEveryMode) {
  for (const Rounding direction : kDirections) {
    const FloatRounding policy{direction, false, false};
    for (uint32_t value = 0; value < 0x10000u; ++value) {
      // Each 16-bit integer is exact in F32, so the host rounds once.
      const auto as_signed = static_cast<int16_t>(value);
      const uint32_t magnitude =
          as_signed < 0 ? 0u - static_cast<uint32_t>(as_signed) : static_cast<uint32_t>(as_signed);
      ASSERT_EQ(integer_to_f16(magnitude, as_signed < 0, policy),
                host_f32_to_f16(std::bit_cast<uint32_t>(static_cast<float>(as_signed)), direction))
          << as_signed;
      ASSERT_EQ(integer_to_f16(value, false, policy),
                host_f32_to_f16(std::bit_cast<uint32_t>(static_cast<float>(value)), direction))
          << value;
    }
  }
}
#endif

TEST(FloatRoundingTest, IntegerToF32MatchesIeeeInEveryMode) {
  std::mt19937_64 random(2);
  for (const Rounding direction : kDirections) {
    const FloatRounding policy{direction, false, false};
    HostRounding host(direction);
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

TEST(FloatRoundingTest, WidenF32ToF64IsExact) {
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
TEST(FloatRoundingTest, TininessIsJudgedAfterRounding) {
  const FloatRounding flush{Rounding::NEAREST_EVEN, true, false};
  EXPECT_EQ(f32_to_f16(0x387ff000u, flush), 0x0400u);
  EXPECT_EQ(f32_to_f16(0xb87ff000u, flush), 0x8400u);
  EXPECT_EQ(f32_to_f16(0x387fe000u, flush), 0x0000u);
  EXPECT_EQ(f32_to_f16(0xb87fe000u, flush), 0x8000u);
  // Without flushing, that tiny value's subnormal encoding rounds up to the smallest normal.
  EXPECT_EQ(f32_to_f16(0x387fe000u, FloatRounding{}), 0x0400u);
  // Directed rounding decides whether the boundary value carries.
  const FloatRounding flush_zero{Rounding::TOWARD_ZERO, true, false};
  const FloatRounding flush_up{Rounding::CEIL, true, false};
  EXPECT_EQ(f32_to_f16(0x387ff000u, flush_zero), 0x0000u);
  EXPECT_EQ(f32_to_f16(0x387fe001u, flush_up), 0x0400u);
  // F64 to F32 uses the same rule.
  EXPECT_EQ(f64_to_f32(uint64_t{0x380fffffe0000000}, flush), 0u);
  EXPECT_EQ(f64_to_f32(uint64_t{0x380ffffff0000000}, flush), uint64_t{0x00800000});
  EXPECT_EQ(f64_to_f32(uint64_t{0xb6a0000000000000}, flush), uint64_t{0x80000000});
}

TEST(FloatRoundingTest, OverflowFollowsModeUnlessSaturated) {
  const uint32_t too_large = 0x47800000u; // 65536
  EXPECT_EQ(f32_to_f16(too_large, {Rounding::NEAREST_EVEN}), 0x7c00u);
  EXPECT_EQ(f32_to_f16(too_large, {Rounding::TOWARD_ZERO}), 0x7bffu);
  EXPECT_EQ(f32_to_f16(too_large | 0x80000000u, {Rounding::CEIL}), 0xfbffu);
  EXPECT_EQ(f32_to_f16(too_large | 0x80000000u, {Rounding::FLOOR}), 0xfc00u);
  const FloatRounding saturate{Rounding::NEAREST_EVEN, false, true};
  EXPECT_EQ(f32_to_f16(too_large, saturate), 0x7bffu);
  EXPECT_EQ(integer_to_f16(65535u, true, saturate), 0xfbffu);
  // Saturation applies to finite values only.
  EXPECT_EQ(f32_to_f16(0xff800000u, saturate), 0xfc00u);
}

// A prescale multiplies the exact value before its single rounding.
TEST(FloatRoundingTest, PrescaleAppliesBeforeRounding) {
  FloatRounding halve;
  halve.prescale = -1;
  // 131008 overflows F16, but 65504 is its largest finite value.
  EXPECT_EQ(f32_to_f16(0x47ffe000u, FloatRounding{}), 0x7c00u);
  EXPECT_EQ(f32_to_f16(0x47ffe000u, halve), 0x7bffu);
  // Halving the smallest normal gives a subnormal, not a flushed or rounded value.
  EXPECT_EQ(f32_to_f16(0x38800000u, halve), 0x0200u);
  FloatRounding quadruple;
  quadruple.prescale = 2;
  EXPECT_EQ(integer_to_f16(2u, false, quadruple), 0x4800u);
  EXPECT_EQ(integer_to_f16(32767u, true, quadruple), 0xfc00u);
}

// Pins the SDWA ordering that simd_glue.h's sdwa_conversion_to_f16 documents as
// a known hardware mismatch: scaling the exact value first rounds 65535 div:2 to
// 32768, while gfx1030 rounds to infinity first and halves that. Measured on
// gfx1030 (RX 6800 XT) for SDWA v_cvt_f16_{u16,i16,f32} at mul:2/mul:4/div:2,
// which match VOP3 on every lane; left for the SDWA F16 migration follow-up.
TEST(FloatRoundingTest, PrescaleBeforeRoundingDisagreesWithGfx1030Sdwa) {
  FloatRounding halve;
  halve.prescale = -1;
  EXPECT_EQ(integer_to_f16(65535u, false, FloatRounding{}), 0x7c00u);
  EXPECT_EQ(integer_to_f16(65535u, false, halve), 0x7800u);
}

TEST(FloatRoundingTest, NarrowQuietsNanAndKeepsLeadingPayload) {
  EXPECT_EQ(f32_to_f16(0x7f800001u, {}), 0x7e00u);
  EXPECT_EQ(f32_to_f16(0xffa12345u, {}), 0xff09u);
  EXPECT_EQ(f64_to_f32(uint64_t{0x7ff0000000000001}, {}), uint64_t{0x7fc00000});
  // FP16_OVFL does not change infinities or NaNs.
  EXPECT_EQ(f32_to_f16(0x7f800000u, {Rounding::NEAREST_EVEN, false, true}), 0x7c00u);
}

TEST(FloatRoundingTest, SimdLanesMatchScalarLanes) {
  using V32 = util::native<uint32_t>;
#if UTIL_SIMD_BROKEN_NATIVE_64BIT_MASKS
  using V64 = util::stdx::fixed_size_simd<uint64_t, 1>;
#else
  using V64 = util::native<uint64_t>;
#endif
  std::mt19937_64 random(4);
  for (const Rounding direction : kDirections) {
    for (bool flush : {false, true}) {
      const FloatRounding policy{direction, flush, !flush};
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

} // namespace
