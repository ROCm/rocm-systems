// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file output_modifier_test.cpp
/// @brief OMOD and CLAMP on raw encodings: a reference model, gfx1201 regression
/// cases, and scalar/SIMD agreement.

#include "rocjitsu/isa/arch/amdgpu/shared/fp_format.h"
#include "rocjitsu/isa/arch/amdgpu/shared/input_denormal.h"
#include "rocjitsu/isa/arch/amdgpu/shared/minmax.h"
#include "rocjitsu/isa/arch/amdgpu/shared/output_modifier.h"
#include "util/simd.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace {

namespace denorm = rocjitsu::amdgpu::input_denormal;
namespace fmt = rocjitsu::amdgpu::fp_format;
namespace mm = rocjitsu::amdgpu::minmax;
namespace om = rocjitsu::amdgpu::output_modifier;

// ---------------------------------------------------------------------------
// Reference model: scale decoded values rather than adjust encoded exponents.
// Handle underflow before scaling so host F64 subnormal rounding cannot affect it.
// ---------------------------------------------------------------------------

template <typename Fmt> struct Ref {
  using L = typename Fmt::Lane;
  static constexpr L kMant = (L{1} << Fmt::kMantissaBits) - 1;
  static constexpr int kBias = (1 << (Fmt::kExponentBits - 1)) - 1;

  static int exponent(L x) { return static_cast<int>((x & Fmt::kInfinity) >> Fmt::kMantissaBits); }

  // The magnitude of a finite normal encoding.
  static double magnitude(L x) {
    return std::ldexp(static_cast<double>((x & kMant) | (L{1} << Fmt::kMantissaBits)),
                      exponent(x) - kBias - static_cast<int>(Fmt::kMantissaBits));
  }

  // Encode a magnitude in the normal range; exact for every value scaled here.
  static L encode(double m) {
    int e;
    const double fraction = std::frexp(m, &e); // m = fraction * 2^e, fraction in [0.5, 1)
    const L significand =
        static_cast<L>(std::ldexp(fraction, static_cast<int>(Fmt::kMantissaBits) + 1)) & kMant;
    return (static_cast<L>(e - 1 + kBias) << Fmt::kMantissaBits) | significand;
  }

  // Overflow follows the guest rounding direction; FP16_OVFL saturates F16.
  static L overflow(bool negative, const om::Policy &p) {
    bool infinite =
        p.round_mode == 0 || (p.round_mode == 1 && !negative) || (p.round_mode == 2 && negative);
    if (Fmt::kWidth == 16 && p.fp16_ovfl)
      infinite = false;
    const L m = infinite ? Fmt::kInfinity : Fmt::kInfinity - 1;
    return negative ? (m | Fmt::kSign) : m;
  }

  static L scale(L x, const om::Policy &p) {
    if (p.omod == 0 || exponent(x) == static_cast<int>(Fmt::kExponentMax))
      return x;
    if (exponent(x) == 0)
      return 0;
    const bool negative = (x & Fmt::kSign) != 0;
    const double factor = p.omod == 1 ? 2.0 : p.omod == 2 ? 4.0 : 0.5;
    const double input_magnitude = magnitude(x);
    const double min_normal = std::ldexp(1.0, 1 - kBias);
    // input * factor < min_normal, checked before host rounding can occur.
    if (input_magnitude < min_normal / factor)
      return negative ? Fmt::kSign : 0;
    const double scaled_magnitude = input_magnitude * factor;
    if (scaled_magnitude > magnitude(Fmt::kInfinity - 1))
      return overflow(negative, p);
    return encode(scaled_magnitude) | (negative ? Fmt::kSign : 0);
  }

  static L clamp(L x, const om::Policy &p) {
    if (!p.clamp)
      return x;
    if ((x & Fmt::kMagnitude) > Fmt::kInfinity)
      return p.clamp_nan_to_zero ? 0 : x;
    if ((x & Fmt::kSign) != 0 || x == 0)
      return 0;
    const L one = static_cast<L>(kBias) << Fmt::kMantissaBits;
    return x > one ? one : x;
  }

  static L apply(L x, const om::Policy &p) { return clamp(scale(x, p), p); }
};

/// Every combination of OMOD, CLAMP, NaN clamping, rounding mode and FP16_OVFL.
std::vector<om::Policy> policies() {
  std::vector<om::Policy> result;
  for (uint32_t omod = 0; omod < 4; ++omod)
    for (const bool clamp : {false, true})
      for (const bool nan_to_zero : {false, true})
        for (uint32_t round_mode = 0; round_mode < 4; ++round_mode)
          for (const bool fp16_ovfl : {false, true})
            result.push_back({omod, clamp, nan_to_zero, round_mode, fp16_ovfl});
  return result;
}

/// Both signs of each exponent boundary OMOD can cross, with the smallest and
/// largest significands, plus the exponents around 1.0 that CLAMP compares with.
template <typename Fmt> std::vector<typename Fmt::Lane> specials() {
  using L = typename Fmt::Lane;
  const L top = Fmt::kExponentMax;
  const L bias = top >> 1;
  std::vector<L> values;
  for (const L e :
       {L{0}, L{1}, L{2}, L{3}, bias - 1, bias, bias + 1, top - 3, top - 2, top - 1, top})
    for (const L significand : {L{0}, L{1}, Ref<Fmt>::kMant >> 1, Fmt::kQuiet, Ref<Fmt>::kMant}) {
      const L x = (e << Fmt::kMantissaBits) | significand;
      values.push_back(x);
      values.push_back(x | Fmt::kSign);
    }
  return values;
}

template <typename Fmt> std::vector<typename Fmt::Lane> specials_and_random() {
  using L = typename Fmt::Lane;
  std::vector<L> values = specials<Fmt>();
  std::mt19937_64 rng(0x0aad);
  for (int i = 0; i < 20000; ++i)
    values.push_back(static_cast<L>(rng()) & Fmt::kBits);
  return values;
}

template <typename Fmt> void expect_matches_reference(typename Fmt::Lane x) {
  for (const om::Policy &p : policies())
    ASSERT_EQ(om::apply<Fmt>(x, p), Ref<Fmt>::apply(x, p))
        << std::hex << "x 0x" << x << std::dec << " omod " << p.omod << " clamp " << p.clamp
        << " nan_to_zero " << p.clamp_nan_to_zero << " round " << p.round_mode << " fp16_ovfl "
        << p.fp16_ovfl;
}

TEST(OutputModifierTest, F16EveryEncodingMatchesReference) {
  for (uint32_t x = 0; x <= 0xffffu; ++x)
    expect_matches_reference<fmt::F16>(x);
}

TEST(OutputModifierTest, F32MatchesReference) {
  for (const uint32_t x : specials_and_random<fmt::F32>())
    expect_matches_reference<fmt::F32>(x);
}

TEST(OutputModifierTest, F64MatchesReference) {
  for (const uint64_t x : specials_and_random<fmt::F64>())
    expect_matches_reference<fmt::F64>(x);
}

// ---------------------------------------------------------------------------
// Regression cases from gfx1201 v_max_num results with mul:2, mul:4, div:2 and
// clamp, through the same selection and output stages as the executors.
// ---------------------------------------------------------------------------

template <typename Fmt> struct Witness {
  uint32_t mode;
  uint32_t omod;
  bool clamp;
  typename Fmt::Lane a, b, expected;
};

template <typename Fmt> void expect_witnesses(const std::vector<Witness<Fmt>> &witnesses) {
  for (const Witness<Fmt> &w : witnesses) {
    const bool f32 = Fmt::kWidth == 32;
    const auto input_policy = denorm::Policy::make(f32 ? (w.mode >> 4) & 3u : (w.mode >> 6) & 3u);
    // GFX12 CLAMP turns NaN into +0 regardless of MODE.DX10_CLAMP.
    const om::Policy output{w.omod, w.clamp, true, f32 ? w.mode & 3u : (w.mode >> 2) & 3u,
                            ((w.mode >> 23) & 1u) != 0};
    const auto selected = mm::evaluate<Fmt, mm::MaxNum>(input_policy, w.a, w.b);
    EXPECT_EQ(om::apply<Fmt>(selected, output), w.expected)
        << std::hex << "mode 0x" << w.mode << " omod " << w.omod << " clamp " << w.clamp << " a 0x"
        << w.a << " b 0x" << w.b;
  }
}

TEST(OutputModifierTest, MatchesGfx1201F16) {
  expect_witnesses<fmt::F16>({
      // Active OMOD maps zero and subnormal inputs to +0.
      {0xf0u, 1, false, 0x0001u, 0x0000u, 0x0000u},
      {0xf0u, 3, false, 0x0001u, 0x0000u, 0x0000u},
      {0xf0u, 1, false, 0x8000u, 0x8000u, 0x0000u},
      // Halving a normal into the subnormal range produces signed zero.
      {0xf0u, 3, false, 0x8400u, 0xbc00u, 0x8000u},
      {0xf0u, 2, false, 0x3c00u, 0x0000u, 0x4400u},
      {0xf0u, 3, false, 0x3c00u, 0x0000u, 0x3800u},
      // Overflow rounds in the guest mode; FP16_OVFL saturates.
      {0xf0u, 1, false, 0x7bffu, 0x0000u, 0x7c00u},
      {0xf0u, 2, false, 0xfbffu, 0xfc00u, 0xfc00u},
      {0xffu, 1, false, 0x7bffu, 0x0000u, 0x7bffu},
      {0xffu, 2, false, 0xfbffu, 0xfc00u, 0xfbffu},
      {0x8000f0u, 1, false, 0x7bffu, 0x0000u, 0x7bffu},
      {0x8000f0u, 2, false, 0xfbffu, 0xfc00u, 0xfbffu},
      {0xf0u, 1, false, 0x7e00u, 0x7e00u, 0x7e00u},
      {0xf0u, 0, true, 0x7e00u, 0x7e00u, 0x0000u},
      {0xf0u, 0, true, 0x7bffu, 0x0000u, 0x3c00u},
      {0xf0u, 0, true, 0x83ffu, 0x83ffu, 0x0000u},
      {0xf0u, 0, true, 0x0400u, 0x0000u, 0x0400u},
      // gfx1201 capture (2026-10-02): equal max-finite inputs, both rounding
      // directions and signs, with FP16_OVFL disabled and enabled.
      {0xf5u, 1, false, 0x7bffu, 0x7bffu, 0x7c00u},
      {0x8000f5u, 1, false, 0x7bffu, 0x7bffu, 0x7bffu},
      {0xfau, 1, false, 0x7bffu, 0x7bffu, 0x7bffu},
      {0x8000fau, 1, false, 0x7bffu, 0x7bffu, 0x7bffu},
      {0xf5u, 1, false, 0xfbffu, 0xfbffu, 0xfbffu},
      {0x8000f5u, 1, false, 0xfbffu, 0xfbffu, 0xfbffu},
      {0xfau, 1, false, 0xfbffu, 0xfbffu, 0xfc00u},
      {0x8000fau, 1, false, 0xfbffu, 0xfbffu, 0xfbffu},
      {0xf5u, 2, false, 0x7bffu, 0x7bffu, 0x7c00u},
      {0x8000f5u, 2, false, 0x7bffu, 0x7bffu, 0x7bffu},
      {0xfau, 2, false, 0x7bffu, 0x7bffu, 0x7bffu},
      {0x8000fau, 2, false, 0x7bffu, 0x7bffu, 0x7bffu},
      {0xf5u, 2, false, 0xfbffu, 0xfbffu, 0xfbffu},
      {0x8000f5u, 2, false, 0xfbffu, 0xfbffu, 0xfbffu},
      {0xfau, 2, false, 0xfbffu, 0xfbffu, 0xfc00u},
      {0x8000fau, 2, false, 0xfbffu, 0xfbffu, 0xfbffu},
  });
}

TEST(OutputModifierTest, MatchesGfx1201F32) {
  expect_witnesses<fmt::F32>({
      {0xf0u, 1, false, 0x00000001u, 0x00000000u, 0x00000000u},
      {0xf0u, 1, false, 0x80000000u, 0x80000000u, 0x00000000u},
      {0xf0u, 3, false, 0x80800000u, 0xbf800000u, 0x80000000u},
      {0xf0u, 2, false, 0x3f800000u, 0x00000000u, 0x40800000u},
      {0xf0u, 1, false, 0x7f7fffffu, 0x00000000u, 0x7f800000u},
      {0xffu, 1, false, 0xff7fffffu, 0xff800000u, 0xff7fffffu},
      // FP16_OVFL leaves F32 overflow infinite.
      {0x8000f0u, 1, false, 0x7f7fffffu, 0x00000000u, 0x7f800000u},
      {0xf0u, 1, false, 0x7fc00000u, 0x7fc00000u, 0x7fc00000u},
      {0xf0u, 0, true, 0x7fc00000u, 0x7fc00000u, 0x00000000u},
      {0xf0u, 0, true, 0x7f7fffffu, 0x00000000u, 0x3f800000u},
      {0xf0u, 0, true, 0x807fffffu, 0x807fffffu, 0x00000000u},
      // gfx1201 capture (2026-10-02): equal max-finite inputs, both rounding
      // directions and signs, with FP16_OVFL disabled and enabled.
      {0xf5u, 1, false, 0x7f7fffffu, 0x7f7fffffu, 0x7f800000u},
      {0x8000f5u, 1, false, 0x7f7fffffu, 0x7f7fffffu, 0x7f800000u},
      {0xfau, 1, false, 0x7f7fffffu, 0x7f7fffffu, 0x7f7fffffu},
      {0x8000fau, 1, false, 0x7f7fffffu, 0x7f7fffffu, 0x7f7fffffu},
      {0xf5u, 1, false, 0xff7fffffu, 0xff7fffffu, 0xff7fffffu},
      {0x8000f5u, 1, false, 0xff7fffffu, 0xff7fffffu, 0xff7fffffu},
      {0xfau, 1, false, 0xff7fffffu, 0xff7fffffu, 0xff800000u},
      {0x8000fau, 1, false, 0xff7fffffu, 0xff7fffffu, 0xff800000u},
      {0xf5u, 2, false, 0x7f7fffffu, 0x7f7fffffu, 0x7f800000u},
      {0x8000f5u, 2, false, 0x7f7fffffu, 0x7f7fffffu, 0x7f800000u},
      {0xfau, 2, false, 0x7f7fffffu, 0x7f7fffffu, 0x7f7fffffu},
      {0x8000fau, 2, false, 0x7f7fffffu, 0x7f7fffffu, 0x7f7fffffu},
      {0xf5u, 2, false, 0xff7fffffu, 0xff7fffffu, 0xff7fffffu},
      {0x8000f5u, 2, false, 0xff7fffffu, 0xff7fffffu, 0xff7fffffu},
      {0xfau, 2, false, 0xff7fffffu, 0xff7fffffu, 0xff800000u},
      {0x8000fau, 2, false, 0xff7fffffu, 0xff7fffffu, 0xff800000u},
  });
}

TEST(OutputModifierTest, MatchesGfx1201F64) {
  expect_witnesses<fmt::F64>({
      {0xf0u, 1, false, 0x1u, 0x0u, 0x0u},
      {0xf0u, 3, false, 0x8010000000000000u, 0xbff0000000000000u, 0x8000000000000000u},
      {0xf0u, 2, false, 0x3ff0000000000000u, 0x0u, 0x4010000000000000u},
      {0xf0u, 1, false, 0x7fefffffffffffffu, 0x0u, 0x7ff0000000000000u},
      {0xffu, 2, false, 0xffefffffffffffffu, 0xfff0000000000000u, 0xffefffffffffffffu},
      {0x8000f0u, 1, false, 0xffefffffffffffffu, 0xfff0000000000000u, 0xfff0000000000000u},
      {0xf0u, 0, true, 0x7ff8000000000000u, 0x7ff8000000000000u, 0x0u},
      {0xf0u, 0, true, 0x7fefffffffffffffu, 0x0u, 0x3ff0000000000000u},
      {0xf0u, 0, true, 0x800fffffffffffffu, 0x800fffffffffffffu, 0x0u},
      // gfx1201 capture (2026-10-02): equal max-finite inputs, both rounding
      // directions and signs, with FP16_OVFL disabled and enabled.
      {0xf5u, 1, false, 0x7fefffffffffffffu, 0x7fefffffffffffffu, 0x7ff0000000000000u},
      {0x8000f5u, 1, false, 0x7fefffffffffffffu, 0x7fefffffffffffffu, 0x7ff0000000000000u},
      {0xfau, 1, false, 0x7fefffffffffffffu, 0x7fefffffffffffffu, 0x7fefffffffffffffu},
      {0x8000fau, 1, false, 0x7fefffffffffffffu, 0x7fefffffffffffffu, 0x7fefffffffffffffu},
      {0xf5u, 1, false, 0xffefffffffffffffu, 0xffefffffffffffffu, 0xffefffffffffffffu},
      {0x8000f5u, 1, false, 0xffefffffffffffffu, 0xffefffffffffffffu, 0xffefffffffffffffu},
      {0xfau, 1, false, 0xffefffffffffffffu, 0xffefffffffffffffu, 0xfff0000000000000u},
      {0x8000fau, 1, false, 0xffefffffffffffffu, 0xffefffffffffffffu, 0xfff0000000000000u},
      {0xf5u, 2, false, 0x7fefffffffffffffu, 0x7fefffffffffffffu, 0x7ff0000000000000u},
      {0x8000f5u, 2, false, 0x7fefffffffffffffu, 0x7fefffffffffffffu, 0x7ff0000000000000u},
      {0xfau, 2, false, 0x7fefffffffffffffu, 0x7fefffffffffffffu, 0x7fefffffffffffffu},
      {0x8000fau, 2, false, 0x7fefffffffffffffu, 0x7fefffffffffffffu, 0x7fefffffffffffffu},
      {0xf5u, 2, false, 0xffefffffffffffffu, 0xffefffffffffffffu, 0xffefffffffffffffu},
      {0x8000f5u, 2, false, 0xffefffffffffffffu, 0xffefffffffffffffu, 0xffefffffffffffffu},
      {0xfau, 2, false, 0xffefffffffffffffu, 0xffefffffffffffffu, 0xfff0000000000000u},
      {0x8000fau, 2, false, 0xffefffffffffffffu, 0xffefffffffffffffu, 0xfff0000000000000u},
  });
}

// ---------------------------------------------------------------------------
// SIMD and scalar output stages agree for the same bits and policy.
// ---------------------------------------------------------------------------

#if __has_include(<experimental/simd>)
template <typename Fmt, typename V> void expect_simd_matches_scalar() {
  using L = typename Fmt::Lane;
  constexpr std::size_t W = V::size();
  std::vector<L> values = specials_and_random<Fmt>();
  values.resize(values.size() - values.size() % W);
  for (const om::Policy &p : policies())
    for (std::size_t base = 0; base < values.size(); base += W) {
      const V r = om::apply<Fmt>(V(values.data() + base, util::stdx::element_aligned), p);
      for (std::size_t i = 0; i < W; ++i)
        ASSERT_EQ(L(r[i]), om::apply<Fmt>(values[base + i], p))
            << std::hex << "x 0x" << values[base + i] << std::dec << " omod " << p.omod;
    }
}
#endif

TEST(OutputModifierTest, SimdMatchesScalar) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable";
  } else {
#if __has_include(<experimental/simd>)
    expect_simd_matches_scalar<fmt::F16, util::native<uint32_t>>();
    expect_simd_matches_scalar<fmt::F32, util::native<uint32_t>>();
#if UTIL_SIMD_BROKEN_NATIVE_64BIT_MASKS
    expect_simd_matches_scalar<fmt::F64, util::stdx::fixed_size_simd<uint64_t, 1>>();
#else
    expect_simd_matches_scalar<fmt::F64, util::native<uint64_t>>();
#endif
#endif
  }
}

} // namespace
