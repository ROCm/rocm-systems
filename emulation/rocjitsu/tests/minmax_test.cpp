// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file minmax_test.cpp
/// @brief Min/max selection rules, gfx1201 regression cases, and scalar/SIMD agreement.

#include "rocjitsu/isa/arch/amdgpu/shared/fp_format.h"
#include "rocjitsu/isa/arch/amdgpu/shared/minmax.h"
#include "util/simd.h"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace {

namespace cmp = rocjitsu::amdgpu::comparison;
namespace fmt = rocjitsu::amdgpu::fp_format;
namespace mm = rocjitsu::amdgpu::minmax;
using mm::Nan;

constexpr cmp::Policy kKeep{false};
constexpr cmp::Policy kFlush{true};

// ---------------------------------------------------------------------------
// Reference model: compare decoded values, with gfx1201 NaN and signed-zero rules.
// ---------------------------------------------------------------------------

template <typename Fmt> struct Ref {
  using L = typename Fmt::Lane;
  static constexpr L kMant = (L{1} << Fmt::kMantissaBits) - 1;

  static bool nan(L x) { return (x & Fmt::kInfinity) == Fmt::kInfinity && (x & kMant) != 0; }

  static L prepare(L x, bool flush) {
    x &= Fmt::kBits;
    if (flush && (x & Fmt::kInfinity) == 0)
      x &= Fmt::kSign;
    return x;
  }

  static double value(L x) {
    const bool negative = (x & Fmt::kSign) != 0;
    const int bias = (1 << (Fmt::kExponentBits - 1)) - 1;
    const int exponent = static_cast<int>((x & Fmt::kInfinity) >> Fmt::kMantissaBits);
    const double mantissa = static_cast<double>(x & kMant);
    double v;
    if (exponent == 0)
      v = std::ldexp(mantissa, 1 - bias - static_cast<int>(Fmt::kMantissaBits));
    else if ((x & Fmt::kInfinity) == Fmt::kInfinity)
      v = HUGE_VAL;
    else
      v = std::ldexp(mantissa + std::ldexp(1.0, Fmt::kMantissaBits),
                     exponent - bias - static_cast<int>(Fmt::kMantissaBits));
    return negative ? -v : v;
  }

  // Inputs have already been masked and flushed.
  static L select(bool maximum, Nan rule, L a, L b) {
    const L quiet = L{1} << (Fmt::kMantissaBits - 1);
    if (rule == Nan::PROPAGATE) {
      if (nan(a))
        return a | quiet;
      if (nan(b))
        return b | quiet;
    } else {
      if (nan(a) && nan(b))
        return a | quiet;
      if (nan(a))
        return b;
      if (nan(b))
        return a;
    }
    const double va = value(a);
    const double vb = value(b);
    if (va == vb) {
      if (va != 0.0)
        return a;
      // Equal numeric values can still be different signed zeros: -0 < +0.
      const bool a_negative = (a & Fmt::kSign) != 0;
      return (a_negative == maximum) ? b : a;
    }
    return maximum ? (va > vb ? a : b) : (va < vb ? a : b);
  }

  static L select(bool maximum, Nan rule, L a, L b, bool flush) {
    return select(maximum, rule, prepare(a, flush), prepare(b, flush));
  }

  static L median3(L a, L b, L c, bool flush) {
    a = prepare(a, flush);
    b = prepare(b, flush);
    c = prepare(c, flush);
    if (nan(a) || nan(b) || nan(c))
      return select(false, Nan::NUMBER, select(false, Nan::NUMBER, a, b), c);
    const L largest = select(true, Nan::NUMBER, select(true, Nan::NUMBER, a, b), c);
    // Match by encoding so -0 and +0 remain distinct, as observed on gfx1201.
    if (largest == a)
      return select(true, Nan::NUMBER, b, c);
    if (largest == b)
      return select(true, Nan::NUMBER, a, c);
    return select(true, Nan::NUMBER, a, b);
  }
};

template <typename Fmt>
using Fn2 = typename Fmt::Lane (*)(typename Fmt::Lane, typename Fmt::Lane, const cmp::Policy &);

template <typename Fmt, typename Op>
typename Fmt::Lane evaluate2(typename Fmt::Lane a, typename Fmt::Lane b, const cmp::Policy &p) {
  return mm::evaluate<Fmt, Op>(p, a, b);
}

struct Op2 {
  bool maximum;
  Nan rule;
};
constexpr std::array<Op2, 4> kOps = {
    {{false, Nan::PROPAGATE}, {true, Nan::PROPAGATE}, {false, Nan::NUMBER}, {true, Nan::NUMBER}}};

template <typename Fmt> std::array<Fn2<Fmt>, 4> ops2() {
  return {&evaluate2<Fmt, mm::Minimum>, &evaluate2<Fmt, mm::Maximum>, &evaluate2<Fmt, mm::MinNum>,
          &evaluate2<Fmt, mm::MaxNum>};
}

/// Signed zeros, the subnormal and normal boundaries, infinities, quiet and signaling NaNs.
template <typename Fmt> std::vector<typename Fmt::Lane> specials() {
  using L = typename Fmt::Lane;
  const L max_subnormal = Fmt::kMinNormal - 1;
  const std::vector<L> magnitudes = {0,
                                     1,
                                     2,
                                     max_subnormal,
                                     Fmt::kMinNormal,
                                     Fmt::kMinNormal + 1,
                                     Fmt::kInfinity - 1,
                                     Fmt::kInfinity,
                                     Fmt::kInfinity | 1,
                                     Fmt::kInfinity | 5,
                                     Fmt::kInfinity | Fmt::kQuiet,
                                     Fmt::kInfinity | Fmt::kQuiet | 3,
                                     Fmt::kMagnitude};
  std::vector<L> values;
  for (const L magnitude : magnitudes) {
    values.push_back(magnitude);
    values.push_back(magnitude | Fmt::kSign);
  }
  return values;
}

template <typename Fmt> void expect_binary(typename Fmt::Lane a, typename Fmt::Lane b) {
  const auto fns = ops2<Fmt>();
  for (const bool flush : {false, true})
    for (std::size_t i = 0; i < kOps.size(); ++i)
      ASSERT_EQ(fns[i](a, b, cmp::Policy{flush}),
                Ref<Fmt>::select(kOps[i].maximum, kOps[i].rule, a, b, flush))
          << "op " << i << " a 0x" << std::hex << a << " b 0x" << b << " flush " << flush;
}

template <typename Fmt>
void expect_ternary(typename Fmt::Lane a, typename Fmt::Lane b, typename Fmt::Lane c) {
  using R = Ref<Fmt>;
  for (const bool flush : {false, true}) {
    const cmp::Policy p{flush};
    const auto nested = [&](bool first, bool second, Nan rule) {
      return R::select(second, rule, R::select(first, rule, a, b, flush), c, flush);
    };
    ASSERT_EQ((mm::evaluate<Fmt, mm::Minimum3>(p, a, b, c)), nested(false, false, Nan::PROPAGATE));
    ASSERT_EQ((mm::evaluate<Fmt, mm::Maximum3>(p, a, b, c)), nested(true, true, Nan::PROPAGATE));
    ASSERT_EQ((mm::evaluate<Fmt, mm::MaximumMinimum>(p, a, b, c)),
              nested(true, false, Nan::PROPAGATE));
    ASSERT_EQ((mm::evaluate<Fmt, mm::MinimumMaximum>(p, a, b, c)),
              nested(false, true, Nan::PROPAGATE));
    ASSERT_EQ((mm::evaluate<Fmt, mm::Min3Num>(p, a, b, c)), nested(false, false, Nan::NUMBER));
    ASSERT_EQ((mm::evaluate<Fmt, mm::Max3Num>(p, a, b, c)), nested(true, true, Nan::NUMBER));
    ASSERT_EQ((mm::evaluate<Fmt, mm::MinMaxNum>(p, a, b, c)), nested(false, true, Nan::NUMBER));
    ASSERT_EQ((mm::evaluate<Fmt, mm::MaxMinNum>(p, a, b, c)), nested(true, false, Nan::NUMBER));
    ASSERT_EQ((mm::evaluate<Fmt, mm::Med3Num>(p, a, b, c)), R::median3(a, b, c, flush))
        << std::hex << "a 0x" << a << " b 0x" << b << " c 0x" << c << " flush " << flush;
  }
}

// ---------------------------------------------------------------------------
// Exhaustive F16 pairs against special values, plus random and three-source cases.
// ---------------------------------------------------------------------------

template <typename Fmt> void check_specials_and_random() {
  using L = typename Fmt::Lane;
  const auto values = specials<Fmt>();
  for (const L a : values)
    for (const L b : values) {
      expect_binary<Fmt>(a, b);
      for (const L c : values)
        expect_ternary<Fmt>(a, b, c);
    }
  std::mt19937_64 rng(0x5eed);
  const auto draw = [&](int i) {
    L x = static_cast<L>(rng()) & Fmt::kBits;
    if (i & 1)
      x &= ~Fmt::kInfinity; // Include subnormal inputs frequently.
    if (i & 2)
      x = values[rng() % values.size()];
    return x;
  };
  for (int i = 0; i < 20000; ++i) {
    const L a = draw(i), b = draw(i >> 1), c = draw(i >> 2);
    expect_binary<Fmt>(a, b);
    expect_ternary<Fmt>(a, b, c);
  }
}

TEST(MinmaxTest, F16EveryEncodingAgainstSpecials) {
  for (uint32_t a = 0; a <= 0xffffu; ++a)
    for (const uint32_t b : specials<fmt::F16>()) {
      expect_binary<fmt::F16>(a, b);
      expect_binary<fmt::F16>(b, a);
    }
}

TEST(MinmaxTest, F16SpecialsAndRandom) { check_specials_and_random<fmt::F16>(); }
TEST(MinmaxTest, F32SpecialsAndRandom) { check_specials_and_random<fmt::F32>(); }
TEST(MinmaxTest, F64SpecialsAndRandom) { check_specials_and_random<fmt::F64>(); }

// ---------------------------------------------------------------------------
// Regression cases from gfx1201 hardware results.
// ---------------------------------------------------------------------------

TEST(MinmaxTest, MatchesGfx1201MinimumMaximum) {
  using F32 = fmt::F32;
  // The first NaN in source order wins, quieted, with sign and payload; a quiet
  // src0 beats a signaling src1.
  EXPECT_EQ((mm::evaluate<F32, mm::Maximum>(kKeep, 0x7f800001u, 0u)), 0x7fc00001u);
  EXPECT_EQ((mm::evaluate<F32, mm::Maximum>(kKeep, 0xffc00000u, 0x7f800001u)), 0xffc00000u);
  EXPECT_EQ((mm::evaluate<F32, mm::Minimum>(kKeep, 1u, 0x7f800001u)), 0x7fc00001u);
  // Input flushing preserves the sign; selection does not flush its output.
  EXPECT_EQ((mm::evaluate<F32, mm::Minimum>(kFlush, 0x807fffffu, 0u)), 0x80000000u);
  EXPECT_EQ((mm::evaluate<F32, mm::Minimum>(kKeep, 0x807fffffu, 0u)), 0x807fffffu);
  EXPECT_EQ((mm::evaluate<F32, mm::Maximum>(kKeep, 1u, 0u)), 1u);
  EXPECT_EQ((mm::evaluate<F32, mm::Maximum>(kKeep, 0x80000000u, 0u)), 0u);
  EXPECT_EQ((mm::evaluate<fmt::F64, mm::Maximum>(kKeep, uint64_t{0x7ff0000000000001}, uint64_t{0})),
            uint64_t{0x7ff8000000000001});
  EXPECT_EQ((mm::evaluate<fmt::F16, mm::Maximum>(kKeep, 0x7c01u, 0u)), 0x7e01u);
  EXPECT_EQ((mm::evaluate<fmt::F16, mm::Minimum>(kFlush, 0x83ffu, 0u)), 0x8000u);
  // Nested selections keep the inner NaN's payload when src2 is also NaN.
  EXPECT_EQ((mm::evaluate<F32, mm::MaximumMinimum>(kKeep, 0u, 0x7f800001u, 0x7fc00000u)),
            0x7fc00001u);
  EXPECT_EQ((mm::evaluate<fmt::F16, mm::Maximum3>(kKeep, 0xfd2du, 0x21c8u, 0xb723u)), 0xff2du);
}

TEST(MinmaxTest, MatchesGfx1201NumForms) {
  using F32 = fmt::F32;
  EXPECT_EQ((mm::evaluate<F32, mm::MaxNum>(kKeep, 0x80000000u, 0u)), 0u);
  EXPECT_EQ((mm::evaluate<F32, mm::MinNum>(kKeep, 0x7f800001u, 0x7fc00000u)), 0x7fc00001u);
  EXPECT_EQ((mm::evaluate<F32, mm::MinNum>(kKeep, 0x7f800001u, 0x3f800000u)), 0x3f800000u);
  EXPECT_EQ((mm::evaluate<fmt::F64, mm::MaxNum>(kFlush, uint64_t{1}, uint64_t{0})), uint64_t{0});
  EXPECT_EQ((mm::evaluate<fmt::F16, mm::MinNum>(kKeep, 0xfe00u, 0x7e00u)), 0xfe00u);
  EXPECT_EQ((mm::evaluate<F32, mm::MinMaxNum>(kKeep, 0x7f800001u, 0x7fc00000u, 0x3f800000u)),
            0x3f800000u);
  EXPECT_EQ((mm::evaluate<F32, mm::Med3Num>(kKeep, 0x7fc00000u, 1u, 0u)), 0u);
  EXPECT_EQ((mm::evaluate<F32, mm::Med3Num>(kKeep, 0x80000000u, 0u, 0x80000000u)), 0x80000000u);
  EXPECT_EQ((mm::evaluate<fmt::F16, mm::Med3Num>(kFlush, 1u, 0u, 1u)), 0u);
}

// ---------------------------------------------------------------------------
// SIMD lanes select the same source as the scalar path.
// ---------------------------------------------------------------------------

#if __has_include(<experimental/simd>)
template <typename Fmt, typename V> void expect_simd_matches_scalar() {
  using L = typename Fmt::Lane;
  constexpr std::size_t W = V::size();
  std::vector<L> values = specials<Fmt>();
  std::mt19937_64 rng(0xc0ffee);
  while (values.size() % W != 0 || values.size() < 4 * W)
    values.push_back(static_cast<L>(rng()) & ~Fmt::kInfinity);
  for (const cmp::Policy p : {kKeep, kFlush})
    for (std::size_t rotate = 0; rotate < W; ++rotate)
      for (std::size_t base = 0; base < values.size(); base += W) {
        alignas(64) std::array<L, W> a{}, b{}, c{};
        for (std::size_t i = 0; i < W; ++i) {
          a[i] = values[base + i];
          b[i] = values[(base + i + rotate * 3 + 1) % values.size()];
          c[i] = values[(base + i + rotate * 7 + 2) % values.size()];
        }
        const V va(a.data(), util::stdx::element_aligned);
        const V vb(b.data(), util::stdx::element_aligned);
        const V vc(c.data(), util::stdx::element_aligned);
        const V max_prop = mm::evaluate<Fmt, mm::Maximum>(p, va, vb);
        const V min_num = mm::evaluate<Fmt, mm::MinNum>(p, va, vb);
        const V minmax = mm::evaluate<Fmt, mm::MinimumMaximum>(p, va, vb, vc);
        const V med = mm::evaluate<Fmt, mm::Med3Num>(p, va, vb, vc);
        for (std::size_t i = 0; i < W; ++i) {
          ASSERT_EQ(L(max_prop[i]), (mm::evaluate<Fmt, mm::Maximum>(p, a[i], b[i])));
          ASSERT_EQ(L(min_num[i]), (mm::evaluate<Fmt, mm::MinNum>(p, a[i], b[i])));
          ASSERT_EQ(L(minmax[i]), (mm::evaluate<Fmt, mm::MinimumMaximum>(p, a[i], b[i], c[i])));
          ASSERT_EQ(L(med[i]), (mm::evaluate<Fmt, mm::Med3Num>(p, a[i], b[i], c[i])));
        }
      }
}
#endif

TEST(MinmaxTest, SimdMatchesScalar) {
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
