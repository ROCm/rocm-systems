// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file comparison_test.cpp
/// @brief Unit tests for the staged floating-point VOPC relations.

#include "rocjitsu/isa/arch/amdgpu/shared/comparison.h"
#include "util/simd.h"

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

namespace {

namespace cmp = rocjitsu::amdgpu::comparison;

constexpr cmp::Policy kKeep{false};
constexpr cmp::Policy kFlush{true};

// ---------------------------------------------------------------------------
// Reference: decode to double, flush by hand, compare with host IEEE operators.
// ---------------------------------------------------------------------------

template <typename Fmt> double decode(typename Fmt::Lane bits, bool flush) {
  bits &= Fmt::kBits;
  const bool negative = (bits & Fmt::kSign) != 0;
  const uint64_t exponent = (bits & Fmt::kInfinity) >> Fmt::kMantissaBits;
  const uint64_t mantissa = bits & ((typename Fmt::Lane{1} << Fmt::kMantissaBits) - 1);
  double value;
  if (exponent == Fmt::kExponentMax)
    value = mantissa ? std::numeric_limits<double>::quiet_NaN()
                     : std::numeric_limits<double>::infinity();
  else if (exponent == 0)
    value =
        flush ? 0.0
              : std::ldexp(static_cast<double>(mantissa), 2 - (1 << (Fmt::kExponentBits - 1)) -
                                                              static_cast<int>(Fmt::kMantissaBits));
  else
    value = std::ldexp(static_cast<double>(mantissa | (uint64_t{1} << Fmt::kMantissaBits)),
                       static_cast<int>(exponent) - ((1 << (Fmt::kExponentBits - 1)) - 1) -
                           static_cast<int>(Fmt::kMantissaBits));
  return negative ? -value : value;
}

// Indexed by VOPC opcode: F LT EQ LE GT LG GE O U NGE NLG NGT NLE NEQ NLT T.
bool reference(unsigned opcode, double a, double b) {
  switch (opcode) {
  case 0:
    return false;
  case 1:
    return a < b;
  case 2:
    return a == b;
  case 3:
    return a <= b;
  case 4:
    return a > b;
  case 5:
    return a < b || a > b;
  case 6:
    return a >= b;
  case 7:
    return !std::isnan(a) && !std::isnan(b);
  default:
    return !reference(15 - opcode, a, b);
  }
}

template <typename Fmt, typename Rel> bool scalar(uint64_t a, uint64_t b, cmp::Policy policy) {
  using L = typename Fmt::Lane;
  return cmp::evaluate<Fmt, Rel>(static_cast<L>(a), static_cast<L>(b), policy);
}

template <typename Fmt> using RelationFn = bool (*)(uint64_t, uint64_t, cmp::Policy);

template <typename Fmt> std::array<RelationFn<Fmt>, 16> relations() {
  return {&scalar<Fmt, cmp::F>,   &scalar<Fmt, cmp::Lt>,  &scalar<Fmt, cmp::Eq>,
          &scalar<Fmt, cmp::Le>,  &scalar<Fmt, cmp::Gt>,  &scalar<Fmt, cmp::Lg>,
          &scalar<Fmt, cmp::Ge>,  &scalar<Fmt, cmp::O>,   &scalar<Fmt, cmp::U>,
          &scalar<Fmt, cmp::Nge>, &scalar<Fmt, cmp::Nlg>, &scalar<Fmt, cmp::Ngt>,
          &scalar<Fmt, cmp::Nle>, &scalar<Fmt, cmp::Neq>, &scalar<Fmt, cmp::Nlt>,
          &scalar<Fmt, cmp::T>};
}

/// Signed zeros, the subnormal and normal boundaries, infinities and NaNs.
template <typename Fmt> std::vector<uint64_t> specials() {
  using L = typename Fmt::Lane;
  const L max_subnormal = (L{1} << Fmt::kMantissaBits) - 1;
  const L min_normal = L{1} << Fmt::kMantissaBits;
  const L quiet = L{1} << (Fmt::kMantissaBits - 1);
  const std::vector<L> magnitudes = {0,
                                     1,
                                     2,
                                     max_subnormal,
                                     min_normal,
                                     min_normal + 1,
                                     Fmt::kInfinity - 1,
                                     Fmt::kInfinity,
                                     Fmt::kInfinity | 1,
                                     Fmt::kInfinity | quiet,
                                     Fmt::kMagnitude};
  std::vector<uint64_t> values;
  for (const L magnitude : magnitudes) {
    values.push_back(magnitude);
    values.push_back(magnitude | Fmt::kSign);
  }
  return values;
}

template <typename Fmt> void expect_matches_reference(uint64_t a, uint64_t b) {
  const auto fns = relations<Fmt>();
  for (const bool flush : {false, true}) {
    const double da = decode<Fmt>(static_cast<typename Fmt::Lane>(a), flush);
    const double db = decode<Fmt>(static_cast<typename Fmt::Lane>(b), flush);
    for (unsigned opcode = 0; opcode < 16; ++opcode)
      ASSERT_EQ(fns[opcode](a, b, cmp::Policy{flush}), reference(opcode, da, db))
          << "opcode " << opcode << " a 0x" << std::hex << a << " b 0x" << b << " flush " << flush;
  }
}

// ---------------------------------------------------------------------------
// Agreement with IEEE relations.
// ---------------------------------------------------------------------------

TEST(ComparisonTest, F16EveryEncodingAgainstSpecials) {
  for (uint64_t a = 0; a <= 0xffffu; ++a)
    for (const uint64_t b : specials<cmp::F16>()) {
      expect_matches_reference<cmp::F16>(a, b);
      expect_matches_reference<cmp::F16>(b, a);
    }
}

template <typename Fmt> void check_specials_and_random() {
  const auto values = specials<Fmt>();
  for (const uint64_t a : values)
    for (const uint64_t b : values)
      expect_matches_reference<Fmt>(a, b);
  std::mt19937_64 rng(0x5eed);
  for (int i = 0; i < 20000; ++i) {
    // Clear random exponent bits half the time, so subnormals are common.
    uint64_t a = rng() & Fmt::kBits;
    uint64_t b = rng() & Fmt::kBits;
    if (i & 1)
      a &= ~static_cast<uint64_t>(Fmt::kInfinity);
    if (i & 2)
      b &= ~static_cast<uint64_t>(Fmt::kInfinity);
    expect_matches_reference<Fmt>(a, b);
  }
}

TEST(ComparisonTest, F32SpecialsAndRandom) { check_specials_and_random<cmp::F32>(); }

TEST(ComparisonTest, F64SpecialsAndRandom) { check_specials_and_random<cmp::F64>(); }

TEST(ComparisonTest, IgnoresBitsAboveTheF16Encoding) {
  EXPECT_TRUE((cmp::evaluate<cmp::F16, cmp::Eq>(0xabcd3c00u, 0x00003c00u, kKeep)));
  EXPECT_TRUE((cmp::evaluate<cmp::F16, cmp::Lt>(0xffff0000u, 0x00003c00u, kKeep)));
}

// ---------------------------------------------------------------------------
// Physical gfx1201 lanes. With input denormals disabled, a subnormal compares
// as a zero of the same sign; with them enabled it compares by value.
// ---------------------------------------------------------------------------

TEST(ComparisonTest, MatchesGfx1201InputFlush) {
  EXPECT_FALSE((cmp::evaluate<cmp::F32, cmp::Lt>(0x80000001u, 0x00000000u, kFlush)));
  EXPECT_TRUE((cmp::evaluate<cmp::F32, cmp::Lt>(0x80000001u, 0x00000000u, kKeep)));
  EXPECT_FALSE((cmp::evaluate<cmp::F32, cmp::Lt>(0x807fffffu, 0x80000000u, kFlush)));
  EXPECT_TRUE((cmp::evaluate<cmp::F32, cmp::Lt>(0x807fffffu, 0x80000000u, kKeep)));
  EXPECT_FALSE(
      (cmp::evaluate<cmp::F64, cmp::Lt>(uint64_t{0x8000000000000001}, uint64_t{0}, kFlush)));
  EXPECT_TRUE((cmp::evaluate<cmp::F64, cmp::Lt>(uint64_t{0x8000000000000001}, uint64_t{0}, kKeep)));
  EXPECT_FALSE((cmp::evaluate<cmp::F16, cmp::Lt>(0x83ffu, 0x0000u, kFlush)));
  EXPECT_TRUE((cmp::evaluate<cmp::F16, cmp::Lt>(0x83ffu, 0x0000u, kKeep)));
}

TEST(ComparisonTest, AppliesModifiersBeforeTheFlush) {
  // v_cmp_lt_f16 -a, |b| with a = 0x0001: NEG makes -tiny, the flush makes -0.
  EXPECT_FALSE((cmp::evaluate<cmp::F16, cmp::Lt>(0x0001u, 0x0000u, 0u, 1u, kFlush)));
  EXPECT_TRUE((cmp::evaluate<cmp::F16, cmp::Lt>(0x0001u, 0x0000u, 0u, 1u, kKeep)));
  // ABS clears the sign before NEG sets it again.
  EXPECT_EQ(cmp::modify<cmp::F32>(0x80000002u, true, true), 0x80000002u);
  EXPECT_EQ(cmp::modify<cmp::F32>(0x80000002u, true, false), 0x00000002u);
  // The src1 modifiers come from bit 1 of each field.
  EXPECT_FALSE((cmp::evaluate<cmp::F32, cmp::Gt>(0x3f800000u, 0x3f800000u, 0u, 1u, kKeep)));
  EXPECT_TRUE((cmp::evaluate<cmp::F32, cmp::Gt>(0x3f800000u, 0x3f800000u, 0u, 2u, kKeep)));
}

TEST(ComparisonTest, FlushKeepsSignAndSpecials) {
  EXPECT_EQ(cmp::flush_input<cmp::F32>(0x807fffffu, kFlush), 0x80000000u);
  EXPECT_EQ(cmp::flush_input<cmp::F32>(0x007fffffu, kFlush), 0x00000000u);
  EXPECT_EQ(cmp::flush_input<cmp::F32>(0x00800000u, kFlush), 0x00800000u);
  EXPECT_EQ(cmp::flush_input<cmp::F32>(0x7f800001u, kFlush), 0x7f800001u);
  EXPECT_EQ(cmp::flush_input<cmp::F32>(0x807fffffu, kKeep), 0x807fffffu);
  EXPECT_EQ(cmp::flush_input<cmp::F16>(0x83ffu, kFlush), 0x8000u);
  EXPECT_EQ(cmp::flush_input<cmp::F64>(uint64_t{0x800fffffffffffff}, kFlush),
            uint64_t{0x8000000000000000});
}

TEST(ComparisonTest, NegatedRelationsAreTrueOnNaN) {
  constexpr uint32_t kQuiet = 0x7fc00000u;
  constexpr uint32_t kSignaling = 0xff800001u;
  EXPECT_FALSE((cmp::evaluate<cmp::F32, cmp::Lt>(kQuiet, 0u, kKeep)));
  EXPECT_TRUE((cmp::evaluate<cmp::F32, cmp::Nlt>(kQuiet, 0u, kKeep)));
  EXPECT_FALSE((cmp::evaluate<cmp::F32, cmp::Eq>(kSignaling, kSignaling, kKeep)));
  EXPECT_TRUE((cmp::evaluate<cmp::F32, cmp::Neq>(kSignaling, kSignaling, kKeep)));
  EXPECT_FALSE((cmp::evaluate<cmp::F32, cmp::Lg>(0u, kQuiet, kKeep)));
  EXPECT_TRUE((cmp::evaluate<cmp::F32, cmp::U>(0u, kQuiet, kKeep)));
  EXPECT_FALSE((cmp::evaluate<cmp::F32, cmp::O>(0u, kQuiet, kKeep)));
  EXPECT_FALSE((cmp::evaluate<cmp::F32, cmp::F>(kQuiet, kQuiet, kKeep)));
  EXPECT_TRUE((cmp::evaluate<cmp::F32, cmp::T>(kQuiet, kQuiet, kKeep)));
}

TEST(ComparisonTest, SignedZerosAreEqual) {
  EXPECT_TRUE((cmp::evaluate<cmp::F32, cmp::Eq>(0x80000000u, 0x00000000u, kKeep)));
  EXPECT_FALSE((cmp::evaluate<cmp::F32, cmp::Lt>(0x80000000u, 0x00000000u, kKeep)));
  EXPECT_FALSE((cmp::evaluate<cmp::F32, cmp::Lg>(0x80000000u, 0x00000000u, kKeep)));
}

// ---------------------------------------------------------------------------
// MODE gating.
// ---------------------------------------------------------------------------

TEST(ComparisonTest, FlushesWhenInputDenormalsAreDisabled) {
  EXPECT_TRUE(cmp::Policy::make(0u).flush_inputs);
  EXPECT_TRUE(cmp::Policy::make(2u).flush_inputs);
  EXPECT_FALSE(cmp::Policy::make(1u).flush_inputs);
  EXPECT_FALSE(cmp::Policy::make(3u).flush_inputs);
}

// ---------------------------------------------------------------------------
// SIMD lanes evaluate the same relation as the scalar path.
// ---------------------------------------------------------------------------

#if __has_include(<experimental/simd>)
template <typename Fmt, typename Rel, typename V> void expect_simd_matches_scalar() {
  using L = typename Fmt::Lane;
  constexpr std::size_t W = V::size();
  std::vector<L> values;
  for (const uint64_t v : specials<Fmt>())
    values.push_back(static_cast<L>(v));
  std::mt19937_64 rng(0xc0ffee);
  while (values.size() % W != 0 || values.size() < 4 * W)
    values.push_back(static_cast<L>(rng()) & ~Fmt::kInfinity);
  for (const cmp::Policy policy : {kKeep, kFlush})
    for (std::size_t rotate = 0; rotate < W; ++rotate)
      for (std::size_t base = 0; base < values.size(); base += W) {
        alignas(64) std::array<L, W> a{};
        alignas(64) std::array<L, W> b{};
        for (std::size_t i = 0; i < W; ++i) {
          a[i] = values[base + i];
          b[i] = values[(base + i + rotate * 3 + 1) % values.size()];
        }
        const V va(a.data(), util::stdx::element_aligned);
        const V vb(b.data(), util::stdx::element_aligned);
        const uint64_t bits = util::simd_mask_to_bits(cmp::evaluate<Fmt, Rel>(va, vb, policy));
        for (std::size_t i = 0; i < W; ++i)
          ASSERT_EQ(((bits >> i) & 1u) != 0, (cmp::evaluate<Fmt, Rel>(a[i], b[i], policy)))
              << "lane " << i << " a 0x" << std::hex << a[i] << " b 0x" << b[i];
      }
}

template <typename Fmt, typename V> void expect_every_simd_relation_matches_scalar() {
  expect_simd_matches_scalar<Fmt, cmp::F, V>();
  expect_simd_matches_scalar<Fmt, cmp::Lt, V>();
  expect_simd_matches_scalar<Fmt, cmp::Eq, V>();
  expect_simd_matches_scalar<Fmt, cmp::Le, V>();
  expect_simd_matches_scalar<Fmt, cmp::Gt, V>();
  expect_simd_matches_scalar<Fmt, cmp::Lg, V>();
  expect_simd_matches_scalar<Fmt, cmp::Ge, V>();
  expect_simd_matches_scalar<Fmt, cmp::O, V>();
  expect_simd_matches_scalar<Fmt, cmp::U, V>();
  expect_simd_matches_scalar<Fmt, cmp::Nge, V>();
  expect_simd_matches_scalar<Fmt, cmp::Nlg, V>();
  expect_simd_matches_scalar<Fmt, cmp::Ngt, V>();
  expect_simd_matches_scalar<Fmt, cmp::Nle, V>();
  expect_simd_matches_scalar<Fmt, cmp::Neq, V>();
  expect_simd_matches_scalar<Fmt, cmp::Nlt, V>();
  expect_simd_matches_scalar<Fmt, cmp::T, V>();
}
#endif

TEST(ComparisonTest, SimdMatchesScalar) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable";
  } else {
#if __has_include(<experimental/simd>)
    expect_every_simd_relation_matches_scalar<cmp::F16, util::native<uint32_t>>();
    expect_every_simd_relation_matches_scalar<cmp::F32, util::native<uint32_t>>();
#if UTIL_SIMD_BROKEN_NATIVE_64BIT_MASKS
    expect_every_simd_relation_matches_scalar<cmp::F64, util::stdx::fixed_size_simd<uint64_t, 1>>();
#else
    expect_every_simd_relation_matches_scalar<cmp::F64, util::native<uint64_t>>();
#endif
#endif
  }
}

} // namespace
