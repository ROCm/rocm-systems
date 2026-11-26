// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file division_test.cpp
/// @brief Numerical goldens for division instructions through public arithmetic APIs.

#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_fp.h"

#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

namespace {
// Pack one active lane so the existing numerical goldens exercise public APIs.
// CPU baseline is intentional: SIMD coverage lives in the arithmetic tests.
template <typename Float> struct DivisionOperands {
  uint32_t words[4][2][32]{};
  const uint32_t *a[2] = {words[0][0], words[0][1]};
  const uint32_t *b[2] = {words[1][0], words[1][1]};
  const uint32_t *c[2] = {words[2][0], words[2][1]};
  uint32_t *d[2] = {words[3][0], words[3][1]};

  DivisionOperands(Float x, Float y, Float z) {
    const Float values[] = {x, y, z};
    for (int operand = 0; operand < 3; ++operand) {
      if constexpr (sizeof(Float) == 4) {
        words[operand][0][0] = std::bit_cast<uint32_t>(values[operand]);
      } else {
        const uint64_t bits = std::bit_cast<uint64_t>(values[operand]);
        words[operand][0][0] = static_cast<uint32_t>(bits);
        words[operand][1][0] = static_cast<uint32_t>(bits >> 32);
      }
    }
  }

  Float result() const {
    if constexpr (sizeof(Float) == 4) {
      return std::bit_cast<float>(words[3][0][0]);
    } else {
      return std::bit_cast<double>(uint64_t(words[3][0][0]) | (uint64_t(words[3][1][0]) << 32));
    }
  }
};

// Translate the legacy guest MODE representation without changing host controls.
uint64_t division_flags(uint32_t rounding, uint32_t denorm) {
  return RJ_ARITH_SEMANTICS_COMPATIBILITY | (uint64_t(rounding) << 24) |
         ((denorm & 1) ? 0 : RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) |
         ((denorm & 2) ? 0 : RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS);
}

template <typename Float> struct DivisionScaleResult {
  Float value;
  bool post_scale;
};

// Return the rescaled operand and its lane's condition bit.
template <typename Float>
DivisionScaleResult<Float> div_scale(Float a, Float b, Float c, uint32_t rounding = 0,
                                     uint32_t denorm = 3) {
  DivisionOperands<Float> operands(a, b, c);
  uint32_t condition = 0;
  auto api = [] {
    if constexpr (sizeof(Float) == 4) {
      return rj_arith_v_div_scale_f32_w32_gfx942;
    } else {
      return rj_arith_v_div_scale_f64_w32_gfx942;
    }
  }();
  EXPECT_EQ(api(division_flags(rounding, denorm), 1, 0, operands.d, &condition, operands.a,
                operands.b, operands.c),
            RJ_ARITH_SUCCESS);
  return {operands.result(), (condition & 1) != 0};
}

// Return the inherited fused, conditionally scaled FMA result.
template <typename Float>
Float div_fmas(Float a, Float b, Float c, bool scale, uint32_t rounding, uint32_t denorm) {
  DivisionOperands<Float> operands(a, b, c);
  auto api = [] {
    if constexpr (sizeof(Float) == 4) {
      return rj_arith_v_div_fmas_f32_w32_gfx942;
    } else {
      return rj_arith_v_div_fmas_f64_w32_gfx942;
    }
  }();
  uint32_t status = 0;
  EXPECT_EQ(api(division_flags(rounding, denorm), 1, 0, operands.d, operands.a, operands.b,
                operands.c, scale ? 1 : 0, &status),
            RJ_ARITH_SUCCESS);
  return operands.result();
}

// Apply quotient recovery and exceptional-operand handling to one lane.
template <typename Float>
Float div_fixup(Float a, Float b, Float c, uint32_t rounding, uint32_t denorm) {
  DivisionOperands<Float> operands(a, b, c);
  auto api = [] {
    if constexpr (sizeof(Float) == 4) {
      return rj_arith_v_div_fixup_f32_w32_gfx942;
    } else {
      return rj_arith_v_div_fixup_f64_w32_gfx942;
    }
  }();
  uint32_t status = 0;
  EXPECT_EQ(api(division_flags(rounding, denorm), 1, 0, operands.d, operands.a, operands.b,
                operands.c, &status),
            RJ_ARITH_SUCCESS);
  return operands.result();
}

// Expected instruction results were checked on physical gfx1100 and gfx1201.
// Unlike scalar-vs-SIMD comparisons, these are independent numerical oracles.
TEST(DivisionTest, ScaleUsesEncodedExponentBoundaries) {
  struct Case {
    float n, d, ns, ds;
    bool vcc;
  };
  constexpr Case cases[] = {
      {0x1p-140f, 1.f, 0x1p-76f, 1.f, true},
      {0x1p-10f, 0x1p127f, 0x1p-10f, 0x1p63f, true},
      {0x1p-30f, 0x1p70f, 0x1p34f, 0x1p70f, true},
      {0x1p-24f, 0x1p65f, 0x1p-24f, 0x1p65f, false},
      {0x1p100f, 0x1p-20f, 0x1p100f, 0x1p44f, true},
      {0x1p-53f, 0x1p-149f, 0x1p11f, 0x1p-85f, false},
      {0x1p-103f, 0x1p-126f, 0x1p-39f, 0x1p-62f, false},
      {0x1p-102f, 0x1p-126f, 0x1p-102f, 0x1p-126f, false},
      {1.f, 0x1p126f, 1.f, 0x1p62f, true},
      {0x1p31f, 0x1p126f, 0x1p-33f, 0x1p62f, false},
  };
  for (const Case &test : cases) {
    const DivisionScaleResult<float> n = div_scale(test.n, test.d, test.n);
    const DivisionScaleResult<float> d = div_scale(test.d, test.d, test.n);
    EXPECT_EQ(n.value, test.ns);
    EXPECT_EQ(d.value, test.ds);
    EXPECT_EQ(n.post_scale, test.vcc);
    EXPECT_EQ(d.post_scale, test.vcc);
  }
  EXPECT_EQ(div_scale(0x1p-1040, 1., 0x1p-1040).value, 0x1p-912);
  EXPECT_EQ(div_scale(0x1p1023, 0x1p1023, 0x1p-10).value, 0x1p895);
  EXPECT_EQ(div_scale(0x1p-970, 0x1p-1022, 0x1p-970).value, 0x1p-842);
  EXPECT_EQ(div_scale(0x1p-969, 0x1p-1022, 0x1p-969).value, 0x1p-969);

  // Preserve infinities and quiet signaling NaNs, unless a zero original
  // operand takes precedence and produces the canonical negative quiet NaN.
  const float inf32 = std::numeric_limits<float>::infinity();
  const double inf64 = std::numeric_limits<double>::infinity();
  const float snan32 = std::bit_cast<float>(0x7f800123u);
  const double snan64 = std::bit_cast<double>(0x7ff0000000000123ULL);
  EXPECT_EQ(div_scale(inf32, 1.f, inf32).value, inf32);
  EXPECT_EQ(div_scale(inf64, 1., inf64).value, inf64);
  EXPECT_EQ(std::bit_cast<uint32_t>(div_scale(snan32, 1.f, snan32).value), 0x7fc00123u);
  EXPECT_EQ(std::bit_cast<uint64_t>(div_scale(snan64, 1., snan64).value), 0x7ff8000000000123ULL);
  EXPECT_EQ(std::bit_cast<uint32_t>(div_scale(inf32, 0.f, inf32).value), 0xffc00000u);
  EXPECT_EQ(std::bit_cast<uint64_t>(div_scale(inf64, 0., inf64).value), 0xfff8000000000000ULL);
  EXPECT_EQ(std::bit_cast<uint32_t>(div_scale(inf32, inf32, 0.f).value), 0xffc00000u);
  EXPECT_EQ(std::bit_cast<uint64_t>(div_scale(inf64, inf64, 0.).value), 0xfff8000000000000ULL);
}

TEST(DivisionTest, FmasScaleDirectionUsesAddendNotSum) {
  EXPECT_EQ(div_fmas(0.f, 0.f, 1.f, true, 0, 3), 0x1p-64f);
  EXPECT_EQ(div_fmas(0.f, 0.f, 2.f, true, 0, 3), 0x1p65f);
  EXPECT_EQ(div_fmas(1.f, 2.f, 0.f, true, 0, 3), 0x1p-63f);
  EXPECT_EQ(div_fmas(0.5f, 1.f, 1.5f, true, 0, 3), 0x1p-63f);
  EXPECT_EQ(div_fmas(0., 0., 1., true, 0, 3), 0x1p-128);
  EXPECT_EQ(div_fmas(0., 0., 2., true, 0, 3), 0x1p129);
  EXPECT_EQ(div_fmas(1., 2., 0., true, 0, 3), 0x1p-127);
}

TEST(DivisionTest, ScalingIsFusedWithRounding) {
  // The small product puts the result above/below a subnormal midpoint. Rounding
  // the FMA to its destination format before scaling would lose that distinction.
  EXPECT_EQ(std::bit_cast<uint32_t>(div_fmas(0x1p-100f, 0x1p-100f, 0x1p-86f, true, 0, 3)), 1u);
  EXPECT_EQ(std::bit_cast<uint32_t>(div_fmas(-0x1p-100f, 0x1p-100f, 0x1p-86f, true, 0, 3)), 0u);
  EXPECT_EQ(std::bit_cast<uint64_t>(div_fmas(0x1p-600, 0x1p-600, 0x1p-947, true, 0, 3)), 1u);
  EXPECT_EQ(std::bit_cast<uint64_t>(div_fmas(-0x1p-600, 0x1p-600, 0x1p-947, true, 0, 3)), 0u);
  // The unscaled product overflows, but the instruction's fused downscale does not.
  EXPECT_EQ(div_fmas(0x1p100f, 0x1p50f, 0.f, true, 0, 3), 0x1p86f);
  EXPECT_EQ(div_fmas(0x1p800, 0x1p300, 0., true, 0, 3), 0x1p972);
  EXPECT_EQ(div_fmas(0x1.000002p0f, 0x1.fffffcp-1f, -1.f, false, 0, 3), -0x1p-46f);
  EXPECT_EQ(div_fmas(0x1.0000000000001p0, 0x1.ffffffffffffep-1, -1., false, 0, 3), -0x1p-104);
}

TEST(DivisionTest, RoundingAndDenormModesAreExplicit) {
  // Guest rounding is explicit; the public API assumes the default host environment.
  for (uint32_t mode = 0; mode < 4; ++mode) {
    EXPECT_EQ(std::bit_cast<uint32_t>(div_fmas(1.f, 1.f, -1.f, false, mode, 3)),
              mode == 2 ? 0x80000000u : 0u);
    EXPECT_EQ(std::bit_cast<uint32_t>(div_fmas(0.f, 0.f, 0x1p-86f, true, mode, 3)),
              mode == 1 ? 1u : 0u);
    EXPECT_EQ(std::bit_cast<uint64_t>(div_fmas(0., 0., -0x1p-947, true, mode, 3)),
              mode == 2 ? 0x8000000000000001ULL : 0x8000000000000000ULL);
  }
  const float subnormal = std::bit_cast<float>(1u);
  for (uint32_t mode = 0; mode < 4; ++mode) {
    const DivisionScaleResult<float> scaled = div_scale(subnormal, 1.f, subnormal, 0, mode);
    EXPECT_EQ(std::isnan(scaled.value), !(mode & 1u));
    EXPECT_TRUE(scaled.post_scale);
    EXPECT_EQ(std::bit_cast<uint32_t>(div_fmas(subnormal, 1.f, 0.f, false, 0, mode)),
              (mode & 2u) ? 1u : 0u); // FMAS never flushes its inputs.
    EXPECT_EQ(std::bit_cast<uint32_t>(div_fixup(subnormal, 1.f, 1.f, 0, mode)), 1u);
    EXPECT_EQ(std::bit_cast<uint32_t>(div_fixup(1.f, 1.f, subnormal, 0, mode)),
              (mode & 1u) ? 0x3f800000u : 0u);
  }
}

TEST(DivisionTest, FixupHandlesSignAndExceptionalQuotients) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  EXPECT_EQ(div_fixup(-3.f, 1.f, 1.f, 0, 3), 3.f);
  EXPECT_EQ(div_fixup(3., -1., 1., 0, 3), -3.);
  EXPECT_EQ(div_fixup(nan, 0x1p-130f, 0x1p100f, 0, 3), INFINITY);
  EXPECT_EQ(div_fixup(nan, 0x1p120f, 0x1p-100f, 0, 3), 0.f);
  EXPECT_EQ(std::bit_cast<uint32_t>(div_fixup(nan, 0x1p120f, -0x1p-100f, 2, 3)), 0x80000001u);
  EXPECT_EQ(div_fixup(nan, 0x1p-130f, 0x1p100f, 3, 3), std::numeric_limits<float>::max());
  EXPECT_TRUE(std::isnan(div_fixup(1.f, 0.f, 0.f, 0, 3)));
  EXPECT_EQ(std::bit_cast<uint32_t>(div_fixup(1.f, -1.f, 0.f, 0, 3)), 0x80000000u);
  EXPECT_EQ(std::bit_cast<uint32_t>(div_fmas(0.f, INFINITY, nan, false, 0, 3)), 0xffc00000u);
  EXPECT_EQ(std::bit_cast<uint32_t>(div_fixup(1.f, nan, std::bit_cast<float>(0xff800123u), 0, 3)),
            0xffc00123u);
}

} // namespace
