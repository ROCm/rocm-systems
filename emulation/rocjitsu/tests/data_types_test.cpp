// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "util/data_types.h"

#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <limits>

namespace {

using NarrowRne = uint8_t (*)(float);
using NarrowSr = uint8_t (*)(float, uint32_t);

void expect_deepest_subnormal_boundary(const char *name, float smallest_subnormal, NarrowRne rne,
                                       NarrowSr sr) {
  SCOPED_TRACE(name);
  const float midpoint = 0.5f * smallest_subnormal;
  const float above_midpoint = 0.75f * smallest_subnormal;
  EXPECT_EQ(rne(midpoint), 0x00);
  EXPECT_EQ(rne(above_midpoint), 0x01);
  EXPECT_EQ(sr(above_midpoint, 0), 0x00);
  EXPECT_EQ(sr(above_midpoint, 0xFFFFFFFFu), 0x01);
}

TEST(MxFloatNarrow, DeepestSubnormalBoundary) {
  expect_deepest_subnormal_boundary("fp8_e4m3", std::ldexp(1.0f, -9), util::f32_to_fp8_e4m3_rne,
                                    util::f32_to_fp8_e4m3_sr);
  expect_deepest_subnormal_boundary("fp8_e5m3", std::ldexp(1.0f, -17), util::f32_to_fp8_e5m3_rne,
                                    util::f32_to_fp8_e5m3_sr);
  expect_deepest_subnormal_boundary("bf8_e5m2", std::ldexp(1.0f, -16), util::f32_to_bf8_e5m2_rne,
                                    util::f32_to_bf8_e5m2_sr);
  expect_deepest_subnormal_boundary("fp4_e2m1", std::ldexp(1.0f, -1), util::f32_to_fp4_e2m1_rne,
                                    util::f32_to_fp4_e2m1_sr);
  expect_deepest_subnormal_boundary("fp6_e2m3", std::ldexp(1.0f, -3), util::f32_to_fp6_e2m3_rne,
                                    util::f32_to_fp6_e2m3_sr);
  expect_deepest_subnormal_boundary("bf6_e3m2", std::ldexp(1.0f, -4), util::f32_to_bf6_e3m2_rne,
                                    util::f32_to_bf6_e3m2_sr);
}

// ---- FP4 E2M1 (exhaustive — 16 values) ----

// Hardcoded truth table for positive FP4 E2M1 codes (independent of data_types.h).
// FP4 E2M1: 1s+2e+1m, bias=1, no NaN/Inf, max=6.0
// code -> value: exp=0 denorm: 0.0, 0.5; exp=1..3 normal: (1+m)*2^(exp-1)
static constexpr float kFp4TruthPositive[8] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
};

TEST(Fp4E2M1, ExhaustiveWiden) {
  for (uint8_t code = 0; code < 8; ++code) {
    EXPECT_EQ(util::fp4_e2m1_to_f32(code), kFp4TruthPositive[code]) << "code=" << int(code);
    float neg = util::fp4_e2m1_to_f32(code | 0x8);
    if (code == 0)
      EXPECT_TRUE(std::signbit(neg) && neg == 0.0f);
    else
      EXPECT_EQ(neg, -kFp4TruthPositive[code]) << "neg code=" << int(code);
  }
}

TEST(Fp4E2M1, RneRoundTrip) {
  for (uint8_t code = 0; code < 8; ++code) {
    float val = kFp4TruthPositive[code];
    uint8_t narrow = util::f32_to_fp4_e2m1_rne(val);
    EXPECT_EQ(narrow, code) << "code=" << int(code) << " val=" << val;
  }
}

TEST(Fp4E2M1, RneEdgeCases) {
  EXPECT_EQ(util::f32_to_fp4_e2m1_rne(std::numeric_limits<float>::quiet_NaN()), 0);
  EXPECT_EQ(util::f32_to_fp4_e2m1_rne(std::numeric_limits<float>::infinity()), 0x7);
  EXPECT_EQ(util::f32_to_fp4_e2m1_rne(-std::numeric_limits<float>::infinity()), 0xF);
  EXPECT_EQ(util::f32_to_fp4_e2m1_rne(7.0f), 0x7);
  EXPECT_EQ(util::f32_to_fp4_e2m1_rne(100.0f), 0x7);
  // 4.0 must map to code 0x6 (exp=3, mant=0), NOT saturate to 6.0
  EXPECT_EQ(util::f32_to_fp4_e2m1_rne(4.0f), 0x6);
}

TEST(Fp4E2M1, SrDeterministic) {
  uint8_t r = util::f32_to_fp4_e2m1_sr(1.0f, 0);
  EXPECT_EQ(r, 0x2);
  r = util::f32_to_fp4_e2m1_sr(6.0f, 0xFFFFFFFF);
  EXPECT_EQ(r, 0x7);
}

// ---- FP6 E2M3 (exhaustive — 64 values) ----

static float fp6_reference(uint8_t code) {
  uint32_t sign = (code >> 5) & 1;
  uint32_t exp = (code >> 3) & 0x3;
  uint32_t mant = code & 0x7;
  if (exp == 0 && mant == 0)
    return sign ? -0.0f : 0.0f;
  if (exp == 0) {
    float v = std::ldexp(static_cast<float>(mant), -3);
    return sign ? -v : v;
  }
  uint32_t f = (sign << 31) | ((exp + 127 - 1) << 23) | (mant << 20);
  return std::bit_cast<float>(f);
}

TEST(Fp6E2M3, ExhaustiveWiden) {
  for (uint8_t code = 0; code < 64; ++code) {
    float got = util::fp6_e2m3_to_f32(code);
    float ref = fp6_reference(code);
    EXPECT_EQ(std::bit_cast<uint32_t>(got), std::bit_cast<uint32_t>(ref)) << "code=" << int(code);
  }
}

TEST(Fp6E2M3, RneRoundTrip) {
  for (uint8_t code = 0; code < 32; ++code) {
    float val = util::fp6_e2m3_to_f32(code);
    uint8_t narrow = util::f32_to_fp6_e2m3_rne(val);
    EXPECT_EQ(narrow, code) << "code=" << int(code) << " val=" << val;
  }
}

TEST(Fp6E2M3, RneEdgeCases) {
  EXPECT_EQ(util::f32_to_fp6_e2m3_rne(std::numeric_limits<float>::quiet_NaN()), 0);
  EXPECT_EQ(util::f32_to_fp6_e2m3_rne(std::numeric_limits<float>::infinity()), 0x1F);
  EXPECT_EQ(util::f32_to_fp6_e2m3_rne(10.0f), 0x1F);
  // 4.0 maps to exp=3, mant=0 (code 0x18)
  EXPECT_EQ(util::f32_to_fp6_e2m3_rne(4.0f), 0x18);
}

TEST(Fp6E2M3, SrDeterministic) {
  uint8_t r = util::f32_to_fp6_e2m3_sr(1.0f, 0);
  EXPECT_EQ(r, 0x08);
}

TEST(Fp6E2M3, SrDenormRoundTrip) {
  for (uint8_t code = 1; code < 8; ++code) {
    float val = util::fp6_e2m3_to_f32(code);
    uint8_t narrow = util::f32_to_fp6_e2m3_sr(val, 0);
    EXPECT_EQ(narrow, code) << "code=" << int(code) << " val=" << val;
  }
}

// ---- BF6 E3M2 (exhaustive — 64 values) ----

static float bf6_reference(uint8_t code) {
  uint32_t sign = (code >> 5) & 1;
  uint32_t exp = (code >> 2) & 0x7;
  uint32_t mant = code & 0x3;
  if (exp == 0 && mant == 0)
    return sign ? -0.0f : 0.0f;
  if (exp == 0) {
    float v = std::ldexp(static_cast<float>(mant), -4);
    return sign ? -v : v;
  }
  uint32_t f = (sign << 31) | ((exp + 127 - 3) << 23) | (mant << 21);
  return std::bit_cast<float>(f);
}

TEST(Bf6E3M2, ExhaustiveWiden) {
  for (uint8_t code = 0; code < 64; ++code) {
    float got = util::bf6_e3m2_to_f32(code);
    float ref = bf6_reference(code);
    EXPECT_EQ(std::bit_cast<uint32_t>(got), std::bit_cast<uint32_t>(ref)) << "code=" << int(code);
  }
}

TEST(Bf6E3M2, RneRoundTrip) {
  for (uint8_t code = 0; code < 32; ++code) {
    float val = util::bf6_e3m2_to_f32(code);
    uint8_t narrow = util::f32_to_bf6_e3m2_rne(val);
    EXPECT_EQ(narrow, code) << "code=" << int(code) << " val=" << val;
  }
}

TEST(Bf6E3M2, RneEdgeCases) {
  EXPECT_EQ(util::f32_to_bf6_e3m2_rne(std::numeric_limits<float>::quiet_NaN()), 0);
  EXPECT_EQ(util::f32_to_bf6_e3m2_rne(std::numeric_limits<float>::infinity()), 0x1F);
  EXPECT_EQ(util::f32_to_bf6_e3m2_rne(50.0f), 0x1F);
  // 16.0 maps to exp=7, mant=0 (code 0x1C)
  EXPECT_EQ(util::f32_to_bf6_e3m2_rne(16.0f), 0x1C);
}

TEST(Bf6E3M2, SrDeterministic) {
  uint8_t r = util::f32_to_bf6_e3m2_sr(1.0f, 0);
  EXPECT_EQ(r, 0x0C);
}

TEST(Bf6E3M2, SrDenormRoundTrip) {
  for (uint8_t code = 1; code < 4; ++code) {
    float val = util::bf6_e3m2_to_f32(code);
    uint8_t narrow = util::f32_to_bf6_e3m2_sr(val, 0);
    EXPECT_EQ(narrow, code) << "code=" << int(code) << " val=" << val;
  }
}

// ---- FP8 E4M3 (OCP E4M3FN) ----

TEST(Fp8E4M3, KeyValues) {
  EXPECT_EQ(util::fp8_e4m3_to_f32(0x00), 0.0f);
  EXPECT_EQ(util::fp8_e4m3_ocp_to_f32(0x38), util::fp8_e4m3_to_f32(0x38));
  EXPECT_EQ(util::fp8_e4m3_to_f32(0x38), 1.0f);
  EXPECT_EQ(util::fp8_e4m3_to_f32(0x7E), 448.0f);
  EXPECT_TRUE(std::isnan(util::fp8_e4m3_to_f32(0x7F)));
  EXPECT_TRUE(std::isnan(util::fp8_e4m3_to_f32(0xFF)));
  // exp=15, mant=0..6 are valid normals (256..448)
  EXPECT_FALSE(std::isnan(util::fp8_e4m3_to_f32(0x78)));
  EXPECT_EQ(util::fp8_e4m3_to_f32(0x78), 256.0f);
  EXPECT_EQ(util::fp8_e4m3_to_f32(0x79), 288.0f);
}

TEST(Fp8E4M3Fnuz, KeyValues) {
  EXPECT_EQ(util::fp8_e4m3_fnuz_to_f32(0x00), 0.0f);
  EXPECT_EQ(util::fp8_e4m3_fnuz_to_f32(0x40), 1.0f);
  EXPECT_EQ(util::fp8_e4m3_fnuz_to_f32(0xC0), -1.0f);
  EXPECT_EQ(util::fp8_e4m3_fnuz_to_f32(0x01), std::ldexp(1.0f, -10));
  EXPECT_EQ(util::fp8_e4m3_fnuz_to_f32(0x7F), 240.0f);
  EXPECT_TRUE(std::isnan(util::fp8_e4m3_fnuz_to_f32(0x80)));
}

TEST(Fp8E4M3Fnuz, BlockMatchesScalar) {
  const uint8_t src[] = {0x00, 0x01, 0x40, 0x7F, 0x80, 0xC0};
  float dst[std::size(src)] = {};
  util::fp8_e4m3_fnuz_to_f32_block(src, dst, std::size(src));
  for (size_t i = 0; i < std::size(src); ++i) {
    float scalar = util::fp8_e4m3_fnuz_to_f32(src[i]);
    if (std::isnan(scalar))
      EXPECT_TRUE(std::isnan(dst[i])) << "i=" << i;
    else
      EXPECT_EQ(dst[i], scalar) << "i=" << i;
  }
}

TEST(Fp8E4M3Fnuz, RneNarrow) {
  EXPECT_EQ(util::f32_to_fp8_e4m3_fnuz_rne(std::numeric_limits<float>::quiet_NaN()), 0x80);
  EXPECT_EQ(util::f32_to_fp8_e4m3_fnuz_rne(std::numeric_limits<float>::infinity()), 0x7F);
  EXPECT_EQ(util::f32_to_fp8_e4m3_fnuz_rne(-std::numeric_limits<float>::infinity()), 0xFF);
  EXPECT_EQ(util::f32_to_fp8_e4m3_fnuz_rne(0.0f), 0x00);
  EXPECT_EQ(util::f32_to_fp8_e4m3_fnuz_rne(-0.0f), 0x00);
  EXPECT_EQ(util::f32_to_fp8_e4m3_fnuz_rne(1.0f), 0x40);
  EXPECT_EQ(util::f32_to_fp8_e4m3_fnuz_rne(240.0f), 0x7F);
  EXPECT_EQ(util::f32_to_fp8_e4m3_fnuz_rne(300.0f), 0x7F);
  EXPECT_EQ(util::f32_to_fp8_e4m3_fnuz_rne(-300.0f), 0xFF);
  EXPECT_EQ(util::f32_to_fp8_e4m3_fnuz_rne(1e-12f), 0x00);

  float largest_denorm = util::fp8_e4m3_fnuz_to_f32(0x07);
  float smallest_normal = util::fp8_e4m3_fnuz_to_f32(0x08);
  float midpoint = (largest_denorm + smallest_normal) / 2.0f;
  EXPECT_EQ(util::f32_to_fp8_e4m3_fnuz_rne(midpoint + 1e-9f), 0x08);
}

TEST(Fp8E4M3Fnuz, SrNarrow) {
  EXPECT_EQ(util::f32_to_fp8_e4m3_fnuz_sr(1.0f, 0), 0x40);
  EXPECT_EQ(util::f32_to_fp8_e4m3_fnuz_sr(std::numeric_limits<float>::quiet_NaN(), 0), 0x80);
  EXPECT_EQ(util::f32_to_fp8_e4m3_fnuz_sr(std::numeric_limits<float>::infinity(), 0), 0x7F);

  const float normal_lo = util::fp8_e4m3_fnuz_to_f32(0x40);
  const float normal_hi = util::fp8_e4m3_fnuz_to_f32(0x41);
  const float normal_quarter = normal_lo + 0.25f * (normal_hi - normal_lo);
  EXPECT_EQ(util::f32_to_fp8_e4m3_fnuz_sr(normal_quarter, 0), 0x40);
  EXPECT_EQ(util::f32_to_fp8_e4m3_fnuz_sr(normal_quarter, 0xFFFFFFFFu), 0x41);

  const float subnormal_lo = util::fp8_e4m3_fnuz_to_f32(0x01);
  const float subnormal_hi = util::fp8_e4m3_fnuz_to_f32(0x02);
  const float subnormal_quarter = subnormal_lo + 0.25f * (subnormal_hi - subnormal_lo);
  EXPECT_EQ(util::f32_to_fp8_e4m3_fnuz_sr(subnormal_quarter, 0), 0x01);
  EXPECT_EQ(util::f32_to_fp8_e4m3_fnuz_sr(subnormal_quarter, 0xFFFFFFFFu), 0x02);

  for (uint8_t code = 1; code < 8; ++code) {
    float val = util::fp8_e4m3_fnuz_to_f32(code);
    EXPECT_EQ(util::f32_to_fp8_e4m3_fnuz_sr(val, 0), code) << "code=" << int(code);
  }
}

TEST(Fp8E4M3, RneNarrow) {
  EXPECT_EQ(util::f32_to_fp8_e4m3_rne(std::numeric_limits<float>::quiet_NaN()), 0x7F);
  EXPECT_EQ(util::f32_to_fp8_e4m3_rne(std::numeric_limits<float>::infinity()), 0x7E);
  EXPECT_EQ(util::f32_to_fp8_e4m3_rne(0.0f), 0x00);
  EXPECT_EQ(util::f32_to_fp8_e4m3_rne(1.0f), 0x38);
  EXPECT_EQ(util::f32_to_fp8_e4m3_rne(448.0f), 0x7E);
  EXPECT_EQ(util::f32_to_fp8_e4m3_rne(500.0f), 0x7E);
  // Denorm overflow: largest denorm (code 7 = 7*2^-9) rounds up to
  // smallest normal (code 8 = 2^-6) for values just above midpoint.
  float largest_denorm = util::fp8_e4m3_to_f32(0x07);
  float smallest_normal = util::fp8_e4m3_to_f32(0x08);
  float midpoint = (largest_denorm + smallest_normal) / 2.0f;
  EXPECT_EQ(util::f32_to_fp8_e4m3_rne(midpoint + 1e-9f), 0x08);
}

TEST(Fp8E4M3, RneExp15) {
  // exp=15, mant=0..6 are valid normals: 256, 288, 320, 352, 384, 416, 448
  static constexpr float kExp15Values[] = {256.0f, 288.0f, 320.0f, 352.0f, 384.0f, 416.0f, 448.0f};
  for (int m = 0; m < 7; ++m) {
    uint8_t code = 0x78 | m;
    EXPECT_EQ(util::fp8_e4m3_to_f32(code), kExp15Values[m]) << "decode m=" << m;
    EXPECT_EQ(util::f32_to_fp8_e4m3_rne(kExp15Values[m]), code) << "encode m=" << m;
  }
  // 272.0 is midpoint between 256 (m=0) and 288 (m=1) — RNE to even (m=0)
  EXPECT_EQ(util::f32_to_fp8_e4m3_rne(272.0f), 0x78);
  // 280.0 > midpoint(272), rounds up to 288
  EXPECT_EQ(util::f32_to_fp8_e4m3_rne(280.0f), 0x79);
  // 304.0 is midpoint between 288 (m=1) and 320 (m=2) — RNE to even (m=2)
  EXPECT_EQ(util::f32_to_fp8_e4m3_rne(304.0f), 0x7A);
  // Values above 448 saturate to 0x7E (max finite, not NaN 0x7F)
  EXPECT_EQ(util::f32_to_fp8_e4m3_rne(449.0f), 0x7E);
  EXPECT_EQ(util::f32_to_fp8_e4m3_rne(1000.0f), 0x7E);
}

TEST(Fp8E4M3, SrExp15) {
  // Exact exp=15 values should round-trip with seed=0
  static constexpr float kExp15Values[] = {256.0f, 288.0f, 320.0f, 352.0f, 384.0f, 416.0f, 448.0f};
  for (int m = 0; m < 7; ++m) {
    uint8_t code = 0x78 | m;
    EXPECT_EQ(util::f32_to_fp8_e4m3_sr(kExp15Values[m], 0), code) << "sr m=" << m;
  }
  // Values above max saturate to 0x7E
  EXPECT_EQ(util::f32_to_fp8_e4m3_sr(500.0f, 0), 0x7E);
  EXPECT_EQ(util::f32_to_fp8_e4m3_sr(500.0f, 0xFFFFFFFF), 0x7E);
}

TEST(Fp8E4M3, SrNarrow) {
  uint8_t r = util::f32_to_fp8_e4m3_sr(1.0f, 0);
  EXPECT_EQ(r, 0x38);
  EXPECT_EQ(util::f32_to_fp8_e4m3_sr(std::numeric_limits<float>::quiet_NaN(), 0), 0x7F);
  EXPECT_EQ(util::f32_to_fp8_e4m3_sr(std::numeric_limits<float>::infinity(), 0), 0x7E);
}

TEST(Fp8E4M3, SrDenormRoundTrip) {
  for (uint8_t code = 1; code < 8; ++code) {
    float val = util::fp8_e4m3_to_f32(code);
    uint8_t narrow = util::f32_to_fp8_e4m3_sr(val, 0);
    EXPECT_EQ(narrow, code) << "code=" << int(code) << " val=" << val;
  }
}

// ---- OCP-MX E8M0 unsigned exponent scale ----

TEST(E8M0, KeyValues) {
  EXPECT_EQ(util::e8m0_to_f32(0x00), std::bit_cast<float>(0x00400000u));
  EXPECT_EQ(util::e8m0_to_f32(0x7e), 0.5f);
  EXPECT_EQ(util::e8m0_to_f32(0x7f), 1.0f);
  EXPECT_EQ(util::e8m0_to_f32(0x80), 2.0f);
  EXPECT_EQ(util::e8m0_to_f32(0xfe), std::ldexp(1.0f, 127));
  EXPECT_TRUE(std::isnan(util::e8m0_to_f32(0xff)));
}

// ---- FP8 E5M3 (gfx1250 unsigned scale format) ----

TEST(Fp8E5M3, KeyValues) {
  EXPECT_EQ(util::fp8_e5m3_to_f32(0x00), 0.0f);
  EXPECT_EQ(util::fp8_e5m3_to_f32(0x01), std::ldexp(1.0f, -17));
  EXPECT_EQ(util::fp8_e5m3_to_f32(0x07), 0.875f * std::ldexp(1.0f, -14));
  EXPECT_EQ(util::fp8_e5m3_to_f32(0x08), std::ldexp(1.0f, -14));
  EXPECT_EQ(util::fp8_e5m3_to_f32(0x38), std::ldexp(1.0f, -8));
  EXPECT_EQ(util::fp8_e5m3_to_f32(0xFE), 114688.0f);
  EXPECT_TRUE(std::isnan(util::fp8_e5m3_to_f32(0xFF)));
  EXPECT_TRUE(std::isfinite(util::fp8_e5m3_to_f32(0xFE)));
}

TEST(Fp8E5M3, RneNarrow) {
  EXPECT_EQ(util::f32_to_fp8_e5m3_rne(std::numeric_limits<float>::quiet_NaN()), 0xFF);
  EXPECT_EQ(util::f32_to_fp8_e5m3_rne(std::numeric_limits<float>::infinity()), 0xFF);
  EXPECT_EQ(util::f32_to_fp8_e5m3_rne(0.0f), 0x00);
  const float smallest_subnormal = std::ldexp(1.0f, -17);
  EXPECT_EQ(util::f32_to_fp8_e5m3_rne(std::ldexp(1.0f, -18)), 0x00);
  EXPECT_EQ(util::f32_to_fp8_e5m3_rne(0.75f * smallest_subnormal), 0x01);
  EXPECT_EQ(util::f32_to_fp8_e5m3_rne(std::nextafter(std::ldexp(1.0f, -18), smallest_subnormal)),
            0x01);
  EXPECT_EQ(util::f32_to_fp8_e5m3_rne(std::ldexp(1.0f, -17)), 0x01);
  EXPECT_EQ(util::f32_to_fp8_e5m3_rne(std::ldexp(1.0f, -14)), 0x08);
  EXPECT_EQ(util::f32_to_fp8_e5m3_rne(std::ldexp(1.0f, -8)), 0x38);
  EXPECT_EQ(util::f32_to_fp8_e5m3_rne(-std::ldexp(1.0f, -8)), 0x38);
  EXPECT_EQ(util::f32_to_fp8_e5m3_rne(114688.0f), 0xFE);
}

TEST(Fp8E5M3, OverflowMapsToNaNCode) {
  const float exponent_overflow = std::ldexp(1.0f, 17);
  EXPECT_EQ(util::f32_to_fp8_e5m3_rne(exponent_overflow), 0xFF);
  EXPECT_EQ(util::f32_to_fp8_e5m3_sr(exponent_overflow, 0), 0xFF);
  EXPECT_EQ(util::f32_to_fp8_e5m3_sr(exponent_overflow, 0xFFFFFFFFu), 0xFF);
  EXPECT_TRUE(std::isnan(util::fp8_e5m3_to_f32(util::f32_to_fp8_e5m3_rne(exponent_overflow))));

  const float near_overflow = 120000.0f;
  EXPECT_EQ(util::f32_to_fp8_e5m3_rne(near_overflow), 0xFF);
  EXPECT_EQ(util::f32_to_fp8_e5m3_sr(near_overflow, 0), 0xFE);
  EXPECT_EQ(util::f32_to_fp8_e5m3_sr(near_overflow, 0xFFFFFFFFu), 0xFF);
}

TEST(Fp8E5M3, SrExactRoundTrip) {
  static constexpr uint8_t kCodes[] = {0x01, 0x07, 0x08, 0x38, 0x80, 0xC0, 0xFE};
  for (uint8_t code : kCodes) {
    float val = util::fp8_e5m3_to_f32(code);
    EXPECT_EQ(util::f32_to_fp8_e5m3_sr(val, 0x12345678u), code) << "code=" << int(code);
  }
}

TEST(Fp8E5M3, SrUsesSeedForNormalAndSubnormalRounding) {
  const float normal_lo = util::fp8_e5m3_to_f32(0x38);
  const float normal_hi = util::fp8_e5m3_to_f32(0x39);
  const float normal_quarter = normal_lo + 0.25f * (normal_hi - normal_lo);
  EXPECT_EQ(util::f32_to_fp8_e5m3_sr(normal_quarter, 0), 0x38);
  EXPECT_EQ(util::f32_to_fp8_e5m3_sr(normal_quarter, 0xFFFFFFFFu), 0x39);

  const float subnormal_lo = util::fp8_e5m3_to_f32(0x02);
  const float subnormal_hi = util::fp8_e5m3_to_f32(0x03);
  const float subnormal_quarter = subnormal_lo + 0.25f * (subnormal_hi - subnormal_lo);
  EXPECT_EQ(util::f32_to_fp8_e5m3_sr(subnormal_quarter, 0), 0x02);
  EXPECT_EQ(util::f32_to_fp8_e5m3_sr(subnormal_quarter, 0xFFFFFFFFu), 0x03);
}

// ---- BF8 E5M2 ----

TEST(Bf8E5M2, KeyValues) {
  EXPECT_EQ(util::bf8_e5m2_to_f32(0x00), 0.0f);
  EXPECT_EQ(util::bf8_e5m2_ocp_to_f32(0x3C), util::bf8_e5m2_to_f32(0x3C));
  EXPECT_EQ(util::bf8_e5m2_to_f32(0x3C), 1.0f);
  EXPECT_TRUE(std::isinf(util::bf8_e5m2_to_f32(0x7C)));
  EXPECT_TRUE(std::isnan(util::bf8_e5m2_to_f32(0x7F)));
  EXPECT_EQ(util::bf8_e5m2_to_f32(0x7B), 57344.0f);
}

TEST(Bf8E5M2Fnuz, KeyValues) {
  EXPECT_EQ(util::bf8_e5m2_fnuz_to_f32(0x00), 0.0f);
  EXPECT_EQ(util::bf8_e5m2_fnuz_to_f32(0x40), 1.0f);
  EXPECT_EQ(util::bf8_e5m2_fnuz_to_f32(0xC0), -1.0f);
  EXPECT_EQ(util::bf8_e5m2_fnuz_to_f32(0x01), std::ldexp(1.0f, -17));
  EXPECT_EQ(util::bf8_e5m2_fnuz_to_f32(0x7F), 57344.0f);
  EXPECT_TRUE(std::isnan(util::bf8_e5m2_fnuz_to_f32(0x80)));
}

TEST(Bf8E5M2Fnuz, BlockMatchesScalar) {
  const uint8_t src[] = {0x00, 0x01, 0x40, 0x7F, 0x80, 0xC0};
  float dst[std::size(src)] = {};
  util::bf8_e5m2_fnuz_to_f32_block(src, dst, std::size(src));
  for (size_t i = 0; i < std::size(src); ++i) {
    float scalar = util::bf8_e5m2_fnuz_to_f32(src[i]);
    if (std::isnan(scalar))
      EXPECT_TRUE(std::isnan(dst[i])) << "i=" << i;
    else
      EXPECT_EQ(dst[i], scalar) << "i=" << i;
  }
}

TEST(Bf8E5M2Fnuz, RneNarrow) {
  EXPECT_EQ(util::f32_to_bf8_e5m2_fnuz_rne(std::numeric_limits<float>::quiet_NaN()), 0x80);
  EXPECT_EQ(util::f32_to_bf8_e5m2_fnuz_rne(std::numeric_limits<float>::infinity()), 0x7F);
  EXPECT_EQ(util::f32_to_bf8_e5m2_fnuz_rne(-std::numeric_limits<float>::infinity()), 0xFF);
  EXPECT_EQ(util::f32_to_bf8_e5m2_fnuz_rne(0.0f), 0x00);
  EXPECT_EQ(util::f32_to_bf8_e5m2_fnuz_rne(-0.0f), 0x00);
  EXPECT_EQ(util::f32_to_bf8_e5m2_fnuz_rne(1.0f), 0x40);
  EXPECT_EQ(util::f32_to_bf8_e5m2_fnuz_rne(57344.0f), 0x7F);
  EXPECT_EQ(util::f32_to_bf8_e5m2_fnuz_rne(70000.0f), 0x7F);
  EXPECT_EQ(util::f32_to_bf8_e5m2_fnuz_rne(-70000.0f), 0xFF);
  EXPECT_EQ(util::f32_to_bf8_e5m2_fnuz_rne(1e-12f), 0x00);
}

TEST(Bf8E5M2Fnuz, SrNarrow) {
  EXPECT_EQ(util::f32_to_bf8_e5m2_fnuz_sr(1.0f, 0), 0x40);
  EXPECT_EQ(util::f32_to_bf8_e5m2_fnuz_sr(std::numeric_limits<float>::quiet_NaN(), 0), 0x80);
  EXPECT_EQ(util::f32_to_bf8_e5m2_fnuz_sr(std::numeric_limits<float>::infinity(), 0), 0x7F);

  const float normal_lo = util::bf8_e5m2_fnuz_to_f32(0x40);
  const float normal_hi = util::bf8_e5m2_fnuz_to_f32(0x41);
  const float normal_quarter = normal_lo + 0.25f * (normal_hi - normal_lo);
  EXPECT_EQ(util::f32_to_bf8_e5m2_fnuz_sr(normal_quarter, 0), 0x40);
  EXPECT_EQ(util::f32_to_bf8_e5m2_fnuz_sr(normal_quarter, 0xFFFFFFFFu), 0x41);

  const float subnormal_lo = util::bf8_e5m2_fnuz_to_f32(0x01);
  const float subnormal_hi = util::bf8_e5m2_fnuz_to_f32(0x02);
  const float subnormal_quarter = subnormal_lo + 0.25f * (subnormal_hi - subnormal_lo);
  EXPECT_EQ(util::f32_to_bf8_e5m2_fnuz_sr(subnormal_quarter, 0), 0x01);
  EXPECT_EQ(util::f32_to_bf8_e5m2_fnuz_sr(subnormal_quarter, 0xFFFFFFFFu), 0x02);

  for (uint8_t code = 1; code < 4; ++code) {
    float val = util::bf8_e5m2_fnuz_to_f32(code);
    EXPECT_EQ(util::f32_to_bf8_e5m2_fnuz_sr(val, 0), code) << "code=" << int(code);
  }
}

TEST(Bf8E5M2, RneNarrow) {
  EXPECT_EQ(util::f32_to_bf8_e5m2_rne(std::numeric_limits<float>::quiet_NaN()), 0x7F);
  EXPECT_EQ(util::f32_to_bf8_e5m2_rne(std::numeric_limits<float>::infinity()), 0x7C);
  EXPECT_EQ(util::f32_to_bf8_e5m2_rne(0.0f), 0x00);
  EXPECT_EQ(util::f32_to_bf8_e5m2_rne(1.0f), 0x3C);
}

TEST(Bf8E5M2, SrNarrow) {
  EXPECT_EQ(util::f32_to_bf8_e5m2_sr(1.0f, 0), 0x3C);
  EXPECT_EQ(util::f32_to_bf8_e5m2_sr(std::numeric_limits<float>::quiet_NaN(), 0), 0x7F);
  EXPECT_EQ(util::f32_to_bf8_e5m2_sr(std::numeric_limits<float>::infinity(), 0), 0x7C);
}

TEST(Bf8E5M2, SrDenormRoundTrip) {
  for (uint8_t code = 1; code < 4; ++code) {
    float val = util::bf8_e5m2_to_f32(code);
    uint8_t narrow = util::f32_to_bf8_e5m2_sr(val, 0);
    EXPECT_EQ(narrow, code) << "code=" << int(code) << " val=" << val;
  }
}

// ---- FP16 ----

TEST(Fp16, RoundTrip) {
  auto rt = [](float v) { return util::f16_to_f32(util::f32_to_f16(v)); };
  EXPECT_EQ(rt(0.0f), 0.0f);
  EXPECT_EQ(rt(1.0f), 1.0f);
  EXPECT_EQ(rt(65504.0f), 65504.0f);
  EXPECT_TRUE(std::isinf(rt(std::numeric_limits<float>::infinity())));
  EXPECT_TRUE(std::isnan(rt(std::numeric_limits<float>::quiet_NaN())));
  EXPECT_TRUE(std::signbit(rt(-0.0f)));
}

// ---- BF16 ----

TEST(Bf16, RoundTrip) {
  auto rt = [](float v) { return util::bf16_to_f32(util::f32_to_bf16(v)); };
  EXPECT_EQ(rt(0.0f), 0.0f);
  EXPECT_EQ(rt(1.0f), 1.0f);
  EXPECT_TRUE(std::signbit(rt(-0.0f)));
}

// ---- 6-bit pack/unpack ----

TEST(Pack6Bit, RoundTrip) {
  uint8_t vals[32];
  for (int i = 0; i < 32; ++i)
    vals[i] = static_cast<uint8_t>(i * 2);
  uint32_t dwords[6];
  util::pack_6bit(vals, dwords);
  uint8_t unpacked[32];
  util::unpack_6bit(dwords, unpacked);
  for (int i = 0; i < 32; ++i)
    EXPECT_EQ(unpacked[i], vals[i] & 0x3F) << "i=" << i;
}

TEST(Pack6Bit, AllOnes) {
  uint8_t vals[32];
  for (int i = 0; i < 32; ++i)
    vals[i] = 0x3F;
  uint32_t dwords[6];
  util::pack_6bit(vals, dwords);
  uint8_t unpacked[32];
  util::unpack_6bit(dwords, unpacked);
  for (int i = 0; i < 32; ++i)
    EXPECT_EQ(unpacked[i], 0x3F) << "i=" << i;
}

// ---- PRNG LFSR ----

TEST(PrngAdvance, KnownSequence) {
  uint32_t s = 1;
  s = util::prng_advance(s);
  EXPECT_EQ(s, 2u);
  s = util::prng_advance(s);
  EXPECT_EQ(s, 4u);
}

TEST(PrngAdvance, ZeroFixedPoint) { EXPECT_EQ(util::prng_advance(0), 0u); }

TEST(PrngAdvance, HighBitSet) {
  uint32_t s = 0x80000000u;
  s = util::prng_advance(s);
  EXPECT_EQ(s, 197u);
}

} // namespace
