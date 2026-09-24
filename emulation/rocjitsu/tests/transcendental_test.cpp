// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file transcendental_test.cpp
/// @brief Phase C unit tests for shared transcendental functions.

#include "rocjitsu/isa/arch/amdgpu/shared/transcendental.h"

#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>

namespace {

using namespace rocjitsu::amdgpu::transcendental;

// ---------------------------------------------------------------------------
// Special-case tests (±0, ±Inf, NaN, denormals)
// ---------------------------------------------------------------------------

TEST(TranscendentalTest, RcpF32SpecialCases) {
  EXPECT_EQ(rcp_f32(0.0f), std::numeric_limits<float>::infinity());
  EXPECT_EQ(rcp_f32(-0.0f), -std::numeric_limits<float>::infinity());
  EXPECT_EQ(rcp_f32(std::numeric_limits<float>::infinity()), 0.0f);
  EXPECT_EQ(rcp_f32(-std::numeric_limits<float>::infinity()), -0.0f);
  EXPECT_TRUE(std::isnan(rcp_f32(std::numeric_limits<float>::quiet_NaN())));
  EXPECT_FLOAT_EQ(rcp_f32(2.0f), 0.5f);
}

TEST(TranscendentalTest, RcpF32MatchesPhysicalRdna3AndRdna4) {
  const uint32_t cases[][2] = {
      {0x3f800000u, 0x3f800000u}, {0x3f800001u, 0x3f7ffffeu}, {0x3f81ffffu, 0x3f7c0fc3u},
      {0x3f820000u, 0x3f7c0fc1u}, {0x3f83ffffu, 0x3f783e12u}, {0x3f840000u, 0x3f783e10u},
      {0x3fc00000u, 0x3f2aaaaau}, {0x3fffffffu, 0x3f000001u}, {0x3f802922u, 0x3f7fadd6u},
      {0x3f932eddu, 0x3f5ea262u}, {0x3fb0333cu, 0x3f39f868u}, {0x3ffff486u, 0x3f0005beu},
      {0x00800000u, 0x7e800000u}, {0x7f000000u, 0x00000000u}, {0x00000001u, 0x7f800000u},
      {0x7fa12345u, 0x7fe12345u},
  };
  for (const auto &test : cases)
    for (uint32_t sign : {0u, 0x80000000u})
      EXPECT_EQ(std::bit_cast<uint32_t>(rcp_f32(std::bit_cast<float>(test[0] | sign))),
                test[1] | sign)
          << std::hex << test[0] << " sign=" << sign;
}

TEST(TranscendentalTest, RcpF32CompleteNormalizedHardwareDigest) {
  // FNV-style hash of raw result words captured independently on gfx1100 and gfx1201.
  // Check the complete mantissa domain without storing the 32 MiB capture.
  uint64_t digest = 14695981039346656037ull;
  for (uint32_t mantissa = 0; mantissa < (1u << 23); ++mantissa) {
    const float input = std::bit_cast<float>(0x3f800000u | mantissa);
    digest = (digest ^ std::bit_cast<uint32_t>(rcp_f32(input))) * 1099511628211ull;
  }
  EXPECT_EQ(digest, 0xd54ec24992572df9ull);
}

TEST(TranscendentalTest, RsqF32SpecialCases) {
  EXPECT_EQ(rsq_f32(0.0f), std::numeric_limits<float>::infinity());
  EXPECT_TRUE(std::isnan(rsq_f32(-1.0f)));
  EXPECT_EQ(rsq_f32(std::numeric_limits<float>::infinity()), 0.0f);
  EXPECT_TRUE(std::isnan(rsq_f32(std::numeric_limits<float>::quiet_NaN())));
  EXPECT_FLOAT_EQ(rsq_f32(4.0f), 0.5f);
}

TEST(TranscendentalTest, RsqF32MatchesPhysicalRdna3AndRdna4) {
  const uint32_t cases[][2] = {
      {0x3f800000u, 0x3f800000u}, {0x3f800001u, 0x3f7fffffu}, {0x3f83ffffu, 0x3f7c1765u},
      {0x3f840000u, 0x3f7c1764u}, {0x3f8040c4u, 0x3f7fbf55u}, {0x3f80cab3u, 0x3f7f363du},
      {0x3fbfffffu, 0x3f5105ecu}, {0x3fc00000u, 0x3f5105ecu}, {0x3fffffffu, 0x3f3504f3u},
      {0x40000000u, 0x3f3504f3u}, {0x40000001u, 0x3f3504f2u}, {0x4003ffffu, 0x3f32416au},
      {0x40040000u, 0x3f32416au}, {0x403fffffu, 0x3f13cd3bu}, {0x40400000u, 0x3f13cd3au},
      {0x407fffffu, 0x3f000000u}, {0x00000000u, 0x7f800000u}, {0x80000000u, 0xff800000u},
      {0x00000001u, 0x7f800000u}, {0x80000001u, 0xff800000u}, {0xbf800000u, 0xffc00000u},
      {0xff800000u, 0xffc00000u}, {0x7f800000u, 0x00000000u}, {0x7fa12345u, 0x7fe12345u},
      {0xffa12345u, 0xffe12345u},
  };
  for (const auto &test : cases)
    EXPECT_EQ(std::bit_cast<uint32_t>(rsq_f32(std::bit_cast<float>(test[0]))), test[1])
        << std::hex << test[0];
}

TEST(TranscendentalTest, RsqF32CompleteNormalizedHardwareDigest) {
  // FNV-style hash of raw result words captured independently on gfx1100 and gfx1201.
  // Cover every mantissa in both exponent parities without storing the 64 MiB capture.
  uint64_t digest = 14695981039346656037ull;
  for (uint32_t index = 0; index < (1u << 24); ++index) {
    const float input = std::bit_cast<float>(0x3f800000u + index);
    digest = (digest ^ std::bit_cast<uint32_t>(rsq_f32(input))) * 1099511628211ull;
  }
  EXPECT_EQ(digest, 0x010bc79eb6e48cafull);
}

TEST(TranscendentalTest, SqrtF32SpecialCases) {
  EXPECT_TRUE(std::isnan(sqrt_f32(-1.0f)));
  EXPECT_EQ(sqrt_f32(0.0f), 0.0f);
  EXPECT_EQ(sqrt_f32(std::numeric_limits<float>::infinity()),
            std::numeric_limits<float>::infinity());
  EXPECT_FLOAT_EQ(sqrt_f32(4.0f), 2.0f);
}

TEST(TranscendentalTest, LogF32SpecialCases) {
  EXPECT_EQ(log_f32(0.0f), -std::numeric_limits<float>::infinity());
  EXPECT_TRUE(std::isnan(log_f32(-1.0f)));
  EXPECT_EQ(log_f32(std::numeric_limits<float>::infinity()),
            std::numeric_limits<float>::infinity());
  EXPECT_FLOAT_EQ(log_f32(1.0f), 0.0f);
  EXPECT_FLOAT_EQ(log_f32(4.0f), 2.0f);
}

TEST(TranscendentalTest, ExpF32SpecialCases) {
  EXPECT_EQ(exp_f32(-std::numeric_limits<float>::infinity()), 0.0f);
  EXPECT_EQ(exp_f32(std::numeric_limits<float>::infinity()),
            std::numeric_limits<float>::infinity());
  EXPECT_FLOAT_EQ(exp_f32(0.0f), 1.0f);
  EXPECT_FLOAT_EQ(exp_f32(1.0f), 2.0f);
}

TEST(TranscendentalTest, SinCosF32SpecialCases) {
  // sin(2*pi*0) = 0, cos(2*pi*0) = 1
  EXPECT_NEAR(sin_f32(0.0f), 0.0f, 1e-6f);
  EXPECT_NEAR(cos_f32(0.0f), 1.0f, 1e-6f);
  // sin(2*pi*0.25) = 1, cos(2*pi*0.25) = 0
  EXPECT_NEAR(sin_f32(0.25f), 1.0f, 1e-6f);
  EXPECT_NEAR(cos_f32(0.25f), 0.0f, 1e-6f);
  // NaN/Inf inputs
  EXPECT_TRUE(std::isnan(sin_f32(std::numeric_limits<float>::infinity())));
  EXPECT_TRUE(std::isnan(cos_f32(std::numeric_limits<float>::infinity())));
}

TEST(TranscendentalTest, SinCosF32CapturedRangeBoundaries) {
  // Raw outputs captured independently on gfx1100 and gfx1201 with denormals
  // preserved. Include small-input and polynomial boundaries, reflected
  // intervals, exact quadrants, and finite inputs too large for integer casts.
  const uint32_t cases[][3] = {
      {0x00000000u, 0x00000000u, 0x3f800000u}, {0x80000000u, 0x80000000u, 0x3f800000u},
      {0x00000001u, 0x00000006u, 0x3f800000u}, {0x80000001u, 0x80000006u, 0x3f800000u},
      {0x007fffffu, 0x01c90fd3u, 0x3f800000u}, {0x00800000u, 0x01c90fd5u, 0x3f800000u},
      {0x39bfffffu, 0x3b16cbdfu, 0x3f7fffd3u}, {0x39c00000u, 0x3b16cbdfu, 0x3f7fffd3u},
      {0x39ffffffu, 0x3b490fbeu, 0x3f7fffb1u}, {0x3a000000u, 0x3b490fcbu, 0x3f7fffb1u},
      {0x3e000000u, 0x3f3504f4u, 0x3f3504f3u}, {0x3e7fffffu, 0x3f800000u, 0x33c90fd9u},
      {0x3e800000u, 0x3f800000u, 0x00000000u}, {0x3f000000u, 0x00000000u, 0xbf800000u},
      {0x3f800000u, 0x00000000u, 0x3f800000u}, {0x7f7fffffu, 0x00000000u, 0x3f800000u},
      {0xff7fffffu, 0x00000000u, 0x3f800000u}, {0x7f800000u, 0xffc00000u, 0xffc00000u},
      {0xff800000u, 0xffc00000u, 0xffc00000u},
  };
  for (const auto &test : cases) {
    SCOPED_TRACE(test[0]);
    EXPECT_EQ(std::bit_cast<uint32_t>(sin_f32(std::bit_cast<float>(test[0]))), test[1]);
    EXPECT_EQ(std::bit_cast<uint32_t>(cos_f32(std::bit_cast<float>(test[0]))), test[2]);
  }
}

TEST(TranscendentalTest, SinCosF32CapturedApproximationBound) {
  // The staged polynomial still differs from some captured hardware results
  // by one ULP. Keep those controls explicit instead of treating it as an
  // exhaustive bit-exact mapping.
  const uint32_t cases[][3] = {
      {0x3aab9885u, 0x3c06c500u, 0x3f7ffdc8u},
      {0x3b8419b1u, 0x3ccf7b0du, 0x3f7feaf9u},
      {0x3bcf188bu, 0x3d229c2du, 0x3f7fcc55u},
      {0x3c98be2du, 0x3def6125u, 0x3f7e3ec8u},
  };
  for (const auto &test : cases)
    for (unsigned op = 0; op < 2; ++op) {
      const float input = std::bit_cast<float>(test[0]);
      const uint32_t actual = std::bit_cast<uint32_t>(op ? cos_f32(input) : sin_f32(input));
      EXPECT_LE(std::abs(int64_t{actual} - test[op + 1]), 1) << std::hex << test[0];
    }
}

TEST(TranscendentalTest, RcpF64SpecialCases) {
  EXPECT_EQ(rcp_f64(0.0), std::numeric_limits<double>::infinity());
  EXPECT_EQ(rcp_f64(-0.0), -std::numeric_limits<double>::infinity());
  EXPECT_DOUBLE_EQ(rcp_f64(2.0), 0.5);
}

TEST(TranscendentalTest, SqrtF64SpecialCases) {
  EXPECT_TRUE(std::isnan(sqrt_f64(-1.0)));
  EXPECT_DOUBLE_EQ(sqrt_f64(4.0), 2.0);
}

// ---------------------------------------------------------------------------
// ULP accuracy tests (pseudorandom inputs)
// ---------------------------------------------------------------------------

TEST(TranscendentalTest, RcpF32Ulp) {
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(0.001f, 1000.0f);
  for (int i = 0; i < 10000; ++i) {
    float x = dist(rng);
    float result = rcp_f32(x);
    float expected = 1.0f / x;
    // Allow 1 ULP difference
    ASSERT_NEAR(result, expected, std::abs(expected) * 1.2e-7f)
        << "rcp_f32(" << x << ") = " << result << " expected " << expected;
  }
}

TEST(TranscendentalTest, SqrtF32Ulp) {
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(0.0f, 1e6f);
  for (int i = 0; i < 10000; ++i) {
    float x = dist(rng);
    float result = sqrt_f32(x);
    float expected = std::sqrt(x);
    ASSERT_NEAR(result, expected, std::abs(expected) * 1.2e-7f)
        << "sqrt_f32(" << x << ") = " << result << " expected " << expected;
  }
}

} // namespace
