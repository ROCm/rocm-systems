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

TEST(TranscendentalTest, LogF32MatchesPhysicalRdna3AndRdna4) {
  // Raw witnesses distinguish near-one row boundaries, the reflected inner
  // coordinate, intermediate product precision and exact-grid packing.
  const uint32_t captured[][2] = {
      {0x3f000000u, 0xbf800000u}, {0x3f7bffffu, 0xbcba1fa3u}, {0x3f7c0000u, 0xbcba1f74u},
      {0x3f7c0001u, 0xbcba1f46u}, {0x3f7dffecu, 0xbc396b23u}, {0x3f7dffffu, 0xbc39643au},
      {0x3f7e0000u, 0xbc3963ddu}, {0x3f7e0001u, 0xbc396380u}, {0x3f7ff5cbu, 0xb96ba0e4u},
      {0x3f7ffffdu, 0xb48a7faeu}, {0x3f7ffffeu, 0xb438aa3cu}, {0x3f7fffffu, 0xb3b8aa3cu},
      {0x3f800000u, 0x00000000u}, {0x3f800001u, 0x3438aa3bu}, {0x3f800005u, 0x3566d4c6u},
      {0x3f800006u, 0x358a7faau}, {0x3f800043u, 0x37415204u}, {0x3f800c30u, 0x3a0ca2fau},
      {0x3f80ffffu, 0x3c37f1ceu}, {0x3f810000u, 0x3c37f286u}, {0x3f810001u, 0x3c37f33du},
      {0x3f81ffffu, 0x3cb73c5au}, {0x3f820000u, 0x3cb73cb4u}, {0x3f820001u, 0x3cb73d0fu},
      {0x3f835d48u, 0x3d19508bu}, {0x3f83ffffu, 0x3d35d66fu}, {0x3f840000u, 0x3d35d69cu},
      {0x3f85ffffu, 0x3d8759afu}, {0x3f860000u, 0x3d8759c5u}, {0x3f860001u, 0x3d8759dbu},
      {0x3fffffffu, 0x3f7fffffu},
  };
  for (const auto &sample : captured)
    EXPECT_EQ(std::bit_cast<uint32_t>(log_f32(std::bit_cast<float>(sample[0]))), sample[1])
        << std::hex << sample[0];
}

TEST(TranscendentalTest, LogF32CompleteMantissaHardwareDigest) {
  // Independently captured on both cards over [0.5,2), including both
  // near-one paths and every ordinary polynomial interval.
  uint64_t digest = 14695981039346656037ull;
  for (uint32_t bits = 0x3f000000u; bits < 0x40000000u; ++bits)
    digest =
        (digest ^ std::bit_cast<uint32_t>(log_f32(std::bit_cast<float>(bits)))) * 1099511628211ull;
  EXPECT_EQ(digest, 0xc33a54efbae098beull);
}

TEST(TranscendentalTest, ExpF32SpecialCases) {
  EXPECT_EQ(exp_f32(-std::numeric_limits<float>::infinity()), 0.0f);
  EXPECT_EQ(exp_f32(std::numeric_limits<float>::infinity()),
            std::numeric_limits<float>::infinity());
  EXPECT_FLOAT_EQ(exp_f32(0.0f), 1.0f);
  EXPECT_FLOAT_EQ(exp_f32(1.0f), 2.0f);
}

TEST(TranscendentalTest, ExpF32MatchesPhysicalRdna3AndRdna4) {
  // Captures cover staged polynomial rounding, signed reduction, the small
  // input cutoff, and finite overflow/underflow boundaries.
  const uint32_t captured[][2] = {
      {0x337fffffu, 0x3f800000u}, {0x33800000u, 0x3f800000u}, {0x33800001u, 0x3f800000u},
      {0xb37fffffu, 0x3f800000u}, {0xb3800000u, 0x3f7fffffu}, {0xb3800001u, 0x3f7fffffu},
      {0x42ffffffu, 0x7f7fffa7u}, {0x43000000u, 0x7f800000u}, {0xc2fc0000u, 0x00800000u},
      {0xc2fc0001u, 0x00000000u}, {0x3f0567ecu, 0x3fb7b03du}, {0xc114ed44u, 0x3acecc1eu},
      {0x35500000u, 0x3f800004u}, {0x3e000090u, 0x3f8b95d0u}, {0x3e80001eu, 0x3f9837f6u},
      {0x3ec0001au, 0x3fa5fedcu}, {0x3f000013u, 0x3fb504fcu}, {0x3f20002au, 0x3fc56740u},
      {0x3f40006fu, 0x3fd7453eu}, {0x3f600022u, 0x3feac0dcu},
  };
  for (const auto &sample : captured)
    EXPECT_EQ(std::bit_cast<uint32_t>(exp_f32(std::bit_cast<float>(sample[0]))), sample[1])
        << std::hex << sample[0];
}

TEST(TranscendentalTest, ExpF32CompleteFractionHardwareDigests) {
  // Independent raw captures of 2^24 equally spaced inputs in each signed
  // unit interval, covering all 32 polynomial intervals and their boundaries.
  const uint64_t expected[] = {0x68959bb684f42e5full, 0x88b75684494db8d3ull};
  for (uint32_t sign = 0; sign < 2; ++sign) {
    uint64_t digest = 14695981039346656037ull;
    for (uint32_t index = 0; index < (1u << 24); ++index) {
      const float positive = static_cast<float>(index) * 0x1p-24f;
      const uint32_t bits = std::bit_cast<uint32_t>(positive) | (sign << 31);
      digest = (digest ^ std::bit_cast<uint32_t>(exp_f32(std::bit_cast<float>(bits)))) *
               1099511628211ull;
    }
    EXPECT_EQ(digest, expected[sign]) << "sign=" << sign;
  }
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

TEST(TranscendentalTest, SinCosF32CapturedArithmeticStages) {
  // Captured on gfx1100 and gfx1201. These inputs distinguish intermediate
  // coordinate/product rounding, normalized linear saturation and the
  // reflected cosine cutoff from a single wide polynomial evaluation.
  const uint32_t cases[][3] = {
      {0x3aab9885u, 0x3c06c500u, 0x3f7ffdc8u}, {0x3b8419b1u, 0x3ccf7b0du, 0x3f7feaf9u},
      {0x3bcf188bu, 0x3d229c2du, 0x3f7fcc55u}, {0x3c98be2du, 0x3def6125u, 0x3f7e3ec8u},
      {0x3d2001fbu, 0x3e78d2d0u, 0x3f7853c7u}, {0x3d6002f3u, 0x3eac7f01u, 0x3f7108a3u},
      {0x3dc084adu, 0x3f0e9072u, 0x3f54a13bu}, {0x3df0010du, 0x3f2bebe5u, 0x3f3dae6bu},
      {0x3e7f2800u, 0x3f7fff1fu, 0x3ba9a533u}, {0x3e7ff600u, 0x3f7fffffu, 0x397b53d7u},
      {0x3e7fff08u, 0x3f800000u, 0x37c2c761u}, {0x3e7ffff1u, 0x3f800000u, 0x35bc7ee2u},
      {0x3e7ffa50u, 0x3f800000u, 0x390ef149u}, {0x3efffd28u, 0x390ef149u, 0xbf800000u},
      {0x3f3ffe94u, 0xbf800000u, 0xb90ef149u}, {0x3f7ffe94u, 0xb90ef149u, 0x3f800000u},
  };
  for (const auto &test : cases)
    for (unsigned op = 0; op < 2; ++op) {
      const float input = std::bit_cast<float>(test[0]);
      const uint32_t actual = std::bit_cast<uint32_t>(op ? cos_f32(input) : sin_f32(input));
      EXPECT_EQ(actual, test[op + 1]) << std::hex << test[0];
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

TEST(TranscendentalTest, HalfLogExpCompleteHardwareDigests) {
  struct Captured {
    uint32_t denorm_mode;
    bool overflow;
    uint64_t digest[4];
  };
  // FNV hashes of raw F16 hardware captures over all 65536 input encodings.
  // Order: LOG signaling/quiet policy, then EXP signaling/quiet policy.
  const Captured captured[] = {
      {0u,
       false,
       {0x7ec126a2efb7cc29ull, 0x7b139780c1becc29ull, 0xc64da58f553026d8ull,
        0xa9b68cfe7b8372d8ull}},
      {0u,
       true,
       {0xf5160d9ac46aec29ull, 0x3999dfce791d6c29ull, 0x689733bef391b6d8ull,
        0xbdc5546b38d182d8ull}},
      {1u,
       false,
       {0x1a7dc0df37544143ull, 0xab561022e31ba143ull, 0xc64da58f553026d8ull,
        0xa9b68cfe7b8372d8ull}},
      {1u,
       true,
       {0x07029e1e69dff32bull, 0xab21f15c2191eb2bull, 0x689733bef391b6d8ull,
        0xbdc5546b38d182d8ull}},
      {2u,
       false,
       {0x7ec126a2efb7cc29ull, 0x7b139780c1becc29ull, 0x590d5c6485394589ull,
        0x7a49d8175530c589ull}},
      {2u,
       true,
       {0xf5160d9ac46aec29ull, 0x3999dfce791d6c29ull, 0x610235de125ed589ull,
        0x36d64dcb7d64d589ull}},
      {3u,
       false,
       {0x1a7dc0df37544143ull, 0xab561022e31ba143ull, 0x590d5c6485394589ull,
        0x7a49d8175530c589ull}},
      {3u,
       true,
       {0x07029e1e69dff32bull, 0xab21f15c2191eb2bull, 0x610235de125ed589ull,
        0x36d64dcb7d64d589ull}},
  };
  for (const auto &sample : captured) {
    for (bool logarithm : {true, false}) {
      for (bool quiet : {false, true}) {
        uint64_t digest = 14695981039346656037ull;
        for (uint32_t input = 0; input < 65536; ++input) {
          const float value = util::f16_to_f32(static_cast<uint16_t>(input));
          const float result =
              logarithm ? log_exp_f16<true>(value, sample.denorm_mode, sample.overflow, quiet)
                        : log_exp_f16<false>(value, sample.denorm_mode, sample.overflow, quiet);
          digest = (digest ^ util::f32_to_f16(result)) * 1099511628211ull;
        }
        EXPECT_EQ(digest, sample.digest[(logarithm ? 0 : 2) + quiet])
            << "log=" << logarithm << " quiet=" << quiet << " denorm=" << sample.denorm_mode
            << " overflow=" << sample.overflow;
      }
    }
  }
}

} // namespace
