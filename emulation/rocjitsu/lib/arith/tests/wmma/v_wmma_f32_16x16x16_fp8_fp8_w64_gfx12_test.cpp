// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for v_wmma_f32_16x16x16_fp8_fp8_w64_gfx12.
// Keeps this instruction's matrix inputs and expected results beside its aliasing and flag
// checks.

#include "common/cpu.h"
#include "common/floating_point.h"
#include "common/registers.h"
#include "common/semantics.h"
#include "rj_arith/rj_arith.h"
#include "wmma/helpers.h"
#include <limits>

#include <gtest/gtest.h>
#include <random>
#include <stdint.h>

namespace {

using Registers = rocjitsu::arith::test::Registers<64, 64>;

} // namespace

// Check matrix results, operand aliases, and instruction-flag contracts across supported CPUs
// and semantics.
TEST(WmmaF32FP8FP8W64Gfx12, Arithmetic) {
  std::minstd_rand random;
  uint32_t a[16][16], b[16][16], c[16][16];
  for (int row = 0; row < 16; ++row) {
    for (int k = 0; k < 16; ++k) {
      a[row][k] = rocjitsu::arith::test::random_fp8_bits(random);
    }
  }
  for (int k = 0; k < 16; ++k) {
    for (int col = 0; col < 16; ++col) {
      b[k][col] = rocjitsu::arith::test::random_fp8_bits(random);
    }
  }
  for (int row = 0; row < 16; ++row) {
    for (int col = 0; col < 16; ++col) {
      c[row][col] = rocjitsu::arith::test::random_f32_bits(random);
    }
  }
  // One NaN and one infinity per logical operand where the format has Inf.
  a[1][2] = rocjitsu::arith::test::kFP8QuietNaNBits;
  // FP8 E4M3FN has no infinity encoding.
  b[6][5] = rocjitsu::arith::test::kFP8QuietNaNBits;
  // FP8 E4M3FN has no infinity encoding.
  c[9][10] = rocjitsu::arith::test::kF32QuietNaNBits;
  c[11][12] = rocjitsu::arith::test::kF32PositiveInfinityBits;

  Registers input;
  for (int row = 0; row < 16; ++row) {
    for (int k = 0; k < 16; ++k) {
      rocjitsu::arith::test::set_wmma_gfx12_small<8, 64>(input.v, row, k, a[row][k]);
    }
  }
  for (int k = 0; k < 16; ++k) {
    for (int col = 0; col < 16; ++col) {
      rocjitsu::arith::test::set_wmma_gfx12_small<8, 64>(input.v + 1, col, k, b[k][col]);
    }
  }
  // Map each logical accumulator to its destination register and lane.
  for (int row = 0; row < 16; ++row) {
    for (int col = 0; col < 16; ++col) {
      input.v[2 + (row % 4)][col + 16 * (row / 8) + 32 * ((row / 4) % 2)] = c[row][col];
    }
  }

  // Goldens from rocjitsu's gfx12_dot8_f32 model, using logical matrices
  // independently of VGPR packing. Exact comparisons include NaN sign and payload.
  constexpr float factor_nan = std::bit_cast<float>(0xffc00a3dU);
  constexpr float accumulator_nan = std::bit_cast<float>(0x7fc12345U);
  constexpr float inf = std::numeric_limits<float>::infinity();
  // clang-format off
  const float golden[16][16] = {
      { 0x1.125618p+6f,   0x1.21fcap+4f,   0x1.3299ep+4f,  -0x1.b040ep+6f,  0x1.d2d788p+5f, factor_nan,  -0x1.77f27p+4f, -0x1.0302b8p+6f,  -0x1.2d1a6p+6f,  0x1.06c502p+7f,   0x1.8d327p+5f,  0x1.c45198p+6f,   -0x1.a94cp+0f,  0x1.707074p+6f,   0x1.763fbp+5f,   0x1.3aa98p+6f},
      {     factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan, factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan},
      { -0x1.16231p+4f,   0x1.8ce45p+5f,  -0x1.e022dp+6f, -0x1.ec6ff4p+6f,    0x1.1b94p+4f, factor_nan,  -0x1.e627ep+5f,  0x1.08a9ecp+6f,  -0x1.0c95ap+4f,  0x1.d83bd8p+6f, -0x1.0cd3ccp+6f, -0x1.f06a88p+5f,   0x1.30605p+4f,   0x1.a771fp+6f,  0x1.1d32c4p+6f,   -0x1.c915p+6f},
      {-0x1.a3adccp+6f,  0x1.39d90cp+6f,  -0x1.63f96p+3f, -0x1.8c3a58p+5f,   0x1.aea97p+4f, factor_nan,  0x1.0267dcp+6f,  -0x1.da023p+5f,  0x1.812368p+5f,  -0x1.2133ep+3f,  -0x1.b053bp+6f,  -0x1.7efa7p+6f,  -0x1.a9bd8p+3f,  -0x1.f1509p+4f, -0x1.0edcf4p+6f,  0x1.bf1c1cp+6f},
      { 0x1.6d3498p+5f,   0x1.318dcp+6f,  0x1.346d9cp+6f, -0x1.d04d28p+5f, -0x1.999efcp+6f, factor_nan,  -0x1.836e1p+5f,   0x1.01d66p+3f,   0x1.4c9a1p+4f,  0x1.30c6d8p+6f, -0x1.bf9c98p+5f, -0x1.15e058p+5f,   -0x1.6e86p+6f,  0x1.a56398p+5f, -0x1.273cf8p+5f,  -0x1.5a47bp+4f},
      { -0x1.a77b3p+5f,  -0x1.b19ddp+4f,   0x1.fe419p+6f,   0x1.25975p+6f,  -0x1.d552cp+5f, factor_nan,   0x1.62cc7p+5f,  0x1.d16608p+6f,  -0x1.2cf28p+6f,   0x1.48a5cp+4f,  0x1.d18b28p+5f,  0x1.bf3f88p+6f, -0x1.7bc304p+6f,   -0x1.5c8fp+6f,  0x1.d4362cp+6f,  -0x1.6858dp+5f},
      { -0x1.4c5dap+5f,   0x1.c3d34p+4f,  0x1.04b7b8p+6f,  0x1.6d877cp+6f,  -0x1.d13d8p+4f, factor_nan,   0x1.4ecc2p+4f,   0x1.91c34p+3f,   0x1.76417p+5f,   0x1.54662p+6f, -0x1.623f88p+5f,  0x1.3b3484p+6f, -0x1.0da8fep+7f, -0x1.ac6778p+5f,  0x1.7f33a4p+6f,   0x1.c56d9p+6f},
      {-0x1.29baecp+6f,  -0x1.80b87p+6f, -0x1.10fd98p+6f,  0x1.bed158p+5f, -0x1.349e78p+5f, factor_nan,  0x1.24c312p+7f, -0x1.22263cp+6f, -0x1.68b934p+6f, -0x1.55cc34p+6f,  -0x1.ca18ep+4f, -0x1.4a6c9cp+6f,  -0x1.bd3a9p+5f,   0x1.c7b92p+6f, -0x1.e414b8p+6f,   0x1.ed611p+4f},
      { -0x1.b7f05p+4f,  0x1.c4e3a8p+6f,   0x1.3877cp+6f,  0x1.c4c698p+6f,  -0x1.8590fp+5f, factor_nan,   0x1.0a664p+3f,    0x1.64cap+4f,   -0x1.c981p+2f, -0x1.700e38p+6f, -0x1.9f9928p+5f,  0x1.f1ff08p+5f, -0x1.ca88c8p+5f,  0x1.8beac8p+6f,  -0x1.38fadp+5f,  -0x1.8b264p+4f},
      { 0x1.6ac278p+6f, -0x1.e98764p+6f,   0x1.78456p+3f, -0x1.dee9d8p+5f,  0x1.034ef4p+7f, factor_nan,   0x1.eb016p+4f,   0x1.db27ap+5f,  0x1.affd74p+6f,   0x1.e3f54p+2f, accumulator_nan,  0x1.a98908p+6f,  0x1.801fe4p+6f,  -0x1.be4b2p+4f,  0x1.d5486cp+6f,   0x1.3c067p+6f},
      { -0x1.47f61p+4f,   0x1.7f766p+3f, -0x1.ab804cp+6f,  -0x1.6ac4fp+5f, -0x1.075df6p+7f, factor_nan, -0x1.d9f4fcp+6f,  0x1.25ca54p+6f,  0x1.026a2cp+6f,   0x1.1bcb6p+4f,   0x1.7afa1p+6f,  -0x1.eeb48p+2f, -0x1.0b004ep+7f, -0x1.3ba808p+5f,  0x1.d343f8p+5f, -0x1.66c588p+6f},
      { 0x1.7589d4p+6f,  -0x1.73032p+4f,  0x1.f9c378p+6f,  -0x1.15ba6p+6f,  0x1.9bcbdcp+6f, factor_nan,  -0x1.bbc25p+5f, -0x1.1075e8p+5f,   0x1.61cfap+4f,   0x1.2028ap+6f, -0x1.6d3238p+6f,  0x1.5f2424p+6f,             inf, -0x1.c44a98p+6f, -0x1.534e08p+6f, -0x1.3b107cp+6f},
      {-0x1.703f88p+6f,   0x1.80803p+5f,  0x1.892cfcp+6f, -0x1.2f19b8p+6f,   0x1.694fdp+4f, factor_nan,  -0x1.1558dp+4f, -0x1.cd37d8p+5f, -0x1.c48ae8p+5f,   -0x1.32adp+3f, -0x1.7ebfe8p+5f,  0x1.4e5078p+5f,  0x1.8314d4p+6f,   0x1.8e87dp+5f,  -0x1.d0685p+6f,   0x1.7bc28p+4f},
      {-0x1.82ab2cp+6f,  -0x1.b2a43p+6f, -0x1.b072f8p+6f,   0x1.83cdfp+5f,   0x1.a4468p+4f, factor_nan,  0x1.afc2b8p+5f, -0x1.c21a8cp+6f,   0x1.c43b7p+4f,  -0x1.504d5p+6f,   0x1.72814p+4f,   0x1.f85e8p+4f,  -0x1.95adcp+5f,   -0x1.1937p+1f,  -0x1.9af97p+6f,  -0x1.21b97p+6f},
      { -0x1.189e4p+3f,   0x1.b7bb4p+2f,  -0x1.2c105p+6f,  -0x1.ee618p+5f,   0x1.7c63fp+5f, factor_nan,   0x1.b5dcap+4f,  0x1.854c98p+5f, -0x1.748f48p+6f,   0x1.48e3ap+3f, -0x1.76ea84p+6f,   0x1.1cd87p+5f,  0x1.9a078cp+6f,  -0x1.93878p+1f,  0x1.84e06cp+6f,  0x1.9de524p+6f},
      {-0x1.36dbfcp+6f,  0x1.2c9494p+6f,  -0x1.033edp+4f,  0x1.be4084p+6f,  -0x1.84719p+6f, factor_nan,  0x1.aa5b1cp+6f, -0x1.4f67acp+6f,  0x1.70f3c8p+6f,   0x1.941fcp+2f,   0x1.1345ep+3f,  0x1.4aecd8p+5f,   0x1.1beccp+5f,   0x1.dc464p+6f, -0x1.3772f4p+6f,  0x1.119354p+6f},
  };
  // clang-format on

  rocjitsu::arith::test::expect_matrix_invalid_flags(rj_arith_v_wmma_f32_16x16x16_fp8_fp8_w64_gfx12,
                                                     input, 7, 0, 1, 2,
                                                     RJ_ARITH_WMMA_NEG_C | RJ_ARITH_WMMA_ABS_C);

  // A occupies 1 registers, B 1, C/D 4. Reuse the same arithmetic
  // data for overlapping and disjoint destination ranges.
  const uint64_t modes[] = {0, RJ_ARITH_WMMA_NEG_C | RJ_ARITH_WMMA_ABS_C};
  const auto cpu_levels = rocjitsu::arith::test::supported_cpu_levels();
  for (uint64_t cpu : cpu_levels) {
    for (uint64_t semantics : rocjitsu::arith::test::semantics_modes()) {
      for (uint64_t mode : modes) {
        for (int aliasing_offset : {0, 1, 2, 7}) {
          SCOPED_TRACE(::testing::Message()
                       << cpu << '/' << semantics << '/' << mode << '/' << aliasing_offset);
          Registers r = input;
          ASSERT_EQ(rj_arith_v_wmma_f32_16x16x16_fp8_fp8_w64_gfx12(
                        cpu | semantics, mode, r.v + aliasing_offset, r.v, r.v + 1, r.v + 2),
                    RJ_ARITH_SUCCESS);
          for (int row = 0; row < 16; ++row) {
            for (int col = 0; col < 16; ++col) {
              uint32_t actual =
                  r.v[aliasing_offset + (row % 4)][col + 16 * (row / 8) + 32 * ((row / 4) % 2)];
              if (semantics != RJ_ARITH_SEMANTICS_LOOSE) {
                EXPECT_EQ(actual, std::bit_cast<uint32_t>(golden[row][col]));
              } else {
                rocjitsu::arith::test::expect_f32(actual, golden[row][col], semantics,
                                                  /*atol=*/1e-5,
                                                  /*rtol=*/1e-5);
              }
            }
          }
          r.expect_preserved(input, aliasing_offset, 4);
        }
      }
    }
  }
}
