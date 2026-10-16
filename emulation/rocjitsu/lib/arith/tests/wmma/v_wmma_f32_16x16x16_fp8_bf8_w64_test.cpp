// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for v_wmma_f32_16x16x16_fp8_bf8_w64.
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
TEST(WmmaF32FP8BF8W64, Arithmetic) {
  std::minstd_rand random;
  uint32_t a[16][16], b[16][16], c[16][16];
  for (int row = 0; row < 16; ++row) {
    for (int k = 0; k < 16; ++k) {
      a[row][k] = rocjitsu::arith::test::random_fp8_bits(random);
    }
  }
  for (int k = 0; k < 16; ++k) {
    for (int col = 0; col < 16; ++col) {
      b[k][col] = rocjitsu::arith::test::random_bf8_bits(random);
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
  b[6][5] = rocjitsu::arith::test::kBF8QuietNaNBits;
  b[8][7] = rocjitsu::arith::test::kBF8PositiveInfinityBits;
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

  // Offline goldens from rocjitsu ad2bb973ec4's original DOT models.
  // Reference computation uses logical matrices, independently of the VGPR
  // packing below, and does not call rj-arith. NaNs are compared by classification.
  constexpr float nan = std::numeric_limits<float>::quiet_NaN();
  constexpr float inf = std::numeric_limits<float>::infinity();
  // clang-format off
  const float golden[16][16] = {
      { 0x1.0ce018p+6f,   0x1.2aa4ap+4f,   0x1.ec29ep+4f,  -0x1.ac2bep+6f,  0x1.9a6988p+5f, nan,  -0x1.a6e27p+4f,  inf,  -0x1.1ae76p+6f,  0x1.f4cd04p+6f,   0x1.ac327p+5f,  0x1.c3c198p+6f,  -0x1.8b898p+3f,  0x1.815c74p+6f,   0x1.b277bp+5f,   0x1.3aa08p+6f},
      {            nan,             nan,             nan,             nan,             nan, nan,             nan,  nan,             nan,             nan,             nan,             nan,             nan,             nan,             nan,             nan},
      {-0x1.066f88p+5f,   0x1.44305p+5f, -0x1.210868p+7f, -0x1.dda1f4p+6f,    0x1.2dc4p+4f, nan,  -0x1.f147ep+5f, -inf,  -0x1.3a35ap+4f,  0x1.d3f5d8p+6f, -0x1.0cf7ccp+6f, -0x1.cf5a88p+5f,   0x1.2c805p+4f,   0x1.979bfp+6f,  0x1.fd0588p+5f,   -0x1.ba28p+6f},
      {-0x1.ca43ccp+6f,  0x1.36790cp+6f,  -0x1.c2b2cp+2f, -0x1.5ab258p+5f,  0x1.0a24b8p+5f, nan,  0x1.20b7dcp+6f,  inf,  0x1.7b6b68p+5f,  -0x1.3703ep+3f,  -0x1.c033bp+6f,  -0x1.953e7p+6f,  -0x1.192ecp+4f, -0x1.15f048p+5f, -0x1.1b87f4p+6f,  0x1.cc411cp+6f},
      { 0x1.6ea698p+5f,   0x1.3825cp+6f,  0x1.28cd9cp+6f, -0x1.ff6b28p+5f, -0x1.a1c8fcp+6f, nan,  -0x1.a0761p+5f, -inf,   0x1.6c0a1p+4f,  0x1.5274d8p+6f, -0x1.808498p+5f, -0x1.274058p+5f,   -0x1.62e6p+6f,  0x1.b8df98p+5f, -0x1.191ef8p+5f,  -0x1.7893bp+4f},
      { -0x1.d16f3p+5f, -0x1.079ae8p+5f,  0x1.1092c8p+7f,   0x1.4d095p+6f,  -0x1.cebecp+5f, nan,   0x1.734c7p+5f,  inf,  -0x1.6d248p+6f,   0x1.488dcp+4f,  0x1.36e528p+5f,  0x1.c1bd88p+6f, -0x1.9ec704p+6f,   -0x1.8f3ap+6f,  0x1.9bda2cp+6f,  -0x1.3fb8dp+5f},
      { -0x1.3219ap+5f,   0x1.660f4p+4f,   0x1.9f1b7p+5f,  0x1.9bd17cp+6f,  -0x1.2e10cp+5f, nan,   0x1.634c2p+4f, -inf,   0x1.a9e97p+5f,   0x1.37e42p+6f, -0x1.2eab88p+5f,  0x1.371084p+6f, -0x1.1104fep+7f, -0x1.062bbcp+6f,  0x1.78d5a4p+6f,   0x1.bee39p+6f},
      {-0x1.088eecp+6f,  -0x1.51807p+6f,  -0x1.a79b3p+5f,  0x1.e88158p+5f, -0x1.473e78p+5f, nan,  0x1.0b9b12p+7f,  inf, -0x1.679534p+6f, -0x1.3c2034p+6f,  -0x1.27b47p+5f, -0x1.85949cp+6f,  -0x1.94ea9p+5f,   0x1.c2632p+6f, -0x1.f2b8b8p+6f,  0x1.2f3088p+5f},
      { -0x1.b3605p+4f,  0x1.b4f3a8p+6f,   0x1.23b3cp+6f,  0x1.bcde98p+6f,  -0x1.89ccfp+5f, nan,   0x1.34ce4p+3f, -inf,  -0x1.64e08p+3f, -0x1.770a38p+6f, -0x1.95e528p+5f,  0x1.0ba184p+6f, -0x1.f1d8c8p+5f,  0x1.7d20c8p+6f,  -0x1.345edp+5f,  -0x1.c14e4p+4f},
      { 0x1.4e4878p+6f, -0x1.f5a164p+6f,   0x1.5a456p+3f, -0x1.b7c5d8p+5f,  0x1.db7fe8p+6f, nan,   0x1.e2a96p+4f, -inf,  0x1.c2fd74p+6f,  -0x1.90958p+1f,             nan,  0x1.b43708p+6f,  0x1.6f37e4p+6f,  -0x1.07d39p+5f,  0x1.ee126cp+6f,   0x1.42187p+6f},
      { -0x1.4ab21p+4f,   0x1.c8366p+3f, -0x1.d21c4cp+6f,  -0x1.9e44fp+5f, -0x1.e347ecp+6f, nan, -0x1.ce8efcp+6f, -inf,  0x1.a97058p+5f,   0x1.0233bp+5f,   0x1.58761p+6f,  -0x1.01748p+2f, -0x1.e7f29cp+6f, -0x1.453208p+5f,  0x1.9db3f8p+5f, -0x1.63ba88p+6f},
      { 0x1.9da3d4p+6f,  -0x1.2e932p+4f,  0x1.2622bcp+7f,  -0x1.af2ccp+5f,  0x1.8fc7dcp+6f, nan,  -0x1.dd0a5p+5f,  inf,   0x1.7c2fap+4f,   0x1.1176ap+6f, -0x1.7ddf38p+6f,  0x1.4db424p+6f,             inf, -0x1.b3d298p+6f, -0x1.472608p+6f, -0x1.31f27cp+6f},
      {-0x1.7dcf88p+6f,   0x1.2f583p+5f,  0x1.8d2cfcp+6f, -0x1.399db8p+6f,   0x1.7417ap+3f, nan,  -0x1.fcc0dp+4f, -inf, -0x1.09c474p+6f,   -0x1.ae74p+1f, -0x1.9397e8p+5f,  0x1.2a0078p+5f,  0x1.6cc4d4p+6f,   0x1.a783dp+5f,  -0x1.e15a5p+6f,   0x1.5d9a8p+4f},
      {-0x1.8c0b2cp+6f,  -0x1.bac83p+6f, -0x1.a326f8p+6f,   0x1.8505fp+5f,   0x1.7ef68p+4f, nan,  0x1.07f45cp+6f,  inf,  0x1.1273b8p+5f,  -0x1.7c755p+6f,   0x1.72714p+4f,   0x1.32b74p+5f,  -0x1.9529cp+5f,  -0x1.1d7b8p+2f,  -0x1.a4c37p+6f,  -0x1.20e97p+6f},
      { -0x1.dcf64p+3f,  -0x1.41098p+1f,  -0x1.2b905p+6f,  -0x1.b6998p+5f,   0x1.582bfp+5f, nan,   0x1.f3c4ap+4f,  inf, -0x1.789d48p+6f,   0x1.4b674p+2f, -0x1.888884p+6f,   0x1.51287p+5f,  0x1.60b78cp+6f,  -0x1.3ed1ep+3f,  0x1.545a6cp+6f,  0x1.9c1624p+6f},
      {-0x1.269dfcp+6f,  0x1.216894p+6f,  -0x1.eabb4p+2f,  0x1.db6a84p+6f,  -0x1.62359p+6f, nan,  0x1.b2931cp+6f,  inf,  0x1.5d87c8p+6f,   0x1.8aafep+3f,     0x1.e2fp-4f,  0x1.51ecd8p+5f,   0x1.29bccp+5f,   0x1.d1024p+6f, -0x1.4a17f4p+6f,  0x1.124c54p+6f},
  };
  // clang-format on

  rocjitsu::arith::test::expect_matrix_invalid_flags(rj_arith_v_wmma_f32_16x16x16_fp8_bf8_w64,
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
          ASSERT_EQ(rj_arith_v_wmma_f32_16x16x16_fp8_bf8_w64(
                        cpu | semantics, mode, r.v + aliasing_offset, r.v, r.v + 1, r.v + 2),
                    RJ_ARITH_SUCCESS);
          for (int row = 0; row < 16; ++row) {
            for (int col = 0; col < 16; ++col) {
              uint32_t actual =
                  r.v[aliasing_offset + (row % 4)][col + 16 * (row / 8) + 32 * ((row / 4) % 2)];
              rocjitsu::arith::test::expect_f32(actual, golden[row][col], semantics, /*atol=*/1e-5,
                                                /*rtol=*/1e-5);
            }
          }
          r.expect_preserved(input, aliasing_offset, 4);
        }
      }
    }
  }
}
