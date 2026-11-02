// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for v_wmma_f32_16x16x16_bf8_bf8_w32_gfx12.
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

using Registers = rocjitsu::arith::test::Registers<64, 32>;

} // namespace

// Check matrix results, operand aliases, and instruction-flag contracts across supported CPUs
// and semantics.
TEST(WmmaF32BF8BF8W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::expect_wmma_fp8_subnormals<true, true, 32>(
      rj_arith_v_wmma_f32_16x16x16_bf8_bf8_w32_gfx12);
  std::minstd_rand random;
  uint32_t a[16][16], b[16][16], c[16][16];
  for (int row = 0; row < 16; ++row) {
    for (int k = 0; k < 16; ++k) {
      a[row][k] = rocjitsu::arith::test::random_bf8_bits(random);
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
  a[1][2] = rocjitsu::arith::test::kBF8QuietNaNBits;
  a[3][4] = rocjitsu::arith::test::kBF8PositiveInfinityBits;
  b[6][5] = rocjitsu::arith::test::kBF8QuietNaNBits;
  b[8][7] = rocjitsu::arith::test::kBF8PositiveInfinityBits;
  c[9][10] = rocjitsu::arith::test::kF32QuietNaNBits;
  c[11][12] = rocjitsu::arith::test::kF32PositiveInfinityBits;

  Registers input;
  for (int row = 0; row < 16; ++row) {
    for (int k = 0; k < 16; ++k) {
      rocjitsu::arith::test::set_wmma_gfx12_small<8, 32>(input.v, row, k, a[row][k]);
    }
  }
  for (int k = 0; k < 16; ++k) {
    for (int col = 0; col < 16; ++col) {
      rocjitsu::arith::test::set_wmma_gfx12_small<8, 32>(input.v + 2, col, k, b[k][col]);
    }
  }
  // Map each logical accumulator to its destination register and lane.
  for (int row = 0; row < 16; ++row) {
    for (int col = 0; col < 16; ++col) {
      input.v[4 + (row % 8)][col + 16 * (row / 8)] = c[row][col];
    }
  }

  // Goldens from rocjitsu's gfx12_dot8_f32 model, using logical matrices
  // independently of VGPR packing. Exact comparisons include NaN sign and payload.
  constexpr float factor_nan = std::bit_cast<float>(0xffc00a3dU);
  constexpr float accumulator_nan = std::bit_cast<float>(0x7fc12345U);
  constexpr float inf = std::numeric_limits<float>::infinity();
  // clang-format off
  const float golden[16][16] = {
      {  0x1.fa983p+5f,   0x1.7544ap+4f,   0x1.da59ep+4f,  -0x1.b897ep+6f,  0x1.a4e588p+5f, factor_nan, -0x1.19a138p+5f,        inf,  -0x1.3c236p+6f,  0x1.039182p+7f,   0x1.87727p+5f,  0x1.9e3598p+6f,  -0x1.d2698p+3f,  0x1.708474p+6f,   0x1.9a9fbp+5f,   0x1.54fa8p+6f},
      {     factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan, factor_nan,      factor_nan, factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan},
      { -0x1.7e1f1p+4f,   0x1.51a05p+5f, -0x1.14e868p+7f, -0x1.bdcdf4p+6f,    0x1.836cp+4f, factor_nan,  -0x1.9ec7ep+5f,       -inf,  -0x1.5425ap+4f,  0x1.ccf5d8p+6f, -0x1.2f57ccp+6f, -0x1.9c3288p+5f,  0x1.18a028p+5f,   0x1.d295fp+6f,  0x1.e41188p+5f,   -0x1.aa64p+6f},
      {           -inf,             inf,             inf,            -inf,             inf, factor_nan,            -inf, factor_nan,            -inf,            -inf,            -inf,             inf,            -inf,             inf,            -inf,             inf},
      { 0x1.ac7e98p+5f,   0x1.2f1dcp+6f,  0x1.4b599cp+6f, -0x1.f5d328p+5f, -0x1.afdefcp+6f, factor_nan,  -0x1.88061p+5f,       -inf,   0x1.6d2a1p+4f,  0x1.5cd4d8p+6f, -0x1.59d498p+5f, -0x1.45b058p+5f,   -0x1.8926p+6f,  0x1.6d6f98p+5f, -0x1.01c2f8p+5f,  -0x1.2d3bbp+4f},
      { -0x1.e3533p+5f, -0x1.1beae8p+5f,  0x1.1a0ec8p+7f,   0x1.3ea15p+6f,  -0x1.9562cp+5f, factor_nan,   0x1.562c7p+5f,        inf,  -0x1.7ef48p+6f,   0x1.1c35cp+4f,  0x1.294528p+5f,  0x1.c47188p+6f, -0x1.84fb04p+6f,   -0x1.7a16p+6f,  0x1.ad2c2cp+6f,  -0x1.7008dp+5f},
      { -0x1.2e09ap+5f,   0x1.27774p+4f,   0x1.aa037p+5f,  0x1.ab057cp+6f,  -0x1.72e0cp+5f, factor_nan,   0x1.27bc2p+4f,       -inf,   0x1.ae597p+5f,   0x1.26f82p+6f, -0x1.227788p+5f,  0x1.490884p+6f, -0x1.0480fep+7f, -0x1.ff9378p+5f,  0x1.6df1a4p+6f,   0x1.d50f9p+6f},
      {-0x1.c2bdd8p+5f,  -0x1.46b07p+6f,  -0x1.ac8b3p+5f,  0x1.d56158p+5f, -0x1.6e3678p+5f, factor_nan,  0x1.198312p+7f,        inf, -0x1.4b6534p+6f, -0x1.4c2834p+6f,  -0x1.3c747p+5f, -0x1.8f649cp+6f,  -0x1.b87a9p+5f,   0x1.b4232p+6f, -0x1.01be5cp+7f,  0x1.6c3088p+5f},
      { -0x1.5f605p+4f,  0x1.dd07a8p+6f,   0x1.3cc7cp+6f,  0x1.cb4a98p+6f,  -0x1.926cfp+5f, factor_nan,   0x1.fd5e4p+3f,       -inf,  -0x1.26208p+3f, -0x1.685e38p+6f, -0x1.936d28p+5f,  0x1.ede308p+5f, -0x1.fc48c8p+5f,  0x1.6c4ac8p+6f,  -0x1.36f6dp+5f,  -0x1.7c7e4p+4f},
      { 0x1.11ac78p+6f, -0x1.0242b2p+7f,  -0x1.3aea8p+1f, -0x1.f30dd8p+5f,  0x1.d0abe8p+6f, factor_nan,   0x1.36996p+4f,       -inf,  0x1.d04d74p+6f,  -0x1.c30acp+2f, accumulator_nan,  0x1.b6db08p+6f,  0x1.7cf3e4p+6f,  -0x1.891f2p+4f,  0x1.e5926cp+6f,   0x1.313c7p+6f},
      { -0x1.d4ba1p+4f,  0x1.27dd98p+5f, -0x1.fd044cp+6f,  -0x1.da54fp+5f, -0x1.e243ecp+6f, factor_nan, -0x1.fa4afcp+6f,       -inf,  0x1.aec858p+5f,   0x1.9fc76p+4f,   0x1.3e561p+6f,  -0x1.ecba4p+3f, -0x1.01614ep+7f, -0x1.500608p+5f,  0x1.ad03f8p+5f, -0x1.5e6688p+6f},
      { 0x1.b4cfd4p+6f,  -0x1.99d32p+4f,  0x1.074ebcp+7f,  -0x1.d1eccp+5f,  0x1.9d57dcp+6f, factor_nan, -0x1.053d28p+6f,        inf,   0x1.e14fap+4f,   0x1.d9f54p+5f, -0x1.8b2b38p+6f,  0x1.6db424p+6f,             inf, -0x1.acfe98p+6f, -0x1.6c2e08p+6f, -0x1.38ba7cp+6f},
      {-0x1.7da388p+6f,   0x1.e8a06p+4f,  0x1.ae94fcp+6f, -0x1.633db8p+6f,   0x1.22dbdp+4f, factor_nan, -0x1.1ae868p+5f,       -inf, -0x1.211874p+6f,   -0x1.763ap+2f, -0x1.a42fe8p+5f,  0x1.41a078p+5f,  0x1.8fa4d4p+6f,   0x1.935bdp+5f,  -0x1.de225p+6f,   0x1.795a8p+4f},
      {-0x1.86632cp+6f,  -0x1.c7383p+6f, -0x1.aba2f8p+6f,   0x1.45e5fp+5f,   0x1.1e168p+4f, factor_nan,  0x1.ecdcb8p+5f,        inf,  0x1.2973b8p+5f,  -0x1.80015p+6f,   0x1.96314p+4f,   0x1.ffce8p+4f,  -0x1.b189cp+5f,    0x1.eb89p+1f,  -0x1.a8bb7p+6f,  -0x1.082d7p+6f},
      { -0x1.df2b2p+4f,   0x1.6b9dap+3f,  -0x1.33885p+6f,  -0x1.ce018p+5f,   0x1.a58ffp+5f, factor_nan,   0x1.b474ap+4f,        inf, -0x1.957148p+6f,   0x1.e1a74p+2f, -0x1.84c484p+6f,   0x1.2c007p+5f,  0x1.4db78cp+6f,  -0x1.84b8fp+4f,  0x1.65c86cp+6f,  0x1.851624p+6f},
      {-0x1.30edfcp+6f,  0x1.397894p+6f,  -0x1.0e9dap+3f,  0x1.d91684p+6f,  -0x1.55d99p+6f, factor_nan,  0x1.90731cp+6f,        inf,  0x1.50dbc8p+6f,   0x1.634fep+3f,   0x1.1d178p+1f,  0x1.383cd8p+5f,   0x1.0bbccp+5f,   0x1.ca4a4p+6f, -0x1.55dff4p+6f,  0x1.1f7454p+6f},
  };
  // clang-format on

  rocjitsu::arith::test::expect_matrix_guest_controls<true, false>(
      rj_arith_v_wmma_f32_16x16x16_bf8_bf8_w32_gfx12, input, 13, 0, 2, 4);

  rocjitsu::arith::test::expect_matrix_invalid_flags(rj_arith_v_wmma_f32_16x16x16_bf8_bf8_w32_gfx12,
                                                     input, 13, 0, 2, 4,
                                                     RJ_ARITH_WMMA_NEG_C | RJ_ARITH_WMMA_ABS_C);

  // A occupies 2 registers, B 2, C/D 8. Reuse the same arithmetic
  // data for overlapping and disjoint destination ranges.
  const uint64_t modes[] = {0, RJ_ARITH_WMMA_NEG_C | RJ_ARITH_WMMA_ABS_C};
  const auto cpu_levels = rocjitsu::arith::test::supported_cpu_levels();
  for (uint64_t cpu : cpu_levels) {
    for (uint64_t semantics : rocjitsu::arith::test::semantics_modes()) {
      for (uint64_t mode : modes) {
        for (int aliasing_offset : {0, 1, 2, 3, 4, 13}) {
          SCOPED_TRACE(::testing::Message()
                       << cpu << '/' << semantics << '/' << mode << '/' << aliasing_offset);
          Registers r = input;
          ASSERT_EQ(rj_arith_v_wmma_f32_16x16x16_bf8_bf8_w32_gfx12(
                        cpu | semantics, mode, r.v + aliasing_offset, r.v, r.v + 2, r.v + 4),
                    RJ_ARITH_SUCCESS);
          for (int row = 0; row < 16; ++row) {
            for (int col = 0; col < 16; ++col) {
              uint32_t actual = r.v[aliasing_offset + (row % 8)][col + 16 * (row / 8)];
              if (semantics != RJ_ARITH_SEMANTICS_LOOSE) {
                EXPECT_EQ(actual, std::bit_cast<uint32_t>(golden[row][col]));
              } else {
                rocjitsu::arith::test::expect_f32(actual, golden[row][col], semantics,
                                                  /*atol=*/1e-5,
                                                  /*rtol=*/1e-5);
              }
            }
          }
          r.expect_preserved(input, aliasing_offset, 8);
        }
      }
    }
  }
}
