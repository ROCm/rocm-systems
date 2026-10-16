// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for v_wmma_f32_16x16x16_bf8_fp8_w64.
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
TEST(WmmaF32BF8FP8W64, Arithmetic) {
  std::minstd_rand random;
  uint32_t a[16][16], b[16][16], c[16][16];
  for (int row = 0; row < 16; ++row) {
    for (int k = 0; k < 16; ++k) {
      a[row][k] = rocjitsu::arith::test::random_bf8_bits(random);
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
  a[1][2] = rocjitsu::arith::test::kBF8QuietNaNBits;
  a[3][4] = rocjitsu::arith::test::kBF8PositiveInfinityBits;
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

  // Offline goldens from rocjitsu ad2bb973ec4's original DOT models.
  // Reference computation uses logical matrices, independently of the VGPR
  // packing below, and does not call rj-arith. NaNs are compared by classification.
  constexpr float nan = std::numeric_limits<float>::quiet_NaN();
  constexpr float inf = std::numeric_limits<float>::infinity();
  // clang-format off
  const float golden[16][16] = {
      { 0x1.151818p+6f,   0x1.9034ap+4f,   0x1.75a9ep+4f,  -0x1.ccc9ep+6f,  0x1.a2dd88p+5f, nan,  -0x1.8d727p+4f,  -0x1.eccf7p+5f,  -0x1.3daf6p+6f,  0x1.0fdf82p+7f,   0x1.878a7p+5f,  0x1.b3a598p+6f,    0x1.3b5ap+1f,  0x1.5d4074p+6f,   0x1.878fbp+5f,   0x1.48308p+6f},
      {            nan,             nan,             nan,             nan,             nan, nan,             nan,             nan,             nan,             nan,             nan,             nan,             nan,             nan,             nan,             nan},
      { -0x1.bc0e2p+3f,   0x1.62c05p+5f,  -0x1.e1a0dp+6f, -0x1.c599f4p+6f,    0x1.f9b8p+3f, nan,  -0x1.a3dfep+5f,  0x1.fb3bd8p+5f,  -0x1.5a15ap+4f,  0x1.c2bfd8p+6f, -0x1.23e7ccp+6f, -0x1.9a9288p+5f,   0x1.83505p+4f,   0x1.ceeffp+6f,  0x1.020ac4p+6f,   -0x1.bf38p+6f},
      {           -inf,             inf,             inf,            -inf,             inf, nan,            -inf,            -inf,            -inf,            -inf,            -inf,             inf,            -inf,             inf,            -inf,             inf},
      { 0x1.aac298p+5f,   0x1.3c05cp+6f,  0x1.45819cp+6f, -0x1.f7e728p+5f, -0x1.95b4fcp+6f, nan,  -0x1.65ce1p+5f,  -0x1.2ca68p+1f,   0x1.39821p+4f,  0x1.5370d8p+6f, -0x1.a4bc98p+5f, -0x1.2e8058p+5f,   -0x1.8196p+6f,  0x1.5a6398p+5f, -0x1.18eaf8p+5f,  -0x1.101bbp+4f},
      { -0x1.9d033p+5f,  -0x1.fe85dp+4f,  0x1.09e4c8p+7f,   0x1.28315p+6f,  -0x1.c04acp+5f, nan,   0x1.31ac7p+5f,  0x1.c96d08p+6f,  -0x1.462c8p+6f,   0x1.133dcp+4f,  0x1.c16528p+5f,  0x1.bb8d88p+6f, -0x1.6ea704p+6f,   -0x1.5ffep+6f,  0x1.e2222cp+6f,  -0x1.8884dp+5f},
      { -0x1.2bf9ap+5f,   0x1.7c274p+4f,  0x1.0ec9b8p+6f,  0x1.a38d7cp+6f,  -0x1.5aaccp+5f, nan,   0x1.f8284p+3f,   0x1.3eb34p+3f,   0x1.90f17p+5f,   0x1.399c2p+6f, -0x1.479788p+5f,  0x1.456884p+6f, -0x1.04defep+7f, -0x1.d07378p+5f,  0x1.7781a4p+6f,   0x1.dc839p+6f},
      {-0x1.14b4ecp+6f,  -0x1.69447p+6f,  -0x1.fdcb3p+5f,  0x1.ac4958p+5f, -0x1.630e78p+5f, nan,  0x1.233312p+7f, -0x1.214e3cp+6f, -0x1.560534p+6f, -0x1.5bd034p+6f,  -0x1.17447p+5f, -0x1.65049cp+6f,  -0x1.e6ba9p+5f,   0x1.c6e32p+6f, -0x1.fdc2b8p+6f,  0x1.4e9088p+5f},
      { -0x1.5d605p+4f,  0x1.ef2da8p+6f,   0x1.42f3cp+6f,  0x1.b9fa98p+6f,  -0x1.870cfp+5f, nan,   0x1.09872p+4f,    0x1.4caap+4f,   -0x1.1cc1p+2f, -0x1.5bfa38p+6f, -0x1.86dd28p+5f,  0x1.e03308p+5f, -0x1.bd20c8p+5f,  0x1.7d6ec8p+6f,  -0x1.293adp+5f,  -0x1.59164p+4f},
      { 0x1.3e6878p+6f, -0x1.ea4964p+6f,    0x1.48acp-2f, -0x1.cf1dd8p+5f,  0x1.f997e8p+6f, nan,   0x1.42a96p+4f,   0x1.35d9dp+6f,  0x1.ca2174p+6f,  -0x1.01158p+1f,             nan,  0x1.a57f08p+6f,  0x1.9873e4p+6f,  -0x1.3c572p+4f,  0x1.ee566cp+6f,   0x1.302c7p+6f},
      { -0x1.904a1p+4f,  0x1.3f9d98p+5f, -0x1.b2e44cp+6f,  -0x1.8dd4fp+5f, -0x1.0821f6p+7f, nan, -0x1.07857ep+7f,  0x1.3b3a54p+6f,  0x1.06842cp+6f,   0x1.d5aecp+3f,   0x1.63621p+6f,  -0x1.9a9d2p+4f, -0x1.01654ep+7f, -0x1.35ae08p+5f,  0x1.fdd3f8p+5f, -0x1.4fe288p+6f},
      { 0x1.970fd4p+6f,  -0x1.5d032p+4f,  0x1.c7b978p+6f,  -0x1.0d6e6p+6f,  0x1.ab91dcp+6f, nan,  -0x1.ee425p+5f,  -0x1.4717dp+4f,   0x1.a0cfap+4f,   0x1.e2654p+5f, -0x1.8d0938p+6f,  0x1.7eb424p+6f,             inf, -0x1.c27e98p+6f, -0x1.757208p+6f, -0x1.3b527cp+6f},
      {-0x1.73d788p+6f,   0x1.31cc3p+5f,  0x1.8e3cfcp+6f, -0x1.4217b8p+6f,   0x1.6603dp+4f, nan,  -0x1.da11ap+3f, -0x1.25e1ecp+6f, -0x1.ee44e8p+5f,   -0x1.967dp+3f, -0x1.85a7e8p+5f,  0x1.a20078p+5f,  0x1.9d80d4p+6f,   0x1.6193dp+5f,  -0x1.d3225p+6f,   0x1.93128p+4f},
      {-0x1.7b4f2cp+6f,  -0x1.9cd43p+6f, -0x1.d30ef8p+6f,   0x1.3c01fp+5f,   0x1.990e8p+4f, nan,  0x1.8bc8b8p+5f, -0x1.af808cp+6f,   0x1.c57f7p+4f,  -0x1.509f5p+6f,   0x1.73314p+4f,   0x1.940e8p+4f,  -0x1.7699cp+5f,    0x1.4109p+1f,  -0x1.997f7p+6f,  -0x1.048b7p+6f},
      { -0x1.7a3b2p+4f,   0x1.016edp+4f,  -0x1.0f685p+6f,  -0x1.0f38cp+6f,   0x1.e117fp+5f, nan,   0x1.e214ap+4f,  0x1.635898p+5f, -0x1.8b1948p+6f,   0x1.e993ap+3f, -0x1.767c84p+6f,   0x1.06e07p+5f,  0x1.93df8cp+6f,  -0x1.7961ep+3f,  0x1.9cca6cp+6f,  0x1.88c824p+6f},
      {-0x1.31b5fcp+6f,  0x1.4d1894p+6f,  -0x1.243dap+3f,  0x1.c93684p+6f,  -0x1.781b9p+6f, nan,  0x1.8ee71cp+6f, -0x1.4c53acp+6f,  0x1.6229c8p+6f,   0x1.187fcp+2f,   0x1.becbcp+2f,  0x1.3504d8p+5f,   0x1.ece98p+4f,   0x1.cc4a4p+6f, -0x1.526bf4p+6f,  0x1.23c054p+6f},
  };
  // clang-format on

  rocjitsu::arith::test::expect_matrix_invalid_flags(rj_arith_v_wmma_f32_16x16x16_bf8_fp8_w64,
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
          ASSERT_EQ(rj_arith_v_wmma_f32_16x16x16_bf8_fp8_w64(
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
