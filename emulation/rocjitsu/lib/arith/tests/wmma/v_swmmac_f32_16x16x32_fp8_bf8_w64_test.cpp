// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for v_swmmac_f32_16x16x32_fp8_bf8_w64.
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
TEST(SwmmacF32FP8BF8K32W64, Arithmetic) {
  std::minstd_rand random;
  uint32_t a[16][16], b[32][16], c[16][16];
  for (int row = 0; row < 16; ++row) {
    for (int k = 0; k < 16; ++k) {
      a[row][k] = rocjitsu::arith::test::random_fp8_bits(random);
    }
  }
  for (int k = 0; k < 32; ++k) {
    for (int col = 0; col < 16; ++col) {
      b[k][col] = rocjitsu::arith::test::random_bf8_bits(random);
    }
  }
  for (int row = 0; row < 16; ++row) {
    for (int col = 0; col < 16; ++col) {
      c[row][col] = rocjitsu::arith::test::random_f32_bits(random);
    }
  }
  // Two increasing positions per group of four, using high engine bits.
  int selected[16][16];
  rocjitsu::arith::test::make_sparse_indices(random, selected);
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
      rocjitsu::arith::test::set_swmmac8_a<64>(input.v, row, k, a[row][k]);
    }
  }
  for (int k = 0; k < 32; ++k) {
    for (int col = 0; col < 16; ++col) {
      rocjitsu::arith::test::set_swmmac8_b<64>(input.v + 1, col, k, b[k][col]);
    }
  }
  // Map each logical accumulator to its destination register and lane.
  for (int row = 0; row < 16; ++row) {
    for (int col = 0; col < 16; ++col) {
      input.v[3 + ((row % 8) % 4)][col + 16 * (row / 8) + 32 * ((row % 8) / 4)] = c[row][col];
    }
  }
  // Replicate valid pairs in each metadata bank; each call below clears
  // the unused banks so a wrong selector cannot read equivalent inputs.
  std::fill_n(input.v[7], 64, 0);
  for (int row = 0; row < 16; ++row) {
    for (int k = 0; k < 16; ++k) {
      int lane = row + 16 * (k / 8) + 32 * ((k / 4) & 1), local = k % 4;
      for (int key = 0; key < 32; key += 8) {
        input.v[7][lane] |= uint32_t(selected[row][k] % 4) << (key + 2 * local);
      }
    }
  }

  // Goldens from rocjitsu's gfx12_dot8_f32 model, using logical matrices
  // and sparse indices independently of VGPR packing; exact NaN encodings included.
  constexpr float factor_nan = std::bit_cast<float>(0xffc00a3dU);
  constexpr float accumulator_nan = std::bit_cast<float>(0x7fc12345U);
  constexpr float inf = std::numeric_limits<float>::infinity();
  // clang-format off
  const float golden[16][16] = {
      {  0x1.0cc8ap+4f,  0x1.90ec34p+6f,    0x1.003dp+6f,   0x1.a42c4p+3f,  -0x1.86d8cp+6f,  -0x1.011d8p+2f,  0x1.58eca4p+6f,            -inf,  0x1.e79528p+5f, -0x1.df247cp+6f,  -0x1.eb068p+2f, -0x1.a95544p+6f,  -0x1.5f5f6p+3f,   0x1.a8e17p+4f,  0x1.470594p+6f,   -0x1.327dp+6f},
      {     factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan},
      { 0x1.62d9f4p+6f, -0x1.f62aa8p+5f, -0x1.743c14p+6f,  0x1.3d9da4p+6f,   -0x1.fffcp+1f,     0x1.afcp-4f, -0x1.197c98p+6f,   0x1.70cf5p+4f,  -0x1.579fdp+5f,   0x1.cedbdp+6f,  0x1.157abcp+6f,   0x1.b4872p+6f,  0x1.8237b4p+6f,  0x1.277068p+6f, -0x1.1be02cp+6f,  -0x1.92a75p+6f},
      {-0x1.763f14p+6f,   0x1.100ccp+6f,  -0x1.235f8p+4f,  0x1.444578p+5f, -0x1.89e6a8p+5f,      factor_nan, -0x1.df0b28p+6f, -0x1.ef3ab8p+5f, -0x1.6a12e8p+5f, -0x1.21f504p+6f,  0x1.1f04a4p+6f,  0x1.307a18p+5f,  -0x1.ae59ep+5f,  0x1.ce96d8p+6f,   0x1.7326ep+4f,   0x1.40b2bp+4f},
      { -0x1.033e2p+5f,   0x1.bac1ep+4f,  0x1.554858p+6f, -0x1.4620c8p+5f,  0x1.f4fa08p+5f,      factor_nan,  -0x1.75c91p+5f,  0x1.e8d618p+6f,   0x1.4fa0cp+3f,   0x1.7e1dep+6f, -0x1.b2734cp+6f, -0x1.323938p+5f,   0x1.bff2bp+4f,  -0x1.68a0bp+5f,   0x1.6c132p+5f, -0x1.e97584p+6f},
      { 0x1.e63f04p+6f,   0x1.6698bp+5f,  0x1.06483cp+6f,   0x1.8b391p+5f,  -0x1.9c88cp+3f,      factor_nan,    0x1.4b58p+2f,            -inf, -0x1.0aad88p+5f,   0x1.3e88ep+6f,  -0x1.c7134p+4f,  0x1.bd47f8p+6f,  0x1.8b7c18p+5f,   0x1.0650fp+6f,  0x1.057f98p+7f, -0x1.b9079cp+6f},
      {  0x1.9abf4p+5f,  -0x1.95a24p+6f,   0x1.2ce18p+5f,  -0x1.6657ap+3f,  0x1.0954bcp+6f,   0x1.51c8ep+3f, -0x1.a8ff94p+6f,             inf,  0x1.528a78p+5f,  -0x1.e479cp+6f,  0x1.65d49cp+6f,  0x1.8d32dcp+6f, -0x1.1bddfep+7f,  0x1.62d178p+6f,    0x1.2f95p+4f,    0x1.4289p+2f},
      {  0x1.b91b4p+4f,  -0x1.e3901p+4f,   0x1.9ae83p+4f,  -0x1.3d9fap+5f,  0x1.c40858p+5f,      factor_nan, -0x1.f5c394p+6f,             inf, -0x1.49c574p+6f,   0x1.bf0cep+5f,  -0x1.19e23p+5f, -0x1.02fcdcp+7f,   0x1.c66e5p+5f,  -0x1.18ac2p+4f, -0x1.1b1b1ep+7f,  0x1.fe9818p+6f},
      { 0x1.626108p+5f,  -0x1.b368cp+4f,  0x1.3151dcp+6f, -0x1.5f6c08p+5f, -0x1.a4cc08p+6f, -0x1.289104p+6f, -0x1.c3e13cp+6f,   0x1.d7dbep+5f,   0x1.e57f8p+5f,  0x1.eee8a8p+5f, -0x1.0982fcp+6f,   0x1.4987cp+3f, -0x1.f2d894p+6f,  0x1.f8dac8p+6f,   0x1.29a34p+3f,  0x1.698f48p+6f},
      {-0x1.dc3a14p+6f, -0x1.14bb5cp+6f,  0x1.2859f8p+5f,  0x1.0a6b38p+6f, -0x1.1eba8cp+7f,  -0x1.14c96p+7f, -0x1.a8feb4p+6f,   0x1.b7e13p+6f, -0x1.121bc6p+7f,  0x1.316378p+5f, accumulator_nan,   0x1.2a1fcp+2f, -0x1.8be248p+6f, -0x1.5dae48p+5f,  0x1.08eb3cp+6f,  -0x1.bc9bap+3f},
      { 0x1.5d85a8p+6f,   0x1.a3696p+6f,  0x1.0ee1c6p+7f,   0x1.26468p+6f,   0x1.a236ap+5f,      factor_nan,  0x1.3437b8p+6f,             inf,  -0x1.ec6f5p+4f,  -0x1.13a4dp+6f,  -0x1.a42bfp+6f,  0x1.33a518p+6f,   -0x1.e7e8p-2f,  -0x1.74f9dp+5f, -0x1.164ae4p+6f,   0x1.6c90cp+2f},
      {  0x1.760acp+2f,  -0x1.770b5p+6f,  -0x1.7576ap+4f,  0x1.376284p+6f,  -0x1.1bf3fp+5f,      factor_nan, -0x1.0a7b8cp+6f, -0x1.103edcp+6f,   0x1.46f2ap+4f, -0x1.70ea58p+6f,  0x1.04fb2cp+6f,  -0x1.0e05fp+7f,             inf,   0x1.4fb58p+5f,   0x1.2cee6p+4f,  -0x1.ededdp+4f},
      {-0x1.97a498p+5f, -0x1.499fdcp+6f,   0x1.0e27dp+4f,   0x1.713c2p+4f,  -0x1.036b6p+4f,  -0x1.05e68p+3f,   0x1.05fd7p+4f,   0x1.6a279p+4f,  0x1.0b0a6cp+6f,  -0x1.3821cp+3f,   0x1.3b371p+6f,   -0x1.ae2bp+4f,  -0x1.fa374p+5f,   0x1.9e9c7p+5f,   0x1.f787cp+2f,  0x1.613cbcp+6f},
      { 0x1.c1cb38p+6f,  -0x1.ed56bp+5f, -0x1.d8a234p+6f,  0x1.a4265cp+6f, -0x1.a8e5bcp+6f,  0x1.422384p+6f,  -0x1.411ecp+4f,            -inf,   0x1.66472p+6f,  0x1.c5bee4p+6f, -0x1.e0145cp+6f,   0x1.ac328p+2f,  0x1.d1583cp+6f,   0x1.7eb6fp+4f,  0x1.e2896cp+6f,  0x1.356dd4p+6f},
      {  0x1.a26e4p+5f,  -0x1.91ff4p+3f,   -0x1.d7f1p+6f, -0x1.a345c4p+6f,   0x1.623c7p+5f,      factor_nan,  0x1.407458p+6f,            -inf,  -0x1.f042fp+5f,  0x1.01c2bep+7f,   0x1.ae38bp+5f,  0x1.0595c8p+6f, -0x1.155e74p+7f,  0x1.27f9fcp+6f,   0x1.16a6ap+4f, -0x1.eb61e8p+5f},
      { 0x1.3e5638p+6f,   0x1.191bcp+6f,  -0x1.c1e6cp+6f,  0x1.d77048p+5f,  0x1.3d5dbcp+6f,      factor_nan, -0x1.c26868p+6f,             inf,   0x1.0f5cap+3f, -0x1.a01068p+6f,  -0x1.8434ep+5f,   0x1.f1f97p+6f,    0x1.0e73p+5f,   -0x1.1df7p+1f, -0x1.a98d58p+6f,   0x1.a36dcp+5f},
  };
  // clang-format on

  rocjitsu::arith::test::expect_matrix_invalid_flags(
      rj_arith_v_swmmac_f32_16x16x32_fp8_bf8_w64, input, 3, 0, 1, 7,
      RJ_ARITH_SWMMAC_INDEX_KEY_1 | RJ_ARITH_SWMMAC_INDEX_KEY_2);

  // A occupies 1 registers, B 2, C/D 4. Reuse the same arithmetic
  // data for overlapping and disjoint destination ranges.
  const uint64_t modes[] = {0, RJ_ARITH_SWMMAC_INDEX_KEY_1, (0) | RJ_ARITH_SWMMAC_INDEX_KEY_2,
                            RJ_ARITH_SWMMAC_INDEX_KEY_1 | RJ_ARITH_SWMMAC_INDEX_KEY_2};
  const auto cpu_levels = rocjitsu::arith::test::supported_cpu_levels();
  for (uint64_t cpu : cpu_levels) {
    for (uint64_t semantics : rocjitsu::arith::test::semantics_modes()) {
      for (uint64_t mode : modes) {
        Registers selected_input = input;
        rocjitsu::arith::test::isolate_sparse_selector<64>(selected_input.v[7], mode);
        for (int aliasing_offset : {3}) {
          SCOPED_TRACE(::testing::Message()
                       << cpu << '/' << semantics << '/' << mode << '/' << aliasing_offset);
          Registers r = selected_input;
          ASSERT_EQ(rj_arith_v_swmmac_f32_16x16x32_fp8_bf8_w64(
                        cpu | semantics, mode, r.v + aliasing_offset, r.v, r.v + 1, r.v + 7),
                    RJ_ARITH_SUCCESS);
          for (int row = 0; row < 16; ++row) {
            for (int col = 0; col < 16; ++col) {
              uint32_t actual = r.v[aliasing_offset + ((row % 8) % 4)]
                                   [col + 16 * (row / 8) + 32 * ((row % 8) / 4)];
              if (semantics != RJ_ARITH_SEMANTICS_LOOSE) {
                EXPECT_EQ(actual, std::bit_cast<uint32_t>(golden[row][col]));
              } else {
                rocjitsu::arith::test::expect_f32(actual, golden[row][col], semantics,
                                                  /*atol=*/1e-5,
                                                  /*rtol=*/1e-5);
              }
            }
          }
          r.expect_preserved(selected_input, aliasing_offset, 4);
        }
      }
    }
  }
  rocjitsu::arith::test::expect_sparse_aliasing(rj_arith_v_swmmac_f32_16x16x32_fp8_bf8_w64, input,
                                                /*outputs=*/4, /*b=*/1, /*index=*/7, {0, 1, 1, 7},
                                                cpu_levels);
}
