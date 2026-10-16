// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for v_swmmac_f32_16x16x32_bf8_bf8_w32.
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
TEST(SwmmacF32BF8BF8K32W32, Arithmetic) {
  std::minstd_rand random;
  uint32_t a[16][16], b[32][16], c[16][16];
  for (int row = 0; row < 16; ++row) {
    for (int k = 0; k < 16; ++k) {
      a[row][k] = rocjitsu::arith::test::random_bf8_bits(random);
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
  a[1][2] = rocjitsu::arith::test::kBF8QuietNaNBits;
  a[3][4] = rocjitsu::arith::test::kBF8PositiveInfinityBits;
  b[6][5] = rocjitsu::arith::test::kBF8QuietNaNBits;
  b[8][7] = rocjitsu::arith::test::kBF8PositiveInfinityBits;
  c[9][10] = rocjitsu::arith::test::kF32QuietNaNBits;
  c[11][12] = rocjitsu::arith::test::kF32PositiveInfinityBits;

  Registers input;
  for (int row = 0; row < 16; ++row) {
    for (int k = 0; k < 16; ++k) {
      rocjitsu::arith::test::set_swmmac8_a<32>(input.v, row, k, a[row][k]);
    }
  }
  for (int k = 0; k < 32; ++k) {
    for (int col = 0; col < 16; ++col) {
      rocjitsu::arith::test::set_swmmac8_b<32>(input.v + 2, col, k, b[k][col]);
    }
  }
  // Map each logical accumulator to its destination register and lane.
  for (int row = 0; row < 16; ++row) {
    for (int col = 0; col < 16; ++col) {
      input.v[6 + (row % 8)][col + 16 * (row / 8)] = c[row][col];
    }
  }
  // Replicate valid pairs in each metadata bank; each call below clears
  // the unused banks so a wrong selector cannot read equivalent inputs.
  std::fill_n(input.v[14], 32, 0);
  for (int row = 0; row < 16; ++row) {
    for (int k = 0; k < 16; ++k) {
      int lane = row + 16 * ((k / 8) & 1), local = 8 * (k / 16) + k % 8;
      for (int key = 0; key < 32; key += 16) {
        input.v[14][lane] |= uint32_t(selected[row][k] % 4) << (key + 2 * local);
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
      {  0x1.95a0ap+4f,  0x1.8f5c34p+6f,    0x1.2a8dp+6f,   0x1.2a0c4p+3f,  -0x1.9cf4cp+6f,  -0x1.729d8p+2f,  0x1.70d0a4p+6f,            -inf,  0x1.f03d28p+5f, -0x1.f3887cp+6f,  -0x1.8011ap+4f, -0x1.c58144p+6f,  -0x1.2ebecp+2f,   0x1.5a197p+4f,  0x1.4efd94p+6f,   -0x1.1de5p+6f},
      {     factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan},
      { 0x1.8d3df4p+6f, -0x1.fa82a8p+5f, -0x1.5e4014p+6f,  0x1.68dda4p+6f,    0x1.9e42p+2f,   0x1.a65f8p+3f, -0x1.13a898p+6f,   0x1.c4df5p+4f,  -0x1.600fdp+5f,   0x1.ccfbdp+6f,  0x1.1e82bcp+6f,   0x1.ea212p+6f,  0x1.9c37b4p+6f,  0x1.338e68p+6f, -0x1.15822cp+6f,  -0x1.981f5p+6f},
      {            inf,            -inf,            -inf,             inf,            -inf,      factor_nan,             inf,             inf,             inf,            -inf,            -inf,             inf,             inf,             inf,            -inf,             inf},
      { -0x1.3b1e2p+5f,   0x1.3168fp+5f,  0x1.580858p+6f, -0x1.3820c8p+5f,  0x1.b9e608p+5f,      factor_nan,  -0x1.56991p+5f,  0x1.c57218p+6f,   0x1.6160cp+3f,   0x1.76e9ep+6f, -0x1.c0294cp+6f, -0x1.2a1138p+5f,   0x1.8a82bp+4f,  -0x1.9650bp+5f,   0x1.67172p+5f, -0x1.d71584p+6f},
      { 0x1.de9704p+6f,   0x1.5138bp+5f,  0x1.eb2878p+5f,   0x1.61b91p+5f,  -0x1.d7d18p+2f,      factor_nan,     0x1.f9fp+1f,            -inf, -0x1.10cd88p+5f,   0x1.54e0ep+6f,  -0x1.036dap+5f,  0x1.a0fff8p+6f,  0x1.7ed018p+5f,   0x1.fd79ep+5f,  0x1.0c5798p+7f, -0x1.d2439cp+6f},
      {  0x1.a7e74p+5f,  -0x1.a3cc4p+6f,   0x1.87358p+5f,  -0x1.6fb7ap+3f,  0x1.146abcp+6f,   0x1.ea68ep+3f, -0x1.916994p+6f,             inf,  0x1.7e3278p+5f,  -0x1.ceb3cp+6f,  0x1.75f89cp+6f,  0x1.aa4cdcp+6f, -0x1.0f7bfep+7f,  0x1.4e4d78p+6f,    0x1.8c45p+4f,    0x1.8224p+0f},
      {  0x1.9adb4p+4f,  -0x1.acf01p+4f,   0x1.b4683p+4f,  -0x1.32cfap+5f,  0x1.939858p+5f,      factor_nan, -0x1.d37b94p+6f,             inf, -0x1.4f9574p+6f,   0x1.b8bcep+5f,  -0x1.e0646p+4f, -0x1.0c86dcp+7f,  0x1.034728p+6f,  -0x1.3f4c2p+4f, -0x1.13491ep+7f,  0x1.ea1818p+6f},
      { 0x1.81a908p+5f,  -0x1.8978cp+4f,  0x1.1511dcp+6f, -0x1.1c3c08p+5f, -0x1.ba0608p+6f, -0x1.24e904p+6f, -0x1.bed93cp+6f,   0x1.ee67ep+5f,   0x1.ed5b8p+5f,  0x1.e01ca8p+5f, -0x1.115efcp+6f,   0x1.7c0f8p+2f, -0x1.02104ap+7f,  0x1.0bee64p+7f,    0x1.210dp+1f,  0x1.754b48p+6f},
      {-0x1.d91214p+6f, -0x1.cc36b8p+5f,  0x1.49d9f8p+5f,   0x1.d7f67p+5f, -0x1.03c88cp+7f,  -0x1.25b96p+7f, -0x1.b8c6b4p+6f,   0x1.e35d3p+6f, -0x1.114dc6p+7f,  0x1.6cd378p+5f, accumulator_nan,   0x1.cf5fcp+2f, -0x1.8b0a48p+6f, -0x1.551e48p+5f,  0x1.ed7e78p+5f,  -0x1.8d9ddp+4f},
      { 0x1.98c5a8p+6f,   0x1.a2596p+6f,  0x1.25ddc6p+7f,   0x1.276e8p+6f,   0x1.5c46ap+5f,      factor_nan,  0x1.6d7db8p+6f,             inf,  -0x1.5c0f5p+4f,  -0x1.ea41ap+5f,  -0x1.b9a1fp+6f,   0x1.fe023p+5f,  -0x1.7dfe8p+2f,  -0x1.5679dp+5f, -0x1.325ae4p+6f,   0x1.24c86p+3f},
      {  0x1.0ce2bp+4f,  -0x1.749b5p+6f,  -0x1.a796ap+4f,  0x1.4dd684p+6f,  -0x1.85bffp+5f,      factor_nan, -0x1.09978cp+6f, -0x1.d269b8p+5f,   0x1.b0f2ap+4f, -0x1.5cd658p+6f,  0x1.21e32cp+6f,  -0x1.0fb5fp+7f,             inf,   0x1.7a058p+5f,   0x1.5d9e6p+4f,  -0x1.9caddp+4f},
      {-0x1.9b2498p+5f, -0x1.60ebdcp+6f,   0x1.1e77dp+4f,   0x1.765c2p+4f,  -0x1.478b6p+4f,  -0x1.32968p+3f,   0x1.1be57p+4f,   0x1.67179p+4f,  0x1.263a6cp+6f,  -0x1.9ce1cp+3f,   0x1.45e71p+6f,   -0x1.d42bp+4f,  -0x1.af474p+5f,   0x1.174c7p+5f,   0x1.658f8p+1f,  0x1.5f50bcp+6f},
      { 0x1.de0338p+6f,  -0x1.c80ebp+5f, -0x1.d42034p+6f,  0x1.94a45cp+6f, -0x1.a9cbbcp+6f,  0x1.32df84p+6f,  -0x1.088ecp+4f,            -inf,   0x1.62032p+6f,  0x1.a4d8e4p+6f, -0x1.e10e5cp+6f,   0x1.02c94p+3f,  0x1.f7f83cp+6f,   0x1.bc4efp+4f,  0x1.ca856cp+6f,  0x1.3d0dd4p+6f},
      {  0x1.64064p+5f,  -0x1.70be8p+2f,   -0x1.c309p+6f, -0x1.d6bdc4p+6f,   0x1.5b147p+5f,      factor_nan,  0x1.750458p+6f,            -inf,  -0x1.ad12fp+5f,  0x1.0ecebep+7f,   0x1.96bcbp+5f,   0x1.bacb9p+5f, -0x1.1baa74p+7f,  0x1.c4fff8p+5f,   0x1.9806ap+4f, -0x1.0258f4p+6f},
      { 0x1.4b4e38p+6f,   0x1.013bcp+6f,  -0x1.c2e6cp+6f,  0x1.ac3048p+5f,  0x1.2f9dbcp+6f,      factor_nan, -0x1.afe868p+6f,             inf,   0x1.2c3cap+3f, -0x1.9cb068p+6f,  -0x1.a22cep+5f,   0x1.d8497p+6f,    0x1.12d3p+5f,   0x1.01a24p+3f, -0x1.b54558p+6f,   0x1.bb35cp+5f},
  };
  // clang-format on

  rocjitsu::arith::test::expect_matrix_invalid_flags(
      rj_arith_v_swmmac_f32_16x16x32_bf8_bf8_w32, input, 6, 0, 2, 14, RJ_ARITH_SWMMAC_INDEX_KEY_1);

  // A occupies 2 registers, B 4, C/D 8. Reuse the same arithmetic
  // data for overlapping and disjoint destination ranges.
  const uint64_t modes[] = {0, RJ_ARITH_SWMMAC_INDEX_KEY_1};
  const auto cpu_levels = rocjitsu::arith::test::supported_cpu_levels();
  for (uint64_t cpu : cpu_levels) {
    for (uint64_t semantics : rocjitsu::arith::test::semantics_modes()) {
      for (uint64_t mode : modes) {
        Registers selected_input = input;
        rocjitsu::arith::test::isolate_sparse_selector<32>(selected_input.v[14], mode);
        for (int aliasing_offset : {6}) {
          SCOPED_TRACE(::testing::Message()
                       << cpu << '/' << semantics << '/' << mode << '/' << aliasing_offset);
          Registers r = selected_input;
          ASSERT_EQ(rj_arith_v_swmmac_f32_16x16x32_bf8_bf8_w32(
                        cpu | semantics, mode, r.v + aliasing_offset, r.v, r.v + 2, r.v + 14),
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
          r.expect_preserved(selected_input, aliasing_offset, 8);
        }
      }
    }
  }
  rocjitsu::arith::test::expect_sparse_aliasing(rj_arith_v_swmmac_f32_16x16x32_bf8_bf8_w32, input,
                                                /*outputs=*/8, /*b=*/2, /*index=*/14, {0, 1, 2, 14},
                                                cpu_levels);
}
