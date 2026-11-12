// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for v_swmmac_f16_16x16x32_f16_w64.
// Keeps this instruction's matrix inputs and expected results beside its aliasing and flag
// checks.

#include "common/cpu.h"
#include "common/floating_point.h"
#include "common/registers.h"
#include "common/semantics.h"
#include "rj_arith/rj_arith.h"
#include "wmma/helpers.h"
#include <bit>
#include <limits>

#include <gtest/gtest.h>
#include <random>
#include <stdint.h>

namespace {

using Registers = rocjitsu::arith::test::Registers<64, 64>;

} // namespace

// Check matrix results, operand aliases, and instruction-flag contracts across supported CPUs
// and semantics.
TEST(SwmmacF16F16K32W64, Arithmetic) {
  rocjitsu::arith::test::expect_sparse_modifiers<false, 64, true>(
      rj_arith_v_swmmac_f16_16x16x32_f16_w64);

  rocjitsu::arith::test::expect_sparse_half_overflow<64>(rj_arith_v_swmmac_f16_16x16x32_f16_w64);

  std::minstd_rand random;
  uint32_t a[16][16], b[32][16], c[16][16];
  for (int row = 0; row < 16; ++row) {
    for (int k = 0; k < 16; ++k) {
      a[row][k] = rocjitsu::arith::test::random_f16_bits(random);
    }
  }
  for (int k = 0; k < 32; ++k) {
    for (int col = 0; col < 16; ++col) {
      b[k][col] = rocjitsu::arith::test::random_f16_bits(random);
    }
  }
  for (int row = 0; row < 16; ++row) {
    for (int col = 0; col < 16; ++col) {
      c[row][col] = rocjitsu::arith::test::random_f16_bits(random);
    }
  }
  // Two increasing positions per group of four, using high engine bits.
  int selected[16][16];
  rocjitsu::arith::test::make_sparse_indices(random, selected);
  // One NaN and one infinity per logical operand where the format has Inf.
  a[1][2] = rocjitsu::arith::test::kF16QuietNaNBits;
  a[3][4] = rocjitsu::arith::test::kF16PositiveInfinityBits;
  b[6][5] = rocjitsu::arith::test::kF16QuietNaNBits;
  b[8][7] = rocjitsu::arith::test::kF16PositiveInfinityBits;
  c[9][10] = rocjitsu::arith::test::kF16QuietNaNBits;
  c[11][12] = rocjitsu::arith::test::kF16PositiveInfinityBits;

  Registers input;
  for (int row = 0; row < 16; ++row) {
    for (int k = 0; k < 16; ++k) {
      rocjitsu::arith::test::set_swmmac16_a<64>(input.v, row, k, a[row][k]);
    }
  }
  for (int k = 0; k < 32; ++k) {
    for (int col = 0; col < 16; ++col) {
      rocjitsu::arith::test::set_swmmac16_b<64>(input.v + 2, col, k, b[k][col]);
    }
  }
  // C/D layout: register and lane select a row/column; packed gfx12
  // stores adjacent rows in the two halves of each word.
  for (int row = 0; row < 16; ++row) {
    for (int col = 0; col < 16; ++col) {
      auto &word =
          input.v[6 + (((row % 8) / 2) % 2)][col + 16 * (row / 8) + 32 * (((row % 8) / 2) / 2)];
      int shift = 16 * ((row % 8) % 2);
      word = (word & ~(0xffffU << shift)) | (c[row][col] << shift);
    }
  }
  // Replicate valid pairs in each metadata bank; each call below clears
  // the unused banks so a wrong selector cannot read equivalent inputs.
  rocjitsu::arith::test::set_swmmac_indices<16, 64>(input.v[8], selected);

  // Goldens from rocjitsu's gfx12_dot4_packed16 model, using logical matrices
  // and sparse indices independently of VGPR packing; exact NaN encodings included.
  constexpr float factor_nan = std::bit_cast<float>(0xffc7a000U);
  constexpr float accumulator_nan = std::bit_cast<float>(0x7fc02000U);
  constexpr float inf = std::numeric_limits<float>::infinity();
  // clang-format off
  const float golden[16][16] = {
      {-0x1.5ecp-2f, -0x1.d3cp+2f,  -0x1.92p-1f,  0x1.534p+1f, -0x1.7e8p+1f,  0x1.844p+1f,  0x1.a84p+1f,         -inf,  0x1.a74p+0f, -0x1.284p+0f,    -0x1.078p+2f,  0x1.4a4p+0f,  0x1.56cp+0f,  0x1.144p-3f, -0x1.fb8p+0f,  0x1.da8p+0f},
      {  factor_nan,   factor_nan,   factor_nan,   factor_nan,   factor_nan,   factor_nan,   factor_nan,   factor_nan,   factor_nan,   factor_nan,      factor_nan,   factor_nan,   factor_nan,   factor_nan,   factor_nan,   factor_nan},
      {  0x1.34p+2f, -0x1.d34p+0f,  0x1.338p+2f,  0x1.844p+2f,  -0x1.43p+1f,  0x1.84cp+1f,   0x1.59p+0f,  0x1.2b4p+1f, -0x1.7c8p+0f, -0x1.ec8p+0f,    -0x1.ad8p+1f,   0x1.34p-1f,  0x1.258p+0f,  0x1.468p+2f, -0x1.554p+0f,  0x1.714p+2f},
      {         inf,         -inf,         -inf,          inf,         -inf,   factor_nan,          inf,          inf,          inf,         -inf,            -inf,          inf,          inf,          inf,         -inf,          inf},
      {  0x1.21p-1f,  0x1.a88p-1f,   -0x1.ap+0f, -0x1.838p+1f,  0x1.358p+0f,   factor_nan,   0x1.35p+0f, -0x1.9b4p-1f, -0x1.6ccp+0f, -0x1.1a4p+2f,     -0x1.83p-5f,  0x1.074p+0f,  0x1.294p+1f,  0x1.2f4p+1f,   -0x1.4p+1f,  0x1.b74p-1f},
      { -0x1.78p+1f,   0x1.3ap-6f, -0x1.53cp-3f,  -0x1.3fp+0f,  0x1.078p+0f,   factor_nan, -0x1.388p+3f,         -inf, -0x1.84cp+1f, -0x1.02cp+2f,    -0x1.2d4p+2f,  0x1.0dcp+2f, -0x1.1b4p+3f, -0x1.c08p+0f,  0x1.9a4p-2f,   0x1.2bp-1f},
      {-0x1.bd4p+1f, -0x1.3c4p-1f,  -0x1.37p+2f,   0x1.11p+2f, -0x1.efcp+1f, -0x1.62cp-4f,  0x1.7b4p+1f,          inf, -0x1.9d8p+2f,  0x1.288p+0f,    -0x1.af4p+0f, -0x1.0c8p+2f, -0x1.3e8p+1f,  0x1.014p+1f,  0x1.344p-1f, -0x1.934p+1f},
      {-0x1.88cp-1f,  0x1.6b4p+2f, -0x1.00cp+1f, -0x1.01cp+1f,   0x1.21p+0f,   factor_nan,  0x1.458p-1f,          inf,   0x1.bfp-1f,  -0x1.4ap+1f,    -0x1.6f8p+0f, -0x1.eacp+0f,  0x1.3b4p+1f, -0x1.214p+1f, -0x1.ce4p+1f,  0x1.688p+1f},
      { -0x1.31p-1f,  0x1.938p-2f,   0x1.21p-5f,  0x1.894p+0f, -0x1.bb4p+0f, -0x1.844p-1f, -0x1.3b4p-1f, -0x1.028p+1f, -0x1.9ecp+0f,  0x1.b24p-1f,     0x1.55cp+0f,  0x1.8c8p-4f, -0x1.64cp+1f,   0x1.b3p+1f,  -0x1.0dp-2f, -0x1.218p+0f},
      {-0x1.494p+1f,   0x1.09p+2f,  0x1.5b4p-3f,  -0x1.91p+0f, -0x1.174p+0f, -0x1.bbcp+2f, -0x1.5ccp+2f,  0x1.02cp-1f, -0x1.69cp+1f,   0x1.0cp+1f, accumulator_nan,  -0x1.59p+2f, -0x1.804p+1f,  0x1.2a8p+2f,  -0x1.89p+1f, -0x1.4ccp+1f},
      { 0x1.a34p+2f,  0x1.3acp+1f,  0x1.09cp+3f,  0x1.d04p-1f,  -0x1.88p+2f,   factor_nan,  0x1.46cp+1f,          inf,   0x1.71p+2f,  0x1.018p+2f,      0x1.2dp+2f,  0x1.31cp+1f, -0x1.7e8p+0f,  0x1.06cp+2f, -0x1.2a4p-8f,  0x1.968p+2f},
      {-0x1.488p-2f,   0x1.45p-1f,   0x1.4ap+2f, -0x1.348p+1f,  0x1.fe8p+0f,   factor_nan,  -0x1.96p+1f, -0x1.a68p-2f, -0x1.3f8p+1f,  0x1.034p-1f,    -0x1.4fcp+1f, -0x1.6a4p+1f,          inf,  -0x1.f3p+1f,  0x1.00cp+1f,  0x1.d28p+0f},
      { 0x1.decp+2f,  0x1.184p+2f, -0x1.218p-3f, -0x1.4ccp+0f, -0x1.efcp+0f, -0x1.8b4p+2f,  0x1.cc8p-1f, -0x1.db8p+2f,  0x1.228p+2f, -0x1.d3cp-2f,     0x1.158p+1f,   0x1.ecp+1f,  0x1.28cp+0f,   0x1.cap-1f, -0x1.34cp+2f,   0x1.37p+2f},
      {-0x1.318p+1f,  0x1.65cp-2f, -0x1.0d8p+0f, -0x1.e54p-2f,  0x1.598p+2f,  0x1.148p+0f, -0x1.5ccp+1f,         -inf, -0x1.184p+3f,  -0x1.acp+1f,     -0x1.67p+1f,  0x1.52cp+2f, -0x1.8d4p+0f, -0x1.3fcp+1f, -0x1.e14p-1f, -0x1.714p+2f},
      { 0x1.21cp+1f,   0x1.1fp-1f,  0x1.0f4p+1f,  0x1.508p-2f, -0x1.0c8p+2f,   factor_nan,  0x1.cbcp+2f,         -inf,  0x1.404p+3f,  0x1.078p+2f,    -0x1.358p-3f, -0x1.478p+1f, -0x1.348p+3f, -0x1.f88p+2f, -0x1.fe8p-1f, -0x1.a8cp+0f},
      { 0x1.96cp+2f,  -0x1.5fp-1f,  0x1.774p+1f,  0x1.38cp+0f,   0x1.2ap+0f,   factor_nan,  0x1.c0cp+1f,          inf, -0x1.eecp-2f,      0x1p+1f,     0x1.ab8p+0f,  -0x1.27p+0f, -0x1.138p-1f,  0x1.208p+1f, -0x1.3a4p-4f,  0x1.3dcp+0f},
  };
  // clang-format on

  rocjitsu::arith::test::expect_matrix_guest_controls<true, false, true>(
      rj_arith_v_swmmac_f16_16x16x32_f16_w64, input, 6, 0, 2, 8);

  rocjitsu::arith::test::expect_matrix_invalid_flags(
      rj_arith_v_swmmac_f16_16x16x32_f16_w64, input, 6, 0, 2, 8,
      rocjitsu::arith::test::kFloatingWmmaModifiers | RJ_ARITH_SWMMAC_INDEX_KEY_1 |
          RJ_ARITH_SWMMAC_INDEX_KEY_2);

  // A occupies 2 registers, B 4, C/D 2. Reuse the same arithmetic
  // data for overlapping and disjoint destination ranges.
  const uint64_t modes[] = {0, RJ_ARITH_SWMMAC_INDEX_KEY_1, (0) | RJ_ARITH_SWMMAC_INDEX_KEY_2,
                            RJ_ARITH_SWMMAC_INDEX_KEY_1 | RJ_ARITH_SWMMAC_INDEX_KEY_2};
  const auto cpu_levels = rocjitsu::arith::test::supported_cpu_levels();
  for (uint64_t cpu : cpu_levels) {
    for (uint64_t semantics : rocjitsu::arith::test::semantics_modes()) {
      for (uint64_t mode : modes) {
        Registers selected_input = input;
        rocjitsu::arith::test::isolate_sparse_selector<64>(selected_input.v[8], mode);
        for (int aliasing_offset : {6}) {
          SCOPED_TRACE(::testing::Message()
                       << cpu << '/' << semantics << '/' << mode << '/' << aliasing_offset);
          Registers r = selected_input;
          ASSERT_EQ(rj_arith_v_swmmac_f16_16x16x32_f16_w64(
                        cpu | semantics, mode, r.v + aliasing_offset, r.v, r.v + 2, r.v + 8),
                    RJ_ARITH_SUCCESS);
          for (int row = 0; row < 16; ++row) {
            for (int col = 0; col < 16; ++col) {
              uint32_t actual = r.v[aliasing_offset + (((row % 8) / 2) % 2)]
                                   [col + 16 * (row / 8) + 32 * (((row % 8) / 2) / 2)];
              float value =
                  rocjitsu::arith::test::f16_value(uint16_t(actual >> (16 * ((row % 8) % 2))));
              if (semantics != RJ_ARITH_SEMANTICS_LOOSE) {
                EXPECT_EQ(std::bit_cast<uint32_t>(value),
                          std::bit_cast<uint32_t>(golden[row][col]));
              } else {
                rocjitsu::arith::test::expect_f32(std::bit_cast<uint32_t>(value), golden[row][col],
                                                  semantics,
                                                  /*atol=*/1e-5, /*rtol=*/5e-4);
              }
            }
          }
          r.expect_preserved(selected_input, aliasing_offset, 2);
        }
      }
    }
  }
  rocjitsu::arith::test::expect_sparse_aliasing(rj_arith_v_swmmac_f16_16x16x32_f16_w64, input,
                                                /*outputs=*/2, /*b=*/2, /*index=*/8, {0, 1, 2, 8},
                                                cpu_levels);
}
