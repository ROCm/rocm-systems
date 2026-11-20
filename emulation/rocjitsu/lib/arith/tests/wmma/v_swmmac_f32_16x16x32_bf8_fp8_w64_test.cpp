// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for v_swmmac_f32_16x16x32_bf8_fp8_w64.
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
TEST(SwmmacF32BF8FP8K32W64, Arithmetic) {
  std::minstd_rand random;
  uint32_t a[16][16], b[32][16], c[16][16];
  for (int row = 0; row < 16; ++row) {
    for (int k = 0; k < 16; ++k) {
      a[row][k] = rocjitsu::arith::test::random_bf8_bits(random);
    }
  }
  for (int k = 0; k < 32; ++k) {
    for (int col = 0; col < 16; ++col) {
      b[k][col] = rocjitsu::arith::test::random_fp8_bits(random);
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
  b[6][5] = rocjitsu::arith::test::kFP8QuietNaNBits;
  // FP8 E4M3FN has no infinity encoding.
  c[9][10] = rocjitsu::arith::test::kF32QuietNaNBits;
  c[11][12] = rocjitsu::arith::test::kF32PositiveInfinityBits;

  // A uses the dense small-element layout with compressed K coordinates.
  Registers input;
  for (int row = 0; row < 16; ++row) {
    for (int k = 0; k < 16; ++k) {
      rocjitsu::arith::test::set_wmma_gfx12_small<8, 64>(input.v, row, k, a[row][k]);
    }
  }
  for (int k = 0; k < 32; ++k) {
    for (int col = 0; col < 16; ++col) {
      rocjitsu::arith::test::set_swmmac_small_b<8, 64>(input.v + 1, col, k, b[k][col]);
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
  rocjitsu::arith::test::set_swmmac_indices<8, 64>(input.v[7], selected);

  // Goldens from rocjitsu's gfx12_dot8_f32 model, using logical matrices
  // and sparse indices independently of VGPR packing; exact NaN encodings included.
  constexpr float factor_nan = std::bit_cast<float>(0xffc00a3dU);
  constexpr float accumulator_nan = std::bit_cast<float>(0x7fc12345U);
  constexpr float inf = std::numeric_limits<float>::infinity();
  // clang-format off
  const float golden[16][16] = {
      {  0x1.124c5p+5f,  0x1.adec34p+6f,    0x1.21ddp+6f,   0x1.316c4p+3f,  -0x1.6294cp+6f,  -0x1.e25d8p+2f,  0x1.8e7aa4p+6f,  0x1.7e031cp+6f,  0x1.0eac94p+6f, -0x1.f1a47cp+6f,  -0x1.bcc1ap+4f, -0x1.c78144p+6f,  -0x1.3bafbp+4f,   0x1.99797p+4f,  0x1.43ad94p+6f,   -0x1.44b9p+6f},
      {     factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan},
      { 0x1.9cfff4p+6f, -0x1.9e92a8p+5f, -0x1.619014p+6f,  0x1.5265a4p+6f,    0x1.5322p+2f,    0x1.ab7fp+2f, -0x1.0bae98p+6f,   0x1.b42f5p+4f,  -0x1.3cefdp+5f,   0x1.bde3dp+6f,  0x1.1102bcp+6f,   0x1.04609p+7f,  0x1.9383b4p+6f,  0x1.04d468p+6f, -0x1.20ea2cp+6f,  -0x1.af8f5p+6f},
      {            inf,            -inf,            -inf,             inf,            -inf,      factor_nan,             inf,             inf,             inf,            -inf,            -inf,             inf,             inf,             inf,            -inf,             inf},
      { -0x1.4c222p+5f,   0x1.e689ep+4f,  0x1.4a6458p+6f, -0x1.4878c8p+5f,  0x1.b39a08p+5f,      factor_nan,  -0x1.60191p+5f,  0x1.d71618p+6f,   0x1.06c0cp+3f,   0x1.83c7ep+6f, -0x1.d2174cp+6f, -0x1.11a938p+5f,   0x1.3c32bp+4f,  -0x1.af6cbp+5f,   0x1.7ca72p+5f, -0x1.d26784p+6f},
      { 0x1.dfdd04p+6f,   0x1.6448bp+5f,  0x1.33d83cp+6f,   0x1.aed11p+5f,  -0x1.556c6p+4f,      factor_nan,    0x1.0cdcp+3f,  0x1.9f4ea8p+5f,  -0x1.851b1p+4f,   0x1.53d0ep+6f,  -0x1.232dap+5f,  0x1.a667f8p+6f,  0x1.899018p+5f,   0x1.f3b1ep+5f,  0x1.014598p+7f, -0x1.a4ef9cp+6f},
      {  0x1.8a9f4p+5f,  -0x1.a3584p+6f,   0x1.c8418p+5f,  -0x1.ff37ap+3f,  0x1.1ca8bcp+6f,   0x1.55b8ep+3f, -0x1.86f194p+6f,  0x1.99ade8p+6f,  0x1.907678p+5f,  -0x1.b6bdcp+6f,  0x1.61189cp+6f,  0x1.bac2dcp+6f, -0x1.fb17fcp+6f,  0x1.450178p+6f,    0x1.67a5p+4f,   -0x1.36f7p+2f},
      {  0x1.89134p+4f, -0x1.0c1808p+5f,   0x1.d5683p+4f,  -0x1.1413ap+5f,  0x1.213858p+5f,      factor_nan, -0x1.d60394p+6f,  0x1.ffc4c4p+6f, -0x1.4c6974p+6f,   0x1.6f5cep+5f,  -0x1.979c6p+4f, -0x1.0f1cdcp+7f,  0x1.25b328p+6f,  -0x1.545c2p+4f, -0x1.0fc41ep+7f,  0x1.d97018p+6f},
      { 0x1.6abd08p+5f,  -0x1.5f00cp+4f,  0x1.1341dcp+6f, -0x1.514408p+5f, -0x1.bc0408p+6f, -0x1.271504p+6f, -0x1.a3073cp+6f,   0x1.f51fep+5f,   0x1.f67f8p+5f,  0x1.ec1ca8p+5f, -0x1.146afcp+6f,   0x1.6647cp+3f, -0x1.f66894p+6f,  0x1.034664p+7f,   -0x1.28f3p+1f,  0x1.738f48p+6f},
      {-0x1.a99a14p+6f, -0x1.c286b8p+5f,  0x1.9c99f8p+5f,   0x1.634e7p+5f, -0x1.07038cp+7f,  -0x1.1f356p+7f, -0x1.8102b4p+6f,   0x1.e2993p+6f, -0x1.e9dd8cp+6f,  0x1.4c1378p+5f, accumulator_nan,   0x1.43dfcp+2f, -0x1.918a48p+6f, -0x1.1c5e48p+5f,  0x1.14073cp+6f,  -0x1.b3dddp+4f},
      { 0x1.6365a8p+6f,   0x1.a6b16p+6f,  0x1.e1db8cp+6f,    0x1.c8cdp+5f,   0x1.6ba6ap+5f,      factor_nan,  0x1.8489b8p+6f, -0x1.ff5064p+6f,  -0x1.8e3f5p+4f,  -0x1.0fbcdp+6f,  -0x1.9d51fp+6f,  0x1.0f5118p+6f,   -0x1.eefdp+1f,  -0x1.86e1dp+5f, -0x1.187ae4p+6f,   0x1.f2086p+3f},
      {  0x1.25e2bp+4f,  -0x1.75bb5p+6f,  -0x1.edb6ap+4f,  0x1.553c84p+6f,  -0x1.9d37fp+5f,      factor_nan, -0x1.ff3f18p+5f, -0x1.d991b8p+5f,   0x1.58d2ap+4f, -0x1.685858p+6f,  0x1.25372cp+6f,  -0x1.fef3ep+6f,             inf,   0x1.a98d8p+5f,   0x1.751ccp+3f,  -0x1.7ce5dp+4f},
      {-0x1.b9d898p+5f, -0x1.71cddcp+6f,   0x1.1dc7dp+4f,   0x1.712c2p+4f,  -0x1.27ab6p+4f,   -0x1.3adap+1f,   0x1.945aep+3f,   0x1.1b779p+4f,  0x1.0eea6cp+6f,  -0x1.e2038p+2f,   0x1.311f1p+6f,   -0x1.a62bp+4f,  -0x1.12ebap+6f,   0x1.35a47p+5f,   0x1.87cf8p+1f,  0x1.75acbcp+6f},
      { 0x1.ceed38p+6f, -0x1.027358p+6f, -0x1.af8234p+6f,  0x1.95305cp+6f, -0x1.cbc3bcp+6f,  0x1.596184p+6f,  -0x1.a00d8p+3f,   0x1.92febp+4f,   0x1.97af2p+6f,  0x1.a3d4e4p+6f, -0x1.e2d25cp+6f,    0x1.d965p+1f,  0x1.d3983cp+6f,   0x1.581efp+4f,  0x1.df376cp+6f,  0x1.5c0dd4p+6f},
      {  0x1.08ee4p+5f,  -0x1.8f5f4p+3f,   -0x1.9addp+6f, -0x1.dcf5c4p+6f,   0x1.6d3c7p+5f,      factor_nan,  0x1.273458p+6f,  -0x1.3dc17p+6f,  -0x1.6d62fp+5f,  0x1.092abep+7f,   0x1.e044bp+5f,   0x1.8dbb9p+5f, -0x1.132274p+7f,  0x1.0bc3fcp+6f,   0x1.157b5p+5f, -0x1.bca9e8p+5f},
      { 0x1.3cc038p+6f,   0x1.e0878p+5f,  -0x1.de72cp+6f,  0x1.914c48p+5f,  0x1.e0ef78p+5f,      factor_nan, -0x1.e46068p+6f,  -0x1.b5aeap+4f,   0x1.207cap+3f, -0x1.a4ac68p+6f,  -0x1.93f4ep+5f,   0x1.ca017p+6f,    0x1.2bc3p+5f,   -0x1.fdf7p+1f, -0x1.c3ab58p+6f,   0x1.a9a1cp+5f},
  };
  // clang-format on

  rocjitsu::arith::test::expect_matrix_guest_controls<true, false>(
      rj_arith_v_swmmac_f32_16x16x32_bf8_fp8_w64, input, 3, 0, 1, 7);

  rocjitsu::arith::test::expect_matrix_invalid_flags(
      rj_arith_v_swmmac_f32_16x16x32_bf8_fp8_w64, input, 3, 0, 1, 7,
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
          ASSERT_EQ(rj_arith_v_swmmac_f32_16x16x32_bf8_fp8_w64(
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
  rocjitsu::arith::test::expect_sparse_aliasing(rj_arith_v_swmmac_f32_16x16x32_bf8_fp8_w64, input,
                                                /*outputs=*/4, /*b=*/1, /*index=*/7, {0, 1, 1, 7},
                                                cpu_levels);
}
