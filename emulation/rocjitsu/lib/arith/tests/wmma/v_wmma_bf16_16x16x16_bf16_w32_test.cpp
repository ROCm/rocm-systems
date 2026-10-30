// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for v_wmma_bf16_16x16x16_bf16_w32.
// Keeps this instruction's matrix inputs and expected results beside its aliasing and flag
// checks.

#include "common/cpu.h"
#include "common/floating_point.h"
#include "common/registers.h"
#include "common/semantics.h"
#include "rj_arith/rj_arith.h"
#include "wmma/exact_simd.h"
#include "wmma/helpers.h"
#include <bit>
#include <limits>

#include <gtest/gtest.h>
#include <random>
#include <stdint.h>

namespace {

using Registers = rocjitsu::arith::test::Registers<64, 32>;

} // namespace

// Check matrix results, operand aliases, and instruction-flag contracts across supported CPUs
// and semantics.
TEST(WmmaBF16BF16W32Gfx11, Arithmetic) {
  std::minstd_rand random;
  uint32_t a[16][16], b[16][16], c[16][16];
  for (int row = 0; row < 16; ++row) {
    for (int k = 0; k < 16; ++k) {
      a[row][k] = rocjitsu::arith::test::random_bf16_bits(random);
    }
  }
  for (int k = 0; k < 16; ++k) {
    for (int col = 0; col < 16; ++col) {
      b[k][col] = rocjitsu::arith::test::random_bf16_bits(random);
    }
  }
  for (int row = 0; row < 16; ++row) {
    for (int col = 0; col < 16; ++col) {
      c[row][col] = rocjitsu::arith::test::random_bf16_bits(random);
    }
  }
  // One NaN and one infinity per logical operand where the format has Inf.
  a[1][2] = rocjitsu::arith::test::kBF16QuietNaNBits;
  a[3][4] = rocjitsu::arith::test::kBF16PositiveInfinityBits;
  b[6][5] = rocjitsu::arith::test::kBF16QuietNaNBits;
  b[8][7] = rocjitsu::arith::test::kBF16PositiveInfinityBits;
  c[9][10] = rocjitsu::arith::test::kBF16QuietNaNBits;
  c[11][12] = rocjitsu::arith::test::kBF16PositiveInfinityBits;

  Registers input;
  for (int row = 0; row < 16; ++row) {
    for (int k = 0; k < 16; ++k) {
      rocjitsu::arith::test::set_wmma_gfx11<16, 32>(input.v, row, k, a[row][k]);
    }
  }
  for (int k = 0; k < 16; ++k) {
    for (int col = 0; col < 16; ++col) {
      rocjitsu::arith::test::set_wmma_gfx11<16, 32>(input.v + 8, col, k, b[k][col]);
    }
  }
  // C/D layout: register and lane select a row/column; packed gfx12
  // stores adjacent rows in halves, while gfx11 OPSEL preserves the other half.
  for (int row = 0; row < 16; ++row) {
    for (int col = 0; col < 16; ++col) {
      input.v[16 + (row / 2)][col + 16 * (row % 2)] = c[row][col] | (c[row][col] << 16);
    }
  }

  // Goldens from rocjitsu's gfx11_dot2_packed16 model, using logical matrices
  // independently of VGPR packing. Exact comparisons include NaN sign and payload.
  constexpr float factor_nan = std::bit_cast<float>(0xfffd0000U);
  constexpr float invalid_product_nan = std::bit_cast<float>(0xffc00000U);
  constexpr float accumulator_nan = std::bit_cast<float>(0x7fc10000U);
  constexpr float inf = std::numeric_limits<float>::infinity();
  // clang-format off
  const float golden[16][16] = {
      {-0x1.6ap+2f, -0x1.acp+2f, -0x1.32p+2f,   0x1.bp+1f, -0x1.1cp-2f, factor_nan, -0x1.14p+2f,   inf, -0x1.0cp+1f,  0x1.6ep+0f,  -0x1.5p+1f, -0x1.54p+1f,  0x1.38p-1f, -0x1.e8p+0f,  0x1.6ep+2f, -0x1.92p+0f},
      {      factor_nan,       factor_nan,       factor_nan,       factor_nan,       factor_nan, factor_nan,       factor_nan, factor_nan,       factor_nan,       factor_nan,       factor_nan,       factor_nan,       factor_nan,       factor_nan,       factor_nan,       factor_nan},
      { 0x1.16p-1f, -0x1.26p+2f, -0x1.9ap+1f,  0x1.94p+0f, -0x1.bcp+1f, factor_nan, -0x1.6cp+1f,  -inf,  0x1.7ap+1f, -0x1.e6p-1f, -0x1.2ep-1f, -0x1.14p+1f,  0x1.42p+0f,  0x1.fep-1f, -0x1.b6p+1f,  0x1.8ap+0f},
      {       -inf,         inf,         inf,        -inf,         inf, factor_nan,        -inf, invalid_product_nan,        -inf,        -inf,        -inf,         inf,        -inf,         inf,        -inf,         inf},
      {-0x1.9cp+1f, -0x1.06p+2f, -0x1.04p+1f, -0x1.a2p+0f,   0x1.fp-1f, factor_nan,  0x1.44p-1f,  -inf,  0x1.08p+0f,  0x1.54p+1f,  0x1.4ep+1f,  0x1.c2p-1f,  0x1.9cp+1f,  0x1.68p-3f,  0x1.e6p+1f, -0x1.94p+0f},
      { 0x1.58p+2f, -0x1.5cp+1f,  0x1.94p+1f, -0x1.62p+1f, -0x1.fcp+0f, factor_nan, -0x1.7cp-3f,   inf,  0x1.36p+2f, -0x1.c4p-1f, -0x1.4ap+0f,  0x1.fep-3f, -0x1.1ep-1f,  0x1.c8p+0f, -0x1.1cp+1f,  0x1.7ep+0f},
      { 0x1.4ap+2f,  0x1.44p+2f, -0x1.04p+1f, -0x1.22p+0f,  0x1.0cp-1f, factor_nan, -0x1.72p+2f,  -inf, -0x1.b4p-1f, -0x1.f4p+2f, -0x1.76p+1f,   0x1.8p+0f, -0x1.0ep+1f,  0x1.e6p-4f, -0x1.12p+0f, -0x1.c2p+0f},
      {-0x1.9ap+1f, -0x1.76p-1f, -0x1.d8p+0f, -0x1.16p+1f,  0x1.36p+0f, factor_nan,  0x1.3ap+0f,   inf, -0x1.78p-1f,  0x1.2ap+2f,  0x1.2ep+1f,  0x1.3ep+0f,   0x1.fp+0f,  0x1.b2p+0f, -0x1.48p-1f,  0x1.48p+0f},
      {-0x1.64p-1f,  -0x1.2p+1f,  0x1.2cp-2f, -0x1.92p-2f, -0x1.0cp+1f, factor_nan, -0x1.8cp+1f,  -inf,  0x1.46p-1f, -0x1.d2p-2f,  0x1.2cp+0f, -0x1.9cp+1f, -0x1.24p+2f,  -0x1.ap+1f,  0x1.d8p+1f,  0x1.b4p+0f},
      {-0x1.ccp+1f,  0x1.d8p-2f, -0x1.2cp+2f, -0x1.42p+1f,  0x1.2ap+1f, factor_nan, -0x1.aap+2f,  -inf, -0x1.1cp+2f, -0x1.02p+1f,       accumulator_nan, -0x1.7ep-4f, -0x1.bap+2f, -0x1.e4p-1f,  0x1.76p+2f, -0x1.fap+2f},
      {-0x1.12p+2f, -0x1.02p+1f,  -0x1.2p-1f,  0x1.eep-1f, -0x1.02p+0f, factor_nan, -0x1.96p+1f,  -inf,  0x1.a8p+0f,  -0x1.1p+2f,  -0x1.9p+0f,  -0x1.ep+0f,  -0x1.ap+0f,  0x1.04p-4f, -0x1.94p+1f,  0x1.2ap+0f},
      {-0x1.6ap+1f, -0x1.aap+1f,  -0x1.4p+0f, -0x1.c4p+0f, -0x1.9cp-1f, factor_nan,  0x1.a4p+1f,   inf,  0x1.22p+0f,  0x1.3cp+0f,  0x1.c8p-2f, -0x1.2cp-2f,         inf,  0x1.48p+1f,  0x1.5cp+1f, -0x1.32p+1f},
      { 0x1.8cp+2f,  0x1.2ap+1f,  0x1.92p-2f,  0x1.eep+1f, -0x1.66p+1f, factor_nan,  0x1.2cp-3f,  -inf,  0x1.1ap+1f, -0x1.dap+1f, -0x1.a8p-1f, -0x1.42p-2f,   0x1.dp-2f,  0x1.1ep+1f,  -0x1.7p+1f,  0x1.74p+1f},
      {  0x1.ep+1f,  0x1.66p+2f,   0x1.1p-4f, -0x1.c4p+1f,  -0x1.ep-1f, factor_nan,  0x1.d2p+0f,   inf,  0x1.06p+1f, -0x1.12p+0f,  0x1.acp-6f,  0x1.eep+0f, -0x1.6ep-2f, -0x1.06p-2f, -0x1.0cp-2f, -0x1.72p-1f},
      { 0x1.a2p-1f,   0x1.2p+1f,  0x1.1ep+1f, -0x1.0ap+2f, -0x1.cep-5f, factor_nan,  0x1.c6p+0f,   inf,  0x1.cap+0f,  0x1.e6p+1f,  0x1.5ep+1f, -0x1.54p+2f, -0x1.fep+1f, -0x1.3cp+2f,  0x1.0ep+3f,    -0x1p+3f},
      { 0x1.16p+1f, -0x1.5cp+1f,  0x1.52p-3f, -0x1.32p+0f, -0x1.f4p+0f, factor_nan, -0x1.7cp+1f,   inf,   0x1.8p-3f,  -0x1.3p+1f, -0x1.52p+1f,  0x1.48p-2f,  0x1.e4p+0f,  0x1.98p+1f, -0x1.5ep+1f,  0x1.94p+0f},
  };
  // clang-format on

  rocjitsu::arith::test::expect_matrix_invalid_flags(
      rj_arith_v_wmma_bf16_16x16x16_bf16_w32, input, 25, 0, 8, 16,
      rocjitsu::arith::test::kFloatingWmmaModifiers | RJ_ARITH_WMMA_HIGH_C_D);

  // A occupies 8 registers, B 8, C/D 8. Reuse the same arithmetic
  // data for overlapping and disjoint destination ranges.
  const uint64_t modes[] = {0, RJ_ARITH_WMMA_HIGH_C_D};
  const auto cpu_levels = rocjitsu::arith::test::supported_cpu_levels();
  for (uint64_t cpu : cpu_levels) {
    for (uint64_t semantics : rocjitsu::arith::test::semantics_modes()) {
      for (uint64_t mode : modes) {
        for (int aliasing_offset : {0, 4, 8, 12, 16, 25}) {
          SCOPED_TRACE(::testing::Message()
                       << cpu << '/' << semantics << '/' << mode << '/' << aliasing_offset);
          Registers r = input;
          ASSERT_EQ(rj_arith_v_wmma_bf16_16x16x16_bf16_w32(
                        cpu | semantics, mode, r.v + aliasing_offset, r.v, r.v + 8, r.v + 16),
                    RJ_ARITH_SUCCESS);
          for (int row = 0; row < 16; ++row) {
            for (int col = 0; col < 16; ++col) {
              uint32_t actual = r.v[aliasing_offset + (row / 2)][col + 16 * (row % 2)];
              uint32_t before = input.v[aliasing_offset + (row / 2)][col + 16 * (row % 2)];
              uint32_t preserved = mode & RJ_ARITH_WMMA_HIGH_C_D ? 0xffffU : 0xffff0000U;
              EXPECT_EQ(actual & preserved, before & preserved);
              float value = rocjitsu::arith::test::bf16_value(
                  uint16_t(actual >> (((mode & RJ_ARITH_WMMA_HIGH_C_D) ? 16 : 0))));
              if (semantics != RJ_ARITH_SEMANTICS_LOOSE) {
                EXPECT_EQ(std::bit_cast<uint32_t>(value),
                          std::bit_cast<uint32_t>(golden[row][col]));
              } else {
                rocjitsu::arith::test::expect_f32(std::bit_cast<uint32_t>(value), golden[row][col],
                                                  semantics,
                                                  /*atol=*/2e-4, /*rtol=*/4e-3);
              }
            }
          }
          r.expect_preserved(input, aliasing_offset, 8);
        }
      }
    }
  }
  rocjitsu::arith::test::expect_matrix_modifiers<true, 32, true, true>(
      rj_arith_v_wmma_bf16_16x16x16_bf16_w32);
  rocjitsu::arith::test::expect_wmma_fixed_controls<true, 32, true, true>(
      rj_arith_v_wmma_bf16_16x16x16_bf16_w32);
  rocjitsu::arith::test::expect_packed_wmma_capture<true, 32, true>(
      rj_arith_v_wmma_bf16_16x16x16_bf16_w32);
  rocjitsu::arith::test::expect_wmma_exact_simd<true, 32, true, true>(
      rj_arith_v_wmma_bf16_16x16x16_bf16_w32);
}
