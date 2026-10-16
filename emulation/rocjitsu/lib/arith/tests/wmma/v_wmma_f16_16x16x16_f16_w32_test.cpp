// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for v_wmma_f16_16x16x16_f16_w32.
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
TEST(WmmaF16F16W32Gfx11, Arithmetic) {
  std::minstd_rand random;
  uint32_t a[16][16], b[16][16], c[16][16];
  for (int row = 0; row < 16; ++row) {
    for (int k = 0; k < 16; ++k) {
      a[row][k] = rocjitsu::arith::test::random_f16_bits(random);
    }
  }
  for (int k = 0; k < 16; ++k) {
    for (int col = 0; col < 16; ++col) {
      b[k][col] = rocjitsu::arith::test::random_f16_bits(random);
    }
  }
  for (int row = 0; row < 16; ++row) {
    for (int col = 0; col < 16; ++col) {
      c[row][col] = rocjitsu::arith::test::random_f16_bits(random);
    }
  }
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
  constexpr float factor_nan = std::bit_cast<float>(0xffc7a000U);
  constexpr float invalid_product_nan = std::bit_cast<float>(0xffc00000U);
  constexpr float accumulator_nan = std::bit_cast<float>(0x7fc02000U);
  constexpr float inf = std::numeric_limits<float>::infinity();
  // clang-format off
  const float golden[16][16] = {
      {-0x1.664p+1f,  -0x1.7bp-1f,  -0x1.06p-1f, -0x1.384p-1f,   0x1.38p+0f, factor_nan, -0x1.9d4p+2f,   inf,  -0x1.5bp+2f,  0x1.3bcp-4f,  -0x1.88p+0f, -0x1.4e4p+1f, -0x1.134p+1f,  0x1.438p+0f,  0x1.328p+2f,  -0x1.2fp-1f},
      {       factor_nan,        factor_nan,        factor_nan,        factor_nan,        factor_nan, factor_nan,        factor_nan, factor_nan,        factor_nan,        factor_nan,        factor_nan,        factor_nan,        factor_nan,        factor_nan,        factor_nan,        factor_nan},
      { 0x1.09cp+0f,  -0x1.98p+0f, -0x1.f4cp+1f,   0x1.0fp+2f,  0x1.30cp+1f, factor_nan,  -0x1.3ep+0f,  -inf,  0x1.0dcp-1f,  0x1.70cp-2f, -0x1.328p+0f,  0x1.3c8p+0f,  0x1.068p-1f,  0x1.014p-1f, -0x1.014p+2f, -0x1.90cp-1f},
      {        -inf,          inf,          inf,         -inf,          inf, factor_nan,         -inf, invalid_product_nan,         -inf,         -inf,         -inf,          inf,         -inf,          inf,         -inf,          inf},
      {-0x1.4e8p+0f,  -0x1.33p+1f,  -0x1.a3p+0f,  -0x1.3fp+2f,  0x1.37cp+0f, factor_nan,  -0x1.9dp+0f,  -inf, -0x1.35cp-1f, -0x1.1ccp-1f,   0x1.fap+1f,   0x1.5ep-4f,   0x1.8ap+0f,  0x1.7a4p-4f, -0x1.f2cp-6f,  0x1.3dcp+0f},
      { 0x1.878p+1f, -0x1.bb8p+0f,  0x1.928p+2f,  0x1.dfcp+1f, -0x1.b6cp+1f, factor_nan,   0x1.72p+0f,   inf,  0x1.d0cp+1f,  0x1.b38p+1f, -0x1.3bcp+1f, -0x1.e64p+0f, -0x1.818p+1f, -0x1.1d4p+2f,   0x1.6ap-2f,  -0x1.e2p-2f},
      { 0x1.468p+0f,  0x1.a28p+0f, -0x1.f54p+1f,   0x1.a4p-1f,  0x1.394p+1f, factor_nan, -0x1.134p-1f,  -inf,   0x1.4fp-3f, -0x1.6f8p+1f, -0x1.134p+1f,  -0x1.fbp-4f, -0x1.15cp+1f, -0x1.f74p+0f, -0x1.0b8p-1f,   0x1.8ap-1f},
      { 0x1.1d4p-2f,   0x1.adp+1f,  0x1.44cp-1f,  0x1.308p-2f,  0x1.634p+1f, factor_nan,   0x1.29p+2f,   inf,  0x1.b34p+0f,   0x1.02p+1f, -0x1.e78p-4f,  0x1.5a4p-1f,  0x1.524p+2f,  0x1.538p+1f, -0x1.00cp+1f,  0x1.a58p+0f},
      { 0x1.a78p+0f,  0x1.328p+1f,  0x1.f1cp-2f,  0x1.8c8p-3f, -0x1.98cp+0f, factor_nan, -0x1.d7cp-1f,  -inf,  0x1.0f8p-1f,  0x1.788p-2f, -0x1.a34p+0f, -0x1.03cp+0f,  -0x1.41p+2f, -0x1.64cp+1f,  0x1.138p+1f,   0x1.09p+1f},
      {-0x1.d9cp+2f, -0x1.278p+1f,  -0x1.81p+2f,  0x1.0f8p-1f,  0x1.714p+1f, factor_nan, -0x1.cb4p+2f,  -inf,  -0x1.6fp+1f,  0x1.dc4p-2f,        accumulator_nan, -0x1.8a8p-2f,  -0x1.b8p+2f,  0x1.23cp-3f,   0x1.33p+2f, -0x1.ebcp+2f},
      {-0x1.7b4p+2f,  0x1.8c4p+0f, -0x1.3f8p+1f, -0x1.43cp+0f,  -0x1.5cp-1f, factor_nan, -0x1.2ecp+1f,  -inf,  0x1.3acp+0f, -0x1.5b8p+0f, -0x1.70cp+2f,  -0x1.1dp+1f, -0x1.694p-1f,  0x1.d0cp+1f, -0x1.808p+2f,  0x1.33cp-1f},
      {  0x1.f8p+0f, -0x1.ef4p+1f,   0x1.6ep+0f, -0x1.ca4p-2f,  0x1.f9cp-4f, factor_nan,  0x1.184p+1f,   inf, -0x1.f28p+0f, -0x1.ec4p-2f, -0x1.c94p+0f,  0x1.99cp+0f,          inf,  0x1.28cp+2f,  0x1.1f8p-2f, -0x1.24cp-1f},
      {-0x1.284p+1f,  0x1.a04p+1f,  0x1.6ecp+1f, -0x1.7e4p-2f, -0x1.60cp+0f, factor_nan, -0x1.adcp+1f,  -inf, -0x1.5ecp-3f,  0x1.004p-1f, -0x1.024p+1f,  0x1.adcp+0f, -0x1.a94p+1f,  -0x1.13p+1f, -0x1.47cp+1f,   0x1.04p+2f},
      {  0x1.c2p+2f, -0x1.f14p+0f,  -0x1.1ap-1f,   0x1.5ep+0f, -0x1.45cp+2f, factor_nan,   0x1.79p+1f,   inf,  0x1.ce4p+1f, -0x1.c04p+0f,  0x1.7ccp+0f, -0x1.138p+0f,  0x1.d68p-2f,  -0x1.0bp+1f,  0x1.f74p+1f,  0x1.35cp+0f},
      {-0x1.eacp+0f,  0x1.448p+3f, -0x1.374p+2f, -0x1.018p-1f,  0x1.3a8p-1f, factor_nan, -0x1.41cp+0f,   inf,   0x1.06p+1f, -0x1.aecp-2f,  0x1.898p+0f, -0x1.1d4p+2f, -0x1.568p+2f, -0x1.3c8p+2f,  0x1.828p+0f,  -0x1.05p+2f},
      { 0x1.1e8p+2f,  0x1.db8p+1f, -0x1.318p-2f, -0x1.9a8p-1f,  0x1.becp+0f, factor_nan, -0x1.858p-3f,   inf,  -0x1.15p-2f, -0x1.054p+1f, -0x1.5e8p+2f, -0x1.facp-1f,   0x1.a8p-2f,  0x1.938p+1f, -0x1.c84p+1f,   0x1.51p+0f},
  };
  // clang-format on

  rocjitsu::arith::test::expect_matrix_invalid_flags(rj_arith_v_wmma_f16_16x16x16_f16_w32, input,
                                                     25, 0, 8, 16, 63 | RJ_ARITH_WMMA_HIGH_C_D);

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
          ASSERT_EQ(rj_arith_v_wmma_f16_16x16x16_f16_w32(
                        cpu | semantics, mode, r.v + aliasing_offset, r.v, r.v + 8, r.v + 16),
                    RJ_ARITH_SUCCESS);
          for (int row = 0; row < 16; ++row) {
            for (int col = 0; col < 16; ++col) {
              uint32_t actual = r.v[aliasing_offset + (row / 2)][col + 16 * (row % 2)];
              uint32_t before = input.v[aliasing_offset + (row / 2)][col + 16 * (row % 2)];
              uint32_t preserved = mode & RJ_ARITH_WMMA_HIGH_C_D ? 0xffffU : 0xffff0000U;
              EXPECT_EQ(actual & preserved, before & preserved);
              float value = rocjitsu::arith::test::f16_value(
                  uint16_t(actual >> (((mode & RJ_ARITH_WMMA_HIGH_C_D) ? 16 : 0))));
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
          r.expect_preserved(input, aliasing_offset, 8);
        }
      }
    }
  }
  rocjitsu::arith::test::expect_matrix_modifiers<false, 32, true, true>(
      rj_arith_v_wmma_f16_16x16x16_f16_w32);
  rocjitsu::arith::test::expect_packed_wmma_capture<false, 32, true>(
      rj_arith_v_wmma_f16_16x16x16_f16_w32);
  rocjitsu::arith::test::expect_wmma_exact_simd<false, 32, true, true>(
      rj_arith_v_wmma_f16_16x16x16_f16_w32);
}
