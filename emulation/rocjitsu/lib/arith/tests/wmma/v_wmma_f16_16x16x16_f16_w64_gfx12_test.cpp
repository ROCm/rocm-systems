// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for v_wmma_f16_16x16x16_f16_w64_gfx12.
// Keeps this instruction's matrix inputs and expected results beside its aliasing and flag
// checks.

#include "common/cpu.h"
#include "common/floating_point.h"
#include "common/registers.h"
#include "common/semantics.h"
#include "rj_arith/rj_arith.h"
#include "wmma/exact_simd.h"
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
TEST(WmmaF16F16W64Gfx12, Arithmetic) {
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
      rocjitsu::arith::test::set_wmma_gfx12_half<64>(input.v, row, k, a[row][k]);
    }
  }
  for (int k = 0; k < 16; ++k) {
    for (int col = 0; col < 16; ++col) {
      rocjitsu::arith::test::set_wmma_gfx12_half<64>(input.v + 2, col, k, b[k][col]);
    }
  }
  // C/D layout: register and lane select a row/column; packed gfx12
  // stores adjacent rows in halves, while gfx11 OPSEL preserves the other half.
  for (int row = 0; row < 16; ++row) {
    for (int col = 0; col < 16; ++col) {
      auto &word = input.v[4 + ((row % 4) / 2)][col + 16 * (row / 8) + 32 * ((row / 4) % 2)];
      int shift = 16 * ((row % 4) % 2);
      word = (word & ~(0xffffU << shift)) | (c[row][col] << shift);
    }
  }

  // Offline goldens from rocjitsu ad2bb973ec4's original DOT models.
  // Reference computation uses logical matrices, independently of the VGPR
  // packing below, and does not call rj-arith. NaNs are compared by classification.
  constexpr float nan = std::numeric_limits<float>::quiet_NaN();
  constexpr float inf = std::numeric_limits<float>::infinity();
  // clang-format off
  const float golden[16][16] = {
      {-0x1.664p+1f, -0x1.7b4p-1f,  -0x1.06p-1f,  -0x1.38p-1f,   0x1.38p+0f, nan, -0x1.9d4p+2f,  inf,  -0x1.5bp+2f,  0x1.3bcp-4f,  -0x1.88p+0f, -0x1.4ecp+1f, -0x1.134p+1f,  0x1.438p+0f,  0x1.328p+2f,  -0x1.2fp-1f},
      {         nan,          nan,          nan,          nan,          nan, nan,          nan,  nan,          nan,          nan,          nan,          nan,          nan,          nan,          nan,          nan},
      { 0x1.09cp+0f, -0x1.97cp+0f, -0x1.f4cp+1f,   0x1.0fp+2f,   0x1.31p+1f, nan,  -0x1.3ep+0f, -inf,   0x1.0ep-1f,  0x1.70cp-2f, -0x1.32cp+0f,  0x1.3c4p+0f,  0x1.064p-1f,   0x1.01p-1f, -0x1.014p+2f, -0x1.90cp-1f},
      {        -inf,          inf,          inf,         -inf,          inf, nan,         -inf,  nan,         -inf,         -inf,         -inf,          inf,         -inf,          inf,         -inf,          inf},
      { -0x1.4ep+0f,  -0x1.33p+1f,  -0x1.a3p+0f,  -0x1.3fp+2f,   0x1.38p+0f, nan,  -0x1.9dp+0f, -inf, -0x1.358p-1f, -0x1.1c8p-1f,   0x1.fap+1f,   0x1.5fp-4f,   0x1.8ap+0f,  0x1.7a4p-4f, -0x1.eacp-6f,  0x1.3dcp+0f},
      { 0x1.878p+1f, -0x1.bb8p+0f,  0x1.928p+2f,  0x1.e08p+1f, -0x1.b6cp+1f, nan,   0x1.72p+0f,  inf,  0x1.d0cp+1f,  0x1.b3cp+1f, -0x1.3bcp+1f,  -0x1.e6p+0f, -0x1.81cp+1f, -0x1.1d4p+2f,  0x1.68cp-2f, -0x1.e28p-2f},
      { 0x1.468p+0f,  0x1.a2cp+0f,  -0x1.f5p+1f,  0x1.a3cp-1f,  0x1.394p+1f, nan, -0x1.12cp-1f, -inf,    0x1.5p-3f, -0x1.6f4p+1f, -0x1.134p+1f,  -0x1.fdp-4f, -0x1.15cp+1f,  -0x1.f7p+0f, -0x1.0b8p-1f,  0x1.89cp-1f},
      {  0x1.1dp-2f,  0x1.ad4p+1f,  0x1.444p-1f,  0x1.308p-2f,  0x1.634p+1f, nan,   0x1.29p+2f,  inf,   0x1.b3p+0f,   0x1.02p+1f, -0x1.e48p-4f,   0x1.5ap-1f,  0x1.524p+2f,  0x1.53cp+1f, -0x1.008p+1f,  0x1.a54p+0f},
      { 0x1.a78p+0f,  0x1.324p+1f,   0x1.f2p-2f,  0x1.8c8p-3f,  -0x1.99p+0f, nan, -0x1.d7cp-1f, -inf,  0x1.0fcp-1f,  0x1.78cp-2f,  -0x1.a3p+0f, -0x1.03cp+0f, -0x1.418p+2f, -0x1.64cp+1f,  0x1.138p+1f,   0x1.09p+1f},
      {-0x1.d98p+2f, -0x1.27cp+1f, -0x1.80cp+2f,  0x1.108p-1f,  0x1.714p+1f, nan, -0x1.cb4p+2f, -inf, -0x1.6ecp+1f,  0x1.dc8p-2f,          nan,  -0x1.88p-2f,  -0x1.b8p+2f,  0x1.238p-3f,   0x1.33p+2f,  -0x1.ecp+2f},
      { -0x1.7bp+2f,  0x1.8ccp+0f, -0x1.3f8p+1f, -0x1.43cp+0f,  -0x1.5bp-1f, nan, -0x1.2ecp+1f, -inf,  0x1.3acp+0f, -0x1.5b4p+0f, -0x1.70cp+2f,  -0x1.1dp+1f, -0x1.6acp-1f,  0x1.d08p+1f, -0x1.80cp+2f,  0x1.33cp-1f},
      {  0x1.f8p+0f, -0x1.ef4p+1f,  0x1.6dcp+0f, -0x1.ca4p-2f,  0x1.f58p-4f, nan,  0x1.188p+1f,  inf,  -0x1.f3p+0f, -0x1.ec4p-2f, -0x1.c94p+0f,  0x1.99cp+0f,          inf,  0x1.28cp+2f,  0x1.1ecp-2f, -0x1.24cp-1f},
      { -0x1.28p+1f,  0x1.a04p+1f,   0x1.6fp+1f, -0x1.7e4p-2f, -0x1.60cp+0f, nan,  -0x1.aep+1f, -inf, -0x1.5ecp-3f,  0x1.014p-1f,  -0x1.02p+1f,  0x1.adcp+0f, -0x1.a94p+1f,  -0x1.13p+1f, -0x1.478p+1f,   0x1.04p+2f},
      { 0x1.c1cp+2f, -0x1.f14p+0f, -0x1.19cp-1f,  0x1.5e4p+0f,  -0x1.46p+2f, nan,   0x1.79p+1f,  inf,   0x1.cep+1f, -0x1.c08p+0f,  0x1.7d8p+0f, -0x1.13cp+0f,  0x1.d58p-2f, -0x1.0acp+1f,  0x1.f7cp+1f,  0x1.35cp+0f},
      { -0x1.ebp+0f,  0x1.444p+3f, -0x1.374p+2f, -0x1.024p-1f,  0x1.3a8p-1f, nan,  -0x1.42p+0f,  inf,   0x1.06p+1f, -0x1.af4p-2f,  0x1.894p+0f, -0x1.1d4p+2f, -0x1.56cp+2f, -0x1.3c8p+2f,  0x1.82cp+0f, -0x1.04cp+2f},
      { 0x1.1ecp+2f,   0x1.dbp+1f, -0x1.314p-2f, -0x1.9acp-1f,  0x1.bdcp+0f, nan, -0x1.874p-3f,  inf, -0x1.154p-2f, -0x1.058p+1f, -0x1.5e4p+2f, -0x1.facp-1f,   0x1.a7p-2f,  0x1.93cp+1f,  -0x1.c8p+1f,  0x1.50cp+0f},
  };
  // clang-format on

  rocjitsu::arith::test::expect_matrix_invalid_flags(rj_arith_v_wmma_f16_16x16x16_f16_w64_gfx12,
                                                     input, 7, 0, 2, 4, 63);

  // A occupies 2 registers, B 2, C/D 2. Reuse the same arithmetic
  // data for overlapping and disjoint destination ranges.
  const uint64_t modes[] = {0};
  const auto cpu_levels = rocjitsu::arith::test::supported_cpu_levels();
  for (uint64_t cpu : cpu_levels) {
    for (uint64_t semantics : rocjitsu::arith::test::semantics_modes()) {
      for (uint64_t mode : modes) {
        for (int aliasing_offset : {0, 1, 2, 3, 4, 7}) {
          SCOPED_TRACE(::testing::Message()
                       << cpu << '/' << semantics << '/' << mode << '/' << aliasing_offset);
          Registers r = input;
          ASSERT_EQ(rj_arith_v_wmma_f16_16x16x16_f16_w64_gfx12(
                        cpu | semantics, mode, r.v + aliasing_offset, r.v, r.v + 2, r.v + 4),
                    RJ_ARITH_SUCCESS);
          for (int row = 0; row < 16; ++row) {
            for (int col = 0; col < 16; ++col) {
              uint32_t actual = r.v[aliasing_offset + ((row % 4) / 2)]
                                   [col + 16 * (row / 8) + 32 * ((row / 4) % 2)];
              float value =
                  rocjitsu::arith::test::f16_value(uint16_t(actual >> (16 * ((row % 4) % 2))));
              rocjitsu::arith::test::expect_f32(rocjitsu::arith::test::bit_cast<uint32_t>(value),
                                                golden[row][col], semantics,
                                                /*atol=*/1e-5, /*rtol=*/5e-4);
            }
          }
          r.expect_preserved(input, aliasing_offset, 2);
        }
      }
    }
  }
  rocjitsu::arith::test::expect_wmma_exact_simd<false, 64, false, true>(
      rj_arith_v_wmma_f16_16x16x16_f16_w64_gfx12);
}
