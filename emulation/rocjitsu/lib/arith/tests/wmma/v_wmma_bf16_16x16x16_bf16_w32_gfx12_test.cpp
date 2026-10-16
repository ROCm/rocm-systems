// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for v_wmma_bf16_16x16x16_bf16_w32_gfx12.
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

using Registers = rocjitsu::arith::test::Registers<64, 32>;

} // namespace

// Check matrix results, operand aliases, and instruction-flag contracts across supported CPUs
// and semantics.
TEST(WmmaBF16BF16W32Gfx12, Arithmetic) {
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
      rocjitsu::arith::test::set_wmma_gfx12_half<32>(input.v, row, k, a[row][k]);
    }
  }
  for (int k = 0; k < 16; ++k) {
    for (int col = 0; col < 16; ++col) {
      rocjitsu::arith::test::set_wmma_gfx12_half<32>(input.v + 4, col, k, b[k][col]);
    }
  }
  // C/D layout: register and lane select a row/column; packed gfx12
  // stores adjacent rows in halves, while gfx11 OPSEL preserves the other half.
  for (int row = 0; row < 16; ++row) {
    for (int col = 0; col < 16; ++col) {
      auto &word = input.v[8 + ((row % 8) / 2)][col + 16 * (row / 8)];
      int shift = 16 * ((row % 8) % 2);
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
      {-0x1.6ep+2f, -0x1.b2p+2f, -0x1.34p+2f,  0x1.b2p+1f, -0x1.2cp-2f, nan, -0x1.18p+2f,  inf, -0x1.0ap+1f,  0x1.6ep+0f, -0x1.54p+1f, -0x1.58p+1f,  0x1.3ap-1f, -0x1.eep+0f,  0x1.6ep+2f, -0x1.9ep+0f},
      {        nan,         nan,         nan,         nan,         nan, nan,         nan,  nan,         nan,         nan,         nan,         nan,         nan,         nan,         nan,         nan},
      { 0x1.26p-1f, -0x1.26p+2f, -0x1.9ap+1f,  0x1.96p+0f,  -0x1.cp+1f, nan,  -0x1.7p+1f, -inf,  0x1.7ep+1f, -0x1.dap-1f, -0x1.2ap-1f, -0x1.16p+1f,  0x1.44p+0f,  0x1.fep-1f, -0x1.bap+1f,  0x1.8cp+0f},
      {       -inf,         inf,         inf,        -inf,         inf, nan,        -inf,  nan,        -inf,        -inf,        -inf,         inf,        -inf,         inf,        -inf,         inf},
      {-0x1.9cp+1f, -0x1.06p+2f, -0x1.06p+1f, -0x1.9ep+0f,  0x1.f4p-1f, nan,  0x1.46p-1f, -inf,  0x1.0cp+0f,  0x1.52p+1f,  0x1.4ep+1f,  0x1.bep-1f,   0x1.ap+1f,  0x1.78p-3f,  0x1.ecp+1f, -0x1.96p+0f},
      { 0x1.5cp+2f, -0x1.5ep+1f,  0x1.94p+1f, -0x1.62p+1f,    -0x1p+1f, nan, -0x1.7ap-3f,  inf,  0x1.3ep+2f,  -0x1.cp-1f, -0x1.4ap+0f,  0x1.ecp-3f, -0x1.22p-1f,  0x1.c6p+0f,  -0x1.2p+1f,  0x1.82p+0f},
      { 0x1.4cp+2f,  0x1.44p+2f, -0x1.06p+1f, -0x1.1cp+0f,  0x1.14p-1f, nan, -0x1.72p+2f, -inf, -0x1.c4p-1f, -0x1.fcp+2f, -0x1.76p+1f,  0x1.86p+0f, -0x1.0ep+1f,   0x1.fp-4f, -0x1.12p+0f, -0x1.c2p+0f},
      {-0x1.9ap+1f,  -0x1.7p-1f, -0x1.d6p+0f, -0x1.16p+1f,  0x1.3ap+0f, nan,  0x1.3cp+0f,  inf, -0x1.78p-1f,  0x1.2cp+2f,   0x1.3p+1f,  0x1.42p+0f,   0x1.fp+0f,  0x1.b4p+0f, -0x1.46p-1f,  0x1.48p+0f},
      {-0x1.64p-1f, -0x1.24p+1f,   0x1.2p-2f, -0x1.92p-2f, -0x1.0ep+1f, nan,  -0x1.9p+1f, -inf,  0x1.5ap-1f, -0x1.cap-2f,  0x1.2ep+0f, -0x1.9ep+1f, -0x1.24p+2f,  -0x1.ap+1f,  0x1.dap+1f,  0x1.b8p+0f},
      {-0x1.d2p+1f,  0x1.eep-2f, -0x1.2ep+2f, -0x1.42p+1f,  0x1.2ap+1f, nan,  -0x1.bp+2f, -inf, -0x1.1ep+2f, -0x1.02p+1f,         nan, -0x1.b6p-4f,  -0x1.cp+2f, -0x1.e4p-1f,  0x1.78p+2f, -0x1.fep+2f},
      {-0x1.12p+2f, -0x1.04p+1f, -0x1.26p-1f,  0x1.eap-1f, -0x1.fcp-1f, nan, -0x1.98p+1f, -inf,  0x1.a6p+0f,  -0x1.1p+2f, -0x1.9ap+0f,  -0x1.ep+0f, -0x1.a4p+0f,  0x1.46p-4f, -0x1.96p+1f,  0x1.26p+0f},
      { -0x1.7p+1f, -0x1.aap+1f, -0x1.42p+0f, -0x1.c4p+0f, -0x1.a2p-1f, nan,  0x1.a6p+1f,  inf,  0x1.24p+0f,  0x1.44p+0f,  0x1.dap-2f, -0x1.52p-2f,         inf,  0x1.48p+1f,   0x1.6p+1f, -0x1.32p+1f},
      {  0x1.9p+2f,  0x1.2cp+1f,  0x1.84p-2f,  0x1.fap+1f, -0x1.6ap+1f, nan,  0x1.3ap-3f, -inf,   0x1.2p+1f, -0x1.dcp+1f, -0x1.a2p-1f,  -0x1.4p-2f,  0x1.ecp-2f,   0x1.2p+1f, -0x1.74p+1f,  0x1.7ep+1f},
      {  0x1.ep+1f,  0x1.68p+2f,  0x1.1cp-4f, -0x1.c8p+1f, -0x1.e6p-1f, nan,  0x1.dap+0f,  inf,  0x1.08p+1f, -0x1.12p+0f,  0x1.dcp-6f,   0x1.fp+0f, -0x1.78p-2f, -0x1.0ep-2f, -0x1.0ap-2f, -0x1.68p-1f},
      { 0x1.a2p-1f,   0x1.2p+1f,  0x1.1ep+1f, -0x1.0ap+2f, -0x1.18p-4f, nan,  0x1.ccp+0f,  inf,   0x1.dp+0f,  0x1.e8p+1f,   0x1.6p+1f, -0x1.56p+2f, -0x1.fep+1f, -0x1.3ep+2f,   0x1.1p+3f,    -0x1p+3f},
      { 0x1.16p+1f, -0x1.5cp+1f,  0x1.66p-3f, -0x1.34p+0f, -0x1.f8p+0f, nan,  -0x1.8p+1f,  inf,  0x1.42p-3f, -0x1.32p+1f, -0x1.58p+1f,  0x1.44p-2f,  0x1.e4p+0f,  0x1.96p+1f, -0x1.5cp+1f,  0x1.94p+0f},
  };
  // clang-format on

  rocjitsu::arith::test::expect_matrix_invalid_flags(rj_arith_v_wmma_bf16_16x16x16_bf16_w32_gfx12,
                                                     input, 13, 0, 4, 8, 63);

  // A occupies 4 registers, B 4, C/D 4. Reuse the same arithmetic
  // data for overlapping and disjoint destination ranges.
  const uint64_t modes[] = {0};
  const auto cpu_levels = rocjitsu::arith::test::supported_cpu_levels();
  for (uint64_t cpu : cpu_levels) {
    for (uint64_t semantics : rocjitsu::arith::test::semantics_modes()) {
      for (uint64_t mode : modes) {
        for (int aliasing_offset : {0, 2, 4, 6, 8, 13}) {
          SCOPED_TRACE(::testing::Message()
                       << cpu << '/' << semantics << '/' << mode << '/' << aliasing_offset);
          Registers r = input;
          ASSERT_EQ(rj_arith_v_wmma_bf16_16x16x16_bf16_w32_gfx12(
                        cpu | semantics, mode, r.v + aliasing_offset, r.v, r.v + 4, r.v + 8),
                    RJ_ARITH_SUCCESS);
          for (int row = 0; row < 16; ++row) {
            for (int col = 0; col < 16; ++col) {
              uint32_t actual = r.v[aliasing_offset + ((row % 8) / 2)][col + 16 * (row / 8)];
              float value =
                  rocjitsu::arith::test::bf16_value(uint16_t(actual >> (16 * ((row % 8) % 2))));
              rocjitsu::arith::test::expect_f32(rocjitsu::arith::test::bit_cast<uint32_t>(value),
                                                golden[row][col], semantics,
                                                /*atol=*/2e-4, /*rtol=*/4e-3);
            }
          }
          r.expect_preserved(input, aliasing_offset, 4);
        }
      }
    }
  }
  rocjitsu::arith::test::expect_wmma_exact_simd<true, 32, false, true>(
      rj_arith_v_wmma_bf16_16x16x16_bf16_w32_gfx12);
}
