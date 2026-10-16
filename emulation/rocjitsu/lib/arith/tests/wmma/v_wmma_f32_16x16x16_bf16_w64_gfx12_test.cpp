// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for v_wmma_f32_16x16x16_bf16_w64_gfx12.
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
TEST(WmmaF32BF16W64Gfx12, Arithmetic) {
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
      c[row][col] = rocjitsu::arith::test::random_f32_bits(random);
    }
  }
  // One NaN and one infinity per logical operand where the format has Inf.
  a[1][2] = rocjitsu::arith::test::kBF16QuietNaNBits;
  a[3][4] = rocjitsu::arith::test::kBF16PositiveInfinityBits;
  b[6][5] = rocjitsu::arith::test::kBF16QuietNaNBits;
  b[8][7] = rocjitsu::arith::test::kBF16PositiveInfinityBits;
  c[9][10] = rocjitsu::arith::test::kF32QuietNaNBits;
  c[11][12] = rocjitsu::arith::test::kF32PositiveInfinityBits;

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
  // Map each logical accumulator to its destination register and lane.
  for (int row = 0; row < 16; ++row) {
    for (int col = 0; col < 16; ++col) {
      input.v[4 + (row % 4)][col + 16 * (row / 8) + 32 * ((row / 4) % 2)] = c[row][col];
    }
  }

  // Offline goldens from rocjitsu ad2bb973ec4's original DOT models.
  // Reference computation uses logical matrices, independently of the VGPR
  // packing below, and does not call rj-arith. NaNs are compared by classification.
  constexpr float nan = std::numeric_limits<float>::quiet_NaN();
  constexpr float inf = std::numeric_limits<float>::infinity();
  // clang-format off
  const float golden[16][16] = {
      { 0x1.07d42ep+6f,  0x1.2c6c48p+4f,   0x1.6c255p+4f,  -0x1.a1ee3p+6f,  0x1.8225c4p+5f, nan, -0x1.219f8ap+4f,  inf, -0x1.ed7c1cp+5f,  0x1.fa4ca8p+6f,  0x1.745522p+5f,  0x1.aa3638p+6f,  -0x1.388fap+2f,   0x1.5508fp+6f,  0x1.236ebep+5f,  0x1.4ebebcp+6f},
      {            nan,             nan,             nan,             nan,             nan, nan,             nan,  nan,             nan,             nan,             nan,             nan,             nan,             nan,             nan,             nan},
      {-0x1.94cd1cp+4f,  0x1.446b56p+5f, -0x1.fbcafep+6f, -0x1.f01b48p+6f,  0x1.212e58p+3f, nan, -0x1.94ef88p+5f, -inf, -0x1.cec4eep+3f,  0x1.c15abcp+6f, -0x1.114eb4p+6f, -0x1.9f8208p+5f,  0x1.f85d8cp+4f,   0x1.d7f7ep+6f,   0x1.17b86p+6f,  -0x1.90efbp+6f},
      {           -inf,             inf,             inf,            -inf,             inf, nan,            -inf,  nan,            -inf,            -inf,            -inf,             inf,            -inf,             inf,            -inf,             inf},
      { 0x1.ac43cep+5f,  0x1.4128e6p+6f,  0x1.533514p+6f,  -0x1.8a776p+5f, -0x1.bac854p+6f, nan, -0x1.47574cp+5f, -inf,  0x1.919c38p+4f,  0x1.71e0ccp+6f, -0x1.9879f8p+5f, -0x1.39132cp+5f, -0x1.82498ap+6f,  0x1.559478p+5f, -0x1.c01654p+4f, -0x1.3161e4p+4f},
      { -0x1.f4231p+5f, -0x1.e6b3eep+4f,  0x1.e87aa2p+6f,  0x1.04c7fcp+6f, -0x1.2c1e74p+5f, nan,  0x1.febe98p+4f,  inf, -0x1.9ae43ep+6f,   0x1.c6f98p+1f,   0x1.4e5cdp+5f,  0x1.dc76d6p+6f, -0x1.48dfc2p+6f,  -0x1.3736fp+6f,   0x1.9993ap+6f, -0x1.5289fcp+5f},
      {-0x1.1226d2p+5f,  0x1.f8365ep+3f,  0x1.03b71ep+6f,  0x1.95d828p+6f, -0x1.99f272p+5f, nan,  0x1.d3e322p+3f, -inf,   0x1.9e873p+5f,  0x1.1b8bfcp+6f, -0x1.03e9fcp+5f,  0x1.532ce8p+6f, -0x1.f07504p+6f,  -0x1.afbedp+5f,  0x1.6a2202p+6f,   0x1.c3563p+6f},
      { -0x1.d9628p+5f, -0x1.78e348p+6f, -0x1.beb992p+5f,  0x1.b4958ap+5f, -0x1.95f49ap+5f, nan,  0x1.efcd0cp+6f,  inf,  -0x1.5a434p+6f,  -0x1.4b04dp+6f, -0x1.f19fdcp+4f, -0x1.89ba2cp+6f, -0x1.162642p+6f,  0x1.9951e8p+6f, -0x1.f200e8p+6f,  0x1.445852p+5f},
      {-0x1.a6cd54p+4f,  0x1.aa3d28p+6f,  0x1.2ef2b8p+6f,  0x1.bcf954p+6f, -0x1.7f17dcp+5f, nan,  0x1.fdad34p+3f, -inf,  -0x1.279c2p+3f,  -0x1.6fc52p+6f, -0x1.5567f8p+5f,  0x1.d46944p+5f, -0x1.a10bfcp+5f,  0x1.84e7a4p+6f,  -0x1.4ed48p+5f, -0x1.c34c46p+4f},
      { 0x1.60ac6ap+6f,  -0x1.dd6d4p+6f,  0x1.1a7fe8p+3f, -0x1.08ee62p+6f,  0x1.aaa1dcp+6f, nan,  0x1.28e7d2p+5f, -inf,  0x1.dc7698p+6f, -0x1.3a73ccp+3f,             nan,  0x1.b22f9cp+6f,  0x1.a4e5b2p+6f, -0x1.a82db4p+4f,  0x1.be7882p+6f,   0x1.7240dp+6f},
      {-0x1.ba41b6p+3f,  0x1.aeee4cp+4f,  -0x1.d1699p+6f,  -0x1.9573bp+5f, -0x1.d1fdc2p+6f, nan,  -0x1.dc802p+6f, -inf,  0x1.9604b6p+5f,   0x1.87d2fp+4f,  0x1.805a18p+6f,  -0x1.de97dp+2f,  -0x1.fcbbep+6f, -0x1.9c40c6p+5f,  0x1.17843ap+6f, -0x1.596374p+6f},
      { 0x1.8b5604p+6f, -0x1.db88bcp+3f,  0x1.edb982p+6f, -0x1.da251cp+5f,  0x1.80456ap+6f, nan, -0x1.08ea08p+6f,  inf,  0x1.2d88fap+5f,  0x1.deeae4p+5f, -0x1.6c19b4p+6f,  0x1.558c26p+6f,             inf, -0x1.dc4b7cp+6f, -0x1.6aee94p+6f, -0x1.3884b2p+6f},
      { -0x1.2d63ap+6f,  0x1.5a602cp+4f,  0x1.89cc1ap+6f, -0x1.4f38ecp+6f,  0x1.2b6688p+4f, nan,  -0x1.17bedp+4f, -inf, -0x1.fde234p+5f,  -0x1.ebf8dp+2f, -0x1.82d0c6p+5f,   0x1.1c98bp+5f,  0x1.b27884p+6f,  0x1.d5722ap+5f, -0x1.cafaf6p+6f,   0x1.81e9fp+3f},
      { -0x1.cf4f1p+6f, -0x1.8f5954p+6f, -0x1.95b554p+6f,  0x1.02d962p+5f,  0x1.e4ed04p+4f, nan,  0x1.a6dc44p+5f,  inf,  0x1.9e3738p+4f, -0x1.6a476ep+6f,  0x1.300b4cp+4f,  0x1.011c8cp+5f, -0x1.af9c98p+5f,  0x1.2ca594p+3f, -0x1.cfb0cap+6f, -0x1.19f932p+6f},
      { -0x1.2ba08p+4f, -0x1.f5f2e8p+3f, -0x1.cb07cep+5f, -0x1.cd6766p+5f,  0x1.94dcacp+5f, nan,  0x1.eaa1cep+4f,  inf, -0x1.a79d2ep+6f,  0x1.6f5e38p+3f, -0x1.7b149ep+6f,  0x1.7f6ae8p+5f,  0x1.690454p+6f,  -0x1.adc6ap+2f,   0x1.72d16p+6f,  0x1.91cd88p+6f},
      {-0x1.5e0778p+6f,  0x1.d50c0cp+5f, -0x1.6834b8p+2f,  0x1.d69a7cp+6f, -0x1.6e64d2p+6f, nan,  0x1.87cdecp+6f,  inf,     0x1.3d6p+6f,  0x1.7b0c76p+3f,  0x1.123b1ep+4f,  0x1.47a7ccp+5f,   0x1.1a3e6p+5f,  0x1.ad1258p+6f,   -0x1.2b71p+6f,  0x1.1724fcp+6f},
  };
  // clang-format on

  rocjitsu::arith::test::expect_matrix_invalid_flags(rj_arith_v_wmma_f32_16x16x16_bf16_w64_gfx12,
                                                     input, 9, 0, 2, 4, 0);

  // A occupies 2 registers, B 2, C/D 4. Reuse the same arithmetic
  // data for overlapping and disjoint destination ranges.
  const uint64_t modes[] = {0};
  const auto cpu_levels = rocjitsu::arith::test::supported_cpu_levels();
  for (uint64_t cpu : cpu_levels) {
    for (uint64_t semantics : rocjitsu::arith::test::semantics_modes()) {
      for (uint64_t mode : modes) {
        for (int aliasing_offset : {0, 1, 2, 3, 4, 9}) {
          SCOPED_TRACE(::testing::Message()
                       << cpu << '/' << semantics << '/' << mode << '/' << aliasing_offset);
          Registers r = input;
          ASSERT_EQ(rj_arith_v_wmma_f32_16x16x16_bf16_w64_gfx12(
                        cpu | semantics, mode, r.v + aliasing_offset, r.v, r.v + 2, r.v + 4),
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
