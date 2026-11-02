// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for v_wmma_f32_16x16x16_bf16_w32.
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
TEST(WmmaF32BF16W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::expect_bf16_loose_subnormal<32, true>(
      rj_arith_v_wmma_f32_16x16x16_bf16_w32);
  rocjitsu::arith::test::expect_wmma_exact_simd<true, 32, true, false>(
      rj_arith_v_wmma_f32_16x16x16_bf16_w32);

  rocjitsu::arith::test::expect_matrix_modifiers<true, 32, true>(
      rj_arith_v_wmma_f32_16x16x16_bf16_w32);
  rocjitsu::arith::test::expect_wmma_fixed_controls<true, 32, true, false>(
      rj_arith_v_wmma_f32_16x16x16_bf16_w32);

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
  c[9][10] = rocjitsu::arith::test::kF32SignalingNaNBits;
  c[11][12] = rocjitsu::arith::test::kF32PositiveInfinityBits;

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
  // Map each logical accumulator to its destination register and lane.
  for (int row = 0; row < 16; ++row) {
    for (int col = 0; col < 16; ++col) {
      rocjitsu::arith::test::wmma_gfx11_acc<32>(input.v + 16, row, col) = c[row][col];
    }
  }

  // Golden words from the in-tree gfx11_dot2_f32 model, accumulated in
  // K order 0,2,...,14. No rj-arith reference computation.
  constexpr float factor_nan = std::bit_cast<float>(0xffc00a3dU);
  constexpr float accumulator_nan = std::bit_cast<float>(0x7fc12345U);
  constexpr float invalid_product_nan = std::bit_cast<float>(0xffc00000U);
  constexpr float inf = std::numeric_limits<float>::infinity();
  // clang-format off
  const float golden[16][16] = {
      { 0x1.07d42ep+6f,  0x1.2c6c48p+4f,   0x1.6c255p+4f, -0x1.a1ee2cp+6f,  0x1.8225c2p+5f, factor_nan, -0x1.219f8ap+4f,                 inf, -0x1.ed7c1cp+5f,  0x1.fa4ca6p+6f,  0x1.745522p+5f,  0x1.aa3638p+6f,  -0x1.388fap+2f,   0x1.5508fp+6f,  0x1.236ebep+5f,  0x1.4ebebap+6f},
      {     factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan, factor_nan,      factor_nan,          factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan},
      {-0x1.94cd1cp+4f,  0x1.446b54p+5f, -0x1.fbcafcp+6f, -0x1.f01b48p+6f,  0x1.212e58p+3f, factor_nan, -0x1.94ef88p+5f,                -inf, -0x1.cec4ecp+3f,  0x1.c15abcp+6f, -0x1.114eb4p+6f, -0x1.9f8208p+5f,  0x1.f85d8cp+4f,  0x1.d7f7dep+6f,   0x1.17b86p+6f,  -0x1.90efbp+6f},
      {           -inf,             inf,             inf,            -inf,             inf, factor_nan,            -inf, invalid_product_nan,            -inf,            -inf,            -inf,             inf,            -inf,             inf,            -inf,             inf},
      { 0x1.ac43cep+5f,  0x1.4128e4p+6f,  0x1.533514p+6f,  -0x1.8a776p+5f, -0x1.bac852p+6f, factor_nan, -0x1.47574cp+5f,                -inf,  0x1.919c38p+4f,  0x1.71e0ccp+6f, -0x1.9879f6p+5f, -0x1.39132cp+5f, -0x1.82498ap+6f,  0x1.559478p+5f, -0x1.c01654p+4f, -0x1.3161e4p+4f},
      { -0x1.f4231p+5f, -0x1.e6b3eep+4f,   0x1.e87aap+6f,  0x1.04c7fcp+6f, -0x1.2c1e72p+5f, factor_nan,  0x1.febe98p+4f,                 inf, -0x1.9ae43ap+6f,   0x1.c6f98p+1f,   0x1.4e5cdp+5f,  0x1.dc76d4p+6f, -0x1.48dfc2p+6f,  -0x1.3736fp+6f,  0x1.99939ep+6f, -0x1.5289fcp+5f},
      {-0x1.1226d2p+5f,  0x1.f8365ep+3f,  0x1.03b71cp+6f,  0x1.95d826p+6f,  -0x1.99f27p+5f, factor_nan,   0x1.d3e32p+3f,                -inf,  0x1.9e872cp+5f,  0x1.1b8bfap+6f, -0x1.03e9fap+5f,  0x1.532ce6p+6f, -0x1.f07502p+6f,  -0x1.afbedp+5f,  0x1.6a2202p+6f,   0x1.c3563p+6f},
      { -0x1.d9628p+5f, -0x1.78e348p+6f, -0x1.beb992p+5f,  0x1.b4958ap+5f, -0x1.95f49ap+5f, factor_nan,  0x1.efcd0cp+6f,                 inf, -0x1.5a433ep+6f,  -0x1.4b04dp+6f, -0x1.f19fdcp+4f, -0x1.89ba2ap+6f,  -0x1.16264p+6f,  0x1.9951e8p+6f, -0x1.f200e4p+6f,  0x1.445852p+5f},
      {-0x1.a6cd54p+4f,  0x1.aa3d26p+6f,  0x1.2ef2b8p+6f,  0x1.bcf954p+6f, -0x1.7f17dcp+5f, factor_nan,  0x1.fdad34p+3f,                -inf,  -0x1.279c2p+3f, -0x1.6fc51ep+6f, -0x1.5567f8p+5f,  0x1.d46944p+5f, -0x1.a10bfcp+5f,   0x1.84e7ap+6f, -0x1.4ed47ep+5f, -0x1.c34c46p+4f},
      { 0x1.60ac6ap+6f,  -0x1.dd6d4p+6f,  0x1.1a7fe8p+3f, -0x1.08ee62p+6f,  0x1.aaa1dap+6f, factor_nan,  0x1.28e7d2p+5f,                -inf,  0x1.dc7698p+6f, -0x1.3a73ccp+3f, accumulator_nan,  0x1.b22f9cp+6f,  0x1.a4e5aep+6f, -0x1.a82db4p+4f,  0x1.be7882p+6f,   0x1.7240dp+6f},
      {-0x1.ba41b6p+3f,  0x1.aeee4cp+4f,  -0x1.d1699p+6f, -0x1.9573aep+5f,  -0x1.d1fdcp+6f, factor_nan,  -0x1.dc802p+6f,                -inf,  0x1.9604b6p+5f,   0x1.87d2fp+4f,  0x1.805a16p+6f,  -0x1.de97dp+2f, -0x1.fcbbdap+6f, -0x1.9c40c4p+5f,  0x1.17843ap+6f, -0x1.596372p+6f},
      { 0x1.8b5602p+6f, -0x1.db88bcp+3f,  0x1.edb982p+6f, -0x1.da251ap+5f,  0x1.804568p+6f, factor_nan, -0x1.08ea06p+6f,                 inf,  0x1.2d88fap+5f,  0x1.deeae4p+5f, -0x1.6c19b4p+6f,  0x1.558c26p+6f,             inf, -0x1.dc4b7cp+6f, -0x1.6aee94p+6f,  -0x1.3884bp+6f},
      {-0x1.2d639cp+6f,  0x1.5a602cp+4f,  0x1.89cc1ap+6f, -0x1.4f38eap+6f,  0x1.2b6688p+4f, factor_nan, -0x1.17becep+4f,                -inf, -0x1.fde234p+5f,  -0x1.ebf8dp+2f, -0x1.82d0c6p+5f,  0x1.1c98aep+5f,   0x1.b2788p+6f,  0x1.d57228p+5f, -0x1.cafaf8p+6f,   0x1.81e9fp+3f},
      {-0x1.cf4f0ep+6f, -0x1.8f5952p+6f, -0x1.95b554p+6f,  0x1.02d962p+5f,  0x1.e4ed04p+4f, factor_nan,  0x1.a6dc44p+5f,                 inf,  0x1.9e3738p+4f, -0x1.6a476ep+6f,  0x1.300b4cp+4f,  0x1.011c8cp+5f, -0x1.af9c96p+5f,  0x1.2ca594p+3f, -0x1.cfb0cap+6f, -0x1.19f932p+6f},
      { -0x1.2ba08p+4f, -0x1.f5f2e8p+3f, -0x1.cb07cep+5f, -0x1.cd6766p+5f,  0x1.94dcaap+5f, factor_nan,  0x1.eaa1cep+4f,                 inf, -0x1.a79d2ep+6f,  0x1.6f5e38p+3f, -0x1.7b149ep+6f,  0x1.7f6ae6p+5f,  0x1.690456p+6f,  -0x1.adc6ap+2f,   0x1.72d16p+6f,  0x1.91cd88p+6f},
      {-0x1.5e0778p+6f,  0x1.d50c08p+5f, -0x1.6834b8p+2f,  0x1.d69a7ap+6f,  -0x1.6e64dp+6f, factor_nan,  0x1.87cde8p+6f,                 inf,  0x1.3d5ffep+6f,  0x1.7b0c76p+3f,  0x1.123b1ep+4f,  0x1.47a7cap+5f,   0x1.1a3e6p+5f,  0x1.ad1258p+6f,   -0x1.2b71p+6f,  0x1.1724fcp+6f},
  };
  // clang-format on

  rocjitsu::arith::test::expect_matrix_invalid_flags(rj_arith_v_wmma_f32_16x16x16_bf16_w32, input,
                                                     25, 0, 8, 16,
                                                     rocjitsu::arith::test::kFloatingWmmaModifiers);

  // A occupies 8 registers, B 8, C/D 8. Reuse the same arithmetic
  // data for overlapping and disjoint destination ranges.
  rocjitsu::arith::test::expect_wmma_f32_arithmetic</*Gfx11=*/true>(
      rj_arith_v_wmma_f32_16x16x16_bf16_w32, input, golden, {0, 4, 8, 12, 16, 25});
}

// BF16 factors, FP32 accumulators, and tiny results obey mandatory flushing.
TEST(WmmaF32BF16W32Gfx11, InputDenormal) {
  rocjitsu::arith::test::expect_gfx11_bf16_denormal<32>(rj_arith_v_wmma_f32_16x16x16_bf16_w32);
}
