// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for v_wmma_f32_16x16x16_f16_w64.
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
TEST(WmmaF32F16W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::expect_matrix_modifiers<false, 64, true>(
      rj_arith_v_wmma_f32_16x16x16_f16_w64);

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
      c[row][col] = rocjitsu::arith::test::random_f32_bits(random);
    }
  }
  // One NaN and one infinity per logical operand where the format has Inf.
  a[1][2] = rocjitsu::arith::test::kF16QuietNaNBits;
  a[3][4] = rocjitsu::arith::test::kF16PositiveInfinityBits;
  b[6][5] = rocjitsu::arith::test::kF16QuietNaNBits;
  b[8][7] = rocjitsu::arith::test::kF16PositiveInfinityBits;
  c[9][10] = rocjitsu::arith::test::kF32SignalingNaNBits;
  c[11][12] = rocjitsu::arith::test::kF32PositiveInfinityBits;

  Registers input;
  for (int row = 0; row < 16; ++row) {
    for (int k = 0; k < 16; ++k) {
      rocjitsu::arith::test::set_wmma_gfx11<16, 64>(input.v, row, k, a[row][k]);
    }
  }
  for (int k = 0; k < 16; ++k) {
    for (int col = 0; col < 16; ++col) {
      rocjitsu::arith::test::set_wmma_gfx11<16, 64>(input.v + 8, col, k, b[k][col]);
    }
  }
  // Map each logical accumulator to its destination register and lane.
  for (int row = 0; row < 16; ++row) {
    for (int col = 0; col < 16; ++col) {
      rocjitsu::arith::test::wmma_gfx11_acc<64>(input.v + 16, row, col) = c[row][col];
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
      {  0x1.12f76p+6f,  0x1.79c654p+4f,  0x1.b6e44ep+4f, -0x1.ac583ap+6f,  0x1.8d8b08p+5f, factor_nan, -0x1.30491cp+4f,                 inf, -0x1.00d7d2p+6f,   0x1.f824bp+6f,   0x1.812ebp+5f,  0x1.adbf72p+6f, -0x1.f302aap+2f,  0x1.61f56cp+6f,  0x1.265f64p+5f,   0x1.525fcp+6f},
      {     factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan, factor_nan,      factor_nan,          factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan},
      {-0x1.789e1cp+4f,  0x1.58dd12p+5f, -0x1.fa080ep+6f, -0x1.e50bdcp+6f,  0x1.fa6552p+3f, factor_nan, -0x1.86e984p+5f,                -inf, -0x1.20c028p+4f,  0x1.c5720ap+6f, -0x1.174462p+6f, -0x1.8f1e54p+5f,  0x1.0109c6p+5f,  0x1.d1a93ep+6f,  0x1.162d88p+6f,  -0x1.97c5dp+6f},
      {           -inf,             inf,             inf,            -inf,             inf, factor_nan,            -inf, invalid_product_nan,            -inf,            -inf,            -inf,             inf,            -inf,             inf,            -inf,             inf},
      { 0x1.be5404p+5f,  0x1.453486p+6f,   0x1.5572ep+6f,  -0x1.9bc0fp+5f, -0x1.b927d4p+6f, factor_nan, -0x1.595d8cp+5f,                -inf,   0x1.72268p+4f,  0x1.5f7f64p+6f, -0x1.922c08p+5f, -0x1.4245c8p+5f,  -0x1.884e6p+6f,   0x1.5979ap+5f, -0x1.fd3d8ep+4f, -0x1.15a564p+4f},
      {-0x1.01d044p+6f,  -0x1.d5602p+4f,  0x1.f55f88p+6f,  0x1.1cda68p+6f, -0x1.3cb644p+5f, factor_nan,   0x1.14963p+5f,                 inf, -0x1.a07e88p+6f,  0x1.fa5b64p+2f,    0x1.48c4p+5f,  0x1.d466dap+6f, -0x1.57ec44p+6f, -0x1.4c1978p+6f,  0x1.a56f0ap+6f, -0x1.630b34p+5f},
      {-0x1.3032cap+5f,  0x1.9b2c1ap+3f,  0x1.f7ac7cp+5f,  0x1.9e4356p+6f, -0x1.89a3a2p+5f, factor_nan,  0x1.3311ccp+4f,                -inf,  0x1.a75558p+5f,  0x1.30a4f4p+6f, -0x1.f6f0cep+4f,  0x1.4bb43ep+6f, -0x1.f16cfcp+6f, -0x1.c565f4p+5f,    0x1.6daap+6f,   0x1.cd676p+6f},
      {-0x1.c02e84p+5f, -0x1.662178p+6f, -0x1.afc2e8p+5f,  0x1.c2edcap+5f, -0x1.85e1aap+5f, factor_nan,  0x1.ffa19cp+6f,                 inf, -0x1.52dab4p+6f, -0x1.556d26p+6f, -0x1.1521dap+5f, -0x1.8f805cp+6f, -0x1.07debcp+6f,   0x1.99b96p+6f, -0x1.f1fd5cp+6f,  0x1.474704p+5f},
      {-0x1.82a374p+4f,  0x1.bbe6dcp+6f,  0x1.315674p+6f,   0x1.be32bp+6f, -0x1.7085bap+5f, factor_nan,  0x1.1ffa86p+4f,                -inf, -0x1.323a82p+3f, -0x1.69f49ep+6f, -0x1.6c794ap+5f,  0x1.eb75c2p+5f, -0x1.9f1428p+5f,  0x1.8727d4p+6f, -0x1.5a027cp+5f, -0x1.bd832ep+4f},
      { 0x1.52b958p+6f, -0x1.e4b8f4p+6f,  0x1.09211ap+3f, -0x1.f823a8p+5f,  0x1.ae2fc8p+6f, factor_nan,   0x1.1c478p+5f,                -inf,  0x1.e460d8p+6f, -0x1.e8ddaap+2f, accumulator_nan,  0x1.b496c8p+6f,  0x1.a96f44p+6f,  -0x1.8c556p+4f,  0x1.b4c326p+6f,  0x1.73c1ccp+6f},
      { -0x1.dd43cp+3f,  0x1.e2daf8p+4f, -0x1.dc96eap+6f, -0x1.addd84p+5f, -0x1.d2bd68p+6f, factor_nan, -0x1.dca6cep+6f,                -inf,  0x1.90cefep+5f,  0x1.b301ecp+4f,  0x1.6dfc24p+6f, -0x1.08afb6p+3f, -0x1.f7d632p+6f, -0x1.812e24p+5f,  0x1.08a266p+6f, -0x1.5b4c64p+6f},
      { 0x1.9da704p+6f, -0x1.dd49cap+3f,  0x1.f7c4e2p+6f,  -0x1.cb9bdp+5f,  0x1.86a69ap+6f, factor_nan, -0x1.0f305cp+6f,                 inf,  0x1.152adcp+5f,  0x1.d568c2p+5f, -0x1.72806cp+6f,  0x1.5707c4p+6f,             inf, -0x1.d43ff8p+6f, -0x1.726238p+6f, -0x1.313708p+6f},
      {-0x1.551b68p+6f,  0x1.651ce2p+4f,  0x1.913c3cp+6f, -0x1.5d3138p+6f,  0x1.3f9534p+4f, factor_nan,  -0x1.59f59p+4f,                -inf, -0x1.0e3082p+6f, -0x1.152d72p+2f, -0x1.852c3ep+5f,  0x1.28cbaep+5f,   0x1.a6badp+6f,  0x1.b4f442p+5f, -0x1.c91486p+6f,  0x1.ab67ccp+3f},
      {-0x1.c30e2ap+6f, -0x1.ac3348p+6f, -0x1.994c3ep+6f,   0x1.1c47cp+5f,  0x1.b76826p+4f, factor_nan,   0x1.ae8e9p+5f,                 inf,   0x1.b3747p+4f, -0x1.6b6d0ap+6f,  0x1.45977cp+4f,  0x1.e79b6ep+4f,  -0x1.ac73dp+5f,   0x1.e87eep+2f, -0x1.c23c68p+6f, -0x1.0fd91ap+6f},
      {-0x1.62e1f2p+4f, -0x1.257738p+3f, -0x1.019376p+6f, -0x1.b1e588p+5f,  0x1.92ce44p+5f, factor_nan,   0x1.c7613p+4f,                 inf,  -0x1.a6bbep+6f,   0x1.c5f88p+2f, -0x1.850cb2p+6f,  0x1.7c76cep+5f,  0x1.693254p+6f,  -0x1.01142p+3f,  0x1.572684p+6f,  0x1.a18228p+6f},
      { -0x1.5131ep+6f,   0x1.076b9p+6f, -0x1.aac7ccp+2f,   0x1.d8bdep+6f, -0x1.5fb0f6p+6f, factor_nan,  0x1.8ed4dap+6f,                 inf,  0x1.40dfa8p+6f,  0x1.95dc1cp+3f,  0x1.bd76c6p+3f,  0x1.3fca7ep+5f,  0x1.0be5c2p+5f,  0x1.ab9718p+6f, -0x1.3039cep+6f,  0x1.16832ap+6f},
  };
  // clang-format on

  rocjitsu::arith::test::expect_matrix_invalid_flags(rj_arith_v_wmma_f32_16x16x16_f16_w64, input,
                                                     21, 0, 8, 16, 63);

  // A occupies 8 registers, B 8, C/D 4. Reuse the same arithmetic
  // data for overlapping and disjoint destination ranges.
  rocjitsu::arith::test::expect_wmma_f32_arithmetic</*Gfx11=*/true>(
      rj_arith_v_wmma_f32_16x16x16_f16_w64, input, golden, {0, 4, 8, 12, 16, 21});
}
