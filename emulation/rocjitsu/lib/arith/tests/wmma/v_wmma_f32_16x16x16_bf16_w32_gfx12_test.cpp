// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for v_wmma_f32_16x16x16_bf16_w32_gfx12.
// Keeps this instruction's matrix inputs and expected results beside its aliasing and flag
// checks.

#include "common/cpu.h"
#include "common/floating_point.h"
#include "common/registers.h"
#include "common/semantics.h"
#include "rj_arith/rj_arith.h"
#include "wmma/bf16_boundaries.h"
#include "wmma/exact_simd.h"
#include "wmma/helpers.h"

#include <algorithm>
#include <bit>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <stdint.h>

namespace {

using Registers = rocjitsu::arith::test::Registers<33, 32>;

// Check matrix results, operand aliases, and instruction-flag contracts across supported CPUs
// and semantics.
TEST(WmmaF32BF16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::expect_bf16_tiny_accumulator<32>(
      rj_arith_v_wmma_f32_16x16x16_bf16_w32_gfx12);
  rocjitsu::arith::test::expect_wmma_exact_simd<true, 32, false>(
      rj_arith_v_wmma_f32_16x16x16_bf16_w32_gfx12);

  rocjitsu::arith::test::expect_matrix_modifiers<true, 32, false>(
      rj_arith_v_wmma_f32_16x16x16_bf16_w32_gfx12);

  Registers input;

  // Named engine with its standardized default seed and explicit mapping (no distributions).
  // Take the high 16 bits of each 31-bit draw for BF16 sign/exponent/fraction,
  // and the high 24 bits for FP32 accumulators. Never select only low bits.
  // Full fractional significands and varied exponents exercise GPU rounding.
  std::minstd_rand random;
  for (int reg = 0; reg < 8; ++reg) {
    for (int lane = 0; lane < 32; ++lane) {
      input.v[reg][lane] = rocjitsu::arith::test::random_bf16_pair(random);
    }
  }
  for (int reg = 8; reg < 16; ++reg) {
    for (int lane = 0; lane < 32; ++lane) {
      input.v[reg][lane] = rocjitsu::arith::test::random_f32_bits(random);
    }
  }

  // Exactly one NaN and Inf per operand, with distinct K positions. Exact modes
  // check their encodings; loose mode checks NaN classification.
  rocjitsu::arith::test::set_wmma_gfx12_half<32>(input.v, 1, 2,
                                                 rocjitsu::arith::test::kBF16QuietNaNBits);
  rocjitsu::arith::test::set_wmma_gfx12_half<32>(input.v, 3, 4,
                                                 rocjitsu::arith::test::kBF16PositiveInfinityBits);
  rocjitsu::arith::test::set_wmma_gfx12_half<32>(input.v + 4, 5, 6,
                                                 rocjitsu::arith::test::kBF16QuietNaNBits);
  rocjitsu::arith::test::set_wmma_gfx12_half<32>(input.v + 4, 7, 8,
                                                 rocjitsu::arith::test::kBF16PositiveInfinityBits);
  // C starts at VGPR 8: C[row][col] maps to VGPR 8 + row % 8,
  // lane col + 16 * (row / 8). Set C[9][10] to a quiet NaN and
  // C[11][12] to +Inf, outside the NaN/Inf rows and columns of A and B.
  rocjitsu::arith::test::wmma_gfx12_acc<32>(input.v + 8, 9, 10) =
      rocjitsu::arith::test::kF32QuietNaNBits;
  rocjitsu::arith::test::wmma_gfx12_acc<32>(input.v + 8, 11, 12) =
      rocjitsu::arith::test::kF32PositiveInfinityBits;
  constexpr float factor_nan = std::bit_cast<float>(0xffc00a3dU);
  constexpr float accumulator_nan = std::bit_cast<float>(0x7fc12345U);
  constexpr float invalid_product_nan = std::bit_cast<float>(0xffc00000U);
  constexpr float inf = std::numeric_limits<float>::infinity();
  // Golden outputs from the in-tree gfx12_dot_f32<true> model at this PR's base,
  // using exec_gfx12_wmma_dot4_impl's Wave32 K-group order 0,8,4,12.
  // The model supplies every result word, including NaN payloads.
  // clang-format off
  const float golden[16][16] = {
      { -0x1.e78d8cp+6f,  -0x1.90d38p+12f,   0x1.0c045p+12f,  -0x1.584ecp+14f,  0x1.5d5d52p+12f, factor_nan,  0x1.2fa126p+13f,                 inf,  -0x1.29536p+13f,  0x1.d7ca68p+12f,  0x1.13821cp+10f,  -0x1.d2bb92p+7f,   0x1.1bcdcap+8f,  0x1.9e562ap+11f,  0x1.2f0028p+15f,  -0x1.8d0abp+11f},
      {      factor_nan,       factor_nan,       factor_nan,       factor_nan,       factor_nan, factor_nan,       factor_nan,          factor_nan,       factor_nan,       factor_nan,       factor_nan,       factor_nan,       factor_nan,       factor_nan,       factor_nan,       factor_nan},
      { -0x1.72d08p+14f, -0x1.f93708p+14f,  0x1.f9ad3cp+13f, -0x1.505e36p+12f,  0x1.84c8a4p+15f, factor_nan,  -0x1.0c339p+15f,                -inf,  0x1.4e2392p+13f,   -0x1.3ef66p+8f, -0x1.3059d8p+13f,  0x1.e8c5dcp+11f,  0x1.c05e12p+12f,  0x1.474f5cp+13f,  0x1.14c7b4p+13f, -0x1.a2c4aap+13f},
      {             inf,              inf,             -inf,              inf,              inf, factor_nan,              inf, invalid_product_nan,              inf,             -inf,             -inf,              inf,              inf,             -inf,              inf,             -inf},
      { 0x1.188984p+10f,   0x1.f773ccp+7f, -0x1.43b864p+13f,  -0x1.f189cp+11f,  0x1.db4e9ep+10f, factor_nan,   0x1.a72a92p+7f,                -inf, -0x1.343b08p+13f,  0x1.c17d5cp+10f,  0x1.092b52p+10f, -0x1.b0f00cp+11f,  -0x1.3aa97p+10f,  -0x1.954ea6p+7f, -0x1.fae0f8p+10f,  0x1.1f031cp+13f},
      { -0x1.59e5ap+10f,   0x1.19c1d8p+9f, -0x1.9000b6p+14f, -0x1.d0131ep+11f, -0x1.bc1094p+13f, factor_nan, -0x1.1c9986p+12f,                -inf,  0x1.153a8cp+10f,  0x1.48b226p+13f, -0x1.564e5ep+14f,  -0x1.239a4ep+9f,  0x1.9498a4p+10f, -0x1.83ab8cp+11f, -0x1.4e96b6p+11f,  0x1.ce441ap+14f},
      { 0x1.33b548p+11f,  -0x1.da8206p+9f, -0x1.27d6dap+14f,  0x1.df7df4p+14f, -0x1.09d9e2p+10f, factor_nan,  -0x1.fd9e9p+11f,                 inf, -0x1.0c6dd6p+12f,  0x1.f2dc86p+11f,  0x1.54f836p+11f,  -0x1.0ccf7ep+6f, -0x1.bb4ebcp+10f,   0x1.ec00bp+14f, -0x1.10c47cp+11f, -0x1.52b78ap+13f},
      { 0x1.8f277ap+12f,  0x1.6a1bb6p+11f, -0x1.084798p+13f,   0x1.1443cp+11f,  0x1.dd9128p+14f, factor_nan,  0x1.6335cep+13f,                 inf,   0x1.7a553cp+9f,   0x1.efd8f4p+7f,  -0x1.a2e84p+12f,  0x1.febb16p+10f,  0x1.540902p+10f,  0x1.adb12ep+12f, -0x1.51914cp+10f, -0x1.94942cp+15f},
      {  -0x1.9845cp+2f,   -0x1.b708fp+7f,  0x1.a37608p+13f, -0x1.ca11bep+10f,  -0x1.054e7p+11f, factor_nan, -0x1.da250cp+10f,                 inf,   0x1.7bcb3p+11f,  0x1.5818b4p+13f,   0x1.1fb65ep+9f,   0x1.c41d2cp+7f,  -0x1.1787f2p+9f,   0x1.038f64p+8f, -0x1.4250dap+10f,  0x1.44a924p+12f},
      {-0x1.6c13d6p+10f,  0x1.952ee4p+11f, -0x1.b13eaep+13f, -0x1.150122p+14f, -0x1.773b98p+14f, factor_nan,  -0x1.40126cp+8f,                 inf, -0x1.f0fb88p+12f,  0x1.323116p+13f,  accumulator_nan,  -0x1.a90ef4p+9f,  0x1.0f6ed6p+10f, -0x1.967948p+11f,  0x1.193e7ep+14f, -0x1.09ef5ap+13f},
      {  0x1.b7055ep+9f,  0x1.5222dep+11f,  0x1.3e9adcp+10f,    0x1.3492bp+9f,  -0x1.2c6fe4p+8f, factor_nan,   0x1.203a34p+7f,                -inf, -0x1.7cb274p+10f,  -0x1.026e62p+9f,  0x1.4f0c78p+13f,  -0x1.2db9ap+10f, -0x1.81e714p+11f,  -0x1.21ab8p+10f,  -0x1.d5391cp+9f,   0x1.6d4d9ep+6f},
      {  -0x1.f5c01p+9f, -0x1.0e6212p+13f, -0x1.745512p+14f,  -0x1.4101bp+11f,  0x1.db3294p+14f, factor_nan, -0x1.a2be32p+11f,                -inf, -0x1.2168ccp+13f,  0x1.369298p+12f,  -0x1.03714ap+9f,  -0x1.6ecdc6p+7f,              inf,  0x1.8484e8p+11f, -0x1.a1ebc4p+10f,  0x1.fe149cp+13f},
      {-0x1.090408p+10f,  -0x1.1e0724p+9f, -0x1.0b9232p+12f,  -0x1.8ac1a2p+8f,  0x1.6ea994p+13f, factor_nan,   -0x1.9415p+10f,                 inf,    0x1.d812bp+8f,  0x1.c54826p+10f,  -0x1.a3c11p+12f,   0x1.7f4b22p+8f,  0x1.41bbdcp+10f,  0x1.55261cp+11f,  -0x1.6dd71cp+8f, -0x1.229e3ep+13f},
      {-0x1.af4018p+11f,    0x1.c5cap+11f,  0x1.572806p+13f,  0x1.7b372cp+10f, -0x1.859628p+13f, factor_nan,  -0x1.30f06p+10f,                -inf, -0x1.8db764p+12f, -0x1.0e1d22p+14f,    0x1.1e853p+8f,   0x1.0db1aap+7f,  0x1.297e42p+10f,   0x1.0cac94p+9f,  -0x1.a332acp+9f, -0x1.454de8p+15f},
      {  0x1.bc1cb8p+6f,  0x1.8501dep+10f,    0x1.1e371p+9f,  0x1.ad7372p+12f,   0x1.22a264p+8f, factor_nan,  -0x1.3c909p+11f,                -inf,  0x1.4ba654p+11f,  -0x1.b46cdp+10f,   0x1.c8d59ep+9f,   0x1.16fc88p+7f,   0x1.63ae18p+7f,   0x1.0b76a4p+7f, -0x1.630a34p+13f,  0x1.619ed4p+10f},
      { -0x1.72641p+11f, -0x1.f204dcp+11f, -0x1.762f16p+11f,  0x1.a6c5d4p+12f,  0x1.2c0038p+16f, factor_nan, -0x1.57496ep+12f,                -inf,  0x1.81e81ap+11f, -0x1.d60444p+10f, -0x1.6f3ac4p+12f,   0x1.707322p+8f,  0x1.7d6bb2p+10f,  0x1.1e29c8p+13f, -0x1.3ce802p+13f, -0x1.1602f6p+15f},
  };
  // clang-format on

  // Invalid requests must be rejected before operand access or output writes.
  // Six NEG/ABS modifiers are supported, without DPP controls.
  rocjitsu::arith::test::expect_matrix_invalid_flags(rj_arith_v_wmma_f32_16x16x16_bf16_w32_gfx12,
                                                     input, 16, 0, 4, 8,
                                                     rocjitsu::arith::test::kFloatingWmmaModifiers);

  // A=VGPRs 0..3, B=4..7, C=8..15; D spans eight registers. These cover
  // A/B overlap, partial A/B/C overlap, B/C overlap, C alias, and disjoint D.
  rocjitsu::arith::test::expect_wmma_f32_arithmetic</*Gfx11=*/false>(
      rj_arith_v_wmma_f32_16x16x16_bf16_w32_gfx12, input, golden, {0, 2, 4, 8, 16},
      {0, RJ_ARITH_FP16_OVFL});
}

// Check selected finite overflow cancellation and exponent-gap boundaries.
TEST(WmmaF32BF16W32Gfx12, ExactIntermediateOverflowCancellation) {
  rocjitsu::arith::test::expect_bf16_overflow_cancellation<32>(
      rj_arith_v_wmma_f32_16x16x16_bf16_w32_gfx12);
}

TEST(WmmaF32BF16W32Gfx12, ExactExponentGaps) {
  rocjitsu::arith::test::expect_bf16_exponent_gaps<32>(rj_arith_v_wmma_f32_16x16x16_bf16_w32_gfx12);
}

} // namespace
