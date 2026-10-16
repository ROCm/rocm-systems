// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for v_wmma_f32_16x16x16_f16_w32.
// Keeps this instruction's matrix inputs and expected results beside its aliasing and flag
// checks.

#include "common/cpu.h"
#include "common/floating_point.h"
#include "common/registers.h"
#include "common/semantics.h"
#include "rj_arith/rj_arith.h"
#include "wmma/helpers.h"

#include <bit>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <stdint.h>

namespace {

using Registers = rocjitsu::arith::test::Registers<33, 32>;

} // namespace

// Check matrix results, operand aliases, and instruction-flag contracts across supported CPUs
// and semantics.
TEST(WmmaF32F16W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::expect_matrix_modifiers<false, 32, true>(
      rj_arith_v_wmma_f32_16x16x16_f16_w32);

  Registers input;

  // Named engine with its standardized default seed and explicit mapping (no distributions).
  // Take the high 16 bits of each 31-bit draw for FP16 sign/exponent/fraction,
  // and the high 24 bits for FP32 accumulators. Never select only low bits.
  // Full fractional significands and varied exponents exercise GPU rounding.
  std::minstd_rand random;
  // One logical A/B matrix each, physically replicated in the two lane groups.
  for (int reg = 0; reg < 16; ++reg) {
    for (int lane = 0; lane < 16; ++lane) {
      uint32_t word = rocjitsu::arith::test::random_f16_pair(random);
      input.v[reg][lane] = input.v[reg][lane + 16] = word;
    }
  }
  for (int reg = 16; reg < 24; ++reg) {
    for (int lane = 0; lane < 32; ++lane) {
      input.v[reg][lane] = rocjitsu::arith::test::random_f32_bits(random);
    }
  }

  // Exactly one logical NaN and Inf per operand, with distinct K positions.
  // Exact modes check NaN words; loose mode checks their classification.
  rocjitsu::arith::test::set_wmma_gfx11<16, 32>(input.v, 1, 2,
                                                rocjitsu::arith::test::kF16QuietNaNBits);
  rocjitsu::arith::test::set_wmma_gfx11<16, 32>(input.v, 3, 4,
                                                rocjitsu::arith::test::kF16PositiveInfinityBits);
  rocjitsu::arith::test::set_wmma_gfx11<16, 32>(input.v + 8, 5, 6,
                                                rocjitsu::arith::test::kF16QuietNaNBits);
  rocjitsu::arith::test::set_wmma_gfx11<16, 32>(input.v + 8, 7, 8,
                                                rocjitsu::arith::test::kF16PositiveInfinityBits);
  // C starts at VGPR 16: reg=row/2, lane=col+16*(row%2).
  // These isolated C values lie outside A/B's exceptional rows and columns.
  rocjitsu::arith::test::wmma_gfx11_acc<32>(input.v + 16, 9, 10) =
      rocjitsu::arith::test::kF32SignalingNaNBits;
  rocjitsu::arith::test::wmma_gfx11_acc<32>(input.v + 16, 11, 12) =
      rocjitsu::arith::test::kF32PositiveInfinityBits;
  constexpr float factor_nan = std::bit_cast<float>(0xffc00a3dU);
  constexpr float accumulator_nan = std::bit_cast<float>(0x7fc12345U);
  constexpr float inf = std::numeric_limits<float>::infinity();
  // Golden words from the in-tree gfx11_dot2_f32 model, accumulated in
  // K order 0,2,...,14. No rj-arith reference computation.
  // clang-format off
  const float golden[16][16] = {
      {-0x1.afb00cp+20f,   0x1.0c3fdp+21f, -0x1.287e62p+21f,   0x1.fadb4p+25f, -0x1.3d5514p+21f, factor_nan, -0x1.8ecc84p+18f,       -inf,   0x1.b3cb6p+12f,  0x1.b44d6ap+18f,   0x1.fd1ccp+20f,  0x1.70f68ap+17f, -0x1.3947f2p+19f,   0x1.7a931p+26f,  0x1.306ddep+19f, -0x1.5ab2acp+27f},
      {      factor_nan,       factor_nan,       factor_nan,       factor_nan,       factor_nan, factor_nan,       factor_nan, factor_nan,       factor_nan,       factor_nan,       factor_nan,       factor_nan,       factor_nan,       factor_nan,       factor_nan,       factor_nan},
      { 0x1.e93658p+31f,  0x1.546a48p+30f,  0x1.aa25c6p+20f, -0x1.3f6668p+25f, -0x1.f52eecp+27f, factor_nan, -0x1.d3252ep+23f,       -inf,   0x1.d0fe8p+15f, -0x1.da8794p+23f, -0x1.127f76p+21f,  0x1.16298cp+29f,  0x1.965e18p+23f,  0x1.c15c6ep+29f,   0x1.2bb0ap+14f,  -0x1.3fe04p+27f},
      {            -inf,              inf,              inf,             -inf,             -inf, factor_nan,             -inf,       -inf,              inf,             -inf,             -inf,              inf,              inf,              inf,              inf,             -inf},
      {-0x1.261a4ep+26f, -0x1.c45768p+18f,   0x1.04853p+29f,  0x1.d88272p+18f, -0x1.5a1c28p+26f, factor_nan,  0x1.7c78c2p+26f,       -inf,  -0x1.90c2ep+25f,  0x1.c4f5b4p+25f, -0x1.5d1a74p+25f, -0x1.7c5138p+26f, -0x1.78d592p+19f,  -0x1.b38c9p+15f, -0x1.1bc4aap+27f,  0x1.09fcb2p+22f},
      { -0x1.345a6p+30f,  0x1.a39304p+15f,  0x1.de31ccp+19f, -0x1.048b22p+17f, -0x1.c43174p+18f, factor_nan, -0x1.012e5ep+21f,       -inf, -0x1.7e4c86p+18f,  0x1.2422c8p+19f, -0x1.bad858p+26f,  0x1.6aeeccp+22f,  0x1.26f744p+24f,   0x1.c0f31p+18f,  0x1.2bcf0cp+22f, -0x1.ecebbcp+16f},
      {-0x1.6657eap+27f,  -0x1.f99dbp+23f, -0x1.487222p+22f, -0x1.67836ep+21f,  0x1.8af1f6p+27f, factor_nan, -0x1.39b746p+16f,       -inf, -0x1.35bb62p+17f,  0x1.cf23e6p+20f,  0x1.64d3b2p+25f, -0x1.28295cp+16f, -0x1.4b50c6p+16f,  0x1.f091dcp+24f,  0x1.9d3ec8p+24f,  0x1.2a9d24p+24f},
      { 0x1.054b3cp+15f, -0x1.1886c4p+15f, -0x1.890ddcp+15f,    0x1.e7063p+1f, -0x1.a524e8p+12f, factor_nan, -0x1.411becp+26f,        inf,  -0x1.c93b5p+19f, -0x1.bd40fep+23f,  0x1.1cffdcp+27f,  0x1.077394p+31f,  -0x1.b85d7p+17f,   0x1.1fda6p+14f, -0x1.09b6c8p+12f, -0x1.ba5108p+25f},
      {-0x1.fbb0acp+22f,  -0x1.86584p+21f,  -0x1.7c763p+18f,  0x1.b4696ap+15f,  0x1.7facb6p+25f, factor_nan,  0x1.e3376cp+19f,       -inf, -0x1.006b5ap+14f,  0x1.17deb4p+16f,  0x1.349132p+20f,   0x1.3d9d8p+23f,  -0x1.80aaep+17f, -0x1.71b4a4p+26f,  0x1.2436cep+19f,  0x1.910df4p+27f},
      {-0x1.0a521ap+27f, -0x1.0020d8p+24f,  0x1.33c2b8p+21f, -0x1.6bce4ap+29f, -0x1.081264p+27f, factor_nan,   0x1.05237p+13f,       -inf,  0x1.99f728p+15f, -0x1.31bf9ep+20f,  accumulator_nan, -0x1.11d496p+23f, -0x1.bc1e54p+14f,  0x1.2732cep+30f,    0x1.bbb6p+23f,  0x1.31435ep+21f},
      { 0x1.8ac3e4p+31f,  0x1.12addcp+30f,  0x1.1891a4p+22f,   0x1.6da4ep+26f, -0x1.8e45fap+26f, factor_nan, -0x1.6b6828p+23f,       -inf, -0x1.fb66d8p+18f, -0x1.61fa16p+23f,  0x1.a07f58p+27f,  0x1.88992cp+17f,  0x1.397368p+23f, -0x1.8aca0ap+27f, -0x1.b97b86p+19f,  0x1.134504p+15f},
      {-0x1.8d108cp+25f,  0x1.4cce44p+13f,   0x1.2eef3p+26f,  0x1.4d1062p+16f,  0x1.145456p+20f, factor_nan,  0x1.4b7f3ep+19f,       -inf, -0x1.166fb2p+23f,  0x1.4ae696p+23f, -0x1.fb6a2cp+26f,  0x1.64123ep+23f,              inf,  0x1.5897bep+25f, -0x1.5d83f8p+23f, -0x1.750b9ep+21f},
      { 0x1.ea6e48p+30f,  0x1.83c06ep+28f,  0x1.b6371cp+30f, -0x1.3223aap+29f, -0x1.e3bf22p+30f, factor_nan,  0x1.36a0f8p+25f,       -inf,  0x1.187efcp+25f,  -0x1.fcdc3p+23f,  0x1.eda298p+28f, -0x1.0a9e48p+27f, -0x1.351c76p+29f,  0x1.06a88ep+30f, -0x1.a0cb6ep+26f, -0x1.02e62cp+18f},
      { 0x1.152c1ep+26f, -0x1.f137dep+17f,  0x1.9e083cp+29f,  0x1.52c042p+23f,  -0x1.b4b26p+29f, factor_nan,  0x1.12ec1cp+17f,        inf,  0x1.5e0486p+22f,   -0x1.d275ep+9f, -0x1.e1942cp+28f, -0x1.84fadep+21f,  0x1.a5581ap+25f,  -0x1.5aa98p+24f, -0x1.34682ep+23f, -0x1.12ddccp+22f},
      {  0x1.80647p+30f,  0x1.634984p+27f, -0x1.95f17ep+28f,  0x1.b78fd8p+25f,  0x1.b04122p+28f, factor_nan,  0x1.07cda2p+19f,        inf, -0x1.1d9dacp+22f,  0x1.ec71bcp+16f,  0x1.d69ee2p+27f,  0x1.2e7888p+25f,   0x1.33f12p+11f,  -0x1.21321p+28f, -0x1.ac87a2p+27f, -0x1.660c6cp+26f},
      { 0x1.48d01ep+24f,  0x1.51c6fap+24f, -0x1.0678c4p+24f,  0x1.20c7b6p+26f,  0x1.a073f6p+29f, factor_nan, -0x1.23977ep+24f,       -inf, -0x1.370a62p+19f,  0x1.1a0192p+22f,  0x1.41ccb8p+27f,  0x1.44e722p+29f, -0x1.d4c298p+15f,  0x1.63bdaap+26f, -0x1.7bd352p+19f, -0x1.dd8afcp+21f},
  };
  // clang-format on

  // Invalid requests must be rejected before operand access or output writes.
  // This instruction supports the six NEG/ABS modifier bits, with no DPP.
  rocjitsu::arith::test::expect_matrix_invalid_flags(rj_arith_v_wmma_f32_16x16x16_f16_w32, input,
                                                     24, 0, 8, 16, 63);

  // A=VGPRs 0..7, B=8..15, C=16..23; D spans eight registers. Offsets
  // cover A alias, partial A/B overlap, B alias, partial B/C overlap, C alias,
  // and disjoint D.
  rocjitsu::arith::test::expect_wmma_f32_arithmetic</*Gfx11=*/true>(
      rj_arith_v_wmma_f32_16x16x16_f16_w32, input, golden, {0, 4, 8, 12, 16, 24});
}
