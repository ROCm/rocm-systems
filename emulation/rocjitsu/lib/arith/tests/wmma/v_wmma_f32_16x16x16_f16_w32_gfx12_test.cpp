// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for v_wmma_f32_16x16x16_f16_w32_gfx12.
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
#include <vector>

namespace {

using Registers = rocjitsu::arith::test::Registers<33, 32>;

// Check matrix results, operand aliases, and instruction-flag contracts across supported CPUs
// and semantics.
TEST(WmmaF32F16W32Gfx12, Arithmetic) {
  Registers input;

  // Named engine with its standardized default seed and explicit mapping (no distributions).
  // Take the high 16 bits of each 31-bit draw for FP16 sign/exponent/fraction,
  // and the high 24 bits for FP32 accumulators. Never select only low bits.
  // Full fractional significands and varied exponents exercise GPU rounding.
  std::minstd_rand random;
  for (int reg = 0; reg < 8; ++reg) {
    for (int lane = 0; lane < 32; ++lane) {
      input.v[reg][lane] = rocjitsu::arith::test::random_f16_pair(random);
    }
  }
  for (int reg = 8; reg < 16; ++reg) {
    for (int lane = 0; lane < 32; ++lane) {
      input.v[reg][lane] = rocjitsu::arith::test::random_f32_bits(random);
    }
  }

  // Exactly one NaN and Inf per operand, with distinct K positions. NaN
  // encodings are checked exactly outside loose semantics.
  rocjitsu::arith::test::set_wmma_gfx12_half<32>(input.v, 1, 2,
                                                 rocjitsu::arith::test::kF16QuietNaNBits);
  rocjitsu::arith::test::set_wmma_gfx12_half<32>(input.v, 3, 4,
                                                 rocjitsu::arith::test::kF16PositiveInfinityBits);
  rocjitsu::arith::test::set_wmma_gfx12_half<32>(input.v + 4, 5, 6,
                                                 rocjitsu::arith::test::kF16QuietNaNBits);
  rocjitsu::arith::test::set_wmma_gfx12_half<32>(input.v + 4, 7, 8,
                                                 rocjitsu::arith::test::kF16PositiveInfinityBits);
  // C starts at VGPR 8: C[row][col] maps to VGPR 8 + row % 8,
  // lane col + 16 * (row / 8). Set C[9][10] to a quiet NaN and
  // C[11][12] to +Inf, outside the NaN/Inf rows and columns of A and B.
  input.v[8 + 9 % 8][10 + 16 * (9 / 8)] = rocjitsu::arith::test::kF32QuietNaNBits;
  input.v[8 + 11 % 8][12 + 16 * (11 / 8)] = rocjitsu::arith::test::kF32PositiveInfinityBits;
  // Numeric goldens from the rocjitsu model, confirmed word-for-word on gfx1201.
  constexpr float factor_nan = std::bit_cast<float>(0xffc00a3dU);
  constexpr float accumulator_nan = std::bit_cast<float>(0x7fc12345U);
  constexpr float invalid_product_nan = std::bit_cast<float>(0xffc00000U);
  constexpr float inf = std::numeric_limits<float>::infinity();
  // Golden outputs from gfx12_dot_f32<false> at this PR's base, using the
  // exec_gfx12_wmma_dot4_impl Wave32 K-group order 0,8,4,12 and default controls.
  // wmma_swap_k_bits_2_3 defines that order; no rj-arith output supplies the reference.
  // clang-format off
  const float golden[16][16] = {
      {-0x1.afb00ep+20f,   0x1.0c3fdp+21f, -0x1.287e66p+21f,  0x1.fadb3cp+25f, -0x1.3d5548p+21f, factor_nan, -0x1.8ecc8cp+18f,                 inf,  0x1.b3cb62p+12f,  0x1.b44d68p+18f,   0x1.fd1ccp+20f,  0x1.70f688p+17f, -0x1.3947f4p+19f,   0x1.7a931p+26f,   0x1.306dep+19f, -0x1.5ab2aep+27f},
      {      factor_nan,       factor_nan,       factor_nan,       factor_nan,       factor_nan, factor_nan,       factor_nan,          factor_nan,       factor_nan,       factor_nan,       factor_nan,       factor_nan,       factor_nan,       factor_nan,       factor_nan,       factor_nan},
      { 0x1.e93656p+31f,  0x1.546a4ap+30f,   0x1.aa32dp+20f,  -0x1.3f664p+25f, -0x1.f52efep+27f, factor_nan, -0x1.d32524p+23f,                -inf,  0x1.d14e9ap+15f, -0x1.da87c6p+23f, -0x1.127f08p+21f,  0x1.162988p+29f,  0x1.965d18p+23f,  0x1.c15c6ep+29f,   0x1.2a0e3p+14f, -0x1.3fe038p+27f},
      {             inf,              inf,             -inf,              inf,              inf, factor_nan,              inf, invalid_product_nan,              inf,             -inf,             -inf,              inf,              inf,             -inf,              inf,             -inf},
      {-0x1.261a64p+26f, -0x1.c4515ep+18f,  0x1.04852ep+29f,   0x1.d8a98p+18f, -0x1.5a1c16p+26f, factor_nan,  0x1.7c78d2p+26f,                -inf, -0x1.90c2eep+25f,  0x1.c4f558p+25f, -0x1.5d1a74p+25f, -0x1.7c5124p+26f, -0x1.78cf28p+19f, -0x1.b3188ep+15f, -0x1.1bc4aep+27f,  0x1.09fc7ep+22f},
      { -0x1.345a6p+30f,  0x1.a400f4p+15f,  0x1.de1486p+19f, -0x1.04c668p+17f, -0x1.c44568p+18f, factor_nan, -0x1.0132f8p+21f,                -inf, -0x1.7e257ap+18f,  0x1.2425b4p+19f, -0x1.bad84cp+26f,  0x1.6aecdcp+22f,  0x1.26f714p+24f,  0x1.c0fa12p+18f,  0x1.2bce8ap+22f, -0x1.ed16e6p+16f},
      { -0x1.6657fp+27f, -0x1.f99da2p+23f, -0x1.4871aap+22f, -0x1.678958p+21f,    0x1.8af2p+27f, factor_nan, -0x1.39db1ep+16f,                 inf, -0x1.35f68ap+17f,  0x1.cf1eccp+20f,  0x1.64d3a8p+25f, -0x1.285902p+16f, -0x1.4a6afap+16f,  0x1.f09248p+24f,  0x1.9d3dfep+24f,  0x1.2a9cbap+24f},
      { 0x1.0594bcp+15f, -0x1.17f0f2p+15f, -0x1.8918aap+15f,  -0x1.a2813ap+6f, -0x1.9eebacp+12f, factor_nan, -0x1.411c04p+26f,                 inf, -0x1.c93e0cp+19f, -0x1.bd4044p+23f,  0x1.1cffd4p+27f,  0x1.077394p+31f, -0x1.b80b18p+17f,  0x1.1e4668p+14f, -0x1.f9a8ccp+11f, -0x1.ba50eap+25f},
      {-0x1.fbafdap+22f, -0x1.865e26p+21f, -0x1.7c7a96p+18f,  0x1.b3de18p+15f,   0x1.7facbp+25f, factor_nan,  0x1.e33bf2p+19f,                 inf,  -0x1.fe98dp+13f,  0x1.186ca6p+16f,   0x1.34904p+20f,  0x1.3d9c22p+23f,  -0x1.80a5fp+17f, -0x1.71b4a8p+26f,  0x1.2437fcp+19f,  0x1.910df8p+27f},
      {-0x1.0a5236p+27f, -0x1.001ff4p+24f,   0x1.33c16p+21f, -0x1.6bce4cp+29f, -0x1.08126cp+27f, factor_nan,  0x1.05fa02p+13f,                 inf,   0x1.995a7p+15f, -0x1.31c0d8p+20f,  accumulator_nan, -0x1.11d622p+23f, -0x1.bdf7dep+14f,  0x1.2732cep+30f,  0x1.bbb482p+23f,  0x1.314404p+21f},
      { 0x1.8ac3e4p+31f,  0x1.12addap+30f,  0x1.18954ep+22f,  0x1.6da4fcp+26f, -0x1.8e45e8p+26f, factor_nan,   -0x1.6b67p+23f,                -inf, -0x1.fb8deap+18f, -0x1.61fa44p+23f,   0x1.a07f5p+27f,  0x1.88d71cp+17f,   0x1.3973cp+23f,  -0x1.8aca1p+27f, -0x1.b9776cp+19f,  0x1.139b4ap+15f},
      {-0x1.8d10dcp+25f,  0x1.4a3eacp+13f,  0x1.2eef02p+26f,  0x1.4d8082p+16f,  0x1.144b3cp+20f, factor_nan,  0x1.4b972ep+19f,                -inf, -0x1.1670a8p+23f,  0x1.4ae574p+23f, -0x1.fb6a1ep+26f,  0x1.6410d4p+23f,              inf,  0x1.58982ap+25f, -0x1.5d8434p+23f,  -0x1.75081p+21f},
      { 0x1.ea6e4ap+30f,  0x1.83c064p+28f,  0x1.b6371ap+30f, -0x1.3223aap+29f,  -0x1.e3bf2p+30f, factor_nan,  0x1.36a116p+25f,                 inf,  0x1.187f5cp+25f, -0x1.fcdc38p+23f,  0x1.eda288p+28f, -0x1.0a9e3ep+27f, -0x1.351c76p+29f,  0x1.06a88ep+30f, -0x1.a0cb3ap+26f, -0x1.02cf68p+18f},
      { 0x1.152c54p+26f, -0x1.f109eep+17f,  0x1.9e0842p+29f,  0x1.52bf8ap+23f, -0x1.b4b25ep+29f, factor_nan,  0x1.12b036p+17f,                -inf,  0x1.5e04b8p+22f,  -0x1.88f4e4p+9f, -0x1.e19438p+28f, -0x1.84f92ep+21f,  0x1.a55834p+25f, -0x1.5aaa04p+24f, -0x1.346802p+23f, -0x1.12dde8p+22f},
      {  0x1.80647p+30f,  0x1.634978p+27f, -0x1.95f182p+28f,  0x1.b79004p+25f,  0x1.b0411ep+28f, factor_nan,  0x1.07d064p+19f,                -inf, -0x1.1d9ba2p+22f,  0x1.ec1204p+16f,   0x1.d69efp+27f,  0x1.2e787cp+25f,  0x1.2187f8p+11f,  -0x1.21321p+28f, -0x1.ac87bap+27f, -0x1.660c98p+26f},
      { 0x1.48d01cp+24f,  0x1.51c6f8p+24f, -0x1.0678c4p+24f,  0x1.20c7b4p+26f,  0x1.a073f6p+29f, factor_nan, -0x1.23977cp+24f,                -inf, -0x1.370a62p+19f,  0x1.1a0194p+22f,  0x1.41ccb8p+27f,  0x1.44e722p+29f, -0x1.d4c296p+15f,  0x1.63bdaap+26f, -0x1.7bd352p+19f, -0x1.dd8b04p+21f},
};
  // clang-format on

  // Invalid requests must be rejected before operand access or output writes.
  // Nonzero instruction modifiers are not implemented yet.
  rocjitsu::arith::test::expect_matrix_invalid_flags(rj_arith_v_wmma_f32_16x16x16_f16_w32_gfx12,
                                                     input, 16, 0, 4, 8, 0);

  // A=VGPRs 0..3, B=4..7, C=8..15; D spans eight registers. These cover
  // A/B overlap, partial A/B/C overlap, B/C overlap, C alias, and disjoint D.
  const std::vector<uint64_t> cpu_levels = rocjitsu::arith::test::supported_cpu_levels();
  for (uint64_t cpu : cpu_levels) {
    for (uint64_t semantics : rocjitsu::arith::test::semantics_modes()) {
      for (uint64_t fp_flags : {uint64_t{0}, RJ_ARITH_FP16_OVFL}) {
        for (int aliasing_offset : {0, 2, 4, 8, 16}) {
          SCOPED_TRACE(::testing::Message() << cpu << '/' << semantics << '/' << aliasing_offset);
          Registers r = input;
          ASSERT_EQ(rj_arith_v_wmma_f32_16x16x16_f16_w32_gfx12(cpu | semantics | fp_flags, 0,
                                                               r.v + aliasing_offset, r.v, r.v + 4,
                                                               r.v + 8),
                    RJ_ARITH_SUCCESS);
          for (int row = 0; row < 16; ++row) {
            for (int col = 0; col < 16; ++col) {
              float expected = golden[row][col];
              uint32_t actual = r.v[aliasing_offset + row % 8][col + 16 * (row / 8)];
              if (semantics == RJ_ARITH_SEMANTICS_LOOSE) {
                rocjitsu::arith::test::expect_f32(actual, expected, semantics, /*atol=*/1e-5,
                                                  /*rtol=*/1e-5);
              } else {
                EXPECT_EQ(actual, std::bit_cast<uint32_t>(expected));
              }
            }
          }
          r.expect_preserved(input, aliasing_offset, 8);
        }
      }
    }
  }
}

// Reject controls not implemented by this slice, before accessing operands.
TEST(WmmaF32F16W32Gfx12, UnsupportedGuestControls) {
  for (uint64_t flag :
       {RJ_ARITH_FP_FLUSH_INPUT_DENORMALS, RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS,
        RJ_ARITH_FP_IEEE_MODE, RJ_ARITH_FP_DX10_CLAMP_OFF, RJ_ARITH_FP_ROUND_UP,
        RJ_ARITH_FP_ROUND_DOWN, RJ_ARITH_FP_ROUND_ZERO, RJ_ARITH_FP_FLUSH_INPUT_DENORMALS_F16,
        RJ_ARITH_FP_PRESERVE_INPUT_DENORMALS_F16}) {
    EXPECT_EQ(
        rj_arith_v_wmma_f32_16x16x16_f16_w32_gfx12(flag, 0, nullptr, nullptr, nullptr, nullptr),
        RJ_ARITH_ERROR_INVALID_FLAGS);
  }
}

} // namespace
