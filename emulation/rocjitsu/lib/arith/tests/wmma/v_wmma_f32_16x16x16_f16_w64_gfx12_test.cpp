// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for v_wmma_f32_16x16x16_f16_w64_gfx12.
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

using Registers = rocjitsu::arith::test::Registers<17, 64>;

} // namespace

// Check matrix results, operand aliases, and instruction-flag contracts across supported CPUs
// and semantics.
TEST(WmmaF32F16W64Gfx12, Arithmetic) {
  Registers input;

  // Named engine with its standardized default seed and explicit mapping (no distributions).
  // Take the high 16 bits of each 31-bit draw for FP16 sign/exponent/fraction,
  // and the high 24 bits for FP32 accumulators. Never select only low bits.
  // Full fractional significands and varied exponents exercise GPU rounding.
  std::minstd_rand random;
  for (int reg = 0; reg < 4; ++reg) {
    for (int lane = 0; lane < 64; ++lane) {
      input.v[reg][lane] = rocjitsu::arith::test::random_f16_pair(random);
    }
  }
  for (int reg = 4; reg < 8; ++reg) {
    for (int lane = 0; lane < 64; ++lane) {
      input.v[reg][lane] = rocjitsu::arith::test::random_f32_bits(random);
    }
  }

  // Exactly one NaN and Inf per operand, with distinct K positions. Exact modes
  // check their encodings; loose mode checks NaN classification.
  rocjitsu::arith::test::set_wmma_gfx12_half<64>(input.v, 1, 2,
                                                 rocjitsu::arith::test::kF16QuietNaNBits);
  rocjitsu::arith::test::set_wmma_gfx12_half<64>(input.v, 3, 4,
                                                 rocjitsu::arith::test::kF16PositiveInfinityBits);
  rocjitsu::arith::test::set_wmma_gfx12_half<64>(input.v + 2, 5, 6,
                                                 rocjitsu::arith::test::kF16QuietNaNBits);
  rocjitsu::arith::test::set_wmma_gfx12_half<64>(input.v + 2, 7, 8,
                                                 rocjitsu::arith::test::kF16PositiveInfinityBits);
  // C begins at VGPR 4. Rows use reg=row%4 and
  // lane=col+16*(row/8)+32*((row/4)%2); inject at (9,10) and (11,12).
  rocjitsu::arith::test::wmma_gfx12_acc<64>(input.v + 4, 9, 10) =
      rocjitsu::arith::test::kF32QuietNaNBits;
  rocjitsu::arith::test::wmma_gfx12_acc<64>(input.v + 4, 11, 12) =
      rocjitsu::arith::test::kF32PositiveInfinityBits;
  constexpr float factor_nan = std::bit_cast<float>(0xffc00a3dU);
  constexpr float accumulator_nan = std::bit_cast<float>(0x7fc12345U);
  constexpr float inf = std::numeric_limits<float>::infinity();
  // Golden words from the in-tree gfx12_dot_f32<false> model, with FP16
  // DOT4 steps in Wave64 K-group order 0,4,8,12. No rj-arith reference computation.
  // clang-format off
  const float golden[16][16] = {
      {-0x1.afb00cp+20f,  0x1.0c3fcep+21f, -0x1.287e66p+21f,  0x1.fadb3cp+25f, -0x1.3d5528p+21f, factor_nan, -0x1.8ecc8cp+18f,        inf,  0x1.b3cb62p+12f,  0x1.b44d68p+18f,  0x1.fd1cbep+20f,  0x1.70f68ap+17f, -0x1.3947f4p+19f,  0x1.7a9312p+26f,   0x1.306dep+19f, -0x1.5ab2aep+27f},
      {      factor_nan,       factor_nan,       factor_nan,       factor_nan,       factor_nan, factor_nan,       factor_nan, factor_nan,       factor_nan,       factor_nan,       factor_nan,       factor_nan,       factor_nan,       factor_nan,       factor_nan,       factor_nan},
      { 0x1.e93654p+31f,  0x1.546a48p+30f,  0x1.aa3216p+20f, -0x1.3f65f4p+25f, -0x1.f52ef6p+27f, factor_nan, -0x1.d324aep+23f,        inf,  0x1.d10c9ep+15f, -0x1.da892cp+23f, -0x1.127eb8p+21f,  0x1.16298cp+29f,  0x1.965d7ep+23f,  0x1.c15c6ep+29f,  0x1.29d80ep+14f, -0x1.3fe038p+27f},
      {            -inf,              inf,             -inf,              inf,              inf, factor_nan,              inf,       -inf,              inf,              inf,              inf,              inf,             -inf,              inf,              inf,             -inf},
      {-0x1.261a64p+26f, -0x1.c46158p+18f,  0x1.048528p+29f,  0x1.d86eb6p+18f, -0x1.5a1c0ap+26f, factor_nan,  0x1.7c78c2p+26f,       -inf, -0x1.90c2f4p+25f,   0x1.c4f5cp+25f,  -0x1.5d1a8p+25f,  -0x1.7c514p+26f, -0x1.78c58cp+19f, -0x1.b2f9d6p+15f, -0x1.1bc49ep+27f,  0x1.09fb66p+22f},
      {-0x1.345a5ep+30f,   0x1.a3de7p+15f,  0x1.de2b4cp+19f, -0x1.047994p+17f, -0x1.c4354ap+18f, factor_nan, -0x1.012ebep+21f,        inf, -0x1.7e248ap+18f,  0x1.242bf4p+19f, -0x1.bad86cp+26f,  0x1.6aee3cp+22f,  0x1.26f718p+24f,  0x1.c0f92cp+18f,  0x1.2bced6p+22f,  -0x1.ec4dcp+16f},
      {-0x1.6657e6p+27f, -0x1.f99d8ep+23f, -0x1.4874fep+22f, -0x1.678842p+21f,  0x1.8af1eep+27f, factor_nan, -0x1.3a3e82p+16f,       -inf, -0x1.35bd42p+17f,  0x1.cf20c4p+20f,   0x1.64d3fp+25f, -0x1.288166p+16f, -0x1.4b5404p+16f,  0x1.f091dcp+24f,  0x1.9d3eb4p+24f,  0x1.2a9c5cp+24f},
      { 0x1.0594bcp+15f, -0x1.17f0f2p+15f, -0x1.8918acp+15f,  -0x1.a2813cp+6f, -0x1.9eebacp+12f, factor_nan, -0x1.411c04p+26f,        inf, -0x1.c93e0ap+19f, -0x1.bd4044p+23f,  0x1.1cffd4p+27f,  0x1.077394p+31f, -0x1.b80b18p+17f,  0x1.1e4668p+14f, -0x1.f9a8ccp+11f, -0x1.ba50eap+25f},
      {-0x1.fbafd8p+22f, -0x1.865e28p+21f, -0x1.7c7a94p+18f,  0x1.b3de18p+15f,  0x1.7facb2p+25f, factor_nan,  0x1.e33bf2p+19f,       -inf, -0x1.fe98d2p+13f,  0x1.186ca6p+16f,   0x1.34904p+20f,   0x1.3d9c2p+23f,  -0x1.80a5fp+17f, -0x1.71b4a8p+26f,  0x1.2437fcp+19f,  0x1.910df8p+27f},
      { -0x1.0a523p+27f, -0x1.00207cp+24f,    0x1.33c6p+21f, -0x1.6bce4ap+29f, -0x1.081276p+27f, factor_nan,  0x1.04ce1ep+13f,        inf,  0x1.982a0ap+15f, -0x1.31bee2p+20f,  accumulator_nan, -0x1.11d486p+23f, -0x1.bf210ep+14f,   0x1.2732dp+30f,    0x1.bbb6p+23f,  0x1.313edap+21f},
      { 0x1.8ac3e4p+31f,  0x1.12add8p+30f,  0x1.1893a8p+22f,  0x1.6da4dep+26f, -0x1.8e45c4p+26f, factor_nan, -0x1.6b66eap+23f,       -inf, -0x1.fb5446p+18f, -0x1.61fa5cp+23f,  0x1.a07f48p+27f,  0x1.88d1fep+17f,   0x1.39754p+23f, -0x1.8aca0ap+27f, -0x1.b9777ep+19f,  0x1.14bb44p+15f},
      {-0x1.8d10fcp+25f,  0x1.49deb8p+13f,  0x1.2eeef6p+26f,  0x1.4d6ae4p+16f,  0x1.14504ap+20f, factor_nan,  0x1.4b8e36p+19f,       -inf, -0x1.166fcep+23f,   0x1.4ae57p+23f,  -0x1.fb6a1p+26f,  0x1.6411d8p+23f,              inf,    0x1.5898p+25f, -0x1.5d8428p+23f, -0x1.750b6ap+21f},
      { 0x1.ea6e48p+30f,  0x1.83c072p+28f,  0x1.b63718p+30f, -0x1.3223aap+29f, -0x1.e3bf22p+30f, factor_nan,  0x1.36a122p+25f,        inf,  0x1.187f34p+25f,  -0x1.fcdc6p+23f,  0x1.eda284p+28f, -0x1.0a9e56p+27f, -0x1.351c7ap+29f,  0x1.06a88ep+30f, -0x1.a0cb68p+26f, -0x1.02ca18p+18f},
      { 0x1.152c2cp+26f, -0x1.f13308p+17f,  0x1.9e083cp+29f,  0x1.52c06cp+23f, -0x1.b4b264p+29f, factor_nan,     0x1.131p+17f,       -inf,   0x1.5e02dp+22f,   -0x1.d17e4p+9f, -0x1.e19434p+28f, -0x1.84feeap+21f,  0x1.a5580ep+25f, -0x1.5aa928p+24f, -0x1.34683cp+23f,  -0x1.12dc2p+22f},
      { 0x1.806474p+30f,  0x1.634984p+27f, -0x1.95f176p+28f,  0x1.b78fd8p+25f,  0x1.b04122p+28f, factor_nan,  0x1.07c16cp+19f,       -inf, -0x1.1d9b6ep+22f,  0x1.eca504p+16f,  0x1.d69ee2p+27f,  0x1.2e7898p+25f,  0x1.287708p+11f, -0x1.213218p+28f, -0x1.ac87b8p+27f, -0x1.660c9ap+26f},
      { 0x1.48d01ep+24f,  0x1.51c6f8p+24f, -0x1.0678c6p+24f,  0x1.20c7b4p+26f,  0x1.a073f8p+29f, factor_nan, -0x1.23977cp+24f,        inf, -0x1.370a64p+19f,  0x1.1a0196p+22f,  0x1.41ccb8p+27f,  0x1.44e722p+29f, -0x1.d4c296p+15f,  0x1.63bdaap+26f, -0x1.7bd354p+19f,   -0x1.dd8bp+21f},
  };
  // clang-format on

  // Invalid requests must be rejected before operand access or output writes.
  // Nonzero instruction modifiers are rejected in this slice.
  rocjitsu::arith::test::expect_matrix_invalid_flags(rj_arith_v_wmma_f32_16x16x16_f16_w64_gfx12,
                                                     input, 8, 0, 2, 4, 0);

  // A=VGPRs 0..1, B=2..3, C=4..7; D spans four registers. Offsets
  // cover A/B overlap, partial A/B/C overlap, B/C overlap, C alias, and disjoint D.
  const auto cpu_levels = rocjitsu::arith::test::supported_cpu_levels();
  for (uint64_t cpu : cpu_levels) {
    for (uint64_t semantics : rocjitsu::arith::test::semantics_modes()) {
      for (uint64_t fp_flags : {uint64_t{0}, RJ_ARITH_FP16_OVFL}) {
        for (int aliasing_offset : {0, 1, 2, 4, 8}) {
          SCOPED_TRACE(::testing::Message() << cpu << '/' << semantics << '/' << aliasing_offset);
          Registers r = input;
          ASSERT_EQ(rj_arith_v_wmma_f32_16x16x16_f16_w64_gfx12(cpu | semantics | fp_flags, 0,
                                                               r.v + aliasing_offset, r.v, r.v + 2,
                                                               r.v + 4),
                    RJ_ARITH_SUCCESS);
          for (int row = 0; row < 16; ++row) {
            for (int col = 0; col < 16; ++col) {
              float expected = golden[row][col];
              uint32_t actual =
                  rocjitsu::arith::test::wmma_gfx12_acc<64>(r.v + aliasing_offset, row, col);
              if (semantics == RJ_ARITH_SEMANTICS_LOOSE) {
                rocjitsu::arith::test::expect_f32(actual, expected, semantics, /*atol=*/1e-5,
                                                  /*rtol=*/1e-5);
              } else {
                EXPECT_EQ(actual, std::bit_cast<uint32_t>(expected));
              }
            }
          }
          r.expect_preserved(input, aliasing_offset, 4);
        }
      }
    }
  }
}

// Reject unsupported guest controls before operand access.
TEST(WmmaF32F16W64Gfx12, UnsupportedGuestControls) {
  rocjitsu::arith::test::expect_unsupported_guest_controls([&](uint64_t flag) {
    return rj_arith_v_wmma_f32_16x16x16_f16_w64_gfx12(flag, 0, nullptr, nullptr, nullptr, nullptr);
  });
}
