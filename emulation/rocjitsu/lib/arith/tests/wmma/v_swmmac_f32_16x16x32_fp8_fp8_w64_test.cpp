// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for v_swmmac_f32_16x16x32_fp8_fp8_w64.
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
TEST(SwmmacF32FP8FP8K32W64, Arithmetic) {
  std::minstd_rand random;
  uint32_t a[16][16], b[32][16], c[16][16];
  for (int row = 0; row < 16; ++row) {
    for (int k = 0; k < 16; ++k) {
      a[row][k] = rocjitsu::arith::test::random_fp8_bits(random);
    }
  }
  for (int k = 0; k < 32; ++k) {
    for (int col = 0; col < 16; ++col) {
      b[k][col] = rocjitsu::arith::test::random_fp8_bits(random);
    }
  }
  for (int row = 0; row < 16; ++row) {
    for (int col = 0; col < 16; ++col) {
      c[row][col] = rocjitsu::arith::test::random_f32_bits(random);
    }
  }
  // Two increasing positions per group of four, using high engine bits.
  int selected[16][16];
  rocjitsu::arith::test::make_sparse_indices(random, selected);
  // One NaN and one infinity per logical operand where the format has Inf.
  a[1][2] = rocjitsu::arith::test::kFP8QuietNaNBits;
  // FP8 E4M3FN has no infinity encoding.
  b[6][5] = rocjitsu::arith::test::kFP8QuietNaNBits;
  // FP8 E4M3FN has no infinity encoding.
  c[9][10] = rocjitsu::arith::test::kF32QuietNaNBits;
  c[11][12] = rocjitsu::arith::test::kF32PositiveInfinityBits;

  Registers input;
  for (int row = 0; row < 16; ++row) {
    for (int k = 0; k < 16; ++k) {
      rocjitsu::arith::test::set_swmmac8_a<64>(input.v, row, k, a[row][k]);
    }
  }
  for (int k = 0; k < 32; ++k) {
    for (int col = 0; col < 16; ++col) {
      rocjitsu::arith::test::set_swmmac8_b<64>(input.v + 1, col, k, b[k][col]);
    }
  }
  // Map each logical accumulator to its destination register and lane.
  for (int row = 0; row < 16; ++row) {
    for (int col = 0; col < 16; ++col) {
      input.v[3 + ((row % 8) % 4)][col + 16 * (row / 8) + 32 * ((row % 8) / 4)] = c[row][col];
    }
  }
  // Replicate valid pairs in each metadata bank; each call below clears
  // the unused banks so a wrong selector cannot read equivalent inputs.
  std::fill_n(input.v[7], 64, 0);
  for (int row = 0; row < 16; ++row) {
    for (int k = 0; k < 16; ++k) {
      int lane = row + 16 * (k / 8) + 32 * ((k / 4) & 1), local = k % 4;
      for (int key = 0; key < 32; key += 8) {
        input.v[7][lane] |= uint32_t(selected[row][k] % 4) << (key + 2 * local);
      }
    }
  }

  // Goldens from rocjitsu's gfx12_dot8_f32 model, using logical matrices
  // and sparse indices independently of VGPR packing; exact NaN encodings included.
  constexpr float factor_nan = std::bit_cast<float>(0xffc00a3dU);
  constexpr float accumulator_nan = std::bit_cast<float>(0x7fc12345U);
  constexpr float inf = std::numeric_limits<float>::infinity();
  // clang-format off
  const float golden[16][16] = {
      {  0x1.5190ap+4f,  0x1.9a2634p+6f,    0x1.01adp+6f,   0x1.eb988p+2f,  -0x1.42c8cp+6f,  -0x1.5d5ecp+3f,  0x1.80fda4p+6f,  0x1.afa91cp+6f,  0x1.df3b28p+5f, -0x1.da217cp+6f,  -0x1.6eb34p+3f, -0x1.8c3544p+6f,  -0x1.0c77bp+4f,  0x1.036eb8p+5f,  0x1.3dcb94p+6f,   -0x1.69d8p+6f},
      {     factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan},
      { 0x1.7c68f4p+6f, -0x1.b322a8p+5f, -0x1.683c14p+6f,  0x1.1a2ea4p+6f,    0x1.19c8p+0f,  -0x1.7b608p+3f,  -0x1.f8093p+5f,   0x1.1a0f5p+4f,  -0x1.1df3dp+5f,   0x1.c30ddp+6f,  0x1.0b77bcp+6f,   0x1.d77f2p+6f,  0x1.814bb4p+6f,  0x1.0c4168p+6f, -0x1.13452cp+6f,  -0x1.a7ff5p+6f},
      {-0x1.762b14p+6f,   0x1.1514cp+6f,  -0x1.f84f8p+4f,  0x1.6f9d78p+5f, -0x1.5688a8p+5f,      factor_nan, -0x1.a51f28p+6f, -0x1.02935cp+6f,  -0x1.baa5dp+4f, -0x1.33a104p+6f,  0x1.26e2a4p+6f,  0x1.3eda18p+5f,  -0x1.92fdep+5f,  0x1.b8ead8p+6f,   0x1.00277p+5f,   0x1.9ef2bp+4f},
      { -0x1.10f82p+5f,   0x1.9619ep+4f,  0x1.4b8858p+6f, -0x1.68c4c8p+5f,  0x1.c3f608p+5f,      factor_nan,  -0x1.b4e11p+5f,  0x1.06650cp+7f,   0x1.e4e18p+2f,   0x1.8b75ep+6f, -0x1.d8584cp+6f, -0x1.223138p+5f,   0x1.605abp+4f,  -0x1.6df8bp+5f,   0x1.536b2p+5f, -0x1.ed1584p+6f},
      { 0x1.def104p+6f,   0x1.6558bp+5f,  0x1.3f783cp+6f,  0x1.078288p+6f,  -0x1.ebd06p+4f,      factor_nan,    0x1.4ad4p+3f,  0x1.76eca8p+5f,  -0x1.664b1p+4f,   0x1.2f99ep+6f,  -0x1.0689ap+5f,  0x1.c283f8p+6f,  0x1.866c18p+5f,   0x1.eac5ep+5f,   0x1.e2ea3p+6f, -0x1.95419cp+6f},
      {  0x1.804b4p+5f,  -0x1.95854p+6f,   0x1.92b78p+5f,  -0x1.2f4bdp+4f,  0x1.2445bcp+6f,   0x1.01e1cp+2f, -0x1.8eea94p+6f,  0x1.9918e8p+6f,  0x1.581c78p+5f,  -0x1.c050cp+6f,  0x1.54649cp+6f,  0x1.8be5dcp+6f, -0x1.0cfafep+7f,  0x1.53bc78p+6f,    0x1.3525p+4f,    0x1.9f24p+0f},
      {  0x1.c98b4p+4f,  -0x1.e8301p+4f,  0x1.0c4c18p+5f,  -0x1.2227ap+5f,  0x1.6c8858p+5f,      factor_nan, -0x1.e7ab94p+6f,  0x1.12b462p+7f, -0x1.3b7174p+6f,   0x1.7764ep+5f,  -0x1.0cda3p+5f, -0x1.f6d1b8p+6f,   0x1.fb9e5p+5f,  -0x1.ec584p+3f, -0x1.16791ep+7f,  0x1.e00418p+6f},
      { 0x1.4e8d08p+5f,  -0x1.d228cp+4f,  0x1.2ca3dcp+6f, -0x1.882c08p+5f, -0x1.b33e08p+6f, -0x1.293504p+6f, -0x1.b34f3cp+6f,   0x1.f1dbep+5f,   0x1.fd278p+5f,  0x1.f162a8p+5f, -0x1.1448fcp+6f,   0x1.9df7cp+3f, -0x1.f00e94p+6f,  0x1.f9b4c8p+6f,   0x1.063b4p+3f,  0x1.737848p+6f},
      {-0x1.a36814p+6f, -0x1.e49eb8p+5f,  0x1.42f1f8p+5f,   0x1.ddd67p+5f, -0x1.ff7c18p+6f,  -0x1.1e6e6p+7f, -0x1.606ab4p+6f,   0x1.a23d3p+6f, -0x1.066d46p+7f,  0x1.420f78p+5f, accumulator_nan,   0x1.4f9fcp+2f, -0x1.8c5648p+6f, -0x1.314a48p+5f,  0x1.0c133cp+6f,  -0x1.226ddp+4f},
      { 0x1.4640a8p+6f,   0x1.ad916p+6f,  0x1.e7c78cp+6f,   0x1.04268p+6f,   0x1.9676ap+5f,      factor_nan,  0x1.50d4b8p+6f, -0x1.d84464p+6f, -0x1.47a3a8p+5f,  -0x1.1d80dp+6f,  -0x1.9fc9fp+6f,  0x1.30a118p+6f,    0x1.a603p+1f,  -0x1.ccb7dp+5f, -0x1.f475c8p+5f,   0x1.89586p+3f},
      {  0x1.58956p+3f,  -0x1.908d5p+6f,  -0x1.e326ap+4f,  0x1.4f9e84p+6f,  -0x1.162ffp+5f,      factor_nan, -0x1.bd0f18p+5f, -0x1.31ccdcp+6f,   0x1.4252ap+4f, -0x1.727558p+6f,  0x1.0c032cp+6f,  -0x1.d783ep+6f,             inf,   0x1.86c58p+5f,   0x1.017ccp+3f, -0x1.0f9ae8p+5f},
      {-0x1.d5c498p+5f, -0x1.4e4fdcp+6f,   0x1.ba9fap+3f,   0x1.91684p+3f,  -0x1.4686cp+3f,    -0x1.3edp-2f,   0x1.998aep+3f,   0x1.3aef9p+4f,  0x1.ccc4d8p+5f,   -0x1.da87p+1f,   0x1.0ced1p+6f,   -0x1.821bp+4f,  -0x1.3229ap+6f,   0x1.8ac47p+5f,   0x1.26a7cp+2f,  0x1.8922bcp+6f},
      { 0x1.bd3d38p+6f, -0x1.029b58p+6f, -0x1.a0d534p+6f,  0x1.b12f5cp+6f, -0x1.beaabcp+6f,  0x1.61bc84p+6f,  -0x1.c24d8p+3f,   0x1.53c2bp+4f,   0x1.8b342p+6f,  0x1.c4dfe4p+6f, -0x1.d96a5cp+6f,   0x1.81c28p+2f,  0x1.be743cp+6f,   0x1.6f9efp+4f,  0x1.e5b96cp+6f,  0x1.4375d4p+6f},
      {  0x1.4e8e4p+5f,  -0x1.07d7ap+4f,   -0x1.b2ffp+6f, -0x1.b125c4p+6f,   0x1.3ab87p+5f,      factor_nan,  0x1.01d258p+6f,  -0x1.03297p+6f,  -0x1.d252fp+5f,  0x1.ff357cp+6f,   0x1.e846bp+5f,   0x1.ce7b9p+5f, -0x1.160774p+7f,  0x1.2c61fcp+6f,   0x1.4c2aap+4f, -0x1.b865e8p+5f},
      { 0x1.257838p+6f,   0x1.f1078p+5f,  -0x1.dd8ecp+6f,  0x1.cae048p+5f,  0x1.fd3b78p+5f,      factor_nan, -0x1.fbc868p+6f,  -0x1.6e0eap+4f,   0x1.e1dcap+3f, -0x1.b28868p+6f,  -0x1.7760ep+5f,   0x1.c9697p+6f,    0x1.249bp+5f,  -0x1.202dcp+3f, -0x1.b76e58p+6f,   0x1.9787cp+5f},
  };
  // clang-format on

  rocjitsu::arith::test::expect_matrix_invalid_flags(
      rj_arith_v_swmmac_f32_16x16x32_fp8_fp8_w64, input, 3, 0, 1, 7,
      RJ_ARITH_SWMMAC_INDEX_KEY_1 | RJ_ARITH_SWMMAC_INDEX_KEY_2);

  // A occupies 1 registers, B 2, C/D 4. Reuse the same arithmetic
  // data for overlapping and disjoint destination ranges.
  const uint64_t modes[] = {0, RJ_ARITH_SWMMAC_INDEX_KEY_1, (0) | RJ_ARITH_SWMMAC_INDEX_KEY_2,
                            RJ_ARITH_SWMMAC_INDEX_KEY_1 | RJ_ARITH_SWMMAC_INDEX_KEY_2};
  const auto cpu_levels = rocjitsu::arith::test::supported_cpu_levels();
  for (uint64_t cpu : cpu_levels) {
    for (uint64_t semantics : rocjitsu::arith::test::semantics_modes()) {
      for (uint64_t mode : modes) {
        Registers selected_input = input;
        rocjitsu::arith::test::isolate_sparse_selector<64>(selected_input.v[7], mode);
        for (int aliasing_offset : {3}) {
          SCOPED_TRACE(::testing::Message()
                       << cpu << '/' << semantics << '/' << mode << '/' << aliasing_offset);
          Registers r = selected_input;
          ASSERT_EQ(rj_arith_v_swmmac_f32_16x16x32_fp8_fp8_w64(
                        cpu | semantics, mode, r.v + aliasing_offset, r.v, r.v + 1, r.v + 7),
                    RJ_ARITH_SUCCESS);
          for (int row = 0; row < 16; ++row) {
            for (int col = 0; col < 16; ++col) {
              uint32_t actual = r.v[aliasing_offset + ((row % 8) % 4)]
                                   [col + 16 * (row / 8) + 32 * ((row % 8) / 4)];
              if (semantics != RJ_ARITH_SEMANTICS_LOOSE) {
                EXPECT_EQ(actual, std::bit_cast<uint32_t>(golden[row][col]));
              } else {
                rocjitsu::arith::test::expect_f32(actual, golden[row][col], semantics,
                                                  /*atol=*/1e-5,
                                                  /*rtol=*/1e-5);
              }
            }
          }
          r.expect_preserved(selected_input, aliasing_offset, 4);
        }
      }
    }
  }
  rocjitsu::arith::test::expect_sparse_aliasing(rj_arith_v_swmmac_f32_16x16x32_fp8_fp8_w64, input,
                                                /*outputs=*/4, /*b=*/1, /*index=*/7, {0, 1, 1, 7},
                                                cpu_levels);
}
