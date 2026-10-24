// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for v_swmmac_f32_16x16x32_f16_w32.
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

using Registers = rocjitsu::arith::test::Registers<64, 32>;

} // namespace

// Check matrix results, operand aliases, and instruction-flag contracts across supported CPUs
// and semantics.
TEST(SwmmacF32F16K32W32, Arithmetic) {
  rocjitsu::arith::test::expect_sparse_modifiers<false, 32, false>(
      rj_arith_v_swmmac_f32_16x16x32_f16_w32);

  // Selected gfx1201 captures from rocjitsu's RdnaSwmmacFloat tests: B sign
  // follows compressed slots, and accumulator ABS precedes NEG.
  const struct {
    uint32_t a, b, c, mode, expected;
  } captures[] = {{0xbc003c00U, 0x42004000U, 0xbf000000U,
                   RJ_ARITH_SWMMAC_INDEX_KEY_1 | RJ_ARITH_WMMA_NEG_HI_B, 0x422e0000U},
                  {0x3c003c00U, 0x40004000U, 0xbf000000U,
                   RJ_ARITH_SWMMAC_INDEX_KEY_1 | RJ_ARITH_WMMA_ABS_C | RJ_ARITH_WMMA_NEG_C,
                   0x41fc0000U}};
  for (uint64_t cpu : rocjitsu::arith::test::supported_cpu_levels()) {
    for (const auto &capture : captures) {
      Registers r;
      for (int reg = 0; reg < 8; ++reg) {
        std::fill_n(r.v[reg], 32, capture.a);
        std::fill_n(r.v[reg + 8], 32, capture.b);
        std::fill_n(r.v[reg + 24], 32, capture.c);
      }
      std::fill_n(r.v[16], 32, 0xeeddc988U);
      ASSERT_EQ(rj_arith_v_swmmac_f32_16x16x32_f16_w32(
                    cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS, capture.mode,
                    r.v + 24, r.v, r.v + 8, r.v + 16),
                RJ_ARITH_SUCCESS);
      EXPECT_EQ(r.v[24][0], capture.expected);
    }
  }
  std::minstd_rand random;
  uint32_t a[16][16], b[32][16], c[16][16];
  for (int row = 0; row < 16; ++row) {
    for (int k = 0; k < 16; ++k) {
      a[row][k] = rocjitsu::arith::test::random_f16_bits(random);
    }
  }
  for (int k = 0; k < 32; ++k) {
    for (int col = 0; col < 16; ++col) {
      b[k][col] = rocjitsu::arith::test::random_f16_bits(random);
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
  a[1][2] = rocjitsu::arith::test::kF16QuietNaNBits;
  a[3][4] = rocjitsu::arith::test::kF16PositiveInfinityBits;
  b[6][5] = rocjitsu::arith::test::kF16QuietNaNBits;
  b[8][7] = rocjitsu::arith::test::kF16PositiveInfinityBits;
  c[9][10] = rocjitsu::arith::test::kF32QuietNaNBits;
  c[11][12] = rocjitsu::arith::test::kF32PositiveInfinityBits;

  Registers input;
  for (int row = 0; row < 16; ++row) {
    for (int k = 0; k < 16; ++k) {
      rocjitsu::arith::test::set_swmmac16_a<32>(input.v, row, k, a[row][k]);
    }
  }
  for (int k = 0; k < 32; ++k) {
    for (int col = 0; col < 16; ++col) {
      rocjitsu::arith::test::set_swmmac16_b<32>(input.v + 4, col, k, b[k][col]);
    }
  }
  // Map each logical accumulator to its destination register and lane.
  for (int row = 0; row < 16; ++row) {
    for (int col = 0; col < 16; ++col) {
      input.v[12 + (row % 8)][col + 16 * (row / 8)] = c[row][col];
    }
  }
  // Replicate valid pairs in each metadata bank; each call below clears
  // the unused banks so a wrong selector cannot read equivalent inputs.
  std::fill_n(input.v[20], 32, 0);
  for (int row = 0; row < 16; ++row) {
    for (int k = 0; k < 16; ++k) {
      int lane = row + 16 * ((k / 4) & 1), local = 2 * ((k / 2) & 1) + 4 * (k / 8) + k % 2;
      for (int key = 0; key < 32; key += 16) {
        input.v[20][lane] |= uint32_t(selected[row][k] % 4) << (key + 2 * local);
      }
    }
  }

  // Goldens from rocjitsu's gfx12_dot_f32 model, using logical matrices
  // and sparse indices independently of VGPR packing; exact NaN encodings included.
  constexpr float factor_nan = std::bit_cast<float>(0xffc00a3dU);
  constexpr float accumulator_nan = std::bit_cast<float>(0x7fc12345U);
  constexpr float inf = std::numeric_limits<float>::infinity();
  // clang-format off
  const float golden[16][16] = {
      { 0x1.84eaa6p+4f,  0x1.d13a08p+6f,  0x1.2790b8p+6f,  0x1.7c136ep+1f, -0x1.7f751ep+6f,  -0x1.91cdbp+3f,  0x1.5145e4p+6f,            -inf,  0x1.bfe138p+5f, -0x1.e121fcp+6f,  -0x1.55814p+3f, -0x1.c61edcp+6f, -0x1.be4208p+2f,  0x1.0c8d28p+4f,  0x1.5dff54p+6f, -0x1.2cd7ccp+6f},
      {     factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan},
      { 0x1.54022cp+6f, -0x1.cab674p+5f, -0x1.867d68p+6f,  0x1.1d02aep+6f,  0x1.604f2cp+3f,  0x1.94388cp+2f, -0x1.212334p+6f,  0x1.63965ap+4f, -0x1.413adcp+5f,   0x1.dcd07p+6f,  0x1.4170cep+6f,  0x1.deaad4p+6f,  0x1.95ae48p+6f,  0x1.fd1b7cp+5f, -0x1.0579bcp+6f, -0x1.cdb994p+6f},
      {            inf,            -inf,            -inf,             inf,            -inf,      factor_nan,             inf,             inf,             inf,            -inf,            -inf,             inf,             inf,             inf,            -inf,             inf},
      {-0x1.3f181cp+5f,  0x1.2980d2p+5f,  0x1.5fdc18p+6f, -0x1.ff366cp+4f,   0x1.87ff9p+5f,      factor_nan, -0x1.6592a4p+5f,  0x1.c975a4p+6f,  0x1.8a53fap+3f,  0x1.990796p+6f, -0x1.bee4bcp+6f, -0x1.2f95c8p+5f,  0x1.0408d8p+4f, -0x1.bc026ap+5f,  0x1.725e2ap+5f,  -0x1.d780cp+6f},
      { 0x1.f2529cp+6f,  0x1.33eb54p+5f,  0x1.e0739cp+5f,  0x1.7d835ap+5f,  -0x1.3a537p+3f,      factor_nan,  0x1.6cf13cp+4f,            -inf, -0x1.6e7bf6p+4f,  0x1.6d73c8p+6f, -0x1.44900cp+4f,  0x1.773e14p+6f,   0x1.09022p+6f,  0x1.0ed46ep+6f,  0x1.03328cp+7f, -0x1.c99042p+6f},
      { 0x1.d3b474p+5f,  -0x1.9af0dp+6f,  0x1.d0beaep+5f, -0x1.379a64p+4f,  0x1.2cbd1cp+6f,  0x1.a9eaeep+3f, -0x1.abe0a2p+6f,             inf,  0x1.f26d38p+5f, -0x1.cfff36p+6f,  0x1.7ac222p+6f,  0x1.ce5cd4p+6f, -0x1.03b222p+7f,  0x1.32bf78p+6f,  0x1.720bacp+4f,  0x1.be74acp+2f},
      { 0x1.97d382p+4f, -0x1.52e67ep+5f,  0x1.efb7cap+4f, -0x1.0ca166p+5f,  0x1.6d0f96p+5f,      factor_nan, -0x1.d2ea78p+6f,             inf, -0x1.47f1e8p+6f,  0x1.d4491ap+5f,  -0x1.9f70ap+4f, -0x1.038ad4p+7f,  0x1.d32fb6p+5f, -0x1.b4449ap+3f,  -0x1.ffdb8p+6f,  0x1.ccd3fcp+6f},
      { 0x1.88d72ap+5f, -0x1.8cdd52p+4f,  0x1.036b8ep+6f, -0x1.32c55ep+5f, -0x1.a142f6p+6f, -0x1.10b51ap+6f, -0x1.ba94d6p+6f,   0x1.07ae1p+6f,  0x1.03bd9ap+6f,  0x1.c8e934p+5f, -0x1.14d2dcp+6f,  0x1.2743fcp+2f, -0x1.d8ef6cp+6f,   0x1.f36adp+6f,  0x1.3c9414p+1f,  0x1.76a284p+6f},
      {-0x1.bceffcp+6f, -0x1.fb1ed6p+5f,  0x1.3a50d4p+5f,  0x1.fa1b3cp+5f, -0x1.000692p+7f, -0x1.023ea6p+7f,  -0x1.805e7p+6f,  0x1.db6072p+6f, -0x1.02b35cp+7f,   0x1.43fc5p+5f, accumulator_nan,  0x1.1ae5a6p+4f, -0x1.6ad1a8p+6f, -0x1.a1b654p+5f,  0x1.118ebap+6f,  -0x1.1ebd3p+4f},
      { 0x1.5d8f04p+6f,  0x1.83318cp+6f,   0x1.fcf59p+6f,  0x1.17acdap+6f,  0x1.bc67d2p+5f,      factor_nan,  0x1.552358p+6f,             inf, -0x1.093c3ep+5f, -0x1.13f412p+6f, -0x1.dd750ap+6f,  0x1.bf9e88p+5f, -0x1.26da7ap+0f, -0x1.8d9abcp+5f,   -0x1.3606p+6f, -0x1.7b06a8p+2f},
      { 0x1.22e6fep+4f, -0x1.78fa76p+6f, -0x1.3575a4p+5f,  0x1.5db208p+6f, -0x1.981f3cp+5f,      factor_nan, -0x1.c2e948p+5f, -0x1.b834c8p+5f,  0x1.0d9e06p+5f, -0x1.5f9c68p+6f,  0x1.2d30bep+6f, -0x1.ff731ep+6f,             inf,  0x1.b9170cp+5f,  0x1.cdded6p+3f, -0x1.dbc348p+4f},
      {-0x1.0b6aa6p+6f, -0x1.82e412p+6f,  0x1.211b54p+4f,  0x1.845e62p+4f, -0x1.dcac8ap+3f,  0x1.0b0b4cp+2f,  0x1.a065c4p+3f,  0x1.288486p+5f,  0x1.f22e32p+5f, -0x1.34c46ap+3f,  0x1.2d376ep+6f, -0x1.1b608cp+5f, -0x1.b2c694p+5f,   0x1.0a87ap+5f,  0x1.5731e6p+3f,  0x1.30f4cap+6f},
      { 0x1.ecc478p+6f, -0x1.cfdaa2p+5f,  -0x1.c346ap+6f,  0x1.83d198p+6f, -0x1.dc87dcp+6f,  0x1.254104p+6f, -0x1.88bd36p+2f,            -inf,  0x1.a19d4ap+6f,  0x1.c1b232p+6f, -0x1.bebdbep+6f,  -0x1.593d3p+1f,  0x1.f785fap+6f,  0x1.cd1ebep+4f,  0x1.d30658p+6f,  0x1.714c46p+6f},
      { 0x1.310b06p+5f, -0x1.7ddd56p+2f, -0x1.ce9dacp+6f, -0x1.c8bcfcp+6f,  0x1.9d8d0ap+5f,      factor_nan,  0x1.2ca5ecp+6f,            -inf, -0x1.375084p+6f,  0x1.f075f2p+6f,  0x1.7f8986p+5f,  0x1.ef8952p+5f,  -0x1.d571bp+6f,  0x1.21af6ep+6f,  0x1.9c5a6ap+4f, -0x1.dc95f4p+5f},
      {  0x1.0cacdp+6f,   0x1.01dd3p+6f, -0x1.da9ab6p+6f,   0x1.81fccp+5f,  0x1.22e0a6p+6f,      factor_nan, -0x1.c38f2ap+6f,             inf,  0x1.287b42p+3f, -0x1.ad262ap+6f, -0x1.b5881cp+5f,  0x1.de455ap+6f,  0x1.0ab346p+5f,  0x1.8f4688p+1f, -0x1.b03d4cp+6f,  0x1.9a085ap+5f},
  };
  // clang-format on

  rocjitsu::arith::test::expect_matrix_invalid_flags(rj_arith_v_swmmac_f32_16x16x32_f16_w32, input,
                                                     12, 0, 4, 20,
                                                     63 | RJ_ARITH_SWMMAC_INDEX_KEY_1);

  // A occupies 4 registers, B 8, C/D 8. Reuse the same arithmetic
  // data for overlapping and disjoint destination ranges.
  const uint64_t modes[] = {0, RJ_ARITH_SWMMAC_INDEX_KEY_1};
  const auto cpu_levels = rocjitsu::arith::test::supported_cpu_levels();
  for (uint64_t cpu : cpu_levels) {
    for (uint64_t semantics : rocjitsu::arith::test::semantics_modes()) {
      for (uint64_t mode : modes) {
        Registers selected_input = input;
        rocjitsu::arith::test::isolate_sparse_selector<32>(selected_input.v[20], mode);
        for (int aliasing_offset : {12}) {
          SCOPED_TRACE(::testing::Message()
                       << cpu << '/' << semantics << '/' << mode << '/' << aliasing_offset);
          Registers r = selected_input;
          ASSERT_EQ(rj_arith_v_swmmac_f32_16x16x32_f16_w32(
                        cpu | semantics, mode, r.v + aliasing_offset, r.v, r.v + 4, r.v + 20),
                    RJ_ARITH_SUCCESS);
          for (int row = 0; row < 16; ++row) {
            for (int col = 0; col < 16; ++col) {
              uint32_t actual = r.v[aliasing_offset + (row % 8)][col + 16 * (row / 8)];
              if (semantics != RJ_ARITH_SEMANTICS_LOOSE) {
                EXPECT_EQ(actual, std::bit_cast<uint32_t>(golden[row][col]));
              } else {
                rocjitsu::arith::test::expect_f32(actual, golden[row][col], semantics,
                                                  /*atol=*/1e-5,
                                                  /*rtol=*/1e-5);
              }
            }
          }
          r.expect_preserved(selected_input, aliasing_offset, 8);
        }
      }
    }
  }
  rocjitsu::arith::test::expect_sparse_aliasing(rj_arith_v_swmmac_f32_16x16x32_f16_w32, input,
                                                /*outputs=*/8, /*b=*/4, /*index=*/20, {0, 2, 4, 20},
                                                cpu_levels);
}
