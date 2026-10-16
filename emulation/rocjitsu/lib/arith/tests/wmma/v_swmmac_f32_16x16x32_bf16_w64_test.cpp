// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for v_swmmac_f32_16x16x32_bf16_w64.
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
TEST(SwmmacF32BF16K32W64, Arithmetic) {
  rocjitsu::arith::test::expect_sparse_modifiers<true, 64, false>(
      rj_arith_v_swmmac_f32_16x16x32_bf16_w64);

  std::minstd_rand random;
  uint32_t a[16][16], b[32][16], c[16][16];
  for (int row = 0; row < 16; ++row) {
    for (int k = 0; k < 16; ++k) {
      a[row][k] = rocjitsu::arith::test::random_bf16_bits(random);
    }
  }
  for (int k = 0; k < 32; ++k) {
    for (int col = 0; col < 16; ++col) {
      b[k][col] = rocjitsu::arith::test::random_bf16_bits(random);
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
  a[1][2] = rocjitsu::arith::test::kBF16QuietNaNBits;
  a[3][4] = rocjitsu::arith::test::kBF16PositiveInfinityBits;
  b[6][5] = rocjitsu::arith::test::kBF16QuietNaNBits;
  b[8][7] = rocjitsu::arith::test::kBF16PositiveInfinityBits;
  c[9][10] = rocjitsu::arith::test::kF32QuietNaNBits;
  c[11][12] = rocjitsu::arith::test::kF32PositiveInfinityBits;

  Registers input;
  for (int row = 0; row < 16; ++row) {
    for (int k = 0; k < 16; ++k) {
      rocjitsu::arith::test::set_swmmac16_a<64>(input.v, row, k, a[row][k]);
    }
  }
  for (int k = 0; k < 32; ++k) {
    for (int col = 0; col < 16; ++col) {
      rocjitsu::arith::test::set_swmmac16_b<64>(input.v + 2, col, k, b[k][col]);
    }
  }
  // Map each logical accumulator to its destination register and lane.
  for (int row = 0; row < 16; ++row) {
    for (int col = 0; col < 16; ++col) {
      input.v[6 + ((row % 8) % 4)][col + 16 * (row / 8) + 32 * ((row % 8) / 4)] = c[row][col];
    }
  }
  // Replicate valid pairs in each metadata bank; each call below clears
  // the unused banks so a wrong selector cannot read equivalent inputs.
  std::fill_n(input.v[10], 64, 0);
  for (int row = 0; row < 16; ++row) {
    for (int k = 0; k < 16; ++k) {
      int lane = row + 16 * (k / 4), local = k % 4;
      for (int key = 0; key < 32; key += 8) {
        input.v[10][lane] |= uint32_t(selected[row][k] % 4) << (key + 2 * local);
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
      { 0x1.6aea58p+4f,   0x1.d5be6p+6f,   0x1.16498p+6f,  0x1.38a818p+2f, -0x1.746704p+6f,  -0x1.91138p+2f,  0x1.571f12p+6f,            -inf,  0x1.a667c4p+5f, -0x1.f231bcp+6f,  -0x1.659cbp+3f, -0x1.ba6f1ap+6f, -0x1.3eb358p+3f,  0x1.205b98p+4f,  0x1.6fd20cp+6f, -0x1.4e93e2p+6f},
      {     factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan,      factor_nan},
      { 0x1.4448d6p+6f,  -0x1.a8127p+5f, -0x1.964f88p+6f,  0x1.f515bcp+5f,  0x1.988ffcp+2f,   0x1.8a9e7p+0f,  -0x1.339bbp+6f,  0x1.47461cp+4f, -0x1.227214p+5f,  0x1.dbfa8ep+6f,  0x1.3f67dap+6f,  0x1.e58b94p+6f,  0x1.896a16p+6f,  0x1.d5ec72p+5f, -0x1.0b6a56p+6f, -0x1.df30acp+6f},
      {            inf,            -inf,            -inf,             inf,            -inf,      factor_nan,             inf,             inf,             inf,            -inf,            -inf,             inf,             inf,             inf,            -inf,             inf},
      {-0x1.1bdb8cp+5f,  0x1.065402p+5f,  0x1.5a6f98p+6f, -0x1.f4c6d2p+4f,  0x1.a5ce88p+5f,      factor_nan,  -0x1.6bd9dp+5f,  0x1.d1011ap+6f,   0x1.8df75p+3f,  0x1.8de134p+6f, -0x1.c64f4cp+6f, -0x1.29d2dcp+5f,  0x1.179138p+4f, -0x1.9e260ap+5f,  0x1.63ef94p+5f, -0x1.dc63bcp+6f},
      { 0x1.edaf88p+6f,   0x1.3f6e8p+5f,  0x1.e1f3f8p+5f,  0x1.b54f8cp+5f, -0x1.aa9152p+3f,      factor_nan,    0x1.9c9cp+4f,            -inf, -0x1.4f2ddep+4f,  0x1.653ca8p+6f, -0x1.e8e794p+3f,  0x1.6e0c48p+6f,  0x1.1af85cp+6f,   0x1.0d52fp+6f,  0x1.f33dacp+6f, -0x1.c8c348p+6f},
      { 0x1.0bf8a4p+6f, -0x1.8ef838p+6f,  0x1.e8998ep+5f, -0x1.178dbap+4f,  0x1.22d1f4p+6f,   0x1.fc5abp+3f,  -0x1.b03dep+6f,             inf,  0x1.fedaaap+5f, -0x1.d3170ep+6f,   0x1.8375bp+6f,  0x1.d7b48cp+6f, -0x1.f91a6ep+6f,  0x1.2a3388p+6f,   0x1.46405p+4f,  0x1.582bfap+3f},
      { 0x1.8a15cep+4f, -0x1.8ce1cep+5f,  0x1.0fc95ep+5f, -0x1.e4ab8cp+4f,  0x1.609914p+5f,      factor_nan, -0x1.d57062p+6f,             inf, -0x1.4a1a56p+6f,   0x1.ddb59p+5f, -0x1.8c55c2p+4f,  -0x1.f88adp+6f,  0x1.d9a8c6p+5f, -0x1.51156ap+3f, -0x1.f5eedep+6f,  0x1.c9b182p+6f},
      { 0x1.70d2bep+5f,  -0x1.8f681p+4f,  0x1.0ac09ap+6f, -0x1.5e96b6p+5f, -0x1.98d11ep+6f, -0x1.0b691cp+6f, -0x1.b8cfbcp+6f,  0x1.f1b9d4p+5f,  0x1.e8d8d6p+5f,  0x1.f38b54p+5f, -0x1.0859e8p+6f,  0x1.2054f8p+2f, -0x1.d13e34p+6f,  0x1.e275bcp+6f,   0x1.34e99p+2f,  0x1.733e08p+6f},
      {-0x1.a4fbf6p+6f, -0x1.01fd7ep+6f,  0x1.3b36e6p+5f,  0x1.fd7514p+5f, -0x1.09c2ecp+7f, -0x1.fba17cp+6f, -0x1.6df28ep+6f,  0x1.d2580ap+6f, -0x1.fd6668p+6f,   0x1.48135p+5f, accumulator_nan,  0x1.095d9cp+4f, -0x1.632feep+6f, -0x1.b8a72ap+5f,    0x1.126fp+6f, -0x1.0273ccp+4f},
      { 0x1.5a1d08p+6f,   0x1.8565ap+6f,  0x1.f60014p+6f,   0x1.156bap+6f,  0x1.b7f584p+5f,      factor_nan,   0x1.52b22p+6f,             inf, -0x1.25adc4p+5f, -0x1.2514a6p+6f, -0x1.ef2b68p+6f,  0x1.9ecf78p+5f,   0x1.3bc83p+0f, -0x1.9e5eb8p+5f,  -0x1.36f0ep+6f, -0x1.437588p+3f},
      { 0x1.46ef4cp+4f, -0x1.6c2c68p+6f, -0x1.47e13cp+5f,  0x1.5a6e14p+6f, -0x1.90077ap+5f,      factor_nan, -0x1.9dba34p+5f, -0x1.afe208p+5f,  0x1.1c53b4p+5f,  -0x1.5c41dp+6f,  0x1.2e37bep+6f, -0x1.e89898p+6f,             inf,  0x1.c4b1fap+5f,  0x1.5181ecp+3f, -0x1.df2b4ep+4f},
      {-0x1.1e48acp+6f, -0x1.8d8e94p+6f,  0x1.1373bap+4f,  0x1.656a2cp+4f,  -0x1.dd60ep+3f,  0x1.06e8ecp+3f,  0x1.3c3bd6p+3f,  0x1.5253bcp+5f,  0x1.c7acacp+5f, -0x1.23c14cp+2f,  0x1.240b04p+6f,  -0x1.1f7fdp+5f, -0x1.b2d138p+5f,  0x1.1f434cp+5f,  0x1.f26748p+2f,  0x1.24ee0cp+6f},
      { 0x1.04e5f2p+7f, -0x1.c6e396p+5f, -0x1.b25d06p+6f,  0x1.85e5a6p+6f, -0x1.ec0992p+6f,  0x1.2b902ep+6f,  -0x1.7b37ap+0f,            -inf,  0x1.bfc4c4p+6f,  0x1.ca545cp+6f, -0x1.ad7f76p+6f,  -0x1.cd0a1p+2f,  0x1.f31b0ap+6f,  0x1.be5616p+4f,  0x1.d4c3acp+6f,   0x1.849a6p+6f},
      { 0x1.18bc8cp+5f,  -0x1.d6859p+3f, -0x1.c904e8p+6f, -0x1.c55bf2p+6f,  0x1.bbff24p+5f,      factor_nan,  0x1.2a9c34p+6f,            -inf, -0x1.3fdc7cp+6f,  0x1.de4932p+6f,  0x1.8a7ce4p+5f,  0x1.f44fd6p+5f, -0x1.bb9b4cp+6f,  0x1.33d566p+6f,  0x1.710764p+4f,  -0x1.ae1b7p+5f},
      { 0x1.eae8fcp+5f,  0x1.fe82e2p+5f, -0x1.e64958p+6f,  0x1.8c1da4p+5f,  0x1.118da8p+6f,      factor_nan, -0x1.d0c6e4p+6f,             inf,  0x1.6fd4b8p+2f, -0x1.b2c39cp+6f, -0x1.ba4a7cp+5f,  0x1.dd3028p+6f,   0x1.14e4dp+5f,   0x1.63f8dp+0f, -0x1.a00084p+6f,  0x1.82e296p+5f},
  };
  // clang-format on

  rocjitsu::arith::test::expect_matrix_invalid_flags(
      rj_arith_v_swmmac_f32_16x16x32_bf16_w64, input, 6, 0, 2, 10,
      63 | RJ_ARITH_SWMMAC_INDEX_KEY_1 | RJ_ARITH_SWMMAC_INDEX_KEY_2);

  // A occupies 2 registers, B 4, C/D 4. Reuse the same arithmetic
  // data for overlapping and disjoint destination ranges.
  const uint64_t modes[] = {0, RJ_ARITH_SWMMAC_INDEX_KEY_1, (0) | RJ_ARITH_SWMMAC_INDEX_KEY_2,
                            RJ_ARITH_SWMMAC_INDEX_KEY_1 | RJ_ARITH_SWMMAC_INDEX_KEY_2};
  const auto cpu_levels = rocjitsu::arith::test::supported_cpu_levels();
  for (uint64_t cpu : cpu_levels) {
    for (uint64_t semantics : rocjitsu::arith::test::semantics_modes()) {
      for (uint64_t mode : modes) {
        Registers selected_input = input;
        rocjitsu::arith::test::isolate_sparse_selector<64>(selected_input.v[10], mode);
        for (int aliasing_offset : {6}) {
          SCOPED_TRACE(::testing::Message()
                       << cpu << '/' << semantics << '/' << mode << '/' << aliasing_offset);
          Registers r = selected_input;
          ASSERT_EQ(rj_arith_v_swmmac_f32_16x16x32_bf16_w64(
                        cpu | semantics, mode, r.v + aliasing_offset, r.v, r.v + 2, r.v + 10),
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
  rocjitsu::arith::test::expect_sparse_aliasing(rj_arith_v_swmmac_f32_16x16x32_bf16_w64, input,
                                                /*outputs=*/4, /*b=*/2, /*index=*/10, {0, 1, 2, 10},
                                                cpu_levels);
}
