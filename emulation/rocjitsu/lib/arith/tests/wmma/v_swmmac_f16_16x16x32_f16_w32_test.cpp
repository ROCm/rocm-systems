// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for v_swmmac_f16_16x16x32_f16_w32.
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

// ISA operand packing, independent of the arithmetic oracle. Gfx11 repeats
// logical input values across lane groups; gfx12 partitions K between groups.
void set_a(uint32_t *const *v, int coordinate, int k, uint32_t bits) {
  auto &word = v[2 * (k / 8) + ((k / 2) & 1)][coordinate + 16 * ((k / 4) & 1)];
  int shift = 16 * (k % 2);
  word = (word & ~(0xffffU << shift)) | (bits << shift);
}

// Place one B element at its matrix coordinate in this sparse WMMA register layout.
void set_b(uint32_t *const *v, int coordinate, int k, uint32_t bits) {
  auto &word = v[4 * (k / 16) + (k / 2) % 4][coordinate + 16 * ((k / 8) & 1)];
  int shift = 16 * (k % 2);
  word = (word & ~(0xffffU << shift)) | (bits << shift);
}

} // namespace

// Check matrix results, operand aliases, and instruction-flag contracts across supported CPUs
// and semantics.
TEST(SwmmacF16F16K32W32, Arithmetic) {
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
      c[row][col] = rocjitsu::arith::test::random_f16_bits(random);
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
  c[9][10] = rocjitsu::arith::test::kF16QuietNaNBits;
  c[11][12] = rocjitsu::arith::test::kF16PositiveInfinityBits;

  Registers input;
  for (int row = 0; row < 16; ++row) {
    for (int k = 0; k < 16; ++k) {
      set_a(input.v, row, k, a[row][k]);
    }
  }
  for (int k = 0; k < 32; ++k) {
    for (int col = 0; col < 16; ++col) {
      set_b(input.v + 4, col, k, b[k][col]);
    }
  }
  // C/D layout: register and lane select a row/column; packed gfx12
  // stores adjacent rows in halves, while gfx11 OPSEL preserves the other half.
  for (int row = 0; row < 16; ++row) {
    for (int col = 0; col < 16; ++col) {
      auto &word = input.v[12 + ((row % 8) / 2)][col + 16 * (row / 8)];
      int shift = 16 * ((row % 8) % 2);
      word = (word & ~(0xffffU << shift)) | (c[row][col] << shift);
    }
  }
  // Each selector set encodes the same pairs, exercising KEY selection
  // without additional golden matrices. Every pair is strictly increasing.
  std::fill_n(input.v[16], 32, 0);
  for (int row = 0; row < 16; ++row) {
    for (int k = 0; k < 16; ++k) {
      int lane = row + 16 * ((k / 4) & 1), local = 2 * ((k / 2) & 1) + 4 * (k / 8) + k % 2;
      for (int key = 0; key < 32; key += 16) {
        input.v[16][lane] |= uint32_t(selected[row][k] % 4) << (key + 2 * local);
      }
    }
  }

  // Offline goldens from rocjitsu ad2bb973ec4's original DOT models.
  // Reference computation uses logical matrices, independently of the VGPR
  // packing below, and does not call rj-arith. NaNs are compared by classification.
  constexpr float nan = std::numeric_limits<float>::quiet_NaN();
  constexpr float inf = std::numeric_limits<float>::infinity();
  // clang-format off
  const float golden[16][16] = {
      {-0x1.5ecp-2f, -0x1.d3cp+2f,  -0x1.92p-1f,  0x1.534p+1f, -0x1.7e8p+1f,  0x1.844p+1f,  0x1.a84p+1f,         -inf,  0x1.a74p+0f, -0x1.284p+0f, -0x1.078p+2f,  0x1.4a4p+0f,  0x1.56cp+0f,  0x1.144p-3f, -0x1.fb8p+0f,  0x1.da8p+0f},
      {         nan,          nan,          nan,          nan,          nan,          nan,          nan,          nan,          nan,          nan,          nan,          nan,          nan,          nan,          nan,          nan},
      {  0x1.34p+2f, -0x1.d34p+0f,  0x1.338p+2f,  0x1.844p+2f,  -0x1.43p+1f,  0x1.84cp+1f,   0x1.59p+0f,  0x1.2b4p+1f, -0x1.7c8p+0f, -0x1.ec8p+0f, -0x1.ad8p+1f,   0x1.34p-1f,  0x1.258p+0f,  0x1.468p+2f, -0x1.554p+0f,  0x1.714p+2f},
      {         inf,         -inf,         -inf,          inf,         -inf,          nan,          inf,          inf,          inf,         -inf,         -inf,          inf,          inf,          inf,         -inf,          inf},
      {  0x1.21p-1f,  0x1.a88p-1f,   -0x1.ap+0f, -0x1.838p+1f,  0x1.358p+0f,          nan,   0x1.35p+0f, -0x1.9b4p-1f, -0x1.6ccp+0f, -0x1.1a4p+2f,  -0x1.83p-5f,  0x1.074p+0f,  0x1.294p+1f,  0x1.2f4p+1f,   -0x1.4p+1f,  0x1.b74p-1f},
      { -0x1.78p+1f,   0x1.3ap-6f, -0x1.53cp-3f,  -0x1.3fp+0f,  0x1.078p+0f,          nan, -0x1.388p+3f,         -inf, -0x1.84cp+1f, -0x1.02cp+2f, -0x1.2d4p+2f,  0x1.0dcp+2f, -0x1.1b4p+3f, -0x1.c08p+0f,  0x1.9a4p-2f,   0x1.2bp-1f},
      {-0x1.bd4p+1f, -0x1.3c4p-1f,  -0x1.37p+2f,   0x1.11p+2f, -0x1.efcp+1f, -0x1.62cp-4f,  0x1.7b4p+1f,          inf, -0x1.9d8p+2f,  0x1.288p+0f, -0x1.af4p+0f, -0x1.0c8p+2f, -0x1.3e8p+1f,  0x1.014p+1f,  0x1.344p-1f, -0x1.934p+1f},
      {-0x1.88cp-1f,  0x1.6b4p+2f, -0x1.00cp+1f, -0x1.01cp+1f,   0x1.21p+0f,          nan,  0x1.458p-1f,          inf,   0x1.bfp-1f,  -0x1.4ap+1f, -0x1.6f8p+0f, -0x1.eacp+0f,  0x1.3b4p+1f, -0x1.214p+1f, -0x1.ce4p+1f,  0x1.688p+1f},
      { -0x1.31p-1f,  0x1.938p-2f,   0x1.21p-5f,  0x1.894p+0f, -0x1.bb4p+0f, -0x1.844p-1f, -0x1.3b4p-1f, -0x1.028p+1f, -0x1.9ecp+0f,  0x1.b24p-1f,  0x1.55cp+0f,  0x1.8c8p-4f, -0x1.64cp+1f,   0x1.b3p+1f,  -0x1.0dp-2f, -0x1.218p+0f},
      {-0x1.494p+1f,   0x1.09p+2f,  0x1.5b4p-3f,  -0x1.91p+0f, -0x1.174p+0f, -0x1.bbcp+2f, -0x1.5ccp+2f,  0x1.02cp-1f, -0x1.69cp+1f,   0x1.0cp+1f,          nan,  -0x1.59p+2f, -0x1.804p+1f,  0x1.2a8p+2f,  -0x1.89p+1f, -0x1.4ccp+1f},
      { 0x1.a34p+2f,  0x1.3acp+1f,  0x1.09cp+3f,  0x1.d04p-1f,  -0x1.88p+2f,          nan,  0x1.46cp+1f,          inf,   0x1.71p+2f,  0x1.018p+2f,   0x1.2dp+2f,  0x1.31cp+1f, -0x1.7e8p+0f,  0x1.06cp+2f, -0x1.2a4p-8f,  0x1.968p+2f},
      {-0x1.488p-2f,   0x1.45p-1f,   0x1.4ap+2f, -0x1.348p+1f,  0x1.fe8p+0f,          nan,  -0x1.96p+1f, -0x1.a68p-2f, -0x1.3f8p+1f,  0x1.034p-1f, -0x1.4fcp+1f, -0x1.6a4p+1f,          inf,  -0x1.f3p+1f,  0x1.00cp+1f,  0x1.d28p+0f},
      { 0x1.decp+2f,  0x1.184p+2f, -0x1.218p-3f, -0x1.4ccp+0f, -0x1.efcp+0f, -0x1.8b4p+2f,  0x1.cc8p-1f, -0x1.db8p+2f,  0x1.228p+2f, -0x1.d3cp-2f,  0x1.158p+1f,   0x1.ecp+1f,  0x1.28cp+0f,   0x1.cap-1f, -0x1.34cp+2f,   0x1.37p+2f},
      {-0x1.318p+1f,  0x1.65cp-2f, -0x1.0d8p+0f, -0x1.e54p-2f,  0x1.598p+2f,  0x1.148p+0f, -0x1.5ccp+1f,         -inf, -0x1.184p+3f,  -0x1.acp+1f,  -0x1.67p+1f,  0x1.52cp+2f, -0x1.8d4p+0f, -0x1.3fcp+1f, -0x1.e14p-1f, -0x1.714p+2f},
      { 0x1.21cp+1f,   0x1.1fp-1f,  0x1.0f4p+1f,  0x1.508p-2f, -0x1.0c8p+2f,          nan,  0x1.cbcp+2f,         -inf,  0x1.404p+3f,  0x1.078p+2f, -0x1.358p-3f, -0x1.478p+1f, -0x1.348p+3f, -0x1.f88p+2f, -0x1.fe8p-1f, -0x1.a8cp+0f},
      { 0x1.96cp+2f,  -0x1.5fp-1f,  0x1.774p+1f,  0x1.38cp+0f,   0x1.2ap+0f,          nan,  0x1.c0cp+1f,          inf, -0x1.eecp-2f,      0x1p+1f,  0x1.ab8p+0f,  -0x1.27p+0f, -0x1.138p-1f,  0x1.208p+1f, -0x1.3a4p-4f,  0x1.3dcp+0f},
  };
  // clang-format on

  rocjitsu::arith::test::expect_matrix_invalid_flags(rj_arith_v_swmmac_f16_16x16x32_f16_w32, input,
                                                     12, 0, 4, 16,
                                                     63 | RJ_ARITH_SWMMAC_INDEX_KEY_1);

  // A occupies 4 registers, B 8, C/D 4. Reuse the same arithmetic
  // data for overlapping and disjoint destination ranges.
  const uint64_t modes[] = {0, RJ_ARITH_SWMMAC_INDEX_KEY_1};
  const auto cpu_levels = rocjitsu::arith::test::supported_cpu_levels();
  for (uint64_t cpu : cpu_levels) {
    for (uint64_t semantics : rocjitsu::arith::test::semantics_modes()) {
      for (uint64_t mode : modes) {
        for (int aliasing_offset : {12}) {
          SCOPED_TRACE(::testing::Message()
                       << cpu << '/' << semantics << '/' << mode << '/' << aliasing_offset);
          Registers r = input;
          ASSERT_EQ(rj_arith_v_swmmac_f16_16x16x32_f16_w32(
                        cpu | semantics, mode, r.v + aliasing_offset, r.v, r.v + 4, r.v + 16),
                    RJ_ARITH_SUCCESS);
          for (int row = 0; row < 16; ++row) {
            for (int col = 0; col < 16; ++col) {
              uint32_t actual = r.v[aliasing_offset + ((row % 8) / 2)][col + 16 * (row / 8)];
              float value =
                  rocjitsu::arith::test::f16_value(uint16_t(actual >> (16 * ((row % 8) % 2))));
              rocjitsu::arith::test::expect_f32(rocjitsu::arith::test::bit_cast<uint32_t>(value),
                                                golden[row][col], semantics,
                                                /*atol=*/1e-5, /*rtol=*/5e-4);
            }
          }
          r.expect_preserved(input, aliasing_offset, 4);
        }
      }
    }
  }
  // Tied C/D makes an overlapping destination change the initial accumulator.
  // Compare each alias with a disjoint snapshot of those same effective inputs;
  // this supplements (and does not replace) the independent goldens above.
  for (uint64_t cpu : cpu_levels) {
    for (uint64_t semantics : {RJ_ARITH_SEMANTICS_LOOSE, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL}) {
      for (int aliasing_offset : {0, 2, 4, 16}) {
        Registers r = input, reference = input;
        for (int reg = 0; reg < 4; ++reg) {
          for (int lane = 0; lane < 32; ++lane) {
            reference.v[48 + reg][lane] = r.v[aliasing_offset + reg][lane];
          }
        }
        ASSERT_EQ(rj_arith_v_swmmac_f16_16x16x32_f16_w32(cpu | semantics, 0, reference.v + 48,
                                                         reference.v, reference.v + 4,
                                                         reference.v + 16),
                  RJ_ARITH_SUCCESS);
        ASSERT_EQ(rj_arith_v_swmmac_f16_16x16x32_f16_w32(cpu | semantics, 0, r.v + aliasing_offset,
                                                         r.v, r.v + 4, r.v + 16),
                  RJ_ARITH_SUCCESS);
        for (int reg = 0; reg < 4; ++reg) {
          for (int lane = 0; lane < 32; ++lane) {
            EXPECT_EQ(r.v[aliasing_offset + reg][lane], reference.v[48 + reg][lane]);
          }
        }
        r.expect_preserved(input, aliasing_offset, 4);
      }
    }
  }
}
