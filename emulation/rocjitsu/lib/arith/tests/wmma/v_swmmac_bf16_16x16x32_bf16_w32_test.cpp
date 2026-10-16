// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for v_swmmac_bf16_16x16x32_bf16_w32.
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
TEST(SwmmacBF16BF16K32W32, Arithmetic) {
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
      c[row][col] = rocjitsu::arith::test::random_bf16_bits(random);
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
  c[9][10] = rocjitsu::arith::test::kBF16QuietNaNBits;
  c[11][12] = rocjitsu::arith::test::kBF16PositiveInfinityBits;

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
      {-0x1.9ap+0f, -0x1.6cp+2f, -0x1.8ap+2f,   0x1.2p+2f,  0x1.8cp-2f,  0x1.2cp+3f,  0x1.1ep+2f,        -inf,  -0x1.8p+1f, -0x1.3ep+2f, -0x1.4cp+2f,  0x1.76p+1f, -0x1.fep+0f,  0x1.52p+1f,  0x1.4ep+1f, -0x1.86p+2f},
      {        nan,         nan,         nan,         nan,         nan,         nan,         nan,         nan,         nan,         nan,         nan,         nan,         nan,         nan,         nan,         nan},
      { 0x1.1ep+1f,  0x1.3ap+1f,  0x1.1ep-2f,  -0x1.9p+0f, -0x1.8ep+2f, -0x1.06p+1f, -0x1.a4p+1f, -0x1.ecp-2f,  0x1.bap+0f, -0x1.d4p+0f, -0x1.24p+2f,  0x1.34p+1f, -0x1.c8p+0f,  -0x1.6p-1f, -0x1.64p+1f,   0x1.bp+0f},
      {        inf,        -inf,        -inf,         inf,        -inf,         nan,         inf,         inf,         inf,        -inf,        -inf,         inf,         inf,         inf,        -inf,         inf},
      { 0x1.24p+2f, -0x1.bap+1f, -0x1.6ap+1f, -0x1.ccp+0f,   0x1.9p+2f,         nan,  0x1.b2p-2f,  0x1.a8p-1f, -0x1.d2p-2f, -0x1.08p+3f, -0x1.cap+0f,  0x1.3cp+1f,  0x1.14p+2f,  0x1.7ap+2f, -0x1.88p+1f, -0x1.f4p-1f},
      {-0x1.5ap+1f,  0x1.b2p+0f,  0x1.0ep-3f,  0x1.5ep+2f, -0x1.d8p+0f,         nan, -0x1.5ep+2f,        -inf,  -0x1.7p+0f, -0x1.9ap+2f,  0x1.cap-2f,  0x1.04p+1f,  -0x1.ep+1f, -0x1.36p+1f,  -0x1.bp+1f, -0x1.1ap-1f},
      { 0x1.74p+2f,   0x1.ep+0f, -0x1.9ep+0f,  0x1.84p+2f, -0x1.8ap+2f,  0x1.acp+1f,  0x1.dcp+0f,         inf, -0x1.34p+2f,  0x1.08p+0f,  0x1.d4p+0f, -0x1.1cp+1f,  0x1.4ep+0f, -0x1.3ep+0f, -0x1.e4p+0f,  0x1.6cp-2f},
      {-0x1.04p+1f, -0x1.8cp+0f,  0x1.6ep-1f,  0x1.68p+0f, -0x1.26p-1f,         nan,  0x1.c2p-3f,         inf, -0x1.e6p-1f, -0x1.0ap-1f,  0x1.2ep+0f,  0x1.c8p+0f,   0x1.6p+1f,  0x1.fap-1f,  0x1.96p-2f,  0x1.f6p+0f},
      {-0x1.1cp+2f,  0x1.16p-1f,  0x1.86p+1f, -0x1.0ap+2f, -0x1.f4p-3f, -0x1.9cp-1f, -0x1.a8p-3f, -0x1.b4p+2f, -0x1.5cp+2f,  0x1.44p+2f,  0x1.eap+1f,  0x1.0ap-4f, -0x1.1ep+1f, -0x1.4cp-1f,  0x1.02p+1f, -0x1.8ap+0f},
      { 0x1.12p+2f,  0x1.b8p+1f,  0x1.6ap-2f, -0x1.54p+0f, -0x1.56p+2f, -0x1.3ap+2f,  0x1.8ep-3f, -0x1.a6p+1f,  0x1.2cp-2f,  0x1.68p+1f,         nan, -0x1.98p+2f,  -0x1.4p+0f,   0x1.4p+1f, -0x1.76p+1f, -0x1.22p-1f},
      { 0x1.5cp+2f,  0x1.d8p+1f,  0x1.aap+2f,   0x1.2p-2f, -0x1.72p+2f,         nan,   0x1.ep+0f,         inf,  0x1.88p+1f,   0x1.dp-3f,  0x1.94p-3f, -0x1.0ep+1f,  0x1.acp-1f,   0x1.bp+0f,  -0x1.6p-4f,  0x1.e8p+1f},
      { 0x1.eap+0f,  0x1.1cp+2f,  0x1.66p+1f,  -0x1.ap+1f,  0x1.f8p+1f,         nan,  0x1.04p+1f,  0x1.16p-5f, -0x1.26p+1f,  0x1.f6p-1f, -0x1.8ep+0f,  0x1.a8p+1f,         inf, -0x1.d8p+0f, -0x1.e6p+0f,  0x1.c2p+0f},
      { 0x1.3ep+1f,  0x1.3cp+1f, -0x1.d8p-1f, -0x1.8ap+1f, -0x1.6ap+0f, -0x1.58p+1f, -0x1.02p+1f, -0x1.2cp+1f,  -0x1.7p-1f,  0x1.38p+2f,  0x1.4ap-3f,  0x1.c2p+1f,  0x1.1cp+0f,  0x1.b2p+1f, -0x1.96p+2f,  0x1.c6p+0f},
      { 0x1.6ap+2f,  0x1.38p+0f,  0x1.9ap+1f,  0x1.b2p-1f,  0x1.58p+1f,  0x1.a6p+0f,  0x1.18p-1f,        -inf, -0x1.b4p-3f, -0x1.fcp+0f,  0x1.0ep+1f,  0x1.f8p-2f, -0x1.86p+1f, -0x1.d2p+1f, -0x1.02p+0f, -0x1.48p+1f},
      { -0x1.cp+0f, -0x1.16p+3f,  0x1.9cp+1f,  0x1.46p-3f, -0x1.a2p-1f,         nan,  0x1.5ap+2f,        -inf,  0x1.e4p+2f, -0x1.02p+0f,  0x1.48p+1f, -0x1.02p+1f,  -0x1.9p+1f, -0x1.fcp+0f, -0x1.8ap+1f,  0x1.fep+1f},
      {-0x1.02p-3f, -0x1.acp+0f,  0x1.f6p-9f,  0x1.54p+1f, -0x1.a8p+1f,         nan, -0x1.2ep-1f,         inf, -0x1.f6p+1f,  0x1.acp-1f,  0x1.6ep-1f,  -0x1.1p+0f,  0x1.d6p-1f,  0x1.3ep-2f,  0x1.f2p+1f, -0x1.8cp+1f},
  };
  // clang-format on

  rocjitsu::arith::test::expect_matrix_invalid_flags(rj_arith_v_swmmac_bf16_16x16x32_bf16_w32,
                                                     input, 12, 0, 4, 16,
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
          ASSERT_EQ(rj_arith_v_swmmac_bf16_16x16x32_bf16_w32(
                        cpu | semantics, mode, r.v + aliasing_offset, r.v, r.v + 4, r.v + 16),
                    RJ_ARITH_SUCCESS);
          for (int row = 0; row < 16; ++row) {
            for (int col = 0; col < 16; ++col) {
              uint32_t actual = r.v[aliasing_offset + ((row % 8) / 2)][col + 16 * (row / 8)];
              float value =
                  rocjitsu::arith::test::bf16_value(uint16_t(actual >> (16 * ((row % 8) % 2))));
              rocjitsu::arith::test::expect_f32(rocjitsu::arith::test::bit_cast<uint32_t>(value),
                                                golden[row][col], semantics,
                                                /*atol=*/2e-4, /*rtol=*/4e-3);
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
        ASSERT_EQ(rj_arith_v_swmmac_bf16_16x16x32_bf16_w32(cpu | semantics, 0, reference.v + 48,
                                                           reference.v, reference.v + 4,
                                                           reference.v + 16),
                  RJ_ARITH_SUCCESS);
        ASSERT_EQ(rj_arith_v_swmmac_bf16_16x16x32_bf16_w32(
                      cpu | semantics, 0, r.v + aliasing_offset, r.v, r.v + 4, r.v + 16),
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
