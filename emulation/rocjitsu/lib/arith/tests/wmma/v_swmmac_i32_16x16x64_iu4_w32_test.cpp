// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for v_swmmac_i32_16x16x64_iu4_w32.
// Keeps this instruction's matrix inputs and expected results beside its aliasing and flag
// checks.

#include "common/cpu.h"
#include "common/integer.h"
#include "common/registers.h"
#include "common/semantics.h"
#include "rj_arith/rj_arith.h"
#include "wmma/helpers.h"
#include <algorithm>

#include <gtest/gtest.h>
#include <random>
#include <stdint.h>

namespace {

using Registers = rocjitsu::arith::test::Registers<64, 32>;

} // namespace

// Check matrix results, operand aliases, and instruction-flag contracts across supported CPUs
// and semantics.
TEST(SwmmacI32Iu4K64W32, Arithmetic) {
  std::minstd_rand random;
  uint32_t a[16][32], b[64][16], c[16][16];
  for (int row = 0; row < 16; ++row) {
    for (int k = 0; k < 32; ++k) {
      a[row][k] = uint32_t(random()) >> 27;
    }
  }
  for (int k = 0; k < 64; ++k) {
    for (int col = 0; col < 16; ++col) {
      b[k][col] = uint32_t(random()) >> 27;
    }
  }
  // Accumulators near both signed limits expose wraparound and staged CLAMP.
  for (int row = 0; row < 16; ++row) {
    for (int col = 0; col < 16; ++col) {
      int64_t distance = (random() >> 16) % 128;
      c[row][col] = uint32_t((row + col) % 2 ? 2147483647LL - distance : -2147483648LL + distance);
    }
  }
  int selected[16][32];
  rocjitsu::arith::test::make_sparse_indices(random, selected);

  // A uses the dense small-element layout with compressed K coordinates.
  Registers input;
  for (int row = 0; row < 16; ++row) {
    for (int k = 0; k < 32; ++k) {
      rocjitsu::arith::test::set_wmma_gfx12_small<4, 32>(input.v, row, k, a[row][k]);
    }
  }
  for (int k = 0; k < 64; ++k) {
    for (int col = 0; col < 16; ++col) {
      rocjitsu::arith::test::set_swmmac_small_b<4, 32>(input.v + 2, col, k, b[k][col]);
    }
  }
  // Map each logical accumulator to its destination register and lane.
  for (int row = 0; row < 16; ++row) {
    for (int col = 0; col < 16; ++col) {
      input.v[6 + (row % 8)][col + 16 * (row / 8)] = c[row][col];
    }
  }
  // Each selector set encodes the same pairs, exercising KEY selection
  // without additional golden matrices. Every pair is strictly increasing.
  rocjitsu::arith::test::set_swmmac_indices<4, 32>(input.v[14], selected);

  rocjitsu::arith::test::expect_matrix_guest_controls<false, false>(
      rj_arith_v_swmmac_i32_16x16x64_iu4_w32, input, 6, 0, 2, 14);

  rocjitsu::arith::test::expect_matrix_invalid_flags(
      rj_arith_v_swmmac_i32_16x16x64_iu4_w32, input, 6, 0, 2, 14,
      RJ_ARITH_WMMA_SIGNED_A | RJ_ARITH_WMMA_SIGNED_B | RJ_ARITH_WMMA_CLAMP);

  // A occupies 2 registers, B 4, C/D 8. Reuse the same arithmetic
  // data for overlapping and disjoint destination ranges.
  const uint64_t modes[] = {0,
                            RJ_ARITH_WMMA_SIGNED_A,
                            RJ_ARITH_WMMA_SIGNED_B,
                            RJ_ARITH_WMMA_SIGNED_A | RJ_ARITH_WMMA_SIGNED_B,
                            RJ_ARITH_WMMA_CLAMP,
                            RJ_ARITH_WMMA_CLAMP | RJ_ARITH_WMMA_SIGNED_A,
                            RJ_ARITH_WMMA_CLAMP | RJ_ARITH_WMMA_SIGNED_B,
                            RJ_ARITH_WMMA_CLAMP | RJ_ARITH_WMMA_SIGNED_A | RJ_ARITH_WMMA_SIGNED_B};
  const auto cpu_levels = rocjitsu::arith::test::supported_cpu_levels();
  for (uint64_t cpu : cpu_levels) {
    for (uint64_t semantics : rocjitsu::arith::test::semantics_modes()) {
      for (uint64_t mode : modes) {
        for (int aliasing_offset : {0, 1, 2, 4, 6, 14, 15}) {
          SCOPED_TRACE(::testing::Message()
                       << cpu << '/' << semantics << '/' << mode << '/' << aliasing_offset);
          Registers r = input;
          // Exact ISA reference: wide signed sum, then clamp at each native
          // accumulation boundary, or wrap to uint32 at the final store.
          uint32_t expected[16][16];
          for (int row = 0; row < 16; ++row) {
            for (int col = 0; col < 16; ++col) {
              int64_t acc = rocjitsu::arith::test::extend_bits(
                  r.v[aliasing_offset + (row % 8)][col + 16 * (row / 8)], 32, true);
              for (int stage = 0; stage < 2; ++stage) {
                for (int i = 0; i < 16; ++i) {
                  int k = 16 * (i / 8) + 8 * stage + i % 8;
                  acc += rocjitsu::arith::test::extend_bits(a[row][k], 4,
                                                            mode & RJ_ARITH_WMMA_SIGNED_A) *
                         rocjitsu::arith::test::extend_bits(b[selected[row][k]][col], 4,
                                                            mode & RJ_ARITH_WMMA_SIGNED_B);
                }
                if (mode & RJ_ARITH_WMMA_CLAMP) {
                  acc = std::clamp(acc, int64_t(-2147483648LL), int64_t(2147483647));
                }
              }
              expected[row][col] = uint32_t(acc);
            }
          }
          ASSERT_EQ(rj_arith_v_swmmac_i32_16x16x64_iu4_w32(
                        cpu | semantics, mode, r.v + aliasing_offset, r.v, r.v + 2, r.v + 14),
                    RJ_ARITH_SUCCESS);
          for (int row = 0; row < 16; ++row) {
            for (int col = 0; col < 16; ++col) {
              uint32_t actual = r.v[aliasing_offset + (row % 8)][col + 16 * (row / 8)];
              EXPECT_EQ(actual, expected[row][col]);
            }
          }
          r.expect_preserved(input, aliasing_offset, 8);
        }
      }
    }
  }
}
