// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for v_wmma_i32_16x16x16_iu8_w64_gfx12.
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

using Registers = rocjitsu::arith::test::Registers<64, 64>;

} // namespace

// Check matrix results, operand aliases, and instruction-flag contracts across supported CPUs
// and semantics.
TEST(WmmaI32Iu8W64Gfx12, Arithmetic) {
  std::minstd_rand random;
  uint32_t a[16][16], b[16][16], c[16][16];
  for (int row = 0; row < 16; ++row) {
    for (int k = 0; k < 16; ++k) {
      a[row][k] = uint32_t(random()) >> 23;
    }
  }
  for (int k = 0; k < 16; ++k) {
    for (int col = 0; col < 16; ++col) {
      b[k][col] = uint32_t(random()) >> 23;
    }
  }
  // Accumulators near both signed limits expose wraparound and staged CLAMP.
  for (int row = 0; row < 16; ++row) {
    for (int col = 0; col < 16; ++col) {
      int64_t distance = (random() >> 16) % 128;
      c[row][col] = uint32_t((row + col) % 2 ? 2147483647LL - distance : -2147483648LL + distance);
    }
  }

  Registers input;
  for (int row = 0; row < 16; ++row) {
    for (int k = 0; k < 16; ++k) {
      rocjitsu::arith::test::set_wmma_gfx12_small<8, 64>(input.v, row, k, a[row][k]);
    }
  }
  for (int k = 0; k < 16; ++k) {
    for (int col = 0; col < 16; ++col) {
      rocjitsu::arith::test::set_wmma_gfx12_small<8, 64>(input.v + 1, col, k, b[k][col]);
    }
  }
  // Map each logical accumulator to its destination register and lane.
  for (int row = 0; row < 16; ++row) {
    for (int col = 0; col < 16; ++col) {
      input.v[2 + (row % 4)][col + 16 * (row / 8) + 32 * ((row / 4) % 2)] = c[row][col];
    }
  }

  rocjitsu::arith::test::expect_matrix_guest_controls<false, false>(
      rj_arith_v_wmma_i32_16x16x16_iu8_w64_gfx12, input, 7, 0, 1, 2);

  rocjitsu::arith::test::expect_matrix_invalid_flags(
      rj_arith_v_wmma_i32_16x16x16_iu8_w64_gfx12, input, 7, 0, 1, 2,
      RJ_ARITH_WMMA_SIGNED_A | RJ_ARITH_WMMA_SIGNED_B | RJ_ARITH_WMMA_CLAMP);

  // A occupies 1 registers, B 1, C/D 4. Reuse the same arithmetic
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
        for (int aliasing_offset : {0, 1, 2, 7}) {
          SCOPED_TRACE(::testing::Message()
                       << cpu << '/' << semantics << '/' << mode << '/' << aliasing_offset);
          Registers r = input;
          // Exact ISA reference: wide signed sum, then clamp at each native
          // accumulation boundary, or wrap to uint32 at the final store.
          uint32_t expected[16][16];
          for (int row = 0; row < 16; ++row) {
            for (int col = 0; col < 16; ++col) {
              int64_t acc = rocjitsu::arith::test::extend_bits(c[row][col], 32, true);
              for (int stage = 0; stage < 2; ++stage) {
                for (int i = 0; i < 8; ++i) {
                  int k = 16 * (i / 8) + 8 * stage + i % 8;
                  acc += rocjitsu::arith::test::extend_bits(a[row][k], 8,
                                                            mode & RJ_ARITH_WMMA_SIGNED_A) *
                         rocjitsu::arith::test::extend_bits(b[k][col], 8,
                                                            mode & RJ_ARITH_WMMA_SIGNED_B);
                }
                if (mode & RJ_ARITH_WMMA_CLAMP) {
                  acc = std::clamp(acc, int64_t(-2147483648LL), int64_t(2147483647));
                }
              }
              expected[row][col] = uint32_t(acc);
            }
          }
          ASSERT_EQ(rj_arith_v_wmma_i32_16x16x16_iu8_w64_gfx12(
                        cpu | semantics, mode, r.v + aliasing_offset, r.v, r.v + 1, r.v + 2),
                    RJ_ARITH_SUCCESS);
          for (int row = 0; row < 16; ++row) {
            for (int col = 0; col < 16; ++col) {
              uint32_t actual =
                  r.v[aliasing_offset + (row % 4)][col + 16 * (row / 8) + 32 * ((row / 4) % 2)];
              EXPECT_EQ(actual, expected[row][col]);
            }
          }
          r.expect_preserved(input, aliasing_offset, 4);
        }
      }
    }
  }
}
