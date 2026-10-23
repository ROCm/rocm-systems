// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for v_swmmac_i32_16x16x32_iu8_w32.
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
TEST(SwmmacI32Iu8K32W32, Arithmetic) {
  std::minstd_rand random;
  uint32_t a[16][16], b[32][16], c[16][16];
  for (int row = 0; row < 16; ++row) {
    for (int k = 0; k < 16; ++k) {
      a[row][k] = uint32_t(random()) >> 23;
    }
  }
  for (int k = 0; k < 32; ++k) {
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
  int selected[16][16];
  rocjitsu::arith::test::make_sparse_indices(random, selected);

  Registers input;
  for (int row = 0; row < 16; ++row) {
    for (int k = 0; k < 16; ++k) {
      rocjitsu::arith::test::set_swmmac8_a<32>(input.v, row, k, a[row][k]);
    }
  }
  for (int k = 0; k < 32; ++k) {
    for (int col = 0; col < 16; ++col) {
      rocjitsu::arith::test::set_swmmac8_b<32>(input.v + 2, col, k, b[k][col]);
    }
  }
  // Map each logical accumulator to its destination register and lane.
  for (int row = 0; row < 16; ++row) {
    for (int col = 0; col < 16; ++col) {
      input.v[6 + (row % 8)][col + 16 * (row / 8)] = c[row][col];
    }
  }
  // Replicate valid pairs in each metadata bank; clear unused banks per call
  // so a wrong selector cannot read equivalent inputs.
  std::fill_n(input.v[14], 32, 0);
  for (int row = 0; row < 16; ++row) {
    for (int k = 0; k < 16; ++k) {
      int lane = row + 16 * ((k / 8) & 1), local = 8 * (k / 16) + k % 8;
      for (int key = 0; key < 32; key += 16) {
        input.v[14][lane] |= uint32_t(selected[row][k] % 4) << (key + 2 * local);
      }
    }
  }

  rocjitsu::arith::test::expect_matrix_invalid_flags(
      rj_arith_v_swmmac_i32_16x16x32_iu8_w32, input, 6, 0, 2, 14,
      RJ_ARITH_WMMA_SIGNED_A | RJ_ARITH_WMMA_SIGNED_B | RJ_ARITH_WMMA_CLAMP |
          RJ_ARITH_SWMMAC_INDEX_KEY_1);

  // A occupies 2 registers, B 4, C/D 8. Reuse the same arithmetic
  // data for overlapping and disjoint destination ranges.
  const uint64_t modes[] = {
      0,
      RJ_ARITH_WMMA_SIGNED_A,
      RJ_ARITH_WMMA_SIGNED_B,
      RJ_ARITH_WMMA_SIGNED_A | RJ_ARITH_WMMA_SIGNED_B,
      RJ_ARITH_WMMA_CLAMP,
      RJ_ARITH_WMMA_CLAMP | RJ_ARITH_WMMA_SIGNED_A,
      RJ_ARITH_WMMA_CLAMP | RJ_ARITH_WMMA_SIGNED_B,
      RJ_ARITH_WMMA_CLAMP | RJ_ARITH_WMMA_SIGNED_A | RJ_ARITH_WMMA_SIGNED_B,
      (RJ_ARITH_WMMA_CLAMP | RJ_ARITH_WMMA_SIGNED_A | RJ_ARITH_WMMA_SIGNED_B) |
          RJ_ARITH_SWMMAC_INDEX_KEY_1};
  const auto cpu_levels = rocjitsu::arith::test::supported_cpu_levels();
  for (uint64_t cpu : cpu_levels) {
    for (uint64_t semantics : rocjitsu::arith::test::semantics_modes()) {
      for (uint64_t mode : modes) {
        Registers selected_input = input;
        rocjitsu::arith::test::isolate_sparse_selector<32, 16>(selected_input.v[14], mode);
        for (int aliasing_offset : {0, 1, 2, 4, 6, 14, 15}) {
          SCOPED_TRACE(::testing::Message()
                       << cpu << '/' << semantics << '/' << mode << '/' << aliasing_offset);
          Registers r = selected_input;
          // Exact ISA reference: wide signed sum, then clamp at each native
          // accumulation boundary, or wrap to uint32 at the final store.
          uint32_t expected[16][16];
          for (int row = 0; row < 16; ++row) {
            for (int col = 0; col < 16; ++col) {
              int64_t acc = rocjitsu::arith::test::extend_bits(
                  r.v[aliasing_offset + (row % 8)][col + 16 * (row / 8)], 32, true);
              for (int stage = 0; stage < 2; ++stage) {
                for (int i = 0; i < 8; ++i) {
                  int k = 16 * (i / 8) + 8 * stage + i % 8;
                  acc += rocjitsu::arith::test::extend_bits(a[row][k], 8,
                                                            mode & RJ_ARITH_WMMA_SIGNED_A) *
                         rocjitsu::arith::test::extend_bits(b[selected[row][k]][col], 8,
                                                            mode & RJ_ARITH_WMMA_SIGNED_B);
                }
                if (mode & RJ_ARITH_WMMA_CLAMP) {
                  acc = std::clamp(acc, int64_t(-2147483648LL), int64_t(2147483647));
                }
              }
              expected[row][col] = uint32_t(acc);
            }
          }
          ASSERT_EQ(rj_arith_v_swmmac_i32_16x16x32_iu8_w32(
                        cpu | semantics, mode, r.v + aliasing_offset, r.v, r.v + 2, r.v + 14),
                    RJ_ARITH_SUCCESS);
          for (int row = 0; row < 16; ++row) {
            for (int col = 0; col < 16; ++col) {
              uint32_t actual = r.v[aliasing_offset + (row % 8)][col + 16 * (row / 8)];
              EXPECT_EQ(actual, expected[row][col]);
            }
          }
          r.expect_preserved(selected_input, aliasing_offset, 8);
        }
      }
    }
  }
}
