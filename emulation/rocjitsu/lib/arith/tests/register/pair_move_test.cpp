// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Pair moves test selected words, masked lanes and overlap in both directions.
#include "common/cpu.h"
#include "common/registers.h"
#include "common/semantics.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_register.h"
#include <random>

namespace {
// Exercise both words using one deterministic register bank for all CPU paths.
template <int Lanes, bool Packed, typename Function> void check(Function function) {
  using rocjitsu::arith::test::Registers;
  Registers<6, Lanes> input;
  std::minstd_rand rng;
  for (int reg = 0; reg < 6; ++reg) {
    for (int lane = 0; lane < Lanes; ++lane) {
      input.v[reg][lane] = (uint32_t(rng()) >> 15) | (uint32_t(rng()) >> 15 << 16);
    }
  }
  for (auto cpu : rocjitsu::arith::test::supported_cpu_levels()) {
    for (auto semantics : rocjitsu::arith::test::semantics_modes()) {
      for (int destination : {0, 1, 3, 4}) {
        for (int selection = 0; selection < (Packed ? 4 : 1); ++selection) {
          for (uint64_t exec_mask : {0ULL, 0xaaaaaaaaaaaaaaaaULL, ~0ULL}) {
            auto actual = input;
            auto expected = input;
            uint64_t mode = (selection & 1 ? RJ_ARITH_PK_LO_A_HIGH : 0) |
                            (selection & 2 ? RJ_ARITH_PK_LO_B_HIGH : 0);
            for (int lane = 0; lane < Lanes; ++lane) {
              if ((exec_mask >> lane) & 1) {
                expected.v[destination][lane] = input.v[Packed ? selection & 1 : 0][lane];
                expected.v[destination + 1][lane] =
                    input.v[Packed ? 2 + (selection >> 1) : 1][lane];
              }
            }
            int error;
            if constexpr (Packed) {
              error = function(cpu | semantics, exec_mask, mode, actual.v + destination, actual.v,
                               actual.v + 2);
            } else {
              error = function(cpu | semantics, exec_mask, mode, actual.v + destination, actual.v);
            }
            ASSERT_EQ(error, RJ_ARITH_SUCCESS);
            for (int word = 0; word < 2; ++word) {
              for (int lane = 0; lane < Lanes; ++lane) {
                EXPECT_EQ(actual.v[destination + word][lane], expected.v[destination + word][lane]);
              }
            }
            actual.expect_preserved(input, destination, 2);
          }
        }
      }
    }
  }
}
} // namespace
TEST(PairMove, Mov64Wave32) { check<32, false>(rj_arith_v_mov_b64_w32); }
TEST(PairMove, Mov64Wave64) { check<64, false>(rj_arith_v_mov_b64_w64); }
TEST(PairMove, PackedWave32) { check<32, true>(rj_arith_v_pk_mov_b32_w32); }
TEST(PairMove, PackedWave64) { check<64, true>(rj_arith_v_pk_mov_b32_w64); }
