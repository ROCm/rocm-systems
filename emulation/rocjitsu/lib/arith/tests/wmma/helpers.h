// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Dense WMMA operand layouts and shared arithmetic, aliasing, and invalid-flag checks.

#pragma once

#include "common/contracts.h"
#include "common/cpu.h"
#include "common/floating_point.h"
#include "common/registers.h"
#include "common/semantics.h"
#include "rj_arith/rj_arith.h"

#include <algorithm>
#include <bit>
#include <gtest/gtest.h>
#include <initializer_list>
#include <random>
#include <stdint.h>

namespace rocjitsu::arith::test {

// Check invalid flags with real storage (all preserved) and null operands
// (rejected before access). Offsets identify the dense WMMA D/A/B/C operands.
template <typename Call, int RegistersCount, int Lanes>
void expect_matrix_invalid_flags(Call call, const Registers<RegistersCount, Lanes> &input, int d,
                                 int a, int b, int c, uint64_t supported_modifiers) {
  expect_invalid_flags(
      [&](uint64_t flags, uint64_t modifiers) {
        auto r = input;
        int result = call(flags, modifiers, r.v + d, r.v + a, r.v + b, r.v + c);
        r.expect_preserved(input, 0, 0);
        EXPECT_EQ(call(flags, modifiers, nullptr, nullptr, nullptr, nullptr),
                  RJ_ARITH_ERROR_INVALID_FLAGS);
        return result;
      },
      supported_modifiers);
}

// Write an A/B element in the gfx11 dense WMMA layout. Coordinate is the
// A row or B column; each 16-lane group repeats the complete K vector.
template <int Bits, int Lanes>
void set_wmma_gfx11(uint32_t *const *v, int coordinate, int k, uint32_t bits) {
  static_assert(Bits == 4 || Bits == 8 || Bits == 16);
  static_assert(Lanes == 32 || Lanes == 64);
  for (int group = 0; group < Lanes / 16; ++group) {
    auto &word = v[k * Bits / 32][coordinate + 16 * group];
    int shift = k * Bits % 32;
    word = (word & ~(((1U << Bits) - 1) << shift)) | (bits << shift);
  }
}

// Write a gfx12 dense WMMA FP16/BF16 A/B element. K is partitioned
// between 16-lane groups, with adjacent halves packed in each word.
template <int Lanes>
void set_wmma_gfx12_half(uint32_t *const *v, int coordinate, int k, uint32_t bits) {
  static_assert(Lanes == 32 || Lanes == 64);
  constexpr int group_k = 16 / (Lanes / 16);
  auto &word = v[(k % group_k) / 2][coordinate + 16 * (k / group_k)];
  int shift = 16 * (k % 2);
  word = (word & ~(0xffffU << shift)) | (bits << shift);
}

// Access a gfx12 FP32 accumulator element in the wave's ISA register layout.
template <int Lanes> uint32_t &wmma_gfx12_acc(uint32_t *const *v, int row, int col) {
  static_assert(Lanes == 32 || Lanes == 64);
  constexpr int regs = 256 / Lanes;
  int lane = col + 16 * (row / 8) + (Lanes == 64 ? 32 * ((row / 4) % 2) : 0);
  return v[row % regs][lane];
}

// Access a gfx11 FP32 accumulator element in the wave's ISA register layout.
template <int Lanes> uint32_t &wmma_gfx11_acc(uint32_t *const *v, int row, int col) {
  static_assert(Lanes == 32 || Lanes == 64);
  constexpr int groups = Lanes / 16;
  return v[row / groups][col + 16 * (row % groups)];
}

// Check mandatory gfx11 flushing of BF16 factors, FP32 accumulators, and tiny results.
template <int Lanes, typename Call> void expect_gfx11_bf16_denormal(Call call) {
  struct Case {
    int k;
    uint16_t a, b;
    uint32_t c, expected;
  };
  // The last two cases match a gfx1100 Wave32 observation; Wave64 checks the same
  // logical layout. Keep K=0 for input flushing and K=15 for result flushing:
  // intervening DOT2 steps flush the accumulator again.
  constexpr Case cases[] = {{14, 0x0001, 0x4980, 0x00000000, 0x00000000},
                            {0, 0x0080, 0xbf80, 0x00000001, 0x80800000},
                            {15, 0x0080, 0x3f80, 0x80c00000, 0x00000000}};
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics : {RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_ALLOW_LOOSE,
                               RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS}) {
      for (const auto &sample : cases) {
        SCOPED_TRACE(testing::Message() << cpu << '/' << semantics << "/K=" << sample.k);
        Registers<32, Lanes> r;
        for (auto *reg : r.v) {
          std::fill_n(reg, Lanes, 0U);
        }
        for (int coordinate = 0; coordinate < 16; ++coordinate) {
          set_wmma_gfx11<16, Lanes>(r.v, coordinate, sample.k, sample.a);
          set_wmma_gfx11<16, Lanes>(r.v + 8, coordinate, sample.k, sample.b);
          for (int col = 0; col < 16; ++col) {
            wmma_gfx11_acc<Lanes>(r.v + 16, coordinate, col) = sample.c;
          }
        }
        ASSERT_EQ(call(cpu | semantics, 0, r.v + 24, r.v, r.v + 8, r.v + 16), RJ_ARITH_SUCCESS);
        for (int row = 0; row < 16; ++row) {
          for (int col = 0; col < 16; ++col) {
            EXPECT_EQ(wmma_gfx11_acc<Lanes>(r.v + 24, row, col), sample.expected);
          }
        }
      }
    }
  }
}

// Check a gfx11 FP32-output WMMA against its golden matrix across CPUs, semantics,
// and destination aliases. A/B/C begin at registers 0/8/16; all other storage is preserved.
template <typename Call, int RegistersCount, int Lanes>
void expect_gfx11_wmma_arithmetic(Call call, const Registers<RegistersCount, Lanes> &input,
                                  const float (&golden)[16][16],
                                  std::initializer_list<int> aliasing_offsets) {
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics : semantics_modes()) {
      for (int aliasing_offset : aliasing_offsets) {
        SCOPED_TRACE(testing::Message() << cpu << '/' << semantics << '/' << aliasing_offset);
        auto r = input;
        ASSERT_EQ(call(cpu | semantics, 0, r.v + aliasing_offset, r.v, r.v + 8, r.v + 16),
                  RJ_ARITH_SUCCESS);
        for (int row = 0; row < 16; ++row) {
          for (int col = 0; col < 16; ++col) {
            uint32_t actual = wmma_gfx11_acc<Lanes>(r.v + aliasing_offset, row, col);
            if (semantics == RJ_ARITH_SEMANTICS_LOOSE) {
              expect_f32(actual, golden[row][col], semantics, /*atol=*/1e-5, /*rtol=*/1e-5);
            } else {
              EXPECT_EQ(actual, std::bit_cast<uint32_t>(golden[row][col]));
            }
          }
        }
        r.expect_preserved(input, aliasing_offset, 256 / Lanes);
      }
    }
  }
}

} // namespace rocjitsu::arith::test
