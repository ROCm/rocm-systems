// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Dense WMMA operand layouts and shared invalid-flag checks.

#pragma once

#include "common/contracts.h"
#include "common/registers.h"
#include "rj_arith/rj_arith.h"

#include <gtest/gtest.h>
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

} // namespace rocjitsu::arith::test
