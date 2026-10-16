// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Matrix operand layouts, sparse indices, and shared invalid-flag checks.

#pragma once

#include "common/contracts.h"
#include "common/registers.h"
#include "rj_arith/rj_arith.h"

#include <gtest/gtest.h>
#include <random>
#include <stdint.h>

namespace rocjitsu::arith::test {

// Check invalid flags with real storage (all preserved) and null operands
// (rejected before access). The last offset denotes C for WMMA or sparse
// indices for SWMMAC; SWMMAC's D is also its accumulator input.
template <class Call, int RegistersCount, int Lanes>
void expect_matrix_invalid_flags(Call call, const Registers<RegistersCount, Lanes> &input, int d,
                                 int a, int b, int c_or_indices, uint64_t supported_modifiers) {
  expect_invalid_flags(
      [&](uint64_t flags, uint64_t modifiers) {
        auto r = input;
        int result = call(flags, modifiers, r.v + d, r.v + a, r.v + b, r.v + c_or_indices);
        r.expect_preserved(input, 0, 0);
        EXPECT_EQ(call(flags, modifiers, nullptr, nullptr, nullptr, nullptr),
                  RJ_ARITH_ERROR_INVALID_FLAGS);
        return result;
      },
      supported_modifiers);
}

// Select two increasing K positions per group of four, using high engine bits.
// Draw order is row-major and part of the deterministic golden-input contract.
template <int K> void make_sparse_indices(std::minstd_rand &random, int (&selected)[16][K]) {
  static_assert(K % 2 == 0);
  const uint32_t pairs[] = {0x4, 0x8, 0xc, 0x9, 0xd, 0xe};
  for (int row = 0; row < 16; ++row) {
    for (int k = 0; k < K; k += 2) {
      uint32_t pair = pairs[(random() >> 16) % 6];
      selected[row][k] = 2 * k + (pair & 3);
      selected[row][k + 1] = 2 * k + (pair >> 2);
    }
  }
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

// Write a gfx12 dense WMMA 4/8-bit A/B element. Wave64 distributes
// the Wave32 register words into the upper 32 lanes.
template <int Bits, int Lanes>
void set_wmma_gfx12_small(uint32_t *const *v, int coordinate, int k, uint32_t bits) {
  static_assert(Bits == 4 || Bits == 8);
  static_assert(Lanes == 32 || Lanes == 64);
  int bit = (k % 8 + 8 * (k / 16)) * Bits;
  int lane = coordinate + 16 * ((k / 8) % 2);
  auto &word = v[Lanes == 32 ? bit / 32 : 0][lane + (Lanes == 64 ? 32 * (bit / 32) : 0)];
  int shift = bit % 32;
  word = (word & ~(((1U << Bits) - 1) << shift)) | (bits << shift);
}

} // namespace rocjitsu::arith::test
