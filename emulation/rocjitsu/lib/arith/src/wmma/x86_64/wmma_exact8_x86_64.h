// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Dense gfx12 FP8/BF8 WMMA using Wave32 register sections for both wave sizes.
#pragma once

#include "wmma/layout.h"
#include "wmma/x86_64/wmma_dot8_x86_64.h"

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::wmma_exact8 {

using namespace wmma_dot8;

constexpr int Slices = 16 / Columns;

// Decode the eight K factors of one DOT8 stage for one slice of lanes. A and B
// share this packed layout, so the lanes are rows for A and columns for B.
template <bool Bf8>
inline uint32_t decode_stage(Factor *factors, const uint32_t *const *operand, int stage,
                             int slice) {
  Vec largest = constant<Vec>(0);
  for (int k = 0; k < 8; k += 4) {
    int reg = wmma_gfx12_packed_local(8 * stage + k) / 4;
    Vec words = load<Vec>(operand[reg] + 16 * stage + slice * Columns);
    RJ_ARITH_UNROLL(4)
    for (int i = 0; i < 4; ++i) {
      Vec bits = bit_and(shift_right(words, 8 * i), constant<Vec>(255));
      factors[k + i] = decode_factor<Bf8>(bits, largest);
    }
  }
  return uint32_t(lane_mask(greater(largest, constant<Vec>(FirstSpecial<Bf8> - 1))));
}

// Fixed dense K lets each row broadcast its A lane straight from the decoded
// vectors, which are stored by K with one lane per row.
struct Left {
  alignas(Vec) int32_t units[16][16];
  alignas(Vec) int32_t grid[16][16];
  // One bit per row holding a NaN or infinity in that stage.
  uint32_t special[2];
};

template <bool Bf8> inline void prepare_left(Left &left, const uint32_t *const *a) {
  for (int stage = 0; stage < 2; ++stage) {
    left.special[stage] = 0;
    for (int slice = 0; slice < Slices; ++slice) {
      Factor factors[8];
      uint32_t special = decode_stage<Bf8>(factors, a, stage, slice);
      RJ_ARITH_UNROLL(8)
      for (int k = 0; k < 8; ++k) {
        store(left.units[8 * stage + k] + slice * Columns, factors[k].units);
        store(left.grid[8 * stage + k] + slice * Columns, factors[k].grid);
      }
      left.special[stage] |= special << (slice * Columns);
    }
  }
}

// Apply one DOT8 stage to a row. The A factors are rows of units and grids
// indexed by the stage's K, and b holds the matching decoded B columns.
inline Vec dot8(const int32_t (*units)[16], const int32_t (*grid)[16], int row, const Factor *b,
                Vec acc) {
  Group groups[2];
  RJ_ARITH_UNROLL(2)
  for (int group = 0; group < 2; ++group) {
    Vec products[4], grids[4];
    RJ_ARITH_UNROLL(4)
    for (int i = 0; i < 4; ++i) {
      int k = 4 * group + i;
      products[i] = mul(constant<Vec>(units[k][row]), b[k].units);
      grids[i] = add(constant<Vec>(grid[k][row]), b[k].grid);
    }
    groups[group] = align4(products, grids);
  }
  return round_groups(groups[0], groups[1], acc);
}

inline Vec exceptional(const int32_t (*units)[16], int row, const Factor *b, Vec acc, Vec result) {
  Exceptions seen{constant<Vec>(0), constant<Vec>(0), constant<Vec>(0)};
  for (int k = 0; k < 8; ++k) {
    classify(seen, constant<Vec>(units[k][row]), b[k].units);
  }
  return resolve(seen, acc, result);
}

// Apply one DOT8 stage to every row of a column slice. Select the finite
// loop once per slice, outside the row loop.
template <bool CheckSpecial>
inline void stage_rows(Result &result, const Left &left, const Factor *right, int stage, int slice,
                       const uint32_t *const *c, uint32_t special_rows) {
  for (int row = 0; row < 16; ++row) {
    int reg = row % 8, lane = 16 * (row / 8) + slice * Columns;
    Vec &out = result[reg][lane / Columns];
    Vec acc = stage ? out : load<Vec>(c[reg] + lane);
    Vec next = dot8(left.units + 8 * stage, left.grid + 8 * stage, row, right, acc);
    if constexpr (CheckSpecial) {
      if ((special_rows >> row) & 1) {
        next = exceptional(left.units + 8 * stage, row, right, acc, next);
      }
    }
    out = next;
  }
}

inline void run(bool bf8_a, bool bf8_b, uint32_t *const *d, const uint32_t *const *a,
                const uint32_t *const *b, const uint32_t *const *c) {
  Left left;
  if (bf8_a) {
    prepare_left<true>(left, a);
  } else {
    prepare_left<false>(left, a);
  }

  // Each stage keeps its eight decoded B factors live across all sixteen rows.
  Result result;
  for (int stage = 0; stage < 2; ++stage) {
    for (int slice = 0; slice < Slices; ++slice) {
      Factor right[8];
      uint32_t special_b = bf8_b ? decode_stage<true>(right, b, stage, slice)
                                 : decode_stage<false>(right, b, stage, slice);
      uint32_t special_rows = special_b ? 0xffff : left.special[stage];
      if (special_rows) [[unlikely]] {
        stage_rows<true>(result, left, right, stage, slice, c, special_rows);
      } else {
        stage_rows<false>(result, left, right, stage, slice, c, 0);
      }
    }
  }
  store_result(d, result);
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::wmma_exact8
