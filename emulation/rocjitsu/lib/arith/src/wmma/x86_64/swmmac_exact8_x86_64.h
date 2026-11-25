// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// gfx12 sparse FP8 DOT8 computed across independent output columns.
#pragma once

#include "wmma/sparse_layout.h"
#include "wmma/x86_64/wmma_dot8_x86_64.h"

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::sparse_exact8 {

using namespace wmma_dot8;

struct FactorTable {
  ScalarFactor values[256];
};

// Decode each A byte once into signed Q12 units and their binary grid.
template <bool Bf8>
inline constexpr auto factors = [] {
  FactorTable table{};
  for (unsigned bits = 0; bits < 256; ++bits) {
    Encoding<Bf8> value{uint16_t(bits)};
    int units = value.nan()   ? Nan
                : value.inf() ? Infinity
                              : int(value.significand()) << (12 - Encoding<Bf8>::fraction_bits);
    table.values[bits] = {value.negative() ? -units : units,
                          units ? value.alignment_exponent() - 12 : Empty};
  }
  return table;
}();

struct Input {
  ScalarFactor a[16][16];
  uint8_t k[16][16];
  Factor b[32][16 / Columns];
  bool special_a[16][2];
  bool special_b;
};

template <bool Bf8, int Wave>
inline void prepare_left(Input &in, const uint32_t *const *a, const uint32_t *const *index,
                         int key) {
  for (int row = 0; row < 16; ++row) {
    for (int stage = 0; stage < 2; ++stage) {
      bool any_special = false;
      for (int batch = 0; batch < 2; ++batch) {
        int first = stage * 8 + batch * 4;
        auto field = swmmac_a_field<8, Wave>(row, first);
        uint32_t word = a[field.reg][field.lane];
        int first_slot = swmmac_index_slot<8, Wave>(first);
        uint32_t selectors = index[0][field.lane] >> (key + 2 * first_slot);
        RJ_ARITH_UNROLL(4)
        for (int slot = 0; slot < 4; ++slot) {
          int ck = first + slot;
          uint8_t bits = uint8_t(word >> (8 * slot));
          any_special |= (bits & 127) >= FirstSpecial<Bf8>;
          in.a[row][ck] = factors<Bf8>.values[bits];
          in.k[row][ck] = uint8_t(4 * (ck / 2) + ((selectors >> (2 * slot)) & 3));
        }
      }
      in.special_a[row][stage] = any_special;
    }
  }
}

template <bool Bf8, int Wave> inline void prepare_right(Input &in, const uint32_t *const *b) {
  Vec largest = constant<Vec>(0);
  for (int k = 0; k < 32; ++k) {
    for (int col = 0; col < 16; col += Columns) {
      auto field = swmmac_b_field<8, Wave>(col, k);
      Vec packed = load<Vec>(b[field.reg] + field.lane);
      Vec bits = bit_and(shift_right(packed, constant<Vec>(field.shift)), constant<Vec>(255));
      in.b[k][col / Columns] = decode_factor<Bf8>(bits, largest);
    }
  }
  in.special_b = lane_mask(greater(largest, constant<Vec>(FirstSpecial<Bf8> - 1))) != 0;
}

template <int Wave>
inline void prepare(Input &in, bool bf8_a, bool bf8_b, int key, const uint32_t *const *a,
                    const uint32_t *const *b, const uint32_t *const *index) {
  if (bf8_a) {
    prepare_left<true, Wave>(in, a, index, key);
  } else {
    prepare_left<false, Wave>(in, a, index, key);
  }
  if (bf8_b) {
    prepare_right<true, Wave>(in, b);
  } else {
    prepare_right<false, Wave>(in, b);
  }
}

// Form four products of broadcast A factors with the B columns selected by k.
inline Group group4(const ScalarFactor *a, const uint8_t *k, const Factor (*b)[16 / Columns],
                    int slice) {
  Vec products[4], grids[4];
  RJ_ARITH_UNROLL(4)
  for (int i = 0; i < 4; ++i) {
    const auto &factor = b[k[i]][slice];
    products[i] = mul(constant<Vec>(a[i].units), factor.units);
    grids[i] = add(constant<Vec>(a[i].grid), factor.grid);
  }
  return align4(products, grids);
}

inline Vec exceptional(const ScalarFactor *a, const uint8_t *k, const Factor (*b)[16 / Columns],
                       int slice, Vec acc, Vec result) {
  Exceptions seen{constant<Vec>(0), constant<Vec>(0), constant<Vec>(0)};
  for (int i = 0; i < 8; ++i) {
    classify(seen, constant<Vec>(a[i].units), b[k[i]][slice].units);
  }
  return resolve(seen, acc, result);
}

inline Vec dot8(const ScalarFactor *a, const uint8_t *k, const Factor (*b)[16 / Columns], int slice,
                Vec acc, bool check_special) {
  Vec result = round_groups(group4(a, k, b, slice), group4(a + 4, k + 4, b, slice), acc);
  if (check_special) [[unlikely]] {
    result = exceptional(a, k, b, slice, acc, result);
  }
  return result;
}

// Apply both DOT8 stages to a complete 16-by-16 output tile.
inline void run(bool bf8_a, bool bf8_b, bool wave64, int key, uint32_t *const *d,
                const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *index) {
  Input in;
  if (wave64) {
    prepare<64>(in, bf8_a, bf8_b, key, a, b, index);
  } else {
    prepare<32>(in, bf8_a, bf8_b, key, a, b, index);
  }
  Result result;
  for (int ck = 0; ck < 16; ck += 8) {
    for (int row = 0; row < 16; ++row) {
      for (int slice = 0; slice < 16 / Columns; ++slice) {
        int reg = row % 8, lane = 16 * (row / 8) + slice * Columns;
        Vec acc = ck ? result[reg][lane / Columns] : load<Vec>(d[reg] + lane);
        result[reg][lane / Columns] = dot8(in.a[row] + ck, in.k[row] + ck, in.b, slice, acc,
                                           in.special_b || in.special_a[row][ck / 8]);
      }
    }
  }
  store_result(d, result);
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::sparse_exact8
