// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// gfx12 sparse FP8 DOT8 computed across independent output columns.
#pragma once

#include "wmma/sparse_layout.h"
#include "wmma/x86_64/wmma_dot8_x86_64.h"

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::sparse_exact8 {

using namespace wmma_dot8;

constexpr int Slices = 16 / Columns;

struct FactorTable {
  ScalarFactor values[256];
};

// Decode each A byte once into Q12 units and their binary grid.
// On AVX2, units are stored zero-extended from signed int16_t for VPMADDWD.
template <bool Bf8>
inline constexpr auto factors = [] {
  FactorTable table{};
  for (unsigned bits = 0; bits < 256; ++bits) {
    Encoding<Bf8> value{uint16_t(bits)};
    int units = value.nan()   ? Nan
                : value.inf() ? Infinity
                              : int(value.significand()) << (12 - Encoding<Bf8>::fraction_bits);
    int signed_units = value.negative() ? -units : units;
#if defined(RJ_ARITH_TARGET_X86_64_V4)
    int32_t stored_units = signed_units;
#else
    int32_t stored_units = int32_t(uint16_t(int16_t(signed_units)));
#endif
    table.values[bits] = {stored_units, units ? value.alignment_exponent() - 12 : Empty};
  }
  return table;
}();

struct Input {
  ScalarFactor a[16][16];
  uint8_t k[16][16];
  Factor b[Slices][32];
  uint32_t special_a[2];
  uint32_t special_b[Slices];
};

template <bool Bf8, int Wave>
inline void prepare_left(Input &in, const uint32_t *const *a, const uint32_t *const *index,
                         int key) {
  in.special_a[0] = 0;
  in.special_a[1] = 0;
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
      if (any_special) {
        in.special_a[stage] |= 1u << row;
      }
    }
  }
}

// Decode all 32 dense B K-positions, loading each packed 4-byte word vector once.
template <bool Bf8, int Wave> inline void prepare_right(Input &in, const uint32_t *const *b) {
  for (int slice = 0; slice < Slices; ++slice) {
    int col = slice * Columns;
    Vec largest = constant<Vec>(0);
    for (int k = 0; k < 32; k += 4) {
      auto field = swmmac_b_field<8, Wave>(col, k);
      Vec words = load<Vec>(b[field.reg] + field.lane);
      Factor batch[4];
      decode_word4<Bf8>(words, batch, largest);
      RJ_ARITH_UNROLL(4)
      for (int i = 0; i < 4; ++i) {
        in.b[slice][k + i] = batch[i];
      }
    }
    in.special_b[slice] = special_mask_from_largest16<Bf8>(largest);
  }
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
inline Group group4(const ScalarFactor *a, const uint8_t *k, const Factor *b) {
  Vec products[4], grids[4];
  RJ_ARITH_UNROLL(4)
  for (int i = 0; i < 4; ++i) {
    const Factor &factor = b[k[i]];
    products[i] = mul_units(a[i].units, factor.units);
    grids[i] = add(constant<Vec>(a[i].grid), factor.grid);
  }
  return align4(products, grids);
}

inline Vec exceptional(const ScalarFactor *a, const uint8_t *k, const Factor *b, Vec acc,
                       Vec result) {
  Exceptions seen{constant<Vec>(0), constant<Vec>(0), constant<Vec>(0)};
  for (int i = 0; i < 8; ++i) {
    int32_t u = int32_t(int16_t(a[i].units));
    classify(seen, constant<Vec>(u), b[k[i]].units);
  }
  return resolve(seen, acc, result);
}

// Apply one DOT8 stage to all 16 rows of one output slice. Special checking is
// hoisted outside the row loop so finite tiles execute branchlessly.
template <bool CheckSpecial>
inline void stage_rows(Result &result, const Input &in, int stage, int slice,
                       const uint32_t *const *d, uint32_t special_rows) {
  const int ck = 8 * stage;
  const Factor *b = in.b[slice];
  for (int row = 0; row < 16; ++row) {
    int reg = row % 8, lane = 16 * (row / 8) + slice * Columns;
    Vec &out = result[reg][lane / Columns];
    Vec acc = stage ? out : load<Vec>(d[reg] + lane);
    const ScalarFactor *arow = in.a[row] + ck;
    const uint8_t *krow = in.k[row] + ck;
    Vec next = round_groups(group4(arow, krow, b), group4(arow + 4, krow + 4, b), acc);
    if constexpr (CheckSpecial) {
      if ((special_rows >> row) & 1) {
        next = exceptional(arow, krow, b, acc, next);
      }
    }
    out = next;
  }
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
  for (int slice = 0; slice < Slices; ++slice) {
    for (int stage = 0; stage < 2; ++stage) {
      uint32_t special_rows = in.special_b[slice] ? 0xffffu : in.special_a[stage];
      if (special_rows) [[unlikely]] {
        stage_rows<true>(result, in, stage, slice, d, special_rows);
      } else {
        stage_rows<false>(result, in, stage, slice, d, 0);
      }
    }
  }
  store_result(d, result);
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::sparse_exact8
