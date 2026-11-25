// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Gather each row's sparse B factors in compressed K order. The two x86
// widths supply their existing dense DOT4 stage and packed-result convention.
#pragma once

#include "common/compiler.h"
#include "dot/x86_64/exact_dot_simd.h"
#include "wmma/sparse_layout.h"

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::sparse_half {

// Each DOT4 stage transposes A into four columns. Each retained value has
// a two-bit selector for its matching B row within the dense group of four.
struct LeftRows {
  uint32_t bits[16];
  uint32_t selector[16];
  uint32_t units[16];
  uint32_t grid[16];
};

// Inspect the four selected pairs only when a row may contain a special value.
template <typename V, typename Factor>
V selected_max(const Factor (&left)[4], const Factor (&right)[4]) {
  using namespace exact_dot;
  auto magnitude_mask = constant<V>(0x7fff);
  auto largest = constant<V>(0);
  RJ_ARITH_UNROLL(4)
  for (int j = 0; j < 4; ++j) {
    auto a = bit_and(left[j].bits, magnitude_mask);
    auto b = bit_and(right[j].bits, magnitude_mask);
    largest = maximum(largest, maximum(a, b));
  }
  return largest;
}

template <class Stage, int WaveSize, bool Packed>
void tile(uint32_t mode, bool saturate, uint32_t *const *dest, const uint32_t *const *a,
          const uint32_t *const *b, const uint32_t *const *index) {
  using namespace exact_dot;
  using V = typename Stage::V;
  using Factor = typename Stage::Factor;
  constexpr int Width = sizeof(V) / sizeof(uint32_t);
  constexpr unsigned Sign = Packed ? 0x8000U : 0x80000000U;
  const auto keep = constant<V>(mode & RJ_ARITH_WMMA_ABS_C ? int(~Sign) : -1);
  const auto sign = constant<V>(mode & RJ_ARITH_WMMA_NEG_C ? int(Sign) : 0);
  const int key = SwmmacIndex<16, 32, WaveSize>::shift(mode);
  // B negation follows the compressed pair position. Fold it into A since only the
  // XOR of the factor signs affects a finite product or an infinite product.
  const uint32_t negate[2] = {
      bool(mode & RJ_ARITH_WMMA_NEG_LO_A) != bool(mode & RJ_ARITH_WMMA_NEG_LO_B) ? 0x8000U : 0,
      bool(mode & RJ_ARITH_WMMA_NEG_HI_A) != bool(mode & RJ_ARITH_WMMA_NEG_HI_B) ? 0x8000U : 0};
  V acc[16 / Width][16];
  for (int slice = 0; slice < 16 / Width; ++slice) {
    int col = slice * Width;
    for (int row = 0; row < 16; ++row) {
      int reg = row % 8, lane = col + 16 * (row / 8);
      auto bits = load<V>(dest[Packed ? reg / 2 : reg] + lane);
      if constexpr (Packed) {
        bits = bit_and(shift_right(bits, 16 * (reg % 2)), constant<V>(0xffff));
      }
      acc[slice][row] = Stage::initial(bit_xor(bit_and(bits, keep), sign));
    }
  }
  for (int step = 0; step < 16; step += 4) {
    alignas(64) LeftRows columns[4];
    unsigned exceptional_rows = 0;
    for (int base = 0; base < 16; base += Width) {
      V amax = constant<V>(0);
      auto first = swmmac_a_field<16, WaveSize>(base, step);
      int slot_base = swmmac_index_slot<16, WaveSize>(step);
      auto selectors = shift_right(load<V>(index[0] + first.lane), key + 2 * slot_base);
      RJ_ARITH_UNROLL(4)
      for (int j = 0; j < 4; ++j) {
        int ck = step + j;
        auto field = swmmac_a_field<16, WaveSize>(base, ck);
        auto bits = bit_xor(bit_and(shift_right(load<V>(a[field.reg] + field.lane), field.shift),
                                    constant<V>(0xffff)),
                            constant<V>(negate[j % 2]));
        auto factor = Stage::factor(bits, 0);
        amax = maximum(amax, bit_and(bits, constant<V>(0x7fff)));
        auto select = bit_and(shift_right(selectors, 2 * j), constant<V>(3));
        auto &column = columns[j];
        store(column.bits + base, bits);
        store(column.selector + base, select);
        store(column.units + base, factor.units);
        store(column.grid + base, factor.grid);
      }
      exceptional_rows |= lane_mask(greater(amax, constant<V>(Stage::Inf - 1))) << base;
    }
    // Each pair of compressed A elements selects two of four dense B rows.
    // Decode the eight B choices once for all sixteen output rows.
    for (int slice = 0; slice < 16 / Width; ++slice) {
      int col = slice * Width;
      Factor choices[8];
      V bmax = constant<V>(0);
      for (int k = 0; k < 8; ++k) {
        auto field = swmmac_b_field<16, WaveSize>(col, 2 * step + k);
        auto bits = bit_and(shift_right(load<V>(b[field.reg] + field.lane), field.shift),
                            constant<V>(0xffff));
        bmax = maximum(bmax, bit_and(bits, constant<V>(0x7fff)));
        choices[k] = Stage::factor(bits, Stage::Bias);
      }
      const unsigned special_rows =
          lane_mask(greater(bmax, constant<V>(Stage::Inf - 1))) ? 0xffffU : exceptional_rows;
      for (int row = 0; row < 16; ++row) {
        Factor left[4]{}, right[4]{};
        RJ_ARITH_UNROLL(4)
        for (int j = 0; j < 4; ++j) {
          const auto &column = columns[j];
          const auto &selected = choices[4 * (j / 2) + column.selector[row]];
          left[j].units = constant<V>(column.units[row]);
          left[j].grid = constant<V>(column.grid[row]);
          right[j].units = selected.units;
          right[j].grid = selected.grid;
        }
        const auto addend = acc[slice][row];
        typename Stage::Mask special;
        auto result = Stage::finite(left, right, addend, saturate, special);
        if ((special_rows & (1U << row)) || Stage::any(special)) [[unlikely]] {
          // Only the nonfinite repair inspects the original operand encodings.
          RJ_ARITH_UNROLL(4)
          for (int j = 0; j < 4; ++j) {
            const auto &column = columns[j];
            left[j].bits = constant<V>(column.bits[row]);
            right[j].bits = choices[4 * (j / 2) + column.selector[row]].bits;
          }
          result = Stage::repair(left, right, addend, result, special);
        }
        acc[slice][row] = result;
      }
    }
  }
  // Commit physical sections in scalar order, after the last A/B/metadata read.
  for (int reg = 0; reg < (Packed ? 4 : 8); ++reg) {
    for (int lane = 0; lane < 32; lane += Width) {
      int row = (Packed ? 2 * reg : reg) + 8 * (lane / 16);
      auto value = acc[(lane % 16) / Width][row];
      if constexpr (Packed) {
        value = bit_or(bit_and(value, constant<V>(0xffff)),
                       shift_left(acc[(lane % 16) / Width][row + 1], 16));
      }
      store(dest[reg] + lane, value);
    }
  }
}
} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::sparse_half
