// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Gather each row's sparse B factors in compressed K order for gfx12 DOT4.
// Both x86 widths share single-load A/B decoding. FP32 (v3 & v4) and Packed
// BF16 (v4) keep unpacked accumulators across stages with branchless 16-row
// loops and deferred cold repair; Packed FP16 (and v3 BF16) use the single-load
// staged path directly.
#pragma once

#include "common/compiler.h"
#include "common/packed16.h"
#include "dot/dot.h"
#include "dot/x86_64/exact_dot_simd.h"
#include "wmma/sparse_layout.h"

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::sparse_half {

// Each DOT4 stage transposes A into four columns. Each retained value has
// a two-bit selector for its matching B row within the dense group of four.
struct alignas(64) LeftRows {
  uint32_t bits[16];
  uint32_t selector[16];
  uint32_t units[16];
  int32_t grid[16];
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

// Read one row's initial C vector (with ABS_C / NEG_C applied) directly from dest.
template <typename V, bool Packed>
inline V load_c_row(uint32_t mode, int col, int row, const uint32_t *const *dest) {
  using namespace exact_dot;
  constexpr unsigned Sign = Packed ? 0x8000U : 0x80000000U;
  int reg = row % 8, lane = col + 16 * (row / 8);
  V bits = load<V>(dest[Packed ? reg / 2 : reg] + lane);
  if constexpr (Packed) {
    bits = bit_and(shift_right(bits, 16 * (reg % 2)), constant<V>(0xffff));
  }
  if (mode & RJ_ARITH_WMMA_ABS_C) {
    bits = bit_and(bits, constant<V>(int(~Sign)));
  }
  if (mode & RJ_ARITH_WMMA_NEG_C) {
    bits = bit_xor(bits, constant<V>(int(Sign)));
  }
  return bits;
}

// Cold exact fallback for rows in one output slice that encounter infinite
// factors, nonfinite B, intermediate overflow, or gradual subnormal underflow.
template <class Stage, int WaveSize, bool Packed>
[[gnu::noinline]] void repair_slice(uint32_t row_mask, int col, uint32_t mode, bool saturate,
                                    typename Stage::V (&acc)[16], uint32_t *const *dest,
                                    const uint32_t *const *a, const uint32_t *const *b,
                                    const uint32_t *const *index) {
  using namespace exact_dot;
  using V = typename Stage::V;
  using Factor = typename Stage::Factor;
  const int key = SwmmacIndex<16, 32, WaveSize>::shift(mode);
  const uint32_t negate[2] = {
      bool(mode & RJ_ARITH_WMMA_NEG_LO_A) != bool(mode & RJ_ARITH_WMMA_NEG_LO_B) ? 0x8000U : 0,
      bool(mode & RJ_ARITH_WMMA_NEG_HI_A) != bool(mode & RJ_ARITH_WMMA_NEG_HI_B) ? 0x8000U : 0};

  for (uint32_t m = row_mask; m; m &= m - 1) {
    int row = __builtin_ctz(m);
    acc[row] = Stage::initial(load_c_row<V, Packed>(mode, col, row, dest));
  }
  for (int step = 0; step < 16; step += 4) {
    Factor choices[8];
    V bmax = constant<V>(0);
    RJ_ARITH_UNROLL(4)
    for (int p = 0; p < 4; ++p) {
      SparseField field = swmmac_b_field<16, WaveSize>(col, 2 * step + 2 * p);
      V packed = load<V>(b[field.reg] + field.lane);
      V bh[2] = {bit_and(packed, constant<V>(0xffff)), shift_right(packed, 16)};
      RJ_ARITH_UNROLL(2)
      for (int h = 0; h < 2; ++h) {
        bmax = maximum(bmax, bit_and(bh[h], constant<V>(0x7fff)));
        choices[2 * p + h] = Stage::factor(bh[h], Stage::Bias);
      }
    }
    const bool b_exc = lane_mask(greater(bmax, constant<V>(Stage::Inf - 1))) != 0;
    const int slot_base = swmmac_index_slot<16, WaveSize>(step);
    for (uint32_t m = row_mask; m; m &= m - 1) {
      int row = __builtin_ctz(m);
      SparseField first = swmmac_a_field<16, WaveSize>(row, step);
      uint32_t selectors = index[0][first.lane] >> (key + 2 * slot_base);
      Factor left[4]{}, right[4]{};
      int amax = 0;
      RJ_ARITH_UNROLL(4)
      for (int j = 0; j < 4; ++j) {
        SparseField field = swmmac_a_field<16, WaveSize>(row, step + j);
        uint32_t abits = ((a[field.reg][field.lane] >> field.shift) & 0xffffU) ^ negate[j % 2];
        int amag = int(abits & 0x7fffU);
        if (amag > amax) {
          amax = amag;
        }
        left[j] = Stage::factor(constant<V>(int(abits)), 0);
        right[j] = choices[4 * (j / 2) + ((selectors >> (2 * j)) & 3U)];
      }
      V old = acc[row];
      typename Stage::Mask special;
      V result = Stage::finite(left, right, old, saturate, special);
      if (b_exc || amax >= Stage::Inf || Stage::any(special)) {
        result = Stage::repair(left, right, old, result, special);
      }
      acc[row] = result;
    }
  }
}

// Decode A for one DOT4 stage (single load per packed pair).
template <class Stage, int WaveSize>
[[gnu::always_inline]] inline void
decode_a_stage(int step, int key, const uint32_t (&negate)[2], const uint32_t *const *a,
               const uint32_t *const *index, LeftRows (&columns)[4], uint32_t &a_nan_rows,
               uint32_t &a_inf_rows) {
  using namespace exact_dot;
  using V = typename Stage::V;
  constexpr int Width = sizeof(V) / sizeof(uint32_t);
  V mask_lo = constant<V>(0xffff);
  V mask_mag = constant<V>(0x7fff);
  V neg_lo = constant<V>(int(negate[0]));
  V neg_hi = constant<V>(int(negate[1]));
  for (int base = 0; base < 16; base += Width) {
    V amax = constant<V>(0);
    V a_mags[4];
    SparseField first = swmmac_a_field<16, WaveSize>(base, step);
    int slot_base = swmmac_index_slot<16, WaveSize>(step);
    V selectors = shift_right(load<V>(index[0] + first.lane), key + 2 * slot_base);
    RJ_ARITH_UNROLL(2)
    for (int p = 0; p < 2; ++p) {
      SparseField field = swmmac_a_field<16, WaveSize>(base, step + 2 * p);
      V packed = load<V>(a[field.reg] + field.lane);
      V ah[2] = {bit_xor(bit_and(packed, mask_lo), neg_lo),
                 bit_xor(shift_right(packed, 16), neg_hi)};
      RJ_ARITH_UNROLL(2)
      for (int h = 0; h < 2; ++h) {
        int j = 2 * p + h;
        typename Stage::Factor factor = Stage::factor(ah[h], 0);
        a_mags[j] = bit_and(ah[h], mask_mag);
        amax = maximum(amax, a_mags[j]);
        V select = bit_and(shift_right(selectors, 2 * j), constant<V>(3));
        auto &column = columns[j];
        store(column.selector + base, select);
        store(column.units + base, Stage::pack_a_unit(factor.units));
        store(column.grid + base, factor.grid);
      }
    }
    uint32_t stage_special = uint32_t(lane_mask(greater(amax, constant<V>(Stage::Inf - 1))))
                             << base;
    if (stage_special) [[unlikely]] {
      a_nan_rows |= uint32_t(lane_mask(greater(amax, constant<V>(Stage::Inf)))) << base;
      RJ_ARITH_UNROLL(4)
      for (int j = 0; j < 4; ++j) {
        a_inf_rows |= uint32_t(lane_mask(equal(a_mags[j], constant<V>(Stage::Inf)))) << base;
      }
      for (uint32_t m = stage_special; m; m &= m - 1) {
        int r = __builtin_ctz(m);
        RJ_ARITH_UNROLL(4)
        for (int j = 0; j < 4; ++j) {
          columns[j].units[r] = 0;
          columns[j].grid[r] = gfx12_dot_detail::kEmptyExponent;
        }
      }
    }
  }
}

// Decode the 8 dense B choices for one slice (single load per packed pair).
template <class Stage, int WaveSize>
[[gnu::always_inline]] inline void
decode_b_stage(int step, int col, int bias, const uint32_t *const *b,
               typename Stage::V (&b_units)[8], typename Stage::V (&b_grid)[8],
               uint32_t &b_special) {
  using namespace exact_dot;
  using V = typename Stage::V;
  V mask_lo = constant<V>(0xffff);
  V mask_mag = constant<V>(0x7fff);
  V bmax = constant<V>(0);
  RJ_ARITH_UNROLL(4)
  for (int p = 0; p < 4; ++p) {
    SparseField field = swmmac_b_field<16, WaveSize>(col, 2 * step + 2 * p);
    V packed = load<V>(b[field.reg] + field.lane);
    V bh[2] = {bit_and(packed, mask_lo), shift_right(packed, 16)};
    RJ_ARITH_UNROLL(2)
    for (int h = 0; h < 2; ++h) {
      int k = 2 * p + h;
      typename Stage::Factor f = Stage::factor(bh[h], bias);
      b_units[k] = f.units;
      b_grid[k] = f.grid;
      bmax = maximum(bmax, bit_and(bh[h], mask_mag));
    }
  }
  b_special |= uint32_t(lane_mask(greater(bmax, constant<V>(Stage::Inf - 1))));
}

template <class Stage, int WaveSize, bool Packed>
void tile(uint32_t mode, bool saturate, uint32_t *const *dest, const uint32_t *const *a,
          const uint32_t *const *b, const uint32_t *const *index) {
  using namespace exact_dot;
  using V = typename Stage::V;
  using Mask = typename Stage::Mask;
  using Factor = typename Stage::Factor;
  constexpr int Width = sizeof(V) / sizeof(uint32_t);
  constexpr int Slices = 16 / Width;
  // Unpacked accumulators win on FP32 (v3 & v4) and Packed BF16 (v4). Packed
  // FP16 and AVX2 Packed BF16 stay on the single-load staged DOT4 loop.
  constexpr bool UseUnpacked = !Packed || (Stage::Bf16 && Width == 16);
  constexpr unsigned Sign = Packed ? 0x8000U : 0x80000000U;
  const V keep = constant<V>(mode & RJ_ARITH_WMMA_ABS_C ? int(~Sign) : -1);
  const V sign = constant<V>(mode & RJ_ARITH_WMMA_NEG_C ? int(Sign) : 0);
  const int key = SwmmacIndex<16, 32, WaveSize>::shift(mode);
  const uint32_t negate[2] = {
      bool(mode & RJ_ARITH_WMMA_NEG_LO_A) != bool(mode & RJ_ARITH_WMMA_NEG_LO_B) ? 0x8000U : 0,
      bool(mode & RJ_ARITH_WMMA_NEG_HI_A) != bool(mode & RJ_ARITH_WMMA_NEG_HI_B) ? 0x8000U : 0};

  V acc[Slices][16];

  if constexpr (UseUnpacked) {
    V c_grid[Slices][16];
    Mask c_special_mask[Slices][16];
    uint32_t c_special_rows[Slices] = {};
    for (int slice = 0; slice < Slices; ++slice) {
      int col = slice * Width;
      for (int row = 0; row < 16; ++row) {
        int reg = row % 8, lane = col + 16 * (row / 8);
        V bits = load<V>(dest[Packed ? reg / 2 : reg] + lane);
        if constexpr (Packed) {
          bits = bit_and(shift_right(bits, 16 * (reg % 2)), constant<V>(0xffff));
        }
        V cv = bit_xor(bit_and(bits, keep), sign);
        Stage::unpack_c(cv, acc[slice][row], c_grid[slice][row], c_special_mask[slice][row]);
        if (Stage::any(c_special_mask[slice][row])) [[unlikely]] {
          c_special_rows[slice] |= 1u << row;
        }
      }
    }

    uint32_t a_nan_rows = 0, a_inf_rows = 0, b_special[Slices] = {};
    Mask any_bad[Slices] = {};

    for (int step = 0; step < 16; step += 4) {
      LeftRows columns[4];
      decode_a_stage<Stage, WaveSize>(step, key, negate, a, index, columns, a_nan_rows, a_inf_rows);
      const bool final_stage = (step == 12);
      for (int slice = 0; slice < Slices; ++slice) {
        V b_units[8], b_grid[8];
        decode_b_stage<Stage, WaveSize>(step, slice * Width, 0, b, b_units, b_grid,
                                        b_special[slice]);
        Mask bad_acc = any_bad[slice];
        if (!final_stage) {
          for (int row = 0; row < 16; ++row) {
            uint32_t s0 = columns[0].selector[row];
            uint32_t s1 = columns[1].selector[row];
            uint32_t s2 = 4u + columns[2].selector[row];
            uint32_t s3 = 4u + columns[3].selector[row];
            V total, grid;
            Stage::dot4_sum(columns[0].units[row], columns[0].grid[row], b_units[s0], b_grid[s0],
                            columns[1].units[row], columns[1].grid[row], b_units[s1], b_grid[s1],
                            columns[2].units[row], columns[2].grid[row], b_units[s2], b_grid[s2],
                            columns[3].units[row], columns[3].grid[row], b_units[s3], b_grid[s3],
                            acc[slice][row], c_grid[slice][row], total, grid);
            Stage::next_unpacked(total, grid, acc[slice][row], c_grid[slice][row], bad_acc);
          }
        } else {
          for (int row = 0; row < 16; ++row) {
            uint32_t s0 = columns[0].selector[row];
            uint32_t s1 = columns[1].selector[row];
            uint32_t s2 = 4u + columns[2].selector[row];
            uint32_t s3 = 4u + columns[3].selector[row];
            V total, grid;
            Stage::dot4_sum(columns[0].units[row], columns[0].grid[row], b_units[s0], b_grid[s0],
                            columns[1].units[row], columns[1].grid[row], b_units[s1], b_grid[s1],
                            columns[2].units[row], columns[2].grid[row], b_units[s2], b_grid[s2],
                            columns[3].units[row], columns[3].grid[row], b_units[s3], b_grid[s3],
                            acc[slice][row], c_grid[slice][row], total, grid);
            acc[slice][row] = Stage::pack_final(total, grid, bad_acc);
          }
        }
        any_bad[slice] = bad_acc;
      }
    }

    for (int slice = 0; slice < Slices; ++slice) {
      bool bad_any = Stage::any(any_bad[slice]);
      if (b_special[slice] || a_inf_rows || bad_any) [[unlikely]] {
        uint32_t repair_mask = (b_special[slice] || bad_any)
                                   ? 0xffffu
                                   : (a_inf_rows | a_nan_rows | c_special_rows[slice]);
        repair_slice<Stage, WaveSize, Packed>(repair_mask, slice * Width, mode, saturate,
                                              acc[slice], dest, a, b, index);
      } else if (c_special_rows[slice] | a_nan_rows) [[unlikely]] {
        Stage::apply_simple_specials(mode, slice * Width, c_special_rows[slice], a_nan_rows,
                                     c_special_mask[slice], dest, acc[slice]);
      }
    }
  } else {
    // Packed FP16 (v3 & v4) and Packed BF16 (v3): single-load A/B staging
    // feeding the direct per-stage empirical DOT4 reduction.
    for (int slice = 0; slice < Slices; ++slice) {
      int col = slice * Width;
      for (int row = 0; row < 16; ++row) {
        int reg = row % 8, lane = col + 16 * (row / 8);
        V bits = load<V>(dest[reg / 2] + lane);
        bits = bit_and(shift_right(bits, 16 * (reg % 2)), constant<V>(0xffff));
        acc[slice][row] = Stage::initial(bit_xor(bit_and(bits, keep), sign));
      }
    }
    V mask_lo = constant<V>(0xffff);
    V mask_mag = constant<V>(0x7fff);
    V neg_lo = constant<V>(int(negate[0]));
    V neg_hi = constant<V>(int(negate[1]));
    for (int step = 0; step < 16; step += 4) {
      LeftRows columns[4];
      unsigned exceptional_rows = 0;
      for (int base = 0; base < 16; base += Width) {
        V amax = constant<V>(0);
        SparseField first = swmmac_a_field<16, WaveSize>(base, step);
        int slot_base = swmmac_index_slot<16, WaveSize>(step);
        V selectors = shift_right(load<V>(index[0] + first.lane), key + 2 * slot_base);
        RJ_ARITH_UNROLL(2)
        for (int p = 0; p < 2; ++p) {
          SparseField field = swmmac_a_field<16, WaveSize>(base, step + 2 * p);
          V packed = load<V>(a[field.reg] + field.lane);
          V ah[2] = {bit_xor(bit_and(packed, mask_lo), neg_lo),
                     bit_xor(shift_right(packed, 16), neg_hi)};
          RJ_ARITH_UNROLL(2)
          for (int h = 0; h < 2; ++h) {
            int j = 2 * p + h;
            Factor factor = Stage::factor(ah[h], 0);
            amax = maximum(amax, bit_and(ah[h], mask_mag));
            V select = bit_and(shift_right(selectors, 2 * j), constant<V>(3));
            LeftRows &column = columns[j];
            store(column.bits + base, ah[h]);
            store(column.selector + base, select);
            store(column.units + base, factor.units);
            store(column.grid + base, factor.grid);
          }
        }
        exceptional_rows |= lane_mask(greater(amax, constant<V>(Stage::Inf - 1))) << base;
      }
      for (int slice = 0; slice < Slices; ++slice) {
        int col = slice * Width;
        Factor choices[8];
        V bmax = constant<V>(0);
        RJ_ARITH_UNROLL(4)
        for (int p = 0; p < 4; ++p) {
          SparseField field = swmmac_b_field<16, WaveSize>(col, 2 * step + 2 * p);
          V packed = load<V>(b[field.reg] + field.lane);
          V bh[2] = {bit_and(packed, mask_lo), shift_right(packed, 16)};
          RJ_ARITH_UNROLL(2)
          for (int h = 0; h < 2; ++h) {
            bmax = maximum(bmax, bit_and(bh[h], mask_mag));
            choices[2 * p + h] = Stage::factor(bh[h], Stage::Bias);
          }
        }
        const unsigned special_rows =
            lane_mask(greater(bmax, constant<V>(Stage::Inf - 1))) ? 0xffffU : exceptional_rows;
        for (int row = 0; row < 16; ++row) {
          Factor left[4]{}, right[4]{};
          RJ_ARITH_UNROLL(4)
          for (int j = 0; j < 4; ++j) {
            const auto &column = columns[j];
            const Factor &selected = choices[4 * (j / 2) + column.selector[row]];
            left[j].units = constant<V>(int(column.units[row]));
            left[j].grid = constant<V>(column.grid[row]);
            right[j].units = selected.units;
            right[j].grid = selected.grid;
          }
          const V addend = acc[slice][row];
          Mask special;
          V result = Stage::finite(left, right, addend, saturate, special);
          if ((special_rows & (1U << row)) || Stage::any(special)) [[unlikely]] {
            RJ_ARITH_UNROLL(4)
            for (int j = 0; j < 4; ++j) {
              const LeftRows &column = columns[j];
              left[j].bits = constant<V>(int(column.bits[row]));
              right[j].bits = choices[4 * (j / 2) + column.selector[row]].bits;
            }
            result = Stage::repair(left, right, addend, result, special);
          }
          acc[slice][row] = result;
        }
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
