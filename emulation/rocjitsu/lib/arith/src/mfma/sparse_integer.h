// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Inherited CDNA sparse byte layouts and integer accumulation, isolated by CPU ISA.
#pragma once
#include "mfma/integer.h"
#include "rj_arith/detail/rj_arith_wmma.h"
namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {
struct SmfmacByteLocation {
  int lane, nibble, element;
};

// Locate compressed A and its selector nibble in compact or wide SMFMAC operands.
template <int M, int K> SmfmacByteLocation smfmac_byte_location(int row, int q, int slot) {
  constexpr bool wide = M * K == 2048;
  if constexpr (!wide) {
    constexpr int half_groups = K / 8;
    return {((q % half_groups) / 2) * M + row, 2 * (q / half_groups) + q % 2,
            4 * (q / half_groups) + (2 * q + slot) % 4};
  } else if constexpr (M == 16) {
    return {(2 * (q / 16) + (q / 4) % 2) * M + row, 2 * ((q / 8) % 2) + 4 * ((q % 4) / 2) + q % 2,
            4 * (2 * ((q / 2) % 2) + (q / 8) % 2) + (2 * q + slot) % 4};
  } else {
    return {q / 8 * M + row, q % 2 + 2 * ((q / 4) % 2) + 4 * ((q / 2) % 2),
            4 * (2 * ((q / 2) % 2) + (q / 4) % 2) + (2 * q + slot) % 4};
  }
}

// Flatten the legacy dense B byte layout, independent of metadata selection.
template <int M, bool Wmma> int sparse_matrix_b_index(int col, int k) {
  int lane, element;
  if constexpr (Wmma) {
    // gfx1250 SWMMAC uses 32-element blocks, unlike dense WMMA's 16-element blocks.
    lane = col + 16 * ((k / 32) % 2);
    element = k % 32 + 32 * (k / 64);
  } else {
    lane = (M == 16 ? 16 * ((k % 32) / 8) : 32 * ((k % 16) / 8)) + col;
    element = 8 * (k / (M == 16 ? 32 : 16)) + k % 8;
  }
  return (element / 4 * (Wmma ? 32 : 64) + lane) * 4 + element % 4;
}

// Evaluate the tied accumulator without publishing any output until every input is consumed.
template <int M, int K, bool Wmma>
void sparse_matrix_i8_kernel(uint64_t controls, bool clamp, uint32_t *const *d,
                             const uint32_t *const *a, const uint32_t *const *b,
                             const uint32_t *const *index) {
  constexpr int lanes = Wmma ? 32 : 64;
  alignas(64) uint32_t left[M * K / 2], right[M * K], result[M][M];
  matrix_i8_decode<M * K / 2, lanes>(a, left, !Wmma || (controls & RJ_ARITH_WMMA_SIGNED_A));
  matrix_i8_decode<M * K, lanes>(b, right, !Wmma || (controls & RJ_ARITH_WMMA_SIGNED_B));
  for (int row = 0; row < M; ++row) {
    alignas(64) uint32_t values[K / 2], columns[K / 2][M];
    for (int ck = 0; ck < K / 2; ++ck) {
      int selected, ai;
      if constexpr (Wmma) {
        // Metadata uses contiguous 32-entry blocks; A switches halfwaves every 16 elements.
        int lane = row + 16 * (ck / 32);
        selected = int(index[(ck % 32) / 16][lane] >> (2 * (ck % 16)) & 3);
        ai = matrix_i8_index<16, 64, 1, true>(row, ck, 0, 0, true);
      } else {
        auto loc = smfmac_byte_location<M, K>(row, ck / 2, ck % 2);
        int shift = (M * K < 2048 && (controls & 3) == 0) ? 16 * int(controls >> 3 & 1) : 0;
        selected = int(index[0][loc.lane] >> (shift + 4 * loc.nibble + 2 * (ck % 2)) & 3);
        ai = (loc.element / 4 * 64 + loc.lane) * 4 + loc.element % 4;
      }
      values[ck] = left[ai];
      int k = 4 * (ck / 2) + selected;
      for (int col = 0; col < M; ++col) {
        columns[ck][col] = right[sparse_matrix_b_index<M, Wmma>(col, k)];
      }
    }
    matrix_i8_dot_row<M, K / 2>(values, columns, result[row]);
    for (int col = 0; col < M; ++col) {
      auto out = matrix_i8_output<M, 1, Wmma>(row, col, 0);
      uint32_t initial = d[out.reg][out.lane];
      {
        if (clamp) {
          int64_t sum = int64_t(bit_cast<int32_t>(result[row][col])) + bit_cast<int32_t>(initial);
          result[row][col] = sum > 0x7fffffffLL    ? 0x7fffffffU
                             : sum < -0x80000000LL ? 0x80000000U
                                                   : uint32_t(sum);
          continue;
        }
      }
      result[row][col] += initial;
    }
  }
  for (int row = 0; row < M; ++row) {
    for (int col = 0; col < M; ++col) {
      auto out = matrix_i8_output<M, 1, Wmma>(row, col, 0);
      d[out.reg][out.lane] = result[row][col];
    }
  }
}
} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
