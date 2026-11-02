// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// ISA-local sparse operand preparation and result layout helpers.
#pragma once

#include "common/isa.h"
#include "rj_arith/rj_arith.h"
#include "wmma/layout.h"
#include "wmma/sparse_layout.h"
#include "wmma/swmmac_integer.h"

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Snapshot signed/unsigned factors, sparse indices, and signed D into logical
// matrices. D uses eight 32-lane sections; metadata pairs must select two
// increasing positions in each group of four. K is the dense reduction size.
template <int Bits, int K, int WaveSize = 32>
inline void swmmac_integer_prepare(SwmmacIntegerInputs &input, uint32_t mode, uint32_t *const *d,
                                   const uint32_t *const *a, const uint32_t *const *b,
                                   const uint32_t *const *index) {
  // Interpret one extracted Bits-wide value with the operand's signedness.
  auto extend = [](uint32_t x, bool sign) {
    return int32_t(x) - (sign && (x & (1u << (Bits - 1))) ? int32_t(1u << Bits) : 0);
  };
  int key = rocjitsu::arith::SwmmacIndex<Bits, K, WaveSize>::shift(mode);
  for (int row = 0; row < 16; ++row) {
    for (int ck = 0; ck < K / 2; ++ck) {
      // Keep the accumulator kernels' stage ordering. Wave64 K=64 places
      // consecutive compressed halves in stages 0 and 1, unlike Wave32.
      int position = WaveSize == 64 && K == 64 ? 16 * ((ck / 8) & 1) + 8 * (ck / 16) + ck % 8 : ck;
      if constexpr (Bits == 8) {
        auto field = swmmac_a_field<8, WaveSize>(row, ck);
        input.a[row][ck] = extend(uint8_t(field.read(a)), mode & RJ_ARITH_WMMA_SIGNED_A);
        input.selected[row][ck] = uint8_t(swmmac_selected_k<8, WaveSize>(index, row, ck, key));
      } else {
        int lane = row + 16 * ((ck / 8) & 1), local = 8 * (ck / 16) + ck % 8;
        if constexpr (WaveSize == 64) {
          int block = position / 8;
          lane = row + 16 * (K == 32 ? block : 2 * (block & 1) + block / 2);
          local = position % 8;
        }
        uint32_t raw = (a[local * Bits / 32][lane] >> (local * Bits % 32)) & ((1u << Bits) - 1);
        input.a[row][ck] = extend(raw, mode & RJ_ARITH_WMMA_SIGNED_A);
        input.selected[row][ck] =
            uint8_t(4 * (position / 2) + ((index[0][lane] >> (key + 2 * local)) & 3));
      }
    }
  }

  // B remains dense; selected[row][ck] will choose its reduction coordinate.
  for (int k = 0; k < K; ++k) {
    for (int col = 0; col < 16; ++col) {
      if constexpr (Bits == 8) {
        input.b[k][col] =
            extend(swmmac_b_bits<8, WaveSize>(b, col, k), mode & RJ_ARITH_WMMA_SIGNED_B);
      } else {
        int lane = col + 16 * ((k / 16) & 1), local = 16 * (k / 32) + k % 16;
        if constexpr (WaveSize == 64) {
          int block = k / (K / 4);
          lane = col + 32 * (block & 1) + 16 * (block / 2);
          local = k % (K / 4);
        }
        uint32_t raw = (b[local * Bits / 32][lane] >> (local * Bits % 32)) & ((1u << Bits) - 1);
        input.b[k][col] = extend(raw, mode & RJ_ARITH_WMMA_SIGNED_B);
      }
    }
  }
  for (int row = 0; row < 16; ++row) {
    for (int col = 0; col < 16; ++col) {
      uint32_t raw = d[row % 8][col + 16 * (row / 8)];
      input.acc[row][col] = int32_t(int64_t(raw) - ((raw >> 31) ? (INT64_C(1) << 32) : 0));
    }
  }
}

// Return the compressed product position within one hardware accumulation stage.
inline int swmmac_integer_ck(int stage, int local) {
  return wmma_gfx12_integer_stage_k(stage, local);
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
