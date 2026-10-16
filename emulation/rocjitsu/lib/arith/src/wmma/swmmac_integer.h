// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Shared decoded operands, sparse-index selection, and accumulation-stage
// mapping for integer SWMMAC scalar and SIMD backends. Layouts are adapted
// from rocjitsu shared/mma_exec.h; CLAMP ordering and width-specific index
// selection follow gfx1201 Wave32/Wave64 captures.

#pragma once

#include "rj_arith/rj_arith.h"
#include "wmma/sparse_layout.h"

#include <stdint.h>

namespace rocjitsu::arith {

struct SwmmacIntegerInputs {
  int32_t a[16][32];
  int32_t b[64][16];
  int32_t acc[16][16];
  uint8_t selected[16][32];
};

// Snapshot signed/unsigned factors, sparse indices, and signed D into logical
// matrices. D uses eight 32-lane sections; metadata pairs must select two
// increasing positions in each group of four. K is the dense reduction size.
template <int Bits, int K, bool Wave64 = false>
inline void swmmac_integer_prepare(SwmmacIntegerInputs &input, uint32_t mode, uint32_t *const *d,
                                   const uint32_t *const *a, const uint32_t *const *b,
                                   const uint32_t *const *index) {
  // Interpret one extracted Bits-wide value with the operand's signedness.
  auto extend = [](uint32_t x, bool sign) {
    return int32_t(x) - (sign && (x & (1u << (Bits - 1))) ? int32_t(1u << Bits) : 0);
  };
  unsigned key =
      Wave64 && Bits == 8 ? 8 * ((mode >> 7) & 3) : (mode & RJ_ARITH_SWMMAC_INDEX_KEY_1 ? 16 : 0);
  for (int row = 0; row < 16; ++row) {
    for (int ck = 0; ck < K / 2; ++ck) {
      // Keep the accumulator kernels' stage ordering. Wave64 K=64 places
      // consecutive compressed halves in stages 0 and 1, unlike Wave32.
      int position = Wave64 && K == 64 ? 16 * ((ck / 8) & 1) + 8 * (ck / 16) + ck % 8 : ck;
      if constexpr (Bits == 8) {
        constexpr int WaveSize = Wave64 ? 64 : 32;
        auto field = swmmac_a_field<8, WaveSize>(row, ck);
        input.a[row][ck] = extend(uint8_t(field.read(a)), mode & RJ_ARITH_WMMA_SIGNED_A);
        input.selected[row][ck] = uint8_t(swmmac_selected_k<8, WaveSize>(index, row, ck, key));
      } else {
        int lane = row + 16 * ((ck / 8) & 1), local = 8 * (ck / 16) + ck % 8;
        if constexpr (Wave64) {
          {
            int block = position / 8;
            lane = row + 16 * (K == 32 ? block : 2 * (block & 1) + block / 2);
            local = position % 8;
          }
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
        input.b[k][col] = extend(swmmac_b_bits < 8, Wave64 ? 64 : 32 > (b, col, k),
                                 mode & RJ_ARITH_WMMA_SIGNED_B);
      } else {
        int lane = col + 16 * ((k / 16) & 1), local = 16 * (k / 32) + k % 16;
        if constexpr (Wave64) {
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
  return 16 * (local / 8) + 8 * stage + local % 8;
}

// Accumulate decoded sparse INT8/INT4 operands and write eight 32-lane D
// sections. k is the dense reduction size (32 or 64); clamp saturates each
// stage, otherwise results wrap. Input snapshots must not overlap D.
void swmmac_integer_x86_64_v3(int k, bool clamp, uint32_t *const *d,
                              const SwmmacIntegerInputs &input);
// Accumulate decoded sparse INT8/INT4 operands and write eight 32-lane D
// sections. k is the dense reduction size (32 or 64); clamp saturates each
// stage, otherwise results wrap. Input snapshots must not overlap D.
void swmmac_integer_x86_64_v4(int k, bool clamp, uint32_t *const *d,
                              const SwmmacIntegerInputs &input);

} // namespace rocjitsu::arith
