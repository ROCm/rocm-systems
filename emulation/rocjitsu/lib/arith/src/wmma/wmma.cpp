// SPDX-License-Identifier: MIT

// Dense gfx12 Wave32 FP16 WMMA with FP32 accumulation.
// Stage the complete result before writing to permit input/output aliases.

#include "common/float_formats.h"
#include "common/internal.h"
#include "dot/dot.h"
#include "rj_arith/rj_arith.h"

#include <cmath>
#include <stdint.h>

// Evaluate one complete matrix tile. Reject unsupported controls before accessing operands.
template <bool Bf16>
static int wmma(uint64_t flags, uint64_t instruction_flags, uint32_t *const *d,
                const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  if (instruction_flags) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (int error = rocjitsu::arith::validate(flags, 0, true)) {
    return error;
  }
  uint32_t result[8][32];
  // Eight K elements per source lane; two adjacent elements share a word.
  const auto read = [](const uint32_t *const *v, int index, int k) {
    return uint16_t(v[(k % 8) / 2][index + 16 * (k / 8)] >> (16 * (k % 2)));
  };
  for (int row = 0; row < 16; ++row) {
    for (int col = 0; col < 16; ++col) {
      int lane = col + 16 * (row / 8);
      uint32_t acc_bits = c[row % 8][lane];
      if (rocjitsu::arith::is_exact(flags)) {
        // gfx12 Wave32 consumes physical K groups in order 0,8,4,12.
        for (int k = 0; k < 16; k += 4) {
          std::array<uint16_t, 4> left, right;
          for (int j = 0; j < 4; ++j) {
            int logical = k + j;
            int physical = (logical & 3) | ((logical & 4) << 1) | ((logical & 8) >> 1);
            left[j] = read(a, row, physical);
            right[j] = read(b, col, physical);
          }
          acc_bits = rocjitsu::arith::gfx12_dot_bits<Bf16, 4, false>(left, right, acc_bits);
        }
      } else {
        float acc = rocjitsu::arith::as_float(c[row % 8][lane]);
        for (int k = 0; k < 16; ++k) {
          acc = std::fma(Bf16 ? rocjitsu::arith::bf16_to_float(read(a, row, k))
                              : rocjitsu::arith::f16_to_float(read(a, row, k)),
                         Bf16 ? rocjitsu::arith::bf16_to_float(read(b, col, k))
                              : rocjitsu::arith::f16_to_float(read(b, col, k)),
                         acc);
        }
        acc_bits = rocjitsu::arith::as_bits(acc);
      }
      result[row % 8][lane] = acc_bits;
    }
  }
  for (int reg = 0; reg < 8; ++reg) {
    for (int lane = 0; lane < 32; ++lane) {
      d[reg][lane] = result[reg][lane];
    }
  }
  return RJ_ARITH_SUCCESS;
}

// Evaluate one gfx12 Wave32 F16 tile into FP32 destination registers.
int rj_arith_v_wmma_f32_16x16x16_f16_w32_gfx12(uint64_t flags, uint64_t instruction_flags,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b, const uint32_t *const *c) {
  return wmma<false>(flags, instruction_flags, d, a, b, c);
}

// Evaluate one gfx12 Wave32 BF16 tile into FP32 destination registers.
int rj_arith_v_wmma_f32_16x16x16_bf16_w32_gfx12(uint64_t flags, uint64_t instruction_flags,
                                                uint32_t *const *d, const uint32_t *const *a,
                                                const uint32_t *const *b,
                                                const uint32_t *const *c) {
  return wmma<true>(flags, instruction_flags, d, a, b, c);
}
