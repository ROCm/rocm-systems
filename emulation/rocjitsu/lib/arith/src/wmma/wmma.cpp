// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Dense gfx12 Wave32/Wave64 FP16/BF16 WMMA with FP32 accumulation.
// Stage the complete result before writing to permit input/output aliases.

#include "common/float_formats.h"
#include "common/internal.h"
#include "dot/dot.h"
#include "rj_arith/rj_arith.h"

#include <array>
#include <cmath>
#include <stdint.h>

// Evaluate one complete matrix tile. Reject unsupported controls before accessing operands.
template <bool Bf16, int WaveSize>
static int wmma(uint64_t flags, uint64_t instruction_flags, uint32_t *const *d,
                const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  if (instruction_flags) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (int error = rocjitsu::arith::validate(flags)) {
    return error;
  }
  static_assert(WaveSize == 32 || WaveSize == 64);
  constexpr int acc_regs = 256 / WaveSize;
  constexpr int group_k = 16 / (WaveSize / 16);
  uint32_t result[acc_regs][WaveSize];
  // Each source lane holds eight K elements in Wave32 or four in Wave64;
  // two adjacent elements share a word.
  const auto read = [](const uint32_t *const *v, int index, int k) {
    return uint16_t(v[(k % group_k) / 2][index + 16 * (k / group_k)] >> (16 * (k % 2)));
  };
  const auto convert = [](uint16_t bits) {
    return Bf16 ? rocjitsu::arith::bf16_to_float(bits) : rocjitsu::arith::f16_to_float(bits);
  };
  for (int row = 0; row < 16; ++row) {
    for (int col = 0; col < 16; ++col) {
      int lane = col + 16 * (row / 8) + (WaveSize == 64 ? 32 * ((row / 4) % 2) : 0);
      uint32_t acc_bits = c[row % acc_regs][lane];
      if (rocjitsu::arith::is_exact(flags)) {
        // gfx12 Wave32 consumes K groups in order 0,8,4,12; Wave64 uses 0,4,8,12.
        for (int k = 0; k < 16; k += 4) {
          std::array<uint16_t, 4> left, right;
          for (int j = 0; j < 4; ++j) {
            int logical = k + j;
            int physical = WaveSize == 64
                               ? logical
                               : (logical & 3) | ((logical & 4) << 1) | ((logical & 8) >> 1);
            left[j] = read(a, row, physical);
            right[j] = read(b, col, physical);
          }
          acc_bits = rocjitsu::arith::gfx12_dot_bits<Bf16, 4, false>(left, right, acc_bits);
        }
      } else {
        float acc = rocjitsu::arith::as_float(c[row % acc_regs][lane]);
        for (int k = 0; k < 16; ++k) {
          acc = std::fma(convert(read(a, row, k)), convert(read(b, col, k)), acc);
        }
        acc_bits = rocjitsu::arith::as_bits(acc);
      }
      result[row % acc_regs][lane] = acc_bits;
    }
  }
  for (int reg = 0; reg < acc_regs; ++reg) {
    for (int lane = 0; lane < WaveSize; ++lane) {
      d[reg][lane] = result[reg][lane];
    }
  }
  return RJ_ARITH_SUCCESS;
}

// Evaluate a gfx12 Wave32 F16 tile into FP32 destination registers.
int rj_arith_v_wmma_f32_16x16x16_f16_w32_gfx12(uint64_t flags, uint64_t instruction_flags,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b, const uint32_t *const *c) {
  return wmma<false, 32>(flags, instruction_flags, d, a, b, c);
}

// Evaluate a gfx12 Wave64 F16 tile into FP32 destination registers.
int rj_arith_v_wmma_f32_16x16x16_f16_w64_gfx12(uint64_t flags, uint64_t instruction_flags,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b, const uint32_t *const *c) {
  return wmma<false, 64>(flags, instruction_flags, d, a, b, c);
}

// Evaluate a gfx12 Wave32 BF16 tile into FP32 destination registers.
int rj_arith_v_wmma_f32_16x16x16_bf16_w32_gfx12(uint64_t flags, uint64_t instruction_flags,
                                                uint32_t *const *d, const uint32_t *const *a,
                                                const uint32_t *const *b,
                                                const uint32_t *const *c) {
  return wmma<true, 32>(flags, instruction_flags, d, a, b, c);
}

// Evaluate a gfx12 Wave64 BF16 tile into FP32 destination registers.
int rj_arith_v_wmma_f32_16x16x16_bf16_w64_gfx12(uint64_t flags, uint64_t instruction_flags,
                                                uint32_t *const *d, const uint32_t *const *a,
                                                const uint32_t *const *b,
                                                const uint32_t *const *c) {
  return wmma<true, 64>(flags, instruction_flags, d, a, b, c);
}
