// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// gfx11 WMMA: replicated factor layouts and two-product accumulation stages.
// The gfx12 counterparts have distinct layouts and arithmetic and retain their suffix.

#include "common/float_formats.h"
#include "common/internal.h"
#include "dot/gfx11_dot2.h"
#include "rj_arith/rj_arith.h"

#include <cmath>

// Evaluate a gfx11 tile, reading all operands before any destination stores.
template <bool Bf16, int Lanes>
static int wmma(uint64_t flags, uint64_t instruction_flags, uint32_t *const *d,
                const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  if (instruction_flags) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (int error = rocjitsu::arith::validate(flags)) {
    return error;
  }
  constexpr int Groups = Lanes / 16, Regs = 256 / Lanes;
  uint32_t result[Regs][Lanes];
  for (int row = 0; row < 16; ++row) {
    for (int col = 0; col < 16; ++col) {
      int group = row % Groups, lane = group * 16 + col, reg = row / Groups;
      uint32_t acc = c[reg][lane];
      float loose = rocjitsu::arith::as_float(acc);
      for (int k = 0; k < 16; k += 2) {
        uint32_t av = a[k / 2][group * 16 + row];
        uint32_t bv = b[k / 2][group * 16 + col];
        if (rocjitsu::arith::is_exact(flags)) {
          acc = rocjitsu::arith::gfx11_dot2_f32<Bf16>(av, bv, av >> 16, bv >> 16, acc);
        } else {
          for (int half = 0; half < 2; ++half) {
            uint16_t x = uint16_t(av >> (16 * half)), y = uint16_t(bv >> (16 * half));
            loose = std::fma(
                Bf16 ? rocjitsu::arith::bf16_to_float(x) : rocjitsu::arith::f16_to_float(x),
                Bf16 ? rocjitsu::arith::bf16_to_float(y) : rocjitsu::arith::f16_to_float(y), loose);
          }
        }
      }
      result[reg][lane] = rocjitsu::arith::is_exact(flags) ? acc : rocjitsu::arith::as_bits(loose);
    }
  }
  for (int reg = 0; reg < Regs; ++reg) {
    for (int lane = 0; lane < Lanes; ++lane) {
      d[reg][lane] = result[reg][lane];
    }
  }
  return RJ_ARITH_SUCCESS;
}

// Evaluate gfx11 Wave32 F16 WMMA with FP32 results.
int rj_arith_v_wmma_f32_16x16x16_f16_w32(uint64_t flags, uint64_t instruction_flags,
                                         uint32_t *const *d, const uint32_t *const *a,
                                         const uint32_t *const *b, const uint32_t *const *c) {
  return wmma<false, 32>(flags, instruction_flags, d, a, b, c);
}

// Evaluate gfx11 Wave64 F16 WMMA with FP32 results.
int rj_arith_v_wmma_f32_16x16x16_f16_w64(uint64_t flags, uint64_t instruction_flags,
                                         uint32_t *const *d, const uint32_t *const *a,
                                         const uint32_t *const *b, const uint32_t *const *c) {
  return wmma<false, 64>(flags, instruction_flags, d, a, b, c);
}

// Evaluate gfx11 Wave32 BF16 WMMA with FP32 results.
int rj_arith_v_wmma_f32_16x16x16_bf16_w32(uint64_t flags, uint64_t instruction_flags,
                                          uint32_t *const *d, const uint32_t *const *a,
                                          const uint32_t *const *b, const uint32_t *const *c) {
  return wmma<true, 32>(flags, instruction_flags, d, a, b, c);
}

// Evaluate gfx11 Wave64 BF16 WMMA with FP32 results.
int rj_arith_v_wmma_f32_16x16x16_bf16_w64(uint64_t flags, uint64_t instruction_flags,
                                          uint32_t *const *d, const uint32_t *const *a,
                                          const uint32_t *const *b, const uint32_t *const *c) {
  return wmma<true, 64>(flags, instruction_flags, d, a, b, c);
}
