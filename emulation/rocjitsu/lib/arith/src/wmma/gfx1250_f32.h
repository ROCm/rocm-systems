// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// FP32 gfx1250 WMMA operand staging and reduction; each ISA owns its instantiation.
#pragma once
#include "wmma/gfx1250_half.h"
#if defined(__AVX2__)
#include "wmma/x86_64/gfx1250_reduction.h"
#endif

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {
// Compute the complete result before publishing any destination, permitting operand overlap.
inline void wmma_gfx1250_f32_kernel(uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, const uint32_t *const *c) {
  alignas(64) float rows[16][4], columns[4][16], result[16][16];
  for (int i = 0; i < 16; ++i) {
    for (int k = 0; k < 4; ++k) {
      // Two adjacent K elements occupy two VGPRs; the next pair uses the upper halfwave.
      rows[i][k] = as_float(a[k % 2][i + 16 * (k / 2)]);
      columns[k][i] = as_float(b[k % 2][i + 16 * (k / 2)]);
    }
    for (int j = 0; j < 16; ++j) {
      result[i][j] = gfx1250_read_c<Wmma1250Form::F32F16>(c, i, j, controls);
    }
  }
#if defined(__AVX2__)
  gfx1250_reduce_rows<4, false>(rows, columns, result);
#else
  for (int row = 0; row < 16; ++row) {
    for (int col = 0; col < 16; ++col) {
      for (int k = 0; k < 4; ++k) {
        result[row][col] = __builtin_fmaf(rows[row][k], columns[k][col], result[row][col]);
      }
    }
  }
#endif
  gfx1250_write_result<Wmma1250Form::F32F16>(0, d, result);
}
} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
