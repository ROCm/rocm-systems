// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// gfx1250 Wave32 half-input WMMA using AVX2 or AVX-512 row FMAs.
#pragma once
#include "common/internal.h"
#include "wmma/gfx1250_half.h"
#include "wmma/x86_64/gfx1250_reduction.h"
#include <immintrin.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {
// Reduce K in order with C added at the point prescribed by the legacy format policy.
template <Wmma1250Form Form>
void wmma_gfx1250_half_kernel(uint64_t flags, uint64_t controls, uint32_t *const *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              const uint32_t *const *c) {
  constexpr bool Bf16 = Form != Wmma1250Form::F32F16 && Form != Wmma1250Form::F16;
  constexpr bool DotFirst = Form == Wmma1250Form::F32Bf16 || Form == Wmma1250Form::Bf16F32;
  alignas(64) float left[512], right[512];
  for (int reg = 0; reg < 8; ++reg) {
    for (int i = 0; i < 64; i += 8) {
      auto x = _mm_loadu_si128(reinterpret_cast<const __m128i *>(a[reg] + i / 2));
      auto y = _mm_loadu_si128(reinterpret_cast<const __m128i *>(b[reg] + i / 2));
      if constexpr (Bf16) {
        _mm256_store_ps(left + reg * 64 + i,
                        _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(x), 16)));
        _mm256_store_ps(right + reg * 64 + i,
                        _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(y), 16)));
      } else {
        _mm256_store_ps(left + reg * 64 + i, _mm256_cvtph_ps(x));
        _mm256_store_ps(right + reg * 64 + i, _mm256_cvtph_ps(y));
      }
    }
  }
  alignas(64) float rows[16][32], columns[32][16], result[16][16];
  for (int i = 0; i < 16; ++i) {
    for (int k = 0; k < 32; ++k) {
      rows[i][k] = left[gfx1250_half_index(i, k)];
      columns[k][i] = right[gfx1250_half_index(i, k)];
    }
    for (int col = 0; col < 16; ++col) {
      result[i][col] = gfx1250_read_c<Form>(c, i, col, controls);
    }
  }
  gfx1250_reduce_rows<32, DotFirst>(rows, columns, result);
  if constexpr (Form == Wmma1250Form::Bf16F32) {
    for (int row = 0; row < 16; ++row) {
      for (int col = 0; col < 16; ++col) {
        if ((as_bits(result[row][col]) & 0x7fffffffU) > 0x7f800000U) {
          result[row][col] = gfx1250_repair_nan(result[row][col], left, right, row, col,
                                                gfx1250_read_c<Form>(c, row, col, controls));
        }
      }
    }
  }
  gfx1250_write_result<Form>(flags, d, result);
}
} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
