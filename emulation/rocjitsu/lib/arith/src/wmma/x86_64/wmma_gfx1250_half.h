// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// gfx1250 Wave32 half-input WMMA using AVX2 or AVX-512 row FMAs.
#pragma once
#include "common/internal.h"
#if defined(__AVX512F__)
#include "common/x86_64/half_x86_64_v4.h"
#endif
#include "wmma/gfx1250_half.h"
#include "wmma/x86_64/gfx1250_accumulator.h"
#include "wmma/x86_64/gfx1250_reduction.h"
#include <immintrin.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {
// Widen the K pair that 16 adjacent VGPR lanes hold for every row or column into two tiles.
template <bool Bf16>
inline void gfx1250_widen_half_pair(const uint32_t *words, float *even, float *odd) {
#if defined(__AVX512F__)
  const auto packed = _mm512_loadu_si512(words);
  if constexpr (Bf16) {
    _mm512_store_ps(even, _mm512_castsi512_ps(_mm512_slli_epi32(packed, 16)));
    _mm512_store_ps(odd, _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_srli_epi32(packed, 16), 16)));
  } else {
    _mm512_store_ps(even, half_input(packed, 0, 0));
    _mm512_store_ps(odd, half_input(packed, 16, 0));
  }
#else
  for (int lane = 0; lane < 16; lane += 8) {
    const auto packed = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(words + lane));
    if constexpr (Bf16) {
      _mm256_store_ps(even + lane, _mm256_castsi256_ps(_mm256_slli_epi32(packed, 16)));
      _mm256_store_ps(odd + lane,
                      _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_srli_epi32(packed, 16), 16)));
    } else {
      // Collect the even halves in the low and the odd halves in the high 128 bits.
      const auto order = _mm256_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15, 0,
                                          1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15);
      const auto split = _mm256_permute4x64_epi64(_mm256_shuffle_epi8(packed, order), 0xd8);
      _mm256_store_ps(even + lane, _mm256_cvtph_ps(_mm256_castsi256_si128(split)));
      _mm256_store_ps(odd + lane, _mm256_cvtph_ps(_mm256_extracti128_si256(split, 1)));
    }
  }
#endif
}

// Return whether the mixed-result row needs the scalar NaN-priority replay.
inline bool gfx1250_row_has_nan(const float *row) {
#if defined(__AVX512F__)
  auto value = _mm512_load_ps(row);
  return _mm512_cmp_ps_mask(value, value, _CMP_UNORD_Q) != 0;
#else
  auto lo = _mm256_load_ps(row);
  auto hi = _mm256_load_ps(row + 8);
  return _mm256_movemask_ps(_mm256_or_ps(_mm256_cmp_ps(lo, lo, _CMP_UNORD_Q),
                                         _mm256_cmp_ps(hi, hi, _CMP_UNORD_Q))) != 0;
#endif
}

// Reduce K in order with C added at the point prescribed by the legacy format policy.
template <Wmma1250Form Form>
void wmma_gfx1250_half_kernel(uint64_t flags, uint64_t controls, uint32_t *const *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              const uint32_t *const *c) {
  constexpr bool Bf16 = Form != Wmma1250Form::F32F16 && Form != Wmma1250Form::F16;
  constexpr bool DotFirst = Form == Wmma1250Form::F32Bf16 || Form == Wmma1250Form::Bf16F32;
  // Stage every source before publication: left[k][row], right[k][column] and C.
  alignas(64) float left[32][16], right[32][16], result[16][16];
  for (int k = 0; k < 32; k += 2) {
    const int half_index = gfx1250_half_index(0, k);
    const int reg = half_index / 64;
    const int lane = half_index / 2 % 32;
    gfx1250_widen_half_pair<Bf16>(a[reg] + lane, left[k], left[k + 1]);
    gfx1250_widen_half_pair<Bf16>(b[reg] + lane, right[k], right[k + 1]);
  }
  gfx1250_load_accumulators<Form>(c, controls, result);
  constexpr int exponent = Bf16 ? 0x7f80 : 0x7c00;
  if (gfx1250_has_nonfinite<8, false, exponent, exponent>(a, b)) {
    gfx1250_reduce_rows<32, DotFirst, true>(left, right, result);
  } else {
    gfx1250_reduce_interleaved<32, DotFirst>(left, right, result);
  }
  if constexpr (Form == Wmma1250Form::Bf16F32) {
    // Recompute mixed-result NaNs in the prescribed A, B, accumulator order.
    for (int row = 0; row < 16; ++row) {
      if (!gfx1250_row_has_nan(result[row])) {
        continue;
      }
      for (int col = 0; col < 16; ++col) {
        if ((as_bits(result[row][col]) & 0x7fffffffU) > 0x7f800000U) {
          float dot = 0.0f;
          for (int k = 0; k < 32; ++k) {
            dot = matrix_nan_fma(left[k][row], right[k][col], dot);
          }
          result[row][col] = matrix_nan_fma(dot, 1.0f, gfx1250_read_c<Form>(c, row, col, controls));
        }
      }
    }
  }
  gfx1250_store_results<Form>(flags, d, result);
}
} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
