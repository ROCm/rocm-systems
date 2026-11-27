// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// gfx1250 Wave32 half-input WMMA using AVX2 or AVX-512 row FMAs.
#pragma once
#include "common/compiler.h"
#include "common/internal.h"
#include "wmma/gfx1250_half.h"
#include "wmma/x86_64/gfx1250_accumulator.h"
#include "wmma/x86_64/gfx1250_reduction.h"
#include <immintrin.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {
// Widen the K pair that 16 adjacent VGPR lanes hold for every row or column into two tiles.
template <bool Bf16>
inline void gfx1250_widen_half_pair(const uint32_t *words, float *even, float *odd) {
#if defined(RJ_ARITH_TARGET_X86_64_V4)
  const auto packed = _mm512_loadu_si512(words);
  if constexpr (Bf16) {
    _mm512_store_ps(even, _mm512_castsi512_ps(_mm512_slli_epi32(packed, 16)));
    _mm512_store_ps(
        odd, _mm512_castsi512_ps(_mm512_and_si512(packed, _mm512_set1_epi32(int(0xffff0000U)))));
  } else {
    _mm512_store_ps(even, _mm512_cvtph_ps(_mm512_cvtepi32_epi16(packed)));
    _mm512_store_ps(odd, _mm512_cvtph_ps(_mm512_cvtepi32_epi16(_mm512_srli_epi32(packed, 16))));
  }
#else
  for (int lane = 0; lane < 16; lane += 8) {
    const auto packed = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(words + lane));
    if constexpr (Bf16) {
      _mm256_store_ps(even + lane, _mm256_castsi256_ps(_mm256_slli_epi32(packed, 16)));
      _mm256_store_ps(odd + lane, _mm256_castsi256_ps(_mm256_and_si256(
                                      packed, _mm256_set1_epi32(int(0xffff0000U)))));
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

// Recompute mixed-result NaNs in the prescribed A, B, accumulator order.
[[gnu::noinline]] inline void gfx1250_repair_bf16f32_nans(const float left[32][16],
                                                          const float right[32][16],
                                                          const uint32_t *const *c,
                                                          uint64_t controls, float result[16][16]) {
#if defined(RJ_ARITH_TARGET_X86_64_V4)
  const __m512i keep = _mm512_set1_epi32((controls & RJ_ARITH_WMMA_ABS_C) ? INT32_MAX : -1);
  const __m512i negate = _mm512_set1_epi32((controls & RJ_ARITH_WMMA_NEG_C) ? INT32_MIN : 0);
  const __m512 one = _mm512_set1_ps(1.0f);
  for (int row = 0; row < 16; ++row) {
    const __m512 acc_row = _mm512_load_ps(result[row]);
    const __mmask16 row_nan = _mm512_cmp_ps_mask(acc_row, acc_row, _CMP_UNORD_Q);
    if (!row_nan) {
      continue;
    }
    __m512 dot = _mm512_setzero_ps();
    for (int k = 0; k < 32; ++k) {
      dot = gfx1250_matrix_nan_fma_v4(_mm512_set1_ps(left[k][row]), _mm512_load_ps(right[k]), dot);
    }
    const __m512i c_bits = _mm512_loadu_si512(c[row & 7] + 16 * (row >> 3));
    const __m512 init =
        _mm512_castsi512_ps(_mm512_xor_si512(_mm512_and_si512(c_bits, keep), negate));
    const __m512 repaired = gfx1250_matrix_nan_fma_v4(dot, one, init);
    _mm512_store_ps(result[row], _mm512_mask_blend_ps(row_nan, acc_row, repaired));
  }
#else
  for (int row = 0; row < 16; ++row) {
    for (int col = 0; col < 16; ++col) {
      if ((as_bits(result[row][col]) & 0x7fffffffU) > 0x7f800000U) {
        float dot = 0.0f;
        for (int k = 0; k < 32; ++k) {
          dot = matrix_nan_fma(left[k][row], right[k][col], dot);
        }
        result[row][col] =
            matrix_nan_fma(dot, 1.0f, gfx1250_read_c<Wmma1250Form::Bf16F32>(c, row, col, controls));
      }
    }
  }
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
  alignas(64) float left[32][16], right[32][16];
  for (int k = 0; k < 32; k += 2) {
    const int reg = 4 * (k / 16) + (k % 8) / 2;
    const int lane = 16 * ((k / 8) % 2);
    gfx1250_widen_half_pair<Bf16>(a[reg] + lane, left[k], left[k + 1]);
    gfx1250_widen_half_pair<Bf16>(b[reg] + lane, right[k], right[k + 1]);
  }
#if defined(RJ_ARITH_TARGET_X86_64_V4)
  __m512 acc[16];
  if constexpr (DotFirst) {
    RJ_ARITH_UNROLL(16)
    for (int row = 0; row < 16; ++row) {
      acc[row] = _mm512_setzero_ps();
    }
  } else {
    gfx1250_load_acc_zmm<Form>(c, controls, acc);
  }
  for (int k = 0; k < 32; ++k) {
    const __m512 column = _mm512_load_ps(right[k]);
    RJ_ARITH_UNROLL(16)
    for (int row = 0; row < 16; ++row) {
      acc[row] = _mm512_fmadd_ps(_mm512_set1_ps(left[k][row]), column, acc[row]);
    }
  }
  if constexpr (DotFirst) {
    const __m512i keep = _mm512_set1_epi32((controls & RJ_ARITH_WMMA_ABS_C) ? INT32_MAX : -1);
    const __m512i negate = _mm512_set1_epi32((controls & RJ_ARITH_WMMA_NEG_C) ? INT32_MIN : 0);
    RJ_ARITH_UNROLL(8)
    for (int reg = 0; reg < 8; ++reg) {
      RJ_ARITH_UNROLL(2)
      for (int group = 0; group < 2; ++group) {
        __m512i bits = _mm512_loadu_si512(c[reg] + 16 * group);
        __m512 init = _mm512_castsi512_ps(_mm512_xor_si512(_mm512_and_si512(bits, keep), negate));
        acc[reg + 8 * group] = _mm512_add_ps(acc[reg + 8 * group], init);
      }
    }
  }
  if constexpr (Form == Wmma1250Form::Bf16F32) {
    if ((flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_COMPATIBILITY) {
      __mmask16 any_nan = 0;
      RJ_ARITH_UNROLL(16)
      for (int row = 0; row < 16; ++row) {
        any_nan |= _mm512_cmp_ps_mask(acc[row], acc[row], _CMP_UNORD_Q);
      }
      if (any_nan) [[unlikely]] {
        alignas(64) float result[16][16];
        RJ_ARITH_UNROLL(16)
        for (int row = 0; row < 16; ++row) {
          _mm512_store_ps(result[row], acc[row]);
        }
        gfx1250_repair_bf16f32_nans(left, right, c, controls, result);
        gfx1250_store_results<Form>(flags, d, result);
        return;
      }
    }
  }
  gfx1250_store_acc_zmm<Form>(flags, d, acc);
#else
  alignas(64) float result[16][16];
  gfx1250_load_accumulators<Form>(c, controls, result);
  gfx1250_reduce_interleaved<32, DotFirst>(left, right, result);
  if constexpr (Form == Wmma1250Form::Bf16F32) {
    if ((flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_COMPATIBILITY) {
      __m256 any_nan = _mm256_setzero_ps();
      for (int row = 0; row < 16; ++row) {
        __m256 lo = _mm256_load_ps(result[row]);
        __m256 hi = _mm256_load_ps(result[row] + 8);
        any_nan = _mm256_or_ps(any_nan, _mm256_or_ps(_mm256_cmp_ps(lo, lo, _CMP_UNORD_Q),
                                                     _mm256_cmp_ps(hi, hi, _CMP_UNORD_Q)));
      }
      if (_mm256_movemask_ps(any_nan)) [[unlikely]] {
        gfx1250_repair_bf16f32_nans(left, right, c, controls, result);
      }
    }
  }
  gfx1250_store_results<Form>(flags, d, result);
#endif
}
} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
