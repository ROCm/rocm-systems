// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#if defined(__AVX512F__) && defined(__AVX512VL__) && defined(__FMA__)
#include <immintrin.h>

#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace rocjitsu::amdgpu {

// The fixed kernel keeps scaling entirely in integer registers. Every FP4
// 32-term dot product is a multiple of 1/4 with magnitude at most 1152, so
// combined exponents in [-124, 117] make every nonzero scaled sum normal.
// The general-scale specialization handles values outside this range.
inline bool mfma_fp4_avx512_normal_scales(const uint8_t *a, const uint8_t *b) {
  for (uint32_t block = 0; block < 4; ++block) {
    uint32_t amin = 255, amax = 0, bmin = 255, bmax = 0;
    for (uint32_t i = 0; i < 16; ++i) {
      amin = std::min(amin, uint32_t(a[block * 16 + i]));
      amax = std::max(amax, uint32_t(a[block * 16 + i]));
      bmin = std::min(bmin, uint32_t(b[block * 16 + i]));
      bmax = std::max(bmax, uint32_t(b[block * 16 + i]));
    }
    if (amax == 255 || bmax == 255 || amin + bmin < 130 || amax + bmax > 371)
      return false;
  }
  return true;
}

// This FP4-only specialization omits the generic input-exponent checks.
// Sums of finite FP4 products are normal or signed zero. Scale those directly
// when the result stays normal; preserve libm rounding, errno and exception
// behavior for the exceptional nonzero lanes. Zero rows commonly carry small
// scale bytes in padded tiles, so do not reject an entire instruction for them.
inline __m512 mfma_fp4_scale_avx512(__m512 sums, uint8_t a, const uint8_t *b) {
  const auto qnan = _mm512_set1_epi32(0x7fc00000);
  if (a == 255)
    return _mm512_castsi512_ps(qnan);
  const auto scales = _mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(b)));
  const auto bits = _mm512_castps_si512(sums);
  const auto exponent = _mm512_and_si512(_mm512_srli_epi32(bits, 23), _mm512_set1_epi32(255));
  const auto scaled =
      _mm512_add_epi32(exponent, _mm512_add_epi32(scales, _mm512_set1_epi32(int(a) - 254)));
  const auto normal = _mm512_cmp_epi32_mask(scaled, _mm512_setzero_si512(), _MM_CMPINT_GT) &
                      _mm512_cmp_epi32_mask(scaled, _mm512_set1_epi32(255), _MM_CMPINT_LT);
  const auto zero = _mm512_cmpeq_epi32_mask(_mm512_and_si512(bits, _mm512_set1_epi32(0x7fffffff)),
                                            _mm512_setzero_si512());
  assert(((_mm512_cmp_epi32_mask(exponent, _mm512_setzero_si512(), _MM_CMPINT_GT) &
           _mm512_cmp_epi32_mask(exponent, _mm512_set1_epi32(255), _MM_CMPINT_LT)) |
          zero) == 0xffffu &&
         "FP4 block sums must be normal or signed zero");
  const auto nan = _mm512_cmpeq_epi32_mask(scales, _mm512_set1_epi32(255));
  auto output =
      _mm512_mask_mov_epi32(bits, normal & ~zero,
                            _mm512_or_si512(_mm512_and_si512(bits, _mm512_set1_epi32(0x807fffff)),
                                            _mm512_slli_epi32(scaled, 23)));
  output = _mm512_mask_mov_epi32(output, nan, qnan);
  unsigned fallback = static_cast<__mmask16>(~(normal | zero | nan));
  if (fallback) {
    alignas(64) float source[16], result[16];
    _mm512_store_ps(source, sums);
    _mm512_store_ps(result, _mm512_castsi512_ps(output));
    do {
      const unsigned lane = std::countr_zero(fallback);
      result[lane] = std::ldexp(source[lane], int(a) + int(b[lane]) - 254);
      fallback &= fallback - 1;
    } while (fallback);
    return _mm512_load_ps(result);
  }
  return _mm512_castsi512_ps(output);
}

// Preserve the accumulator's NaN payload when the scale also produces NaN.
// C++ addition may commute the two inputs; fix their hardware operand order.
inline __m512 mfma_fp4_add_avx512(__m512 accumulator, __m512 scaled) {
  __m512 result;
  asm volatile("vaddps %2, %1, %0" : "=v"(result) : "v"(accumulator), "v"(scaled));
  return result;
}

// One register contains eight packed K values per lane; each group of sixteen
// lanes supplies one 32-K block. Decode A to row-major and B to K-major storage.
inline void mfma_fp4_stage_a_avx512(const uint32_t *lanes, uint32_t reg, float *a) {
  const __m256 positive = _mm256_setr_ps(0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f);
  const __m256 negative = _mm256_setr_ps(-0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f);
  const __m256i shifts = _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28);
  for (uint32_t block = 0; block < 4; ++block)
    for (uint32_t row = 0; row < 16; ++row) {
      const __m256i indices = _mm256_srlv_epi32(_mm256_set1_epi32(lanes[block * 16 + row]), shifts);
      // VPERMI2PS uses the low four index bits, including the FP4 sign bit.
      _mm256_storeu_ps(a + row * 128 + block * 32 + reg * 8,
                       _mm256_permutex2var_ps(positive, indices, negative));
    }
}

inline void mfma_fp4_stage_b_avx512(const uint32_t *lanes, uint32_t reg, float *b) {
  const __m512 values = _mm512_setr_ps(0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f, -0.0f, -0.5f,
                                       -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f);
  for (uint32_t block = 0; block < 4; ++block) {
    const __m512i words = _mm512_loadu_si512(lanes + block * 16);
    [&]<std::size_t... Element>(std::index_sequence<Element...>) {
      ((_mm512_storeu_ps(b + (block * 32 + reg * 8 + Element) * 16,
                         _mm512_permutexvar_ps(_mm512_srli_epi32(words, Element * 4), values))),
       ...);
    }(std::make_index_sequence<8>{});
  }
}

// Preconditions: dense A[16][128], B[128][16], C[16][16]. NormalScales
// requires scales admitted above. Each output retains increasing K order and four separate block
// adds. Sixteen independent ZMM accumulators share each B load and cover FMA latency.
template <bool NormalScales = true>
inline void mfma_fp4_16x16x128_avx512(const float *a, const float *b, float *c,
                                      const uint8_t *a_scales, const uint8_t *b_scales) {
  for (uint32_t block = 0; block < 4; ++block) {
    __m512 sums[16];
    for (auto &sum : sums)
      sum = _mm512_setzero_ps();
    for (uint32_t k = block * 32; k < (block + 1) * 32; ++k) {
      const __m512 bv = _mm512_loadu_ps(b + k * 16);
      [&]<std::size_t... Row>(std::index_sequence<Row...>) __attribute__((always_inline)) {
        ((sums[Row] = _mm512_fmadd_ps(_mm512_set1_ps(a[Row * 128 + k]), bv, sums[Row])), ...);
      }
      (std::make_index_sequence<16>{});
    }
    const __m512i bs = _mm512_cvtepu8_epi32(
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(b_scales + block * 16)));
    [&]<std::size_t... Row>(std::index_sequence<Row...>) __attribute__((always_inline)) {
      (
          [&] {
            __m512 scaled;
            if constexpr (NormalScales) {
              const __m512i exponent =
                  _mm512_add_epi32(bs, _mm512_set1_epi32(int(a_scales[block * 16 + Row]) - 254));
              const __m512i bits = _mm512_castps_si512(sums[Row]);
              const __mmask16 nonzero =
                  _mm512_cmp_ps_mask(sums[Row], _mm512_setzero_ps(), _CMP_NEQ_OQ);
              scaled = _mm512_castsi512_ps(
                  _mm512_mask_add_epi32(bits, nonzero, bits, _mm512_slli_epi32(exponent, 23)));
            } else {
              scaled = mfma_fp4_scale_avx512(sums[Row], a_scales[block * 16 + Row],
                                             b_scales + block * 16);
            }
            _mm512_storeu_ps(c + Row * 16,
                             mfma_fp4_add_avx512(_mm512_loadu_ps(c + Row * 16), scaled));
          }(),
          ...);
    }
    (std::make_index_sequence<16>{});
  }
}

} // namespace rocjitsu::amdgpu
#endif
