// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef UTIL_DETAIL_CONVERT_X86_AVX2_H_
#define UTIL_DETAIL_CONVERT_X86_AVX2_H_

#include <cstddef>
#include <cstdint>
#include <immintrin.h>

namespace util::detail {

inline void bf16_to_f32_block_avx2(const uint16_t *src, float *dst, size_t n, size_t &i) {
  for (; i + 8 <= n; i += 8) {
    __m256i w = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(src + i)));
    _mm256_storeu_ps(&dst[i], _mm256_castsi256_ps(_mm256_slli_epi32(w, 16)));
  }
}

inline void i8_to_i32_block_avx2(const int8_t *src, int32_t *dst, size_t n, size_t &i) {
  for (; i + 8 <= n; i += 8)
    _mm256_storeu_si256(
        reinterpret_cast<__m256i *>(&dst[i]),
        _mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(src + i))));
}

inline void u8_to_i32_block_avx2(const uint8_t *src, int32_t *dst, size_t n, size_t &i) {
  for (; i + 8 <= n; i += 8)
    _mm256_storeu_si256(
        reinterpret_cast<__m256i *>(&dst[i]),
        _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(src + i))));
}

} // namespace util::detail

#endif // UTIL_DETAIL_CONVERT_X86_AVX2_H_
