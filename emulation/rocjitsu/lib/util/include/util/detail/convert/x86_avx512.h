// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef UTIL_DETAIL_CONVERT_X86_AVX512_H_
#define UTIL_DETAIL_CONVERT_X86_AVX512_H_

#include <cstddef>
#include <cstdint>
#include <immintrin.h>

namespace util::detail {

inline void f16_to_f32_block_avx512(const uint16_t *src, float *dst, size_t n, size_t &i) {
  for (; i + 16 <= n; i += 16)
    _mm512_storeu_ps(
        &dst[i], _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(src + i))));
}

inline void bf16_to_f32_block_avx512(const uint16_t *src, float *dst, size_t n, size_t &i) {
  for (; i + 16 <= n; i += 16) {
    __m512i w =
        _mm512_cvtepu16_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(src + i)));
    _mm512_storeu_ps(&dst[i], _mm512_castsi512_ps(_mm512_slli_epi32(w, 16)));
  }
}

inline void i8_to_i32_block_avx512(const int8_t *src, int32_t *dst, size_t n, size_t &i) {
  for (; i + 16 <= n; i += 16)
    _mm512_storeu_si512(
        reinterpret_cast<__m512i *>(&dst[i]),
        _mm512_cvtepi8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(src + i))));
}

inline void u8_to_i32_block_avx512(const uint8_t *src, int32_t *dst, size_t n, size_t &i) {
  for (; i + 16 <= n; i += 16)
    _mm512_storeu_si512(
        reinterpret_cast<__m512i *>(&dst[i]),
        _mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(src + i))));
}

} // namespace util::detail

#endif // UTIL_DETAIL_CONVERT_X86_AVX512_H_
