// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Bulk CDNA FP8/FNUZ widening without a lookup table.
#pragma once
#include "common/x86_64/fp8.h"
#include "mfma/x86_64/reduction.h"
#include <immintrin.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {
// Decode eight zero-extended source bytes in integer lanes.
template <bool Bf8, bool Fnuz> inline __m256 mfma_widen_fp8(__m256i bytes) {
  if constexpr (!Fnuz) {
    return widen_fp8<Bf8>(bytes);
  }
  constexpr int fraction_bits = Bf8 ? 2 : 3;
  constexpr int bias = Bf8 ? 16 : 8;
  auto magnitude = _mm256_and_si256(bytes, _mm256_set1_epi32(127));
  auto fraction = _mm256_and_si256(magnitude, _mm256_set1_epi32((1 << fraction_bits) - 1));
  auto normal = _mm256_add_epi32(_mm256_slli_epi32(magnitude, 23 - fraction_bits),
                                 _mm256_set1_epi32((127 - bias) << 23));
  auto small = _mm256_cmpgt_epi32(_mm256_set1_epi32(1 << fraction_bits), magnitude);
  auto subnormal =
      _mm256_mul_ps(_mm256_cvtepi32_ps(fraction), _mm256_set1_ps(Bf8 ? 0x1p-17f : 0x1p-10f));
  auto result = _mm256_blendv_epi8(normal, _mm256_castps_si256(subnormal), small);
  auto sign = _mm256_slli_epi32(_mm256_and_si256(bytes, _mm256_set1_epi32(128)), 24);
  result = _mm256_or_si256(result, sign);
  auto nan = _mm256_cmpeq_epi32(bytes, _mm256_set1_epi32(128));
  return _mm256_castsi256_ps(_mm256_blendv_epi8(result, _mm256_set1_epi32(int(0xffc00000U)), nan));
}

// Decode both packed factors once, then reuse the ISA-specific row reduction.
template <int M, int K, bool Bf8A, bool Bf8B, bool Fnuz>
void mfma_fp8_kernel(uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
                     const uint32_t *const *b, const uint32_t *const *c) {
  alignas(64) float left[M * K], right[M * K];
  for (int i = 0; i < M * K; i += 8) {
    auto x = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(a[i / 256] + i / 4 % 64));
    auto y = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(b[i / 256] + i / 4 % 64));
    _mm256_store_ps(left + i, mfma_widen_fp8<Bf8A, Fnuz>(_mm256_cvtepu8_epi32(x)));
    _mm256_store_ps(right + i, mfma_widen_fp8<Bf8B, Fnuz>(_mm256_cvtepu8_epi32(y)));
  }
  mfma_reduce_simd<M, K, 1, 4>(controls, d, left, right, c);
}
} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
