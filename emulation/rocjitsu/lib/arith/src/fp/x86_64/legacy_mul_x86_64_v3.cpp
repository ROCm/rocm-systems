// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 inherited multiply, including directed guest rounding and output controls.
#include "fp/x86_64/legacy_mul_x86_64.h"

namespace rocjitsu::arith {
namespace {

// Correct nearest FP32 products toward the requested guest rounding direction.
// FP64 holds every finite FP32 product exactly, including FP32 overflow/underflow.
__m128 directed_product(__m128 x, __m128 y, __m128 nearest, unsigned rounding) {
  auto exact = _mm256_mul_pd(_mm256_cvtps_pd(x), _mm256_cvtps_pd(y));
  auto widened = _mm256_cvtps_pd(nearest);
  auto below = _mm256_cmp_pd(widened, exact, _CMP_LT_OQ);
  auto above = _mm256_cmp_pd(widened, exact, _CMP_GT_OQ);
  auto bits = _mm_castps_si128(nearest);
  auto negative = _mm_srai_epi32(bits, 31);
  auto up = _mm_srai_epi32(
      _mm_sllv_epi32(_mm_set1_epi32(_mm256_movemask_pd(below)), _mm_setr_epi32(31, 30, 29, 28)),
      31);
  auto down = _mm_srai_epi32(
      _mm_sllv_epi32(_mm_set1_epi32(_mm256_movemask_pd(above)), _mm_setr_epi32(31, 30, 29, 28)),
      31);
  if (rounding == 1) {
    down = _mm_setzero_si128();
  } else if (rounding == 2) {
    up = _mm_setzero_si128();
  } else {
    up = _mm_and_si128(up, negative);
    down = _mm_andnot_si128(negative, down);
  }
  // Moving upward decrements negative encodings and increments positive ones.
  auto step = _mm_or_si128(negative, _mm_set1_epi32(1));
  bits = _mm_add_epi32(bits, _mm_and_si128(up, step));
  bits = _mm_sub_epi32(bits, _mm_and_si128(down, step));
  return _mm_castsi128_ps(bits);
}

} // namespace

// Execute one Wave32 block; source/destination aliases and inactive lanes are valid.
void legacy_mul_x86_64_v3(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                          const uint32_t *a, const uint32_t *b) {
  const unsigned rounding = unsigned((flags & RJ_ARITH_FP_ROUND_MASK) >> 24);
  for (int lane = 0; lane < 32; lane += 8) {
    auto x = legacy_mul_input(a + lane, mode, flags);
    auto y = legacy_mul_input(b + lane, mode >> 1, flags);
    auto value = _mm256_mul_ps(x, y);
    if (rounding) {
      auto low = directed_product(_mm256_castps256_ps128(x), _mm256_castps256_ps128(y),
                                  _mm256_castps256_ps128(value), rounding);
      auto high = directed_product(_mm256_extractf128_ps(x, 1), _mm256_extractf128_ps(y, 1),
                                   _mm256_extractf128_ps(value, 1), rounding);
      value = _mm256_insertf128_ps(_mm256_castps128_ps256(low), high, 1);
    }
    auto result = legacy_mul_finish(value, x, y, flags, mode, a + lane, b + lane);
    auto active = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                                    _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), active, result);
  }
}

} // namespace rocjitsu::arith
