// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Stepwise FP32 rounding toward zero for BVH projection. AVX-512 supplies local
// rounding controls; AVX2 corrects default-rounded results using FP64 residuals.
// Host controls must be default and are never changed. Only the low four lanes
// are consumed; backend-local vector types do not cross compilation targets.
#pragma once

#include <immintrin.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::bvh::bvh_rounding {
#if defined(RJ_ARITH_BVH_AVX512)
inline namespace avx512 {
using Vector = __m512;

// Round each operation directly, suppressing host exception side effects.
[[gnu::always_inline]] inline Vector add(Vector a, Vector b) {
  return _mm512_add_round_ps(a, b, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
// Subtract with per-instruction rounding toward zero.
[[gnu::always_inline]] inline Vector subtract(Vector a, Vector b) {
  return _mm512_sub_round_ps(a, b, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
// Multiply with per-instruction rounding toward zero.
[[gnu::always_inline]] inline Vector multiply(Vector a, Vector b) {
  return _mm512_mul_round_ps(a, b, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
#else
inline namespace avx2 {
using Vector = __m128;

// Move an away-from-zero nearest result one FP32 encoding toward zero.
[[gnu::always_inline]] inline Vector correct(Vector nearest, __m256d away) {
  const auto packed = _mm256_permutevar8x32_epi32(_mm256_castpd_si256(away),
                                                  _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6));
  const auto decrement = _mm_srli_epi32(_mm256_castsi256_si128(packed), 31);
  return _mm_castsi128_ps(_mm_sub_epi32(_mm_castps_si128(nearest), decrement));
}

// FastTwoSum with the larger input first recovers the residual's sign, even
// when adding a tiny opposite-sign value would disappear in an FP64 sum.
[[gnu::always_inline]] inline Vector correct_sum(Vector a, Vector b, Vector nearest) {
  const auto sign = _mm_set1_ps(-0.0f);
  const auto a_is_larger = _mm_cmp_ps(_mm_andnot_ps(sign, a), _mm_andnot_ps(sign, b), _CMP_GE_OQ);
  const auto larger = _mm_blendv_ps(b, a, a_is_larger);
  const auto smaller = _mm_blendv_ps(a, b, a_is_larger);
  const auto wide_nearest = _mm256_cvtps_pd(nearest);
  const auto error =
      _mm256_add_pd(_mm256_sub_pd(_mm256_cvtps_pd(larger), wide_nearest), _mm256_cvtps_pd(smaller));
  const auto signed_error = _mm256_xor_pd(error, _mm256_and_pd(wide_nearest, _mm256_set1_pd(-0.0)));
  return correct(nearest, _mm256_cmp_pd(signed_error, _mm256_setzero_pd(), _CMP_LT_OQ));
}

// Preserve native zero/NaN selection while correcting finite nearest rounding.
[[gnu::always_inline]] inline Vector add(Vector a, Vector b) {
  return correct_sum(a, b, _mm_add_ps(a, b));
}
// Keep native subtraction's NaN sign while correcting its rounding.
[[gnu::always_inline]] inline Vector subtract(Vector a, Vector b) {
  return correct_sum(a, _mm_xor_ps(b, _mm_set1_ps(-0.0f)), _mm_sub_ps(a, b));
}

// Every finite FP32 product fits exactly in FP64, including under/overflow.
[[gnu::always_inline]] inline Vector multiply(Vector a, Vector b) {
  const auto nearest = _mm_mul_ps(a, b);
  const auto exact = _mm256_mul_pd(_mm256_cvtps_pd(a), _mm256_cvtps_pd(b));
  const auto sign = _mm256_set1_pd(-0.0);
  return correct(nearest, _mm256_cmp_pd(_mm256_andnot_pd(sign, _mm256_cvtps_pd(nearest)),
                                        _mm256_andnot_pd(sign, exact), _CMP_GT_OQ));
}
#endif

// Pack the three vertices, with a spare fourth lane.
inline Vector pack(float a, float b, float c) {
  const auto low = _mm_setr_ps(a, b, c, 0.0f);
#if defined(RJ_ARITH_BVH_AVX512)
  return _mm512_zextps128_ps512(low);
#else
  return low;
#endif
}

// Repeat a ray coordinate across the packed vertices.
inline Vector broadcast(float value) {
#if defined(RJ_ARITH_BVH_AVX512)
  return _mm512_set1_ps(value);
#else
  return _mm_set1_ps(value);
#endif
}

// Permute within each four-lane group, using an immediate index encoding.
template <int Indices> inline Vector permute(Vector value) {
#if defined(RJ_ARITH_BVH_AVX512)
  return _mm512_permute_ps(value, Indices);
#else
  return _mm_shuffle_ps(value, value, Indices);
#endif
}

// Interleave distances and edge weights for their two ordered reductions.
inline Vector interleave_low(Vector a, Vector b) {
#if defined(RJ_ARITH_BVH_AVX512)
  return _mm512_unpacklo_ps(a, b);
#else
  return _mm_unpacklo_ps(a, b);
#endif
}
// Interleave the remaining vertex and spare lane for the final reduction.
inline Vector interleave_high(Vector a, Vector b) {
#if defined(RJ_ARITH_BVH_AVX512)
  return _mm512_unpackhi_ps(a, b);
#else
  return _mm_unpackhi_ps(a, b);
#endif
}

// Publish the low four lanes into scalar geometry/result storage.
inline void store(float *destination, Vector value) {
#if defined(RJ_ARITH_BVH_AVX512)
  _mm_storeu_ps(destination, _mm512_castps512_ps128(value));
#else
  _mm_storeu_ps(destination, value);
#endif
}

#if defined(RJ_ARITH_BVH_AVX512)
} // namespace avx512
#else
} // namespace avx2
#endif
} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::bvh::bvh_rounding
