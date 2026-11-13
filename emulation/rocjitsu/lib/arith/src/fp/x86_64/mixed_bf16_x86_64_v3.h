// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 mixed-source decoding, compensated BF16 rounding and selected-lane repair.
// AVX512 reuses output handling while widening the FP64 arithmetic to eight lanes.
#pragma once

#include "common/isa.h"
#include "fp/mixed_bf16.h"
#include "rj_arith/detail/rj_arith_common.h"
#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Decode eight raw mixed operands, then apply their source ABS/NEG modifiers.
inline __m256 mixed_bf16_input8(const uint32_t *source, uint32_t mode) {
  auto bits = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(source));
  if (mode & RJ_ARITH_MIX_BF16_A) {
    if (mode & RJ_ARITH_ALU_HIGH_A) {
      bits = _mm256_srli_epi32(bits, 16);
    }
    bits = _mm256_slli_epi32(bits, 16);
  }
  if (mode & RJ_ARITH_ALU_ABS_A) {
    bits = _mm256_and_si256(bits, _mm256_set1_epi32(0x7fffffff));
  }
  if (mode & RJ_ARITH_ALU_NEG_A) {
    bits = _mm256_xor_si256(bits, _mm256_set1_epi32(INT32_MIN));
  }
  return _mm256_castsi256_ps(bits);
}

// Round four exact finite product-plus-addend sums to guest BF16 encodings.
// Product and addend supply the exact-zero sign; error retains lost sum bits.
inline __m128i mixed_bf16_narrow4(__m256d sum, __m256d error, __m256d product, __m256d addend,
                                  unsigned rounding, bool clamp) {
  const auto zero = _mm256_setzero_si256();
  const auto one = _mm256_set1_epi64x(1);
  const auto magnitude_mask = _mm256_set1_epi64x(INT64_MAX);
  auto bits = _mm256_castpd_si256(sum);
  const auto even = _mm256_cmpeq_epi64(_mm256_and_si256(bits, one), zero);
  const auto inexact = _mm256_castpd_si256(_mm256_cmp_pd(error, _mm256_setzero_pd(), _CMP_NEQ_OQ));
  const auto same_sign = _mm256_cmpeq_epi64(
      _mm256_srli_epi64(_mm256_xor_si256(bits, _mm256_castpd_si256(error)), 63), zero);
  const auto step = _mm256_blendv_epi8(_mm256_set1_epi64x(-1), one, same_sign);
  bits = _mm256_add_epi64(bits, _mm256_and_si256(step, _mm256_and_si256(even, inexact)));

  // Default-host +0 cancellation becomes -0 only for downward guest rounding;
  // equal-sign zero summands retain that sign for every rounding mode.
  const auto product_bits = _mm256_castpd_si256(product);
  const auto addend_bits = _mm256_castpd_si256(addend);
  const auto matching = _mm256_and_si256(
      _mm256_cmpeq_epi64(_mm256_and_si256(addend_bits, magnitude_mask), zero),
      _mm256_cmpeq_epi64(_mm256_srli_epi64(_mm256_xor_si256(product_bits, addend_bits), 63), zero));
  const auto sign_mask = _mm256_set1_epi64x(INT64_MIN);
  const auto zero_sign = _mm256_blendv_epi8(rounding == 2 ? sign_mask : zero,
                                            _mm256_and_si256(product_bits, sign_mask), matching);
  bits = _mm256_blendv_epi8(bits, zero_sign,
                            _mm256_cmpeq_epi64(_mm256_and_si256(bits, magnitude_mask), zero));
  if (clamp) {
    bits = _mm256_castpd_si256(_mm256_min_pd(
        _mm256_max_pd(_mm256_castsi256_pd(bits), _mm256_setzero_pd()), _mm256_set1_pd(1)));
  }
  const auto sign = _mm256_and_si256(_mm256_srli_epi64(bits, 48), _mm256_set1_epi64x(0x8000));
  const auto negative = _mm256_cmpgt_epi64(sign, zero);
  const auto away = rounding == 1   ? _mm256_xor_si256(negative, _mm256_set1_epi64x(-1))
                    : rounding == 2 ? negative
                                    : zero;
  const auto exponent = _mm256_and_si256(_mm256_srli_epi64(bits, 52), _mm256_set1_epi64x(2047));
  const auto normal = _mm256_cmpgt_epi64(exponent, _mm256_set1_epi64x(896));
  const auto not_tiny = _mm256_cmpgt_epi64(exponent, _mm256_set1_epi64x(888));
  auto shift = _mm256_blendv_epi8(_mm256_sub_epi64(_mm256_set1_epi64x(942), exponent),
                                  _mm256_set1_epi64x(45), normal);
  shift = _mm256_blendv_epi8(_mm256_set1_epi64x(53), shift, not_tiny);
  const auto significand =
      _mm256_or_si256(_mm256_and_si256(bits, _mm256_set1_epi64x(0xfffffffffffffULL)),
                      _mm256_set1_epi64x(1ULL << 52));
  auto rounded = _mm256_srlv_epi64(significand, shift);
  rounded = _mm256_add_epi64(
      rounded,
      _mm256_and_si256(normal,
                       _mm256_slli_epi64(_mm256_sub_epi64(exponent, _mm256_set1_epi64x(897)), 7)));
  const auto unit = _mm256_sllv_epi64(one, shift);
  const auto remainder = _mm256_and_si256(significand, _mm256_sub_epi64(unit, one));
  __m256i increment;
  if (rounding) {
    increment = _mm256_and_si256(away, _mm256_cmpgt_epi64(remainder, zero));
  } else {
    const auto midpoint = _mm256_srli_epi64(unit, 1);
    increment =
        _mm256_or_si256(_mm256_cmpgt_epi64(remainder, midpoint),
                        _mm256_and_si256(_mm256_cmpeq_epi64(remainder, midpoint),
                                         _mm256_sub_epi64(zero, _mm256_and_si256(rounded, one))));
  }
  rounded = _mm256_sub_epi64(rounded, increment);
  const auto tiny_result = _mm256_and_si256(
      away,
      _mm256_andnot_si256(_mm256_cmpeq_epi64(_mm256_and_si256(bits, magnitude_mask), zero), one));
  rounded = _mm256_blendv_epi8(tiny_result, rounded, not_tiny);
  const auto overflow = _mm256_or_si256(_mm256_cmpgt_epi64(exponent, _mm256_set1_epi64x(1150)),
                                        _mm256_cmpgt_epi64(rounded, _mm256_set1_epi64x(0x7f7f)));
  const auto overflow_value =
      rounding ? _mm256_blendv_epi8(_mm256_set1_epi64x(0x7f7f), _mm256_set1_epi64x(0x7f80), away)
               : _mm256_set1_epi64x(0x7f80);
  rounded = _mm256_or_si256(sign, _mm256_blendv_epi8(rounded, overflow_value, overflow));
  const auto packed = _mm256_shuffle_epi32(rounded, _MM_SHUFFLE(2, 0, 2, 0));
  return _mm_unpacklo_epi64(_mm256_castsi256_si128(packed), _mm256_extracti128_si256(packed, 1));
}

// Clamp ordinary FP32 output, then repair only nonfinite-input lanes through
// the baseline model to retain its host-FMA NaN policy across SIMD variants.
inline __m256i mixed_bf16_finish8(MixedFma destination, uint64_t flags, bool clamp, __m256 a,
                                  __m256 b, __m256 c, __m256i result) {
  if (destination == MixedFma::Float && clamp) {
    auto value = _mm256_castsi256_ps(result);
    value = _mm256_min_ps(_mm256_max_ps(value, _mm256_setzero_ps()), _mm256_set1_ps(1));
    result = _mm256_castps_si256(value);
  }
  const auto abs_mask = _mm256_set1_epi32(0x7fffffff);
  auto largest = _mm256_max_epi32(_mm256_and_si256(_mm256_castps_si256(a), abs_mask),
                                  _mm256_and_si256(_mm256_castps_si256(b), abs_mask));
  largest = _mm256_max_epi32(largest, _mm256_and_si256(_mm256_castps_si256(c), abs_mask));
  const unsigned special = unsigned(_mm256_movemask_ps(
      _mm256_castsi256_ps(_mm256_cmpgt_epi32(largest, _mm256_set1_epi32(0x7f7fffff)))));
  if (special) {
    uint32_t av[8], bv[8], cv[8], out[8];
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(av), _mm256_castps_si256(a));
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(bv), _mm256_castps_si256(b));
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(cv), _mm256_castps_si256(c));
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(out), result);
    for (int lane = 0; lane < 8; ++lane) {
      if ((special >> lane) & 1) {
        out[lane] = rocjitsu::arith::mixed_bf16_value(destination, flags, clamp, av[lane], bv[lane],
                                                      cv[lane]);
      }
    }
    result = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(out));
  }
  return result;
}

// Merge BF16 results with the unselected destination half; FP32 replaces the word.
inline __m256i mixed_bf16_merge8(MixedFma destination, __m256i result, const uint32_t *d) {
  if (destination == MixedFma::Float) {
    return result;
  }
  const auto original = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(d));
  if (destination == MixedFma::High) {
    return _mm256_or_si256(_mm256_and_si256(original, _mm256_set1_epi32(0xffff)),
                           _mm256_slli_epi32(result, 16));
  }
  return _mm256_or_si256(_mm256_and_si256(original, _mm256_set1_epi32(-65536)), result);
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
