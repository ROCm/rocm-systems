// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Shared AVX2-width source preparation and half output for CDNA mixed MAD.
#pragma once
#include "common/x86_64/alu_x86_64_v3.h"
#include "common/x86_64/half_x86_64_v3.h"
#include "fp/mixed_mad.h"
#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Decode selected source halves and ABS/NEG without denormal flushing.
inline __m256 mixed_mad_input(const uint32_t *p, uint32_t mode) {
  auto words = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(p));
  if (mode & RJ_ARITH_MIX_F16_A) {
    return half_input(words, mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0, mode);
  }
  return _mm256_castsi256_ps(_mm256_xor_si256(
      _mm256_and_si256(words, _mm256_set1_epi32(mode & RJ_ARITH_ALU_ABS_A ? INT32_MAX : -1)),
      _mm256_set1_epi32(mode & RJ_ARITH_ALU_NEG_A ? INT32_MIN : 0)));
}

// Apply CLAMP and optional directed FP16 narrowing to the FP32 intermediate.
// Nonfinite input lanes use baseline NaN handling, without a whole-vector fallback.
inline __m256i mixed_mad_finish(MixedFma destination, uint64_t flags, uint32_t mode, __m256 value,
                                __m256 x, __m256 y, __m256 z, const uint32_t *a, const uint32_t *b,
                                const uint32_t *c) {
  if (mode & RJ_ARITH_ALU_CLAMP) {
    value = clamp_f32(value, flags & RJ_ARITH_FP_DX10_CLAMP_OFF);
  }
  __m256i result = _mm256_castps_si256(value);
  if (destination != MixedFma::Float) {
    __m128i half;
    switch ((flags & RJ_ARITH_FP_ROUND_MASK) >> 24) {
    case 1:
      half = _mm256_cvtps_ph(value, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
      break;
    case 2:
      half = _mm256_cvtps_ph(value, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
      break;
    case 3:
      half = _mm256_cvtps_ph(value, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
      break;
    default:
      half = _mm256_cvtps_ph(value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
      break;
    }
    result = _mm256_cvtepu16_epi32(half);
    if (flags & RJ_ARITH_FP16_OVFL) {
      auto finite = _mm256_cmpgt_epi32(
          _mm256_set1_epi32(0x7f800000),
          _mm256_and_si256(_mm256_castps_si256(value), _mm256_set1_epi32(INT32_MAX)));
      auto infinity = _mm256_cmpeq_epi32(_mm256_and_si256(result, _mm256_set1_epi32(0x7fff)),
                                         _mm256_set1_epi32(0x7c00));
      result = _mm256_add_epi32(result, _mm256_and_si256(finite, infinity));
    }
  }
  const auto magnitude = _mm256_set1_epi32(INT32_MAX);
  auto largest =
      _mm256_max_epi32(_mm256_and_si256(_mm256_castps_si256(x), magnitude),
                       _mm256_max_epi32(_mm256_and_si256(_mm256_castps_si256(y), magnitude),
                                        _mm256_and_si256(_mm256_castps_si256(z), magnitude)));
  unsigned repair = unsigned(_mm256_movemask_ps(
      _mm256_castsi256_ps(_mm256_cmpgt_epi32(largest, _mm256_set1_epi32(0x7f7fffff)))));
  if (repair) {
    uint32_t lanes[8];
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(lanes), result);
    for (int lane = 0; lane < 8; ++lane) {
      if ((repair >> lane) & 1) {
        lanes[lane] = mixed_mad_value(destination, flags, mode, a[lane], b[lane], c[lane]);
      }
    }
    result = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(lanes));
  }
  return result;
}

// Merge only the selected half, preserving unrelated destination bits.
inline __m256i mixed_mad_merge(MixedFma destination, __m256i result, const uint32_t *d) {
  if (destination == MixedFma::Float) {
    return result;
  }
  const int shift = destination == MixedFma::High ? 16 : 0;
  auto old = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(d));
  auto keep = _mm256_andnot_si256(_mm256_set1_epi32(int(0xffffU << shift)), old);
  return _mm256_or_si256(keep, _mm256_sll_epi32(result, _mm_cvtsi32_si128(shift)));
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
