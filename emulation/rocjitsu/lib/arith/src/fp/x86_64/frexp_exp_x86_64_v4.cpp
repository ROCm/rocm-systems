// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 F32/F64 exponent extraction, including preserved subnormals under
// the default host FP environment. Invalid/special inputs produce integer zero.

#include "fp/frexp_exp.h"
#include "rj_arith/detail/rj_arith_common.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {
namespace {

// Extract unbiased frexp exponents, staging inputs before masked stores for aliases.
template <bool Fp64>
void run(uint32_t exec_mask, uint32_t *d, const uint32_t *const *a, bool flush_input) {
  uint32_t result[32];
  for (int lane = 0; lane < 32; lane += Fp64 ? 8 : 16) {
    if constexpr (Fp64) {
      auto low =
          _mm512_cvtepu32_epi64(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(a[0] + lane)));
      auto high =
          _mm512_cvtepu32_epi64(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(a[1] + lane)));
      auto magnitude = _mm512_and_si512(_mm512_or_si512(low, _mm512_slli_epi64(high, 32)),
                                        _mm512_set1_epi64(INT64_MAX));
      auto valid = _mm512_cmpneq_epi64_mask(magnitude, _mm512_setzero_si512()) &
                   _mm512_cmplt_epi64_mask(magnitude, _mm512_set1_epi64(0x7ff0000000000000LL));
      if (flush_input) {
        valid &= _mm512_cmpge_epi64_mask(magnitude, _mm512_set1_epi64(0x0010000000000000LL));
      }
      // Mask specials before FP evaluation; VGETEXP normalizes preserved subnormals.
      const auto exponent = _mm512_maskz_getexp_pd(valid, _mm512_castsi512_pd(magnitude));
      auto value = _mm256_add_epi32(_mm512_cvttpd_epi32(exponent), _mm256_set1_epi32(1));
      _mm256_storeu_si256(reinterpret_cast<__m256i *>(result + lane),
                          _mm256_maskz_mov_epi32(valid, value));
    } else {
      auto magnitude =
          _mm512_and_si512(_mm512_loadu_si512(a[0] + lane), _mm512_set1_epi32(INT32_MAX));
      auto valid = _mm512_cmpneq_epi32_mask(magnitude, _mm512_setzero_si512()) &
                   _mm512_cmplt_epi32_mask(magnitude, _mm512_set1_epi32(0x7f800000));
      if (flush_input) {
        valid &= _mm512_cmpge_epi32_mask(magnitude, _mm512_set1_epi32(0x00800000));
      }
      const auto exponent = _mm512_maskz_getexp_ps(valid, _mm512_castsi512_ps(magnitude));
      const auto value =
          _mm512_maskz_add_epi32(valid, _mm512_cvttps_epi32(exponent), _mm512_set1_epi32(1));
      _mm512_storeu_si512(result + lane, value);
    }
  }
  for (int lane = 0; lane < 32; lane += 16) {
    _mm512_mask_storeu_epi32(d + lane, __mmask16(exec_mask >> lane),
                             _mm512_loadu_si512(result + lane));
  }
}

} // namespace

// Select the declared source width with validated controls and named CPU features.
void frexp_exp_x86_64_v4(bool fp64, uint32_t exec_mask, uint32_t *d, const uint32_t *const *a,
                         bool flush_input) {
  if (fp64) {
    run<true>(exec_mask, d, a, flush_input);
  } else {
    run<false>(exec_mask, d, a, flush_input);
  }
}

} // namespace rocjitsu::arith
