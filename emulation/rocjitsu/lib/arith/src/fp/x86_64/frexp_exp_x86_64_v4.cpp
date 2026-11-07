// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 exponent extraction using integer leading-zero counts for subnormals.

#include "fp/frexp_exp.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {
namespace {

// Stage integer exponents before writes, preserving aliases with either input word.
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
      auto field = _mm512_srli_epi64(magnitude, 52);
      auto exponent = _mm512_sub_epi64(field, _mm512_set1_epi64(1022));
      auto subnormal = _mm512_cmpeq_epi64_mask(field, _mm512_setzero_si512());
      exponent = _mm512_mask_mov_epi64(
          exponent, subnormal,
          _mm512_sub_epi64(_mm512_set1_epi64(-1010), _mm512_lzcnt_epi64(magnitude)));
      uint64_t minimum = flush_input ? 0x0010000000000000ULL : 1;
      auto valid = _mm512_cmple_epu64_mask(_mm512_sub_epi64(magnitude, _mm512_set1_epi64(minimum)),
                                           _mm512_set1_epi64(0x7fefffffffffffffULL - minimum));
      exponent = _mm512_maskz_mov_epi64(valid, exponent);
      _mm256_storeu_si256(reinterpret_cast<__m256i *>(result + lane),
                          _mm512_cvtepi64_epi32(exponent));
    } else {
      auto magnitude =
          _mm512_and_si512(_mm512_loadu_si512(a[0] + lane), _mm512_set1_epi32(INT32_MAX));
      auto field = _mm512_srli_epi32(magnitude, 23);
      auto exponent = _mm512_sub_epi32(field, _mm512_set1_epi32(126));
      auto subnormal = _mm512_cmpeq_epi32_mask(field, _mm512_setzero_si512());
      exponent = _mm512_mask_mov_epi32(
          exponent, subnormal,
          _mm512_sub_epi32(_mm512_set1_epi32(-117), _mm512_lzcnt_epi32(magnitude)));
      uint32_t minimum = flush_input ? 0x00800000U : 1;
      auto valid = _mm512_cmple_epu32_mask(_mm512_sub_epi32(magnitude, _mm512_set1_epi32(minimum)),
                                           _mm512_set1_epi32(0x7f7fffffU - minimum));
      _mm512_storeu_si512(result + lane, _mm512_maskz_mov_epi32(valid, exponent));
    }
  }
  for (int lane = 0; lane < 32; lane += 16) {
    _mm512_mask_storeu_epi32(d + lane, __mmask16(exec_mask >> lane),
                             _mm512_loadu_si512(result + lane));
  }
}
} // namespace

// Extract finite exponents; zeros, NaNs, infinities and guest-flushed inputs yield zero.
void frexp_exp_x86_64_v4(bool fp64, uint32_t exec_mask, uint32_t *d, const uint32_t *const *a,
                         bool flush_input) {
  if (fp64) {
    run<true>(exec_mask, d, a, flush_input);
  } else {
    run<false>(exec_mask, d, a, flush_input);
  }
}
} // namespace rocjitsu::arith
