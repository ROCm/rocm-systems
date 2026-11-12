// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 32-bit products added to split-word 64-bit accumulators, with carry output.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "integer/mad64.h"
#include "rj_arith/rj_arith.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {

// Add 32-bit products to 64-bit accumulators. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <bool Signed>
uint32_t mad64_x86_64_v3(uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
                         const uint32_t *const *a, const uint32_t *const *b,
                         const uint32_t *const *c) {
  uint32_t staged[2][32], output_carry = 0;
  for (int lane = 0; lane < 32; lane += 4) {
    auto av =
        _mm256_cvtepu32_epi64(_mm_loadu_si128(reinterpret_cast<const __m128i *>(a[0] + lane)));
    auto bv =
        _mm256_cvtepu32_epi64(_mm_loadu_si128(reinterpret_cast<const __m128i *>(b[0] + lane)));
    auto low =
        _mm256_cvtepu32_epi64(_mm_loadu_si128(reinterpret_cast<const __m128i *>(c[0] + lane)));
    auto high =
        _mm256_cvtepu32_epi64(_mm_loadu_si128(reinterpret_cast<const __m128i *>(c[1] + lane)));
    auto cv = _mm256_or_si256(low, _mm256_slli_epi64(high, 32));
    __m256i product;
    if constexpr (Signed) {
      product = _mm256_mul_epi32(av, bv);
    } else {
      product = _mm256_mul_epu32(av, bv);
    }
    auto result = _mm256_add_epi64(product, cv);
    if constexpr (Signed) {
      auto overflow =
          _mm256_and_si256(_mm256_xor_si256(product, result), _mm256_xor_si256(cv, result));

      // Bit 64 is the sign of the exact sum, including signed-overflow cases.
      auto sign = _mm256_xor_si256(result, overflow);
      output_carry |= uint32_t(_mm256_movemask_pd(_mm256_castsi256_pd(sign))) << lane;
      if (mode & RJ_ARITH_ALU_CLAMP) {
        auto zero = _mm256_setzero_si256();
        auto limit =
            _mm256_xor_si256(_mm256_cmpgt_epi64(zero, product), _mm256_set1_epi64x(INT64_MAX));
        result = _mm256_blendv_epi8(result, limit, _mm256_cmpgt_epi64(zero, overflow));
      }
    } else {
      auto sign = _mm256_set1_epi64x(INT64_MIN);
      auto carry =
          _mm256_cmpgt_epi64(_mm256_xor_si256(product, sign), _mm256_xor_si256(result, sign));
      output_carry |= uint32_t(_mm256_movemask_pd(_mm256_castsi256_pd(carry))) << lane;
      if (mode & RJ_ARITH_ALU_CLAMP) {
        result = _mm256_or_si256(result, carry);
      }
    }
    auto words = _mm256_permutevar8x32_epi32(result, _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7));
    _mm_storeu_si128(reinterpret_cast<__m128i *>(staged[0] + lane), _mm256_castsi256_si128(words));
    _mm_storeu_si128(reinterpret_cast<__m128i *>(staged[1] + lane),
                     _mm256_extracti128_si256(words, 1));
  }
  // Commit both halves only after all source reads, with D1 winning overlaps.
  for (int reg = 0; reg < 2; ++reg) {
    for (int lane = 0; lane < 32; lane += 8) {
      auto result = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(staged[reg] + lane));
      auto lane_exec_mask = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                                              _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
      _mm256_maskstore_epi32(reinterpret_cast<int *>(d[reg] + lane), lane_exec_mask, result);
    }
  }
  return output_carry & exec_mask;
}

template uint32_t mad64_x86_64_v3<false>(uint32_t, uint32_t, uint32_t *const *,
                                         const uint32_t *const *, const uint32_t *const *,
                                         const uint32_t *const *);
template uint32_t mad64_x86_64_v3<true>(uint32_t, uint32_t, uint32_t *const *,
                                        const uint32_t *const *, const uint32_t *const *,
                                        const uint32_t *const *);

} // namespace rocjitsu::arith
