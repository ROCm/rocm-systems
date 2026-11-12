// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 three-input integer arithmetic, selection, and shift/boolean combinations.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "integer/integer_ternary.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {

// Evaluate three-input integer operations. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <IntegerTernary Op>
void integer_ternary_x86_64_v3(uint32_t exec_mask, uint32_t *d, const uint32_t *a,
                               const uint32_t *b, const uint32_t *c) {
  for (int lane = 0; lane < 32; lane += 8) {
    auto x = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane)),
         y = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + lane)),
         z = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(c + lane));
    __m256i result;
    if constexpr (Op == IntegerTernary::ShiftAdd) {
      result =
          _mm256_add_epi32(_mm256_sllv_epi32(x, _mm256_and_si256(y, _mm256_set1_epi32(31))), z);
    }
    if constexpr (Op == IntegerTernary::AddShift) {
      result =
          _mm256_sllv_epi32(_mm256_add_epi32(x, y), _mm256_and_si256(z, _mm256_set1_epi32(31)));
    }
    if constexpr (Op == IntegerTernary::ShiftOr) {
      result = _mm256_or_si256(_mm256_sllv_epi32(x, _mm256_and_si256(y, _mm256_set1_epi32(31))), z);
    }
    if constexpr (Op == IntegerTernary::AndOr) {
      result = _mm256_or_si256(_mm256_and_si256(x, y), z);
    }
    if constexpr (Op == IntegerTernary::Or3) {
      result = _mm256_or_si256(_mm256_or_si256(x, y), z);
    }
    if constexpr (Op == IntegerTernary::Xor3) {
      result = _mm256_xor_si256(_mm256_xor_si256(x, y), z);
    }
    if constexpr (Op == IntegerTernary::XorAdd) {
      result = _mm256_add_epi32(_mm256_xor_si256(x, y), z);
    }
    if constexpr (Op == IntegerTernary::Lerp) {
      // Native byte average rounds up. Subtract one only for an odd sum
      // whose C byte requests truncation; higher C bits are ignored.
      auto correction =
          _mm256_and_si256(_mm256_andnot_si256(z, _mm256_xor_si256(x, y)), _mm256_set1_epi8(1));
      result = _mm256_sub_epi8(_mm256_avg_epu8(x, y), correction);
    }
    auto lane_exec_mask = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                                            _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), lane_exec_mask, result);
  }
}

template void integer_ternary_x86_64_v3<IntegerTernary::ShiftAdd>(uint32_t, uint32_t *,
                                                                  const uint32_t *,
                                                                  const uint32_t *,
                                                                  const uint32_t *);
template void integer_ternary_x86_64_v3<IntegerTernary::AddShift>(uint32_t, uint32_t *,
                                                                  const uint32_t *,
                                                                  const uint32_t *,
                                                                  const uint32_t *);
template void integer_ternary_x86_64_v3<IntegerTernary::ShiftOr>(uint32_t, uint32_t *,
                                                                 const uint32_t *, const uint32_t *,
                                                                 const uint32_t *);
template void integer_ternary_x86_64_v3<IntegerTernary::Lerp>(uint32_t, uint32_t *,
                                                              const uint32_t *, const uint32_t *,
                                                              const uint32_t *);

template void integer_ternary_x86_64_v3<IntegerTernary::AndOr>(uint32_t, uint32_t *,
                                                               const uint32_t *, const uint32_t *,
                                                               const uint32_t *);
template void integer_ternary_x86_64_v3<IntegerTernary::Or3>(uint32_t, uint32_t *, const uint32_t *,
                                                             const uint32_t *, const uint32_t *);
template void integer_ternary_x86_64_v3<IntegerTernary::Xor3>(uint32_t, uint32_t *,
                                                              const uint32_t *, const uint32_t *,
                                                              const uint32_t *);
template void integer_ternary_x86_64_v3<IntegerTernary::XorAdd>(uint32_t, uint32_t *,
                                                                const uint32_t *, const uint32_t *,
                                                                const uint32_t *);

} // namespace rocjitsu::arith
