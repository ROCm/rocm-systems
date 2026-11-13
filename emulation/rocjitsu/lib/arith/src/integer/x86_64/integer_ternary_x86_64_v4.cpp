// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 three-input integer arithmetic, selection, and shift/boolean combinations.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "integer/integer_ternary.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {

// Evaluate three-input integer operations. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <IntegerTernary Op>
void integer_ternary_x86_64_v4(uint32_t exec_mask, uint32_t *d, const uint32_t *a,
                               const uint32_t *b, const uint32_t *c) {
  for (int lane = 0; lane < 32; lane += 16) {
    auto x = _mm512_loadu_si512(a + lane), y = _mm512_loadu_si512(b + lane),
         z = _mm512_loadu_si512(c + lane);
    __m512i result;
    if constexpr (Op == IntegerTernary::ShiftAdd) {
      result =
          _mm512_add_epi32(_mm512_sllv_epi32(x, _mm512_and_si512(y, _mm512_set1_epi32(31))), z);
    }
    if constexpr (Op == IntegerTernary::AddShift) {
      result =
          _mm512_sllv_epi32(_mm512_add_epi32(x, y), _mm512_and_si512(z, _mm512_set1_epi32(31)));
    }
    if constexpr (Op == IntegerTernary::ShiftOr) {
      result = _mm512_or_si512(_mm512_sllv_epi32(x, _mm512_and_si512(y, _mm512_set1_epi32(31))), z);
    }
    if constexpr (Op == IntegerTernary::AndOr) {
      result = _mm512_ternarylogic_epi32(x, y, z, 0xea);
    }
    if constexpr (Op == IntegerTernary::Or3) {
      result = _mm512_ternarylogic_epi32(x, y, z, 0xfe);
    }
    if constexpr (Op == IntegerTernary::Xor3) {
      result = _mm512_ternarylogic_epi32(x, y, z, 0x96);
    }
    if constexpr (Op == IntegerTernary::XorAdd) {
      result = _mm512_add_epi32(_mm512_xor_si512(x, y), z);
    }
    if constexpr (Op == IntegerTernary::Lerp) {
      // Native byte average rounds up. Subtract one only for an odd sum
      // whose C byte requests truncation; higher C bits are ignored.
      auto correction =
          _mm512_and_si512(_mm512_ternarylogic_epi32(x, y, z, 0x14), _mm512_set1_epi8(1));
      result = _mm512_sub_epi8(_mm512_avg_epu8(x, y), correction);
    }
    _mm512_mask_storeu_epi32(d + lane, __mmask16(exec_mask >> lane), result);
  }
}

template void integer_ternary_x86_64_v4<IntegerTernary::ShiftAdd>(uint32_t, uint32_t *,
                                                                  const uint32_t *,
                                                                  const uint32_t *,
                                                                  const uint32_t *);
template void integer_ternary_x86_64_v4<IntegerTernary::AddShift>(uint32_t, uint32_t *,
                                                                  const uint32_t *,
                                                                  const uint32_t *,
                                                                  const uint32_t *);
template void integer_ternary_x86_64_v4<IntegerTernary::ShiftOr>(uint32_t, uint32_t *,
                                                                 const uint32_t *, const uint32_t *,
                                                                 const uint32_t *);
template void integer_ternary_x86_64_v4<IntegerTernary::AndOr>(uint32_t, uint32_t *,
                                                               const uint32_t *, const uint32_t *,
                                                               const uint32_t *);
template void integer_ternary_x86_64_v4<IntegerTernary::Or3>(uint32_t, uint32_t *, const uint32_t *,
                                                             const uint32_t *, const uint32_t *);
template void integer_ternary_x86_64_v4<IntegerTernary::Xor3>(uint32_t, uint32_t *,
                                                              const uint32_t *, const uint32_t *,
                                                              const uint32_t *);
template void integer_ternary_x86_64_v4<IntegerTernary::XorAdd>(uint32_t, uint32_t *,
                                                                const uint32_t *, const uint32_t *,
                                                                const uint32_t *);
template void integer_ternary_x86_64_v4<IntegerTernary::Lerp>(uint32_t, uint32_t *,
                                                              const uint32_t *, const uint32_t *,
                                                              const uint32_t *);

} // namespace rocjitsu::arith
