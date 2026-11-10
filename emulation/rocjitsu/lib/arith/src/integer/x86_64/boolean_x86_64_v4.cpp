// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 wordwise boolean operations, including three-input truth tables.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "integer/boolean.h"
#include "rj_arith/rj_arith.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {

// Evaluate wordwise boolean operations. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <Boolean Op, bool Half>
void boolean_x86_64_v4(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                       const uint32_t *b) {
  auto sa = _mm_cvtsi32_si128(mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0);
  auto sb = _mm_cvtsi32_si128(mode & RJ_ARITH_ALU_HIGH_B ? 16 : 0);
  auto sd = _mm_cvtsi32_si128(mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0);
  auto destination_mask =
      _mm512_set1_epi32(int(uint32_t(65535) << (mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0)));
  for (int lane = 0; lane < 32; lane += 16) {
    __m512i x;
    x = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(a + lane));
    if constexpr (Half) {
      x = _mm512_srl_epi32(x, sa);
    }
    __m512i y;
    if constexpr (Op != Boolean::Not) {
      y = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(b + lane));
      if constexpr (Half) {
        y = _mm512_srl_epi32(y, sb);
      }
    }
    __m512i result;
    if constexpr (Op == Boolean::And) {
      result = _mm512_and_si512(x, y);
    }
    if constexpr (Op == Boolean::Or) {
      result = _mm512_or_si512(x, y);
    }
    if constexpr (Op == Boolean::Xor) {
      result = _mm512_xor_si512(x, y);
    }
    if constexpr (Op == Boolean::Xnor) {
      result = _mm512_xor_si512(_mm512_xor_si512(x, y), _mm512_set1_epi32(-1));
    }
    if constexpr (Op == Boolean::Not) {
      result = _mm512_xor_si512(x, _mm512_set1_epi32(-1));
    }
    if constexpr (Half) {
      auto original = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(d + lane));
      auto selected = _mm512_sll_epi32(result, sd);
      result = _mm512_ternarylogic_epi32(destination_mask, selected, original, 0xca);
    }
    _mm512_mask_storeu_epi32(d + lane, __mmask16(exec_mask >> lane), result);
  }
}

template void boolean_x86_64_v4<Boolean::And, false>(uint32_t exec_mask, uint32_t mode, uint32_t *d,
                                                     const uint32_t *a, const uint32_t *b);
template void boolean_x86_64_v4<Boolean::Or, false>(uint32_t exec_mask, uint32_t mode, uint32_t *d,
                                                    const uint32_t *a, const uint32_t *b);
template void boolean_x86_64_v4<Boolean::Xor, false>(uint32_t exec_mask, uint32_t mode, uint32_t *d,
                                                     const uint32_t *a, const uint32_t *b);
template void boolean_x86_64_v4<Boolean::Not, false>(uint32_t exec_mask, uint32_t mode, uint32_t *d,
                                                     const uint32_t *a, const uint32_t *b);
template void boolean_x86_64_v4<Boolean::And, true>(uint32_t exec_mask, uint32_t mode, uint32_t *d,
                                                    const uint32_t *a, const uint32_t *b);
template void boolean_x86_64_v4<Boolean::Or, true>(uint32_t exec_mask, uint32_t mode, uint32_t *d,
                                                   const uint32_t *a, const uint32_t *b);
template void boolean_x86_64_v4<Boolean::Xor, true>(uint32_t exec_mask, uint32_t mode, uint32_t *d,
                                                    const uint32_t *a, const uint32_t *b);
template void boolean_x86_64_v4<Boolean::Not, true>(uint32_t exec_mask, uint32_t mode, uint32_t *d,
                                                    const uint32_t *a, const uint32_t *b);

template void boolean_x86_64_v4<Boolean::Xnor, false>(uint32_t, uint32_t, uint32_t *,
                                                      const uint32_t *, const uint32_t *);

} // namespace rocjitsu::arith
