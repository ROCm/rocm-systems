// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 saturating integer narrowing and signed/unsigned halfword extension.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "conversion/integer_conversion.h"
#include "rj_arith/rj_arith.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {
namespace {

// Saturate a signed/unsigned 32-bit integer to 16 bits; return its low-halfword encoding.
template <bool Unsigned> __m512i narrow(__m512i raw) {
  if constexpr (Unsigned) {
    return _mm512_min_epu32(raw, _mm512_set1_epi32(65535));
  } else {
    return _mm512_and_si512(_mm512_min_epi32(_mm512_max_epi32(raw, _mm512_set1_epi32(-32768)),
                                             _mm512_set1_epi32(32767)),
                            _mm512_set1_epi32(65535));
  }
}

} // namespace

// Narrow or extend integer halfwords. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <bool Unsigned, bool Packed>
void integer_conversion_x86_64_v4(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                                  const uint32_t *b) {
  auto shift = _mm_cvtsi32_si128(mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0);
  for (int lane = 0; lane < 32; lane += 16) {
    auto raw = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(a + lane));
    __m512i result;
    if constexpr (Packed) {
      auto vb = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(b + lane));
      result = _mm512_or_si512(narrow<Unsigned>(raw), _mm512_slli_epi32(narrow<Unsigned>(vb), 16));
    } else {
      result = _mm512_and_si512(_mm512_srl_epi32(raw, shift), _mm512_set1_epi32(65535));
      if constexpr (!Unsigned) {
        result = _mm512_srai_epi32(_mm512_slli_epi32(result, 16), 16);
      }
    }
    _mm512_mask_storeu_epi32(d + lane, __mmask16(exec_mask >> lane), result);
  }
}

template void integer_conversion_x86_64_v4<false, false>(uint32_t, uint32_t, uint32_t *,
                                                         const uint32_t *, const uint32_t *);
template void integer_conversion_x86_64_v4<true, false>(uint32_t, uint32_t, uint32_t *,
                                                        const uint32_t *, const uint32_t *);
template void integer_conversion_x86_64_v4<false, true>(uint32_t, uint32_t, uint32_t *,
                                                        const uint32_t *, const uint32_t *);
template void integer_conversion_x86_64_v4<true, true>(uint32_t, uint32_t, uint32_t *,
                                                       const uint32_t *, const uint32_t *);

} // namespace rocjitsu::arith
