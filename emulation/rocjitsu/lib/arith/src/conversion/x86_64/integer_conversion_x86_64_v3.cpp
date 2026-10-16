// SPDX-License-Identifier: MIT

// AVX2 saturating integer narrowing and signed/unsigned halfword extension.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "conversion/integer_conversion.h"
#include "rj_arith/rj_arith.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {
namespace {

// Saturate a signed/unsigned 32-bit integer to 16 bits; return its low-halfword encoding.
template <bool Unsigned> __m256i narrow(__m256i raw) {
  if constexpr (Unsigned) {
    return _mm256_min_epu32(raw, _mm256_set1_epi32(65535));
  } else {
    return _mm256_and_si256(_mm256_min_epi32(_mm256_max_epi32(raw, _mm256_set1_epi32(-32768)),
                                             _mm256_set1_epi32(32767)),
                            _mm256_set1_epi32(65535));
  }
}

} // namespace

// Narrow or extend integer halfwords. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <bool Unsigned, bool Packed>
void integer_conversion_x86_64_v3(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                                  const uint32_t *b) {
  for (int lane = 0; lane < 32; lane += 8) {
    auto va = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane));
    __m256i result;
    if constexpr (Packed) {
      auto vb = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + lane));
      result = _mm256_or_si256(narrow<Unsigned>(va), _mm256_slli_epi32(narrow<Unsigned>(vb), 16));
    } else {
      va = _mm256_srl_epi32(va, _mm_cvtsi32_si128(mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0));
      if constexpr (Unsigned) {
        result = _mm256_and_si256(va, _mm256_set1_epi32(65535));
      } else {
        result = _mm256_srai_epi32(_mm256_slli_epi32(va, 16), 16);
      }
    }
    auto lane_exec_mask = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                                            _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), lane_exec_mask, result);
  }
}

template void integer_conversion_x86_64_v3<false, false>(uint32_t, uint32_t, uint32_t *,
                                                         const uint32_t *, const uint32_t *);
template void integer_conversion_x86_64_v3<false, true>(uint32_t, uint32_t, uint32_t *,
                                                        const uint32_t *, const uint32_t *);
template void integer_conversion_x86_64_v3<true, false>(uint32_t, uint32_t, uint32_t *,
                                                        const uint32_t *, const uint32_t *);
template void integer_conversion_x86_64_v3<true, true>(uint32_t, uint32_t, uint32_t *,
                                                       const uint32_t *, const uint32_t *);

} // namespace rocjitsu::arith
