// SPDX-License-Identifier: MIT

// AVX-512 saturating INT16-to-U8 packing and selection of two FP16 source halfwords.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "conversion/pack.h"
#include "rj_arith/rj_arith.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {

// Pack saturated bytes or selected FP16 halves. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <bool Saturate>
void pack_x86_64_v4(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                    const uint32_t *b) {
  for (int lane = 0; lane < 32; lane += 16) {
    auto x = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(a + lane));
    __m512i result;
    if constexpr (Saturate) {
      auto lo = _mm512_srai_epi32(_mm512_slli_epi32(x, 16), 16);
      auto hi = _mm512_srai_epi32(x, 16);
      lo = _mm512_min_epi32(_mm512_max_epi32(lo, _mm512_setzero_si512()), _mm512_set1_epi32(255));
      hi = _mm512_min_epi32(_mm512_max_epi32(hi, _mm512_setzero_si512()), _mm512_set1_epi32(255));
      auto packed = _mm512_or_si512(lo, _mm512_slli_epi32(hi, 8));
      auto old = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(d + lane));
      unsigned shift = mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
      auto selected = _mm512_sll_epi32(packed, _mm_cvtsi32_si128(shift));
      result = _mm512_or_si512(_mm512_andnot_si512(_mm512_set1_epi32(int(65535u << shift)), old),
                               selected);
    } else {
      auto y = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(b + lane));
      x = _mm512_srl_epi32(x, _mm_cvtsi32_si128(mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0));
      y = _mm512_srl_epi32(y, _mm_cvtsi32_si128(mode & RJ_ARITH_ALU_HIGH_B ? 16 : 0));
      x = _mm512_and_si512(x, _mm512_set1_epi32(mode & RJ_ARITH_ALU_ABS_A ? 32767 : 65535));
      y = _mm512_and_si512(y, _mm512_set1_epi32(mode & RJ_ARITH_ALU_ABS_B ? 32767 : 65535));
      x = _mm512_mask_mov_epi32(x,
                                _mm512_cmp_epi32_mask(_mm512_and_si512(x, _mm512_set1_epi32(32767)),
                                                      _mm512_set1_epi32(0x7c00), _MM_CMPINT_GT),
                                _mm512_or_si512(x, _mm512_set1_epi32(0x200)));
      y = _mm512_mask_mov_epi32(y,
                                _mm512_cmp_epi32_mask(_mm512_and_si512(y, _mm512_set1_epi32(32767)),
                                                      _mm512_set1_epi32(0x7c00), _MM_CMPINT_GT),
                                _mm512_or_si512(y, _mm512_set1_epi32(0x200)));
      x = _mm512_xor_si512(x, _mm512_set1_epi32(mode & RJ_ARITH_ALU_NEG_A ? 32768 : 0));
      y = _mm512_xor_si512(y, _mm512_set1_epi32(mode & RJ_ARITH_ALU_NEG_B ? 32768 : 0));
      result = _mm512_or_si512(x, _mm512_slli_epi32(y, 16));
    }
    _mm512_mask_storeu_epi32(d + lane, __mmask16(exec_mask >> lane), result);
  }
}

template void pack_x86_64_v4<true>(uint32_t, uint32_t, uint32_t *, const uint32_t *,
                                   const uint32_t *);
template void pack_x86_64_v4<false>(uint32_t, uint32_t, uint32_t *, const uint32_t *,
                                    const uint32_t *);

} // namespace rocjitsu::arith
