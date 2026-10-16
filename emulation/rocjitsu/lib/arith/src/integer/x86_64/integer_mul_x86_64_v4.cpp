// SPDX-License-Identifier: MIT

// AVX-512 low/high-half integer multiplication with signed or unsigned operands.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "integer/integer_mul.h"
#include "rj_arith/rj_arith.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {
namespace {

// Return the high 32 bits of each signed/unsigned 32-bit product.
template <bool Signed> __m512i high_product(__m512i a, __m512i b) {
  // The widening multiply covers even words; shift odd words into place and interleave their
  // high halves.
  __m512i even, odd;
  if constexpr (Signed) {
    even = _mm512_mul_epi32(a, b);
    odd = _mm512_mul_epi32(_mm512_srli_epi64(a, 32), _mm512_srli_epi64(b, 32));
  } else {
    even = _mm512_mul_epu32(a, b);
    odd = _mm512_mul_epu32(_mm512_srli_epi64(a, 32), _mm512_srli_epi64(b, 32));
  }
  return _mm512_or_si512(_mm512_srli_epi64(even, 32),
                         _mm512_and_si512(odd, _mm512_set1_epi64(-INT64_C(4294967296))));
}

} // namespace

// Multiply integers and select result halves. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <int Bits, bool Signed, bool High>
void integer_mul_x86_64_v4(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                           const uint32_t *b) {
  for (int lane = 0; lane < 32; lane += 16) {
    auto x = _mm512_loadu_si512(a + lane);
    auto y = _mm512_loadu_si512(b + lane);
    if constexpr (Bits == 24) {
      if constexpr (Signed) {
        x = _mm512_srai_epi32(_mm512_slli_epi32(x, 8), 8);
        y = _mm512_srai_epi32(_mm512_slli_epi32(y, 8), 8);
      } else {
        x = _mm512_and_si512(x, _mm512_set1_epi32(0x00ffffff));
        y = _mm512_and_si512(y, _mm512_set1_epi32(0x00ffffff));
      }
    }
    __m512i value;
    if constexpr (High) {
      value = high_product<Signed>(x, y);
    } else {
      value = _mm512_mullo_epi32(x, y);
      if constexpr (Bits == 24) {
        if (mode & RJ_ARITH_ALU_CLAMP) {
          auto high = high_product<Signed>(x, y);
          if constexpr (Signed) {
            // A signed product fits iff its high word equals the sign extension
            // of its low word. The high word also determines saturation's sign.
            auto fits = _mm512_maskz_set1_epi32(
                _mm512_cmpeq_epi32_mask(high, _mm512_srai_epi32(value, 31)), -1);
            auto saturated =
                _mm512_xor_si512(_mm512_set1_epi32(INT32_MAX), _mm512_srai_epi32(high, 31));
            value = _mm512_or_si512(_mm512_and_si512(fits, value),
                                    _mm512_andnot_si512(fits, saturated));
          } else {
            auto overflow =
                _mm512_maskz_set1_epi32(_mm512_cmpgt_epi32_mask(high, _mm512_setzero_si512()), -1);
            value = _mm512_or_si512(value, overflow);
          }
        }
      }
    }
    _mm512_mask_storeu_epi32(d + lane, __mmask16(exec_mask >> lane), value);
  }
}

template void integer_mul_x86_64_v4<32, false, false>(uint32_t, uint32_t, uint32_t *,
                                                      const uint32_t *, const uint32_t *);
template void integer_mul_x86_64_v4<32, false, true>(uint32_t, uint32_t, uint32_t *,
                                                     const uint32_t *, const uint32_t *);
template void integer_mul_x86_64_v4<32, true, true>(uint32_t, uint32_t, uint32_t *,
                                                    const uint32_t *, const uint32_t *);
template void integer_mul_x86_64_v4<24, true, false>(uint32_t, uint32_t, uint32_t *,
                                                     const uint32_t *, const uint32_t *);
template void integer_mul_x86_64_v4<24, true, true>(uint32_t, uint32_t, uint32_t *,
                                                    const uint32_t *, const uint32_t *);
template void integer_mul_x86_64_v4<24, false, false>(uint32_t, uint32_t, uint32_t *,
                                                      const uint32_t *, const uint32_t *);
template void integer_mul_x86_64_v4<24, false, true>(uint32_t, uint32_t, uint32_t *,
                                                     const uint32_t *, const uint32_t *);

} // namespace rocjitsu::arith
