// SPDX-License-Identifier: MIT

// AVX2 low/high-half integer multiplication with signed or unsigned operands.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "integer/integer_mul.h"
#include "rj_arith/rj_arith.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {
namespace {

// Return the high 32 bits of each signed/unsigned 32-bit product.
template <bool Signed> __m256i high_product(__m256i a, __m256i b) {
  // The widening multiply covers even words; shift odd words into place and interleave their
  // high halves.
  __m256i even, odd;
  if constexpr (Signed) {
    even = _mm256_mul_epi32(a, b);
    odd = _mm256_mul_epi32(_mm256_srli_epi64(a, 32), _mm256_srli_epi64(b, 32));
  } else {
    even = _mm256_mul_epu32(a, b);
    odd = _mm256_mul_epu32(_mm256_srli_epi64(a, 32), _mm256_srli_epi64(b, 32));
  }
  return _mm256_or_si256(_mm256_srli_epi64(even, 32),
                         _mm256_and_si256(odd, _mm256_set1_epi64x(-INT64_C(4294967296))));
}

} // namespace

// Multiply integers and select result halves. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <int Bits, bool Signed, bool High>
void integer_mul_x86_64_v3(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                           const uint32_t *b) {
  for (int lane = 0; lane < 32; lane += 8) {
    auto x = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane));
    auto y = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + lane));
    if constexpr (Bits == 24) {
      if constexpr (Signed) {
        x = _mm256_srai_epi32(_mm256_slli_epi32(x, 8), 8);
        y = _mm256_srai_epi32(_mm256_slli_epi32(y, 8), 8);
      } else {
        x = _mm256_and_si256(x, _mm256_set1_epi32(0x00ffffff));
        y = _mm256_and_si256(y, _mm256_set1_epi32(0x00ffffff));
      }
    }
    __m256i value;
    if constexpr (High) {
      value = high_product<Signed>(x, y);
    } else {
      value = _mm256_mullo_epi32(x, y);
      if constexpr (Bits == 24) {
        if (mode & RJ_ARITH_ALU_CLAMP) {
          auto high = high_product<Signed>(x, y);
          if constexpr (Signed) {
            // A signed product fits iff its high word equals the sign extension
            // of its low word. The high word also determines saturation's sign.
            auto fits = _mm256_cmpeq_epi32(high, _mm256_srai_epi32(value, 31));
            auto saturated =
                _mm256_xor_si256(_mm256_set1_epi32(INT32_MAX), _mm256_srai_epi32(high, 31));
            value = _mm256_or_si256(_mm256_and_si256(fits, value),
                                    _mm256_andnot_si256(fits, saturated));
          } else {
            auto overflow = _mm256_cmpgt_epi32(high, _mm256_setzero_si256());
            value = _mm256_or_si256(value, overflow);
          }
        }
      }
    }
    auto lane_exec_mask = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                                            _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), lane_exec_mask, value);
  }
}

template void integer_mul_x86_64_v3<32, false, false>(uint32_t, uint32_t, uint32_t *,
                                                      const uint32_t *, const uint32_t *);
template void integer_mul_x86_64_v3<32, false, true>(uint32_t, uint32_t, uint32_t *,
                                                     const uint32_t *, const uint32_t *);
template void integer_mul_x86_64_v3<32, true, true>(uint32_t, uint32_t, uint32_t *,
                                                    const uint32_t *, const uint32_t *);
template void integer_mul_x86_64_v3<24, true, false>(uint32_t, uint32_t, uint32_t *,
                                                     const uint32_t *, const uint32_t *);
template void integer_mul_x86_64_v3<24, true, true>(uint32_t, uint32_t, uint32_t *,
                                                    const uint32_t *, const uint32_t *);
template void integer_mul_x86_64_v3<24, false, false>(uint32_t, uint32_t, uint32_t *,
                                                      const uint32_t *, const uint32_t *);
template void integer_mul_x86_64_v3<24, false, true>(uint32_t, uint32_t, uint32_t *,
                                                     const uint32_t *, const uint32_t *);

} // namespace rocjitsu::arith
