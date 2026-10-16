// SPDX-License-Identifier: MIT

// AVX-512 32-bit products added to split-word 64-bit accumulators, with carry output.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "integer/mad64.h"
#include "rj_arith/rj_arith.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {

// Add 32-bit products to 64-bit accumulators. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <bool Signed>
uint32_t mad64_x86_64_v4(uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
                         const uint32_t *const *a, const uint32_t *const *b,
                         const uint32_t *const *c) {
  uint32_t staged[2][32], output_carry = 0;
  for (int lane = 0; lane < 32; lane += 8) {
    auto av =
        _mm512_cvtepu32_epi64(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(a[0] + lane)));
    auto bv =
        _mm512_cvtepu32_epi64(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(b[0] + lane)));
    auto low =
        _mm512_cvtepu32_epi64(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(c[0] + lane)));
    auto high =
        _mm512_cvtepu32_epi64(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(c[1] + lane)));
    auto cv = _mm512_or_si512(low, _mm512_slli_epi64(high, 32));
    __m512i product;
    if constexpr (Signed) {
      product = _mm512_mul_epi32(av, bv);
    } else {
      product = _mm512_mul_epu32(av, bv);
    }
    auto result = _mm512_add_epi64(product, cv);
    if constexpr (Signed) {
      auto overflow =
          _mm512_and_si512(_mm512_xor_si512(product, result), _mm512_xor_si512(cv, result));

      // Bit 64 is the sign of the exact sum, including signed-overflow cases.
      auto sign = _mm512_xor_si512(result, overflow);
      output_carry |= uint32_t(_mm512_movepi64_mask(sign)) << lane;
      if (mode & RJ_ARITH_ALU_CLAMP) {
        auto limit = _mm512_mask_set1_epi64(_mm512_set1_epi64(INT64_MAX),
                                            _mm512_movepi64_mask(product), INT64_MIN);
        result = _mm512_mask_mov_epi64(result, _mm512_movepi64_mask(overflow), limit);
      }
    } else {
      auto carry = _mm512_cmplt_epu64_mask(result, product);
      output_carry |= uint32_t(carry) << lane;
      if (mode & RJ_ARITH_ALU_CLAMP) {
        result = _mm512_mask_set1_epi64(result, carry, -1);
      }
    }
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(staged[0] + lane),
                        _mm512_cvtepi64_epi32(result));
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(staged[1] + lane),
                        _mm512_cvtepi64_epi32(_mm512_srli_epi64(result, 32)));
  }
  // Commit both halves only after all source reads, with D1 winning overlaps.
  for (int reg = 0; reg < 2; ++reg) {
    for (int lane = 0; lane < 32; lane += 16) {
      auto result = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(staged[reg] + lane));
      _mm512_mask_storeu_epi32(d[reg] + lane, __mmask16(exec_mask >> lane), result);
    }
  }
  return output_carry & exec_mask;
}

template uint32_t mad64_x86_64_v4<false>(uint32_t, uint32_t, uint32_t *const *,
                                         const uint32_t *const *, const uint32_t *const *,
                                         const uint32_t *const *);
template uint32_t mad64_x86_64_v4<true>(uint32_t, uint32_t, uint32_t *const *,
                                        const uint32_t *const *, const uint32_t *const *,
                                        const uint32_t *const *);

} // namespace rocjitsu::arith
