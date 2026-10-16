// SPDX-License-Identifier: MIT

// AVX-512 32-bit add/subtract and three-input addition, with optional saturation.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "common/x86_64/alu_x86_64_v4.h"
#include "integer/integer_add.h"
#include "rj_arith/rj_arith.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {

// Add or subtract integers. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <IntegerAdd Op, bool Signed>
void integer_add_x86_64_v4(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                           const uint32_t *b, const uint32_t *c) {
  if constexpr (Op == IntegerAdd::Subrev) {
    const auto *temporary = a;
    a = b;
    b = temporary;
  }
  for (int lane = 0; lane < 32; lane += 16) {
    auto x = _mm512_loadu_si512(a + lane), y = _mm512_loadu_si512(b + lane);
    __m512i result;
    if constexpr (Op == IntegerAdd::Add || Op == IntegerAdd::Add3) {
      result = _mm512_add_epi32(x, y);
    } else {
      result = _mm512_sub_epi32(x, y);
    }
    if constexpr (Op == IntegerAdd::Add3) {
      result = _mm512_add_epi32(result, _mm512_loadu_si512(c + lane));
    } else if (mode & RJ_ARITH_ALU_CLAMP) {
      if constexpr (Signed && Op == IntegerAdd::Add) {
        result = saturate_signed_sum(x, y, result);
      } else if constexpr (Signed) {
        auto different_inputs = _mm512_xor_si512(x, y);
        auto different_result = _mm512_xor_si512(x, result);
        __m512i overflow;

        // Subtraction overflows when opposite-sign inputs flip A's sign.
        overflow = _mm512_and_si512(different_inputs, different_result);
        overflow = _mm512_srai_epi32(overflow, 31);
        auto saturated = _mm512_xor_si512(_mm512_set1_epi32(INT32_MAX), _mm512_srai_epi32(x, 31));
        result = _mm512_or_si512(_mm512_and_si512(overflow, saturated),
                                 _mm512_andnot_si512(overflow, result));
      } else {
        auto bias = _mm512_set1_epi32(INT32_MIN);
        auto ordered_x = _mm512_xor_si512(x, bias);
        if constexpr (Op == IntegerAdd::Add) {
          auto overflow = _mm512_maskz_set1_epi32(
              _mm512_cmpgt_epi32_mask(ordered_x, _mm512_xor_si512(result, bias)), -1);
          result = _mm512_or_si512(result, overflow);
        } else {
          auto borrow = _mm512_maskz_set1_epi32(
              _mm512_cmpgt_epi32_mask(_mm512_xor_si512(y, bias), ordered_x), -1);
          result = _mm512_andnot_si512(borrow, result);
        }
      }
    }
    _mm512_mask_storeu_epi32(d + lane, __mmask16(exec_mask >> lane), result);
  }
}

template void integer_add_x86_64_v4<IntegerAdd::Add, false>(uint32_t, uint32_t, uint32_t *,
                                                            const uint32_t *, const uint32_t *,
                                                            const uint32_t *);
template void integer_add_x86_64_v4<IntegerAdd::Sub, false>(uint32_t, uint32_t, uint32_t *,
                                                            const uint32_t *, const uint32_t *,
                                                            const uint32_t *);
template void integer_add_x86_64_v4<IntegerAdd::Subrev, false>(uint32_t, uint32_t, uint32_t *,
                                                               const uint32_t *, const uint32_t *,
                                                               const uint32_t *);
template void integer_add_x86_64_v4<IntegerAdd::Add, true>(uint32_t, uint32_t, uint32_t *,
                                                           const uint32_t *, const uint32_t *,
                                                           const uint32_t *);
template void integer_add_x86_64_v4<IntegerAdd::Sub, true>(uint32_t, uint32_t, uint32_t *,
                                                           const uint32_t *, const uint32_t *,
                                                           const uint32_t *);
template void integer_add_x86_64_v4<IntegerAdd::Add3, false>(uint32_t, uint32_t, uint32_t *,
                                                             const uint32_t *, const uint32_t *,
                                                             const uint32_t *);

} // namespace rocjitsu::arith
