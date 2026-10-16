// SPDX-License-Identifier: MIT

// AVX-512 integer add/subtract with per-lane carry/borrow input and output masks.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "integer/carry.h"
#include "rj_arith/rj_arith.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {

// Add/subtract with carry or borrow masks. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <CarryOp Op, bool WithCarry>
uint32_t carry_x86_64_v4(uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
                         const uint32_t *const *a, const uint32_t *const *b, uint32_t input_carry) {
  uint32_t output_carry = 0;
  for (int lane = 0; lane < 32; lane += 16) {
    auto av = _mm512_loadu_si512(
        reinterpret_cast<const __m512i *>((Op == CarryOp::Subrev ? b : a)[0] + lane));
    auto bv = _mm512_loadu_si512(
        reinterpret_cast<const __m512i *>((Op == CarryOp::Subrev ? a : b)[0] + lane));
    __m512i result;
    __mmask16 co;
    if constexpr (Op == CarryOp::Add) {
      result = _mm512_add_epi32(av, bv);
      co = _mm512_cmplt_epu32_mask(result, av);
    } else {
      result = _mm512_sub_epi32(av, bv);
      co = _mm512_cmplt_epu32_mask(av, bv);
    }
    if constexpr (WithCarry) {
      auto ci = _mm512_maskz_set1_epi32(__mmask16(input_carry >> lane), 1);
      if constexpr (Op == CarryOp::Add) {
        auto sum = _mm512_add_epi32(result, ci);
        co = co | _mm512_cmplt_epu32_mask(sum, result);
        result = sum;
      } else {
        co = co | _mm512_cmplt_epu32_mask(result, ci);
        result = _mm512_sub_epi32(result, ci);
      }
    }
    output_carry |= uint32_t(co) << lane;
    if (mode & RJ_ARITH_ALU_CLAMP) {
      result = _mm512_mask_mov_epi32(result, co, _mm512_set1_epi32(Op == CarryOp::Add ? -1 : 0));
    }
    _mm512_mask_storeu_epi32(d[0] + lane, __mmask16(exec_mask >> lane), result);
  }
  return output_carry & exec_mask;
}

template uint32_t carry_x86_64_v4<CarryOp::Add, false>(uint32_t, uint32_t, uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *, uint32_t);
template uint32_t carry_x86_64_v4<CarryOp::Add, true>(uint32_t, uint32_t, uint32_t *const *,
                                                      const uint32_t *const *,
                                                      const uint32_t *const *, uint32_t);
template uint32_t carry_x86_64_v4<CarryOp::Sub, false>(uint32_t, uint32_t, uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *, uint32_t);
template uint32_t carry_x86_64_v4<CarryOp::Sub, true>(uint32_t, uint32_t, uint32_t *const *,
                                                      const uint32_t *const *,
                                                      const uint32_t *const *, uint32_t);
template uint32_t carry_x86_64_v4<CarryOp::Subrev, false>(uint32_t, uint32_t, uint32_t *const *,
                                                          const uint32_t *const *,
                                                          const uint32_t *const *, uint32_t);
template uint32_t carry_x86_64_v4<CarryOp::Subrev, true>(uint32_t, uint32_t, uint32_t *const *,
                                                         const uint32_t *const *,
                                                         const uint32_t *const *, uint32_t);

} // namespace rocjitsu::arith
