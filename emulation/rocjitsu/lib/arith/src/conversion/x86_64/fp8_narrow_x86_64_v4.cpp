// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 FP16/FP32 narrowing to FP8/BF8 with nearest or seeded stochastic rounding.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "conversion/x86_64/fp8_narrow_x86_64_v4.h"
#include "conversion/fp8_narrow_impl.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_conversion.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {

// Narrow FP16/FP32 to FP8/BF8. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <bool Bf8, bool Stochastic, bool Half>
void fp8_narrow_x86_64_v4(uint32_t exec_mask, uint32_t mode, bool saturate, uint32_t *d,
                          const uint32_t *a, const uint32_t *b, Fp8NarrowPolicy policy) {
  unsigned shift = Stochastic ? ((mode >> 16) & 3) * 8 : mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
  auto count = _mm_cvtsi32_si128(int(shift));
  uint32_t selected_bits = (Stochastic ? 255U : 65535U) << shift;
  auto selected = _mm512_set1_epi32(int(selected_bits));
  auto keep_a = _mm512_set1_epi32(mode & RJ_ARITH_ALU_ABS_A ? INT32_MAX : -1);
  auto flip_a = _mm512_set1_epi32(mode & RJ_ARITH_ALU_NEG_A ? INT32_MIN : 0);
  auto keep_b = _mm512_set1_epi32(mode & RJ_ARITH_ALU_ABS_B ? INT32_MAX : -1);
  auto flip_b = _mm512_set1_epi32(mode & RJ_ARITH_ALU_NEG_B ? INT32_MIN : 0);
  for (int lane = 0; lane < 32; lane += 16) {
    auto av = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(a + lane));
    __m512i bv;
    if constexpr (Half && !Stochastic) {
      bv = _mm512_setzero_si512();
    } else {
      bv = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(b + lane));
    }
    if constexpr (Half) {
      // Native half widening may quiet NaNs; narrowing only depends on their sign.
      auto widen = [](__m512i raw) {
        return _mm512_castps_si512(_mm512_cvtph_ps(_mm512_cvtepi32_epi16(raw)));
      };
      if constexpr (!Stochastic) {
        bv = widen(_mm512_srli_epi32(av, 16));
      }
      if (Stochastic && (mode & RJ_ARITH_ALU_HIGH_A)) {
        av = _mm512_srli_epi32(av, 16);
      }
      av = widen(av);
    }
    auto old = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(d + lane));
    av = _mm512_xor_si512(_mm512_and_si512(av, keep_a), flip_a);
    auto value = x86_64_v4::fp8_narrow_detail::narrow<Bf8, Stochastic>(av, bv, saturate, policy);
    if constexpr (!Stochastic) {
      bv = _mm512_xor_si512(_mm512_and_si512(bv, keep_b), flip_b);
      auto high = x86_64_v4::fp8_narrow_detail::narrow<Bf8, false>(bv, _mm512_setzero_si512(),
                                                                   saturate, policy);
      value = _mm512_or_si512(value, _mm512_slli_epi32(high, 8));
    }
    value = _mm512_or_si512(_mm512_andnot_si512(selected, old), _mm512_sll_epi32(value, count));
    _mm512_mask_storeu_epi32(d + lane, __mmask16(exec_mask >> lane), value);
  }
}

template void fp8_narrow_x86_64_v4<false, false, false>(uint32_t, uint32_t, bool, uint32_t *,
                                                        const uint32_t *, const uint32_t *,
                                                        Fp8NarrowPolicy);

template void fp8_narrow_x86_64_v4<false, false, true>(uint32_t, uint32_t, bool, uint32_t *,
                                                       const uint32_t *, const uint32_t *,
                                                       Fp8NarrowPolicy);

template void fp8_narrow_x86_64_v4<false, true, false>(uint32_t, uint32_t, bool, uint32_t *,
                                                       const uint32_t *, const uint32_t *,
                                                       Fp8NarrowPolicy);

template void fp8_narrow_x86_64_v4<false, true, true>(uint32_t, uint32_t, bool, uint32_t *,
                                                      const uint32_t *, const uint32_t *,
                                                      Fp8NarrowPolicy);

template void fp8_narrow_x86_64_v4<true, false, false>(uint32_t, uint32_t, bool, uint32_t *,
                                                       const uint32_t *, const uint32_t *,
                                                       Fp8NarrowPolicy);

template void fp8_narrow_x86_64_v4<true, false, true>(uint32_t, uint32_t, bool, uint32_t *,
                                                      const uint32_t *, const uint32_t *,
                                                      Fp8NarrowPolicy);

template void fp8_narrow_x86_64_v4<true, true, false>(uint32_t, uint32_t, bool, uint32_t *,
                                                      const uint32_t *, const uint32_t *,
                                                      Fp8NarrowPolicy);

template void fp8_narrow_x86_64_v4<true, true, true>(uint32_t, uint32_t, bool, uint32_t *,
                                                     const uint32_t *, const uint32_t *,
                                                     Fp8NarrowPolicy);

} // namespace rocjitsu::arith
