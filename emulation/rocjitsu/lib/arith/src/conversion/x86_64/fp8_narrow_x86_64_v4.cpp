// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 FP16/FP32 narrowing to FP8/BF8 with nearest or seeded stochastic rounding.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "conversion/fp8_narrow_impl.h"
#include "rj_arith/detail/rj_arith_conversion.h"

#include <immintrin.h>
#include <stdint.h>

namespace {

using rocjitsu::arith::Fp8NarrowPolicy;

// Select yes where signed a > b, and no otherwise.
__m512i select_greater(__m512i a, __m512i b, __m512i yes, __m512i no) {
  return _mm512_mask_blend_epi32(_mm512_cmp_epi32_mask(a, b, _MM_CMPINT_GT), no, yes);
}

// Narrow FP32 bits to FP8/BF8 using nearest or seeded stochastic rounding and optional
// saturation.
template <bool Bf8, bool Stochastic>
__m512i narrow(__m512i raw, __m512i seed, bool saturate, Fp8NarrowPolicy policy) {
  constexpr int fraction_bits = Bf8 ? 2 : 3;
  bool fnuz = policy == Fp8NarrowPolicy::Fnuz;
  bool unsigned_format = policy == Fp8NarrowPolicy::E5M3;
  bool rdna = policy == Fp8NarrowPolicy::Gfx12;
  int terminal = fnuz ? 128 : unsigned_format ? 255 : Bf8 ? 124 : 127;
  int bias = unsigned_format ? 15 : (Bf8 ? 15 : 7) + int(fnuz);
  auto one = _mm512_set1_epi32(1), zero = _mm512_setzero_si512();
  auto magnitude = _mm512_and_si512(raw, _mm512_set1_epi32(0x7fffffff));
  auto exponent = _mm512_sub_epi32(_mm512_srli_epi32(magnitude, 23), _mm512_set1_epi32(127 - bias));
  auto significand = _mm512_or_si512(_mm512_and_si512(magnitude, _mm512_set1_epi32(0x7fffff)),
                                     _mm512_set1_epi32(0x800000));
  auto alignment = _mm512_max_epi32(_mm512_sub_epi32(one, exponent), zero);
  __m512i result;
  if constexpr (Stochastic) {
    if (rdna) {
      significand = _mm512_srlv_epi32(significand, alignment);
      result = _mm512_srli_epi32(
          _mm512_add_epi32(significand, _mm512_srli_epi32(seed, 9 + fraction_bits)),
          23 - fraction_bits);
    } else {
      auto shift = _mm512_add_epi32(alignment, _mm512_set1_epi32(23 - fraction_bits));
      auto random = _mm512_srlv_epi32(seed, _mm512_sub_epi32(_mm512_set1_epi32(32), shift));
      result = _mm512_srlv_epi32(_mm512_add_epi32(significand, random), shift);
      result = select_greater(shift, _mm512_set1_epi32(24), zero, result);
    }
  } else {
    auto shift = _mm512_min_epi32(
        _mm512_add_epi32(alignment, _mm512_set1_epi32(23 - fraction_bits)), _mm512_set1_epi32(31));
    auto half = _mm512_sllv_epi32(one, _mm512_sub_epi32(shift, one));
    auto odd = _mm512_and_si512(_mm512_srlv_epi32(significand, shift), one);
    auto rounded =
        _mm512_add_epi32(significand, _mm512_add_epi32(_mm512_sub_epi32(half, one), odd));
    result = _mm512_srlv_epi32(rounded, shift);
  }
  auto normal_exponent =
      _mm512_slli_epi32(_mm512_max_epi32(_mm512_sub_epi32(exponent, one), zero), fraction_bits);
  result = _mm512_add_epi32(result, normal_exponent);
  auto limit = _mm512_set1_epi32(terminal);
  if (saturate) {
    auto finite_limit = _mm512_set1_epi32(terminal - 1);
    limit = rdna ? select_greater(_mm512_set1_epi32(0x7f800000), magnitude, finite_limit, limit)
                 : finite_limit;
  }
  result = _mm512_min_epu32(result, limit);
  auto sign =
      unsigned_format ? zero : _mm512_and_si512(_mm512_srli_epi32(raw, 24), _mm512_set1_epi32(128));
  if (fnuz) {
    sign = select_greater(result, zero, sign, zero);
    auto signed_result = _mm512_or_si512(result, sign);
    result = select_greater(result, _mm512_set1_epi32(127), _mm512_set1_epi32(128), signed_result);
  } else {
    result = _mm512_or_si512(result, sign);
  }
  auto nan = fnuz              ? _mm512_set1_epi32(128)
             : unsigned_format ? _mm512_set1_epi32(255)
             : rdna            ? _mm512_set1_epi32(Bf8 ? 0xfe : 0xff)
                               : _mm512_or_si512(sign, _mm512_set1_epi32(127));
  return select_greater(magnitude, _mm512_set1_epi32(0x7f800000), nan, result);
}

} // namespace

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
    auto value = narrow<Bf8, Stochastic>(av, bv, saturate, policy);
    if constexpr (!Stochastic) {
      bv = _mm512_xor_si512(_mm512_and_si512(bv, keep_b), flip_b);
      auto high = narrow<Bf8, false>(bv, _mm512_setzero_si512(), saturate, policy);
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
