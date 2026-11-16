// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 scaled FP4/FP8/BF8 narrowing. Source widening, reciprocal scaling,
// rounding and partial writes stay vectorized for all input classes.

#include "conversion/x86_64/scaled_narrow_x86_64_v4.h"
#include "conversion/scaled_narrow.h"
#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {
// Narrow a validated wave, consuming all source vectors before stores so
// whole-register aliases retain the original data, seed and scale values.
template <ScaledNarrowDestination Dst, bool Stochastic>
void scaled_narrow_x86_64_v4(int lanes, uint64_t exec_mask, ScaledNarrowSource source,
                             bool saturate, bool flush_input, int shift, uint32_t *d,
                             const uint32_t *a, const uint32_t *b, const uint32_t *scale,
                             const uint32_t *upper_a) {
  constexpr bool fp4 = Dst == ScaledNarrowDestination::Fp4;
  auto selected = _mm512_set1_epi32(int((Stochastic || fp4 ? 0xffU : 0xffffU) << shift));
  for (int lane = 0; lane < lanes; lane += 16) {
    auto av = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(a + lane));
    auto bv = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(b + lane));
    auto old = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(d + lane));
    auto exponent = _mm512_and_si512(
        _mm512_srli_epi32(_mm512_loadu_si512(reinterpret_cast<const __m512i *>(scale + lane)), 23),
        _mm512_set1_epi32(255));
    auto preserve = _mm512_cmpeq_epi32_mask(exponent, _mm512_set1_epi32(255));
    auto subnormal_scale = _mm512_cmpeq_epi32_mask(exponent, _mm512_set1_epi32(254));
    auto inverse_bits = _mm512_slli_epi32(_mm512_sub_epi32(_mm512_set1_epi32(254), exponent), 23);
    inverse_bits =
        _mm512_mask_blend_epi32(subnormal_scale, inverse_bits, _mm512_set1_epi32(0x00400000));
    // Exponent 255 keeps D, so use a harmless multiplier in those lanes.
    inverse_bits = _mm512_mask_blend_epi32(preserve, inverse_bits, _mm512_set1_epi32(0x3f800000));
    auto inverse_scale = _mm512_castsi512_ps(inverse_bits);
    auto scaled =
        _mm512_mul_ps(x86_64_v4::scaled_narrow_decode(av, source, flush_input), inverse_scale);
    auto value =
        x86_64_v4::scaled_narrow_encode<Dst, Stochastic>(_mm512_castps_si512(scaled), bv, saturate);
    if constexpr (!Stochastic || fp4) {
      auto upper = source == ScaledNarrowSource::F32 ? bv : _mm512_srli_epi32(av, 16);
      if constexpr (Stochastic && fp4) {
        if (source == ScaledNarrowSource::F32) {
          upper = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(upper_a + lane));
        }
      }
      auto next_seed =
          _mm512_xor_si512(_mm512_slli_epi32(bv, 1),
                           _mm512_and_si512(_mm512_srai_epi32(bv, 31), _mm512_set1_epi32(197)));
      auto high_scaled =
          _mm512_mul_ps(x86_64_v4::scaled_narrow_decode(upper, source, flush_input), inverse_scale);
      auto high = x86_64_v4::scaled_narrow_encode<Dst, Stochastic>(_mm512_castps_si512(high_scaled),
                                                                   next_seed, saturate);
      value = _mm512_or_si512(value, _mm512_slli_epi32(high, fp4 ? 4 : 8));
    }
    value = _mm512_or_si512(_mm512_andnot_si512(selected, old),
                            _mm512_sll_epi32(value, _mm_cvtsi32_si128(shift)));
    value = _mm512_mask_blend_epi32(preserve, value, old);
    _mm512_mask_storeu_epi32(d + lane, __mmask16(exec_mask >> lane), value);
  }
}
template void scaled_narrow_x86_64_v4<ScaledNarrowDestination::Fp8, false>(
    int, uint64_t, ScaledNarrowSource, bool, bool, int, uint32_t *, const uint32_t *,
    const uint32_t *, const uint32_t *, const uint32_t *);
template void scaled_narrow_x86_64_v4<ScaledNarrowDestination::Fp8, true>(
    int, uint64_t, ScaledNarrowSource, bool, bool, int, uint32_t *, const uint32_t *,
    const uint32_t *, const uint32_t *, const uint32_t *);
template void scaled_narrow_x86_64_v4<ScaledNarrowDestination::Bf8, false>(
    int, uint64_t, ScaledNarrowSource, bool, bool, int, uint32_t *, const uint32_t *,
    const uint32_t *, const uint32_t *, const uint32_t *);
template void scaled_narrow_x86_64_v4<ScaledNarrowDestination::Bf8, true>(
    int, uint64_t, ScaledNarrowSource, bool, bool, int, uint32_t *, const uint32_t *,
    const uint32_t *, const uint32_t *, const uint32_t *);
template void scaled_narrow_x86_64_v4<ScaledNarrowDestination::Fp4, false>(
    int, uint64_t, ScaledNarrowSource, bool, bool, int, uint32_t *, const uint32_t *,
    const uint32_t *, const uint32_t *, const uint32_t *);
template void scaled_narrow_x86_64_v4<ScaledNarrowDestination::Fp4, true>(
    int, uint64_t, ScaledNarrowSource, bool, bool, int, uint32_t *, const uint32_t *,
    const uint32_t *, const uint32_t *, const uint32_t *);
} // namespace rocjitsu::arith
