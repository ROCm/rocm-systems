// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 scaled FP4/FP8/BF8 widening, rounding and packing. All input
// classes and selectors stay vectorized, with masked destination stores.

#include "conversion/x86_64/scaled_widen_x86_64_v4.h"
#include "common/x86_64/fp8.h"
#include "conversion/scaled_widen.h"
#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith::x86_64_v4 {

// Widen narrow codes, retaining sign for zero, infinity and canonical NaN.
inline __m512 decode(__m512i raw, ScaledSource source) {
  if (source == ScaledSource::Fp8) {
    return widen_fp8<false>(raw);
  }
  if (source == ScaledSource::Bf8) {
    return widen_fp8<true>(raw);
  }
  const auto magnitude = _mm512_setr_ps(0, 0.5f, 1, 1.5f, 2, 3, 4, 6, 0, 0.5f, 1, 1.5f, 2, 3, 4, 6);
  auto value = _mm512_permutexvar_ps(_mm512_and_si512(raw, _mm512_set1_epi32(7)), magnitude);
  auto sign = _mm512_slli_epi32(_mm512_and_si512(raw, _mm512_set1_epi32(8)), 28);
  return _mm512_castsi512_ps(_mm512_or_si512(_mm512_castps_si512(value), sign));
}

} // namespace rocjitsu::arith::x86_64_v4

namespace rocjitsu::arith {
// Execute a validated wave, reading both sources before any possibly aliased
// destination store. All formats share scaling, masking and layout handling.
void scaled_widen_x86_64_v4(int lanes, uint64_t exec_mask, ScaledSource source,
                            ScaledDestination destination, bool packed, bool saturate, int shift,
                            int destination_shift, uint32_t *const *d, const uint32_t *a,
                            const uint32_t *b) {
  for (int lane = 0; lane < lanes; lane += 16) {
    auto raw = _mm512_srl_epi32(_mm512_loadu_si512(reinterpret_cast<const __m512i *>(a + lane)),
                                _mm_cvtsi32_si128(shift));
    auto exponent = _mm512_and_si512(
        _mm512_srli_epi32(_mm512_loadu_si512(reinterpret_cast<const __m512i *>(b + lane)), 23),
        _mm512_set1_epi32(255));
    auto exceptional = _mm512_cmpeq_epi32_mask(exponent, _mm512_set1_epi32(255));
    auto zero = _mm512_cmpeq_epi32_mask(exponent, _mm512_set1_epi32(0));
    auto scale_bits = _mm512_mask_blend_epi32(zero, _mm512_slli_epi32(exponent, 23),
                                              _mm512_set1_epi32(0x00400000));
    // A special scale overrides the final result. Substitute one here to avoid
    // unnecessary infinity arithmetic before that override.
    scale_bits = _mm512_mask_blend_epi32(exceptional, scale_bits, _mm512_set1_epi32(0x3f800000));
    auto scale = _mm512_castsi512_ps(scale_bits);
    const int nan = destination == ScaledDestination::F32   ? 0x7fc00000
                    : destination == ScaledDestination::F16 ? 0x7e00
                                                            : 0x7fc0;
    auto low = x86_64_v4::scaled_widen_detail::encode(
        _mm512_mul_ps(x86_64_v4::decode(raw, source), scale), destination, saturate);
    low = _mm512_mask_blend_epi32(exceptional, low, _mm512_set1_epi32(nan));
    auto high = _mm512_setzero_si512();
    if (packed) {
      auto upper = _mm512_srl_epi32(raw, _mm_cvtsi32_si128(source == ScaledSource::Fp4 ? 4 : 8));
      high = x86_64_v4::scaled_widen_detail::encode(
          _mm512_mul_ps(x86_64_v4::decode(upper, source), scale), destination, saturate);
      high = _mm512_mask_blend_epi32(exceptional, high, _mm512_set1_epi32(nan));
      if (destination != ScaledDestination::F32) {
        low = _mm512_or_si512(low, _mm512_slli_epi32(high, 16));
      }
    } else if (destination == ScaledDestination::F16) {
      auto old = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(d[0] + lane));
      auto keep = _mm512_set1_epi32(int(~(0xffffU << destination_shift)));
      low = _mm512_or_si512(_mm512_and_si512(old, keep),
                            _mm512_sll_epi32(low, _mm_cvtsi32_si128(destination_shift)));
    }
    auto active = __mmask16(exec_mask >> lane);
    _mm512_mask_storeu_epi32(d[0] + lane, active, low);
    if (packed && destination == ScaledDestination::F32) {
      _mm512_mask_storeu_epi32(d[1] + lane, active, high);
    }
  }
}
} // namespace rocjitsu::arith
