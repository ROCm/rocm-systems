// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 scaled FP4/FP8/BF8 widening, rounding and packing. All input
// classes and selectors stay vectorized, with masked destination stores.

#include "conversion/x86_64/scaled_widen_x86_64_v3.h"
#include "common/x86_64/fp8.h"
#include "conversion/scaled_widen.h"
#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith::x86_64_v3 {

// Widen narrow codes, retaining sign for zero, infinity and canonical NaN.
inline __m256 decode(__m256i raw, ScaledSource source) {
  if (source == ScaledSource::Fp8) {
    return widen_fp8<false>(raw);
  }
  if (source == ScaledSource::Bf8) {
    return widen_fp8<true>(raw);
  }
  const auto magnitude = _mm256_setr_ps(0, 0.5f, 1, 1.5f, 2, 3, 4, 6);
  auto value = _mm256_permutevar8x32_ps(magnitude, _mm256_and_si256(raw, _mm256_set1_epi32(7)));
  auto sign = _mm256_slli_epi32(_mm256_and_si256(raw, _mm256_set1_epi32(8)), 28);
  return _mm256_castsi256_ps(_mm256_or_si256(_mm256_castps_si256(value), sign));
}

} // namespace rocjitsu::arith::x86_64_v3

namespace rocjitsu::arith {
// Execute a validated wave, reading both sources before any possibly aliased
// destination store. All formats share scaling, masking and layout handling.
void scaled_widen_x86_64_v3(int lanes, uint64_t exec_mask, ScaledSource source,
                            ScaledDestination destination, bool packed, bool saturate, int shift,
                            int destination_shift, uint32_t *const *d, const uint32_t *a,
                            const uint32_t *b) {
  for (int lane = 0; lane < lanes; lane += 8) {
    auto raw = _mm256_srl_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane)),
                                _mm_cvtsi32_si128(shift));
    auto exponent = _mm256_and_si256(
        _mm256_srli_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + lane)), 23),
        _mm256_set1_epi32(255));
    auto exceptional = _mm256_cmpeq_epi32(exponent, _mm256_set1_epi32(255));
    auto zero = _mm256_cmpeq_epi32(exponent, _mm256_set1_epi32(0));
    auto scale_bits =
        _mm256_blendv_epi8(_mm256_slli_epi32(exponent, 23), _mm256_set1_epi32(0x00400000), zero);
    // A special scale overrides the final result. Substitute one here to avoid
    // unnecessary infinity arithmetic before that override.
    scale_bits = _mm256_blendv_epi8(scale_bits, _mm256_set1_epi32(0x3f800000), exceptional);
    auto scale = _mm256_castsi256_ps(scale_bits);
    const int nan = destination == ScaledDestination::F32   ? 0x7fc00000
                    : destination == ScaledDestination::F16 ? 0x7e00
                                                            : 0x7fc0;
    auto low = x86_64_v3::scaled_widen_detail::encode(
        _mm256_mul_ps(x86_64_v3::decode(raw, source), scale), destination, saturate);
    low = _mm256_blendv_epi8(low, _mm256_set1_epi32(nan), exceptional);
    auto high = _mm256_setzero_si256();
    if (packed) {
      auto upper = _mm256_srl_epi32(raw, _mm_cvtsi32_si128(source == ScaledSource::Fp4 ? 4 : 8));
      high = x86_64_v3::scaled_widen_detail::encode(
          _mm256_mul_ps(x86_64_v3::decode(upper, source), scale), destination, saturate);
      high = _mm256_blendv_epi8(high, _mm256_set1_epi32(nan), exceptional);
      if (destination != ScaledDestination::F32) {
        low = _mm256_or_si256(low, _mm256_slli_epi32(high, 16));
      }
    } else if (destination == ScaledDestination::F16) {
      auto old = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(d[0] + lane));
      auto keep = _mm256_set1_epi32(int(~(0xffffU << destination_shift)));
      low = _mm256_or_si256(_mm256_and_si256(old, keep),
                            _mm256_sll_epi32(low, _mm_cvtsi32_si128(destination_shift)));
    }
    auto active = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                                    _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d[0] + lane), active, low);
    if (packed && destination == ScaledDestination::F32) {
      _mm256_maskstore_epi32(reinterpret_cast<int *>(d[1] + lane), active, high);
    }
  }
}
} // namespace rocjitsu::arith
