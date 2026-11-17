// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 scaled FP4/FP8/BF8 narrowing. Source widening, reciprocal scaling,
// rounding and partial writes stay vectorized for all input classes.

#include "conversion/x86_64/scaled_narrow_x86_64_v3.h"
#include "conversion/scaled_narrow.h"
#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {
// Narrow a validated wave, consuming all source vectors before stores so
// whole-register aliases retain the original data, seed and scale values.
template <ScaledNarrowDestination Dst, bool Stochastic>
void scaled_narrow_x86_64_v3(int lanes, uint64_t exec_mask, ScaledNarrowSource source,
                             bool saturate, bool flush_input, int shift, uint32_t *d,
                             const uint32_t *a, const uint32_t *b, const uint32_t *scale,
                             const uint32_t *upper_a) {
  constexpr bool fp4 = Dst == ScaledNarrowDestination::Fp4;
  auto selected = _mm256_set1_epi32(int((Stochastic || fp4 ? 0xffU : 0xffffU) << shift));
  for (int lane = 0; lane < lanes; lane += 8) {
    auto av = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane));
    auto bv = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + lane));
    auto old = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(d + lane));
    auto exponent = _mm256_and_si256(
        _mm256_srli_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(scale + lane)), 23),
        _mm256_set1_epi32(255));
    auto preserve = _mm256_cmpeq_epi32(exponent, _mm256_set1_epi32(255));
    auto subnormal_scale = _mm256_cmpeq_epi32(exponent, _mm256_set1_epi32(254));
    auto inverse_bits = _mm256_slli_epi32(_mm256_sub_epi32(_mm256_set1_epi32(254), exponent), 23);
    inverse_bits = _mm256_blendv_epi8(inverse_bits, _mm256_set1_epi32(0x00400000), subnormal_scale);
    // Exponent 255 keeps D, so use a harmless multiplier in those lanes.
    inverse_bits = _mm256_blendv_epi8(inverse_bits, _mm256_set1_epi32(0x3f800000), preserve);
    auto inverse_scale = _mm256_castsi256_ps(inverse_bits);
    auto scaled =
        _mm256_mul_ps(x86_64_v3::scaled_narrow_decode(av, source, flush_input), inverse_scale);
    auto value =
        x86_64_v3::scaled_narrow_encode<Dst, Stochastic>(_mm256_castps_si256(scaled), bv, saturate);
    if constexpr (!Stochastic || fp4) {
      auto upper = source == ScaledNarrowSource::F32 ? bv : _mm256_srli_epi32(av, 16);
      if constexpr (Stochastic && fp4) {
        if (source == ScaledNarrowSource::F32) {
          upper = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(upper_a + lane));
        }
      }
      auto next_seed =
          _mm256_xor_si256(_mm256_slli_epi32(bv, 1),
                           _mm256_and_si256(_mm256_srai_epi32(bv, 31), _mm256_set1_epi32(197)));
      auto high_scaled =
          _mm256_mul_ps(x86_64_v3::scaled_narrow_decode(upper, source, flush_input), inverse_scale);
      auto high = x86_64_v3::scaled_narrow_encode<Dst, Stochastic>(_mm256_castps_si256(high_scaled),
                                                                   next_seed, saturate);
      value = _mm256_or_si256(value, _mm256_slli_epi32(high, fp4 ? 4 : 8));
    }
    value = _mm256_or_si256(_mm256_andnot_si256(selected, old),
                            _mm256_sll_epi32(value, _mm_cvtsi32_si128(shift)));
    value = _mm256_blendv_epi8(value, old, preserve);
    auto active = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                                    _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), active, value);
  }
}
template void scaled_narrow_x86_64_v3<ScaledNarrowDestination::Fp8, false>(
    int, uint64_t, ScaledNarrowSource, bool, bool, int, uint32_t *, const uint32_t *,
    const uint32_t *, const uint32_t *, const uint32_t *);
template void scaled_narrow_x86_64_v3<ScaledNarrowDestination::Fp8, true>(
    int, uint64_t, ScaledNarrowSource, bool, bool, int, uint32_t *, const uint32_t *,
    const uint32_t *, const uint32_t *, const uint32_t *);
template void scaled_narrow_x86_64_v3<ScaledNarrowDestination::Bf8, false>(
    int, uint64_t, ScaledNarrowSource, bool, bool, int, uint32_t *, const uint32_t *,
    const uint32_t *, const uint32_t *, const uint32_t *);
template void scaled_narrow_x86_64_v3<ScaledNarrowDestination::Bf8, true>(
    int, uint64_t, ScaledNarrowSource, bool, bool, int, uint32_t *, const uint32_t *,
    const uint32_t *, const uint32_t *, const uint32_t *);
template void scaled_narrow_x86_64_v3<ScaledNarrowDestination::Fp4, false>(
    int, uint64_t, ScaledNarrowSource, bool, bool, int, uint32_t *, const uint32_t *,
    const uint32_t *, const uint32_t *, const uint32_t *);
template void scaled_narrow_x86_64_v3<ScaledNarrowDestination::Fp4, true>(
    int, uint64_t, ScaledNarrowSource, bool, bool, int, uint32_t *, const uint32_t *,
    const uint32_t *, const uint32_t *, const uint32_t *);
} // namespace rocjitsu::arith
