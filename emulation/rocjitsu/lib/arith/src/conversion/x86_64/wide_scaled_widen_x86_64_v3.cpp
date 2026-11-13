// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 wide low-precision widening: vector field extraction, finite decoding and
// shared output rounding. Stage each lane group's results before masked stores.

#include "common/x86_64/fp8.h"
#include "conversion/scaled_widen.h"
#include "conversion/x86_64/scaled_widen_x86_64_v3.h"
#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith::x86_64_v3 {
// Decode narrow encodings, retaining signed zero, subnormals and the OCP NaN sign.
template <ScaledSource Src> inline __m256 wide_scaled_decode(__m256i raw) {
  if constexpr (Src == ScaledSource::Fp8 || Src == ScaledSource::Bf8) {
    return widen_fp8<Src == ScaledSource::Bf8>(raw);
  } else if constexpr (Src == ScaledSource::Fp4) {
    auto values = _mm256_setr_ps(0, 0.5f, 1, 1.5f, 2, 3, 4, 6);
    auto value = _mm256_permutevar8x32_ps(values, raw);
    auto sign = _mm256_slli_epi32(_mm256_and_si256(raw, _mm256_set1_epi32(8)), 28);
    return _mm256_castsi256_ps(_mm256_or_si256(_mm256_castps_si256(value), sign));
  }
  constexpr bool Bf6 = Src == ScaledSource::Bf6;
  constexpr int fraction = Bf6 ? 2 : 3, bias = Bf6 ? 3 : 1;
  auto sign = _mm256_slli_epi32(_mm256_and_si256(raw, _mm256_set1_epi32(32)), 26);
  auto exponent = _mm256_srli_epi32(_mm256_and_si256(raw, _mm256_set1_epi32(31)), fraction);
  auto mantissa = _mm256_and_si256(raw, _mm256_set1_epi32((1 << fraction) - 1));
  auto normal = _mm256_or_si256(
      _mm256_slli_epi32(_mm256_add_epi32(exponent, _mm256_set1_epi32(127 - bias)), 23),
      _mm256_slli_epi32(mantissa, 23 - fraction));
  auto subnormal = _mm256_castps_si256(
      _mm256_mul_ps(_mm256_cvtepi32_ps(mantissa), _mm256_set1_ps(Bf6 ? 0.0625f : 0.125f)));
  auto value =
      _mm256_blendv_epi8(normal, subnormal, _mm256_cmpeq_epi32(exponent, _mm256_setzero_si256()));
  return _mm256_castsi256_ps(_mm256_or_si256(value, sign));
}
} // namespace rocjitsu::arith::x86_64_v3

namespace rocjitsu::arith {
// Expand validated source registers. Whole-register aliases are safe because
// every source in a lane group is consumed before any destination is written.
template <ScaledSource Src, bool ByteScale>
void wide_scaled_widen_x86_64_v3(int lanes, uint64_t exec_mask, ScaledDestination destination,
                                 bool saturate, int scale_shift, uint32_t *const *d,
                                 const uint32_t *const *a, const uint32_t *b) {
  constexpr int bits = Src == ScaledSource::Fp4                                 ? 4
                       : (Src == ScaledSource::Fp6 || Src == ScaledSource::Bf6) ? 6
                                                                                : 8;
  constexpr int elements = ByteScale ? (bits == 6 ? 16 : 8) : 32;
  constexpr int input_words = elements * bits / 32;
  int words = destination == ScaledDestination::F32 ? elements : elements / 2;
  for (int lane = 0; lane < lanes; lane += 8) {
    __m256i source[input_words], result[elements];
    for (int word = 0; word < input_words; ++word) {
      source[word] = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a[word] + lane));
    }
    auto exponent = _mm256_and_si256(
        _mm256_srl_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + lane)),
                         _mm_cvtsi32_si128(scale_shift)),
        _mm256_set1_epi32(255));
    auto special = _mm256_cmpeq_epi32(exponent, _mm256_set1_epi32(255));
    auto scale_bits =
        _mm256_blendv_epi8(_mm256_slli_epi32(exponent, 23), _mm256_set1_epi32(0x00400000),
                           _mm256_cmpeq_epi32(exponent, _mm256_setzero_si256()));
    scale_bits = _mm256_blendv_epi8(
        scale_bits, _mm256_set1_epi32(ByteScale ? 0x7fc00000 : 0x3f800000), special);
    auto scale = _mm256_castsi256_ps(scale_bits);
    for (int element = 0; element < elements; ++element) {
      int bit = bits * element, word = bit / 32, shift = bit % 32;
      auto raw = _mm256_srl_epi32(source[word], _mm_cvtsi32_si128(shift));
      if (shift + bits > 32) {
        raw =
            _mm256_or_si256(raw, _mm256_sll_epi32(source[word + 1], _mm_cvtsi32_si128(32 - shift)));
      }
      auto scaled_bits =
          _mm256_castps_si256(_mm256_mul_ps(x86_64_v3::wide_scaled_decode<Src>(raw), scale));
      if constexpr (ByteScale) {
        // Canonicalize the E8M0 NaN scale, including when both operands are NaNs.
        scaled_bits = _mm256_blendv_epi8(scaled_bits, _mm256_set1_epi32(0x7fc00000), special);
      }
      auto value = x86_64_v3::scaled_widen_detail::encode(_mm256_castsi256_ps(scaled_bits),
                                                          destination, saturate);
      if (destination == ScaledDestination::F32) {
        result[element] = value;
      } else if (element % 2) {
        result[element / 2] = _mm256_or_si256(result[element / 2], _mm256_slli_epi32(value, 16));
      } else {
        result[element / 2] = value;
      }
    }
    // gfx950 special scales suppress writes; gfx1250 writes the NaN results.
    auto active = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                                    _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    if constexpr (!ByteScale) {
      active = _mm256_andnot_si256(special, active);
    }
    for (int word = 0; word < words; ++word) {
      _mm256_maskstore_epi32(reinterpret_cast<int *>(d[word] + lane), active, result[word]);
    }
  }
}
template void wide_scaled_widen_x86_64_v3<ScaledSource::Fp4, true>(int, uint64_t, ScaledDestination,
                                                                   bool, int, uint32_t *const *,
                                                                   const uint32_t *const *,
                                                                   const uint32_t *);
template void wide_scaled_widen_x86_64_v3<ScaledSource::Fp8, true>(int, uint64_t, ScaledDestination,
                                                                   bool, int, uint32_t *const *,
                                                                   const uint32_t *const *,
                                                                   const uint32_t *);
template void wide_scaled_widen_x86_64_v3<ScaledSource::Bf8, true>(int, uint64_t, ScaledDestination,
                                                                   bool, int, uint32_t *const *,
                                                                   const uint32_t *const *,
                                                                   const uint32_t *);
template void wide_scaled_widen_x86_64_v3<ScaledSource::Fp6, false>(int, uint64_t,
                                                                    ScaledDestination, bool, int,
                                                                    uint32_t *const *,
                                                                    const uint32_t *const *,
                                                                    const uint32_t *);
template void wide_scaled_widen_x86_64_v3<ScaledSource::Fp6, true>(int, uint64_t, ScaledDestination,
                                                                   bool, int, uint32_t *const *,
                                                                   const uint32_t *const *,
                                                                   const uint32_t *);
template void wide_scaled_widen_x86_64_v3<ScaledSource::Bf6, false>(int, uint64_t,
                                                                    ScaledDestination, bool, int,
                                                                    uint32_t *const *,
                                                                    const uint32_t *const *,
                                                                    const uint32_t *);
template void wide_scaled_widen_x86_64_v3<ScaledSource::Bf6, true>(int, uint64_t, ScaledDestination,
                                                                   bool, int, uint32_t *const *,
                                                                   const uint32_t *const *,
                                                                   const uint32_t *);
} // namespace rocjitsu::arith
