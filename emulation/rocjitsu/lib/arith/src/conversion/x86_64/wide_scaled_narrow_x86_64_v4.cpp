// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Wide scaled low-precision narrowing, with vector rounding, seed advancement
// and register packing shared across finite and OCP formats.
#include "conversion/scaled_narrow.h"
#include "conversion/x86_64/scaled_narrow_x86_64_v4.h"
#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {
// Consume all source elements before publishing each lane group's results.
// Inactive lanes preserve D; on scale exponent 255, active gfx950 lanes write signed max.
template <ScaledNarrowDestination Dst, bool Stochastic, bool Gfx1250>
void wide_scaled_narrow_x86_64_v4(int lanes, uint64_t exec_mask, ScaledNarrowSource source,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, const uint32_t *scale, bool saturate,
                                  bool flush_input) {
  constexpr int bits =
      Dst == ScaledNarrowDestination::Fp4                                            ? 4
      : (Dst == ScaledNarrowDestination::Fp6 || Dst == ScaledNarrowDestination::Bf6) ? 6
                                                                                     : 8;
  constexpr int elements = Gfx1250 ? (bits == 6 ? 16 : 8) : 32;
  constexpr int words = elements * bits / 32;
  auto sig_mask = _mm512_set1_epi32(int(scaled_sig_mask(source)));
  auto sign_mask = _mm512_set1_epi32(int(0x80000000U));
  auto qnan_mag = _mm512_set1_epi32(0x7fc00000);
  for (int lane = 0; lane < lanes; lane += 16) {
    if (uint16_t(exec_mask >> lane) == 0) {
      continue;
    }
    auto raw_scale = _mm512_loadu_si512(scale + lane);
    auto exponent = _mm512_and_si512(_mm512_srli_epi32(raw_scale, 23), _mm512_set1_epi32(255));
    auto nan_scale = _mm512_cmpeq_epi32_mask(exponent, _mm512_set1_epi32(255));
    auto subnormal_scale = _mm512_cmpeq_epi32_mask(exponent, _mm512_set1_epi32(254));
    auto inverse_bits = _mm512_slli_epi32(_mm512_sub_epi32(_mm512_set1_epi32(254), exponent), 23);
    inverse_bits =
        _mm512_mask_blend_epi32(subnormal_scale, inverse_bits, _mm512_set1_epi32(0x00400000));
    // Exponent 255 is handled separately from the multiply.
    inverse_bits = _mm512_mask_blend_epi32(nan_scale, inverse_bits, _mm512_set1_epi32(0x3f800000));
    auto inverse_scale = _mm512_castsi512_ps(inverse_bits);

    auto seed = _mm512_setzero_si512();
    if constexpr (Stochastic) {
      seed = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(b[0] + lane));
    }
    __m512i result[words];
    for (int word = 0; word < words; ++word) {
      result[word] = _mm512_setzero_si512();
    }
    for (int element = 0; element < elements; ++element) {
      const uint32_t *input;
      if (source == ScaledNarrowSource::F32) {
        if constexpr (Stochastic || Gfx1250) {
          input = a[element];
        } else {
          // Index before selecting to avoid GCC 13.3 miscompiling the array base.
          input = element % 2 ? b[element / 2] : a[element / 2];
        }
      } else {
        input = a[element / 2];
      }
      auto raw = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(input + lane));
      if (source != ScaledNarrowSource::F32 && element % 2) {
        raw = _mm512_srli_epi32(raw, 16);
      }
      auto decoded = x86_64_v4::scaled_narrow_decode(raw, source, flush_input);
      auto scaled = _mm512_castps_si512(_mm512_mul_ps(decoded, inverse_scale));
      if constexpr (Gfx1250) {
        scaled = _mm512_mask_blend_epi32(nan_scale, scaled, _mm512_set1_epi32(kE8M0ScaleNaNBits));
      } else {
        scaled = _mm512_mask_blend_epi32(
            nan_scale, scaled,
            _mm512_or_si512(_mm512_and_si512(_mm512_castps_si512(decoded), sign_mask), qnan_mag));
      }
      auto value = x86_64_v4::scaled_narrow_encode<Dst, Stochastic, !Gfx1250>(scaled, seed,
                                                                              saturate, sig_mask);
      // Six-bit formats can straddle adjacent destination registers.
      int bit = element * bits, word = bit / 32, shift = bit % 32;
      result[word] =
          _mm512_or_si512(result[word], _mm512_sll_epi32(value, _mm_cvtsi32_si128(shift)));
      if (shift > 32 - bits) {
        result[word + 1] = _mm512_or_si512(result[word + 1],
                                           _mm512_srl_epi32(value, _mm_cvtsi32_si128(32 - shift)));
      }
      if constexpr (Stochastic) {
        seed =
            _mm512_xor_si512(_mm512_slli_epi32(seed, 1),
                             _mm512_and_si512(_mm512_srai_epi32(seed, 31), _mm512_set1_epi32(197)));
      }
    }
    auto active = __mmask16(exec_mask >> lane);
    for (int word = 0; word < words; ++word) {
      _mm512_mask_storeu_epi32(d[word] + lane, active, result[word]);
    }
  }
}
template void wide_scaled_narrow_x86_64_v4<ScaledNarrowDestination::Fp6, false, false>(
    int, uint64_t, ScaledNarrowSource, uint32_t *const *, const uint32_t *const *,
    const uint32_t *const *, const uint32_t *, bool, bool);
template void wide_scaled_narrow_x86_64_v4<ScaledNarrowDestination::Fp6, false, true>(
    int, uint64_t, ScaledNarrowSource, uint32_t *const *, const uint32_t *const *,
    const uint32_t *const *, const uint32_t *, bool, bool);
template void wide_scaled_narrow_x86_64_v4<ScaledNarrowDestination::Fp6, true, false>(
    int, uint64_t, ScaledNarrowSource, uint32_t *const *, const uint32_t *const *,
    const uint32_t *const *, const uint32_t *, bool, bool);
template void wide_scaled_narrow_x86_64_v4<ScaledNarrowDestination::Fp6, true, true>(
    int, uint64_t, ScaledNarrowSource, uint32_t *const *, const uint32_t *const *,
    const uint32_t *const *, const uint32_t *, bool, bool);
template void wide_scaled_narrow_x86_64_v4<ScaledNarrowDestination::Bf6, false, false>(
    int, uint64_t, ScaledNarrowSource, uint32_t *const *, const uint32_t *const *,
    const uint32_t *const *, const uint32_t *, bool, bool);
template void wide_scaled_narrow_x86_64_v4<ScaledNarrowDestination::Bf6, false, true>(
    int, uint64_t, ScaledNarrowSource, uint32_t *const *, const uint32_t *const *,
    const uint32_t *const *, const uint32_t *, bool, bool);
template void wide_scaled_narrow_x86_64_v4<ScaledNarrowDestination::Bf6, true, false>(
    int, uint64_t, ScaledNarrowSource, uint32_t *const *, const uint32_t *const *,
    const uint32_t *const *, const uint32_t *, bool, bool);
template void wide_scaled_narrow_x86_64_v4<ScaledNarrowDestination::Bf6, true, true>(
    int, uint64_t, ScaledNarrowSource, uint32_t *const *, const uint32_t *const *,
    const uint32_t *const *, const uint32_t *, bool, bool);
template void wide_scaled_narrow_x86_64_v4<ScaledNarrowDestination::Fp4, false, true>(
    int, uint64_t, ScaledNarrowSource, uint32_t *const *, const uint32_t *const *,
    const uint32_t *const *, const uint32_t *, bool, bool);
template void wide_scaled_narrow_x86_64_v4<ScaledNarrowDestination::Fp4, true, true>(
    int, uint64_t, ScaledNarrowSource, uint32_t *const *, const uint32_t *const *,
    const uint32_t *const *, const uint32_t *, bool, bool);
template void wide_scaled_narrow_x86_64_v4<ScaledNarrowDestination::Fp8, false, true>(
    int, uint64_t, ScaledNarrowSource, uint32_t *const *, const uint32_t *const *,
    const uint32_t *const *, const uint32_t *, bool, bool);
template void wide_scaled_narrow_x86_64_v4<ScaledNarrowDestination::Fp8, true, true>(
    int, uint64_t, ScaledNarrowSource, uint32_t *const *, const uint32_t *const *,
    const uint32_t *const *, const uint32_t *, bool, bool);
template void wide_scaled_narrow_x86_64_v4<ScaledNarrowDestination::Bf8, false, true>(
    int, uint64_t, ScaledNarrowSource, uint32_t *const *, const uint32_t *const *,
    const uint32_t *const *, const uint32_t *, bool, bool);
template void wide_scaled_narrow_x86_64_v4<ScaledNarrowDestination::Bf8, true, true>(
    int, uint64_t, ScaledNarrowSource, uint32_t *const *, const uint32_t *const *,
    const uint32_t *const *, const uint32_t *, bool, bool);
} // namespace rocjitsu::arith
