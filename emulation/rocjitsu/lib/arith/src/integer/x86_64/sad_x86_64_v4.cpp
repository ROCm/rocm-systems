// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 scalar and packed sums of absolute differences with integer accumulators.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "common/compiler.h"
#include "integer/sad_impl.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {
namespace {

// Return the unsigned absolute difference between two values.
template <int Bits, bool Masked> __m512i difference(__m512i a, __m512i b) {
  if constexpr (Bits == 8) {
    auto absolute = _mm512_or_si512(_mm512_subs_epu8(a, b), _mm512_subs_epu8(b, a));
    if constexpr (Masked) {
      absolute =
          _mm512_maskz_mov_epi8(_mm512_cmpneq_epi8_mask(b, _mm512_setzero_si512()), absolute);
    }

    // Sum pairs and then groups of four bytes without mixing GPU lanes.
    auto pairs = _mm512_maddubs_epi16(absolute, _mm512_set1_epi8(1));
    return _mm512_madd_epi16(pairs, _mm512_set1_epi16(1));
  } else if constexpr (Bits == 16) {
    auto absolute = _mm512_sub_epi16(_mm512_max_epu16(a, b), _mm512_min_epu16(a, b));
    return _mm512_add_epi32(_mm512_and_si512(absolute, _mm512_set1_epi32(0xffff)),
                            _mm512_srli_epi32(absolute, 16));
  } else {
    return _mm512_sub_epi32(_mm512_max_epu32(a, b), _mm512_min_epu32(a, b));
  }
}

// Add difference sums to C, saturating unsigned overflow when clamp is enabled.
__m512i accumulate(__m512i sum, __m512i c, bool clamp) {
  auto result = _mm512_add_epi32(sum, c);
  if (clamp) {
    auto carry = _mm512_cmplt_epu32_mask(result, c);
    result = _mm512_mask_mov_epi32(result, carry, _mm512_set1_epi32(-1));
  }
  return result;
}

} // namespace

// Accumulate absolute differences. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <Sad Op>
void sad_x86_64_v4(uint32_t exec_mask, bool clamp, uint32_t *const *d, const uint32_t *const *a,
                   const uint32_t *b, const uint32_t *const *c) {
  constexpr int outputs = sad_outputs(Op);
  constexpr int bits = Op == Sad::U32 ? 32 : Op == Sad::U16 ? 16 : 8;
  for (int lane = 0; lane < 32; lane += 16) {
    auto x = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(a[0] + lane));
    auto y = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(b + lane));
    __m512i accumulators[outputs], result[outputs];
    RJ_ARITH_UNROLL(4)
    for (int reg = 0; reg < outputs; ++reg) {
      accumulators[reg] = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(c[reg] + lane));
    }
    if constexpr (sad_quad(Op)) {
      auto hi = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(a[1] + lane));
      __m512i windows[] = {x, _mm512_or_si512(_mm512_srli_epi32(x, 8), _mm512_slli_epi32(hi, 24)),
                           _mm512_or_si512(_mm512_srli_epi32(x, 16), _mm512_slli_epi32(hi, 16)),
                           _mm512_or_si512(_mm512_srli_epi32(x, 24), _mm512_slli_epi32(hi, 8))};
      __m512i sums[4];
      RJ_ARITH_UNROLL(4)
      for (int window = 0; window < 4; ++window) {
        sums[window] = difference<8, sad_masked(Op)>(windows[window], y);
      }
      RJ_ARITH_UNROLL(4)
      for (int reg = 0; reg < outputs; ++reg) {
        if constexpr (sad_packed(Op)) {
          auto packed = _mm512_or_si512(sums[2 * reg], _mm512_slli_epi32(sums[2 * reg + 1], 16));
          result[reg] = clamp ? _mm512_adds_epu16(packed, accumulators[reg])
                              : _mm512_add_epi16(packed, accumulators[reg]);
        } else {
          result[reg] = accumulate(sums[reg], accumulators[reg], clamp);
        }
      }
    } else {
      auto sum = difference<bits, sad_masked(Op)>(x, y);
      if constexpr (Op == Sad::HighU8) {
        sum = _mm512_slli_epi32(sum, 16);
      }
      result[0] = accumulate(sum, accumulators[0], clamp);
    }
    // All inputs for these lanes precede every destination write, including
    // cross-register aliases and repeated destination addresses.
    const __mmask16 lane_exec_mask = __mmask16(exec_mask >> lane);
    RJ_ARITH_UNROLL(4)
    for (int reg = 0; reg < outputs; ++reg) {
      _mm512_mask_storeu_epi32(d[reg] + lane, lane_exec_mask, result[reg]);
    }
  }
}

template void sad_x86_64_v4<Sad::U8>(uint32_t, bool, uint32_t *const *, const uint32_t *const *,
                                     const uint32_t *, const uint32_t *const *);
template void sad_x86_64_v4<Sad::HighU8>(uint32_t, bool, uint32_t *const *, const uint32_t *const *,
                                         const uint32_t *, const uint32_t *const *);
template void sad_x86_64_v4<Sad::U16>(uint32_t, bool, uint32_t *const *, const uint32_t *const *,
                                      const uint32_t *, const uint32_t *const *);
template void sad_x86_64_v4<Sad::U32>(uint32_t, bool, uint32_t *const *, const uint32_t *const *,
                                      const uint32_t *, const uint32_t *const *);
template void sad_x86_64_v4<Sad::MaskedU8>(uint32_t, bool, uint32_t *const *,
                                           const uint32_t *const *, const uint32_t *,
                                           const uint32_t *const *);
template void sad_x86_64_v4<Sad::QuadU16>(uint32_t, bool, uint32_t *const *,
                                          const uint32_t *const *, const uint32_t *,
                                          const uint32_t *const *);
template void sad_x86_64_v4<Sad::MaskedQuadU16>(uint32_t, bool, uint32_t *const *,
                                                const uint32_t *const *, const uint32_t *,
                                                const uint32_t *const *);
template void sad_x86_64_v4<Sad::MaskedQuadU32>(uint32_t, bool, uint32_t *const *,
                                                const uint32_t *const *, const uint32_t *,
                                                const uint32_t *const *);

} // namespace rocjitsu::arith
