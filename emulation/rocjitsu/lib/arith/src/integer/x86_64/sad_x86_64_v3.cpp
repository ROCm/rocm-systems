// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 scalar and packed sums of absolute differences with integer accumulators.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "integer/sad_impl.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {
namespace {

// Return the unsigned absolute difference between two values.
template <int Bits, bool Masked> __m256i difference(__m256i a, __m256i b) {
  if constexpr (Bits == 8) {
    auto absolute = _mm256_or_si256(_mm256_subs_epu8(a, b), _mm256_subs_epu8(b, a));
    if constexpr (Masked) {
      absolute = _mm256_andnot_si256(_mm256_cmpeq_epi8(b, _mm256_setzero_si256()), absolute);
    }

    // Sum pairs and then groups of four bytes without mixing GPU lanes.
    auto pairs = _mm256_maddubs_epi16(absolute, _mm256_set1_epi8(1));
    return _mm256_madd_epi16(pairs, _mm256_set1_epi16(1));
  } else if constexpr (Bits == 16) {
    auto absolute = _mm256_sub_epi16(_mm256_max_epu16(a, b), _mm256_min_epu16(a, b));
    return _mm256_add_epi32(_mm256_and_si256(absolute, _mm256_set1_epi32(0xffff)),
                            _mm256_srli_epi32(absolute, 16));
  } else {
    return _mm256_sub_epi32(_mm256_max_epu32(a, b), _mm256_min_epu32(a, b));
  }
}

// Add difference sums to C, saturating unsigned overflow when clamp is enabled.
__m256i accumulate(__m256i sum, __m256i c, bool clamp) {
  auto result = _mm256_add_epi32(sum, c);
  if (clamp) {
    auto bias = _mm256_set1_epi32(INT32_MIN);
    auto carry = _mm256_cmpgt_epi32(_mm256_xor_si256(c, bias), _mm256_xor_si256(result, bias));
    result = _mm256_or_si256(result, carry);
  }
  return result;
}

} // namespace

// Accumulate absolute differences. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <Sad Op>
void sad_x86_64_v3(uint32_t exec_mask, bool clamp, uint32_t *const *d, const uint32_t *const *a,
                   const uint32_t *b, const uint32_t *const *c) {
  constexpr int outputs = sad_outputs(Op);
  constexpr int bits = Op == Sad::U32 ? 32 : Op == Sad::U16 ? 16 : 8;
  for (int lane = 0; lane < 32; lane += 8) {
    auto x = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a[0] + lane));
    auto y = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + lane));
    __m256i accumulators[outputs], result[outputs];
    for (int reg = 0; reg < outputs; ++reg) {
      accumulators[reg] = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(c[reg] + lane));
    }
    if constexpr (sad_quad(Op)) {
      auto hi = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a[1] + lane));
      __m256i windows[] = {x, _mm256_or_si256(_mm256_srli_epi32(x, 8), _mm256_slli_epi32(hi, 24)),
                           _mm256_or_si256(_mm256_srli_epi32(x, 16), _mm256_slli_epi32(hi, 16)),
                           _mm256_or_si256(_mm256_srli_epi32(x, 24), _mm256_slli_epi32(hi, 8))};
      __m256i sums[4];
      for (int window = 0; window < 4; ++window) {
        sums[window] = difference<8, sad_masked(Op)>(windows[window], y);
      }
      for (int reg = 0; reg < outputs; ++reg) {
        if constexpr (sad_packed(Op)) {
          auto packed = _mm256_or_si256(sums[2 * reg], _mm256_slli_epi32(sums[2 * reg + 1], 16));
          result[reg] = clamp ? _mm256_adds_epu16(packed, accumulators[reg])
                              : _mm256_add_epi16(packed, accumulators[reg]);
        } else {
          result[reg] = accumulate(sums[reg], accumulators[reg], clamp);
        }
      }
    } else {
      auto sum = difference<bits, sad_masked(Op)>(x, y);
      if constexpr (Op == Sad::HighU8) {
        sum = _mm256_slli_epi32(sum, 16);
      }
      result[0] = accumulate(sum, accumulators[0], clamp);
    }
    // All inputs for these lanes precede every destination write, including
    // cross-register aliases and repeated destination addresses.
    auto lane_exec_mask = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                                            _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    for (int reg = 0; reg < outputs; ++reg) {
      _mm256_maskstore_epi32(reinterpret_cast<int *>(d[reg] + lane), lane_exec_mask, result[reg]);
    }
  }
}

template void sad_x86_64_v3<Sad::U8>(uint32_t, bool, uint32_t *const *, const uint32_t *const *,
                                     const uint32_t *, const uint32_t *const *);
template void sad_x86_64_v3<Sad::HighU8>(uint32_t, bool, uint32_t *const *, const uint32_t *const *,
                                         const uint32_t *, const uint32_t *const *);
template void sad_x86_64_v3<Sad::U16>(uint32_t, bool, uint32_t *const *, const uint32_t *const *,
                                      const uint32_t *, const uint32_t *const *);
template void sad_x86_64_v3<Sad::U32>(uint32_t, bool, uint32_t *const *, const uint32_t *const *,
                                      const uint32_t *, const uint32_t *const *);
template void sad_x86_64_v3<Sad::MaskedU8>(uint32_t, bool, uint32_t *const *,
                                           const uint32_t *const *, const uint32_t *,
                                           const uint32_t *const *);
template void sad_x86_64_v3<Sad::QuadU16>(uint32_t, bool, uint32_t *const *,
                                          const uint32_t *const *, const uint32_t *,
                                          const uint32_t *const *);
template void sad_x86_64_v3<Sad::MaskedQuadU16>(uint32_t, bool, uint32_t *const *,
                                                const uint32_t *const *, const uint32_t *,
                                                const uint32_t *const *);
template void sad_x86_64_v3<Sad::MaskedQuadU32>(uint32_t, bool, uint32_t *const *,
                                                const uint32_t *const *, const uint32_t *,
                                                const uint32_t *const *);

} // namespace rocjitsu::arith
