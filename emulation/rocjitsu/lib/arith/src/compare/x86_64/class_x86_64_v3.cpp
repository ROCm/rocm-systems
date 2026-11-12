// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 floating-point class tests producing one predicate bit per lane.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "compare/class.h"
#include "rj_arith/rj_arith.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {

// Test floating-point classes.
// Requires the named CPU features and validated instruction controls.
template <int Bits>
uint32_t class_x86_64_v3(uint32_t mode, const uint32_t *const *a, const uint32_t *b) {
  constexpr unsigned fraction_bits = Bits == 16 ? 10 : Bits == 32 ? 23 : 20;
  constexpr uint32_t exponent_mask = Bits == 16 ? 31 : Bits == 32 ? 255 : 2047;
  constexpr uint32_t sign = Bits == 16 ? 0x8000 : 0x80000000;
  uint32_t result = 0;
  for (int lane = 0; lane < 32; lane += 8) {
    auto high = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a[Bits == 64 ? 1 : 0] + lane));
    auto classes = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + lane));
    if constexpr (Bits == 16) {
      high = _mm256_srl_epi32(high, _mm_cvtsi32_si128(mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0));
      high = _mm256_and_si256(high, _mm256_set1_epi32(65535));
      classes = _mm256_srl_epi32(classes, _mm_cvtsi32_si128(mode & RJ_ARITH_ALU_HIGH_B ? 16 : 0));
    }
    high = _mm256_and_si256(high,
                            _mm256_set1_epi32(int(mode & RJ_ARITH_ALU_ABS_A ? ~sign : UINT32_MAX)));
    high = _mm256_xor_si256(high, _mm256_set1_epi32(int(mode & RJ_ARITH_ALU_NEG_A ? sign : 0)));
    auto exponent =
        _mm256_and_si256(_mm256_srli_epi32(high, fraction_bits), _mm256_set1_epi32(exponent_mask));
    auto fraction = _mm256_and_si256(high, _mm256_set1_epi32((1u << fraction_bits) - 1));
    if constexpr (Bits == 64) {
      fraction = _mm256_or_si256(
          fraction, _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a[0] + lane)));
    }
    auto signed_high = high;
    if constexpr (Bits == 16) {
      signed_high = _mm256_slli_epi32(high, 16);
    }
    auto quiet = _mm256_and_si256(_mm256_srli_epi32(high, fraction_bits - 1), _mm256_set1_epi32(1));
    auto zero_fraction = _mm256_cmpeq_epi32(fraction, _mm256_setzero_si256());
    auto zero_exponent = _mm256_cmpeq_epi32(exponent, _mm256_setzero_si256());
    auto max_exponent = _mm256_cmpeq_epi32(exponent, _mm256_set1_epi32(exponent_mask));
    auto index =
        _mm256_blendv_epi8(_mm256_set1_epi32(8),
                           _mm256_sub_epi32(_mm256_set1_epi32(7),
                                            _mm256_and_si256(zero_fraction, _mm256_set1_epi32(1))),
                           zero_exponent);
    index = _mm256_blendv_epi8(index, _mm256_set1_epi32(9), max_exponent);
    index = _mm256_blendv_epi8(index, _mm256_sub_epi32(_mm256_set1_epi32(11), index),
                               _mm256_srai_epi32(signed_high, 31));
    index = _mm256_blendv_epi8(index, quiet, _mm256_andnot_si256(zero_fraction, max_exponent));
    auto matches = _mm256_slli_epi32(_mm256_srlv_epi32(classes, index), 31);
    result |= uint32_t(_mm256_movemask_ps(_mm256_castsi256_ps(matches))) << lane;
  }
  return result;
}

template uint32_t class_x86_64_v3<16>(uint32_t, const uint32_t *const *, const uint32_t *);
template uint32_t class_x86_64_v3<32>(uint32_t, const uint32_t *const *, const uint32_t *);
template uint32_t class_x86_64_v3<64>(uint32_t, const uint32_t *const *, const uint32_t *);

} // namespace rocjitsu::arith
