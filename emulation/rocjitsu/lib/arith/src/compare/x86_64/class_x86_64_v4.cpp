// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 floating-point class tests producing one predicate bit per lane.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "compare/class.h"
#include "rj_arith/rj_arith.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {

// Test floating-point classes.
// Requires the named CPU features and validated instruction controls.
template <int Bits>
uint32_t class_x86_64_v4(uint32_t mode, const uint32_t *const *a, const uint32_t *b) {
  constexpr unsigned fraction_bits = Bits == 16 ? 10 : Bits == 32 ? 23 : 20;
  constexpr uint32_t exponent_mask = Bits == 16 ? 31 : Bits == 32 ? 255 : 2047;
  constexpr uint32_t sign = Bits == 16 ? 0x8000 : 0x80000000;
  uint32_t result = 0;
  for (int lane = 0; lane < 32; lane += 16) {
    auto high = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(a[Bits == 64 ? 1 : 0] + lane));
    auto classes = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(b + lane));
    if constexpr (Bits == 16) {
      high = _mm512_srl_epi32(high, _mm_cvtsi32_si128(mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0));
      high = _mm512_and_si512(high, _mm512_set1_epi32(65535));
      classes = _mm512_srl_epi32(classes, _mm_cvtsi32_si128(mode & RJ_ARITH_ALU_HIGH_B ? 16 : 0));
    }
    high = _mm512_and_si512(high,
                            _mm512_set1_epi32(int(mode & RJ_ARITH_ALU_ABS_A ? ~sign : UINT32_MAX)));
    high = _mm512_xor_si512(high, _mm512_set1_epi32(int(mode & RJ_ARITH_ALU_NEG_A ? sign : 0)));
    auto exponent =
        _mm512_and_si512(_mm512_srli_epi32(high, fraction_bits), _mm512_set1_epi32(exponent_mask));
    auto fraction = _mm512_and_si512(high, _mm512_set1_epi32((1u << fraction_bits) - 1));
    if constexpr (Bits == 64) {
      fraction = _mm512_or_si512(
          fraction, _mm512_loadu_si512(reinterpret_cast<const __m512i *>(a[0] + lane)));
    }
    auto signed_high = high;
    if constexpr (Bits == 16) {
      signed_high = _mm512_slli_epi32(high, 16);
    }
    auto quiet = _mm512_and_si512(_mm512_srli_epi32(high, fraction_bits - 1), _mm512_set1_epi32(1));
    auto zero_fraction = _mm512_cmpeq_epi32_mask(fraction, _mm512_setzero_si512());
    auto zero_exponent = _mm512_cmpeq_epi32_mask(exponent, _mm512_setzero_si512());
    auto max_exponent = _mm512_cmpeq_epi32_mask(exponent, _mm512_set1_epi32(exponent_mask));
    auto tiny_index =
        _mm512_mask_blend_epi32(zero_fraction, _mm512_set1_epi32(7), _mm512_set1_epi32(6));
    auto index = _mm512_mask_blend_epi32(zero_exponent, _mm512_set1_epi32(8), tiny_index);
    index = _mm512_mask_mov_epi32(index, max_exponent, _mm512_set1_epi32(9));
    index = _mm512_mask_blend_epi32(_mm512_movepi32_mask(signed_high), index,
                                    _mm512_sub_epi32(_mm512_set1_epi32(11), index));
    index = _mm512_mask_mov_epi32(index, max_exponent & ~zero_fraction, quiet);
    result |=
        uint32_t(_mm512_test_epi32_mask(classes, _mm512_sllv_epi32(_mm512_set1_epi32(1), index)))
        << lane;
  }
  return result;
}

template uint32_t class_x86_64_v4<16>(uint32_t, const uint32_t *const *, const uint32_t *);
template uint32_t class_x86_64_v4<32>(uint32_t, const uint32_t *const *, const uint32_t *);
template uint32_t class_x86_64_v4<64>(uint32_t, const uint32_t *const *, const uint32_t *);

} // namespace rocjitsu::arith
