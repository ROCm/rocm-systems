// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 signed and unsigned integer comparisons producing lane masks.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "compare/integer_compare.h"
#include "rj_arith/rj_arith.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {

// Compare signed/unsigned integers.
// Requires the named CPU features and validated instruction controls.
template <int Bits, bool Signed, unsigned Predicate>
uint32_t integer_compare_x86_64_v4(uint32_t mode, const uint32_t *const *a,
                                   const uint32_t *const *b) {
  uint32_t less = 0, equal = 0;
  for (int lane = 0; lane < 32; lane += 16) {
    auto av = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(a[Bits == 64 ? 1 : 0] + lane));
    auto bv = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(b[Bits == 64 ? 1 : 0] + lane));
    if constexpr (Bits == 16) {
      av = _mm512_and_si512(
          _mm512_srl_epi32(av, _mm_cvtsi32_si128(mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0)),
          _mm512_set1_epi32(65535));
      bv = _mm512_and_si512(
          _mm512_srl_epi32(bv, _mm_cvtsi32_si128(mode & RJ_ARITH_ALU_HIGH_B ? 16 : 0)),
          _mm512_set1_epi32(65535));
    }
    constexpr uint32_t flip = Signed ? (Bits == 16 ? 0x8000 : 0x80000000) : 0;
    av = _mm512_xor_si512(av, _mm512_set1_epi32(int(flip)));
    bv = _mm512_xor_si512(bv, _mm512_set1_epi32(int(flip)));
    auto lt = _mm512_cmplt_epu32_mask(av, bv), eq = _mm512_cmpeq_epi32_mask(av, bv);
    if constexpr (Bits == 64) {
      auto al = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(a[0] + lane));
      auto bl = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(b[0] + lane));
      lt |= eq & _mm512_cmplt_epu32_mask(al, bl);
      eq &= _mm512_cmpeq_epi32_mask(al, bl);
    }
    less |= uint32_t(lt) << lane;
    equal |= uint32_t(eq) << lane;
  }
  return integer_compare_result<Predicate>(less, equal);
}

template uint32_t integer_compare_x86_64_v4<16, true, 1>(uint32_t, const uint32_t *const *,
                                                         const uint32_t *const *);
template uint32_t integer_compare_x86_64_v4<16, true, 2>(uint32_t, const uint32_t *const *,
                                                         const uint32_t *const *);
template uint32_t integer_compare_x86_64_v4<16, true, 3>(uint32_t, const uint32_t *const *,
                                                         const uint32_t *const *);
template uint32_t integer_compare_x86_64_v4<16, true, 4>(uint32_t, const uint32_t *const *,
                                                         const uint32_t *const *);
template uint32_t integer_compare_x86_64_v4<16, true, 5>(uint32_t, const uint32_t *const *,
                                                         const uint32_t *const *);
template uint32_t integer_compare_x86_64_v4<16, true, 6>(uint32_t, const uint32_t *const *,
                                                         const uint32_t *const *);
template uint32_t integer_compare_x86_64_v4<16, false, 1>(uint32_t, const uint32_t *const *,
                                                          const uint32_t *const *);
template uint32_t integer_compare_x86_64_v4<16, false, 2>(uint32_t, const uint32_t *const *,
                                                          const uint32_t *const *);
template uint32_t integer_compare_x86_64_v4<16, false, 3>(uint32_t, const uint32_t *const *,
                                                          const uint32_t *const *);
template uint32_t integer_compare_x86_64_v4<16, false, 4>(uint32_t, const uint32_t *const *,
                                                          const uint32_t *const *);
template uint32_t integer_compare_x86_64_v4<16, false, 5>(uint32_t, const uint32_t *const *,
                                                          const uint32_t *const *);
template uint32_t integer_compare_x86_64_v4<16, false, 6>(uint32_t, const uint32_t *const *,
                                                          const uint32_t *const *);
template uint32_t integer_compare_x86_64_v4<32, true, 1>(uint32_t, const uint32_t *const *,
                                                         const uint32_t *const *);
template uint32_t integer_compare_x86_64_v4<32, true, 2>(uint32_t, const uint32_t *const *,
                                                         const uint32_t *const *);
template uint32_t integer_compare_x86_64_v4<32, true, 3>(uint32_t, const uint32_t *const *,
                                                         const uint32_t *const *);
template uint32_t integer_compare_x86_64_v4<32, true, 4>(uint32_t, const uint32_t *const *,
                                                         const uint32_t *const *);
template uint32_t integer_compare_x86_64_v4<32, true, 5>(uint32_t, const uint32_t *const *,
                                                         const uint32_t *const *);
template uint32_t integer_compare_x86_64_v4<32, true, 6>(uint32_t, const uint32_t *const *,
                                                         const uint32_t *const *);
template uint32_t integer_compare_x86_64_v4<32, false, 1>(uint32_t, const uint32_t *const *,
                                                          const uint32_t *const *);
template uint32_t integer_compare_x86_64_v4<32, false, 2>(uint32_t, const uint32_t *const *,
                                                          const uint32_t *const *);
template uint32_t integer_compare_x86_64_v4<32, false, 3>(uint32_t, const uint32_t *const *,
                                                          const uint32_t *const *);
template uint32_t integer_compare_x86_64_v4<32, false, 4>(uint32_t, const uint32_t *const *,
                                                          const uint32_t *const *);
template uint32_t integer_compare_x86_64_v4<32, false, 5>(uint32_t, const uint32_t *const *,
                                                          const uint32_t *const *);
template uint32_t integer_compare_x86_64_v4<32, false, 6>(uint32_t, const uint32_t *const *,
                                                          const uint32_t *const *);
template uint32_t integer_compare_x86_64_v4<64, true, 1>(uint32_t, const uint32_t *const *,
                                                         const uint32_t *const *);
template uint32_t integer_compare_x86_64_v4<64, true, 2>(uint32_t, const uint32_t *const *,
                                                         const uint32_t *const *);
template uint32_t integer_compare_x86_64_v4<64, true, 3>(uint32_t, const uint32_t *const *,
                                                         const uint32_t *const *);
template uint32_t integer_compare_x86_64_v4<64, true, 4>(uint32_t, const uint32_t *const *,
                                                         const uint32_t *const *);
template uint32_t integer_compare_x86_64_v4<64, true, 5>(uint32_t, const uint32_t *const *,
                                                         const uint32_t *const *);
template uint32_t integer_compare_x86_64_v4<64, true, 6>(uint32_t, const uint32_t *const *,
                                                         const uint32_t *const *);
template uint32_t integer_compare_x86_64_v4<64, false, 1>(uint32_t, const uint32_t *const *,
                                                          const uint32_t *const *);
template uint32_t integer_compare_x86_64_v4<64, false, 2>(uint32_t, const uint32_t *const *,
                                                          const uint32_t *const *);
template uint32_t integer_compare_x86_64_v4<64, false, 3>(uint32_t, const uint32_t *const *,
                                                          const uint32_t *const *);
template uint32_t integer_compare_x86_64_v4<64, false, 4>(uint32_t, const uint32_t *const *,
                                                          const uint32_t *const *);
template uint32_t integer_compare_x86_64_v4<64, false, 5>(uint32_t, const uint32_t *const *,
                                                          const uint32_t *const *);
template uint32_t integer_compare_x86_64_v4<64, false, 6>(uint32_t, const uint32_t *const *,
                                                          const uint32_t *const *);

} // namespace rocjitsu::arith
