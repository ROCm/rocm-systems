// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 signed and unsigned integer comparisons producing lane masks.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "compare/integer_compare.h"
#include "compare/predicate.h"
#include "rj_arith/detail/rj_arith_compare.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {

// Compare signed/unsigned integers.
// Requires the named CPU features and validated instruction controls.
template <int Bits, bool Signed, unsigned Predicate>
uint32_t integer_compare_x86_64_v3(uint32_t mode, const uint32_t *const *a,
                                   const uint32_t *const *b) {
  __m256i results[4];
  for (int lane = 0; lane < 32; lane += 8) {
    auto av = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a[Bits == 64 ? 1 : 0] + lane));
    auto bv = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b[Bits == 64 ? 1 : 0] + lane));
    if constexpr (Bits == 16) {
      av = _mm256_and_si256(
          _mm256_srl_epi32(av, _mm_cvtsi32_si128(mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0)),
          _mm256_set1_epi32(65535));
      bv = _mm256_and_si256(
          _mm256_srl_epi32(bv, _mm_cvtsi32_si128(mode & RJ_ARITH_ALU_HIGH_B ? 16 : 0)),
          _mm256_set1_epi32(65535));
    }
    constexpr uint32_t flip = Bits == 16 ? (Signed ? 0x8000 : 0) : (Signed ? 0 : 0x80000000);
    av = _mm256_xor_si256(av, _mm256_set1_epi32(int(flip)));
    bv = _mm256_xor_si256(bv, _mm256_set1_epi32(int(flip)));
    auto lt = _mm256_cmpgt_epi32(bv, av), eq = _mm256_cmpeq_epi32(av, bv);
    if constexpr (Bits == 64) {
      auto al = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a[0] + lane));
      auto bl = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b[0] + lane));
      auto low_lt = _mm256_cmpgt_epi32(_mm256_xor_si256(bl, _mm256_set1_epi32(int(0x80000000u))),
                                       _mm256_xor_si256(al, _mm256_set1_epi32(int(0x80000000u))));
      lt = _mm256_or_si256(lt, _mm256_and_si256(eq, low_lt));
      eq = _mm256_and_si256(eq, _mm256_cmpeq_epi32(al, bl));
    }
    __m256i bits;
    if constexpr (Predicate == CompareLess || Predicate == CompareGreaterEqual) {
      bits = lt;
    } else if constexpr (Predicate == CompareEqual || Predicate == CompareNotEqual) {
      bits = eq;
    } else {
      bits = _mm256_or_si256(lt, eq);
    }
    results[lane / 8] = bits;
  }
  auto a01 = _mm256_packs_epi32(results[0], results[1]),
       a23 = _mm256_packs_epi32(results[2], results[3]);
  auto bytes = _mm256_packs_epi16(a01, a23);
  bytes = _mm256_permutevar8x32_epi32(bytes, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
  auto mask = uint32_t(_mm256_movemask_epi8(bytes));
  if constexpr (Predicate == CompareGreaterEqual || Predicate == CompareNotEqual ||
                Predicate == CompareGreater) {
    return ~mask;
  }
  return mask;
}

template uint32_t
integer_compare_x86_64_v3<16, true, rocjitsu::arith::CompareLess>(uint32_t, const uint32_t *const *,
                                                                  const uint32_t *const *);
template uint32_t integer_compare_x86_64_v3<16, true, rocjitsu::arith::CompareEqual>(
    uint32_t, const uint32_t *const *, const uint32_t *const *);
template uint32_t integer_compare_x86_64_v3<16, true, rocjitsu::arith::CompareLessEqual>(
    uint32_t, const uint32_t *const *, const uint32_t *const *);
template uint32_t integer_compare_x86_64_v3<16, true, rocjitsu::arith::CompareGreater>(
    uint32_t, const uint32_t *const *, const uint32_t *const *);
template uint32_t integer_compare_x86_64_v3<16, true, rocjitsu::arith::CompareNotEqual>(
    uint32_t, const uint32_t *const *, const uint32_t *const *);
template uint32_t integer_compare_x86_64_v3<16, true, rocjitsu::arith::CompareGreaterEqual>(
    uint32_t, const uint32_t *const *, const uint32_t *const *);
template uint32_t integer_compare_x86_64_v3<16, false, rocjitsu::arith::CompareLess>(
    uint32_t, const uint32_t *const *, const uint32_t *const *);
template uint32_t integer_compare_x86_64_v3<16, false, rocjitsu::arith::CompareEqual>(
    uint32_t, const uint32_t *const *, const uint32_t *const *);
template uint32_t integer_compare_x86_64_v3<16, false, rocjitsu::arith::CompareLessEqual>(
    uint32_t, const uint32_t *const *, const uint32_t *const *);
template uint32_t integer_compare_x86_64_v3<16, false, rocjitsu::arith::CompareGreater>(
    uint32_t, const uint32_t *const *, const uint32_t *const *);
template uint32_t integer_compare_x86_64_v3<16, false, rocjitsu::arith::CompareNotEqual>(
    uint32_t, const uint32_t *const *, const uint32_t *const *);
template uint32_t integer_compare_x86_64_v3<16, false, rocjitsu::arith::CompareGreaterEqual>(
    uint32_t, const uint32_t *const *, const uint32_t *const *);
template uint32_t
integer_compare_x86_64_v3<32, true, rocjitsu::arith::CompareLess>(uint32_t, const uint32_t *const *,
                                                                  const uint32_t *const *);
template uint32_t integer_compare_x86_64_v3<32, true, rocjitsu::arith::CompareEqual>(
    uint32_t, const uint32_t *const *, const uint32_t *const *);
template uint32_t integer_compare_x86_64_v3<32, true, rocjitsu::arith::CompareLessEqual>(
    uint32_t, const uint32_t *const *, const uint32_t *const *);
template uint32_t integer_compare_x86_64_v3<32, true, rocjitsu::arith::CompareGreater>(
    uint32_t, const uint32_t *const *, const uint32_t *const *);
template uint32_t integer_compare_x86_64_v3<32, true, rocjitsu::arith::CompareNotEqual>(
    uint32_t, const uint32_t *const *, const uint32_t *const *);
template uint32_t integer_compare_x86_64_v3<32, true, rocjitsu::arith::CompareGreaterEqual>(
    uint32_t, const uint32_t *const *, const uint32_t *const *);
template uint32_t integer_compare_x86_64_v3<32, false, rocjitsu::arith::CompareLess>(
    uint32_t, const uint32_t *const *, const uint32_t *const *);
template uint32_t integer_compare_x86_64_v3<32, false, rocjitsu::arith::CompareEqual>(
    uint32_t, const uint32_t *const *, const uint32_t *const *);
template uint32_t integer_compare_x86_64_v3<32, false, rocjitsu::arith::CompareLessEqual>(
    uint32_t, const uint32_t *const *, const uint32_t *const *);
template uint32_t integer_compare_x86_64_v3<32, false, rocjitsu::arith::CompareGreater>(
    uint32_t, const uint32_t *const *, const uint32_t *const *);
template uint32_t integer_compare_x86_64_v3<32, false, rocjitsu::arith::CompareNotEqual>(
    uint32_t, const uint32_t *const *, const uint32_t *const *);
template uint32_t integer_compare_x86_64_v3<32, false, rocjitsu::arith::CompareGreaterEqual>(
    uint32_t, const uint32_t *const *, const uint32_t *const *);
template uint32_t
integer_compare_x86_64_v3<64, true, rocjitsu::arith::CompareLess>(uint32_t, const uint32_t *const *,
                                                                  const uint32_t *const *);
template uint32_t integer_compare_x86_64_v3<64, true, rocjitsu::arith::CompareEqual>(
    uint32_t, const uint32_t *const *, const uint32_t *const *);
template uint32_t integer_compare_x86_64_v3<64, true, rocjitsu::arith::CompareLessEqual>(
    uint32_t, const uint32_t *const *, const uint32_t *const *);
template uint32_t integer_compare_x86_64_v3<64, true, rocjitsu::arith::CompareGreater>(
    uint32_t, const uint32_t *const *, const uint32_t *const *);
template uint32_t integer_compare_x86_64_v3<64, true, rocjitsu::arith::CompareNotEqual>(
    uint32_t, const uint32_t *const *, const uint32_t *const *);
template uint32_t integer_compare_x86_64_v3<64, true, rocjitsu::arith::CompareGreaterEqual>(
    uint32_t, const uint32_t *const *, const uint32_t *const *);
template uint32_t integer_compare_x86_64_v3<64, false, rocjitsu::arith::CompareLess>(
    uint32_t, const uint32_t *const *, const uint32_t *const *);
template uint32_t integer_compare_x86_64_v3<64, false, rocjitsu::arith::CompareEqual>(
    uint32_t, const uint32_t *const *, const uint32_t *const *);
template uint32_t integer_compare_x86_64_v3<64, false, rocjitsu::arith::CompareLessEqual>(
    uint32_t, const uint32_t *const *, const uint32_t *const *);
template uint32_t integer_compare_x86_64_v3<64, false, rocjitsu::arith::CompareGreater>(
    uint32_t, const uint32_t *const *, const uint32_t *const *);
template uint32_t integer_compare_x86_64_v3<64, false, rocjitsu::arith::CompareNotEqual>(
    uint32_t, const uint32_t *const *, const uint32_t *const *);
template uint32_t integer_compare_x86_64_v3<64, false, rocjitsu::arith::CompareGreaterEqual>(
    uint32_t, const uint32_t *const *, const uint32_t *const *);

} // namespace rocjitsu::arith
