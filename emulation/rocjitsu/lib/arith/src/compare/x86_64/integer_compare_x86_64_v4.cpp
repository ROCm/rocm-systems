// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 signed and unsigned integer comparisons producing lane masks.
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
uint64_t integer_compare_x86_64_v4(int lanes, uint32_t mode, const uint32_t *const *a,
                                   const uint32_t *const *b) {
  constexpr int pred = Predicate == CompareLess        ? _MM_CMPINT_LT
                       : Predicate == CompareEqual     ? _MM_CMPINT_EQ
                       : Predicate == CompareLessEqual ? _MM_CMPINT_LE
                       : Predicate == CompareGreater   ? _MM_CMPINT_GT
                       : Predicate == CompareNotEqual  ? _MM_CMPINT_NE
                                                       : _MM_CMPINT_GE;
  auto compare_half = [&](int base) {
    uint32_t result = 0;
    for (int lane = 0; lane < 32; lane += 16) {
      auto x = _mm512_loadu_si512(a[Bits == 64 ? 1 : 0] + base + lane);
      auto y = _mm512_loadu_si512(b[Bits == 64 ? 1 : 0] + base + lane);
      if constexpr (Bits == 16) {
        auto extend = [](__m512i bits, bool high) {
          auto top = high ? bits : _mm512_slli_epi32(bits, 16);
          return Signed ? _mm512_srai_epi32(top, 16) : _mm512_srli_epi32(top, 16);
        };
        x = extend(x, mode & RJ_ARITH_ALU_HIGH_A);
        y = extend(y, mode & RJ_ARITH_ALU_HIGH_B);
      }
      __mmask16 selected;
      if constexpr (Bits != 64) {
        selected = Signed ? _mm512_cmp_epi32_mask(x, y, pred) : _mm512_cmp_epu32_mask(x, y, pred);
      } else {
        auto lo_x = _mm512_loadu_si512(a[0] + base + lane),
             lo_y = _mm512_loadu_si512(b[0] + base + lane);
        if constexpr (pred == _MM_CMPINT_EQ || pred == _MM_CMPINT_NE) {
          auto equal = _mm512_mask_cmpeq_epi32_mask(_mm512_cmpeq_epi32_mask(x, y), lo_x, lo_y);
          selected = pred == _MM_CMPINT_EQ ? equal : __mmask16(~equal);
        } else {
          constexpr int strict =
              pred == _MM_CMPINT_LE || pred == _MM_CMPINT_LT ? _MM_CMPINT_LT : _MM_CMPINT_GT;
          selected =
              Signed ? _mm512_cmp_epi32_mask(x, y, strict) : _mm512_cmp_epu32_mask(x, y, strict);
          selected |= _mm512_mask_cmp_epu32_mask(_mm512_cmpeq_epi32_mask(x, y), lo_x, lo_y, pred);
        }
      }
      result |= uint32_t(selected) << lane;
    }
    return result;
  };
  uint64_t result = compare_half(0);
  if (lanes == 64) {
    result |= uint64_t(compare_half(32)) << 32;
  }
  return result;
}

template uint64_t integer_compare_x86_64_v4<16, true, rocjitsu::arith::CompareLess>(
    int, uint32_t, const uint32_t *const *, const uint32_t *const *);
template uint64_t integer_compare_x86_64_v4<16, true, rocjitsu::arith::CompareEqual>(
    int, uint32_t, const uint32_t *const *, const uint32_t *const *);
template uint64_t integer_compare_x86_64_v4<16, true, rocjitsu::arith::CompareLessEqual>(
    int, uint32_t, const uint32_t *const *, const uint32_t *const *);
template uint64_t integer_compare_x86_64_v4<16, true, rocjitsu::arith::CompareGreater>(
    int, uint32_t, const uint32_t *const *, const uint32_t *const *);
template uint64_t integer_compare_x86_64_v4<16, true, rocjitsu::arith::CompareNotEqual>(
    int, uint32_t, const uint32_t *const *, const uint32_t *const *);
template uint64_t integer_compare_x86_64_v4<16, true, rocjitsu::arith::CompareGreaterEqual>(
    int, uint32_t, const uint32_t *const *, const uint32_t *const *);
template uint64_t integer_compare_x86_64_v4<16, false, rocjitsu::arith::CompareLess>(
    int, uint32_t, const uint32_t *const *, const uint32_t *const *);
template uint64_t integer_compare_x86_64_v4<16, false, rocjitsu::arith::CompareEqual>(
    int, uint32_t, const uint32_t *const *, const uint32_t *const *);
template uint64_t integer_compare_x86_64_v4<16, false, rocjitsu::arith::CompareLessEqual>(
    int, uint32_t, const uint32_t *const *, const uint32_t *const *);
template uint64_t integer_compare_x86_64_v4<16, false, rocjitsu::arith::CompareGreater>(
    int, uint32_t, const uint32_t *const *, const uint32_t *const *);
template uint64_t integer_compare_x86_64_v4<16, false, rocjitsu::arith::CompareNotEqual>(
    int, uint32_t, const uint32_t *const *, const uint32_t *const *);
template uint64_t integer_compare_x86_64_v4<16, false, rocjitsu::arith::CompareGreaterEqual>(
    int, uint32_t, const uint32_t *const *, const uint32_t *const *);
template uint64_t integer_compare_x86_64_v4<32, true, rocjitsu::arith::CompareLess>(
    int, uint32_t, const uint32_t *const *, const uint32_t *const *);
template uint64_t integer_compare_x86_64_v4<32, true, rocjitsu::arith::CompareEqual>(
    int, uint32_t, const uint32_t *const *, const uint32_t *const *);
template uint64_t integer_compare_x86_64_v4<32, true, rocjitsu::arith::CompareLessEqual>(
    int, uint32_t, const uint32_t *const *, const uint32_t *const *);
template uint64_t integer_compare_x86_64_v4<32, true, rocjitsu::arith::CompareGreater>(
    int, uint32_t, const uint32_t *const *, const uint32_t *const *);
template uint64_t integer_compare_x86_64_v4<32, true, rocjitsu::arith::CompareNotEqual>(
    int, uint32_t, const uint32_t *const *, const uint32_t *const *);
template uint64_t integer_compare_x86_64_v4<32, true, rocjitsu::arith::CompareGreaterEqual>(
    int, uint32_t, const uint32_t *const *, const uint32_t *const *);
template uint64_t integer_compare_x86_64_v4<32, false, rocjitsu::arith::CompareLess>(
    int, uint32_t, const uint32_t *const *, const uint32_t *const *);
template uint64_t integer_compare_x86_64_v4<32, false, rocjitsu::arith::CompareEqual>(
    int, uint32_t, const uint32_t *const *, const uint32_t *const *);
template uint64_t integer_compare_x86_64_v4<32, false, rocjitsu::arith::CompareLessEqual>(
    int, uint32_t, const uint32_t *const *, const uint32_t *const *);
template uint64_t integer_compare_x86_64_v4<32, false, rocjitsu::arith::CompareGreater>(
    int, uint32_t, const uint32_t *const *, const uint32_t *const *);
template uint64_t integer_compare_x86_64_v4<32, false, rocjitsu::arith::CompareNotEqual>(
    int, uint32_t, const uint32_t *const *, const uint32_t *const *);
template uint64_t integer_compare_x86_64_v4<32, false, rocjitsu::arith::CompareGreaterEqual>(
    int, uint32_t, const uint32_t *const *, const uint32_t *const *);
template uint64_t integer_compare_x86_64_v4<64, true, rocjitsu::arith::CompareLess>(
    int, uint32_t, const uint32_t *const *, const uint32_t *const *);
template uint64_t integer_compare_x86_64_v4<64, true, rocjitsu::arith::CompareEqual>(
    int, uint32_t, const uint32_t *const *, const uint32_t *const *);
template uint64_t integer_compare_x86_64_v4<64, true, rocjitsu::arith::CompareLessEqual>(
    int, uint32_t, const uint32_t *const *, const uint32_t *const *);
template uint64_t integer_compare_x86_64_v4<64, true, rocjitsu::arith::CompareGreater>(
    int, uint32_t, const uint32_t *const *, const uint32_t *const *);
template uint64_t integer_compare_x86_64_v4<64, true, rocjitsu::arith::CompareNotEqual>(
    int, uint32_t, const uint32_t *const *, const uint32_t *const *);
template uint64_t integer_compare_x86_64_v4<64, true, rocjitsu::arith::CompareGreaterEqual>(
    int, uint32_t, const uint32_t *const *, const uint32_t *const *);
template uint64_t integer_compare_x86_64_v4<64, false, rocjitsu::arith::CompareLess>(
    int, uint32_t, const uint32_t *const *, const uint32_t *const *);
template uint64_t integer_compare_x86_64_v4<64, false, rocjitsu::arith::CompareEqual>(
    int, uint32_t, const uint32_t *const *, const uint32_t *const *);
template uint64_t integer_compare_x86_64_v4<64, false, rocjitsu::arith::CompareLessEqual>(
    int, uint32_t, const uint32_t *const *, const uint32_t *const *);
template uint64_t integer_compare_x86_64_v4<64, false, rocjitsu::arith::CompareGreater>(
    int, uint32_t, const uint32_t *const *, const uint32_t *const *);
template uint64_t integer_compare_x86_64_v4<64, false, rocjitsu::arith::CompareNotEqual>(
    int, uint32_t, const uint32_t *const *, const uint32_t *const *);
template uint64_t integer_compare_x86_64_v4<64, false, rocjitsu::arith::CompareGreaterEqual>(
    int, uint32_t, const uint32_t *const *, const uint32_t *const *);

} // namespace rocjitsu::arith
