// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Vector floating-point predicates, returning one comparison bit per lane.
// The caller applies EXEC and handles FP exception reporting.

#include "compare/float_compare.h"
#include "rj_arith/rj_arith.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {
namespace {

template <int Bits> class Lanes {
public:
  using V = __m256i;
  using M = __m256i;

  // Broadcast the supplied bit pattern to each vector lane.
  static V set(uint64_t x) {
    if constexpr (Bits == 64) {
      return _mm256_set1_epi64x(int64_t(x));
    } else {
      return _mm256_set1_epi32(int(x));
    }
  }

  // Load source values in the vector layout expected by this kernel.
  static V load(const uint32_t *const *p, int lane) {
    if constexpr (Bits == 64) {
      auto low =
          _mm256_cvtepu32_epi64(_mm_loadu_si128(reinterpret_cast<const __m128i *>(p[0] + lane)));
      auto high =
          _mm256_cvtepu32_epi64(_mm_loadu_si128(reinterpret_cast<const __m128i *>(p[1] + lane)));
      return _mm256_or_si256(low, _mm256_slli_epi64(high, 32));
    } else {
      return _mm256_loadu_si256(reinterpret_cast<const V *>(p[0] + lane));
    }
  }

  // Return the bitwise intersection of the vector operands.
  static V band(V a, V b) { return _mm256_and_si256(a, b); }

  // Return the bitwise union of the vector operands.
  static V bor(V a, V b) { return _mm256_or_si256(a, b); }

  // Return the bitwise exclusive-or of the vector operands.
  static V bxor(V a, V b) { return _mm256_xor_si256(a, b); }

  // Return the lane mask for equal operands.
  static M eq(V a, V b) {
    if constexpr (Bits == 64) {
      return _mm256_cmpeq_epi64(a, b);
    } else {
      return _mm256_cmpeq_epi32(a, b);
    }
  }

  // Return the lane mask for signed a > b.
  static M gt(V a, V b) {
    if constexpr (Bits == 64) {
      return _mm256_cmpgt_epi64(a, b);
    } else {
      return _mm256_cmpgt_epi32(a, b);
    }
  }

  // Select values lane by lane according to the supplied predicate.
  static V select(M mask, V no, V yes) { return _mm256_blendv_epi8(no, yes, mask); }

  // Collect per-lane predicates into a scalar bit mask.
  static uint32_t mask(M m) {
    if constexpr (Bits == 64) {
      return uint32_t(_mm256_movemask_pd(_mm256_castsi256_pd(m)));
    } else {
      return uint32_t(_mm256_movemask_ps(_mm256_castsi256_ps(m)));
    }
  }

  // Return the lane mask for unsigned a < b.
  static M lt_unsigned(V a, V b) {
    auto flip = set(1ULL << (Bits == 64 ? 63 : 31));
    return gt(bxor(b, flip), bxor(a, flip));
  }
};

} // namespace

// Evaluate ordered/unordered FP predicates.
// Requires the named CPU features and validated instruction controls.
template <int Bits, unsigned Predicate>
uint32_t float_compare_x86_64_v3(uint32_t mode, bool flush, const uint32_t *const *a,
                                 const uint32_t *const *b) {
  using L = Lanes<Bits>;
  constexpr int lanes = 256 / (Bits == 64 ? 64 : 32);
  constexpr uint64_t sign = 1ULL << (Bits - 1), magnitude = sign - 1;
  constexpr uint64_t infinity = Bits == 16   ? 0x7c00
                                : Bits == 32 ? 0x7f800000
                                             : 0x7ff0000000000000ULL;
  auto zero = L::set(0), inf = L::set(infinity), mag = L::set(magnitude), sign_bit = L::set(sign);
  uint32_t less = 0, equal = 0, unordered = 0;
  for (int lane = 0; lane < 32; lane += lanes) {
    auto av = L::load(a, lane), bv = L::load(b, lane);
    if constexpr (Bits == 16) {
      av = L::band(_mm256_srl_epi32(av, _mm_cvtsi32_si128(mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0)),
                   L::set(65535));
      bv = L::band(_mm256_srl_epi32(bv, _mm_cvtsi32_si128(mode & RJ_ARITH_ALU_HIGH_B ? 16 : 0)),
                   L::set(65535));
    }
    av = L::band(av, L::set(mode & RJ_ARITH_ALU_ABS_A ? magnitude : UINT64_MAX));
    bv = L::band(bv, L::set(mode & RJ_ARITH_ALU_ABS_B ? magnitude : UINT64_MAX));
    av = L::bxor(av, L::set(mode & RJ_ARITH_ALU_NEG_A ? sign : 0));
    bv = L::bxor(bv, L::set(mode & RJ_ARITH_ALU_NEG_B ? sign : 0));
    if (flush) {
      av = L::select(L::eq(L::band(av, inf), zero), av, L::band(av, sign_bit));
      bv = L::select(L::eq(L::band(bv, inf), zero), bv, L::band(bv, sign_bit));
    }
    unordered |= (L::mask(L::gt(L::band(av, mag), inf)) | L::mask(L::gt(L::band(bv, mag), inf)))
                 << lane;
    av = L::select(L::eq(L::band(av, mag), zero), av, zero);
    bv = L::select(L::eq(L::band(bv, mag), zero), bv, zero);
    av = L::bxor(
        av, L::select(L::eq(L::band(av, sign_bit), sign_bit), sign_bit, L::set(sign | magnitude)));
    bv = L::bxor(
        bv, L::select(L::eq(L::band(bv, sign_bit), sign_bit), sign_bit, L::set(sign | magnitude)));
    less |= L::mask(L::lt_unsigned(av, bv)) << lane;
    equal |= L::mask(L::eq(av, bv)) << lane;
  }
  return float_compare_result<Predicate>(less, equal, ~unordered);
}

template uint32_t float_compare_x86_64_v3<16, 1>(uint32_t, bool, const uint32_t *const *,
                                                 const uint32_t *const *);
template uint32_t float_compare_x86_64_v3<16, 2>(uint32_t, bool, const uint32_t *const *,
                                                 const uint32_t *const *);
template uint32_t float_compare_x86_64_v3<16, 3>(uint32_t, bool, const uint32_t *const *,
                                                 const uint32_t *const *);
template uint32_t float_compare_x86_64_v3<16, 4>(uint32_t, bool, const uint32_t *const *,
                                                 const uint32_t *const *);
template uint32_t float_compare_x86_64_v3<16, 5>(uint32_t, bool, const uint32_t *const *,
                                                 const uint32_t *const *);
template uint32_t float_compare_x86_64_v3<16, 6>(uint32_t, bool, const uint32_t *const *,
                                                 const uint32_t *const *);
template uint32_t float_compare_x86_64_v3<16, 7>(uint32_t, bool, const uint32_t *const *,
                                                 const uint32_t *const *);
template uint32_t float_compare_x86_64_v3<16, 8>(uint32_t, bool, const uint32_t *const *,
                                                 const uint32_t *const *);
template uint32_t float_compare_x86_64_v3<16, 9>(uint32_t, bool, const uint32_t *const *,
                                                 const uint32_t *const *);
template uint32_t float_compare_x86_64_v3<16, 10>(uint32_t, bool, const uint32_t *const *,
                                                  const uint32_t *const *);
template uint32_t float_compare_x86_64_v3<16, 11>(uint32_t, bool, const uint32_t *const *,
                                                  const uint32_t *const *);
template uint32_t float_compare_x86_64_v3<16, 12>(uint32_t, bool, const uint32_t *const *,
                                                  const uint32_t *const *);
template uint32_t float_compare_x86_64_v3<16, 13>(uint32_t, bool, const uint32_t *const *,
                                                  const uint32_t *const *);
template uint32_t float_compare_x86_64_v3<16, 14>(uint32_t, bool, const uint32_t *const *,
                                                  const uint32_t *const *);
template uint32_t float_compare_x86_64_v3<32, 1>(uint32_t, bool, const uint32_t *const *,
                                                 const uint32_t *const *);
template uint32_t float_compare_x86_64_v3<32, 2>(uint32_t, bool, const uint32_t *const *,
                                                 const uint32_t *const *);
template uint32_t float_compare_x86_64_v3<32, 3>(uint32_t, bool, const uint32_t *const *,
                                                 const uint32_t *const *);
template uint32_t float_compare_x86_64_v3<32, 4>(uint32_t, bool, const uint32_t *const *,
                                                 const uint32_t *const *);
template uint32_t float_compare_x86_64_v3<32, 5>(uint32_t, bool, const uint32_t *const *,
                                                 const uint32_t *const *);
template uint32_t float_compare_x86_64_v3<32, 6>(uint32_t, bool, const uint32_t *const *,
                                                 const uint32_t *const *);
template uint32_t float_compare_x86_64_v3<32, 7>(uint32_t, bool, const uint32_t *const *,
                                                 const uint32_t *const *);
template uint32_t float_compare_x86_64_v3<32, 8>(uint32_t, bool, const uint32_t *const *,
                                                 const uint32_t *const *);
template uint32_t float_compare_x86_64_v3<32, 9>(uint32_t, bool, const uint32_t *const *,
                                                 const uint32_t *const *);
template uint32_t float_compare_x86_64_v3<32, 10>(uint32_t, bool, const uint32_t *const *,
                                                  const uint32_t *const *);
template uint32_t float_compare_x86_64_v3<32, 11>(uint32_t, bool, const uint32_t *const *,
                                                  const uint32_t *const *);
template uint32_t float_compare_x86_64_v3<32, 12>(uint32_t, bool, const uint32_t *const *,
                                                  const uint32_t *const *);
template uint32_t float_compare_x86_64_v3<32, 13>(uint32_t, bool, const uint32_t *const *,
                                                  const uint32_t *const *);
template uint32_t float_compare_x86_64_v3<32, 14>(uint32_t, bool, const uint32_t *const *,
                                                  const uint32_t *const *);
template uint32_t float_compare_x86_64_v3<64, 1>(uint32_t, bool, const uint32_t *const *,
                                                 const uint32_t *const *);
template uint32_t float_compare_x86_64_v3<64, 2>(uint32_t, bool, const uint32_t *const *,
                                                 const uint32_t *const *);
template uint32_t float_compare_x86_64_v3<64, 3>(uint32_t, bool, const uint32_t *const *,
                                                 const uint32_t *const *);
template uint32_t float_compare_x86_64_v3<64, 4>(uint32_t, bool, const uint32_t *const *,
                                                 const uint32_t *const *);
template uint32_t float_compare_x86_64_v3<64, 5>(uint32_t, bool, const uint32_t *const *,
                                                 const uint32_t *const *);
template uint32_t float_compare_x86_64_v3<64, 6>(uint32_t, bool, const uint32_t *const *,
                                                 const uint32_t *const *);
template uint32_t float_compare_x86_64_v3<64, 7>(uint32_t, bool, const uint32_t *const *,
                                                 const uint32_t *const *);
template uint32_t float_compare_x86_64_v3<64, 8>(uint32_t, bool, const uint32_t *const *,
                                                 const uint32_t *const *);
template uint32_t float_compare_x86_64_v3<64, 9>(uint32_t, bool, const uint32_t *const *,
                                                 const uint32_t *const *);
template uint32_t float_compare_x86_64_v3<64, 10>(uint32_t, bool, const uint32_t *const *,
                                                  const uint32_t *const *);
template uint32_t float_compare_x86_64_v3<64, 11>(uint32_t, bool, const uint32_t *const *,
                                                  const uint32_t *const *);
template uint32_t float_compare_x86_64_v3<64, 12>(uint32_t, bool, const uint32_t *const *,
                                                  const uint32_t *const *);
template uint32_t float_compare_x86_64_v3<64, 13>(uint32_t, bool, const uint32_t *const *,
                                                  const uint32_t *const *);
template uint32_t float_compare_x86_64_v3<64, 14>(uint32_t, bool, const uint32_t *const *,
                                                  const uint32_t *const *);

} // namespace rocjitsu::arith
