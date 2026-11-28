// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Vector floating-point predicates, returning one comparison bit per lane.
// The caller applies EXEC and handles FP exception reporting.

#include "common/compiler.h"
#include "common/isa.h"
#include "compare/float_compare.h"
#include "compare/predicate.h"
#include "compare/x86_64/predicate.h"
#include "rj_arith/detail/rj_arith_common.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {
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

  // Compare numeric values, widening FP16 without changing their values.
  template <unsigned Predicate> static uint32_t compare(V a, V b) {
    constexpr int immediate = float_compare_immediate<Predicate>();
    if constexpr (Bits == 64) {
      return uint32_t(_mm256_movemask_pd(
          _mm256_cmp_pd(_mm256_castsi256_pd(a), _mm256_castsi256_pd(b), immediate)));
    } else {
      if constexpr (Bits == 16) {
        auto ah = _mm_packus_epi32(_mm256_castsi256_si128(a), _mm256_extracti128_si256(a, 1));
        auto bh = _mm_packus_epi32(_mm256_castsi256_si128(b), _mm256_extracti128_si256(b, 1));
        return uint32_t(
            _mm256_movemask_ps(_mm256_cmp_ps(_mm256_cvtph_ps(ah), _mm256_cvtph_ps(bh), immediate)));
      } else {
        return uint32_t(_mm256_movemask_ps(
            _mm256_cmp_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b), immediate)));
      }
    }
  }
};

} // namespace
} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE

namespace rocjitsu::arith {

// Evaluate a full wave; the result-mask width selects 32 or 64 lanes.
// Requires the named CPU features and validated instruction controls.
template <int Bits, unsigned Predicate, typename Mask>
Mask float_compare_x86_64_v3(uint32_t mode, bool flush, const uint32_t *const *a,
                             const uint32_t *const *b) {
  using L = RJ_ARITH_IMPL_NAMESPACE::Lanes<Bits>;
  constexpr int lanes = 256 / (Bits == 64 ? 64 : 32);
  constexpr uint64_t sign = 1ULL << (Bits - 1), magnitude = sign - 1;
  constexpr uint64_t infinity = Bits == 16   ? 0x7c00
                                : Bits == 32 ? 0x7f800000
                                             : 0x7ff0000000000000ULL;
  auto zero = L::set(0), inf = L::set(infinity), mag = L::set(magnitude), sign_bit = L::set(sign);
  static_assert(sizeof(Mask) == 4 || sizeof(Mask) == 8);
  Mask result = 0;
  RJ_ARITH_UNROLL(16)
  for (int lane = 0; lane < int(sizeof(Mask) * 8); lane += lanes) {
    auto av = L::load(a, lane), bv = L::load(b, lane);
    if constexpr (Bits == 16) {
      av = L::band(_mm256_srl_epi32(av, _mm_cvtsi32_si128(mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0)),
                   L::set(65535));
      bv = L::band(_mm256_srl_epi32(bv, _mm_cvtsi32_si128(mode & RJ_ARITH_ALU_HIGH_B ? 16 : 0)),
                   L::set(65535));
    }
    if constexpr (Predicate == CompareOrdered || Predicate == CompareUnordered) {
      // Classify raw encodings: sign and denormal controls cannot change NaN-ness.
      result |= Mask(L::mask(L::gt(L::band(av, mag), inf)) | L::mask(L::gt(L::band(bv, mag), inf)))
                << lane;
    } else {
      av = L::band(av, L::set(mode & RJ_ARITH_ALU_ABS_A ? magnitude : UINT64_MAX));
      bv = L::band(bv, L::set(mode & RJ_ARITH_ALU_ABS_B ? magnitude : UINT64_MAX));
      av = L::bxor(av, L::set(mode & RJ_ARITH_ALU_NEG_A ? sign : 0));
      bv = L::bxor(bv, L::set(mode & RJ_ARITH_ALU_NEG_B ? sign : 0));
      if (flush) {
        av = L::select(L::eq(L::band(av, inf), zero), av, L::band(av, sign_bit));
        bv = L::select(L::eq(L::band(bv, inf), zero), bv, L::band(bv, sign_bit));
      }
      result |= Mask(L::template compare<Predicate>(av, bv)) << lane;
    }
  }
  if constexpr (Predicate == CompareOrdered) {
    return ~result;
  }
  return result;
}

template uint32_t
float_compare_x86_64_v3<16, rocjitsu::arith::CompareLess>(uint32_t, bool, const uint32_t *const *,
                                                          const uint32_t *const *);
template uint32_t
float_compare_x86_64_v3<16, rocjitsu::arith::CompareEqual>(uint32_t, bool, const uint32_t *const *,
                                                           const uint32_t *const *);
template uint32_t float_compare_x86_64_v3<16, rocjitsu::arith::CompareLessEqual>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);
template uint32_t float_compare_x86_64_v3<16, rocjitsu::arith::CompareGreater>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);
template uint32_t float_compare_x86_64_v3<16, rocjitsu::arith::CompareNotEqual>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);
template uint32_t float_compare_x86_64_v3<16, rocjitsu::arith::CompareGreaterEqual>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);
template uint32_t float_compare_x86_64_v3<16, rocjitsu::arith::CompareOrdered>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);
template uint32_t float_compare_x86_64_v3<16, rocjitsu::arith::CompareUnordered>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);
template uint32_t float_compare_x86_64_v3<16, rocjitsu::arith::CompareNotGreaterEqual>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);
template uint32_t float_compare_x86_64_v3<16, rocjitsu::arith::CompareNotLessGreater>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);
template uint32_t float_compare_x86_64_v3<16, rocjitsu::arith::CompareNotGreater>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);
template uint32_t float_compare_x86_64_v3<16, rocjitsu::arith::CompareNotLessEqual>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);
template uint32_t float_compare_x86_64_v3<16, rocjitsu::arith::CompareNotEqualUnordered>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);
template uint32_t float_compare_x86_64_v3<16, rocjitsu::arith::CompareNotLess>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);
template uint32_t
float_compare_x86_64_v3<32, rocjitsu::arith::CompareLess>(uint32_t, bool, const uint32_t *const *,
                                                          const uint32_t *const *);
template uint32_t
float_compare_x86_64_v3<32, rocjitsu::arith::CompareEqual>(uint32_t, bool, const uint32_t *const *,
                                                           const uint32_t *const *);
template uint32_t float_compare_x86_64_v3<32, rocjitsu::arith::CompareLessEqual>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);
template uint32_t float_compare_x86_64_v3<32, rocjitsu::arith::CompareGreater>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);
template uint32_t float_compare_x86_64_v3<32, rocjitsu::arith::CompareNotEqual>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);
template uint32_t float_compare_x86_64_v3<32, rocjitsu::arith::CompareGreaterEqual>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);
template uint32_t float_compare_x86_64_v3<32, rocjitsu::arith::CompareOrdered>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);
template uint32_t float_compare_x86_64_v3<32, rocjitsu::arith::CompareUnordered>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);
template uint32_t float_compare_x86_64_v3<32, rocjitsu::arith::CompareNotGreaterEqual>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);
template uint32_t float_compare_x86_64_v3<32, rocjitsu::arith::CompareNotLessGreater>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);
template uint32_t float_compare_x86_64_v3<32, rocjitsu::arith::CompareNotGreater>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);
template uint32_t float_compare_x86_64_v3<32, rocjitsu::arith::CompareNotLessEqual>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);
template uint32_t float_compare_x86_64_v3<32, rocjitsu::arith::CompareNotEqualUnordered>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);
template uint32_t float_compare_x86_64_v3<32, rocjitsu::arith::CompareNotLess>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);
template uint32_t
float_compare_x86_64_v3<64, rocjitsu::arith::CompareLess>(uint32_t, bool, const uint32_t *const *,
                                                          const uint32_t *const *);
template uint32_t
float_compare_x86_64_v3<64, rocjitsu::arith::CompareEqual>(uint32_t, bool, const uint32_t *const *,
                                                           const uint32_t *const *);
template uint32_t float_compare_x86_64_v3<64, rocjitsu::arith::CompareLessEqual>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);
template uint32_t float_compare_x86_64_v3<64, rocjitsu::arith::CompareGreater>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);
template uint32_t float_compare_x86_64_v3<64, rocjitsu::arith::CompareNotEqual>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);
template uint32_t float_compare_x86_64_v3<64, rocjitsu::arith::CompareGreaterEqual>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);
template uint32_t float_compare_x86_64_v3<64, rocjitsu::arith::CompareOrdered>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);
template uint32_t float_compare_x86_64_v3<64, rocjitsu::arith::CompareUnordered>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);
template uint32_t float_compare_x86_64_v3<64, rocjitsu::arith::CompareNotGreaterEqual>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);
template uint32_t float_compare_x86_64_v3<64, rocjitsu::arith::CompareNotLessGreater>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);
template uint32_t float_compare_x86_64_v3<64, rocjitsu::arith::CompareNotGreater>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);
template uint32_t float_compare_x86_64_v3<64, rocjitsu::arith::CompareNotLessEqual>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);
template uint32_t float_compare_x86_64_v3<64, rocjitsu::arith::CompareNotEqualUnordered>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);
template uint32_t float_compare_x86_64_v3<64, rocjitsu::arith::CompareNotLess>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);

template uint64_t float_compare_x86_64_v3<16, rocjitsu::arith::CompareLess, uint64_t>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);
template uint64_t float_compare_x86_64_v3<16, rocjitsu::arith::CompareEqual, uint64_t>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);
template uint64_t float_compare_x86_64_v3<16, rocjitsu::arith::CompareLessEqual, uint64_t>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);
template uint64_t float_compare_x86_64_v3<16, rocjitsu::arith::CompareGreater, uint64_t>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);
template uint64_t float_compare_x86_64_v3<16, rocjitsu::arith::CompareNotEqual, uint64_t>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);
template uint64_t float_compare_x86_64_v3<16, rocjitsu::arith::CompareGreaterEqual, uint64_t>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);
template uint64_t float_compare_x86_64_v3<16, rocjitsu::arith::CompareOrdered, uint64_t>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);
template uint64_t float_compare_x86_64_v3<16, rocjitsu::arith::CompareUnordered, uint64_t>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);
template uint64_t float_compare_x86_64_v3<16, rocjitsu::arith::CompareNotGreaterEqual, uint64_t>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);
template uint64_t float_compare_x86_64_v3<16, rocjitsu::arith::CompareNotLessGreater, uint64_t>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);
template uint64_t float_compare_x86_64_v3<16, rocjitsu::arith::CompareNotGreater, uint64_t>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);
template uint64_t float_compare_x86_64_v3<16, rocjitsu::arith::CompareNotLessEqual, uint64_t>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);
template uint64_t float_compare_x86_64_v3<16, rocjitsu::arith::CompareNotEqualUnordered, uint64_t>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);
template uint64_t float_compare_x86_64_v3<16, rocjitsu::arith::CompareNotLess, uint64_t>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);
template uint64_t float_compare_x86_64_v3<32, rocjitsu::arith::CompareLess, uint64_t>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);
template uint64_t float_compare_x86_64_v3<32, rocjitsu::arith::CompareEqual, uint64_t>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);
template uint64_t float_compare_x86_64_v3<32, rocjitsu::arith::CompareLessEqual, uint64_t>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);
template uint64_t float_compare_x86_64_v3<32, rocjitsu::arith::CompareGreater, uint64_t>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);
template uint64_t float_compare_x86_64_v3<32, rocjitsu::arith::CompareNotEqual, uint64_t>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);
template uint64_t float_compare_x86_64_v3<32, rocjitsu::arith::CompareGreaterEqual, uint64_t>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);
template uint64_t float_compare_x86_64_v3<32, rocjitsu::arith::CompareOrdered, uint64_t>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);
template uint64_t float_compare_x86_64_v3<32, rocjitsu::arith::CompareUnordered, uint64_t>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);
template uint64_t float_compare_x86_64_v3<32, rocjitsu::arith::CompareNotGreaterEqual, uint64_t>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);
template uint64_t float_compare_x86_64_v3<32, rocjitsu::arith::CompareNotLessGreater, uint64_t>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);
template uint64_t float_compare_x86_64_v3<32, rocjitsu::arith::CompareNotGreater, uint64_t>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);
template uint64_t float_compare_x86_64_v3<32, rocjitsu::arith::CompareNotLessEqual, uint64_t>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);
template uint64_t float_compare_x86_64_v3<32, rocjitsu::arith::CompareNotEqualUnordered, uint64_t>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);
template uint64_t float_compare_x86_64_v3<32, rocjitsu::arith::CompareNotLess, uint64_t>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);
template uint64_t float_compare_x86_64_v3<64, rocjitsu::arith::CompareLess, uint64_t>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);
template uint64_t float_compare_x86_64_v3<64, rocjitsu::arith::CompareEqual, uint64_t>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);
template uint64_t float_compare_x86_64_v3<64, rocjitsu::arith::CompareLessEqual, uint64_t>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);
template uint64_t float_compare_x86_64_v3<64, rocjitsu::arith::CompareGreater, uint64_t>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);
template uint64_t float_compare_x86_64_v3<64, rocjitsu::arith::CompareNotEqual, uint64_t>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);
template uint64_t float_compare_x86_64_v3<64, rocjitsu::arith::CompareGreaterEqual, uint64_t>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);
template uint64_t float_compare_x86_64_v3<64, rocjitsu::arith::CompareOrdered, uint64_t>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);
template uint64_t float_compare_x86_64_v3<64, rocjitsu::arith::CompareUnordered, uint64_t>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);
template uint64_t float_compare_x86_64_v3<64, rocjitsu::arith::CompareNotGreaterEqual, uint64_t>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);
template uint64_t float_compare_x86_64_v3<64, rocjitsu::arith::CompareNotLessGreater, uint64_t>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);
template uint64_t float_compare_x86_64_v3<64, rocjitsu::arith::CompareNotGreater, uint64_t>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);
template uint64_t float_compare_x86_64_v3<64, rocjitsu::arith::CompareNotLessEqual, uint64_t>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);
template uint64_t float_compare_x86_64_v3<64, rocjitsu::arith::CompareNotEqualUnordered, uint64_t>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);
template uint64_t float_compare_x86_64_v3<64, rocjitsu::arith::CompareNotLess, uint64_t>(
    uint32_t, bool, const uint32_t *const *, const uint32_t *const *);

} // namespace rocjitsu::arith
