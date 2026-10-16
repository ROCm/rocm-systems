// SPDX-License-Identifier: MIT

// x86 SIMD shared SIMD polynomial, correction lookup, and output-policy helpers for exact
// math.
// Shared vector helpers; entry-point policy and dispatch remain in the family callers.

#pragma once

#include "rj_arith/rj_arith.h"

#include <cstddef>
#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith::sfu_simd {

using U4 [[gnu::vector_size(32)]] = uint64_t;

// Multiply the low unsigned 32-bit factors of each 64-bit lane, returning full 64-bit
// products.
inline U4 mul32(U4 a, U4 b) { return (U4)_mm256_mul_epu32((__m256i)a, (__m256i)b); }
#if defined(__AVX512F__)
using U8 [[gnu::vector_size(64)]] = uint64_t;

// Multiply the low unsigned 32-bit factors of each 64-bit lane, returning full 64-bit
// products.
inline U8 mul32(U8 a, U8 b) { return (U8)_mm512_mul_epu32((__m512i)a, (__m512i)b); }
#endif

// Each field contains 32 dwords. AVX2 selects between four eight-entry
// register tables; AVX512 selects directly from two sixteen-entry tables.
// Only the even dwords are retained as widened uint64_t coefficients.
inline U4 lookup32(U4 index, const uint32_t *words) {
  const auto *table = reinterpret_cast<const __m256i *>(words);
  auto a = _mm256_permutevar8x32_epi32(_mm256_loadu_si256(table), (__m256i)index);
  auto b = _mm256_permutevar8x32_epi32(_mm256_loadu_si256(table + 1), (__m256i)index);
  auto c = _mm256_permutevar8x32_epi32(_mm256_loadu_si256(table + 2), (__m256i)index);
  auto d = _mm256_permutevar8x32_epi32(_mm256_loadu_si256(table + 3), (__m256i)index);
  auto bit3 = (__m256i)((index & 8ULL) != 0ULL);
  auto bit4 = (__m256i)((index & 16ULL) != 0ULL);
  auto lo = _mm256_blendv_epi8(a, b, bit3);
  auto hi = _mm256_blendv_epi8(c, d, bit3);
  return (U4)_mm256_blendv_epi8(lo, hi, bit4) & 0xffffffffULL;
}
#if defined(__AVX512F__)
// Look up 32-bit table entries at the supplied indices.
inline U8 lookup32(U8 indices, const uint32_t *words) {
  auto index = (__m512i)indices;
  const auto *table = words;
  auto selected =
      _mm512_permutex2var_epi32(_mm512_loadu_si512(table), index, _mm512_loadu_si512(table + 16));
  return (U8)selected & 0xffffffffULL;
}
#endif

// Select values lane by lane according to the supplied predicate.
template <class V> inline V choose(V mask, V yes, V no) { return (mask & yes) | (~mask & no); }

// Load unsigned 32-bit source words into consecutive 64-bit vector lanes.
inline U4 load(const uint32_t *p, U4) {
  return (U4)_mm256_cvtepu32_epi64(_mm_loadu_si128(reinterpret_cast<const __m128i *>(p)));
}

// Store the low 32 bits of each 64-bit vector lane into consecutive words.
inline void store(uint32_t *p, U4 value) {
  auto words =
      _mm256_permutevar8x32_epi32((__m256i)value, _mm256_setr_epi32(0, 2, 4, 6, 0, 0, 0, 0));
  _mm_storeu_si128(reinterpret_cast<__m128i *>(p), _mm256_castsi256_si128(words));
}

#if defined(__AVX512F__)

// Load unsigned 32-bit source words into consecutive 64-bit vector lanes.
inline U8 load(const uint32_t *p, U8) {
  return (U8)_mm512_cvtepu32_epi64(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(p)));
}

// Store the low 32 bits of each 64-bit vector lane into consecutive words.
inline void store(uint32_t *p, U8 value) {
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(p), _mm512_cvtepi64_epi32((__m512i)value));
}

#endif

// Compare up to 16 (AVX2) or 32 (AVX512) residual entries. Loads near the table's
// end are shifted backwards; the caller excludes entries outside the bucket.
inline uint32_t matches(const uint16_t *p, uint16_t key, U4) {
  auto entries = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(p));
  auto equal = _mm256_cmpeq_epi16(_mm256_and_si256(entries, _mm256_set1_epi16(0x7fff)),
                                  _mm256_set1_epi16(key));
  return uint32_t(_mm256_movemask_epi8(equal));
}
#if defined(__AVX512F__)
// Match the low 15 bits of 32 consecutive 16-bit correction entries against key.
inline uint32_t matches(const uint16_t *p, uint16_t key, U8) {
  auto entries = _mm512_loadu_si512(p);
  return _mm512_cmpeq_epi16_mask(_mm512_and_si512(entries, _mm512_set1_epi16(0x7fff)),
                                 _mm512_set1_epi16(key));
}
#endif

// Match the low 15 bits of 32 consecutive 16-bit correction entries against key.
inline uint64_t matches32(const uint16_t *p, uint16_t key, U4 v) {
  return uint64_t(matches(p, key, v)) | (uint64_t(matches(p + 16, key, v)) << 32);
}
#if defined(__AVX512F__)
// Match the low 15 bits of 32 consecutive 16-bit correction entries against key.
inline uint64_t matches32(const uint16_t *p, uint16_t key, U8 v) { return matches(p, key, v); }
#endif

// Apply sparse empirical correction entries to the polynomial results.
template <class V, size_t N>
V correct(V mantissa, V result, const uint16_t *offsets, const uint16_t (&corrections)[N]) {
  // AVX2 movemask has two bits per uint16_t; AVX512 has one.
  constexpr int Bits = sizeof(V) == 32 ? 2 : 1;
  constexpr int Count = N;

  // Expose independent bucket loads without fully unrolling an eight-lane block.
#pragma GCC unroll 4
  for (int lane = 0; lane < int(sizeof(V) / sizeof(uint64_t)); ++lane) {
    int bucket = mantissa[lane] >> 15;
    int first = offsets[bucket];
    int end = offsets[bucket + 1];

    // Shift the last load backwards, then exclude both neighboring buckets.
    int base = first < Count - 32 ? first : Count - 32;
    int last_bit = Bits * (end - base);
    uint64_t valid = last_bit == 64 ? ~0ULL : (1ULL << last_bit) - 1;
    int first_bit = Bits * (first - base);
    valid &= first_bit == 64 ? 0 : ~((1ULL << first_bit) - 1);
    uint64_t found = matches32(corrections + base, uint16_t(mantissa[lane] & 0x7fff), V{}) & valid;
    if (found) {
      int index = base + __builtin_ctzll(found) / Bits;
      result[lane] += (corrections[index] & 0x8000) ? 1ULL : -1ULL;
    }
  }
  return result;
}

// Apply guest denormal and output-modifier policies to computed result bits.
template <class V> V output(V value, uint64_t flags, uint32_t mode) {
  V zero{}, magnitude{}, exponent{};
  unsigned omod = (mode >> 6) & 3;
  if (omod) {
    magnitude = value & 0x7fffffffULL;
    exponent = magnitude >> 23;
    V scaled;
    if (omod == 3) {
      scaled = value - 0x800000ULL;
      scaled = choose((V)(exponent == 1ULL), value & 0x80000000ULL, scaled);
    } else {
      scaled = value + (uint64_t(omod) << 23);
      scaled =
          choose((V)(exponent + omod >= 255ULL), (value & 0x80000000ULL) | 0x7f800000ULL, scaled);
    }
    scaled = choose((V)(exponent == 0ULL), zero, scaled);
    value = choose((V)(magnitude < 0x7f800000ULL), scaled, value);
  }
  if (mode & RJ_ARITH_ALU_CLAMP) {
    V nan = (V)((value & 0x7fffffffULL) > 0x7f800000ULL);
    V clamped = choose((V)(value > 0x3f800000ULL), zero + 0x3f800000ULL, value);
    clamped = choose((V)((value & 0x80000000ULL) != 0ULL) | nan, zero, clamped);
    value = flags & RJ_ARITH_FP_DX10_CLAMP_OFF ? choose(nan, value, clamped) : clamped;
  }
  return value;
}

// Shared stages of the inherited RCP/RSQ fixed-point polynomials. Their
// distinct shifts and scaling are selected explicitly below. The only
// >64-bit product is reconstructed from 32-bit limbs; no FP approximation.
template <bool Rsq, class V> V reciprocal_polynomial(V mantissa, V constant, V linear, V packed) {
  V quadratic = packed & 0xffffULL, cubic = packed >> 16;
  constexpr int BaseShift = Rsq ? 13 : 14;
  V fraction = mantissa & 0x3ffffULL;
  V product = mul32(linear, fraction);
  V shift = ((V)(product >= (1ULL << (BaseShift + 24))) & 1ULL) + uint64_t(BaseShift);
  V quotient = product >> shift;
  V one = (product ^ product) + 1ULL;
  V remainder = product & ((one << shift) - 1ULL);
  V midpoint = one << (shift - 1ULL);
  V rounded = quotient + ((V)((remainder > midpoint) |
                              ((remainder == midpoint) & ((quotient & 1ULL) != 0ULL))) &
                          1ULL);
  linear = rounded << (shift - uint64_t(BaseShift));
  V inner = (quadratic << 18) - mul32(cubic, fraction);
  V square = mul32(fraction & ~1ULL, fraction & ~1ULL);

  // (inner * square) >> 51, retaining carries between 32-bit limbs.
  V p00 = mul32(inner, square), p01 = mul32(inner, square >> 32);
  V p10 = mul32(inner >> 32, square), p11 = mul32(inner >> 32, square >> 32);
  V middle = (p00 >> 32) + (p01 & 0xffffffffULL) + (p10 & 0xffffffffULL);
  V high = (p01 >> 32) + (p10 >> 32) + p11 + (middle >> 32);
  quadratic = (high << 13) | ((middle & 0xffffffffULL) >> 19);
  if constexpr (Rsq) {
    return 0x3e800000ULL + ((constant * 32ULL - linear + quadratic * 4ULL + 64ULL) >> 7);
  } else {
    return 0x3e800000ULL + ((constant * 16ULL - linear + quadratic * 2ULL + 32ULL) >> 6);
  }
}

} // namespace rocjitsu::arith::sfu_simd
