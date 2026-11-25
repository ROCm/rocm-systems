// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// SIMD evaluation of the captured FP32 sine model. Keep the normalized
// zero-origin interval and denormal input policies in the scalar model.
#pragma once

#include "math/trig_model.h"
#include "math/x86_64/trans_exact_x86_64.h"

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::trig {
#if defined(__AVX512F__)
using Products = trans_simd::U8;
#else
using Products = trans_simd::U4;
#endif
using Words [[gnu::vector_size(sizeof(Products))]] = uint32_t;
#if defined(__AVX512F__)
using trans_simd::choose;
#else
// All selection predicates fill their dword or qword. AVX2 can blend each
// lane directly from its sign bit.
inline Words choose(Words mask, Words yes, Words no) {
  return (Words)_mm256_blendv_ps((__m256)no, (__m256)yes, (__m256)mask);
}
inline Products choose(Products mask, Products yes, Products no) {
  return (Products)_mm256_blendv_pd((__m256d)no, (__m256d)yes, (__m256d)mask);
}
#endif
using trans_simd::mul32;

// Form the coefficient magnitudes and the Q7 quadratic base at compile time.
struct SineCoefficients {
  alignas(64) uint32_t constant[32], linear[32], inner[32], cubic[32];
};
inline constexpr SineCoefficients sine_coefficients = [] {
  SineCoefficients c{};
  for (int i = 0; i < 32; ++i) {
    int linear = coefficients[1][i], quadratic = coefficients[2][i], cubic = coefficients[3][i];
    c.constant[i] = coefficients[0][i];
    c.linear[i] = linear < 0 ? -linear : linear;
    c.inner[i] = (cubic <= 0 ? -quadratic : -(quadratic + cubic)) * 128;
    c.cubic[i] = cubic < 0 ? -cubic : cubic;
  }
  return c;
}();

// Lookup across the 16 sine and 16 cosine columns using register permutations.
inline Words sine_lookup(const uint32_t *a, Words index, Words cosine) {
#if defined(__AVX512F__)
  return (Words)_mm512_permutex2var_epi32(_mm512_loadu_si512(a), (__m512i)(index | (cosine & 16)),
                                          _mm512_loadu_si512(a + 16));
#else
  const auto permute = [&](int offset) {
    return (Words)_mm256_permutevar8x32_epi32(
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + offset)), (__m256i)index);
  };
  const Words high = (Words)((index & 8) != 0);
  return choose(cosine, choose(high, permute(24), permute(16)),
                choose(high, permute(8), permute(0)));
#endif
}

// Phase is measured in turns with 31 fractional bits. Oversized shifts give zero.
inline Words sine_phase(Words mantissa, Words shift) {
#if defined(__AVX512F__)
  return ((Words)_mm512_sllv_epi32((__m512i)mantissa, (__m512i)shift) |
          (Words)_mm512_srlv_epi32((__m512i)mantissa, (__m512i)-shift)) &
         0x7fffffff;
#else
  return ((Words)_mm256_sllv_epi32((__m256i)mantissa, (__m256i)shift) |
          (Words)_mm256_srlv_epi32((__m256i)mantissa, (__m256i)-shift)) &
         0x7fffffff;
#endif
}

struct SineProducts {
  Products linear, quadratic, small;
};

// Form full 64-bit products from the lower dword of each pair. In the second
// sine interval the final sum uses Q34 instead of the ordinary Q29.
inline SineProducts sine_products(Products linear, Products coordinate, Products square,
                                  Products inner, Products near, Products mantissa) {
  const Products prod = mul32(linear, coordinate);
  const Products quadratic = mul32(inner, mul32(square, square) >> 10);
  const Products large = prod >> 45;
  const Products shift = large + 21;
  const Products round = (large << 20) + 0xfffff + ((prod >> shift) & 1);
  const Products general_linear = ((prod + round) >> shift) << large;
  const Products bias = (quadratic >> 25) & (1u << 22);
  const Products general_quadratic = ((quadratic + ((1u << 29) - 1 - (1u << 22)) - bias) >> 29)
                                     << 1;
  const Products near_linear = (prod >> 20) << 4;
  const Products near_quadratic = (quadratic + 0x3fffff + ((quadratic >> 23) & 1)) >> 23;
  const Products mask = -(near & 1);

  // Normal small inputs multiply by the captured significand before RNE.
  const Products small = mul32(mantissa, Products{} + 0xc90fd5);
  const Products top = small >> 47;
  const Products aligned = small << (top ^ 1);
  const Products rounded = (aligned + 0x7fffff + ((aligned >> 24) & 1)) >> 24;
  return {choose(mask, near_linear, general_linear),
          choose(mask, near_quadratic, general_quadratic), rounded + (top << 23)};
}

inline Words sine_merge(Products even, Products odd) {
  return (Words)((even & 0xffffffffULL) | (odd << 32));
}

// Truncate a Q29 or Q34 sum to FP32. AVX2 extracts its leading position by
// converting at most 24 significant bits, so host rounding and status stay unchanged.
inline Words sine_pack(Words sum, Words near) {
#if defined(__AVX512F__)
  const Words zeros = (Words)_mm512_lzcnt_epi32((__m512i)sum);
  const Words exponent = choose(near, Words{} + 123, Words{} + 128) - zeros;
  return (exponent << 23) + ((sum << (zeros & 31)) >> 8);
#else
  const Words exponent = (Words)_mm256_castps_si256(_mm256_cvtepi32_ps((__m256i)(sum >> 8))) >> 23;
  const Words biased = exponent - choose(near, Words{} + 27, Words{} + 22);
  return (biased << 23) + ((sum << ((150 - exponent) & 31)) >> 8);
#endif
}

inline Words sine_values(Words bits, Words &fallback, bool quiet_snan) {
  const Words magnitude = bits & 0x7fffffff;
  const Words mantissa = (magnitude & 0x7fffff) | 0x800000;
  const Words phase = sine_phase(mantissa, (magnitude >> 23) - 119);
  const Words quadrant = phase >> 29, quarter = phase & 0x1fffffff;
  const Words reflected = (Words)(quarter >= 0x10000000);
  const Words offset = quarter ^ (reflected & 0x1fffffff);
  const Words index = offset >> 24, fraction = offset & 0xffffff;
  const Words cosine = -((quadrant ^ reflected) & 1);
  const Words near = ~cosine & (Words)(index < 2);

  const auto &c = sine_coefficients;
  const Words constant = sine_lookup(c.constant, index, cosine);
  const Words linear = sine_lookup(c.linear, index, cosine);
  const Words quadratic = sine_lookup(c.inner, index, cosine);
  const Words cubic = sine_lookup(c.cubic, index, cosine);
  const Words scaled = cubic * ((fraction ^ (cosine & 0xffffff)) >> 6);
  const Words inner = quadratic + ((scaled + 1023 + ((scaled >> 11) & 1)) >> 11);
  const Words square = fraction >> 7;

  // Round within each 20-bit coordinate block, saturating at its endpoint.
  const Words general_input = (fraction + 8) >> 4;
  const Words near_input = (fraction + 2) >> 2;
  const Words block_end = (fraction >> 2) | 0xfffff;
  const Words coord =
      choose(near, choose((Words)(near_input > block_end), block_end, near_input) << 2,
             choose((Words)(general_input > 0xfffff), Words{} + 0xfffff, general_input) << 4);
  const auto even = sine_products((Products)linear, (Products)coord, (Products)square,
                                  (Products)inner, (Products)near, (Products)mantissa);
  const auto odd =
      sine_products((Products)linear >> 32, (Products)coord >> 32, (Products)square >> 32,
                    (Products)inner >> 32, (Products)near >> 32, (Products)mantissa >> 32);
  const Words linear_term = sine_merge(even.linear, odd.linear);
  const Words sum = (constant << choose(near, Words{} + 8, Words{} + 3)) +
                    ((linear_term ^ cosine) - cosine) - sine_merge(even.quadratic, odd.quadratic);
  const Words unit = cosine & (Words)(offset < 46592), origin = ~cosine & (Words)(quarter == 0);
  const Words sign = ((quadrant << 30) ^ bits) & 0x80000000;
  Words result = (choose(unit, Words{} + 0x3f800000, sine_pack(sum, near)) | sign) & ~origin;
  const Words pending = ~(unit | origin) & ((near & (Words)(index == 0)) |
                                            ((Words)(sum < 256) | (Words)(sum > 0x7fffffff)));

  const Words small = (Words)(magnitude < 0x39c00000), integral = (Words)(magnitude >= 0x4b000000);
  const Words empty = (Words)(magnitude == 0);
  const Words tiny_result =
      ((magnitude & 0x7f800000) + 0x800000 + sine_merge(even.small, odd.small)) |
      (bits & 0x80000000);
  result = choose(empty, bits, choose(small, tiny_result, result) & ~integral);
  fallback = (~empty & (Words)(magnitude < 0x800000)) | (~(small | integral) & pending);
  const Words nonfinite = choose((Words)(magnitude == 0x7f800000), Words{} + 0xffc00000,
                                 bits | uint32_t(quiet_snan ? 0x400000 : 0));
  return choose((Words)(magnitude >= 0x7f800000), nonfinite, result);
}

// Snapshot each input group before writing its results so d and a may alias.
inline void evaluate_wave(uint32_t *d, const uint32_t *a, unsigned denorm, bool quiet_snan) {
  constexpr unsigned lanes = sizeof(Words) / sizeof(uint32_t);
  for (unsigned lane = 0; lane < 32; lane += lanes) {
    Words input, fallback;
    __builtin_memcpy(&input, a + lane, sizeof(input));
    Words values = sine_values(input, fallback, quiet_snan);
    __builtin_memcpy(d + lane, &values, sizeof(values));
#if defined(__AVX512F__)
    unsigned pending = _mm512_movepi32_mask((__m512i)fallback);
#else
    unsigned pending = unsigned(_mm256_movemask_ps(_mm256_castsi256_ps((__m256i)fallback)));
#endif
    while (pending) {
      const unsigned index = bit_count_zero<true>(pending);
      d[lane + index] = evaluate(input[index], false, denorm, quiet_snan);
      pending &= pending - 1;
    }
  }
}
} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::trig
