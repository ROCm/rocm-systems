// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Vector evaluation of the existing EXP fixed-point polynomial and its staged roundings.
// Reuses the scalar model's coefficients; special values stay on the vector path.
#pragma once

#include "math/exp_model.h"
#include "math/x86_64/trans_exact_x86_64.h"
#include "rj_arith/detail/rj_arith_common.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::exp_simd {
using namespace trans_simd;

static_assert(sizeof(detail::exp::Coefficient) == 4 * sizeof(uint32_t));

// Gather one coefficient field from the inherited four-field table.
inline U4 coefficient(U4 index, int field) {
  auto packed = _mm256_permutevar8x32_epi32((__m256i)(index * 4ULL),
                                            _mm256_setr_epi32(0, 2, 4, 6, 0, 0, 0, 0));
  auto value = _mm_i32gather_epi32(reinterpret_cast<const int *>(detail::exp::coefficients) + field,
                                   _mm256_castsi256_si128(packed), 4);
  return (U4)_mm256_cvtepu32_epi64(value);
}

// Commit four low dwords, preserving lanes outside the execution mask.
inline void masked_store(uint32_t *d, U4 value, uint32_t exec_mask) {
  auto packed =
      _mm256_permutevar8x32_epi32((__m256i)value, _mm256_setr_epi32(0, 2, 4, 6, 0, 0, 0, 0));
  auto mask = _mm_sllv_epi32(_mm_set1_epi32(int(exec_mask)), _mm_setr_epi32(31, 30, 29, 28));
  _mm_maskstore_epi32(reinterpret_cast<int *>(d), mask, _mm256_castsi256_si128(packed));
}

#if defined(__AVX512F__)
// Gather eight coefficients without expanding or duplicating the table.
inline U8 coefficient(U8 index, int field) {
  auto indices = _mm512_cvtepi64_epi32((__m512i)(index * 4ULL));
  auto value = _mm256_i32gather_epi32(
      reinterpret_cast<const int *>(detail::exp::coefficients) + field, indices, 4);
  return (U8)_mm512_cvtepu32_epi64(value);
}

// Commit eight low dwords, preserving lanes outside the execution mask.
inline void masked_store(uint32_t *d, U8 value, uint32_t exec_mask) {
  _mm256_mask_storeu_epi32(d, __mmask8(exec_mask), _mm512_cvtepi64_epi32((__m512i)value));
}
#endif

// Round a nonnegative fixed-point value to nearest-even after a fixed right shift.
template <int Shift, typename V> inline V round_even(V value) {
  V whole = value >> Shift;
  V rest = value & ((1ULL << Shift) - 1ULL);
  V increment = (V)((rest > (1ULL << (Shift - 1))) |
                    ((rest == (1ULL << (Shift - 1))) & ((whole & 1ULL) != 0ULL)));
  return whole + (increment & 1ULL);
}

// Compute the inherited EXP result, including all nonfinite and out-of-range inputs.
template <typename V> V evaluate(V input, bool quiet_snan) {
  V zero{}, magnitude = input & 0x7fffffffULL;
  V negative = (V)((input & 0x80000000ULL) != 0ULL);
  // Sanitize exceptional lanes before shifts; restore their answers after the polynomial.
  V safe = choose((V)((magnitude >= 0x33800000ULL) & (magnitude < 0x43000000ULL)), magnitude,
                  zero + 0x3e800000ULL);
  // Q29 reduction shifts the 24-bit significand by exponent_field - 121.
  V exponent = safe >> 23;
  V mantissa = (safe & 0x7fffffULL) | 0x800000ULL;
  V left = choose((V)(exponent >= 121ULL), exponent - 121ULL, zero);
  V right = choose((V)(exponent < 121ULL), 121ULL - exponent, zero);
  V absolute = (mantissa << left) >> right;
  // Ones-complement reduction places exact negative integers in the preceding interval.
  V phase = choose(negative, ~absolute, absolute);
  V fraction = phase & 0xffffffULL, index = (phase >> 24) & 31ULL;
  V constant = coefficient(index, 0), linear = coefficient(index, 1);
  V quadratic = coefficient(index, 2), cubic = coefficient(index, 3);

  // Preserve each scalar-model rounding boundary rather than fusing polynomial terms.
  V lp = mul32(linear, fraction);
  linear = choose((V)(lp >= (1ULL << 47)), round_even<24>(lp), round_even<23>(lp) >> 1);
  V coordinate = fraction >> 6;
  V square = mul32(coordinate, coordinate) >> 12;
  V inner = round_even<7>((quadratic << 7) + mul32(cubic, coordinate));
  V product = mul32(inner, square);
  V quad =
      choose((V)(product >= (1ULL << 47)), round_even<24>(product) * 2ULL, round_even<23>(product));
  V sum = ((constant + linear) << 8) + quad;
  // Only the low dword is retained, so logical and arithmetic phase shifts agree here.
  V value = (((phase >> 29) + 126ULL) * 0x800000ULL + round_even<13>(sum)) & 0xffffffffULL;
  value = choose((V)(magnitude < 0x33800000ULL), zero + 0x3f800000ULL, value);
  value = choose(negative & (V)(magnitude > 0x42fc0000ULL), zero, value);
  value = choose(~negative & (V)(magnitude >= 0x43000000ULL), zero + 0x7f800000ULL, value);
  return choose((V)(magnitude > 0x7f800000ULL), input | uint64_t(quiet_snan ? 0x400000 : 0), value);
}

// Apply source/output modifiers and store active results after reading the complete wave.
template <typename V>
void run(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
         bool quiet_snan) {
  constexpr int Width = sizeof(V) / sizeof(uint64_t);
  uint32_t result[32];
  for (int lane = 0; lane < 32; lane += Width) {
    V input = load(a + lane, V{});
    if (mode & RJ_ARITH_ALU_ABS_A) {
      input &= 0x7fffffffULL;
    }
    if (mode & RJ_ARITH_ALU_NEG_A) {
      input ^= 0x80000000ULL;
    }
    V value = evaluate(input, quiet_snan);
    // The scalar output stage flushes before OMOD, even if the polynomial rounded tiny.
    value = choose((V)((value & 0x7f800000ULL) == 0ULL), value & 0x80000000ULL, value);
    store(result + lane, output(value, flags, mode));
  }
  for (int lane = 0; lane < 32; lane += Width) {
    masked_store(d + lane, load(result + lane, V{}), exec_mask >> lane);
  }
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::exp_simd
