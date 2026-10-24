// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// x86 SIMD vector empirical square-root/reciprocal-square-root evaluation and scalar repair.
// Shared vector helpers; entry-point policy and dispatch remain in the family callers.

#pragma once

#include "math/rsq_model.h"
#include "math/sqrt_model.h"
#include "math/x86_64/sfu_exact_x86_64.h"
#include "rj_arith/rj_arith.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith::root_simd {

using sfu_simd::U4;
#if defined(__AVX512F__)
using sfu_simd::U8;
#endif
using sfu_simd::choose;
using sfu_simd::load;
using sfu_simd::mul32;
using sfu_simd::store;

// Select the requested polynomial coefficient field for each reduced input.
template <bool Reciprocal, class V> V coefficient(V index, int field) {
  if constexpr (Reciprocal && sizeof(V) == 32) {
    // Eight AVX2 register tables cost more than one gather for these 64 entries.
    auto words = _mm256_i64gather_epi32(
        reinterpret_cast<const int *>(detail::rsq::coefficients[field]), (__m256i)index, 4);
    return (U4)_mm256_cvtepu32_epi64(words);
  } else if constexpr (Reciprocal) {
    V lo = sfu_simd::lookup32(index, detail::rsq::coefficients[field]);
    V hi = sfu_simd::lookup32(index, detail::rsq::coefficients[field] + 32);
    return choose((V)(index >= 32ULL), hi, lo);
  } else {
    return sfu_simd::lookup32(index, detail::sqrt::coefficients[field]);
  }
}

// Collect vector lane predicates into a scalar bit mask.
inline uint32_t bits(U4 mask) { return _mm256_movemask_pd(_mm256_castsi256_pd((__m256i)mask)); }
#if defined(__AVX512F__)
// Collect vector lane predicates into a scalar bit mask.
inline uint32_t bits(U8 mask) { return _mm512_movepi64_mask((__m512i)mask); }
#endif

// Arithmetic right shift expressed with unsigned lanes, including on AVX2.
template <class V> V signed_shift(V value, int shift) {
  return (value >> shift) | ((V)((value >> 63) != 0ULL) << (64 - shift));
}

static_assert([] {
  for (int bucket = 0; bucket < 512; ++bucket) {
    int count = detail::kAmdgpuRsqCorrectionOffsets[bucket + 1] -
                detail::kAmdgpuRsqCorrectionOffsets[bucket];
    if (count > 32) {
      return false;
    }
  }
  return true;
}());

// Evaluate corrected reciprocal-square-root bits for reduced indices.
template <class V> V rsq(V index) {
  V segment = index >> 18;
  V value = sfu_simd::reciprocal_polynomial<true>(index, coefficient<true>(segment, 0),
                                                  coefficient<true>(segment, 1),
                                                  coefficient<true>(segment, 2));
  return sfu_simd::correct(index, value, detail::kAmdgpuRsqCorrectionOffsets,
                           detail::kAmdgpuRsqCorrections);
}

// Exact integer translation of the inherited SQRT stages. Only inputs within
// the inherited midpoint window need the sparse scalar residual lookup.
template <class V> V sqrt(V index) {
  V segment = index >> 19;
  V constant = coefficient<false>(segment, 0);
  V linear = coefficient<false>(segment, 1);
  V packed = coefficient<false>(segment, 2);
  V fraction = index & 0x7ffffULL;
  V product = mul32(linear, fraction);
  V bias = choose((V)(product >= (1ULL << 40)), (product ^ product) + (1ULL << 16),
                  (product ^ product) + (1ULL << 15));
  linear = (product + bias) >> 17;
  V square = mul32(fraction & ~3ULL, fraction & ~3ULL) >> 14;
  V inner = ((packed & 0xffffULL) << 8) + signed_shift(256ULL - mul32(packed >> 16, fraction), 10);
  V quadratic = mul32(inner, square);
  V negated =
      choose((V)(quadratic >= (1ULL << 47)), signed_shift((1ULL << 23) - quadratic, 24) << 1,
             signed_shift((1ULL << 22) - quadratic, 23));
  V sum = (constant << 22) + (linear << 19) + (negated << 12);
  V quotient = sum >> 24, remainder = sum & 0xffffffULL;
  V value =
      0x3f800000ULL + quotient +
      ((V)((remainder > 0x800000ULL) | ((remainder == 0x800000ULL) & ((quotient & 1ULL) != 0ULL))) &
       1ULL);
  V repair = (V)(remainder >= 0x800000ULL - 8192) & (V)(remainder <= 0x800000ULL + 8192);
  if (bits(repair)) {
    for (int lane = 0; lane < int(sizeof(V) / sizeof(uint64_t)); ++lane) {
      if (repair[lane]) {
        value[lane] = detail::amdgpu_sqrt_correct(uint32_t(index[lane]), uint32_t(value[lane]));
      }
    }
  }
  return value;
}

// Compute empirical exact roots/reciprocal roots and return requested FP causes from active
// lanes only.
template <bool Reciprocal, class V>
uint32_t run(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
             bool quiet_snan, bool reporting) {
  constexpr int Lanes = sizeof(V) / sizeof(uint64_t);
  uint32_t result[32], invalid = 0;
  for (int base = 0; base < 32; base += Lanes) {
    V input = load(a + base, V{});
    if (mode & RJ_ARITH_ALU_ABS_A) {
      input &= 0x7fffffffULL;
    }
    if (mode & RJ_ARITH_ALU_NEG_A) {
      input ^= 0x80000000ULL;
    }
    V sign = input & 0x80000000ULL, magnitude = input & 0x7fffffffULL, zero{};
    if constexpr (!Reciprocal) {
      if (reporting) {
        invalid |=
            bits((V)(sign != 0ULL) & (V)(magnitude != 0ULL) & (V)(magnitude <= 0x7f800000ULL))
            << base;
      }
    }
    // Special-only blocks need classification/finalization, but no polynomial.
    // This depends on operand values, never on EXEC density. Mixed blocks keep SIMD.
    V value{};
    V ordinary = (V)(input >= 0x800000ULL) & (V)(input < 0x7f800000ULL);
    if (bits(ordinary)) {
      V index = (input ^ 0x800000ULL) & 0xffffffULL;
      V normalized;
      if constexpr (Reciprocal) {
        normalized = rsq(index);
      } else {
        normalized = sqrt(index);
      }

      // floor((biased exponent - 127) / 2), represented with unsigned wrapping.
      V half_exponent = (((magnitude >> 23) + 1ULL) >> 1) - 64ULL;
      V exponent;
      if constexpr (Reciprocal) {
        exponent = (normalized >> 23) - half_exponent;
      } else {
        exponent = (normalized >> 23) + half_exponent;
      }
      value = (exponent << 23) | (normalized & 0x7fffffULL);
    }
    value = choose((V)(magnitude == 0x7f800000ULL), Reciprocal ? zero : input, value);
    value = choose((V)(sign != 0ULL), zero + 0xffc00000ULL, value);
    value = choose((V)(magnitude < 0x800000ULL), Reciprocal ? sign | 0x7f800000ULL : sign, value);
    value = choose((V)(magnitude > 0x7f800000ULL), input | uint64_t(quiet_snan ? 0x400000U : 0U),
                   value);
    store(result + base, sfu_simd::output(value, flags, mode));
  }
  for (int lane = 0; lane < 32; lane += 8) {
    auto mask = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                                  _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), mask,
                           _mm256_loadu_si256(reinterpret_cast<const __m256i *>(result + lane)));
  }
  return (invalid & exec_mask) ? 1U : 0U;
}

} // namespace rocjitsu::arith::root_simd
