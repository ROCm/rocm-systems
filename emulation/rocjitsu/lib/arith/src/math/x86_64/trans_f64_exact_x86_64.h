// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// x86 SIMD vector empirical FP64 reciprocal, reciprocal-square-root and square-root model.
// Shared vector helpers; entry-point policy and dispatch remain in the family callers.

#pragma once

// Lane-parallel form of math/unary_f64_exact.h; the levels match it top-down.
//
// Level 1, run: load split-word sources, ABS/NEG and input flushing, evaluate,
//          then output flushing, OMOD and CLAMP as trans_f64_exact_value does.
// Level 2, evaluate: blocks with an ordinary source run the datapath in every
//          lane; power-of-two neighbor cases and IEEE special sources are
//          blended over its result afterwards.
// Level 3, the datapaths and encoding, all in 64-bit integer lanes.
//
// Exponents are carried biased by 1023 + 64, so they stay positive before the
// final encoding: subnormal sources are normalized by an exact 2^64 scale.

#include "fp/fp64.h"
#include "math/rcp_model.h"
#include "math/trans_f64_model.h"
#include "math/unary_f64_exact.h"
#include "math/x86_64/root_exact_x86_64.h"
#include "math/x86_64/trans_exact_x86_64.h"
#include "rj_arith/rj_arith.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::trans_f64_simd {

using trans_simd::U4;
#if defined(__AVX512F__)
using trans_simd::U8;
#endif
using trans_f64::kFraction;
using trans_f64::kInfinity;
using trans_f64::kMinNormal;
using trans_f64::kQuiet;
using trans_f64::kSign;
using trans_simd::choose;
using trans_simd::load;
using trans_simd::mul32;
using trans_simd::store;

// Signed lanes for the result exponent, which may fall below the normal range.
using S4 [[gnu::vector_size(32)]] = int64_t;
#if defined(__AVX512F__)
using S8 [[gnu::vector_size(64)]] = int64_t;
#endif

// Broadcast a constant to every lane.
template <typename V> V splat(uint64_t value) { return V{} + value; }

// Multiply by 2^64, which is exact and normalizes every subnormal source.
inline U4 scale_2_64(U4 magnitude) {
  return (U4)_mm256_castpd_si256(
      _mm256_mul_pd(_mm256_castsi256_pd((__m256i)magnitude), _mm256_set1_pd(0x1p64)));
}
#if defined(__AVX512F__)
// Multiply by 2^64, which is exact and normalizes every subnormal source.
inline U8 scale_2_64(U8 magnitude) {
  return (U8)_mm512_castpd_si512(
      _mm512_mul_pd(_mm512_castsi512_pd((__m512i)magnitude), _mm512_set1_pd(0x1p64)));
}
#endif

// Round x / 2^shift to nearest, ties to even.
template <typename V> V round_shift(V x, int shift) {
  const uint64_t half = uint64_t{1} << (shift - 1);
  V quotient = x >> shift, remainder = x & ((uint64_t{1} << shift) - 1);
  return quotient +
         ((V)((remainder > half) | ((remainder == half) & ((quotient & 1ULL) != 0ULL))) & 1ULL);
}

// trans_f64::higher_order; every product has 32-bit factors.
template <typename V> V higher_order(V packed, V f, int quadratic_shift, int cubic_shift) {
  V quadratic = packed & 0xffffULL, cubic = packed >> 16;
  V top = f >> 7;
  V square = mul32(top, top) >> 10;
  V inner =
      (quadratic << quadratic_shift) - round_shift(mul32(cubic, (f >> 4) + 1ULL), cubic_shift);
  V product = mul32(inner, square);
  return choose((V)(product >= (1ULL << 47)), round_shift(product, 24) * 2ULL,
                round_shift(product, 23));
}

// trans_f64::rcp_datapath.
template <typename V> V rcp_datapath(V m29) {
  V segment = m29 >> 24, f = m29 & 0xffffffULL;
  V constant = trans_simd::lookup32(segment, detail::rcp_coefficients[0]);
  V linear = trans_simd::lookup32(segment, detail::rcp_coefficients[1]);
  V packed = trans_simd::lookup32(segment, detail::rcp_coefficients[2]);
  V linear_product = mul32(linear, f);
  linear = choose((V)(linear_product >= (1ULL << 44)), round_shift(linear_product, 21),
                  (linear_product + 0x17ffffULL) >> 21);
  constant = constant * 512ULL + ((V)(segment != 0ULL) & 256ULL);
  return constant - linear * 64ULL + higher_order(packed, f, 8, 12);
}

// trans_f64::rsq_datapath.
template <typename V> V rsq_datapath(V parity, V m29) {
  V segment = m29 >> 24, f = m29 & 0xffffffULL, index = (parity << 5) | segment;
  V constant = root_simd::coefficient<true>(index, 0);
  V linear = root_simd::coefficient<true>(index, 1);
  V packed = root_simd::coefficient<true>(index, 2);
  V linear_input = choose((V)(segment == 31ULL), f & ~0x30ULL, f);
  V linear_product = mul32(linear, linear_input);
  linear = (linear_product +
            choose((V)(linear_product >= (1ULL << 43)), splat<V>(0x17ffff), splat<V>(0x1bffff))) >>
           21;
  constant = constant * 1024ULL + ((V)(segment != 0ULL) & 512ULL);
  return constant - linear * 128ULL + higher_order(packed, f, 9, 11);
}

// trans_f64::sqrt_datapath.
template <typename V> V sqrt_datapath(V parity, V m28) {
  V segment = m28 >> 24, f = m28 & 0xffffffULL, index = (parity << 4) | segment;
  V constant = root_simd::coefficient<false>(index, 0);
  V linear = root_simd::coefficient<false>(index, 1);
  V packed = root_simd::coefficient<false>(index, 2);
  V linear_product = mul32(linear, f);
  linear = choose((V)(linear_product >= (1ULL << 45)), round_shift(linear_product, 22),
                  (linear_product + (1ULL << 20)) >> 22);
  constant = (1ULL << 35) + constant * 1024ULL + ((V)(segment != 15ULL) & 512ULL);
  return constant + linear * 128ULL - higher_order(packed, f, 8, 11);
}

// trans_f64::encode, from an IEEE-biased exponent. Only RCP of the largest
// sources leaves the normal range downwards, by at most two places.
template <typename V, typename S> V encode(S biased, V fraction52) {
  V normal = ((V)biased << 52) | fraction52;
  V shift = (V)(1 - biased);
  V subnormal = (kMinNormal | fraction52) >> choose((V)(shift > 63ULL), splat<V>(63), shift);
  V value = choose((V)(biased < 1), subnormal, normal);
  return choose((V)(biased >= 2047), splat<V>(kInfinity), value);
}

// trans_f64::trans_f64_magnitude for every lane.
template <Fp64 Op, typename V, typename S> V magnitude_value(V magnitude) {
  V subnormal = (V)(magnitude < kMinNormal);
  V scaled = choose(subnormal, scale_2_64(magnitude), magnitude);
  // The source exponent e is biased - 1087.
  V biased = (scaled >> 52) + (~subnormal & 64ULL);
  V fraction = scaled & kFraction;
  if constexpr (Op == Fp64::Rcp) {
    V m29 = fraction >> 23;
    V datapath = (rcp_datapath(m29) - (1ULL << 34)) << 18;

    // The scalar model's three power-of-two cases, as exponent and fraction
    // overrides plus a one-unit step that the encoding then takes.
    V exact_power = (V)(fraction == 0ULL);
    V below_power = (V)(m29 == 0ULL) & ~exact_power;
    V above_power = (V)(fraction == kFraction);
    V power = exact_power | below_power | above_power;
    S exponent = (S)(splat<V>(2109) - biased + ((exact_power | below_power) & 1ULL));
    V value = encode(exponent, choose(power, V{}, datapath));
    V step = (above_power & 1ULL) | below_power;
    return choose((V)(value == kInfinity), value, value + step);
  } else {
    // floor(e / 2) is half - 544, and e's parity is that of biased + 1.
    V parity = (biased + 1ULL) & 1ULL, half = (biased + 1ULL) >> 1;
    if constexpr (Op == Fp64::Rsq) {
      V m29 = fraction >> 23;
      V datapath = (rsq_datapath(parity, m29) - (1ULL << 35)) << 17;
      V power = (V)(parity == 0ULL) & (V)(m29 == 0ULL);
      V exponent = 1566ULL - half + (power & 1ULL);
      V value = (exponent << 52) | choose(power, V{}, datapath);
      return value - (power & (V)(fraction != 0ULL) & 1ULL);
    } else {
      V datapath = (sqrt_datapath(parity, fraction >> 24) - (1ULL << 35)) << 17;
      return ((half + 479ULL) << 52) | datapath;
    }
  }
}

// trans_f64::trans_f64_bits for every lane.
template <Fp64 Op, typename V, typename S> V evaluate(V bits, bool quiet_snan) {
  V sign = bits & kSign, magnitude = bits & ~kSign;
  // Special-only blocks need classification, but no datapath. This depends on
  // operand values, never on EXEC density. Mixed blocks keep SIMD.
  V ordinary = (V)(magnitude != 0ULL) & (V)(magnitude < kInfinity);
  if constexpr (Op != Fp64::Rcp) {
    ordinary &= (V)(sign == 0ULL);
  }
  V value{};
  if (root_simd::bits(ordinary)) {
    value = magnitude_value<Op, V, S>(magnitude);
  }
  value |= sign;

  // Special sources, lowest priority first.
  V infinity = (V)(magnitude == kInfinity), zero = (V)(magnitude == 0ULL);
  if constexpr (Op == Fp64::Rcp) {
    value = choose(infinity, sign, value);
    value = choose(zero, sign | kInfinity, value);
  } else {
    value = choose(infinity, Op == Fp64::Rsq ? V{} : splat<V>(kInfinity), value);
    value = choose((V)(sign != 0ULL), splat<V>(kSign | kInfinity | kQuiet), value);
    value = choose(zero, Op == Fp64::Rsq ? sign | kInfinity : bits, value);
  }
  return choose((V)(magnitude > kInfinity), quiet_snan ? bits | kQuiet : bits, value);
}

// trans_f64_exact_output for every lane.
template <typename V> V output(V value, uint64_t flags, uint32_t mode) {
  if (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) {
    value = choose((V)((value & ~kSign) < kMinNormal), value & kSign, value);
  }
  const unsigned omod = (mode >> 6) & 3;
  if (omod) {
    V sign = value & kSign, magnitude = value & ~kSign, exponent = magnitude >> 52;
    V scaled;
    if (omod == 3) {
      scaled = choose((V)(exponent == 1ULL), sign, value - kMinNormal);
    } else {
      scaled =
          choose((V)(exponent + omod >= 2047ULL), sign | kInfinity, value + (uint64_t(omod) << 52));
    }
    scaled = choose((V)(exponent == 0ULL), V{}, scaled);
    value = choose((V)(magnitude < kInfinity), scaled, value);
  }
  if (mode & RJ_ARITH_ALU_CLAMP) {
    V nan = (V)((value & ~kSign) > kInfinity);
    V clamped = choose((V)(value > 0x3ff0000000000000ULL), splat<V>(0x3ff0000000000000ULL), value);
    clamped = choose((V)((value & kSign) != 0ULL) | nan, V{}, clamped);
    value = flags & RJ_ARITH_FP_DX10_CLAMP_OFF ? choose(nan, value, clamped) : clamped;
  }
  return value;
}

// Evaluate the selected lanes, preserving inactive destination lanes. Sources
// can alias either destination half, so every write waits for the last load.
template <Fp64 Op, typename V, typename S>
void run(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
         const uint32_t *const *a, bool quiet_snan) {
  constexpr int Lanes = sizeof(V) / sizeof(uint64_t);
  uint32_t result[2][32];
  for (int base = 0; base < 32; base += Lanes) {
    V bits = load(a[0] + base, V{}) | (load(a[1] + base, V{}) << 32);
    if (mode & RJ_ARITH_ALU_ABS_A) {
      bits &= ~kSign;
    }
    if (mode & RJ_ARITH_ALU_NEG_A) {
      bits ^= kSign;
    }
    if (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) {
      bits = choose((V)((bits & ~kSign) < kMinNormal), bits & kSign, bits);
    }
    V value = output(evaluate<Op, V, S>(bits, quiet_snan), flags, mode);
    store(result[0] + base, value);
    store(result[1] + base, value >> 32);
  }
  for (int reg = 0; reg < 2; ++reg) {
    for (int lane = 0; lane < 32; lane += 8) {
      auto mask = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                                    _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
      _mm256_maskstore_epi32(
          reinterpret_cast<int *>(d[reg] + lane), mask,
          _mm256_loadu_si256(reinterpret_cast<const __m256i *>(result[reg] + lane)));
    }
  }
}

// Dispatch the operation on vectors of type V with signed counterpart S.
template <typename V, typename S>
void dispatch(Fp64 op, uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
              const uint32_t *const *a, bool quiet_snan) {
  switch (op) {
  case Fp64::Rcp:
    return run<Fp64::Rcp, V, S>(flags, exec_mask, mode, d, a, quiet_snan);
  case Fp64::Rsq:
    return run<Fp64::Rsq, V, S>(flags, exec_mask, mode, d, a, quiet_snan);
  default:
    return run<Fp64::Sqrt, V, S>(flags, exec_mask, mode, d, a, quiet_snan);
  }
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::trans_f64_simd
