// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Eight-column integer-grid stages for gfx11 and gfx12 dense FP16/BF16 WMMA.
#pragma once

#include "dot/dot.h"
#include "dot/x86_64/exact_dot_x86_64.h"

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::wmma_v3 {

// Broadcast a grid exponent or register bit pattern across eight columns.
inline __m256i splat(int value) { return exact_dot::constant<__m256i>(value); }

struct Factor {
  __m256i bits, units, grid, sign;
};

// Stage grids carry the destination bias less 128, so that the float exponent
// of a total plus its grid is the encoded exponent minus one. The B factors
// and the accumulator apply it; A grids stay unbiased.
template <bool Bf16, bool Packed>
inline constexpr int kGridBias = (Packed && !Bf16 ? 15 : 127) - 128;

// Scale significands to twelve fraction bits. The product grid is ea + eb - 24.
// Every lane holds one zero-extended 16-bit encoding.
template <bool Bf16, bool Rdna3> inline Factor decode(__m256i bits, int grid_bias = 0) {
  using namespace exact_dot;
  constexpr int Fraction = Bf16 ? 7 : 10, Bias = Bf16 ? 127 : 15;
  auto magnitude = bit_and(bits, splat(0x7fff));
  auto exponent = shift_right(magnitude, Fraction);
  auto fraction = bit_and(magnitude, splat((1 << Fraction) - 1));
  // VPSIGND by the exponent field clears whatever a zero exponent drops.
  auto hidden = splat(1 << Fraction);
  auto units = Bf16 && Rdna3 ? _mm256_sign_epi32(bit_or(fraction, hidden), exponent)
                             : bit_or(fraction, _mm256_sign_epi32(hidden, exponent));
  auto grid = choose(greater(units, splat(0)),
                     sub(maximum(exponent, splat(1)), splat(Bias + 12 - grid_bias)),
                     splat(gfx12_dot_detail::kEmptyExponent));
  auto sign = greater(bits, splat(0x7fff));
  units = shift_left(units, 12 - Fraction);
  if constexpr (!Rdna3) {
    units = sub(bit_xor(units, sign), sign);
  }
  return {bits, units, grid, sign};
}

// The biased leading bit: 127 plus its index, or zero for an empty lane. For
// magnitudes above 24 bits, dropping the low byte preserves that bit and makes
// VCVTDQ2PS exact in every host rounding mode.
inline __m256i biased_top_bit(__m256i magnitude) {
  using namespace exact_dot;
  auto large = greater(magnitude, splat(0xffffff));
  auto exact = andnot(bit_and(large, splat(255)), magnitude);
  return float_exponent(exact);
}

// Round on the biased destination grid. gfx11 FP32/BF16 flushes after
// rounding, whereas gfx12 and packed FP16 round directly on the subnormal grid.
template <bool Bf16, bool Rdna3, bool Packed>
inline __m256i pack_stage(__m256i total, __m256i grid, __m256i frame, bool saturate) {
  using namespace exact_dot;
  constexpr bool Half = Packed && !Bf16, Gradual = !Rdna3 || Half;
  constexpr int Fraction = Half ? 10 : 23, Shift = 30 - Fraction;
  auto magnitude = absolute(total);
  auto top = biased_top_bit(magnitude);
  // The encoded exponent minus one, with the leading bit normalized to bit 30.
  auto exponent = add(top, grid);
  auto normalized = shift_left(magnitude, sub(splat(127 + 30), top));
  // VPSIGND keeps a lane while its keep word is positive and clears it at zero.
  __m256i bits, keep = magnitude;
  if (!Gradual ||
      !_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_sign_epi32(exponent, magnitude)))) {
    // The retained hidden bit and any rounding carry both advance the exponent
    // field. Half an ULP minus one, plus the retained low bit, rounds ties to even.
    auto parity = bit_and(shift_right(normalized, Shift), splat(1));
    auto rounded = shift_right(add(normalized, add(splat((1 << (Shift - 1)) - 1), parity)), Shift);
    bits = add(shift_left(exponent, Fraction), rounded);
    if constexpr (!Gradual) {
      // A carry at exponent zero can still round to the minimum normal.
      keep = bit_and(keep, greater(add(exponent, shift_right(rounded, Fraction)), splat(0)));
    }
  } else {
    // A tiny lane rounds directly on the destination's subnormal grid.
    auto guard = add(splat(29 - Fraction), maximum(sub(splat(0), exponent), splat(0)));
    auto extra = shift_right(normalized, guard);
    auto retained = shift_right(extra, 1);
    auto sticky = nonzero(bit_xor(normalized, shift_left(extra, guard)));
    auto increment = bit_and(bit_and(extra, splat(1)), bit_or(sticky, retained));
    bits = add(shift_left(maximum(exponent, splat(0)), Fraction), add(retained, increment));
    keep = bit_and(keep, nonzero(bits));
  }
  bits = _mm256_min_epu32(bits, splat(Half ? (saturate ? 0x7bff : 0x7c00) : 0x7f800000));
  auto sign = Half ? bit_and(bit_xor(signed_shift_right(total, 31), frame), splat(0x8000))
                   : bit_and(bit_xor(total, frame), splat(int(0x80000000U)));
  bits = _mm256_sign_epi32(bit_or(bits, sign), keep);
  if constexpr (Packed && Bf16) {
    bits = shift_right(bits, 16);
  }
  return bits;
}

// Resolve factor NaNs, invalid products and accumulator NaNs in empirical order.
template <bool Bf16, bool Rdna3, bool Packed, int N>
inline __m256i special(const Factor (&a)[N], const Factor (&b)[N], __m256i factor_max,
                       __m256i acc) {
  using namespace exact_dot;
  constexpr int Infinity = Bf16 ? 0x7f80 : 0x7c00, Zero = Bf16 && Rdna3 ? 127 : 0;
  constexpr int CInfinity = Packed ? Infinity : 0x7f800000,
                Sign = Packed ? 0x8000 : int(0x80000000U);
  constexpr int Quiet = Packed ? (Bf16 ? 0x40 : 0x200) : gfx12_dot_detail::kQuietNanBit;
  constexpr auto FactorNan = gfx12_dot_detail::kFactorNan,
                 InvalidNan = gfx12_dot_detail::kInvalidProductNan;
  auto c_nan = greater(andnot(splat(Sign), acc), splat(CInfinity));
  auto result = bit_or(acc, bit_and(c_nan, splat(Quiet)));
  auto infinity = equal(factor_max, splat(Infinity));
  if (lane_mask(infinity)) {
    auto positive = splat(0), negative = splat(0), invalid = splat(0);
    RJ_ARITH_UNROLL(4)
    for (int j = 0; j < N; ++j) {
      auto am = bit_and(a[j].bits, splat(0x7fff)), bm = bit_and(b[j].bits, splat(0x7fff));
      auto a_inf = equal(am, splat(Infinity)), b_inf = equal(bm, splat(Infinity));
      invalid = bit_or(invalid, bit_or(andnot(greater(bm, splat(Zero)), a_inf),
                                       andnot(greater(am, splat(Zero)), b_inf)));
      auto inf = bit_or(a_inf, b_inf),
           sign = signed_shift_right(shift_left(bit_xor(a[j].bits, b[j].bits), 16), 31);
      positive = bit_or(positive, andnot(sign, inf));
      negative = bit_or(negative, bit_and(sign, inf));
    }
    invalid = bit_or(invalid, bit_and(positive, negative));
    positive = bit_or(positive, equal(acc, splat(CInfinity)));
    negative = bit_or(negative, equal(acc, splat(Sign | CInfinity)));
    result = choose(andnot(c_nan, infinity),
                    bit_or(splat(CInfinity), bit_and(negative, splat(Sign))), result);
    invalid = bit_and(infinity, bit_or(invalid, andnot(c_nan, bit_and(positive, negative))));
    result = choose(invalid, splat(Packed ? packed16::special<Bf16>(InvalidNan) : int(InvalidNan)),
                    result);
  }
  return choose(greater(factor_max, splat(Infinity)),
                splat(Packed ? packed16::special<Bf16>(FactorNan) : int(FactorNan)), result);
}

// Accumulate one empirical DOT2 or DOT4 on its alignment grid. The caller
// replaces nonfinite lanes using the original operands.
template <bool Bf16, bool Rdna3, bool Packed, int N>
inline __m256i stage(const Factor (&a)[N], const Factor (&b)[N], __m256i acc, bool saturate,
                     __m256i &c_exception) {
  using namespace exact_dot;
  constexpr bool Half = Packed && !Bf16, FlushC = Rdna3 && !Half;
  constexpr int CFraction = Packed ? (Bf16 ? 7 : 10) : 23, CBias = Half ? 15 : 127;
  // A flushed C relies on the empty grid sitting far below every finite grid.
  static_assert(gfx12_dot_detail::kEmptyExponent <= -512);
  __m256i grids[N], values[N];
  RJ_ARITH_UNROLL(4)
  for (int j = 0; j < N; ++j) {
    grids[j] = add(a[j].grid, b[j].grid);
    values[j] = mul(a[j].units, b[j].units);
  }
  auto ce = bit_and(shift_right(acc, CFraction), splat(2 * CBias + 1));
  auto cf = shift_left(bit_and(acc, splat((1 << CFraction) - 1)), 26 - CFraction);
  // C's biased grid is max(ce, 1) - (CBias + 26) + (CBias - 128).
  __m256i c_values, c_grid;
  if constexpr (FlushC) {
    // A flushed C keeps stale units on the empty grid: every finite grid
    // shifts them out, and a stage without finite products flushes anyway.
    c_values = bit_or(cf, splat(1 << 26));
    c_grid = choose(greater(ce, splat(0)), sub(ce, splat(128 + 26)),
                    splat(gfx12_dot_detail::kEmptyExponent));
  } else {
    c_values = bit_or(cf, _mm256_sign_epi32(splat(1 << 26), ce));
    c_grid = sub(maximum(ce, splat(1)), splat(128 + 26));
  }
  c_exception = equal(ce, splat(2 * CBias + 1));
  __m256i total, grid, frame;
  if constexpr (Rdna3) {
    // gfx11 complements each aligned product whose sign differs from C.
    frame = Packed ? greater(acc, splat(0x7fff)) : signed_shift_right(acc, 31);
    grid = maximum(maximum(grids[0], grids[1]), c_grid);
    total = shift_right(c_values, sub(grid, c_grid));
    RJ_ARITH_UNROLL(2)
    for (int j = 0; j < N; ++j) {
      auto aligned = shift_right(values[j], sub(grid, grids[j]));
      total = add(total, bit_xor(aligned, bit_xor(bit_xor(a[j].sign, b[j].sign), frame)));
    }
  } else {
    // gfx12 floors signed products on the pair grid before aligning the pair.
    // C's hidden bit is at bit 26, so a shift of 27 or more discards C.
    frame = splat(0);
    __m256i pair_grid[2] = {maximum(grids[0], grids[1]), maximum(grids[2], grids[3])};
    grid = maximum(c_grid, maximum(pair_grid[0], pair_grid[1]));
    auto shift = sub(grid, c_grid);
    auto c_signed = _mm256_sign_epi32(c_values, Packed ? shift_left(acc, 16) : acc);
    total = bit_and(greater(splat(27), shift), _mm256_srav_epi32(c_signed, shift));
    RJ_ARITH_UNROLL(2)
    for (int j = 0; j < N; j += 2) {
      auto group = pair_grid[j / 2];
      auto sum = add(_mm256_srav_epi32(values[j], sub(group, grids[j])),
                     _mm256_srav_epi32(values[j + 1], sub(group, grids[j + 1])));
      total = add(total, _mm256_srav_epi32(sum, sub(grid, group)));
    }
  }
  return pack_stage<Bf16, Rdna3, Packed>(total, grid, frame, saturate);
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::wmma_v3
