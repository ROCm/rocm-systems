// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Bit-exact gfx12 FP8/BF8 DOT8 arithmetic across independent WMMA output columns.
#pragma once

#include "common/compiler.h"
#if defined(__AVX512F__)
#include "dot/dot.h"
#include "dot/x86_64/exact_dot_simd.h"
#else
#include "wmma/x86_64/wmma_exact_x86_64.h"
#endif

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::wmma_dot8 {

using namespace exact_dot;

#if defined(__AVX512F__)
using Vec = __m512i;
inline Vec floor_shift(Vec x, Vec n) { return _mm512_srav_epi32(x, n); }
inline Vec pack_f32(Vec total, Vec grid, Vec acc, Vec magnitude) {
  // Round once with embedded RNE, then scale the normal result by 2^grid.
  constexpr int Nearest = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
  auto bits = _mm512_castps_si512(_mm512_cvt_roundepi32_ps(total, Nearest));
  auto normal =
      _mm512_maskz_add_epi32(_mm512_test_epi32_mask(total, total), bits, shift_left(grid, 23));
  // The smallest nonzero BF8 factor has alignment exponent -14. Two Q12
  // factors give a minimum product grid of 2 * (-14 - 12) = -52. Below it,
  // both groups sum to zero and C passes through, with either zero sign cleared.
  return choose(greater(constant<Vec>(-52), grid), andnot(equal(magnitude, constant<Vec>(0)), acc),
                normal);
}
#else
using Vec = __m256i;
inline Vec floor_shift(Vec x, Vec n) { return _mm256_srav_epi32(x, n); }
inline Vec pack_f32(Vec total, Vec grid, Vec, Vec) {
  // The dense AVX2 FP32 packer expects its grid one below the unit exponent.
  return wmma_v3::pack_stage<false, false, false>(total, sub(grid, constant<Vec>(1)),
                                                  constant<Vec>(0), false);
}
#endif

constexpr int Columns = sizeof(Vec) / sizeof(uint32_t);
constexpr int Empty = gfx12_dot_detail::kEmptyExponent;
// Finite significands have magnitude < 8192. These tags let the cold path
// classify both byte formats using their decoded units alone.
constexpr int Infinity = 8192, Nan = 8193;

struct ScalarFactor {
  // Full-width fields allow direct vector broadcasts of the row factor.
  int32_t units;
  int32_t grid;
};
struct Factor {
  Vec units;
  Vec grid;
};
template <bool Bf8>
using Encoding = gfx12_dot_detail::Factor<Bf8 ? gfx12_dot_detail::InputFormat::Bf8
                                              : gfx12_dot_detail::InputFormat::Fp8>;

template <bool Bf8>
constexpr int FirstNan = Bf8 ? Encoding<Bf8>::infinity + 1 : Encoding<Bf8>::magnitude_mask;
template <bool Bf8> constexpr int FirstSpecial = Bf8 ? Encoding<Bf8>::infinity : FirstNan<Bf8>;

// Decode packed operand bytes into signed Q12 units and their alignment grids.
template <bool Bf8> inline Factor decode_factor(Vec bits, Vec &largest) {
  constexpr int F = Encoding<Bf8>::fraction_bits, Bias = Encoding<Bf8>::bias;
  Vec magnitude = bit_and(bits, constant<Vec>(127));
  largest = maximum(largest, magnitude);
  Vec exp = shift_right(magnitude, F),
      frac = bit_and(magnitude, constant<Vec>(Encoding<Bf8>::fraction_mask));
  Vec units =
      shift_left(bit_or(frac, andnot(equal(exp, constant<Vec>(0)), constant<Vec>(1 << F))), 12 - F);
  Vec grid = choose(equal(units, constant<Vec>(0)), constant<Vec>(Empty),
                    sub(maximum(exp, constant<Vec>(1)), constant<Vec>(Bias + 12)));
  if constexpr (Bf8) {
    units = choose(equal(magnitude, constant<Vec>(Encoding<Bf8>::infinity)),
                   constant<Vec>(Infinity), units);
  }
  units = choose(greater(magnitude, constant<Vec>(FirstNan<Bf8> - 1)), constant<Vec>(Nan), units);
  Vec sign = signed_shift_right(shift_left(bits, 24), 31);
  return {sub(bit_xor(units, sign), sign), grid};
}

struct Group {
  Vec sum;
  Vec grid;
};

// Finite products have magnitude below 2^26. Their four-term sums and the
// final eight terms with C fit signed 32-bit lanes.
// Each group of four first aligns its signed products to its own maximum
// grid. Arithmetic shifts implement floor, including negative discarded units.
inline Group align4(const Vec *products, const Vec *grids) {
  Vec grid = maximum(maximum(grids[0], grids[1]), maximum(grids[2], grids[3]));
  Vec aligned[4];
  RJ_ARITH_UNROLL(4)
  for (int i = 0; i < 4; ++i) {
    aligned[i] = floor_shift(products[i], sub(grid, grids[i]));
  }
  Vec sum = add(add(aligned[0], aligned[1]), add(aligned[2], aligned[3]));
  return {sum, grid};
}

// Nonfinite products seen so far within one DOT8 stage.
struct Exceptions {
  Vec invalid;
  Vec positive;
  Vec negative;
};

// Record a factor NaN, a zero times infinity, or a signed infinite product
// from the decoded units x and y.
inline void classify(Exceptions &seen, Vec x, Vec y) {
  Vec ax = absolute(x), ay = absolute(y);
  Vec xi = equal(ax, constant<Vec>(Infinity)), yi = equal(ay, constant<Vec>(Infinity));
  seen.invalid = bit_or(seen.invalid, greater(maximum(ax, ay), constant<Vec>(Infinity)));
  seen.invalid = bit_or(seen.invalid, bit_or(bit_and(xi, equal(ay, constant<Vec>(0))),
                                             bit_and(yi, equal(ax, constant<Vec>(0)))));
  Vec inf = bit_or(xi, yi), sign = signed_shift_right(bit_xor(x, y), 31);
  seen.positive = bit_or(seen.positive, andnot(sign, inf));
  seen.negative = bit_or(seen.negative, bit_and(sign, inf));
}

// Factor NaNs and invalid products have priority over the incoming payload.
// A lone infinity of either sign propagates through the stage.
inline Vec resolve(const Exceptions &seen, Vec acc, Vec result) {
  Vec invalid = bit_or(seen.invalid, bit_and(seen.positive, seen.negative));
  Vec sign = bit_and(seen.negative, constant<Vec>(INT32_MIN));
  Vec inf = bit_or(sign, constant<Vec>(0x7f800000));
  Vec has_inf = bit_or(seen.positive, seen.negative);
  Vec magnitude = bit_and(acc, constant<Vec>(INT32_MAX));
  Vec c_nan = greater(magnitude, constant<Vec>(0x7f800000));
  Vec opposite = equal(acc, bit_xor(inf, constant<Vec>(INT32_MIN)));
  invalid = bit_or(invalid, bit_and(has_inf, opposite));
  result = choose(andnot(c_nan, has_inf), inf, result);
  return choose(invalid, constant<Vec>(int32_t(gfx12_dot_detail::kFactorNan)), result);
}

// Round two independently aligned group totals and the incoming FP32 accumulator.
inline Vec round_groups(Group lo, Group hi, Vec acc) {
  Vec mag = bit_and(acc, constant<Vec>(INT32_MAX)), exp = shift_right(mag, 23);
  Vec cgrid = sub(maximum(exp, constant<Vec>(1)), constant<Vec>(127 + 26));
  Vec grid = maximum(maximum(lo.grid, hi.grid), cgrid);
  Vec cu = bit_or(bit_and(acc, constant<Vec>(0x7fffff)),
                  andnot(equal(exp, constant<Vec>(0)), constant<Vec>(0x800000)));
  Vec sign = signed_shift_right(acc, 31);
  cu = shift_left(sub(bit_xor(cu, sign), sign), 3);
  Vec shift = sub(grid, cgrid);
  // A normal C's leading bit is 26. The empirical accumulator alignment
  // discards C entirely after that bit has shifted out above its floor.
  Vec total = bit_and(greater(constant<Vec>(27), shift), floor_shift(cu, shift));
  total = add(
      total, add(floor_shift(lo.sum, sub(grid, lo.grid)), floor_shift(hi.sum, sub(grid, hi.grid))));
  Vec result = pack_f32(total, grid, acc, mag);
  Vec special_c = equal(exp, constant<Vec>(255));
  Vec quiet = bit_and(greater(mag, constant<Vec>(0x7f800000)),
                      constant<Vec>(gfx12_dot_detail::kQuietNanBit));
  return choose(special_c, bit_or(acc, quiet), result);
}

// Apply one DOT8 stage of broadcast A factors to the B columns selected by k.
using Result = Vec[8][32 / Columns];

inline void store_result(uint32_t *const *d, const Result &result) {
  // Whole VGPRs can alias each other. Commit snapshotted results in the same
  // ascending 32-lane section order as the scalar entry point.
  for (int reg = 0; reg < 8; ++reg) {
    for (int lane = 0; lane < 32; lane += Columns) {
      store(d[reg] + lane, result[reg][lane / Columns]);
    }
  }
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::wmma_dot8
