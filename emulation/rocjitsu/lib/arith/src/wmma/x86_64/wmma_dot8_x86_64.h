// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Bit-exact gfx12 FP8/BF8 DOT8 arithmetic across independent WMMA output columns.
#pragma once

#include "common/compiler.h"
#include "dot/dot.h"
#include "dot/x86_64/exact_dot_simd.h"

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::wmma_dot8 {

using namespace exact_dot;

#if defined(RJ_ARITH_TARGET_X86_64_V4)
using Vec = __m512i;
inline Vec floor_shift(Vec x, Vec n) { return _mm512_srav_epi32(x, n); }
inline Vec mul_units(int32_t a_unit, Vec b_units) { return mul(constant<Vec>(a_unit), b_units); }
inline Vec pack_f32(Vec total, Vec grid, Vec acc, Vec magnitude) {
  // Round once with embedded RNE, then scale the normal result by 2^grid.
  constexpr int Nearest = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
  auto bits = _mm512_castps_si512(_mm512_cvt_roundepi32_ps(total, Nearest));
  auto normal =
      _mm512_maskz_add_epi32(_mm512_test_epi32_mask(total, total), bits, shift_left(grid, 23));
  // The smallest nonzero BF8 factor has alignment exponent -14. Two Q12
  // factors give a minimum product grid of 2 * (-14 - 12) = -52. Below it,
  // both groups sum to zero and C passes through, with either zero sign cleared.
  __m512i passthrough = _mm512_maskz_mov_epi32(_mm512_test_epi32_mask(magnitude, magnitude), acc);
  return _mm512_mask_mov_epi32(normal, _mm512_cmplt_epi32_mask(grid, constant<Vec>(-52)),
                               passthrough);
}
#else
using Vec = __m256i;
inline Vec floor_shift(Vec x, Vec n) { return _mm256_srav_epi32(x, n); }
// All FP8/BF8 Q12 units (and +/-Infinity, +/-Nan) fit in signed int16_t.
// Zero-extending a_unit lets VPMADDWD compute exact 32-bit products at 2x throughput.
inline Vec mul_units(int32_t a_unit_u16, Vec b_units) {
  return _mm256_madd_epi16(constant<Vec>(a_unit_u16), b_units);
}
inline Vec pack_f32(Vec total, Vec grid, Vec acc, Vec magnitude) {
  // Minimum nonzero FP8/BF8 product grid is -52 and max biased exponent is < 163,
  // so every nonzero normal total rounds to a normal FP32 with one VCVTDQ2PS.
  __m256i bits = _mm256_castps_si256(_mm256_cvtepi32_ps(total));
  __m256i normal = andnot(equal(total, constant<Vec>(0)), add(bits, shift_left(grid, 23)));
  return choose(greater(constant<Vec>(-52), grid), andnot(equal(magnitude, constant<Vec>(0)), acc),
                normal);
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

#if defined(RJ_ARITH_TARGET_X86_64_V4)
// 128-entry int16_t LUTs for OCP FP8 (E4M3FN) and OCP BF8 (E5M2) magnitudes [0..127].
struct Lut128 {
  alignas(64) int16_t units[128];
  alignas(64) int16_t grid[128];
};

template <bool Bf8>
inline constexpr Lut128 kLut = [] {
  Lut128 t{};
  for (unsigned mag = 0; mag < 128; ++mag) {
    Encoding<Bf8> v{uint16_t(mag)};
    int u = v.nan()   ? Nan
            : v.inf() ? Infinity
                      : int(v.significand()) << (12 - Encoding<Bf8>::fraction_bits);
    t.units[mag] = int16_t(u);
    t.grid[mag] = int16_t(u ? v.alignment_exponent() - 12 : Empty);
  }
  return t;
}();

// Decode 4 packed K-slices (16 lanes x 4 bytes) from one 512-bit word vector
// using 16-bit VPERMI2W table lookups (32 elements per lookup pair).
template <bool Bf8> inline void decode_word4(Vec words, Factor (&out)[4], Vec &largest16) {
  const __m512i ut0 = _mm512_load_si512(kLut<Bf8>.units + 0);
  const __m512i ut1 = _mm512_load_si512(kLut<Bf8>.units + 32);
  const __m512i ut2 = _mm512_load_si512(kLut<Bf8>.units + 64);
  const __m512i ut3 = _mm512_load_si512(kLut<Bf8>.units + 96);
  const __m512i gt0 = _mm512_load_si512(kLut<Bf8>.grid + 0);
  const __m512i gt1 = _mm512_load_si512(kLut<Bf8>.grid + 32);
  const __m512i gt2 = _mm512_load_si512(kLut<Bf8>.grid + 64);
  const __m512i gt3 = _mm512_load_si512(kLut<Bf8>.grid + 96);
  const __m512i zero = _mm512_setzero_si512();
  const __m512i mask_mag = _mm512_set1_epi16(0x007f);
  const __m512i bit6 = _mm512_set1_epi16(0x0040);
  const __m512i bit7 = _mm512_set1_epi16(0x0080);

  // Split 16x4 bytes into two 32xuint16 vectors: even = (b0, b2), odd = (b1, b3).
  __m512i pairs[2] = {_mm512_and_si512(words, _mm512_set1_epi16(0x00ff)),
                      _mm512_srli_epi16(words, 8)};
  RJ_ARITH_UNROLL(2)
  for (int p = 0; p < 2; ++p) {
    __m512i bytes = pairs[p];
    largest16 = _mm512_max_epu16(largest16, _mm512_and_si512(bytes, mask_mag));
    __mmask32 hi = _mm512_test_epi16_mask(bytes, bit6);
    __mmask32 neg = _mm512_test_epi16_mask(bytes, bit7);
    // VPERMI2W indexes by bits [5:0] of each 16-bit lane, ignoring bits [15:6].
    __m512i u_lo = _mm512_permutex2var_epi16(ut0, bytes, ut1);
    __m512i u_hi = _mm512_permutex2var_epi16(ut2, bytes, ut3);
    __m512i u16 = _mm512_mask_mov_epi16(u_lo, hi, u_hi);
    u16 = _mm512_mask_sub_epi16(u16, neg, zero, u16);

    __m512i g_lo = _mm512_permutex2var_epi16(gt0, bytes, gt1);
    __m512i g_hi = _mm512_permutex2var_epi16(gt2, bytes, gt3);
    __m512i g16 = _mm512_mask_mov_epi16(g_lo, hi, g_hi);

    out[p].units = _mm512_srai_epi32(_mm512_slli_epi32(u16, 16), 16);
    out[p].grid = _mm512_srai_epi32(_mm512_slli_epi32(g16, 16), 16);
    out[p + 2].units = _mm512_srai_epi32(u16, 16);
    out[p + 2].grid = _mm512_srai_epi32(g16, 16);
  }
}

template <bool Bf8> inline uint32_t special_mask_from_largest16(Vec largest16) {
  // Fold even/odd 16-bit halves within each 32-bit lane before comparing.
  __m512i folded = _mm512_max_epu16(largest16, _mm512_srli_epi32(largest16, 16));
  __m512i lane_max = _mm512_and_si512(folded, constant<Vec>(0x7f));
  return uint32_t(_mm512_cmpge_epi32_mask(lane_max, constant<Vec>(FirstSpecial<Bf8>)));
}
#else
// Decode packed operand bytes into signed Q12 units and their alignment grids.
template <bool Bf8> inline Factor decode_factor(Vec bits, Vec &largest) {
  constexpr int F = Encoding<Bf8>::fraction_bits, Bias = Encoding<Bf8>::bias;
  Vec magnitude = bit_and(bits, constant<Vec>(127));
  largest = maximum(largest, magnitude);
  Vec exp = shift_right(magnitude, F),
      frac = bit_and(magnitude, constant<Vec>(Encoding<Bf8>::fraction_mask));
  Vec units = shift_left(bit_or(frac, _mm256_sign_epi32(constant<Vec>(1 << F), exp)), 12 - F);
  Vec grid = choose(equal(magnitude, constant<Vec>(0)), constant<Vec>(Empty),
                    sub(maximum(exp, constant<Vec>(1)), constant<Vec>(Bias + 12)));
  if constexpr (Bf8) {
    units = choose(equal(magnitude, constant<Vec>(Encoding<Bf8>::infinity)),
                   constant<Vec>(Infinity), units);
  }
  units = choose(greater(magnitude, constant<Vec>(FirstNan<Bf8> - 1)), constant<Vec>(Nan), units);
  Vec sign_src = shift_left(bits, 24);
  // Negate units when bit 7 is set (note: bits == 0x80 -> units == 0 -> -0 == 0).
  Vec sign = signed_shift_right(sign_src, 31);
  return {sub(bit_xor(units, sign), sign), grid};
}

template <bool Bf8> inline void decode_word4(Vec words, Factor (&out)[4], Vec &largest) {
  RJ_ARITH_UNROLL(4)
  for (int i = 0; i < 4; ++i) {
    Vec bits = bit_and(shift_right(words, 8 * i), constant<Vec>(255));
    out[i] = decode_factor<Bf8>(bits, largest);
  }
}

template <bool Bf8> inline uint32_t special_mask_from_largest16(Vec largest) {
  return uint32_t(lane_mask(greater(largest, constant<Vec>(FirstSpecial<Bf8> - 1))));
}
#endif

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
  Vec shift = sub(grid, cgrid);
#if defined(RJ_ARITH_TARGET_X86_64_V4)
  __m512i cf3 = bit_and(shift_left(acc, 3), constant<Vec>(0x7fffff << 3));
  __m512i cu = _mm512_mask_or_epi32(cf3, _mm512_test_epi32_mask(acc, constant<Vec>(255 << 23)), cf3,
                                    constant<Vec>(1 << 26));
  cu = _mm512_mask_sub_epi32(cu, _mm512_movepi32_mask(acc), constant<Vec>(0), cu);
  Vec total = _mm512_maskz_srav_epi32(_mm512_cmplt_epi32_mask(shift, constant<Vec>(27)), cu, shift);
  total = add(
      total, add(floor_shift(lo.sum, sub(grid, lo.grid)), floor_shift(hi.sum, sub(grid, hi.grid))));
  Vec result = pack_f32(total, grid, acc, mag);
  __mmask16 special_c = _mm512_cmpeq_epi32_mask(exp, constant<Vec>(255));
  __m512i quiet_c =
      _mm512_mask_or_epi32(acc, _mm512_cmpgt_epi32_mask(mag, constant<Vec>(0x7f800000)), acc,
                           constant<Vec>(gfx12_dot_detail::kQuietNanBit));
  return _mm512_mask_mov_epi32(result, special_c, quiet_c);
#else
  Vec cf3 = shift_left(bit_and(acc, constant<Vec>(0x7fffff)), 3);
  Vec cu = bit_or(cf3, _mm256_sign_epi32(constant<Vec>(1 << 26), exp));
  cu = _mm256_sign_epi32(cu, acc);
  Vec total = bit_and(greater(constant<Vec>(27), shift), floor_shift(cu, shift));
  total = add(
      total, add(floor_shift(lo.sum, sub(grid, lo.grid)), floor_shift(hi.sum, sub(grid, hi.grid))));
  Vec result = pack_f32(total, grid, acc, mag);
  Vec special_c = equal(exp, constant<Vec>(255));
  Vec quiet = bit_and(greater(mag, constant<Vec>(0x7f800000)),
                      constant<Vec>(gfx12_dot_detail::kQuietNanBit));
  return choose(special_c, bit_or(acc, quiet), result);
#endif
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
