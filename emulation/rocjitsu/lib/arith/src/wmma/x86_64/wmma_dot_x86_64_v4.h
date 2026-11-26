// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 integer-grid reduction for a row of 16 independent WMMA outputs.
#pragma once

#include "dot/dot.h"
#include "dot/x86_64/exact_dot_x86_64.h"

#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::wmma_v4 {

// Broadcast one signed grid exponent or register bit pattern to every column.
inline __m512i splat(int value) { return _mm512_set1_epi32(value); }

// Packed FP16 grids carry the FP32 exponent bias, so the stage rounding scales
// its total with one shift. The B factors and C supply the bias.
template <bool Bf16, bool Packed> inline constexpr int kWmmaGridBias = Packed && !Bf16 ? 127 : 0;

// Packed partial sums stay sign-extended between stages, so bit 31 is the sign
// of C in every format.
template <bool Packed> inline __m512i wmma_extend(__m512i bits) {
  using namespace exact_dot;
  return Packed ? signed_shift_right(shift_left(bits, 16), 16) : bits;
}

// Round the signed grid total once at the architectural stage boundary. Bit 31
// of frame flips the sign of the result.
template <bool Bf16, bool Rdna3, bool Packed>
inline __m512i wmma_pack(__m512i total, __m512i grid, __m512i frame, bool saturate) {
  using namespace exact_dot;
  constexpr int Nearest = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
  constexpr int Sign = int(0x80000000u);
  if constexpr (Packed && !Bf16) {
    // Truncate the signed total to 24 bits and fold the lost bits into the last
    // place. Rounding to odd keeps two bits below every FP16 rounding position,
    // so VCVTPS2PH performs the one RNE rounding, gradual underflow and
    // overflow. Embedded rounding with SAE does not touch MXCSR.
    auto truncated = _mm512_cvt_roundepi32_ps(total, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    auto inexact =
        _mm512_cmpneq_epi32_mask(_mm512_cvtt_roundps_epi32(truncated, _MM_FROUND_NO_EXC), total);
    // Scaling by a signed power of two is exact and keeps a zero total zero.
    auto scale = bit_xor(shift_left(grid, 23), bit_and(frame, splat(Sign)));
    auto value = _mm512_mul_round_ps(truncated, _mm512_castsi512_ps(scale), Nearest);
    // Clamping the magnitude to the largest finite FP16 value saturates.
    auto limit = _mm512_castsi512_ps(splat(saturate ? 0x477fe000 : 0x7f800000));
    value = _mm512_range_round_ps(value, limit, 2, _MM_FROUND_NO_EXC);
    // The sticky bit lands in the same last place after scaling and clamping,
    // which keeps the inexact test off the critical path.
    auto odd = _mm512_castps_si512(value);
    odd = _mm512_mask_or_epi32(odd, inexact, odd, splat(1));
    auto half = _mm512_cvt_roundps_ph(_mm512_castsi512_ps(odd), Nearest);
    // A result that rounds to zero is positive zero.
    return _mm512_maskz_cvtepi16_epi32(_mm256_test_epi16_mask(half, _mm256_set1_epi16(0x7fff)),
                                       half);
  }
  // RNE conversion rounds the integer significand once. Adding grid to the
  // resulting exponent is exact in the normal range and does not touch MXCSR.
  auto rounded = _mm512_castps_si512(_mm512_cvt_roundepi32_ps(total, Nearest));
  auto magnitude = bit_and(rounded, splat(0x7fffffff));
  auto exponent = add(shift_right(magnitude, 23), grid);
  // An overflowing exponent stays above infinity as an unsigned pattern.
  auto bits = _mm512_min_epu32(add(magnitude, shift_left(grid, 23)), splat(0x7f800000));
  auto keep = _mm512_test_epi32_mask(total, total);
  if constexpr (Rdna3) {
    keep = _mm512_mask_cmpgt_epi32_mask(keep, exponent, splat(0));
  } else if (_mm512_mask_cmple_epi32_mask(keep, exponent, splat(0))) [[unlikely]] {
    // Gradual underflow requires rounding directly on the subnormal grid.
    return wmma_extend<Packed>(pack<Bf16, Rdna3, Packed>(total, grid, frame, saturate));
  }
  bits = _mm512_maskz_ternarylogic_epi32(keep, bits, Rdna3 ? bit_xor(rounded, frame) : rounded,
                                         splat(Sign), 0xf8);
  if constexpr (Packed) {
    bits = signed_shift_right(bits, 16);
  }
  return bits;
}

// A finite magnitude uses twelve fractional bits. Multiplying two decoded
// significands puts the product directly on the grid at ea + eb - 24.
struct WmmaFactor {
  __m512i bits;
  __m512i units;
  __m512i grid;
  __m512i sign;
};

// Decode sixteen factors, retaining sign masks for gfx11 and signed units for gfx12.
template <bool Bf16, bool Rdna3> inline WmmaFactor wmma_factor(__m512i bits, int grid_bias = 0) {
  using namespace exact_dot;
  constexpr int Fraction = Bf16 ? 7 : 10, Bias = Bf16 ? 127 : 15;
  auto magnitude = bit_and(bits, splat(0x7fff));
  auto exponent = shift_right(magnitude, Fraction);
  auto fraction = bit_and(magnitude, splat((1 << Fraction) - 1));
  auto units = _mm512_mask_or_epi32(Bf16 && Rdna3 ? splat(0) : fraction,
                                    _mm512_test_epi32_mask(exponent, exponent), fraction,
                                    splat(1 << Fraction));
  // Even paired with the largest BF16 exponent, an empty factor stays below
  // the smallest nonzero product grid.
  auto grid = _mm512_mask_mov_epi32(splat(gfx12_dot_detail::kEmptyExponent),
                                    _mm512_test_epi32_mask(units, units),
                                    sub(maximum(exponent, splat(1)), splat(Bias + 12 - grid_bias)));
  __m512i sign = splat(0);
  if constexpr (Rdna3) {
    sign = signed_shift_right(shift_left(bits, 16), 31);
  } else {
    units =
        _mm512_mask_sub_epi32(units, _mm512_test_epi32_mask(bits, splat(0x8000)), splat(0), units);
  }
  return {bits, shift_left(units, 12 - Fraction), grid, sign};
}

// Resolve exceptional results in their destination format. A factor NaN
// takes priority over invalid products, which in turn take priority over C.
template <bool Bf16, bool Rdna3, bool Packed, int N>
inline __m512i wmma_special(const WmmaFactor (&a)[N], const WmmaFactor (&b)[N], __m512i factor_max,
                            __m512i acc) {
  using namespace exact_dot;
  constexpr int Infinity = Bf16 ? 0x7f80 : 0x7c00;
  constexpr int ZeroLimit = Bf16 && Rdna3 ? 127 : 0;
  constexpr int CInfinity = Packed ? Infinity : 0x7f800000;
  constexpr int CSign = Packed ? int(0xffff8000u) : int(0x80000000u);
  constexpr int Quiet = Packed ? (Bf16 ? 0x40 : 0x200) : gfx12_dot_detail::kQuietNanBit;
  constexpr auto FactorNan = gfx12_dot_detail::kFactorNan;
  constexpr auto InvalidNan = gfx12_dot_detail::kInvalidProductNan;
  auto cnan = _mm512_cmpgt_epi32_mask(andnot(splat(CSign), acc), splat(CInfinity));
  auto result = _mm512_mask_or_epi32(acc, cnan, acc, splat(Quiet));
  auto infinity = _mm512_cmpeq_epi32_mask(factor_max, splat(Infinity));
  if (infinity) {
    __mmask16 positive = 0, negative = 0, invalid = 0;
    RJ_ARITH_UNROLL(4)
    for (int j = 0; j < N; ++j) {
      auto am = bit_and(a[j].bits, splat(0x7fff)), bm = bit_and(b[j].bits, splat(0x7fff));
      auto ai = _mm512_cmpeq_epi32_mask(am, splat(Infinity));
      auto bi = _mm512_cmpeq_epi32_mask(bm, splat(Infinity));
      invalid |= (ai & _mm512_cmple_epi32_mask(bm, splat(ZeroLimit))) |
                 (bi & _mm512_cmple_epi32_mask(am, splat(ZeroLimit)));
      auto sign = _mm512_test_epi32_mask(bit_xor(a[j].bits, b[j].bits), splat(0x8000));
      positive |= (ai | bi) & ~sign;
      negative |= (ai | bi) & sign;
    }
    invalid |= positive & negative;
    positive |= _mm512_cmpeq_epi32_mask(acc, splat(CInfinity));
    negative |= _mm512_cmpeq_epi32_mask(acc, splat(CSign | CInfinity));
    auto inf = _mm512_mask_blend_epi32(negative, splat(CInfinity), splat(CSign | CInfinity));
    result = _mm512_mask_mov_epi32(result, infinity & ~cnan, inf);
    result = _mm512_mask_set1_epi32(result, infinity & (invalid | (positive & negative & ~cnan)),
                                    Packed ? int(int16_t(packed16::special<Bf16>(InvalidNan)))
                                           : int(InvalidNan));
  }
  return _mm512_mask_set1_epi32(result, _mm512_cmpgt_epi32_mask(factor_max, splat(Infinity)),
                                Packed ? int(int16_t(packed16::special<Bf16>(FactorNan)))
                                       : int(FactorNan));
}

// Put every product on its own alignment grid first. All subsequent alignments
// are right shifts, including the signed floors of gfx12 product pairs. The
// result is valid for finite lanes. Lanes with an infinite or NaN C are flagged
// in special and, like exceptional factors, are resolved by wmma_special.
template <bool Bf16, bool Rdna3, bool Packed>
inline __m512i wmma_dot(const WmmaFactor (&a)[Rdna3 ? 2 : 4], const WmmaFactor (&b)[Rdna3 ? 2 : 4],
                        __m512i acc, bool saturate, __mmask16 &special) {
  using namespace exact_dot;
  constexpr int N = Rdna3 ? 2 : 4;
  constexpr bool Half = Packed && !Bf16;
  __m512i values[N], grids[N];
  constexpr int CFrac = Packed ? (Bf16 ? 7 : 10) : 23;
  constexpr int CBias = Half ? 15 : 127;
  RJ_ARITH_UNROLL(4)
  for (int j = 0; j < N; ++j) {
    grids[j] = add(a[j].grid, b[j].grid);
    values[j] = mul(a[j].units, b[j].units);
  }
  auto ce = bit_and(shift_right(acc, CFrac), splat(2 * CBias + 1));
  auto cf = bit_and(acc, splat((1 << CFrac) - 1));
  auto normal = _mm512_test_epi32_mask(acc, splat((2 * CBias + 1) << CFrac));
  special = _mm512_cmpeq_epi32_mask(ce, splat(2 * CBias + 1));
  constexpr bool FlushC = Rdna3 && !Half;
  auto cm = _mm512_mask_or_epi32(FlushC ? splat(0) : cf, normal, cf, splat(1 << CFrac));
  auto cvalues = shift_left(cm, 26 - CFrac);
  // Unless C is flushed, its grid is at least the subnormal grid, which is the
  // floor of every stage grid. Empty products can only fall further below it.
  auto cgrid = add(maximum(ce, splat(1)), splat(kWmmaGridBias<Bf16, Packed> - CBias - 26));
  if constexpr (Rdna3) {
    // A product's leading exponent is at most the sum of its factor exponents
    // plus one. Its own grid (sum minus 24) already dominates leading minus 26.
    auto csign = signed_shift_right(acc, 31);
    auto grid = maximum(
        maximum(grids[0], grids[1]),
        FlushC ? _mm512_mask_mov_epi32(splat(gfx12_dot_detail::kEmptyExponent), normal, cgrid)
               : cgrid);
    auto total = shift_right(cvalues, sub(grid, cgrid));
    RJ_ARITH_UNROLL(2)
    for (int j = 0; j < N; ++j) {
      auto m = shift_right(values[j], sub(grid, grids[j]));
      total = add(total, _mm512_ternarylogic_epi32(m, bit_xor(a[j].sign, b[j].sign), csign, 0x96));
    }
    // All-zero inputs leave at most two units on a grid that rounds to zero,
    // and the stage rounding returns positive zero for them.
    return wmma_pack<Bf16, Rdna3, Packed>(total, grid, csign, saturate);
  } else {
    __m512i pair_grids[N / 2];
    RJ_ARITH_UNROLL(2)
    for (int j = 0; j < N; j += 2) {
      pair_grids[j / 2] = maximum(grids[j], grids[j + 1]);
    }
    auto grid = maximum(maximum(pair_grids[0], pair_grids[1]), cgrid);
    auto cshift = sub(grid, cgrid);
    // gfx12 floors signed terms at each alignment. It drops a fully shifted
    // accumulator when the grid exceeds the minimum normal exponent. The hidden
    // bit of a normal C is bit 26 and a subnormal C sits on the floor grid, so
    // both cases reduce to a shift past bit 26.
    auto signed_c = _mm512_mask_sub_epi32(cvalues, _mm512_movepi32_mask(acc), splat(0), cvalues);
    auto total =
        _mm512_maskz_srav_epi32(_mm512_cmplt_epi32_mask(cshift, splat(27)), signed_c, cshift);
    RJ_ARITH_UNROLL(2)
    for (int j = 0; j < N; j += 2) {
      auto pg = pair_grids[j / 2];
      auto pair = add(_mm512_srav_epi32(values[j], sub(pg, grids[j])),
                      _mm512_srav_epi32(values[j + 1], sub(pg, grids[j + 1])));
      total = add(total, _mm512_srav_epi32(pair, sub(grid, pg)));
    }
    return wmma_pack<Bf16, Rdna3, Packed>(total, grid, splat(0), saturate);
  }
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::wmma_v4
