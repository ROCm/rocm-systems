// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 dense FP16/BF16 WMMA. Each DOT2 or DOT4 stage processes all 16
// independent rows branchlessly with unpacked accumulators kept across stages,
// deferring rare nonfinite/subnormal/overflow repair to a cold helper.

#include "common/simd.h"
#include "rj_arith/detail/rj_arith_wmma.h"
#include "wmma/layout.h"
#include "wmma/x86_64/wmma_dot_x86_64_v4.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {

namespace RJ_ARITH_IMPL_NAMESPACE::wmma_v4 {

// Scalar-addressable decoded A fields for one stage. Only sign is kept on Rdna3.
template <bool Rdna3> struct alignas(64) StageLeft {
  int32_t units[16];
  int32_t grid[16];
  int32_t sign[Rdna3 ? 16 : 1];
};

// Cold exact fallback for rows that encounter infinite factors, nonfinite B,
// intermediate overflow, or gradual FP32 subnormal underflow.
template <bool Bf16, bool Rdna3, bool Packed>
[[gnu::noinline]] void repair_rows(uint32_t row_mask, uint32_t instruction_flags, bool saturate,
                                   __m512i (&acc)[16], const __m512i (&c_init)[16],
                                   const uint32_t *const (&a)[8], const uint32_t *const (&b)[8]) {
  using namespace exact_dot;
  constexpr int Step = Rdna3 ? 2 : 4;
  constexpr int Infinity = Bf16 ? 0x7f80 : 0x7c00;
  for (uint32_t m = row_mask; m; m &= m - 1) {
    int row = __builtin_ctz(m);
    acc[row] = wmma_extend<Packed>(c_init[row]);
  }
  for (int step = 0; step < 16; step += Step) {
    WmmaFactor right_factors[Step];
    __m512i bmax = splat(0);
    RJ_ARITH_UNROLL(2)
    for (int p = 0; p < Step / 2; ++p) {
      int pair = step / 2 + p;
      __m512i b_packed = load<__m512i>(b[pair]);
      __m512i bh[2] = {bit_xor(bit_and(b_packed, splat(0xffff)),
                               splat(instruction_flags & RJ_ARITH_WMMA_NEG_LO_B ? 0x8000 : 0)),
                       bit_xor(shift_right(b_packed, 16),
                               splat(instruction_flags & RJ_ARITH_WMMA_NEG_HI_B ? 0x8000 : 0))};
      RJ_ARITH_UNROLL(2)
      for (int h = 0; h < 2; ++h) {
        int j = 2 * p + h;
        right_factors[j] = wmma_factor<Bf16, Rdna3>(bh[h], kWmmaGridBias<Bf16, Packed>);
        bmax = maximum(bmax, bit_and(bh[h], splat(0x7fff)));
      }
    }
    __mmask16 right_special = _mm512_cmpge_epi32_mask(bmax, splat(Infinity));
    for (uint32_t m = row_mask; m; m &= m - 1) {
      int row = __builtin_ctz(m);
      WmmaFactor left_factors[Step];
      int amax_scalar = 0;
      RJ_ARITH_UNROLL(4)
      for (int j = 0; j < Step; ++j) {
        int pair = (step + j) / 2, shift = 16 * (j % 2);
        uint32_t neg_a = j % 2 ? RJ_ARITH_WMMA_NEG_HI_A : RJ_ARITH_WMMA_NEG_LO_A;
        int abits =
            int(((a[pair][row] >> shift) & 0xffffu) ^ (instruction_flags & neg_a ? 0x8000u : 0u));
        int amag = abits & 0x7fff;
        if (amag > amax_scalar) {
          amax_scalar = amag;
        }
        left_factors[j] = wmma_factor<Bf16, Rdna3>(splat(abits));
      }
      __mmask16 special;
      auto result =
          wmma_dot<Bf16, Rdna3, Packed>(left_factors, right_factors, acc[row], saturate, special);
      special = __mmask16(special | right_special | (amax_scalar >= Infinity ? 0xffff : 0));
      if (special) {
        result = _mm512_mask_mov_epi32(
            result, special,
            wmma_special<Bf16, Rdna3, Packed>(left_factors, right_factors,
                                              maximum(bmax, splat(amax_scalar)), acc[row]));
      }
      acc[row] = result;
    }
  }
}

// Decode one stage (Step = 2 for gfx11, Step = 4 for gfx12), loading each
// packed A/B pair once and classifying nonfinite A rows into NaN vs Inf.
template <bool Bf16, bool Rdna3, bool Packed>
inline void decode_stage(int step, uint32_t instruction_flags, const uint32_t *const (&a)[8],
                         const uint32_t *const (&b)[8], StageLeft<Rdna3> (&left)[Rdna3 ? 2 : 4],
                         __m512i (&b_units)[Rdna3 ? 2 : 4], __m512i (&b_grid)[Rdna3 ? 2 : 4],
                         __m512i (&b_sign)[Rdna3 ? 2 : 1], uint32_t &a_nan_rows,
                         uint32_t &a_inf_rows, uint32_t &b_special) {
  using namespace exact_dot;
  constexpr int Step = Rdna3 ? 2 : 4;
  constexpr int Infinity = Bf16 ? 0x7f80 : 0x7c00;
  __m512i neg_lo_a = splat(instruction_flags & RJ_ARITH_WMMA_NEG_LO_A ? 0x8000 : 0);
  __m512i neg_hi_a = splat(instruction_flags & RJ_ARITH_WMMA_NEG_HI_A ? 0x8000 : 0);
  __m512i neg_lo_b = splat(instruction_flags & RJ_ARITH_WMMA_NEG_LO_B ? 0x8000 : 0);
  __m512i neg_hi_b = splat(instruction_flags & RJ_ARITH_WMMA_NEG_HI_B ? 0x8000 : 0);
  __m512i mask_lo = splat(0xffff);
  __m512i mask_mag = splat(0x7fff);
  __m512i amax = splat(0), bmax = splat(0);
  __m512i a_mags[Step];

  RJ_ARITH_UNROLL(2)
  for (int p = 0; p < Step / 2; ++p) {
    int pair = step / 2 + p;
    __m512i b_packed = load<__m512i>(b[pair]);
    __m512i a_packed = load<__m512i>(a[pair]);
    __m512i b_halves[2] = {bit_xor(bit_and(b_packed, mask_lo), neg_lo_b),
                           bit_xor(shift_right(b_packed, 16), neg_hi_b)};
    __m512i a_halves[2] = {bit_xor(bit_and(a_packed, mask_lo), neg_lo_a),
                           bit_xor(shift_right(a_packed, 16), neg_hi_a)};
    RJ_ARITH_UNROLL(2)
    for (int h = 0; h < 2; ++h) {
      int j = 2 * p + h;
      WmmaFactor bf = wmma_factor<Bf16, Rdna3>(b_halves[h], kWmmaGridBias<Bf16, Packed>);
      b_units[j] = bf.units;
      b_grid[j] = bf.grid;
      if constexpr (Rdna3) {
        b_sign[j] = bf.sign;
      }
      WmmaFactor af = wmma_factor<Bf16, Rdna3>(a_halves[h]);
      _mm512_store_si512(left[j].units, af.units);
      _mm512_store_si512(left[j].grid, af.grid);
      if constexpr (Rdna3) {
        _mm512_store_si512(left[j].sign, af.sign);
      }
      bmax = maximum(bmax, bit_and(b_halves[h], mask_mag));
      a_mags[j] = bit_and(a_halves[h], mask_mag);
      amax = maximum(amax, a_mags[j]);
    }
  }
  uint32_t stage_a_special = uint32_t(_mm512_cmpge_epi32_mask(amax, splat(Infinity)));
  if (stage_a_special) [[unlikely]] {
    uint32_t stage_nan = uint32_t(_mm512_cmpgt_epi32_mask(amax, splat(Infinity)));
    uint32_t stage_inf = 0;
    RJ_ARITH_UNROLL(4)
    for (int j = 0; j < Step; ++j) {
      stage_inf |= uint32_t(_mm512_cmpeq_epi32_mask(a_mags[j], splat(Infinity)));
    }
    a_nan_rows |= stage_nan;
    a_inf_rows |= stage_inf;
    // Zero nonfinite A rows so they do not trigger spurious overflow in any_bad.
    for (uint32_t m = stage_a_special; m; m &= m - 1) {
      int r = __builtin_ctz(m);
      RJ_ARITH_UNROLL(4)
      for (int j = 0; j < Step; ++j) {
        left[j].units[r] = 0;
        left[j].grid[r] = gfx12_dot_detail::kEmptyExponent;
      }
    }
  }
  b_special |= uint32_t(_mm512_cmpge_epi32_mask(bmax, splat(Infinity)));
}

// Accumulate all 16 independent rows per stage. The hot 16-row loops have
// zero branches, zero function calls, and zero scalar-to-vector stalls.
template <bool Bf16, bool Rdna3, bool Packed>
void accumulate_tile(uint32_t instruction_flags, bool saturate, __m512i (&acc)[16],
                     const __m512i (&c_init)[16], const uint32_t *const (&a)[8],
                     const uint32_t *const (&b)[8]) {
  using namespace exact_dot;
  constexpr int Step = Rdna3 ? 2 : 4;
  constexpr bool Half = Packed && !Bf16;
  constexpr int Nearest = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
  constexpr int Sign = int(0x80000000u);

  uint32_t c_special_rows = 0;
  __mmask16 c_special_mask[16];

  if constexpr (!Half) {
    constexpr int CFrac = Packed ? 7 : 23;
    constexpr int CInf = Packed ? 0x7f80 : 0x7f800000;
    constexpr int CQuiet = Packed ? 0x0040 : int(gfx12_dot_detail::kQuietNanBit);
    const int FactorNan = Packed ? int(packed16::special<Bf16>(gfx12_dot_detail::kFactorNan))
                                 : int(gfx12_dot_detail::kFactorNan);
    constexpr int FracMaskShl3 = Packed ? (0x7f0000 << 3) : (0x7fffff << 3);
    __m512i c_val[16], c_grid[16], c_sign[Rdna3 ? 16 : 1];

    for (int row = 0; row < 16; ++row) {
      __m512i ext = wmma_extend<Packed>(c_init[row]);
      __m512i ce = bit_and(shift_right(ext, CFrac), splat(255));
      __m512i cf = bit_and(ext, splat((1 << CFrac) - 1));
      __mmask16 normal = _mm512_test_epi32_mask(ext, splat(255 << CFrac));
      __mmask16 cspec = _mm512_cmpeq_epi32_mask(ce, splat(255));
      c_special_mask[row] = cspec;
      if (cspec) [[unlikely]] {
        c_special_rows |= 1u << row;
      }
      __mmask16 finite = __mmask16(~cspec);
      __mmask16 fnormal = __mmask16(normal & finite);
      __m512i cm = _mm512_maskz_or_epi32(finite, Rdna3 ? splat(0) : cf, splat(0));
      cm = _mm512_mask_or_epi32(cm, fnormal, cf, splat(1 << CFrac));
      __m512i cv = shift_left(cm, 26 - CFrac);
      __m512i cg = add(maximum(ce, splat(1)), splat(-153));
      if constexpr (Rdna3) {
        c_val[row] = cv;
        c_grid[row] = _mm512_mask_mov_epi32(splat(gfx12_dot_detail::kEmptyExponent), fnormal, cg);
        c_sign[row] = _mm512_maskz_srai_epi32(finite, ext, 31);
      } else {
        c_val[row] = _mm512_mask_sub_epi32(cv, _mm512_movepi32_mask(ext), splat(0), cv);
        c_grid[row] = _mm512_mask_mov_epi32(splat(-152), finite, cg);
      }
    }

    uint32_t a_nan_rows = 0, a_inf_rows = 0, b_special = 0;
    __mmask16 any_bad = 0;

    for (int step = 0; step < 16 - Step; step += Step) {
      StageLeft<Rdna3> left[Step];
      __m512i b_units[Step], b_grid[Step], b_sign[Rdna3 ? Step : 1];
      decode_stage<Bf16, Rdna3, Packed>(step, instruction_flags, a, b, left, b_units, b_grid,
                                        b_sign, a_nan_rows, a_inf_rows, b_special);
      for (int row = 0; row < 16; ++row) {
        __m512i total, grid;
        if constexpr (Rdna3) {
          __m512i g0 = add(splat(left[0].grid[row]), b_grid[0]);
          __m512i v0 = mul(splat(left[0].units[row]), b_units[0]);
          __m512i g1 = add(splat(left[1].grid[row]), b_grid[1]);
          __m512i v1 = mul(splat(left[1].units[row]), b_units[1]);
          __m512i cg = c_grid[row];
          __m512i cs = c_sign[row];
          grid = maximum(maximum(g0, g1), cg);
          total = shift_right(c_val[row], sub(grid, cg));
          __m512i m0 = shift_right(v0, sub(grid, g0));
          __m512i ps0 = bit_xor(splat(left[0].sign[row]), b_sign[0]);
          total = add(total, _mm512_ternarylogic_epi32(m0, ps0, cs, 0x96));
          __m512i m1 = shift_right(v1, sub(grid, g1));
          __m512i ps1 = bit_xor(splat(left[1].sign[row]), b_sign[1]);
          total = add(total, _mm512_ternarylogic_epi32(m1, ps1, cs, 0x96));

          __m512i rounded = _mm512_castps_si512(_mm512_cvt_roundepi32_ps(total, Nearest));
          __m512i exp = bit_and(shift_right(rounded, 23), splat(255));
          __m512i ce = add(exp, grid);
          __mmask16 nz = _mm512_test_epi32_mask(total, total);
          __mmask16 keep = _mm512_mask_cmpgt_epi32_mask(nz, ce, splat(0));
          any_bad |= _mm512_mask_cmpge_epi32_mask(keep, ce, splat(255));
          __m512i r3 = shift_left(rounded, 3);
          c_val[row] =
              _mm512_maskz_ternarylogic_epi32(keep, r3, splat(FracMaskShl3), splat(1 << 26), 0xea);
          c_grid[row] =
              _mm512_mask_add_epi32(splat(gfx12_dot_detail::kEmptyExponent), keep, ce, splat(-153));
          c_sign[row] = _mm512_maskz_srai_epi32(keep, bit_xor(total, cs), 31);
        } else {
          __m512i g0 = add(splat(left[0].grid[row]), b_grid[0]);
          __m512i v0 = mul(splat(left[0].units[row]), b_units[0]);
          __m512i g1 = add(splat(left[1].grid[row]), b_grid[1]);
          __m512i v1 = mul(splat(left[1].units[row]), b_units[1]);
          __m512i pg0 = maximum(g0, g1);
          __m512i p0 =
              add(_mm512_srav_epi32(v0, sub(pg0, g0)), _mm512_srav_epi32(v1, sub(pg0, g1)));

          __m512i g2 = add(splat(left[2].grid[row]), b_grid[2]);
          __m512i v2 = mul(splat(left[2].units[row]), b_units[2]);
          __m512i g3 = add(splat(left[3].grid[row]), b_grid[3]);
          __m512i v3 = mul(splat(left[3].units[row]), b_units[3]);
          __m512i pg1 = maximum(g2, g3);
          __m512i p1 =
              add(_mm512_srav_epi32(v2, sub(pg1, g2)), _mm512_srav_epi32(v3, sub(pg1, g3)));

          __m512i cg = c_grid[row];
          grid = maximum(maximum(pg0, pg1), cg);
          __m512i cshift = sub(grid, cg);
          total = _mm512_maskz_srav_epi32(_mm512_cmplt_epi32_mask(cshift, splat(27)), c_val[row],
                                          cshift);
          total = add(total, _mm512_srav_epi32(p0, sub(grid, pg0)));
          total = add(total, _mm512_srav_epi32(p1, sub(grid, pg1)));

          __m512i rounded = _mm512_castps_si512(_mm512_cvt_roundepi32_ps(total, Nearest));
          __m512i exp = bit_and(shift_right(rounded, 23), splat(255));
          __m512i ce = add(exp, grid);
          __mmask16 nz = _mm512_test_epi32_mask(total, total);
          any_bad |= _mm512_mask_cmpge_epu32_mask(nz, sub(ce, splat(1)), splat(254));
          __m512i r3 = shift_left(rounded, 3);
          __m512i cv =
              _mm512_maskz_ternarylogic_epi32(nz, r3, splat(FracMaskShl3), splat(1 << 26), 0xea);
          c_val[row] = _mm512_mask_sub_epi32(cv, _mm512_movepi32_mask(total), splat(0), cv);
          c_grid[row] = _mm512_mask_add_epi32(splat(-152), nz, ce, splat(-153));
        }
      }
    }

    // Final stage: pack branchlessly into FP32 or Packed BF16.
    {
      StageLeft<Rdna3> left[Step];
      __m512i b_units[Step], b_grid[Step], b_sign[Rdna3 ? Step : 1];
      decode_stage<Bf16, Rdna3, Packed>(16 - Step, instruction_flags, a, b, left, b_units, b_grid,
                                        b_sign, a_nan_rows, a_inf_rows, b_special);
      for (int row = 0; row < 16; ++row) {
        __m512i total, grid;
        if constexpr (Rdna3) {
          __m512i g0 = add(splat(left[0].grid[row]), b_grid[0]);
          __m512i v0 = mul(splat(left[0].units[row]), b_units[0]);
          __m512i g1 = add(splat(left[1].grid[row]), b_grid[1]);
          __m512i v1 = mul(splat(left[1].units[row]), b_units[1]);
          __m512i cg = c_grid[row];
          __m512i cs = c_sign[row];
          grid = maximum(maximum(g0, g1), cg);
          total = shift_right(c_val[row], sub(grid, cg));
          __m512i m0 = shift_right(v0, sub(grid, g0));
          __m512i ps0 = bit_xor(splat(left[0].sign[row]), b_sign[0]);
          total = add(total, _mm512_ternarylogic_epi32(m0, ps0, cs, 0x96));
          __m512i m1 = shift_right(v1, sub(grid, g1));
          __m512i ps1 = bit_xor(splat(left[1].sign[row]), b_sign[1]);
          total = add(total, _mm512_ternarylogic_epi32(m1, ps1, cs, 0x96));

          __m512i rounded = _mm512_castps_si512(_mm512_cvt_roundepi32_ps(total, Nearest));
          __m512i mag = bit_and(rounded, splat(0x7fffffff));
          __m512i exp = add(shift_right(mag, 23), grid);
          __m512i bits = _mm512_min_epu32(add(mag, shift_left(grid, 23)), splat(0x7f800000));
          __mmask16 keep =
              _mm512_mask_cmpgt_epi32_mask(_mm512_test_epi32_mask(total, total), exp, splat(0));
          bits =
              _mm512_maskz_ternarylogic_epi32(keep, bits, bit_xor(rounded, cs), splat(Sign), 0xf8);
          if constexpr (Packed) {
            bits = shift_right(bits, 16);
          }
          acc[row] = bits;
        } else {
          __m512i g0 = add(splat(left[0].grid[row]), b_grid[0]);
          __m512i v0 = mul(splat(left[0].units[row]), b_units[0]);
          __m512i g1 = add(splat(left[1].grid[row]), b_grid[1]);
          __m512i v1 = mul(splat(left[1].units[row]), b_units[1]);
          __m512i pg0 = maximum(g0, g1);
          __m512i p0 =
              add(_mm512_srav_epi32(v0, sub(pg0, g0)), _mm512_srav_epi32(v1, sub(pg0, g1)));

          __m512i g2 = add(splat(left[2].grid[row]), b_grid[2]);
          __m512i v2 = mul(splat(left[2].units[row]), b_units[2]);
          __m512i g3 = add(splat(left[3].grid[row]), b_grid[3]);
          __m512i v3 = mul(splat(left[3].units[row]), b_units[3]);
          __m512i pg1 = maximum(g2, g3);
          __m512i p1 =
              add(_mm512_srav_epi32(v2, sub(pg1, g2)), _mm512_srav_epi32(v3, sub(pg1, g3)));

          __m512i cg = c_grid[row];
          grid = maximum(maximum(pg0, pg1), cg);
          __m512i cshift = sub(grid, cg);
          total = _mm512_maskz_srav_epi32(_mm512_cmplt_epi32_mask(cshift, splat(27)), c_val[row],
                                          cshift);
          total = add(total, _mm512_srav_epi32(p0, sub(grid, pg0)));
          total = add(total, _mm512_srav_epi32(p1, sub(grid, pg1)));

          __m512i rounded = _mm512_castps_si512(_mm512_cvt_roundepi32_ps(total, Nearest));
          __m512i mag = bit_and(rounded, splat(0x7fffffff));
          __m512i exp = add(shift_right(mag, 23), grid);
          __mmask16 keep = _mm512_test_epi32_mask(total, total);
          any_bad |= _mm512_mask_cmple_epi32_mask(keep, exp, splat(0));
          __m512i bits = _mm512_min_epu32(add(mag, shift_left(grid, 23)), splat(0x7f800000));
          bits = _mm512_maskz_ternarylogic_epi32(keep, bits, rounded, splat(Sign), 0xf8);
          if constexpr (Packed) {
            bits = shift_right(bits, 16);
          }
          acc[row] = bits;
        }
      }
    }

    if (b_special || a_inf_rows || any_bad) [[unlikely]] {
      uint32_t repair_mask =
          (b_special || any_bad) ? 0xffffu : (a_inf_rows | a_nan_rows | c_special_rows);
      repair_rows<Bf16, Rdna3, Packed>(repair_mask, instruction_flags, saturate, acc, c_init, a, b);
      if constexpr (Packed) {
        for (uint32_t m = repair_mask; m; m &= m - 1) {
          int row = __builtin_ctz(m);
          acc[row] = bit_and(acc[row], splat(0xffff));
        }
      }
    } else if (c_special_rows | a_nan_rows) [[unlikely]] {
      for (uint32_t m = c_special_rows; m; m &= m - 1) {
        int row = __builtin_ctz(m);
        __m512i cv = c_init[row];
        __mmask16 cnan =
            _mm512_cmpgt_epi32_mask(bit_and(cv, splat(Packed ? 0x7fff : 0x7fffffff)), splat(CInf));
        __m512i resolved = _mm512_mask_or_epi32(cv, cnan, cv, splat(CQuiet));
        acc[row] = _mm512_mask_mov_epi32(acc[row], c_special_mask[row], resolved);
      }
      for (uint32_t m = a_nan_rows; m; m &= m - 1) {
        acc[__builtin_ctz(m)] = splat(FactorNan);
      }
    }
  } else {
    // Packed FP16: branchless 16-row stages using fast SAE VCVTPS2PH rounding.
    constexpr int CInf = 0x7c00, CQuiet = 0x0200;
    const int FactorNan = int(packed16::special<false>(gfx12_dot_detail::kFactorNan));
    for (int row = 0; row < 16; ++row) {
      __m512i ext = wmma_extend<true>(c_init[row]);
      __m512i ce = bit_and(shift_right(ext, 10), splat(31));
      __mmask16 cspec = _mm512_cmpeq_epi32_mask(ce, splat(31));
      c_special_mask[row] = cspec;
      if (cspec) [[unlikely]] {
        c_special_rows |= 1u << row;
      }
      acc[row] = _mm512_maskz_mov_epi32(__mmask16(~cspec), ext);
    }
    uint32_t a_nan_rows = 0, a_inf_rows = 0, b_special = 0;
    __mmask16 any_bad = 0;
    for (int step = 0; step < 16; step += Step) {
      StageLeft<Rdna3> left[Step];
      __m512i b_units[Step], b_grid[Step], b_sign[Rdna3 ? Step : 1];
      decode_stage<false, Rdna3, true>(step, instruction_flags, a, b, left, b_units, b_grid, b_sign,
                                       a_nan_rows, a_inf_rows, b_special);
      for (int row = 0; row < 16; ++row) {
        __m512i cur = acc[row];
        __m512i ce = bit_and(shift_right(cur, 10), splat(31));
        any_bad |= _mm512_cmpeq_epi32_mask(ce, splat(31));
        __m512i cf = bit_and(cur, splat(1023));
        __mmask16 normal = _mm512_test_epi32_mask(cur, splat(31 << 10));
        __m512i cm = _mm512_mask_or_epi32(cf, normal, cf, splat(1024));
        __m512i cv = shift_left(cm, 16);
        __m512i cg = add(maximum(ce, splat(1)), splat(127 - 15 - 26));
        __m512i total, grid, frame;
        if constexpr (Rdna3) {
          frame = signed_shift_right(cur, 31);
          __m512i g0 = add(splat(left[0].grid[row]), b_grid[0]);
          __m512i v0 = mul(splat(left[0].units[row]), b_units[0]);
          __m512i g1 = add(splat(left[1].grid[row]), b_grid[1]);
          __m512i v1 = mul(splat(left[1].units[row]), b_units[1]);
          grid = maximum(maximum(g0, g1), cg);
          total = shift_right(cv, sub(grid, cg));
          __m512i m0 = shift_right(v0, sub(grid, g0));
          __m512i ps0 = bit_xor(splat(left[0].sign[row]), b_sign[0]);
          total = add(total, _mm512_ternarylogic_epi32(m0, ps0, frame, 0x96));
          __m512i m1 = shift_right(v1, sub(grid, g1));
          __m512i ps1 = bit_xor(splat(left[1].sign[row]), b_sign[1]);
          total = add(total, _mm512_ternarylogic_epi32(m1, ps1, frame, 0x96));
        } else {
          frame = splat(0);
          __m512i g0 = add(splat(left[0].grid[row]), b_grid[0]);
          __m512i v0 = mul(splat(left[0].units[row]), b_units[0]);
          __m512i g1 = add(splat(left[1].grid[row]), b_grid[1]);
          __m512i v1 = mul(splat(left[1].units[row]), b_units[1]);
          __m512i pg0 = maximum(g0, g1);
          __m512i p0 =
              add(_mm512_srav_epi32(v0, sub(pg0, g0)), _mm512_srav_epi32(v1, sub(pg0, g1)));
          __m512i g2 = add(splat(left[2].grid[row]), b_grid[2]);
          __m512i v2 = mul(splat(left[2].units[row]), b_units[2]);
          __m512i g3 = add(splat(left[3].grid[row]), b_grid[3]);
          __m512i v3 = mul(splat(left[3].units[row]), b_units[3]);
          __m512i pg1 = maximum(g2, g3);
          __m512i p1 =
              add(_mm512_srav_epi32(v2, sub(pg1, g2)), _mm512_srav_epi32(v3, sub(pg1, g3)));
          grid = maximum(maximum(pg0, pg1), cg);
          __m512i cshift = sub(grid, cg);
          __m512i sc = _mm512_mask_sub_epi32(cv, _mm512_movepi32_mask(cur), splat(0), cv);
          total = _mm512_maskz_srav_epi32(_mm512_cmplt_epi32_mask(cshift, splat(27)), sc, cshift);
          total = add(total, _mm512_srav_epi32(p0, sub(grid, pg0)));
          total = add(total, _mm512_srav_epi32(p1, sub(grid, pg1)));
        }
        acc[row] = wmma_pack<false, Rdna3, true>(total, grid, frame, saturate);
      }
    }
    if (b_special || a_inf_rows || any_bad) [[unlikely]] {
      uint32_t repair_mask =
          (b_special || any_bad) ? 0xffffu : (a_inf_rows | a_nan_rows | c_special_rows);
      repair_rows<false, Rdna3, true>(repair_mask, instruction_flags, saturate, acc, c_init, a, b);
    } else if (c_special_rows | a_nan_rows) [[unlikely]] {
      for (uint32_t m = c_special_rows; m; m &= m - 1) {
        int row = __builtin_ctz(m);
        __m512i cv = c_init[row];
        __mmask16 cnan = _mm512_cmpgt_epi32_mask(bit_and(cv, splat(0x7fff)), splat(CInf));
        __m512i resolved = _mm512_mask_or_epi32(cv, cnan, cv, splat(CQuiet));
        acc[row] = _mm512_mask_mov_epi32(acc[row], c_special_mask[row], resolved);
      }
      for (uint32_t m = a_nan_rows; m; m &= m - 1) {
        acc[__builtin_ctz(m)] = splat(FactorNan);
      }
    }
    for (int row = 0; row < 16; ++row) {
      acc[row] = bit_and(acc[row], splat(0xffff));
    }
  }
}

} // namespace RJ_ARITH_IMPL_NAMESPACE::wmma_v4

// Compute bit-exact A*B+C with validated controls. D may alias any input.
// gfx11 packed results preserve the unselected half of each destination.
template <bool Bf16, int Lanes, bool Rdna3, bool Packed>
void wmma_exact_x86_64_v4(uint32_t instruction_flags, bool saturate, uint32_t *const *d,
                          const uint32_t *const *a, const uint32_t *const *b,
                          const uint32_t *const *c) {
  using namespace exact_dot;
  using namespace wmma_v4;
  constexpr int Registers = 256 / Lanes,
                DestinationRegisters = Packed && !Rdna3 ? Registers / 2 : Registers;
  const int high = instruction_flags & RJ_ARITH_WMMA_HIGH_C_D ? 16 : 0;
  auto layout = [](int row, int &reg, int &lane) {
    reg = Rdna3 ? row / (Lanes / 16) : row % Registers;
    lane = Rdna3 ? 16 * (row % (Lanes / 16))
                 : 16 * (row / 8) + (Lanes == 64 ? 32 * ((row / 4) % 2) : 0);
  };
  __m512i c_init[16], acc[16];
  __m512i preserved[Rdna3 && Packed ? 16 : 1];
  for (int row = 0; row < 16; ++row) {
    int reg, lane;
    layout(row, reg, lane);
    int dest_reg = Packed && !Rdna3 ? reg / 2 : reg;
    auto v = load<__m512i>(c[dest_reg] + lane);
    if constexpr (Rdna3 && Packed) {
      __m512i old = load<__m512i>(d[dest_reg] + lane);
      preserved[row] = andnot(splat(int(0xffffu << high)), old);
    }
    if constexpr (Packed) {
      int shift = Rdna3 ? high : 16 * (reg % 2);
      v = bit_and(shift_right(v, splat(shift)), splat(0xffff));
    }
    if (instruction_flags & RJ_ARITH_WMMA_ABS_C) {
      v = bit_and(v, splat(Packed ? 0x7fff : 0x7fffffff));
    }
    if (instruction_flags & RJ_ARITH_WMMA_NEG_C) {
      v = bit_xor(v, splat(Packed ? 0x8000 : int(0x80000000U)));
    }
    c_init[row] = v;
  }
  const uint32_t *left[8], *right[8];
  for (int pair = 0; pair < 8; ++pair) {
    int k = !Rdna3 && Lanes == 32 ? wmma_gfx12_w32_k(2 * pair) : 2 * pair;
    int reg = Rdna3 ? k / 2 : (k % Registers) / 2;
    int lane = Rdna3 ? 0 : 16 * (k / Registers);
    left[pair] = a[reg] + lane;
    right[pair] = b[reg] + lane;
  }
  accumulate_tile<Bf16, Rdna3, Packed>(instruction_flags, saturate, acc, c_init, left, right);

  // All inputs and preserved halves have been read; write destinations in
  // ascending register order so the last store wins if D pointers alias.
  if constexpr (Packed && !Rdna3) {
    constexpr int Groups = Lanes / 16;
    for (int dest_reg = 0; dest_reg < DestinationRegisters; ++dest_reg) {
      for (int g = 0; g < Groups; ++g) {
        int base_row = 2 * dest_reg + (Lanes == 64 ? 8 * (g & 1) + 4 * (g >> 1) : 8 * g);
        __m512i packed = bit_or(acc[base_row], shift_left(acc[base_row + 1], 16));
        store(d[dest_reg] + 16 * g, packed);
      }
    }
  } else {
    constexpr int Groups = Lanes / 16;
    for (int reg = 0; reg < DestinationRegisters; ++reg) {
      for (int g = 0; g < Groups; ++g) {
        int row =
            Rdna3 ? reg * Groups + g : reg + (Lanes == 64 ? 8 * (g & 1) + 4 * (g >> 1) : 8 * g);
        __m512i v = acc[row];
        if constexpr (Rdna3 && Packed) {
          v = bit_or(shift_left(v, splat(high)), preserved[row]);
        }
        store(d[reg] + 16 * g, v);
      }
    }
  }
}

template void wmma_exact_x86_64_v4<false, 32, false, false>(uint32_t, bool, uint32_t *const *,
                                                            const uint32_t *const *,
                                                            const uint32_t *const *,
                                                            const uint32_t *const *);
template void wmma_exact_x86_64_v4<false, 32, false, true>(uint32_t, bool, uint32_t *const *,
                                                           const uint32_t *const *,
                                                           const uint32_t *const *,
                                                           const uint32_t *const *);
template void wmma_exact_x86_64_v4<false, 32, true, false>(uint32_t, bool, uint32_t *const *,
                                                           const uint32_t *const *,
                                                           const uint32_t *const *,
                                                           const uint32_t *const *);
template void wmma_exact_x86_64_v4<false, 32, true, true>(uint32_t, bool, uint32_t *const *,
                                                          const uint32_t *const *,
                                                          const uint32_t *const *,
                                                          const uint32_t *const *);
template void wmma_exact_x86_64_v4<false, 64, false, false>(uint32_t, bool, uint32_t *const *,
                                                            const uint32_t *const *,
                                                            const uint32_t *const *,
                                                            const uint32_t *const *);
template void wmma_exact_x86_64_v4<false, 64, false, true>(uint32_t, bool, uint32_t *const *,
                                                           const uint32_t *const *,
                                                           const uint32_t *const *,
                                                           const uint32_t *const *);
template void wmma_exact_x86_64_v4<false, 64, true, false>(uint32_t, bool, uint32_t *const *,
                                                           const uint32_t *const *,
                                                           const uint32_t *const *,
                                                           const uint32_t *const *);
template void wmma_exact_x86_64_v4<false, 64, true, true>(uint32_t, bool, uint32_t *const *,
                                                          const uint32_t *const *,
                                                          const uint32_t *const *,
                                                          const uint32_t *const *);
template void wmma_exact_x86_64_v4<true, 32, false, false>(uint32_t, bool, uint32_t *const *,
                                                           const uint32_t *const *,
                                                           const uint32_t *const *,
                                                           const uint32_t *const *);
template void wmma_exact_x86_64_v4<true, 32, false, true>(uint32_t, bool, uint32_t *const *,
                                                          const uint32_t *const *,
                                                          const uint32_t *const *,
                                                          const uint32_t *const *);
template void wmma_exact_x86_64_v4<true, 32, true, false>(uint32_t, bool, uint32_t *const *,
                                                          const uint32_t *const *,
                                                          const uint32_t *const *,
                                                          const uint32_t *const *);
template void wmma_exact_x86_64_v4<true, 32, true, true>(uint32_t, bool, uint32_t *const *,
                                                         const uint32_t *const *,
                                                         const uint32_t *const *,
                                                         const uint32_t *const *);
template void wmma_exact_x86_64_v4<true, 64, false, false>(uint32_t, bool, uint32_t *const *,
                                                           const uint32_t *const *,
                                                           const uint32_t *const *,
                                                           const uint32_t *const *);
template void wmma_exact_x86_64_v4<true, 64, false, true>(uint32_t, bool, uint32_t *const *,
                                                          const uint32_t *const *,
                                                          const uint32_t *const *,
                                                          const uint32_t *const *);
template void wmma_exact_x86_64_v4<true, 64, true, false>(uint32_t, bool, uint32_t *const *,
                                                          const uint32_t *const *,
                                                          const uint32_t *const *,
                                                          const uint32_t *const *);
template void wmma_exact_x86_64_v4<true, 64, true, true>(uint32_t, bool, uint32_t *const *,
                                                         const uint32_t *const *,
                                                         const uint32_t *const *,
                                                         const uint32_t *const *);

} // namespace rocjitsu::arith
