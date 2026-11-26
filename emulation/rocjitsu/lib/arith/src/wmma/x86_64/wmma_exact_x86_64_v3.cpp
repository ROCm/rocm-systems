// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Eight-column integer-grid WMMA for the AVX2 backend. Keeps unpacked
// accumulators across stages for FP32 and Packed BF16, uses VPMADDWD for
// 16-bit Q12 unit products, preserves host MXCSR on packed paths, and defers
// rare nonfinite/subnormal/overflow repair to a cold helper.
#include "common/simd.h"
#include "rj_arith/detail/rj_arith_wmma.h"
#include "wmma/layout.h"
#include "wmma/x86_64/wmma_exact_x86_64.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {

namespace RJ_ARITH_IMPL_NAMESPACE::wmma_v3 {

// Scalar-addressable decoded A fields for one stage. Units are zero-extended
// signed 16-bit Q12 values so VPMADDWD computes exact 32-bit products.
template <bool Rdna3> struct alignas(32) StageLeft {
  uint32_t units[16];
  int32_t grid[16];
  int32_t sign[Rdna3 ? 16 : 1];
};

// Multiply zero-extended 16-bit signed A units with signed B units via VPMADDWD.
inline __m256i mul_units(__m256i a_u16_zext, __m256i b_units) {
  return _mm256_madd_epi16(a_u16_zext, b_units);
}

// Cold exact fallback for rows in one 8-column slice that hit infinite factors,
// nonfinite B, intermediate overflow, or gradual subnormal underflow.
template <bool Bf16, bool Rdna3, bool Packed>
[[gnu::noinline]] void repair_slice_rows(uint32_t row_mask, int half, uint32_t flags, bool saturate,
                                         __m256i (&acc)[16], const __m256i (&c_init)[16],
                                         const uint32_t *const (&a)[8],
                                         const uint32_t *const (&b)[8]) {
  using namespace exact_dot;
  constexpr int N = Rdna3 ? 2 : 4;
  constexpr int Infinity = Bf16 ? 0x7f80 : 0x7c00;
  for (uint32_t m = row_mask; m; m &= m - 1) {
    int row = __builtin_ctz(m);
    acc[row] = c_init[row];
  }
  for (int step = 0; step < 16; step += N) {
    Factor right[N];
    __m256i b_max = splat(0);
    RJ_ARITH_UNROLL(2)
    for (int p = 0; p < N / 2; ++p) {
      int pair = step / 2 + p;
      __m256i b_packed = load<__m256i>(b[pair] + 8 * half);
      __m256i bh[2] = {
          bit_xor(bit_and(b_packed, splat(0xffff)),
                  splat(flags & RJ_ARITH_WMMA_NEG_LO_B ? 0x8000 : 0)),
          bit_xor(shift_right(b_packed, 16), splat(flags & RJ_ARITH_WMMA_NEG_HI_B ? 0x8000 : 0))};
      RJ_ARITH_UNROLL(2)
      for (int h = 0; h < 2; ++h) {
        int j = 2 * p + h;
        right[j] = decode<Bf16, Rdna3>(bh[h], kGridBias<Bf16, Packed>);
        b_max = maximum(b_max, bit_and(bh[h], splat(0x7fff)));
      }
    }
    bool exceptional_columns = lane_mask(greater(b_max, splat(Infinity - 1))) != 0;
    for (uint32_t m = row_mask; m; m &= m - 1) {
      int row = __builtin_ctz(m);
      Factor factors[N];
      int row_max = 0;
      RJ_ARITH_UNROLL(4)
      for (int j = 0; j < N; ++j) {
        int pair = (step + j) / 2, shift = 16 * (j % 2);
        uint32_t neg_a =
            (flags & (j % 2 ? RJ_ARITH_WMMA_NEG_HI_A : RJ_ARITH_WMMA_NEG_LO_A)) ? 0x8000u : 0u;
        int abits = int(((a[pair][row] >> shift) & 0xffffu) ^ neg_a);
        int amag = abits & 0x7fff;
        if (amag > row_max) {
          row_max = amag;
        }
        factors[j] = decode<Bf16, Rdna3>(splat(abits));
      }
      __m256i old = acc[row];
      __m256i c_exception;
      __m256i result = stage<Bf16, Rdna3, Packed>(factors, right, old, saturate, c_exception);
      if (exceptional_columns || row_max >= Infinity || lane_mask(c_exception)) {
        __m256i factor_max = maximum(b_max, splat(row_max));
        __m256i mask = bit_or(c_exception, greater(factor_max, splat(Infinity - 1)));
        result =
            choose(mask, special<Bf16, Rdna3, Packed>(factors, right, factor_max, old), result);
      }
      acc[row] = result;
    }
  }
}

// Decode one stage of A (16 rows) and B (both 8-column slices), loading each
// packed A/B pair once and classifying nonfinite A rows into NaN vs Inf.
template <bool Bf16, bool Rdna3, int GridBias>
inline void decode_stage(int step, uint32_t flags, const uint32_t *const (&a)[8],
                         const uint32_t *const (&b)[8], StageLeft<Rdna3> (&left)[Rdna3 ? 2 : 4],
                         __m256i (&b_units)[2][Rdna3 ? 2 : 4], __m256i (&b_grid)[2][Rdna3 ? 2 : 4],
                         __m256i (&b_sign)[2][Rdna3 ? 2 : 1], uint32_t &a_nan_rows,
                         uint32_t &a_inf_rows, uint32_t (&b_special)[2]) {
  using namespace exact_dot;
  constexpr int N = Rdna3 ? 2 : 4;
  constexpr int Infinity = Bf16 ? 0x7f80 : 0x7c00;
  __m256i neg_lo_a = splat(flags & RJ_ARITH_WMMA_NEG_LO_A ? 0x8000 : 0);
  __m256i neg_hi_a = splat(flags & RJ_ARITH_WMMA_NEG_HI_A ? 0x8000 : 0);
  __m256i neg_lo_b = splat(flags & RJ_ARITH_WMMA_NEG_LO_B ? 0x8000 : 0);
  __m256i neg_hi_b = splat(flags & RJ_ARITH_WMMA_NEG_HI_B ? 0x8000 : 0);
  __m256i mask_lo = splat(0xffff);
  __m256i mask_mag = splat(0x7fff);
  __m256i a_max[2] = {splat(0), splat(0)};
  __m256i b_max[2] = {splat(0), splat(0)};
  __m256i a_mags[2][N];

  RJ_ARITH_UNROLL(2)
  for (int p = 0; p < N / 2; ++p) {
    int pair = step / 2 + p;
    for (int half = 0; half < 2; ++half) {
      __m256i b_packed = load<__m256i>(b[pair] + 8 * half);
      __m256i bh[2] = {bit_xor(bit_and(b_packed, mask_lo), neg_lo_b),
                       bit_xor(shift_right(b_packed, 16), neg_hi_b)};
      RJ_ARITH_UNROLL(2)
      for (int h = 0; h < 2; ++h) {
        int j = 2 * p + h;
        Factor bf = decode<Bf16, Rdna3>(bh[h], GridBias);
        b_units[half][j] = bf.units;
        b_grid[half][j] = bf.grid;
        if constexpr (Rdna3) {
          b_sign[half][j] = bf.sign;
        }
        b_max[half] = maximum(b_max[half], bit_and(bh[h], mask_mag));
      }
    }
    for (int k = 0; k < 2; ++k) {
      __m256i a_packed = load<__m256i>(a[pair] + 8 * k);
      __m256i ah[2] = {bit_xor(bit_and(a_packed, mask_lo), neg_lo_a),
                       bit_xor(shift_right(a_packed, 16), neg_hi_a)};
      RJ_ARITH_UNROLL(2)
      for (int h = 0; h < 2; ++h) {
        int j = 2 * p + h;
        Factor af = decode<Bf16, Rdna3>(ah[h]);
        store(left[j].units + 8 * k, bit_and(af.units, mask_lo));
        store(left[j].grid + 8 * k, af.grid);
        if constexpr (Rdna3) {
          store(left[j].sign + 8 * k, af.sign);
        }
        a_mags[k][j] = bit_and(ah[h], mask_mag);
        a_max[k] = maximum(a_max[k], a_mags[k][j]);
      }
    }
  }
  uint32_t stage_a_special = uint32_t(lane_mask(greater(a_max[0], splat(Infinity - 1)))) |
                             (uint32_t(lane_mask(greater(a_max[1], splat(Infinity - 1)))) << 8);
  if (stage_a_special) [[unlikely]] {
    a_nan_rows |= uint32_t(lane_mask(greater(a_max[0], splat(Infinity)))) |
                  (uint32_t(lane_mask(greater(a_max[1], splat(Infinity)))) << 8);
    RJ_ARITH_UNROLL(4)
    for (int j = 0; j < N; ++j) {
      a_inf_rows |= uint32_t(lane_mask(equal(a_mags[0][j], splat(Infinity)))) |
                    (uint32_t(lane_mask(equal(a_mags[1][j], splat(Infinity)))) << 8);
    }
    for (uint32_t m = stage_a_special; m; m &= m - 1) {
      int r = __builtin_ctz(m);
      RJ_ARITH_UNROLL(4)
      for (int j = 0; j < N; ++j) {
        left[j].units[r] = 0;
        left[j].grid[r] = gfx12_dot_detail::kEmptyExponent;
      }
    }
  }
  b_special[0] |= uint32_t(lane_mask(greater(b_max[0], splat(Infinity - 1))));
  b_special[1] |= uint32_t(lane_mask(greater(b_max[1], splat(Infinity - 1))));
}

// Compute one DOT2 or DOT4 integer-grid sum for 8 columns of a single row.
template <bool Rdna3>
inline void stage_sum(int row, const StageLeft<Rdna3> (&left)[Rdna3 ? 2 : 4],
                      const __m256i (&b_units)[Rdna3 ? 2 : 4],
                      const __m256i (&b_grid)[Rdna3 ? 2 : 4],
                      const __m256i (&b_sign)[Rdna3 ? 2 : 1], __m256i c_val, __m256i c_grid,
                      __m256i c_sign, __m256i &total_out, __m256i &grid_out) {
  using namespace exact_dot;
  if constexpr (Rdna3) {
    __m256i g0 = add(splat(left[0].grid[row]), b_grid[0]);
    __m256i v0 = mul_units(splat(int(left[0].units[row])), b_units[0]);
    __m256i g1 = add(splat(left[1].grid[row]), b_grid[1]);
    __m256i v1 = mul_units(splat(int(left[1].units[row])), b_units[1]);
    __m256i grid = maximum(maximum(g0, g1), c_grid);
    __m256i total = shift_right(c_val, sub(grid, c_grid));
    __m256i m0 = shift_right(v0, sub(grid, g0));
    total = add(total, bit_xor(m0, bit_xor(bit_xor(splat(left[0].sign[row]), b_sign[0]), c_sign)));
    __m256i m1 = shift_right(v1, sub(grid, g1));
    total = add(total, bit_xor(m1, bit_xor(bit_xor(splat(left[1].sign[row]), b_sign[1]), c_sign)));
    total_out = total;
    grid_out = grid;
  } else {
    __m256i g0 = add(splat(left[0].grid[row]), b_grid[0]);
    __m256i v0 = mul_units(splat(int(left[0].units[row])), b_units[0]);
    __m256i g1 = add(splat(left[1].grid[row]), b_grid[1]);
    __m256i v1 = mul_units(splat(int(left[1].units[row])), b_units[1]);
    __m256i pg0 = maximum(g0, g1);
    __m256i p0 = add(_mm256_srav_epi32(v0, sub(pg0, g0)), _mm256_srav_epi32(v1, sub(pg0, g1)));

    __m256i g2 = add(splat(left[2].grid[row]), b_grid[2]);
    __m256i v2 = mul_units(splat(int(left[2].units[row])), b_units[2]);
    __m256i g3 = add(splat(left[3].grid[row]), b_grid[3]);
    __m256i v3 = mul_units(splat(int(left[3].units[row])), b_units[3]);
    __m256i pg1 = maximum(g2, g3);
    __m256i p1 = add(_mm256_srav_epi32(v2, sub(pg1, g2)), _mm256_srav_epi32(v3, sub(pg1, g3)));

    __m256i grid = maximum(maximum(pg0, pg1), c_grid);
    __m256i cshift = sub(grid, c_grid);
    __m256i total = bit_and(greater(splat(27), cshift), _mm256_srav_epi32(c_val, cshift));
    total = add(total, _mm256_srav_epi32(p0, sub(grid, pg0)));
    total = add(total, _mm256_srav_epi32(p1, sub(grid, pg1)));
    total_out = total;
    grid_out = grid;
  }
}

// Advance the sixteen rows across all DOT2/DOT4 stages.
// - FP32 outputs use host-RNE VCVTDQ2PS with unpacked (c_val, c_grid).
// - Packed BF16 uses exact-only biased_top_bit + unpacked (c_val, c_grid) to
//   preserve host MXCSR exceptions without repacking/re-unpacking between stages.
// - Packed FP16 preserves host MXCSR exceptions via exact-only biased_top_bit.
template <bool Bf16, bool Rdna3, bool Packed>
void accumulate_tile(uint32_t flags, bool saturate, __m256i (&acc)[2][16],
                     const __m256i (&c_init)[2][16], const uint32_t *const (&a)[8],
                     const uint32_t *const (&b)[8]) {
  using namespace exact_dot;
  constexpr int N = Rdna3 ? 2 : 4;
  constexpr bool Half = Packed && !Bf16;
  constexpr int Sign = int(0x80000000U);

  if constexpr (!Half) {
    constexpr int CFrac = Packed ? 7 : 23;
    constexpr int CInf = Packed ? 0x7f80 : 0x7f800000;
    constexpr int CQuiet = Packed ? 0x0040 : int(gfx12_dot_detail::kQuietNanBit);
    const int FactorNan = Packed ? int(packed16::special<Bf16>(gfx12_dot_detail::kFactorNan))
                                 : int(gfx12_dot_detail::kFactorNan);
    constexpr int FracMaskShl3 = Packed ? (0x7f0000 << 3) : (0x7fffff << 3);

    __m256i c_val[2][16], c_grid[2][16], c_sign[Rdna3 ? 2 : 1][16];
    __m256i c_special_vec[2][16];
    uint32_t c_special_rows[2] = {0, 0};

    for (int half = 0; half < 2; ++half) {
      for (int row = 0; row < 16; ++row) {
        __m256i v = c_init[half][row];
        __m256i ce = bit_and(shift_right(v, CFrac), splat(255));
        __m256i cf = shift_left(bit_and(v, splat((1 << CFrac) - 1)), 26 - CFrac);
        __m256i cspec = equal(ce, splat(255));
        c_special_vec[half][row] = cspec;
        if (lane_mask(cspec)) [[unlikely]] {
          c_special_rows[half] |= 1u << row;
        }
        __m256i v_fin = andnot(cspec, v);
        __m256i ce_fin = andnot(cspec, ce);
        __m256i cf_fin = andnot(cspec, cf);
        if constexpr (Rdna3) {
          __m256i normal = greater(ce_fin, splat(0));
          c_val[half][row] = bit_and(normal, bit_or(cf_fin, splat(1 << 26)));
          c_grid[half][row] =
              choose(normal, sub(ce_fin, splat(153)), splat(gfx12_dot_detail::kEmptyExponent));
          c_sign[half][row] =
              Packed ? greater(v_fin, splat(0x7fff)) : signed_shift_right(v_fin, 31);
        } else {
          __m256i cv = bit_or(cf_fin, _mm256_sign_epi32(splat(1 << 26), ce_fin));
          __m256i vs = Packed ? shift_left(v_fin, 16) : v_fin;
          c_val[half][row] = _mm256_sign_epi32(cv, vs);
          c_grid[half][row] = sub(maximum(ce_fin, splat(1)), splat(153));
        }
      }
    }

    uint32_t a_nan_rows = 0, a_inf_rows = 0, b_special[2] = {0, 0};
    __m256i any_bad[2] = {splat(0), splat(0)};

    for (int step = 0; step < 16 - N; step += N) {
      StageLeft<Rdna3> left[N];
      __m256i b_units[2][N], b_grid[2][N], b_sign[2][Rdna3 ? N : 1];
      decode_stage<Bf16, Rdna3, 0>(step, flags, a, b, left, b_units, b_grid, b_sign, a_nan_rows,
                                   a_inf_rows, b_special);
      for (int half = 0; half < 2; ++half) {
        __m256i bad_acc = any_bad[half];
        for (int row = 0; row < 16; ++row) {
          __m256i total, grid;
          __m256i cs = Rdna3 ? c_sign[half][row] : splat(0);
          stage_sum<Rdna3>(row, left, b_units[half], b_grid[half], b_sign[half], c_val[half][row],
                           c_grid[half][row], cs, total, grid);
          __m256i z_mask = equal(total, splat(0));
          __m256i ce, raw_cv;
          if constexpr (!Packed) {
            __m256i rounded = _mm256_castps_si256(_mm256_cvtepi32_ps(total));
            __m256i mag = bit_and(rounded, splat(0x7fffffff));
            ce = add(shift_right(mag, 23), grid);
            raw_cv = bit_or(bit_and(shift_left(rounded, 3), splat(FracMaskShl3)), splat(1 << 26));
          } else {
            __m256i magnitude = absolute(total);
            __m256i top = biased_top_bit(magnitude);
            __m256i exponent = add(top, grid);
            __m256i normalized = shift_left(magnitude, sub(splat(127 + 30), top));
            __m256i parity = bit_and(shift_right(normalized, 7), splat(1));
            __m256i rounded = shift_right(add(normalized, add(splat(63), parity)), 7);
            ce = add(exponent, shift_right(rounded, 24));
            raw_cv = bit_or(bit_and(shift_left(rounded, 3), splat(FracMaskShl3)), splat(1 << 26));
          }
          if constexpr (Rdna3) {
            __m256i keep = andnot(z_mask, greater(ce, splat(0)));
            bad_acc = bit_or(bad_acc, bit_and(keep, greater(ce, splat(254))));
            c_val[half][row] = bit_and(keep, raw_cv);
            c_grid[half][row] =
                choose(keep, sub(ce, splat(153)), splat(gfx12_dot_detail::kEmptyExponent));
            c_sign[half][row] = bit_and(keep, signed_shift_right(bit_xor(total, cs), 31));
          } else {
            bad_acc = bit_or(
                bad_acc, andnot(z_mask, bit_or(greater(splat(1), ce), greater(ce, splat(254)))));
            c_val[half][row] = _mm256_sign_epi32(raw_cv, total);
            c_grid[half][row] = choose(z_mask, splat(-152), sub(ce, splat(153)));
          }
        }
        any_bad[half] = bad_acc;
      }
    }

    // Final stage: pack branchlessly into FP32 or Packed BF16.
    {
      StageLeft<Rdna3> left[N];
      __m256i b_units[2][N], b_grid[2][N], b_sign[2][Rdna3 ? N : 1];
      decode_stage<Bf16, Rdna3, 0>(16 - N, flags, a, b, left, b_units, b_grid, b_sign, a_nan_rows,
                                   a_inf_rows, b_special);
      for (int half = 0; half < 2; ++half) {
        __m256i bad_acc = any_bad[half];
        for (int row = 0; row < 16; ++row) {
          __m256i total, grid;
          __m256i cs = Rdna3 ? c_sign[half][row] : splat(0);
          stage_sum<Rdna3>(row, left, b_units[half], b_grid[half], b_sign[half], c_val[half][row],
                           c_grid[half][row], cs, total, grid);
          __m256i z_mask = equal(total, splat(0));
          __m256i exp, bits, sign_src;
          if constexpr (!Packed) {
            __m256i rounded = _mm256_castps_si256(_mm256_cvtepi32_ps(total));
            __m256i mag = bit_and(rounded, splat(0x7fffffff));
            exp = add(shift_right(mag, 23), grid);
            bits = _mm256_min_epu32(add(mag, shift_left(grid, 23)), splat(0x7f800000));
            sign_src = rounded;
          } else {
            __m256i magnitude = absolute(total);
            __m256i top = biased_top_bit(magnitude);
            __m256i exponent = add(top, grid);
            __m256i normalized = shift_left(magnitude, sub(splat(127 + 30), top));
            __m256i parity = bit_and(shift_right(normalized, 7), splat(1));
            __m256i rounded = shift_right(add(normalized, add(splat(63), parity)), 7);
            exp = add(exponent, shift_right(rounded, 24));
            __m256i raw = add(shift_left(sub(exponent, splat(1)), 23), rounded);
            bits = _mm256_min_epu32(raw, splat(0x7f800000));
            sign_src = total;
          }
          __m256i res;
          if constexpr (Rdna3) {
            __m256i keep = andnot(z_mask, greater(exp, splat(0)));
            __m256i sign = bit_and(bit_xor(sign_src, cs), splat(Sign));
            res = bit_and(keep, bit_or(bits, sign));
          } else {
            bad_acc = bit_or(bad_acc, andnot(z_mask, greater(splat(1), exp)));
            __m256i sign = bit_and(sign_src, splat(Sign));
            res = andnot(z_mask, bit_or(bits, sign));
          }
          if constexpr (Packed) {
            res = shift_right(res, 16);
          }
          acc[half][row] = res;
        }
        any_bad[half] = bad_acc;
      }
    }

    for (int half = 0; half < 2; ++half) {
      bool bad_any = lane_mask(any_bad[half]) != 0;
      if (b_special[half] || a_inf_rows || bad_any) [[unlikely]] {
        uint32_t repair_mask = (b_special[half] || bad_any)
                                   ? 0xffffu
                                   : (a_inf_rows | a_nan_rows | c_special_rows[half]);
        repair_slice_rows<Bf16, Rdna3, Packed>(repair_mask, half, flags, saturate, acc[half],
                                               c_init[half], a, b);
      } else if (c_special_rows[half] | a_nan_rows) [[unlikely]] {
        for (uint32_t m = c_special_rows[half]; m; m &= m - 1) {
          int row = __builtin_ctz(m);
          __m256i cv = c_init[half][row];
          __m256i cnan = greater(bit_and(cv, splat(Packed ? 0x7fff : 0x7fffffff)), splat(CInf));
          __m256i resolved = bit_or(cv, bit_and(cnan, splat(CQuiet)));
          acc[half][row] = choose(c_special_vec[half][row], resolved, acc[half][row]);
        }
        for (uint32_t m = a_nan_rows; m; m &= m - 1) {
          acc[half][__builtin_ctz(m)] = splat(FactorNan);
        }
      }
    }
  } else {
    // Packed FP16: MXCSR-clean exact-only biased_top_bit + VPMADDWD + branchless
    // normal-range 16-row loops, deferring subnormals/overflows to cold repair.
    constexpr int OutFrac = 10, OutShift = 30 - OutFrac;
    constexpr int CInf = 0x7c00, CQuiet = 0x0200;
    const int FactorNan = int(packed16::special<false>(gfx12_dot_detail::kFactorNan));
    const __m256i clamp_limit = splat(saturate ? 0x7bff : 0x7c00);
    constexpr int GridBias = kGridBias<false, true>;

    __m256i c_special_vec[2][16];
    uint32_t c_special_rows[2] = {0, 0};
    for (int half = 0; half < 2; ++half) {
      for (int row = 0; row < 16; ++row) {
        __m256i v = c_init[half][row];
        __m256i ce = bit_and(shift_right(v, 10), splat(31));
        __m256i cspec = equal(ce, splat(31));
        c_special_vec[half][row] = cspec;
        if (lane_mask(cspec)) [[unlikely]] {
          c_special_rows[half] |= 1u << row;
        }
        acc[half][row] = andnot(cspec, v);
      }
    }

    uint32_t a_nan_rows = 0, a_inf_rows = 0, b_special[2] = {0, 0};
    __m256i any_bad[2] = {splat(0), splat(0)};
    for (int step = 0; step < 16; step += N) {
      StageLeft<Rdna3> left[N];
      __m256i b_units[2][N], b_grid[2][N], b_sign[2][Rdna3 ? N : 1];
      decode_stage<false, Rdna3, GridBias>(step, flags, a, b, left, b_units, b_grid, b_sign,
                                           a_nan_rows, a_inf_rows, b_special);
      const bool check_ovf = (step + N < 16) && !saturate;
      for (int half = 0; half < 2; ++half) {
        __m256i bad_acc = any_bad[half];
        for (int row = 0; row < 16; ++row) {
          __m256i old = acc[half][row];
          __m256i ce = bit_and(shift_right(old, 10), splat(31));
          __m256i cf = shift_left(bit_and(old, splat(1023)), 16);
          __m256i cv = bit_or(cf, _mm256_sign_epi32(splat(1 << 26), ce));
          __m256i c_val, c_grid = sub(maximum(ce, splat(1)), splat(128 + 26)), frame;
          if constexpr (Rdna3) {
            c_val = cv;
            frame = greater(old, splat(0x7fff));
          } else {
            c_val = _mm256_sign_epi32(cv, shift_left(old, 16));
            frame = splat(0);
          }

          __m256i total, grid;
          stage_sum<Rdna3>(row, left, b_units[half], b_grid[half], b_sign[half], c_val, c_grid,
                           frame, total, grid);

          __m256i magnitude = absolute(total);
          __m256i top = biased_top_bit(magnitude);
          __m256i exponent = add(top, grid);
          __m256i normalized = shift_left(magnitude, sub(splat(127 + 30), top));
          __m256i parity = bit_and(shift_right(normalized, OutShift), splat(1));
          __m256i rounded =
              shift_right(add(normalized, add(splat((1 << (OutShift - 1)) - 1), parity)), OutShift);
          __m256i bits = add(shift_left(exponent, OutFrac), rounded);
          bad_acc = bit_or(bad_acc, _mm256_sign_epi32(exponent, magnitude));
          bits = _mm256_min_epu32(bits, clamp_limit);
          if (check_ovf) {
            bad_acc = bit_or(bad_acc, equal(bits, splat(0x7c00)));
          }
          __m256i sign = bit_and(bit_xor(signed_shift_right(total, 31), frame), splat(0x8000));
          acc[half][row] = _mm256_sign_epi32(bit_or(bits, sign), magnitude);
        }
        any_bad[half] = bad_acc;
      }
    }

    for (int half = 0; half < 2; ++half) {
      bool bad_any = lane_mask(any_bad[half]) != 0;
      if (b_special[half] || a_inf_rows || bad_any) [[unlikely]] {
        uint32_t repair_mask = (b_special[half] || bad_any)
                                   ? 0xffffu
                                   : (a_inf_rows | a_nan_rows | c_special_rows[half]);
        repair_slice_rows<false, Rdna3, true>(repair_mask, half, flags, saturate, acc[half],
                                              c_init[half], a, b);
      } else if (c_special_rows[half] | a_nan_rows) [[unlikely]] {
        for (uint32_t m = c_special_rows[half]; m; m &= m - 1) {
          int row = __builtin_ctz(m);
          __m256i cv = c_init[half][row];
          __m256i cnan = greater(bit_and(cv, splat(0x7fff)), splat(CInf));
          __m256i resolved = bit_or(cv, bit_and(cnan, splat(CQuiet)));
          acc[half][row] = choose(c_special_vec[half][row], resolved, acc[half][row]);
        }
        for (uint32_t m = a_nan_rows; m; m &= m - 1) {
          acc[half][__builtin_ctz(m)] = splat(FactorNan);
        }
      }
    }
  }
}
} // namespace RJ_ARITH_IMPL_NAMESPACE::wmma_v3

// Map validated wave operands into the two eight-column slices. The staged
// destination preserves overlapping inputs and the unused half of a gfx11 D.
template <bool Bf16, int Lanes, bool Rdna3, bool Packed>
void wmma_exact_x86_64_v3(uint32_t instruction_flags, bool saturate, uint32_t *const *d,
                          const uint32_t *const *a, const uint32_t *const *b,
                          const uint32_t *const *c) {
  using namespace exact_dot;
  using namespace wmma_v3;
  constexpr int Registers = 256 / Lanes,
                DestinationRegisters = Packed && !Rdna3 ? Registers / 2 : Registers;
  auto layout = [](int row, int &reg, int &lane) {
    reg = Rdna3 ? row / (Lanes / 16) : row % Registers;
    lane = Rdna3 ? 16 * (row % (Lanes / 16))
                 : 16 * (row / 8) + (Lanes == 64 ? 32 * ((row / 4) % 2) : 0);
  };
  int high = instruction_flags & RJ_ARITH_WMMA_HIGH_C_D ? 16 : 0;
  const uint32_t *left[8], *right[8];
  for (int pair = 0; pair < 8; ++pair) {
    int k = !Rdna3 && Lanes == 32 ? wmma_gfx12_w32_k(2 * pair) : 2 * pair;
    int reg = Rdna3 ? k / 2 : (k % Registers) / 2, lane = Rdna3 ? 0 : 16 * (k / Registers);
    left[pair] = a[reg] + lane;
    right[pair] = b[reg] + lane;
  }
  __m256i c_init[2][16], acc[2][16];
  __m256i preserved[Rdna3 && Packed ? 2 : 1][16];
  for (int col = 0; col < 16; col += 8) {
    for (int row = 0; row < 16; ++row) {
      int reg, lane;
      layout(row, reg, lane);
      int dest = Packed && !Rdna3 ? reg / 2 : reg;
      auto v = load<__m256i>(c[dest] + lane + col);
      if constexpr (Rdna3 && Packed) {
        __m256i old = load<__m256i>(d[dest] + lane + col);
        preserved[col / 8][row] = andnot(splat(int(0xffffU << high)), old);
      }
      if constexpr (Packed) {
        v = bit_and(shift_right(v, splat(Rdna3 ? high : 16 * (reg % 2))), splat(0xffff));
      }
      if (instruction_flags & RJ_ARITH_WMMA_ABS_C) {
        v = bit_and(v, splat(Packed ? 0x7fff : 0x7fffffff));
      }
      if (instruction_flags & RJ_ARITH_WMMA_NEG_C) {
        v = bit_xor(v, splat(Packed ? 0x8000 : int(0x80000000U)));
      }
      c_init[col / 8][row] = v;
    }
  }
  accumulate_tile<Bf16, Rdna3, Packed>(instruction_flags, saturate, acc, c_init, left, right);

  // All inputs and preserved halves have been read; store in ascending
  // register order so the last store wins if D pointers alias.
  constexpr int Groups = Lanes / 16;
  if constexpr (Packed && !Rdna3) {
    for (int dest = 0; dest < DestinationRegisters; ++dest) {
      for (int g = 0; g < Groups; ++g) {
        int base_row = 2 * dest + (Lanes == 64 ? 8 * (g & 1) + 4 * (g >> 1) : 8 * g);
        for (int h = 0; h < 2; ++h) {
          __m256i packed = bit_or(acc[h][base_row], shift_left(acc[h][base_row + 1], 16));
          store(d[dest] + 16 * g + 8 * h, packed);
        }
      }
    }
  } else {
    for (int reg = 0; reg < DestinationRegisters; ++reg) {
      for (int g = 0; g < Groups; ++g) {
        int row =
            Rdna3 ? reg * Groups + g : reg + (Lanes == 64 ? 8 * (g & 1) + 4 * (g >> 1) : 8 * g);
        for (int h = 0; h < 2; ++h) {
          __m256i v = acc[h][row];
          if constexpr (Rdna3 && Packed) {
            v = bit_or(shift_left(v, splat(high)), preserved[h][row]);
          }
          store(d[reg] + 16 * g + 8 * h, v);
        }
      }
    }
  }
}

template void wmma_exact_x86_64_v3<false, 32, false, false>(uint32_t, bool, uint32_t *const *,
                                                            const uint32_t *const *,
                                                            const uint32_t *const *,
                                                            const uint32_t *const *);
template void wmma_exact_x86_64_v3<false, 32, false, true>(uint32_t, bool, uint32_t *const *,
                                                           const uint32_t *const *,
                                                           const uint32_t *const *,
                                                           const uint32_t *const *);
template void wmma_exact_x86_64_v3<false, 32, true, false>(uint32_t, bool, uint32_t *const *,
                                                           const uint32_t *const *,
                                                           const uint32_t *const *,
                                                           const uint32_t *const *);
template void wmma_exact_x86_64_v3<false, 32, true, true>(uint32_t, bool, uint32_t *const *,
                                                          const uint32_t *const *,
                                                          const uint32_t *const *,
                                                          const uint32_t *const *);
template void wmma_exact_x86_64_v3<false, 64, false, false>(uint32_t, bool, uint32_t *const *,
                                                            const uint32_t *const *,
                                                            const uint32_t *const *,
                                                            const uint32_t *const *);
template void wmma_exact_x86_64_v3<false, 64, false, true>(uint32_t, bool, uint32_t *const *,
                                                           const uint32_t *const *,
                                                           const uint32_t *const *,
                                                           const uint32_t *const *);
template void wmma_exact_x86_64_v3<false, 64, true, false>(uint32_t, bool, uint32_t *const *,
                                                           const uint32_t *const *,
                                                           const uint32_t *const *,
                                                           const uint32_t *const *);
template void wmma_exact_x86_64_v3<false, 64, true, true>(uint32_t, bool, uint32_t *const *,
                                                          const uint32_t *const *,
                                                          const uint32_t *const *,
                                                          const uint32_t *const *);
template void wmma_exact_x86_64_v3<true, 32, false, false>(uint32_t, bool, uint32_t *const *,
                                                           const uint32_t *const *,
                                                           const uint32_t *const *,
                                                           const uint32_t *const *);
template void wmma_exact_x86_64_v3<true, 32, false, true>(uint32_t, bool, uint32_t *const *,
                                                          const uint32_t *const *,
                                                          const uint32_t *const *,
                                                          const uint32_t *const *);
template void wmma_exact_x86_64_v3<true, 32, true, false>(uint32_t, bool, uint32_t *const *,
                                                          const uint32_t *const *,
                                                          const uint32_t *const *,
                                                          const uint32_t *const *);
template void wmma_exact_x86_64_v3<true, 32, true, true>(uint32_t, bool, uint32_t *const *,
                                                         const uint32_t *const *,
                                                         const uint32_t *const *,
                                                         const uint32_t *const *);
template void wmma_exact_x86_64_v3<true, 64, false, false>(uint32_t, bool, uint32_t *const *,
                                                           const uint32_t *const *,
                                                           const uint32_t *const *,
                                                           const uint32_t *const *);
template void wmma_exact_x86_64_v3<true, 64, false, true>(uint32_t, bool, uint32_t *const *,
                                                          const uint32_t *const *,
                                                          const uint32_t *const *,
                                                          const uint32_t *const *);
template void wmma_exact_x86_64_v3<true, 64, true, false>(uint32_t, bool, uint32_t *const *,
                                                          const uint32_t *const *,
                                                          const uint32_t *const *,
                                                          const uint32_t *const *);
template void wmma_exact_x86_64_v3<true, 64, true, true>(uint32_t, bool, uint32_t *const *,
                                                         const uint32_t *const *,
                                                         const uint32_t *const *,
                                                         const uint32_t *const *);

} // namespace rocjitsu::arith
