// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Vectorised empirical bit-exact gfx12 FP8 (E4M3 x E4M3) WMMA with FP32 D.
//
// The lanes of a vector are output COLUMNS of one row, so every step of the DOT8 model is
// per-lane: A contributes a broadcast, B a contiguous load, and C and D contiguous accesses.
// There is no horizontal reduction, shuffle or gather anywhere in the kernel.
//
// Three properties of the model keep the vector form short:
//   - The accumulator's clamped leading exponent, max(-126, c_leading), equals
//     max(exponent_field, 1) - 127 for every FP32 encoding, so no bit scan is needed. AVX2
//     has no leading-zero count, which would otherwise force a scalar fallback.
//   - A zero factor is given alignment exponent -137. An all-zero DOT8 then lands on product
//     grid -153, below the accumulator floor of -152, so it always loses the grid selection
//     and the empty-exponent sentinel disappears. Any value at or below -137 works; the
//     weaker "never outranks a nonzero product" bound alone does not.
//   - A variable arithmetic right shift saturates its count at 31 and fills with the sign
//     bit, which is exactly the floor a term below the grid needs, so the alignment carries
//     no clamp and no mask.
//
// Every aligned term and the running total stay inside int32: a product needs at most 26 bits
// on its group grid, four of them 28, and two groups plus the accumulator under 30.
//
// The result is rounded by converting the grid-unit count to FP32 and scaling by 2^grid. The
// scale is exact, so the conversion is the model's single round-to-nearest-even. Lanes whose
// result is subnormal or overflows, and grids below the FP32 normal floor, take the integer
// pack instead; that same test makes the kernel independent of MXCSR flush-to-zero, because a
// flushed result reads back with a zero exponent field and is repaired.

#pragma once

#include "dot/dot.h"
#include "wmma/layout.h"

#include <immintrin.h>
#include <stdint.h>

// RJ_ARITH_IMPL_NAMESPACE is set per translation unit by the build (x86_64_v3, x86_64_v4), so
// the AVX2 and AVX-512 kernels are distinct symbols. Sharing one name across both would let the
// linker substitute the AVX-512 build into the AVX2 entry point.
namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {
namespace wmma_gfx12_fp8_exact {

// Alignment exponent given to a zero factor; see the file comment.
inline constexpr int kZeroExponent = -137;
inline constexpr uint32_t kFactorNan = 0xffc00a3d;
inline constexpr int kQuietNanBit = 0x00400000;

#if defined(__AVX2__)

// One decoded matrix: [k][index], so a column block is a contiguous load and a row a
// broadcast. `nan` marks indices carrying a NaN factor at any k.
struct MatrixAvx2 {
  alignas(32) int32_t sig[16][16];
  alignas(32) int32_t exp[16][16];
  alignas(32) int32_t nan[16];
};

// E4M3 -> signed significand, alignment exponent and NaN mask, eight factors at a time.
inline void decode_avx2(__m256i x, __m256i &sig, __m256i &exp, __m256i &nan) {
  const __m256i zero = _mm256_setzero_si256();
  const __m256i f = _mm256_and_si256(_mm256_srli_epi32(x, 3), _mm256_set1_epi32(15));
  const __m256i s =
      _mm256_or_si256(_mm256_and_si256(x, _mm256_set1_epi32(7)),
                      _mm256_and_si256(_mm256_cmpgt_epi32(f, zero), _mm256_set1_epi32(8)));
  const __m256i sign = _mm256_srai_epi32(_mm256_slli_epi32(x, 24), 31);
  sig = _mm256_sub_epi32(_mm256_xor_si256(s, sign), sign);
  exp = _mm256_blendv_epi8(
      _mm256_set1_epi32(kZeroExponent),
      _mm256_sub_epi32(_mm256_max_epi32(f, _mm256_set1_epi32(1)), _mm256_set1_epi32(7)),
      _mm256_cmpgt_epi32(s, zero));
  nan = _mm256_cmpeq_epi32(_mm256_and_si256(x, _mm256_set1_epi32(0x7f)), _mm256_set1_epi32(0x7f));
}

// Decode one gfx12 Wave32 factor tile into [k][row_or_column].
inline void decode_matrix_avx2(const uint32_t *const *v, MatrixAvx2 &out) {
  __m256i nan_low = _mm256_setzero_si256(), nan_high = nan_low;
  for (int k = 0; k < 16; ++k) {
    const int local = wmma_gfx12_packed_local(k);
    const uint32_t *source = v[local / 4] + 16 * (k / 8);
    const __m128i shift = _mm_cvtsi32_si128(8 * (local % 4));
    __m256i sig, exp, nan;
    for (int index = 0; index < 16; index += 8) {
      const __m256i words = _mm256_srl_epi32(
          _mm256_loadu_si256(reinterpret_cast<const __m256i *>(source + index)), shift);
      decode_avx2(_mm256_and_si256(words, _mm256_set1_epi32(0xff)), sig, exp, nan);
      _mm256_store_si256(reinterpret_cast<__m256i *>(out.sig[k] + index), sig);
      _mm256_store_si256(reinterpret_cast<__m256i *>(out.exp[k] + index), exp);
      (index ? nan_high : nan_low) = _mm256_or_si256(index ? nan_high : nan_low, nan);
    }
  }
  _mm256_store_si256(reinterpret_cast<__m256i *>(out.nan), nan_low);
  _mm256_store_si256(reinterpret_cast<__m256i *>(out.nan + 8), nan_high);
}

// One DOT8 stage for eight columns of one row: C plus two independently aligned groups of
// four products, then one FP32 rounding. `taint` marks lanes an outer NaN blend will overwrite.
inline __m256i dot8_avx2(const MatrixAvx2 &A, int row, const MatrixAvx2 &B, int column_base,
                         int k0, __m256i acc, __m256i taint) {
  const __m256i zero = _mm256_setzero_si256();
  __m256i product[8], exponent[8];
  for (int i = 0; i < 8; ++i) {
    const int k = k0 + i;
    product[i] = _mm256_mullo_epi32(
        _mm256_set1_epi32(A.sig[k][row]),
        _mm256_load_si256(reinterpret_cast<const __m256i *>(B.sig[k] + column_base)));
    exponent[i] = _mm256_add_epi32(
        _mm256_set1_epi32(A.exp[k][row]),
        _mm256_load_si256(reinterpret_cast<const __m256i *>(B.exp[k] + column_base)));
  }
  const __m256i top_low = _mm256_max_epi32(_mm256_max_epi32(exponent[0], exponent[1]),
                                           _mm256_max_epi32(exponent[2], exponent[3]));
  const __m256i top_high = _mm256_max_epi32(_mm256_max_epi32(exponent[4], exponent[5]),
                                            _mm256_max_epi32(exponent[6], exponent[7]));
  __m256i group_low = zero, group_high = zero;
  for (int i = 0; i < 4; ++i) {
    group_low = _mm256_add_epi32(
        group_low, _mm256_srav_epi32(_mm256_slli_epi32(product[i], 18),
                                     _mm256_sub_epi32(top_low, exponent[i])));
    group_high = _mm256_add_epi32(
        group_high, _mm256_srav_epi32(_mm256_slli_epi32(product[4 + i], 18),
                                      _mm256_sub_epi32(top_high, exponent[4 + i])));
  }
  const __m256i field = _mm256_and_si256(_mm256_srli_epi32(acc, 23), _mm256_set1_epi32(255));
  const __m256i field1 = _mm256_max_epi32(field, _mm256_set1_epi32(1));
  const __m256i grid = _mm256_max_epi32(
      _mm256_sub_epi32(_mm256_max_epi32(top_low, top_high), _mm256_set1_epi32(24)),
      _mm256_sub_epi32(field1, _mm256_set1_epi32(153)));
  __m256i units = _mm256_add_epi32(
      _mm256_srav_epi32(group_low, _mm256_sub_epi32(grid, _mm256_sub_epi32(top_low,
                                                                          _mm256_set1_epi32(24)))),
      _mm256_srav_epi32(group_high,
                        _mm256_sub_epi32(grid, _mm256_sub_epi32(top_high, _mm256_set1_epi32(24)))));

  const __m256i significand =
      _mm256_or_si256(_mm256_and_si256(acc, _mm256_set1_epi32(0x7fffff)),
                      _mm256_and_si256(_mm256_cmpgt_epi32(field, zero),
                                       _mm256_set1_epi32(0x800000)));
  const __m256i sign = _mm256_srai_epi32(acc, 31);
  const __m256i signed_c = _mm256_sub_epi32(_mm256_xor_si256(significand, sign), sign);
  const __m256i down = _mm256_sub_epi32(grid, _mm256_sub_epi32(field1, _mm256_set1_epi32(150)));
  const __m256i c_units = _mm256_srav_epi32(
      _mm256_sllv_epi32(signed_c, _mm256_max_epi32(_mm256_sub_epi32(zero, down), zero)),
      _mm256_max_epi32(down, zero));
  // A negative accumulator that retains no magnitude bit contributes zero rather than one
  // floor unit, above the FP32 normal floor (gfx1201, rocjitsu #13691).
  const __m256i vanished = _mm256_and_si256(
      _mm256_and_si256(_mm256_cmpeq_epi32(c_units, _mm256_set1_epi32(-1)),
                       _mm256_cmpgt_epi32(down, zero)),
      _mm256_cmpgt_epi32(grid, _mm256_set1_epi32(-126)));
  units = _mm256_add_epi32(units, _mm256_andnot_si256(vanished, c_units));

  const __m256i rounded = _mm256_castps_si256(_mm256_mul_ps(
      _mm256_cvtepi32_ps(units),
      _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(grid, _mm256_set1_epi32(127)), 23))));
  const __m256i magnitude = _mm256_and_si256(acc, _mm256_set1_epi32(0x7fffffff));
  const __m256i acc_nan = _mm256_cmpgt_epi32(magnitude, _mm256_set1_epi32(0x7f800000));
  const __m256i acc_special =
      _mm256_or_si256(acc_nan, _mm256_cmpeq_epi32(magnitude, _mm256_set1_epi32(0x7f800000)));
  __m256i out = _mm256_blendv_epi8(
      rounded, _mm256_or_si256(acc, _mm256_and_si256(acc_nan, _mm256_set1_epi32(kQuietNanBit))),
      acc_special);

  const __m256i result_field =
      _mm256_and_si256(_mm256_srli_epi32(rounded, 23), _mm256_set1_epi32(255));
  __m256i repair = _mm256_or_si256(
      _mm256_andnot_si256(_mm256_cmpeq_epi32(units, zero), _mm256_cmpeq_epi32(result_field, zero)),
      _mm256_or_si256(_mm256_cmpeq_epi32(result_field, _mm256_set1_epi32(255)),
                      _mm256_cmpgt_epi32(_mm256_set1_epi32(-126), grid)));
  repair = _mm256_andnot_si256(_mm256_or_si256(acc_special, taint), repair);
  if (!_mm256_testz_si256(repair, repair)) {
    alignas(32) int32_t unit_lane[8], grid_lane[8], repair_lane[8], out_lane[8];
    _mm256_store_si256(reinterpret_cast<__m256i *>(unit_lane), units);
    _mm256_store_si256(reinterpret_cast<__m256i *>(grid_lane), grid);
    _mm256_store_si256(reinterpret_cast<__m256i *>(repair_lane), repair);
    _mm256_store_si256(reinterpret_cast<__m256i *>(out_lane), out);
    for (int lane = 0; lane < 8; ++lane) {
      if (repair_lane[lane]) {
        out_lane[lane] = int32_t(gfx12_dot_detail::pack(unit_lane[lane], grid_lane[lane]));
      }
    }
    out = _mm256_load_si256(reinterpret_cast<const __m256i *>(out_lane));
  }
  return out;
}

// Evaluate a gfx12 Wave32 FP8 x FP8 tile. D may alias A, B or C: every source read completes
// before the first destination store.
inline void dense_avx2(uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                       const uint32_t *const *c) {
  MatrixAvx2 left, right;
  decode_matrix_avx2(a, left);
  decode_matrix_avx2(b, right);
  alignas(32) uint32_t result[8][32];
  for (int row = 0; row < 16; ++row) {
    const int reg = row % 8;
    for (int column_base = 0; column_base < 16; column_base += 8) {
      const int lane = column_base + 16 * (row / 8);
      const __m256i taint = _mm256_or_si256(
          _mm256_set1_epi32(left.nan[row]),
          _mm256_load_si256(reinterpret_cast<const __m256i *>(right.nan + column_base)));
      __m256i acc = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(c[reg] + lane));
      acc = dot8_avx2(left, row, right, column_base, 0, acc, taint);
      acc = dot8_avx2(left, row, right, column_base, 8, acc, taint);
      _mm256_store_si256(
          reinterpret_cast<__m256i *>(result[reg] + lane),
          _mm256_blendv_epi8(acc, _mm256_set1_epi32(int(kFactorNan)), taint));
    }
  }
  for (int reg = 0; reg < 8; ++reg) {
    for (int lane = 0; lane < 32; ++lane) {
      d[reg][lane] = result[reg][lane];
    }
  }
}
#endif // __AVX2__

#if defined(__AVX512F__)

struct MatrixAvx512 {
  alignas(64) int32_t sig[16][16];
  alignas(64) int32_t exp[16][16];
  __mmask16 nan;
};

inline void decode_matrix_avx512(const uint32_t *const *v, MatrixAvx512 &out) {
  const __m512i zero = _mm512_setzero_si512();
  out.nan = 0;
  for (int k = 0; k < 16; ++k) {
    const int local = wmma_gfx12_packed_local(k);
    const __m512i words = _mm512_srl_epi32(_mm512_loadu_si512(v[local / 4] + 16 * (k / 8)),
                                           _mm_cvtsi32_si128(8 * (local % 4)));
    const __m512i x = _mm512_and_si512(words, _mm512_set1_epi32(0xff));
    const __m512i f = _mm512_and_si512(_mm512_srli_epi32(x, 3), _mm512_set1_epi32(15));
    const __m512i s = _mm512_or_si512(
        _mm512_and_si512(x, _mm512_set1_epi32(7)),
        _mm512_maskz_set1_epi32(_mm512_cmpgt_epi32_mask(f, zero), 8));
    const __m512i sign = _mm512_srai_epi32(_mm512_slli_epi32(x, 24), 31);
    _mm512_store_si512(out.sig[k], _mm512_sub_epi32(_mm512_xor_si512(s, sign), sign));
    _mm512_store_si512(
        out.exp[k],
        _mm512_mask_blend_epi32(
            _mm512_cmpgt_epi32_mask(s, zero), _mm512_set1_epi32(kZeroExponent),
            _mm512_sub_epi32(_mm512_max_epi32(f, _mm512_set1_epi32(1)), _mm512_set1_epi32(7))));
    out.nan |= _mm512_cmpeq_epi32_mask(_mm512_and_si512(x, _mm512_set1_epi32(0x7f)),
                                       _mm512_set1_epi32(0x7f));
  }
}

// One DOT8 stage for all sixteen columns of one row.
inline __m512i dot8_avx512(const MatrixAvx512 &A, int row, const MatrixAvx512 &B, int k0,
                           __m512i acc, __mmask16 taint) {
  const __m512i zero = _mm512_setzero_si512();
  __m512i product[8], exponent[8];
  for (int i = 0; i < 8; ++i) {
    const int k = k0 + i;
    product[i] = _mm512_mullo_epi32(_mm512_set1_epi32(A.sig[k][row]), _mm512_load_si512(B.sig[k]));
    exponent[i] = _mm512_add_epi32(_mm512_set1_epi32(A.exp[k][row]), _mm512_load_si512(B.exp[k]));
  }
  const __m512i top_low = _mm512_max_epi32(_mm512_max_epi32(exponent[0], exponent[1]),
                                           _mm512_max_epi32(exponent[2], exponent[3]));
  const __m512i top_high = _mm512_max_epi32(_mm512_max_epi32(exponent[4], exponent[5]),
                                            _mm512_max_epi32(exponent[6], exponent[7]));
  __m512i group_low = zero, group_high = zero;
  for (int i = 0; i < 4; ++i) {
    group_low = _mm512_add_epi32(
        group_low, _mm512_srav_epi32(_mm512_slli_epi32(product[i], 18),
                                     _mm512_sub_epi32(top_low, exponent[i])));
    group_high = _mm512_add_epi32(
        group_high, _mm512_srav_epi32(_mm512_slli_epi32(product[4 + i], 18),
                                      _mm512_sub_epi32(top_high, exponent[4 + i])));
  }
  const __m512i field = _mm512_and_si512(_mm512_srli_epi32(acc, 23), _mm512_set1_epi32(255));
  const __m512i field1 = _mm512_max_epi32(field, _mm512_set1_epi32(1));
  const __m512i grid = _mm512_max_epi32(
      _mm512_sub_epi32(_mm512_max_epi32(top_low, top_high), _mm512_set1_epi32(24)),
      _mm512_sub_epi32(field1, _mm512_set1_epi32(153)));
  __m512i units = _mm512_add_epi32(
      _mm512_srav_epi32(group_low,
                        _mm512_sub_epi32(grid, _mm512_sub_epi32(top_low, _mm512_set1_epi32(24)))),
      _mm512_srav_epi32(group_high,
                        _mm512_sub_epi32(grid, _mm512_sub_epi32(top_high, _mm512_set1_epi32(24)))));

  const __m512i significand = _mm512_or_si512(
      _mm512_and_si512(acc, _mm512_set1_epi32(0x7fffff)),
      _mm512_maskz_set1_epi32(_mm512_cmpgt_epi32_mask(field, zero), 0x800000));
  const __m512i sign = _mm512_srai_epi32(acc, 31);
  const __m512i signed_c = _mm512_sub_epi32(_mm512_xor_si512(significand, sign), sign);
  const __m512i down = _mm512_sub_epi32(grid, _mm512_sub_epi32(field1, _mm512_set1_epi32(150)));
  const __m512i c_units = _mm512_srav_epi32(
      _mm512_sllv_epi32(signed_c, _mm512_max_epi32(_mm512_sub_epi32(zero, down), zero)),
      _mm512_max_epi32(down, zero));
  const __mmask16 vanished = _mm512_cmpeq_epi32_mask(c_units, _mm512_set1_epi32(-1)) &
                             _mm512_cmpgt_epi32_mask(down, zero) &
                             _mm512_cmpgt_epi32_mask(grid, _mm512_set1_epi32(-126));
  units = _mm512_mask_add_epi32(units, __mmask16(~vanished), units, c_units);

  const __m512i rounded = _mm512_castps_si512(_mm512_mul_ps(
      _mm512_cvtepi32_ps(units),
      _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_add_epi32(grid, _mm512_set1_epi32(127)), 23))));
  const __m512i magnitude = _mm512_and_si512(acc, _mm512_set1_epi32(0x7fffffff));
  const __mmask16 acc_nan = _mm512_cmpgt_epi32_mask(magnitude, _mm512_set1_epi32(0x7f800000));
  const __mmask16 acc_special =
      acc_nan | _mm512_cmpeq_epi32_mask(magnitude, _mm512_set1_epi32(0x7f800000));
  __m512i out = _mm512_mask_blend_epi32(
      acc_special, rounded,
      _mm512_mask_or_epi32(acc, acc_nan, acc, _mm512_set1_epi32(kQuietNanBit)));

  const __m512i result_field =
      _mm512_and_si512(_mm512_srli_epi32(rounded, 23), _mm512_set1_epi32(255));
  const __mmask16 repair = __mmask16(
      ((_mm512_cmpeq_epi32_mask(result_field, zero) & ~_mm512_cmpeq_epi32_mask(units, zero)) |
       _mm512_cmpeq_epi32_mask(result_field, _mm512_set1_epi32(255)) |
       _mm512_cmplt_epi32_mask(grid, _mm512_set1_epi32(-126))) &
      ~(acc_special | taint));
  if (repair) {
    alignas(64) int32_t unit_lane[16], grid_lane[16], out_lane[16];
    _mm512_store_si512(unit_lane, units);
    _mm512_store_si512(grid_lane, grid);
    _mm512_store_si512(out_lane, out);
    for (int lane = 0; lane < 16; ++lane) {
      if ((repair >> lane) & 1) {
        out_lane[lane] = int32_t(gfx12_dot_detail::pack(unit_lane[lane], grid_lane[lane]));
      }
    }
    out = _mm512_load_si512(out_lane);
  }
  return out;
}

inline void dense_avx512(uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                         const uint32_t *const *c) {
  MatrixAvx512 left, right;
  decode_matrix_avx512(a, left);
  decode_matrix_avx512(b, right);
  alignas(64) uint32_t result[8][32];
  for (int row = 0; row < 16; ++row) {
    const int reg = row % 8, lane = 16 * (row / 8);
    const __mmask16 taint = ((left.nan >> row) & 1) ? __mmask16(0xffff) : right.nan;
    __m512i acc = _mm512_loadu_si512(c[reg] + lane);
    acc = dot8_avx512(left, row, right, 0, acc, taint);
    acc = dot8_avx512(left, row, right, 8, acc, taint);
    _mm512_store_si512(result[reg] + lane,
                       _mm512_mask_blend_epi32(taint, acc, _mm512_set1_epi32(int(kFactorNan))));
  }
  for (int reg = 0; reg < 8; ++reg) {
    for (int lane = 0; lane < 32; ++lane) {
      d[reg][lane] = result[reg][lane];
    }
  }
}
#endif // __AVX512F__

} // namespace wmma_gfx12_fp8_exact
} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
