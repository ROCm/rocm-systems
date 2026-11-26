// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// x86-64-v4 (AVX-512) bit-exact implementation of captured gfx942 MFMA.
#include "mfma/backends.h"
#include "mfma/x86_64/gfx942_exact_common.h"
#include <immintrin.h>

namespace rocjitsu::arith {
namespace {
using namespace x86_64_v4;

// Decode 16 packed elements (in the low bits of each 32-bit lane) directly to
// signed mantissas (optionally pre-scaled by 16 for sum_products) and exponents.
template <Gfx942Elem Elem, bool ScaleBy16>
inline __mmask16 decode16_v4(__m512i raw, __m512i &mant, __m512i &exp) {
  const __m512i zero = _mm512_setzero_si512();
  const __m512i one = _mm512_set1_epi32(1);
  if constexpr (Elem == Gfx942Elem::F16) {
    const __m512i u = _mm512_and_si512(raw, _mm512_set1_epi32(0xffff));
    const __m512i e = _mm512_and_si512(_mm512_srli_epi32(u, 10), _mm512_set1_epi32(0x1f));
    const __m512i f = _mm512_and_si512(u, _mm512_set1_epi32(0x3ff));
    const __mmask16 norm = _mm512_cmpneq_epi32_mask(e, zero);
    __m512i mag = _mm512_mask_or_epi32(f, norm, f, _mm512_set1_epi32(0x400));
    if constexpr (ScaleBy16) {
      mag = _mm512_slli_epi32(mag, 4);
    }
    const __mmask16 neg = _mm512_test_epi32_mask(u, _mm512_set1_epi32(0x8000));
    mant = _mm512_mask_sub_epi32(mag, neg, zero, mag);
    exp = _mm512_sub_epi32(_mm512_max_epi32(e, one), _mm512_set1_epi32(15));
    return _mm512_cmpeq_epi32_mask(e, _mm512_set1_epi32(31));
  } else if constexpr (Elem == Gfx942Elem::Bf16) {
    const __m512i u = _mm512_and_si512(raw, _mm512_set1_epi32(0xffff));
    const __m512i e = _mm512_and_si512(_mm512_srli_epi32(u, 7), _mm512_set1_epi32(0xff));
    const __m512i f = _mm512_and_si512(u, _mm512_set1_epi32(0x7f));
    const __mmask16 norm = _mm512_cmpneq_epi32_mask(e, zero);
    __m512i mag = _mm512_mask_or_epi32(f, norm, f, _mm512_set1_epi32(0x80));
    mag = _mm512_slli_epi32(mag, ScaleBy16 ? 7 : 3);
    const __mmask16 neg = _mm512_test_epi32_mask(u, _mm512_set1_epi32(0x8000));
    mant = _mm512_mask_sub_epi32(mag, neg, zero, mag);
    exp = _mm512_sub_epi32(_mm512_max_epi32(e, one), _mm512_set1_epi32(127));
    return _mm512_cmpeq_epi32_mask(e, _mm512_set1_epi32(0xff));
  } else if constexpr (Elem == Gfx942Elem::Xf32) {
    const __m512i e = _mm512_and_si512(_mm512_srli_epi32(raw, 23), _mm512_set1_epi32(0xff));
    const __m512i f = _mm512_and_si512(_mm512_srli_epi32(raw, 13), _mm512_set1_epi32(0x3ff));
    const __mmask16 norm = _mm512_cmpneq_epi32_mask(e, zero);
    __m512i mag = _mm512_mask_or_epi32(f, norm, f, _mm512_set1_epi32(0x400));
    if constexpr (ScaleBy16) {
      mag = _mm512_slli_epi32(mag, 4);
    }
    const __mmask16 neg = _mm512_movepi32_mask(raw);
    mant = _mm512_mask_sub_epi32(mag, neg, zero, mag);
    exp = _mm512_sub_epi32(_mm512_max_epi32(e, one), _mm512_set1_epi32(127));
    return _mm512_cmpeq_epi32_mask(e, _mm512_set1_epi32(0xff));
  } else if constexpr (Elem == Gfx942Elem::Fp8Fnuz) {
    const __m512i u = _mm512_and_si512(raw, _mm512_set1_epi32(0xff));
    const __m512i e = _mm512_and_si512(_mm512_srli_epi32(u, 3), _mm512_set1_epi32(0x0f));
    const __m512i f = _mm512_and_si512(u, _mm512_set1_epi32(0x07));
    const __mmask16 norm = _mm512_cmpneq_epi32_mask(e, zero);
    __m512i mag = _mm512_mask_or_epi32(f, norm, f, _mm512_set1_epi32(8));
    mag = _mm512_slli_epi32(mag, ScaleBy16 ? 11 : 7);
    const __mmask16 neg = _mm512_test_epi32_mask(u, _mm512_set1_epi32(0x80));
    mant = _mm512_mask_sub_epi32(mag, neg, zero, mag);
    exp = _mm512_sub_epi32(_mm512_max_epi32(e, one), _mm512_set1_epi32(8));
    return _mm512_cmpeq_epi32_mask(u, _mm512_set1_epi32(0x80));
  } else {
    const __m512i u = _mm512_and_si512(raw, _mm512_set1_epi32(0xff));
    const __m512i e = _mm512_and_si512(_mm512_srli_epi32(u, 2), _mm512_set1_epi32(0x1f));
    const __m512i f = _mm512_and_si512(u, _mm512_set1_epi32(0x03));
    const __mmask16 norm = _mm512_cmpneq_epi32_mask(e, zero);
    __m512i mag = _mm512_mask_or_epi32(f, norm, f, _mm512_set1_epi32(4));
    mag = _mm512_slli_epi32(mag, ScaleBy16 ? 12 : 8);
    const __mmask16 neg = _mm512_test_epi32_mask(u, _mm512_set1_epi32(0x80));
    mant = _mm512_mask_sub_epi32(mag, neg, zero, mag);
    exp = _mm512_sub_epi32(_mm512_max_epi32(e, one), _mm512_set1_epi32(16));
    return _mm512_cmpeq_epi32_mask(u, _mm512_set1_epi32(0x80));
  }
}

// Align and sum `Products` (4 or 8) products across 16 columns.
// `a_m[i]` carries int16_t(a_mant * 16) in the low 16 bits and 0 in the high 16 bits.
// `b_m[i]` carries int16_t(b_mant) in the low 16 bits of each 32-bit lane.
template <int Products, bool CanOverflow>
inline __mmask16 sum_products16_v4(const __m512i *a_m, const __m512i *a_e, const __m512i *b_m,
                                   const __m512i *b_e, __m512i &sum_out, __m512i &top_exp_out) {
  const __m512i zero = _mm512_setzero_si512();
  const __m512i zero_exp = _mm512_set1_epi32(kGfx942ZeroProductExponent);
  const __m512i ovf_limit = _mm512_set1_epi32(128);
  __m512i prod_mag[Products];
  __m512i prod_exp[Products];
  __mmask16 prod_neg[Products];
  __m512i top_exp = zero_exp;
  __mmask16 may_overflow = 0;

  for (int i = 0; i < Products; ++i) {
    const __m512i prod = _mm512_madd_epi16(a_m[i], b_m[i]);
    const __mmask16 nz = _mm512_cmpneq_epi32_mask(prod, zero);
    const __m512i exp = _mm512_mask_add_epi32(zero_exp, nz, a_e[i], b_e[i]);
    prod_mag[i] = _mm512_abs_epi32(prod);
    prod_exp[i] = exp;
    prod_neg[i] = _mm512_movepi32_mask(prod);
    top_exp = (i == 0) ? exp : _mm512_max_epi32(top_exp, exp);
    if constexpr (CanOverflow) {
      const __m512i ovf = _mm512_add_epi32(exp, _mm512_srli_epi32(prod_mag[i], 25));
      may_overflow |= _mm512_cmpge_epi32_mask(ovf, ovf_limit);
    }
  }

  __m512i sum = zero;
  for (int i = 0; i < Products; ++i) {
    const __m512i shift = _mm512_sub_epi32(top_exp, prod_exp[i]);
    const __m512i q_mag = _mm512_srlv_epi32(prod_mag[i], shift);
    const __m512i q = _mm512_mask_sub_epi32(q_mag, prod_neg[i], zero, q_mag);
    sum = (i == 0) ? q : _mm512_add_epi32(sum, q);
  }
  sum_out = sum;
  top_exp_out = top_exp;
  return may_overflow;
}

// Convert 8 normalized dot+C sums (64-bit) to f32 nearest-even with suppressed exceptions.
template <bool DropFar>
inline __m256i add_accumulators8_core_v4(__m256i dot32, __m256i c_mant32, __m256i top_exp32,
                                         __m256i dot_shift32, __m256i c_shift32, __mmask8 keep_c) {
  const __m512i zero64 = _mm512_setzero_si512();
  const __m512i one64 = _mm512_set1_epi64(1);
  const __m512i top_exp = _mm512_cvtepi32_epi64(top_exp32);

  const __m512i dot_part = _mm512_srav_epi64(_mm512_slli_epi64(_mm512_cvtepi32_epi64(dot32), 8),
                                             _mm512_cvtepu32_epi64(dot_shift32));
  __m512i c_part =
      _mm512_slli_epi64(_mm512_srav_epi64(_mm512_slli_epi64(_mm512_cvtepi32_epi64(c_mant32), 1),
                                          _mm512_cvtepu32_epi64(c_shift32)),
                        8);
  if constexpr (DropFar) {
    c_part = _mm512_maskz_mov_epi64(keep_c, c_part);
  }
  const __m512i total = _mm512_add_epi64(dot_part, c_part);

  // Normalize (rounding down to 31 fraction bits when right > 0) and round to f32 RNE.
  // right = max(total_top - 31, -125 - top_exp) where total_top = 63 - lzcnt(|total| | 1).
  // For r = max(right, 0): shifted = total >> r, exact value is shifted * 2^(top_exp - 32 + r).
  const __m512i total_mag = _mm512_abs_epi64(total);
  const __m512i lz = _mm512_lzcnt_epi64(_mm512_or_si512(total_mag, one64));
  const __m512i r1 = _mm512_sub_epi64(_mm512_set1_epi64(32), lz);
  const __m512i r2 = _mm512_sub_epi64(_mm512_set1_epi64(-125), top_exp);
  const __m512i r = _mm512_max_epi64(_mm512_max_epi64(r1, r2), zero64);
  const __m512i shifted = _mm512_srav_epi64(total, r);

  const __mmask8 nz_shifted = _mm512_cmpneq_epi64_mask(shifted, zero64);
  const __m512d shifted_d = _mm512_cvtepi64_pd(shifted);
  const __m512i exp_shift = _mm512_add_epi64(_mm512_sub_epi64(top_exp, _mm512_set1_epi64(32)), r);
  const __m512i scaled_bits = _mm512_maskz_add_epi64(nz_shifted, _mm512_castpd_si512(shifted_d),
                                                     _mm512_slli_epi64(exp_shift, 52));
  const __m256 rounded_f = _mm512_cvt_roundpd_ps(_mm512_castsi512_pd(scaled_bits),
                                                 _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  return _mm256_castps_si256(rounded_f);
}

template <bool DropFar>
inline __m512i add_accumulators16_v4(__m512i dot, __m512i dot_exp, __m512i c_bits,
                                     __mmask16 &c_inf_nan_out) {
  const __m512i zero32 = _mm512_setzero_si512();
  const __m512i one32 = _mm512_set1_epi32(1);
  const __m512i c_exp_field =
      _mm512_and_si512(_mm512_srli_epi32(c_bits, 23), _mm512_set1_epi32(0xff));
  c_inf_nan_out = _mm512_cmpeq_epi32_mask(c_exp_field, _mm512_set1_epi32(0xff));
  const __m512i c_exp =
      _mm512_sub_epi32(_mm512_max_epi32(c_exp_field, one32), _mm512_set1_epi32(127));
  const __mmask16 c_norm = _mm512_cmpneq_epi32_mask(c_exp_field, zero32);
  const __m512i c_frac = _mm512_and_si512(c_bits, _mm512_set1_epi32(0x7fffff));
  const __m512i c_mag = _mm512_mask_or_epi32(c_frac, c_norm, c_frac, _mm512_set1_epi32(0x800000));
  const __mmask16 c_neg = _mm512_movepi32_mask(c_bits);
  const __m512i c_mant = _mm512_mask_sub_epi32(c_mag, c_neg, zero32, c_mag);

  const __m512i top_exp = _mm512_max_epi32(dot_exp, c_exp);
  const __m512i dot_shift = _mm512_sub_epi32(top_exp, dot_exp);
  const __m512i c_shift = _mm512_sub_epi32(top_exp, c_exp);
  __mmask16 keep_c = 0xffffu;
  if constexpr (DropFar) {
    keep_c = _mm512_cmple_epi32_mask(c_shift, _mm512_set1_epi32(25));
  }

  const __m256i lo = add_accumulators8_core_v4<DropFar>(
      _mm512_castsi512_si256(dot), _mm512_castsi512_si256(c_mant), _mm512_castsi512_si256(top_exp),
      _mm512_castsi512_si256(dot_shift), _mm512_castsi512_si256(c_shift), __mmask8(keep_c));
  const __m256i hi = add_accumulators8_core_v4<DropFar>(
      _mm512_extracti64x4_epi64(dot, 1), _mm512_extracti64x4_epi64(c_mant, 1),
      _mm512_extracti64x4_epi64(top_exp, 1), _mm512_extracti64x4_epi64(dot_shift, 1),
      _mm512_extracti64x4_epi64(c_shift, 1), __mmask8(keep_c >> 8));
  const __m512i res = _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
  // Flush -0.0 (including negative underflow to zero) to +0.0.
  const __mmask16 nz_res = _mm512_test_epi32_mask(res, _mm512_set1_epi32(0x7fffffff));
  return _mm512_maskz_mov_epi32(nz_res, res);
}

// 4x4x4_16b dense specialization: 4 batches (16 lanes) processed per ZMM!
template <MfmaGfx942Format Format>
inline void mfma_gfx942_exact_4x4x4_v4(uint32_t *const *d, const uint32_t *const *a,
                                       const uint32_t *const *b, const uint32_t *const *c) {
  constexpr Gfx942Elem elem = gfx942_elem_a<Format>();
  constexpr bool can_ovf = gfx942_can_product_overflow<Format>();
  alignas(64) uint32_t result[4][64];
  const __m512i low16_mask = _mm512_set1_epi32(0xffff);

  for (int b4 = 0; b4 < 4; ++b4) {
    const int base_lane = b4 * 16;
    __m512i a_m[4], a_e[4], b_m[4], b_e[4];
    __mmask16 a_inf[4], b_inf_any = 0;
    for (int reg = 0; reg < 2; ++reg) {
      const __m512i aw = _mm512_loadu_si512(&a[reg][base_lane]);
      const __m512i bw = _mm512_loadu_si512(&b[reg][base_lane]);
      a_inf[2 * reg] = decode16_v4<elem, true>(aw, a_m[2 * reg], a_e[2 * reg]);
      a_inf[2 * reg + 1] =
          decode16_v4<elem, true>(_mm512_srli_epi32(aw, 16), a_m[2 * reg + 1], a_e[2 * reg + 1]);
      b_inf_any |= decode16_v4<elem, false>(bw, b_m[2 * reg], b_e[2 * reg]);
      b_inf_any |=
          decode16_v4<elem, false>(_mm512_srli_epi32(bw, 16), b_m[2 * reg + 1], b_e[2 * reg + 1]);
      // Zero high 16 bits of a_m so vpmaddwd multiplies low 16 bits cleanly.
      a_m[2 * reg] = _mm512_and_si512(a_m[2 * reg], low16_mask);
      a_m[2 * reg + 1] = _mm512_and_si512(a_m[2 * reg + 1], low16_mask);
    }
    const __mmask16 a_inf_any = a_inf[0] | a_inf[1] | a_inf[2] | a_inf[3];

    // Broadcast lane `Row` of each 4-lane group across that 4-lane group.
    auto run_row = [&]<int Row, _MM_PERM_ENUM ShufImm>() {
      __m512i row_am[4], row_ae[4];
      for (int k = 0; k < 4; ++k) {
        row_am[k] = _mm512_shuffle_epi32(a_m[k], ShufImm);
        row_ae[k] = _mm512_shuffle_epi32(a_e[k], ShufImm);
      }
      __m512i dot, dot_exp;
      const __mmask16 may_ovf =
          sum_products16_v4<4, can_ovf>(row_am, row_ae, b_m, b_e, dot, dot_exp);
      const __m512i c_bits = _mm512_loadu_si512(&c[Row][base_lane]);
      __mmask16 c_inf = 0;
      const __m512i res = add_accumulators16_v4<false>(dot, dot_exp, c_bits, c_inf);
      _mm512_store_si512(&result[Row][base_lane], res);

      // Expand a_inf_any at lane (batch_in_group * 4 + Row) to all 4 columns of that batch.
      uint16_t a_row_inf = 0;
      for (int g = 0; g < 4; ++g) {
        if ((a_inf_any >> (g * 4 + Row)) & 1u) {
          a_row_inf |= uint16_t(0xfu << (g * 4));
        }
      }
      uint32_t special = uint32_t(a_row_inf | b_inf_any | c_inf | may_ovf);
      if (__builtin_expect(special != 0u, 0)) {
        while (special != 0u) {
          const int lane = __builtin_ctz(special);
          special &= special - 1u;
          const int batch = b4 * 4 + (lane >> 2);
          const int col = lane & 3;
          gfx942_non_finite_dense_one<Format, 4, 4, 16>(
              batch, Row, col, 0, 4, a, b, c[Row][base_lane + lane], result[Row][base_lane + lane]);
        }
      }
    };
    run_row.template operator()<0, _MM_PERM_AAAA>();
    run_row.template operator()<1, _MM_PERM_BBBB>();
    run_row.template operator()<2, _MM_PERM_CCCC>();
    run_row.template operator()<3, _MM_PERM_DDDD>();
  }

  for (int reg = 0; reg < 4; ++reg) {
    for (int lane = 0; lane < 64; lane += 16) {
      _mm512_storeu_si512(&d[reg][lane], _mm512_load_si512(&result[reg][lane]));
    }
  }
}

// General dense M=16 or M=32 kernel (covers 18 of the 20 captured gfx942 dense forms).
template <MfmaGfx942Format Format, int M, int K, int Batch>
inline void mfma_gfx942_exact_dense_v4(uint32_t *const *d, const uint32_t *const *a,
                                       const uint32_t *const *b, const uint32_t *const *c) {
  constexpr Gfx942Elem elem_a = gfx942_elem_a<Format>();
  constexpr Gfx942Elem elem_b = gfx942_elem_b<Format>();
  constexpr int per_word = gfx942_per_word<Format>();
  constexpr int step_k = gfx942_products_per_step<Format>();
  constexpr bool even_odd = gfx942_even_odd<Format>();
  constexpr bool can_ovf = gfx942_can_product_overflow<Format>();
  constexpr int per_lane = K * M * Batch / 64;
  constexpr int output_regs = M * M * Batch / 64;
  constexpr int col_chunks = M / 16; // 1 for M=16, 2 for M=32
  constexpr int num_steps = (K + step_k - 1) / step_k;

  alignas(64) uint32_t result[output_regs][64];

  for (int batch = 0; batch < Batch; ++batch) {
    // Decode A and B for this batch directly to (mantissa, exponent) + per-step inf/nan masks.
    alignas(64) int32_t a_mant[K][M];
    alignas(64) int32_t a_exp[K][M];
    alignas(64) __m512i b_mant[K][col_chunks];
    alignas(64) __m512i b_exp[K][col_chunks];
    uint32_t a_inf_mask[num_steps]{};
    uint16_t b_inf_mask[num_steps][col_chunks]{};

    for (int depth = 0; depth < K; depth += per_word) {
      const int step_idx = depth / step_k;
      const int lane_base = batch * M + (depth / per_lane) * M * Batch;
      const int reg = (depth % per_lane) / per_word;
      for (int ch = 0; ch < col_chunks; ++ch) {
        const __m512i aw = _mm512_loadu_si512(&a[reg][lane_base + ch * 16]);
        const __m512i bw = _mm512_loadu_si512(&b[reg][lane_base + ch * 16]);
        for (int sub = 0; sub < per_word; ++sub) {
          const int dk = depth + sub;
          const __m512i a_shifted = (sub == 0) ? aw : _mm512_srli_epi32(aw, sub * (32 / per_word));
          const __m512i b_shifted = (sub == 0) ? bw : _mm512_srli_epi32(bw, sub * (32 / per_word));
          __m512i am, ae, bm, be;
          const __mmask16 a_inf = decode16_v4<elem_a, true>(a_shifted, am, ae);
          const __mmask16 b_inf = decode16_v4<elem_b, false>(b_shifted, bm, be);
          _mm512_store_si512(&a_mant[dk][ch * 16], am);
          _mm512_store_si512(&a_exp[dk][ch * 16], ae);
          b_mant[dk][ch] = bm;
          b_exp[dk][ch] = be;
          a_inf_mask[step_idx] |= uint32_t(a_inf) << (ch * 16);
          b_inf_mask[step_idx][ch] |= uint16_t(b_inf);
        }
      }
    }

    for (int row = 0; row < M; ++row) {
      const int out_reg = batch * (M * M / 64) + (row / (256 / M)) * 4 + (row & 3);
      const int out_lane_base = ((row >> 2) % (64 / M)) * M;
      __m512i acc[col_chunks];
      for (int ch = 0; ch < col_chunks; ++ch) {
        acc[ch] = _mm512_loadu_si512(&c[out_reg][out_lane_base + ch * 16]);
      }

      for (int step_idx = 0; step_idx < num_steps; ++step_idx) {
        const int start = step_idx * step_k;
        constexpr int step_products = (K < step_k) ? K : step_k;
        const bool row_a_inf = ((a_inf_mask[step_idx] >> row) & 1u) != 0u;

        if constexpr (even_odd) {
          constexpr int half_p = step_products / 2;
          __m512i even_am[half_p], even_ae[half_p], odd_am[half_p], odd_ae[half_p];
          for (int i = 0; i < half_p; ++i) {
            even_am[i] = _mm512_set1_epi32(uint16_t(a_mant[start + 2 * i][row]));
            even_ae[i] = _mm512_set1_epi32(a_exp[start + 2 * i][row]);
            odd_am[i] = _mm512_set1_epi32(uint16_t(a_mant[start + 2 * i + 1][row]));
            odd_ae[i] = _mm512_set1_epi32(a_exp[start + 2 * i + 1][row]);
          }
          for (int ch = 0; ch < col_chunks; ++ch) {
            __m512i even_bm[half_p], even_be[half_p], odd_bm[half_p], odd_be[half_p];
            for (int i = 0; i < half_p; ++i) {
              even_bm[i] = b_mant[start + 2 * i][ch];
              even_be[i] = b_exp[start + 2 * i][ch];
              odd_bm[i] = b_mant[start + 2 * i + 1][ch];
              odd_be[i] = b_exp[start + 2 * i + 1][ch];
            }
            __m512i dot, dot_exp, odd, odd_exp;
            sum_products16_v4<half_p, false>(even_am, even_ae, even_bm, even_be, dot, dot_exp);
            sum_products16_v4<half_p, false>(odd_am, odd_ae, odd_bm, odd_be, odd, odd_exp);
            const __m512i top_exp = _mm512_max_epi32(dot_exp, odd_exp);
            dot = _mm512_add_epi32(_mm512_srav_epi32(dot, _mm512_sub_epi32(top_exp, dot_exp)),
                                   _mm512_srav_epi32(odd, _mm512_sub_epi32(top_exp, odd_exp)));
            dot_exp = top_exp;
            __mmask16 c_inf = 0;
            const __m512i prev_acc = acc[ch];
            acc[ch] = add_accumulators16_v4<true>(dot, dot_exp, prev_acc, c_inf);
            uint32_t special =
                row_a_inf ? 0xffffu : uint32_t(b_inf_mask[step_idx][ch] | uint16_t(c_inf));
            if (__builtin_expect(special != 0u, 0)) {
              alignas(64) uint32_t prev_buf[16], res_buf[16];
              _mm512_store_si512(prev_buf, prev_acc);
              _mm512_store_si512(res_buf, acc[ch]);
              while (special != 0u) {
                const int lane = __builtin_ctz(special);
                special &= special - 1u;
                gfx942_non_finite_dense_one<Format, M, K, Batch>(batch, row, ch * 16 + lane, start,
                                                                 step_products, a, b,
                                                                 prev_buf[lane], res_buf[lane]);
              }
              acc[ch] = _mm512_load_si512(res_buf);
            }
          }
        } else {
          __m512i row_am[step_products], row_ae[step_products];
          for (int i = 0; i < step_products; ++i) {
            row_am[i] = _mm512_set1_epi32(uint16_t(a_mant[start + i][row]));
            row_ae[i] = _mm512_set1_epi32(a_exp[start + i][row]);
          }
          for (int ch = 0; ch < col_chunks; ++ch) {
            __m512i col_bm[step_products], col_be[step_products];
            for (int i = 0; i < step_products; ++i) {
              col_bm[i] = b_mant[start + i][ch];
              col_be[i] = b_exp[start + i][ch];
            }
            __m512i dot, dot_exp;
            const __mmask16 may_ovf = sum_products16_v4<step_products, can_ovf>(
                row_am, row_ae, col_bm, col_be, dot, dot_exp);
            __mmask16 c_inf = 0;
            const __m512i prev_acc = acc[ch];
            acc[ch] = add_accumulators16_v4<false>(dot, dot_exp, prev_acc, c_inf);
            uint32_t special =
                row_a_inf
                    ? 0xffffu
                    : uint32_t(b_inf_mask[step_idx][ch] | uint16_t(c_inf) | uint16_t(may_ovf));
            if (__builtin_expect(special != 0u, 0)) {
              alignas(64) uint32_t prev_buf[16], res_buf[16];
              _mm512_store_si512(prev_buf, prev_acc);
              _mm512_store_si512(res_buf, acc[ch]);
              while (special != 0u) {
                const int lane = __builtin_ctz(special);
                special &= special - 1u;
                gfx942_non_finite_dense_one<Format, M, K, Batch>(batch, row, ch * 16 + lane, start,
                                                                 step_products, a, b,
                                                                 prev_buf[lane], res_buf[lane]);
              }
              acc[ch] = _mm512_load_si512(res_buf);
            }
          }
        }
      }

      for (int ch = 0; ch < col_chunks; ++ch) {
        _mm512_store_si512(&result[out_reg][out_lane_base + ch * 16], acc[ch]);
      }
    }
  }

  for (int reg = 0; reg < output_regs; ++reg) {
    for (int lane = 0; lane < 64; lane += 16) {
      _mm512_storeu_si512(&d[reg][lane], _mm512_load_si512(&result[reg][lane]));
    }
  }
}

using SparseRepairFn = void (*)(int, int, int, int, const uint32_t *const *,
                                const uint32_t *const *, const uint32_t *, uint32_t, uint32_t &);

// Format-independent AVX-512 reduction loop for captured gfx942 sparse exact SMFMAC.
// Sharing this loop across FP8/BF8 combinations keeps code size bounded while processing
// 16 output columns per vector across all 12 sparse forms.
template <int M, int K, int StepK, bool EvenOdd, bool CanOvf>
[[gnu::noinline]] void smfmac_gfx942_exact_reduce_v4(
    uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
    const uint32_t *selected_index, const __m512i (*b_mant)[K], const __m512i (*b_exp)[K],
    const uint16_t (*b_inf)[K], const int32_t (*a_mant)[M], const int32_t (*a_exp)[M],
    const uint32_t *a_inf_mask, SparseRepairFn repair_one) {
  constexpr SmfmacLayout layout = EvenOdd ? SmfmacLayout::Cdna3Fp8 : SmfmacLayout::Cdna3F16;
  constexpr int output_regs = M * M / 64;
  constexpr int col_chunks = M / 16;
  constexpr int half_k = K / 2;
  constexpr int num_steps = half_k / StepK;

  alignas(64) uint32_t result[output_regs][64];
  for (int row = 0; row < M; ++row) {
    const int out_reg = (row / (256 / M)) * 4 + (row & 3);
    const int out_lane_base = ((row >> 2) % (64 / M)) * M;
    __m512i acc[col_chunks];
    for (int ch = 0; ch < col_chunks; ++ch) {
      acc[ch] = _mm512_loadu_si512(&d[out_reg][out_lane_base + ch * 16]);
    }

    for (int step_idx = 0; step_idx < num_steps; ++step_idx) {
      const int start = step_idx * StepK;
      const bool row_a_inf = ((a_inf_mask[step_idx] >> row) & 1u) != 0u;
      int sel[StepK];
      for (int i = 0; i < StepK; i += 2) {
        const int t = start + i;
        const int q_log = t / 2;
        const int q =
            (M == 16 && K == 64) ? ((q_log & 3) + ((q_log & 4) << 1) + ((q_log & 8) >> 1)) : q_log;
        const SmfmacSparseLoc sloc = smfmac_sparse_loc<layout, M, K>(row, q, 0);
        const uint32_t nib = selected_index[sloc.lane] >> (4 * sloc.nibble);
        sel[i] = 4 * q + int(nib & 3u);
        sel[i + 1] = 4 * q + int((nib >> 2) & 3u);
      }

      if constexpr (EvenOdd) {
        constexpr int half_p = StepK / 2;
        __m512i even_am[half_p], even_ae[half_p], odd_am[half_p], odd_ae[half_p];
        for (int i = 0; i < half_p; ++i) {
          even_am[i] = _mm512_set1_epi32(uint16_t(a_mant[start + 2 * i][row]));
          even_ae[i] = _mm512_set1_epi32(a_exp[start + 2 * i][row]);
          odd_am[i] = _mm512_set1_epi32(uint16_t(a_mant[start + 2 * i + 1][row]));
          odd_ae[i] = _mm512_set1_epi32(a_exp[start + 2 * i + 1][row]);
        }
        for (int ch = 0; ch < col_chunks; ++ch) {
          __m512i even_bm[half_p], even_be[half_p], odd_bm[half_p], odd_be[half_p];
          uint16_t b_step_inf = 0;
          for (int i = 0; i < half_p; ++i) {
            const int se = sel[2 * i];
            const int so = sel[2 * i + 1];
            even_bm[i] = b_mant[ch][se];
            even_be[i] = b_exp[ch][se];
            odd_bm[i] = b_mant[ch][so];
            odd_be[i] = b_exp[ch][so];
            b_step_inf |= uint16_t(b_inf[ch][se] | b_inf[ch][so]);
          }
          __m512i dot, dot_exp, odd, odd_exp;
          sum_products16_v4<half_p, false>(even_am, even_ae, even_bm, even_be, dot, dot_exp);
          sum_products16_v4<half_p, false>(odd_am, odd_ae, odd_bm, odd_be, odd, odd_exp);
          const __m512i top_exp = _mm512_max_epi32(dot_exp, odd_exp);
          dot = _mm512_add_epi32(_mm512_srav_epi32(dot, _mm512_sub_epi32(top_exp, dot_exp)),
                                 _mm512_srav_epi32(odd, _mm512_sub_epi32(top_exp, odd_exp)));
          dot_exp = top_exp;
          __mmask16 c_inf = 0;
          const __m512i prev_acc = acc[ch];
          acc[ch] = add_accumulators16_v4<true>(dot, dot_exp, prev_acc, c_inf);
          uint32_t special = row_a_inf ? 0xffffu : uint32_t(b_step_inf | uint16_t(c_inf));
          if (__builtin_expect(special != 0u, 0)) {
            alignas(64) uint32_t prev_buf[16], res_buf[16];
            _mm512_store_si512(prev_buf, prev_acc);
            _mm512_store_si512(res_buf, acc[ch]);
            while (special != 0u) {
              const int lane = __builtin_ctz(special);
              special &= special - 1u;
              repair_one(row, ch * 16 + lane, start, StepK, a, b, selected_index, prev_buf[lane],
                         res_buf[lane]);
            }
            acc[ch] = _mm512_load_si512(res_buf);
          }
        }
      } else {
        __m512i row_am[StepK], row_ae[StepK];
        for (int i = 0; i < StepK; ++i) {
          row_am[i] = _mm512_set1_epi32(uint16_t(a_mant[start + i][row]));
          row_ae[i] = _mm512_set1_epi32(a_exp[start + i][row]);
        }
        for (int ch = 0; ch < col_chunks; ++ch) {
          __m512i col_bm[StepK], col_be[StepK];
          uint16_t b_step_inf = 0;
          for (int i = 0; i < StepK; ++i) {
            const int si = sel[i];
            col_bm[i] = b_mant[ch][si];
            col_be[i] = b_exp[ch][si];
            b_step_inf |= b_inf[ch][si];
          }
          __m512i dot, dot_exp;
          const __mmask16 may_ovf =
              sum_products16_v4<StepK, CanOvf>(row_am, row_ae, col_bm, col_be, dot, dot_exp);
          __mmask16 c_inf = 0;
          const __m512i prev_acc = acc[ch];
          acc[ch] = add_accumulators16_v4<false>(dot, dot_exp, prev_acc, c_inf);
          uint32_t special =
              row_a_inf ? 0xffffu : uint32_t(b_step_inf | uint16_t(c_inf) | uint16_t(may_ovf));
          if (__builtin_expect(special != 0u, 0)) {
            alignas(64) uint32_t prev_buf[16], res_buf[16];
            _mm512_store_si512(prev_buf, prev_acc);
            _mm512_store_si512(res_buf, acc[ch]);
            while (special != 0u) {
              const int lane = __builtin_ctz(special);
              special &= special - 1u;
              repair_one(row, ch * 16 + lane, start, StepK, a, b, selected_index, prev_buf[lane],
                         res_buf[lane]);
            }
            acc[ch] = _mm512_load_si512(res_buf);
          }
        }
      }
    }

    for (int ch = 0; ch < col_chunks; ++ch) {
      _mm512_store_si512(&result[out_reg][out_lane_base + ch * 16], acc[ch]);
    }
  }

  for (int reg = 0; reg < output_regs; ++reg) {
    for (int lane = 0; lane < 64; lane += 16) {
      _mm512_storeu_si512(&d[reg][lane], _mm512_load_si512(&result[reg][lane]));
    }
  }
}

template <MfmaGfx942Format Format, int M, int K>
inline void smfmac_gfx942_exact_kernel_v4(uint64_t controls, uint32_t *const *d,
                                          const uint32_t *const *a, const uint32_t *const *b,
                                          const uint32_t *const *index) {
  constexpr bool half = Format == MfmaGfx942Format::F16 || Format == MfmaGfx942Format::Bf16;
  constexpr SmfmacLayout layout = half ? SmfmacLayout::Cdna3F16 : SmfmacLayout::Cdna3Fp8;
  constexpr Gfx942Elem elem_a = gfx942_elem_a<Format>();
  constexpr Gfx942Elem elem_b = gfx942_elem_b<Format>();
  constexpr int per_word = gfx942_per_word<Format>();
  constexpr int step_k = gfx942_products_per_step<Format>();
  constexpr bool even_odd = gfx942_even_odd<Format>();
  constexpr bool can_ovf = gfx942_can_product_overflow<Format>();
  constexpr int set_bits = M * K / 64;
  constexpr int col_chunks = M / 16;
  constexpr int half_k = K / 2;
  constexpr int num_steps = half_k / step_k;

  const int shift = (controls & 3) == 0 ? set_bits * int((controls >> 3) & (32 / set_bits - 1)) : 0;
  alignas(64) uint32_t selected_index[64];
  if (shift == 0) {
    for (int lane = 0; lane < 64; lane += 16) {
      _mm512_store_si512(&selected_index[lane], _mm512_loadu_si512(&index[0][lane]));
    }
  } else {
    const __m128i sh = _mm_cvtsi32_si128(shift);
    for (int lane = 0; lane < 64; lane += 16) {
      _mm512_store_si512(&selected_index[lane],
                         _mm512_srl_epi32(_mm512_loadu_si512(&index[0][lane]), sh));
    }
  }

  alignas(64) __m512i b_mant[col_chunks][K];
  alignas(64) __m512i b_exp[col_chunks][K];
  uint16_t b_inf[col_chunks][K];
  for (int depth = 0; depth < K; depth += per_word) {
    const SmfmacDenseLoc dloc = smfmac_dense_loc<layout, M>(depth, 0);
    const int reg = dloc.element / per_word;
    const int lane_base = dloc.lane;
    for (int ch = 0; ch < col_chunks; ++ch) {
      const __m512i bw = _mm512_loadu_si512(&b[reg][lane_base + ch * 16]);
      for (int sub = 0; sub < per_word; ++sub) {
        const __m512i bs = (sub == 0) ? bw : _mm512_srli_epi32(bw, sub * (32 / per_word));
        b_inf[ch][depth + sub] = uint16_t(
            decode16_v4<elem_b, false>(bs, b_mant[ch][depth + sub], b_exp[ch][depth + sub]));
      }
    }
  }

  alignas(64) int32_t a_mant[half_k][M];
  alignas(64) int32_t a_exp[half_k][M];
  uint32_t a_inf_mask[num_steps]{};
  for (int t = 0; t < half_k; t += 2) {
    const int q_log = t / 2;
    const int q =
        (M == 16 && K == 64) ? ((q_log & 3) + ((q_log & 4) << 1) + ((q_log & 8) >> 1)) : q_log;
    const SmfmacSparseLoc sloc = smfmac_sparse_loc<layout, M, K>(0, q, 0);
    const int reg = sloc.element / per_word;
    const int base_sub = sloc.element % per_word;
    const int lane_base = sloc.lane;
    const int step_idx = t / step_k;
    for (int ch = 0; ch < col_chunks; ++ch) {
      const __m512i aw = _mm512_loadu_si512(&a[reg][lane_base + ch * 16]);
      for (int s = 0; s < 2; ++s) {
        const int sub = base_sub + s;
        const __m512i as = (sub == 0) ? aw : _mm512_srli_epi32(aw, sub * (32 / per_word));
        __m512i am, ae;
        const __mmask16 inf = decode16_v4<elem_a, true>(as, am, ae);
        _mm512_store_si512(&a_mant[t + s][ch * 16], am);
        _mm512_store_si512(&a_exp[t + s][ch * 16], ae);
        a_inf_mask[step_idx] |= uint32_t(uint16_t(inf)) << (ch * 16);
      }
    }
  }

  smfmac_gfx942_exact_reduce_v4<M, K, step_k, even_odd, can_ovf>(
      d, a, b, selected_index, b_mant, b_exp, b_inf, a_mant, a_exp, a_inf_mask,
      &gfx942_non_finite_sparse_one<Format, M, K>);
}

} // namespace

template <MfmaGfx942Format Format, int M, int K, int Batch>
void mfma_gfx942_exact_x86_64_v4(uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c) {
  if constexpr (M == 4 && K == 4 && Batch == 16) {
    mfma_gfx942_exact_4x4x4_v4<Format>(d, a, b, c);
  } else {
    mfma_gfx942_exact_dense_v4<Format, M, K, Batch>(d, a, b, c);
  }
}

template <MfmaGfx942Format Format, int M, int K>
void smfmac_gfx942_exact_x86_64_v4(uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b, const uint32_t *const *index) {
  smfmac_gfx942_exact_kernel_v4<Format, M, K>(controls, d, a, b, index);
}

// Explicit instantiations for all 20 captured gfx942 dense exact MFMA forms.
template void mfma_gfx942_exact_x86_64_v4<MfmaGfx942Format::F16, 32, 8, 1>(uint32_t *const *,
                                                                           const uint32_t *const *,
                                                                           const uint32_t *const *,
                                                                           const uint32_t *const *);
template void mfma_gfx942_exact_x86_64_v4<MfmaGfx942Format::F16, 16, 16, 1>(
    uint32_t *const *, const uint32_t *const *, const uint32_t *const *, const uint32_t *const *);
template void mfma_gfx942_exact_x86_64_v4<MfmaGfx942Format::F16, 32, 4, 2>(uint32_t *const *,
                                                                           const uint32_t *const *,
                                                                           const uint32_t *const *,
                                                                           const uint32_t *const *);
template void mfma_gfx942_exact_x86_64_v4<MfmaGfx942Format::F16, 16, 4, 4>(uint32_t *const *,
                                                                           const uint32_t *const *,
                                                                           const uint32_t *const *,
                                                                           const uint32_t *const *);
template void mfma_gfx942_exact_x86_64_v4<MfmaGfx942Format::F16, 4, 4, 16>(uint32_t *const *,
                                                                           const uint32_t *const *,
                                                                           const uint32_t *const *,
                                                                           const uint32_t *const *);

template void mfma_gfx942_exact_x86_64_v4<MfmaGfx942Format::Bf16, 32, 8, 1>(
    uint32_t *const *, const uint32_t *const *, const uint32_t *const *, const uint32_t *const *);
template void mfma_gfx942_exact_x86_64_v4<MfmaGfx942Format::Bf16, 16, 16, 1>(
    uint32_t *const *, const uint32_t *const *, const uint32_t *const *, const uint32_t *const *);
template void mfma_gfx942_exact_x86_64_v4<MfmaGfx942Format::Bf16, 32, 4, 2>(
    uint32_t *const *, const uint32_t *const *, const uint32_t *const *, const uint32_t *const *);
template void mfma_gfx942_exact_x86_64_v4<MfmaGfx942Format::Bf16, 16, 4, 4>(
    uint32_t *const *, const uint32_t *const *, const uint32_t *const *, const uint32_t *const *);
template void mfma_gfx942_exact_x86_64_v4<MfmaGfx942Format::Bf16, 4, 4, 16>(
    uint32_t *const *, const uint32_t *const *, const uint32_t *const *, const uint32_t *const *);

template void mfma_gfx942_exact_x86_64_v4<MfmaGfx942Format::Xf32, 32, 4, 1>(
    uint32_t *const *, const uint32_t *const *, const uint32_t *const *, const uint32_t *const *);
template void mfma_gfx942_exact_x86_64_v4<MfmaGfx942Format::Xf32, 16, 8, 1>(
    uint32_t *const *, const uint32_t *const *, const uint32_t *const *, const uint32_t *const *);

template void mfma_gfx942_exact_x86_64_v4<MfmaGfx942Format::Fp8Fp8, 16, 32, 1>(
    uint32_t *const *, const uint32_t *const *, const uint32_t *const *, const uint32_t *const *);
template void mfma_gfx942_exact_x86_64_v4<MfmaGfx942Format::Fp8Bf8, 16, 32, 1>(
    uint32_t *const *, const uint32_t *const *, const uint32_t *const *, const uint32_t *const *);
template void mfma_gfx942_exact_x86_64_v4<MfmaGfx942Format::Bf8Fp8, 16, 32, 1>(
    uint32_t *const *, const uint32_t *const *, const uint32_t *const *, const uint32_t *const *);
template void mfma_gfx942_exact_x86_64_v4<MfmaGfx942Format::Bf8Bf8, 16, 32, 1>(
    uint32_t *const *, const uint32_t *const *, const uint32_t *const *, const uint32_t *const *);

template void mfma_gfx942_exact_x86_64_v4<MfmaGfx942Format::Fp8Fp8, 32, 16, 1>(
    uint32_t *const *, const uint32_t *const *, const uint32_t *const *, const uint32_t *const *);
template void mfma_gfx942_exact_x86_64_v4<MfmaGfx942Format::Fp8Bf8, 32, 16, 1>(
    uint32_t *const *, const uint32_t *const *, const uint32_t *const *, const uint32_t *const *);
template void mfma_gfx942_exact_x86_64_v4<MfmaGfx942Format::Bf8Fp8, 32, 16, 1>(
    uint32_t *const *, const uint32_t *const *, const uint32_t *const *, const uint32_t *const *);
template void mfma_gfx942_exact_x86_64_v4<MfmaGfx942Format::Bf8Bf8, 32, 16, 1>(
    uint32_t *const *, const uint32_t *const *, const uint32_t *const *, const uint32_t *const *);

// Explicit instantiations for all 12 captured gfx942 sparse exact SMFMAC forms.
template void smfmac_gfx942_exact_x86_64_v4<MfmaGfx942Format::F16, 16, 32>(uint64_t,
                                                                           uint32_t *const *,
                                                                           const uint32_t *const *,
                                                                           const uint32_t *const *,
                                                                           const uint32_t *const *);
template void smfmac_gfx942_exact_x86_64_v4<MfmaGfx942Format::Bf16, 16, 32>(
    uint64_t, uint32_t *const *, const uint32_t *const *, const uint32_t *const *,
    const uint32_t *const *);
template void smfmac_gfx942_exact_x86_64_v4<MfmaGfx942Format::F16, 32, 16>(uint64_t,
                                                                           uint32_t *const *,
                                                                           const uint32_t *const *,
                                                                           const uint32_t *const *,
                                                                           const uint32_t *const *);
template void smfmac_gfx942_exact_x86_64_v4<MfmaGfx942Format::Bf16, 32, 16>(
    uint64_t, uint32_t *const *, const uint32_t *const *, const uint32_t *const *,
    const uint32_t *const *);
template void smfmac_gfx942_exact_x86_64_v4<MfmaGfx942Format::Fp8Fp8, 16, 64>(
    uint64_t, uint32_t *const *, const uint32_t *const *, const uint32_t *const *,
    const uint32_t *const *);
template void smfmac_gfx942_exact_x86_64_v4<MfmaGfx942Format::Fp8Bf8, 16, 64>(
    uint64_t, uint32_t *const *, const uint32_t *const *, const uint32_t *const *,
    const uint32_t *const *);
template void smfmac_gfx942_exact_x86_64_v4<MfmaGfx942Format::Bf8Fp8, 16, 64>(
    uint64_t, uint32_t *const *, const uint32_t *const *, const uint32_t *const *,
    const uint32_t *const *);
template void smfmac_gfx942_exact_x86_64_v4<MfmaGfx942Format::Bf8Bf8, 16, 64>(
    uint64_t, uint32_t *const *, const uint32_t *const *, const uint32_t *const *,
    const uint32_t *const *);
template void smfmac_gfx942_exact_x86_64_v4<MfmaGfx942Format::Fp8Fp8, 32, 32>(
    uint64_t, uint32_t *const *, const uint32_t *const *, const uint32_t *const *,
    const uint32_t *const *);
template void smfmac_gfx942_exact_x86_64_v4<MfmaGfx942Format::Fp8Bf8, 32, 32>(
    uint64_t, uint32_t *const *, const uint32_t *const *, const uint32_t *const *,
    const uint32_t *const *);
template void smfmac_gfx942_exact_x86_64_v4<MfmaGfx942Format::Bf8Fp8, 32, 32>(
    uint64_t, uint32_t *const *, const uint32_t *const *, const uint32_t *const *,
    const uint32_t *const *);
template void smfmac_gfx942_exact_x86_64_v4<MfmaGfx942Format::Bf8Bf8, 32, 32>(
    uint64_t, uint32_t *const *, const uint32_t *const *, const uint32_t *const *,
    const uint32_t *const *);

} // namespace rocjitsu::arith
