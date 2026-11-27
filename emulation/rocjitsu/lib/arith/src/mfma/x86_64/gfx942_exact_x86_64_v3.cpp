// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// x86-64-v3 (AVX2) bit-exact implementation of captured gfx942 MFMA and SMFMAC.
#include "mfma/backends.h"
#include "mfma/x86_64/gfx942_exact_common.h"
#include <immintrin.h>

namespace rocjitsu::arith {
namespace {
using namespace x86_64_v3;

inline __m256i srav_epi64_v3(__m256i x, __m256i shift) {
  const __m256i sign = _mm256_cmpgt_epi64(_mm256_setzero_si256(), x);
  return _mm256_xor_si256(_mm256_srlv_epi64(_mm256_xor_si256(x, sign), shift), sign);
}

inline __m256i abs_epi64_v3(__m256i x) {
  const __m256i sign = _mm256_cmpgt_epi64(_mm256_setzero_si256(), x);
  return _mm256_sub_epi64(_mm256_xor_si256(x, sign), sign);
}

inline __m256i max_epi64_v3(__m256i a, __m256i b) {
  return _mm256_blendv_epi8(b, a, _mm256_cmpgt_epi64(a, b));
}

// Exact bit-index of highest set bit for 1 <= u < 2^52 using exact double subtraction
// (2^52 + u) - 2^52 = double(u) with zero rounding error and zero FP exceptions.
inline __m256i top_bit_u52_v3(__m256i u) {
  const __m256i magic = _mm256_set1_epi64x(0x4330000000000000LL);
  const __m256d d =
      _mm256_sub_pd(_mm256_castsi256_pd(_mm256_or_si256(u, magic)), _mm256_castsi256_pd(magic));
  return _mm256_sub_epi64(_mm256_srli_epi64(_mm256_castpd_si256(d), 52), _mm256_set1_epi64x(1023));
}

// Decode 8 packed elements (in the low bits of each 32-bit lane) directly to
// signed mantissas (optionally pre-scaled by 16) and exponents.
template <Gfx942Elem Elem, bool ScaleBy16>
inline uint8_t decode8_v3(__m256i raw, __m256i &mant, __m256i &exp) {
  const __m256i zero = _mm256_setzero_si256();
  const __m256i one = _mm256_set1_epi32(1);
  if constexpr (Elem == Gfx942Elem::F16) {
    const __m256i u = _mm256_and_si256(raw, _mm256_set1_epi32(0xffff));
    const __m256i e = _mm256_and_si256(_mm256_srli_epi32(u, 10), _mm256_set1_epi32(0x1f));
    const __m256i f = _mm256_and_si256(u, _mm256_set1_epi32(0x3ff));
    const __m256i norm = _mm256_cmpgt_epi32(e, zero);
    __m256i mag = _mm256_or_si256(f, _mm256_and_si256(norm, _mm256_set1_epi32(0x400)));
    if constexpr (ScaleBy16) {
      mag = _mm256_slli_epi32(mag, 4);
    }
    mant = _mm256_sign_epi32(mag, _mm256_or_si256(_mm256_slli_epi32(u, 16), one));
    exp = _mm256_sub_epi32(_mm256_max_epi32(e, one), _mm256_set1_epi32(15));
    return uint8_t(
        _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(e, _mm256_set1_epi32(31)))));
  } else if constexpr (Elem == Gfx942Elem::Bf16) {
    const __m256i u = _mm256_and_si256(raw, _mm256_set1_epi32(0xffff));
    const __m256i e = _mm256_and_si256(_mm256_srli_epi32(u, 7), _mm256_set1_epi32(0xff));
    const __m256i f = _mm256_and_si256(u, _mm256_set1_epi32(0x7f));
    const __m256i norm = _mm256_cmpgt_epi32(e, zero);
    __m256i mag = _mm256_or_si256(f, _mm256_and_si256(norm, _mm256_set1_epi32(0x80)));
    mag = _mm256_slli_epi32(mag, ScaleBy16 ? 7 : 3);
    mant = _mm256_sign_epi32(mag, _mm256_or_si256(_mm256_slli_epi32(u, 16), one));
    exp = _mm256_sub_epi32(_mm256_max_epi32(e, one), _mm256_set1_epi32(127));
    return uint8_t(
        _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(e, _mm256_set1_epi32(0xff)))));
  } else if constexpr (Elem == Gfx942Elem::Xf32) {
    const __m256i e = _mm256_and_si256(_mm256_srli_epi32(raw, 23), _mm256_set1_epi32(0xff));
    const __m256i f = _mm256_and_si256(_mm256_srli_epi32(raw, 13), _mm256_set1_epi32(0x3ff));
    const __m256i norm = _mm256_cmpgt_epi32(e, zero);
    __m256i mag = _mm256_or_si256(f, _mm256_and_si256(norm, _mm256_set1_epi32(0x400)));
    if constexpr (ScaleBy16) {
      mag = _mm256_slli_epi32(mag, 4);
    }
    mant = _mm256_sign_epi32(mag, _mm256_or_si256(raw, one));
    exp = _mm256_sub_epi32(_mm256_max_epi32(e, one), _mm256_set1_epi32(127));
    return uint8_t(
        _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(e, _mm256_set1_epi32(0xff)))));
  } else if constexpr (Elem == Gfx942Elem::Fp8Fnuz) {
    const __m256i u = _mm256_and_si256(raw, _mm256_set1_epi32(0xff));
    const __m256i e = _mm256_and_si256(_mm256_srli_epi32(u, 3), _mm256_set1_epi32(0x0f));
    const __m256i f = _mm256_and_si256(u, _mm256_set1_epi32(0x07));
    const __m256i norm = _mm256_cmpgt_epi32(e, zero);
    __m256i mag = _mm256_or_si256(f, _mm256_and_si256(norm, _mm256_set1_epi32(8)));
    mag = _mm256_slli_epi32(mag, ScaleBy16 ? 11 : 7);
    mant = _mm256_sign_epi32(mag, _mm256_or_si256(_mm256_slli_epi32(u, 24), one));
    exp = _mm256_sub_epi32(_mm256_max_epi32(e, one), _mm256_set1_epi32(8));
    return uint8_t(
        _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(u, _mm256_set1_epi32(0x80)))));
  } else {
    const __m256i u = _mm256_and_si256(raw, _mm256_set1_epi32(0xff));
    const __m256i e = _mm256_and_si256(_mm256_srli_epi32(u, 2), _mm256_set1_epi32(0x1f));
    const __m256i f = _mm256_and_si256(u, _mm256_set1_epi32(0x03));
    const __m256i norm = _mm256_cmpgt_epi32(e, zero);
    __m256i mag = _mm256_or_si256(f, _mm256_and_si256(norm, _mm256_set1_epi32(4)));
    mag = _mm256_slli_epi32(mag, ScaleBy16 ? 12 : 8);
    mant = _mm256_sign_epi32(mag, _mm256_or_si256(_mm256_slli_epi32(u, 24), one));
    exp = _mm256_sub_epi32(_mm256_max_epi32(e, one), _mm256_set1_epi32(16));
    return uint8_t(
        _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(u, _mm256_set1_epi32(0x80)))));
  }
}

template <int Products, bool CanOverflow>
inline uint8_t sum_products8_v3(const __m256i *a_m, const __m256i *a_e, const __m256i *b_m,
                                const __m256i *b_e, __m256i &sum_out, __m256i &top_exp_out) {
  const __m256i zero = _mm256_setzero_si256();
  const __m256i zero_exp = _mm256_set1_epi32(kGfx942ZeroProductExponent);
  const __m256i ovf_thresh = _mm256_set1_epi32(127);
  __m256i prod_raw[Products];
  __m256i prod_mag[Products];
  __m256i prod_exp[Products];
  __m256i top_exp = zero_exp;
  __m256i may_ovf_v = zero;

  for (int i = 0; i < Products; ++i) {
    const __m256i prod = _mm256_madd_epi16(a_m[i], b_m[i]);
    const __m256i is_zero = _mm256_cmpeq_epi32(prod, zero);
    const __m256i exp = _mm256_blendv_epi8(_mm256_add_epi32(a_e[i], b_e[i]), zero_exp, is_zero);
    prod_raw[i] = prod;
    prod_mag[i] = _mm256_abs_epi32(prod);
    prod_exp[i] = exp;
    top_exp = (i == 0) ? exp : _mm256_max_epi32(top_exp, exp);
    if constexpr (CanOverflow) {
      const __m256i ovf = _mm256_add_epi32(exp, _mm256_srli_epi32(prod_mag[i], 25));
      may_ovf_v = _mm256_or_si256(may_ovf_v, _mm256_cmpgt_epi32(ovf, ovf_thresh));
    }
  }

  __m256i sum = zero;
  for (int i = 0; i < Products; ++i) {
    const __m256i shift = _mm256_sub_epi32(top_exp, prod_exp[i]);
    const __m256i q_mag = _mm256_srlv_epi32(prod_mag[i], shift);
    const __m256i q = _mm256_sign_epi32(q_mag, prod_raw[i]);
    sum = (i == 0) ? q : _mm256_add_epi32(sum, q);
  }
  sum_out = sum;
  top_exp_out = top_exp;
  if constexpr (CanOverflow) {
    return uint8_t(_mm256_movemask_ps(_mm256_castsi256_ps(may_ovf_v)));
  } else {
    return 0;
  }
}

template <bool DropFar>
inline __m128i add_accumulators4_core_v3(__m128i dot32, __m128i c_mant32, __m128i top_exp32,
                                         __m128i dot_shift32, __m128i c_shift32, __m128i keep_c32) {
  const __m256i zero64 = _mm256_setzero_si256();
  const __m256i one64 = _mm256_set1_epi64x(1);
  const __m256i top_exp = _mm256_cvtepi32_epi64(top_exp32);

  const __m256i dot_part = srav_epi64_v3(_mm256_slli_epi64(_mm256_cvtepi32_epi64(dot32), 8),
                                         _mm256_cvtepu32_epi64(dot_shift32));
  __m256i c_part =
      _mm256_slli_epi64(srav_epi64_v3(_mm256_slli_epi64(_mm256_cvtepi32_epi64(c_mant32), 1),
                                      _mm256_cvtepu32_epi64(c_shift32)),
                        8);
  if constexpr (DropFar) {
    c_part = _mm256_and_si256(c_part, _mm256_cvtepi32_epi64(keep_c32));
  }
  const __m256i total = _mm256_add_epi64(dot_part, c_part);

  // Normalize (rounding down to 31 fraction bits when right > 0) and round to f32 RNE.
  // |shifted| < 2^39 < 2^52, so (2^52 + |shifted|) - 2^52 is exact in IEEE-754 double.
  const __m256i total_mag = abs_epi64_v3(total);
  const __m256i total_top = top_bit_u52_v3(_mm256_or_si256(total_mag, one64));
  const __m256i r1 = _mm256_sub_epi64(total_top, _mm256_set1_epi64x(31));
  const __m256i r2 = _mm256_sub_epi64(_mm256_set1_epi64x(-125), top_exp);
  const __m256i r = max_epi64_v3(max_epi64_v3(r1, r2), zero64);
  const __m256i shifted = srav_epi64_v3(total, r);

  const __m256i sh_sign = _mm256_cmpgt_epi64(zero64, shifted);
  const __m256i sh_mag = _mm256_sub_epi64(_mm256_xor_si256(shifted, sh_sign), sh_sign);
  const __m256i magic = _mm256_set1_epi64x(0x4330000000000000LL);
  const __m256d sh_d_mag = _mm256_sub_pd(_mm256_castsi256_pd(_mm256_or_si256(sh_mag, magic)),
                                         _mm256_castsi256_pd(magic));
  const __m256i exp_shift = _mm256_add_epi64(_mm256_sub_epi64(top_exp, _mm256_set1_epi64x(32)), r);
  const __m256i nz_mask = _mm256_cmpgt_epi64(sh_mag, zero64);
  const __m256i scaled_mag = _mm256_and_si256(
      _mm256_add_epi64(_mm256_castpd_si256(sh_d_mag), _mm256_slli_epi64(exp_shift, 52)), nz_mask);
  const __m256i scaled_bits = _mm256_or_si256(
      scaled_mag, _mm256_and_si256(sh_sign, _mm256_set1_epi64x(int64_t(0x8000000000000000ULL))));
  return _mm_castps_si128(_mm256_cvtpd_ps(_mm256_castsi256_pd(scaled_bits)));
}

template <bool DropFar>
inline __m256i add_accumulators8_v3(__m256i dot, __m256i dot_exp, __m256i c_bits,
                                    uint8_t &c_inf_nan_out) {
  const __m256i zero = _mm256_setzero_si256();
  const __m256i one = _mm256_set1_epi32(1);
  const __m256i c_exp_field =
      _mm256_and_si256(_mm256_srli_epi32(c_bits, 23), _mm256_set1_epi32(0xff));
  c_inf_nan_out = uint8_t(_mm256_movemask_ps(
      _mm256_castsi256_ps(_mm256_cmpeq_epi32(c_exp_field, _mm256_set1_epi32(0xff)))));
  const __m256i c_exp =
      _mm256_sub_epi32(_mm256_max_epi32(c_exp_field, one), _mm256_set1_epi32(127));
  const __m256i c_norm = _mm256_cmpgt_epi32(c_exp_field, zero);
  const __m256i c_mag = _mm256_or_si256(_mm256_and_si256(c_bits, _mm256_set1_epi32(0x7fffff)),
                                        _mm256_and_si256(c_norm, _mm256_set1_epi32(0x800000)));
  const __m256i c_mant = _mm256_sign_epi32(c_mag, _mm256_or_si256(c_bits, one));

  const __m256i top_exp = _mm256_max_epi32(dot_exp, c_exp);
  const __m256i dot_shift = _mm256_sub_epi32(top_exp, dot_exp);
  const __m256i c_shift = _mm256_sub_epi32(top_exp, c_exp);
  __m256i keep_c = _mm256_set1_epi32(-1);
  if constexpr (DropFar) {
    keep_c = _mm256_cmpgt_epi32(_mm256_set1_epi32(26), c_shift);
  }

  const __m128i lo = add_accumulators4_core_v3<DropFar>(
      _mm256_castsi256_si128(dot), _mm256_castsi256_si128(c_mant), _mm256_castsi256_si128(top_exp),
      _mm256_castsi256_si128(dot_shift), _mm256_castsi256_si128(c_shift),
      _mm256_castsi256_si128(keep_c));
  const __m128i hi = add_accumulators4_core_v3<DropFar>(
      _mm256_extracti128_si256(dot, 1), _mm256_extracti128_si256(c_mant, 1),
      _mm256_extracti128_si256(top_exp, 1), _mm256_extracti128_si256(dot_shift, 1),
      _mm256_extracti128_si256(c_shift, 1), _mm256_extracti128_si256(keep_c, 1));
  const __m256i res = _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
  // Flush -0.0 (including negative underflow to zero) to +0.0.
  const __m256i is_zero =
      _mm256_cmpeq_epi32(_mm256_and_si256(res, _mm256_set1_epi32(0x7fffffff)), zero);
  return _mm256_andnot_si256(is_zero, res);
}

// 4x4x4_16b dense specialization: 2 batches (8 lanes) processed per YMM.
template <MfmaGfx942Format Format>
inline void mfma_gfx942_exact_4x4x4_v3(uint32_t *const *d, const uint32_t *const *a,
                                       const uint32_t *const *b, const uint32_t *const *c) {
  constexpr Gfx942Elem elem = gfx942_elem_a<Format>();
  constexpr bool can_ovf = gfx942_can_product_overflow<Format>();
  alignas(32) uint32_t result[4][64];
  const __m256i low16_mask = _mm256_set1_epi32(0xffff);

  for (int b2 = 0; b2 < 8; ++b2) {
    const int base_lane = b2 * 8;
    __m256i a_m[4], a_e[4], b_m[4], b_e[4];
    uint8_t a_inf_any = 0;
    uint8_t b_inf_any = 0;
    for (int reg = 0; reg < 2; ++reg) {
      const __m256i aw = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&a[reg][base_lane]));
      const __m256i bw = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&b[reg][base_lane]));
      a_inf_any |= decode8_v3<elem, true>(aw, a_m[2 * reg], a_e[2 * reg]);
      a_inf_any |=
          decode8_v3<elem, true>(_mm256_srli_epi32(aw, 16), a_m[2 * reg + 1], a_e[2 * reg + 1]);
      b_inf_any |= decode8_v3<elem, false>(bw, b_m[2 * reg], b_e[2 * reg]);
      b_inf_any |=
          decode8_v3<elem, false>(_mm256_srli_epi32(bw, 16), b_m[2 * reg + 1], b_e[2 * reg + 1]);
      a_m[2 * reg] = _mm256_and_si256(a_m[2 * reg], low16_mask);
      a_m[2 * reg + 1] = _mm256_and_si256(a_m[2 * reg + 1], low16_mask);
    }

    auto run_row = [&]<int Row, int ShufImm>() {
      __m256i row_am[4], row_ae[4];
      for (int k = 0; k < 4; ++k) {
        row_am[k] = _mm256_shuffle_epi32(a_m[k], ShufImm);
        row_ae[k] = _mm256_shuffle_epi32(a_e[k], ShufImm);
      }
      __m256i dot, dot_exp;
      const uint8_t may_ovf = sum_products8_v3<4, can_ovf>(row_am, row_ae, b_m, b_e, dot, dot_exp);
      const __m256i c_bits =
          _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&c[Row][base_lane]));
      uint8_t c_inf = 0;
      const __m256i res = add_accumulators8_v3<false>(dot, dot_exp, c_bits, c_inf);
      _mm256_store_si256(reinterpret_cast<__m256i *>(&result[Row][base_lane]), res);

      uint8_t a_row_inf = 0;
      if ((a_inf_any >> Row) & 1u) {
        a_row_inf |= 0x0fu;
      }
      if ((a_inf_any >> (4 + Row)) & 1u) {
        a_row_inf |= 0xf0u;
      }
      uint32_t special = uint32_t(a_row_inf | b_inf_any | c_inf | may_ovf);
      if (__builtin_expect(special != 0u, 0)) {
        while (special != 0u) {
          const int lane = __builtin_ctz(special);
          special &= special - 1u;
          const int batch = b2 * 2 + (lane >> 2);
          const int col = lane & 3;
          gfx942_non_finite_dense_one<Format, 4, 4, 16>(
              batch, Row, col, 0, 4, a, b, c[Row][base_lane + lane], result[Row][base_lane + lane]);
        }
      }
    };
    run_row.template operator()<0, 0x00>();
    run_row.template operator()<1, 0x55>();
    run_row.template operator()<2, 0xaa>();
    run_row.template operator()<3, 0xff>();
  }

  for (int reg = 0; reg < 4; ++reg) {
    for (int lane = 0; lane < 64; lane += 8) {
      _mm256_storeu_si256(reinterpret_cast<__m256i *>(&d[reg][lane]),
                          _mm256_load_si256(reinterpret_cast<const __m256i *>(&result[reg][lane])));
    }
  }
}

template <MfmaGfx942Format Format, int M, int K, int Batch>
inline void mfma_gfx942_exact_dense_v3(uint32_t *const *d, const uint32_t *const *a,
                                       const uint32_t *const *b, const uint32_t *const *c) {
  constexpr Gfx942Elem elem_a = gfx942_elem_a<Format>();
  constexpr Gfx942Elem elem_b = gfx942_elem_b<Format>();
  constexpr int per_word = gfx942_per_word<Format>();
  constexpr int step_k = gfx942_products_per_step<Format>();
  constexpr bool even_odd = gfx942_even_odd<Format>();
  constexpr bool can_ovf = gfx942_can_product_overflow<Format>();
  constexpr int per_lane = K * M * Batch / 64;
  constexpr int output_regs = M * M * Batch / 64;
  constexpr int col_chunks = M / 8; // 2 for M=16, 4 for M=32
  constexpr int num_steps = (K + step_k - 1) / step_k;

  alignas(32) uint32_t result[output_regs][64];

  for (int batch = 0; batch < Batch; ++batch) {
    alignas(32) int32_t a_mant[K][M];
    alignas(32) int32_t a_exp[K][M];
    alignas(32) __m256i b_mant[col_chunks][K];
    alignas(32) __m256i b_exp[col_chunks][K];
    uint32_t a_inf_mask[num_steps]{};
    uint8_t b_inf_mask[num_steps][col_chunks]{};

    for (int depth = 0; depth < K; depth += per_word) {
      const int step_idx = depth / step_k;
      const int lane_base = batch * M + (depth / per_lane) * M * Batch;
      const int reg = (depth % per_lane) / per_word;
      for (int ch = 0; ch < col_chunks; ++ch) {
        const __m256i aw =
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&a[reg][lane_base + ch * 8]));
        const __m256i bw =
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&b[reg][lane_base + ch * 8]));
        for (int sub = 0; sub < per_word; ++sub) {
          const int dk = depth + sub;
          const __m256i a_shifted = (sub == 0) ? aw : _mm256_srli_epi32(aw, sub * (32 / per_word));
          const __m256i b_shifted = (sub == 0) ? bw : _mm256_srli_epi32(bw, sub * (32 / per_word));
          __m256i am, ae, bm, be;
          const uint8_t a_inf = decode8_v3<elem_a, true>(a_shifted, am, ae);
          const uint8_t b_inf = decode8_v3<elem_b, false>(b_shifted, bm, be);
          _mm256_store_si256(reinterpret_cast<__m256i *>(&a_mant[dk][ch * 8]), am);
          _mm256_store_si256(reinterpret_cast<__m256i *>(&a_exp[dk][ch * 8]), ae);
          b_mant[ch][dk] = bm;
          b_exp[ch][dk] = be;
          a_inf_mask[step_idx] |= uint32_t(a_inf) << (ch * 8);
          b_inf_mask[step_idx][ch] |= b_inf;
        }
      }
    }

    for (int row = 0; row < M; ++row) {
      const int out_reg = batch * (M * M / 64) + (row / (256 / M)) * 4 + (row & 3);
      const int out_lane_base = ((row >> 2) % (64 / M)) * M;
      __m256i acc[col_chunks];
      for (int ch = 0; ch < col_chunks; ++ch) {
        acc[ch] = _mm256_loadu_si256(
            reinterpret_cast<const __m256i *>(&c[out_reg][out_lane_base + ch * 8]));
      }

      for (int step_idx = 0; step_idx < num_steps; ++step_idx) {
        const int start = step_idx * step_k;
        constexpr int step_products = (K < step_k) ? K : step_k;
        const bool row_a_inf = ((a_inf_mask[step_idx] >> row) & 1u) != 0u;

        if constexpr (even_odd) {
          constexpr int half_p = step_products / 2;
          __m256i even_am[half_p], even_ae[half_p], odd_am[half_p], odd_ae[half_p];
          for (int i = 0; i < half_p; ++i) {
            even_am[i] = _mm256_set1_epi32(uint16_t(a_mant[start + 2 * i][row]));
            even_ae[i] = _mm256_set1_epi32(a_exp[start + 2 * i][row]);
            odd_am[i] = _mm256_set1_epi32(uint16_t(a_mant[start + 2 * i + 1][row]));
            odd_ae[i] = _mm256_set1_epi32(a_exp[start + 2 * i + 1][row]);
          }
          for (int ch = 0; ch < col_chunks; ++ch) {
            __m256i even_bm[half_p], even_be[half_p], odd_bm[half_p], odd_be[half_p];
            for (int i = 0; i < half_p; ++i) {
              even_bm[i] = b_mant[ch][start + 2 * i];
              even_be[i] = b_exp[ch][start + 2 * i];
              odd_bm[i] = b_mant[ch][start + 2 * i + 1];
              odd_be[i] = b_exp[ch][start + 2 * i + 1];
            }
            __m256i dot, dot_exp, odd, odd_exp;
            sum_products8_v3<half_p, false>(even_am, even_ae, even_bm, even_be, dot, dot_exp);
            sum_products8_v3<half_p, false>(odd_am, odd_ae, odd_bm, odd_be, odd, odd_exp);
            const __m256i top_exp = _mm256_max_epi32(dot_exp, odd_exp);
            dot = _mm256_add_epi32(_mm256_srav_epi32(dot, _mm256_sub_epi32(top_exp, dot_exp)),
                                   _mm256_srav_epi32(odd, _mm256_sub_epi32(top_exp, odd_exp)));
            dot_exp = top_exp;
            uint8_t c_inf = 0;
            const __m256i prev_acc = acc[ch];
            acc[ch] = add_accumulators8_v3<true>(dot, dot_exp, prev_acc, c_inf);
            uint32_t special = row_a_inf ? 0xffu : uint32_t(b_inf_mask[step_idx][ch] | c_inf);
            if (__builtin_expect(special != 0u, 0)) {
              alignas(32) uint32_t prev_buf[8], res_buf[8];
              _mm256_store_si256(reinterpret_cast<__m256i *>(prev_buf), prev_acc);
              _mm256_store_si256(reinterpret_cast<__m256i *>(res_buf), acc[ch]);
              while (special != 0u) {
                const int lane = __builtin_ctz(special);
                special &= special - 1u;
                gfx942_non_finite_dense_one<Format, M, K, Batch>(batch, row, ch * 8 + lane, start,
                                                                 step_products, a, b,
                                                                 prev_buf[lane], res_buf[lane]);
              }
              acc[ch] = _mm256_load_si256(reinterpret_cast<const __m256i *>(res_buf));
            }
          }
        } else {
          __m256i row_am[step_products], row_ae[step_products];
          for (int i = 0; i < step_products; ++i) {
            row_am[i] = _mm256_set1_epi32(uint16_t(a_mant[start + i][row]));
            row_ae[i] = _mm256_set1_epi32(a_exp[start + i][row]);
          }
          for (int ch = 0; ch < col_chunks; ++ch) {
            __m256i dot, dot_exp;
            const uint8_t may_ovf = sum_products8_v3<step_products, can_ovf>(
                row_am, row_ae, &b_mant[ch][start], &b_exp[ch][start], dot, dot_exp);
            uint8_t c_inf = 0;
            const __m256i prev_acc = acc[ch];
            acc[ch] = add_accumulators8_v3<false>(dot, dot_exp, prev_acc, c_inf);
            uint32_t special =
                row_a_inf ? 0xffu : uint32_t(b_inf_mask[step_idx][ch] | c_inf | may_ovf);
            if (__builtin_expect(special != 0u, 0)) {
              alignas(32) uint32_t prev_buf[8], res_buf[8];
              _mm256_store_si256(reinterpret_cast<__m256i *>(prev_buf), prev_acc);
              _mm256_store_si256(reinterpret_cast<__m256i *>(res_buf), acc[ch]);
              while (special != 0u) {
                const int lane = __builtin_ctz(special);
                special &= special - 1u;
                gfx942_non_finite_dense_one<Format, M, K, Batch>(batch, row, ch * 8 + lane, start,
                                                                 step_products, a, b,
                                                                 prev_buf[lane], res_buf[lane]);
              }
              acc[ch] = _mm256_load_si256(reinterpret_cast<const __m256i *>(res_buf));
            }
          }
        }
      }

      for (int ch = 0; ch < col_chunks; ++ch) {
        _mm256_store_si256(reinterpret_cast<__m256i *>(&result[out_reg][out_lane_base + ch * 8]),
                           acc[ch]);
      }
    }
  }

  for (int reg = 0; reg < output_regs; ++reg) {
    for (int lane = 0; lane < 64; lane += 8) {
      _mm256_storeu_si256(reinterpret_cast<__m256i *>(&d[reg][lane]),
                          _mm256_load_si256(reinterpret_cast<const __m256i *>(&result[reg][lane])));
    }
  }
}

template <MfmaGfx942Format Format, int M, int K>
inline void smfmac_gfx942_exact_kernel_v3(uint64_t controls, uint32_t *const *d,
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
  constexpr int output_regs = M * M / 64;
  constexpr int col_chunks = M / 8;
  constexpr int half_k = K / 2;
  constexpr int num_steps = half_k / step_k;

  const int shift = (controls & 3) == 0 ? set_bits * int((controls >> 3) & (32 / set_bits - 1)) : 0;
  alignas(32) uint32_t selected_index[64];
  if (shift == 0) {
    for (int lane = 0; lane < 64; lane += 8) {
      _mm256_store_si256(reinterpret_cast<__m256i *>(&selected_index[lane]),
                         _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&index[0][lane])));
    }
  } else {
    const __m128i sh = _mm_cvtsi32_si128(shift);
    for (int lane = 0; lane < 64; lane += 8) {
      _mm256_store_si256(
          reinterpret_cast<__m256i *>(&selected_index[lane]),
          _mm256_srl_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(&index[0][lane])),
                           sh));
    }
  }

  alignas(32) __m256i b_mant[col_chunks][K];
  alignas(32) __m256i b_exp[col_chunks][K];
  uint8_t b_inf[col_chunks][K];
  for (int depth = 0; depth < K; depth += per_word) {
    const SmfmacDenseLoc dloc = smfmac_dense_loc<layout, M>(depth, 0);
    const int reg = dloc.element / per_word;
    const int lane_base = dloc.lane;
    for (int ch = 0; ch < col_chunks; ++ch) {
      const __m256i bw =
          _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&b[reg][lane_base + ch * 8]));
      for (int sub = 0; sub < per_word; ++sub) {
        const __m256i bs = (sub == 0) ? bw : _mm256_srli_epi32(bw, sub * (32 / per_word));
        b_inf[ch][depth + sub] =
            decode8_v3<elem_b, false>(bs, b_mant[ch][depth + sub], b_exp[ch][depth + sub]);
      }
    }
  }

  alignas(32) int32_t a_mant[half_k][M];
  alignas(32) int32_t a_exp[half_k][M];
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
      const __m256i aw =
          _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&a[reg][lane_base + ch * 8]));
      for (int s = 0; s < 2; ++s) {
        const int sub = base_sub + s;
        const __m256i as = (sub == 0) ? aw : _mm256_srli_epi32(aw, sub * (32 / per_word));
        __m256i am, ae;
        const uint8_t inf = decode8_v3<elem_a, true>(as, am, ae);
        _mm256_store_si256(reinterpret_cast<__m256i *>(&a_mant[t + s][ch * 8]), am);
        _mm256_store_si256(reinterpret_cast<__m256i *>(&a_exp[t + s][ch * 8]), ae);
        a_inf_mask[step_idx] |= uint32_t(inf) << (ch * 8);
      }
    }
  }

  alignas(32) uint32_t result[output_regs][64];
  for (int row = 0; row < M; ++row) {
    const int out_reg = (row / (256 / M)) * 4 + (row & 3);
    const int out_lane_base = ((row >> 2) % (64 / M)) * M;
    __m256i acc[col_chunks];
    for (int ch = 0; ch < col_chunks; ++ch) {
      acc[ch] = _mm256_loadu_si256(
          reinterpret_cast<const __m256i *>(&d[out_reg][out_lane_base + ch * 8]));
    }

    for (int step_idx = 0; step_idx < num_steps; ++step_idx) {
      const int start = step_idx * step_k;
      const bool row_a_inf = ((a_inf_mask[step_idx] >> row) & 1u) != 0u;
      int sel[step_k];
      for (int i = 0; i < step_k; i += 2) {
        const int t = start + i;
        const int q_log = t / 2;
        const int q =
            (M == 16 && K == 64) ? ((q_log & 3) + ((q_log & 4) << 1) + ((q_log & 8) >> 1)) : q_log;
        const SmfmacSparseLoc sloc = smfmac_sparse_loc<layout, M, K>(row, q, 0);
        const uint32_t nib = selected_index[sloc.lane] >> (4 * sloc.nibble);
        sel[i] = 4 * q + int(nib & 3u);
        sel[i + 1] = 4 * q + int((nib >> 2) & 3u);
      }

      if constexpr (even_odd) {
        constexpr int half_p = step_k / 2;
        __m256i even_am[half_p], even_ae[half_p], odd_am[half_p], odd_ae[half_p];
        for (int i = 0; i < half_p; ++i) {
          even_am[i] = _mm256_set1_epi32(uint16_t(a_mant[start + 2 * i][row]));
          even_ae[i] = _mm256_set1_epi32(a_exp[start + 2 * i][row]);
          odd_am[i] = _mm256_set1_epi32(uint16_t(a_mant[start + 2 * i + 1][row]));
          odd_ae[i] = _mm256_set1_epi32(a_exp[start + 2 * i + 1][row]);
        }
        for (int ch = 0; ch < col_chunks; ++ch) {
          __m256i even_bm[half_p], even_be[half_p], odd_bm[half_p], odd_be[half_p];
          uint8_t b_step_inf = 0;
          for (int i = 0; i < half_p; ++i) {
            const int se = sel[2 * i];
            const int so = sel[2 * i + 1];
            even_bm[i] = b_mant[ch][se];
            even_be[i] = b_exp[ch][se];
            odd_bm[i] = b_mant[ch][so];
            odd_be[i] = b_exp[ch][so];
            b_step_inf |= uint8_t(b_inf[ch][se] | b_inf[ch][so]);
          }
          __m256i dot, dot_exp, odd, odd_exp;
          sum_products8_v3<half_p, false>(even_am, even_ae, even_bm, even_be, dot, dot_exp);
          sum_products8_v3<half_p, false>(odd_am, odd_ae, odd_bm, odd_be, odd, odd_exp);
          const __m256i top_exp = _mm256_max_epi32(dot_exp, odd_exp);
          dot = _mm256_add_epi32(_mm256_srav_epi32(dot, _mm256_sub_epi32(top_exp, dot_exp)),
                                 _mm256_srav_epi32(odd, _mm256_sub_epi32(top_exp, odd_exp)));
          dot_exp = top_exp;
          uint8_t c_inf = 0;
          const __m256i prev_acc = acc[ch];
          acc[ch] = add_accumulators8_v3<true>(dot, dot_exp, prev_acc, c_inf);
          uint32_t special = row_a_inf ? 0xffu : uint32_t(b_step_inf | c_inf);
          if (__builtin_expect(special != 0u, 0)) {
            alignas(32) uint32_t prev_buf[8], res_buf[8];
            _mm256_store_si256(reinterpret_cast<__m256i *>(prev_buf), prev_acc);
            _mm256_store_si256(reinterpret_cast<__m256i *>(res_buf), acc[ch]);
            while (special != 0u) {
              const int lane = __builtin_ctz(special);
              special &= special - 1u;
              gfx942_non_finite_sparse_one<Format, M, K>(row, ch * 8 + lane, start, step_k, a, b,
                                                         selected_index, prev_buf[lane],
                                                         res_buf[lane]);
            }
            acc[ch] = _mm256_load_si256(reinterpret_cast<const __m256i *>(res_buf));
          }
        }
      } else {
        __m256i row_am[step_k], row_ae[step_k];
        for (int i = 0; i < step_k; ++i) {
          row_am[i] = _mm256_set1_epi32(uint16_t(a_mant[start + i][row]));
          row_ae[i] = _mm256_set1_epi32(a_exp[start + i][row]);
        }
        for (int ch = 0; ch < col_chunks; ++ch) {
          __m256i col_bm[step_k], col_be[step_k];
          uint8_t b_step_inf = 0;
          for (int i = 0; i < step_k; ++i) {
            const int si = sel[i];
            col_bm[i] = b_mant[ch][si];
            col_be[i] = b_exp[ch][si];
            b_step_inf |= b_inf[ch][si];
          }
          __m256i dot, dot_exp;
          const uint8_t may_ovf =
              sum_products8_v3<step_k, can_ovf>(row_am, row_ae, col_bm, col_be, dot, dot_exp);
          uint8_t c_inf = 0;
          const __m256i prev_acc = acc[ch];
          acc[ch] = add_accumulators8_v3<false>(dot, dot_exp, prev_acc, c_inf);
          uint32_t special = row_a_inf ? 0xffu : uint32_t(b_step_inf | c_inf | may_ovf);
          if (__builtin_expect(special != 0u, 0)) {
            alignas(32) uint32_t prev_buf[8], res_buf[8];
            _mm256_store_si256(reinterpret_cast<__m256i *>(prev_buf), prev_acc);
            _mm256_store_si256(reinterpret_cast<__m256i *>(res_buf), acc[ch]);
            while (special != 0u) {
              const int lane = __builtin_ctz(special);
              special &= special - 1u;
              gfx942_non_finite_sparse_one<Format, M, K>(row, ch * 8 + lane, start, step_k, a, b,
                                                         selected_index, prev_buf[lane],
                                                         res_buf[lane]);
            }
            acc[ch] = _mm256_load_si256(reinterpret_cast<const __m256i *>(res_buf));
          }
        }
      }
    }

    for (int ch = 0; ch < col_chunks; ++ch) {
      _mm256_store_si256(reinterpret_cast<__m256i *>(&result[out_reg][out_lane_base + ch * 8]),
                         acc[ch]);
    }
  }

  for (int reg = 0; reg < output_regs; ++reg) {
    for (int lane = 0; lane < 64; lane += 8) {
      _mm256_storeu_si256(reinterpret_cast<__m256i *>(&d[reg][lane]),
                          _mm256_load_si256(reinterpret_cast<const __m256i *>(&result[reg][lane])));
    }
  }
}

} // namespace

template <MfmaGfx942Format Format, int M, int K, int Batch>
void mfma_gfx942_exact_x86_64_v3(uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c) {
  if constexpr (M == 4 && K == 4 && Batch == 16) {
    mfma_gfx942_exact_4x4x4_v3<Format>(d, a, b, c);
  } else {
    mfma_gfx942_exact_dense_v3<Format, M, K, Batch>(d, a, b, c);
  }
}

template <MfmaGfx942Format Format, int M, int K>
void smfmac_gfx942_exact_x86_64_v3(uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b, const uint32_t *const *index) {
  smfmac_gfx942_exact_kernel_v3<Format, M, K>(controls, d, a, b, index);
}

// Explicit instantiations for all 20 captured gfx942 dense exact MFMA forms.
template void mfma_gfx942_exact_x86_64_v3<MfmaGfx942Format::F16, 32, 8, 1>(uint32_t *const *,
                                                                           const uint32_t *const *,
                                                                           const uint32_t *const *,
                                                                           const uint32_t *const *);
template void mfma_gfx942_exact_x86_64_v3<MfmaGfx942Format::F16, 16, 16, 1>(
    uint32_t *const *, const uint32_t *const *, const uint32_t *const *, const uint32_t *const *);
template void mfma_gfx942_exact_x86_64_v3<MfmaGfx942Format::F16, 32, 4, 2>(uint32_t *const *,
                                                                           const uint32_t *const *,
                                                                           const uint32_t *const *,
                                                                           const uint32_t *const *);
template void mfma_gfx942_exact_x86_64_v3<MfmaGfx942Format::F16, 16, 4, 4>(uint32_t *const *,
                                                                           const uint32_t *const *,
                                                                           const uint32_t *const *,
                                                                           const uint32_t *const *);
template void mfma_gfx942_exact_x86_64_v3<MfmaGfx942Format::F16, 4, 4, 16>(uint32_t *const *,
                                                                           const uint32_t *const *,
                                                                           const uint32_t *const *,
                                                                           const uint32_t *const *);

template void mfma_gfx942_exact_x86_64_v3<MfmaGfx942Format::Bf16, 32, 8, 1>(
    uint32_t *const *, const uint32_t *const *, const uint32_t *const *, const uint32_t *const *);
template void mfma_gfx942_exact_x86_64_v3<MfmaGfx942Format::Bf16, 16, 16, 1>(
    uint32_t *const *, const uint32_t *const *, const uint32_t *const *, const uint32_t *const *);
template void mfma_gfx942_exact_x86_64_v3<MfmaGfx942Format::Bf16, 32, 4, 2>(
    uint32_t *const *, const uint32_t *const *, const uint32_t *const *, const uint32_t *const *);
template void mfma_gfx942_exact_x86_64_v3<MfmaGfx942Format::Bf16, 16, 4, 4>(
    uint32_t *const *, const uint32_t *const *, const uint32_t *const *, const uint32_t *const *);
template void mfma_gfx942_exact_x86_64_v3<MfmaGfx942Format::Bf16, 4, 4, 16>(
    uint32_t *const *, const uint32_t *const *, const uint32_t *const *, const uint32_t *const *);

template void mfma_gfx942_exact_x86_64_v3<MfmaGfx942Format::Xf32, 32, 4, 1>(
    uint32_t *const *, const uint32_t *const *, const uint32_t *const *, const uint32_t *const *);
template void mfma_gfx942_exact_x86_64_v3<MfmaGfx942Format::Xf32, 16, 8, 1>(
    uint32_t *const *, const uint32_t *const *, const uint32_t *const *, const uint32_t *const *);

template void mfma_gfx942_exact_x86_64_v3<MfmaGfx942Format::Fp8Fp8, 16, 32, 1>(
    uint32_t *const *, const uint32_t *const *, const uint32_t *const *, const uint32_t *const *);
template void mfma_gfx942_exact_x86_64_v3<MfmaGfx942Format::Fp8Bf8, 16, 32, 1>(
    uint32_t *const *, const uint32_t *const *, const uint32_t *const *, const uint32_t *const *);
template void mfma_gfx942_exact_x86_64_v3<MfmaGfx942Format::Bf8Fp8, 16, 32, 1>(
    uint32_t *const *, const uint32_t *const *, const uint32_t *const *, const uint32_t *const *);
template void mfma_gfx942_exact_x86_64_v3<MfmaGfx942Format::Bf8Bf8, 16, 32, 1>(
    uint32_t *const *, const uint32_t *const *, const uint32_t *const *, const uint32_t *const *);

template void mfma_gfx942_exact_x86_64_v3<MfmaGfx942Format::Fp8Fp8, 32, 16, 1>(
    uint32_t *const *, const uint32_t *const *, const uint32_t *const *, const uint32_t *const *);
template void mfma_gfx942_exact_x86_64_v3<MfmaGfx942Format::Fp8Bf8, 32, 16, 1>(
    uint32_t *const *, const uint32_t *const *, const uint32_t *const *, const uint32_t *const *);
template void mfma_gfx942_exact_x86_64_v3<MfmaGfx942Format::Bf8Fp8, 32, 16, 1>(
    uint32_t *const *, const uint32_t *const *, const uint32_t *const *, const uint32_t *const *);
template void mfma_gfx942_exact_x86_64_v3<MfmaGfx942Format::Bf8Bf8, 32, 16, 1>(
    uint32_t *const *, const uint32_t *const *, const uint32_t *const *, const uint32_t *const *);

// Explicit instantiations for all 12 captured gfx942 sparse exact SMFMAC forms.
template void smfmac_gfx942_exact_x86_64_v3<MfmaGfx942Format::F16, 16, 32>(uint64_t,
                                                                           uint32_t *const *,
                                                                           const uint32_t *const *,
                                                                           const uint32_t *const *,
                                                                           const uint32_t *const *);
template void smfmac_gfx942_exact_x86_64_v3<MfmaGfx942Format::Bf16, 16, 32>(
    uint64_t, uint32_t *const *, const uint32_t *const *, const uint32_t *const *,
    const uint32_t *const *);
template void smfmac_gfx942_exact_x86_64_v3<MfmaGfx942Format::F16, 32, 16>(uint64_t,
                                                                           uint32_t *const *,
                                                                           const uint32_t *const *,
                                                                           const uint32_t *const *,
                                                                           const uint32_t *const *);
template void smfmac_gfx942_exact_x86_64_v3<MfmaGfx942Format::Bf16, 32, 16>(
    uint64_t, uint32_t *const *, const uint32_t *const *, const uint32_t *const *,
    const uint32_t *const *);
template void smfmac_gfx942_exact_x86_64_v3<MfmaGfx942Format::Fp8Fp8, 16, 64>(
    uint64_t, uint32_t *const *, const uint32_t *const *, const uint32_t *const *,
    const uint32_t *const *);
template void smfmac_gfx942_exact_x86_64_v3<MfmaGfx942Format::Fp8Bf8, 16, 64>(
    uint64_t, uint32_t *const *, const uint32_t *const *, const uint32_t *const *,
    const uint32_t *const *);
template void smfmac_gfx942_exact_x86_64_v3<MfmaGfx942Format::Bf8Fp8, 16, 64>(
    uint64_t, uint32_t *const *, const uint32_t *const *, const uint32_t *const *,
    const uint32_t *const *);
template void smfmac_gfx942_exact_x86_64_v3<MfmaGfx942Format::Bf8Bf8, 16, 64>(
    uint64_t, uint32_t *const *, const uint32_t *const *, const uint32_t *const *,
    const uint32_t *const *);
template void smfmac_gfx942_exact_x86_64_v3<MfmaGfx942Format::Fp8Fp8, 32, 32>(
    uint64_t, uint32_t *const *, const uint32_t *const *, const uint32_t *const *,
    const uint32_t *const *);
template void smfmac_gfx942_exact_x86_64_v3<MfmaGfx942Format::Fp8Bf8, 32, 32>(
    uint64_t, uint32_t *const *, const uint32_t *const *, const uint32_t *const *,
    const uint32_t *const *);
template void smfmac_gfx942_exact_x86_64_v3<MfmaGfx942Format::Bf8Fp8, 32, 32>(
    uint64_t, uint32_t *const *, const uint32_t *const *, const uint32_t *const *,
    const uint32_t *const *);
template void smfmac_gfx942_exact_x86_64_v3<MfmaGfx942Format::Bf8Bf8, 32, 32>(
    uint64_t, uint32_t *const *, const uint32_t *const *, const uint32_t *const *,
    const uint32_t *const *);

} // namespace rocjitsu::arith
