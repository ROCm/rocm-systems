// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// gfx950 mixed-format MFMA staging and reduction, isolated by CPU ISA.
#pragma once

#include "mfma/layout.h"
#include "mfma/mixed_formats.h"
#include "rj_arith/detail/rj_arith_mfma.h"
#if defined(RJ_ARITH_TARGET_X86_64_V3)
#include "mfma/x86_64/mixed_decode.h"
#include "mfma/x86_64/reduction.h"
#endif

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Decode an operand in the inherited byte-chunked or contiguous-subbyte lane layout.
template <int M, int K>
void mfma_decode_mixed(const uint32_t *const *input, int format, float output[M][K]) {
  constexpr int lanes_per_row = 64 / M, per_lane = K / lanes_per_row;
  for (int row = 0; row < M; ++row) {
    for (int k = 0; k < K; ++k) {
      int lane, element;
      if (matrix_format_bits(format) == 8 && per_lane > 16) {
        lane = (k / 16 % lanes_per_row) * M + row;
        element = (k / (16 * lanes_per_row)) * 16 + k % 16;
      } else {
        lane = (k / per_lane) * M + row;
        element = k % per_lane;
      }
      output[row][k] = matrix_read_packed(input, lane, element, format);
    }
  }
}

#if defined(RJ_ARITH_TARGET_X86_64_V3)
// Extract the selected E8M0 exponent bytes for all 64 lanes and report whether
// every pair (ea[i], eb[j]) maps to a normal exact FP32 power-of-2 multiplier.
inline bool mfma_extract_e8m0_scales(uint64_t controls, const uint32_t *scale_a,
                                     const uint32_t *scale_b, uint32_t (&ea)[64],
                                     uint32_t (&eb)[64], int extra_shift = 0) {
  const int a_shift = 8 * int((controls >> RJ_ARITH_MFMA_SCALE_A_BYTE_SHIFT) & 3);
  const int b_shift = 8 * int((controls >> RJ_ARITH_MFMA_SCALE_B_BYTE_SHIFT) & 3);
  uint32_t ea_min = 255, ea_max = 0, eb_min = 255, eb_max = 0;
  for (int i = 0; i < 64; ++i) {
    const uint32_t a = (scale_a[i] >> a_shift) & 255u;
    const uint32_t b = (scale_b[i] >> b_shift) & 255u;
    ea[i] = a;
    eb[i] = b;
    if (a < ea_min) {
      ea_min = a;
    }
    if (a > ea_max) {
      ea_max = a;
    }
    if (b < eb_min) {
      eb_min = b;
    }
    if (b > eb_max) {
      eb_max = b;
    }
  }
  // Single-multiply FP32 factor 2^(ea+eb-254+extra_shift) has biased exponent
  // ea + eb - 127 + extra_shift, which is normal and exact in [1, 254].
  const int lo = int(ea_min) + int(eb_min) - 127 + extra_shift;
  const int hi = int(ea_max) + int(eb_max) - 127 + extra_shift;
  return ea_max < 255 && eb_max < 255 && lo >= 1 && hi <= 254;
}

// Cold vectorized FP64 power-of-2 scaling: exact for every E8M0 pair (0..255, 0..255).
// Kept out-of-line so the common normal-scale fast path suffers zero register spills.
#if defined(RJ_ARITH_TARGET_X86_64_V4)
[[gnu::noinline]] inline __m512
mfma_scale_nonplain_vec(__m512 sum, uint32_t a_exp, const uint32_t *eb_ptr, int extra_shift = 0) {
  const __m512i eb = _mm512_load_si512(eb_ptr);
  const __mmask16 nan_mask =
      a_exp == 255u ? __mmask16(0xffffu) : _mm512_cmpeq_epi32_mask(eb, _mm512_set1_epi32(255));
  const __m512i base = _mm512_set1_epi64(int64_t(a_exp) + 769 + extra_shift);
  const __m512i e_lo = _mm512_add_epi64(_mm512_cvtepu32_epi64(_mm512_castsi512_si256(eb)), base);
  const __m512i e_hi =
      _mm512_add_epi64(_mm512_cvtepu32_epi64(_mm512_extracti32x8_epi32(eb, 1)), base);
  const __m512d d_lo = _mm512_cvtps_pd(_mm512_castps512_ps256(sum));
  const __m512d d_hi = _mm512_cvtps_pd(_mm512_extractf32x8_ps(sum, 1));
  const __m256 s_lo =
      _mm512_cvtpd_ps(_mm512_mul_pd(d_lo, _mm512_castsi512_pd(_mm512_slli_epi64(e_lo, 52))));
  const __m256 s_hi =
      _mm512_cvtpd_ps(_mm512_mul_pd(d_hi, _mm512_castsi512_pd(_mm512_slli_epi64(e_hi, 52))));
  const __m512 scaled = _mm512_insertf32x8(_mm512_castps256_ps512(s_lo), s_hi, 1);
  return _mm512_mask_blend_ps(nan_mask, scaled, _mm512_castsi512_ps(_mm512_set1_epi32(0x7fc00000)));
}
#else
[[gnu::noinline]] inline __m256
mfma_scale_nonplain_vec(__m256 sum, uint32_t a_exp, const uint32_t *eb_ptr, int extra_shift = 0) {
  const __m256i eb = _mm256_load_si256(reinterpret_cast<const __m256i *>(eb_ptr));
  const __m256i nan_mask =
      a_exp == 255u ? _mm256_set1_epi32(-1) : _mm256_cmpeq_epi32(eb, _mm256_set1_epi32(255));
  const __m256i base = _mm256_set1_epi64x(int64_t(a_exp) + 769 + extra_shift);
  const __m256i e_lo = _mm256_add_epi64(_mm256_cvtepu32_epi64(_mm256_castsi256_si128(eb)), base);
  const __m256i e_hi =
      _mm256_add_epi64(_mm256_cvtepu32_epi64(_mm256_extracti128_si256(eb, 1)), base);
  const __m256d d_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(sum));
  const __m256d d_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(sum, 1));
  const __m128 s_lo =
      _mm256_cvtpd_ps(_mm256_mul_pd(d_lo, _mm256_castsi256_pd(_mm256_slli_epi64(e_lo, 52))));
  const __m128 s_hi =
      _mm256_cvtpd_ps(_mm256_mul_pd(d_hi, _mm256_castsi256_pd(_mm256_slli_epi64(e_hi, 52))));
  const __m256 scaled = _mm256_insertf128_ps(_mm256_castps128_ps256(s_lo), s_hi, 1);
  return _mm256_blendv_ps(scaled, _mm256_castsi256_ps(_mm256_set1_epi32(0x7fc00000)),
                          _mm256_castsi256_ps(nan_mask));
}
#endif

// Load one row slice of C with optional ABS/NEG modifiers.
inline MfmaNative mfma_mixed_load_c(const uint32_t *src, uint32_t abs_mask, uint32_t neg_mask) {
#if defined(RJ_ARITH_TARGET_X86_64_V4)
  __m512i bits = _mm512_loadu_si512(src);
  bits = _mm512_xor_si512(_mm512_and_si512(bits, _mm512_set1_epi32(int(abs_mask))),
                          _mm512_set1_epi32(int(neg_mask)));
  return _mm512_castsi512_ps(bits);
#else
  __m256i bits = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(src));
  bits = _mm256_xor_si256(_mm256_and_si256(bits, _mm256_set1_epi32(int(abs_mask))),
                          _mm256_set1_epi32(int(neg_mask)));
  return _mm256_castsi256_ps(bits);
#endif
}

// Multi-row register-blocked mixed/scaled SIMD reduction over K-major decoded matrices.
template <int M, int K, bool Scaled>
void mfma_mixed_reduce_simd(uint64_t controls, uint32_t *const *d, const float (&left)[K][M],
                            const float (&right)[K][M], const uint32_t *const *c,
                            const uint32_t *scale_a, const uint32_t *scale_b, bool clamp_overflow) {
  constexpr int registers = M * M / 64;
  const uint32_t abs_mask =
      (Scaled && (controls & RJ_ARITH_MFMA_SCALE_ABS_C)) ? 0x7fffffffU : 0xffffffffU;
  const uint32_t neg_mask = (Scaled && (controls & RJ_ARITH_MFMA_SCALE_NEG_C)) ? 0x80000000U : 0U;
  alignas(64) uint32_t ea[64], eb[64];
  bool plain_scales = false;
  if constexpr (Scaled) {
    plain_scales = mfma_extract_e8m0_scales(controls, scale_a, scale_b, ea, eb);
  }
  const bool direct = mfma_direct_store_safe<registers>(d, c);
  alignas(64) float result[registers][64];

#if defined(RJ_ARITH_TARGET_X86_64_V4)
  constexpr int col_vecs = M / 16;
  if constexpr (!Scaled) {
    // Unscaled: 16 rows for M=16 (16 ZMMs), 8 rows x 2 cols for M=32 (16 ZMMs).
    constexpr int row_tile = M == 16 ? 16 : 8;
    for (int rb = 0; rb < M; rb += row_tile) {
      __m512 acc[row_tile][col_vecs];
      RJ_ARITH_UNROLL(16)
      for (int r = 0; r < row_tile; ++r) {
        const MfmaWord out = mfma_output<M, 1>(rb + r, 0, 0);
        for (int p = 0; p < col_vecs; ++p) {
          acc[r][p] =
              _mm512_loadu_ps(reinterpret_cast<const float *>(c[out.reg] + out.lane + 16 * p));
        }
      }
      for (int k = 0; k < K; ++k) {
        __m512 b_vec[col_vecs];
        for (int p = 0; p < col_vecs; ++p) {
          b_vec[p] = _mm512_load_ps(right[k] + 16 * p);
        }
        RJ_ARITH_UNROLL(16)
        for (int r = 0; r < row_tile; ++r) {
          const __m512 a_bcast = _mm512_set1_ps(left[k][rb + r]);
          for (int p = 0; p < col_vecs; ++p) {
            acc[r][p] = _mm512_fmadd_ps(a_bcast, b_vec[p], acc[r][p]);
          }
        }
      }
      RJ_ARITH_UNROLL(16)
      for (int r = 0; r < row_tile; ++r) {
        const MfmaWord out = mfma_output<M, 1>(rb + r, 0, 0);
        for (int p = 0; p < col_vecs; ++p) {
          if (direct) {
            _mm512_storeu_ps(reinterpret_cast<float *>(d[out.reg] + out.lane + 16 * p), acc[r][p]);
          } else {
            _mm512_store_ps(result[out.reg] + out.lane + 16 * p, acc[r][p]);
          }
        }
      }
    }
  } else {
    // Scaled: keep acc and block sum in 16 ZMMs (8x1 for M=16, 4x2 for M=32).
    constexpr int row_tile = M == 16 ? 8 : 4;
    const __m512i f32_abs = _mm512_set1_epi32(0x7fffffff);
    const __m512i f32_inf = _mm512_set1_epi32(0x7f800000);
    for (int rb = 0; rb < M; rb += row_tile) {
      __m512 acc[row_tile][col_vecs];
      __mmask16 clamp_mask[row_tile][col_vecs];
      for (int r = 0; r < row_tile; ++r) {
        const MfmaWord out = mfma_output<M, 1>(rb + r, 0, 0);
        for (int p = 0; p < col_vecs; ++p) {
          acc[r][p] = mfma_mixed_load_c(c[out.reg] + out.lane + 16 * p, abs_mask, neg_mask);
          if (clamp_overflow) {
            clamp_mask[r][p] = _mm512_cmplt_epu32_mask(
                _mm512_and_si512(_mm512_castps_si512(acc[r][p]), f32_abs), f32_inf);
          } else {
            clamp_mask[r][p] = 0;
          }
        }
      }
      for (int block = 0; block < K / 32; ++block) {
        __m512 sum[row_tile][col_vecs];
        for (int r = 0; r < row_tile; ++r) {
          for (int p = 0; p < col_vecs; ++p) {
            sum[r][p] = _mm512_setzero_ps();
          }
        }
        RJ_ARITH_UNROLL(4)
        for (int k = 32 * block; k < 32 * (block + 1); ++k) {
          __m512 b_vec[col_vecs];
          for (int p = 0; p < col_vecs; ++p) {
            b_vec[p] = _mm512_load_ps(right[k] + 16 * p);
          }
          RJ_ARITH_UNROLL(8)
          for (int r = 0; r < row_tile; ++r) {
            const __m512 a_bcast = _mm512_set1_ps(left[k][rb + r]);
            for (int p = 0; p < col_vecs; ++p) {
              sum[r][p] = _mm512_fmadd_ps(a_bcast, b_vec[p], sum[r][p]);
            }
          }
        }
        for (int r = 0; r < row_tile; ++r) {
          const uint32_t a_exp = ea[M * block + rb + r];
          for (int p = 0; p < col_vecs; ++p) {
            if (clamp_overflow) {
              clamp_mask[r][p] &= _mm512_cmplt_epu32_mask(
                  _mm512_and_si512(_mm512_castps_si512(sum[r][p]), f32_abs), f32_inf);
            }
            __m512 scaled;
            if (__builtin_expect(plain_scales, 1)) {
              const __m512i eb_vec = _mm512_load_si512(eb + M * block + 16 * p);
              const __m512i exp = _mm512_add_epi32(eb_vec, _mm512_set1_epi32(int(a_exp) - 127));
              const __m512 factor = _mm512_castsi512_ps(_mm512_slli_epi32(exp, 23));
              scaled = _mm512_mul_ps(sum[r][p], factor);
            } else {
              scaled = mfma_scale_nonplain_vec(sum[r][p], a_exp, eb + M * block + 16 * p);
            }
            acc[r][p] = _mm512_add_ps(acc[r][p], scaled);
          }
        }
      }
      for (int r = 0; r < row_tile; ++r) {
        const MfmaWord out = mfma_output<M, 1>(rb + r, 0, 0);
        for (int p = 0; p < col_vecs; ++p) {
          if (clamp_overflow) {
            const __mmask16 inf_m = _mm512_cmpeq_epi32_mask(
                _mm512_and_si512(_mm512_castps_si512(acc[r][p]), f32_abs), f32_inf);
            const int sat_m = clamp_mask[r][p] & inf_m;
            acc[r][p] = _mm512_castsi512_ps(
                _mm512_mask_sub_epi32(_mm512_castps_si512(acc[r][p]), sat_m,
                                      _mm512_castps_si512(acc[r][p]), _mm512_set1_epi32(1)));
          }
          if (direct) {
            _mm512_storeu_ps(reinterpret_cast<float *>(d[out.reg] + out.lane + 16 * p), acc[r][p]);
          } else {
            _mm512_store_ps(result[out.reg] + out.lane + 16 * p, acc[r][p]);
          }
        }
      }
    }
  }
  if (!direct) {
    for (int reg = 0; reg < registers; ++reg) {
      for (int lane = 0; lane < 64; lane += 16) {
        _mm512_storeu_ps(reinterpret_cast<float *>(d[reg] + lane),
                         _mm512_load_ps(result[reg] + lane));
      }
    }
  }
#else
  // AVX2: 4 rows x 16 cols (8 YMM accumulators) -> zero register spills.
  const __m256i f32_abs = _mm256_set1_epi32(0x7fffffff);
  const __m256i f32_inf = _mm256_set1_epi32(0x7f800000);
  for (int col_base = 0; col_base < M; col_base += 16) {
    for (int rb = 0; rb < M; rb += 4) {
      alignas(32) float tile_acc[4][16];
      __m256i tile_clamp[4][2];
      for (int r = 0; r < 4; ++r) {
        const MfmaWord out = mfma_output<M, 1>(rb + r, col_base, 0);
        const MfmaNative c0 = mfma_mixed_load_c(c[out.reg] + out.lane, abs_mask, neg_mask);
        const MfmaNative c1 = mfma_mixed_load_c(c[out.reg] + out.lane + 8, abs_mask, neg_mask);
        _mm256_store_ps(tile_acc[r] + 0, c0);
        _mm256_store_ps(tile_acc[r] + 8, c1);
        if constexpr (Scaled) {
          if (clamp_overflow) {
            tile_clamp[r][0] =
                _mm256_cmpgt_epi32(f32_inf, _mm256_and_si256(_mm256_castps_si256(c0), f32_abs));
            tile_clamp[r][1] =
                _mm256_cmpgt_epi32(f32_inf, _mm256_and_si256(_mm256_castps_si256(c1), f32_abs));
          } else {
            tile_clamp[r][0] = _mm256_setzero_si256();
            tile_clamp[r][1] = _mm256_setzero_si256();
          }
        }
      }
      if constexpr (!Scaled) {
        __m256 acc[4][2];
        for (int r = 0; r < 4; ++r) {
          acc[r][0] = _mm256_load_ps(tile_acc[r] + 0);
          acc[r][1] = _mm256_load_ps(tile_acc[r] + 8);
        }
        for (int k = 0; k < K; ++k) {
          const __m256 b0 = _mm256_load_ps(right[k] + col_base);
          const __m256 b1 = _mm256_load_ps(right[k] + col_base + 8);
          RJ_ARITH_UNROLL(4)
          for (int r = 0; r < 4; ++r) {
            const __m256 x = _mm256_broadcast_ss(&left[k][rb + r]);
            acc[r][0] = _mm256_fmadd_ps(x, b0, acc[r][0]);
            acc[r][1] = _mm256_fmadd_ps(x, b1, acc[r][1]);
          }
        }
        for (int r = 0; r < 4; ++r) {
          _mm256_store_ps(tile_acc[r] + 0, acc[r][0]);
          _mm256_store_ps(tile_acc[r] + 8, acc[r][1]);
        }
      } else {
        for (int block = 0; block < K / 32; ++block) {
          __m256 sum[4][2];
          for (int r = 0; r < 4; ++r) {
            sum[r][0] = _mm256_setzero_ps();
            sum[r][1] = _mm256_setzero_ps();
          }
          RJ_ARITH_UNROLL(4)
          for (int k = 32 * block; k < 32 * (block + 1); ++k) {
            const __m256 b0 = _mm256_load_ps(right[k] + col_base);
            const __m256 b1 = _mm256_load_ps(right[k] + col_base + 8);
            RJ_ARITH_UNROLL(4)
            for (int r = 0; r < 4; ++r) {
              const __m256 x = _mm256_broadcast_ss(&left[k][rb + r]);
              sum[r][0] = _mm256_fmadd_ps(x, b0, sum[r][0]);
              sum[r][1] = _mm256_fmadd_ps(x, b1, sum[r][1]);
            }
          }
          for (int r = 0; r < 4; ++r) {
            const uint32_t a_exp = ea[M * block + rb + r];
            for (int h = 0; h < 2; ++h) {
              if (clamp_overflow) {
                const __m256i fin = _mm256_cmpgt_epi32(
                    f32_inf, _mm256_and_si256(_mm256_castps_si256(sum[r][h]), f32_abs));
                tile_clamp[r][h] = _mm256_and_si256(tile_clamp[r][h], fin);
              }
              __m256 scaled;
              if (__builtin_expect(plain_scales, 1)) {
                const __m256i eb_vec = _mm256_load_si256(
                    reinterpret_cast<const __m256i *>(eb + M * block + col_base + 8 * h));
                const __m256i exp = _mm256_add_epi32(eb_vec, _mm256_set1_epi32(int(a_exp) - 127));
                const __m256 factor = _mm256_castsi256_ps(_mm256_slli_epi32(exp, 23));
                scaled = _mm256_mul_ps(sum[r][h], factor);
              } else {
                scaled =
                    mfma_scale_nonplain_vec(sum[r][h], a_exp, eb + M * block + col_base + 8 * h);
              }
              const __m256 cur = _mm256_load_ps(tile_acc[r] + 8 * h);
              _mm256_store_ps(tile_acc[r] + 8 * h, _mm256_add_ps(cur, scaled));
            }
          }
        }
        if (clamp_overflow) {
          for (int r = 0; r < 4; ++r) {
            for (int h = 0; h < 2; ++h) {
              const __m256i bits = _mm256_castps_si256(_mm256_load_ps(tile_acc[r] + 8 * h));
              const __m256i inf_m = _mm256_cmpeq_epi32(_mm256_and_si256(bits, f32_abs), f32_inf);
              const __m256i sat = _mm256_and_si256(inf_m, tile_clamp[r][h]);
              // Adding -1 (all-1s mask) decrements saturated infinities to FLT_MAX of same sign.
              _mm256_store_ps(tile_acc[r] + 8 * h,
                              _mm256_castsi256_ps(_mm256_add_epi32(bits, sat)));
            }
          }
        }
      }
      for (int r = 0; r < 4; ++r) {
        const MfmaWord out = mfma_output<M, 1>(rb + r, col_base, 0);
        if (direct) {
          _mm256_storeu_ps(reinterpret_cast<float *>(d[out.reg] + out.lane),
                           _mm256_load_ps(tile_acc[r] + 0));
          _mm256_storeu_ps(reinterpret_cast<float *>(d[out.reg] + out.lane + 8),
                           _mm256_load_ps(tile_acc[r] + 8));
        } else {
          _mm256_store_ps(result[out.reg] + out.lane, _mm256_load_ps(tile_acc[r] + 0));
          _mm256_store_ps(result[out.reg] + out.lane + 8, _mm256_load_ps(tile_acc[r] + 8));
        }
      }
    }
  }
  if (!direct) {
    for (int reg = 0; reg < registers; ++reg) {
      std::memcpy(d[reg], result[reg], sizeof(result[reg]));
    }
  }
#endif
}
#endif

// Stage every output before stores; preserve inherited reductions and optional block scaling.
template <int M, int K, bool Scaled = false>
void mfma_mixed_kernel(uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
                       const uint32_t *const *b, const uint32_t *const *c,
                       const uint32_t *scale_a = nullptr, const uint32_t *scale_b = nullptr,
                       bool clamp_overflow = false) {
#if defined(RJ_ARITH_TARGET_X86_64_V3)
  alignas(64) float left[K][M], right[K][M];
  mixed_x86::decode<64 / M, M>(a, int(controls & 7), false, left);
  mixed_x86::decode<64 / M, M>(b, int(controls >> 3 & 7), false, right);
  mfma_mixed_reduce_simd<M, K, Scaled>(controls, d, left, right, c, scale_a, scale_b,
                                       clamp_overflow);
#else
  alignas(64) float result[M][M];
  alignas(64) float left[M][K], right_rows[M][K], right[K][M];
  mfma_decode_mixed<M, K>(a, int(controls & 7), left);
  mfma_decode_mixed<M, K>(b, int(controls >> 3 & 7), right_rows);
  for (int k = 0; k < K; ++k) {
    for (int col = 0; col < M; ++col) {
      right[k][col] = right_rows[col][k];
    }
  }
  for (int row = 0; row < M; ++row) {
    bool clamp[M];
    for (int col = 0; col < M; ++col) {
      auto out = mfma_output<M, 1>(row, col, 0);
      uint32_t initial = c[out.reg][out.lane];
      if constexpr (Scaled) {
        if (controls & RJ_ARITH_MFMA_SCALE_ABS_C) {
          initial &= 0x7fffffffU;
        }
        if (controls & RJ_ARITH_MFMA_SCALE_NEG_C) {
          initial ^= 0x80000000U;
        }
        clamp[col] = clamp_overflow && __builtin_isfinite(as_float(initial));
      }
      result[row][col] = as_float(initial);
    }
    if constexpr (Scaled) {
      // Scale each fused 32-product partial once, then add it separately to C.
      for (int block = 0; block < K / 32; ++block) {
        alignas(64) float sums[M] = {};
        for (int col = 0; col < M; ++col) {
          for (int k = 32 * block; k < 32 * (block + 1); ++k) {
            sums[col] = __builtin_fmaf(left[row][k], right[k][col], sums[col]);
          }
        }
        int a_exp = int(scale_a[M * block + row] >> (8 * ((controls >> 6) & 3)) & 255);
        for (int col = 0; col < M; ++col) {
          int b_exp = int(scale_b[M * block + col] >> (8 * ((controls >> 8) & 3)) & 255);
          float scaled = a_exp == 255 || b_exp == 255
                             ? as_float(0x7fc00000U)
                             : __builtin_ldexpf(sums[col], a_exp + b_exp - 254);
          // A finite low-precision dot fits in FP32 before scaling, so a
          // nonfinite block identifies a nonfinite input to this output.
          clamp[col] = clamp[col] && __builtin_isfinite(sums[col]);
          result[row][col] += scaled;
        }
      }
      for (int col = 0; col < M; ++col) {
        if (clamp[col] && __builtin_isinf(result[row][col])) {
          result[row][col] = as_float(as_bits(result[row][col]) - 1u);
        }
      }
    } else {
      for (int col = 0; col < M; ++col) {
        for (int k = 0; k < K; ++k) {
          result[row][col] += left[row][k] * right[k][col];
        }
      }
    }
  }
  for (int row = 0; row < M; ++row) {
    for (int col = 0; col < M; ++col) {
      auto out = mfma_output<M, 1>(row, col, 0);
      d[out.reg][out.lane] = as_bits(result[row][col]);
    }
  }
#endif
}
} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
