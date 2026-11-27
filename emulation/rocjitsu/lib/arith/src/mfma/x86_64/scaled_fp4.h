// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/compiler.h"
#include "mfma/layout.h"
#include "mfma/mixed.h"
#include "rj_arith/detail/rj_arith_mfma.h"
#include <cstring>
#include <immintrin.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

struct alignas(32) MfmaFp4Pairs {
  uint16_t lane[16];
};

// Widen 16 packed FP4 lanes into K-pair order. The j-th vector
// holds one 16-bit (element 2j, element 2j+1) byte pair per lane, in lane order.
inline void mfma_fp4_transpose(const uint32_t *const *input, int lane, __m256i table,
                               MfmaFp4Pairs out[16]) {
  const __m128i gather128 = _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
  const __m256i gather = _mm256_broadcastsi128_si256(gather128);
  const __m256i nibble = _mm256_set1_epi8(15);
  for (int reg = 0; reg < 4; ++reg) {
    const uint32_t *p = input[reg] + lane;
    // Lanes 0-3|8-11 and 4-7|12-15, so the in-lane unpacks below end in lane order.
    __m256i x0 = _mm256_inserti128_si256(
        _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i *>(p))),
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(p + 8)), 1);
    __m256i x1 = _mm256_inserti128_si256(
        _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i *>(p + 4))),
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(p + 12)), 1);
    // Byte q of each lane word moves to dword q, then dwords interleave to 8 lanes per qword.
    x0 = _mm256_shuffle_epi8(x0, gather);
    x1 = _mm256_shuffle_epi8(x1, gather);
    const __m256i z[2] = {_mm256_unpacklo_epi32(x0, x1), _mm256_unpackhi_epi32(x0, x1)};
    for (int h = 0; h < 2; ++h) {
      __m256i lo = _mm256_shuffle_epi8(table, _mm256_and_si256(z[h], nibble));
      __m256i hi = _mm256_shuffle_epi8(table, _mm256_and_si256(_mm256_srli_epi16(z[h], 4), nibble));
      _mm256_store_si256(reinterpret_cast<__m256i *>(out[4 * reg + 2 * h].lane),
                         _mm256_unpacklo_epi8(lo, hi));
      _mm256_store_si256(reinterpret_cast<__m256i *>(out[4 * reg + 2 * h + 1].lane),
                         _mm256_unpackhi_epi8(lo, hi));
    }
  }
}

// Compute exact FP4 block dots using multi-row blocked PMADDUBSW (AVX-512BW on v4, AVX2 on v3),
// supporting both normal and non-plain E8M0 scales via vectorized FP32/FP64 scaling.
template <int M>
bool mfma_scaled_fp4(uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
                     const uint32_t *const *b, const uint32_t *const *c, const uint32_t *scale_a,
                     const uint32_t *scale_b, bool clamp_overflow) {
  constexpr uint64_t formats =
      RJ_ARITH_MATRIX_FORMAT_FP4 | (RJ_ARITH_MATRIX_FORMAT_FP4 << RJ_ARITH_MATRIX_FORMAT_B_SHIFT);
  if ((controls & 0x3f) != formats) {
    return false;
  }
  static_assert(M == 16 || M == 32);
  constexpr int blocks = 64 / M;
  constexpr int registers = M * M / 64;
  alignas(64) uint32_t ea[64], eb[64];
  // The integer dot is 4x the FP4 dot, so subtract 2 from the scale exponent.
  const bool plain_scales = mfma_extract_e8m0_scales(controls, scale_a, scale_b, ea, eb, -2);
  const uint32_t abs_mask = (controls & RJ_ARITH_MFMA_SCALE_ABS_C) ? 0x7fffffffU : 0xffffffffU;
  const uint32_t neg_mask = (controls & RJ_ARITH_MFMA_SCALE_NEG_C) ? 0x80000000U : 0U;

  // Stage A and B integer pairs for all blocks before any D stores (preserves arbitrary aliasing).
  MfmaFp4Pairs left[blocks][M / 16][16], right[blocks][M / 16][16];
  __m256i bias[blocks][M / 16];
  const __m256i values = _mm256_broadcastsi128_si256(
      _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12));
  const __m256i offset = _mm256_set1_epi8(12);
  const __m256i a_table = _mm256_add_epi8(values, offset);
  for (int block = 0; block < blocks; ++block) {
    for (int part = 0; part < M / 16; ++part) {
      const int lane = block * M + part * 16;
      mfma_fp4_transpose(a, lane, a_table, left[block][part]);
      mfma_fp4_transpose(b, lane, values, right[block][part]);
      __m256i bsum = _mm256_setzero_si256();
      for (const MfmaFp4Pairs &pair : right[block][part]) {
        const __m256i pv = _mm256_load_si256(reinterpret_cast<const __m256i *>(pair.lane));
        bsum = _mm256_sub_epi16(bsum, _mm256_maddubs_epi16(offset, pv));
      }
      bias[block][part] = bsum;
    }
  }

  const bool direct = mfma_ascending_store_safe<registers>(d, c);
  alignas(64) float result[registers][64];

#if defined(RJ_ARITH_TARGET_X86_64_V4)
  const __m512i f32_abs = _mm512_set1_epi32(0x7fffffff);
  const __m512i f32_inf = _mm512_set1_epi32(0x7f800000);
  if constexpr (M == 16) {
    // Process 8 rows at a time; each ZMM pairs two rows across all 16 columns.
    for (int rb = 0; rb < 16; rb += 8) {
      __m512 acc[8];
      __mmask16 c_finite[8];
      for (int r = 0; r < 8; ++r) {
        const MfmaWord out = mfma_output<16, 1>(rb + r, 0, 0);
        acc[r] = mfma_mixed_load_c(c[out.reg] + out.lane, abs_mask, neg_mask);
        c_finite[r] = clamp_overflow
                          ? _mm512_cmplt_epu32_mask(
                                _mm512_and_si512(_mm512_castps_si512(acc[r]), f32_abs), f32_inf)
                          : __mmask16(0);
      }
      for (int block = 0; block < 4; ++block) {
        const __m512i bias512 = _mm512_broadcast_i64x4(bias[block][0]);
        __m512i sum[4] = {bias512, bias512, bias512, bias512};
        RJ_ARITH_UNROLL(8)
        for (int pair = 0; pair < 16; ++pair) {
          const __m512i b512 = _mm512_broadcast_i64x4(
              _mm256_load_si256(reinterpret_cast<const __m256i *>(right[block][0][pair].lane)));
          RJ_ARITH_UNROLL(4)
          for (int rp = 0; rp < 4; ++rp) {
            const __m256i lo = _mm256_set1_epi16(int16_t(left[block][0][pair].lane[rb + 2 * rp]));
            const __m256i hi =
                _mm256_set1_epi16(int16_t(left[block][0][pair].lane[rb + 2 * rp + 1]));
            const __m512i lhs = _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
            sum[rp] = _mm512_add_epi16(sum[rp], _mm512_maddubs_epi16(lhs, b512));
          }
        }
        const __m512i eb_vec = _mm512_load_si512(eb + 16 * block);
        for (int rp = 0; rp < 4; ++rp) {
          const __m256i dots[2] = {_mm512_castsi512_si256(sum[rp]),
                                   _mm512_extracti64x4_epi64(sum[rp], 1)};
          for (int h = 0; h < 2; ++h) {
            const int r = 2 * rp + h;
            const __m512 fdots = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(dots[h]));
            const uint32_t a_exp = ea[16 * block + rb + r];
            __m512 scaled;
            if (__builtin_expect(plain_scales, 1)) {
              const __m512i exp = _mm512_add_epi32(eb_vec, _mm512_set1_epi32(int(a_exp) - 129));
              scaled = _mm512_mul_ps(fdots, _mm512_castsi512_ps(_mm512_slli_epi32(exp, 23)));
            } else {
              scaled = mfma_scale_nonplain_vec(fdots, a_exp, eb + 16 * block, -2);
            }
            acc[r] = _mm512_add_ps(acc[r], scaled);
          }
        }
      }
      for (int r = 0; r < 8; ++r) {
        if (clamp_overflow) {
          const __mmask16 inf_m = _mm512_cmpeq_epi32_mask(
              _mm512_and_si512(_mm512_castps_si512(acc[r]), f32_abs), f32_inf);
          acc[r] = _mm512_castsi512_ps(
              _mm512_mask_sub_epi32(_mm512_castps_si512(acc[r]), c_finite[r] & inf_m,
                                    _mm512_castps_si512(acc[r]), _mm512_set1_epi32(1)));
        }
        const MfmaWord out = mfma_output<16, 1>(rb + r, 0, 0);
        if (direct) {
          _mm512_storeu_ps(reinterpret_cast<float *>(d[out.reg] + out.lane), acc[r]);
        } else {
          _mm512_store_ps(result[out.reg] + out.lane, acc[r]);
        }
      }
    }
  } else {
    // M == 32: each ZMM holds all 32 columns as 32 int16 dot products; process 8 rows at a time.
    for (int rb = 0; rb < 32; rb += 8) {
      __m512 acc[8][2];
      __mmask16 c_finite[8][2];
      for (int r = 0; r < 8; ++r) {
        const MfmaWord out = mfma_output<32, 1>(rb + r, 0, 0);
        for (int p = 0; p < 2; ++p) {
          acc[r][p] = mfma_mixed_load_c(c[out.reg] + out.lane + 16 * p, abs_mask, neg_mask);
          c_finite[r][p] =
              clamp_overflow
                  ? _mm512_cmplt_epu32_mask(
                        _mm512_and_si512(_mm512_castps_si512(acc[r][p]), f32_abs), f32_inf)
                  : __mmask16(0);
        }
      }
      for (int block = 0; block < 2; ++block) {
        const __m512i bias512 =
            _mm512_inserti64x4(_mm512_castsi256_si512(bias[block][0]), bias[block][1], 1);
        __m512i sum[8];
        for (int r = 0; r < 8; ++r) {
          sum[r] = bias512;
        }
        const int a_part = rb / 16;
        const int a_off = rb % 16;
        RJ_ARITH_UNROLL(8)
        for (int pair = 0; pair < 16; ++pair) {
          const __m512i b512 = _mm512_inserti64x4(
              _mm512_castsi256_si512(
                  _mm256_load_si256(reinterpret_cast<const __m256i *>(right[block][0][pair].lane))),
              _mm256_load_si256(reinterpret_cast<const __m256i *>(right[block][1][pair].lane)), 1);
          RJ_ARITH_UNROLL(8)
          for (int r = 0; r < 8; ++r) {
            const __m512i lhs =
                _mm512_set1_epi16(int16_t(left[block][a_part][pair].lane[a_off + r]));
            sum[r] = _mm512_add_epi16(sum[r], _mm512_maddubs_epi16(lhs, b512));
          }
        }
        const __m512i eb0 = _mm512_load_si512(eb + 32 * block + 0);
        const __m512i eb1 = _mm512_load_si512(eb + 32 * block + 16);
        for (int r = 0; r < 8; ++r) {
          const __m256i dots[2] = {_mm512_castsi512_si256(sum[r]),
                                   _mm512_extracti64x4_epi64(sum[r], 1)};
          const __m512i ebs[2] = {eb0, eb1};
          const uint32_t a_exp = ea[32 * block + rb + r];
          for (int p = 0; p < 2; ++p) {
            const __m512 fdots = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(dots[p]));
            __m512 scaled;
            if (__builtin_expect(plain_scales, 1)) {
              const __m512i exp = _mm512_add_epi32(ebs[p], _mm512_set1_epi32(int(a_exp) - 129));
              scaled = _mm512_mul_ps(fdots, _mm512_castsi512_ps(_mm512_slli_epi32(exp, 23)));
            } else {
              scaled = mfma_scale_nonplain_vec(fdots, a_exp, eb + 32 * block + 16 * p, -2);
            }
            acc[r][p] = _mm512_add_ps(acc[r][p], scaled);
          }
        }
      }
      for (int r = 0; r < 8; ++r) {
        const MfmaWord out = mfma_output<32, 1>(rb + r, 0, 0);
        for (int p = 0; p < 2; ++p) {
          if (clamp_overflow) {
            const __mmask16 inf_m = _mm512_cmpeq_epi32_mask(
                _mm512_and_si512(_mm512_castps_si512(acc[r][p]), f32_abs), f32_inf);
            acc[r][p] = _mm512_castsi512_ps(
                _mm512_mask_sub_epi32(_mm512_castps_si512(acc[r][p]), c_finite[r][p] & inf_m,
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
  // AVX2: 4-row blocked PMADDUBSW per 16-column part.
  const __m256i f32_abs = _mm256_set1_epi32(0x7fffffff);
  const __m256i f32_inf = _mm256_set1_epi32(0x7f800000);
  for (int part = 0; part < M / 16; ++part) {
    for (int rb = 0; rb < M; rb += 4) {
      alignas(32) float tile_acc[4][16];
      alignas(32) uint32_t tile_fin[4][16];
      for (int r = 0; r < 4; ++r) {
        const MfmaWord out = mfma_output<M, 1>(rb + r, 16 * part, 0);
        const MfmaNative c0 = mfma_mixed_load_c(c[out.reg] + out.lane, abs_mask, neg_mask);
        const MfmaNative c1 = mfma_mixed_load_c(c[out.reg] + out.lane + 8, abs_mask, neg_mask);
        _mm256_store_ps(tile_acc[r] + 0, c0);
        _mm256_store_ps(tile_acc[r] + 8, c1);
        if (clamp_overflow) {
          _mm256_store_si256(
              reinterpret_cast<__m256i *>(tile_fin[r] + 0),
              _mm256_cmpgt_epi32(f32_inf, _mm256_and_si256(_mm256_castps_si256(c0), f32_abs)));
          _mm256_store_si256(
              reinterpret_cast<__m256i *>(tile_fin[r] + 8),
              _mm256_cmpgt_epi32(f32_inf, _mm256_and_si256(_mm256_castps_si256(c1), f32_abs)));
        }
      }
      const int a_part = rb / 16;
      const int a_off = rb % 16;
      for (int block = 0; block < blocks; ++block) {
        __m256i sum[4] = {bias[block][part], bias[block][part], bias[block][part],
                          bias[block][part]};
        RJ_ARITH_UNROLL(8)
        for (int pair = 0; pair < 16; ++pair) {
          const __m256i rhs =
              _mm256_load_si256(reinterpret_cast<const __m256i *>(right[block][part][pair].lane));
          RJ_ARITH_UNROLL(4)
          for (int r = 0; r < 4; ++r) {
            const __m256i lhs =
                _mm256_set1_epi16(int16_t(left[block][a_part][pair].lane[a_off + r]));
            sum[r] = _mm256_add_epi16(sum[r], _mm256_maddubs_epi16(lhs, rhs));
          }
        }
        for (int r = 0; r < 4; ++r) {
          const uint32_t a_exp = ea[block * M + rb + r];
          for (int half = 0; half < 2; ++half) {
            const int col = part * 16 + half * 8;
            const __m128i dots =
                half ? _mm256_extracti128_si256(sum[r], 1) : _mm256_castsi256_si128(sum[r]);
            const __m256 fdots = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(dots));
            __m256 scaled;
            if (__builtin_expect(plain_scales, 1)) {
              const __m256i exponents = _mm256_add_epi32(
                  _mm256_set1_epi32(int(a_exp) - 129),
                  _mm256_load_si256(reinterpret_cast<const __m256i *>(eb + block * M + col)));
              const __m256 factor = _mm256_castsi256_ps(_mm256_slli_epi32(exponents, 23));
              scaled = _mm256_mul_ps(fdots, factor);
            } else {
              scaled = mfma_scale_nonplain_vec(fdots, a_exp, eb + block * M + col, -2);
            }
            const __m256 cur = _mm256_load_ps(tile_acc[r] + 8 * half);
            _mm256_store_ps(tile_acc[r] + 8 * half, _mm256_add_ps(cur, scaled));
          }
        }
      }
      for (int r = 0; r < 4; ++r) {
        if (clamp_overflow) {
          for (int half = 0; half < 2; ++half) {
            const __m256i bits = _mm256_castps_si256(_mm256_load_ps(tile_acc[r] + 8 * half));
            const __m256i inf_m = _mm256_cmpeq_epi32(_mm256_and_si256(bits, f32_abs), f32_inf);
            const __m256i cm =
                _mm256_load_si256(reinterpret_cast<const __m256i *>(tile_fin[r] + 8 * half));
            const __m256i sat = _mm256_and_si256(inf_m, cm);
            _mm256_store_ps(tile_acc[r] + 8 * half,
                            _mm256_castsi256_ps(_mm256_add_epi32(bits, sat)));
          }
        }
        const MfmaWord out = mfma_output<M, 1>(rb + r, 16 * part, 0);
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
  return true;
}
} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
