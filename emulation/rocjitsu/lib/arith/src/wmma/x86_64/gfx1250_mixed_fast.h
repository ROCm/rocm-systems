// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Fast x86-64 (AVX2 / AVX-512) gfx1250 mixed-format and block-scaled WMMA kernels:
// - Vectorized C load/store and E8M0/E5M3/E4M3 scale decoding.
// - Zero-transpose vpshufb + vpmaddubsw exact integer dot for FP4xFP4 (note: for unscaled
//   Block == 0, integer 0 converts to +0.0f so C = -0.0f with all -0.0f products yields +0.0f,
//   which is permitted under RJ_ARITH_SEMANTICS_LOOSE).
// - Fast FP8/BF8/FP6/BF6/FP4 decoding and 16-row (v4) / 4x2 (v3) FMA tiling.
#pragma once

#include "common/compiler.h"
#include "common/internal.h"
#include "common/x86_64/fp8.h"
#include "mfma/mixed_formats.h"
#include "rj_arith/detail/rj_arith_mfma.h"
#include "rj_arith/detail/rj_arith_wmma.h"
#include "wmma/x86_64/gfx1250_bf8.h"
#include "wmma/x86_64/gfx1250_fp8_decode.h"
#include <cstring>
#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Decode 16 scale bytes for one block (lane_base..lane_base+15) into 16 FP32 values.
inline void gfx1250_decode_scales_16(const uint32_t *const *scale_vgprs, int lane_base, int block,
                                     int format, float out[16]) {
  const uint32_t *src = scale_vgprs[block / 4] + lane_base;
  const int shift = 8 * (block % 4);
#if defined(RJ_ARITH_TARGET_X86_64_V4)
  const __m512i words = _mm512_srli_epi32(_mm512_loadu_si512(src), shift);
  const __m512i raw = _mm512_and_si512(words, _mm512_set1_epi32(255));
  if (format == 0) {
    // E8M0: 255 -> qNaN, 0 -> 2^-127 (0x00400000), 1..254 -> raw << 23.
    __m512i bits = _mm512_slli_epi32(raw, 23);
    bits = _mm512_mask_mov_epi32(bits, _mm512_cmpeq_epi32_mask(raw, _mm512_setzero_si512()),
                                 _mm512_set1_epi32(0x00400000));
    bits = _mm512_mask_mov_epi32(bits, _mm512_cmpeq_epi32_mask(raw, _mm512_set1_epi32(255)),
                                 _mm512_set1_epi32(0x7fc00000));
    _mm512_store_ps(out, _mm512_castsi512_ps(bits));
  } else if (format == 1) {
    _mm512_store_ps(out, widen_fp8_e5m3(raw));
  } else {
    const __m128i bytes = _mm512_cvtepi32_epi8(raw);
    _mm512_store_ps(out, gfx1250_widen_fp8_16_v4(bytes));
  }
#else
  for (int half = 0; half < 16; half += 8) {
    const __m256i words =
        _mm256_srli_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(src + half)), shift);
    const __m256i raw = _mm256_and_si256(words, _mm256_set1_epi32(255));
    if (format == 0) {
      __m256i bits = _mm256_slli_epi32(raw, 23);
      bits = _mm256_blendv_epi8(bits, _mm256_set1_epi32(0x00400000),
                                _mm256_cmpeq_epi32(raw, _mm256_setzero_si256()));
      bits = _mm256_blendv_epi8(bits, _mm256_set1_epi32(0x7fc00000),
                                _mm256_cmpeq_epi32(raw, _mm256_set1_epi32(255)));
      _mm256_store_ps(out + half, _mm256_castsi256_ps(bits));
    } else if (format == 1) {
      _mm256_store_ps(out + half, widen_fp8_e5m3(raw));
    } else {
      const __m128i lo128 = _mm256_castsi256_si128(raw);
      const __m128i hi128 = _mm256_extracti128_si256(raw, 1);
      const __m128i pack16 = _mm_packus_epi32(lo128, hi128);
      const __m128i pack8 = _mm_packus_epi16(pack16, pack16);
      _mm256_store_ps(out + half, gfx1250_widen_fp8_8_v3(pack8));
    }
  }
#endif
}

template <int M>
inline void gfx1250_mixed_load_c(const uint32_t *const *c, uint64_t controls, float result[M][16]) {
  const uint32_t keep = (controls & RJ_ARITH_WMMA_MIXED_ABS_C) ? 0x7fffffffU : 0xffffffffU;
  const uint32_t negate = (controls & RJ_ARITH_WMMA_MIXED_NEG_C) ? 0x80000000U : 0U;
#if defined(RJ_ARITH_TARGET_X86_64_V4)
  const __m512i kv = _mm512_set1_epi32(int(keep));
  const __m512i nv = _mm512_set1_epi32(int(negate));
  for (int mh = 0; mh < M / 16; ++mh) {
    for (int r = 0; r < 8; ++r) {
      const __m512i lo = _mm512_loadu_si512(c[8 * mh + r]);
      const __m512i hi = _mm512_loadu_si512(c[8 * mh + r] + 16);
      _mm512_store_ps(result[16 * mh + r],
                      _mm512_castsi512_ps(_mm512_xor_si512(_mm512_and_si512(lo, kv), nv)));
      _mm512_store_ps(result[16 * mh + 8 + r],
                      _mm512_castsi512_ps(_mm512_xor_si512(_mm512_and_si512(hi, kv), nv)));
    }
  }
#else
  const __m256i kv = _mm256_set1_epi32(int(keep));
  const __m256i nv = _mm256_set1_epi32(int(negate));
  for (int mh = 0; mh < M / 16; ++mh) {
    for (int r = 0; r < 8; ++r) {
      for (int g = 0; g < 2; ++g) {
        const uint32_t *src = c[8 * mh + r] + 16 * g;
        float *dst = result[16 * mh + 8 * g + r];
        const __m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(src));
        const __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(src + 8));
        _mm256_store_ps(dst, _mm256_castsi256_ps(_mm256_xor_si256(_mm256_and_si256(v0, kv), nv)));
        _mm256_store_ps(dst + 8,
                        _mm256_castsi256_ps(_mm256_xor_si256(_mm256_and_si256(v1, kv), nv)));
      }
    }
  }
#endif
}

template <int M> inline void gfx1250_mixed_store_d(uint32_t *const *d, const float result[M][16]) {
  for (int mh = 0; mh < M / 16; ++mh) {
    for (int r = 0; r < 8; ++r) {
      std::memcpy(d[8 * mh + r], result[16 * mh + r], 16 * sizeof(float));
      std::memcpy(d[8 * mh + r] + 16, result[16 * mh + 8 + r], 16 * sizeof(float));
    }
  }
}

// FP4 E2M1 integer lookup tables (scaled by 2):
// A is signed int8 in [-12, 12]; B is biased unsigned uint8 = s8 + 12 in [0, 24].
// Zero-transpose vpshufb maps low nibbles -> (k0,k2,k4,k6) and high nibbles -> (k1,k3,k5,k7).
template <int M, int Block>
void gfx1250_fp4_exact_int_kernel(uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, const uint32_t *const *c,
                                  const uint32_t *const *scale_a, const uint32_t *const *scale_b) {
  alignas(64) float result[M][16];
  gfx1250_mixed_load_c<M>(c, controls, result);

  constexpr int groups = Block ? 128 / (Block ? Block : 1) : 1;
  constexpr int vgprs_per_block = Block ? Block / 8 : 8;
  constexpr int halfwaves_per_block = Block ? 1 : 2;
  constexpr int steps_per_block = vgprs_per_block * halfwaves_per_block * 2;
  const int asf = Block ? int((controls >> 8) & 3) : 0;
  const int bsf = Block ? int((controls >> 10) & 3) : 0;
  const int b_scale_lane = Block ? 16 * int((controls >> 13) & 1) : 0;

  alignas(64) float sa[groups][M];
  alignas(64) float sb[groups][16];
  if constexpr (Block != 0) {
    for (int blk = 0; blk < groups; ++blk) {
      for (int mh = 0; mh < M / 16; ++mh) {
        const int a_lane = M == 32 ? 16 * mh : 16 * int((controls >> 12) & 1);
        gfx1250_decode_scales_16(scale_a, a_lane, blk, asf, sa[blk] + 16 * mh);
      }
      gfx1250_decode_scales_16(scale_b, b_scale_lane, blk, bsf, sb[blk]);
    }
  }

  // Table for signed A (2*val in [-12, 12]) and biased unsigned B (2*val + 12 in [0, 24]).
  const __m128i lut_a128 = _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
  const __m128i lut_b128 = _mm_setr_epi8(12, 13, 14, 15, 16, 18, 20, 24, 12, 11, 10, 9, 8, 6, 4, 0);

  for (int blk = 0; blk < groups; ++blk) {
    const int hw0 = Block == 32 ? (blk % 2) : Block == 16 ? ((blk / 2) % 2) : 0;
    const int vbase = Block == 32 ? 4 * (blk / 2) : Block == 16 ? 4 * (blk / 4) + 2 * (blk % 2) : 0;

#if defined(RJ_ARITH_TARGET_X86_64_V4)
    const __m512i lut_a = _mm512_broadcast_i32x4(lut_a128);
    const __m512i lut_b = _mm512_broadcast_i32x4(lut_b128);
    const __m512i mask_0f = _mm512_set1_epi8(0x0f);
    const __m512i bias12 = _mm512_set1_epi8(12);
    const __m512i ones16 = _mm512_set1_epi16(1);
    const __m512 unit = _mm512_set1_ps(0.25f);

    alignas(64) uint32_t b_u8[steps_per_block][16];
    int step = 0;
    for (int hw = 0; hw < halfwaves_per_block; ++hw) {
      const int lane = 16 * (hw0 + hw);
      for (int v = 0; v < vgprs_per_block; ++v) {
        const __m512i bw = _mm512_loadu_si512(b[vbase + v] + lane);
        const __m512i blo = _mm512_shuffle_epi8(lut_b, _mm512_and_si512(bw, mask_0f));
        const __m512i bhi =
            _mm512_shuffle_epi8(lut_b, _mm512_and_si512(_mm512_srli_epi16(bw, 4), mask_0f));
        _mm512_store_si512(b_u8[step++], blo);
        _mm512_store_si512(b_u8[step++], bhi);
      }
    }

    for (int mh = 0; mh < M / 16; ++mh) {
      const uint32_t *const *a_bank = a + 8 * mh;
      alignas(64) uint32_t a_s8[steps_per_block][16];
      alignas(64) int32_t a_bias_arr[16];
      __m512i a_bias_16_0 = _mm512_setzero_si512();
      __m512i a_bias_16_1 = _mm512_setzero_si512();
      step = 0;
      for (int hw = 0; hw < halfwaves_per_block; ++hw) {
        const int lane = 16 * (hw0 + hw);
        for (int v = 0; v < vgprs_per_block; ++v) {
          const __m512i aw = _mm512_loadu_si512(a_bank[vbase + v] + lane);
          const __m512i alo = _mm512_shuffle_epi8(lut_a, _mm512_and_si512(aw, mask_0f));
          const __m512i ahi =
              _mm512_shuffle_epi8(lut_a, _mm512_and_si512(_mm512_srli_epi16(aw, 4), mask_0f));
          _mm512_store_si512(a_s8[step], alo);
          _mm512_store_si512(a_s8[step + 1], ahi);
          // 12 * sum_k a_s8 fits in int16 for <=16 steps; split into two accumulators for K=128.
          const __m512i p0 = _mm512_maddubs_epi16(bias12, alo);
          const __m512i p1 = _mm512_maddubs_epi16(bias12, ahi);
          if (step < 16) {
            a_bias_16_0 = _mm512_add_epi16(a_bias_16_0, _mm512_add_epi16(p0, p1));
          } else {
            a_bias_16_1 = _mm512_add_epi16(a_bias_16_1, _mm512_add_epi16(p0, p1));
          }
          step += 2;
        }
      }
      __m512i a_bias_32 = _mm512_madd_epi16(a_bias_16_0, ones16);
      if constexpr (steps_per_block > 16) {
        a_bias_32 = _mm512_add_epi32(a_bias_32, _mm512_madd_epi16(a_bias_16_1, ones16));
      }
      _mm512_store_si512(a_bias_arr, a_bias_32);

      // Accumulate in int16 (max 16 steps * 576 = 9216 << 32767) and widen once per chunk.
      __m512i idot[16];
      RJ_ARITH_UNROLL(16)
      for (int r = 0; r < 16; ++r) {
        idot[r] = _mm512_sub_epi32(_mm512_setzero_si512(), _mm512_set1_epi32(a_bias_arr[r]));
      }
      constexpr int chunk = steps_per_block > 16 ? 16 : steps_per_block;
      for (int s0 = 0; s0 < steps_per_block; s0 += chunk) {
        __m512i acc16[16];
        RJ_ARITH_UNROLL(16)
        for (int r = 0; r < 16; ++r) {
          acc16[r] = _mm512_setzero_si512();
        }
        for (int s = s0; s < s0 + chunk; ++s) {
          const __m512i bv = _mm512_load_si512(b_u8[s]);
          RJ_ARITH_UNROLL(16)
          for (int r = 0; r < 16; ++r) {
            const __m512i av = _mm512_set1_epi32(int(a_s8[s][r]));
            acc16[r] = _mm512_add_epi16(acc16[r], _mm512_maddubs_epi16(bv, av));
          }
        }
        RJ_ARITH_UNROLL(16)
        for (int r = 0; r < 16; ++r) {
          idot[r] = _mm512_add_epi32(idot[r], _mm512_madd_epi16(acc16[r], ones16));
        }
      }
      const __m512 sb_v = Block ? _mm512_load_ps(sb[blk]) : _mm512_setzero_ps();
      RJ_ARITH_UNROLL(16)
      for (int r = 0; r < 16; ++r) {
        __m512 partial = _mm512_mul_ps(_mm512_cvtepi32_ps(idot[r]), unit);
        if constexpr (Block != 0) {
          partial =
              _mm512_mul_ps(_mm512_mul_ps(partial, _mm512_set1_ps(sa[blk][16 * mh + r])), sb_v);
        }
        const __m512 cur = _mm512_load_ps(result[16 * mh + r]);
        _mm512_store_ps(result[16 * mh + r], _mm512_add_ps(cur, partial));
      }
    }
#else
    const __m256i lut_a = _mm256_broadcastsi128_si256(lut_a128);
    const __m256i lut_b = _mm256_broadcastsi128_si256(lut_b128);
    const __m256i mask_0f = _mm256_set1_epi8(0x0f);
    const __m256i bias12 = _mm256_set1_epi8(12);
    const __m256i ones16 = _mm256_set1_epi16(1);
    const __m256 unit = _mm256_set1_ps(0.25f);

    alignas(32) uint32_t b_u8[steps_per_block][16];
    int step = 0;
    for (int hw = 0; hw < halfwaves_per_block; ++hw) {
      const int lane = 16 * (hw0 + hw);
      for (int v = 0; v < vgprs_per_block; ++v) {
        for (int col = 0; col < 16; col += 8) {
          const __m256i bw =
              _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b[vbase + v] + lane + col));
          const __m256i blo = _mm256_shuffle_epi8(lut_b, _mm256_and_si256(bw, mask_0f));
          const __m256i bhi =
              _mm256_shuffle_epi8(lut_b, _mm256_and_si256(_mm256_srli_epi16(bw, 4), mask_0f));
          _mm256_store_si256(reinterpret_cast<__m256i *>(b_u8[step] + col), blo);
          _mm256_store_si256(reinterpret_cast<__m256i *>(b_u8[step + 1] + col), bhi);
        }
        step += 2;
      }
    }

    for (int mh = 0; mh < M / 16; ++mh) {
      const uint32_t *const *a_bank = a + 8 * mh;
      alignas(32) uint32_t a_s8[steps_per_block][16];
      alignas(32) int32_t a_bias_arr[16];
      for (int r0 = 0; r0 < 16; r0 += 8) {
        __m256i bias0 = _mm256_setzero_si256();
        __m256i bias1 = _mm256_setzero_si256();
        step = 0;
        for (int hw = 0; hw < halfwaves_per_block; ++hw) {
          const int lane = 16 * (hw0 + hw);
          for (int v = 0; v < vgprs_per_block; ++v) {
            const __m256i aw = _mm256_loadu_si256(
                reinterpret_cast<const __m256i *>(a_bank[vbase + v] + lane + r0));
            const __m256i alo = _mm256_shuffle_epi8(lut_a, _mm256_and_si256(aw, mask_0f));
            const __m256i ahi =
                _mm256_shuffle_epi8(lut_a, _mm256_and_si256(_mm256_srli_epi16(aw, 4), mask_0f));
            _mm256_store_si256(reinterpret_cast<__m256i *>(a_s8[step] + r0), alo);
            _mm256_store_si256(reinterpret_cast<__m256i *>(a_s8[step + 1] + r0), ahi);
            const __m256i p0 = _mm256_maddubs_epi16(bias12, alo);
            const __m256i p1 = _mm256_maddubs_epi16(bias12, ahi);
            if (step < 16) {
              bias0 = _mm256_add_epi16(bias0, _mm256_add_epi16(p0, p1));
            } else {
              bias1 = _mm256_add_epi16(bias1, _mm256_add_epi16(p0, p1));
            }
            step += 2;
          }
        }
        __m256i b32 = _mm256_madd_epi16(bias0, ones16);
        if constexpr (steps_per_block > 16) {
          b32 = _mm256_add_epi32(b32, _mm256_madd_epi16(bias1, ones16));
        }
        _mm256_store_si256(reinterpret_cast<__m256i *>(a_bias_arr + r0), b32);
      }

      const __m256 sb0 = Block ? _mm256_load_ps(sb[blk]) : _mm256_setzero_ps();
      const __m256 sb1 = Block ? _mm256_load_ps(sb[blk] + 8) : _mm256_setzero_ps();
      constexpr int chunk = steps_per_block > 16 ? 16 : steps_per_block;
      for (int base = 0; base < 16; base += 4) {
        __m256i idot[4][2];
        RJ_ARITH_UNROLL(4)
        for (int r = 0; r < 4; ++r) {
          const __m256i neg_b =
              _mm256_sub_epi32(_mm256_setzero_si256(), _mm256_set1_epi32(a_bias_arr[base + r]));
          idot[r][0] = neg_b;
          idot[r][1] = neg_b;
        }
        for (int s0 = 0; s0 < steps_per_block; s0 += chunk) {
          __m256i acc16[4][2];
          RJ_ARITH_UNROLL(4)
          for (int r = 0; r < 4; ++r) {
            acc16[r][0] = _mm256_setzero_si256();
            acc16[r][1] = _mm256_setzero_si256();
          }
          for (int s = s0; s < s0 + chunk; ++s) {
            const __m256i bv0 = _mm256_load_si256(reinterpret_cast<const __m256i *>(b_u8[s]));
            const __m256i bv1 = _mm256_load_si256(reinterpret_cast<const __m256i *>(b_u8[s] + 8));
            RJ_ARITH_UNROLL(4)
            for (int r = 0; r < 4; ++r) {
              const __m256i av = _mm256_set1_epi32(int(a_s8[s][base + r]));
              acc16[r][0] = _mm256_add_epi16(acc16[r][0], _mm256_maddubs_epi16(bv0, av));
              acc16[r][1] = _mm256_add_epi16(acc16[r][1], _mm256_maddubs_epi16(bv1, av));
            }
          }
          RJ_ARITH_UNROLL(4)
          for (int r = 0; r < 4; ++r) {
            idot[r][0] = _mm256_add_epi32(idot[r][0], _mm256_madd_epi16(acc16[r][0], ones16));
            idot[r][1] = _mm256_add_epi32(idot[r][1], _mm256_madd_epi16(acc16[r][1], ones16));
          }
        }
        RJ_ARITH_UNROLL(4)
        for (int r = 0; r < 4; ++r) {
          __m256 p0 = _mm256_mul_ps(_mm256_cvtepi32_ps(idot[r][0]), unit);
          __m256 p1 = _mm256_mul_ps(_mm256_cvtepi32_ps(idot[r][1]), unit);
          if constexpr (Block != 0) {
            const __m256 sav = _mm256_set1_ps(sa[blk][16 * mh + base + r]);
            p0 = _mm256_mul_ps(_mm256_mul_ps(p0, sav), sb0);
            p1 = _mm256_mul_ps(_mm256_mul_ps(p1, sav), sb1);
          }
          float *row_ptr = result[16 * mh + base + r];
          _mm256_store_ps(row_ptr, _mm256_add_ps(_mm256_load_ps(row_ptr), p0));
          _mm256_store_ps(row_ptr + 8, _mm256_add_ps(_mm256_load_ps(row_ptr + 8), p1));
        }
      }
    }
#endif
  }
  gfx1250_mixed_store_d<M>(d, result);
}

template <int Fmt> struct Gfx1250Mag6Table {
  uint16_t bits[32]{};
  constexpr Gfx1250Mag6Table() {
    for (int i = 0; i < 32; ++i) {
      bits[i] = uint16_t(bit_cast<uint32_t>(matrix_decode_subbyte(i, Fmt)) >> 16);
    }
  }
};

// Fast decode of 16-row gfx1250 mixed operand into K-major float[128][16].
// FP8/BF8 use vpermi2w (v4) / F16C (v3); FP6/BF6 load each 3-VGPR group (16 codes) once;
// FP4 loads each VGPR (8 codes) once.
template <bool InterleaveFour>
inline void gfx1250_mixed_decode_16(const uint32_t *const *input, int format, float out[128][16]) {
  if (format == RJ_ARITH_MATRIX_FORMAT_FP8 || format == RJ_ARITH_MATRIX_FORMAT_BF8) {
    const bool is_bf8 = format == RJ_ARITH_MATRIX_FORMAT_BF8;
    for (int k = 0; k < 128; k += 4) {
      const int reg = 4 * (k / 32) + (k % 16) / 4;
      const int lane = 16 * ((k / 16) % 2);
#if defined(RJ_ARITH_TARGET_X86_64_V4)
      const __m512i words = _mm512_loadu_si512(input[reg] + lane);
      const __m512i hi_mask = _mm512_set1_epi32(int(0xffff0000U));
      if (is_bf8) {
        const __m512i lo = gfx1250_fp8_to_bf16_pairs<true>(_mm512_cvtepi32_epi16(words));
        const __m512i hi =
            gfx1250_fp8_to_bf16_pairs<true>(_mm512_cvtepi32_epi16(_mm512_srli_epi32(words, 16)));
        _mm512_store_ps(out[k], _mm512_castsi512_ps(_mm512_slli_epi32(lo, 16)));
        _mm512_store_ps(out[k + 1], _mm512_castsi512_ps(_mm512_and_si512(lo, hi_mask)));
        _mm512_store_ps(out[k + 2], _mm512_castsi512_ps(_mm512_slli_epi32(hi, 16)));
        _mm512_store_ps(out[k + 3], _mm512_castsi512_ps(_mm512_and_si512(hi, hi_mask)));
      } else {
        const __m512i lo = gfx1250_fp8_to_bf16_pairs<false>(_mm512_cvtepi32_epi16(words));
        const __m512i hi =
            gfx1250_fp8_to_bf16_pairs<false>(_mm512_cvtepi32_epi16(_mm512_srli_epi32(words, 16)));
        _mm512_store_ps(out[k], _mm512_castsi512_ps(_mm512_slli_epi32(lo, 16)));
        _mm512_store_ps(out[k + 1], _mm512_castsi512_ps(_mm512_and_si512(lo, hi_mask)));
        _mm512_store_ps(out[k + 2], _mm512_castsi512_ps(_mm512_slli_epi32(hi, 16)));
        _mm512_store_ps(out[k + 3], _mm512_castsi512_ps(_mm512_and_si512(hi, hi_mask)));
      }
#else
      for (int off = 0; off < 16; off += 8) {
        __m256i words =
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(input[reg] + lane + off));
        if (is_bf8) {
          const __m256i mag = _mm256_and_si256(words, _mm256_set1_epi8(0x7f));
          const __m256i nan = _mm256_cmpgt_epi8(mag, _mm256_set1_epi8(0x7c));
          if (_mm256_movemask_epi8(nan)) [[unlikely]] {
            const __m256i canon = _mm256_or_si256(
                _mm256_and_si256(words, _mm256_set1_epi8(static_cast<int8_t>(0x80))),
                _mm256_set1_epi8(0x7e));
            words = _mm256_blendv_epi8(words, canon, nan);
          }
          __m256 e0, e1, e2, e3;
          gfx1250_widen_bf8_pair(bit_cast<Gfx1250PackedWords>(words), e0, e1);
          gfx1250_widen_bf8_pair(bit_cast<Gfx1250PackedWords>(_mm256_srli_epi32(words, 16)), e2,
                                 e3);
          _mm256_store_ps(out[k] + off, e0);
          _mm256_store_ps(out[k + 1] + off, e1);
          _mm256_store_ps(out[k + 2] + off, e2);
          _mm256_store_ps(out[k + 3] + off, e3);
        } else {
          __m256 e0, e1, e2, e3;
          gfx1250_widen_fp8_quad_v3(words, e0, e1, e2, e3);
          _mm256_store_ps(out[k] + off, e0);
          _mm256_store_ps(out[k + 1] + off, e1);
          _mm256_store_ps(out[k + 2] + off, e2);
          _mm256_store_ps(out[k + 3] + off, e3);
        }
      }
#endif
    }
    return;
  }

  if (format == RJ_ARITH_MATRIX_FORMAT_FP4) {
#if defined(RJ_ARITH_TARGET_X86_64_V4)
    const __m512 lut =
        _mm512_setr_ps(0, 0.5f, 1, 1.5f, 2, 3, 4, 6, -0.0f, -0.5f, -1, -1.5f, -2, -3, -4, -6);
    for (int hw = 0; hw < 2; ++hw) {
      const int lane = 16 * hw;
      for (int v = 0; v < 8; ++v) {
        __m512i w = _mm512_loadu_si512(input[v] + lane);
        const int b0 = InterleaveFour ? 16 * v + 4 * hw : (v / 4) * 64 + 32 * hw + (v % 4) * 8;
        const int b1 = InterleaveFour ? b0 + 8 : b0 + 4;
        RJ_ARITH_UNROLL(4)
        for (int j = 0; j < 4; ++j) {
          _mm512_store_ps(out[b0 + j], _mm512_permutexvar_ps(w, lut));
          w = _mm512_srli_epi32(w, 4);
        }
        RJ_ARITH_UNROLL(4)
        for (int j = 0; j < 4; ++j) {
          _mm512_store_ps(out[b1 + j], _mm512_permutexvar_ps(w, lut));
          w = _mm512_srli_epi32(w, 4);
        }
      }
    }
#else
    const __m256 lut = _mm256_setr_ps(0, 0.5f, 1, 1.5f, 2, 3, 4, 6);
    const __m256i sign_bit = _mm256_set1_epi32(8);
    for (int hw = 0; hw < 2; ++hw) {
      const int lane = 16 * hw;
      for (int v = 0; v < 8; ++v) {
        const int b0 = InterleaveFour ? 16 * v + 4 * hw : (v / 4) * 64 + 32 * hw + (v % 4) * 8;
        const int b1 = InterleaveFour ? b0 + 8 : b0 + 4;
        for (int off = 0; off < 16; off += 8) {
          __m256i w = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(input[v] + lane + off));
          RJ_ARITH_UNROLL(4)
          for (int j = 0; j < 4; ++j) {
            __m256 mag = _mm256_permutevar8x32_ps(lut, w);
            __m256i sgn = _mm256_slli_epi32(_mm256_and_si256(w, sign_bit), 28);
            _mm256_store_ps(out[b0 + j] + off,
                            _mm256_castsi256_ps(_mm256_or_si256(_mm256_castps_si256(mag), sgn)));
            w = _mm256_srli_epi32(w, 4);
          }
          RJ_ARITH_UNROLL(4)
          for (int j = 0; j < 4; ++j) {
            __m256 mag = _mm256_permutevar8x32_ps(lut, w);
            __m256i sgn = _mm256_slli_epi32(_mm256_and_si256(w, sign_bit), 28);
            _mm256_store_ps(out[b1 + j] + off,
                            _mm256_castsi256_ps(_mm256_or_si256(_mm256_castps_si256(mag), sgn)));
            w = _mm256_srli_epi32(w, 4);
          }
        }
      }
    }
#endif
    return;
  }

  // FP6 / BF6: load each 3-VGPR group (16 codes) once for both layouts.
  auto decode_6 = [&]<int Fmt>() {
#if defined(RJ_ARITH_TARGET_X86_64_V4)
    alignas(64) static constexpr Gfx1250Mag6Table<Fmt> table;
    const __m512i lut = _mm512_load_si512(table.bits);
    auto widen6 = [&](__m512i code) {
      const __m512i h = _mm512_permutexvar_epi16(code, lut);
      return _mm512_castsi512_ps(
          _mm512_or_si512(_mm512_slli_epi32(h, 16),
                          _mm512_slli_epi32(_mm512_and_si512(code, _mm512_set1_epi32(32)), 26)));
    };
    for (int hw = 0; hw < 2; ++hw) {
      const int lane = 16 * hw;
      for (int g = 0; g < 4; ++g) {
        const int vb = 3 * g;
        const __m512i w0 = _mm512_loadu_si512(input[vb] + lane);
        const __m512i w1 = _mm512_loadu_si512(input[vb + 1] + lane);
        const __m512i w2 = _mm512_loadu_si512(input[vb + 2] + lane);
        const int b0 = InterleaveFour ? 32 * g + 4 * hw : (g / 2) * 64 + 32 * hw + (g % 2) * 16;
        constexpr int stride = InterleaveFour ? 8 : 4;
        const int b1 = b0 + stride, b2 = b0 + 2 * stride, b3 = b0 + 3 * stride;
        _mm512_store_ps(out[b0 + 0], widen6(w0));
        _mm512_store_ps(out[b0 + 1], widen6(_mm512_srli_epi32(w0, 6)));
        _mm512_store_ps(out[b0 + 2], widen6(_mm512_srli_epi32(w0, 12)));
        _mm512_store_ps(out[b0 + 3], widen6(_mm512_srli_epi32(w0, 18)));
        _mm512_store_ps(out[b1 + 0], widen6(_mm512_srli_epi32(w0, 24)));
        _mm512_store_ps(out[b1 + 1], widen6(_mm512_or_si512(_mm512_srli_epi32(w0, 30),
                                                            _mm512_slli_epi32(w1, 2))));
        _mm512_store_ps(out[b1 + 2], widen6(_mm512_srli_epi32(w1, 4)));
        _mm512_store_ps(out[b1 + 3], widen6(_mm512_srli_epi32(w1, 10)));
        _mm512_store_ps(out[b2 + 0], widen6(_mm512_srli_epi32(w1, 16)));
        _mm512_store_ps(out[b2 + 1], widen6(_mm512_srli_epi32(w1, 22)));
        _mm512_store_ps(out[b2 + 2], widen6(_mm512_or_si512(_mm512_srli_epi32(w1, 28),
                                                            _mm512_slli_epi32(w2, 4))));
        _mm512_store_ps(out[b2 + 3], widen6(_mm512_srli_epi32(w2, 2)));
        _mm512_store_ps(out[b3 + 0], widen6(_mm512_srli_epi32(w2, 8)));
        _mm512_store_ps(out[b3 + 1], widen6(_mm512_srli_epi32(w2, 14)));
        _mm512_store_ps(out[b3 + 2], widen6(_mm512_srli_epi32(w2, 20)));
        _mm512_store_ps(out[b3 + 3], widen6(_mm512_srli_epi32(w2, 26)));
      }
    }
#else
    constexpr int fraction = Fmt == RJ_ARITH_MATRIX_FORMAT_FP6 ? 3 : 2;
    constexpr int bias = Fmt == RJ_ARITH_MATRIX_FORMAT_FP6 ? 1 : 3;
    const __m256i mask31 = _mm256_set1_epi32(31);
    const __m256i mask32 = _mm256_set1_epi32(32);
    const __m256i exp_bias = _mm256_set1_epi32((127 - bias) << 23);
    const __m256i sub_thresh = _mm256_set1_epi32(1 << fraction);
    const __m256 sub_scale = _mm256_set1_ps(Fmt == RJ_ARITH_MATRIX_FORMAT_FP6 ? 0x1p-3f : 0x1p-4f);
    auto widen6 = [&](__m256i code) {
      const __m256i mag = _mm256_and_si256(code, mask31);
      const __m256i normal = _mm256_add_epi32(_mm256_slli_epi32(mag, 23 - fraction), exp_bias);
      const __m256 small_v = _mm256_mul_ps(_mm256_cvtepi32_ps(mag), sub_scale);
      const __m256i is_small = _mm256_cmpgt_epi32(sub_thresh, mag);
      const __m256i val = _mm256_blendv_epi8(normal, _mm256_castps_si256(small_v), is_small);
      const __m256i sgn = _mm256_slli_epi32(_mm256_and_si256(code, mask32), 26);
      return _mm256_castsi256_ps(_mm256_or_si256(val, sgn));
    };
    for (int hw = 0; hw < 2; ++hw) {
      const int lane = 16 * hw;
      for (int g = 0; g < 4; ++g) {
        const int vb = 3 * g;
        const int b0 = InterleaveFour ? 32 * g + 4 * hw : (g / 2) * 64 + 32 * hw + (g % 2) * 16;
        constexpr int stride = InterleaveFour ? 8 : 4;
        const int b1 = b0 + stride, b2 = b0 + 2 * stride, b3 = b0 + 3 * stride;
        for (int off = 0; off < 16; off += 8) {
          const __m256i w0 =
              _mm256_loadu_si256(reinterpret_cast<const __m256i *>(input[vb] + lane + off));
          const __m256i w1 =
              _mm256_loadu_si256(reinterpret_cast<const __m256i *>(input[vb + 1] + lane + off));
          const __m256i w2 =
              _mm256_loadu_si256(reinterpret_cast<const __m256i *>(input[vb + 2] + lane + off));
          _mm256_store_ps(out[b0 + 0] + off, widen6(w0));
          _mm256_store_ps(out[b0 + 1] + off, widen6(_mm256_srli_epi32(w0, 6)));
          _mm256_store_ps(out[b0 + 2] + off, widen6(_mm256_srli_epi32(w0, 12)));
          _mm256_store_ps(out[b0 + 3] + off, widen6(_mm256_srli_epi32(w0, 18)));
          _mm256_store_ps(out[b1 + 0] + off, widen6(_mm256_srli_epi32(w0, 24)));
          _mm256_store_ps(out[b1 + 1] + off, widen6(_mm256_or_si256(_mm256_srli_epi32(w0, 30),
                                                                    _mm256_slli_epi32(w1, 2))));
          _mm256_store_ps(out[b1 + 2] + off, widen6(_mm256_srli_epi32(w1, 4)));
          _mm256_store_ps(out[b1 + 3] + off, widen6(_mm256_srli_epi32(w1, 10)));
          _mm256_store_ps(out[b2 + 0] + off, widen6(_mm256_srli_epi32(w1, 16)));
          _mm256_store_ps(out[b2 + 1] + off, widen6(_mm256_srli_epi32(w1, 22)));
          _mm256_store_ps(out[b2 + 2] + off, widen6(_mm256_or_si256(_mm256_srli_epi32(w1, 28),
                                                                    _mm256_slli_epi32(w2, 4))));
          _mm256_store_ps(out[b2 + 3] + off, widen6(_mm256_srli_epi32(w2, 2)));
          _mm256_store_ps(out[b3 + 0] + off, widen6(_mm256_srli_epi32(w2, 8)));
          _mm256_store_ps(out[b3 + 1] + off, widen6(_mm256_srli_epi32(w2, 14)));
          _mm256_store_ps(out[b3 + 2] + off, widen6(_mm256_srli_epi32(w2, 20)));
          _mm256_store_ps(out[b3 + 3] + off, widen6(_mm256_srli_epi32(w2, 26)));
        }
      }
    }
#endif
  };
  if (format == RJ_ARITH_MATRIX_FORMAT_FP6) {
    decode_6.template operator()<RJ_ARITH_MATRIX_FORMAT_FP6>();
  } else {
    decode_6.template operator()<RJ_ARITH_MATRIX_FORMAT_BF6>();
  }
}

// Fast 16-row (AVX-512) / 4x2 (AVX2) FMA reduction for general gfx1250 mixed/scaled WMMA.
template <int M, int Block>
void gfx1250_mixed_fast_kernel(uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, const uint32_t *const *c,
                               const uint32_t *const *scale_a = nullptr,
                               const uint32_t *const *scale_b = nullptr) {
  const int fa = M == 32 ? RJ_ARITH_MATRIX_FORMAT_FP4 : int(controls & 7);
  const int fb = M == 32 ? RJ_ARITH_MATRIX_FORMAT_FP4 : int((controls >> 3) & 7);
  if (fa == RJ_ARITH_MATRIX_FORMAT_FP4 && fb == RJ_ARITH_MATRIX_FORMAT_FP4) {
    gfx1250_fp4_exact_int_kernel<M, Block>(controls, d, a, b, c, scale_a, scale_b);
    return;
  }

  // Only M == 16 can reach here (M == 32 is always FP4xFP4).
  constexpr int groups = Block ? 128 / (Block ? Block : 1) : 1;
  constexpr int span = Block ? Block : 128;
  alignas(64) float left[128][16], right[128][16], result[16][16];
  if constexpr (Block == 0) {
    if (fa >= 2 && fb >= 2) {
      gfx1250_mixed_decode_16<true>(a, fa, left);
      gfx1250_mixed_decode_16<true>(b, fb, right);
    } else {
      gfx1250_mixed_decode_16<false>(a, fa, left);
      gfx1250_mixed_decode_16<false>(b, fb, right);
    }
  } else {
    gfx1250_mixed_decode_16<false>(a, fa, left);
    gfx1250_mixed_decode_16<false>(b, fb, right);
  }
  gfx1250_mixed_load_c<16>(c, controls, result);

  alignas(64) float sa[groups][16], sb[groups][16];
  if constexpr (Block != 0) {
    const int asf = int((controls >> 8) & 3);
    const int bsf = int((controls >> 10) & 3);
    const int a_lane = 16 * int((controls >> 12) & 1);
    const int b_lane = 16 * int((controls >> 13) & 1);
    for (int blk = 0; blk < groups; ++blk) {
      gfx1250_decode_scales_16(scale_a, a_lane, blk, asf, sa[blk]);
      gfx1250_decode_scales_16(scale_b, b_lane, blk, bsf, sb[blk]);
    }
  }

#if defined(RJ_ARITH_TARGET_X86_64_V4)
  if constexpr (Block == 0) {
    __m512 acc[16];
    RJ_ARITH_UNROLL(16)
    for (int r = 0; r < 16; ++r) {
      acc[r] = _mm512_load_ps(result[r]);
    }
    for (int k = 0; k < 128; ++k) {
      const __m512 bv = _mm512_load_ps(right[k]);
      RJ_ARITH_UNROLL(16)
      for (int r = 0; r < 16; ++r) {
        acc[r] = _mm512_fmadd_ps(_mm512_set1_ps(left[k][r]), bv, acc[r]);
      }
    }
    RJ_ARITH_UNROLL(16)
    for (int r = 0; r < 16; ++r) {
      _mm512_store_ps(result[r], acc[r]);
    }
  } else {
    // Keep all 16 block sums register-resident during the inner K loop (1 pass over B),
    // updating result[0..15] in L1 only at the 4 or 8 block boundaries.
    for (int blk = 0; blk < groups; ++blk) {
      __m512 sum[16];
      RJ_ARITH_UNROLL(16)
      for (int r = 0; r < 16; ++r) {
        sum[r] = _mm512_setzero_ps();
      }
      for (int k = blk * span; k < (blk + 1) * span; ++k) {
        const __m512 bv = _mm512_load_ps(right[k]);
        RJ_ARITH_UNROLL(16)
        for (int r = 0; r < 16; ++r) {
          sum[r] = _mm512_fmadd_ps(_mm512_set1_ps(left[k][r]), bv, sum[r]);
        }
      }
      const __m512 sb_v = _mm512_load_ps(sb[blk]);
      RJ_ARITH_UNROLL(16)
      for (int r = 0; r < 16; ++r) {
        const __m512 scaled =
            _mm512_mul_ps(_mm512_mul_ps(sum[r], _mm512_set1_ps(sa[blk][r])), sb_v);
        _mm512_store_ps(result[r], _mm512_add_ps(_mm512_load_ps(result[r]), scaled));
      }
    }
  }
#else
  // AVX2: 4 rows x 2 column halves (8 YMMs) share each A broadcast across both halves.
  for (int base = 0; base < 16; base += 4) {
    if constexpr (Block == 0) {
      __m256 acc[4][2];
      RJ_ARITH_UNROLL(4)
      for (int r = 0; r < 4; ++r) {
        acc[r][0] = _mm256_load_ps(result[base + r]);
        acc[r][1] = _mm256_load_ps(result[base + r] + 8);
      }
      for (int k = 0; k < 128; ++k) {
        const __m256 b0 = _mm256_load_ps(right[k]);
        const __m256 b1 = _mm256_load_ps(right[k] + 8);
        RJ_ARITH_UNROLL(4)
        for (int r = 0; r < 4; ++r) {
          const __m256 av = _mm256_broadcast_ss(&left[k][base + r]);
          acc[r][0] = _mm256_fmadd_ps(av, b0, acc[r][0]);
          acc[r][1] = _mm256_fmadd_ps(av, b1, acc[r][1]);
        }
      }
      RJ_ARITH_UNROLL(4)
      for (int r = 0; r < 4; ++r) {
        _mm256_store_ps(result[base + r], acc[r][0]);
        _mm256_store_ps(result[base + r] + 8, acc[r][1]);
      }
    } else {
      for (int blk = 0; blk < groups; ++blk) {
        __m256 sum[4][2];
        RJ_ARITH_UNROLL(4)
        for (int r = 0; r < 4; ++r) {
          sum[r][0] = _mm256_setzero_ps();
          sum[r][1] = _mm256_setzero_ps();
        }
        for (int k = blk * span; k < (blk + 1) * span; ++k) {
          const __m256 b0 = _mm256_load_ps(right[k]);
          const __m256 b1 = _mm256_load_ps(right[k] + 8);
          RJ_ARITH_UNROLL(4)
          for (int r = 0; r < 4; ++r) {
            const __m256 av = _mm256_broadcast_ss(&left[k][base + r]);
            sum[r][0] = _mm256_fmadd_ps(av, b0, sum[r][0]);
            sum[r][1] = _mm256_fmadd_ps(av, b1, sum[r][1]);
          }
        }
        const __m256 sb0 = _mm256_load_ps(sb[blk]);
        const __m256 sb1 = _mm256_load_ps(sb[blk] + 8);
        RJ_ARITH_UNROLL(4)
        for (int r = 0; r < 4; ++r) {
          const __m256 sav = _mm256_set1_ps(sa[blk][base + r]);
          const __m256 s0 = _mm256_mul_ps(_mm256_mul_ps(sum[r][0], sav), sb0);
          const __m256 s1 = _mm256_mul_ps(_mm256_mul_ps(sum[r][1], sav), sb1);
          _mm256_store_ps(result[base + r], _mm256_add_ps(_mm256_load_ps(result[base + r]), s0));
          _mm256_store_ps(result[base + r] + 8,
                          _mm256_add_ps(_mm256_load_ps(result[base + r] + 8), s1));
        }
      }
    }
  }
#endif
  gfx1250_mixed_store_d<16>(d, result);
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
