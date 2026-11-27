// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 VNNI (Zen4+) exact integer FP4xFP4 backend for gfx1250 mixed and scaled WMMA.
// Maps signed FP4 nibbles directly in-place via vpshufb into signed int8 A (2*val in [-12, 12])
// and biased unsigned uint8 B (2*val + 12 in [0, 24]), accumulating exact 4*dot(A, B) with
// vpdpbusd (EVEX {1to16} broadcast folded into the signed A operand) and subtracting the
// per-row bias 12 * sum_k a_s8[k]. Scales and C are applied in exact FP32.
#include "common/compiler.h"
#include "rj_arith/detail/rj_arith_wmma.h"
#include "wmma/gfx1250_backends.h"
#include "wmma/x86_64/gfx1250_mixed_fast.h"
#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {
namespace {
using RJ_ARITH_IMPL_NAMESPACE::gfx1250_decode_scales_16;
using RJ_ARITH_IMPL_NAMESPACE::gfx1250_mixed_load_c;
using RJ_ARITH_IMPL_NAMESPACE::gfx1250_mixed_store_d;
} // namespace

template <int M, int Block>
void wmma_gfx1250_fp4_avx512vnni(uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c,
                                 const uint32_t *const *scale_a, const uint32_t *const *scale_b) {
  constexpr int groups = Block ? 128 / Block : 1;
  constexpr int vgprs_per_block = Block == 32 ? 4 : Block == 16 ? 2 : 8;
  constexpr int halfwaves_per_block = Block ? 1 : 2;
  constexpr int steps_per_block = vgprs_per_block * halfwaves_per_block * 2;

  const __m128i lut_a128 = _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
  const __m128i lut_b128 = _mm_setr_epi8(12, 13, 14, 15, 16, 18, 20, 24, 12, 11, 10, 9, 8, 6, 4, 0);
  const __m512i lut_a = _mm512_broadcast_i32x4(lut_a128);
  const __m512i lut_b = _mm512_broadcast_i32x4(lut_b128);
  const __m512i mask_0f = _mm512_set1_epi8(0x0f);
  const __m512i bias12 = _mm512_set1_epi8(12);
  const __m512 unit = _mm512_set1_ps(0.25f);

  if constexpr (M == 16 && Block == 0) {
    // Zero-stack-result unscaled 16x16x128 FP4xFP4: read A, B, C completely before storing D.
    alignas(64) uint32_t b_u8[32][16];
    alignas(64) uint32_t a_s8[32][16];
    alignas(64) int32_t a_bias_arr[16];
    __m512i a_bias0 = _mm512_setzero_si512();
    __m512i a_bias1 = _mm512_setzero_si512();
    int step = 0;
    for (int hw = 0; hw < 2; ++hw) {
      const int lane = 16 * hw;
      for (int v = 0; v < 8; ++v) {
        const __m512i bw = _mm512_loadu_si512(b[v] + lane);
        _mm512_store_si512(b_u8[step], _mm512_shuffle_epi8(lut_b, _mm512_and_si512(bw, mask_0f)));
        _mm512_store_si512(
            b_u8[step + 1],
            _mm512_shuffle_epi8(lut_b, _mm512_and_si512(_mm512_srli_epi16(bw, 4), mask_0f)));
        const __m512i aw = _mm512_loadu_si512(a[v] + lane);
        const __m512i alo = _mm512_shuffle_epi8(lut_a, _mm512_and_si512(aw, mask_0f));
        const __m512i ahi =
            _mm512_shuffle_epi8(lut_a, _mm512_and_si512(_mm512_srli_epi16(aw, 4), mask_0f));
        _mm512_store_si512(a_s8[step], alo);
        _mm512_store_si512(a_s8[step + 1], ahi);
        a_bias0 = _mm512_dpbusd_epi32(a_bias0, bias12, alo);
        a_bias1 = _mm512_dpbusd_epi32(a_bias1, bias12, ahi);
        step += 2;
      }
    }
    _mm512_store_si512(
        a_bias_arr, _mm512_sub_epi32(_mm512_setzero_si512(), _mm512_add_epi32(a_bias0, a_bias1)));
    // Keep a_s8 and a_bias_arr in L1 so vpdpbusd folds {1to16} broadcasts directly from memory.
    asm volatile("" : "+m"(a_s8), "+m"(a_bias_arr));

    __m512i acc[16];
    RJ_ARITH_UNROLL(16)
    for (int r = 0; r < 16; ++r) {
      acc[r] = _mm512_set1_epi32(a_bias_arr[r]);
    }
    for (int s = 0; s < 32; ++s) {
      const __m512i bv = _mm512_load_si512(b_u8[s]);
      RJ_ARITH_UNROLL(16)
      for (int r = 0; r < 16; ++r) {
        acc[r] = _mm512_dpbusd_epi32(acc[r], bv, _mm512_set1_epi32(int(a_s8[s][r])));
      }
    }

    const __m512i kv =
        _mm512_set1_epi32(int((controls & RJ_ARITH_WMMA_MIXED_ABS_C) ? 0x7fffffffU : 0xffffffffU));
    const __m512i nv =
        _mm512_set1_epi32(int((controls & RJ_ARITH_WMMA_MIXED_NEG_C) ? 0x80000000U : 0U));
    __m512 out[16];
    RJ_ARITH_UNROLL(8)
    for (int r = 0; r < 8; ++r) {
      const __m512 c_lo =
          _mm512_castsi512_ps(_mm512_xor_si512(_mm512_and_si512(_mm512_loadu_si512(c[r]), kv), nv));
      const __m512 c_hi = _mm512_castsi512_ps(
          _mm512_xor_si512(_mm512_and_si512(_mm512_loadu_si512(c[r] + 16), kv), nv));
      out[r] = _mm512_add_ps(c_lo, _mm512_mul_ps(_mm512_cvtepi32_ps(acc[r]), unit));
      out[8 + r] = _mm512_add_ps(c_hi, _mm512_mul_ps(_mm512_cvtepi32_ps(acc[8 + r]), unit));
    }
    RJ_ARITH_UNROLL(8)
    for (int r = 0; r < 8; ++r) {
      _mm512_storeu_ps(d[r], out[r]);
      _mm512_storeu_ps(d[r] + 16, out[8 + r]);
    }
    return;
  }

  alignas(64) float result[M][16];
  gfx1250_mixed_load_c<M>(c, controls, result);

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

  for (int blk = 0; blk < groups; ++blk) {
    const int hw0 = Block == 32 ? (blk % 2) : Block == 16 ? ((blk / 2) % 2) : 0;
    const int vbase = Block == 32 ? 4 * (blk / 2) : Block == 16 ? 4 * (blk / 4) + 2 * (blk % 2) : 0;

    alignas(64) uint32_t b_u8[steps_per_block][16];
    int step = 0;
    for (int hw = 0; hw < halfwaves_per_block; ++hw) {
      const int lane = 16 * (hw0 + hw);
      for (int v = 0; v < vgprs_per_block; ++v) {
        const __m512i bw = _mm512_loadu_si512(b[vbase + v] + lane);
        _mm512_store_si512(b_u8[step++], _mm512_shuffle_epi8(lut_b, _mm512_and_si512(bw, mask_0f)));
        _mm512_store_si512(
            b_u8[step++],
            _mm512_shuffle_epi8(lut_b, _mm512_and_si512(_mm512_srli_epi16(bw, 4), mask_0f)));
      }
    }

    const __m512 sbv = Block ? _mm512_load_ps(sb[blk]) : _mm512_setzero_ps();
    for (int mh = 0; mh < M / 16; ++mh) {
      const uint32_t *const *a_bank = a + 8 * mh;
      alignas(64) uint32_t a_s8[steps_per_block][16];
      alignas(64) int32_t a_bias_arr[16];
      __m512i a_bias0 = _mm512_setzero_si512();
      __m512i a_bias1 = _mm512_setzero_si512();
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
          a_bias0 = _mm512_dpbusd_epi32(a_bias0, bias12, alo);
          a_bias1 = _mm512_dpbusd_epi32(a_bias1, bias12, ahi);
          step += 2;
        }
      }
      _mm512_store_si512(
          a_bias_arr, _mm512_sub_epi32(_mm512_setzero_si512(), _mm512_add_epi32(a_bias0, a_bias1)));
      // Keep a_s8 and a_bias_arr in L1 so vpdpbusd folds {1to16} broadcasts instead of vpermd.
      asm volatile("" : "+m"(a_s8), "+m"(a_bias_arr));

      __m512i acc[16];
      RJ_ARITH_UNROLL(16)
      for (int r = 0; r < 16; ++r) {
        acc[r] = _mm512_set1_epi32(a_bias_arr[r]);
      }
      for (int s = 0; s < steps_per_block; ++s) {
        const __m512i bv = _mm512_load_si512(b_u8[s]);
        RJ_ARITH_UNROLL(16)
        for (int r = 0; r < 16; ++r) {
          acc[r] = _mm512_dpbusd_epi32(acc[r], bv, _mm512_set1_epi32(int(a_s8[s][r])));
        }
      }

      RJ_ARITH_UNROLL(16)
      for (int r = 0; r < 16; ++r) {
        __m512 partial = _mm512_mul_ps(_mm512_cvtepi32_ps(acc[r]), unit);
        if constexpr (Block != 0) {
          partial =
              _mm512_mul_ps(_mm512_mul_ps(partial, _mm512_set1_ps(sa[blk][16 * mh + r])), sbv);
        }
        float *row_ptr = result[16 * mh + r];
        _mm512_store_ps(row_ptr, _mm512_add_ps(_mm512_load_ps(row_ptr), partial));
      }
    }
  }
  gfx1250_mixed_store_d<M>(d, result);
}

template void wmma_gfx1250_fp4_avx512vnni<16, 0>(uint64_t, uint32_t *const *,
                                                 const uint32_t *const *, const uint32_t *const *,
                                                 const uint32_t *const *, const uint32_t *const *,
                                                 const uint32_t *const *);
template void wmma_gfx1250_fp4_avx512vnni<32, 0>(uint64_t, uint32_t *const *,
                                                 const uint32_t *const *, const uint32_t *const *,
                                                 const uint32_t *const *, const uint32_t *const *,
                                                 const uint32_t *const *);
template void wmma_gfx1250_fp4_avx512vnni<16, 16>(uint64_t, uint32_t *const *,
                                                  const uint32_t *const *, const uint32_t *const *,
                                                  const uint32_t *const *, const uint32_t *const *,
                                                  const uint32_t *const *);
template void wmma_gfx1250_fp4_avx512vnni<16, 32>(uint64_t, uint32_t *const *,
                                                  const uint32_t *const *, const uint32_t *const *,
                                                  const uint32_t *const *, const uint32_t *const *,
                                                  const uint32_t *const *);
template void wmma_gfx1250_fp4_avx512vnni<32, 16>(uint64_t, uint32_t *const *,
                                                  const uint32_t *const *, const uint32_t *const *,
                                                  const uint32_t *const *, const uint32_t *const *,
                                                  const uint32_t *const *);
template void wmma_gfx1250_fp4_avx512vnni<32, 32>(uint64_t, uint32_t *const *,
                                                  const uint32_t *const *, const uint32_t *const *,
                                                  const uint32_t *const *, const uint32_t *const *,
                                                  const uint32_t *const *);
} // namespace rocjitsu::arith
