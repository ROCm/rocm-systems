// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 BF16 pair-dot (vdpbf16ps) backend for CDNA4 mixed and scaled f8f6f4 MFMA.
#include "common/compiler.h"
#include "common/x86_64/bf16_broadcast.h"
#include "mfma/backends.h"
#include "mfma/mixed.h"
#include "mfma/x86_64/bf16_dot.h"
#include "mfma/x86_64/fp8.h"
#include "rj_arith/detail/rj_arith_mfma.h"
#include <cstring>
#include <immintrin.h>

#if !defined(RJ_ARITH_TARGET_X86_64_V4_BF16)
#error "mfma_mixed_x86_64_v4_bf16.cpp requires AVX-512 BF16 (-mavx512bf16)"
#endif

namespace rocjitsu::arith {

namespace {

// Decode mixed f8/f6/f4 operand directly into K-paired BF16 [K/2][M] with halves swapped
// (even K in high 16 bits, odd K in low 16 bits) to match vdpbf16ps's high-half-first order.
template <int Format, int M, int K>
void decode_mixed_bf16_pairs(const uint32_t *const *input, uint32_t (&out)[K / 2][M]) {
  constexpr int groups = 64 / M;
  constexpr int block = Format <= RJ_ARITH_MATRIX_FORMAT_BF8 ? 16 : 32;
  for (int k = 0; k < K; k += block) {
    const int lane = (k / block % groups) * M;
    const int first = (k / (block * groups)) * block;
    for (int row = 0; row < M; row += 16) {
      const int first_lane = lane + row;
      if constexpr (Format <= RJ_ARITH_MATRIX_FORMAT_BF8) {
        constexpr bool Bf8 = (Format == RJ_ARITH_MATRIX_FORMAT_BF8);
        for (int j = 0; j < block; j += 4) {
          const int reg = (first + j) / 4;
          const __m512i words = _mm512_loadu_si512(input[reg] + first_lane);
          const __m512i p01 = mfma_fp8_to_bf16_pairs<Bf8, false>(words, 0);
          const __m512i p23 = mfma_fp8_to_bf16_pairs<Bf8, false>(words, 1);
          _mm512_store_si512(out[(k + j) / 2 + 0] + row, _mm512_rol_epi32(p01, 16));
          _mm512_store_si512(out[(k + j) / 2 + 1] + row, _mm512_rol_epi32(p23, 16));
        }
      } else if constexpr (Format == RJ_ARITH_MATRIX_FORMAT_FP6 ||
                           Format == RJ_ARITH_MATRIX_FORMAT_BF6) {
        alignas(64) static constexpr mixed_x86::SubbyteBf16Table<Format> table;
        const __m512i lut = _mm512_load_si512(table.bits);
        const int base_reg = (first * 6) / 32;
        __m512i cached[6];
        for (int r = 0; r < 6; ++r) {
          cached[r] = _mm512_loadu_si512(input[base_reg + r] + first_lane);
        }
        auto extract6 = [&](int j) {
          const int bit = (first + j) * 6;
          const int r = bit / 32 - base_reg;
          const int shift = bit % 32;
          __m512i code = _mm512_srli_epi32(cached[r], shift);
          if (shift > 26) {
            code = _mm512_or_si512(code, _mm512_slli_epi32(cached[r + 1], 32 - shift));
          }
          return code;
        };
        const __m512i m31 = _mm512_set1_epi32(31);
        const __m512i m32 = _mm512_set1_epi32(32);
        for (int j = 0; j < 32; j += 2) {
          const __m512i c0 = extract6(j);
          const __m512i c1 = extract6(j + 1);
          // Place even K (c0) in high 16-bit half, odd K (c1) in low 16-bit half.
          const __m512i idx = _mm512_or_si512(_mm512_and_si512(c1, m31),
                                              _mm512_slli_epi32(_mm512_and_si512(c0, m31), 16));
          const __m512i mag = _mm512_permutexvar_epi16(idx, lut);
          const __m512i signs = _mm512_or_si512(_mm512_slli_epi32(_mm512_and_si512(c1, m32), 10),
                                                _mm512_slli_epi32(_mm512_and_si512(c0, m32), 26));
          _mm512_store_si512(out[(k + j) / 2] + row, _mm512_or_si512(mag, signs));
        }
      } else {
        // FP4: 16 signed BF16 entries fit in one ZMM (duplicated to 32 uint16s for vpermw).
        alignas(64) static constexpr uint16_t fp4_bf16[32] = {
            0x0000, 0x3f00, 0x3f80, 0x3fc0, 0x4000, 0x4040, 0x4080, 0x40c0, 0x8000, 0xbf00, 0xbf80,
            0xbfc0, 0xc000, 0xc040, 0xc080, 0xc0c0, 0x0000, 0x3f00, 0x3f80, 0x3fc0, 0x4000, 0x4040,
            0x4080, 0x40c0, 0x8000, 0xbf00, 0xbf80, 0xbfc0, 0xc000, 0xc040, 0xc080, 0xc0c0,
        };
        const __m512i lut = _mm512_load_si512(fp4_bf16);
        const __m512i m15 = _mm512_set1_epi32(15);
        for (int reg_off = 0; reg_off < 4; ++reg_off) {
          const int reg = first / 8 + reg_off;
          const __m512i w = _mm512_loadu_si512(input[reg] + first_lane);
          RJ_ARITH_UNROLL(4)
          for (int p = 0; p < 4; ++p) {
            const __m512i c0 = _mm512_and_si512(_mm512_srli_epi32(w, 8 * p), m15);
            const __m512i c1 = _mm512_and_si512(_mm512_srli_epi32(w, 8 * p + 4), m15);
            const __m512i idx = _mm512_or_si512(c1, _mm512_slli_epi32(c0, 16));
            _mm512_store_si512(out[(k + 8 * reg_off + 2 * p) / 2] + row,
                               _mm512_permutexvar_epi16(idx, lut));
          }
        }
      }
    }
  }
}

template <int M, int K>
void decode_mixed_bf16_dispatch(const uint32_t *const *input, int format,
                                uint32_t (&out)[K / 2][M]) {
  switch (format) {
  case RJ_ARITH_MATRIX_FORMAT_FP8:
    return decode_mixed_bf16_pairs<RJ_ARITH_MATRIX_FORMAT_FP8, M, K>(input, out);
  case RJ_ARITH_MATRIX_FORMAT_BF8:
    return decode_mixed_bf16_pairs<RJ_ARITH_MATRIX_FORMAT_BF8, M, K>(input, out);
  case RJ_ARITH_MATRIX_FORMAT_FP6:
    return decode_mixed_bf16_pairs<RJ_ARITH_MATRIX_FORMAT_FP6, M, K>(input, out);
  case RJ_ARITH_MATRIX_FORMAT_BF6:
    return decode_mixed_bf16_pairs<RJ_ARITH_MATRIX_FORMAT_BF6, M, K>(input, out);
  default:
    return decode_mixed_bf16_pairs<RJ_ARITH_MATRIX_FORMAT_FP4, M, K>(input, out);
  }
}

} // namespace

// Unscaled mixed f8f6f4 MFMA via vdpbf16ps. Falls back to v4 only if C has a subnormal lane.
template <int M, int K>
bool mfma_mixed_x86_64_v4_bf16(uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, const uint32_t *const *c) {
  constexpr int registers = M * M / 64;
  constexpr int col_vecs = M / 16;
  constexpr int row_tile = M == 16 ? 16 : 8;
  if (__builtin_expect(mfma_c_has_subnormal<registers>(c), 0)) {
    return false;
  }
  alignas(64) uint32_t left[K / 2][M], right[K / 2][M];
  decode_mixed_bf16_dispatch<M, K>(a, int(controls & 7), left);
  decode_mixed_bf16_dispatch<M, K>(b, int((controls >> 3) & 7), right);
  const bool direct = mfma_direct_store_safe<registers>(d, c);
  alignas(64) float result[registers][64];
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
    RJ_ARITH_UNROLL(4)
    for (int kp = 0; kp < K / 2; ++kp) {
      __m512i b_vec[col_vecs];
      for (int p = 0; p < col_vecs; ++p) {
        b_vec[p] = _mm512_load_si512(right[kp] + 16 * p);
      }
      RJ_ARITH_UNROLL(16)
      for (int r = 0; r < row_tile; ++r) {
        const __m512i a_bcast = broadcast_bf16_pair(&left[kp][rb + r]);
        for (int p = 0; p < col_vecs; ++p) {
          acc[r][p] = _mm512_dpbf16_ps(acc[r][p], bit_cast<__m512bh>(b_vec[p]),
                                       bit_cast<__m512bh>(a_bcast));
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
  if (!direct) {
    for (int reg = 0; reg < registers; ++reg) {
      for (int lane = 0; lane < 64; lane += 16) {
        _mm512_storeu_ps(reinterpret_cast<float *>(d[reg] + lane),
                         _mm512_load_ps(result[reg] + lane));
      }
    }
  }
  return true;
}

// Block-scaled mixed f8f6f4 MFMA via vdpbf16ps. Each 32-element block dot starts at 0.0f and
// all finite f8/f6/f4 products are multiples of >= 2^-32, so vdpbf16ps never flushes any
// nonzero value; C is added afterwards with regular FP32 vaddps (handles subnormal C cleanly).
template <int M, int K>
void mfma_scaled_x86_64_v4_bf16(uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c,
                                const uint32_t *scale_a, const uint32_t *scale_b,
                                bool clamp_overflow) {
  constexpr int registers = M * M / 64;
  constexpr int col_vecs = M / 16;
  constexpr int row_tile = M == 16 ? 8 : 4;
  alignas(64) uint32_t left[K / 2][M], right[K / 2][M];
  decode_mixed_bf16_dispatch<M, K>(a, int(controls & 7), left);
  decode_mixed_bf16_dispatch<M, K>(b, int((controls >> 3) & 7), right);

  alignas(64) uint32_t ea[64], eb[64];
  const bool plain_scales = mfma_extract_e8m0_scales(controls, scale_a, scale_b, ea, eb);
  const uint32_t abs_mask = (controls & RJ_ARITH_MFMA_SCALE_ABS_C) ? 0x7fffffffU : 0xffffffffU;
  const uint32_t neg_mask = (controls & RJ_ARITH_MFMA_SCALE_NEG_C) ? 0x80000000U : 0U;
  const bool direct = mfma_direct_store_safe<registers>(d, c);
  alignas(64) float result[registers][64];
  const __m512i f32_abs = _mm512_set1_epi32(0x7fffffff);
  const __m512i f32_inf = _mm512_set1_epi32(0x7f800000);

  for (int rb = 0; rb < M; rb += row_tile) {
    __m512 acc[row_tile][col_vecs];
    __mmask16 clamp_mask[row_tile][col_vecs];
    for (int r = 0; r < row_tile; ++r) {
      const MfmaWord out = mfma_output<M, 1>(rb + r, 0, 0);
      for (int p = 0; p < col_vecs; ++p) {
        acc[r][p] = mfma_mixed_load_c(c[out.reg] + out.lane + 16 * p, abs_mask, neg_mask);
        clamp_mask[r][p] =
            clamp_overflow ? _mm512_cmplt_epu32_mask(
                                 _mm512_and_si512(_mm512_castps_si512(acc[r][p]), f32_abs), f32_inf)
                           : __mmask16(0);
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
      for (int kp = 16 * block; kp < 16 * (block + 1); ++kp) {
        __m512i b_vec[col_vecs];
        for (int p = 0; p < col_vecs; ++p) {
          b_vec[p] = _mm512_load_si512(right[kp] + 16 * p);
        }
        RJ_ARITH_UNROLL(8)
        for (int r = 0; r < row_tile; ++r) {
          const __m512i a_bcast = broadcast_bf16_pair(&left[kp][rb + r]);
          for (int p = 0; p < col_vecs; ++p) {
            sum[r][p] = _mm512_dpbf16_ps(sum[r][p], bit_cast<__m512bh>(b_vec[p]),
                                         bit_cast<__m512bh>(a_bcast));
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
            scaled = _mm512_mul_ps(sum[r][p], _mm512_castsi512_ps(_mm512_slli_epi32(exp, 23)));
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
          acc[r][p] = _mm512_castsi512_ps(
              _mm512_mask_sub_epi32(_mm512_castps_si512(acc[r][p]), clamp_mask[r][p] & inf_m,
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
  if (!direct) {
    for (int reg = 0; reg < registers; ++reg) {
      for (int lane = 0; lane < 64; lane += 16) {
        _mm512_storeu_ps(reinterpret_cast<float *>(d[reg] + lane),
                         _mm512_load_ps(result[reg] + lane));
      }
    }
  }
}

template bool mfma_mixed_x86_64_v4_bf16<16, 128>(uint64_t, uint32_t *const *,
                                                 const uint32_t *const *, const uint32_t *const *,
                                                 const uint32_t *const *);
template bool mfma_mixed_x86_64_v4_bf16<32, 64>(uint64_t, uint32_t *const *,
                                                const uint32_t *const *, const uint32_t *const *,
                                                const uint32_t *const *);
template void mfma_scaled_x86_64_v4_bf16<16, 128>(uint64_t, uint32_t *const *,
                                                  const uint32_t *const *, const uint32_t *const *,
                                                  const uint32_t *const *, const uint32_t *,
                                                  const uint32_t *, bool);
template void mfma_scaled_x86_64_v4_bf16<32, 64>(uint64_t, uint32_t *const *,
                                                 const uint32_t *const *, const uint32_t *const *,
                                                 const uint32_t *const *, const uint32_t *,
                                                 const uint32_t *, bool);

} // namespace rocjitsu::arith
