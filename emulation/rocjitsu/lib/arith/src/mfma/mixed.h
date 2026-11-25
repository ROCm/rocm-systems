// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// gfx950 mixed-format MFMA staging and reduction, isolated by CPU ISA.
#pragma once

#include "mfma/layout.h"
#include "mfma/mixed_formats.h"
#include "rj_arith/detail/rj_arith_mfma.h"
#include <cstring>
#if defined(__AVX2__)
#include "mfma/x86_64/mixed_decode.h"
#include "mfma/x86_64/reduction.h"
#endif

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

#if defined(__AVX512F__)
// Scale a 16-lane FP32 block dot by 2^(a_exp + eb - 254), returning canonical qNaN when either
// E8M0 scale byte is 255.
inline __m512 mfma_scale_e8m0(__m512 sum, int a_exp, __m512i eb_v) {
  const auto v255 = _mm512_set1_epi32(255);
  const auto qnan = _mm512_castsi512_ps(_mm512_set1_epi32(0x7fc00000));
  const __mmask16 valid = a_exp == 255 ? __mmask16{0} : _mm512_cmpneq_epi32_mask(eb_v, v255);
  const auto exp_i = _mm512_add_epi32(_mm512_set1_epi32(a_exp - 254), eb_v);
  return _mm512_mask_scalef_ps(qnan, valid, sum, _mm512_cvtepi32_ps(exp_i));
}
#elif defined(__AVX2__)
// Scale an 8-lane FP32 block dot by 2^(a_exp + eb - 254) using two normal power-of-two multiplies.
// Every nonzero finite F8/F6/F4 32-element dot has unbiased exponent in [-32, +36], so splitting
// s = a_exp + eb into h1 = max(s >> 1, 1) and h2 = max(s - h1, 1) never double-rounds and matches
// ldexpf across all scales [0..254], special values, and IEEE rounding modes.
inline __m256 mfma_scale_e8m0(__m256 sum, int a_exp, __m256i eb_v) {
  const auto v255 = _mm256_set1_epi32(255);
  const auto one = _mm256_set1_epi32(1);
  const auto qnan = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fc00000));
  const auto ea_v = _mm256_set1_epi32(a_exp);
  const auto s = _mm256_add_epi32(ea_v, eb_v);
  const auto h1 = _mm256_max_epi32(_mm256_srli_epi32(s, 1), one);
  const auto h2 = _mm256_max_epi32(_mm256_sub_epi32(s, h1), one);
  const auto f1 = _mm256_castsi256_ps(_mm256_slli_epi32(h1, 23));
  const auto f2 = _mm256_castsi256_ps(_mm256_slli_epi32(h2, 23));
  const auto scaled = _mm256_mul_ps(_mm256_mul_ps(sum, f1), f2);
  const auto nan_mask = _mm256_cmpeq_epi32(_mm256_max_epi32(ea_v, eb_v), v255);
  return _mm256_blendv_ps(scaled, qnan, _mm256_castsi256_ps(nan_mask));
}
#else
inline float mfma_scale_e8m0_scalar(float sum, int a_exp, int b_exp) {
  if (a_exp == 255 || b_exp == 255) {
    return as_float(0x7fc00000U);
  }
  const int s = a_exp + b_exp;
  const int h1 = s > 2 ? (s >> 1) : 1;
  const int h2 = s - h1 > 1 ? (s - h1) : 1;
  return (sum * as_float(uint32_t(h1) << 23)) * as_float(uint32_t(h2) << 23);
}
#endif

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

#if defined(__AVX2__)
template <int M, int K, bool ClampOverflow>
inline void mfma_mixed_scaled_simd(uint64_t controls, const float left[K][M],
                                   const float right[K][M], const uint32_t *const *c,
                                   const uint32_t ea[64], const uint32_t eb[64],
                                   float result[M][M]) {
  static_assert(M % 4 == 0 && M % kMfmaWidth == 0);
  const uint32_t c_and = (controls & RJ_ARITH_MFMA_SCALE_ABS_C) ? 0x7fffffffU : 0xffffffffU;
  const uint32_t c_xor = (controls & RJ_ARITH_MFMA_SCALE_NEG_C) ? 0x80000000U : 0u;
#if defined(__AVX512F__)
  const auto v_and = _mm512_set1_epi32(int(c_and));
  const auto v_xor = _mm512_set1_epi32(int(c_xor));
  const auto exp_mask = _mm512_set1_epi32(0x7f800000);
  const auto abs_mask = _mm512_set1_epi32(0x7fffffff);
  const auto one = _mm512_set1_epi32(1);
#else
  const auto v_and = _mm256_set1_epi32(int(c_and));
  const auto v_xor = _mm256_set1_epi32(int(c_xor));
  const auto exp_mask = _mm256_set1_epi32(0x7f800000);
  const auto abs_mask = _mm256_set1_epi32(0x7fffffff);
#endif
  for (int col = 0; col < M; col += kMfmaWidth) {
    for (int row = 0; row < M; row += 4) {
#if defined(__AVX512F__)
      __m512 acc[4];
      __mmask16 clamp_mask[4];
      RJ_ARITH_UNROLL(4)
      for (int r = 0; r < 4; ++r) {
        const auto out = mfma_output<M, 1>(row + r, col, 0);
        auto bits = _mm512_loadu_si512(c[out.reg] + out.lane);
        bits = _mm512_xor_si512(_mm512_and_si512(bits, v_and), v_xor);
        if constexpr (ClampOverflow) {
          clamp_mask[r] = _mm512_cmpneq_epi32_mask(_mm512_and_si512(bits, exp_mask), exp_mask);
        }
        acc[r] = _mm512_castsi512_ps(bits);
      }
      for (int block = 0; block < K / 32; ++block) {
        __m512 sum[4] = {_mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(),
                         _mm512_setzero_ps()};
        for (int k = 32 * block; k < 32 * (block + 1); ++k) {
          const auto rhs = _mm512_load_ps(right[k] + col);
          RJ_ARITH_UNROLL(4)
          for (int r = 0; r < 4; ++r) {
            sum[r] = _mm512_fmadd_ps(_mm512_set1_ps(left[k][row + r]), rhs, sum[r]);
          }
        }
        const auto eb_v = _mm512_load_si512(eb + M * block + col);
        RJ_ARITH_UNROLL(4)
        for (int r = 0; r < 4; ++r) {
          if constexpr (ClampOverflow) {
            clamp_mask[r] &= _mm512_cmpneq_epi32_mask(
                _mm512_and_si512(_mm512_castps_si512(sum[r]), exp_mask), exp_mask);
          }
          acc[r] =
              _mm512_add_ps(acc[r], mfma_scale_e8m0(sum[r], int(ea[M * block + row + r]), eb_v));
        }
      }
      RJ_ARITH_UNROLL(4)
      for (int r = 0; r < 4; ++r) {
        if constexpr (ClampOverflow) {
          auto res_i = _mm512_castps_si512(acc[r]);
          const auto inf_mask = _mm512_mask_cmpeq_epi32_mask(
              clamp_mask[r], _mm512_and_si512(res_i, abs_mask), exp_mask);
          acc[r] = _mm512_castsi512_ps(_mm512_mask_sub_epi32(res_i, inf_mask, res_i, one));
        }
        _mm512_store_ps(result[row + r] + col, acc[r]);
      }
#else
      __m256i nonfinite[4];
      RJ_ARITH_UNROLL(4)
      for (int r = 0; r < 4; ++r) {
        const auto out = mfma_output<M, 1>(row + r, col, 0);
        auto bits = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(c[out.reg] + out.lane));
        bits = _mm256_xor_si256(_mm256_and_si256(bits, v_and), v_xor);
        if constexpr (ClampOverflow) {
          nonfinite[r] = _mm256_cmpeq_epi32(_mm256_and_si256(bits, exp_mask), exp_mask);
        }
        _mm256_store_ps(result[row + r] + col, _mm256_castsi256_ps(bits));
      }
      for (int block = 0; block < K / 32; ++block) {
        __m256 sum[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(),
                         _mm256_setzero_ps()};
        for (int k = 32 * block; k < 32 * (block + 1); ++k) {
          const auto rhs = _mm256_load_ps(right[k] + col);
          RJ_ARITH_UNROLL(4)
          for (int r = 0; r < 4; ++r) {
            sum[r] = _mm256_fmadd_ps(_mm256_set1_ps(left[k][row + r]), rhs, sum[r]);
          }
        }
        const auto eb_v =
            _mm256_load_si256(reinterpret_cast<const __m256i *>(eb + M * block + col));
        RJ_ARITH_UNROLL(4)
        for (int r = 0; r < 4; ++r) {
          if constexpr (ClampOverflow) {
            nonfinite[r] = _mm256_or_si256(
                nonfinite[r],
                _mm256_cmpeq_epi32(_mm256_and_si256(_mm256_castps_si256(sum[r]), exp_mask),
                                   exp_mask));
          }
          const auto scaled = mfma_scale_e8m0(sum[r], int(ea[M * block + row + r]), eb_v);
          const auto acc = _mm256_add_ps(_mm256_load_ps(result[row + r] + col), scaled);
          _mm256_store_ps(result[row + r] + col, acc);
        }
      }
      if constexpr (ClampOverflow) {
        RJ_ARITH_UNROLL(4)
        for (int r = 0; r < 4; ++r) {
          auto res_i = _mm256_castps_si256(_mm256_load_ps(result[row + r] + col));
          const auto is_inf = _mm256_cmpeq_epi32(_mm256_and_si256(res_i, abs_mask), exp_mask);
          const auto do_clamp = _mm256_andnot_si256(nonfinite[r], is_inf);
          _mm256_store_ps(result[row + r] + col,
                          _mm256_castsi256_ps(_mm256_add_epi32(res_i, do_clamp)));
        }
      }
#endif
    }
  }
}
#endif

// Stage every output before stores; preserve inherited reductions and optional block scaling.
template <int M, int K, bool Scaled = false>
void mfma_mixed_kernel(uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
                       const uint32_t *const *b, const uint32_t *const *c,
                       const uint32_t *scale_a = nullptr, const uint32_t *scale_b = nullptr,
                       bool clamp_overflow = false) {
  alignas(64) float result[M][M];
  alignas(64) uint32_t ea[64], eb[64];
  if constexpr (Scaled) {
    const int a_shift = 8 * int((controls >> 6) & 3);
    const int b_shift = 8 * int((controls >> 8) & 3);
    for (int i = 0; i < 64; ++i) {
      ea[i] = (scale_a[i] >> a_shift) & 255u;
      eb[i] = (scale_b[i] >> b_shift) & 255u;
    }
  }
#if defined(__AVX2__)
  alignas(64) float left[K][M], right[K][M];
  mixed_x86::decode<64 / M, M>(a, int(controls & 7), false, left);
  mixed_x86::decode<64 / M, M>(b, int(controls >> 3 & 7), false, right);
  if constexpr (Scaled) {
    if (clamp_overflow) {
      mfma_mixed_scaled_simd<M, K, true>(controls, left, right, c, ea, eb, result);
    } else {
      mfma_mixed_scaled_simd<M, K, false>(controls, left, right, c, ea, eb, result);
    }
  } else {
    for (int col = 0; col < M; col += kMfmaWidth) {
      for (int row = 0; row < M; row += 4) {
        MfmaNative acc[4];
        RJ_ARITH_UNROLL(4)
        for (int r = 0; r < 4; ++r) {
          const auto out = mfma_output<M, 1>(row + r, col, 0);
          alignas(64) float tmp[kMfmaWidth];
          std::memcpy(tmp, c[out.reg] + out.lane, sizeof(tmp));
          acc[r] = mfma_load(tmp);
        }
        for (int k = 0; k < K; ++k) {
          const auto rhs = mfma_load(right[k] + col);
          RJ_ARITH_UNROLL(4)
          for (int r = 0; r < 4; ++r) {
            acc[r] = mfma_multiply_add(mfma_broadcast(left[k][row + r]), rhs, acc[r]);
          }
        }
        RJ_ARITH_UNROLL(4)
        for (int r = 0; r < 4; ++r) {
          mfma_store(result[row + r] + col, acc[r]);
        }
      }
    }
  }
#else
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
    const auto out = mfma_output<M, 1>(row, 0, 0);
    for (int col = 0; col < M; ++col) {
      uint32_t initial = c[out.reg][out.lane + col];
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
        const int a_exp = int(ea[M * block + row]);
        for (int col = 0; col < M; ++col) {
          const int b_exp = int(eb[M * block + col]);
          const float scaled = mfma_scale_e8m0_scalar(sums[col], a_exp, b_exp);
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
#endif
  for (int row = 0; row < M; ++row) {
    const auto out = mfma_output<M, 1>(row, 0, 0);
    std::memcpy(d[out.reg] + out.lane, result[row], M * sizeof(uint32_t));
  }
}
} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
