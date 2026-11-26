// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/compiler.h"
#include "mfma/layout.h"
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
      out[4 * reg + 2 * h] = bit_cast<MfmaFp4Pairs>(_mm256_unpacklo_epi8(lo, hi));
      out[4 * reg + 2 * h + 1] = bit_cast<MfmaFp4Pairs>(_mm256_unpackhi_epi8(lo, hi));
    }
  }
}

// Compute exact FP4 block dots before applying the original scale and C modifiers.
template <int M>
bool mfma_scaled_fp4(uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
                     const uint32_t *const *b, const uint32_t *const *c, const uint32_t *scale_a,
                     const uint32_t *scale_b, bool clamp_overflow) {
  constexpr uint64_t formats =
      RJ_ARITH_MATRIX_FORMAT_FP4 | (RJ_ARITH_MATRIX_FORMAT_FP4 << RJ_ARITH_MATRIX_FORMAT_B_SHIFT);
  if ((controls & 0x3f) != formats) {
    return false;
  }
  if ((_mm_getcsr() & _MM_ROUND_MASK) == _MM_ROUND_DOWN) {
    return false; // Exact cancellation can produce negative zero in this rounding mode.
  }
  static_assert(M == 16 || M == 32);
  constexpr int blocks = 64 / M;
  const int a_shift = 8 * int(controls >> RJ_ARITH_MFMA_SCALE_A_BYTE_SHIFT & 3);
  const int b_shift = 8 * int(controls >> RJ_ARITH_MFMA_SCALE_B_BYTE_SHIFT & 3);
  bool normal_scales = true;
  alignas(32) uint32_t ea[64], eb[64];
  for (int i = 0; i < 64; ++i) {
    ea[i] = scale_a[i] >> (a_shift) & 255;
    eb[i] = scale_b[i] >> (b_shift) & 255;
    normal_scales &= (ea[i] - 65u <= 120u) & (eb[i] - 65u <= 120u);
  }
  // Exponents 65..185 keep every nonzero scaled block (0.25..1152 in magnitude)
  // normal and finite. Other scales use ldexpf, including its range-error handling.
  if (!normal_scales) {
    return false;
  }
  alignas(32) uint32_t result[M][M];
  for (int row = 0; row < M; ++row) {
    const auto out = mfma_output<M, 1>(row, 0, 0);
    for (int col = 0; col < M; ++col) {
      uint32_t initial = c[out.reg][out.lane + col];
      if (controls & RJ_ARITH_MFMA_SCALE_ABS_C) {
        initial &= 0x7fffffffU;
      }
      if (controls & RJ_ARITH_MFMA_SCALE_NEG_C) {
        initial ^= 0x80000000U;
      }
      result[row][col] = initial;
    }
  }
  for (int block = 0; block < blocks; ++block) {
    MfmaFp4Pairs left[M / 16][16], right[M / 16][16];
    __m256i bias[M / 16];
    // A is unsigned for PMADDUBSW. Subtract the column's bias contribution once.
    auto values = _mm256_broadcastsi128_si256(
        _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12));
    auto offset = _mm256_set1_epi8(12);
    for (int part = 0; part < M / 16; ++part) {
      int lane = block * M + part * 16;
      mfma_fp4_transpose(a, lane, _mm256_add_epi8(values, offset), left[part]);
      mfma_fp4_transpose(b, lane, values, right[part]);
      bias[part] = _mm256_setzero_si256();
      for (const auto &pair : right[part]) {
        bias[part] =
            _mm256_sub_epi16(bias[part], _mm256_maddubs_epi16(offset, bit_cast<__m256i>(pair)));
      }
    }
    for (int row = 0; row < M; ++row) {
      for (int part = 0; part < M / 16; ++part) {
        auto sum = bias[part];
        // With 32 products bounded by 24 * 12 before bias correction, int16 is exact.
        RJ_ARITH_UNROLL(16)
        for (int pair = 0; pair < 16; ++pair) {
          auto lhs = _mm256_set1_epi16(int16_t(left[row / 16][pair].lane[row % 16]));
          sum = _mm256_add_epi16(sum,
                                 _mm256_maddubs_epi16(lhs, bit_cast<__m256i>(right[part][pair])));
        }
        for (int half = 0; half < 2; ++half) {
          int col = part * 16 + half * 8;
          auto dots = half ? _mm256_extracti128_si256(sum, 1) : _mm256_castsi256_si128(sum);
          // The scale bias is 127 and the integer dot is four times the FP4 dot.
          auto exponents = _mm256_add_epi32(
              _mm256_set1_epi32(int(ea[block * M + row]) - 129),
              _mm256_load_si256(reinterpret_cast<const __m256i *>(eb + block * M + col)));
          auto factor = _mm256_castsi256_ps(_mm256_slli_epi32(exponents, 23));
          auto scaled = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(dots)), factor);
          auto *p = reinterpret_cast<__m256i *>(result[row] + col);
          auto accumulated = _mm256_add_ps(_mm256_castsi256_ps(_mm256_load_si256(p)), scaled);
          _mm256_store_si256(p, _mm256_castps_si256(accumulated));
        }
      }
    }
  }
  if (clamp_overflow) {
    for (int row = 0; row < M; ++row) {
      const auto out = mfma_output<M, 1>(row, 0, 0);
      for (int col = 0; col < M; ++col) {
        if (__builtin_isfinite(as_float(c[out.reg][out.lane + col])) &&
            __builtin_isinf(as_float(result[row][col]))) {
          --result[row][col];
        }
      }
    }
  }
  for (int row = 0; row < M; ++row) {
    const auto out = mfma_output<M, 1>(row, 0, 0);
    std::memcpy(d[out.reg] + out.lane, result[row], M * sizeof(uint32_t));
  }
  return true;
}
} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
