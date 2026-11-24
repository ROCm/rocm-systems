// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Inherited CDNA sparse byte layouts and integer accumulation, isolated by CPU ISA.
#pragma once
#include "common/compiler.h"
#include "mfma/integer.h"
#include "rj_arith/detail/rj_arith_wmma.h"
namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {
struct SmfmacByteLocation {
  int lane, nibble, element;
};

// Locate compressed A and its selector nibble in compact or wide SMFMAC operands.
template <int M, int K> SmfmacByteLocation smfmac_byte_location(int row, int q, int slot) {
  constexpr bool wide = M * K == 2048;
  if constexpr (!wide) {
    constexpr int half_groups = K / 8;
    return {((q % half_groups) / 2) * M + row, 2 * (q / half_groups) + q % 2,
            4 * (q / half_groups) + (2 * q + slot) % 4};
  } else if constexpr (M == 16) {
    return {(2 * (q / 16) + (q / 4) % 2) * M + row, 2 * ((q / 8) % 2) + 4 * ((q % 4) / 2) + q % 2,
            4 * (2 * ((q / 2) % 2) + (q / 8) % 2) + (2 * q + slot) % 4};
  } else {
    return {q / 8 * M + row, q % 2 + 2 * ((q / 4) % 2) + 4 * ((q / 2) % 2),
            4 * (2 * ((q / 2) % 2) + (q / 4) % 2) + (2 * q + slot) % 4};
  }
}

// Flatten the legacy dense B byte layout, independent of metadata selection.
template <int M, bool Wmma> int sparse_matrix_b_index(int col, int k) {
  int lane, element;
  if constexpr (Wmma) {
    // gfx1250 SWMMAC uses 32-element blocks, unlike dense WMMA's 16-element blocks.
    lane = col + 16 * ((k / 32) % 2);
    element = k % 32 + 32 * (k / 64);
  } else {
    lane = (M == 16 ? 16 * ((k % 32) / 8) : 32 * ((k % 16) / 8)) + col;
    element = 8 * (k / (M == 16 ? 32 : 16)) + k % 8;
  }
  return (element / 4 * (Wmma ? 32 : 64) + lane) * 4 + element % 4;
}

// Evaluate the tied accumulator without publishing any output until every input is consumed.
template <int M, int K, bool Wmma>
void sparse_matrix_i8_kernel(uint64_t controls, bool clamp, uint32_t *const *d,
                             const uint32_t *const *a, const uint32_t *const *b,
                             const uint32_t *const *index) {
  constexpr int lanes = Wmma ? 32 : 64;
  alignas(64) uint32_t by_k[K][M], result[M][M];
  const uint32_t a_sign = !Wmma || (controls & RJ_ARITH_WMMA_SIGNED_A) ? 0x80u : 0u;
  const uint32_t b_sign = !Wmma || (controls & RJ_ARITH_WMMA_SIGNED_B) ? 0x80u : 0u;
  // The same byte position supplies all columns. Decode B directly into the order used by the dot.
  for (int k = 0; k < K; ++k) {
    const int first = sparse_matrix_b_index<M, Wmma>(0, k);
    const uint32_t *words = b[first / (4 * lanes)] + (first / 4) % lanes;
    for (int col = 0; col < M; ++col) {
      uint32_t byte = (words[col] >> (8 * (first % 4))) & 255;
      uint32_t value = (byte ^ b_sign) - b_sign;
#if defined(__AVX512F__)
      // A selected B byte can occupy either half of a paired multiply-add.
      by_k[k][col] = (value & 0xffffu) * 0x10001u;
#else
      by_k[k][col] = value;
#endif
    }
  }
  for (int row = 0; row < M; ++row) {
    alignas(64) uint32_t accum[M] = {};
#if defined(__AVX512F__)
    __m512i sums[2][M / 16] = {};
#elif defined(__AVX2__)
    __m256i sums[M / 8] = {};
#endif
#if defined(__AVX512F__)
    RJ_ARITH_UNROLL(32)
#endif
    for (int group = 0; group < K / 4; ++group) {
      int selectors, ai;
      if constexpr (Wmma) {
        int lane = row + 16 * (group / 16);
        selectors = int(index[(group % 16) / 8][lane] >> (4 * (group % 8))) & 15;
        ai = matrix_i8_index<16, 64, 1, true>(row, 2 * group, 0, 0, true);
      } else {
        auto loc = smfmac_byte_location<M, K>(row, group, 0);
        int shift = (M * K < 2048 && (controls & 3) == 0) ? 16 * int(controls >> 3 & 1) : 0;
        selectors = int(index[0][loc.lane] >> (shift + 4 * loc.nibble)) & 15;
        ai = (loc.element / 4 * 64 + loc.lane) * 4 + loc.element % 4;
      }
      uint32_t pair = a[ai / (4 * lanes)][ai / 4 % lanes] >> (8 * (ai % 4));
      uint32_t a0 = ((pair & 255) ^ a_sign) - a_sign;
      uint32_t a1 = (((pair >> 8) & 255) ^ a_sign) - a_sign;
      int k0 = 4 * group + (selectors & 3), k1 = 4 * group + (selectors >> 2);
#if defined(__AVX512F__)
      auto left = _mm512_set1_epi32(int((a0 & 0xffffu) | (a1 << 16)));
      for (int col = 0; col < M; col += 16) {
        auto right = _mm512_mask_blend_epi16(0xaaaaaaaau, _mm512_load_si512(by_k[k0] + col),
                                             _mm512_load_si512(by_k[k1] + col));
        sums[group & 1][col / 16] =
            _mm512_add_epi32(sums[group & 1][col / 16], _mm512_madd_epi16(left, right));
      }
#elif defined(__AVX2__)
      for (int col = 0; col < M; col += 8) {
        auto x = _mm256_load_si256(reinterpret_cast<const __m256i *>(by_k[k0] + col));
        auto y = _mm256_load_si256(reinterpret_cast<const __m256i *>(by_k[k1] + col));
        auto products = _mm256_add_epi32(_mm256_mullo_epi32(_mm256_set1_epi32(a0), x),
                                         _mm256_mullo_epi32(_mm256_set1_epi32(a1), y));
        sums[col / 8] = _mm256_add_epi32(sums[col / 8], products);
      }
#else
      for (int col = 0; col < M; ++col) {
        accum[col] += a0 * by_k[k0][col] + a1 * by_k[k1][col];
      }
#endif
    }
#if defined(__AVX512F__)
    for (int col = 0; col < M; col += 16) {
      _mm512_store_si512(accum + col, _mm512_add_epi32(sums[0][col / 16], sums[1][col / 16]));
    }
#elif defined(__AVX2__)
    for (int col = 0; col < M; col += 8) {
      _mm256_store_si256(reinterpret_cast<__m256i *>(accum + col), sums[col / 8]);
    }
#endif
    for (int col = 0; col < M; ++col) {
      auto out = matrix_i8_output<M, 1, Wmma>(row, col, 0);
      uint32_t initial = d[out.reg][out.lane];
      if (clamp) {
        int64_t sum = int64_t(bit_cast<int32_t>(accum[col])) + bit_cast<int32_t>(initial);
        result[row][col] = sum > 0x7fffffffLL    ? 0x7fffffffU
                           : sum < -0x80000000LL ? 0x80000000U
                                                 : uint32_t(sum);
        continue;
      }
      result[row][col] = accum[col] + initial;
    }
  }
  for (int row = 0; row < M; ++row) {
    for (int col = 0; col < M; ++col) {
      auto out = matrix_i8_output<M, 1, Wmma>(row, col, 0);
      d[out.reg][out.lane] = result[row][col];
    }
  }
}
} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
