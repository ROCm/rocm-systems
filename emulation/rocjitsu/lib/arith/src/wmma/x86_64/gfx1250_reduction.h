// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Interleaved gfx1250 WMMA reductions shared by half and byte input formats.
// Half and byte operands store each K slice in the lane order of a Wave32 VGPR.
#pragma once
#include "common/compiler.h"
#include "common/internal.h"
#include <immintrin.h>
namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {
// Row-at-a-time FMA reduction used by the K=4 FP32 kernel (which stages A by row).
template <int K, bool DotFirst, bool KMajor = false>
inline void gfx1250_reduce_rows(const float left[][KMajor ? 16 : K], const float columns[K][16],
                                float result[16][16]) {
  auto value = [&](int row, int k) {
    if constexpr (KMajor) {
      return left[k][row];
    } else {
      return left[row][k];
    }
  };
  for (int row = 0; row < 16; ++row) {
#if defined(RJ_ARITH_TARGET_X86_64_V4)
    auto initial = _mm512_load_ps(result[row]);
    auto acc = DotFirst ? _mm512_setzero_ps() : initial;
    RJ_ARITH_UNROLL(128)
    for (int k = 0; k < K; ++k) {
      acc = _mm512_fmadd_ps(_mm512_set1_ps(value(row, k)), _mm512_load_ps(columns[k]), acc);
    }
    if constexpr (DotFirst) {
      acc = _mm512_add_ps(acc, initial);
    }
    _mm512_store_ps(result[row], acc);
#else
    for (int col = 0; col < 16; col += 8) {
      auto initial = _mm256_load_ps(&result[row][col]);
      auto acc = DotFirst ? _mm256_setzero_ps() : initial;
      RJ_ARITH_UNROLL(128)
      for (int k = 0; k < K; ++k) {
        acc = _mm256_fmadd_ps(_mm256_set1_ps(value(row, k)), _mm256_load_ps(&columns[k][col]), acc);
      }
      if constexpr (DotFirst) {
        acc = _mm256_add_ps(acc, initial);
      }
      _mm256_store_ps(&result[row][col], acc);
    }
#endif
  }
}

// Accumulate independent rows together, reusing each vector from B. Keep one
// copy of this loop per K and C placement across all input and output formats.
template <int K, bool DotFirst>
[[gnu::noinline]] inline void gfx1250_reduce_interleaved(const float left[K][16],
                                                         const float right[K][16],
                                                         float result[16][16]) {
#if defined(RJ_ARITH_TARGET_X86_64_V4)
  __m512 acc[16];
  for (int row = 0; row < 16; ++row) {
    acc[row] = DotFirst ? _mm512_setzero_ps() : _mm512_load_ps(result[row]);
  }
  for (int k = 0; k < K; ++k) {
    const auto column = _mm512_load_ps(right[k]);
    RJ_ARITH_UNROLL(16)
    for (int row = 0; row < 16; ++row) {
      acc[row] = _mm512_fmadd_ps(_mm512_set1_ps(left[k][row]), column, acc[row]);
    }
  }
  for (int row = 0; row < 16; ++row) {
    if constexpr (DotFirst) {
      acc[row] = _mm512_add_ps(acc[row], _mm512_load_ps(result[row]));
    }
    _mm512_store_ps(result[row], acc[row]);
  }
#else
  // Sixteen registers hold a four-row block, both column halves and the row broadcasts.
  for (int base = 0; base < 16; base += 4) {
    __m256 acc[4][2];
    for (int row = 0; row < 4; ++row) {
      for (int half = 0; half < 2; ++half) {
        acc[row][half] =
            DotFirst ? _mm256_setzero_ps() : _mm256_load_ps(&result[base + row][8 * half]);
      }
    }
    for (int k = 0; k < K; ++k) {
      const auto low = _mm256_load_ps(&right[k][0]);
      const auto high = _mm256_load_ps(&right[k][8]);
      RJ_ARITH_UNROLL(4)
      for (int row = 0; row < 4; ++row) {
        const auto scale = _mm256_broadcast_ss(&left[k][base + row]);
        acc[row][0] = _mm256_fmadd_ps(scale, low, acc[row][0]);
        acc[row][1] = _mm256_fmadd_ps(scale, high, acc[row][1]);
      }
    }
    for (int row = 0; row < 4; ++row) {
      for (int half = 0; half < 2; ++half) {
        if constexpr (DotFirst) {
          acc[row][half] =
              _mm256_add_ps(acc[row][half], _mm256_load_ps(&result[base + row][8 * half]));
        }
        _mm256_store_ps(&result[base + row][8 * half], acc[row][half]);
      }
    }
  }
#endif
}
} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
