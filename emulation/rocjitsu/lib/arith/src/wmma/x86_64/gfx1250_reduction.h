// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Row-wise gfx1250 WMMA reduction shared by half and byte input formats.
#pragma once
#include "common/compiler.h"
#include "common/internal.h"
#include <immintrin.h>
namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {
// Reduce aligned row/column matrices in K order, preserving the prescribed C placement.
template <int K, bool DotFirst>
inline void gfx1250_reduce_rows(const float rows[16][K], const float columns[K][16],
                                float result[16][16]) {
  for (int row = 0; row < 16; ++row) {
#if defined(__AVX512F__)
    auto initial = _mm512_load_ps(result[row]);
    auto acc = DotFirst ? _mm512_setzero_ps() : initial;
    RJ_ARITH_UNROLL(128)
    for (int k = 0; k < K; ++k) {
      acc = _mm512_fmadd_ps(_mm512_set1_ps(rows[row][k]), _mm512_load_ps(columns[k]), acc);
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
        acc = _mm256_fmadd_ps(_mm256_set1_ps(rows[row][k]), _mm256_load_ps(&columns[k][col]), acc);
      }
      if constexpr (DotFirst) {
        acc = _mm256_add_ps(acc, initial);
      }
      _mm256_store_ps(&result[row][col], acc);
    }
#endif
  }
}
} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
