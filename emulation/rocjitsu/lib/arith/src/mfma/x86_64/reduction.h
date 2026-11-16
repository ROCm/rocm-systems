// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// MFMA row reductions shared by ISA-isolated input-format backends.
#pragma once
#include "common/compiler.h"
#include "common/internal.h"
#include "mfma/layout.h"
#include <cstring>
#include <immintrin.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {
// Operate across output columns at the width selected for this translation unit.
#if defined(__AVX512F__)
using MfmaNative = __m512;
inline constexpr int kMfmaWidth = 16;
inline MfmaNative mfma_load(const float *p) { return _mm512_load_ps(p); }
inline void mfma_store(float *p, MfmaNative v) { _mm512_store_ps(p, v); }
inline MfmaNative mfma_broadcast(float v) { return _mm512_set1_ps(v); }
inline MfmaNative mfma_multiply_add(MfmaNative a, MfmaNative b, MfmaNative c) {
  return _mm512_fmadd_ps(a, b, c);
}
#else
using MfmaNative = __m256;
inline constexpr int kMfmaWidth = 8;
inline MfmaNative mfma_load(const float *p) { return _mm256_load_ps(p); }
inline void mfma_store(float *p, MfmaNative v) { _mm256_store_ps(p, v); }
inline MfmaNative mfma_broadcast(float v) { return _mm256_set1_ps(v); }
inline MfmaNative mfma_multiply_add(MfmaNative a, MfmaNative b, MfmaNative c) {
  return _mm256_fmadd_ps(a, b, c);
}
#endif

// Reduce decoded packed sources with FMA, staging all outputs to preserve overlap.
template <int M, int K, int Batch, int PerWord>
void mfma_reduce_simd(uint64_t controls, uint32_t *const *d, const float *left, const float *right,
                      const uint32_t *const *c, bool clamp_overflow = false) {
  // Each matrix row is a contiguous slice of one physical accumulator. Stage it
  // in register order so source aliases remain valid through overflow checking.
  constexpr int registers = M * M * Batch / 64;
  alignas(64) float result[registers][64];
  for (int reg = 0; reg < registers; ++reg) {
    std::memcpy(result[reg], c[reg], sizeof(result[reg]));
  }
  for (int batch = 0; batch < Batch; ++batch) {
    alignas(64) float rows[M][K], columns[K][M];
    for (int i = 0; i < M; ++i) {
      for (int k = 0; k < K; ++k) {
        int index = mfma_input_index<M, K, Batch, PerWord>(i, k, batch);
        if (!(controls & ~RJ_ARITH_MFMA_A_ADJACENT_BLOCKS)) {
          rows[i][k] = left[index];
          columns[k][i] = right[index];
        } else {
          rows[i][k] = left[mfma_permute_input<PerWord, M>(index, controls, true)];
          columns[k][i] = right[mfma_permute_input<PerWord, M>(index, controls, false)];
        }
      }
    }
    for (int row = 0; row < M; ++row) {
      const auto first = mfma_output<M, Batch>(row, 0, batch);
      float *output = result[first.reg] + first.lane;
      if constexpr (M >= kMfmaWidth) {
        // Independent output columns share each A broadcast without serializing
        // their reduction chains. Each accumulator still visits K in order.
        // At most four vector accumulators: keep each at a constant register index.
        MfmaNative acc[M / kMfmaWidth];
        RJ_ARITH_UNROLL(4)
        for (int part = 0; part < M / kMfmaWidth; ++part) {
          acc[part] = mfma_load(output + part * kMfmaWidth);
        }
        RJ_ARITH_UNROLL(128)
        for (int k = 0; k < K; ++k) {
          auto x = mfma_broadcast(rows[row][k]);
          RJ_ARITH_UNROLL(4)
          for (int part = 0; part < M / kMfmaWidth; ++part) {
            acc[part] = mfma_multiply_add(x, mfma_load(&columns[k][part * kMfmaWidth]), acc[part]);
          }
        }
        RJ_ARITH_UNROLL(4)
        for (int part = 0; part < M / kMfmaWidth; ++part) {
          mfma_store(output + part * kMfmaWidth, acc[part]);
        }
      } else {
        // Four-column batched forms fit one XMM vector rather than a full YMM/ZMM.
        __m128 acc = _mm_load_ps(output);
        RJ_ARITH_UNROLL(128)
        for (int k = 0; k < K; ++k) {
          acc = _mm_fmadd_ps(_mm_set1_ps(rows[row][k]), _mm_load_ps(columns[k]), acc);
        }
        _mm_store_ps(output, acc);
      }
    }
  }
  if (clamp_overflow) {
    for (int batch = 0; batch < Batch; ++batch) {
      for (int row = 0; row < M; ++row) {
        const auto first = mfma_output<M, Batch>(row, 0, batch);
        for (int col = 0; col < M; ++col) {
          result[first.reg][first.lane + col] = mfma_clamp_overflow<M, K, Batch, PerWord>(
              result[first.reg][first.lane + col], as_float(c[first.reg][first.lane + col]), left,
              right, row, col, batch, controls);
        }
      }
    }
  }
  for (int reg = 0; reg < registers; ++reg) {
    std::memcpy(d[reg], result[reg], sizeof(result[reg]));
  }
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
