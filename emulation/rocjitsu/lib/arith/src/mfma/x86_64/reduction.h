// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// MFMA row reductions shared by ISA-isolated input-format backends.
#pragma once
#include "common/compiler.h"
#include "common/internal.h"
#include "mfma/layout.h"
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
  alignas(64) float result[Batch][M][M];
  for (int batch = 0; batch < Batch; ++batch) {
    alignas(64) float rows[M][K], columns[K][M];
    for (int i = 0; i < M; ++i) {
      for (int k = 0; k < K; ++k) {
        int index = mfma_input_index<M, K, Batch, PerWord>(i, k, batch);
        rows[i][k] = left[mfma_permute_input<PerWord>(index, controls, true)];
        columns[k][i] = right[mfma_permute_input<PerWord>(index, controls, false)];
      }
    }
    for (int row = 0; row < M; ++row) {
      for (int col = 0; col < M; ++col) {
        auto out = mfma_output<M, Batch>(row, col, batch);
        result[batch][row][col] = as_float(c[out.reg][out.lane]);
      }
      if constexpr (M >= kMfmaWidth) {
        // Independent output columns share each A broadcast without serializing
        // their reduction chains. Each accumulator still visits K in order.
        // At most four vector accumulators: keep each at a constant register index.
        MfmaNative acc[M / kMfmaWidth];
        RJ_ARITH_UNROLL(4)
        for (int part = 0; part < M / kMfmaWidth; ++part) {
          acc[part] = mfma_load(&result[batch][row][part * kMfmaWidth]);
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
          mfma_store(&result[batch][row][part * kMfmaWidth], acc[part]);
        }
      } else {
        // Four-column batched forms fit one XMM vector rather than a full YMM/ZMM.
        __m128 acc = _mm_load_ps(result[batch][row]);
        RJ_ARITH_UNROLL(128)
        for (int k = 0; k < K; ++k) {
          acc = _mm_fmadd_ps(_mm_set1_ps(rows[row][k]), _mm_load_ps(columns[k]), acc);
        }
        _mm_store_ps(result[batch][row], acc);
      }
    }
  }
  uint32_t words[M * M * Batch / 64][64];
  for (int batch = 0; batch < Batch; ++batch) {
    for (int row = 0; row < M; ++row) {
      for (int col = 0; col < M; ++col) {
        auto out = mfma_output<M, Batch>(row, col, batch);
        float value = result[batch][row][col];
        if (clamp_overflow) {
          value = mfma_clamp_overflow<M, K, Batch, PerWord>(value, as_float(c[out.reg][out.lane]),
                                                            left, right, row, col, batch, controls);
        }
        words[out.reg][out.lane] = as_bits(value);
      }
    }
  }
  for (int reg = 0; reg < M * M * Batch / 64; ++reg) {
    for (int lane = 0; lane < 64; ++lane) {
      d[reg][lane] = words[reg][lane];
    }
  }
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
