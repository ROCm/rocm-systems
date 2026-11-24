// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// MFMA row reductions shared by ISA-isolated input-format backends.
#pragma once
#include "common/compiler.h"
#include "common/internal.h"
#include "mfma/layout.h"
#include <cstring>
#include <immintrin.h>

#if defined(__AVX512F__)
#include "mfma/x86_64/batch4.h"
#endif

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

#if defined(__AVX512F__)
// Load one selected element from sixteen consecutive wave lanes.
template <int PerWord> __m512 mfma_lane_values(const float *source, int index) {
  if constexpr (PerWord == 1) {
    return _mm512_load_ps(source + index);
  } else {
    static_assert(PerWord == 2);
    auto base = source + (index & ~1);
    auto even = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
    auto indices = _mm512_add_epi32(even, _mm512_set1_epi32(index & 1));
    return _mm512_permutex2var_ps(_mm512_load_ps(base), indices, _mm512_load_ps(base + 16));
  }
}

// Reduce four independent 4x4 matrices per vector, in guest K order.
template <int K, int PerWord>
void mfma_reduce_batch4(uint64_t controls, float (*result)[64], const float *left,
                        const float *right) {
  for (int lane = 0; lane < 64; lane += 16) {
    int first_a = mfma_a_lane(lane, controls), first_b = mfma_b_lane(lane, controls);
    auto groups = mfma_batch4_groups(lane, first_a, controls);
    __m512 acc[4];
    for (int row = 0; row < 4; ++row) {
      acc[row] = _mm512_load_ps(result[row] + lane);
    }
    RJ_ARITH_UNROLL(4)
    for (int k = 0; k < K; ++k) {
      int word = k / PerWord * 64 * PerWord + k % PerWord;
      auto a = mfma_lane_values<PerWord>(left, word + (first_a & ~15) * PerWord);
      auto b = mfma_lane_values<PerWord>(right, word + first_b * PerWord);
      RJ_ARITH_UNROLL(4)
      for (int row = 0; row < 4; ++row) {
        auto x = _mm512_permutexvar_ps(_mm512_add_epi32(groups, _mm512_set1_epi32(row)), a);
        acc[row] = _mm512_fmadd_ps(x, b, acc[row]);
      }
    }
    for (int row = 0; row < 4; ++row) {
      _mm512_store_ps(result[row] + lane, acc[row]);
    }
  }
}

// A changes lane within its broadcast block. B preserves the low four lane
// bits, so each group of 16 columns is one load (two for interleaved halves).
template <int M, int K, int Batch, int PerWord>
void mfma_reduce_columns(uint64_t controls, float (*result)[64], const float *left,
                         const float *right) {
  int cbsz = int(controls & 7);
  int row_mask = cbsz && cbsz <= 4 ? (64 >> cbsz) - 1 : 63;
  for (int batch = 0; batch < Batch; ++batch) {
    const float *rows[K];
    __m512 columns[K][M / 16];
    for (int k = 0; k < K; ++k) {
      int i = mfma_input_index<M, K, Batch, PerWord>(0, k, batch);
      rows[k] = left + mfma_permute_input<PerWord>(i, controls, true);
      for (int part = 0; part < M / 16; ++part) {
        int j = mfma_permute_input<PerWord>(i + 16 * part * PerWord, controls, false);
        columns[k][part] = mfma_lane_values<PerWord>(right, j);
      }
    }
    for (int row = 0; row < M; ++row) {
      auto first = mfma_output<M, Batch>(row, 0, batch);
      float *out = result[first.reg] + first.lane;
      __m512 acc[M / 16];
      for (int part = 0; part < M / 16; ++part) {
        acc[part] = _mm512_load_ps(out + 16 * part);
      }
      RJ_ARITH_UNROLL(32)
      for (int k = 0; k < K; ++k) {
        auto x = _mm512_set1_ps(rows[k][(row & row_mask) * PerWord]);
        for (int part = 0; part < M / 16; ++part) {
          acc[part] = _mm512_fmadd_ps(x, columns[k][part], acc[part]);
        }
      }
      for (int part = 0; part < M / 16; ++part) {
        _mm512_store_ps(out + 16 * part, acc[part]);
      }
    }
  }
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
#if defined(__AVX512F__)
  if constexpr (M == 4 && Batch == 16 && (PerWord == 1 || PerWord == 2)) {
    mfma_reduce_batch4<K, PerWord>(controls, result, left, right);
  } else if constexpr (M >= 16 && (PerWord == 1 || PerWord == 2)) {
    mfma_reduce_columns<M, K, Batch, PerWord>(controls, result, left, right);
  } else
#endif
    for (int batch = 0; batch < Batch; ++batch) {
      alignas(64) float rows[M][K], columns[K][M];
      for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
          int index = mfma_input_index<M, K, Batch, PerWord>(i, k, batch);
          if (!controls) {
            rows[i][k] = left[index];
            columns[k][i] = right[index];
          } else {
            rows[i][k] = left[mfma_permute_input<PerWord>(index, controls, true)];
            columns[k][i] = right[mfma_permute_input<PerWord>(index, controls, false)];
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
              acc[part] =
                  mfma_multiply_add(x, mfma_load(&columns[k][part * kMfmaWidth]), acc[part]);
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
