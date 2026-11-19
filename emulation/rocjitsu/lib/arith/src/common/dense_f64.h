// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Shared four-product FP64 MFMA/WMMA staging and reduction in each ISA domain.
#pragma once
#include "common/compiler.h"
#include "common/internal.h"
#include <math.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif
namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {
// Load one FP64 register pair, applying absolute value before sign negation.
inline double dense_f64_read(const uint32_t *const *p, int reg, int lane, bool negate,
                             bool absolute = false) {
  uint64_t bits = uint64_t(p[reg][lane]) | (uint64_t(p[reg + 1][lane]) << 32);
  if (absolute) {
    bits &= 0x7fffffffffffffffULL;
  }
  return as_double(bits ^ (negate ? 0x8000000000000000ULL : 0));
}

// Evaluate all batches before destination stores, preserving arbitrary input overlap.
template <int M, int Batch, int Lanes>
void dense_f64_kernel(uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
                      const uint32_t *const *b, const uint32_t *const *c) {
  static_assert((Lanes == 64 && ((M == 16 && Batch == 1) || (M == 4 && Batch == 4))) ||
                (Lanes == 32 && M == 16 && Batch == 1));
  // Wave32 WMMA packs two factors per lane and eight output rows per lane group.
  // Wave64 MFMA packs one factor per lane, with shape-specific output row groups.
  const auto output_reg = [](int batch, int row) {
    if constexpr (Lanes == 32) {
      return 2 * (row % 8);
    } else {
      return 2 * (batch * (M * M / 64) + row / (64 / M));
    }
  };
  const auto output_lane = [](int batch, int row, int col) {
    if constexpr (Lanes == 32) {
      return col + 16 * (row / 8);
    } else {
      constexpr int batch_groups = (64 + M * M - 1) / (M * M);
      return batch % batch_groups * M + row % (64 / M) * batch_groups * M + col;
    }
  };
  alignas(64) double result[Batch][M][M];
  for (int batch = 0; batch < Batch; ++batch) {
    alignas(64) double rows[M][4], columns[4][M];
    for (int i = 0; i < M; ++i) {
      for (int k = 0; k < 4; ++k) {
        const int reg = Lanes == 32 ? 2 * (k % 2) : 0;
        const int lane = Lanes == 32 ? i + 16 * (k / 2) : batch * M + k * M * Batch + i;
        rows[i][k] = dense_f64_read(a, reg, lane, controls & 1);
        columns[k][i] = dense_f64_read(b, reg, lane, controls & 2);
      }
    }
    for (int row = 0; row < M; ++row) {
      int reg = output_reg(batch, row);
      for (int col = 0; col < M; ++col) {
        int lane = output_lane(batch, row, col);
        result[batch][row][col] = dense_f64_read(c, reg, lane, controls & 4, controls & 8);
      }
#if defined(__AVX512F__)
      if constexpr (M >= 8) {
        for (int col = 0; col < M; col += 8) {
          auto acc = _mm512_load_pd(&result[batch][row][col]);
          RJ_ARITH_UNROLL(4)
          for (int k = 0; k < 4; ++k) {
            acc = _mm512_fmadd_pd(_mm512_set1_pd(rows[row][k]), _mm512_load_pd(&columns[k][col]),
                                  acc);
          }
          _mm512_store_pd(&result[batch][row][col], acc);
        }
      } else
#endif
#if defined(__AVX2__)
      {
        for (int col = 0; col < M; col += 4) {
          auto acc = _mm256_load_pd(&result[batch][row][col]);
          RJ_ARITH_UNROLL(4)
          for (int k = 0; k < 4; ++k) {
            acc = _mm256_fmadd_pd(_mm256_set1_pd(rows[row][k]), _mm256_load_pd(&columns[k][col]),
                                  acc);
          }
          _mm256_store_pd(&result[batch][row][col], acc);
        }
      }
#else
      {
        for (int col = 0; col < M; ++col) {
          for (int k = 0; k < 4; ++k) {
            result[batch][row][col] = ::fma(rows[row][k], columns[k][col], result[batch][row][col]);
          }
        }
      }
#endif
    }
  }
  // Pair stores follow the legacy FP64 output map; no source is read here.
  for (int batch = 0; batch < Batch; ++batch) {
    for (int row = 0; row < M; ++row) {
      int reg = output_reg(batch, row);
      for (int col = 0; col < M; ++col) {
        int lane = output_lane(batch, row, col);
        uint64_t bits = bit_cast<uint64_t>(result[batch][row][col]);
        d[reg][lane] = uint32_t(bits);
        d[reg + 1][lane] = uint32_t(bits >> 32);
      }
    }
  }
}
} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
