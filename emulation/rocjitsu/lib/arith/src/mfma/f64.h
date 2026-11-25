// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Dense FP64 MFMA layout and staged reduction, isolated within each ISA domain.
#pragma once
#include "common/compiler.h"
#include "common/internal.h"
#include <math.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif
namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {
// Load one FP64 register pair, applying the selected source sign control.
inline double mfma_f64_read(const uint32_t *const *p, int reg, int lane, bool negate) {
  uint64_t bits = uint64_t(p[reg][lane]) | (uint64_t(p[reg + 1][lane]) << 32);
  return as_double(bits ^ (negate ? 0x8000000000000000ULL : 0));
}

// Evaluate all batches before destination stores, preserving arbitrary input overlap.
template <int M, int Batch>
void mfma_f64_kernel(uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
                     const uint32_t *const *b, const uint32_t *const *c) {
#if defined(__AVX512F__)
  if constexpr (M == 16) {
    // Keep the split VGPRs in lane order. All 16 rows reuse the same four B rows.
    auto load = [](const uint32_t *const *src, int reg, int lane) {
      auto lo = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(src[reg] + lane));
      auto hi = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(src[reg + 1] + lane));
      return _mm512_or_si512(_mm512_cvtepu32_epi64(lo),
                             _mm512_slli_epi64(_mm512_cvtepu32_epi64(hi), 32));
    };
    const auto b_sign = _mm512_set1_epi64(controls & 2 ? INT64_MIN : 0);
    const auto c_sign = _mm512_set1_epi64(controls & 4 ? INT64_MIN : 0);
    __m512d right[4][2];
    for (int k = 0; k < 4; ++k) {
      for (int part = 0; part < 2; ++part) {
        right[k][part] =
            _mm512_castsi512_pd(_mm512_xor_si512(load(b, 0, 16 * k + 8 * part), b_sign));
      }
    }
    alignas(64) uint32_t words[8][64];
    for (int row = 0; row < 16; ++row) {
      int reg = 2 * (row / 4), lane = 16 * (row % 4);
      __m512d acc[2];
      for (int part = 0; part < 2; ++part) {
        acc[part] = _mm512_castsi512_pd(_mm512_xor_si512(load(c, reg, lane + 8 * part), c_sign));
      }
      RJ_ARITH_UNROLL(4)
      for (int k = 0; k < 4; ++k) {
        auto x = _mm512_set1_pd(mfma_f64_read(a, 0, row + 16 * k, controls & 1));
        for (int part = 0; part < 2; ++part) {
          acc[part] = _mm512_fmadd_pd(x, right[k][part], acc[part]);
        }
      }
      for (int part = 0; part < 2; ++part) {
        auto v = _mm512_castpd_si512(acc[part]);
        _mm256_store_si256(reinterpret_cast<__m256i *>(words[reg] + lane + 8 * part),
                           _mm512_cvtepi64_epi32(v));
        _mm256_store_si256(reinterpret_cast<__m256i *>(words[reg + 1] + lane + 8 * part),
                           _mm512_cvtepi64_epi32(_mm512_srli_epi64(v, 32)));
      }
    }
    for (int reg = 0; reg < 8; ++reg) {
      __builtin_memcpy(d[reg], words[reg], sizeof(words[reg]));
    }
    return;
  }
#endif
  constexpr int groups = 64 / M, batch_groups = (64 + M * M - 1) / (M * M);
  alignas(64) double result[Batch][M][M];
  for (int batch = 0; batch < Batch; ++batch) {
    alignas(64) double rows[M][4], columns[4][M];
    for (int i = 0; i < M; ++i) {
      for (int k = 0; k < 4; ++k) {
        // Both supported shapes use one element per lane and two VGPRs per factor.
        int lane = batch * M + k * M * Batch + i;
        rows[i][k] = mfma_f64_read(a, 0, lane, controls & 1);
        columns[k][i] = mfma_f64_read(b, 0, lane, controls & 2);
      }
    }
    for (int row = 0; row < M; ++row) {
      int reg = 2 * (batch * (M * M / 64) + row / groups);
      for (int col = 0; col < M; ++col) {
        int lane = batch % batch_groups * M + row % groups * batch_groups * M + col;
        result[batch][row][col] = mfma_f64_read(c, reg, lane, controls & 4);
      }
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
      int reg = 2 * (batch * (M * M / 64) + row / groups);
      for (int col = 0; col < M; ++col) {
        int lane = batch % batch_groups * M + row % groups * batch_groups * M + col;
        uint64_t bits = bit_cast<uint64_t>(result[batch][row][col]);
        d[reg][lane] = uint32_t(bits);
        d[reg + 1][lane] = uint32_t(bits >> 32);
      }
    }
  }
}
} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
