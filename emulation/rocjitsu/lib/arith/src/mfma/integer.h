// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Dense byte-input MFMA/WMMA staging and wrapping integer reductions in each ISA domain.
#pragma once
#include "common/internal.h"
#include "mfma/layout.h"
#include "rj_arith/detail/rj_arith_wmma.h"
#if defined(__AVX2__)
#include <immintrin.h>
#endif
namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {
// Expand packed bytes once, preserving signed values as modulo-2^32 encodings.
template <int Count, int Lanes>
void matrix_i8_decode(const uint32_t *const *p, uint32_t *out, bool is_signed) {
#if defined(__AVX512F__)
  for (int i = 0; i < Count; i += 16) {
    auto bytes =
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(p[i / (4 * Lanes)] + i / 4 % Lanes));
    auto values = is_signed ? _mm512_cvtepi8_epi32(bytes) : _mm512_cvtepu8_epi32(bytes);
    _mm512_store_si512(out + i, values);
  }
#elif defined(__AVX2__)
  for (int i = 0; i < Count; i += 8) {
    auto bytes =
        _mm_loadl_epi64(reinterpret_cast<const __m128i *>(p[i / (4 * Lanes)] + i / 4 % Lanes));
    auto values = is_signed ? _mm256_cvtepi8_epi32(bytes) : _mm256_cvtepu8_epi32(bytes);
    _mm256_store_si256(reinterpret_cast<__m256i *>(out + i), values);
  }
#else
  for (int i = 0; i < Count; ++i) {
    uint32_t byte = p[i / (4 * Lanes)][i / 4 % Lanes] >> (8 * (i % 4)) & 0xffU;
    out[i] = is_signed ? (byte ^ 0x80U) - 0x80U : byte;
  }
#endif
}

// Locate a decoded byte, respecting MFMA broadcasts or gfx1250 split-K layout.
template <int M, int K, int Batch, bool Wmma>
int matrix_i8_index(int index, int k, int batch, uint64_t controls, bool a) {
  if constexpr (Wmma) {
    int local = k / 32 * 16 + k % 16;
    int lane = index + 16 * (k / 16 % 2);
    return (local / 4 * 32 + lane) * 4 + local % 4;
  } else {
    constexpr int per_lane = K / (64 / (M * Batch));
    int lane = batch * M + k / per_lane * M * Batch + index;
    lane = a ? mfma_a_lane(lane, controls) : mfma_b_lane(lane, controls);
    int local = k % per_lane;
    return (local / 4 * 64 + lane) * 4 + local % 4;
  }
}

// Locate a result word in the architecture's matrix output layout.
template <int M, int Batch, bool Wmma> MfmaWord matrix_i8_output(int row, int col, int batch) {
  if constexpr (Wmma) {
    return {row % 8, col + 16 * (row / 8)};
  } else {
    return mfma_output<M, Batch>(row, col, batch);
  }
}

// Reduce decoded byte matrices and apply optional signed saturation after adding C.
template <int M, int K, int Batch, bool Wmma>
void matrix_i8_kernel(uint64_t controls, bool clamp, uint32_t *const *d, const uint32_t *const *a,
                      const uint32_t *const *b, const uint32_t *const *c) {
  constexpr int lanes = Wmma ? 32 : 64;
  alignas(64) uint32_t left[M * K * Batch], right[M * K * Batch];
  matrix_i8_decode<M * K * Batch, lanes>(a, left, !Wmma || (controls & RJ_ARITH_WMMA_SIGNED_A));
  matrix_i8_decode<M * K * Batch, lanes>(b, right, !Wmma || (controls & RJ_ARITH_WMMA_SIGNED_B));
  alignas(64) uint32_t result[Batch][M][M];
  for (int batch = 0; batch < Batch; ++batch) {
    alignas(64) uint32_t rows[M][K], columns[K][M];
    for (int i = 0; i < M; ++i) {
      for (int k = 0; k < K; ++k) {
        rows[i][k] = left[matrix_i8_index<M, K, Batch, Wmma>(i, k, batch, controls, true)];
        columns[k][i] = right[matrix_i8_index<M, K, Batch, Wmma>(i, k, batch, controls, false)];
      }
    }
    for (int row = 0; row < M; ++row) {
#if defined(__AVX512F__)
      if constexpr (M >= 16) {
        for (int col = 0; col < M; col += 16) {
          auto acc = _mm512_setzero_si512();
          for (int k = 0; k < K; ++k) {
            acc = _mm512_add_epi32(acc, _mm512_mullo_epi32(_mm512_set1_epi32(rows[row][k]),
                                                           _mm512_load_si512(&columns[k][col])));
          }
          _mm512_store_si512(&result[batch][row][col], acc);
        }
      } else
#endif
#if defined(__AVX2__)
          if constexpr (M >= 8) {
        for (int col = 0; col < M; col += 8) {
          auto acc = _mm256_setzero_si256();
          for (int k = 0; k < K; ++k) {
            acc = _mm256_add_epi32(
                acc, _mm256_mullo_epi32(
                         _mm256_set1_epi32(rows[row][k]),
                         _mm256_load_si256(reinterpret_cast<const __m256i *>(&columns[k][col]))));
          }
          _mm256_store_si256(reinterpret_cast<__m256i *>(&result[batch][row][col]), acc);
        }
      } else {
        auto acc = _mm_setzero_si128();
        for (int k = 0; k < K; ++k) {
          acc = _mm_add_epi32(
              acc, _mm_mullo_epi32(_mm_set1_epi32(rows[row][k]),
                                   _mm_load_si128(reinterpret_cast<const __m128i *>(columns[k]))));
        }
        _mm_store_si128(reinterpret_cast<__m128i *>(result[batch][row]), acc);
      }
#else
      for (int col = 0; col < M; ++col) {
        uint32_t sum = 0;
        for (int k = 0; k < K; ++k) {
          sum += rows[row][k] * columns[k][col];
        }
        result[batch][row][col] = sum;
      }
#endif
      for (int col = 0; col < M; ++col) {
        auto out = matrix_i8_output<M, Batch, Wmma>(row, col, batch);
        uint32_t sum = result[batch][row][col], initial = c[out.reg][out.lane];
        if (clamp) {
          // The byte dot fits int32; only adding the accumulator can overflow.
          int64_t wide = int64_t(bit_cast<int32_t>(sum)) + bit_cast<int32_t>(initial);
          result[batch][row][col] = wide > 0x7fffffffLL    ? 0x7fffffffU
                                    : wide < -0x80000000LL ? 0x80000000U
                                                           : uint32_t(wide);
          continue;
        }
        result[batch][row][col] = sum + initial;
      }
    }
  }
  // Defer publication until all batches have consumed their potentially aliased sources.
  for (int batch = 0; batch < Batch; ++batch) {
    for (int row = 0; row < M; ++row) {
      for (int col = 0; col < M; ++col) {
        auto out = matrix_i8_output<M, Batch, Wmma>(row, col, batch);
        d[out.reg][out.lane] = result[batch][row][col];
      }
    }
  }
}
} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
