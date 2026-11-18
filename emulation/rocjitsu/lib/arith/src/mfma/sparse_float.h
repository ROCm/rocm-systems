// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Floating SMFMAC decoding, sparse selection and ordered FMA, isolated by host ISA.
#pragma once

#include "mfma/fp8_formats.h"
#include "mfma/layout.h"
#include "mfma/nan_fma.h"
#include "mfma/sparse_layout.h"
#if defined(__AVX2__)
#include "mfma/x86_64/fp8.h"
#include <immintrin.h>
#endif

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Decode a packed wave once so sparse gathers operate on widened values.
template <int Count, int Bits, bool Bf, bool Fnuz, int Lanes = 64>
void smfmac_decode(const uint32_t *const *input, float *values) {
  constexpr int per_word = 32 / Bits;
  for (int i = 0; i < Count;) {
#if defined(__AVX2__)
    auto p =
        reinterpret_cast<const __m128i *>(input[i / (Lanes * per_word)] + i / per_word % Lanes);
    __m256 decoded;
    if constexpr (Bits == 8) {
      decoded = mfma_widen_fp8<Bf, Fnuz>(_mm256_cvtepu8_epi32(_mm_loadl_epi64(p)));
    } else if constexpr (Bf) {
      decoded =
          _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128(p)), 16));
    } else {
      decoded = _mm256_cvtph_ps(_mm_loadu_si128(p));
    }
    _mm256_store_ps(values + i, decoded);
    i += 8;
#else
    uint32_t word = input[i / (Lanes * per_word)][i / per_word % Lanes] >> (Bits * (i % per_word));
    if constexpr (Bits == 8) {
      values[i] = mfma_decode_fp8<Bf, Fnuz>(uint8_t(word));
    } else {
      values[i] = Bf ? bf16_to_float(uint16_t(word)) : f16_to_float(uint16_t(word));
    }
    ++i;
#endif
  }
}

// Reduce one sparse row, retaining input order and repairing only NaN output lanes.
template <int Columns, int Terms>
void sparse_float_row(const float *values, const float (*right)[Columns], const int *selected,
                      float *result) {
  float initial[Columns];
  for (int col = 0; col < Columns; ++col) {
    initial[col] = result[col];
  }
#if defined(__AVX512F__)
  for (int col = 0; col < Columns; col += 16) {
    auto acc = _mm512_load_ps(result + col);
    for (int t = 0; t < Terms; ++t) {
      acc =
          _mm512_fmadd_ps(_mm512_set1_ps(values[t]), _mm512_load_ps(right[selected[t]] + col), acc);
    }
    _mm512_store_ps(result + col, acc);
  }
#elif defined(__AVX2__)
  for (int col = 0; col < Columns; col += 8) {
    auto acc = _mm256_load_ps(result + col);
    for (int t = 0; t < Terms; ++t) {
      acc =
          _mm256_fmadd_ps(_mm256_set1_ps(values[t]), _mm256_load_ps(right[selected[t]] + col), acc);
    }
    _mm256_store_ps(result + col, acc);
  }
#endif
  for (int col = 0; col < Columns; ++col) {
#if defined(__AVX2__)
    // Replay only NaN outputs to preserve A/B/C payload priority across host FMAs.
    if ((as_bits(result[col]) & 0x7fffffffU) <= 0x7f800000U) {
      continue;
    }
#endif
    float acc = initial[col];
    for (int t = 0; t < Terms; ++t) {
      acc = matrix_nan_fma(values[t], right[selected[t]][col], acc);
    }
    result[col] = acc;
  }
}

// Preserve the inherited q/slot reduction order and stage all results before stores.
template <int M, int K, int Bits, bool BfA, bool BfB, bool Fnuz>
void smfmac_float_kernel(uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                         const uint32_t *const *index, bool clamp_overflow) {
  constexpr bool wide = M * K * Bits == 16384;
  constexpr auto layout = Bits == 16 ? (wide ? SmfmacLayout::Cdna4F16 : SmfmacLayout::Cdna3F16)
                                     : (wide ? SmfmacLayout::Cdna4Fp8 : SmfmacLayout::Cdna3Fp8);
  constexpr int per_word = 32 / Bits;
  alignas(64) float left[M * K / 2], raw_right[M * K], right[K][M], result[M][M];
  smfmac_decode<M * K / 2, Bits, BfA, Fnuz>(a, left);
  smfmac_decode<M * K, Bits, BfB, Fnuz>(b, raw_right);
  for (int k = 0; k < K; ++k) {
    for (int col = 0; col < M; ++col) {
      auto loc = smfmac_dense_loc<layout, M>(k, col);
      right[k][col] =
          raw_right[(loc.element / per_word * 64 + loc.lane) * per_word + loc.element % per_word];
    }
  }
  for (int row = 0; row < M; ++row) {
    float values[K / 2];
    int selected[K / 2];
    for (int t = 0; t < K / 2; ++t) {
      auto loc = smfmac_sparse_loc<layout, M, K>(row, t / 2, t % 2);
      values[t] =
          left[(loc.element / per_word * 64 + loc.lane) * per_word + loc.element % per_word];
      selected[t] = 4 * (t / 2) + int(index[0][loc.lane] >> (4 * loc.nibble + 2 * (t % 2)) & 3);
    }
    for (int col = 0; col < M; ++col) {
      auto out = mfma_output<M, 1>(row, col, 0);
      result[row][col] = as_float(d[out.reg][out.lane]);
    }
    sparse_float_row<M, K / 2>(values, right, selected, result[row]);
    if (clamp_overflow) {
      for (int col = 0; col < M; ++col) {
        auto out = mfma_output<M, 1>(row, col, 0);
        if (!__builtin_isinf(result[row][col]) ||
            !__builtin_isfinite(as_float(d[out.reg][out.lane]))) {
          continue;
        }
        bool finite = true;
        for (int t = 0; t < K / 2; ++t) {
          finite &= __builtin_isfinite(values[t]) && __builtin_isfinite(right[selected[t]][col]);
        }
        if (finite) {
          result[row][col] = as_float(as_bits(result[row][col]) - 1u);
        }
      }
    }
  }
  for (int row = 0; row < M; ++row) {
    for (int col = 0; col < M; ++col) {
      auto out = mfma_output<M, 1>(row, col, 0);
      d[out.reg][out.lane] = as_bits(result[row][col]);
    }
  }
}
} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
