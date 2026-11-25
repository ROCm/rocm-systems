// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// FP16/BF16 MFMA staging and reduction shared by isolated AVX2/AVX-512 backends.
#pragma once
#include "common/internal.h"
#include "mfma/x86_64/reduction.h"
#include <immintrin.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Evaluate all batches before publishing D, preserving arbitrary operand overlap.
template <int M, int K, int Batch, bool Bf16>
void mfma_half_kernel(uint64_t controls, bool clamp_overflow, uint32_t *const *d,
                      const uint32_t *const *a, const uint32_t *const *b,
                      const uint32_t *const *c) {
  constexpr int Elements = M * K * Batch;
  alignas(64) float left[Elements], right[Elements];
  for (int reg = 0; reg < Elements / 128; ++reg) {
    for (int i = 0; i < 128; i += 8) {
      auto x = _mm_loadu_si128(reinterpret_cast<const __m128i *>(a[reg] + i / 2));
      auto y = _mm_loadu_si128(reinterpret_cast<const __m128i *>(b[reg] + i / 2));
      if constexpr (Bf16) {
        _mm256_store_ps(left + reg * 128 + i,
                        _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(x), 16)));
        _mm256_store_ps(right + reg * 128 + i,
                        _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(y), 16)));
      } else {
        _mm256_store_ps(left + reg * 128 + i, _mm256_cvtph_ps(x));
        _mm256_store_ps(right + reg * 128 + i, _mm256_cvtph_ps(y));
      }
    }
  }
  mfma_reduce_simd<M, K, Batch, 2>(controls, d, left, right, c, clamp_overflow);
}
} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
