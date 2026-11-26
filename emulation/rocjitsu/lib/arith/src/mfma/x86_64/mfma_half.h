// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// FP16/BF16 MFMA staging and reduction shared by isolated AVX2/AVX-512 backends.
#pragma once
#include "common/internal.h"
#include "mfma/x86_64/reduction.h"
#include <immintrin.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Evaluate all batches before publishing D, preserving arbitrary operand overlap.
template <int M, int K, int Batch, bool Bf16, MfmaF32Model Model>
void mfma_half_kernel(uint64_t controls, uint64_t flags, uint32_t *const *d,
                      const uint32_t *const *a, const uint32_t *const *b,
                      const uint32_t *const *c) {
  constexpr int Elements = M * K * Batch;
  alignas(64) float left[Elements], right[Elements];
  for (int reg = 0; reg < Elements / 128; ++reg) {
    for (int i = 0; i < 128; i += 8) {
      auto x = _mm_loadu_si128(reinterpret_cast<const __m128i *>(a[reg] + i / 2));
      auto y = _mm_loadu_si128(reinterpret_cast<const __m128i *>(b[reg] + i / 2));
      if constexpr (Model == MfmaF32Model::Gfx90a) {
        // Flush before widening: FP16 subnormals are normal numbers in FP32.
        auto exponent = _mm_set1_epi16(Bf16 ? 0x7f80 : 0x7c00);
        auto sign = _mm_set1_epi16(short(0x8000U));
        auto flush = [&](auto bits) {
          auto tiny = _mm_cmpeq_epi16(_mm_and_si128(bits, exponent), _mm_setzero_si128());
          return _mm_blendv_epi8(bits, _mm_and_si128(bits, sign), tiny);
        };
        x = flush(x);
        y = flush(y);
      }
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
  const bool clamp_overflow =
      Model == MfmaF32Model::Default && Bf16 && (flags & RJ_ARITH_FP16_OVFL);
  mfma_reduce_simd<M, K, Batch, 2, Model>(controls, d, left, right, c, clamp_overflow, flags);
}
} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
