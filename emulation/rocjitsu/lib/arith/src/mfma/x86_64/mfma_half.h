// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// FP16/BF16 MFMA staging and reduction shared by isolated AVX2/AVX-512 backends.
#pragma once
#include "common/internal.h"
#include "mfma/x86_64/reduction.h"
#include <immintrin.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Decode one Wave64 packed half VGPR into two contiguous 64-lane FP32 planes (low/high halves).
template <bool Bf16>
inline void mfma_unpack_half_vgpr(const uint32_t *src, float (&lo_out)[64], float (&hi_out)[64]) {
#if defined(RJ_ARITH_TARGET_X86_64_V4)
  const __m512i hi_mask = _mm512_set1_epi32(int(0xffff0000U));
  for (int lane = 0; lane < 64; lane += 16) {
    const __m512i words = _mm512_loadu_si512(src + lane);
    if constexpr (Bf16) {
      _mm512_store_ps(lo_out + lane, _mm512_castsi512_ps(_mm512_slli_epi32(words, 16)));
      _mm512_store_ps(hi_out + lane, _mm512_castsi512_ps(_mm512_and_si512(words, hi_mask)));
    } else {
      const __m256i lo16 = _mm512_cvtepi32_epi16(words);
      const __m256i hi16 = _mm512_cvtepi32_epi16(_mm512_srli_epi32(words, 16));
      _mm512_store_ps(lo_out + lane, _mm512_cvtph_ps(lo16));
      _mm512_store_ps(hi_out + lane, _mm512_cvtph_ps(hi16));
    }
  }
#else
  const __m256i hi_mask = _mm256_set1_epi32(int(0xffff0000U));
  const __m256i split_halves = _mm256_broadcastsi128_si256(
      _mm_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15));
  for (int lane = 0; lane < 64; lane += 8) {
    const __m256i words = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(src + lane));
    if constexpr (Bf16) {
      _mm256_store_ps(lo_out + lane, _mm256_castsi256_ps(_mm256_slli_epi32(words, 16)));
      _mm256_store_ps(hi_out + lane, _mm256_castsi256_ps(_mm256_and_si256(words, hi_mask)));
    } else {
      const __m256i packed =
          _mm256_permute4x64_epi64(_mm256_shuffle_epi8(words, split_halves), 0xd8);
      _mm256_store_ps(lo_out + lane, _mm256_cvtph_ps(_mm256_castsi256_si128(packed)));
      _mm256_store_ps(hi_out + lane, _mm256_cvtph_ps(_mm256_extracti128_si256(packed, 1)));
    }
  }
#endif
}

// Evaluate all batches before publishing D, preserving arbitrary operand overlap.
template <int M, int K, int Batch, bool Bf16, MfmaF32Model Model>
void mfma_half_kernel(uint64_t controls, uint64_t flags, uint32_t *const *d,
                      const uint32_t *const *a, const uint32_t *const *b,
                      const uint32_t *const *c) {
  if constexpr (Model == MfmaF32Model::Default) {
    const bool clamp_overflow = Bf16 && (flags & RJ_ARITH_FP16_OVFL);
    constexpr int per_lane = K / (64 / (M * Batch));
    constexpr int regs = per_lane / 2;
    alignas(64) float left[per_lane][64], right[per_lane][64];
    for (int reg = 0; reg < regs; ++reg) {
      mfma_unpack_half_vgpr<Bf16>(a[reg], left[2 * reg], left[2 * reg + 1]);
      mfma_unpack_half_vgpr<Bf16>(b[reg], right[2 * reg], right[2 * reg + 1]);
    }
    mfma_reduce_planes<M, K, Batch>(controls, d, left, right, c, clamp_overflow);
  } else {
    constexpr int Elements = M * K * Batch;
    alignas(64) float left[Elements], right[Elements];
    for (int reg = 0; reg < Elements / 128; ++reg) {
      for (int i = 0; i < 128; i += 8) {
        __m128i x = _mm_loadu_si128(reinterpret_cast<const __m128i *>(a[reg] + i / 2));
        __m128i y = _mm_loadu_si128(reinterpret_cast<const __m128i *>(b[reg] + i / 2));
        if constexpr (Model == MfmaF32Model::Gfx90a) {
          // Flush before widening: FP16 subnormals are normal numbers in FP32.
          const __m128i exponent = _mm_set1_epi16(Bf16 ? 0x7f80 : 0x7c00);
          const __m128i sign = _mm_set1_epi16(static_cast<short>(0x8000U));
          auto flush = [&](__m128i bits) {
            const __m128i tiny =
                _mm_cmpeq_epi16(_mm_and_si128(bits, exponent), _mm_setzero_si128());
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
    mfma_reduce_simd<M, K, Batch, 2, Model>(controls, d, left, right, c, false, flags);
  }
}
} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
