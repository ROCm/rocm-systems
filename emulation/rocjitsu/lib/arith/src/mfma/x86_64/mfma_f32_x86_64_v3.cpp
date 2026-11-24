// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// ISA-isolated x86-64-v3 FP32/XF32 MFMA, using the shared FP32 row reduction.
#include "mfma/backends.h"
#include "mfma/x86_64/reduction.h"
#include <immintrin.h>

namespace rocjitsu::arith {
// Stage raw FP32 inputs before invoking this ISA domain's matrix reduction.
template <int M, int K, int Batch, bool Xf32>
void mfma_f32_x86_64_v3(uint64_t controls, bool clamp_overflow, uint32_t *const *d,
                        const uint32_t *const *a, const uint32_t *const *b,
                        const uint32_t *const *c) {
  alignas(64) float left[M * K * Batch], right[M * K * Batch];
  for (int i = 0; i < M * K * Batch; i += 8) {
    auto x = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a[i / 64] + i % 64));
    auto y = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b[i / 64] + i % 64));
    if constexpr (Xf32) {
      const auto mask = _mm256_set1_epi32(static_cast<int>(kMfmaXf32InputMask));
      x = _mm256_and_si256(x, mask);
      y = _mm256_and_si256(y, mask);
    }
    _mm256_store_ps(left + i, _mm256_castsi256_ps(x));
    _mm256_store_ps(right + i, _mm256_castsi256_ps(y));
  }
  mfma_reduce_simd<M, K, Batch, 1>(controls, d, left, right, c, clamp_overflow);
}
template void mfma_f32_x86_64_v3<16, 8, 1, true>(uint64_t controls, bool clamp_overflow,
                                                 uint32_t *const *d, const uint32_t *const *a,
                                                 const uint32_t *const *b,
                                                 const uint32_t *const *c);
template void mfma_f32_x86_64_v3<32, 4, 1, true>(uint64_t controls, bool clamp_overflow,
                                                 uint32_t *const *d, const uint32_t *const *a,
                                                 const uint32_t *const *b,
                                                 const uint32_t *const *c);
template void mfma_f32_x86_64_v3<32, 1, 2>(uint64_t, bool, uint32_t *const *,
                                           const uint32_t *const *, const uint32_t *const *,
                                           const uint32_t *const *);
template void mfma_f32_x86_64_v3<16, 1, 4>(uint64_t, bool, uint32_t *const *,
                                           const uint32_t *const *, const uint32_t *const *,
                                           const uint32_t *const *);
template void mfma_f32_x86_64_v3<4, 1, 16>(uint64_t, bool, uint32_t *const *,
                                           const uint32_t *const *, const uint32_t *const *,
                                           const uint32_t *const *);
template void mfma_f32_x86_64_v3<32, 2, 1>(uint64_t, bool, uint32_t *const *,
                                           const uint32_t *const *, const uint32_t *const *,
                                           const uint32_t *const *);
template void mfma_f32_x86_64_v3<16, 4, 1>(uint64_t, bool, uint32_t *const *,
                                           const uint32_t *const *, const uint32_t *const *,
                                           const uint32_t *const *);
} // namespace rocjitsu::arith
