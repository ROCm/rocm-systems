// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// ISA-isolated x86-64-v4 FP32/XF32 MFMA, using the shared FP32 row reduction.
#include "mfma/backends.h"
#include "mfma/x86_64/reduction.h"
#include <immintrin.h>

namespace rocjitsu::arith {
// Stage raw FP32 inputs before invoking this ISA domain's matrix reduction.
template <int M, int K, int Batch, bool Xf32, MfmaF32Model Model>
void mfma_f32_x86_64_v4(uint64_t controls, uint64_t flags, uint32_t *const *d,
                        const uint32_t *const *a, const uint32_t *const *b,
                        const uint32_t *const *c) {
  const bool clamp_overflow =
      Model == MfmaF32Model::Default && !Xf32 && (flags & RJ_ARITH_FP16_OVFL);
  alignas(64) float left[M * K * Batch], right[M * K * Batch];
  for (int i = 0; i < M * K * Batch; i += 8) {
    auto x = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a[i / 64] + i % 64));
    auto y = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b[i / 64] + i % 64));
    if constexpr (Xf32) {
      const auto mask = _mm256_set1_epi32(static_cast<int>(kMfmaXf32InputMask));
      x = _mm256_and_si256(x, mask);
      y = _mm256_and_si256(y, mask);
    }
    if constexpr (Model == MfmaF32Model::Gfx90a) {
      if (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) {
        x = mfma_flush_f32x8(x);
        y = mfma_flush_f32x8(y);
      }
    }
    _mm256_store_ps(left + i, _mm256_castsi256_ps(x));
    _mm256_store_ps(right + i, _mm256_castsi256_ps(y));
  }
  mfma_reduce_simd<M, K, Batch, 1, Model>(controls, d, left, right, c, clamp_overflow, flags);
}
template void mfma_f32_x86_64_v4<16, 8, 1, true>(uint64_t controls, uint64_t flags,
                                                 uint32_t *const *d, const uint32_t *const *a,
                                                 const uint32_t *const *b,
                                                 const uint32_t *const *c);
template void mfma_f32_x86_64_v4<32, 4, 1, true>(uint64_t controls, uint64_t flags,
                                                 uint32_t *const *d, const uint32_t *const *a,
                                                 const uint32_t *const *b,
                                                 const uint32_t *const *c);
template void mfma_f32_x86_64_v4<32, 1, 2>(uint64_t, uint64_t, uint32_t *const *,
                                           const uint32_t *const *, const uint32_t *const *,
                                           const uint32_t *const *);
template void mfma_f32_x86_64_v4<16, 1, 4>(uint64_t, uint64_t, uint32_t *const *,
                                           const uint32_t *const *, const uint32_t *const *,
                                           const uint32_t *const *);
template void mfma_f32_x86_64_v4<4, 1, 16>(uint64_t, uint64_t, uint32_t *const *,
                                           const uint32_t *const *, const uint32_t *const *,
                                           const uint32_t *const *);
template void mfma_f32_x86_64_v4<32, 2, 1>(uint64_t, uint64_t, uint32_t *const *,
                                           const uint32_t *const *, const uint32_t *const *,
                                           const uint32_t *const *);
template void mfma_f32_x86_64_v4<16, 4, 1>(uint64_t, uint64_t, uint32_t *const *,
                                           const uint32_t *const *, const uint32_t *const *,
                                           const uint32_t *const *);

// Each old target retains its numerical and broadcast policy in this ISA domain.
template void mfma_f32_x86_64_v4<32, 1, 2, false, MfmaF32Model::Gfx908>(uint64_t, uint64_t,
                                                                        uint32_t *const *,
                                                                        const uint32_t *const *,
                                                                        const uint32_t *const *,
                                                                        const uint32_t *const *);
template void mfma_f32_x86_64_v4<16, 1, 4, false, MfmaF32Model::Gfx908>(uint64_t, uint64_t,
                                                                        uint32_t *const *,
                                                                        const uint32_t *const *,
                                                                        const uint32_t *const *,
                                                                        const uint32_t *const *);
template void mfma_f32_x86_64_v4<4, 1, 16, false, MfmaF32Model::Gfx908>(uint64_t, uint64_t,
                                                                        uint32_t *const *,
                                                                        const uint32_t *const *,
                                                                        const uint32_t *const *,
                                                                        const uint32_t *const *);
template void mfma_f32_x86_64_v4<32, 2, 1, false, MfmaF32Model::Gfx908>(uint64_t, uint64_t,
                                                                        uint32_t *const *,
                                                                        const uint32_t *const *,
                                                                        const uint32_t *const *,
                                                                        const uint32_t *const *);
template void mfma_f32_x86_64_v4<16, 4, 1, false, MfmaF32Model::Gfx908>(uint64_t, uint64_t,
                                                                        uint32_t *const *,
                                                                        const uint32_t *const *,
                                                                        const uint32_t *const *,
                                                                        const uint32_t *const *);
template void mfma_f32_x86_64_v4<32, 1, 2, false, MfmaF32Model::Gfx90a>(uint64_t, uint64_t,
                                                                        uint32_t *const *,
                                                                        const uint32_t *const *,
                                                                        const uint32_t *const *,
                                                                        const uint32_t *const *);
template void mfma_f32_x86_64_v4<16, 1, 4, false, MfmaF32Model::Gfx90a>(uint64_t, uint64_t,
                                                                        uint32_t *const *,
                                                                        const uint32_t *const *,
                                                                        const uint32_t *const *,
                                                                        const uint32_t *const *);
template void mfma_f32_x86_64_v4<4, 1, 16, false, MfmaF32Model::Gfx90a>(uint64_t, uint64_t,
                                                                        uint32_t *const *,
                                                                        const uint32_t *const *,
                                                                        const uint32_t *const *,
                                                                        const uint32_t *const *);
template void mfma_f32_x86_64_v4<32, 2, 1, false, MfmaF32Model::Gfx90a>(uint64_t, uint64_t,
                                                                        uint32_t *const *,
                                                                        const uint32_t *const *,
                                                                        const uint32_t *const *,
                                                                        const uint32_t *const *);
template void mfma_f32_x86_64_v4<16, 4, 1, false, MfmaF32Model::Gfx90a>(uint64_t, uint64_t,
                                                                        uint32_t *const *,
                                                                        const uint32_t *const *,
                                                                        const uint32_t *const *,
                                                                        const uint32_t *const *);
} // namespace rocjitsu::arith
