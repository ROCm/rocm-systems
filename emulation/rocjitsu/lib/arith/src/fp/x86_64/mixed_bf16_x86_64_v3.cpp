// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 mixed BF16 FMA: eight FP32 lanes or two groups of four FP64 sums.
#include "fp/x86_64/mixed_bf16_x86_64_v3.h"
#include "fp/mixed_bf16.h"
#include "rj_arith/detail/rj_arith_common.h"
#include <immintrin.h>
#include <stdint.h>

namespace {
// Preserve the exact FP32 product and residual while rounding four results to BF16.
__m128i half4(__m128 a, __m128 b, __m128 c, unsigned rounding, bool clamp) {
  const auto product = _mm256_mul_pd(_mm256_cvtps_pd(a), _mm256_cvtps_pd(b));
  const auto z = _mm256_cvtps_pd(c);
  const auto sum = _mm256_add_pd(product, z);
  const auto virtual_z = _mm256_sub_pd(sum, product);
  const auto error = _mm256_add_pd(_mm256_sub_pd(product, _mm256_sub_pd(sum, virtual_z)),
                                   _mm256_sub_pd(z, virtual_z));
  return rocjitsu::arith::mixed_bf16_narrow4(sum, error, product, z, rounding, clamp);
}
} // namespace

// Execute validated mixed BF16 controls, retaining masked lanes and destination halves.
void rocjitsu::arith::mixed_bf16_x86_64_v3(MixedFma destination, uint64_t flags, uint32_t exec_mask,
                                           uint32_t mode, uint32_t *d, const uint32_t *a,
                                           const uint32_t *b, const uint32_t *c) {
  const bool clamp = mode & RJ_ARITH_ALU_CLAMP;
  const unsigned rounding = unsigned((flags & RJ_ARITH_FP_ROUND_MASK) >> 24);
  for (int lane = 0; lane < 32; lane += 8) {
    const auto x = mixed_bf16_input8(a + lane, mode);
    const auto y = mixed_bf16_input8(b + lane, mode >> 1);
    const auto z = mixed_bf16_input8(c + lane, mode >> 2);
    __m256i result;
    if (destination == MixedFma::Float) {
      result = _mm256_castps_si256(_mm256_fmadd_ps(x, y, z));
    } else {
      const auto lo = half4(_mm256_castps256_ps128(x), _mm256_castps256_ps128(y),
                            _mm256_castps256_ps128(z), rounding, clamp);
      const auto hi = half4(_mm256_extractf128_ps(x, 1), _mm256_extractf128_ps(y, 1),
                            _mm256_extractf128_ps(z, 1), rounding, clamp);
      result = _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
    }
    result = mixed_bf16_finish8(destination, flags, clamp, x, y, z, result);
    result = mixed_bf16_merge8(destination, result, d + lane);
    const auto active = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                                          _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), active, result);
  }
}
