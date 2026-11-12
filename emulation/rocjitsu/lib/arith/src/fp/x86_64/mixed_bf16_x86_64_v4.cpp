// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX512 mixed BF16 FMA: sixteen FP32 lanes or eight compensated FP64 sums.
#include "fp/mixed_bf16.h"
#include "fp/x86_64/mixed_bf16_x86_64_v3.h"
#include "rj_arith/detail/rj_arith_common.h"
#include <immintrin.h>
#include <stdint.h>

// Execute validated mixed BF16 controls without whole-vector exceptional fallbacks.
void rocjitsu::arith::mixed_bf16_x86_64_v4(MixedFma destination, uint64_t flags, uint32_t exec_mask,
                                           uint32_t mode, uint32_t *d, const uint32_t *a,
                                           const uint32_t *b, const uint32_t *c) {
  const bool clamp = mode & RJ_ARITH_ALU_CLAMP;
  const unsigned rounding = unsigned((flags & RJ_ARITH_FP_ROUND_MASK) >> 24);
  const int width = destination == MixedFma::Float ? 16 : 8;
  for (int lane = 0; lane < 32; lane += width) {
    const auto x = mixed_bf16_input8(a + lane, mode);
    const auto y = mixed_bf16_input8(b + lane, mode >> 1);
    const auto z = mixed_bf16_input8(c + lane, mode >> 2);
    if (destination == MixedFma::Float) {
      const auto xh = mixed_bf16_input8(a + lane + 8, mode);
      const auto yh = mixed_bf16_input8(b + lane + 8, mode >> 1);
      const auto zh = mixed_bf16_input8(c + lane + 8, mode >> 2);
      const auto result = _mm512_fmadd_ps(_mm512_insertf32x8(_mm512_castps256_ps512(x), xh, 1),
                                          _mm512_insertf32x8(_mm512_castps256_ps512(y), yh, 1),
                                          _mm512_insertf32x8(_mm512_castps256_ps512(z), zh, 1));
      const auto lo = mixed_bf16_finish8(destination, flags, clamp, x, y, z,
                                         _mm256_castps_si256(_mm512_castps512_ps256(result)));
      const auto hi = mixed_bf16_finish8(destination, flags, clamp, xh, yh, zh,
                                         _mm256_castps_si256(_mm512_extractf32x8_ps(result, 1)));
      _mm512_mask_storeu_epi32(d + lane, __mmask16(exec_mask >> lane),
                               _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1));
    } else {
      const auto product = _mm512_mul_pd(_mm512_cvtps_pd(x), _mm512_cvtps_pd(y));
      const auto addend = _mm512_cvtps_pd(z);
      const auto sum = _mm512_add_pd(product, addend);
      const auto virtual_z = _mm512_sub_pd(sum, product);
      const auto error = _mm512_add_pd(_mm512_sub_pd(product, _mm512_sub_pd(sum, virtual_z)),
                                       _mm512_sub_pd(addend, virtual_z));
      const auto lo = mixed_bf16_narrow4(_mm512_castpd512_pd256(sum), _mm512_castpd512_pd256(error),
                                         _mm512_castpd512_pd256(product),
                                         _mm512_castpd512_pd256(addend), rounding, clamp);
      const auto hi = mixed_bf16_narrow4(
          _mm512_extractf64x4_pd(sum, 1), _mm512_extractf64x4_pd(error, 1),
          _mm512_extractf64x4_pd(product, 1), _mm512_extractf64x4_pd(addend, 1), rounding, clamp);
      auto result = _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
      result = mixed_bf16_finish8(destination, flags, clamp, x, y, z, result);
      result = mixed_bf16_merge8(destination, result, d + lane);
      _mm256_mask_storeu_epi32(d + lane, __mmask8(exec_mask >> lane), result);
    }
  }
}
