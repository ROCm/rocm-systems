// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 directed interpolation: exact FP64 products and compensated sums locate
// the correct side of the nearest FP32 result, without changing host controls.
#include "common/validation.h"
#include "fp/mixed_fma_control.h"
#include "graphics/interp_gfx10.h"
#include <immintrin.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {
// Decode four coefficient/coordinate words and apply source modifiers and flushing.
static __m128 input(const uint32_t *source, uint64_t flags, uint32_t mode) {
  auto bits = _mm_loadu_si128(reinterpret_cast<const __m128i *>(source));
  const bool half = mode & RJ_ARITH_MIX_F16_A;
  if (half) {
    bits = _mm_and_si128(_mm_srli_epi32(bits, mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0),
                         _mm_set1_epi32(0xffff));
    if (mixed_fma_flush_input(flags, mode)) {
      const auto tiny =
          _mm_cmplt_epi32(_mm_and_si128(bits, _mm_set1_epi32(0x7fff)), _mm_set1_epi32(0x400));
      bits = _mm_blendv_epi8(bits, _mm_and_si128(bits, _mm_set1_epi32(0x8000)), tiny);
    }
    bits = _mm_castps_si128(_mm_cvtph_ps(_mm_packus_epi32(bits, bits)));
  } else if (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) {
    const auto tiny = _mm_cmplt_epi32(_mm_and_si128(bits, _mm_set1_epi32(0x7fffffff)),
                                      _mm_set1_epi32(0x00800000));
    bits = _mm_blendv_epi8(bits, _mm_and_si128(bits, _mm_set1_epi32(INT32_MIN)), tiny);
  }
  if (mode & RJ_ARITH_ALU_ABS_A) {
    bits = _mm_and_si128(bits, _mm_set1_epi32(INT32_MAX));
  }
  if (mode & RJ_ARITH_ALU_NEG_A) {
    bits = _mm_xor_si128(bits, _mm_set1_epi32(INT32_MIN));
  }
  return _mm_castsi128_ps(bits);
}
} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE

namespace rocjitsu::arith {
// Implement the directed interpolation contract declared in graphics/interp_gfx10.h.
uint32_t interp_gfx10_directed_x86_64_v3(uint64_t flags, uint32_t exec_mask, uint32_t mode,
                                         uint32_t *d, const uint32_t *a, const uint32_t *b,
                                         const uint32_t *c) {
  uint32_t repair = 0;
  const unsigned rounding = unsigned((flags & RJ_ARITH_FP_ROUND_MASK) >> 24);
  for (int lane = 0; lane < 32; lane += 4) {
    auto x = input(a + lane, flags, mode), y = input(b + lane, flags, mode >> 1),
         z = input(c + lane, flags, mode >> 2);
    auto product = _mm256_mul_pd(_mm256_cvtps_pd(x), _mm256_cvtps_pd(y));
    auto addend = _mm256_cvtps_pd(z);
    auto sum = _mm256_add_pd(product, addend);
    auto virtual_addend = _mm256_sub_pd(sum, product);
    auto residual = _mm256_add_pd(_mm256_sub_pd(product, _mm256_sub_pd(sum, virtual_addend)),
                                  _mm256_sub_pd(addend, virtual_addend));
    auto nearest = _mm256_cvtpd_ps(sum);
    auto back = _mm256_cvtps_pd(nearest);
    auto same = _mm256_cmp_pd(back, sum, _CMP_EQ_OQ);
    auto below =
        _mm256_or_pd(_mm256_cmp_pd(back, sum, _CMP_LT_OQ),
                     _mm256_and_pd(same, _mm256_cmp_pd(residual, _mm256_setzero_pd(), _CMP_GT_OQ)));
    auto above =
        _mm256_or_pd(_mm256_cmp_pd(back, sum, _CMP_GT_OQ),
                     _mm256_and_pd(same, _mm256_cmp_pd(residual, _mm256_setzero_pd(), _CMP_LT_OQ)));
    auto negative = _mm256_cmp_pd(sum, _mm256_setzero_pd(), _CMP_LT_OQ);
    auto adjust = rounding == 1   ? below
                  : rounding == 2 ? above
                                  : _mm256_blendv_pd(above, below, negative);
    auto mask64 = _mm256_castpd_si256(adjust);
    auto packed = _mm256_permutevar8x32_epi32(mask64, _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6));
    auto mask = _mm256_castsi256_si128(packed);
    auto bits = _mm_castps_si128(nearest);
    auto sign = _mm_srai_epi32(bits, 31);
    auto delta = _mm_or_si128(sign, _mm_set1_epi32(1));
    if (rounding == 2) {
      delta = _mm_sub_epi32(_mm_setzero_si128(), delta);
    }
    if (rounding == 3) {
      delta = _mm_set1_epi32(-1);
    }
    bits = _mm_add_epi32(bits, _mm_and_si128(mask, delta));
    // Repair exceptional inputs, cancellation and the underflow/overflow boundaries.
    auto magnitude = _mm_and_si128(bits, _mm_set1_epi32(INT32_MAX));
    auto bad = _mm_or_si128(_mm_cmpgt_epi32(_mm_set1_epi32(0x00800001), magnitude),
                            _mm_cmpgt_epi32(magnitude, _mm_set1_epi32(0x7f7ffffe)));
    const auto any_nonfinite = [&](auto value) {
      return _mm_cmpgt_epi32(_mm_and_si128(_mm_castps_si128(value), _mm_set1_epi32(INT32_MAX)),
                             _mm_set1_epi32(0x7f7fffff));
    };
    bad = _mm_or_si128(
        bad, _mm_or_si128(any_nonfinite(x), _mm_or_si128(any_nonfinite(y), any_nonfinite(z))));
    repair |= uint32_t(_mm_movemask_ps(_mm_castsi128_ps(bad))) << lane;
    _mm_storeu_si128(reinterpret_cast<__m128i *>(d + lane), bits);
  }
  return repair & exec_mask;
}
} // namespace rocjitsu::arith
