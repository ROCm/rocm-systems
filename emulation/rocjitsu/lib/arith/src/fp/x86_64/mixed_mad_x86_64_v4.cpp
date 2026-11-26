// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// V4 mixed MAD with an FP32 FMA intermediate and directed half narrowing.
#include "fp/x86_64/mixed_mad_x86_64.h"

namespace rocjitsu::arith {

// Execute one Wave32 block, preserving inactive lanes and the other output half.
void mixed_mad_x86_64_v4(MixedFma destination, uint64_t flags, uint32_t exec_mask, uint32_t mode,
                         uint32_t *d, const uint32_t *a, const uint32_t *b, const uint32_t *c) {
  for (int lane = 0; lane < 32; lane += 16) {
    auto x = mixed_mad_input(a + lane, mode);
    auto y = mixed_mad_input(b + lane, mode >> 1);
    auto z = mixed_mad_input(c + lane, mode >> 2);
    auto xh = mixed_mad_input(a + lane + 8, mode);
    auto yh = mixed_mad_input(b + lane + 8, mode >> 1);
    auto zh = mixed_mad_input(c + lane + 8, mode >> 2);
    auto wide_x = _mm512_insertf32x8(_mm512_castps256_ps512(x), xh, 1);
    auto wide_y = _mm512_insertf32x8(_mm512_castps256_ps512(y), yh, 1);
    auto wide_z = _mm512_insertf32x8(_mm512_castps256_ps512(z), zh, 1);
    auto value = _mm512_fmadd_ps(wide_x, wide_y, wide_z);
    auto low = mixed_mad_finish(destination, flags, mode, _mm512_castps512_ps256(value), x, y, z,
                                a + lane, b + lane, c + lane);
    auto high = mixed_mad_finish(destination, flags, mode, _mm512_extractf32x8_ps(value, 1), xh, yh,
                                 zh, a + lane + 8, b + lane + 8, c + lane + 8);
    low = mixed_mad_merge(destination, low, d + lane);
    high = mixed_mad_merge(destination, high, d + lane + 8);
    auto result = _mm512_inserti64x4(_mm512_castsi256_si512(low), high, 1);
    _mm512_mask_storeu_epi32(d + lane, __mmask16(exec_mask >> lane), result);
  }
}

// Narrow guest-rounded FP32 results without repeating their arithmetic.
void mixed_narrow_x86_64_v4(MixedFma destination, uint64_t flags, uint32_t exec_mask, uint32_t *d,
                            const uint32_t *source) {
  for (int lane = 0; lane < 32; lane += 16) {
    auto value = _mm512_castsi512_ps(_mm512_loadu_si512(source + lane));
    __m256i half;
    switch ((flags & RJ_ARITH_FP_ROUND_MASK) >> 24) {
    case 1:
      half = _mm512_cvtps_ph(value, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
      break;
    case 2:
      half = _mm512_cvtps_ph(value, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
      break;
    case 3:
      half = _mm512_cvtps_ph(value, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
      break;
    default:
      half = _mm512_cvtps_ph(value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
      break;
    }
    auto result = _mm512_cvtepu16_epi32(half);
    if (flags & RJ_ARITH_FP16_OVFL) {
      auto finite = ~_mm512_fpclass_ps_mask(value, 0x99);
      auto inf = _mm512_cmpeq_epi32_mask(_mm512_and_si512(result, _mm512_set1_epi32(0x7fff)),
                                         _mm512_set1_epi32(0x7c00));
      result = _mm512_mask_sub_epi32(result, __mmask16(finite & inf), result, _mm512_set1_epi32(1));
    }
    auto old = _mm512_loadu_si512(d + lane);
    if (destination == MixedFma::High) {
      result = _mm512_or_si512(_mm512_and_si512(old, _mm512_set1_epi32(0xffff)),
                               _mm512_slli_epi32(result, 16));
    } else if (destination == MixedFma::Low) {
      result = _mm512_or_si512(_mm512_and_si512(old, _mm512_set1_epi32(-65536)), result);
    }
    _mm512_mask_storeu_epi32(d + lane, __mmask16(exec_mask >> lane), result);
  }
}

} // namespace rocjitsu::arith
