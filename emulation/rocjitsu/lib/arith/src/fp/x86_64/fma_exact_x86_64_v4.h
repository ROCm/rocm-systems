// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 directed arithmetic and narrowing shared by FP32, FP16, and mixed FMA.
// Only target-specific translation units include these vector helpers.

#pragma once

#include "common/x86_64/alu_x86_64_v4.h"
#include "rj_arith/rj_arith.h"
#include <immintrin.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {
namespace {

// Compute vector a*b+c with instruction-local rounding selected by rounding.
[[maybe_unused]] inline __m512 fused(__m512 a, __m512 b, __m512 c, unsigned rounding) {
  switch (rounding) {
  case 1:
    return _mm512_fmadd_round_ps(a, b, c, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
  case 2:
    return _mm512_fmadd_round_ps(a, b, c, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
  case 3:
    return _mm512_fmadd_round_ps(a, b, c, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
  default:
    return _mm512_fmadd_round_ps(a, b, c, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  }
}

// Apply source modifiers and guest input flushing; mark unsupported fast-path lanes in bad.
[[maybe_unused]] inline __m512 prepare_fma_input(__m512i x, uint64_t flags, uint32_t mode, int src,
                                                 __mmask16 &bad) {
  if (mode & (RJ_ARITH_ALU_ABS_A << src)) {
    x = _mm512_and_si512(x, _mm512_set1_epi32(0x7fffffff));
  }
  if (mode & (RJ_ARITH_ALU_NEG_A << src)) {
    x = _mm512_xor_si512(x, _mm512_set1_epi32(0x80000000U));
  }
  auto m = _mm512_and_si512(x, _mm512_set1_epi32(0x7fffffff));
  auto sub = _mm512_cmplt_epu32_mask(m, _mm512_set1_epi32(0x00800000)) &
             _mm512_cmpneq_epi32_mask(m, _mm512_setzero_si512());
  bad |= _mm512_cmpge_epu32_mask(m, _mm512_set1_epi32(0x7f800000));
  if (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) {
    x = _mm512_mask_and_epi32(x, sub, x, _mm512_set1_epi32(0x80000000U));
  } else {
    bad |= sub;
  }
  return _mm512_castsi512_ps(x);
}

// Keep the optional scaling work out of the common FMA body. Besides sharing
// the arithmetic, this avoids register pressure on waves without OMOD.
[[maybe_unused, gnu::noinline]] __m512i scale_fma_result(__m512i u, uint32_t mode,
                                                         unsigned rounding) {
  const unsigned omod = (mode >> 6) & 3;
  u = _mm512_castps_si512(prepare_omod_f32(_mm512_castsi512_ps(u), mode));
  auto magnitude = _mm512_and_si512(u, _mm512_set1_epi32(0x7fffffff));
  auto nonzero = _mm512_cmpneq_epi32_mask(magnitude, _mm512_setzero_si512());
  auto scaled = _mm512_add_epi32(u, _mm512_set1_epi32(omod == 3 ? -0x800000 : int(omod << 23)));
  if (omod != 3) {
    auto overflow =
        _mm512_cmpge_epu32_mask(magnitude, _mm512_set1_epi32(0x7f800000 - (omod << 23)));
    auto negative = _mm512_movepi32_mask(u);
    __mmask16 infinity = rounding == 0   ? __mmask16(~0U)
                         : rounding == 1 ? __mmask16(~negative)
                         : rounding == 2 ? negative
                                         : 0;
    auto limit = _mm512_or_si512(_mm512_and_si512(u, _mm512_set1_epi32(INT32_MIN)),
                                 _mm512_mask_blend_epi32(infinity, _mm512_set1_epi32(0x7f7fffff),
                                                         _mm512_set1_epi32(0x7f800000)));
    scaled = _mm512_mask_mov_epi32(scaled, overflow, limit);
  }
  u = _mm512_mask_mov_epi32(u, nonzero, scaled);
  return u;
}

// A zero FP32 result may be exact cancellation or rounded underflow. FP64
// represents every finite FP32 product exactly, so equality with -C proves
// cancellation. Zero products with zero C need no widening.
[[gnu::always_inline]] inline __mmask16 exact_fma_zeros(__m512 a, __m512 b, __m512 c,
                                                        __mmask16 zeros) {
  const auto z = _mm512_setzero_ps();
  __mmask16 exact = (_mm512_cmp_ps_mask(a, z, _CMP_EQ_OQ) | _mm512_cmp_ps_mask(b, z, _CMP_EQ_OQ)) &
                    _mm512_cmp_ps_mask(c, z, _CMP_EQ_OQ);
  if (!(zeros & ~exact)) {
    return exact;
  }
  for (int half = 0; half < 2; ++half) {
    auto aa = half ? _mm512_extractf32x8_ps(a, 1) : _mm512_castps512_ps256(a);
    auto bb = half ? _mm512_extractf32x8_ps(b, 1) : _mm512_castps512_ps256(b);
    auto cc = half ? _mm512_extractf32x8_ps(c, 1) : _mm512_castps512_ps256(c);
    auto product = _mm512_mul_pd(_mm512_cvtps_pd(aa), _mm512_cvtps_pd(bb));
    auto negative_c = _mm512_castsi512_pd(
        _mm512_xor_si512(_mm512_castpd_si512(_mm512_cvtps_pd(cc)), _mm512_set1_epi64(INT64_MIN)));
    exact |= unsigned(_mm512_cmp_pd_mask(product, negative_c, _CMP_EQ_OQ)) << (half * 8);
  }
  return exact;
}

// Keep this hot helper inline: an out-of-line call measurably slows ordinary FMA.
// The caller repairs flagged nonfinite/tiny cases with the inherited scalar model.
[[gnu::always_inline]] inline __m512i fma_result(__m512 aa, __m512 bb, __m512 cc, uint64_t flags,
                                                 uint32_t mode, __mmask16 &bad) {
  const unsigned rounding = unsigned((flags & RJ_ARITH_FP_ROUND_MASK) >> 24);
  auto result = fused(aa, bb, cc, rounding);
  auto u = _mm512_castps_si512(result), m = _mm512_and_si512(u, _mm512_set1_epi32(0x7fffffff));
  auto zeros = _mm512_cmpeq_epi32_mask(m, _mm512_setzero_si512());
  auto exact_zero = zeros ? exact_fma_zeros(aa, bb, cc, zeros) : __mmask16(0);
  bad |= (_mm512_cmple_epu32_mask(m, _mm512_set1_epi32(0x00800000)) & ~exact_zero) |
         _mm512_cmpge_epu32_mask(m, _mm512_set1_epi32(0x7f800000));
  if (((mode >> 6) & 3) != 0) {
    u = scale_fma_result(u, mode, rounding);
  }
  if (mode & RJ_ARITH_ALU_CLAMP) {
    u = _mm512_mask_mov_epi32(u, _mm512_movepi32_mask(u), _mm512_setzero_si512());
    u = _mm512_min_epu32(u, _mm512_set1_epi32(0x3f800000));
  }
  return u;
}

// Round FP32 vector values to packed FP16 using the supplied guest rounding code.
[[maybe_unused]] inline __m256i narrow_half(__m512 value, unsigned rounding) {
  switch (rounding) {
  case 1:
    return _mm512_cvt_roundps_ph(value, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
  case 2:
    return _mm512_cvt_roundps_ph(value, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
  case 3:
    return _mm512_cvt_roundps_ph(value, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
  default:
    return _mm512_cvt_roundps_ph(value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  }
}

// Directed FP32 bounds give a round-to-odd intermediate for FP16 narrowing.
// Mixed FMA must repair FP32 underflow; half-input FMA cannot reach it.
[[maybe_unused]] inline __m512i half_fma_odd(const __m512 (&values)[3], unsigned rounding,
                                             __m512 &down, __m512 &up) {
  down = _mm512_fmadd_round_ps(values[0], values[1], values[2],
                               _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
  up = _mm512_fmadd_round_ps(values[0], values[1], values[2],
                             _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
  auto di = _mm512_castps_si512(down), ui = _mm512_castps_si512(up);
  auto towardzero = _mm512_mask_blend_epi32(_mm512_movepi32_mask(di), di, ui);
  auto different = _mm512_cmp_ps_mask(down, up, _CMP_NEQ_OQ);
  auto odd = _mm512_mask_or_epi32(towardzero, different, towardzero, _mm512_set1_epi32(1));

  // Cancellation follows guest rounding; matching signed zeros keep their sign.
  auto zero = _mm512_cmp_ps_mask(down, _mm512_setzero_ps(), _CMP_EQ_OQ);
  odd = _mm512_mask_mov_epi32(odd, zero, rounding == 2 ? di : ui);
  return odd;
}

// Identify FP16 results tiny before exponent packing under the guest rounding mode.
[[gnu::always_inline]] inline __mmask16 half_fma_tiny_mask(__m512i half, __m512i odd,
                                                           unsigned rounding) {
  const auto magnitude = _mm512_and_si512(odd, _mm512_set1_epi32(0x7fffffff));
  const auto negative = _mm512_test_epi32_mask(half, _mm512_set1_epi32(0x8000));
  const __mmask16 away = rounding == 1 ? __mmask16(~negative) : rounding == 2 ? negative : 0;

  // Same pre-packing tininess thresholds as half_fma_value. Round-to-odd
  // preserves comparisons with these exactly representable FP32 thresholds.
  auto threshold = _mm512_mask_mov_epi32(_mm512_set1_epi32(rounding == 0 ? 0x387ff000 : 0x38800000),
                                         away, _mm512_set1_epi32(0x387fe000));
  auto tiny = _mm512_cmplt_epu32_mask(magnitude, threshold) |
              (away & _mm512_cmpeq_epi32_mask(magnitude, threshold));
  auto hm = _mm512_and_si512(half, _mm512_set1_epi32(0x7fff));
  return _mm512_cmplt_epu32_mask(hm, _mm512_set1_epi32(0x400)) |
         (tiny & _mm512_cmpeq_epi32_mask(hm, _mm512_set1_epi32(0x400)));
}

// Apply guest overflow, output-denormal, and CLAMP rules to narrowed FP16 FMA results.
[[gnu::always_inline]] inline __m512i finish_half_fma(__m512i half, __m512i odd, uint64_t flags,
                                                      uint32_t mode) {
  const unsigned rounding = unsigned((flags & RJ_ARITH_FP_ROUND_MASK) >> 24);
  if (flags & RJ_ARITH_FP16_OVFL) {
    auto infinity = _mm512_cmpeq_epi32_mask(_mm512_and_si512(half, _mm512_set1_epi32(0x7fff)),
                                            _mm512_set1_epi32(0x7c00));
    half = _mm512_mask_sub_epi32(half, infinity, half, _mm512_set1_epi32(1));
  }
  if (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) {
    auto flush = half_fma_tiny_mask(half, odd, rounding);
    half = _mm512_mask_and_epi32(half, flush, half, _mm512_set1_epi32(0x8000));
  }
  if (mode & RJ_ARITH_ALU_CLAMP) {
    auto negative = _mm512_test_epi32_mask(half, _mm512_set1_epi32(0x8000));
    half = _mm512_mask_mov_epi32(half, negative, _mm512_setzero_si512());
    half = _mm512_min_epu32(half, _mm512_set1_epi32(0x3c00));
  }
  return half;
}

} // namespace
} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
