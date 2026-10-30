// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 exact FP32 FMA fast paths using instruction-local rounding and scalar repair.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "common/simd.h"
#include "common/x86_64/alu_x86_64_v4.h"
#include "fp/scalar_repair.h"
#include "rj_arith/rj_arith.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {

namespace {

// Compute vector a*b+c with instruction-local rounding selected by rounding.
__m512 fused(__m512 a, __m512 b, __m512 c, unsigned rounding) {
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
__m512 prepare_fma_input(__m512i x, uint64_t flags, uint32_t instruction_flags, int src,
                         __mmask16 &bad) {
  if (instruction_flags & (RJ_ARITH_ALU_ABS_A << src)) {
    x = _mm512_and_si512(x, _mm512_set1_epi32(0x7fffffff));
  }
  if (instruction_flags & (RJ_ARITH_ALU_NEG_A << src)) {
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
[[gnu::noinline]] __m512i scale_fma_result(__m512i u, uint32_t instruction_flags,
                                           unsigned rounding) {
  const unsigned omod = (instruction_flags >> 6) & 3;
  u = _mm512_castps_si512(prepare_omod_f32(_mm512_castsi512_ps(u), instruction_flags));
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
                                                 uint32_t instruction_flags, __mmask16 &bad) {
  const unsigned rounding = unsigned((flags & RJ_ARITH_FP_ROUND_MASK) >> 24);
  auto result = fused(aa, bb, cc, rounding);
  auto u = _mm512_castps_si512(result), m = _mm512_and_si512(u, _mm512_set1_epi32(0x7fffffff));
  auto zeros = _mm512_cmpeq_epi32_mask(m, _mm512_setzero_si512());
  auto exact_zero = zeros ? exact_fma_zeros(aa, bb, cc, zeros) : __mmask16(0);
  bad |= (_mm512_cmple_epu32_mask(m, _mm512_set1_epi32(0x00800000)) & ~exact_zero) |
         _mm512_cmpge_epu32_mask(m, _mm512_set1_epi32(0x7f800000));
  if (((instruction_flags >> 6) & 3) != 0) {
    u = scale_fma_result(u, instruction_flags, rounding);
  }
  if (instruction_flags & RJ_ARITH_ALU_CLAMP) {
    u = _mm512_mask_mov_epi32(u, _mm512_movepi32_mask(u), _mm512_setzero_si512());
    u = _mm512_min_epu32(u, _mm512_set1_epi32(0x3f800000));
  }
  return u;
}

} // namespace

namespace {

// Keep the inherited scalar machinery out of the normal SIMD loop.
[[gnu::noinline]] __m512i repair_fma(__m512i result, __mmask16 bad, uint64_t flags,
                                     uint32_t instruction_flags, const uint32_t *a,
                                     const uint32_t *b, const uint32_t *c, bool rdna3) {
  uint32_t repaired[16];
  _mm512_storeu_si512(repaired, result);
  for (int i = 0; i < 16; ++i) {
    if ((bad >> i) & 1) {
      repaired[i] = repair_fma_f32(a[i], b[i], c[i], flags, instruction_flags, rdna3);
    }
  }
  return _mm512_loadu_si512(repaired);
}

} // namespace

// In the admitted normal-or-exact-zero domain, locally rounded FMA matches the
// inherited rocjitsu model. Only special/tiny lanes use scalar repair.
// Stage both halves before stores to preserve aliases during repair.
void fma_exact_x86_64_v4(uint64_t flags, uint32_t exec_mask, uint32_t instruction_flags,
                         uint32_t *d, const uint32_t *a, const uint32_t *b, const uint32_t *c,
                         bool rdna3) {
  // Stage both vector chunks before destination writes; repaired lanes must still see original
  // inputs.
  __m512i out[2];
  for (int k = 0; k < 2; ++k) {
    __mmask16 bad = 0;
    __m512 aa = prepare_fma_input(_mm512_loadu_si512(a + 16 * k), flags, instruction_flags, 0, bad);
    __m512 bb = prepare_fma_input(_mm512_loadu_si512(b + 16 * k), flags, instruction_flags, 1, bad);
    __m512 cc = prepare_fma_input(_mm512_loadu_si512(c + 16 * k), flags, instruction_flags, 2, bad);

    out[k] = fma_result(aa, bb, cc, flags, instruction_flags, bad);
    if (bad) {
      out[k] = repair_fma(out[k], bad, flags, instruction_flags, a + 16 * k, b + 16 * k, c + 16 * k,
                          rdna3);
    }
  }
  _mm512_mask_storeu_epi32(d, __mmask16(exec_mask), out[0]);
  _mm512_mask_storeu_epi32(d + 16, __mmask16(exec_mask >> 16), out[1]);
}

} // namespace rocjitsu::arith
