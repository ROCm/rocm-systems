// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Shared AVX-512 FP64 output scaling, directed overflow, CLAMP and denormal policy.
#pragma once

#include "common/isa.h"
#include "rj_arith/detail/rj_arith_common.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Scale encoded normal values, selecting the guest-directed overflow encoding.
// OMOD flushes input/output subnormals independently of the guest denormal mode.
inline __m512d fp64_output_x86_64_v4(__m512d value, uint32_t instruction_flags, uint64_t flags,
                                     unsigned rounding, bool native_math = false) {
  auto bits = _mm512_castpd_si512(value);
  auto sign = _mm512_and_si512(bits, _mm512_set1_epi64(INT64_MIN));
  auto exponent = _mm512_and_si512(_mm512_srli_epi64(bits, 52), _mm512_set1_epi64(2047));
  unsigned omod = (instruction_flags >> 6) & 3;
  if (omod && native_math) {
    // Preserve legacy staging: a subnormal core result may become normal after scaling.
    const double scales[] = {1, 2, 4, 0.5};
    value = _mm512_mul_pd(value, _mm512_set1_pd(scales[omod]));
    bits = _mm512_castpd_si512(value);
    auto tiny = _mm512_cmpeq_epi64_mask(
        _mm512_and_si512(bits, _mm512_set1_epi64(0x7ff0000000000000LL)), _mm512_setzero_si512());
    bits = _mm512_mask_mov_epi64(bits, tiny, _mm512_setzero_si512());
  } else if (omod) {
    auto normal = _mm512_cmpneq_epi64_mask(exponent, _mm512_setzero_si512()) &
                  _mm512_cmpneq_epi64_mask(exponent, _mm512_set1_epi64(2047));
    auto adjusted = _mm512_add_epi64(exponent, _mm512_set1_epi64(omod == 3 ? -1 : int(omod)));
    auto scaled =
        _mm512_or_si512(_mm512_and_si512(bits, _mm512_set1_epi64(INT64_MIN | 0x000fffffffffffffLL)),
                        _mm512_slli_epi64(adjusted, 52));
    bits = _mm512_mask_mov_epi64(bits, normal, scaled);
    bits = _mm512_mask_mov_epi64(bits, _mm512_cmpeq_epi64_mask(exponent, _mm512_setzero_si512()),
                                 _mm512_setzero_si512());
    bits = _mm512_mask_mov_epi64(
        bits, normal & _mm512_cmp_epi64_mask(adjusted, _mm512_setzero_si512(), _MM_CMPINT_LE),
        sign);
    auto negative = _mm512_cmpneq_epi64_mask(sign, _mm512_setzero_si512());
    __mmask8 infinity = rounding == 0   ? __mmask8(0xff)
                        : rounding == 1 ? __mmask8(~negative)
                        : rounding == 2 ? negative
                                        : __mmask8(0);
    auto overflow = _mm512_mask_blend_epi64(infinity, _mm512_set1_epi64(0x7fefffffffffffffLL),
                                            _mm512_set1_epi64(0x7ff0000000000000LL));
    bits = _mm512_mask_mov_epi64(
        bits, normal & _mm512_cmp_epi64_mask(adjusted, _mm512_set1_epi64(2047), _MM_CMPINT_GE),
        _mm512_or_si512(sign, overflow));
  }
  value = _mm512_castsi512_pd(bits);
  if (instruction_flags & RJ_ARITH_ALU_CLAMP) {
    auto clamped = _mm512_min_pd(_mm512_max_pd(value, _mm512_setzero_pd()), _mm512_set1_pd(1));
    value = flags & RJ_ARITH_FP_DX10_CLAMP_OFF
                ? _mm512_mask_mov_pd(clamped, _mm512_cmp_pd_mask(value, value, _CMP_UNORD_Q), value)
                : clamped;
  }
  if (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) {
    bits = _mm512_castpd_si512(value);
    auto tiny = _mm512_cmpeq_epi64_mask(
        _mm512_and_si512(bits, _mm512_set1_epi64(0x7ff0000000000000LL)), _mm512_setzero_si512());
    value =
        _mm512_castsi512_pd(_mm512_mask_and_epi64(bits, tiny, bits, _mm512_set1_epi64(INT64_MIN)));
  }
  return value;
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
