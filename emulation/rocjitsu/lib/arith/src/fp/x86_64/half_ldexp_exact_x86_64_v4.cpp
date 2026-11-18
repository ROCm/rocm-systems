// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 exact FP16 power-of-two scaling with guest rounding and output controls.
// All finite intermediates are exact normal FP32 values; conversion owns rounding.

#include "common/x86_64/half_x86_64_v4.h"
#include "fp/half_exponent.h"
#include "rj_arith/detail/rj_arith_common.h"

#include <immintrin.h>
#include <stdint.h>

namespace {

// Narrow finite values with guest rounding, optionally saturating finite overflow.
__m512i narrow(__m512 value, unsigned rounding, bool saturate) {
  __m256i half;
  switch (rounding) {
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
  if (saturate) {
    auto infinity = _mm512_cmpeq_epi32_mask(_mm512_and_si512(result, _mm512_set1_epi32(0x7fff)),
                                            _mm512_set1_epi32(0x7c00));
    result =
        _mm512_mask_blend_epi32(infinity, result, _mm512_sub_epi32(result, _mm512_set1_epi32(1)));
  }
  return result;
}

} // namespace

namespace rocjitsu::arith {

// Scale selected halves, preserving inactive lanes and the unselected destination half.
// Requires the named ISA level and validated controls; never changes host FP controls.
void half_ldexp_exact_x86_64_v4(uint64_t flags, uint32_t exec_mask, uint32_t modifiers, uint32_t *d,
                                const uint32_t *a, const uint32_t *b, bool quiet_snan) {
  const int a_shift = modifiers & RJ_ARITH_ALU_HIGH_A ? 16 : 0;
  const int b_shift = modifiers & RJ_ARITH_ALU_HIGH_B ? 16 : 0;
  const int d_shift = modifiers & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
  const unsigned rounding = unsigned((flags & RJ_ARITH_FP_ROUND_MASK) >> 24);
  const unsigned omod = (modifiers >> 6) & 3;
  const bool saturate = flags & RJ_ARITH_FP16_OVFL;
  const float scales[] = {1, 2, 4, 0.5f};
  for (int lane = 0; lane < 32; lane += 16) {
    auto raw = _mm512_loadu_si512(a + lane);
    auto bits =
        _mm512_and_si512(_mm512_srl_epi32(raw, _mm_cvtsi32_si128(a_shift)),
                         _mm512_set1_epi32(modifiers & RJ_ARITH_ALU_ABS_A ? 0x7fff : 0xffff));
    bits = _mm512_xor_si512(bits, _mm512_set1_epi32(modifiers & RJ_ARITH_ALU_NEG_A ? 0x8000 : 0));
    auto magnitude = _mm512_and_si512(bits, _mm512_set1_epi32(0x7fff));
    auto special = _mm512_cmpgt_epi32_mask(magnitude, _mm512_set1_epi32(0x7bff));
    auto nan = _mm512_cmpgt_epi32_mask(magnitude, _mm512_set1_epi32(0x7c00));
    auto finite = _mm512_mask_blend_epi32(special, bits, _mm512_setzero_si512());
    if (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) {
      auto tiny_input = _mm512_cmplt_epi32_mask(magnitude, _mm512_set1_epi32(0x400));
      finite = _mm512_mask_blend_epi32(tiny_input, finite,
                                       _mm512_and_si512(finite, _mm512_set1_epi32(0x8000)));
    }
    auto x = half_input(finite, 0, 0);
    auto exponent = _mm512_loadu_si512(b + lane);
    exponent = _mm512_sra_epi32(_mm512_sll_epi32(exponent, _mm_cvtsi32_si128(16 - b_shift)),
                                _mm_cvtsi32_si128(16));
    exponent =
        _mm512_min_epi32(_mm512_max_epi32(exponent, _mm512_set1_epi32(-64)), _mm512_set1_epi32(64));
    auto factor = _mm512_castsi512_ps(
        _mm512_slli_epi32(_mm512_add_epi32(exponent, _mm512_set1_epi32(127)), 23));
    auto value = _mm512_mul_ps(x, factor);
    auto tiny = _mm512_cmplt_epi32_mask(
        _mm512_and_si512(_mm512_castps_si512(value), _mm512_set1_epi32(0x7fffffff)),
        _mm512_set1_epi32(0x38800000));
    auto result = narrow(value, rounding, saturate);
    if (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) {
      result = _mm512_mask_blend_epi32(tiny, result,
                                       _mm512_and_si512(result, _mm512_set1_epi32(0x8000)));
    }
    if (omod) {
      // Scaling follows the first FP16 rounding; it cannot recover overflow or tininess.
      auto rounded = _mm512_mask_blend_epi32(tiny, result, _mm512_setzero_si512());
      value = _mm512_mul_ps(half_input(rounded, 0, 0), _mm512_set1_ps(scales[omod]));
      result = narrow(value, rounding, saturate);
      auto tiny_output = _mm512_cmplt_epi32_mask(
          _mm512_and_si512(_mm512_castps_si512(value), _mm512_set1_epi32(0x7fffffff)),
          _mm512_set1_epi32(0x38800000));
      result = _mm512_mask_blend_epi32(tiny_output, result,
                                       _mm512_and_si512(result, _mm512_set1_epi32(0x8000)));
    }
    // Original infinities bypass saturation. Quiet NaNs only when required by the architecture and
    // IEEE mode.
    auto exceptional = _mm512_mask_blend_epi32(
        nan, bits, _mm512_or_si512(bits, _mm512_set1_epi32(quiet_snan ? 0x200 : 0)));
    result = _mm512_mask_blend_epi32(special, result, exceptional);
    if (modifiers & RJ_ARITH_ALU_CLAMP) {
      auto negative = _mm512_cmpgt_epi32_mask(_mm512_and_si512(result, _mm512_set1_epi32(0x8000)),
                                              _mm512_setzero_si512());
      auto clamped = _mm512_min_epi32(result, _mm512_set1_epi32(0x3c00));
      clamped = _mm512_mask_blend_epi32(negative, clamped, _mm512_setzero_si512());
      clamped = _mm512_mask_blend_epi32(nan, clamped, _mm512_setzero_si512());
      result = (flags & RJ_ARITH_FP_DX10_CLAMP_OFF) ? _mm512_mask_blend_epi32(nan, clamped, result)
                                                    : clamped;
    }
    auto old = _mm512_loadu_si512(d + lane);
    result = _mm512_or_si512(
        _mm512_sll_epi32(result, _mm_cvtsi32_si128(d_shift)),
        _mm512_and_si512(old, _mm512_set1_epi32(d_shift ? 0xffff : int(0xffff0000U))));
    _mm512_mask_storeu_epi32(d + lane, __mmask16(exec_mask >> lane), result);
  }
}

} // namespace rocjitsu::arith
