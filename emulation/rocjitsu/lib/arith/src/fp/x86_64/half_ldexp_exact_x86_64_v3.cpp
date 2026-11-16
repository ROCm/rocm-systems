// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 exact FP16 power-of-two scaling with guest rounding and output controls.
// All finite intermediates are exact normal FP32 values; conversion owns rounding.

#include "common/x86_64/half_x86_64_v3.h"
#include "fp/half_exponent.h"
#include "rj_arith/detail/rj_arith_common.h"

#include <immintrin.h>
#include <stdint.h>

namespace {

// Narrow finite values with guest rounding, optionally saturating finite overflow.
__m256i narrow(__m256 value, unsigned rounding, bool saturate) {
  __m128i half;
  switch (rounding) {
  case 1:
    half = _mm256_cvtps_ph(value, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
    break;
  case 2:
    half = _mm256_cvtps_ph(value, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
    break;
  case 3:
    half = _mm256_cvtps_ph(value, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    break;
  default:
    half = _mm256_cvtps_ph(value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    break;
  }
  auto result = _mm256_cvtepu16_epi32(half);
  if (saturate) {
    auto infinity = _mm256_cmpeq_epi32(_mm256_and_si256(result, _mm256_set1_epi32(0x7fff)),
                                       _mm256_set1_epi32(0x7c00));
    result = _mm256_blendv_epi8(result, _mm256_sub_epi32(result, _mm256_set1_epi32(1)), infinity);
  }
  return result;
}

} // namespace

namespace rocjitsu::arith {

// Scale selected halves, preserving inactive lanes and the unselected destination half.
// Requires the named ISA level and validated controls; never changes host FP controls.
void half_ldexp_exact_x86_64_v3(uint64_t flags, uint32_t exec_mask, uint32_t modifiers, uint32_t *d,
                                const uint32_t *a, const uint32_t *b, bool quiet_snan) {
  const int a_shift = modifiers & RJ_ARITH_ALU_HIGH_A ? 16 : 0;
  const int b_shift = modifiers & RJ_ARITH_ALU_HIGH_B ? 16 : 0;
  const int d_shift = modifiers & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
  const unsigned rounding = unsigned((flags & RJ_ARITH_FP_ROUND_MASK) >> 24);
  const unsigned omod = (modifiers >> 6) & 3;
  const bool saturate = flags & RJ_ARITH_FP16_OVFL;
  const float scales[] = {1, 2, 4, 0.5f};
  for (int lane = 0; lane < 32; lane += 8) {
    auto raw = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane));
    auto bits =
        _mm256_and_si256(_mm256_srl_epi32(raw, _mm_cvtsi32_si128(a_shift)),
                         _mm256_set1_epi32(modifiers & RJ_ARITH_ALU_ABS_A ? 0x7fff : 0xffff));
    bits = _mm256_xor_si256(bits, _mm256_set1_epi32(modifiers & RJ_ARITH_ALU_NEG_A ? 0x8000 : 0));
    auto magnitude = _mm256_and_si256(bits, _mm256_set1_epi32(0x7fff));
    auto special = _mm256_cmpgt_epi32(magnitude, _mm256_set1_epi32(0x7bff));
    auto nan = _mm256_cmpgt_epi32(magnitude, _mm256_set1_epi32(0x7c00));
    auto finite = _mm256_blendv_epi8(bits, _mm256_setzero_si256(), special);
    if (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) {
      auto tiny_input = _mm256_cmpgt_epi32(_mm256_set1_epi32(0x400), magnitude);
      finite = _mm256_blendv_epi8(finite, _mm256_and_si256(finite, _mm256_set1_epi32(0x8000)),
                                  tiny_input);
    }
    auto x = half_input(finite, 0, 0);
    auto exponent = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + lane));
    exponent = _mm256_sra_epi32(_mm256_sll_epi32(exponent, _mm_cvtsi32_si128(16 - b_shift)),
                                _mm_cvtsi32_si128(16));
    exponent =
        _mm256_min_epi32(_mm256_max_epi32(exponent, _mm256_set1_epi32(-64)), _mm256_set1_epi32(64));
    auto factor = _mm256_castsi256_ps(
        _mm256_slli_epi32(_mm256_add_epi32(exponent, _mm256_set1_epi32(127)), 23));
    auto value = _mm256_mul_ps(x, factor);
    auto tiny = _mm256_cmpgt_epi32(
        _mm256_set1_epi32(0x38800000),
        _mm256_and_si256(_mm256_castps_si256(value), _mm256_set1_epi32(0x7fffffff)));
    auto result = narrow(value, rounding, saturate);
    if (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) {
      result =
          _mm256_blendv_epi8(result, _mm256_and_si256(result, _mm256_set1_epi32(0x8000)), tiny);
    }
    if (omod) {
      // Scaling follows the first FP16 rounding; it cannot recover overflow or tininess.
      auto rounded = _mm256_blendv_epi8(result, _mm256_setzero_si256(), tiny);
      value = _mm256_mul_ps(half_input(rounded, 0, 0), _mm256_set1_ps(scales[omod]));
      result = narrow(value, rounding, saturate);
      auto tiny_output = _mm256_cmpgt_epi32(
          _mm256_set1_epi32(0x38800000),
          _mm256_and_si256(_mm256_castps_si256(value), _mm256_set1_epi32(0x7fffffff)));
      result = _mm256_blendv_epi8(result, _mm256_and_si256(result, _mm256_set1_epi32(0x8000)),
                                  tiny_output);
    }
    // Original infinities bypass saturation. Quiet NaNs only when required by the architecture and
    // IEEE mode.
    auto exceptional = _mm256_blendv_epi8(
        bits, _mm256_or_si256(bits, _mm256_set1_epi32(quiet_snan ? 0x200 : 0)), nan);
    result = _mm256_blendv_epi8(result, exceptional, special);
    if (modifiers & RJ_ARITH_ALU_CLAMP) {
      auto negative = _mm256_cmpgt_epi32(_mm256_and_si256(result, _mm256_set1_epi32(0x8000)),
                                         _mm256_setzero_si256());
      auto clamped = _mm256_min_epi32(result, _mm256_set1_epi32(0x3c00));
      clamped = _mm256_blendv_epi8(clamped, _mm256_setzero_si256(), negative);
      clamped = _mm256_blendv_epi8(clamped, _mm256_setzero_si256(), nan);
      result =
          (flags & RJ_ARITH_FP_DX10_CLAMP_OFF) ? _mm256_blendv_epi8(clamped, result, nan) : clamped;
    }
    auto old = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(d + lane));
    result = _mm256_or_si256(
        _mm256_sll_epi32(result, _mm_cvtsi32_si128(d_shift)),
        _mm256_and_si256(old, _mm256_set1_epi32(d_shift ? 0xffff : int(0xffff0000U))));
    auto lanes = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                                   _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), lanes, result);
  }
}

} // namespace rocjitsu::arith
