// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 ordinary and packed FP16 arithmetic with guest rounding and modifiers.
// Compatibility retains the inherited intermediate rounding and NaN priority.

#include "common/packed_alu.h"
#include "common/x86_64/half_x86_64_v4.h"
#include "fp/half_binary.h"
#include "rj_arith/detail/rj_arith_common.h"

#include <immintrin.h>
#include <stdint.h>

namespace {

// SIMD equivalent of half_rounds_below for a positive power-of-two FP32 limit.
// An FP16 ULP below the boundary spans 0x2000 FP32 encodings; away rounding
// includes that boundary, hence the +1 encoded in 0x1fff.
__mmask16 rounds_below(__m512 value, int limit, unsigned rounding) {
  auto bits = _mm512_castps_si512(value);
  auto threshold = _mm512_set1_epi32(limit - (rounding == 0 ? 0x1000 : 0));
  if (rounding == 1 || rounding == 2) {
    auto negative = _mm512_cmplt_epi32_mask(bits, _mm512_setzero_si512());
    auto away = rounding == 2 ? negative : __mmask16(~negative);
    threshold = _mm512_mask_mov_epi32(threshold, away, _mm512_set1_epi32(limit - 0x1fff));
  }
  return _mm512_cmplt_epi32_mask(_mm512_and_si512(bits, _mm512_set1_epi32(INT32_MAX)), threshold);
}

// Round a sum toward the guest direction, including the sign of cancellation zero.
__m512 add(__m512 x, __m512 y, unsigned rounding) {
  switch (rounding) {
  case 1:
    return _mm512_add_round_ps(x, y, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
  case 2:
    return _mm512_add_round_ps(x, y, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
  case 3:
    return _mm512_add_round_ps(x, y, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
  default:
    return _mm512_add_round_ps(x, y, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  }
}

// Narrow with guest rounding and finite overflow saturation; retain NaN payloads.
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
    auto magnitude = _mm512_and_si512(_mm512_castps_si512(value), _mm512_set1_epi32(0x7fffffff));
    auto finite = _mm512_cmplt_epi32_mask(magnitude, _mm512_set1_epi32(0x7f800000));
    auto infinity = _mm512_cmpeq_epi32_mask(_mm512_and_si512(result, _mm512_set1_epi32(0x7fff)),
                                            _mm512_set1_epi32(0x7c00));
    result = _mm512_mask_sub_epi32(result, finite & infinity, result, _mm512_set1_epi32(1));
  }
  return result;
}

// Flush promoted FP16 subnormal inputs while retaining zero signs.
__m512 flush(__m512 value) {
  auto bits = _mm512_castps_si512(value);
  auto tiny = _mm512_cmplt_epi32_mask(_mm512_and_si512(bits, _mm512_set1_epi32(0x7fffffff)),
                                      _mm512_set1_epi32(0x38800000));
  return _mm512_castsi512_ps(
      _mm512_mask_and_epi32(bits, tiny, bits, _mm512_set1_epi32(0x80000000U)));
}

} // namespace

namespace rocjitsu::arith {

// Evaluate ordinary/packed half arithmetic without scalar fallbacks for guest modes.
// Requires validated controls and AVX-512; inactive and unselected storage is preserved.
template <Binary Op, bool Packed>
void half_binary_x86_64_v4(uint64_t flags, uint32_t exec_mask, uint32_t instruction_flags,
                           uint32_t *d, const uint32_t *a, const uint32_t *b, bool compatibility,
                           bool preserve_input_snan, HalfOutputPolicy output) {
  const unsigned rounding = unsigned((flags & RJ_ARITH_FP_ROUND_MASK) >> 24);
  const bool saturate = flags & RJ_ARITH_FP16_OVFL;
  const int d_shift = instruction_flags & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
  const float scales[] = {1, 2, 4, 0.5f};
  for (int lane = 0; lane < 32; lane += 16) {
    auto a_words = _mm512_loadu_si512(a + lane), b_words = _mm512_loadu_si512(b + lane);
    auto result = _mm512_setzero_si512();
    for (int half = 0; half < (Packed ? 2 : 1); ++half) {
      const uint32_t m =
          Packed ? packed_half_instruction_flags(instruction_flags, half != 0) : instruction_flags;
      auto x = half_input(a_words, m & RJ_ARITH_ALU_HIGH_A ? 16 : 0, m);
      auto y = half_input(b_words, m & RJ_ARITH_ALU_HIGH_B ? 16 : 0, m >> 1);
      if (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) {
        x = flush(x);
        y = flush(y);
      }
      if constexpr (Op == Binary::Sub) {
        y = _mm512_xor_ps(y, _mm512_set1_ps(-0.0f));
      }
      if constexpr (Op == Binary::Subrev) {
        x = _mm512_xor_ps(x, _mm512_set1_ps(-0.0f));
      }
      // Half products are exact in FP32. Directed sums and final narrowing use
      // the same direction, so an inexact FP32 sum cannot cross a half boundary.
      __m512 value;
      if constexpr (Op == Binary::Mul) {
        value = _mm512_mul_ps(x, y);
      } else {
        value = add(x, y, rounding);
      }
      if (compatibility) {
        value = _mm512_mask_mov_ps(value, _mm512_cmp_ps_mask(y, y, _CMP_UNORD_Q), y);
        value = _mm512_mask_mov_ps(value, _mm512_cmp_ps_mask(x, x, _CMP_UNORD_Q), x);
      }
      const unsigned omod = (m >> 6) & 3;
      // Classify before narrowing: a subnormal-grid round-up can still be tiny.
      auto tiny = (omod || (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS))
                      ? rounds_below(value, 0x38800000, rounding)
                      : __mmask16(0);
      if (output.rounded_tininess && (omod || (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS))) {
        auto encoded = narrow(value, rounding, saturate);
        tiny = _mm512_cmplt_epi32_mask(_mm512_and_si512(encoded, _mm512_set1_epi32(0x7fff)),
                                       _mm512_set1_epi32(0x400));
      }
      if (omod) {
        if (compatibility && !Packed && !output.scale_before_rounding) {
          value = half_input(narrow(value, rounding, saturate), 0, 0);
        }
        auto bits = _mm512_castps_si512(value);

        if (omod == 3 && !output.scale_before_rounding) {
          auto underflow = rounds_below(value, 0x39000000, rounding);
          bits = _mm512_mask_and_epi32(bits, underflow, bits, _mm512_set1_epi32(0x80000000U));
        }
        if (!output.scale_before_rounding) {
          bits = _mm512_mask_mov_epi32(bits, tiny, _mm512_setzero_si512());
        }
        value = _mm512_mul_ps(_mm512_castsi512_ps(bits), _mm512_set1_ps(scales[omod]));
      }
      if (m & RJ_ARITH_ALU_CLAMP) {
        auto clamped = _mm512_min_ps(_mm512_max_ps(value, _mm512_setzero_ps()), _mm512_set1_ps(1));
        value =
            flags & RJ_ARITH_FP_DX10_CLAMP_OFF
                ? _mm512_mask_mov_ps(clamped, _mm512_cmp_ps_mask(value, value, _CMP_UNORD_Q), value)
                : clamped;
      }
      auto encoded = narrow(value, rounding, saturate);
      {
        if (preserve_input_snan &&
            (!(m & RJ_ARITH_ALU_CLAMP) || (flags & RJ_ARITH_FP_DX10_CLAMP_OFF))) {
          // Conversion quiets sNaNs; restore the selected source with its input modifiers.
          auto raw_half = [](__m512i words, int shift, bool abs, bool negate) {
            auto bits = _mm512_and_si512(_mm512_srl_epi32(words, _mm_cvtsi32_si128(shift)),
                                         _mm512_set1_epi32(abs ? 0x7fff : 0xffff));
            return _mm512_xor_si512(bits, _mm512_set1_epi32(negate ? 0x8000 : 0));
          };
          auto raw_x = raw_half(a_words, m & RJ_ARITH_ALU_HIGH_A ? 16 : 0, m & RJ_ARITH_ALU_ABS_A,
                                bool(m & RJ_ARITH_ALU_NEG_A) != (Op == Binary::Subrev));
          auto raw_y = raw_half(b_words, m & RJ_ARITH_ALU_HIGH_B ? 16 : 0, m & RJ_ARITH_ALU_ABS_B,
                                bool(m & RJ_ARITH_ALU_NEG_B) != (Op == Binary::Sub));
          auto nan_x = _mm512_cmpgt_epi32_mask(_mm512_and_si512(raw_x, _mm512_set1_epi32(0x7fff)),
                                               _mm512_set1_epi32(0x7c00));
          auto nan_y = _mm512_cmpgt_epi32_mask(_mm512_and_si512(raw_y, _mm512_set1_epi32(0x7fff)),
                                               _mm512_set1_epi32(0x7c00));
          encoded = _mm512_mask_mov_epi32(encoded, nan_y, raw_y);
          encoded = _mm512_mask_mov_epi32(encoded, nan_x, raw_x);
        }
      }
      if (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) {

        if (output.rounded_tininess) {
          tiny = _mm512_cmplt_epi32_mask(_mm512_and_si512(encoded, _mm512_set1_epi32(0x7fff)),
                                         _mm512_set1_epi32(0x400));
        }
        encoded = _mm512_mask_and_epi32(encoded, tiny, encoded, _mm512_set1_epi32(0x8000));
      }
      if (output.scale_before_rounding && omod) {
        auto tiny_result = _mm512_cmplt_epi32_mask(
            _mm512_and_si512(encoded, _mm512_set1_epi32(0x7fff)), _mm512_set1_epi32(0x400));
        encoded = _mm512_mask_mov_epi32(encoded, tiny_result, _mm512_setzero_si512());
      }
      result = _mm512_or_si512(
          result, _mm512_sll_epi32(encoded, _mm_cvtsi32_si128(Packed ? half * 16 : d_shift)));
    }
    if constexpr (!Packed) {
      auto old = _mm512_loadu_si512(d + lane);
      result = _mm512_or_si512(
          result, _mm512_and_si512(old, _mm512_set1_epi32(d_shift ? 0xffff : int(0xffff0000U))));
    }
    _mm512_mask_storeu_epi32(d + lane, __mmask16(exec_mask >> lane), result);
  }
}

template void half_binary_x86_64_v4<Binary::Add, false>(uint64_t, uint32_t, uint32_t, uint32_t *,
                                                        const uint32_t *, const uint32_t *, bool,
                                                        bool, HalfOutputPolicy);
template void half_binary_x86_64_v4<Binary::Sub, false>(uint64_t, uint32_t, uint32_t, uint32_t *,
                                                        const uint32_t *, const uint32_t *, bool,
                                                        bool, HalfOutputPolicy);
template void half_binary_x86_64_v4<Binary::Subrev, false>(uint64_t, uint32_t, uint32_t, uint32_t *,
                                                           const uint32_t *, const uint32_t *, bool,
                                                           bool, HalfOutputPolicy);
template void half_binary_x86_64_v4<Binary::Mul, false>(uint64_t, uint32_t, uint32_t, uint32_t *,
                                                        const uint32_t *, const uint32_t *, bool,
                                                        bool, HalfOutputPolicy);
template void half_binary_x86_64_v4<Binary::Add, true>(uint64_t, uint32_t, uint32_t, uint32_t *,
                                                       const uint32_t *, const uint32_t *, bool,
                                                       bool, HalfOutputPolicy);
template void half_binary_x86_64_v4<Binary::Mul, true>(uint64_t, uint32_t, uint32_t, uint32_t *,
                                                       const uint32_t *, const uint32_t *, bool,
                                                       bool, HalfOutputPolicy);

} // namespace rocjitsu::arith
