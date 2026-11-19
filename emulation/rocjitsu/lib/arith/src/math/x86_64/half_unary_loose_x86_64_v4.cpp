// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 Loose half reciprocal, roots and EXP/LOG through FP32 arithmetic.
// Keeps selected halves, modifiers and masked publication in vector registers.

#include "common/x86_64/half_x86_64_v4.h"
#include "math/half_unary.h"
#include "math/x86_64/log_exp_loose_x86_64_v4.h"
#include "rj_arith/detail/rj_arith_math.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {

// Evaluate half RCP/RSQ/SQRT/EXP/LOG using FP32 SIMD under default host controls.
// Requires validated controls and v4; preserve inactive lanes and the unused half.
template <Unary Op>
void half_unary_loose_x86_64_v4(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                                const uint32_t *a) {
  static_assert(half_unary_rcp_rsq_sqrt<Op> || Op == Unary::Exp || Op == Unary::Log);
  const bool saturate = flags & RJ_ARITH_FP16_OVFL;
  const unsigned omod = (mode >> 6) & 3;
  const auto sign = _mm512_set1_epi32(INT32_MIN);
  for (int lane = 0; lane < 32; lane += 16) {
    auto x = half_input(_mm512_loadu_si512(a + lane), mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0, mode);
    if (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) {
      auto bits = _mm512_castps_si512(x);
      auto tiny =
          _mm512_cmplt_epi32_mask(_mm512_andnot_si512(sign, bits), _mm512_set1_epi32(0x38800000));
      x = _mm512_castsi512_ps(_mm512_mask_and_epi32(bits, tiny, bits, sign));
    }
    __m512 value;
    if constexpr (Op == Unary::Rcp) {
      value = _mm512_div_ps(_mm512_set1_ps(1), x);
    } else if constexpr (Op == Unary::Rsq) {
      value = _mm512_div_ps(_mm512_set1_ps(1), _mm512_sqrt_ps(x));
    } else if constexpr (Op == Unary::Sqrt) {
      value = _mm512_sqrt_ps(x);
    } else if constexpr (Op == Unary::Exp) {
      value = half_exp2_loose(x);
    } else {
      value = half_log2_loose(x);
      if (saturate) {
        value = _mm512_mask_mov_ps(value, _mm512_cmp_ps_mask(x, _mm512_setzero_ps(), _CMP_EQ_OQ),
                                   _mm512_set1_ps(-65504));
      }
    }
    if constexpr (Op == Unary::Exp || Op == Unary::Log) {
      // EXP/LOG flush their rounded half result; reuse that conversion for OMOD.
      if (omod || (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
        auto rounded = half_narrow(value, saturate);
        if (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) {
          auto tiny = _mm512_cmplt_epi32_mask(_mm512_and_si512(rounded, _mm512_set1_epi32(0x7fff)),
                                              _mm512_set1_epi32(0x400));
          rounded = _mm512_mask_and_epi32(rounded, tiny, rounded, _mm512_set1_epi32(0x8000));
        }
        value = _mm512_cvtph_ps(_mm512_cvtepi32_epi16(rounded));
      }
    } else if (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) {
      auto bits = _mm512_castps_si512(value);
      auto tiny =
          _mm512_cmplt_epi32_mask(_mm512_andnot_si512(sign, bits), _mm512_set1_epi32(0x38800000));
      value = _mm512_castsi512_ps(_mm512_mask_and_epi32(bits, tiny, bits, sign));
    }
    if constexpr (Op == Unary::Rcp || Op == Unary::Rsq) {
      if (saturate) {
        // A zero (including a flushed input) saturates to signed maximum finite.
        auto maximum = _mm512_or_si512(_mm512_set1_epi32(0x477fe000),
                                       _mm512_and_si512(_mm512_castps_si512(value), sign));
        value = _mm512_mask_mov_ps(value, _mm512_cmp_ps_mask(x, _mm512_setzero_ps(), _CMP_EQ_OQ),
                                   _mm512_castsi512_ps(maximum));
      }
    }
    if (omod) {
      // Match the existing Loose path's rounded overflow unless CLAMP discards it.
      if (half_unary_rcp_rsq_sqrt<Op> && !(mode & RJ_ARITH_ALU_CLAMP)) {
        value = _mm512_cvtph_ps(_mm512_cvtepi32_epi16(half_narrow(value, saturate)));
      }
      auto bits = _mm512_castps_si512(value);
      auto magnitude = _mm512_andnot_si512(sign, bits);
      auto tiny = _mm512_cmplt_epi32_mask(magnitude, _mm512_set1_epi32(0x387ff000));
      if (omod == 3) {
        auto underflow = _mm512_cmplt_epi32_mask(magnitude, _mm512_set1_epi32(0x38fff000));
        bits = _mm512_mask_and_epi32(bits, underflow, bits, sign);
      }
      value = _mm512_castsi512_ps(_mm512_mask_mov_epi32(bits, tiny, _mm512_setzero_si512()));
      value = _mm512_mul_ps(value, _mm512_set1_ps(omod == 3 ? 0.5f : omod == 1 ? 2.0f : 4.0f));
    }
    if (mode & RJ_ARITH_ALU_CLAMP) {
      if (flags & RJ_ARITH_FP_DX10_CLAMP_OFF) {
        value = _mm512_min_ps(_mm512_set1_ps(1), _mm512_max_ps(_mm512_setzero_ps(), value));
        value = _mm512_mask_mov_ps(
            value, _mm512_cmp_ps_mask(value, _mm512_setzero_ps(), _CMP_EQ_OQ), _mm512_setzero_ps());
      } else {
        value = _mm512_min_ps(_mm512_max_ps(value, _mm512_setzero_ps()), _mm512_set1_ps(1));
      }
    }
    const int shift = mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
    auto result = _mm512_sll_epi32(half_narrow(value, saturate), _mm_cvtsi32_si128(shift));
    result =
        _mm512_or_si512(result, _mm512_and_si512(_mm512_loadu_si512(d + lane),
                                                 _mm512_set1_epi32(shift ? 0xffff : 0xffff0000U)));
    _mm512_mask_storeu_epi32(d + lane, __mmask16(exec_mask >> lane), result);
  }
}

template void half_unary_loose_x86_64_v4<Unary::Rcp>(uint64_t, uint32_t, uint32_t, uint32_t *,
                                                     const uint32_t *);
template void half_unary_loose_x86_64_v4<Unary::Rsq>(uint64_t, uint32_t, uint32_t, uint32_t *,
                                                     const uint32_t *);
template void half_unary_loose_x86_64_v4<Unary::Sqrt>(uint64_t, uint32_t, uint32_t, uint32_t *,
                                                      const uint32_t *);

template void half_unary_loose_x86_64_v4<Unary::Exp>(uint64_t, uint32_t, uint32_t, uint32_t *,
                                                     const uint32_t *);
template void half_unary_loose_x86_64_v4<Unary::Log>(uint64_t, uint32_t, uint32_t, uint32_t *,
                                                     const uint32_t *);

} // namespace rocjitsu::arith
