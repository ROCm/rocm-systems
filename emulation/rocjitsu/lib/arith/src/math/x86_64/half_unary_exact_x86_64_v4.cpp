// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 exact FP16 integral rounding, mantissa extraction and fractional parts.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "common/x86_64/half_x86_64_v4.h"
#include "math/half_unary.h"
#include "rj_arith/detail/rj_arith_math.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {

// Integral and normalized-mantissa results are exactly representable in FP16.
// Preserve selected halves, inactive lanes and the requested NaN/output policy.
template <Unary Op>
void half_unary_exact_x86_64_v4(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                                const uint32_t *a) {
  for (int lane = 0; lane < 32; lane += 16) {
    auto value =
        half_input(_mm512_loadu_si512(a + lane), mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0, mode);
    if (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) {
      auto bits = _mm512_castps_si512(value);
      auto tiny = _mm512_cmplt_epi32_mask(_mm512_and_si512(bits, _mm512_set1_epi32(INT32_MAX)),
                                          _mm512_set1_epi32(0x38800000));
      value = _mm512_castsi512_ps(
          _mm512_mask_and_epi32(bits, tiny, bits, _mm512_set1_epi32(INT32_MIN)));
    }
    if constexpr (Op == Unary::FrexpMant) {
      // Widened nonzero FP16 values are normal FP32 values, even for half subnormals.
      auto bits = _mm512_castps_si512(value);
      auto magnitude = _mm512_and_si512(bits, _mm512_set1_epi32(INT32_MAX));
      auto finite_nonzero = _mm512_cmpgt_epi32_mask(magnitude, _mm512_setzero_si512()) &
                            _mm512_cmplt_epi32_mask(magnitude, _mm512_set1_epi32(0x7f800000));
      auto mantissa = _mm512_or_si512(_mm512_and_si512(bits, _mm512_set1_epi32(0x807fffffU)),
                                      _mm512_set1_epi32(0x3f000000));
      value = _mm512_castsi512_ps(_mm512_mask_mov_epi32(bits, finite_nonzero, mantissa));
    } else if constexpr (Op == Unary::Fract) {
      // The FP32 difference is exact for every finite half input. Narrow with
      // guest rounding, cap below one, then apply the inherited output flush.
      value = _mm512_sub_ps(value, _mm512_floor_ps(value));
      value = _mm512_mask_min_ps(value, _mm512_cmp_ps_mask(value, value, _CMP_ORD_Q), value,
                                 _mm512_set1_ps(0x1.ffcp-1f));
      auto rounded = half_narrow(value, false, unsigned((flags & RJ_ARITH_FP_ROUND_MASK) >> 24));
      value = _mm512_cvtph_ps(_mm512_cvtepi32_epi16(rounded));
      if (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) {
        value = _mm512_mask_mov_ps(value,
                                   _mm512_cmp_ps_mask(value, _mm512_set1_ps(0x1p-14f), _CMP_LT_OQ),
                                   _mm512_setzero_ps());
      }
    } else {
      constexpr int rounding = Op == Unary::Trunc   ? _MM_FROUND_TO_ZERO
                               : Op == Unary::Ceil  ? _MM_FROUND_TO_POS_INF
                               : Op == Unary::Floor ? _MM_FROUND_TO_NEG_INF
                                                    : _MM_FROUND_TO_NEAREST_INT;
      value = _mm512_roundscale_ps(value, rounding | _MM_FROUND_NO_EXC);
    }
    if (unsigned omod = (mode >> 6) & 3) {
      if constexpr (Op == Unary::Fract) {
        // OMOD flushes tiny sources, and halving a minimum-exponent normal
        // produces +0 rather than a new half subnormal.
        value = _mm512_mask_mov_ps(
            value,
            _mm512_cmp_ps_mask(value, _mm512_set1_ps(omod == 3 ? 0x1p-13f : 0x1p-14f), _CMP_LT_OQ),
            _mm512_setzero_ps());
      }
      value = _mm512_mask_mov_ps(value, _mm512_cmp_ps_mask(value, _mm512_setzero_ps(), _CMP_EQ_OQ),
                                 _mm512_setzero_ps());
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
    int shift = mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
    auto narrowed = half_narrow(value, flags & RJ_ARITH_FP16_OVFL);
    if (!(flags & RJ_ARITH_FP_IEEE_MODE) &&
        (!(mode & RJ_ARITH_ALU_CLAMP) || (flags & RJ_ARITH_FP_DX10_CLAMP_OFF))) {
      auto raw =
          _mm512_and_si512(_mm512_srl_epi32(_mm512_loadu_si512(a + lane),
                                            _mm_cvtsi32_si128(mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0)),
                           _mm512_set1_epi32(mode & RJ_ARITH_ALU_ABS_A ? 0x7fff : 0xffff));
      raw = _mm512_xor_si512(raw, _mm512_set1_epi32(mode & RJ_ARITH_ALU_NEG_A ? 0x8000 : 0));
      auto nan = _mm512_cmpgt_epi32_mask(_mm512_and_si512(raw, _mm512_set1_epi32(0x7fff)),
                                         _mm512_set1_epi32(0x7c00));
      narrowed = _mm512_mask_mov_epi32(narrowed, nan, raw);
    }
    auto result = _mm512_sll_epi32(narrowed, _mm_cvtsi32_si128(shift));
    result =
        _mm512_or_si512(result, _mm512_and_si512(_mm512_loadu_si512(d + lane),
                                                 _mm512_set1_epi32(shift ? 0xffff : 0xffff0000U)));
    _mm512_mask_storeu_epi32(d + lane, __mmask16(exec_mask >> lane), result);
  }
}

template void half_unary_exact_x86_64_v4<Unary::Trunc>(uint64_t, uint32_t, uint32_t, uint32_t *,
                                                       const uint32_t *);
template void half_unary_exact_x86_64_v4<Unary::Ceil>(uint64_t, uint32_t, uint32_t, uint32_t *,
                                                      const uint32_t *);
template void half_unary_exact_x86_64_v4<Unary::Floor>(uint64_t, uint32_t, uint32_t, uint32_t *,
                                                       const uint32_t *);
template void half_unary_exact_x86_64_v4<Unary::Rndne>(uint64_t, uint32_t, uint32_t, uint32_t *,
                                                       const uint32_t *);

template void half_unary_exact_x86_64_v4<Unary::FrexpMant>(uint64_t, uint32_t, uint32_t, uint32_t *,
                                                           const uint32_t *);

template void half_unary_exact_x86_64_v4<Unary::Fract>(uint64_t, uint32_t, uint32_t, uint32_t *,
                                                       const uint32_t *);

} // namespace rocjitsu::arith
