// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 bit-exact mixed-precision FMA with per-lane scalar repair.

#include "common/internal.h"
#include "fp/mixed_fma.h"
#include "fp/mixed_fma_control.h"
#include "fp/scalar_repair.h"
#include "fp/x86_64/fma_exact_x86_64_v4.h"

namespace rocjitsu::arith {

// Evaluate mixed fma exact for the caller-selected operation. Preserve destination lanes
// outside exec_mask.
// Requires the named CPU features and validated instruction controls.
void mixed_fma_exact_x86_64_v4(MixedFma destination, bool rdna3, uint64_t flags, uint32_t exec_mask,
                               uint32_t mode, uint32_t *d, const uint32_t *a, const uint32_t *b,
                               const uint32_t *c) {
  alignas(64) uint32_t result[32];
  const uint32_t *inputs[] = {a, b, c};
  const unsigned rounding = unsigned((flags & RJ_ARITH_FP_ROUND_MASK) >> 24);
  for (int lane = 0; lane < 32; lane += 16) {
    __m512 values[3];
    __mmask16 bad = 0;
    for (int operand = 0; operand < 3; ++operand) {
      auto bits = _mm512_loadu_si512(inputs[operand] + lane);
      uint32_t source_mode = mode >> operand;
      bool half = source_mode & RJ_ARITH_MIX_F16_A;
      bool flush = mixed_fma_flush_input(flags, source_mode);
      if (half) {
        bits = _mm512_srli_epi32(bits, source_mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0);
        bits = _mm512_and_si512(bits, _mm512_set1_epi32(0xffff));
        auto magnitude = _mm512_and_si512(bits, _mm512_set1_epi32(0x7fff));
        bad |= _mm512_cmpge_epu32_mask(magnitude, _mm512_set1_epi32(0x7c00));
        if (flush) {
          auto sub = _mm512_cmplt_epu32_mask(magnitude, _mm512_set1_epi32(0x400));
          bits = _mm512_mask_and_epi32(bits, sub, bits, _mm512_set1_epi32(0x8000));
        }
        bits = _mm512_castps_si512(
            _mm512_cvt_roundph_ps(_mm512_cvtepi32_epi16(bits), _MM_FROUND_NO_EXC));
      }
      // Half denormals were handled in their own format. Do not flush their
      // widened values using the independent FP32 policy.
      uint64_t input_flags = flags & ~RJ_ARITH_FP_FLUSH_INPUT_DENORMALS;
      if (!half && flush) {
        input_flags |= RJ_ARITH_FP_FLUSH_INPUT_DENORMALS;
      }
      values[operand] = prepare_fma_input(bits, input_flags, source_mode, 0, bad);
    }
    __m512i out;
    if (destination == MixedFma::Float) {
      out = fma_result(values[0], values[1], values[2], flags, mode & RJ_ARITH_ALU_CLAMP, bad);
    } else {
      __m512 down, up;
      auto odd = half_fma_odd(values, rounding, down, up);
      auto magnitude = _mm512_and_si512(odd, _mm512_set1_epi32(0x7fffffff));

      // Repair FP32 tininess; a rounded-to-zero bound alone does not prove
      // cancellation. Finite overflow leaves a max-finite toward-zero bound,
      // already beyond FP16 range, so directed narrowing remains correct.
      auto exact_zero = _mm512_cmp_ps_mask(down, _mm512_setzero_ps(), _CMP_EQ_OQ) &
                        _mm512_cmp_ps_mask(up, _mm512_setzero_ps(), _CMP_EQ_OQ);
      bad |= (_mm512_cmple_epu32_mask(magnitude, _mm512_set1_epi32(0x00800000)) & ~exact_zero) |
             _mm512_cmpge_epu32_mask(magnitude, _mm512_set1_epi32(0x7f800000));
      auto half = _mm512_cvtepu16_epi32(narrow_half(_mm512_castsi512_ps(odd), rounding));
      half = finish_half_fma(half, odd, flags, mode);
      int shift = destination == MixedFma::High ? 16 : 0;
      out = _mm512_or_si512(
          _mm512_slli_epi32(half, shift),
          _mm512_and_si512(_mm512_loadu_si512(d + lane), _mm512_set1_epi32(~(0xffffU << shift))));
    }
    _mm512_store_si512(result + lane, out);
    bad &= __mmask16(exec_mask >> lane);
    if (bad) {
      for (int i = 0; i < 16; ++i) {
        if (!((bad >> i) & 1)) {
          continue;
        }
        result[lane + i] = repair_mixed_fma(destination, rdna3, flags, mode, a[lane + i],
                                            b[lane + i], c[lane + i], d[lane + i]);
      }
    }
  }
  for (int lane = 0; lane < 32; lane += 16) {
    _mm512_mask_storeu_epi32(d + lane, __mmask16(exec_mask >> lane),
                             _mm512_load_si512(result + lane));
  }
}

} // namespace rocjitsu::arith
