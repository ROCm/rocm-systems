// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 mixed-precision FMA with shared decoding and semantics-specific rounding.
// Loose uses native FP32 FMA; exact arithmetic retains lane-local scalar repair.

#include "common/compiler.h"
#include "common/validation.h"
#include "common/x86_64/alu_x86_64_v4.h"
#include "fp/mixed_fma.h"
#include "fp/mixed_fma_control.h"
#include "fp/scalar_repair.h"
#include "fp/x86_64/fma_exact_x86_64_v4.h"
#include "fp/x86_64/loose_half_output_x86_64_v4.h"

namespace rocjitsu::arith {

// Apply validated mixed-FMA controls, preserving aliases and inactive destination storage.
template <bool Exact>
static void mixed_fma_x86_64_v4_impl(MixedFma destination, bool rdna3, uint64_t flags,
                                     uint32_t exec_mask, uint32_t mode, uint32_t *d,
                                     const uint32_t *a, const uint32_t *b, const uint32_t *c) {
  alignas(64) uint32_t result[32];
  const uint32_t *inputs[] = {a, b, c};
  const unsigned rounding = unsigned((flags & RJ_ARITH_FP_ROUND_MASK) >> 24);
  for (int lane = 0; lane < 32; lane += 16) {
    __m512 values[3];
    __mmask16 bad = 0;
    RJ_ARITH_UNROLL(3)
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
    if constexpr (!Exact) {
      auto value = _mm512_fmadd_ps(values[0], values[1], values[2]);
      if (destination == MixedFma::Float) {
        if (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) {
          value = flush_denorm_f32(value);
        }
        if (mode & RJ_ARITH_ALU_CLAMP) {
          value = clamp_f32(value, flags & RJ_ARITH_FP_DX10_CLAMP_OFF);
        }
        out = _mm512_castps_si512(value);
      } else {
        auto half = loose_half_output(value, flags, mode);
        const int shift = destination == MixedFma::High ? 16 : 0;
        out = _mm512_or_si512(
            _mm512_slli_epi32(half, shift),
            _mm512_and_si512(_mm512_loadu_si512(d + lane), _mm512_set1_epi32(~(0xffffU << shift))));
      }
    } else if (destination == MixedFma::Float) {
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
      if (bad) {
        auto unresolved = bad;
        auto bits = fma_nan_operands(_mm512_setzero_si512(), values[0], values[1], values[2], flags,
                                     mode, false, unresolved);
        auto payload = _mm512_and_si512(_mm512_srli_epi32(bits, 13), _mm512_set1_epi32(1023));
        auto sign = _mm512_and_si512(_mm512_srli_epi32(bits, 16), _mm512_set1_epi32(0x8000));
        auto nan = _mm512_or_si512(sign, _mm512_or_si512(payload, _mm512_set1_epi32(0x7e00)));
        if ((mode & RJ_ARITH_ALU_CLAMP) && !(flags & RJ_ARITH_FP_DX10_CLAMP_OFF)) {
          nan = _mm512_setzero_si512();
        }
        half = _mm512_mask_mov_epi32(half, __mmask16(bad & ~unresolved), nan);
        bad = unresolved;
      }
      out = _mm512_or_si512(
          _mm512_slli_epi32(half, shift),
          _mm512_and_si512(_mm512_loadu_si512(d + lane), _mm512_set1_epi32(~(0xffffU << shift))));
    }
    if constexpr (Exact) {
      // FP16 input conversion may have quieted a signaling NaN. Preserve the
      // raw-source repair in that case; all-FP32 inputs retain their encodings.
      if (bad && destination == MixedFma::Float &&
          !(mode & (RJ_ARITH_MIX_F16_A | RJ_ARITH_MIX_F16_B | RJ_ARITH_MIX_F16_C))) {
        out = fma_nan_operands(out, values[0], values[1], values[2], flags, mode, rdna3, bad);
      }
      _mm512_store_si512(result + lane, out);
      bad &= __mmask16(exec_mask >> lane);
      if (bad) {
        for (; bad; bad &= bad - 1) {
          int i = __builtin_ctz(bad);
          result[lane + i] = repair_mixed_fma(destination, rdna3, flags, mode, a[lane + i],
                                              b[lane + i], c[lane + i], d[lane + i]);
        }
      }
    } else {
      // One-VGPR outputs permit chunk-local publication for whole-register aliases.
      _mm512_mask_storeu_epi32(d + lane, __mmask16(exec_mask >> lane), out);
    }
  }
  if constexpr (Exact) {
    for (int lane = 0; lane < 32; lane += 16) {
      _mm512_mask_storeu_epi32(d + lane, __mmask16(exec_mask >> lane),
                               _mm512_load_si512(result + lane));
    }
  }
}

// Compute the guest-rounded result, repairing exceptional lanes when needed.
void mixed_fma_exact_x86_64_v4(MixedFma destination, bool rdna3, uint64_t flags, uint32_t exec_mask,
                               uint32_t mode, uint32_t *d, const uint32_t *a, const uint32_t *b,
                               const uint32_t *c) {
  mixed_fma_x86_64_v4_impl<true>(destination, rdna3, flags, exec_mask, mode, d, a, b, c);
}

// Compute a native FP32 FMA with the selected mixed-format destination policy.
void mixed_fma_loose_x86_64_v4(MixedFma destination, bool rdna3, uint64_t flags, uint32_t exec_mask,
                               uint32_t mode, uint32_t *d, const uint32_t *a, const uint32_t *b,
                               const uint32_t *c) {
  mixed_fma_x86_64_v4_impl<false>(destination, rdna3, flags, exec_mask, mode, d, a, b, c);
}

} // namespace rocjitsu::arith
