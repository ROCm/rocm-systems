// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 exact FP16 FMA fast paths using instruction-local rounding and scalar repair.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "common/internal.h"
#include "common/simd.h"
#include "fp/scalar_repair.h"
#include "rj_arith/rj_arith.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {

namespace {

// Replace bad FP16 FMA lanes with scalar-model results and accumulate causes for reporting
// lanes.
[[gnu::noinline]] __m512i repair_half_fma(__m512i result, __mmask16 bad, uint64_t flags,
                                          uint32_t instruction_flags, const uint32_t *a,
                                          const uint32_t *b, const uint32_t *c, bool rdna3,
                                          __mmask16 reporting, uint32_t &causes) {
  uint32_t repaired[16];
  _mm512_storeu_si512(repaired, result);
  for (int i = 0; i < 16; ++i) {
    if (!((bad >> i) & 1)) {
      continue;
    }
    uint16_t x = uint16_t(a[i] >> (instruction_flags & RJ_ARITH_ALU_HIGH_A ? 16 : 0));
    uint16_t y = uint16_t(b[i] >> (instruction_flags & RJ_ARITH_ALU_HIGH_B ? 16 : 0));
    uint16_t z = uint16_t(c[i] >> (instruction_flags & RJ_ARITH_ALU_HIGH_C ? 16 : 0));
    auto status = (reporting >> i) & 1 ? &causes : nullptr;
    repaired[i] = repair_fma_f16(x, y, z, flags, instruction_flags, rdna3, status);
  }
  return _mm512_loadu_si512(repaired);
}

} // namespace

namespace {

// Round FP32 vector values to packed FP16 using the supplied guest rounding code.
__m256i narrow_half(__m512 value, unsigned rounding) {
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
__m512i half_fma_odd(const __m512 (&values)[3], unsigned rounding, __m512 &down, __m512 &up) {
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
                                                      uint32_t instruction_flags) {
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
  if (instruction_flags & RJ_ARITH_ALU_CLAMP) {
    auto negative = _mm512_test_epi32_mask(half, _mm512_set1_epi32(0x8000));
    half = _mm512_mask_mov_epi32(half, negative, _mm512_setzero_si512());
    half = _mm512_min_epu32(half, _mm512_set1_epi32(0x3c00));
  }
  return half;
}

} // namespace

// Round-to-odd staging implements the inherited fp_mode.h
// finite FP16 FMA model without FP32/FP64 double rounding. Every finite FP16
// product and nonzero cancellation is normal in FP32. Existing rj-arith status
// rules are derived from the same bounds; CLAMP suppresses all half-FMA causes.
void half_fma_exact_x86_64_v4(uint64_t flags, uint32_t exec_mask, uint32_t instruction_flags,
                              uint32_t *d, const uint32_t *a, const uint32_t *b, const uint32_t *c,
                              uint32_t *state, bool rdna3) {
  const unsigned omod = (instruction_flags >> 6) & 3;
  const bool report = reports_fp_status(flags) && !(instruction_flags & RJ_ARITH_ALU_CLAMP);
  const unsigned rounding = unsigned((flags & RJ_ARITH_FP_ROUND_MASK) >> 24);
  __m512i results[2];
  uint32_t causes = 0;
  for (int lane = 0; lane < 32; lane += 16) {
    const uint32_t *inputs[] = {a, b, c};
    __m512 values[3];
    __mmask16 input_denormal = 0, bad = 0;
    for (int operand = 0; operand < 3; ++operand) {
      auto bits = _mm512_loadu_si512(inputs[operand] + lane);
      if (instruction_flags & (RJ_ARITH_ALU_HIGH_A << operand)) {
        bits = _mm512_srli_epi32(bits, 16);
      }
      bits = _mm512_and_si512(bits, _mm512_set1_epi32(0xffff));
      if (instruction_flags & (RJ_ARITH_ALU_ABS_A << operand)) {
        bits = _mm512_and_si512(bits, _mm512_set1_epi32(0x7fff));
      }
      if (instruction_flags & (RJ_ARITH_ALU_NEG_A << operand)) {
        bits = _mm512_xor_si512(bits, _mm512_set1_epi32(0x8000));
      }
      auto magnitude = _mm512_and_si512(bits, _mm512_set1_epi32(0x7fff));
      bad |= _mm512_cmpge_epu32_mask(magnitude, _mm512_set1_epi32(0x7c00));
      auto denormal = _mm512_cmplt_epu32_mask(_mm512_sub_epi32(magnitude, _mm512_set1_epi32(1)),
                                              _mm512_set1_epi32(0x3ff));
      if (report) {
        input_denormal |= denormal;
      }
      if (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) {
        bits = _mm512_mask_and_epi32(bits, denormal, bits, _mm512_set1_epi32(0x8000));
      }
      values[operand] = _mm512_cvt_roundph_ps(_mm512_cvtepi32_epi16(bits), _MM_FROUND_NO_EXC);
    }
    __m512 down, up;
    auto odd = half_fma_odd(values, rounding, down, up);
    auto different = _mm512_cmp_ps_mask(down, up, _CMP_NEQ_OQ);
    auto h = narrow_half(_mm512_castsi512_ps(odd), rounding);
    auto half = _mm512_cvtepu16_epi32(h);
    if (report) {
      const __mmask16 active = __mmask16(exec_mask >> lane) & ~bad;
      auto rounded = _mm512_cvt_roundph_ps(h, _MM_FROUND_NO_EXC);
      auto inexact = active & (different | _mm512_cmp_ps_mask(rounded, down, _CMP_NEQ_OQ));
      auto magnitude = _mm512_and_si512(odd, _mm512_set1_epi32(0x7fffffff));

      // Existing tininess threshold: 2^-14 - 2^-26, excluding the midpoint.
      auto tiny = _mm512_cmplt_epu32_mask(magnitude, _mm512_set1_epi32(0x387ff000));
      auto hm = _mm512_and_si512(half, _mm512_set1_epi32(0x7fff));
      if (!omod && inexact) {
        causes |= RJ_ARITH_EXCEPTION_INEXACT;
      }
      if (!omod && (inexact & tiny)) {
        causes |= RJ_ARITH_EXCEPTION_UNDERFLOW;
      }
      if (active & _mm512_cmpeq_epi32_mask(hm, _mm512_set1_epi32(0x7c00))) {
        causes |= RJ_ARITH_EXCEPTION_OVERFLOW;
      }
      if ((omod == 1 || omod == 2) &&
          (active & _mm512_cmplt_epi32_mask(hm, _mm512_set1_epi32(0x7c00)) &
           _mm512_cmpge_epi32_mask(_mm512_srli_epi32(hm, 10), _mm512_set1_epi32(31 - omod)))) {
        causes |= RJ_ARITH_EXCEPTION_OVERFLOW;
      }
      if (active & input_denormal) {
        causes |= RJ_ARITH_EXCEPTION_INPUT_DENORM;
      }
    }
    if (omod) {
      // Inherited OMOD acts on the already-rounded half, flushes tiny
      // intermediates, then scales and narrows with guest rounding.
      half = finish_half_fma(half, odd, flags & ~RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS,
                             instruction_flags & ~RJ_ARITH_ALU_CLAMP);
      auto magnitude = _mm512_and_si512(half, _mm512_set1_epi32(0x7fff));
      auto tiny = half_fma_tiny_mask(half, odd, rounding);
      half = _mm512_mask_mov_epi32(half, tiny, _mm512_setzero_si512());
      if (omod == 3) {
        half = _mm512_mask_and_epi32(half,
                                     _mm512_cmplt_epi32_mask(magnitude, _mm512_set1_epi32(0x800)),
                                     half, _mm512_set1_epi32(0x8000));
      }
      auto wide = _mm512_cvt_roundph_ps(_mm512_cvtepi32_epi16(half), _MM_FROUND_NO_EXC);
      wide = _mm512_mul_ps(wide, _mm512_set1_ps(omod == 3 ? 0.5f : omod == 1 ? 2.0f : 4.0f));
      half = _mm512_cvtepu16_epi32(narrow_half(wide, rounding));
      half = finish_half_fma(half, odd, flags & ~RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS,
                             instruction_flags);
    } else {
      half = finish_half_fma(half, odd, flags, instruction_flags);
    }
    if (bad) {
      half = repair_half_fma(half, bad, flags, instruction_flags, a + lane, b + lane, c + lane,
                             rdna3, report ? __mmask16(exec_mask >> lane) : 0, causes);
    }
    const unsigned shift = instruction_flags & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
    results[lane / 16] = _mm512_or_si512(
        _mm512_slli_epi32(half, shift),
        _mm512_and_si512(_mm512_loadu_si512(d + lane), _mm512_set1_epi32(~(0xffffU << shift))));
  }
  for (int lane = 0; lane < 32; lane += 16) {
    _mm512_mask_storeu_epi32(d + lane, __mmask16(exec_mask >> lane), results[lane / 16]);
  }
  if (reports_fp_status(flags)) {
    *state |= causes;
  }
}

} // namespace rocjitsu::arith
