// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 exact FP16 FMA fast paths using instruction-local rounding and scalar repair.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "common/internal.h"
#include "common/simd.h"
#include "common/x86_64/alu_x86_64_v4.h"
#include "fp/scalar_repair.h"
#include "fp/x86_64/fma_exact_x86_64_v4.h"
#include "rj_arith/rj_arith.h"

#include <immintrin.h>

namespace rocjitsu::arith {

namespace {

// Replace bad FP16 FMA lanes with scalar-model results and accumulate causes for reporting
// lanes.
[[gnu::noinline]] __m512i repair_half_fma(__m512i result, __mmask16 bad, uint64_t flags,
                                          uint32_t mode, const uint32_t *a, const uint32_t *b,
                                          const uint32_t *c, bool rdna3, __mmask16 reporting,
                                          uint32_t &causes) {
  uint32_t repaired[16];
  _mm512_storeu_si512(repaired, result);
  for (int i = 0; i < 16; ++i) {
    if (!((bad >> i) & 1)) {
      continue;
    }
    uint16_t x = uint16_t(a[i] >> (mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0));
    uint16_t y = uint16_t(b[i] >> (mode & RJ_ARITH_ALU_HIGH_B ? 16 : 0));
    uint16_t z = uint16_t(c[i] >> (mode & RJ_ARITH_ALU_HIGH_C ? 16 : 0));
    auto status = (reporting >> i) & 1 ? &causes : nullptr;
    repaired[i] = repair_fma_f16(x, y, z, mode, flags, rdna3, status);
  }
  return _mm512_loadu_si512(repaired);
}

} // namespace

// Round-to-odd staging implements the inherited fp_mode.h
// finite FP16 FMA model without FP32/FP64 double rounding. Every finite FP16
// product and nonzero cancellation is normal in FP32. Existing rj-arith status
// rules are derived from the same bounds; CLAMP suppresses all half-FMA causes.
bool half_fma_exact_x86_64_v4(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                              const uint32_t *a, const uint32_t *b, const uint32_t *c,
                              uint32_t *state, bool rdna3) {
  const unsigned omod = (mode >> 6) & 3;
  const bool report = reports_fp_status(flags) && !(mode & RJ_ARITH_ALU_CLAMP);
  const unsigned rounding = unsigned((flags & RJ_ARITH_FP_ROUND_MASK) >> 24);
  if (report &&
      ((flags & (RJ_ARITH_FP_FLUSH_INPUT_DENORMALS | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) ||
       rounding)) {
    return false;
  }
  __m512i results[2];
  uint32_t causes = 0;
  for (int lane = 0; lane < 32; lane += 16) {
    const uint32_t *inputs[] = {a, b, c};
    __m512 values[3];
    __mmask16 input_denormal = 0, bad = 0;
    for (int operand = 0; operand < 3; ++operand) {
      auto bits = _mm512_loadu_si512(inputs[operand] + lane);
      if (mode & (RJ_ARITH_ALU_HIGH_A << operand)) {
        bits = _mm512_srli_epi32(bits, 16);
      }
      bits = _mm512_and_si512(bits, _mm512_set1_epi32(0xffff));
      if (mode & (RJ_ARITH_ALU_ABS_A << operand)) {
        bits = _mm512_and_si512(bits, _mm512_set1_epi32(0x7fff));
      }
      if (mode & (RJ_ARITH_ALU_NEG_A << operand)) {
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
                             mode & ~RJ_ARITH_ALU_CLAMP);
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
      half = finish_half_fma(half, odd, flags & ~RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS, mode);
    } else {
      half = finish_half_fma(half, odd, flags, mode);
    }
    if (bad) {
      half = repair_half_fma(half, bad, flags, mode, a + lane, b + lane, c + lane, rdna3,
                             report ? __mmask16(exec_mask >> lane) : 0, causes);
    }
    const unsigned shift = mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
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
  return true;
}

// Preserve original sources and accumulated status until both packed halves
// succeed, including when D aliases an input selected by the second half.
bool packed_fma_exact_x86_64_v4(uint64_t flags, uint32_t exec_mask, const uint32_t modes[2],
                                uint32_t *d, const uint32_t *a, const uint32_t *b,
                                const uint32_t *c, uint32_t *state, bool rdna3) {
  uint32_t staged[32], causes = 0;
  for (int lane = 0; lane < 32; lane += 16) {
    _mm512_storeu_si512(staged + lane, _mm512_loadu_si512(d + lane));
  }
  if (!half_fma_exact_x86_64_v4(flags, exec_mask, modes[0], staged, a, b, c, &causes, rdna3) ||
      !half_fma_exact_x86_64_v4(flags, exec_mask, modes[1] | RJ_ARITH_ALU_HIGH_D, staged, a, b, c,
                                &causes, rdna3)) {
    return false;
  }
  for (int lane = 0; lane < 32; lane += 16) {
    _mm512_mask_storeu_epi32(d + lane, __mmask16(exec_mask >> lane),
                             _mm512_loadu_si512(staged + lane));
  }
  if (reports_fp_status(flags)) {
    *state |= causes;
  }
  return true;
}

} // namespace rocjitsu::arith
