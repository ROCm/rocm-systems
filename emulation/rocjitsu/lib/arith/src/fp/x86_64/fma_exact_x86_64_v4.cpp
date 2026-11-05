// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 exact FP32 FMA and MUL fast paths using instruction-local rounding and scalar repair.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "fp/x86_64/fma_exact_x86_64_v4.h"
#include "common/internal.h"
#include "common/simd.h"
#include "common/x86_64/alu_x86_64_v4.h"
#include "fp/binary.h"
#include "fp/div_fmas.h"
#include "fp/scalar_repair.h"
#include "rj_arith/rj_arith.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {
namespace {

// Round wide products to FP32 using the requested instruction-local rounding mode.
__m256 narrow_product_f32(__m512d p, unsigned instruction_flags) {
  switch (instruction_flags) {
  case 1:
    return _mm512_cvt_roundpd_ps(p, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
  case 2:
    return _mm512_cvt_roundpd_ps(p, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
  case 3:
    return _mm512_cvt_roundpd_ps(p, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
  default:
    return _mm512_cvt_roundpd_ps(p, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  }
}

} // namespace

// Adapted from rj_arith-x86-lab wider512; matches the inherited rocjitsu MUL model.
// Combines exact FP64 products with embedded narrowing. Within this normal
// domain, the exact-product comparison also yields inherited MUL exceptions.
bool mul_exact_x86_64_v4(uint64_t flags, uint32_t exec_mask, uint32_t instruction_flags,
                         uint32_t *d, const uint32_t *a, const uint32_t *b, uint32_t *state) {
  if (instruction_flags & RJ_ARITH_ALU_OMOD_HALF) {
    return false;
  }
  const unsigned rounding = unsigned((flags & RJ_ARITH_FP_ROUND_MASK) >> 24);
  const bool report = reports_fp_status(flags);
  if (report && rounding) {
    return false;
  }

  // Zero and product exponents [-126, 126] cannot underflow or overflow.
  __m256i out[4];
  unsigned causes = 0;
  for (int i = 0; i < 32; i += 8) {
    auto aa = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + i)),
         bb = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + i));
    auto ma = _mm256_and_si256(aa, _mm256_set1_epi32(0x7fffffff)),
         mb = _mm256_and_si256(bb, _mm256_set1_epi32(0x7fffffff));
    if (_mm256_cmplt_epu32_mask(_mm256_sub_epi32(ma, _mm256_set1_epi32(1)),
                                _mm256_set1_epi32(0x007fffff)) ||
        _mm256_cmpge_epu32_mask(ma, _mm256_set1_epi32(0x7f800000)) ||
        _mm256_cmplt_epu32_mask(_mm256_sub_epi32(mb, _mm256_set1_epi32(1)),
                                _mm256_set1_epi32(0x007fffff)) ||
        _mm256_cmpge_epu32_mask(mb, _mm256_set1_epi32(0x7f800000))) {
      return false;
    }
    if (instruction_flags & RJ_ARITH_ALU_ABS_A) {
      aa = ma;
    }
    if (instruction_flags & RJ_ARITH_ALU_ABS_B) {
      bb = mb;
    }
    if (instruction_flags & RJ_ARITH_ALU_NEG_A) {
      aa = _mm256_xor_si256(aa, _mm256_set1_epi32(0x80000000U));
    }
    if (instruction_flags & RJ_ARITH_ALU_NEG_B) {
      bb = _mm256_xor_si256(bb, _mm256_set1_epi32(0x80000000U));
    }
    auto p = _mm512_mul_round_pd(_mm512_cvtps_pd(_mm256_castsi256_ps(aa)),
                                 _mm512_cvtps_pd(_mm256_castsi256_ps(bb)),
                                 _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    auto e =
        _mm512_and_si512(_mm512_srli_epi64(_mm512_castpd_si512(p), 52), _mm512_set1_epi64(2047));
    if ((_mm512_cmplt_epi64_mask(e, _mm512_set1_epi64(897)) &
         _mm512_cmp_pd_mask(p, _mm512_setzero_pd(), _CMP_NEQ_OQ)) ||
        _mm512_cmpgt_epi64_mask(e, _mm512_set1_epi64(1149))) {
      return false;
    }
    auto rounded = narrow_product_f32(p, rounding);
    auto u = _mm256_castps_si256(rounded);
    if (report &&
        (_mm512_cmp_pd_mask(p, _mm512_cvtps_pd(rounded), _CMP_NEQ_OQ) & ((exec_mask >> i) & 255))) {
      causes |= RJ_ARITH_EXCEPTION_INEXACT;
    }
    if (instruction_flags & RJ_ARITH_ALU_CLAMP) {
      u = _mm256_min_epu32(
          _mm256_mask_mov_epi32(u, _mm256_movepi32_mask(u), _mm256_setzero_si256()),
          _mm256_set1_epi32(0x3f800000));
    }
    out[i / 8] = u;
  }
  for (int i = 0; i < 4; ++i) {
    _mm256_mask_storeu_epi32(d + i * 8, __mmask8(exec_mask >> (i * 8)), out[i]);
  }
  if (report) {
    *state |= causes;
  }
  return true;
}

namespace {

// Keep the inherited scalar machinery out of the normal SIMD loop.
[[gnu::noinline]] __m512i repair_fma(__m512i result, __mmask16 bad, uint64_t flags,
                                     uint32_t instruction_flags, const uint32_t *a,
                                     const uint32_t *b, const uint32_t *c, bool rdna3,
                                     bool dx9_zero) {
  uint32_t repaired[16];
  _mm512_storeu_si512(repaired, result);
  for (int i = 0; i < 16; ++i) {
    if ((bad >> i) & 1) {
      repaired[i] = repair_fma_f32(a[i], b[i], c[i], flags, instruction_flags, rdna3, dx9_zero);
    }
  }
  return _mm512_loadu_si512(repaired);
}

} // namespace

// In the admitted normal-or-exact-zero domain, locally rounded FMA matches the
// inherited rocjitsu model. Only special/tiny lanes use scalar repair.
// Stage both halves before stores to preserve aliases during repair.
void fma_exact_x86_64_v4(uint64_t flags, uint32_t exec_mask, uint32_t instruction_flags,
                         uint32_t *d, const uint32_t *a, const uint32_t *b, const uint32_t *c,
                         bool dx9_zero, bool rdna3) {
  // Stage both vector chunks before destination writes; repaired lanes must still see original
  // inputs.
  __m512i out[2];
  for (int k = 0; k < 2; ++k) {
    __mmask16 bad = 0;
    __m512 aa = prepare_fma_input(_mm512_loadu_si512(a + 16 * k), flags, instruction_flags, 0, bad);
    __m512 bb = prepare_fma_input(_mm512_loadu_si512(b + 16 * k), flags, instruction_flags, 1, bad);
    __m512 cc = prepare_fma_input(_mm512_loadu_si512(c + 16 * k), flags, instruction_flags, 2, bad);
    if (dx9_zero) {
      const auto zero = _mm512_setzero_ps();
      auto has_zero =
          _mm512_cmp_ps_mask(aa, zero, _CMP_EQ_OQ) | _mm512_cmp_ps_mask(bb, zero, _CMP_EQ_OQ);
      aa = _mm512_mask_mov_ps(aa, has_zero, zero);
      bb = _mm512_mask_mov_ps(bb, has_zero, _mm512_set1_ps(1));
    }
    out[k] = fma_result(aa, bb, cc, flags, instruction_flags, bad);
    if (bad) {
      out[k] = repair_fma(out[k], bad, flags, instruction_flags, a + 16 * k, b + 16 * k, c + 16 * k,
                          rdna3, dx9_zero);
    }
  }
  _mm512_mask_storeu_epi32(d, __mmask16(exec_mask), out[0]);
  _mm512_mask_storeu_epi32(d + 16, __mmask16(exec_mask >> 16), out[1]);
}

// DIV_FMAS without conditional post-scaling. This admitted exponent range permits only
// INEXACT in the existing classifier. FastTwoSum retains a small addend lost
// by FP64 addition, so reporting never mistakes a rounded FP64 sum for exact.
bool div_fmas_exact_x86_64_v4(uint64_t flags, uint32_t exec_mask, uint32_t *d, const uint32_t *a,
                              const uint32_t *b, const uint32_t *c, uint32_t *state,
                              uint32_t mode) {
  const uint32_t *inputs[] = {a, b, c};
  for (const auto *input : inputs) {
    for (int lane = 0; lane < 32; lane += 16) {
      auto magnitude =
          _mm512_and_si512(_mm512_loadu_si512(input + lane), _mm512_set1_epi32(0x7fffffff));
      auto exponent = _mm512_srli_epi32(magnitude, 23);
      if (_mm512_cmp_epu32_mask(_mm512_sub_epi32(exponent, _mm512_set1_epi32(96)),
                                _mm512_set1_epi32(62), _MM_CMPINT_GT) &
          _mm512_cmpneq_epi32_mask(magnitude, _mm512_setzero_si512())) {
        return false;
      }
    }
  }
  // Stage numeric results so reporting still sees original aliased sources.
  uint32_t result[32];
  fma_exact_x86_64_v4(flags, ~0U, mode, result, a, b, c, /*dx9_zero=*/false, /*rdna3=*/false);
  uint32_t causes = 0;
  if (reports_fp_status(flags) && !(mode & (RJ_ARITH_ALU_CLAMP | RJ_ARITH_ALU_OMOD_HALF))) {
    for (int lane = 0; lane < 32; lane += 8) {
      // Apply source modifiers to FP32 inputs, then widen to FP64.
      auto widen = [mode](const uint32_t *p, int operand) {
        auto bits = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(p));
        if (mode & (RJ_ARITH_ALU_ABS_A << operand)) {
          bits = _mm256_and_si256(bits, _mm256_set1_epi32(0x7fffffff));
        }
        if (mode & (RJ_ARITH_ALU_NEG_A << operand)) {
          bits = _mm256_xor_si256(bits, _mm256_set1_epi32(INT32_MIN));
        }
        return _mm512_cvtps_pd(_mm256_castsi256_ps(bits));
      };
      auto product = _mm512_mul_pd(widen(a + lane, 0), widen(b + lane, 1));
      auto addend = widen(c + lane, 2);
      // Return the absolute FP64 values.
      auto magnitude = [](__m512d value) {
        return _mm512_castsi512_pd(
            _mm512_and_si512(_mm512_castpd_si512(value), _mm512_set1_epi64(0x7fffffffffffffffLL)));
      };
      auto swap = _mm512_cmp_pd_mask(magnitude(product), magnitude(addend), _CMP_LT_OQ);
      auto big = _mm512_mask_blend_pd(swap, product, addend);
      auto small = _mm512_mask_blend_pd(swap, addend, product);
      auto sum = _mm512_add_pd(big, small);
      auto residual = _mm512_sub_pd(small, _mm512_sub_pd(sum, big));
      auto inexact = _mm512_cmp_pd_mask(sum,
                                        _mm512_cvtps_pd(_mm256_castsi256_ps(_mm256_loadu_si256(
                                            reinterpret_cast<const __m256i *>(result + lane)))),
                                        _CMP_NEQ_OQ) |
                     _mm512_cmp_pd_mask(residual, _mm512_setzero_pd(), _CMP_NEQ_OQ);
      if (inexact & __mmask8(exec_mask >> lane)) {
        causes |= RJ_ARITH_EXCEPTION_INEXACT;
      }
    }
  }
  for (int lane = 0; lane < 32; lane += 16) {
    _mm512_mask_storeu_epi32(d + lane, __mmask16(exec_mask >> lane),
                             _mm512_loadu_si512(result + lane));
  }
  if (reports_fp_status(flags)) {
    *state |= causes;
  }
  return true;
}

} // namespace rocjitsu::arith
