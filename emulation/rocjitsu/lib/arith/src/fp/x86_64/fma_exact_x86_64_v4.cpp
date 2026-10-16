// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 exact FMA and FP32 MUL fast paths using instruction-local rounding and scalar repair.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "common/internal.h"
#include "common/simd.h"
#include "common/x86_64/alu_x86_64_v4.h"
#include "fp/mixed_fma.h"
#include "fp/mixed_fma_control.h"
#include "fp/scalar_repair.h"
#include "rj_arith/rj_arith.h"

#include <immintrin.h>

namespace rocjitsu::arith {
namespace {

// Round wide products to FP32 using the requested instruction-local rounding mode.
__m256 narrow_product_f32(__m512d p, unsigned mode) {
  switch (mode) {
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
bool mul_exact_x86_64_v4(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                         const uint32_t *a, const uint32_t *b, uint32_t *state) {
  if (mode & RJ_ARITH_ALU_OMOD_HALF) {
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
    if (mode & RJ_ARITH_ALU_ABS_A) {
      aa = ma;
    }
    if (mode & RJ_ARITH_ALU_ABS_B) {
      bb = mb;
    }
    if (mode & RJ_ARITH_ALU_NEG_A) {
      aa = _mm256_xor_si256(aa, _mm256_set1_epi32(0x80000000U));
    }
    if (mode & RJ_ARITH_ALU_NEG_B) {
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
    if (mode & RJ_ARITH_ALU_CLAMP) {
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

// Compute vector a*b+c with instruction-local rounding selected by rounding.
__m512 fused(__m512 a, __m512 b, __m512 c, unsigned rounding) {
  switch (rounding) {
  case 1:
    return _mm512_fmadd_round_ps(a, b, c, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
  case 2:
    return _mm512_fmadd_round_ps(a, b, c, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
  case 3:
    return _mm512_fmadd_round_ps(a, b, c, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
  default:
    return _mm512_fmadd_round_ps(a, b, c, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  }
}

// Apply source modifiers and guest input flushing; mark unsupported fast-path lanes in bad.
__m512 prepare_fma_input(__m512i x, uint64_t flags, uint32_t mode, int src, __mmask16 &bad) {
  if (mode & (RJ_ARITH_ALU_ABS_A << src)) {
    x = _mm512_and_si512(x, _mm512_set1_epi32(0x7fffffff));
  }
  if (mode & (RJ_ARITH_ALU_NEG_A << src)) {
    x = _mm512_xor_si512(x, _mm512_set1_epi32(0x80000000U));
  }
  auto m = _mm512_and_si512(x, _mm512_set1_epi32(0x7fffffff));
  auto sub = _mm512_cmplt_epu32_mask(m, _mm512_set1_epi32(0x00800000)) &
             _mm512_cmpneq_epi32_mask(m, _mm512_setzero_si512());
  bad |= _mm512_cmpge_epu32_mask(m, _mm512_set1_epi32(0x7f800000));
  if (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) {
    x = _mm512_mask_and_epi32(x, sub, x, _mm512_set1_epi32(0x80000000U));
  } else {
    bad |= sub;
  }
  return _mm512_castsi512_ps(x);
}

// Keep the optional scaling work out of the common FMA body. Besides sharing
// the arithmetic, this avoids register pressure on waves without OMOD.
[[gnu::noinline]] __m512i scale_fma_result(__m512i u, uint32_t mode, unsigned rounding) {
  const unsigned omod = (mode >> 6) & 3;
  u = _mm512_castps_si512(prepare_omod_f32(_mm512_castsi512_ps(u), mode));
  auto magnitude = _mm512_and_si512(u, _mm512_set1_epi32(0x7fffffff));
  auto nonzero = _mm512_cmpneq_epi32_mask(magnitude, _mm512_setzero_si512());
  auto scaled = _mm512_add_epi32(u, _mm512_set1_epi32(omod == 3 ? -0x800000 : int(omod << 23)));
  if (omod != 3) {
    auto overflow =
        _mm512_cmpge_epu32_mask(magnitude, _mm512_set1_epi32(0x7f800000 - (omod << 23)));
    auto negative = _mm512_movepi32_mask(u);
    __mmask16 infinity = rounding == 0   ? __mmask16(~0U)
                         : rounding == 1 ? __mmask16(~negative)
                         : rounding == 2 ? negative
                                         : 0;
    auto limit = _mm512_or_si512(_mm512_and_si512(u, _mm512_set1_epi32(INT32_MIN)),
                                 _mm512_mask_blend_epi32(infinity, _mm512_set1_epi32(0x7f7fffff),
                                                         _mm512_set1_epi32(0x7f800000)));
    scaled = _mm512_mask_mov_epi32(scaled, overflow, limit);
  }
  u = _mm512_mask_mov_epi32(u, nonzero, scaled);
  return u;
}

// A zero FP32 result may be exact cancellation or rounded underflow. FP64
// represents every finite FP32 product exactly, so equality with -C proves
// cancellation. Zero products with zero C need no widening.
[[gnu::always_inline]] inline __mmask16 exact_fma_zeros(__m512 a, __m512 b, __m512 c,
                                                        __mmask16 zeros) {
  const auto z = _mm512_setzero_ps();
  __mmask16 exact = (_mm512_cmp_ps_mask(a, z, _CMP_EQ_OQ) | _mm512_cmp_ps_mask(b, z, _CMP_EQ_OQ)) &
                    _mm512_cmp_ps_mask(c, z, _CMP_EQ_OQ);
  if (!(zeros & ~exact)) {
    return exact;
  }
  for (int half = 0; half < 2; ++half) {
    auto aa = half ? _mm512_extractf32x8_ps(a, 1) : _mm512_castps512_ps256(a);
    auto bb = half ? _mm512_extractf32x8_ps(b, 1) : _mm512_castps512_ps256(b);
    auto cc = half ? _mm512_extractf32x8_ps(c, 1) : _mm512_castps512_ps256(c);
    auto product = _mm512_mul_pd(_mm512_cvtps_pd(aa), _mm512_cvtps_pd(bb));
    auto negative_c = _mm512_castsi512_pd(
        _mm512_xor_si512(_mm512_castpd_si512(_mm512_cvtps_pd(cc)), _mm512_set1_epi64(INT64_MIN)));
    exact |= unsigned(_mm512_cmp_pd_mask(product, negative_c, _CMP_EQ_OQ)) << (half * 8);
  }
  return exact;
}

// Keep this hot helper inline: an out-of-line call measurably slows ordinary FMA.
// Normal-result FMA arithmetic. The caller
// handles the flagged nonfinite/tiny cases with its inherited scalar model.
[[gnu::always_inline]] inline __m512i fma_result(__m512 aa, __m512 bb, __m512 cc, uint64_t flags,
                                                 uint32_t mode, __mmask16 &bad) {
  const unsigned rounding = unsigned((flags & RJ_ARITH_FP_ROUND_MASK) >> 24);
  auto result = fused(aa, bb, cc, rounding);
  auto u = _mm512_castps_si512(result), m = _mm512_and_si512(u, _mm512_set1_epi32(0x7fffffff));
  auto zeros = _mm512_cmpeq_epi32_mask(m, _mm512_setzero_si512());
  auto exact_zero = zeros ? exact_fma_zeros(aa, bb, cc, zeros) : __mmask16(0);
  bad |= (_mm512_cmple_epu32_mask(m, _mm512_set1_epi32(0x00800000)) & ~exact_zero) |
         _mm512_cmpge_epu32_mask(m, _mm512_set1_epi32(0x7f800000));
  if (mode & RJ_ARITH_ALU_OMOD_HALF) {
    u = scale_fma_result(u, mode, rounding);
  }
  if (mode & RJ_ARITH_ALU_CLAMP) {
    u = _mm512_mask_mov_epi32(u, _mm512_movepi32_mask(u), _mm512_setzero_si512());
    u = _mm512_min_epu32(u, _mm512_set1_epi32(0x3f800000));
  }
  return u;
}

} // namespace

namespace {

// Keep the inherited scalar machinery out of the normal SIMD loop.
[[gnu::noinline]] __m512i repair_fma(__m512i result, __mmask16 bad, uint64_t flags, uint32_t mode,
                                     const uint32_t *a, const uint32_t *b, const uint32_t *c,
                                     bool rdna3, bool dx9_zero) {
  uint32_t repaired[16];
  _mm512_storeu_si512(repaired, result);
  for (int i = 0; i < 16; ++i) {
    if ((bad >> i) & 1) {
      repaired[i] = repair_fma_f32(a[i], b[i], c[i], flags, mode, rdna3, dx9_zero);
    }
  }
  return _mm512_loadu_si512(repaired);
}

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

// In the admitted normal-or-exact-zero domain, locally rounded FMA matches the
// inherited rocjitsu model. Only special/tiny lanes use scalar repair.
// Stage both halves before stores to preserve aliases during repair.
bool fma_exact_x86_64_v4(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                         const uint32_t *a, const uint32_t *b, const uint32_t *c, bool dx9_zero,
                         bool rdna3) {
  // Stage both vector chunks before destination writes; repaired lanes must still see original
  // inputs.
  __m512i out[2];
  for (int k = 0; k < 2; ++k) {
    __mmask16 bad = 0;
    __m512 aa = prepare_fma_input(_mm512_loadu_si512(a + 16 * k), flags, mode, 0, bad);
    __m512 bb = prepare_fma_input(_mm512_loadu_si512(b + 16 * k), flags, mode, 1, bad);
    __m512 cc = prepare_fma_input(_mm512_loadu_si512(c + 16 * k), flags, mode, 2, bad);
    if (dx9_zero) {
      const auto zero = _mm512_setzero_ps();
      auto has_zero =
          _mm512_cmp_ps_mask(aa, zero, _CMP_EQ_OQ) | _mm512_cmp_ps_mask(bb, zero, _CMP_EQ_OQ);
      aa = _mm512_mask_mov_ps(aa, has_zero, zero);
      bb = _mm512_mask_mov_ps(bb, has_zero, _mm512_set1_ps(1));
    }
    out[k] = fma_result(aa, bb, cc, flags, mode, bad);
    if (bad) {
      out[k] =
          repair_fma(out[k], bad, flags, mode, a + 16 * k, b + 16 * k, c + 16 * k, rdna3, dx9_zero);
    }
  }
  _mm512_mask_storeu_epi32(d, __mmask16(exec_mask), out[0]);
  _mm512_mask_storeu_epi32(d + 16, __mmask16(exec_mask >> 16), out[1]);
  return true;
}

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
                                                      uint32_t mode) {
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
  if (mode & RJ_ARITH_ALU_CLAMP) {
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
  if (!fma_exact_x86_64_v4(flags, ~0U, mode, result, a, b, c)) {
    return false;
  }
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
