// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// FP64 ADD/MUL/FMA using AVX-512 instruction-local guest rounding, without MXCSR changes.
// Operand preparation and masked stores preserve the public split-VGPR layout.
#include "fp/fp64_impl.h"
#include "fp/fp64_round.h"
#include "rj_arith/detail/rj_arith_fp.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {
namespace {

// Load eight split-word values and apply ABS, NEG and optional input flushing.
__m512d input(const uint32_t *const *source, int lane, uint32_t instruction_flags, bool flush) {
  auto low = _mm512_cvtepu32_epi64(
      _mm256_loadu_si256(reinterpret_cast<const __m256i *>(source[0] + lane)));
  auto high = _mm512_cvtepu32_epi64(
      _mm256_loadu_si256(reinterpret_cast<const __m256i *>(source[1] + lane)));
  auto bits = _mm512_or_si512(low, _mm512_slli_epi64(high, 32));
  if (instruction_flags & RJ_ARITH_ALU_ABS_A) {
    bits = _mm512_and_si512(bits, _mm512_set1_epi64(INT64_MAX));
  }
  if (instruction_flags & RJ_ARITH_ALU_NEG_A) {
    bits = _mm512_xor_si512(bits, _mm512_set1_epi64(INT64_MIN));
  }
  if (flush) {
    auto exponent = _mm512_and_si512(bits, _mm512_set1_epi64(0x7ff0000000000000LL));
    auto tiny = _mm512_cmpeq_epi64_mask(exponent, _mm512_setzero_si512());
    bits = _mm512_mask_and_epi64(bits, tiny, bits, _mm512_set1_epi64(INT64_MIN));
  }
  return _mm512_castsi512_pd(bits);
}

// Flush tiny values while preserving sign, or canonicalizing to positive zero after fused OMOD.
__m512d flush_denorm(__m512d value, bool positive_zero) {
  auto bits = _mm512_castpd_si512(value);
  auto tiny = _mm512_cmpeq_epi64_mask(
      _mm512_and_si512(bits, _mm512_set1_epi64(0x7ff0000000000000LL)), _mm512_setzero_si512());
  auto zero =
      positive_zero ? _mm512_setzero_si512() : _mm512_and_si512(bits, _mm512_set1_epi64(INT64_MIN));
  return _mm512_castsi512_pd(_mm512_mask_mov_epi64(bits, tiny, zero));
}

// Apply CDNA OMOD with instruction-local guest rounding, preserving selected NaN encodings.
__m512d scale_cdna(__m512d value, unsigned omod, unsigned rounding) {
  const double scales[] = {1, 2, 4, 0.5};
  auto scale = _mm512_set1_pd(scales[omod]);
  __m512d scaled;
  switch (rounding) {
  case 0:
    scaled = _mm512_mul_round_pd(value, scale, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    break;
  case 1:
    scaled = _mm512_mul_round_pd(value, scale, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
    break;
  case 2:
    scaled = _mm512_mul_round_pd(value, scale, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
    break;
  default:
    scaled = _mm512_mul_round_pd(value, scale, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    break;
  }
  return _mm512_mask_mov_pd(scaled, _mm512_cmp_pd_mask(value, value, _CMP_UNORD_Q), value);
}

// Scale encoded normal values, selecting the guest-directed overflow encoding.
// OMOD flushes input/output subnormals independently of the guest denormal mode.
__m512d output(__m512d value, uint32_t instruction_flags, uint64_t flags, unsigned rounding) {
  auto bits = _mm512_castpd_si512(value);
  auto sign = _mm512_and_si512(bits, _mm512_set1_epi64(INT64_MIN));
  auto exponent = _mm512_and_si512(_mm512_srli_epi64(bits, 52), _mm512_set1_epi64(2047));
  unsigned omod = (instruction_flags >> 6) & 3;
  if (omod) {
    auto normal = _mm512_cmpneq_epi64_mask(exponent, _mm512_setzero_si512()) &
                  _mm512_cmpneq_epi64_mask(exponent, _mm512_set1_epi64(2047));
    auto adjusted = _mm512_add_epi64(exponent, _mm512_set1_epi64(omod == 3 ? -1 : int(omod)));
    auto scaled =
        _mm512_or_si512(_mm512_and_si512(bits, _mm512_set1_epi64(INT64_MIN | 0x000fffffffffffffLL)),
                        _mm512_slli_epi64(adjusted, 52));
    bits = _mm512_mask_mov_epi64(bits, normal, scaled);
    bits = _mm512_mask_mov_epi64(bits, _mm512_cmpeq_epi64_mask(exponent, _mm512_setzero_si512()),
                                 _mm512_setzero_si512());
    bits = _mm512_mask_mov_epi64(
        bits, normal & _mm512_cmp_epi64_mask(adjusted, _mm512_setzero_si512(), _MM_CMPINT_LE),
        sign);
    auto negative = _mm512_cmpneq_epi64_mask(sign, _mm512_setzero_si512());
    __mmask8 infinity = rounding == 0   ? __mmask8(0xff)
                        : rounding == 1 ? __mmask8(~negative)
                        : rounding == 2 ? negative
                                        : __mmask8(0);
    auto overflow = _mm512_mask_blend_epi64(infinity, _mm512_set1_epi64(0x7fefffffffffffffLL),
                                            _mm512_set1_epi64(0x7ff0000000000000LL));
    bits = _mm512_mask_mov_epi64(
        bits, normal & _mm512_cmp_epi64_mask(adjusted, _mm512_set1_epi64(2047), _MM_CMPINT_GE),
        _mm512_or_si512(sign, overflow));
  }
  value = _mm512_castsi512_pd(bits);
  if (instruction_flags & RJ_ARITH_ALU_CLAMP) {
    auto clamped = _mm512_min_pd(_mm512_max_pd(value, _mm512_setzero_pd()), _mm512_set1_pd(1));
    value = flags & RJ_ARITH_FP_DX10_CLAMP_OFF
                ? _mm512_mask_mov_pd(clamped, _mm512_cmp_pd_mask(value, value, _CMP_UNORD_Q), value)
                : clamped;
  }
  if (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) {
    bits = _mm512_castpd_si512(value);
    auto tiny = _mm512_cmpeq_epi64_mask(
        _mm512_and_si512(bits, _mm512_set1_epi64(0x7ff0000000000000LL)), _mm512_setzero_si512());
    value =
        _mm512_castsi512_pd(_mm512_mask_and_epi64(bits, tiny, bits, _mm512_set1_epi64(INT64_MIN)));
  }
  return value;
}

// Re-evaluate the selected lanes with the integer model. Kept out of line so that the
// common case carries no stack frame for it.
[[gnu::cold, gnu::noinline]] __m512d flush_tiny_lanes(__m512d value, __m512d x, __m512d y,
                                                      __m512d z, unsigned lanes,
                                                      unsigned rounding) {
  uint64_t result[8], a[8], b[8], c[8];
  _mm512_storeu_pd(result, value);
  _mm512_storeu_pd(a, x);
  _mm512_storeu_pd(b, y);
  _mm512_storeu_pd(c, z);
  for (; lanes; lanes &= lanes - 1) {
    int i = __builtin_ctz(lanes);
    result[i] = fma_flush_tiny_f64(result[i], a[i], b[i], c[i], rounding);
  }
  return _mm512_loadu_pd(result);
}

// A product rounded to the smallest normal may still be tiny after rounding. Such lanes are
// rare, so only they leave the SIMD path.
inline __m512d flush_tiny_products(__m512d value, __m512d x, __m512d y, __m512d z,
                                   unsigned rounding) {
  auto magnitude = _mm512_and_si512(_mm512_castpd_si512(value), _mm512_set1_epi64(INT64_MAX));
  unsigned boundary = _mm512_cmpeq_epi64_mask(magnitude, _mm512_set1_epi64(0x0010000000000000LL));
  return __builtin_expect(boundary != 0, 0) ? flush_tiny_lanes(value, x, y, z, boundary, rounding)
                                            : value;
}

// Normalize finite values to [0.5,1), retaining signs and quieting NaNs.
__m512d mantissa(__m512d x) {
  auto original = _mm512_castpd_si512(x);
  auto magnitude = _mm512_and_si512(original, _mm512_set1_epi64(INT64_MAX));
  auto tiny = _mm512_cmplt_epi64_mask(magnitude, _mm512_set1_epi64(0x0010000000000000LL));
  // Scaling only subnormal lanes avoids unnecessary overflow in ordinary lanes.
  auto normalized = _mm512_mask_mul_pd(x, tiny, x, _mm512_set1_pd(0x1p54));
  auto bits = _mm512_or_si512(_mm512_and_si512(_mm512_castpd_si512(normalized),
                                               _mm512_set1_epi64(int64_t(0x800fffffffffffffULL))),
                              _mm512_set1_epi64(0x3fe0000000000000LL));
  auto special = _mm512_cmpeq_epi64_mask(magnitude, _mm512_setzero_si512()) |
                 _mm512_cmpge_epi64_mask(magnitude, _mm512_set1_epi64(0x7ff0000000000000LL));
  auto nan = _mm512_cmpgt_epi64_mask(magnitude, _mm512_set1_epi64(0x7ff0000000000000LL));
  original = _mm512_mask_or_epi64(original, nan, original, _mm512_set1_epi64(0x0008000000000000LL));
  return _mm512_castsi512_pd(_mm512_mask_mov_epi64(bits, special, original));
}

// Named integral operations have fixed rounding; FRACT's subtraction uses the guest mode.
// ADD/MUL/FMA likewise use embedded rounding, with host exceptions suppressed.
template <int Rounding> __m512d arithmetic(Fp64 op, __m512d x, __m512d y, __m512d z) {
  if (op == Fp64::Add) {
    return _mm512_add_round_pd(x, y, Rounding | _MM_FROUND_NO_EXC);
  }
  if (op == Fp64::Mul) {
    return _mm512_mul_round_pd(x, y, Rounding | _MM_FROUND_NO_EXC);
  }
  if (op == Fp64::Fma) {
    return _mm512_fmadd_round_pd(x, y, z, Rounding | _MM_FROUND_NO_EXC);
  }
  switch (op) {
  case Fp64::Trunc:
    return _mm512_roundscale_pd(x, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
  case Fp64::Ceil:
    return _mm512_roundscale_pd(x, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
  case Fp64::Rndne:
    return _mm512_roundscale_pd(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  case Fp64::Floor:
    return _mm512_roundscale_pd(x, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
  case Fp64::FrexpMant:
    return mantissa(x);
  default: {
    auto integral = _mm512_roundscale_pd(x, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
    auto value = _mm512_sub_round_pd(x, integral, Rounding | _MM_FROUND_NO_EXC);
    value = _mm512_mask_mov_pd(value, _mm512_cmp_pd_mask(value, _mm512_setzero_pd(), _CMP_EQ_OQ),
                               _mm512_setzero_pd());
    auto limit = _mm512_castsi512_pd(_mm512_set1_epi64(0x3fefffffffffffffLL));
    return _mm512_mask_mov_pd(value, _mm512_cmp_pd_mask(value, limit, _CMP_GT_OQ), limit);
  }
  }
}

// Apply fp64_nan_priority to eight lanes.
__m512d nan_priority(Fp64 op, __m512d value, __m512d x, __m512d y, __m512d z, bool quiet_snan) {
  auto quiet = _mm512_set1_epi64(quiet_snan ? INT64_C(0x0008000000000000) : 0);
  auto pick = [&](__m512d result, __m512d source) {
    return _mm512_mask_mov_pd(
        result, _mm512_cmp_pd_mask(source, source, _CMP_UNORD_Q),
        _mm512_castsi512_pd(_mm512_or_si512(_mm512_castpd_si512(source), quiet)));
  };
  if (op == Fp64::Fma) {
    value = pick(value, z);
    auto zero = _mm512_setzero_pd(), inf = _mm512_set1_pd(__builtin_inf());
    auto invalid = (_mm512_cmp_pd_mask(x, zero, _CMP_EQ_OQ) &
                    _mm512_cmp_pd_mask(_mm512_abs_pd(y), inf, _CMP_EQ_OQ)) |
                   (_mm512_cmp_pd_mask(_mm512_abs_pd(x), inf, _CMP_EQ_OQ) &
                    _mm512_cmp_pd_mask(y, zero, _CMP_EQ_OQ));
    value = _mm512_mask_mov_pd(
        value, invalid, _mm512_castsi512_pd(_mm512_set1_epi64(int64_t(0xfff8000000000000ULL))));
  }
  return pick(pick(value, y), x);
}

// Evaluate all lanes into result. NanPriority applies nan_priority; otherwise host NaNs are kept,
// and the return value tells whether any lane produced a NaN.
template <bool NanPriority>
bool evaluate(Fp64 op, uint64_t flags, uint32_t instruction_flags, const uint32_t *const *a,
              const uint32_t *const *b, const uint32_t *const *c, uint32_t (&result)[2][32],
              bool quiet_snan, bool cdna) {
  const unsigned rounding = unsigned((flags & RJ_ARITH_FP_ROUND_MASK) >> 24);
  const bool flush = flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS;
  __mmask8 nan = 0;
  for (int lane = 0; lane < 32; lane += 8) {
    auto x = input(a, lane, instruction_flags, flush);
    auto y = (op == Fp64::Add || op == Fp64::Mul || op == Fp64::Fma)
                 ? input(b, lane, instruction_flags >> 1, flush)
                 : _mm512_setzero_pd();
    auto z = op == Fp64::Fma ? input(c, lane, instruction_flags >> 2, flush) : _mm512_setzero_pd();
    __m512d value;
    switch (rounding) {
    case 0:
      value = arithmetic<_MM_FROUND_TO_NEAREST_INT>(op, x, y, z);
      break;
    case 1:
      value = arithmetic<_MM_FROUND_TO_POS_INF>(op, x, y, z);
      break;
    case 2:
      value = arithmetic<_MM_FROUND_TO_NEG_INF>(op, x, y, z);
      break;
    default:
      value = arithmetic<_MM_FROUND_TO_ZERO>(op, x, y, z);
      break;
    }
    // Products judge tininess after rounding; ADD results are exact on the subnormal grid.
    if (!cdna && (op == Fp64::Mul || op == Fp64::Fma) &&
        ((flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) ||
         (instruction_flags & RJ_ARITH_ALU_OMOD_MASK))) {
      value = flush_tiny_products(value, x, y, z, rounding);
    }
    if constexpr (NanPriority) {
      value = nan_priority(op, value, x, y, z, quiet_snan);
    } else {
      nan |= _mm512_cmp_pd_mask(value, value, _CMP_UNORD_Q);
    }
    // Rounding and mantissa extraction preserve source sNaNs when gfx11 IEEE is clear.
    if (!quiet_snan && op != Fp64::Add && op != Fp64::Mul && op != Fp64::Fma) {
      value = _mm512_mask_mov_pd(value, _mm512_cmp_pd_mask(x, x, _CMP_UNORD_Q), x);
    }
    uint32_t output_mode = instruction_flags;
    if (cdna && op == Fp64::Fma && (output_mode & RJ_ARITH_ALU_OMOD_MASK)) {
      if (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) {
        value = flush_denorm(value, false);
      }
      value = flush_denorm(scale_cdna(value, (output_mode >> 6) & 3, rounding), true);
      output_mode &= ~RJ_ARITH_ALU_OMOD_MASK;
    }
    auto bits = _mm512_castpd_si512(output(value, output_mode, flags, rounding));
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(result[0] + lane), _mm512_cvtepi64_epi32(bits));
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(result[1] + lane),
                        _mm512_cvtepi64_epi32(_mm512_srli_epi64(bits, 32)));
  }
  return nan != 0;
}

} // namespace

void fp64_x86_64_v4(Fp64 op, uint64_t flags, uint32_t exec_mask, uint32_t instruction_flags,
                    uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                    const uint32_t *const *c, bool quiet_snan, bool cdna) {
  uint32_t result[2][32];
  // A NaN source or an invalid operation always yields a NaN. Such results are rare, so this
  // re-evaluates with NaN priority only when the host produced one.
  if (evaluate<false>(op, flags, instruction_flags, a, b, c, result, quiet_snan, cdna)) {
    evaluate<true>(op, flags, instruction_flags, a, b, c, result, quiet_snan, cdna);
  }
  // Delay writes until every input has been consumed, including cross-half aliases.
  for (int reg = 0; reg < 2; ++reg) {
    for (int lane = 0; lane < 32; lane += 8) {
      _mm256_mask_storeu_epi32(
          d[reg] + lane, __mmask8(exec_mask >> lane),
          _mm256_loadu_si256(reinterpret_cast<const __m256i *>(result[reg] + lane)));
    }
  }
}

} // namespace rocjitsu::arith
