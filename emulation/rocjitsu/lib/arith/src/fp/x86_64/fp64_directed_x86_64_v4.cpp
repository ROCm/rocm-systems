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

// Use embedded rounding for the selected arithmetic operation, with exceptions masked.
template <int Rounding> __m512d arithmetic(Fp64 op, __m512d x, __m512d y, __m512d z) {
  if (op == Fp64::Add) {
    return _mm512_add_round_pd(x, y, Rounding | _MM_FROUND_NO_EXC);
  }
  if (op == Fp64::Mul) {
    return _mm512_mul_round_pd(x, y, Rounding | _MM_FROUND_NO_EXC);
  }
  return _mm512_fmadd_round_pd(x, y, z, Rounding | _MM_FROUND_NO_EXC);
}

// Apply fp64_nan_priority to eight lanes.
__m512d nan_priority(Fp64 op, __m512d value, __m512d x, __m512d y, __m512d z) {
  auto quiet = _mm512_set1_epi64(INT64_C(0x0008000000000000));
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
              const uint32_t *const *b, const uint32_t *const *c, uint32_t (&result)[2][32]) {
  const unsigned rounding = unsigned((flags & RJ_ARITH_FP_ROUND_MASK) >> 24);
  const bool flush = flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS;
  __mmask8 nan = 0;
  for (int lane = 0; lane < 32; lane += 8) {
    auto x = input(a, lane, instruction_flags, flush);
    auto y = input(b, lane, instruction_flags >> 1, flush);
    auto z = op == Fp64::Fma ? input(c, lane, instruction_flags >> 2, flush) : _mm512_setzero_pd();
    __m512d value;
    switch (rounding) {
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
    if ((op == Fp64::Mul || op == Fp64::Fma) && ((flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) ||
                                                 (instruction_flags & RJ_ARITH_ALU_OMOD_MASK))) {
      value = flush_tiny_products(value, x, y, z, rounding);
    }
    if constexpr (NanPriority) {
      value = nan_priority(op, value, x, y, z);
    } else {
      nan |= _mm512_cmp_pd_mask(value, value, _CMP_UNORD_Q);
    }
    auto bits = _mm512_castpd_si512(output(value, instruction_flags, flags, rounding));
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(result[0] + lane), _mm512_cvtepi64_epi32(bits));
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(result[1] + lane),
                        _mm512_cvtepi64_epi32(_mm512_srli_epi64(bits, 32)));
  }
  return nan != 0;
}

} // namespace

void fp64_directed_x86_64_v4(Fp64 op, uint64_t flags, uint32_t exec_mask,
                             uint32_t instruction_flags, uint32_t *const *d,
                             const uint32_t *const *a, const uint32_t *const *b,
                             const uint32_t *const *c, bool rdna3) {
  uint32_t result[2][32];
  // A NaN source or an invalid operation always yields a NaN. Such results are rare, so gfx12
  // re-evaluates with NaN priority only when the host produced one.
  if (evaluate<false>(op, flags, instruction_flags, a, b, c, result) && !rdna3) {
    evaluate<true>(op, flags, instruction_flags, a, b, c, result);
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
