// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// FP64 FMA using AVX-512 instruction-local guest rounding, without MXCSR changes.
// Operand preparation and masked stores preserve the public split-VGPR layout.
#include "fp/fp64_impl.h"
#include "rj_arith/rj_arith.h"

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

} // namespace

void fma_f64_directed_x86_64_v4(uint64_t flags, uint32_t exec_mask, uint32_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c) {
  const unsigned rounding = unsigned((flags & RJ_ARITH_FP_ROUND_MASK) >> 24);
  const bool flush = flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS;
  uint32_t result[2][32];
  for (int lane = 0; lane < 32; lane += 8) {
    auto x = input(a, lane, instruction_flags, flush);
    auto y = input(b, lane, instruction_flags >> 1, flush);
    auto z = input(c, lane, instruction_flags >> 2, flush);
    __m512d value;
    switch (rounding) {
    case 1:
      value = _mm512_fmadd_round_pd(x, y, z, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
      break;
    case 2:
      value = _mm512_fmadd_round_pd(x, y, z, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
      break;
    default:
      value = _mm512_fmadd_round_pd(x, y, z, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
      break;
    }
    auto bits = _mm512_castpd_si512(output(value, instruction_flags, flags, rounding));
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(result[0] + lane), _mm512_cvtepi64_epi32(bits));
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(result[1] + lane),
                        _mm512_cvtepi64_epi32(_mm512_srli_epi64(bits, 32)));
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
