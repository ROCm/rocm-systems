// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 FP32 scaling by powers of two with guest rounding and denormal policies.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "common/x86_64/alu_x86_64_v3.h"
#include "fp/ldexp.h"
#include "rj_arith/rj_arith.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {
namespace {

// Clamp signed 64-bit lanes to the supplied inclusive bounds.
__m256i clamp64(__m256i value, int low, int high) {
  auto lo = _mm256_set1_epi64x(low), hi = _mm256_set1_epi64x(high);
  value = _mm256_blendv_epi8(value, lo, _mm256_cmpgt_epi64(lo, value));
  return _mm256_blendv_epi8(value, hi, _mm256_cmpgt_epi64(value, hi));
}

// Scale FP32 by powers of two.
template <bool GuestDenormals>
void run32(uint64_t flags, uint32_t instruction_flags, uint32_t result[2][32],
           const uint32_t *const *a, const uint32_t *b) {
  auto keep = _mm256_set1_epi32(instruction_flags & RJ_ARITH_ALU_ABS_A ? INT32_MAX : -1);
  auto flip = _mm256_set1_epi32(instruction_flags & RJ_ARITH_ALU_NEG_A ? INT32_MIN : 0);
  const double scales[] = {1, 2, 4, 0.5};
  for (int lane = 0; lane < 32; lane += 8) {
    auto raw = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a[0] + lane));
    auto adjustment = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + lane));
    adjustment = _mm256_min_epi32(_mm256_max_epi32(adjustment, _mm256_set1_epi32(-4096)),
                                  _mm256_set1_epi32(4096));
    raw = _mm256_xor_si256(_mm256_and_si256(raw, keep), flip);
    if (GuestDenormals && (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS)) {
      raw = _mm256_castps_si256(flush_denorm_f32(_mm256_castsi256_ps(raw)));
    }
    auto magnitude = _mm256_and_si256(raw, _mm256_set1_epi32(INT32_MAX));
    auto subnormal = _mm256_cmpgt_epi32(_mm256_set1_epi32(0x800000), magnitude);
    auto small = _mm256_and_si256(raw, subnormal);
    auto scaled =
        _mm256_castps_si256(_mm256_mul_ps(_mm256_castsi256_ps(small), _mm256_set1_ps(0x1p24)));
    auto normalized = _mm256_blendv_epi8(raw, scaled, subnormal);
    auto exponent =
        _mm256_srli_epi32(_mm256_and_si256(normalized, _mm256_set1_epi32(INT32_MAX)), 23);
    exponent = _mm256_add_epi32(exponent, adjustment);
    exponent = _mm256_sub_epi32(exponent, _mm256_and_si256(subnormal, _mm256_set1_epi32(24)));
    auto significand = _mm256_and_si256(normalized, _mm256_set1_epi32((INT32_MIN | 0x7fffff)));
    auto normal = _mm256_or_si256(
        significand,
        _mm256_slli_epi32(_mm256_min_epi32(_mm256_max_epi32(exponent, _mm256_set1_epi32(1)),
                                           _mm256_set1_epi32(254)),
                          23));

    // Move an underflowing value into the normal range, then round once on
    // scaling back down. Clamping the integer exponent prevents wraparound.
    auto raised = _mm256_add_epi32(exponent, _mm256_set1_epi32(24));
    auto tiny = _mm256_or_si256(
        significand,
        _mm256_slli_epi32(_mm256_min_epi32(_mm256_max_epi32(raised, _mm256_set1_epi32(1)),
                                           _mm256_set1_epi32(254)),
                          23));
    auto underflow =
        _mm256_castps_si256(_mm256_mul_ps(_mm256_castsi256_ps(tiny), _mm256_set1_ps(0x1p-24)));
    auto bits =
        _mm256_blendv_epi8(normal, underflow, _mm256_cmpgt_epi32(_mm256_set1_epi32(1), exponent));
    auto sign = _mm256_and_si256(raw, _mm256_set1_epi32(INT32_MIN));
    bits = _mm256_blendv_epi8(bits, sign, _mm256_cmpgt_epi32(_mm256_set1_epi32(-23), exponent));
    bits = _mm256_blendv_epi8(bits, _mm256_or_si256(sign, _mm256_set1_epi32(0x7f800000)),
                              _mm256_cmpgt_epi32(exponent, _mm256_set1_epi32(254)));
    auto special =
        _mm256_or_si256(_mm256_cmpeq_epi32(magnitude, _mm256_setzero_si256()),
                        _mm256_cmpgt_epi32(magnitude, _mm256_set1_epi32(0x7f800000 - 1)));
    auto quiet = _mm256_and_si256(_mm256_cmpgt_epi32(magnitude, _mm256_set1_epi32(0x7f800000)),
                                  _mm256_set1_epi32(0x400000));
    bits = _mm256_blendv_epi8(bits, _mm256_or_si256(raw, quiet), special);
    auto value = _mm256_castsi256_ps(bits);
    if ((instruction_flags & RJ_ARITH_ALU_OMOD_HALF) ||
        (GuestDenormals && (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS))) {
      auto finite = _mm256_cmpgt_epi32(_mm256_set1_epi32(0x7f800000), magnitude);
      auto tiny_before_rounding =
          _mm256_and_si256(finite, _mm256_cmpgt_epi32(_mm256_set1_epi32(1), exponent));
      auto zero = instruction_flags & RJ_ARITH_ALU_OMOD_HALF ? _mm256_setzero_ps()
                                                             : _mm256_castsi256_ps(sign);
      value = _mm256_blendv_ps(value, zero, _mm256_castsi256_ps(tiny_before_rounding));
    }
    if (instruction_flags & RJ_ARITH_ALU_OMOD_HALF) {
      value = prepare_omod_f32(value, instruction_flags);
      value = _mm256_mul_ps(value, _mm256_set1_ps(scales[(instruction_flags >> 6) & 3]));
    }
    if (instruction_flags & RJ_ARITH_ALU_CLAMP) {
      auto clamped = _mm256_min_ps(_mm256_max_ps(value, _mm256_setzero_ps()), _mm256_set1_ps(1));
      if (GuestDenormals && (flags & RJ_ARITH_FP_DX10_CLAMP_OFF)) {
        clamped = _mm256_blendv_ps(clamped, value, _mm256_cmp_ps(value, value, _CMP_UNORD_Q));
      }
      value = clamped;
    }
    if (GuestDenormals && (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
      value = flush_denorm_f32(value);
    }
    bits = _mm256_castps_si256(value);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(result[0] + lane), bits);
  }
}

// Scale FP32 by powers of two.
template <bool GuestDenormals>
void run64(uint64_t flags, uint32_t instruction_flags, uint32_t result[2][32],
           const uint32_t *const *a, const uint32_t *b) {
  auto keep = _mm256_set1_epi64x(instruction_flags & RJ_ARITH_ALU_ABS_A ? INT64_MAX : -1);
  auto flip = _mm256_set1_epi64x(instruction_flags & RJ_ARITH_ALU_NEG_A ? INT64_MIN : 0);
  const double scales[] = {1, 2, 4, 0.5};
  for (int lane = 0; lane < 32; lane += 4) {
    auto low =
        _mm256_cvtepu32_epi64(_mm_loadu_si128(reinterpret_cast<const __m128i *>(a[0] + lane)));
    auto high =
        _mm256_cvtepu32_epi64(_mm_loadu_si128(reinterpret_cast<const __m128i *>(a[1] + lane)));
    auto raw = _mm256_or_si256(low, _mm256_slli_epi64(high, 32));
    auto power = _mm_loadu_si128(reinterpret_cast<const __m128i *>(b + lane));
    power = _mm_min_epi32(_mm_max_epi32(power, _mm_set1_epi32(-4096)), _mm_set1_epi32(4096));
    auto adjustment = _mm256_cvtepi32_epi64(power);
    raw = _mm256_xor_si256(_mm256_and_si256(raw, keep), flip);
    if (GuestDenormals && (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS)) {
      raw = _mm256_castpd_si256(flush_denorm_f64(_mm256_castsi256_pd(raw)));
    }
    auto magnitude = _mm256_and_si256(raw, _mm256_set1_epi64x(INT64_MAX));
    auto subnormal = _mm256_cmpgt_epi64(_mm256_set1_epi64x(INT64_C(0x10000000000000)), magnitude);
    auto small = _mm256_and_si256(raw, subnormal);
    auto scaled =
        _mm256_castpd_si256(_mm256_mul_pd(_mm256_castsi256_pd(small), _mm256_set1_pd(0x1p54)));
    auto normalized = _mm256_blendv_epi8(raw, scaled, subnormal);
    auto exponent =
        _mm256_srli_epi64(_mm256_and_si256(normalized, _mm256_set1_epi64x(INT64_MAX)), 52);
    exponent = _mm256_add_epi64(exponent, adjustment);
    exponent = _mm256_sub_epi64(exponent, _mm256_and_si256(subnormal, _mm256_set1_epi64x(54)));
    auto significand =
        _mm256_and_si256(normalized, _mm256_set1_epi64x((INT64_MIN | INT64_C(0xfffffffffffff))));
    auto normal = _mm256_or_si256(significand, _mm256_slli_epi64(clamp64(exponent, 1, 2046), 52));

    // Move an underflowing value into the normal range, then round once on
    // scaling back down. Clamping the integer exponent prevents wraparound.
    auto raised = _mm256_add_epi64(exponent, _mm256_set1_epi64x(54));
    auto tiny = _mm256_or_si256(significand, _mm256_slli_epi64(clamp64(raised, 1, 2046), 52));
    auto underflow =
        _mm256_castpd_si256(_mm256_mul_pd(_mm256_castsi256_pd(tiny), _mm256_set1_pd(0x1p-54)));
    auto bits =
        _mm256_blendv_epi8(normal, underflow, _mm256_cmpgt_epi64(_mm256_set1_epi64x(1), exponent));
    auto sign = _mm256_and_si256(raw, _mm256_set1_epi64x(INT64_MIN));
    bits = _mm256_blendv_epi8(bits, sign, _mm256_cmpgt_epi64(_mm256_set1_epi64x(-52), exponent));
    bits = _mm256_blendv_epi8(
        bits, _mm256_or_si256(sign, _mm256_set1_epi64x(INT64_C(0x7ff0000000000000))),
        _mm256_cmpgt_epi64(exponent, _mm256_set1_epi64x(2046)));
    auto special = _mm256_or_si256(
        _mm256_cmpeq_epi64(magnitude, _mm256_setzero_si256()),
        _mm256_cmpgt_epi64(magnitude, _mm256_set1_epi64x(INT64_C(0x7ff0000000000000) - 1)));
    auto quiet = _mm256_and_si256(
        _mm256_cmpgt_epi64(magnitude, _mm256_set1_epi64x(INT64_C(0x7ff0000000000000))),
        _mm256_set1_epi64x(INT64_C(0x8000000000000)));
    bits = _mm256_blendv_epi8(bits, _mm256_or_si256(raw, quiet), special);
    auto value = _mm256_castsi256_pd(bits);
    if ((instruction_flags & RJ_ARITH_ALU_OMOD_HALF) ||
        (GuestDenormals && (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS))) {
      auto finite = _mm256_cmpgt_epi64(_mm256_set1_epi64x(INT64_C(0x7ff0000000000000)), magnitude);
      auto tiny_before_rounding =
          _mm256_and_si256(finite, _mm256_cmpgt_epi64(_mm256_set1_epi64x(1), exponent));
      auto zero = instruction_flags & RJ_ARITH_ALU_OMOD_HALF ? _mm256_setzero_pd()
                                                             : _mm256_castsi256_pd(sign);
      value = _mm256_blendv_pd(value, zero, _mm256_castsi256_pd(tiny_before_rounding));
    }
    if (instruction_flags & RJ_ARITH_ALU_OMOD_HALF) {
      value = prepare_omod_f64(value, instruction_flags);
      value = _mm256_mul_pd(value, _mm256_set1_pd(scales[(instruction_flags >> 6) & 3]));
    }
    if (instruction_flags & RJ_ARITH_ALU_CLAMP) {
      auto clamped = _mm256_min_pd(_mm256_max_pd(value, _mm256_setzero_pd()), _mm256_set1_pd(1));
      if (GuestDenormals && (flags & RJ_ARITH_FP_DX10_CLAMP_OFF)) {
        clamped = _mm256_blendv_pd(clamped, value, _mm256_cmp_pd(value, value, _CMP_UNORD_Q));
      }
      value = clamped;
    }
    if (GuestDenormals && (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
      value = flush_denorm_f64(value);
    }
    bits = _mm256_castpd_si256(value);
    auto packed = _mm256_permutevar8x32_epi32(bits, _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7));
    _mm_storeu_si128(reinterpret_cast<__m128i *>(result[0] + lane), _mm256_castsi256_si128(packed));
    _mm_storeu_si128(reinterpret_cast<__m128i *>(result[1] + lane),
                     _mm256_extracti128_si256(packed, 1));
  }
}

// Scale FP32 by powers of two for the selected lanes. Requires validated controls; preserve
// inactive destination lanes.
template <bool GuestDenormals>
void run(bool fp64, uint64_t flags, uint32_t exec_mask, uint32_t instruction_flags,
         uint32_t *const *d, const uint32_t *const *a, const uint32_t *b) {
  uint32_t result[2][32];
  if (fp64) {
    run64<GuestDenormals>(flags, instruction_flags, result, a, b);
  } else {
    run32<GuestDenormals>(flags, instruction_flags, result, a, b);
  }
  for (int reg = 0; reg < (fp64 ? 2 : 1); ++reg) {
    for (int lane = 0; lane < 32; lane += 8) {
      auto lane_exec_mask = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                                              _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
      _mm256_maskstore_epi32(
          reinterpret_cast<int *>(d[reg] + lane), lane_exec_mask,
          _mm256_loadu_si256(reinterpret_cast<const __m256i *>(result[reg] + lane)));
    }
  }
}

} // namespace

// Scale FP32 by powers of two. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
void ldexp_x86_64_v3(bool fp64, uint32_t exec_mask, uint32_t instruction_flags, uint32_t *const *d,
                     const uint32_t *const *a, const uint32_t *b) {
  run<false>(fp64, 0, exec_mask, instruction_flags, d, a, b);
}

// Scale FP32 by powers of two. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
void ldexp_guest_x86_64_v3(bool fp64, uint64_t flags, uint32_t exec_mask,
                           uint32_t instruction_flags, uint32_t *const *d, const uint32_t *const *a,
                           const uint32_t *b) {
  run<true>(fp64, flags, exec_mask, instruction_flags, d, a, b);
}

} // namespace rocjitsu::arith
