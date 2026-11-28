// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 FP32/FP64 scaling by powers of two with guest rounding and denormal policies.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "common/bits.h"
#include "common/compiler.h"
#include "common/x86_64/alu_x86_64_v3.h"
#include "fp/ldexp.h"
#include "rj_arith/detail/rj_arith_fp.h"

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

// Scale FP64 by powers of two, including subnormal inputs and outputs.
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

// Scale FP64 by powers of two for the selected lanes. Requires validated controls; preserve
// inactive destination lanes.
template <bool GuestDenormals>
void run(uint64_t flags, uint32_t exec_mask, uint32_t instruction_flags, uint32_t *const *d,
         const uint32_t *const *a, const uint32_t *b) {
  uint32_t result[2][32];
  // NaNs qualify for exponent-only scaling; range/subnormal cases retain general SIMD.
  const bool general = (instruction_flags & (RJ_ARITH_ALU_OMOD_MASK | RJ_ARITH_ALU_CLAMP)) ||
                       ldexp_f64_normal_x86_64_v3(flags, instruction_flags, true, result, a, b);
  if (general) {
    run64<GuestDenormals>(flags, instruction_flags, result, a, b);
  }
  for (int reg = 0; reg < 2; ++reg) {
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

// Stage exact normal scales and preserved/quieted NaNs. Return only the lanes
// whose subnormal input or rounded result needs the baseline integer helper.
uint32_t ldexp_f64_normal_x86_64_v3(uint64_t flags, uint32_t instruction_flags, bool quiet_snan,
                                    uint32_t result[2][32], const uint32_t *const *a,
                                    const uint32_t *b) {
  uint32_t scalar_lanes = 0;
  const auto zero = _mm256_setzero_si256(), maximum = _mm256_set1_epi64x(2047);
  RJ_ARITH_UNROLL(8)
  for (int lane = 0; lane < 32; lane += 4) {
    auto low =
        _mm256_cvtepu32_epi64(_mm_loadu_si128(reinterpret_cast<const __m128i *>(a[0] + lane)));
    auto high =
        _mm256_cvtepu32_epi64(_mm_loadu_si128(reinterpret_cast<const __m128i *>(a[1] + lane)));
    auto raw = _mm256_or_si256(low, _mm256_slli_epi64(high, 32));
    if (instruction_flags & RJ_ARITH_ALU_ABS_A) {
      raw = _mm256_and_si256(raw, _mm256_set1_epi64x(INT64_MAX));
    }
    if (instruction_flags & RJ_ARITH_ALU_NEG_A) {
      raw = _mm256_xor_si256(raw, _mm256_set1_epi64x(INT64_MIN));
    }
    auto magnitude = _mm256_and_si256(raw, _mm256_set1_epi64x(INT64_MAX));
    auto exponent = _mm256_srli_epi64(magnitude, 52);
    auto adjustment =
        _mm256_cvtepi32_epi64(_mm_loadu_si128(reinterpret_cast<const __m128i *>(b + lane)));
    auto scaled_exponent = _mm256_add_epi64(exponent, clamp64(adjustment, -4096, 4096));
    auto normal =
        _mm256_and_si256(_mm256_cmpgt_epi64(exponent, zero), _mm256_cmpgt_epi64(maximum, exponent));
    normal =
        _mm256_and_si256(normal, _mm256_and_si256(_mm256_cmpgt_epi64(scaled_exponent, zero),
                                                  _mm256_cmpgt_epi64(maximum, scaled_exponent)));
    auto bits =
        _mm256_or_si256(_mm256_and_si256(raw, _mm256_set1_epi64x(INT64_MIN | 0x000fffffffffffffLL)),
                        _mm256_slli_epi64(scaled_exponent, 52));
    bits = _mm256_blendv_epi8(raw, bits, normal);
    if (quiet_snan) {
      bits = _mm256_or_si256(
          bits,
          _mm256_and_si256(_mm256_cmpgt_epi64(magnitude, _mm256_set1_epi64x(0x7ff0000000000000LL)),
                           _mm256_set1_epi64x(0x0008000000000000LL)));
    }
    auto resolved = _mm256_or_si256(normal, _mm256_or_si256(_mm256_cmpeq_epi64(magnitude, zero),
                                                            _mm256_cmpeq_epi64(exponent, maximum)));
    if (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) {
      auto tiny = _mm256_cmpeq_epi64(exponent, zero);
      bits = _mm256_blendv_epi8(bits, _mm256_and_si256(raw, _mm256_set1_epi64x(INT64_MIN)), tiny);
      resolved = _mm256_or_si256(resolved, tiny);
    }
    scalar_lanes |= uint32_t(~_mm256_movemask_pd(_mm256_castsi256_pd(resolved)) & 15) << lane;
    auto packed = _mm256_permutevar8x32_epi32(bits, _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7));
    _mm_storeu_si128(reinterpret_cast<__m128i *>(result[0] + lane), _mm256_castsi256_si128(packed));
    _mm_storeu_si128(reinterpret_cast<__m128i *>(result[1] + lane),
                     _mm256_extracti128_si256(packed, 1));
  }
  return scalar_lanes;
}

// Publish staged words only after every source and scalar repair has been read.
void ldexp_f64_store_x86_64_v3(uint32_t exec_mask, uint32_t *const *d,
                               const uint32_t result[2][32]) {
  RJ_ARITH_UNROLL(2)
  for (int reg = 0; reg < 2; ++reg) {
    RJ_ARITH_UNROLL(4)
    for (int lane = 0; lane < 32; lane += 8) {
      auto lane_exec_mask = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                                              _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
      _mm256_maskstore_epi32(
          reinterpret_cast<int *>(d[reg] + lane), lane_exec_mask,
          _mm256_loadu_si256(reinterpret_cast<const __m256i *>(result[reg] + lane)));
    }
  }
}

// Scale FP64 with nearest rounding, preserving complete VGPR aliases and inactive lanes.
void ldexp_f64_loose_x86_64_v3(uint64_t flags, uint32_t exec_mask, uint32_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a, const uint32_t *b) {
  if (flags & (RJ_ARITH_FP_FLUSH_INPUT_DENORMALS | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS |
               RJ_ARITH_FP_DX10_CLAMP_OFF)) {
    run<true>(flags, exec_mask, instruction_flags, d, a, b);
  } else {
    run<false>(0, exec_mask, instruction_flags, d, a, b);
  }
}

// Exact normal scales change only the exponent. Boundary and subnormal lanes
// retain the baseline rounding model; no lane forces a whole-wave fallback.
void ldexp_f32_exact_x86_64_v3(uint64_t flags, uint32_t exec_mask, uint32_t instruction_flags,
                               uint32_t *d, const uint32_t *a, const uint32_t *b, bool quiet_snan) {
  uint32_t result[32];
  uint32_t repairs = 0;
  const unsigned omod = (instruction_flags >> 6) & 3;
  const auto delta = _mm256_set1_epi32(omod == 3 ? -1 : int(omod));
  const auto zero = _mm256_setzero_si256(), infinity = _mm256_set1_epi32(0x7f800000);
  RJ_ARITH_UNROLL(4)
  for (int lane = 0; lane < 32; lane += 8) {
    auto raw = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane));
    if (instruction_flags & RJ_ARITH_ALU_ABS_A) {
      raw = _mm256_and_si256(raw, _mm256_set1_epi32(INT32_MAX));
    }
    if (instruction_flags & RJ_ARITH_ALU_NEG_A) {
      raw = _mm256_xor_si256(raw, _mm256_set1_epi32(INT32_MIN));
    }
    if (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) {
      raw = _mm256_castps_si256(flush_denorm_f32(_mm256_castsi256_ps(raw)));
    }
    const auto magnitude = _mm256_and_si256(raw, _mm256_set1_epi32(INT32_MAX));
    const auto exponent = _mm256_srli_epi32(magnitude, 23);
    auto adjustment = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + lane));
    adjustment = _mm256_min_epi32(_mm256_max_epi32(adjustment, _mm256_set1_epi32(-4096)),
                                  _mm256_set1_epi32(4096));
    const auto scaled = _mm256_add_epi32(exponent, adjustment);
    const auto final_exponent = _mm256_add_epi32(scaled, delta);
    // Check pre-OMOD and final ranges independently: OMOD cannot rescue a tiny
    // or overflowing intermediate. All admitted normal operations are exact.
    auto normal = _mm256_and_si256(_mm256_cmpgt_epi32(exponent, zero),
                                   _mm256_cmpgt_epi32(_mm256_set1_epi32(255), exponent));
    normal = _mm256_and_si256(normal,
                              _mm256_and_si256(_mm256_cmpgt_epi32(scaled, zero),
                                               _mm256_cmpgt_epi32(_mm256_set1_epi32(255), scaled)));
    normal = _mm256_and_si256(
        normal, _mm256_and_si256(_mm256_cmpgt_epi32(final_exponent, zero),
                                 _mm256_cmpgt_epi32(_mm256_set1_epi32(255), final_exponent)));
    const auto zeros = _mm256_cmpeq_epi32(magnitude, zero);
    const auto special =
        _mm256_or_si256(zeros, _mm256_cmpgt_epi32(magnitude, _mm256_set1_epi32(0x7f7fffff)));
    const auto quiet = _mm256_and_si256(_mm256_cmpgt_epi32(magnitude, infinity),
                                        _mm256_set1_epi32(quiet_snan ? 0x00400000 : 0));
    auto special_value = _mm256_or_si256(raw, quiet);
    if (omod) {
      special_value = _mm256_andnot_si256(zeros, special_value);
    }
    auto value = _mm256_or_si256(_mm256_and_si256(raw, _mm256_set1_epi32(INT32_MIN | 0x7fffff)),
                                 _mm256_slli_epi32(final_exponent, 23));
    value = _mm256_blendv_epi8(value, special_value, special);
    if (instruction_flags & RJ_ARITH_ALU_CLAMP) {
      value = _mm256_castps_si256(
          clamp_f32(_mm256_castsi256_ps(value), flags & RJ_ARITH_FP_DX10_CLAMP_OFF));
    }
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(result + lane), value);
    repairs |=
        (~unsigned(_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_or_si256(normal, special)))) &
         0xffU)
        << lane;
  }
  while (repairs) {
    const int lane = int(bit_count_zero<true>(repairs));
    repairs &= repairs - 1;
    result[lane] = ldexp_f32_value_baseline(a[lane], bit_cast<int32_t>(b[lane]), flags,
                                            instruction_flags, quiet_snan);
  }
  for (int lane = 0; lane < 32; lane += 8) {
    const auto active = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                                          _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), active,
                           _mm256_loadu_si256(reinterpret_cast<const __m256i *>(result + lane)));
  }
}

} // namespace rocjitsu::arith
