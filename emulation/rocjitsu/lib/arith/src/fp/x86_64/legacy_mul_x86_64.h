// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Shared eight-lane preparation and finalization for inherited CDNA multiply.
#pragma once

#include "common/x86_64/alu_x86_64_v3.h"
#include "fp/legacy_mul.h"
#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Apply OMOD using exponent arithmetic. Only finite normal lanes are scaled;
// prepared zeros and nonfinite inputs retain their bits. No host mode changes.
inline __m256 scale_omod(__m256 value, uint32_t instruction_flags, unsigned rounding) {
  const unsigned omod = (instruction_flags >> 6) & 3;
  value = prepare_omod_f32(value, instruction_flags);
  auto bits = _mm256_castps_si256(value);
  auto magnitude = _mm256_and_si256(bits, _mm256_set1_epi32(INT32_MAX));
  auto normal = _mm256_and_si256(_mm256_cmpgt_epi32(magnitude, _mm256_setzero_si256()),
                                 _mm256_cmpgt_epi32(_mm256_set1_epi32(0x7f800000), magnitude));
  auto scaled = _mm256_add_epi32(bits, _mm256_set1_epi32(omod == 3 ? -0x800000 : int(omod << 23)));
  if (omod != 3) {
    auto overflow =
        _mm256_cmpgt_epi32(magnitude, _mm256_set1_epi32(0x7f800000 - int(omod << 23) - 1));
    auto negative = _mm256_srai_epi32(bits, 31);
    auto infinity = rounding == 0   ? _mm256_set1_epi32(-1)
                    : rounding == 1 ? _mm256_xor_si256(negative, _mm256_set1_epi32(-1))
                    : rounding == 2 ? negative
                                    : _mm256_setzero_si256();
    auto limit = _mm256_or_si256(
        _mm256_and_si256(bits, _mm256_set1_epi32(INT32_MIN)),
        _mm256_blendv_epi8(_mm256_set1_epi32(0x7f7fffff), _mm256_set1_epi32(0x7f800000), infinity));
    scaled = _mm256_blendv_epi8(scaled, limit, overflow);
  }
  return _mm256_castsi256_ps(_mm256_blendv_epi8(bits, scaled, normal));
}

// Apply source sign controls and the guest input-denormal policy.
inline __m256 legacy_mul_input(const uint32_t *p, uint32_t mode, uint64_t flags) {
  auto bits = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(p));
  bits = _mm256_and_si256(bits, _mm256_set1_epi32(mode & RJ_ARITH_ALU_ABS_A ? INT32_MAX : -1));
  bits = _mm256_xor_si256(bits, _mm256_set1_epi32(mode & RJ_ARITH_ALU_NEG_A ? INT32_MIN : 0));
  auto value = _mm256_castsi256_ps(bits);
  return flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS ? flush_denorm_f32(value) : value;
}

// Finish a rounded product, preserving the legacy zero-product and OMOD rules.
// Repair only nonfinite-input lanes through the baseline NaN-selection policy.
inline __m256i legacy_mul_finish(__m256 value, __m256 x, __m256 y, uint64_t flags, uint32_t mode,
                                 const uint32_t *a, const uint32_t *b) {
  auto zero = _mm256_or_ps(_mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_EQ_OQ),
                           _mm256_cmp_ps(y, _mm256_setzero_ps(), _CMP_EQ_OQ));
  value = _mm256_andnot_ps(zero, value);
  if (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) {
    value = flush_denorm_f32(value);
  }
  if (mode & RJ_ARITH_ALU_OMOD_MASK) {
    value = scale_omod(value, mode, unsigned((flags & RJ_ARITH_FP_ROUND_MASK) >> 24));
  }
  if (mode & RJ_ARITH_ALU_CLAMP) {
    value = clamp_f32(value, flags & RJ_ARITH_FP_DX10_CLAMP_OFF);
  }
  auto abs_mask = _mm256_set1_epi32(INT32_MAX);
  auto ax = _mm256_and_si256(_mm256_castps_si256(x), abs_mask);
  auto ay = _mm256_and_si256(_mm256_castps_si256(y), abs_mask);
  auto exceptional = _mm256_or_si256(_mm256_cmpgt_epi32(ax, _mm256_set1_epi32(0x7f7fffff)),
                                     _mm256_cmpgt_epi32(ay, _mm256_set1_epi32(0x7f7fffff)));
  unsigned repair = unsigned(_mm256_movemask_ps(_mm256_castsi256_ps(exceptional)));
  if (repair) {
    uint32_t result[8];
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(result), _mm256_castps_si256(value));
    for (int lane = 0; lane < 8; ++lane) {
      if ((repair >> lane) & 1) {
        result[lane] = legacy_mul_value(flags, mode, a[lane], b[lane]);
      }
    }
    return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(result));
  }
  return _mm256_castps_si256(value);
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
