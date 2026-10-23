// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 exponent extraction from FP16, FP32, and FP64 encodings.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "fp/frexp_exp.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {
namespace {

// Extract floating-point exponents for the selected lanes. Requires validated controls;
// preserve inactive destination lanes.
template <bool Fp64>
void run(uint32_t exec_mask, uint32_t *d, const uint32_t *const *a, bool flush_input) {
  uint32_t result[32];
  for (int lane = 0; lane < 32; lane += Fp64 ? 4 : 8) {
    if constexpr (Fp64) {
      auto low =
          _mm256_cvtepu32_epi64(_mm_loadu_si128(reinterpret_cast<const __m128i *>(a[0] + lane)));
      auto high =
          _mm256_cvtepu32_epi64(_mm_loadu_si128(reinterpret_cast<const __m128i *>(a[1] + lane)));
      auto magnitude = _mm256_and_si256(_mm256_or_si256(low, _mm256_slli_epi64(high, 32)),
                                        _mm256_set1_epi64x(INT64_MAX));
      auto subnormal =
          _mm256_cmpgt_epi64(_mm256_set1_epi64x(INT64_C(0x0010000000000000)), magnitude);

      // Convert the integer significand exactly: encoding it as the fraction
      // of 2^52 forms 2^52 + significand. Subtraction recovers an ordinary
      // integer-valued double, even with host DAZ/FTZ enabled, in every rounding mode.
      auto small = _mm256_and_si256(magnitude, subnormal);
      auto biased = _mm256_or_si256(small, _mm256_set1_epi64x(0x4330000000000000LL));
      auto significand =
          _mm256_castpd_si256(_mm256_sub_pd(_mm256_castsi256_pd(biased), _mm256_set1_pd(0x1p52)));
      auto normalized = _mm256_blendv_epi8(magnitude, significand, subnormal);
      auto exponent = _mm256_sub_epi64(_mm256_srli_epi64(normalized, 52), _mm256_set1_epi64x(1022));
      exponent = _mm256_sub_epi64(exponent, _mm256_and_si256(subnormal, _mm256_set1_epi64x(1074)));
      auto special = _mm256_or_si256(
          _mm256_cmpeq_epi64(magnitude, _mm256_setzero_si256()),
          _mm256_cmpgt_epi64(magnitude, _mm256_set1_epi64x(INT64_C(0x7ff0000000000000) - 1)));
      if (flush_input) {
        special = _mm256_or_si256(special, subnormal);
      }
      exponent = _mm256_andnot_si256(special, exponent);
      auto packed =
          _mm256_permutevar8x32_epi32(exponent, _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7));
      _mm_storeu_si128(reinterpret_cast<__m128i *>(result + lane), _mm256_castsi256_si128(packed));
    } else {
      auto magnitude =
          _mm256_and_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(a[0] + lane)),
                           _mm256_set1_epi32(INT32_MAX));
      auto subnormal = _mm256_cmpgt_epi32(_mm256_set1_epi32(0x00800000), magnitude);

      // The 23-bit integer significand converts exactly in every rounding mode,
      // including when host DAZ is enabled; never consume an FP32 subnormal.
      auto fraction = _mm256_and_si256(magnitude, _mm256_set1_epi32(0x007fffff));
      auto leading = _mm256_srli_epi32(_mm256_castps_si256(_mm256_cvtepi32_ps(fraction)), 23);
      auto exponent = _mm256_blendv_epi8(
          _mm256_sub_epi32(_mm256_srli_epi32(magnitude, 23), _mm256_set1_epi32(126)),
          _mm256_sub_epi32(leading, _mm256_set1_epi32(275)), subnormal);

      // Keep only the finite input range [minimum, infinity). Subtracting
      // minimum makes smaller magnitudes wrap above the unsigned upper bound.
      // This single range check also discards guest-flushed subnormal inputs.
      uint32_t minimum = flush_input ? 0x00800000 : 1;
      auto offset = _mm256_sub_epi32(magnitude, _mm256_set1_epi32(minimum));
      auto upper = _mm256_set1_epi32(0x7f7fffff - minimum);
      auto valid = _mm256_cmpeq_epi32(_mm256_min_epu32(offset, upper), offset);
      exponent = _mm256_and_si256(valid, exponent);
      _mm256_storeu_si256(reinterpret_cast<__m256i *>(result + lane), exponent);
    }
  }
  for (int lane = 0; lane < 32; lane += 8) {
    auto lanes = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    auto lane_exec_mask = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                                            _mm256_sub_epi32(_mm256_set1_epi32(31), lanes));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), lane_exec_mask,
                           _mm256_loadu_si256(reinterpret_cast<const __m256i *>(result + lane)));
  }
}

} // namespace

// Extract floating-point exponents. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
void frexp_exp_x86_64_v3(bool fp64, uint32_t exec_mask, uint32_t *d, const uint32_t *const *a,
                         bool flush_input) {
  if (fp64) {
    run<true>(exec_mask, d, a, flush_input);
  } else {
    run<false>(exec_mask, d, a, flush_input);
  }
}

} // namespace rocjitsu::arith
