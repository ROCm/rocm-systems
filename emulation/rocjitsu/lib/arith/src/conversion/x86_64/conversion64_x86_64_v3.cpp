// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 conversions between FP64 and FP32 or 32-bit integers in split VGPR layouts.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "common/x86_64/alu_x86_64_v3.h"
#include "conversion/conversion64.h"
#include "conversion/float_conversion.h"
#include "conversion/x86_64/lanes.h"
#include "rj_arith/rj_arith.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {
namespace {

// Extract each 64-bit lane's low 32-bit word into consecutive lanes.
__m128i low_words(__m256i words) {
  auto packed = _mm256_permutevar8x32_epi32(words, _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7));
  return _mm256_castsi256_si128(packed);
}

} // namespace

// Convert FP64, FP32, or 32-bit integers. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <Conversion64 Op>
void conversion64_x86_64_v3(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
                            const uint32_t *const *a, bool quiet_nan) {
  constexpr bool from_integer =
      Op == Conversion64::SignedToDouble || Op == Conversion64::UnsignedToDouble;
  constexpr bool to_double = from_integer || Op == Conversion64::FloatToDouble;
  if constexpr (to_double || Op == Conversion64::DoubleToFloat) {
    using V = conversion_simd::Lanes<64>;
    uint32_t result[to_double ? 2 : 1][32];
    for (int lane = 0; lane < 32; lane += V::size()) {
      V raw = V::load_words(a[0] + lane);
      if constexpr (!to_double) {
        raw |= V::load_words(a[1] + lane) << 32;
      }
      V bits = conversion_detail::convert64<Op>(raw, flags, mode, quiet_nan);
      bits.store_words(result[0] + lane);
      if constexpr (to_double) {
        (bits >> 32).store_words(result[1] + lane);
      }
    }
    // Snapshot the wave before stores: either destination may alias either source VGPR.
    using W = conversion_simd::Lanes<32>;
    for (int reg = 0; reg < (to_double ? 2 : 1); ++reg) {
      for (int lane = 0; lane < 32; lane += W::size()) {
        W(result[reg] + lane).store(d[reg] + lane, exec_mask >> lane);
      }
    }
    return;
  }
  const auto zero = _mm256_setzero_pd();
  const double scales[] = {1, 2, 4, 0.5};
  uint32_t result[to_double ? 2 : 1][32];
  for (int lane = 0; lane < 32; lane += 4) {
    auto low = _mm_loadu_si128(reinterpret_cast<const __m128i *>(a[0] + lane));
    __m256d value;
    if constexpr (from_integer) {
      if constexpr (Op == Conversion64::SignedToDouble) {
        value = _mm256_cvtepi32_pd(low);
      } else {
        // FP64 represents every uint32_t exactly, so adding 2^32 to
        // the signed conversion for high-bit inputs cannot double-round.
        value = _mm256_cvtepi32_pd(low);
        auto negative = _mm256_cmp_pd(value, zero, _CMP_LT_OQ);
        value = _mm256_add_pd(value, _mm256_and_pd(negative, _mm256_set1_pd(4294967296.0)));
      }
    } else if constexpr (Op == Conversion64::FloatToDouble) {
      low = _mm_and_si128(low, _mm_set1_epi32(mode & RJ_ARITH_ALU_ABS_A ? INT32_MAX : -1));
      low = _mm_xor_si128(low, _mm_set1_epi32(mode & RJ_ARITH_ALU_NEG_A ? INT32_MIN : 0));
      value = _mm256_cvtps_pd(_mm_castsi128_ps(low));
    } else {
      auto high = _mm_loadu_si128(reinterpret_cast<const __m128i *>(a[1] + lane));
      auto raw = _mm256_or_si256(_mm256_cvtepu32_epi64(low),
                                 _mm256_slli_epi64(_mm256_cvtepu32_epi64(high), 32));
      raw = _mm256_and_si256(raw, _mm256_set1_epi64x(mode & RJ_ARITH_ALU_ABS_A ? INT64_MAX : -1));
      raw = _mm256_xor_si256(raw, _mm256_set1_epi64x(mode & RJ_ARITH_ALU_NEG_A ? INT64_MIN : 0));
      value = _mm256_castsi256_pd(raw);
    }
    if constexpr (to_double) {
      if (mode & RJ_ARITH_ALU_OMOD_MASK) {
        value = prepare_omod_f64(value, mode);
        value = _mm256_mul_pd(value, _mm256_set1_pd(scales[(mode >> 6) & 3]));
      }
      if (mode & RJ_ARITH_ALU_CLAMP) {
        value = _mm256_min_pd(_mm256_max_pd(value, zero), _mm256_set1_pd(1));
      }
      auto raw = _mm256_castpd_si256(value);
      _mm_storeu_si128(reinterpret_cast<__m128i *>(result[0] + lane), low_words(raw));
      _mm_storeu_si128(reinterpret_cast<__m128i *>(result[1] + lane),
                       low_words(_mm256_srli_epi64(raw, 32)));
    } else {
      __m128i output;
      if constexpr (Op == Conversion64::DoubleToFloat) {
        auto x = _mm256_cvtpd_ps(value);
        if (mode & RJ_ARITH_ALU_OMOD_MASK) {
          auto magnitude = _mm256_and_pd(value, _mm256_castsi256_pd(_mm256_set1_epi64x(INT64_MAX)));
          auto tiny = _mm256_cmp_pd(magnitude, _mm256_set1_pd(0x1p-126), _CMP_LT_OQ);
          x = _mm_andnot_ps(_mm_castsi128_ps(low_words(_mm256_castpd_si256(tiny))), x);
          x = prepare_omod_f32(x, mode);
          x = _mm_mul_ps(x, _mm_set1_ps(float(scales[(mode >> 6) & 3])));
        }
        if (mode & RJ_ARITH_ALU_CLAMP) {
          x = _mm_min_ps(_mm_max_ps(x, _mm_setzero_ps()), _mm_set1_ps(1));
        }
        output = _mm_castps_si128(x);
      } else if constexpr (Op == Conversion64::DoubleToSigned) {
        auto nan = _mm256_cmp_pd(value, value, _CMP_UNORD_Q);
        value = _mm256_min_pd(_mm256_max_pd(value, _mm256_set1_pd(-2147483648.0)),
                              _mm256_set1_pd(2147483647.0));
        output = _mm_andnot_si128(low_words(_mm256_castpd_si256(nan)), _mm256_cvttpd_epi32(value));
      } else if constexpr (Op == Conversion64::DoubleToUnsigned) {
        value = _mm256_min_pd(_mm256_max_pd(value, zero), _mm256_set1_pd(4294967295.0));
        auto high = _mm256_cmp_pd(value, _mm256_set1_pd(2147483648.0), _CMP_GE_OQ);
        auto adjusted = _mm256_sub_pd(value, _mm256_and_pd(high, _mm256_set1_pd(2147483648.0)));
        output = _mm256_cvttpd_epi32(adjusted);
        output = _mm_or_si128(
            output, _mm_and_si128(low_words(_mm256_castpd_si256(high)), _mm_set1_epi32(INT32_MIN)));
      }
      _mm_storeu_si128(reinterpret_cast<__m128i *>(result[0] + lane), output);
    }
  }
  // Delay stores to allow cross-half source/destination aliases. Store D0
  // before D1, so D1 wins when both destination halves share storage.
  for (int reg = 0; reg < (to_double ? 2 : 1); ++reg) {
    for (int lane = 0; lane < 32; lane += 8) {
      auto lane_exec_mask = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                                              _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
      _mm256_maskstore_epi32(
          reinterpret_cast<int *>(d[reg] + lane), lane_exec_mask,
          _mm256_loadu_si256(reinterpret_cast<const __m256i *>(result[reg] + lane)));
    }
  }
}

template void conversion64_x86_64_v3<Conversion64::SignedToDouble>(uint64_t, uint32_t, uint32_t,
                                                                   uint32_t *const *,
                                                                   const uint32_t *const *, bool);
template void conversion64_x86_64_v3<Conversion64::UnsignedToDouble>(uint64_t, uint32_t, uint32_t,
                                                                     uint32_t *const *,
                                                                     const uint32_t *const *, bool);
template void conversion64_x86_64_v3<Conversion64::DoubleToUnsigned>(uint64_t, uint32_t, uint32_t,
                                                                     uint32_t *const *,
                                                                     const uint32_t *const *, bool);
template void conversion64_x86_64_v3<Conversion64::FloatToDouble>(uint64_t, uint32_t, uint32_t,
                                                                  uint32_t *const *,
                                                                  const uint32_t *const *, bool);
template void conversion64_x86_64_v3<Conversion64::DoubleToFloat>(uint64_t, uint32_t, uint32_t,
                                                                  uint32_t *const *,
                                                                  const uint32_t *const *, bool);

template void conversion64_x86_64_v3<Conversion64::DoubleToSigned>(uint64_t, uint32_t, uint32_t,
                                                                   uint32_t *const *,
                                                                   const uint32_t *const *, bool);

} // namespace rocjitsu::arith
