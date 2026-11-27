// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// SIMD conversion with guest rounding and denormal policies shared with scalar execution.
#include "conversion/float_conversion.h"
#include "conversion/x86_64/lanes.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {
// Convert a wave and write full 32-bit results, preserving inactive lanes.
template <Conversion32 Op>
void conversion32_x86_64_v3(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                            const uint32_t *a) {
  using V = conversion_simd::Lanes<32>;
  for (int lane = 0; lane < 32; lane += V::size()) {
    V raw(a + lane);
    V result;
    if constexpr (Op == Conversion32::FloatToSigned || Op == Conversion32::FloatToUnsigned) {
      // Truncation maps subnormals to zero regardless of the guest flushing policy.
      // Apply source modifiers before selecting saturation and NaN results.
      raw = (raw & V(mode & RJ_ARITH_ALU_ABS_A ? 0x7fffffffU : 0xffffffffU)) ^
            V(mode & RJ_ARITH_ALU_NEG_A ? 0x80000000U : 0U);
      auto value = _mm256_castsi256_ps(raw.bits);
      if constexpr (Op == Conversion32::FloatToSigned) {
        auto bits = _mm256_cvttps_epi32(value);
        bits = _mm256_blendv_epi8(
            bits, _mm256_set1_epi32(INT32_MAX),
            _mm256_castps_si256(_mm256_cmp_ps(value, _mm256_set1_ps(0x1p31f), _CMP_GE_OQ)));
        result =
            V(_mm256_and_si256(bits, _mm256_castps_si256(_mm256_cmp_ps(value, value, _CMP_ORD_Q))));
      } else {
        value = _mm256_max_ps(value, _mm256_setzero_ps()); // Negative values and NaNs become zero.
        // Values in the upper half of uint32_t become signed-convertible after an exact
        // subtraction.
        auto high = _mm256_cmp_ps(value, _mm256_set1_ps(0x1p31f), _CMP_GE_OQ);
        auto adjusted = _mm256_sub_ps(value, _mm256_and_ps(high, _mm256_set1_ps(0x1p31f)));
        auto bits = _mm256_xor_si256(
            _mm256_cvttps_epi32(adjusted),
            _mm256_and_si256(_mm256_castps_si256(high), _mm256_set1_epi32(INT32_MIN)));
        result = V(_mm256_or_si256(
            bits, _mm256_castps_si256(_mm256_cmp_ps(value, _mm256_set1_ps(0x1p32f), _CMP_GE_OQ))));
      }
    } else {
      result = conversion_detail::convert32<Op>(raw, flags, mode);
    }
    result.store(d + lane, exec_mask >> lane);
  }
}

template void conversion32_x86_64_v3<Conversion32::SignedToFloat>(uint64_t, uint32_t, uint32_t,
                                                                  uint32_t *, const uint32_t *);
template void conversion32_x86_64_v3<Conversion32::UnsignedToFloat>(uint64_t, uint32_t, uint32_t,
                                                                    uint32_t *, const uint32_t *);
template void conversion32_x86_64_v3<Conversion32::Nearest>(uint64_t, uint32_t, uint32_t,
                                                            uint32_t *, const uint32_t *);
template void conversion32_x86_64_v3<Conversion32::Floor>(uint64_t, uint32_t, uint32_t, uint32_t *,
                                                          const uint32_t *);

template void conversion32_x86_64_v3<Conversion32::FloatToSigned>(uint64_t, uint32_t, uint32_t,
                                                                  uint32_t *, const uint32_t *);
template void conversion32_x86_64_v3<Conversion32::FloatToUnsigned>(uint64_t, uint32_t, uint32_t,
                                                                    uint32_t *, const uint32_t *);

} // namespace rocjitsu::arith
