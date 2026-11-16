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
void conversion32_x86_64_v4(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
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
      auto value = _mm512_castsi512_ps(raw.bits);
      if constexpr (Op == Conversion32::FloatToSigned) {
        auto bits = _mm512_cvttps_epi32(value);
        bits = _mm512_mask_mov_epi32(bits,
                                     _mm512_cmp_ps_mask(value, _mm512_set1_ps(0x1p31f), _CMP_GE_OQ),
                                     _mm512_set1_epi32(INT32_MAX));
        result = V(_mm512_maskz_mov_epi32(_mm512_cmp_ps_mask(value, value, _CMP_ORD_Q), bits));
      } else {
        value = _mm512_max_ps(value, _mm512_setzero_ps()); // Negative values and NaNs become zero.
        auto bits = _mm512_cvttps_epu32(value);
        result = V(_mm512_mask_mov_epi32(
            bits, _mm512_cmp_ps_mask(value, _mm512_set1_ps(0x1p32f), _CMP_GE_OQ),
            _mm512_set1_epi32(-1)));
      }
    } else {
      result = conversion_detail::convert32<Op>(raw, flags, mode);
    }
    result.store(d + lane, exec_mask >> lane);
  }
}

template void conversion32_x86_64_v4<Conversion32::SignedToFloat>(uint64_t, uint32_t, uint32_t,
                                                                  uint32_t *, const uint32_t *);
template void conversion32_x86_64_v4<Conversion32::UnsignedToFloat>(uint64_t, uint32_t, uint32_t,
                                                                    uint32_t *, const uint32_t *);
template void conversion32_x86_64_v4<Conversion32::FloatToSigned>(uint64_t, uint32_t, uint32_t,
                                                                  uint32_t *, const uint32_t *);
template void conversion32_x86_64_v4<Conversion32::FloatToUnsigned>(uint64_t, uint32_t, uint32_t,
                                                                    uint32_t *, const uint32_t *);
template void conversion32_x86_64_v4<Conversion32::Nearest>(uint64_t, uint32_t, uint32_t,
                                                            uint32_t *, const uint32_t *);
template void conversion32_x86_64_v4<Conversion32::Floor>(uint64_t, uint32_t, uint32_t, uint32_t *,
                                                          const uint32_t *);

} // namespace rocjitsu::arith
