// SPDX-License-Identifier: MIT

// AVX-512 conversions between FP32 and signed/unsigned 32-bit integers.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "conversion/conversion32.h"
#include "rj_arith/rj_arith.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {

// Convert FP32 and 32-bit integers. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <Conversion32 Op>
void conversion32_x86_64_v4(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a) {
  constexpr bool to_float =
      Op == Conversion32::SignedToFloat || Op == Conversion32::UnsignedToFloat;
  const auto zero = _mm512_setzero_ps();
  const float scales[] = {1, 2, 4, 0.5f};
  const auto scale = _mm512_set1_ps(scales[(mode >> 6) & 3]);
  const auto abs_mask = _mm512_set1_epi32(mode & RJ_ARITH_ALU_ABS_A ? 0x7fffffff : -1);
  const auto neg_mask = _mm512_set1_epi32(mode & RJ_ARITH_ALU_NEG_A ? int32_t(INT32_MIN) : 0);
  for (int lane = 0; lane < 32; lane += 16) {
    auto raw = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(a + lane));
    __m512i result;
    if constexpr (to_float) {
      __m512 x;
      if constexpr (Op == Conversion32::SignedToFloat) {
        x = _mm512_cvtepi32_ps(raw);
      } else {
        x = _mm512_cvtepu32_ps(raw);
      }
      x = _mm512_mul_ps(x, scale);
      if (mode & RJ_ARITH_ALU_CLAMP) {
        x = _mm512_min_ps(_mm512_max_ps(x, zero), _mm512_set1_ps(1));
      }
      result = _mm512_castps_si512(x);
    } else {
      raw = _mm512_xor_si512(_mm512_and_si512(raw, abs_mask), neg_mask);
      auto x = _mm512_castsi512_ps(raw);
      if constexpr (Op == Conversion32::FloatToUnsigned) {
        x = _mm512_max_ps(x, zero);
        auto overflow = _mm512_cmp_ps_mask(x, _mm512_set1_ps(4294967296.0f), _CMP_GE_OQ);
        x = _mm512_min_ps(x, _mm512_set1_ps(4294967040.0f));
        result = _mm512_cvttps_epu32(x);
        result = _mm512_mask_mov_epi32(result, overflow, _mm512_set1_epi32(-1));
      } else {
        auto nan = _mm512_cmp_ps_mask(x, x, _CMP_UNORD_Q);
        auto overflow = _mm512_cmp_ps_mask(x, _mm512_set1_ps(2147483648.0f), _CMP_GE_OQ);
        x = _mm512_min_ps(_mm512_max_ps(x, _mm512_set1_ps(-2147483648.0f)),
                          _mm512_set1_ps(2147483520.0f));
        if constexpr (Op == Conversion32::Nearest) {
          auto lower = _mm512_floor_ps(x);
          auto up = _mm512_cmp_ps_mask(_mm512_sub_ps(x, lower), _mm512_set1_ps(0.5f), _CMP_GE_OQ);
          x = _mm512_mask_add_ps(lower, up, lower, _mm512_set1_ps(1));
        }
        if constexpr (Op == Conversion32::Floor) {
          x = _mm512_floor_ps(x);
        }
        result = _mm512_cvttps_epi32(x);
        result = _mm512_mask_mov_epi32(result, overflow, _mm512_set1_epi32(INT32_MAX));
        if constexpr (Op == Conversion32::Nearest || Op == Conversion32::Floor) {
          auto nan_result =
              _mm512_xor_si512(_mm512_srai_epi32(raw, 31), _mm512_set1_epi32(INT32_MAX));
          result = _mm512_mask_mov_epi32(result, nan, nan_result);
        } else {
          result = _mm512_mask_mov_epi32(result, nan, _mm512_setzero_si512());
        }
      }
    }
    _mm512_mask_storeu_epi32(d + lane, __mmask16(exec_mask >> lane), result);
  }
}

template void conversion32_x86_64_v4<Conversion32::SignedToFloat>(uint32_t, uint32_t, uint32_t *,
                                                                  const uint32_t *);
template void conversion32_x86_64_v4<Conversion32::UnsignedToFloat>(uint32_t, uint32_t, uint32_t *,
                                                                    const uint32_t *);
template void conversion32_x86_64_v4<Conversion32::FloatToSigned>(uint32_t, uint32_t, uint32_t *,
                                                                  const uint32_t *);
template void conversion32_x86_64_v4<Conversion32::FloatToUnsigned>(uint32_t, uint32_t, uint32_t *,
                                                                    const uint32_t *);
template void conversion32_x86_64_v4<Conversion32::Nearest>(uint32_t, uint32_t, uint32_t *,
                                                            const uint32_t *);
template void conversion32_x86_64_v4<Conversion32::Floor>(uint32_t, uint32_t, uint32_t *,
                                                          const uint32_t *);

} // namespace rocjitsu::arith
