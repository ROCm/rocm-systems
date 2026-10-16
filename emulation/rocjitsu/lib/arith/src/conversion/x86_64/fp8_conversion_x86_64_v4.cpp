// SPDX-License-Identifier: MIT

// AVX-512 FP8/BF8 widening to FP32, including packed pairs.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "common/x86_64/fp8_x86_64_v4.h"
#include "conversion/fp8_conversion.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {

// Widen FP8/BF8 factors to FP32. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <bool Bf8, bool Packed>
void fp8_conversion_x86_64_v4(uint32_t exec_mask, unsigned shift, uint32_t *const *d,
                              const uint32_t *a) {
  for (int lane = 0; lane < 32; lane += 16) {
    auto raw = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(a + lane));
    raw = _mm512_srl_epi32(raw, _mm_cvtsi32_si128(int(shift)));
    auto low = _mm512_castps_si512(widen_fp8<Bf8, true>(raw));
    auto lane_exec_mask = __mmask16(exec_mask >> lane);
    if constexpr (Packed) {
      auto high = _mm512_castps_si512(widen_fp8<Bf8, true>(_mm512_srli_epi32(raw, 8)));
      _mm512_mask_storeu_epi32(d[0] + lane, lane_exec_mask, low);
      _mm512_mask_storeu_epi32(d[1] + lane, lane_exec_mask, high);
    } else {
      _mm512_mask_storeu_epi32(d[0] + lane, lane_exec_mask, low);
    }
  }
}

template void fp8_conversion_x86_64_v4<false, false>(uint32_t, unsigned, uint32_t *const *,
                                                     const uint32_t *);
template void fp8_conversion_x86_64_v4<true, false>(uint32_t, unsigned, uint32_t *const *,
                                                    const uint32_t *);
template void fp8_conversion_x86_64_v4<false, true>(uint32_t, unsigned, uint32_t *const *,
                                                    const uint32_t *);
template void fp8_conversion_x86_64_v4<true, true>(uint32_t, unsigned, uint32_t *const *,
                                                   const uint32_t *);

} // namespace rocjitsu::arith
