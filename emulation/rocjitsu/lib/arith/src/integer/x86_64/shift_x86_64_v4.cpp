// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 32/64-bit logical and arithmetic shifts, including reversed operand forms.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "integer/shift.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {

// Shift integer words. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <int Bits, Shift Op>
void shift_x86_64_v4(uint32_t exec_mask, uint32_t *const *d, const uint32_t *a,
                     const uint32_t *const *b) {
  for (int lane = 0; lane < 32; lane += 16) {
    const __m512i count =
        _mm512_and_si512(_mm512_loadu_si512(reinterpret_cast<const __m512i *>(a + lane)),
                         _mm512_set1_epi32(Bits - 1));
    const __m512i lo = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(b[0] + lane));
    __m512i low, high;
    if constexpr (Bits == 32) {
      if constexpr (Op == Shift::Left) {
        low = _mm512_sllv_epi32(lo, count);
      }
      if constexpr (Op == Shift::LogicalRight) {
        low = _mm512_srlv_epi32(lo, count);
      }
      if constexpr (Op == Shift::ArithmeticRight) {
        low = _mm512_srav_epi32(lo, count);
      }
    } else {
      const __m512i hi = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(b[1] + lane));
      const __m512i complement = _mm512_sub_epi32(_mm512_set1_epi32(32), count);
      const __m512i excess = _mm512_sub_epi32(count, _mm512_set1_epi32(32));

      // AVX-512 logical shifts zero lanes whose unsigned count is >=32. That
      // also covers negative complement/excess counts and the count=0 seam.
      if constexpr (Op == Shift::Left) {
        low = _mm512_sllv_epi32(lo, count);
        high = _mm512_or_si512(
            _mm512_sllv_epi32(hi, count),
            _mm512_or_si512(_mm512_srlv_epi32(lo, complement), _mm512_sllv_epi32(lo, excess)));
      } else {
        low = _mm512_or_si512(_mm512_srlv_epi32(lo, count), _mm512_sllv_epi32(hi, complement));
        if constexpr (Op == Shift::LogicalRight) {
          low = _mm512_or_si512(low, _mm512_srlv_epi32(hi, excess));
          high = _mm512_srlv_epi32(hi, count);
        } else {
          low = _mm512_mask_mov_epi32(low, _mm512_cmpgt_epi32_mask(count, _mm512_set1_epi32(31)),
                                      _mm512_srav_epi32(hi, excess));
          high = _mm512_srav_epi32(hi, count);
        }
      }
    }
    // Both source halves and the count have been read before either store,
    // including when D aliases B in reverse order or both D halves coincide.
    const __mmask16 lane_exec_mask = __mmask16(exec_mask >> lane);
    _mm512_mask_storeu_epi32(d[0] + lane, lane_exec_mask, low);
    if constexpr (Bits == 64) {
      _mm512_mask_storeu_epi32(d[1] + lane, lane_exec_mask, high);
    }
  }
}

template void shift_x86_64_v4<32, Shift::Left>(uint32_t, uint32_t *const *, const uint32_t *,
                                               const uint32_t *const *);
template void shift_x86_64_v4<32, Shift::LogicalRight>(uint32_t, uint32_t *const *,
                                                       const uint32_t *, const uint32_t *const *);
template void shift_x86_64_v4<32, Shift::ArithmeticRight>(uint32_t, uint32_t *const *,
                                                          const uint32_t *,
                                                          const uint32_t *const *);
template void shift_x86_64_v4<64, Shift::Left>(uint32_t, uint32_t *const *, const uint32_t *,
                                               const uint32_t *const *);
template void shift_x86_64_v4<64, Shift::LogicalRight>(uint32_t, uint32_t *const *,
                                                       const uint32_t *, const uint32_t *const *);
template void shift_x86_64_v4<64, Shift::ArithmeticRight>(uint32_t, uint32_t *const *,
                                                          const uint32_t *,
                                                          const uint32_t *const *);

} // namespace rocjitsu::arith
