// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 32/64-bit logical and arithmetic shifts, including reversed operand forms.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "integer/shift.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {

// Shift integer words. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <int Bits, Shift Op>
void shift_x86_64_v3(uint32_t exec_mask, uint32_t *const *d, const uint32_t *a,
                     const uint32_t *const *b) {
  for (int lane = 0; lane < 32; lane += 8) {
    const __m256i count =
        _mm256_and_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane)),
                         _mm256_set1_epi32(Bits - 1));
    const __m256i lo = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b[0] + lane));
    __m256i low, high;
    if constexpr (Bits == 32) {
      if constexpr (Op == Shift::Left) {
        low = _mm256_sllv_epi32(lo, count);
      }
      if constexpr (Op == Shift::LogicalRight) {
        low = _mm256_srlv_epi32(lo, count);
      }
      if constexpr (Op == Shift::ArithmeticRight) {
        low = _mm256_srav_epi32(lo, count);
      }
    } else {
      const __m256i hi = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b[1] + lane));
      const __m256i complement = _mm256_sub_epi32(_mm256_set1_epi32(32), count);
      const __m256i excess = _mm256_sub_epi32(count, _mm256_set1_epi32(32));

      // AVX2 logical shifts zero lanes whose unsigned count is >=32. That
      // also covers negative complement/excess counts and the count=0 seam.
      if constexpr (Op == Shift::Left) {
        low = _mm256_sllv_epi32(lo, count);
        high = _mm256_or_si256(
            _mm256_sllv_epi32(hi, count),
            _mm256_or_si256(_mm256_srlv_epi32(lo, complement), _mm256_sllv_epi32(lo, excess)));
      } else {
        low = _mm256_or_si256(_mm256_srlv_epi32(lo, count), _mm256_sllv_epi32(hi, complement));
        if constexpr (Op == Shift::LogicalRight) {
          low = _mm256_or_si256(low, _mm256_srlv_epi32(hi, excess));
          high = _mm256_srlv_epi32(hi, count);
        } else {
          low = _mm256_blendv_epi8(low, _mm256_srav_epi32(hi, excess),
                                   _mm256_cmpgt_epi32(count, _mm256_set1_epi32(31)));
          high = _mm256_srav_epi32(hi, count);
        }
      }
    }
    // Both source halves and the count have been read before either store,
    // including when D aliases B in reverse order or both D halves coincide.
    const __m256i lane_exec_mask =
        _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                          _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d[0] + lane), lane_exec_mask, low);
    if constexpr (Bits == 64) {
      _mm256_maskstore_epi32(reinterpret_cast<int *>(d[1] + lane), lane_exec_mask, high);
    }
  }
}

template void shift_x86_64_v3<32, Shift::Left>(uint32_t, uint32_t *const *, const uint32_t *,
                                               const uint32_t *const *);
template void shift_x86_64_v3<32, Shift::LogicalRight>(uint32_t, uint32_t *const *,
                                                       const uint32_t *, const uint32_t *const *);
template void shift_x86_64_v3<32, Shift::ArithmeticRight>(uint32_t, uint32_t *const *,
                                                          const uint32_t *,
                                                          const uint32_t *const *);
template void shift_x86_64_v3<64, Shift::Left>(uint32_t, uint32_t *const *, const uint32_t *,
                                               const uint32_t *const *);
template void shift_x86_64_v3<64, Shift::LogicalRight>(uint32_t, uint32_t *const *,
                                                       const uint32_t *, const uint32_t *const *);
template void shift_x86_64_v3<64, Shift::ArithmeticRight>(uint32_t, uint32_t *const *,
                                                          const uint32_t *,
                                                          const uint32_t *const *);

} // namespace rocjitsu::arith
