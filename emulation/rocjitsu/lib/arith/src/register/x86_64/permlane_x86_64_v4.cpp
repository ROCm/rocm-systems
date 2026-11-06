// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 lane permutations across rows, half-waves, and whole waves.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "register/permlane.h"
#include "rj_arith/rj_arith.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {
// Permute source lanes. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <bool Cross, bool Var>
void permlane_x86_64_v4(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                        const uint32_t *b, uint32_t lo, uint32_t hi) {
  __m512i results[2];
  auto ones = _mm512_set1_epi32(1);
  auto bits = _mm512_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
                                16384, 32768);
  auto shifts = _mm512_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28, 0, 4, 8, 12, 16, 20, 24, 28);
  for (int lane = 0; lane < 32; lane += 16) {
    __m512i index;
    if constexpr (Var) {
      index = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(b + lane));
    } else {
      auto packed =
          _mm512_mask_blend_epi32(0xff00, _mm512_set1_epi32(int(lo)), _mm512_set1_epi32(int(hi)));
      index = _mm512_srlv_epi32(packed, shifts);
    }
    index = _mm512_and_si512(index, _mm512_set1_epi32(15));
    int base = (lane & 16) ^ (Cross ? 16 : 0);
    auto value = _mm512_permutexvar_epi32(index, _mm512_loadu_si512(a + base));
    if (!(mode & RJ_ARITH_PERMLANE_FI)) {
      auto lane_exec_mask = _mm512_and_si512(
          _mm512_srlv_epi32(_mm512_set1_epi32(int(exec_mask >> base)), index), ones);
      auto fallback = mode & RJ_ARITH_PERMLANE_BOUND_CTRL
                          ? _mm512_setzero_si512()
                          : _mm512_loadu_si512(reinterpret_cast<const __m512i *>(d + lane));
      value = _mm512_mask_blend_epi32(
          _mm512_cmpneq_epi32_mask(lane_exec_mask, _mm512_setzero_si512()), fallback, value);
    }
    results[lane / 16] = value;
  }
  // Defer every store until both source rows and all indices have been read.
  for (int lane = 0; lane < 32; lane += 16) {
    (void)bits;
    _mm512_mask_storeu_epi32(d + lane, __mmask16(exec_mask >> lane), results[lane / 16]);
  }
}

template void permlane_x86_64_v4<false, false>(uint32_t, uint32_t, uint32_t *, const uint32_t *,
                                               const uint32_t *, uint32_t, uint32_t);
template void permlane_x86_64_v4<false, true>(uint32_t, uint32_t, uint32_t *, const uint32_t *,
                                              const uint32_t *, uint32_t, uint32_t);
template void permlane_x86_64_v4<true, false>(uint32_t, uint32_t, uint32_t *, const uint32_t *,
                                              const uint32_t *, uint32_t, uint32_t);
template void permlane_x86_64_v4<true, true>(uint32_t, uint32_t, uint32_t *, const uint32_t *,
                                             const uint32_t *, uint32_t, uint32_t);

} // namespace rocjitsu::arith
