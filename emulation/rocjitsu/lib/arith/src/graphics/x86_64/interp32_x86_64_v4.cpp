// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 FP32 interpolation stages with source-lane selection.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "graphics/interp.h"
#include "rj_arith/rj_arith.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {

// Evaluate an FP32 interpolation stage. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <bool P2>
void interp32_x86_64_v4(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                        const uint32_t *b, const uint32_t *c) {
  auto flip_a = _mm512_set1_epi32(mode & RJ_ARITH_ALU_NEG_A ? INT32_MIN : 0);
  auto flip_b = _mm512_set1_epi32(mode & RJ_ARITH_ALU_NEG_B ? INT32_MIN : 0);
  auto flip_c = _mm512_set1_epi32(mode & RJ_ARITH_ALU_NEG_C ? INT32_MIN : 0);
  for (int lane = 0; lane < 32; lane += 16) {
    auto x = _mm512_castsi512_ps(
        _mm512_xor_si512(_mm512_loadu_si512(reinterpret_cast<const __m512i *>(a + lane)), flip_a));
    auto y = _mm512_castsi512_ps(
        _mm512_xor_si512(_mm512_loadu_si512(reinterpret_cast<const __m512i *>(b + lane)), flip_b));
    auto z = _mm512_castsi512_ps(
        _mm512_xor_si512(_mm512_loadu_si512(reinterpret_cast<const __m512i *>(c + lane)), flip_c));

    // Each shuffle reads all four source lanes, regardless of EXEC.
    x = _mm512_permute_ps(x, P2 ? _MM_SHUFFLE(2, 2, 2, 2) : _MM_SHUFFLE(1, 1, 1, 1));
    if constexpr (!P2) {
      z = _mm512_permute_ps(z, _MM_SHUFFLE(0, 0, 0, 0));
    }
    auto value = _mm512_fmadd_ps(x, y, z);
    if (mode & RJ_ARITH_ALU_CLAMP) {
      value = _mm512_min_ps(_mm512_max_ps(value, _mm512_setzero_ps()), _mm512_set1_ps(1));
    }
    _mm512_mask_storeu_epi32(d + lane, __mmask16(exec_mask >> lane), _mm512_castps_si512(value));
  }
}

template void interp32_x86_64_v4<true>(uint32_t, uint32_t, uint32_t *, const uint32_t *,
                                       const uint32_t *, const uint32_t *);
template void interp32_x86_64_v4<false>(uint32_t, uint32_t, uint32_t *, const uint32_t *,
                                        const uint32_t *, const uint32_t *);

} // namespace rocjitsu::arith
