// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 loose FP8/BF8 vector dot products with FP32 accumulation.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "common/simd.h"
#include "common/x86_64/fp8.h"
#include "rj_arith/rj_arith.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {
namespace {

// Accumulate loose FP8/BF8 dot products for the selected lanes. Requires validated controls;
// preserve inactive destination lanes.
template <bool Bf8A, bool Bf8B>
void dot(uint32_t exec_mask, uint32_t modifiers, uint32_t *d, const uint32_t *a, const uint32_t *b,
         const uint32_t *c) {
  const auto keep = _mm512_set1_epi32(modifiers & RJ_ARITH_DOT_ABS_C ? 0x7fffffff : -1);
  const auto flip = _mm512_set1_epi32(modifiers & RJ_ARITH_DOT_NEG_C ? INT32_MIN : 0);
  for (int lane = 0; lane < 32; lane += 16) {
    auto va = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(a + lane));
    auto vb = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(b + lane));
    auto acc = _mm512_castsi512_ps(_mm512_xor_si512(
        _mm512_and_si512(_mm512_loadu_si512(reinterpret_cast<const __m512i *>(c + lane)), keep),
        flip));
    for (int shift = 0; shift < 32; shift += 8) {
      acc = _mm512_fmadd_ps(widen_fp8<Bf8A>(_mm512_srl_epi32(va, _mm_cvtsi32_si128(shift))),
                            widen_fp8<Bf8B>(_mm512_srl_epi32(vb, _mm_cvtsi32_si128(shift))), acc);
    }
    _mm512_mask_storeu_epi32(d + lane, __mmask16(exec_mask >> lane), _mm512_castps_si512(acc));
  }
}

} // namespace

// Accumulate loose FP8/BF8 dot products. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
void fp8_dot_x86_64_v4(bool bf8_a, bool bf8_b, uint32_t exec_mask, uint32_t modifiers, uint32_t *d,
                       const uint32_t *a, const uint32_t *b, const uint32_t *c) {
  if (bf8_a) {
    if (bf8_b) {
      dot<true, true>(exec_mask, modifiers, d, a, b, c);
    } else {
      dot<true, false>(exec_mask, modifiers, d, a, b, c);
    }
  } else {
    if (bf8_b) {
      dot<false, true>(exec_mask, modifiers, d, a, b, c);
    } else {
      dot<false, false>(exec_mask, modifiers, d, a, b, c);
    }
  }
}

} // namespace rocjitsu::arith
