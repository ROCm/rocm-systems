// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 per-lane conditional selection between two source registers.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "register/cndmask.h"
#include "rj_arith/rj_arith.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {

// Select between two sources by lane condition. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <bool Half>
void cndmask_x86_64_v4(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                       const uint32_t *b, uint32_t condition) {
  const uint32_t sign = Half ? 0x8000 : 0x80000000;
  unsigned sa = Half && (mode & RJ_ARITH_ALU_HIGH_A) ? 16 : 0;
  unsigned sb = Half && (mode & RJ_ARITH_ALU_HIGH_B) ? 16 : 0;
  unsigned sd = Half && (mode & RJ_ARITH_ALU_HIGH_D) ? 16 : 0;
  auto clear_a = _mm512_set1_epi32(int(mode & RJ_ARITH_ALU_ABS_A ? ~sign : UINT32_MAX));
  auto clear_b = _mm512_set1_epi32(int(mode & RJ_ARITH_ALU_ABS_B ? ~sign : UINT32_MAX));
  auto flip_a = _mm512_set1_epi32(int(mode & RJ_ARITH_ALU_NEG_A ? sign : 0));
  auto flip_b = _mm512_set1_epi32(int(mode & RJ_ARITH_ALU_NEG_B ? sign : 0));
  for (int lane = 0; lane < 32; lane += 16) {
    auto x = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(a + lane));
    auto y = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(b + lane));
    if constexpr (Half) {
      x = _mm512_srl_epi32(x, _mm_cvtsi32_si128(sa));
      y = _mm512_srl_epi32(y, _mm_cvtsi32_si128(sb));
    }
    x = _mm512_xor_si512(_mm512_and_si512(x, clear_a), flip_a);
    y = _mm512_xor_si512(_mm512_and_si512(y, clear_b), flip_b);
    auto value = _mm512_mask_blend_epi32(__mmask16(condition >> lane), x, y);
    if constexpr (Half) {
      auto old = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(d + lane));
      value = _mm512_sll_epi32(_mm512_and_si512(value, _mm512_set1_epi32(65535)),
                               _mm_cvtsi32_si128(sd));
      value =
          _mm512_or_si512(value, _mm512_andnot_si512(_mm512_set1_epi32(int(65535u << sd)), old));
    }
    _mm512_mask_storeu_epi32(d + lane, __mmask16(exec_mask >> lane), value);
  }
}

template void cndmask_x86_64_v4<true>(uint32_t, uint32_t, uint32_t *, const uint32_t *,
                                      const uint32_t *, uint32_t);
template void cndmask_x86_64_v4<false>(uint32_t, uint32_t, uint32_t *, const uint32_t *,
                                       const uint32_t *, uint32_t);

} // namespace rocjitsu::arith
