// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 SIMD halfword moves and swaps.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "common/simd.h"
#include "rj_arith/detail/rj_arith_common.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {

// Evaluate move half for the caller-selected operation. Preserve destination lanes outside
// exec_mask.
// Requires the named CPU features and validated instruction controls.
void move_half_x86_64_v4(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a) {
  auto sa = _mm_cvtsi32_si128(mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0);
  auto sd = _mm_cvtsi32_si128(mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0);
  auto destination = _mm512_sll_epi32(_mm512_set1_epi32(65535), sd);
  auto keep = _mm512_set1_epi32(mode & RJ_ARITH_ALU_ABS_A ? 32767 : 65535);
  auto sign = _mm512_set1_epi32(mode & RJ_ARITH_ALU_NEG_A ? 32768 : 0);
  for (int lane = 0; lane < 32; lane += 16) {
    auto value = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(a + lane));
    value = _mm512_xor_si512(_mm512_and_si512(_mm512_srl_epi32(value, sa), keep), sign);
    auto original = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(d + lane));
    value =
        _mm512_or_si512(_mm512_andnot_si512(destination, original), _mm512_sll_epi32(value, sd));
    const __mmask16 active = __mmask16(exec_mask >> lane);
    _mm512_mask_storeu_epi32(d + lane, active, value);
  }
}

// Evaluate swap half for the caller-selected operation. Preserve destination lanes outside
// exec_mask.
// Requires the named CPU features and validated instruction controls.
void swap_half_x86_64_v4(int lanes, uint64_t exec_mask, uint32_t mode, uint32_t *d, uint32_t *a) {
  auto sa = _mm_cvtsi32_si128(mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0);
  auto sd = _mm_cvtsi32_si128(mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0);
  auto destination = _mm512_sll_epi32(_mm512_set1_epi32(65535), sd);
  auto source = _mm512_sll_epi32(_mm512_set1_epi32(65535), sa);
  for (int lane = 0; lane < lanes; lane += 16) {
    auto old_a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(a + lane));
    auto old_d = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(d + lane));
    auto new_d = _mm512_or_si512(
        _mm512_andnot_si512(destination, old_d),
        _mm512_and_si512(destination, _mm512_sll_epi32(_mm512_srl_epi32(old_a, sa), sd)));
    const __mmask16 active = __mmask16(exec_mask >> lane);
    _mm512_mask_storeu_epi32(d + lane, active, new_d);

    // Preserve the first write within one VGPR without reloading a masked store.
    // Inactive results are immaterial: both destination stores apply EXEC.
    auto preserved = (d == a ? new_d : old_a);
    auto new_a = _mm512_or_si512(
        _mm512_andnot_si512(source, preserved),
        _mm512_and_si512(source, _mm512_sll_epi32(_mm512_srl_epi32(old_d, sd), sa)));
    _mm512_mask_storeu_epi32(a + lane, active, new_a);
  }
}

} // namespace rocjitsu::arith
