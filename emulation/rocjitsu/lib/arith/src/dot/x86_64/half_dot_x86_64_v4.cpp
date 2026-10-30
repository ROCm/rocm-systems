// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 loose FP16/BF16 dot products with packed 16-bit results.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "common/simd.h"
#include "dot/x86_64/half_conversion_x86_64_v4.h"
#include "rj_arith/rj_arith.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {
namespace {

// Accumulate loose DOT2 into 16-bit results for the selected lanes. Requires validated
// controls; preserve inactive destination lanes.
template <bool Bf16>
void dot(bool saturate, uint32_t exec_mask, uint32_t instruction_flags, uint32_t *d,
         const uint32_t *a, const uint32_t *b, const uint32_t *c) {
  int c_shift = instruction_flags & RJ_ARITH_ALU_HIGH_C ? 16 : 0;
  int d_shift = instruction_flags & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
  auto keep = _mm512_set1_epi32(d_shift ? 65535 : -65536);
  for (int lane = 0; lane < 32; lane += 16) {
    auto va = _mm512_loadu_si512(a + lane);
    auto vb = _mm512_loadu_si512(b + lane);
    auto vc = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(c + lane));
    auto p0 = _mm512_mul_ps(dot_half_input<Bf16>(va, 0, instruction_flags),
                            dot_half_input<Bf16>(vb, 0, instruction_flags >> 1));
    auto p1 = _mm512_mul_ps(dot_half_input<Bf16>(va, 16, instruction_flags),
                            dot_half_input<Bf16>(vb, 16, instruction_flags >> 1));
    auto value = _mm512_add_ps(_mm512_add_ps(p0, p1),
                               dot_half_input<Bf16>(vc, c_shift, instruction_flags >> 2));
    auto result =
        _mm512_sll_epi32(dot_half_narrow<Bf16>(value, saturate), _mm_cvtsi32_si128(d_shift));
    auto old = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(d + lane));
    result = _mm512_or_si512(result, _mm512_and_si512(old, keep));
    _mm512_mask_storeu_epi32(d + lane, __mmask16(exec_mask >> lane), result);
  }
}

} // namespace

// Accumulate loose DOT2 into 16-bit results. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
void half_dot_x86_64_v4(bool bf16, bool saturate, uint32_t exec_mask, uint32_t instruction_flags,
                        uint32_t *d, const uint32_t *a, const uint32_t *b, const uint32_t *c) {
  if (bf16) {
    dot<true>(saturate, exec_mask, instruction_flags, d, a, b, c);
  } else {
    dot<false>(saturate, exec_mask, instruction_flags, d, a, b, c);
  }
}

} // namespace rocjitsu::arith
