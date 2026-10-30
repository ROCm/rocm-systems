// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 loose FP32 FMA with source modifiers, OMOD, CLAMP, and masked stores.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "common/simd.h"
#include "common/x86_64/alu_x86_64_v4.h"
#include "rj_arith/rj_arith.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {

// Evaluate FP32 fused multiply-add for the selected lanes. Requires validated controls;
// preserve inactive destination lanes.
void fma_x86_64_v4(uint64_t flags, uint32_t exec_mask, uint32_t instruction_flags, uint32_t *d,
                   const uint32_t *a, const uint32_t *b, const uint32_t *c) {
  const __m512i keep_a =
      _mm512_set1_epi32((instruction_flags & RJ_ARITH_ALU_ABS_A) ? 0x7fffffff : -1);
  const __m512i flip_a =
      _mm512_set1_epi32((instruction_flags & RJ_ARITH_ALU_NEG_A) ? INT32_MIN : 0);
  const __m512i keep_b =
      _mm512_set1_epi32((instruction_flags & RJ_ARITH_ALU_ABS_B) ? 0x7fffffff : -1);
  const __m512i flip_b =
      _mm512_set1_epi32((instruction_flags & RJ_ARITH_ALU_NEG_B) ? INT32_MIN : 0);
  const __m512i keep_c =
      _mm512_set1_epi32((instruction_flags & RJ_ARITH_ALU_ABS_C) ? 0x7fffffff : -1);
  const __m512i flip_c =
      _mm512_set1_epi32((instruction_flags & RJ_ARITH_ALU_NEG_C) ? INT32_MIN : 0);
  const float scales[] = {1, 2, 4, 0.5f};
  const uint32_t omod = (instruction_flags >> 6) & 3;
  const auto scale = _mm512_set1_ps(scales[omod]);
  for (int i = 0; i < 32; i += 16) {
    auto va = _mm512_castsi512_ps(
        _mm512_xor_si512(_mm512_and_si512(_mm512_loadu_si512(a + i), keep_a), flip_a));
    auto vb = _mm512_castsi512_ps(
        _mm512_xor_si512(_mm512_and_si512(_mm512_loadu_si512(b + i), keep_b), flip_b));
    __m512 vc;

    vc = _mm512_castsi512_ps(
        _mm512_xor_si512(_mm512_and_si512(_mm512_loadu_si512(c + i), keep_c), flip_c));

    if ((flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS)) {
      va = flush_denorm_f32(va);
      vb = flush_denorm_f32(vb);
      vc = flush_denorm_f32(vc);
    }

    auto result = _mm512_fmadd_ps(va, vb, vc);
    if ((flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
      result = flush_denorm_f32(result);
    }
    if (omod) {
      result = prepare_omod_f32(result, instruction_flags);
      result = _mm512_mul_ps(result, scale);
    }
    if (instruction_flags & RJ_ARITH_ALU_CLAMP) {
      result = clamp_f32(result, flags & RJ_ARITH_FP_DX10_CLAMP_OFF);
    }
    _mm512_mask_storeu_epi32(d + i, static_cast<__mmask16>(exec_mask >> i),
                             _mm512_castps_si512(result));
  }
}

} // namespace rocjitsu::arith
