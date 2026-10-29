// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Loose AVX2 FP32 fused multiply-add with modifiers and guest flushing.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "common/simd.h"
#include "common/x86_64/alu_x86_64_v3.h"
#include "rj_arith/rj_arith.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {

// Evaluate FP32 fused multiply-add for the selected lanes. Requires validated controls;
// preserve inactive destination lanes.
void fma_x86_64_v3(uint64_t flags, uint32_t exec_mask, uint32_t instruction_flags, uint32_t *d,
                   const uint32_t *a, const uint32_t *b, const uint32_t *c) {
  const __m256i keep_a =
      _mm256_set1_epi32((instruction_flags & RJ_ARITH_ALU_ABS_A) ? 0x7fffffff : -1);
  const __m256i flip_a =
      _mm256_set1_epi32((instruction_flags & RJ_ARITH_ALU_NEG_A) ? INT32_MIN : 0);
  const __m256i keep_b =
      _mm256_set1_epi32((instruction_flags & RJ_ARITH_ALU_ABS_B) ? 0x7fffffff : -1);
  const __m256i flip_b =
      _mm256_set1_epi32((instruction_flags & RJ_ARITH_ALU_NEG_B) ? INT32_MIN : 0);
  const __m256i keep_c =
      _mm256_set1_epi32((instruction_flags & RJ_ARITH_ALU_ABS_C) ? 0x7fffffff : -1);
  const __m256i flip_c =
      _mm256_set1_epi32((instruction_flags & RJ_ARITH_ALU_NEG_C) ? INT32_MIN : 0);
  const float scales[] = {1, 2, 4, 0.5f};
  const uint32_t omod = (instruction_flags >> 6) & 3;
  const auto scale = _mm256_set1_ps(scales[omod]);
  for (int i = 0; i < 32; i += 8) {
    auto va = _mm256_castsi256_ps(_mm256_xor_si256(
        _mm256_and_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + i)), keep_a),
        flip_a));
    auto vb = _mm256_castsi256_ps(_mm256_xor_si256(
        _mm256_and_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + i)), keep_b),
        flip_b));
    auto vc = _mm256_castsi256_ps(_mm256_xor_si256(
        _mm256_and_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(c + i)), keep_c),
        flip_c));

    if ((flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS)) {
      va = flush_denorm_f32(va);
      vb = flush_denorm_f32(vb);
      vc = flush_denorm_f32(vc);
    }

    auto result = _mm256_fmadd_ps(va, vb, vc);
    if ((flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
      result = flush_denorm_f32(result);
    }
    if (omod) {
      result = prepare_omod_f32(result, instruction_flags);
      result = _mm256_mul_ps(result, scale);
    }
    if (instruction_flags & RJ_ARITH_ALU_CLAMP) {
      result = clamp_f32(result, flags & RJ_ARITH_FP_DX10_CLAMP_OFF);
    }
    __m256i lane_exec_mask = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> i)),
                                               _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + i), lane_exec_mask,
                           _mm256_castps_si256(result));
  }
}

} // namespace rocjitsu::arith
