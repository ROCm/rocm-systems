// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 fused FMA and separately rounded MAD share operand and output handling.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "common/simd.h"
#include "common/x86_64/alu_x86_64_v3.h"
#include "fp/fma.h"
#include "rj_arith/detail/rj_arith_fp.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {

namespace {

// Round and flush the product before adding C. Quiet the first NaN at
// each step, matching the inherited separately rounded scalar model.
__m256 mad_value(__m256 a, __m256 b, __m256 c) {
  auto quiet = _mm256_castsi256_ps(_mm256_set1_epi32(0x00400000));
  auto product = _mm256_mul_ps(a, b);
  product = _mm256_blendv_ps(product, _mm256_or_ps(a, quiet), _mm256_cmp_ps(a, a, _CMP_UNORD_Q));
  product = flush_denorm_f32(product);
  auto value = _mm256_add_ps(product, c);
  return _mm256_blendv_ps(value, _mm256_or_ps(product, quiet),
                          _mm256_cmp_ps(product, product, _CMP_UNORD_Q));
}

// Evaluate FP32 multiply/add, fused unless Separate is selected. Requires validated controls;
// preserve inactive destination lanes.
template <bool Dx9Zero, FmaOperands Operands = FmaOperands::Registers, bool Separate = false>
void run(uint32_t exec_mask, uint32_t instruction_flags, uint32_t *d, const uint32_t *a,
         const uint32_t *b, const uint32_t *c, uint32_t literal = 0, uint64_t flags = 0) {
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
    __m256 vc;
    if constexpr (Operands == FmaOperands::MultiplyLiteral) {
      vc = vb;
      vb = _mm256_castsi256_ps(_mm256_set1_epi32(int(literal)));
    } else if constexpr (Operands == FmaOperands::AddLiteral) {
      vc = _mm256_castsi256_ps(_mm256_set1_epi32(int(literal)));
    } else {
      vc = _mm256_castsi256_ps(_mm256_xor_si256(
          _mm256_and_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(c + i)), keep_c),
          flip_c));
    }
    if (Dx9Zero || (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS)) {
      va = flush_denorm_f32(va);
      vb = flush_denorm_f32(vb);
      vc = flush_denorm_f32(vc);
    }
    if constexpr (Dx9Zero) {
      auto zero = _mm256_setzero_ps();
      auto has_zero =
          _mm256_or_ps(_mm256_cmp_ps(va, zero, _CMP_EQ_OQ), _mm256_cmp_ps(vb, zero, _CMP_EQ_OQ));
      va = _mm256_blendv_ps(va, zero, has_zero);
      vb = _mm256_blendv_ps(vb, _mm256_set1_ps(1), has_zero);
    }
    __m256 result;
    if constexpr (Separate) {
      result = mad_value(va, vb, vc);
    } else {
      result = _mm256_fmadd_ps(va, vb, vc);
    }
    if (Dx9Zero || (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
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

} // namespace

// Evaluate two separately rounded FP32 operations, preserving inactive lanes.
// Requires validated controls and the named ISA; flushing is forced by the caller.
void mad_f32_x86_64_v3(bool legacy, uint64_t flags, uint32_t exec_mask, uint32_t instruction_flags,
                       uint32_t *d, const uint32_t *a, const uint32_t *b, const uint32_t *c) {
  if (legacy) {
    run<true, FmaOperands::Registers, true>(exec_mask, instruction_flags, d, a, b, c, 0, flags);
  } else {
    run<false, FmaOperands::Registers, true>(exec_mask, instruction_flags, d, a, b, c, 0, flags);
  }
}

// Evaluate FP32 fused multiply-add. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
void fma_x86_64_v3(uint64_t flags, uint32_t exec_mask, uint32_t instruction_flags, uint32_t *d,
                   const uint32_t *a, const uint32_t *b, const uint32_t *c) {
  run<false>(exec_mask, instruction_flags, d, a, b, c, 0, flags);
}

// Evaluate FP32 fused multiply-add. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
void fma_dx9_zero_x86_64_v3(uint64_t flags, uint32_t exec_mask, uint32_t instruction_flags,
                            uint32_t *d, const uint32_t *a, const uint32_t *b, const uint32_t *c) {
  run<true>(exec_mask, instruction_flags, d, a, b, c, 0, flags);
}

// Evaluate FP32 fused multiply-add. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <rocjitsu::arith::FmaOperands Operands>
void literal_fma_x86_64_v3(uint64_t flags, uint32_t exec_mask, uint32_t literal, uint32_t *d,
                           const uint32_t *a, const uint32_t *b) {
  run<false, Operands>(exec_mask, 0, d, a, b, nullptr, literal, flags);
}

template void literal_fma_x86_64_v3<rocjitsu::arith::FmaOperands::AddLiteral>(uint64_t, uint32_t,
                                                                              uint32_t, uint32_t *,
                                                                              const uint32_t *,
                                                                              const uint32_t *);
template void literal_fma_x86_64_v3<rocjitsu::arith::FmaOperands::MultiplyLiteral>(
    uint64_t, uint32_t, uint32_t, uint32_t *, const uint32_t *, const uint32_t *);

} // namespace rocjitsu::arith
