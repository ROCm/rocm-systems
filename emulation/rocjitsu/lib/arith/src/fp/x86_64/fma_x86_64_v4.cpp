// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 FMA and separately rounded MAD with shared modifiers and masked stores.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "common/simd.h"
#include "common/x86_64/alu_x86_64_v4.h"
#include "fp/fma.h"
#include "rj_arith/detail/rj_arith_fp.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {
namespace {

// Apply guest rounding independently to the multiply and add, without
// touching host controls. Keep the inherited first-NaN priority at each step.
template <int Round> __m512 mad_value(__m512 a, __m512 b, __m512 c) {
  auto quiet = _mm512_castsi512_ps(_mm512_set1_epi32(0x00400000));
  auto product = _mm512_mul_round_ps(a, b, Round | _MM_FROUND_NO_EXC);
  product =
      _mm512_mask_mov_ps(product, _mm512_cmp_ps_mask(a, a, _CMP_UNORD_Q), _mm512_or_ps(a, quiet));
  product = flush_denorm_f32(product);
  auto value = _mm512_add_round_ps(product, c, Round | _MM_FROUND_NO_EXC);
  return _mm512_mask_mov_ps(value, _mm512_cmp_ps_mask(product, product, _CMP_UNORD_Q),
                            _mm512_or_ps(product, quiet));
}

// Select the instruction-local rounding immediate required by guest MODE.
__m512 mad_value(__m512 a, __m512 b, __m512 c, unsigned rounding) {
  switch (rounding) {
  case 1:
    return mad_value<_MM_FROUND_TO_POS_INF>(a, b, c);
  case 2:
    return mad_value<_MM_FROUND_TO_NEG_INF>(a, b, c);
  case 3:
    return mad_value<_MM_FROUND_TO_ZERO>(a, b, c);
  default:
    return mad_value<_MM_FROUND_TO_NEAREST_INT>(a, b, c);
  }
}

// Round output scaling with the same guest direction as both MAD operations.
__m512 mad_scale(__m512 value, __m512 scale, unsigned rounding) {
  switch (rounding) {
  case 1:
    return _mm512_mul_round_ps(value, scale, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
  case 2:
    return _mm512_mul_round_ps(value, scale, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
  case 3:
    return _mm512_mul_round_ps(value, scale, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
  default:
    return _mm512_mul_round_ps(value, scale, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  }
}

// Evaluate FP32 multiply/add, fused unless Separate is selected. Requires validated controls;
// preserve inactive destination lanes.
template <bool Dx9Zero, FmaOperands Operands = FmaOperands::Registers, bool Separate = false>
void run(uint32_t exec_mask, uint32_t instruction_flags, uint32_t *d, const uint32_t *a,
         const uint32_t *b, const uint32_t *c, uint32_t literal = 0, uint64_t flags = 0) {
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
    if constexpr (Operands == FmaOperands::MultiplyLiteral) {
      vc = vb;
      vb = _mm512_castsi512_ps(_mm512_set1_epi32(int(literal)));
    } else if constexpr (Operands == FmaOperands::AddLiteral) {
      vc = _mm512_castsi512_ps(_mm512_set1_epi32(int(literal)));
    } else {
      vc = _mm512_castsi512_ps(
          _mm512_xor_si512(_mm512_and_si512(_mm512_loadu_si512(c + i), keep_c), flip_c));
    }
    auto evaluate = [&] {
      if constexpr (Separate) {
        return mad_value(va, vb, vc, unsigned((flags & RJ_ARITH_FP_ROUND_MASK) >> 24));
      } else {
        return _mm512_fmadd_ps(va, vb, vc);
      }
    };
    __m512 result;
    if constexpr (Dx9Zero) {
      auto magnitude = [](__m512 v) {
        return _mm512_and_si512(_mm512_castps_si512(v), _mm512_set1_epi32(INT32_MAX));
      };
      auto lower = _mm512_set1_epi32(0x00800000);
      auto zero_product =
          _mm512_cmplt_epu32_mask(_mm512_min_epu32(magnitude(va), magnitude(vb)), lower);
      auto subnormal = [&](auto v) {
        return _mm512_cmplt_epu32_mask(_mm512_sub_epi32(magnitude(v), _mm512_set1_epi32(1)),
                                       _mm512_set1_epi32(0x007fffff));
      };
      if (zero_product) {
        va = _mm512_mask_mov_ps(va, zero_product, _mm512_setzero_ps());
        vb = _mm512_mask_mov_ps(vb, zero_product, _mm512_set1_ps(1));
      }
      if (subnormal(vc)) {
        vc = flush_denorm_f32(vc);
      }
      result = flush_denorm_f32(evaluate());
    } else {
      if (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) {
        va = flush_denorm_f32(va);
        vb = flush_denorm_f32(vb);
        vc = flush_denorm_f32(vc);
      }
      result = evaluate();
      if (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) {
        result = flush_denorm_f32(result);
      }
    }
    if (omod) {
      result = prepare_omod_f32(result, instruction_flags);
      if constexpr (Separate) {
        result = mad_scale(result, scale, unsigned((flags & RJ_ARITH_FP_ROUND_MASK) >> 24));
      } else {
        result = _mm512_mul_ps(result, scale);
      }
    }
    if (instruction_flags & RJ_ARITH_ALU_CLAMP) {
      result = clamp_f32(result, flags & RJ_ARITH_FP_DX10_CLAMP_OFF);
    }
    _mm512_mask_storeu_epi32(d + i, static_cast<__mmask16>(exec_mask >> i),
                             _mm512_castps_si512(result));
  }
}

} // namespace

// Evaluate two separately rounded FP32 operations, preserving inactive lanes.
// Requires validated controls and the named ISA; flushing is forced by the caller.
void mad_f32_x86_64_v4(bool legacy, uint64_t flags, uint32_t exec_mask, uint32_t instruction_flags,
                       uint32_t *d, const uint32_t *a, const uint32_t *b, const uint32_t *c) {
  if (legacy) {
    run<true, FmaOperands::Registers, true>(exec_mask, instruction_flags, d, a, b, c, 0, flags);
  } else {
    run<false, FmaOperands::Registers, true>(exec_mask, instruction_flags, d, a, b, c, 0, flags);
  }
}

// Evaluate FP32 fused multiply-add. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
void fma_x86_64_v4(uint64_t flags, uint32_t exec_mask, uint32_t instruction_flags, uint32_t *d,
                   const uint32_t *a, const uint32_t *b, const uint32_t *c) {
  run<false>(exec_mask, instruction_flags, d, a, b, c, 0, flags);
}

// Evaluate FP32 fused multiply-add. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
void fma_dx9_zero_x86_64_v4(uint64_t flags, uint32_t exec_mask, uint32_t instruction_flags,
                            uint32_t *d, const uint32_t *a, const uint32_t *b, const uint32_t *c) {
  run<true>(exec_mask, instruction_flags, d, a, b, c, 0, flags);
}

// Evaluate FP32 fused multiply-add. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <rocjitsu::arith::FmaOperands Operands>
void literal_fma_x86_64_v4(uint64_t flags, uint32_t exec_mask, uint32_t literal, uint32_t *d,
                           const uint32_t *a, const uint32_t *b) {
  run<false, Operands>(exec_mask, 0, d, a, b, nullptr, literal, flags);
}

template void literal_fma_x86_64_v4<rocjitsu::arith::FmaOperands::AddLiteral>(uint64_t, uint32_t,
                                                                              uint32_t, uint32_t *,
                                                                              const uint32_t *,
                                                                              const uint32_t *);
template void literal_fma_x86_64_v4<rocjitsu::arith::FmaOperands::MultiplyLiteral>(
    uint64_t, uint32_t, uint32_t, uint32_t *, const uint32_t *, const uint32_t *);

} // namespace rocjitsu::arith
