// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// gfx1250 FP16/BF16 WMMA validation, baseline reduction and CPU dispatch.
#include "common/float_formats.h"
#include "common/internal.h"
#include "rj_arith/rj_arith.h"
#include "wmma/gfx1250_backends.h"
#include "wmma/gfx1250_half.h"

namespace {
using rocjitsu::arith::Wmma1250Form;
// Evaluate a complete Wave32 matrix, or reject unsupported flags before accessing operands.
template <Wmma1250Form Form>
int evaluate(uint64_t flags, uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
             const uint32_t *const *b, const uint32_t *const *c) {
  using namespace rocjitsu::arith;
  constexpr bool Bf16 = Form != Wmma1250Form::F32F16 && Form != Wmma1250Form::F16;
  constexpr bool PackedC = Form == Wmma1250Form::F16 || Form == Wmma1250Form::Bf16;
  constexpr bool DotFirst = Bf16 && !PackedC;
  const bool compatibility = (flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_COMPATIBILITY;
  const uint64_t packing_flags = flags;
  flags = resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = validate(
          flags,
          controls & ~uint64_t(kWmmaInstructionFlags | (PackedC ? RJ_ARITH_WMMA_INLINE_C : 0)),
          kMatrixIgnoredFpFlags | RJ_ARITH_FP_DX10_CLAMP_OFF)) {
    return error;
  }
  if ((flags & RJ_ARITH_SEMANTICS_MASK) >= RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS) {
    return RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS;
  }
  uint32_t inline_word = 0;
  const uint32_t *inline_c[] = {&inline_word};
  if constexpr (PackedC) {
    if (compatibility) {
      // Preserve legacy same-type C: ignored signs and full-width inline FP32.
      controls &= ~(RJ_ARITH_WMMA_ABS_C | RJ_ARITH_WMMA_NEG_C);
    } else if (controls & RJ_ARITH_WMMA_INLINE_C) {
      // Loose inline C is a packed half; widen before the shared backend path.
      inline_word =
          as_bits(Bf16 ? bf16_to_float(uint16_t(c[0][0])) : f16_to_float(uint16_t(c[0][0])));
      c = inline_c;
    }
  }
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
    wmma_gfx1250_half_x86_64_v4<Form>(packing_flags, controls, d, a, b, c);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    wmma_gfx1250_half_x86_64_v3<Form>(packing_flags, controls, d, a, b, c);
    return RJ_ARITH_SUCCESS;
  }
#endif
  float left[512], right[512];
  for (int i = 0; i < 512; ++i) {
    uint16_t x = uint16_t(a[i / 64][i / 2 % 32] >> (16 * (i % 2)));
    uint16_t y = uint16_t(b[i / 64][i / 2 % 32] >> (16 * (i % 2)));
    left[i] = Bf16 ? bf16_to_float(x) : f16_to_float(x);
    right[i] = Bf16 ? bf16_to_float(y) : f16_to_float(y);
  }
  float result[16][16];
  for (int row = 0; row < 16; ++row) {
    for (int col = 0; col < 16; ++col) {
      float initial = gfx1250_read_c<Form>(c, row, col, controls);
      float acc = DotFirst ? 0.0f : initial;
      for (int k = 0; k < 32; ++k) {
        float x = left[gfx1250_half_index(row, k)], y = right[gfx1250_half_index(col, k)];
        // BF16 preserves fused updates, including cancellation of an overflowing product.
        if constexpr (Bf16) {
          acc = __builtin_fmaf(x, y, acc);
        } else {
          acc += x * y;
        }
      }
      if constexpr (DotFirst) {
        acc += initial;
      }
      if constexpr (Form == Wmma1250Form::Bf16F32) {
        acc = gfx1250_repair_nan(acc, left, right, row, col, initial);
      }
      result[row][col] = acc;
    }
  }
  gfx1250_write_result<Form>(packing_flags, d, result);
  return RJ_ARITH_SUCCESS;
}
} // namespace

// Evaluate gfx1250 F16-input WMMA with FP32 output.
int rj_arith_v_wmma_f32_16x16x32_f16_w32(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                         const uint32_t *const *a, const uint32_t *const *b,
                                         const uint32_t *const *c) {
  return evaluate<Wmma1250Form::F32F16>(flags, controls, d, a, b, c);
}

// Evaluate gfx1250 BF16-input WMMA with FP32 output.
int rj_arith_v_wmma_f32_16x16x32_bf16_w32(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                          const uint32_t *const *a, const uint32_t *const *b,
                                          const uint32_t *const *c) {
  return evaluate<Wmma1250Form::F32Bf16>(flags, controls, d, a, b, c);
}

// Evaluate gfx1250 F16-result WMMA with final-only narrowing.
int rj_arith_v_wmma_f16_16x16x32_f16_w32(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                         const uint32_t *const *a, const uint32_t *const *b,
                                         const uint32_t *const *c) {
  return evaluate<Wmma1250Form::F16>(flags, controls, d, a, b, c);
}

// Evaluate gfx1250 BF16-result WMMA with final-only narrowing.
int rj_arith_v_wmma_bf16_16x16x32_bf16_w32(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                           const uint32_t *const *a, const uint32_t *const *b,
                                           const uint32_t *const *c) {
  return evaluate<Wmma1250Form::Bf16>(flags, controls, d, a, b, c);
}

// Evaluate gfx1250 BF16F32-result WMMA with final-only narrowing.
int rj_arith_v_wmma_bf16f32_16x16x32_bf16_w32(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                              const uint32_t *const *a, const uint32_t *const *b,
                                              const uint32_t *const *c) {
  return evaluate<Wmma1250Form::Bf16F32>(flags, controls, d, a, b, c);
}
