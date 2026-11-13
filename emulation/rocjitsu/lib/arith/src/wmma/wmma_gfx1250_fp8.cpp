// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// gfx1250 OCP FP8/BF8 WMMA validation, baseline arithmetic and CPU dispatch.
#include "common/float_formats.h"
#include "common/internal.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_wmma.h"
#include "wmma/flags.h"
#include "wmma/gfx1250_backends.h"
#include "wmma/gfx1250_fp8.h"
#include "wmma/gfx1250_half.h"
namespace {
// Evaluate the legacy approximate model, rejecting strict exact modes before operand access.
template <int K, bool Bf8A, bool Bf8B, bool Packed>
int evaluate(uint64_t flags, uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
             const uint32_t *const *b, const uint32_t *const *c) {
  using namespace rocjitsu::arith;
  const bool compatibility = (flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_COMPATIBILITY;
  flags = resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = validate(
          flags,
          controls & ~uint64_t(kWmmaInstructionFlags | (Packed ? RJ_ARITH_WMMA_INLINE_C : 0)),
          kMatrixIgnoredFpFlags | RJ_ARITH_FP_DX10_CLAMP_OFF)) {
    return error;
  }
  if ((flags & RJ_ARITH_SEMANTICS_MASK) >= RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS) {
    return RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS;
  }
  uint32_t inline_word = 0;
  const uint32_t *inline_c[] = {&inline_word};
  if constexpr (Packed) {
    if (compatibility) {
      // Preserve the legacy packed-C signs and full-width inline constant.
      controls &= ~(RJ_ARITH_WMMA_ABS_C | RJ_ARITH_WMMA_NEG_C);
    } else if (controls & RJ_ARITH_WMMA_INLINE_C) {
      inline_word = as_bits(f16_to_float(uint16_t(c[0][0])));
      c = inline_c;
    }
  }
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
    wmma_gfx1250_fp8_x86_64_v4<K, Bf8A, Bf8B, Packed>(flags, controls, d, a, b, c);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    wmma_gfx1250_fp8_x86_64_v3<K, Bf8A, Bf8B, Packed>(flags, controls, d, a, b, c);
    return RJ_ARITH_SUCCESS;
  }
#endif
  constexpr auto form = Packed ? Wmma1250Form::F16 : Wmma1250Form::F32F16;
  float left[16 * K], right[16 * K], result[16][16];
  for (int i = 0; i < 16 * K; ++i) {
    left[i] = fp8_to_float<Bf8A>(uint8_t(a[i / 128][i / 4 % 32] >> (8 * (i % 4))));
    right[i] = fp8_to_float<Bf8B>(uint8_t(b[i / 128][i / 4 % 32] >> (8 * (i % 4))));
  }
  for (int row = 0; row < 16; ++row) {
    for (int col = 0; col < 16; ++col) {
      float acc = gfx1250_read_c<form>(c, row, col, controls);
      for (int k = 0; k < K; ++k) {
        acc += left[gfx1250_fp8_index(row, k)] * right[gfx1250_fp8_index(col, k)];
      }
      result[row][col] = acc;
    }
  }
  gfx1250_write_result<form>(flags, d, result, flags & RJ_ARITH_FP16_OVFL);
  return RJ_ARITH_SUCCESS;
}
} // namespace

// Evaluate gfx1250 FP32 output with OCP byte inputs.
int rj_arith_v_wmma_f32_16x16x64_fp8_fp8_w32(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                             const uint32_t *const *a, const uint32_t *const *b,
                                             const uint32_t *const *c) {
  return evaluate<64, false, false, false>(flags, controls, d, a, b, c);
}

// Evaluate gfx1250 packed FP16 output with OCP byte inputs.
int rj_arith_v_wmma_f16_16x16x64_fp8_fp8_w32(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                             const uint32_t *const *a, const uint32_t *const *b,
                                             const uint32_t *const *c) {
  return evaluate<64, false, false, true>(flags, controls, d, a, b, c);
}

// Evaluate gfx1250 FP32 output with OCP byte inputs.
int rj_arith_v_wmma_f32_16x16x64_fp8_bf8_w32(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                             const uint32_t *const *a, const uint32_t *const *b,
                                             const uint32_t *const *c) {
  return evaluate<64, false, true, false>(flags, controls, d, a, b, c);
}

// Evaluate gfx1250 packed FP16 output with OCP byte inputs.
int rj_arith_v_wmma_f16_16x16x64_fp8_bf8_w32(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                             const uint32_t *const *a, const uint32_t *const *b,
                                             const uint32_t *const *c) {
  return evaluate<64, false, true, true>(flags, controls, d, a, b, c);
}

// Evaluate gfx1250 FP32 output with OCP byte inputs.
int rj_arith_v_wmma_f32_16x16x64_bf8_fp8_w32(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                             const uint32_t *const *a, const uint32_t *const *b,
                                             const uint32_t *const *c) {
  return evaluate<64, true, false, false>(flags, controls, d, a, b, c);
}

// Evaluate gfx1250 packed FP16 output with OCP byte inputs.
int rj_arith_v_wmma_f16_16x16x64_bf8_fp8_w32(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                             const uint32_t *const *a, const uint32_t *const *b,
                                             const uint32_t *const *c) {
  return evaluate<64, true, false, true>(flags, controls, d, a, b, c);
}

// Evaluate gfx1250 FP32 output with OCP byte inputs.
int rj_arith_v_wmma_f32_16x16x64_bf8_bf8_w32(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                             const uint32_t *const *a, const uint32_t *const *b,
                                             const uint32_t *const *c) {
  return evaluate<64, true, true, false>(flags, controls, d, a, b, c);
}

// Evaluate gfx1250 packed FP16 output with OCP byte inputs.
int rj_arith_v_wmma_f16_16x16x64_bf8_bf8_w32(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                             const uint32_t *const *a, const uint32_t *const *b,
                                             const uint32_t *const *c) {
  return evaluate<64, true, true, true>(flags, controls, d, a, b, c);
}

// Evaluate gfx1250 FP32 output with OCP byte inputs.
int rj_arith_v_wmma_f32_16x16x128_fp8_fp8_w32(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                              const uint32_t *const *a, const uint32_t *const *b,
                                              const uint32_t *const *c) {
  return evaluate<128, false, false, false>(flags, controls, d, a, b, c);
}

// Evaluate gfx1250 packed FP16 output with OCP byte inputs.
int rj_arith_v_wmma_f16_16x16x128_fp8_fp8_w32(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                              const uint32_t *const *a, const uint32_t *const *b,
                                              const uint32_t *const *c) {
  return evaluate<128, false, false, true>(flags, controls, d, a, b, c);
}

// Evaluate gfx1250 FP32 output with OCP byte inputs.
int rj_arith_v_wmma_f32_16x16x128_fp8_bf8_w32(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                              const uint32_t *const *a, const uint32_t *const *b,
                                              const uint32_t *const *c) {
  return evaluate<128, false, true, false>(flags, controls, d, a, b, c);
}

// Evaluate gfx1250 packed FP16 output with OCP byte inputs.
int rj_arith_v_wmma_f16_16x16x128_fp8_bf8_w32(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                              const uint32_t *const *a, const uint32_t *const *b,
                                              const uint32_t *const *c) {
  return evaluate<128, false, true, true>(flags, controls, d, a, b, c);
}

// Evaluate gfx1250 FP32 output with OCP byte inputs.
int rj_arith_v_wmma_f32_16x16x128_bf8_fp8_w32(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                              const uint32_t *const *a, const uint32_t *const *b,
                                              const uint32_t *const *c) {
  return evaluate<128, true, false, false>(flags, controls, d, a, b, c);
}

// Evaluate gfx1250 packed FP16 output with OCP byte inputs.
int rj_arith_v_wmma_f16_16x16x128_bf8_fp8_w32(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                              const uint32_t *const *a, const uint32_t *const *b,
                                              const uint32_t *const *c) {
  return evaluate<128, true, false, true>(flags, controls, d, a, b, c);
}

// Evaluate gfx1250 FP32 output with OCP byte inputs.
int rj_arith_v_wmma_f32_16x16x128_bf8_bf8_w32(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                              const uint32_t *const *a, const uint32_t *const *b,
                                              const uint32_t *const *c) {
  return evaluate<128, true, true, false>(flags, controls, d, a, b, c);
}

// Evaluate gfx1250 packed FP16 output with OCP byte inputs.
int rj_arith_v_wmma_f16_16x16x128_bf8_bf8_w32(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                              const uint32_t *const *a, const uint32_t *const *b,
                                              const uint32_t *const *c) {
  return evaluate<128, true, true, true>(flags, controls, d, a, b, c);
}
