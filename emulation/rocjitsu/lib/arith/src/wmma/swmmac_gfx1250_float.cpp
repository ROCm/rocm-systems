// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// gfx1250 floating sparse matrix validation and runtime ISA dispatch.
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_wmma.h"
#include "wmma/gfx1250_backends.h"
#include "wmma/gfx1250_sparse_float.h"
namespace {
// Validate semantics before accessing the tied destination or source pointers.
template <int Bits, bool BfA, bool BfB, int Form>
int evaluate(uint64_t flags, uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
             const uint32_t *const *b, const uint32_t *const *index) {
  using namespace rocjitsu::arith;
  flags = resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = validate(flags, controls & ~uint64_t(RJ_ARITH_SWMMAC_INDEX_KEY_1),
                           kMatrixIgnoredFpFlags | RJ_ARITH_FP_DX10_CLAMP_OFF)) {
    return error;
  }
  if ((flags & RJ_ARITH_SEMANTICS_MASK) >= RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS) {
    return RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS;
  }
  // LLVM folds a per-word logical metadata shift into index_key:1. Normalize
  // that encoding before the common scalar/SIMD layout decoder consumes it.
  uint32_t shifted[Bits == 16 ? 1 : 2][32];
  const uint32_t *selected[Bits == 16 ? 1 : 2];
  if (controls & RJ_ARITH_SWMMAC_INDEX_KEY_1) {
    for (int reg = 0; reg < (Bits == 16 ? 1 : 2); ++reg) {
      for (int lane = 0; lane < 32; ++lane) {
        shifted[reg][lane] = index[reg][lane] >> 16;
      }
      selected[reg] = shifted[reg];
    }
    index = selected;
  }
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
    swmmac_float_x86_64_v4<Bits, BfA, BfB, Form>(flags, d, a, b, index);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    swmmac_float_x86_64_v3<Bits, BfA, BfB, Form>(flags, d, a, b, index);
    return RJ_ARITH_SUCCESS;
  }
#endif
  swmmac1250_float_kernel<Bits, BfA, BfB, Form>(flags, d, a, b, index);
  return RJ_ARITH_SUCCESS;
}
} // namespace
// Accumulate selected sparse products and pack the result once.
int rj_arith_v_swmmac_f32_16x16x64_f16_w32(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                           const uint32_t *const *a, const uint32_t *const *b,
                                           const uint32_t *const *index) {
  return evaluate<16, false, false, 0>(flags, controls, d, a, b, index);
}
// Accumulate selected sparse products and pack the result once.
int rj_arith_v_swmmac_f32_16x16x64_bf16_w32(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                            const uint32_t *const *a, const uint32_t *const *b,
                                            const uint32_t *const *index) {
  return evaluate<16, true, true, 0>(flags, controls, d, a, b, index);
}
// Accumulate selected sparse products and pack the result once.
int rj_arith_v_swmmac_f16_16x16x64_f16_w32(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                           const uint32_t *const *a, const uint32_t *const *b,
                                           const uint32_t *const *index) {
  return evaluate<16, false, false, 1>(flags, controls, d, a, b, index);
}
// Accumulate selected sparse products and pack the result once.
int rj_arith_v_swmmac_bf16_16x16x64_bf16_w32(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                             const uint32_t *const *a, const uint32_t *const *b,
                                             const uint32_t *const *index) {
  return evaluate<16, true, true, 2>(flags, controls, d, a, b, index);
}
// Accumulate selected sparse products and pack the result once.
int rj_arith_v_swmmac_bf16f32_16x16x64_bf16_w32(uint64_t flags, uint64_t controls,
                                                uint32_t *const *d, const uint32_t *const *a,
                                                const uint32_t *const *b,
                                                const uint32_t *const *index) {
  return evaluate<16, true, true, 3>(flags, controls, d, a, b, index);
}
// Accumulate selected sparse products and pack the result once.
int rj_arith_v_swmmac_f32_16x16x128_fp8_fp8_w32(uint64_t flags, uint64_t controls,
                                                uint32_t *const *d, const uint32_t *const *a,
                                                const uint32_t *const *b,
                                                const uint32_t *const *index) {
  return evaluate<8, false, false, 0>(flags, controls, d, a, b, index);
}
// Accumulate selected sparse products and pack the result once.
int rj_arith_v_swmmac_f16_16x16x128_fp8_fp8_w32(uint64_t flags, uint64_t controls,
                                                uint32_t *const *d, const uint32_t *const *a,
                                                const uint32_t *const *b,
                                                const uint32_t *const *index) {
  return evaluate<8, false, false, 1>(flags, controls, d, a, b, index);
}
// Accumulate selected sparse products and pack the result once.
int rj_arith_v_swmmac_f32_16x16x128_fp8_bf8_w32(uint64_t flags, uint64_t controls,
                                                uint32_t *const *d, const uint32_t *const *a,
                                                const uint32_t *const *b,
                                                const uint32_t *const *index) {
  return evaluate<8, false, true, 0>(flags, controls, d, a, b, index);
}
// Accumulate selected sparse products and pack the result once.
int rj_arith_v_swmmac_f16_16x16x128_fp8_bf8_w32(uint64_t flags, uint64_t controls,
                                                uint32_t *const *d, const uint32_t *const *a,
                                                const uint32_t *const *b,
                                                const uint32_t *const *index) {
  return evaluate<8, false, true, 1>(flags, controls, d, a, b, index);
}
// Accumulate selected sparse products and pack the result once.
int rj_arith_v_swmmac_f32_16x16x128_bf8_fp8_w32(uint64_t flags, uint64_t controls,
                                                uint32_t *const *d, const uint32_t *const *a,
                                                const uint32_t *const *b,
                                                const uint32_t *const *index) {
  return evaluate<8, true, false, 0>(flags, controls, d, a, b, index);
}
// Accumulate selected sparse products and pack the result once.
int rj_arith_v_swmmac_f16_16x16x128_bf8_fp8_w32(uint64_t flags, uint64_t controls,
                                                uint32_t *const *d, const uint32_t *const *a,
                                                const uint32_t *const *b,
                                                const uint32_t *const *index) {
  return evaluate<8, true, false, 1>(flags, controls, d, a, b, index);
}
// Accumulate selected sparse products and pack the result once.
int rj_arith_v_swmmac_f32_16x16x128_bf8_bf8_w32(uint64_t flags, uint64_t controls,
                                                uint32_t *const *d, const uint32_t *const *a,
                                                const uint32_t *const *b,
                                                const uint32_t *const *index) {
  return evaluate<8, true, true, 0>(flags, controls, d, a, b, index);
}
// Accumulate selected sparse products and pack the result once.
int rj_arith_v_swmmac_f16_16x16x128_bf8_bf8_w32(uint64_t flags, uint64_t controls,
                                                uint32_t *const *d, const uint32_t *const *a,
                                                const uint32_t *const *b,
                                                const uint32_t *const *index) {
  return evaluate<8, true, true, 1>(flags, controls, d, a, b, index);
}
