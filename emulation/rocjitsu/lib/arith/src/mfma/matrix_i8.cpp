// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// gfx942/gfx950 signed-byte MFMA and gfx1250 mixed-signedness byte WMMA.
#include "common/internal.h"
#include "mfma/backends.h"
#include "mfma/integer.h"
#include "rj_arith/rj_arith.h"
namespace {
// Validate source controls and dispatch the exact integer reduction.
template <int M, int K, int Batch, bool Wmma>
int evaluate(uint64_t flags, uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
             const uint32_t *const *b, const uint32_t *const *c) {
  using namespace rocjitsu::arith;
  flags = resolve_compatibility(flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL);
  constexpr uint64_t allowed =
      Wmma ? RJ_ARITH_WMMA_SIGNED_A | RJ_ARITH_WMMA_SIGNED_B | RJ_ARITH_WMMA_CLAMP : 0x3ffULL;
  if (int error = validate(flags, controls & ~allowed,
                           kMatrixIgnoredFpFlags | RJ_ARITH_FP_DX10_CLAMP_OFF)) {
    return error;
  }
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
    matrix_i8_x86_64_v4<M, K, Batch, Wmma>(controls, d, a, b, c);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    matrix_i8_x86_64_v3<M, K, Batch, Wmma>(controls, d, a, b, c);
    return RJ_ARITH_SUCCESS;
  }
#endif
  matrix_i8_kernel<M, K, Batch, Wmma>(controls, d, a, b, c);
  return RJ_ARITH_SUCCESS;
}
} // namespace

// Multiply signed bytes with wrapping 32-bit accumulation.
int rj_arith_v_mfma_i32_32x32x4_2b_i8_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                          const uint32_t *const *a, const uint32_t *const *b,
                                          const uint32_t *const *c) {
  return evaluate<32, 4, 2, false>(flags, controls, d, a, b, c);
}

// Multiply signed bytes with wrapping 32-bit accumulation.
int rj_arith_v_mfma_i32_16x16x4_4b_i8_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                          const uint32_t *const *a, const uint32_t *const *b,
                                          const uint32_t *const *c) {
  return evaluate<16, 4, 4, false>(flags, controls, d, a, b, c);
}

// Multiply signed bytes with wrapping 32-bit accumulation.
int rj_arith_v_mfma_i32_4x4x4_16b_i8_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                         const uint32_t *const *a, const uint32_t *const *b,
                                         const uint32_t *const *c) {
  return evaluate<4, 4, 16, false>(flags, controls, d, a, b, c);
}

// Multiply signed bytes with wrapping 32-bit accumulation.
int rj_arith_v_mfma_i32_32x32x16_i8_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        const uint32_t *const *c) {
  return evaluate<32, 16, 1, false>(flags, controls, d, a, b, c);
}

// Multiply signed bytes with wrapping 32-bit accumulation.
int rj_arith_v_mfma_i32_16x16x32_i8_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        const uint32_t *const *c) {
  return evaluate<16, 32, 1, false>(flags, controls, d, a, b, c);
}

// Multiply signed bytes with wrapping 32-bit accumulation.
int rj_arith_v_mfma_i32_16x16x64_i8_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        const uint32_t *const *c) {
  return evaluate<16, 64, 1, false>(flags, controls, d, a, b, c);
}

// Multiply signed bytes with wrapping 32-bit accumulation.
int rj_arith_v_mfma_i32_32x32x32_i8_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        const uint32_t *const *c) {
  return evaluate<32, 32, 1, false>(flags, controls, d, a, b, c);
}

// Multiply independently signed/unsigned bytes, optionally saturating the final sum.
int rj_arith_v_wmma_i32_16x16x64_iu8_w32(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                         const uint32_t *const *a, const uint32_t *const *b,
                                         const uint32_t *const *c) {
  return evaluate<16, 64, 1, true>(flags, controls, d, a, b, c);
}
