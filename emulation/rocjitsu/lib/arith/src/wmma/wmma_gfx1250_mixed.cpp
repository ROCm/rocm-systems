// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// gfx1250 mixed-format WMMA validation and CPU dispatch.
#include "common/internal.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_wmma.h"
#include "wmma/gfx1250_backends.h"
#include "wmma/gfx1250_mixed.h"
namespace {
// Validate formats and policy before accessing any register storage.
template <int M, int Block = 0>
int evaluate(uint64_t flags, uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
             const uint32_t *const *b, const uint32_t *const *c,
             const uint32_t *const *scale_a = nullptr, const uint32_t *const *scale_b = nullptr) {
  using namespace rocjitsu::arith;
  flags = resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error =
          validate(flags, controls & ~((M == 32 ? 0xc0ULL : 0xffULL) | (Block ? 0x3f00ULL : 0)),
                   kMatrixIgnoredFpFlags | RJ_ARITH_FP_DX10_CLAMP_OFF)) {
    return error;
  }
  if constexpr (M == 16) {
    if ((controls & 7) > 4 || ((controls >> 3) & 7) > 4) {
      return RJ_ARITH_ERROR_INVALID_FLAGS;
    }
  }
  if constexpr (Block != 0) {
    if (((controls >> 8) & 3) == 3 || ((controls >> 10) & 3) == 3) {
      return RJ_ARITH_ERROR_INVALID_FLAGS;
    }
  }
  if ((flags & RJ_ARITH_SEMANTICS_MASK) >= RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS) {
    return RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS;
  }
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
    if constexpr (Block != 0) {
      wmma_gfx1250_scaled_x86_64_v4<M, Block>(controls, d, a, b, c, scale_a, scale_b);
    } else {
      wmma_gfx1250_mixed_x86_64_v4<M>(controls, d, a, b, c);
    }
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    if constexpr (Block != 0) {
      wmma_gfx1250_scaled_x86_64_v3<M, Block>(controls, d, a, b, c, scale_a, scale_b);
    } else {
      wmma_gfx1250_mixed_x86_64_v3<M>(controls, d, a, b, c);
    }
    return RJ_ARITH_SUCCESS;
  }
#endif
  gfx1250_mixed_kernel<M, Block>(controls, d, a, b, c, scale_a, scale_b);
  return RJ_ARITH_SUCCESS;
}
} // namespace
// Evaluate the inherited approximate WMMA model with staged destination publication.
int rj_arith_v_wmma_f32_16x16x128_f8f6f4_w32(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                             const uint32_t *const *a, const uint32_t *const *b,
                                             const uint32_t *const *c) {
  return evaluate<16>(flags, controls, d, a, b, c);
}
// Evaluate the inherited approximate WMMA model with staged destination publication.
int rj_arith_v_wmma_f32_32x16x128_f4_w32(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                         const uint32_t *const *a, const uint32_t *const *b,
                                         const uint32_t *const *c) {
  return evaluate<32>(flags, controls, d, a, b, c);
}

// Evaluate ordered block dots, separate scale multiplies and accumulator addition.
int rj_arith_v_wmma_scale_f32_16x16x128_f8f6f4_w32(uint64_t flags, uint64_t controls,
                                                   uint32_t *const *d, const uint32_t *const *a,
                                                   const uint32_t *const *b,
                                                   const uint32_t *const *c,
                                                   const uint32_t *const *scale_a,
                                                   const uint32_t *const *scale_b) {
  return evaluate<16, 32>(flags, controls, d, a, b, c, scale_a, scale_b);
}

// Evaluate ordered block dots, separate scale multiplies and accumulator addition.
int rj_arith_v_wmma_scale16_f32_16x16x128_f8f6f4_w32(uint64_t flags, uint64_t controls,
                                                     uint32_t *const *d, const uint32_t *const *a,
                                                     const uint32_t *const *b,
                                                     const uint32_t *const *c,
                                                     const uint32_t *const *scale_a,
                                                     const uint32_t *const *scale_b) {
  return evaluate<16, 16>(flags, controls, d, a, b, c, scale_a, scale_b);
}

// Evaluate ordered block dots, separate scale multiplies and accumulator addition.
int rj_arith_v_wmma_scale_f32_32x16x128_f4_w32(uint64_t flags, uint64_t controls,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b, const uint32_t *const *c,
                                               const uint32_t *const *scale_a,
                                               const uint32_t *const *scale_b) {
  return evaluate<32, 32>(flags, controls, d, a, b, c, scale_a, scale_b);
}

// Evaluate ordered block dots, separate scale multiplies and accumulator addition.
int rj_arith_v_wmma_scale16_f32_32x16x128_f4_w32(uint64_t flags, uint64_t controls,
                                                 uint32_t *const *d, const uint32_t *const *a,
                                                 const uint32_t *const *b, const uint32_t *const *c,
                                                 const uint32_t *const *scale_a,
                                                 const uint32_t *const *scale_b) {
  return evaluate<32, 16>(flags, controls, d, a, b, c, scale_a, scale_b);
}
