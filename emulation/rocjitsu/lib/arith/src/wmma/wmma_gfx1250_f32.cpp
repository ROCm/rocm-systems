// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// gfx1250 FP32 WMMA validation, baseline arithmetic and CPU dispatch.
#include "common/float_formats.h"
#include "common/internal.h"
#include "rj_arith/rj_arith.h"
#include "wmma/gfx1250_backends.h"
#include "wmma/gfx1250_f32.h"
#include "wmma/gfx1250_half.h"
// Evaluate the legacy approximate model, rejecting strict exact modes before operand access.
int rj_arith_v_wmma_f32_16x16x4_f32_w32(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        const uint32_t *const *c) {
  using namespace rocjitsu::arith;
  flags = resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = validate(flags, controls & ~uint64_t(kWmmaInstructionFlags),
                           kMatrixIgnoredFpFlags | RJ_ARITH_FP_DX10_CLAMP_OFF)) {
    return error;
  }
  if ((flags & RJ_ARITH_SEMANTICS_MASK) >= RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS) {
    return RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS;
  }
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
    wmma_gfx1250_f32_x86_64_v4(controls, d, a, b, c);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    wmma_gfx1250_f32_x86_64_v3(controls, d, a, b, c);
    return RJ_ARITH_SUCCESS;
  }
#endif
  wmma_gfx1250_f32_kernel(controls, d, a, b, c);
  return RJ_ARITH_SUCCESS;
}
