// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// gfx1251 FP64 WMMA validation and dispatch using the inherited fixed native-FMA model.
#include "common/dense_f64.h"
#include "common/dense_f64_backends.h"
#include "common/internal.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_wmma.h"

// Evaluate the whole Wave32 tile, ignoring guest FP controls. D may overlap A/B/C.
// No explicit empirical qualification is claimed for this inherited fixed arithmetic model.
int rj_arith_v_wmma_f64_16x16x4_f64_w32(uint64_t flags, uint64_t instruction_flags,
                                        uint32_t *const *d, const uint32_t *const *a,
                                        const uint32_t *const *b, const uint32_t *const *c) {
  using namespace rocjitsu::arith;
  flags = resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  constexpr uint64_t supported =
      RJ_ARITH_WMMA_NEG_LO_A | RJ_ARITH_WMMA_NEG_LO_B | RJ_ARITH_WMMA_NEG_C | RJ_ARITH_WMMA_ABS_C;
  if (int error = validate(flags, instruction_flags & ~supported,
                           kMatrixIgnoredFpFlags | RJ_ARITH_FP_DX10_CLAMP_OFF)) {
    return error;
  }
  if ((flags & RJ_ARITH_SEMANTICS_MASK) >= RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS) {
    return RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS;
  }
  const uint64_t controls =
      (instruction_flags & 7) | ((instruction_flags & RJ_ARITH_WMMA_ABS_C) ? 8 : 0);
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
    dense_f64_x86_64_v4<16, 1, 32>(controls, d, a, b, c);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    dense_f64_x86_64_v3<16, 1, 32>(controls, d, a, b, c);
    return RJ_ARITH_SUCCESS;
  }
#endif
  dense_f64_kernel<16, 1, 32>(controls, d, a, b, c);
  return RJ_ARITH_SUCCESS;
}
