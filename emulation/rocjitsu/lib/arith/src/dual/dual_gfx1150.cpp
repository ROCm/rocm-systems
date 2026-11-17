// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// gfx1150 dual-slot arithmetic delegates to the inherited scalar/SIMD models.
// The VM owns pair-wide reads and commits; these APIs evaluate a single Wave32 slot.
#include "common/validation.h"
#include "rj_arith/detail/rj_arith_dual.h"
#include "rj_arith/detail/rj_arith_fp.h"

// Evaluate one gfx1150 add slot; invalid instruction fields preserve all outputs.
int rj_arith_v_dual_add_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *trapsts) {
  if (instruction_flags) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return rj_arith_v_add_f32_w32_gfx1150(flags, exec_mask, 0, d, a, b, trapsts);
}

// Evaluate one gfx1150 sub slot; invalid instruction fields preserve all outputs.
int rj_arith_v_dual_sub_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *trapsts) {
  if (instruction_flags) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return rj_arith_v_sub_f32_w32_gfx1150(flags, exec_mask, 0, d, a, b, trapsts);
}

// Evaluate one gfx1150 subrev slot; invalid instruction fields preserve all outputs.
int rj_arith_v_dual_subrev_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                           uint64_t instruction_flags, uint32_t *const *d,
                                           const uint32_t *const *a, const uint32_t *const *b,
                                           uint32_t *trapsts) {
  if (instruction_flags) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return rj_arith_v_subrev_f32_w32_gfx1150(flags, exec_mask, 0, d, a, b, trapsts);
}

// Evaluate one gfx1150 mul slot; invalid instruction fields preserve all outputs.
int rj_arith_v_dual_mul_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        uint32_t *trapsts) {
  if (instruction_flags) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return rj_arith_v_mul_f32_w32_gfx1150(flags, exec_mask, 0, d, a, b, trapsts);
}

// Evaluate one gfx1150 fmac slot; invalid instruction fields preserve all outputs.
int rj_arith_v_dual_fmac_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                         uint64_t instruction_flags, uint32_t *const *d,
                                         const uint32_t *const *a, const uint32_t *const *b,
                                         uint32_t *trapsts) {
  if (instruction_flags) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  // C++ qualification conversion preserves D's pre-instruction accumulator storage.
  return rj_arith_v_fma_f32_w32_gfx1150(flags, exec_mask, 0, d, a, b, d, trapsts);
}

// Evaluate one gfx1150 fmaak slot; invalid instruction fields preserve all outputs.
int rj_arith_v_dual_fmaak_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                          uint64_t instruction_flags, uint32_t *const *d,
                                          const uint32_t *const *a, const uint32_t *const *b,
                                          uint32_t literal, uint32_t *trapsts) {
  if (instruction_flags) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  uint32_t literals[32];
  for (auto &value : literals) {
    value = literal;
  }
  const uint32_t *source = literals;
  return rj_arith_v_fma_f32_w32_gfx1150(flags, exec_mask, 0, d, a, b, &source, trapsts);
}

// Evaluate one gfx1150 fmamk slot; invalid instruction fields preserve all outputs.
int rj_arith_v_dual_fmamk_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                          uint64_t instruction_flags, uint32_t *const *d,
                                          const uint32_t *const *a, uint32_t literal,
                                          const uint32_t *const *b, uint32_t *trapsts) {
  if (instruction_flags) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  uint32_t literals[32];
  for (auto &value : literals) {
    value = literal;
  }
  const uint32_t *source = literals;
  return rj_arith_v_fma_f32_w32_gfx1150(flags, exec_mask, 0, d, a, &source, b, trapsts);
}

// Retain native legacy MUL's post-round flushing and absence of Compatibility status.
int rj_arith_v_dual_mul_dx9_zero_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                                 uint64_t instruction_flags, uint32_t *const *d,
                                                 const uint32_t *const *a, const uint32_t *const *b,
                                                 uint32_t *trapsts) {
  if (instruction_flags) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  const uint64_t semantics = flags & RJ_ARITH_SEMANTICS_MASK;
  if (semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS ||
      semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL) {
    return RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS;
  }
  if (!(flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
    return rj_arith_v_dual_mul_dx9_zero_f32_w32_gfx12(flags, exec_mask, 0, d, a, b, trapsts);
  }
  // Stage before stores so overlapping operands remain intact. The inherited native
  // multiply rounds first, then flushes subnormals, unlike the ordinary GPU FMA policy.
  uint32_t results[32]{};
  uint32_t *output = results;
  if (int error = rj_arith_v_dual_mul_dx9_zero_f32_w32_gfx12(
          flags & ~RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS, exec_mask, 0, &output, a, b, trapsts)) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  for (int lane = 0; lane < 32; ++lane) {
    if (exec_mask & (1U << lane)) {
      const uint32_t value = results[lane];
      d[0][lane] = (value & 0x7f800000U) ? value : value & 0x80000000U;
    }
  }
  return RJ_ARITH_SUCCESS;
}
