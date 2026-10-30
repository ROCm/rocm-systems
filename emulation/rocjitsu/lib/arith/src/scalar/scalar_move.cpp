// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Scalar moves, conditional selection, and sign-extended immediates.
// Owns public entry points and their validation/dispatch policy.

// Register transfer semantics follow rocjitsu generated/shared/execute_shared.h.

#include "common/internal.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace {

// Validate flags and optionally write a to D; write=false preserves D.
template <typename T> int move(uint64_t flags, uint64_t mode, T *d, T a, bool write) {
  if (mode >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (int error = rocjitsu::arith::validate(flags, uint32_t(mode))) {
    return error;
  }
  if (write) {
    *d = a;
  }
  return RJ_ARITH_SUCCESS;
}

// Sign-extend the instruction's 16-bit immediate into a 32-bit scalar word.
uint32_t signed_immediate(uint16_t immediate) {
  return uint32_t(immediate) | ((immediate & 0x8000U) ? 0xffff0000U : 0);
}

} // namespace

// Copy the selected source value.
// Return errors before changing outputs.
int rj_arith_s_mov_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return move(flags, instruction_flags, d, a, true);
}

// Copy the selected source value.
// Return errors before changing outputs.
int rj_arith_s_mov_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return move(flags, instruction_flags, d, a, true);
}

// Copy the source when SCC is set.
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_cmov_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                        uint32_t input_scc) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return move(flags, instruction_flags, d, a, (input_scc & 1) != 0);
}

// Copy the source when SCC is set.
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_cmov_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                        uint32_t input_scc) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return move(flags, instruction_flags, d, a, (input_scc & 1) != 0);
}

// Copy a sign-extended immediate.
// Return errors before changing outputs.
int rj_arith_s_movk_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                        uint16_t immediate) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return move(flags, instruction_flags, d, signed_immediate(immediate), true);
}

// Copy a sign-extended immediate when SCC is set.
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_cmovk_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                         uint16_t immediate, uint32_t input_scc) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return move(flags, instruction_flags, d, signed_immediate(immediate), (input_scc & 1) != 0);
}

// Return the supplied guest program-counter value.
// Return errors before changing outputs.
int rj_arith_s_getpc_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t pc) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return rj_arith_s_mov_b64(flags, instruction_flags, d, pc + 4);
}
