// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Scalar boolean operations on EXEC and saved-mask/SCC results.
// Owns public entry points and their validation/dispatch policy.

// Semantics follow rocjitsu generated/shared/execute_shared.h for RDNA3/RDNA4.

#include "common/internal.h"
#include "rj_arith/rj_arith.h"

#include <cstring>
#include <stdint.h>

namespace {

// Validate flags, then commit the result, updated EXEC, and SCC together.
template <typename T>
int store_exec(uint64_t flags, uint64_t mode, T *d, T result, T new_exec, T *exec, uint32_t *scc) {
  if (mode >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (int error = rocjitsu::arith::validate(flags, uint32_t(mode), true)) {
    return error;
  }
  uint32_t cc = new_exec != 0;
  *d = result;
  if (exec) {
    *exec = new_exec;
  }
  std::memcpy(scc, &cc, sizeof(cc));
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Set EXEC to exec_mask & a and save the incoming EXEC in D.
// Set SCC when the new EXEC is nonzero; errors preserve all outputs.
int rj_arith_s_and_saveexec_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                                uint32_t exec_mask, uint32_t *exec, uint32_t *scc) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  uint32_t new_exec = exec_mask & a;
  return store_exec(flags, instruction_flags, d, exec_mask, new_exec, exec, scc);
}

// Set EXEC to exec_mask | a and save the incoming EXEC in D.
// Set SCC when the new EXEC is nonzero; errors preserve all outputs.
int rj_arith_s_or_saveexec_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                               uint32_t exec_mask, uint32_t *exec, uint32_t *scc) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  uint32_t new_exec = exec_mask | a;
  return store_exec(flags, instruction_flags, d, exec_mask, new_exec, exec, scc);
}

// Set EXEC to exec_mask ^ a and save the incoming EXEC in D.
// Set SCC when the new EXEC is nonzero; errors preserve all outputs.
int rj_arith_s_xor_saveexec_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                                uint32_t exec_mask, uint32_t *exec, uint32_t *scc) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  uint32_t new_exec = exec_mask ^ a;
  return store_exec(flags, instruction_flags, d, exec_mask, new_exec, exec, scc);
}

// Set EXEC to ~(exec_mask & a) and save the incoming EXEC in D.
// Set SCC when the new EXEC is nonzero; errors preserve all outputs.
int rj_arith_s_nand_saveexec_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                                 uint32_t a, uint32_t exec_mask, uint32_t *exec, uint32_t *scc) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  uint32_t new_exec = ~(exec_mask & a);
  return store_exec(flags, instruction_flags, d, exec_mask, new_exec, exec, scc);
}

// Set EXEC to ~(exec_mask | a) and save the incoming EXEC in D.
// Set SCC when the new EXEC is nonzero; errors preserve all outputs.
int rj_arith_s_nor_saveexec_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                                uint32_t exec_mask, uint32_t *exec, uint32_t *scc) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  uint32_t new_exec = ~(exec_mask | a);
  return store_exec(flags, instruction_flags, d, exec_mask, new_exec, exec, scc);
}

// Set EXEC to ~(exec_mask ^ a) and save the incoming EXEC in D.
// Set SCC when the new EXEC is nonzero; errors preserve all outputs.
int rj_arith_s_xnor_saveexec_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                                 uint32_t a, uint32_t exec_mask, uint32_t *exec, uint32_t *scc) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  uint32_t new_exec = ~(exec_mask ^ a);
  return store_exec(flags, instruction_flags, d, exec_mask, new_exec, exec, scc);
}

// Set EXEC to exec_mask & ~a and save the incoming EXEC in D.
// Set SCC when the new EXEC is nonzero; errors preserve all outputs.
int rj_arith_s_and_not0_saveexec_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                                     uint32_t a, uint32_t exec_mask, uint32_t *exec,
                                     uint32_t *scc) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  uint32_t new_exec = exec_mask & ~a;
  return store_exec(flags, instruction_flags, d, exec_mask, new_exec, exec, scc);
}

// Set EXEC to exec_mask | ~a and save the incoming EXEC in D.
// Set SCC when the new EXEC is nonzero; errors preserve all outputs.
int rj_arith_s_or_not0_saveexec_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                                    uint32_t a, uint32_t exec_mask, uint32_t *exec, uint32_t *scc) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  uint32_t new_exec = exec_mask | ~a;
  return store_exec(flags, instruction_flags, d, exec_mask, new_exec, exec, scc);
}

// Set EXEC to a & ~exec_mask and save the incoming EXEC in D.
// Set SCC when the new EXEC is nonzero; errors preserve all outputs.
int rj_arith_s_and_not1_saveexec_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                                     uint32_t a, uint32_t exec_mask, uint32_t *exec,
                                     uint32_t *scc) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  uint32_t new_exec = a & ~exec_mask;
  return store_exec(flags, instruction_flags, d, exec_mask, new_exec, exec, scc);
}

// Set EXEC to a | ~exec_mask and save the incoming EXEC in D.
// Set SCC when the new EXEC is nonzero; errors preserve all outputs.
int rj_arith_s_or_not1_saveexec_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                                    uint32_t a, uint32_t exec_mask, uint32_t *exec, uint32_t *scc) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  uint32_t new_exec = a | ~exec_mask;
  return store_exec(flags, instruction_flags, d, exec_mask, new_exec, exec, scc);
}

// Set EXEC to exec_mask & ~a and save the updated EXEC in D.
// Set SCC when the new EXEC is nonzero; errors preserve all outputs.
int rj_arith_s_and_not0_wrexec_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                                   uint32_t a, uint32_t exec_mask, uint32_t *exec, uint32_t *scc) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  uint32_t new_exec = exec_mask & ~a;
  return store_exec(flags, instruction_flags, d, new_exec, new_exec, exec, scc);
}

// Set EXEC to a & ~exec_mask and save the updated EXEC in D.
// Set SCC when the new EXEC is nonzero; errors preserve all outputs.
int rj_arith_s_and_not1_wrexec_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                                   uint32_t a, uint32_t exec_mask, uint32_t *exec, uint32_t *scc) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  uint32_t new_exec = a & ~exec_mask;
  return store_exec(flags, instruction_flags, d, new_exec, new_exec, exec, scc);
}

// Set EXEC to exec_mask & a and save the incoming EXEC in D.
// Set SCC when the new EXEC is nonzero; errors preserve all outputs.
int rj_arith_s_and_saveexec_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                                uint64_t exec_mask, uint64_t *exec, uint32_t *scc) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  uint64_t new_exec = exec_mask & a;
  return store_exec(flags, instruction_flags, d, exec_mask, new_exec, exec, scc);
}

// Set EXEC to exec_mask | a and save the incoming EXEC in D.
// Set SCC when the new EXEC is nonzero; errors preserve all outputs.
int rj_arith_s_or_saveexec_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                               uint64_t exec_mask, uint64_t *exec, uint32_t *scc) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  uint64_t new_exec = exec_mask | a;
  return store_exec(flags, instruction_flags, d, exec_mask, new_exec, exec, scc);
}

// Set EXEC to exec_mask ^ a and save the incoming EXEC in D.
// Set SCC when the new EXEC is nonzero; errors preserve all outputs.
int rj_arith_s_xor_saveexec_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                                uint64_t exec_mask, uint64_t *exec, uint32_t *scc) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  uint64_t new_exec = exec_mask ^ a;
  return store_exec(flags, instruction_flags, d, exec_mask, new_exec, exec, scc);
}

// Set EXEC to ~(exec_mask & a) and save the incoming EXEC in D.
// Set SCC when the new EXEC is nonzero; errors preserve all outputs.
int rj_arith_s_nand_saveexec_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d,
                                 uint64_t a, uint64_t exec_mask, uint64_t *exec, uint32_t *scc) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  uint64_t new_exec = ~(exec_mask & a);
  return store_exec(flags, instruction_flags, d, exec_mask, new_exec, exec, scc);
}

// Set EXEC to ~(exec_mask | a) and save the incoming EXEC in D.
// Set SCC when the new EXEC is nonzero; errors preserve all outputs.
int rj_arith_s_nor_saveexec_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                                uint64_t exec_mask, uint64_t *exec, uint32_t *scc) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  uint64_t new_exec = ~(exec_mask | a);
  return store_exec(flags, instruction_flags, d, exec_mask, new_exec, exec, scc);
}

// Set EXEC to ~(exec_mask ^ a) and save the incoming EXEC in D.
// Set SCC when the new EXEC is nonzero; errors preserve all outputs.
int rj_arith_s_xnor_saveexec_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d,
                                 uint64_t a, uint64_t exec_mask, uint64_t *exec, uint32_t *scc) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  uint64_t new_exec = ~(exec_mask ^ a);
  return store_exec(flags, instruction_flags, d, exec_mask, new_exec, exec, scc);
}

// Set EXEC to exec_mask & ~a and save the incoming EXEC in D.
// Set SCC when the new EXEC is nonzero; errors preserve all outputs.
int rj_arith_s_and_not0_saveexec_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d,
                                     uint64_t a, uint64_t exec_mask, uint64_t *exec,
                                     uint32_t *scc) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  uint64_t new_exec = exec_mask & ~a;
  return store_exec(flags, instruction_flags, d, exec_mask, new_exec, exec, scc);
}

// Set EXEC to exec_mask | ~a and save the incoming EXEC in D.
// Set SCC when the new EXEC is nonzero; errors preserve all outputs.
int rj_arith_s_or_not0_saveexec_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d,
                                    uint64_t a, uint64_t exec_mask, uint64_t *exec, uint32_t *scc) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  uint64_t new_exec = exec_mask | ~a;
  return store_exec(flags, instruction_flags, d, exec_mask, new_exec, exec, scc);
}

// Set EXEC to a & ~exec_mask and save the incoming EXEC in D.
// Set SCC when the new EXEC is nonzero; errors preserve all outputs.
int rj_arith_s_and_not1_saveexec_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d,
                                     uint64_t a, uint64_t exec_mask, uint64_t *exec,
                                     uint32_t *scc) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  uint64_t new_exec = a & ~exec_mask;
  return store_exec(flags, instruction_flags, d, exec_mask, new_exec, exec, scc);
}

// Set EXEC to a | ~exec_mask and save the incoming EXEC in D.
// Set SCC when the new EXEC is nonzero; errors preserve all outputs.
int rj_arith_s_or_not1_saveexec_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d,
                                    uint64_t a, uint64_t exec_mask, uint64_t *exec, uint32_t *scc) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  uint64_t new_exec = a | ~exec_mask;
  return store_exec(flags, instruction_flags, d, exec_mask, new_exec, exec, scc);
}

// Set EXEC to exec_mask & ~a and save the updated EXEC in D.
// Set SCC when the new EXEC is nonzero; errors preserve all outputs.
int rj_arith_s_and_not0_wrexec_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d,
                                   uint64_t a, uint64_t exec_mask, uint64_t *exec, uint32_t *scc) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  uint64_t new_exec = exec_mask & ~a;
  return store_exec(flags, instruction_flags, d, new_exec, new_exec, exec, scc);
}

// Set EXEC to a & ~exec_mask and save the updated EXEC in D.
// Set SCC when the new EXEC is nonzero; errors preserve all outputs.
int rj_arith_s_and_not1_wrexec_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d,
                                   uint64_t a, uint64_t exec_mask, uint64_t *exec, uint32_t *scc) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  uint64_t new_exec = a & ~exec_mask;
  return store_exec(flags, instruction_flags, d, new_exec, new_exec, exec, scc);
}
