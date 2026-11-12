// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Scalar boolean operations and bit reversal.
// Owns public entry points and their validation/dispatch policy.

// Models follow rocjitsu generated/shared/execute_shared.h, verified on GFX1201.

#include "common/bits.h"
#include "common/internal.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

// Compute bitwise AND.
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_and_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                       uint32_t b, uint32_t *scc) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (int error = rocjitsu::arith::validate(flags, instruction_flags)) {
    return error;
  }
  uint32_t result = a & b;
  uint32_t cc = result != 0;
  *d = result;
  __builtin_memcpy(scc, &cc, sizeof(cc));
  return RJ_ARITH_SUCCESS;
}

// Compute bitwise AND.
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_and_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                       uint64_t b, uint32_t *scc) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (int error = rocjitsu::arith::validate(flags, instruction_flags)) {
    return error;
  }
  uint64_t result = a & b;
  uint32_t cc = result != 0;
  *d = result;
  __builtin_memcpy(scc, &cc, sizeof(cc));
  return RJ_ARITH_SUCCESS;
}

// Compute bitwise OR.
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_or_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                      uint32_t b, uint32_t *scc) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (int error = rocjitsu::arith::validate(flags, instruction_flags)) {
    return error;
  }
  uint32_t result = a | b;
  uint32_t cc = result != 0;
  *d = result;
  __builtin_memcpy(scc, &cc, sizeof(cc));
  return RJ_ARITH_SUCCESS;
}

// Compute bitwise OR.
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_or_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                      uint64_t b, uint32_t *scc) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (int error = rocjitsu::arith::validate(flags, instruction_flags)) {
    return error;
  }
  uint64_t result = a | b;
  uint32_t cc = result != 0;
  *d = result;
  __builtin_memcpy(scc, &cc, sizeof(cc));
  return RJ_ARITH_SUCCESS;
}

// Compute bitwise XOR.
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_xor_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                       uint32_t b, uint32_t *scc) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (int error = rocjitsu::arith::validate(flags, instruction_flags)) {
    return error;
  }
  uint32_t result = a ^ b;
  uint32_t cc = result != 0;
  *d = result;
  __builtin_memcpy(scc, &cc, sizeof(cc));
  return RJ_ARITH_SUCCESS;
}

// Compute bitwise XOR.
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_xor_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                       uint64_t b, uint32_t *scc) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (int error = rocjitsu::arith::validate(flags, instruction_flags)) {
    return error;
  }
  uint64_t result = a ^ b;
  uint32_t cc = result != 0;
  *d = result;
  __builtin_memcpy(scc, &cc, sizeof(cc));
  return RJ_ARITH_SUCCESS;
}

// Complement bitwise AND.
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_nand_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                        uint32_t b, uint32_t *scc) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (int error = rocjitsu::arith::validate(flags, instruction_flags)) {
    return error;
  }
  uint32_t result = ~(a & b);
  uint32_t cc = result != 0;
  *d = result;
  __builtin_memcpy(scc, &cc, sizeof(cc));
  return RJ_ARITH_SUCCESS;
}

// Complement bitwise AND.
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_nand_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                        uint64_t b, uint32_t *scc) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (int error = rocjitsu::arith::validate(flags, instruction_flags)) {
    return error;
  }
  uint64_t result = ~(a & b);
  uint32_t cc = result != 0;
  *d = result;
  __builtin_memcpy(scc, &cc, sizeof(cc));
  return RJ_ARITH_SUCCESS;
}

// Complement bitwise OR.
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_nor_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                       uint32_t b, uint32_t *scc) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (int error = rocjitsu::arith::validate(flags, instruction_flags)) {
    return error;
  }
  uint32_t result = ~(a | b);
  uint32_t cc = result != 0;
  *d = result;
  __builtin_memcpy(scc, &cc, sizeof(cc));
  return RJ_ARITH_SUCCESS;
}

// Complement bitwise OR.
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_nor_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                       uint64_t b, uint32_t *scc) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (int error = rocjitsu::arith::validate(flags, instruction_flags)) {
    return error;
  }
  uint64_t result = ~(a | b);
  uint32_t cc = result != 0;
  *d = result;
  __builtin_memcpy(scc, &cc, sizeof(cc));
  return RJ_ARITH_SUCCESS;
}

// Compute bitwise equivalence.
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_xnor_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                        uint32_t b, uint32_t *scc) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (int error = rocjitsu::arith::validate(flags, instruction_flags)) {
    return error;
  }
  uint32_t result = ~(a ^ b);
  uint32_t cc = result != 0;
  *d = result;
  __builtin_memcpy(scc, &cc, sizeof(cc));
  return RJ_ARITH_SUCCESS;
}

// Compute bitwise equivalence.
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_xnor_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                        uint64_t b, uint32_t *scc) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (int error = rocjitsu::arith::validate(flags, instruction_flags)) {
    return error;
  }
  uint64_t result = ~(a ^ b);
  uint32_t cc = result != 0;
  *d = result;
  __builtin_memcpy(scc, &cc, sizeof(cc));
  return RJ_ARITH_SUCCESS;
}

// Compute bitwise AND.
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_and_not1_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                            uint32_t b, uint32_t *scc) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (int error = rocjitsu::arith::validate(flags, instruction_flags)) {
    return error;
  }
  uint32_t result = a & ~b;
  uint32_t cc = result != 0;
  *d = result;
  __builtin_memcpy(scc, &cc, sizeof(cc));
  return RJ_ARITH_SUCCESS;
}

// Compute bitwise AND.
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_and_not1_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                            uint64_t b, uint32_t *scc) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (int error = rocjitsu::arith::validate(flags, instruction_flags)) {
    return error;
  }
  uint64_t result = a & ~b;
  uint32_t cc = result != 0;
  *d = result;
  __builtin_memcpy(scc, &cc, sizeof(cc));
  return RJ_ARITH_SUCCESS;
}

// Compute bitwise OR.
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_or_not1_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                           uint32_t b, uint32_t *scc) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (int error = rocjitsu::arith::validate(flags, instruction_flags)) {
    return error;
  }
  uint32_t result = a | ~b;
  uint32_t cc = result != 0;
  *d = result;
  __builtin_memcpy(scc, &cc, sizeof(cc));
  return RJ_ARITH_SUCCESS;
}

// Compute bitwise OR.
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_or_not1_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                           uint64_t b, uint32_t *scc) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (int error = rocjitsu::arith::validate(flags, instruction_flags)) {
    return error;
  }
  uint64_t result = a | ~b;
  uint32_t cc = result != 0;
  *d = result;
  __builtin_memcpy(scc, &cc, sizeof(cc));
  return RJ_ARITH_SUCCESS;
}

// Complement the source bits.
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_not_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                       uint32_t *scc) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (int error = rocjitsu::arith::validate(flags, instruction_flags)) {
    return error;
  }
  uint32_t result = ~a;
  uint32_t cc = result != 0;
  *d = result;
  __builtin_memcpy(scc, &cc, sizeof(cc));
  return RJ_ARITH_SUCCESS;
}

// Complement the source bits.
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_not_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                       uint32_t *scc) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (int error = rocjitsu::arith::validate(flags, instruction_flags)) {
    return error;
  }
  uint64_t result = ~a;
  uint32_t cc = result != 0;
  *d = result;
  __builtin_memcpy(scc, &cc, sizeof(cc));
  return RJ_ARITH_SUCCESS;
}

// Reverse the bits.
// Return errors before changing outputs.
int rj_arith_s_brev_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (int error = rocjitsu::arith::validate(flags, instruction_flags)) {
    return error;
  }
  uint32_t result = rocjitsu::arith::bit_reverse(a);
  *d = result;
  return RJ_ARITH_SUCCESS;
}

// Reverse the bits.
// Return errors before changing outputs.
int rj_arith_s_brev_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (int error = rocjitsu::arith::validate(flags, instruction_flags)) {
    return error;
  }
  uint64_t result = rocjitsu::arith::bit_reverse(a);
  *d = result;
  return RJ_ARITH_SUCCESS;
}
