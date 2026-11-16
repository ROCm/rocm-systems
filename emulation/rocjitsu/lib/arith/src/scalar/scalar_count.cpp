// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Scalar population counts and leading/trailing/redundant-sign bit counts.
// Owns public entry points and their validation/dispatch policy.

// Models follow rocjitsu generated/shared/execute_shared.h, verified on GFX1201.

#include "common/bits.h"
#include "common/validation.h"
#include "rj_arith/detail/rj_arith_scalar.h"

#include <stdint.h>

// Count zero bits.
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_bcnt0_i32_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
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
  uint32_t result = rocjitsu::arith::bit_population(uint32_t(~a));
  *d = result;
  uint32_t cc = result != 0;
  __builtin_memcpy(scc, &cc, sizeof(cc));
  return RJ_ARITH_SUCCESS;
}

// Count zero bits.
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_bcnt0_i32_b64(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint64_t a,
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
  uint32_t result = rocjitsu::arith::bit_population(uint64_t(~a));
  *d = result;
  uint32_t cc = result != 0;
  __builtin_memcpy(scc, &cc, sizeof(cc));
  return RJ_ARITH_SUCCESS;
}

// Count set bits.
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_bcnt1_i32_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
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
  uint32_t result = rocjitsu::arith::bit_population(uint32_t(a));
  *d = result;
  uint32_t cc = result != 0;
  __builtin_memcpy(scc, &cc, sizeof(cc));
  return RJ_ARITH_SUCCESS;
}

// Count set bits.
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_bcnt1_i32_b64(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint64_t a,
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
  uint32_t result = rocjitsu::arith::bit_population(uint64_t(a));
  *d = result;
  uint32_t cc = result != 0;
  __builtin_memcpy(scc, &cc, sizeof(cc));
  return RJ_ARITH_SUCCESS;
}

// Count trailing zeros.
// Return errors before changing outputs.
int rj_arith_s_ctz_i32_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a) {
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
  uint32_t result = rocjitsu::arith::bit_count_zero<true>(a);
  *d = result;
  return RJ_ARITH_SUCCESS;
}

// Count trailing zeros.
// Return errors before changing outputs.
int rj_arith_s_ctz_i32_b64(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint64_t a) {
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
  uint32_t result = rocjitsu::arith::bit_count_zero<true>(a);
  *d = result;
  return RJ_ARITH_SUCCESS;
}

// Count leading zeros.
// Return errors before changing outputs.
int rj_arith_s_clz_i32_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a) {
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
  uint32_t result = rocjitsu::arith::bit_count_zero<false>(a);
  *d = result;
  return RJ_ARITH_SUCCESS;
}

// Count leading zeros.
// Return errors before changing outputs.
int rj_arith_s_clz_i32_u64(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint64_t a) {
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
  uint32_t result = rocjitsu::arith::bit_count_zero<false>(a);
  *d = result;
  return RJ_ARITH_SUCCESS;
}

// Count redundant sign bits.
// Return errors before changing outputs.
int rj_arith_s_cls_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a) {
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
  a ^= uint32_t(0) - (a >> 31);
  uint32_t result = rocjitsu::arith::bit_count_zero<false>(a);
  *d = result;
  return RJ_ARITH_SUCCESS;
}

// Count redundant sign bits.
// Return errors before changing outputs.
int rj_arith_s_cls_i32_i64(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint64_t a) {
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
  a ^= uint64_t(0) - (a >> 63);
  uint32_t result = rocjitsu::arith::bit_count_zero<false>(a);
  *d = result;
  return RJ_ARITH_SUCCESS;
}

// Find the least-significant zero bit, with the same validation as CTZ.
int rj_arith_s_ff0_i32_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a) {
  return rj_arith_s_ctz_i32_b32(flags, instruction_flags, d, ~a);
}

// Find the least-significant zero bit, with the same validation as CTZ.
int rj_arith_s_ff0_i32_b64(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint64_t a) {
  return rj_arith_s_ctz_i32_b64(flags, instruction_flags, d, ~a);
}

// Search bits using the legacy mnemonic; same result and validation as s_ctz_i32_b32.
int rj_arith_s_ff1_i32_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a) {
  return rj_arith_s_ctz_i32_b32(flags, instruction_flags, d, a);
}

// Search bits using the legacy mnemonic; same result and validation as s_ctz_i32_b64.
int rj_arith_s_ff1_i32_b64(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint64_t a) {
  return rj_arith_s_ctz_i32_b64(flags, instruction_flags, d, a);
}

// Search bits using the legacy mnemonic; same result and validation as s_clz_i32_u32.
int rj_arith_s_flbit_i32_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a) {
  return rj_arith_s_clz_i32_u32(flags, instruction_flags, d, a);
}

// Search bits using the legacy mnemonic; same result and validation as s_clz_i32_u64.
int rj_arith_s_flbit_i32_b64(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint64_t a) {
  return rj_arith_s_clz_i32_u64(flags, instruction_flags, d, a);
}

// Search bits using the legacy mnemonic; same result and validation as s_cls_i32.
int rj_arith_s_flbit_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a) {
  return rj_arith_s_cls_i32(flags, instruction_flags, d, a);
}

// Search bits using the legacy mnemonic; same result and validation as s_cls_i32_i64.
int rj_arith_s_flbit_i32_i64(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint64_t a) {
  return rj_arith_s_cls_i32_i64(flags, instruction_flags, d, a);
}
