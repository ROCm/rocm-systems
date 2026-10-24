// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Scalar integer arithmetic and boolean operations with SCC updates.
// Owns public entry points and their validation/dispatch policy.

// Integer models adapted from rocjitsu generated/shared/execute_shared.h.
// RDNA3 ISA 16.1 and GFX1201 captures correct its ABSDIFF model: subtraction
// wraps before ABS. Both architectures have the same result and SCC rules.

#include "common/bits.h"
#include "common/internal.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

// Add the operands (S_ADD_CO_U32).
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_add_co_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                          uint32_t b, uint32_t *scc) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (int error = rocjitsu::arith::validate(flags, instruction_flags, true)) {
    return error;
  }
  uint64_t wide = uint64_t(a) + b;
  uint32_t result = uint32_t(wide), cc = wide >> 32;
  *d = result;
  *scc = cc;
  return RJ_ARITH_SUCCESS;
}

// Subtract B from A (S_SUB_CO_U32).
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_sub_co_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                          uint32_t b, uint32_t *scc) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (int error = rocjitsu::arith::validate(flags, instruction_flags, true)) {
    return error;
  }
  uint32_t result = a - b, cc = a < b;
  *d = result;
  *scc = cc;
  return RJ_ARITH_SUCCESS;
}

// Add the operands (S_ADD_CO_I32).
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_add_co_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                          uint32_t b, uint32_t *scc) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (int error = rocjitsu::arith::validate(flags, instruction_flags, true)) {
    return error;
  }
  uint32_t result = a + b, cc = (~(a ^ b) & (a ^ result)) >> 31;
  *d = result;
  *scc = cc;
  return RJ_ARITH_SUCCESS;
}

// Subtract B from A (S_SUB_CO_I32).
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_sub_co_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                          uint32_t b, uint32_t *scc) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (int error = rocjitsu::arith::validate(flags, instruction_flags, true)) {
    return error;
  }
  uint32_t result = a - b, cc = ((a ^ b) & (a ^ result)) >> 31;
  *d = result;
  *scc = cc;
  return RJ_ARITH_SUCCESS;
}

// Add the operands (S_ADD_CO_CI_U32).
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_add_co_ci_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                             uint32_t b, uint32_t *scc, uint32_t input_scc) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (int error = rocjitsu::arith::validate(flags, instruction_flags, true)) {
    return error;
  }
  uint64_t wide = uint64_t(a) + b + (input_scc & 1);
  uint32_t result = uint32_t(wide), cc = wide >> 32;
  *d = result;
  *scc = cc;
  return RJ_ARITH_SUCCESS;
}

// Subtract B from A (S_SUB_CO_CI_U32).
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_sub_co_ci_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                             uint32_t b, uint32_t *scc, uint32_t input_scc) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (int error = rocjitsu::arith::validate(flags, instruction_flags, true)) {
    return error;
  }
  uint64_t subtrahend = uint64_t(b) + (input_scc & 1);
  uint32_t result = a - uint32_t(subtrahend), cc = uint64_t(a) < subtrahend;
  *d = result;
  *scc = cc;
  return RJ_ARITH_SUCCESS;
}

// Compute the signed absolute value (S_ABS_I32).
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_abs_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                       uint32_t *scc) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (int error = rocjitsu::arith::validate(flags, instruction_flags, true)) {
    return error;
  }
  uint32_t result = a & 0x80000000u ? 0u - a : a, cc = result != 0;
  *d = result;
  *scc = cc;
  return RJ_ARITH_SUCCESS;
}

// Compute the absolute difference (S_ABSDIFF_I32).
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_absdiff_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                           uint32_t b, uint32_t *scc) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (int error = rocjitsu::arith::validate(flags, instruction_flags, true)) {
    return error;
  }
  uint32_t difference = a - b;
  uint32_t result = difference & 0x80000000u ? 0u - difference : difference, cc = result != 0;
  *d = result;
  *scc = cc;
  return RJ_ARITH_SUCCESS;
}

// Select the minimum (S_MIN_I32).
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_min_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                       uint32_t b, uint32_t *scc) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (int error = rocjitsu::arith::validate(flags, instruction_flags, true)) {
    return error;
  }
  uint32_t cc = (a ^ 0x80000000u) < (b ^ 0x80000000u), result = cc ? a : b;
  *d = result;
  *scc = cc;
  return RJ_ARITH_SUCCESS;
}

// Select the minimum (S_MIN_U32).
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_min_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                       uint32_t b, uint32_t *scc) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (int error = rocjitsu::arith::validate(flags, instruction_flags, true)) {
    return error;
  }
  uint32_t cc = a < b, result = cc ? a : b;
  *d = result;
  *scc = cc;
  return RJ_ARITH_SUCCESS;
}

// Select the maximum (S_MAX_I32).
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_max_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                       uint32_t b, uint32_t *scc) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (int error = rocjitsu::arith::validate(flags, instruction_flags, true)) {
    return error;
  }
  uint32_t cc = (a ^ 0x80000000u) > (b ^ 0x80000000u), result = cc ? a : b;
  *d = result;
  *scc = cc;
  return RJ_ARITH_SUCCESS;
}

// Select the maximum (S_MAX_U32).
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_max_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                       uint32_t b, uint32_t *scc) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (int error = rocjitsu::arith::validate(flags, instruction_flags, true)) {
    return error;
  }
  uint32_t cc = a > b, result = cc ? a : b;
  *d = result;
  *scc = cc;
  return RJ_ARITH_SUCCESS;
}

// Multiply the operands (S_MUL_I32).
// Return errors before changing outputs.
int rj_arith_s_mul_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                       uint32_t b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (int error = rocjitsu::arith::validate(flags, instruction_flags, true)) {
    return error;
  }
  uint32_t result = a * b;
  *d = result;
  return RJ_ARITH_SUCCESS;
}

// Multiply the operands (S_MUL_HI_U32).
// Return errors before changing outputs.
int rj_arith_s_mul_hi_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                          uint32_t b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (int error = rocjitsu::arith::validate(flags, instruction_flags, true)) {
    return error;
  }
  uint32_t result = (uint64_t(a) * b) >> 32;
  *d = result;
  return RJ_ARITH_SUCCESS;
}

// Multiply the operands (S_MUL_HI_I32).
// Return errors before changing outputs.
int rj_arith_s_mul_hi_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                          uint32_t b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (int error = rocjitsu::arith::validate(flags, instruction_flags, true)) {
    return error;
  }
  int64_t x = rocjitsu::arith::extend_integer<32, true>(a);
  int64_t y = rocjitsu::arith::extend_integer<32, true>(b);
  uint32_t result = uint64_t(x * y) >> 32;
  *d = result;
  return RJ_ARITH_SUCCESS;
}

// Add the operands (S_ADD_NC_U64).
// Return errors before changing outputs.
int rj_arith_s_add_nc_u64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                          uint64_t b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (int error = rocjitsu::arith::validate(flags, instruction_flags, true)) {
    return error;
  }
  uint64_t result = a + b;
  *d = result;
  return RJ_ARITH_SUCCESS;
}

// Subtract B from A (S_SUB_NC_U64).
// Return errors before changing outputs.
int rj_arith_s_sub_nc_u64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                          uint64_t b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (int error = rocjitsu::arith::validate(flags, instruction_flags, true)) {
    return error;
  }
  uint64_t result = a - b;
  *d = result;
  return RJ_ARITH_SUCCESS;
}

// Multiply the operands (S_MUL_U64).
// Return errors before changing outputs.
int rj_arith_s_mul_u64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                       uint64_t b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (int error = rocjitsu::arith::validate(flags, instruction_flags, true)) {
    return error;
  }
  uint64_t result = a * b;
  *d = result;
  return RJ_ARITH_SUCCESS;
}

// Execute ADDK_CO_I32 on scalar values.
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_addk_co_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                           uint16_t immediate, uint32_t *scc) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (int error = rocjitsu::arith::validate(flags, instruction_flags, true)) {
    return error;
  }
  uint32_t a = *d, b = uint32_t(immediate) | ((immediate & 0x8000) ? 0xffff0000u : 0);
  uint32_t result = a + b, cc = (~(a ^ b) & (a ^ result)) >> 31;
  *d = result;
  *scc = cc;
  return RJ_ARITH_SUCCESS;
}

// Execute MULK_I32 on scalar values.
// Return errors before changing outputs.
int rj_arith_s_mulk_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                        uint16_t immediate) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (int error = rocjitsu::arith::validate(flags, instruction_flags, true)) {
    return error;
  }
  uint32_t b = uint32_t(immediate) | ((immediate & 0x8000) ? 0xffff0000u : 0);
  uint32_t result = *d * b;
  *d = result;
  return RJ_ARITH_SUCCESS;
}

// Execute SEXT_I32_I8 on scalar values.
// Return errors before changing outputs.
int rj_arith_s_sext_i32_i8(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (int error = rocjitsu::arith::validate(flags, instruction_flags, true)) {
    return error;
  }
  *d = rocjitsu::arith::sign_extend_word<8>(a);
  return RJ_ARITH_SUCCESS;
}

// Execute SEXT_I32_I16 on scalar values.
// Return errors before changing outputs.
int rj_arith_s_sext_i32_i16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (int error = rocjitsu::arith::validate(flags, instruction_flags, true)) {
    return error;
  }
  *d = rocjitsu::arith::sign_extend_word<16>(a);
  return RJ_ARITH_SUCCESS;
}

// RDNA3 spellings share the RDNA4 carry/overflow implementations.
int rj_arith_s_add_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                       uint32_t b, uint32_t *scc) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return rj_arith_s_add_co_u32(flags, instruction_flags, d, a, b, scc);
}

// Subtract B from A (S_SUB_U32).
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_sub_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                       uint32_t b, uint32_t *scc) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return rj_arith_s_sub_co_u32(flags, instruction_flags, d, a, b, scc);
}

// Add the operands (S_ADD_I32).
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_add_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                       uint32_t b, uint32_t *scc) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return rj_arith_s_add_co_i32(flags, instruction_flags, d, a, b, scc);
}

// Subtract B from A (S_SUB_I32).
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_sub_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                       uint32_t b, uint32_t *scc) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return rj_arith_s_sub_co_i32(flags, instruction_flags, d, a, b, scc);
}

// Execute ADDC_U32 on scalar values.
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_addc_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                        uint32_t b, uint32_t input_scc, uint32_t *scc) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return rj_arith_s_add_co_ci_u32(flags, instruction_flags, d, a, b, scc, input_scc);
}

// Execute SUBB_U32 on scalar values.
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_subb_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                        uint32_t b, uint32_t input_scc, uint32_t *scc) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return rj_arith_s_sub_co_ci_u32(flags, instruction_flags, d, a, b, scc, input_scc);
}

// Execute ADDK_I32 on scalar values.
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_addk_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint16_t immediate,
                        uint32_t *scc) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return rj_arith_s_addk_co_i32(flags, instruction_flags, d, immediate, scc);
}
