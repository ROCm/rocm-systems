// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Scalar integer/FP comparisons and bit tests producing SCC.
// Owns public entry points and their validation/dispatch policy.

// Raw float ordering follows rocjitsu shared/comparison.h and reuses the vector
// predicate helpers. Integer and bit tests follow generated/execute_shared.h.

#include "common/internal.h"
#include "compare/float_compare_impl.h"
#include "compare/integer_compare_impl.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace {

// Validate the instruction's accepted flags before operand access or output writes.
int validate(uint64_t flags, uint32_t mode) {
  return rocjitsu::arith::validate(
      flags, mode, RJ_ARITH_FP_FLUSH_INPUT_DENORMALS | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS);
}

// Byte stores also allow SCC to overlap a word of a caller's uint64_t storage.
void store(uint32_t *scc, uint32_t result) { __builtin_memcpy(scc, &result, sizeof(result)); }

// Evaluate the scalar integer predicate and write SCC; errors preserve SCC.
template <bool Signed, unsigned Predicate, typename T>
int integer(uint64_t flags, uint32_t mode, uint32_t *scc, T a, T b) {
  if (int error = validate(flags, mode)) {
    return error;
  }
  if constexpr (Signed) {
    a ^= T(1) << (sizeof(T) * 8 - 1);
    b ^= T(1) << (sizeof(T) * 8 - 1);
  }
  store(scc, rocjitsu::arith::integer_compare_result<Predicate>(a < b, a == b) & 1);
  return RJ_ARITH_SUCCESS;
}

// Test the selected bit against BitValue and write SCC; errors preserve SCC.
template <unsigned BitValue, typename T>
int bit(uint64_t flags, uint32_t mode, uint32_t *scc, T a, uint32_t b) {
  if (int error = validate(flags, mode)) {
    return error;
  }
  store(scc, ((a >> (b & (sizeof(T) * 8 - 1))) & 1) == BitValue);
  return RJ_ARITH_SUCCESS;
}

// Evaluate the scalar FP predicate into SCC and accumulate requested FP causes; errors
// preserve outputs.
template <int Bits, unsigned Predicate>
int floating(uint64_t flags, uint32_t mode, uint32_t *scc, uint32_t a, uint32_t b,
             uint32_t *excp_flag_user) {
  if (int error = validate(flags, mode)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    *excp_flag_user |= rocjitsu::arith::float_compare_exceptions<Bits>(
        a, b, flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS);
  }
  constexpr uint32_t sign = 1u << (Bits - 1), magnitude = sign - 1,
                     infinity = Bits == 16 ? 0x7c00 : 0x7f800000;
  if constexpr (Bits == 16) {
    a &= 65535;
    b &= 65535;
  }
  if (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) {
    if (!(a & infinity)) {
      a &= sign;
    }
    if (!(b & infinity)) {
      b &= sign;
    }
  }
  bool ordered = (a & magnitude) <= infinity && (b & magnitude) <= infinity;
  if (!(a & magnitude)) {
    a = 0;
  }
  if (!(b & magnitude)) {
    b = 0;
  }
  a ^= a & sign ? (sign | magnitude) : sign;
  b ^= b & sign ? (sign | magnitude) : sign;
  store(scc, rocjitsu::arith::float_compare_result<Predicate>(a < b, a == b, ordered) & 1);
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Execute CMP_EQ_I32 on scalar values.
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_cmp_eq_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                          uint32_t b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return integer<true, 2>(flags, instruction_flags, scc, a, b);
}

// Execute CMP_LG_I32 on scalar values.
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_cmp_lg_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                          uint32_t b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return integer<true, 5>(flags, instruction_flags, scc, a, b);
}

// Execute CMP_GT_I32 on scalar values.
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_cmp_gt_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                          uint32_t b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return integer<true, 4>(flags, instruction_flags, scc, a, b);
}

// Execute CMP_GE_I32 on scalar values.
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_cmp_ge_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                          uint32_t b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return integer<true, 6>(flags, instruction_flags, scc, a, b);
}

// Execute CMP_LT_I32 on scalar values.
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_cmp_lt_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                          uint32_t b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return integer<true, 1>(flags, instruction_flags, scc, a, b);
}

// Execute CMP_LE_I32 on scalar values.
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_cmp_le_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                          uint32_t b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return integer<true, 3>(flags, instruction_flags, scc, a, b);
}

// Execute CMP_EQ_U32 on scalar values.
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_cmp_eq_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                          uint32_t b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return integer<false, 2>(flags, instruction_flags, scc, a, b);
}

// Execute CMP_LG_U32 on scalar values.
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_cmp_lg_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                          uint32_t b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return integer<false, 5>(flags, instruction_flags, scc, a, b);
}

// Execute CMP_GT_U32 on scalar values.
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_cmp_gt_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                          uint32_t b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return integer<false, 4>(flags, instruction_flags, scc, a, b);
}

// Execute CMP_GE_U32 on scalar values.
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_cmp_ge_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                          uint32_t b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return integer<false, 6>(flags, instruction_flags, scc, a, b);
}

// Execute CMP_LT_U32 on scalar values.
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_cmp_lt_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                          uint32_t b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return integer<false, 1>(flags, instruction_flags, scc, a, b);
}

// Execute CMP_LE_U32 on scalar values.
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_cmp_le_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                          uint32_t b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return integer<false, 3>(flags, instruction_flags, scc, a, b);
}

// Test whether the selected bit is clear.
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_bitcmp0_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                           uint32_t b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return bit<0>(flags, instruction_flags, scc, a, b);
}

// Test whether the selected bit is set.
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_bitcmp1_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                           uint32_t b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return bit<1>(flags, instruction_flags, scc, a, b);
}

// Test whether the selected bit is clear.
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_bitcmp0_b64(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint64_t a,
                           uint32_t b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return bit<0>(flags, instruction_flags, scc, a, b);
}

// Test whether the selected bit is set.
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_bitcmp1_b64(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint64_t a,
                           uint32_t b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return bit<1>(flags, instruction_flags, scc, a, b);
}

// Execute CMP_EQ_U64 on scalar values.
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_cmp_eq_u64(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint64_t a,
                          uint64_t b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return integer<false, 2>(flags, instruction_flags, scc, a, b);
}

// Execute CMP_LG_U64 on scalar values.
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_cmp_lg_u64(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint64_t a,
                          uint64_t b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return integer<false, 5>(flags, instruction_flags, scc, a, b);
}

// Execute CMP_LT_F32 on scalar values.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_cmp_lt_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                          uint32_t b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return floating<32, 1>(flags, instruction_flags, scc, a, b, excp_flag_user);
}

// Execute CMP_LT_F16 on scalar values.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_cmp_lt_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                          uint32_t b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return floating<16, 1>(flags, instruction_flags, scc, a, b, excp_flag_user);
}

// Execute CMP_EQ_F32 on scalar values.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_cmp_eq_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                          uint32_t b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return floating<32, 2>(flags, instruction_flags, scc, a, b, excp_flag_user);
}

// Execute CMP_EQ_F16 on scalar values.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_cmp_eq_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                          uint32_t b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return floating<16, 2>(flags, instruction_flags, scc, a, b, excp_flag_user);
}

// Execute CMP_LE_F32 on scalar values.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_cmp_le_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                          uint32_t b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return floating<32, 3>(flags, instruction_flags, scc, a, b, excp_flag_user);
}

// Execute CMP_LE_F16 on scalar values.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_cmp_le_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                          uint32_t b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return floating<16, 3>(flags, instruction_flags, scc, a, b, excp_flag_user);
}

// Execute CMP_GT_F32 on scalar values.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_cmp_gt_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                          uint32_t b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return floating<32, 4>(flags, instruction_flags, scc, a, b, excp_flag_user);
}

// Execute CMP_GT_F16 on scalar values.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_cmp_gt_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                          uint32_t b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return floating<16, 4>(flags, instruction_flags, scc, a, b, excp_flag_user);
}

// Execute CMP_LG_F32 on scalar values.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_cmp_lg_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                          uint32_t b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return floating<32, 5>(flags, instruction_flags, scc, a, b, excp_flag_user);
}

// Execute CMP_LG_F16 on scalar values.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_cmp_lg_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                          uint32_t b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return floating<16, 5>(flags, instruction_flags, scc, a, b, excp_flag_user);
}

// Execute CMP_GE_F32 on scalar values.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_cmp_ge_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                          uint32_t b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return floating<32, 6>(flags, instruction_flags, scc, a, b, excp_flag_user);
}

// Execute CMP_GE_F16 on scalar values.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_cmp_ge_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                          uint32_t b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return floating<16, 6>(flags, instruction_flags, scc, a, b, excp_flag_user);
}

// Execute CMP_O_F32 on scalar values.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_cmp_o_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                         uint32_t b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return floating<32, 7>(flags, instruction_flags, scc, a, b, excp_flag_user);
}

// Execute CMP_O_F16 on scalar values.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_cmp_o_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                         uint32_t b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return floating<16, 7>(flags, instruction_flags, scc, a, b, excp_flag_user);
}

// Execute CMP_U_F32 on scalar values.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_cmp_u_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                         uint32_t b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return floating<32, 8>(flags, instruction_flags, scc, a, b, excp_flag_user);
}

// Execute CMP_U_F16 on scalar values.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_cmp_u_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                         uint32_t b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return floating<16, 8>(flags, instruction_flags, scc, a, b, excp_flag_user);
}

// Execute CMP_NGE_F32 on scalar values.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_cmp_nge_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                           uint32_t b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return floating<32, 9>(flags, instruction_flags, scc, a, b, excp_flag_user);
}

// Execute CMP_NGE_F16 on scalar values.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_cmp_nge_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                           uint32_t b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return floating<16, 9>(flags, instruction_flags, scc, a, b, excp_flag_user);
}

// Execute CMP_NLG_F32 on scalar values.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_cmp_nlg_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                           uint32_t b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return floating<32, 10>(flags, instruction_flags, scc, a, b, excp_flag_user);
}

// Execute CMP_NLG_F16 on scalar values.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_cmp_nlg_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                           uint32_t b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return floating<16, 10>(flags, instruction_flags, scc, a, b, excp_flag_user);
}

// Execute CMP_NGT_F32 on scalar values.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_cmp_ngt_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                           uint32_t b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return floating<32, 11>(flags, instruction_flags, scc, a, b, excp_flag_user);
}

// Execute CMP_NGT_F16 on scalar values.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_cmp_ngt_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                           uint32_t b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return floating<16, 11>(flags, instruction_flags, scc, a, b, excp_flag_user);
}

// Execute CMP_NLE_F32 on scalar values.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_cmp_nle_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                           uint32_t b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return floating<32, 12>(flags, instruction_flags, scc, a, b, excp_flag_user);
}

// Execute CMP_NLE_F16 on scalar values.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_cmp_nle_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                           uint32_t b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return floating<16, 12>(flags, instruction_flags, scc, a, b, excp_flag_user);
}

// Execute CMP_NEQ_F32 on scalar values.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_cmp_neq_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                           uint32_t b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return floating<32, 13>(flags, instruction_flags, scc, a, b, excp_flag_user);
}

// Execute CMP_NEQ_F16 on scalar values.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_cmp_neq_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                           uint32_t b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return floating<16, 13>(flags, instruction_flags, scc, a, b, excp_flag_user);
}

// Execute CMP_NLT_F32 on scalar values.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_cmp_nlt_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                           uint32_t b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return floating<32, 14>(flags, instruction_flags, scc, a, b, excp_flag_user);
}

// Execute CMP_NLT_F16 on scalar values.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_cmp_nlt_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                           uint32_t b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return floating<16, 14>(flags, instruction_flags, scc, a, b, excp_flag_user);
}

// RDNA3 SOPK immediate comparisons: signed extends, unsigned zero-extends.
int rj_arith_s_cmpk_eq_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                           uint16_t immediate) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return rj_arith_s_cmp_eq_i32(flags, instruction_flags, scc, a,
                               uint32_t(immediate) | ((immediate & 0x8000U) ? 0xffff0000U : 0));
}

// Execute CMPK_LG_I32 on scalar values.
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_cmpk_lg_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                           uint16_t immediate) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return rj_arith_s_cmp_lg_i32(flags, instruction_flags, scc, a,
                               uint32_t(immediate) | ((immediate & 0x8000U) ? 0xffff0000U : 0));
}

// Execute CMPK_GT_I32 on scalar values.
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_cmpk_gt_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                           uint16_t immediate) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return rj_arith_s_cmp_gt_i32(flags, instruction_flags, scc, a,
                               uint32_t(immediate) | ((immediate & 0x8000U) ? 0xffff0000U : 0));
}

// Execute CMPK_GE_I32 on scalar values.
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_cmpk_ge_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                           uint16_t immediate) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return rj_arith_s_cmp_ge_i32(flags, instruction_flags, scc, a,
                               uint32_t(immediate) | ((immediate & 0x8000U) ? 0xffff0000U : 0));
}

// Execute CMPK_LT_I32 on scalar values.
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_cmpk_lt_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                           uint16_t immediate) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return rj_arith_s_cmp_lt_i32(flags, instruction_flags, scc, a,
                               uint32_t(immediate) | ((immediate & 0x8000U) ? 0xffff0000U : 0));
}

// Execute CMPK_LE_I32 on scalar values.
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_cmpk_le_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                           uint16_t immediate) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return rj_arith_s_cmp_le_i32(flags, instruction_flags, scc, a,
                               uint32_t(immediate) | ((immediate & 0x8000U) ? 0xffff0000U : 0));
}

// Execute CMPK_EQ_U32 on scalar values.
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_cmpk_eq_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                           uint16_t immediate) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return rj_arith_s_cmp_eq_u32(flags, instruction_flags, scc, a, uint32_t(immediate));
}

// Execute CMPK_LG_U32 on scalar values.
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_cmpk_lg_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                           uint16_t immediate) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return rj_arith_s_cmp_lg_u32(flags, instruction_flags, scc, a, uint32_t(immediate));
}

// Execute CMPK_GT_U32 on scalar values.
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_cmpk_gt_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                           uint16_t immediate) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return rj_arith_s_cmp_gt_u32(flags, instruction_flags, scc, a, uint32_t(immediate));
}

// Execute CMPK_GE_U32 on scalar values.
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_cmpk_ge_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                           uint16_t immediate) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return rj_arith_s_cmp_ge_u32(flags, instruction_flags, scc, a, uint32_t(immediate));
}

// Execute CMPK_LT_U32 on scalar values.
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_cmpk_lt_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                           uint16_t immediate) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return rj_arith_s_cmp_lt_u32(flags, instruction_flags, scc, a, uint32_t(immediate));
}

// Execute CMPK_LE_U32 on scalar values.
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_cmpk_le_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                           uint16_t immediate) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return rj_arith_s_cmp_le_u32(flags, instruction_flags, scc, a, uint32_t(immediate));
}
