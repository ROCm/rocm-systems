// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Scalar integral rounding and ties-to-even FP rounding.
// Owns public entry points and their validation/dispatch policy.

// Integer rounding extends rocjitsu util::rndne_scalar to directed rounding.
// GFX1201 verifies all FP16 patterns and raw FP32 boundary/random values.

#include "common/internal.h"
#include "common/round_integral.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace {

using rocjitsu::arith::IntegralRound;

// Round a scalar FP value. Reject invalid controls before changing outputs.
template <IntegralRound Op, bool Half>
int run(uint64_t flags, uint32_t mode, uint32_t *d, uint32_t a, uint32_t *excp_flag_user) {
  if (int error = rocjitsu::arith::validate(
          flags, mode, RJ_ARITH_FP_FLUSH_INPUT_DENORMALS | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
    return error;
  }
  constexpr unsigned fraction = Half ? 10 : 23;
  constexpr uint32_t sign_bit = Half ? 0x8000 : 0x80000000, inf = Half ? 0x7c00 : 0x7f800000;
  if constexpr (Half) {
    a &= 65535;
  }

  // GFX1201: rounding instructions report input classification, not INEXACT
  // when the fractional part is discarded. Loose mode skips this work.
  if (rocjitsu::arith::reports_fp_status(flags)) {
    const uint32_t magnitude = a & (sign_bit - 1);
    if (magnitude > inf && !(magnitude & (1U << (fraction - 1)))) {
      *excp_flag_user |= RJ_ARITH_EXCEPTION_INVALID;
    } else if (!(flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) && magnitude && !(magnitude & inf)) {
      *excp_flag_user |= RJ_ARITH_EXCEPTION_INPUT_DENORM;
    }
  }
  *d = rocjitsu::arith::round_integral_bits<Op, Half>(a, flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS);
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Round toward positive infinity.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_ceil_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                        uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run<IntegralRound::Ceil, false>(flags, instruction_flags, d, a, excp_flag_user);
}

// Round toward positive infinity.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_ceil_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                        uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run<IntegralRound::Ceil, true>(flags, instruction_flags, d, a, excp_flag_user);
}

// Round toward negative infinity.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_floor_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                         uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run<IntegralRound::Floor, false>(flags, instruction_flags, d, a, excp_flag_user);
}

// Round toward negative infinity.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_floor_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                         uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run<IntegralRound::Floor, true>(flags, instruction_flags, d, a, excp_flag_user);
}

// Round toward zero.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_trunc_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                         uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run<IntegralRound::Trunc, false>(flags, instruction_flags, d, a, excp_flag_user);
}

// Round toward zero.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_trunc_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                         uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run<IntegralRound::Trunc, true>(flags, instruction_flags, d, a, excp_flag_user);
}

// Round to the nearest integer with ties to even.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_rndne_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                         uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run<IntegralRound::Nearest, false>(flags, instruction_flags, d, a, excp_flag_user);
}

// Round to the nearest integer with ties to even.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_rndne_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                         uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run<IntegralRound::Nearest, true>(flags, instruction_flags, d, a, excp_flag_user);
}
