// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Scalar floating-point arithmetic and selection, with optional FP status.
// Owns public entry points and their validation/dispatch policy.

// Scalar operations follow rocjitsu generated/shared/execute_shared.h. GFX1201
// captures establish the output-flush exceptions and multiplication threshold.

#include "common/float_formats.h"
#include "common/internal.h"
#include "fp/binary.h"
#include "fp/minmax.h"
#include "rj_arith/rj_arith.h"

#include <cmath>
#include <stdint.h>

namespace {

// Evaluate scalar FP arithmetic. Reject invalid controls before changing outputs.
template <rocjitsu::arith::Binary Op, bool Half>
int run(uint64_t flags, uint32_t mode, uint32_t *d, uint32_t a, uint32_t b) {
  if (int error = rocjitsu::arith::validate(flags, mode, false,
                                            RJ_ARITH_FP_FLUSH_INPUT_DENORMALS |
                                                RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
    return error;
  }
  constexpr uint32_t inf = Half ? 0x7c00 : 0x7f800000, sign = Half ? 0x8000 : 0x80000000;
  if constexpr (Half) {
    a &= 65535;
    b &= 65535;
  }
  if (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) {
    if (!(a & inf)) {
      a &= sign;
    }
    if (!(b & inf)) {
      b &= sign;
    }
  }
  float x = Half ? rocjitsu::arith::f16_to_float(uint16_t(a)) : rocjitsu::arith::as_float(a);
  float y = Half ? rocjitsu::arith::f16_to_float(uint16_t(b)) : rocjitsu::arith::as_float(b), value;
  if constexpr (Op == rocjitsu::arith::Binary::Add) {
    value = x + y;
  }
  if constexpr (Op == rocjitsu::arith::Binary::Sub) {
    value = x - y;
  }
  if constexpr (Op == rocjitsu::arith::Binary::Mul) {
    value = x * y;
  }
  if constexpr (Op == rocjitsu::arith::Binary::MinNum) {
    value = rocjitsu::arith::minmax<false, false>(x, y);
  }
  if constexpr (Op == rocjitsu::arith::Binary::MaxNum) {
    value = rocjitsu::arith::minmax<true, false>(x, y);
  }
  if constexpr (Op == rocjitsu::arith::Binary::Minimum) {
    value = rocjitsu::arith::minmax<false, true>(x, y);
  }
  if constexpr (Op == rocjitsu::arith::Binary::Maximum) {
    value = rocjitsu::arith::minmax<true, true>(x, y);
  }
  uint32_t result = Half ? rocjitsu::arith::float_to_f16(value, flags & RJ_ARITH_FP16_OVFL)
                         : rocjitsu::arith::as_bits(value);
  if constexpr (Op == rocjitsu::arith::Binary::Add || Op == rocjitsu::arith::Binary::Sub ||
                Op == rocjitsu::arith::Binary::Mul) {
    if (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) {
      if (!(result & inf)) {
        result &= sign;
      }
      if constexpr (Op == rocjitsu::arith::Binary::Mul) {
        // GPU tininess detection precedes destination subnormal rounding. The
        // threshold is a quarter subnormal ULP below the smallest normal; ties
        // round to the normal value. Double holds the exact FP32 product.
        constexpr double threshold = Half ? (0x1p-14 - 0x1p-26) : (0x1p-126 - 0x1p-151);
        if (std::abs(double(x) * double(y)) < threshold) {
          result &= sign;
        }
      }
    }
  }
  *d = result;
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Add the operands.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_add_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                       uint32_t b, [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run<rocjitsu::arith::Binary::Add, false>(flags, instruction_flags, d, a, b);
}

// Add the operands.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_add_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                       uint32_t b, [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run<rocjitsu::arith::Binary::Add, true>(flags, instruction_flags, d, a, b);
}

// Subtract B from A.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_sub_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                       uint32_t b, [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run<rocjitsu::arith::Binary::Sub, false>(flags, instruction_flags, d, a, b);
}

// Subtract B from A.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_sub_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                       uint32_t b, [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run<rocjitsu::arith::Binary::Sub, true>(flags, instruction_flags, d, a, b);
}

// Multiply the operands.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_mul_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                       uint32_t b, [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run<rocjitsu::arith::Binary::Mul, false>(flags, instruction_flags, d, a, b);
}

// Multiply the operands.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_mul_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                       uint32_t b, [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run<rocjitsu::arith::Binary::Mul, true>(flags, instruction_flags, d, a, b);
}

// Select the minimum with numeric-preferred NaN selection.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_min_num_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                           uint32_t b, [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run<rocjitsu::arith::Binary::MinNum, false>(flags, instruction_flags, d, a, b);
}

// Select the minimum with numeric-preferred NaN selection.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_min_num_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                           uint32_t b, [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run<rocjitsu::arith::Binary::MinNum, true>(flags, instruction_flags, d, a, b);
}

// Select the maximum with numeric-preferred NaN selection.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_max_num_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                           uint32_t b, [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run<rocjitsu::arith::Binary::MaxNum, false>(flags, instruction_flags, d, a, b);
}

// Select the maximum with numeric-preferred NaN selection.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_max_num_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                           uint32_t b, [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run<rocjitsu::arith::Binary::MaxNum, true>(flags, instruction_flags, d, a, b);
}

// Select the NaN-propagating minimum.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_minimum_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                           uint32_t b, [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run<rocjitsu::arith::Binary::Minimum, false>(flags, instruction_flags, d, a, b);
}

// Select the NaN-propagating minimum.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_minimum_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                           uint32_t b, [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run<rocjitsu::arith::Binary::Minimum, true>(flags, instruction_flags, d, a, b);
}

// Select the NaN-propagating maximum.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_maximum_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                           uint32_t b, [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run<rocjitsu::arith::Binary::Maximum, false>(flags, instruction_flags, d, a, b);
}

// Select the NaN-propagating maximum.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_maximum_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                           uint32_t b, [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run<rocjitsu::arith::Binary::Maximum, true>(flags, instruction_flags, d, a, b);
}
