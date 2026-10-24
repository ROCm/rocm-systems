// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Scalar-result FP16/FP32 special functions with guest modifiers and denormal rules.
// Owns public entry points and their validation/dispatch policy.

// Stage ordering follows rocjitsu shared/pseudo_scalar.h and transcendental.h.
// GFX1201 captures correct half OMOD underflow: newly created -0 keeps its sign.

#include "common/float_formats.h"
#include "common/internal.h"
#include "fp/half_fma_scalar.h"
#include "math/half_unary.h"
#include "math/half_unary_exact.h"
#include "math/unary.h"
#include "math/unary_exact.h"
#include "rj_arith/rj_arith.h"
#include <bit>

#include <algorithm>
#include <cmath>
#include <stdint.h>

namespace {

// Evaluate a scalar-result special function. Reject invalid controls before changing outputs.
template <rocjitsu::arith::Unary Op, bool Half>
int run(uint64_t flags, uint32_t mode, uint32_t *d, uint32_t raw) {
  const uint32_t known =
      RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_OMOD_HALF | RJ_ARITH_ALU_CLAMP;
  constexpr bool supports_exact = !Half || rocjitsu::arith::half_unary_sfu<Op>;
  if (int error = rocjitsu::arith::validate(flags, mode & ~known, supports_exact,
                                            RJ_ARITH_FP_FLUSH_INPUT_DENORMALS |
                                                RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
    return error;
  }
  if constexpr (supports_exact) {
    if (rocjitsu::arith::is_exact(flags)) {
      if constexpr (Half) {
        *d = rocjitsu::arith::half_unary_exact_value<Op>(flags, mode, uint16_t(raw));
      } else {
        *d = rocjitsu::arith::unary_exact_value<Op>(flags, mode, raw, false);
      }
      return RJ_ARITH_SUCCESS;
    }
  }
  constexpr uint32_t sign = Half ? 0x8000 : 0x80000000, mag = sign - 1,
                     inf = Half ? 0x7c00 : 0x7f800000;
  if constexpr (Half) {
    raw &= 65535;
  }
  if (mode & RJ_ARITH_ALU_ABS_A) {
    raw &= mag;
  }
  if (mode & RJ_ARITH_ALU_NEG_A) {
    raw ^= sign;
  }

  // Remember whether infinity was already present in the input, for finite-overflow saturation
  // below.
  const bool finite = (raw & mag) < inf;
  if ((!Half || (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS)) && !(raw & inf)) {
    raw &= sign;
  }
  double x = Half ? rocjitsu::arith::f16_to_float(uint16_t(raw)) : rocjitsu::arith::as_float(raw),
         value;
  if constexpr (Op == rocjitsu::arith::Unary::Exp) {
    value = std::exp2(std::isfinite(x) ? std::clamp(x, -256., 256.) : x);
  }
  if constexpr (Op == rocjitsu::arith::Unary::Log) {
    value = std::log2(x);
  }
  if constexpr (Op == rocjitsu::arith::Unary::Rcp) {
    value = 1 / x;
  }
  if constexpr (Op == rocjitsu::arith::Unary::Rsq) {
    value = 1 / std::sqrt(x);
  }
  if constexpr (Op == rocjitsu::arith::Unary::Sqrt) {
    value = std::sqrt(x);
  }
  const unsigned omod = (mode >> 6) & 3;
  if constexpr (Half) {
    bool ovfl = flags & RJ_ARITH_FP16_OVFL;
    uint16_t result = std::isfinite(value)
                          ? rocjitsu::arith::half_fma_narrow(std::bit_cast<uint64_t>(value), ovfl)
                          : rocjitsu::arith::float_to_f16(float(value), false);
    if (ovfl && finite && std::isinf(value)) {
      result = (std::signbit(value) ? 0x8000 : 0) | 0x7bff;
    }
    if ((flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) && !(result & 0x7c00)) {
      result &= 0x8000;
    }
    if (omod) {
      if ((result & 0x7fff) < 0x400) {
        result = 0;
      }
      float scaled = rocjitsu::arith::f16_to_float(result) * (omod == 1 ? 2 : omod == 2 ? 4 : 0.5f);
      result = rocjitsu::arith::float_to_f16(scaled, ovfl);
      if (!(result & 0x7c00)) {
        result &= 0x8000;
      }
    }
    if (mode & RJ_ARITH_ALU_CLAMP) {
      result = rocjitsu::arith::half_fma_clamp(result);
    }
    *d = result;
  } else {
    float rounded = float(value);
    uint32_t result = rocjitsu::arith::as_bits(rounded);
    if (!(result & 0x7f800000)) {
      result &= 0x80000000;
    }
    rounded = rocjitsu::arith::as_float(result);
    if (omod) {
      rounded *= omod == 1 ? 2 : omod == 2 ? 4 : 0.5f;
      if (rounded == 0) {
        rounded = 0;
      }
    }
    if (mode & RJ_ARITH_ALU_CLAMP) {
      rounded = !(rounded > 0) ? 0 : rounded > 1 ? 1 : rounded;
    }
    result = rocjitsu::arith::as_bits(rounded);
    if (!(result & 0x7f800000)) {
      result &= 0x80000000;
    }
    *d = result;
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Compute base-two exponentiation.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_s_exp_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                         [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run<rocjitsu::arith::Unary::Exp, false>(flags, instruction_flags, d, a);
}

// Compute base-two exponentiation.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_s_exp_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                         [[maybe_unused]] uint32_t *excp_flag_user) {
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
  return run<rocjitsu::arith::Unary::Exp, true>(flags, instruction_flags, d, a);
}

// Compute the base-two logarithm.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_s_log_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                         [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run<rocjitsu::arith::Unary::Log, false>(flags, instruction_flags, d, a);
}

// Compute the base-two logarithm.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_s_log_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                         [[maybe_unused]] uint32_t *excp_flag_user) {
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
  return run<rocjitsu::arith::Unary::Log, true>(flags, instruction_flags, d, a);
}

// Compute the reciprocal.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_s_rcp_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                         [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run<rocjitsu::arith::Unary::Rcp, false>(flags, instruction_flags, d, a);
}

// Compute the reciprocal.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_s_rcp_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                         [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run<rocjitsu::arith::Unary::Rcp, true>(flags, instruction_flags, d, a);
}

// Compute the reciprocal square root.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_s_rsq_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                         [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run<rocjitsu::arith::Unary::Rsq, false>(flags, instruction_flags, d, a);
}

// Compute the reciprocal square root.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_s_rsq_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                         [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run<rocjitsu::arith::Unary::Rsq, true>(flags, instruction_flags, d, a);
}

// Compute the square root.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_s_sqrt_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                          [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run<rocjitsu::arith::Unary::Sqrt, false>(flags, instruction_flags, d, a);
}

// Compute the square root.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_s_sqrt_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                          [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run<rocjitsu::arith::Unary::Sqrt, true>(flags, instruction_flags, d, a);
}
