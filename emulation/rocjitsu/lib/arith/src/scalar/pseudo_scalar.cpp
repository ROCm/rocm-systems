// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Scalar-result FP16/FP32 special functions with guest modifiers and denormal rules.
// Owns public entry points and their validation/dispatch policy.

// Exact stage ordering follows rocjitsu shared/pseudo_scalar.h and transcendental.h.
// GFX1201 captures correct half OMOD underflow: newly created -0 keeps its sign.

#include "common/float_encoding.h"
#include "common/validation.h"
#include "fp/half_fma_scalar.h"
#include "math/half_unary.h"
#include "math/half_unary_exact.h"
#include "math/unary_exact.h"
#include "math/unary_impl.h"
#include "rj_arith/detail/rj_arith_scalar.h"

#include <stdint.h>

namespace {

constexpr uint64_t kModifiers =
    RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_OMOD_MASK | RJ_ARITH_ALU_CLAMP;

// Apply scalar sign modifiers and quiet a NaN, or produce zero under CLAMP.
template <bool Half> uint32_t nan_result(uint32_t raw, uint32_t mode) {
  constexpr uint32_t sign = Half ? 0x8000U : 0x80000000U;
  raw &= Half ? 0xffffU : 0xffffffffU;
  if (mode & RJ_ARITH_ALU_ABS_A) {
    raw &= sign - 1;
  }
  if (mode & RJ_ARITH_ALU_NEG_A) {
    raw ^= sign;
  }
  return mode & RJ_ARITH_ALU_CLAMP ? 0 : raw | (Half ? 0x200U : 0x00400000U);
}

// Encode an infinite result from finite input, applying Loose saturation and CLAMP.
template <bool Half> uint32_t infinite_result(bool negative, uint64_t flags, uint32_t mode) {
  if (mode & RJ_ARITH_ALU_CLAMP) {
    return negative ? 0 : Half ? 0x3c00U : 0x3f800000U;
  }
  return (negative ? (Half ? 0x8000U : 0x80000000U) : 0) |
         (Half && (flags & RJ_ARITH_FP16_OVFL) ? 0x7bffU
          : Half                               ? 0x7c00U
                                               : 0x7f800000U);
}

// Evaluate one Loose scalar result with validated controls.
template <rocjitsu::arith::Unary Op, bool Half>
[[gnu::always_inline]] inline int evaluate_loose(uint64_t flags, uint32_t mode, uint32_t *d,
                                                 uint32_t raw) {
  constexpr uint32_t sign = Half ? 0x8000 : 0x80000000, mag = sign - 1,
                     inf = Half ? 0x7c00 : 0x7f800000;
  // Skip conversions and libm on NaNs; native FP32 arithmetic handles them cheaply.
  if ((Half || Op == rocjitsu::arith::Unary::Exp || Op == rocjitsu::arith::Unary::Log) &&
      (raw & mag) > inf) {
    *d = nan_result<Half>(raw, mode);
    return RJ_ARITH_SUCCESS;
  }
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
  if constexpr (Op != rocjitsu::arith::Unary::Exp) {
    // Zero and negative domain results need neither host math nor half conversion.
    if (!(raw & mag)) {
      if constexpr (Op == rocjitsu::arith::Unary::Sqrt) {
        *d = mode & (RJ_ARITH_ALU_CLAMP | (Half ? 0 : RJ_ARITH_ALU_OMOD_MASK)) ? 0 : raw;
      } else {
        *d = infinite_result<Half>(Op == rocjitsu::arith::Unary::Log || (raw & sign), flags, mode);
      }
      return RJ_ARITH_SUCCESS;
    }
    if constexpr (Op != rocjitsu::arith::Unary::Rcp) {
      if (raw & sign) {
        *d = mode & RJ_ARITH_ALU_CLAMP ? 0 : inf | (Half ? 0x200U : 0x00400000U);
        return RJ_ARITH_SUCCESS;
      }
    }
  }
  // EXP/LOG use FP32 libm; native divide/sqrt retain their FP64 intermediates.
  double x = Half ? rocjitsu::arith::f16_to_float(uint16_t(raw)) : rocjitsu::arith::as_float(raw),
         value;
  if constexpr (Op == rocjitsu::arith::Unary::Exp) {
    value = __builtin_exp2f(__builtin_isfinite(x)
                                ? rocjitsu::arith::clamp(x, Half ? -32. : -256., Half ? 32. : 256.)
                                : x);
  }
  if constexpr (Op == rocjitsu::arith::Unary::Log) {
    value = __builtin_log2f(x);
  }
  if constexpr (Op == rocjitsu::arith::Unary::Rcp) {
    value = 1 / x;
  }
  if constexpr (Op == rocjitsu::arith::Unary::Rsq) {
    value = 1 / __builtin_sqrt(x);
  }
  if constexpr (Op == rocjitsu::arith::Unary::Sqrt) {
    value = __builtin_sqrt(x);
  }
  const unsigned omod = (mode >> 6) & 3;
  if constexpr (Half) {
    // Loose scales and clamps before its single narrowing to FP16.
    if (omod) {
      value *= omod == 1 ? 2 : omod == 2 ? 4 : 0.5f;
    }
    if (mode & RJ_ARITH_ALU_CLAMP) {
      value = !(value > 0) ? 0 : value > 1 ? 1 : value;
    }
    const bool ovfl = flags & RJ_ARITH_FP16_OVFL;
    uint16_t result = rocjitsu::arith::float_to_f16(value, ovfl);
    if (ovfl && finite && __builtin_isinf(value)) {
      result = (__builtin_signbit(value) ? 0x8000 : 0) | 0x7bff;
    }
    if ((omod || (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) && !(result & 0x7c00)) {
      result &= 0x8000;
    }
    *d = result;
  } else {
    float rounded = value;
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

// Validate an ordinary scalar-result instruction before numerical evaluation.
template <rocjitsu::arith::Unary Op, bool Half>
int run(uint64_t flags, uint32_t mode, uint32_t *d, uint32_t raw) {
  if (int error = rocjitsu::arith::validate(flags, mode & ~kModifiers,
                                            RJ_ARITH_FP_FLUSH_INPUT_DENORMALS |
                                                RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
    return error;
  }
  // Every FP32 and FP16 form has a hardware model: the VALU EXP/LOG F16 model also matches the
  // gfx1201 V_S_EXP_F16/V_S_LOG_F16 captures.
  if (rocjitsu::arith::is_exact(flags)) {
    if constexpr (Half) {
      *d = rocjitsu::arith::half_unary_exact_value<Op>(flags, mode, uint16_t(raw));
    } else {
      *d = rocjitsu::arith::unary_exact_value<Op>(flags, mode, raw, false);
    }
    return RJ_ARITH_SUCCESS;
  }
  return evaluate_loose<Op, Half>(flags, mode, d, raw);
}

// Preserve the inherited gfx1250 model without claiming hardware-validated exactness.
// Strict semantics reject before outputs; fallback semantics use the ordinary Loose path.
template <rocjitsu::arith::Unary Op, bool Half>
int run_gfx1250(uint64_t flags, uint64_t mode, uint32_t *d, uint32_t raw) {
  const bool compatibility = (flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_COMPATIBILITY;
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate(flags, mode & ~kModifiers,
                                            RJ_ARITH_FP_FLUSH_INPUT_DENORMALS |
                                                RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
    return error;
  }
  if ((flags & RJ_ARITH_SEMANTICS_MASK) >= RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS) {
    return RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS;
  }
  // Both policies can publish a NaN without entering their numerical helpers.
  constexpr uint32_t magnitude = Half ? 0x7fffU : 0x7fffffffU;
  constexpr uint32_t infinity = Half ? 0x7c00U : 0x7f800000U;
  if ((raw & magnitude) > infinity) {
    *d = nan_result<Half>(raw, uint32_t(mode));
    return RJ_ARITH_SUCCESS;
  }
  if (compatibility) {
    if constexpr (Half) {
      *d = rocjitsu::arith::half_unary_exact_value<Op>(flags, uint32_t(mode), uint16_t(raw));
    } else {
      *d = rocjitsu::arith::unary_exact_value<Op>(flags, uint32_t(mode), raw, false);
    }
    return RJ_ARITH_SUCCESS;
  }
  return evaluate_loose<Op, Half>(flags & ~RJ_ARITH_SEMANTICS_MASK, uint32_t(mode), d, raw);
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

// Evaluate gfx1250 EXP with inherited Compatibility or Loose arithmetic.
int rj_arith_v_s_exp_f16_gfx1250(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                                 uint32_t a, [[maybe_unused]] uint32_t *excp_flag_user) {
  return run_gfx1250<rocjitsu::arith::Unary::Exp, true>(flags, instruction_flags, d, a);
}

// Evaluate gfx1250 EXP with inherited Compatibility or Loose arithmetic.
int rj_arith_v_s_exp_f32_gfx1250(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                                 uint32_t a, [[maybe_unused]] uint32_t *excp_flag_user) {
  return run_gfx1250<rocjitsu::arith::Unary::Exp, false>(flags, instruction_flags, d, a);
}

// Evaluate gfx1250 LOG with inherited Compatibility or Loose arithmetic.
int rj_arith_v_s_log_f16_gfx1250(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                                 uint32_t a, [[maybe_unused]] uint32_t *excp_flag_user) {
  return run_gfx1250<rocjitsu::arith::Unary::Log, true>(flags, instruction_flags, d, a);
}

// Evaluate gfx1250 LOG with inherited Compatibility or Loose arithmetic.
int rj_arith_v_s_log_f32_gfx1250(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                                 uint32_t a, [[maybe_unused]] uint32_t *excp_flag_user) {
  return run_gfx1250<rocjitsu::arith::Unary::Log, false>(flags, instruction_flags, d, a);
}

// Evaluate gfx1250 RCP with inherited Compatibility or Loose arithmetic.
int rj_arith_v_s_rcp_f16_gfx1250(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                                 uint32_t a, [[maybe_unused]] uint32_t *excp_flag_user) {
  return run_gfx1250<rocjitsu::arith::Unary::Rcp, true>(flags, instruction_flags, d, a);
}

// Evaluate gfx1250 RCP with inherited Compatibility or Loose arithmetic.
int rj_arith_v_s_rcp_f32_gfx1250(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                                 uint32_t a, [[maybe_unused]] uint32_t *excp_flag_user) {
  return run_gfx1250<rocjitsu::arith::Unary::Rcp, false>(flags, instruction_flags, d, a);
}

// Evaluate gfx1250 RSQ with inherited Compatibility or Loose arithmetic.
int rj_arith_v_s_rsq_f16_gfx1250(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                                 uint32_t a, [[maybe_unused]] uint32_t *excp_flag_user) {
  return run_gfx1250<rocjitsu::arith::Unary::Rsq, true>(flags, instruction_flags, d, a);
}

// Evaluate gfx1250 RSQ with inherited Compatibility or Loose arithmetic.
int rj_arith_v_s_rsq_f32_gfx1250(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                                 uint32_t a, [[maybe_unused]] uint32_t *excp_flag_user) {
  return run_gfx1250<rocjitsu::arith::Unary::Rsq, false>(flags, instruction_flags, d, a);
}

// Evaluate gfx1250 SQRT with inherited Compatibility or Loose arithmetic.
int rj_arith_v_s_sqrt_f16_gfx1250(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                                  uint32_t a, [[maybe_unused]] uint32_t *excp_flag_user) {
  return run_gfx1250<rocjitsu::arith::Unary::Sqrt, true>(flags, instruction_flags, d, a);
}

// Evaluate gfx1250 SQRT with inherited Compatibility or Loose arithmetic.
int rj_arith_v_s_sqrt_f32_gfx1250(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                                  uint32_t a, [[maybe_unused]] uint32_t *excp_flag_user) {
  return run_gfx1250<rocjitsu::arith::Unary::Sqrt, false>(flags, instruction_flags, d, a);
}
