// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Scalar fused multiply-add under guest FP semantics.
// Owns public entry points and their validation/dispatch policy.

// Fused rounding and tininess detection follow rocjitsu shared/fp_mode.h.
// FP16 uses the existing rocjitsu-derived half_fma_value model.

#include "common/float_formats.h"
#include "common/internal.h"
#include "fp/fma_scalar.h"
#include "fp/half_fma_scalar.h"
#include "rj_arith/rj_arith.h"

#include <cmath>
#include <stdint.h>

namespace {

// Evaluate scalar fused multiply-add. Reject invalid controls before changing outputs.
template <bool Half, bool Accumulate>
[[gnu::always_inline]] inline int run(uint64_t flags, uint32_t mode, uint32_t *d, uint32_t a,
                                      uint32_t b, uint32_t c) {
  if (int error = rocjitsu::arith::validate(flags, mode, !Half,
                                            RJ_ARITH_FP_FLUSH_INPUT_DENORMALS |
                                                RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS |
                                                (Half ? 0 : RJ_ARITH_FP_ROUND_MASK))) {
    return error;
  }
  if constexpr (Accumulate) {
    c = *d;
  }
  constexpr uint32_t sign = Half ? 0x8000 : 0x80000000, inf = Half ? 0x7c00 : 0x7f800000;
  if constexpr (Half) {
    a &= 65535;
    b &= 65535;
    c &= 65535;
  }
  if (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) {
    if (!(a & inf)) {
      a &= sign;
    }
    if (!(b & inf)) {
      b &= sign;
    }
    if (!(c & inf)) {
      c &= sign;
    }
  }
  if constexpr (!Half) {
    if (rocjitsu::arith::is_exact(flags)) {
      *d = rocjitsu::arith::fma_value_f32(a, b, c, flags, false);
      return RJ_ARITH_SUCCESS;
    }
  }
  float x = Half ? rocjitsu::arith::f16_to_float(uint16_t(a)) : rocjitsu::arith::as_float(a);
  float y = Half ? rocjitsu::arith::f16_to_float(uint16_t(b)) : rocjitsu::arith::as_float(b);
  float z = Half ? rocjitsu::arith::f16_to_float(uint16_t(c)) : rocjitsu::arith::as_float(c);
  uint32_t result;
  if constexpr (Half) {
    result = rocjitsu::arith::half_fma_value(uint16_t(a), uint16_t(b), uint16_t(c), 0,
                                             flags & RJ_ARITH_FP16_OVFL);
  } else {
    result = rocjitsu::arith::as_bits(std::fma(x, y, z));
  }
  if (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) {
    uint32_t magnitude = result & (sign - 1);
    constexpr uint32_t normal = Half ? 0x400 : 0x800000;
    if (magnitude < normal) {
      result &= sign;
    }
    if (magnitude == normal) {
      // TwoSum retains tiny addends when deciding which side of the tininess
      // threshold the exact result occupies. The product is exact in double.
      double product = double(x) * double(y), sum = product + double(z), part = sum - product;
      double residual = (product - (sum - part)) + (double(z) - part);
      double absolute = (result & sign) ? -sum : sum, tail = (result & sign) ? -residual : residual;
      constexpr double threshold = Half ? (0x1p-14 - 0x1p-26) : (0x1p-126 - 0x1p-151);
      if (absolute < threshold || (absolute == threshold && tail < 0)) {
        result &= sign;
      }
    }
  }
  *d = result;
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Accumulate fused A*B into D.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_fmac_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                        uint32_t b, [[maybe_unused]] uint32_t *excp_flag_user) {
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
  return run<false, true>(flags, instruction_flags, d, a, b, 0);
}

// Accumulate fused A*B into D.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_fmac_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
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
  return run<true, true>(flags, instruction_flags, d, a, b, 0);
}

// Compute fused A*B+literal.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_fmaak_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                         uint32_t b, uint32_t literal, [[maybe_unused]] uint32_t *excp_flag_user) {
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
  return run<false, false>(flags, instruction_flags, d, a, b, literal);
}

// Compute fused A*literal+B.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_fmamk_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                         uint32_t literal, uint32_t c, [[maybe_unused]] uint32_t *excp_flag_user) {
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
  return run<false, false>(flags, instruction_flags, d, a, literal, c);
}
